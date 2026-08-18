"""A changed RULER must change the generation, not just a changed SET.

`holdout_generation` is the identity the best-model comparison keys on, but it
only ever moved when the holdout SET moved -- a drift reset, or a restart that
could not restore the rows. The MEASUREMENT applied to the set was outside the
counter entirely, so PR #277's swap from 2560 draws with replacement
(WDL-rebalanced, half priority-weighted) to one deterministic pass over the
same 2000 rows left the generation at 1 on both sides:

    iter 160-162: best_loss 4.90535  test_size 2560  holdout_generation 1
    iter 165:     best_loss 4.85326  test_size 2000  holdout_generation 1

`_update_best_model` took its SAME-RULER branch and promoted across two
instruments. The step was -0.156 nats: -5.70 sd on policy and +0.27 sd (flat)
on WDL, which is the fingerprint of dropping priority weighting rather than of
learning.

These tests pin the mechanism that makes that impossible by construction, and
-- just as important -- pin the failure mode in the opposite direction: a
ruler that did NOT change must not bump, or every restart hands the best model
over and the counter stops meaning anything.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from chess_anti_engine.config_keys import TRAINER_WEIGHT_KEYS
from chess_anti_engine.train.eval_ruler import (
    _UNAVAILABLE,
    active_loss_terms,
    call_closure,
    digest_source,
    eval_ruler_id,
    semantic_source_digest,
)
from chess_anti_engine.train.trainer import TrainMetrics, Trainer
from chess_anti_engine.tune.trainable import _maybe_bump_generation_on_ruler_change
from chess_anti_engine.tune.trainable_init import _apply_restored_holdout_scalars
from chess_anti_engine.tune.trial_config import RestoreResult

  # The covered set is DERIVED from the call graph, not listed here -- listing
  # it is what failed twice. These names exist so the tests can say WHICH
  # frames must be in the derived set, which is a different (and checkable)
  # claim from "here is the set".
FULL_PASS_FNS: tuple[Callable[..., Any], ...] = call_closure(
    Trainer._compute_metrics, owner=Trainer, skip=(Trainer._iter_prefetched_batches,),
)
SAMPLED_FNS: tuple[Callable[..., Any], ...] = call_closure(
    Trainer._compute_metrics, owner=Trainer, skip=(Trainer._iter_full_pass_batches,),
)

  # ⚑ THE PINS BELOW ARE TAKEN WITH EVERY LOSS WEIGHT AT ZERO, ON PURPOSE.
  # The id now depends on the SET of non-zero weights as well as on the covered
  # source (see `eval_ruler.active_loss_terms`), and the golden constants exist
  # to catch a SOURCE change. Pinning them against a live weight map would make
  # CI red whenever an operator edits a weight in `configs/pbt2_small.yaml` --
  # a real edit to the live yaml, reported as a code regression. With the map
  # empty the weight field of the descriptor is constant, so these two values
  # move if and only if the measurement code moves, which is exactly the
  # property they had before the weights entered the id at all.
NO_TERMS: dict[str, float] = dict.fromkeys(TRAINER_WEIGHT_KEYS, 0.0)


def full_pass(
    *, batch_size: int = 512, steps: int = 0,
    measured_by: Sequence[Callable[..., Any]] = FULL_PASS_FNS,
    loss_weights: Mapping[str, float] = NO_TERMS,
) -> str:
    """The production ruler: a deterministic pass at the production batch size."""
    return eval_ruler_id(
        mode="full_pass", batch_size=batch_size, steps=steps, mirror_prob=0.0,
        measured_by=measured_by, loss_weights=loss_weights,
    )


def sampled(
    *, batch_size: int = 512, steps: int = 5,
    loss_weights: Mapping[str, float] = NO_TERMS,
) -> str:
    """The pre-PR-277 ruler: `steps` x `batch_size` draws with replacement."""
    return eval_ruler_id(
        mode="sampled", batch_size=batch_size, steps=steps, mirror_prob=0.0,
        measured_by=SAMPLED_FNS, loss_weights=loss_weights,
    )


def _ruler_at(loss_weights: Mapping[str, float]) -> Callable[..., str]:
    """`Trainer.eval_ruler_id_for` with one weight map held fixed.

    Written as a closure rather than a `functools.partial` so the four
    descriptor arguments stay spelled out at every call site below -- they are
    what those tests vary.
    """
    def call(*, batch_size: int, steps: int, mirror_prob: float, full_pass: bool) -> str:
        return Trainer.eval_ruler_id_for(
            batch_size=batch_size, steps=steps, mirror_prob=mirror_prob,
            full_pass=full_pass, loss_weights=loss_weights,
        )
    return call


# --- the identity itself --------------------------------------------------


def test_the_pr_277_swap_produces_a_different_ruler_id() -> None:
    """The exact change that went unnoticed: same set, same batch size, two
    instruments."""
    assert full_pass() != sampled()


def test_the_same_measurement_produces_the_same_ruler_id() -> None:
    """The opposite failure. A ruler id that is not reproducible would bump the
    generation on every iteration and hand the best model over forever."""
    assert full_pass() == full_pass()
    assert full_pass(batch_size=512, steps=0) == full_pass()


def test_the_production_measurement_functions_have_readable_source() -> None:
    """The digest degrades to a constant when `inspect.getsource` fails, and a
    constant cannot detect anything. Pin that the real production functions
    are not silently on that path -- otherwise the deploy check still sees a
    well-formed id while half the mechanism is inert."""
    for fn in (*FULL_PASS_FNS, *SAMPLED_FNS):
        assert semantic_source_digest(fn) != _UNAVAILABLE

    assert full_pass().startswith("v1:full_pass:")


def test_the_trainer_covers_exactly_the_documented_measurement_set() -> None:
    """The production method must hash the frames this module says it does.
    Drop one there and this fails, rather than the coverage quietly shrinking
    while the docstring still promises it."""
    ruler = _ruler_at(NO_TERMS)

    assert ruler(batch_size=512, steps=0, mirror_prob=0.0, full_pass=True) == full_pass()
    assert ruler(batch_size=512, steps=5, mirror_prob=0.0, full_pass=False) == sampled()


def test_every_covered_function_is_load_bearing() -> None:
    """Each frame must actually reach the id. A frame that does not is a
    claim, not a check -- which is what the `pooling="row_weighted"` string
    literal was: the row-weighted denominator lives in `_compute_metrics`, and
    with that function unhashed the denominator could be changed with the id
    sitting still (verified against the real code: the pooling mutation moved
    `v1:full_pass:e0a17086400b76dc` to `...d2c40c8cb2d95c83` only after
    `_compute_metrics` joined the set)."""
    for fn in FULL_PASS_FNS:
        reduced = tuple(f for f in FULL_PASS_FNS if f is not fn)

        assert full_pass(measured_by=reduced) != full_pass(), (
            f"{fn.__qualname__} is in the covered set but does not reach the id"
        )


def test_the_metric_assembly_tail_is_covered() -> None:
    """The regression pin for the second wrong boundary. `_compute_metrics`
    computes the pooling denominator but does NOT apply it: the division is
    `_loss_sums_to_metric_kwargs`, reached through `_build_metrics`, and the
    scalar extraction is `_extract_loss_scalars`. With those three outside the
    covered set `test_loss` could be DOUBLED with this whole file green, which
    is how the second hand-written list failed. They are named here because
    the derived closure must not silently stop reaching them."""
    from chess_anti_engine.train import trainer as trainer_mod

    covered = {fn.__qualname__ for fn in FULL_PASS_FNS}

    assert "Trainer._build_metrics" in covered
    assert "Trainer._extract_loss_scalars" in covered
    assert "_loss_sums_to_metric_kwargs" in covered
  # ...and the frames the first wrong boundary missed, still covered.
    assert "Trainer._compute_metrics" in covered
    assert "Trainer._prepare_host_arrays" in covered
    assert "Trainer._host_batch_to_tensors" in covered
  # ...and one nobody enumerated in either round: the autocast context is
  # part of the measurement, and the closure found it without being told.
    assert "Trainer._amp_context" in covered
    assert trainer_mod._loss_sums_to_metric_kwargs in FULL_PASS_FNS


def test_the_closure_is_derived_rather_than_enumerated() -> None:
    """The property that makes the boundary hold as the code moves: a call
    added on the path is covered without anyone updating a list."""

    class _Fake:
        def entry(self) -> None:
            self.middle()

        def middle(self) -> None:
            self.leaf()

        def leaf(self) -> None: ...

        def unrelated(self) -> None: ...

    covered = {fn.__qualname__.split(".")[-1] for fn in call_closure(_Fake.entry, owner=_Fake)}

    assert covered == {"entry", "middle", "leaf"}, (
        "the closure must follow calls transitively and stop at what is not called"
    )


def test_the_closure_stops_at_the_module_boundary() -> None:
    """The stated edge. A frame from another module is not followed, which is
    why `compute_loss` and the encoders are declared uncovered rather than
    quietly half-covered."""
    covered_modules = {fn.__module__ for fn in FULL_PASS_FNS}

    assert covered_modules == {"chess_anti_engine.train.trainer"}


def test_the_untaken_branch_is_pruned_so_the_modes_stay_isolated() -> None:
    """A change to the legacy sampled path must not hand the best model over
    on the full-pass ruler. Verified against the real code by mutation:
    `_sample_batch_host` moves the sampled id and leaves the full-pass id
    byte-identical."""
    full = {fn.__qualname__ for fn in FULL_PASS_FNS}
    samp = {fn.__qualname__ for fn in SAMPLED_FNS}

    assert "Trainer._sample_batch_host" not in full
    assert "Trainer._iter_prefetched_batches" not in full
    assert "Trainer._full_pass_host_batch" not in samp
    assert "Trainer._iter_full_pass_batches" not in samp


def test_batch_size_is_part_of_the_ruler() -> None:
    """It sets the draw count in sampled mode and the ragged tail in a full
    pass; either way two batch sizes are two measurements."""
    assert full_pass(batch_size=256) != full_pass()


def test_sampled_step_count_is_part_of_the_ruler() -> None:
    """2560 draws and 512 draws from the same 2000 rows are different rulers,
    with different noise floors."""
    assert sampled(steps=1) != sampled()


def test_knobs_that_cannot_reach_the_full_pass_cannot_move_its_ruler() -> None:
    """`test_steps` and `mirror_prob` no longer reach the full pass -- it walks
    every row once and pins mirroring off. A knob that cannot move the number
    must not move its identity, or the handover fires on nothing."""
    assert full_pass(steps=99) != full_pass(), (
        "both are in the descriptor; the pin belongs at the call site"
    )
    ruler = _ruler_at(NO_TERMS)
    baseline = ruler(batch_size=512, steps=5, mirror_prob=0.0, full_pass=True)

    assert ruler(batch_size=512, steps=99, mirror_prob=0.0, full_pass=True) == baseline
    assert ruler(batch_size=512, steps=5, mirror_prob=0.5, full_pass=True) == baseline
  # ...but they do reach the sampled ruler, where both are real.
    assert ruler(batch_size=512, steps=5, mirror_prob=0.5, full_pass=False) != ruler(
        batch_size=512, steps=5, mirror_prob=0.0, full_pass=False,
    )


# --- what the digest is and is not sensitive to ---------------------------


  # A pass, and the same pass after each of the two kinds of edit. Written as
  # source text rather than as real functions so the variants can share one
  # name -- a rename is a difference the digest is allowed to see, and it
  # would mask the difference each of these is actually testing. The column-2
  # comment is the repo's own convention and the reason `textwrap.dedent`
  # cannot be used on a method body.
_PASS = '''
    def _iter(self, buf, *, batch_size):
        bounds = buf.batch_row_bounds(batch_size)
        for start, stop in bounds:
            yield buf.rows_slice_arrays(start, stop)
'''
_PASS_RECOMMENTED = '''
    def _iter(self, buf, *, batch_size):
        """Every row exactly once, oldest first, in fixed order."""
  # A comment the original does not have.
        bounds = buf.batch_row_bounds(batch_size)

        for start, stop in bounds:
            yield buf.rows_slice_arrays(start, stop)
'''
_PASS_REORDERED = '''
    def _iter(self, buf, *, batch_size):
        bounds = list(reversed(buf.batch_row_bounds(batch_size)))
        for start, stop in bounds:
            yield buf.rows_slice_arrays(start, stop)
'''


def test_comments_docstrings_and_layout_do_not_change_the_digest() -> None:
    """This repo annotates heavily and reformats often. If prose moved the
    digest, the handover line would fire on cosmetics and be learned as noise."""
    assert digest_source(_PASS) == digest_source(_PASS_RECOMMENTED)
    assert digest_source(_PASS) != _UNAVAILABLE, "the column-2 comment broke the parse"


def test_a_rewritten_pass_changes_the_digest() -> None:
    """The half a declared descriptor cannot see: the call site is identical
    and the pass itself now reads the rows in a different order."""
    assert digest_source(_PASS) != digest_source(_PASS_REORDERED)


def test_a_rewritten_pass_changes_the_ruler_id() -> None:
    def _stub() -> None: ...

    unchanged = full_pass(measured_by=(_stub,))

    assert unchanged != full_pass(), (
        "the measurement functions' digests must reach the id"
    )


def test_an_unreadable_source_degrades_instead_of_raising() -> None:
    """A frozen install or an exec'd function has no source. The eval path must
    still produce an id -- the descriptor alone -- rather than raise."""
  # No file backs this one, which is what makes it unreadable to `getsource`.
    fn = eval(compile("lambda: None", "<nofile>", "eval"))

    assert semantic_source_digest(fn) == _UNAVAILABLE
    assert full_pass(measured_by=(fn,)).startswith("v1:full_pass:")


# --- the generation bump --------------------------------------------------


def test_a_measurement_change_bumps_the_generation() -> None:
    """The whole point. Same set, new instrument, new generation -- which is
    what routes `_update_best_model` into its handover branch."""
    ruler, gen = _maybe_bump_generation_on_ruler_change(
        observed_ruler=full_pass(),
        known_ruler=sampled(),
        holdout_generation=1,
    )

    assert gen == 2
    assert ruler == full_pass()


def test_an_unchanged_ruler_does_not_bump() -> None:
    """The failure mode in the other direction: a bump per iteration or per
    restart would invalidate a valid best model forever."""
    same = full_pass()

    assert _maybe_bump_generation_on_ruler_change(
        observed_ruler=same, known_ruler=same, holdout_generation=7,
    ) == (same, 7)


def test_no_holdout_eval_this_iteration_changes_nothing() -> None:
    """The G5 refill window, an async cold start and a timed-out collect all
    arrive as `test_metrics=None`. Absence of evidence is not a new ruler."""
    known = full_pass()

    assert _maybe_bump_generation_on_ruler_change(
        observed_ruler="", known_ruler=known, holdout_generation=3,
    ) == (known, 3)


def test_a_checkpoint_with_no_recorded_ruler_adopts_without_bumping() -> None:
    """Migration. Every live checkpoint carries `holdout_generation: 1` under
    the old meaning and no ruler at all; the record it defends was measured on
    the ruler running now. Adopt it, and only bump on the NEXT change."""
    observed = full_pass()

    ruler, gen = _maybe_bump_generation_on_ruler_change(
        observed_ruler=observed, known_ruler="", holdout_generation=1,
    )

    assert (ruler, gen) == (observed, 1)
  # ...and the next change is caught, which is what makes this a migration
  # rather than a hole.
    assert _maybe_bump_generation_on_ruler_change(
        observed_ruler=sampled(), known_ruler=ruler,
        holdout_generation=gen,
    ) == (sampled(), 2)


def test_a_legacy_trial_meta_restores_an_empty_ruler() -> None:
    """The migration's other half: nothing in the checkpoint, no exception,
    and no invented identity."""
    rr = RestoreResult()

    _apply_restored_holdout_scalars(rr, {"holdout_frozen": True, "holdout_generation": 1})

    assert (rr.holdout_frozen, rr.holdout_generation, rr.holdout_ruler) == (True, 1, "")


def test_reverting_a_measurement_bumps_again_rather_than_aliasing() -> None:
    """A -> B -> A is three generations, not two. The counter is monotone on
    purpose: a record earned under A stays handed-over rather than becoming
    silently comparable again after a round trip."""
    a, b = full_pass(), sampled()

    ruler, gen = _maybe_bump_generation_on_ruler_change(
        observed_ruler=b, known_ruler=a, holdout_generation=1,
    )
    _, gen = _maybe_bump_generation_on_ruler_change(
        observed_ruler=a, known_ruler=ruler, holdout_generation=gen,
    )

    assert gen == 3


# --- the wiring that carries it -------------------------------------------


def test_train_metrics_carry_no_ruler_by_default() -> None:
    """Training loss is not a ruler. Only `_compute_metrics` sets the field, so
    a train-path TrainMetrics must not look like an eval."""
    assert TrainMetrics(
        loss=1.0, policy_loss=1.0, soft_policy_loss=0.0, future_policy_loss=0.0,
        wdl_loss=0.0, sf_move_loss=0.0, sf_move_acc=0.0, sf_eval_loss=0.0,
        categorical_loss=0.0, volatility_loss=0.0, sf_volatility_loss=0.0,
        moves_left_loss=0.0,
    ).eval_ruler == ""


def test_the_eval_path_stamps_the_ruler_onto_the_metrics_it_returns() -> None:
    """The id has to come from the code that ran the measurement, not from a
    consumer's assumption about it -- the async holdout path calls
    `_compute_metrics` directly with its own `full_pass` argument, and a
    consumer-side guess would not notice the two diverging."""
    body = inspect.getsource(Trainer._compute_metrics)

    assert "ruler = type(self).eval_ruler_id_for(" in body
    assert "full_pass=bool(full_pass)" in body
    assert "eval_ruler=ruler," in body


def test_the_trial_loop_bumps_before_the_best_model_comparison() -> None:
    """Order is load-bearing: a number measured under a new ruler must meet the
    bumped generation, not the one it was promoted past."""
    from chess_anti_engine.tune import trainable

    src = Path(trainable.__file__).read_text(encoding="utf-8")

    assert src.index("_maybe_bump_generation_on_ruler_change(\n") < src.index(
        "best_loss, best_source = _update_best_model(",
    )
    assert "holdout_ruler = str(restore.holdout_ruler)" in src


# --- the pin that makes a ruler change announce itself --------------------

# Moved 2026-07-28 when PR #283 (live SF target rebuild) merged into this one.
# BOTH ids move, not just the full pass: #283 edits `_prepare_host_arrays` and
# `_compute_metrics`, which are in BOTH `measured_by` lists, plus
# `_sample_batch_host` / `_iter_prefetched_batches` in the sampled branch.
# Updating only the full-pass constant leaves main red.
#
# The MEASUREMENT did not change -- #283 pins `rebuild_sf_targets=False` in
# `_full_pass_host_batch`, so the full pass is byte-identical across it. This
# is the declared false positive (a moved id costs ONE best-model handover),
# and it is why the pin exists: the id moves whenever the covered source moves,
# and a human decides whether the meaning moved with it. Here it did not.
#   full_pass  c8fb48a79e804bb4 -> 2efe658b4e778870
#   sampled    e3cc3241626a581f -> d6f7cabecd8e6f67
#
# Moved again by the #283 review follow-up (fail-closed rebuild default):
# `_prepare_host_arrays` now defaults `rebuild_sf_targets=False` and
# `_sample_batch_host` opts in explicitly — both frames are covered, and the
# sampled/full-pass numbers are unchanged (the full pass still pins False;
# the sampled path still rebuilds exactly when the trainer flag is on), so
# this is another declared false positive of the same shape as #283's.
#   full_pass  2efe658b4e778870 -> bed3d8e3799e997d
#   sampled    d6f7cabecd8e6f67 -> 610f05cf817b4783
#
# Moved again 2026-08-01 by the always-on SF-label contamination column: a
# single line leaves `_prepare_host_arrays`, which no longer prunes
# `has_sf_multipv_raw` from the H2D payload when `sf_policy_sparse_ce` is off.
# That frame is in BOTH lists, so both ids move.
#
# THIRD declared false positive, and this one is proved rather than argued:
# `has_sf_multipv_raw` is consumed by exactly one loss term,
# `sparse_sf_policy_ce`, which is reached only when `sf_sparse_params is not
# None` (i.e. `sf_policy_sparse_ce` ON — a config in which nothing was pruned
# before either) and which returns an all-zero eligibility mask unless
# `sf_multipv_raw` is ALSO present, and that block is still pruned. Pinned by
# `tests/test_sf_no_multipv_metric.py::
# test_adding_the_presence_flag_does_not_move_the_loss`, which runs
# `compute_loss` with and without the vector and requires every scalar in
# `total` to be bitwise equal. The measurement did not move; the source did.
#   full_pass  bed3d8e3799e997d -> b8482e83d3b1c61f
#   sampled    610f05cf817b4783 -> 71ac6f0457876d02
#
# Moved again 2026-08-02 by the per-phase loss split (backlog #124). It used to
# bucket on `moves_left` = plies-remaining / `max_plies` -- the CAP, 450, not
# the game's length -- which put 96.37% of the live window in `end` and made
# `wdl_loss_open` / `_mid` a long-games-only subsample wearing a phase name. It
# now buckets by PIECE COUNT on `eval/audit.py`'s own constant, and the columns
# are renamed accordingly. `compute_loss` and `_build_metrics` both change, so
# both ids move.
#
# FOURTH declared false positive, proved the same way as the third rather than
# argued: `batch["x"]` feeds exactly one thing inside `compute_loss` -- this
# split -- so rewriting the piece planes without touching `outputs` isolates it
# completely. `tests/test_phase_loss_buckets.py::
# test_the_phase_split_cannot_perturb_the_trained_loss` performs that
# intervention and requires that EVERY scalar which moves has `phase_` in its
# name and that `total` is bitwise equal. The measurement did not move; the
# source did, and the reported column names did.
#   full_pass  b8482e83d3b1c61f -> 3a336231d9b5fce5
#   sampled    71ac6f0457876d02 -> 9f9c078dd590db13
#
# Moved again 2026-08-03 by the F11 policy-index-LUT swap (play-path audit
# 2026-08-03; this PR). `Trainer._policy_accuracy_stats` had a module-private
# `lru_cache` over COMPACT_TO_FULL_POLICY / FULL_TO_COMPACT_POLICY -- the
# duplicate `moves/torch_maps.py` exists to prevent (CLAUDE.md: "don't add
# per-module `lru_cache` copies"), and strictly worse, because it keyed on
# `target.device.index` raw and so allocated two copies of both tables for
# `torch.device("cuda")` vs `("cuda", 0)`. `_align_index` -- a closure inside
# `_policy_accuracy_stats`, one of the frames `_compute_metrics` reaches, and
# `digest_source` hashes SOURCE -- now calls
# `torch_maps.policy_index_remap_table(source_width, dst_width, device)` once
# instead of carrying its own width->table dispatch, so both ids move.
#
# FIFTH declared false positive, proved rather than argued -- and proved of the
# RIGHT property, which the first version of this block was not. Review of #318
# showed the round-trip test could NOT catch the two directions being swapped:
# transposing the two branch bodies in `_align_index` left both symbol names
# present and every value-identity test passed; only this pin noticed, and a
# source hash is a tripwire, not a semantic control. So the duplicated dispatch
# was deleted rather than re-worded. `tests/test_trainer_policy_index_lut.py`
# now (a) reconstructs the deleted helper's exact body and requires dtype,
# device, shape and ELEMENT-WISE equality against the shared tables, (b) runs
# BOTH the pre- and post-refactor `_align_index` bodies over in-range, negative
# and out-of-range ids in both directions and requires the mapped indices AND
# the validity masks to agree, and (c) asserts the width-pair -> table binding
# by BEHAVIOUR, so swapping the two returns inside `policy_index_remap_table`
# fails. Same `torch.long`, same source arrays, same values -- the measurement
# cannot have moved, only the source hash. Records stay comparable across the
# handover.
#   full_pass  3a336231d9b5fce5 -> 025b6ef8e537ffcb
#   sampled    9f9c078dd590db13 -> 408bcad98fb150b4
#
# ⚑ OPERATOR-VISIBLE: `holdout_generation` bumps at the deploying restart, so
# the running trial HANDS OVER its best-model record once, adopting the current
# loss instead of comparing to it. Expected, and recorded in the ledger entry.
# RESTART-GATED, but NOT because the id is read once at construction -- it is
# recomputed on EVERY evaluation (`trainer.py`, in `_compute_metrics`). The
# gate is that a running process holds the old module source, so `digest_source`
# keeps hashing the old code until the process restarts onto this checkout.
# Moved again 2026-08-03 by the VALUE-half SF-label detector (PR #326): three
# lines enter `_prepare_host_arrays` to derive `sf_eval_pv_orphan` /
# `sf_eval_pv_checked` from `sf_multipv_raw` + `sf_label_meta` BEFORE the H2D
# payload prune drops the first and while the second is still in `arrs` at all.
# That frame is in BOTH lists, so both ids move.
#   full_pass  025b6ef8e537ffcb -> 44755941fd0bf0c8
#   sampled    408bcad98fb150b4 -> bbae91591858d35c
#
# SIXTH declared false positive, and MEASURED rather than argued. The two new
# arrays are read by exactly one consumer, `losses.sf_eval_pv_orphan_counts`,
# whose two sums leave `compute_loss` as `sf_eval_pv_orphan_rows` /
# `sf_eval_pv_checked_rows` and are mapped by `_RATIO_METRIC_FIELDS` to the
# observation fields `sf_eval_pv_orphan_frac` / `sf_eval_pv_checked_frac`.
# Nothing reaches `total`, so nothing reaches `loss` or `test_loss`. The
# control that proves it: a production full pass over `shard_033130.zarr` run
# twice, once as shipped and once with the three lines patched out (i.e.
# `main`'s behaviour), comparing all 87 scalar `TrainMetrics` fields --
# `loss` bit-identical at 7.238113220214844, and the ONLY field that differs is
# `sf_eval_pv_checked_frac` (0.0 -> 0.9975), which is the new instrument coming
# alive. Records stay comparable across the handover.
# Moved again 2026-08-07 by the terminal-proximal outcome share of the WDL
# target: `_loss_kwargs` is IN the closure (a property, covered deliberately --
# the dict it builds decides what `compute_loss` is even given) and it gains
# three entries, `wdl_terminal_outcome_plies` / `_full_plies` / `_sf_frac` plus
# `moves_left_max_plies`. That frame is in BOTH lists, so both ids move.
#   full_pass  44755941fd0bf0c8 -> 78aaaf430abf66f1
#   sampled    bbae91591858d35c -> afbf4cc1de454249
#
# SEVENTH declared false positive, MEASURED the same way as the sixth. The
# feature is DEFAULT OFF (`wdl_terminal_outcome_plies: 0`, in no config), and
# off it takes the untouched blend expressions verbatim. The control: the same
# 96-row synthetic batch (all optional fields present, `moves_left` populated,
# sf 0.69 / search 0.31, sf_low dampening 0.35) through `origin/main`'s
# `compute_loss` and through this branch's -- all 50 returned scalars BITWISE
# equal, `total` 16.45402717590332 both sides, no key dropped, the only
# difference being two new always-zero instrument keys
# (`wdl_terminal_outcome_weight_sum` / `_rows`) which reach only the
# observation fields `wdl_terminal_outcome_frac` / `_rows` and never `total`.
# The control can fail: the same batch with the knob at the planned production
# 7/2 moves `wdl_ce`, `blended_wdl_ce`, `total` and the wdl phase splits.
# `tests/test_wdl_terminal_outcome.py` pins the off-path equality against a
# verbatim transcription of `main`'s blend so it cannot drift back unnoticed.
# Records stay comparable across the handover.
# EIGHTH declared false positive, MEASURED — and this is a MERGE of two
# independent movements of the same pin, PR #373 (this branch) and PR #375,
# which branched from the same base (`9f551e8c4`) and therefore each computed
# an id on a tree that did not contain the other. Neither input id describes
# the merged code, so BOTH are discarded and the pair below is recomputed on
# the merged tree.
#
# 8a. From #375: `_build_metrics` is in both closures and gained a `_den`
#     helper plus two kwargs (`policy_own_acc_rows`, `policy_future_acc_rows`).
#     Control: `origin/main`'s `trainer.py` loaded as a second module and its
#     `_build_metrics` called with byte-identical `sums`, `acc_sums` and `n`
#     beside this branch's -- all 98 fields main's `TrainMetrics` declares came
#     back BITWISE EQUAL; the branch adds exactly two, both pure denominators.
#     `_den` reads `acc_sums[name][1]`, the same tensor `_acc` already divides
#     by, so it cannot perturb a numerator and no loss field is reachable.
#
# 8b. From #373: `_compute_metrics` is in both closures and its source changed
#     -- it now splats `_eval_loss_kwargs`
#     (`{**_loss_kwargs, "policy_target_temp": 1.0}`) instead of `_loss_kwargs`.
#     That pin is the reason the id can be declared safe. CE's floor IS the
#     target's entropy, so retempering the target raises `policy_ce` for an
#     UNCHANGED model; letting `policy_target_temp` reach the holdout would make
#     the frozen ruler move with the arm under test. Because eval always passes
#     1.0 and `main` passes no key at all (`compute_loss` defaults to 1.0), the
#     two are the same call for EVERY possible yaml, not merely at the default.
#     Control: `origin/main`'s `losses.py` loaded as a second module inside the
#     real package and its `compute_loss` run beside this branch's on one 96-row
#     synthetic batch (targets supported on the legal mask, all optional fields
#     present) -- all 52 returned scalars BITWISE EQUAL, `total`
#     15.71805477142334 both sides, no key on one side only. The control can
#     fail: the same batch at `policy_target_temp=1.3` moves exactly seven keys
#     (`policy_ce`, the three `policy_loss_phase_*` splits, the two
#     `policy_loss_{selfplay,curriculum}` splits, and `total`), which is also
#     the direct evidence that the knob does something unpinned. RE-RUN against
#     `main` after #375 and #376 merged -- a control whose baseline has moved is
#     not a control -- and the scalar count is unchanged at 52 because neither
#     of those PRs touches `losses.py`.
#
# Both movements are source-hash-only in opposite halves of the same closure,
# so the merged ids differ from every id on either input branch:
#   full_pass  78aaaf430abf66f1 -> (#373) a76f440cdb36b72f
#                               -> (#375) 079f56e31e6b9501
#                               -> MERGED 73ff47d368fbe10e
#   sampled    afbf4cc1de454249 -> (#373) 20696c6766732998
#                               -> (#375) 4ca17596e4a59a90
#                               -> MERGED f41625e40b98e987
# ⚑ The sampled id is computed at steps=5, the pre-PR-277 ruler's own budget,
# which is `sampled`'s own default -- NOT at steps=0. steps is hashed into the
# id, so a steps=0 sampled pin is simply a different measurement; a previous PR
# pinned one by accident (9f80a4cda2da0069, that tree's steps=0 value). On this
# tree `sampled(steps=0)` is e0f28fe544f1cbac, and the pin below is not it.
# Nothing the best-model comparison reads (`test_loss` and the per-head losses)
# changed on either side, so records stay comparable across the handover.
# ⚑ MOVED AGAIN by the `sf_policy_floor` term (this PR): `compute_loss` and
# `Trainer._loss_kwargs` are both covered frames, and the term adds a parameter,
# a computation and two reported scalars to the first plus one key to the second.
#
# THE MEASUREMENT IS SOURCE-HASH-ONLY, AND HERE THAT CLAIM IS EXECUTED RATHER
# THAN ARGUED: at the shipped default `w_sf_policy_floor: 0.0` the term is not
# added to `total` at all (an `if`, not a `* 0.0`), and
# `tests/test_sf_policy_floor.py::test_total_is_bit_identical_at_the_default_weight`
# asserts BIT equality of `total` against a call with the term's parameter
# absent. So `test_loss` and every per-head loss the best-model comparison reads
# are numerically unchanged and records stay comparable across the handover --
# but the id moved, so an operator WILL see a `holdout_generation` bump and a
# best-model handover at the next restart. That is the mechanism working.
#   full_pass  73ff47d368fbe10e -> 30cfec0c574e03da
#   sampled    f41625e40b98e987 -> 733b900bc45f524f
#
# ⚑ MOVED AGAIN by the N1 fix in the same PR: the descriptor gains an
# `active_terms` field (the SET of non-zero loss weights), so the id moves even
# with every weight at zero, because the field itself is new. This is the NINTH
# declared false positive and it is the cheapest to argue: the field is
# ADDITIVE and `_compute_metrics` gained one call
# (`loss_weights=self._ruler_loss_weights()`), which reads attributes and
# performs no arithmetic on the batch. Nothing the best-model comparison reads
# changed. The pins below are taken at `NO_TERMS` -- see the note beside that
# constant for why the golden values are deliberately weight-free.
#   full_pass  30cfec0c574e03da -> ba74408d1e9e98fa
#   sampled    733b900bc45f524f -> cceec01bb2efc6d9
# ⚑ RE-PINNED 2026-08-18 (PR #448, Codex F3). `_compute_metrics` now passes
# `loss_shape=` to `eval_ruler_id_for`, so that frame's source digest moved
# and every id moved with it. This is the documented contract, not a
# regression: deploying it fires ONE best-model handover per running trial.
# Previous value: "v1:full_pass:ba74408d1e9e98fa".
PRODUCTION_FULL_PASS_RULER = "v1:full_pass:441d302e9fdeb0fa"
  # ⚑ RENAMED (review #2, N4). This was `PRODUCTION_SAMPLED_RULER`, and that
  # name was false: production NEVER runs the sampled ruler. `trainable_phases`
  # calls `eval_full_pass` (and the async path with `full_pass=True`), and
  # `eval_steps` appears only in scripts/offline_replay_epoch.py. `steps=5` is
  # `sampled`'s OWN default, not a production budget -- and the name asserting
  # otherwise is part of why a previous PR pinned this at steps=0 and called it
  # production. It is still worth pinning: it is the pre-PR-277 instrument, and
  # it hashes a slightly different frame set (it covers
  # `_iter_prefetched_batches` where full_pass covers `_iter_full_pass_batches`),
  # so it catches drift in a frame the production pin cannot see.
# Re-pinned with the above; previous value "v1:sampled:cceec01bb2efc6d9".
PRE_PR277_SAMPLED_RULER = "v1:sampled:43ca561fa385763f"


def test_the_production_ruler_id_is_pinned() -> None:
    """A golden value, and the maintenance contract is the point of it.

    Every other test here checks STRUCTURE -- that a frame is covered, that a
    covered frame reaches the id. Structure cannot notice that the id MOVED,
    which is how the second boundary failed review: `test_loss` was doubled by
    a one-line edit to `_build_metrics` and all 22 tests stayed green.

    So this pins the value. It is expected to fail whenever the measurement
    changes, and that failure is the mechanism working:

        the id moved => `holdout_generation` will bump at the next restart
        => the running trial HANDS OVER its best-model record, adopting the
           current loss instead of comparing to it.

    If you meant to change the measurement: update both constants and add a
    line to the ledger entry, because an operator is going to see a handover.
    If you did not: you have just changed what `test_loss` means, and the
    holdout is a frozen ruler.

    The value depends on the source of the 20 covered frames (21 for the
    sampled ruler; this branch adds `_eval_loss_kwargs`) and NOT on the
    interpreter: `digest_source` is tokenize-based precisely so that a Python
    upgrade cannot move it. Verified equal on CPython 3.10.12, 3.11.14 and
    3.12.12 -- the earlier `ast.unparse` version disagreed on 9 of the 19
    frames between 3.10 and 3.11, which is what made CI red and would have
    fired a handover on a production interpreter bump.
    """
    assert Trainer.eval_ruler_id_for(
        batch_size=512, steps=0, mirror_prob=0.0, full_pass=True,
        loss_weights=NO_TERMS,
    ) == PRODUCTION_FULL_PASS_RULER
    assert Trainer.eval_ruler_id_for(
        batch_size=512, steps=5, mirror_prob=0.0, full_pass=False,
        loss_weights=NO_TERMS,
    ) == PRE_PR277_SAMPLED_RULER


# --- N1: a live weight flip must move the ruler ---------------------------
#
# ⚑⚑ THE DEFECT. Flipping a loss weight live changed WHAT `test_loss` measures
# and did NOT change the ruler id, so `_update_best_model` took its "same ruler
# -> ordinary improvement test" branch and compared a number produced under the
# new objective against a record produced under the old one. `compute_loss`
# adds these terms under an `if` rather than multiplying by the weight, so a
# zero weight means the term is NOT IN `total`; switch on a term that is
# non-negative by construction (`sf_policy_floor` is a sum of `relu`,
# `sf_own_regret` is `sum p*r`) and `test_loss` steps strictly UP with no route
# back down. The best-model record then freezes at the pre-flip checkpoint for
# the rest of the run.
#
# It was PRE-EXISTING, not new: `w_sf_own_regret` has been in
# `TRAINER_WEIGHT_KEYS` throughout and carries the identical hazard.
#
# The tests below assert on BEHAVIOUR -- that ids differ across a flip and hold
# across a no-op -- rather than re-deriving an expected id from the same helper
# the production code calls, which would agree with an implementation that had
# the rule wrong in both places.


class _TinyModel(torch.nn.Module):
    """Smallest model a `Trainer` will build an optimizer over.

    Same shape as `tests/test_sf_policy_floor.py`'s and
    `tests/test_wdl_terminal_outcome.py`'s, which is the in-repo template for a
    config -> Trainer wiring test.
    """

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32),
        }


def _trainer(overrides: dict[str, Any], tmp_path: Path) -> Trainer:
    from chess_anti_engine.train.trainer import trainer_kwargs_from_config
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    flat = flatten_run_config_defaults({
        "train": {"device": "cpu", "lr": 1e-3, "no_amp": True, **overrides},
    })
    ctor = trainer_kwargs_from_config(flat, log_dir=tmp_path)
    return Trainer(_TinyModel(), prefetch_batches=False, **ctor)


def _live_ruler(trainer: Trainer) -> str:
    """The full-pass id for the trainer's CURRENT weights.

    Reads `_ruler_loss_weights()` off the instance, which is what the eval path
    does; `test_the_eval_path_hands_the_ruler_the_live_weights` pins that the
    production call site spells it the same way.
    """
    return type(trainer).eval_ruler_id_for(
        batch_size=512, steps=0, mirror_prob=0.0, full_pass=True,
        loss_weights=trainer._ruler_loss_weights(),
    )


def _push(trainer: Trainer, **weights: float) -> None:
    """The REAL per-iteration live-push path, not a `setattr` stand-in."""
    from chess_anti_engine.tune.trainable_config_ops import _apply_lr_gamma_weights

    _apply_lr_gamma_weights(trainer, dict(weights), rescale_current_lr=True)


def test_switching_a_loss_term_on_live_moves_the_ruler_id(tmp_path: Path) -> None:
    """⚑⚑ THE N1 REGRESSION. Without the fix these two ids are identical.

    Same weights, same code, same holdout set -- and after the push `total`
    contains a term it did not contain before. If the id does not move,
    `holdout_generation` does not bump, `_update_best_model` compares across
    two objectives, and (the term being non-negative) the record freezes.
    """
    trainer = _trainer({}, tmp_path)
    assert trainer.w_sf_policy_floor == 0.0, "setup: the term ships OFF"
    before = _live_ruler(trainer)

    _push(trainer, w_sf_policy_floor=0.5)

    assert trainer._loss_kwargs["sf_policy_floor"].w == 0.5, "setup: the push landed"
    assert _live_ruler(trainer) != before


def test_switching_a_loss_term_off_live_also_moves_the_ruler_id(tmp_path: Path) -> None:
    """The same event in the other direction: `total` LOSES a term.

    Asserted separately because a rule keyed on "did any weight become
    non-zero" would pass the test above and miss this one, and dropping a
    non-negative term steps `test_loss` DOWN -- which promotes a model for the
    edit rather than for learning, the PR #277 failure exactly.
    """
    trainer = _trainer({"w_sf_own_regret": 0.7}, tmp_path)
    assert trainer.w_sf_own_regret == 0.7, "setup: the term starts ON"
    before = _live_ruler(trainer)

    _push(trainer, w_sf_own_regret=0.0)

    assert _live_ruler(trainer) != before


def test_a_no_op_weight_push_leaves_the_ruler_id_alone(tmp_path: Path) -> None:
    """The failure mode in the other direction, and it is the expensive one.

    `_apply_lr_gamma_weights` re-pushes every weight EVERY iteration. An id
    that moved on a push rather than on a change would bump
    `holdout_generation` every iteration, `_update_best_model` would take its
    ADOPT branch forever, and the best-model comparison would stop existing.
    """
    trainer = _trainer({"w_sf_own_regret": 0.7, "w_policy": 1.0}, tmp_path)
    before = _live_ruler(trainer)

    _push(trainer, w_sf_own_regret=0.7, w_policy=1.0)

    assert _live_ruler(trainer) == before


def test_the_pid_driven_weight_cannot_move_the_ruler_across_its_whole_band(
    tmp_path: Path,
) -> None:
    """⚑ THE DESIGN RISK, EXECUTED. `sf_wdl_frac` is a loss weight in
    `TRAINER_WEIGHT_KEYS` that the difficulty controller RECOMPUTES EVERY
    ITERATION from the realized `wdl_regret`, so a rule that hashed weight
    VALUES would move the id on essentially every production iteration -- not a
    noisy handover but the end of the best-model comparison.

    Driven here through `_dynamic_sf_wdl_weight` itself, at the production band
    (`sf_wdl_frac: 0.50`, `sf_wdl_frac_floor: 0.45`,
    `sf_wdl_floor_at_regret: 0.10`), so this reads the controller's real output
    range rather than a hand-picked pair of numbers.
    """
    from chess_anti_engine.tune.trainable_metrics import _dynamic_sf_wdl_weight

    trainer = _trainer({}, tmp_path)
    realized = [
        _dynamic_sf_wdl_weight(
            sf_wdl_start=0.50, sf_wdl_floor=0.45, sf_wdl_floor_at_regret=0.10,
            regret_max=0.30, wdl_regret_used=r,
        )
        for r in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45)
    ]
    assert len({v for v in realized if v is not None}) > 1, "setup: the band must vary"

    ids = set()
    for value in realized:
        assert value is not None
        _push(trainer, sf_wdl_frac=float(value))
        ids.add(_live_ruler(trainer))

    assert len(ids) == 1, f"the PID's own band moved the ruler id: {ids}"


def test_a_magnitude_change_on_an_active_term_does_not_move_the_ruler_id(
    tmp_path: Path,
) -> None:
    """The other half of the rule, stated as a property rather than as a band.

    Doubling a weight REWEIGHTS the objective; it does not change which terms
    it contains, every per-term column stays comparable, and `total` moves by a
    bounded signed amount. That is the definitional drift
    docs/rl_loop_audit.md L15 already accepts. Membership is the unbounded,
    one-way event, and it is the one the id keys on.
    """
    trainer = _trainer({"w_categorical": 0.30}, tmp_path)
    before = _live_ruler(trainer)

    _push(trainer, w_categorical=0.60)

    assert trainer.w_categorical == 0.60, "setup: the push landed"
    assert _live_ruler(trainer) == before


@pytest.mark.parametrize("key", TRAINER_WEIGHT_KEYS)
def test_every_live_pushable_weight_reaches_the_ruler_id(key: str) -> None:
    """⚑ THE ANTI-HAND-LIST TEST, and the reason the covered set is DERIVED.

    Parametrized over `TRAINER_WEIGHT_KEYS` itself -- the one source of truth
    for "which attribute can a live yaml edit move" -- so an implementation
    that hashed a hand-copied subset fails here on precisely the key it
    omitted, with that key's name in the test id. Two hand-written lists have
    already been wrong in this module's history; this makes a third impossible
    to add silently.
    """
    flipped = dict(NO_TERMS)
    flipped[key] = 1.0

    assert full_pass(loss_weights=flipped) != full_pass(loss_weights=NO_TERMS), (
        f"{key} is live-pushable but does not reach the holdout ruler id"
    )


def test_every_live_pushable_weight_is_readable_off_a_real_trainer(
    tmp_path: Path,
) -> None:
    """`_ruler_loss_weights` reads by plain `getattr`, on the eval path.

    A key in `TRAINER_WEIGHT_KEYS` that is not a Trainer attribute would raise
    there -- inside the holdout evaluation, every iteration. The alternative
    (a defaulted `getattr`) would silently drop a renamed weight OUT of the
    ruler, which is the defect class this whole fix is about, so the read stays
    strict and the coverage is asserted here instead.
    """
    weights = _trainer({}, tmp_path)._ruler_loss_weights()

    assert set(weights) == set(TRAINER_WEIGHT_KEYS)
    assert all(isinstance(v, float) for v in weights.values())


def test_the_eval_path_stamps_the_live_weights_onto_the_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ EXECUTED, not grepped. The whole fix is inert unless the eval path
    hands the classmethod the LIVE map -- it is a classmethod and cannot reach
    the instance itself, so a constant `{}` there would pass every test that
    only calls `eval_ruler_id_for` directly.

    `_compute_metrics` computes the ruler BEFORE it iterates batches, so an
    empty buffer runs the real call site (line for line, including
    `self._ruler_loss_weights()`) and stops short of the pooling tail. Only
    that tail is stubbed, and only to reach the `eval_ruler` it is handed.

    Read per evaluation, not captured at construction: the second half asserts
    that a push AFTER the first eval moves the stamped id.
    """
    seen: list[str] = []

    def _capture(*_a: Any, **kw: Any) -> Any:
        seen.append(str(kw["eval_ruler"]))
        return None

    class _EmptyBuf:
        def batch_row_bounds(self, _n: int) -> list[tuple[int, int]]:
            return []

        def __len__(self) -> int:
            return 0

    trainer = _trainer({}, tmp_path)
    monkeypatch.setattr(Trainer, "_build_metrics", staticmethod(_capture))
    monkeypatch.setattr(Trainer, "_log_metrics", lambda *_a, **_k: None)

    trainer._compute_metrics(
        buf=_EmptyBuf(),  # pyright: ignore[reportArgumentType]
        batch_size=8, steps=0, tag="test", full_pass=True,
    )
    _push(trainer, w_sf_policy_floor=0.5)
    trainer._compute_metrics(
        buf=_EmptyBuf(),  # pyright: ignore[reportArgumentType]
        batch_size=8, steps=0, tag="test", full_pass=True,
    )

    assert len(seen) == 2
    assert all(s.startswith("v1:full_pass:") for s in seen)
    assert seen[0] != seen[1], (
        "the eval path is not reading the live weights -- a constant or a "
        "construction-time snapshot would produce the same id twice"
    )


def test_a_zero_weight_and_an_absent_key_are_the_same_ruler() -> None:
    """A term that is off is a term that is not there. Both are 'not in total'.

    Without this, adding a key to `TRAINER_WEIGHT_KEYS` at its inert default
    would move the ruler and hand the best model over for a no-op.
    """
    assert active_loss_terms(NO_TERMS) == ()
    assert full_pass(loss_weights={}) == full_pass(loss_weights=NO_TERMS)


def test_the_active_set_is_order_and_type_insensitive() -> None:
    """The id must key on the SET, not on dict order or on int-vs-float.

    A yaml writes `w_policy: 1` and a PB2 perturbation writes `1.0`; a reload
    can reorder the dict. Neither is a new objective, and either moving the id
    would fire the handover on nothing.
    """
    a = {"w_policy": 1.0, "w_wdl": 0.0, "w_soft": 2.0}
    b = {"w_soft": 2, "w_policy": 1, "w_wdl": 0}

    assert active_loss_terms(a) == active_loss_terms(b) == ("w_policy", "w_soft")


def test_an_unreadable_weight_counts_as_active_rather_than_vanishing() -> None:
    """The declared direction of error, applied to the weight half too.

    A false positive costs one best-model handover; a false negative is a
    promotion across two objectives. So a value that will not convert to a
    float is treated as a term that is present, and never silently dropped.
    """
    assert active_loss_terms({"w_policy": None}) == ("w_policy",)  # pyright: ignore[reportArgumentType]
    assert active_loss_terms({"w_policy": float("nan")}) == ("w_policy",)


# --------------------------------------------------------------------------
# Codex review of PR #448, findings F3 and F4: two ways the ruler could hold
# still across an objective change. Both are the repo's signature defect --
# a value accepted and then silently ignored -- arriving at the ruler rather
# than at the trainer.
# --------------------------------------------------------------------------


def test_negative_weight_is_off_when_the_consumer_clamps_it() -> None:
    """A key its consumer clamps at 0 must leave the term set when it goes negative.

    `compute_loss` does `max(0.0, float(sf_wdl_frac))`, so -0.5 removes the
    component from `total`. A `!= 0.0` membership test would call it ACTIVE and
    freeze the ruler across exactly the event the ruler exists to catch.
    """
    assert "sf_wdl_frac" in active_loss_terms({"sf_wdl_frac": 0.5})
    assert "sf_wdl_frac" not in active_loss_terms({"sf_wdl_frac": -0.5})
    assert "search_wdl_frac" not in active_loss_terms({"search_wdl_frac": -1e-9})
    assert "soft_policy_min_tv" not in active_loss_terms({"soft_policy_min_tv": -0.1})
    # and the ruler MOVES on that transition
    on = {**NO_TERMS, "sf_wdl_frac": 0.5}
    off = {**NO_TERMS, "sf_wdl_frac": -0.5}
    assert full_pass(loss_weights=on) != full_pass(loss_weights=off)
    assert full_pass(loss_weights=off) == full_pass(loss_weights=NO_TERMS)


def test_a_plain_multiplier_stays_active_when_negative() -> None:
    """The <=0 rule is per-key, NOT a blanket sign rule.

    `w_policy` is a plain multiplier: at -0.5 the term is still in `total`,
    sign-flipped. Treating every negative weight as "off" would drop a term
    that is present, which is the same defect pointing the other way.
    """
    assert "w_policy" in active_loss_terms({"w_policy": -0.5})
    assert "w_policy" not in active_loss_terms({"w_policy": 0.0})


def test_active_term_shape_moves_the_ruler() -> None:
    """Retuning WHAT an active term measures must hand the record over."""
    shape_a = {"sf_policy_floor_delta_cp": 20.0, "sf_policy_floor_tau": 0.15}
    shape_b = {"sf_policy_floor_delta_cp": 40.0, "sf_policy_floor_tau": 0.15}
    on = {**NO_TERMS, "w_sf_policy_floor": 0.8}
    a = eval_ruler_id(
        mode="full_pass", batch_size=512, steps=0, mirror_prob=0.0,
        measured_by=FULL_PASS_FNS, loss_weights=on, loss_shape=shape_a,
    )
    b = eval_ruler_id(
        mode="full_pass", batch_size=512, steps=0, mirror_prob=0.0,
        measured_by=FULL_PASS_FNS, loss_weights=on, loss_shape=shape_b,
    )
    assert a != b


def test_absent_shape_is_the_pre_shape_identity() -> None:
    """An empty shape must hash identically to passing none at all.

    Callers omit the shape for terms that are OFF, so the two spellings have to
    agree or a disabled term would get its own ruler.
    """
    on = {**NO_TERMS, "w_policy": 1.0}
    none_ = eval_ruler_id(
        mode="full_pass", batch_size=512, steps=0, mirror_prob=0.0,
        measured_by=FULL_PASS_FNS, loss_weights=on,
    )
    empty = eval_ruler_id(
        mode="full_pass", batch_size=512, steps=0, mirror_prob=0.0,
        measured_by=FULL_PASS_FNS, loss_weights=on, loss_shape={},
    )
    assert none_ == empty


def test_weight_magnitude_still_does_not_move_the_ruler() -> None:
    """Regression guard on the DESIGN, not on the fix.

    The shape exception must not leak into weights: `sf_wdl_frac` is
    recomputed every iteration by the difficulty controller, and hashing its
    VALUE would move the id on essentially every production iteration and
    abolish the best-model comparison entirely.
    """
    lo = {**NO_TERMS, "w_policy": 0.3, "sf_wdl_frac": 0.45}
    hi = {**NO_TERMS, "w_policy": 0.6, "sf_wdl_frac": 0.50}
    assert full_pass(loss_weights=lo) == full_pass(loss_weights=hi)


def test_ruler_loss_shape_reads_the_consumers_object_and_only_when_active() -> None:
    """The production wiring, not just the hash.

    `_ruler_loss_shape` is the only thing that turns a resolved
    `SfPolicyFloorParams` into ruler input, so an untested version of it is the
    repo's signature defect exactly: the hash would be correct and nothing
    would ever reach it.
    """
    from types import SimpleNamespace

    from chess_anti_engine.train.losses import SfPolicyFloorParams

    on = SimpleNamespace(_eval_loss_kwargs={
        "sf_policy_floor": SfPolicyFloorParams(w=0.8, delta_cp=20.0, tau=0.15),
    })
    shape = Trainer._ruler_loss_shape(on)  # pyright: ignore[reportArgumentType]
    assert shape["sf_policy_floor_delta_cp"] == 20.0
    assert shape["sf_policy_floor_tau"] == 0.15
    # `w` is covered by MEMBERSHIP; repeating it here would reintroduce the
    # magnitude hash `active_loss_terms` argues against.
    assert "w_sf_policy_floor" not in shape

    # OFF: the term is not in `total`, so its shape cannot move `test_loss`.
    off = SimpleNamespace(_eval_loss_kwargs={
        "sf_policy_floor": SfPolicyFloorParams(w=0.0, delta_cp=999.0),
    })
    assert Trainer._ruler_loss_shape(off) == {}  # pyright: ignore[reportArgumentType]

    # Absent entirely (a trainer built without the term) must not raise.
    assert Trainer._ruler_loss_shape(SimpleNamespace(_eval_loss_kwargs={})) == {}  # pyright: ignore[reportArgumentType]
