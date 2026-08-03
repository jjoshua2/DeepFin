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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from chess_anti_engine.train.eval_ruler import (
    _UNAVAILABLE,
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


def full_pass(
    *, batch_size: int = 512, steps: int = 0,
    measured_by: Sequence[Callable[..., Any]] = FULL_PASS_FNS,
) -> str:
    """The production ruler: a deterministic pass at the production batch size."""
    return eval_ruler_id(
        mode="full_pass", batch_size=batch_size, steps=steps, mirror_prob=0.0,
        measured_by=measured_by,
    )


def sampled(*, batch_size: int = 512, steps: int = 5) -> str:
    """The pre-PR-277 ruler: `steps` x `batch_size` draws with replacement."""
    return eval_ruler_id(
        mode="sampled", batch_size=batch_size, steps=steps, mirror_prob=0.0,
        measured_by=SAMPLED_FNS,
    )


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
    ruler = Trainer.eval_ruler_id_for

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
    ruler = Trainer.eval_ruler_id_for
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
#
# Moved again 2026-08-03 by the F11 policy-index-LUT swap (play-path audit
# 2026-08-03; this PR). `Trainer._policy_accuracy_stats` had a module-private
# `lru_cache` over COMPACT_TO_FULL_POLICY / FULL_TO_COMPACT_POLICY -- the
# duplicate `moves/torch_maps.py` exists to prevent (CLAUDE.md: "don't add
# per-module `lru_cache` copies"), and strictly worse, because it keyed on
# `target.device.index` raw and so allocated two copies of both tables for
# `torch.device("cuda")` vs `("cuda", 0)`. The two `lut = ...` lines now call
# `torch_maps.full_to_compact_index` / `compact_to_full_index`. Those lines sit
# in `_align_index`, a closure inside `_policy_accuracy_stats`, which is one of
# the frames `_compute_metrics` reaches -- and `digest_source` hashes SOURCE --
# so both ids move.
#
# FIFTH declared false positive, proved rather than argued:
# `tests/test_trainer_policy_index_lut.py` reconstructs the deleted helper's
# exact body and requires dtype, device, shape and ELEMENT-WISE equality against
# the shared tables, plus a round-trip over every real move (which would catch
# the two directions having been swapped). Same `torch.long`, same source
# arrays, same values -- the measurement cannot have moved, only the source hash.
# Records stay comparable across the handover.
#   full_pass  3a336231d9b5fce5 -> 104bee0152a72a68
#   sampled    9f9c078dd590db13 -> 159e5e349a229400
#
# ⚑ OPERATOR-VISIBLE: `holdout_generation` bumps at the deploying restart, so
# the running trial HANDS OVER its best-model record once, adopting the current
# loss instead of comparing to it. Expected, and recorded in the ledger entry.
# RESTART-GATED: merging this PR changes nothing until the run restarts onto it.
PRODUCTION_FULL_PASS_RULER = "v1:full_pass:104bee0152a72a68"
PRODUCTION_SAMPLED_RULER = "v1:sampled:159e5e349a229400"


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

    The value depends on the source of the 19 covered frames and NOT on the
    interpreter: `digest_source` is tokenize-based precisely so that a Python
    upgrade cannot move it. Verified equal on CPython 3.10.12, 3.11.14 and
    3.12.12 -- the earlier `ast.unparse` version disagreed on 9 of the 19
    frames between 3.10 and 3.11, which is what made CI red and would have
    fired a handover on a production interpreter bump.
    """
    assert Trainer.eval_ruler_id_for(
        batch_size=512, steps=0, mirror_prob=0.0, full_pass=True,
    ) == PRODUCTION_FULL_PASS_RULER
    assert Trainer.eval_ruler_id_for(
        batch_size=512, steps=5, mirror_prob=0.0, full_pass=False,
    ) == PRODUCTION_SAMPLED_RULER
