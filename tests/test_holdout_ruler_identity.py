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
from types import SimpleNamespace
from typing import Any

from chess_anti_engine.train.eval_ruler import (
    _UNAVAILABLE,
    digest_source,
    eval_ruler_id,
    semantic_source_digest,
)
from chess_anti_engine.train.trainer import TrainMetrics, Trainer
from chess_anti_engine.tune.trainable import _maybe_bump_generation_on_ruler_change
from chess_anti_engine.tune.trainable_init import _apply_restored_holdout_scalars
from chess_anti_engine.tune.trial_config import RestoreResult

FULL_PASS_FNS: tuple[Callable[..., Any], ...] = (
    Trainer._iter_full_pass_batches, Trainer._full_pass_host_batch,
)
SAMPLED_FNS: tuple[Callable[..., Any], ...] = (
    Trainer._iter_prefetched_batches, Trainer._sample_batch_host,
)


def full_pass(
    *, batch_size: int = 512, steps: int = 0,
    batch_fns: Sequence[Callable[..., Any]] = FULL_PASS_FNS,
) -> str:
    """The production ruler: a deterministic pass at the production batch size."""
    return eval_ruler_id(
        mode="full_pass", batch_size=batch_size, steps=steps, mirror_prob=0.0,
        pooling="row_weighted", batch_fns=batch_fns,
    )


def sampled(*, batch_size: int = 512, steps: int = 5) -> str:
    """The pre-PR-277 ruler: `steps` x `batch_size` draws with replacement."""
    return eval_ruler_id(
        mode="sampled", batch_size=batch_size, steps=steps, mirror_prob=0.0,
        pooling="row_weighted", batch_fns=SAMPLED_FNS,
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


def test_the_production_batch_functions_have_readable_source() -> None:
    """The digest degrades to a constant when `inspect.getsource` fails, and a
    constant cannot detect anything. Pin that the real production functions
    are not silently on that path."""
    for fn in (*FULL_PASS_FNS, *SAMPLED_FNS):
        assert semantic_source_digest(fn) != _UNAVAILABLE

    assert full_pass().startswith("v1:full_pass:")


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
    trainer = SimpleNamespace(
        _iter_full_pass_batches=Trainer._iter_full_pass_batches,
        _full_pass_host_batch=Trainer._full_pass_host_batch,
        _iter_prefetched_batches=Trainer._iter_prefetched_batches,
        _sample_batch_host=Trainer._sample_batch_host,
    )
    ruler = Trainer._eval_ruler_id.__get__(trainer)
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

    unchanged = full_pass(batch_fns=(_stub,))

    assert unchanged != full_pass(), (
        "the batch functions' digests must reach the id"
    )


def test_an_unreadable_source_degrades_instead_of_raising() -> None:
    """A frozen install or an exec'd function has no source. The eval path must
    still produce an id -- the descriptor alone -- rather than raise."""
  # No file backs this one, which is what makes it unreadable to `getsource`.
    fn = eval(compile("lambda: None", "<nofile>", "eval"))

    assert semantic_source_digest(fn) == _UNAVAILABLE
    assert full_pass(batch_fns=(fn,)).startswith("v1:full_pass:")


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

    assert "ruler = self._eval_ruler_id(" in body
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
