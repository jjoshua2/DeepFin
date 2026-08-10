"""The holdout eval must be a PASS over the frozen set, not a resample of it.

Freezing the SET did not freeze the MEASUREMENT (docs/rl_loop_audit.md G14).
`test_wdl_loss` came from ``eval_steps(holdout_buf, batch_size=512, steps=5)``
-- 2560 draws WITH REPLACEMENT from a set capped at 2000 rows -- and
``sample_batch_arrays`` additionally WDL-rebalanced every batch away from the
holdout's own class mix (`draw_cap_frac` 0.90, `wl_max_ratio` 1.5) and drew
half of every batch proportional to `priority`, because `surprise_mix` is
never assigned on the holdout buffer and keeps the class default 0.5.

Measured consequence on the live frozen set (iters 48-61): effective sample
size 1051 of 2000 rows (52.6%), `test_loss` noise floor sd 0.0522 nats, mean
|delta| 0.0708 -- with the composition of the set held constant. A live
pre-registered kill rule read "+>0.026 (4 sigma) over 3 consecutive
iterations"; 0.026 is 0.5 sigma against that floor, so the rule fired on
resampling noise or not at all, and had to be withdrawn.

These tests pin the five properties the fix has to hold, each written so that
removing the behaviour it names makes THIS test fail:

  determinism  -- same weights + same rows twice => bit-identical loss;
  coverage     -- every row scored exactly once, the ragged tail included;
  weighting    -- the reported mean is the ROW-weighted one, not the mean of
                  per-batch means (2000 rows at 512 is 3 x 512 + 464, and the
                  unweighted estimator counts each tail row 1.10x);
  class mix    -- the realized W/D/L histogram is the holdout's, not a
                  rebalanced one;
  wiring       -- BOTH the sync and the async holdout paths take it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from chess_anti_engine.replay.buffer import ArrayReplayBuffer
from chess_anti_engine.train import trainer as trainer_mod
from chess_anti_engine.train.trainer import Trainer

PLANES = 4
POLICY = 1858  # COMPACT_POLICY_SIZE; the shard validator accepts only 4672 or 1858
ROWS = 2000
BATCH = 512
# 2000 rows at batch 512: three full batches and a ragged 464-row tail.
EXPECTED_FULL_PASS_BATCHES = 4
# What production ran before this change: test_steps=5 sampled draws.
LEGACY_SAMPLED_BATCHES = 5


class _MarkedNet(nn.Module):
    """Deterministic tiny net whose outputs are a steep function of the marker.

    The per-row marker lives in ``x[:, 0, 0, 0]`` and rises monotonically with
    the row index (see ``_arrays``), so a loss computed over a batch is a
    function of exactly which rows are in it -- which is what makes "the same
    rows" and "a resample of the same rows" distinguishable at all. The weights
    are SET rather than initialized so the per-row loss spread is large and
    fixed: without a real gradient across the buffer, the ragged tail's mean
    barely differs from the whole set's and the row-weighted and batch-weighted
    estimators agree to 1e-4, i.e. the weighting test would prove nothing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.policy = nn.Linear(PLANES * 64, POLICY)
        self.wdl = nn.Linear(PLANES * 64, 3)
        with torch.no_grad():
            for layer in (self.policy, self.wdl):
                layer.weight.zero_()
                layer.bias.zero_()
  # h[0] is x[:, 0, 0, 0] after the flatten: the row marker.
            self.wdl.weight[0, 0] = 6.0
            self.policy.weight[0, 0] = 3.0

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = x.flatten(1)
        return {"policy": self.policy(h), "wdl": self.wdl(h)}


def _arrays(n: int, *, first_row: int = 0) -> dict[str, np.ndarray]:
    """n rows whose identity and class are readable back out of the batch.

    `wdl_target` is deliberately LOPSIDED -- 80% draws, 19% wins, 1% losses --
    so a WDL rebalance cannot be mistaken for the set's own mix. `priority`
    rises steeply with the row index so priority-proportional sampling is
    likewise visible: it would pull the tail of the buffer and starve the head.
    """
    rows = np.arange(first_row, first_row + n, dtype=np.int64)
    pol = np.zeros((n, POLICY), dtype=np.float32)
    pol[np.arange(n), rows % POLICY] = 1.0
    x = np.zeros((n, PLANES, 8, 8), dtype=np.float32)
  # Normalized by ROWS, not by n, so a slice of the set carries the same
  # marker values it carries inside the whole set.
    x[:, 0, 0, 0] = rows.astype(np.float32) / float(ROWS)
    x[:, 1, 0, 0] = 1.0
  # 0=win, 1=draw, 2=loss -> 380 / 1600 / 20 over 2000 rows (19/80/1%).
    wdl = np.where(rows % 100 == 0, 2, np.where(rows % 5 == 0, 0, 1)).astype(np.int8)
    return {
        "x": x,
        "policy_target": pol,
        "wdl_target": wdl,
        "priority": (rows.astype(np.float32) + 1.0) ** 2,
        "has_policy": np.ones((n,), dtype=np.uint8),
    }


def _buffer(rows: int = ROWS) -> ArrayReplayBuffer:
    buf = ArrayReplayBuffer(rows, rng=np.random.default_rng(0))
  # Two chunks so the pass has to cross a chunk boundary, which is where a
  # naive index walk over `_chunks` would silently drop or repeat rows.
    half = rows // 2
    buf.add_many_arrays(_arrays(half))
    buf.add_many_arrays(_arrays(rows - half, first_row=half))
    return buf


def _trainer(tmp_path: Path, **kw: Any) -> Trainer:
    torch.manual_seed(1234)
    trainer = Trainer(
        _MarkedNet(),
        device="cpu",
        lr=1e-3,
        optimizer="adamw",
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=10_000,
        mirror_prob=0.5,  # eval must pin this to 0 regardless
        **kw,
    )
    trainer.model.eval()
    return trainer


class _BatchRecorder:
    """Records what each scored batch actually contained.

    Wraps ``compute_loss`` where the trainer calls it, which is the last point
    at which a batch is still identifiable -- the model only ever sees ``x``.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.markers: list[np.ndarray] = []
        self.wdl: list[np.ndarray] = []
        real = trainer_mod.compute_loss

        def _wrapped(outputs, batch, **kwargs):
            self.markers.append(batch["x"][:, 0, 0, 0].detach().cpu().numpy().copy())
            self.wdl.append(batch["wdl_t"].detach().cpu().numpy().copy())
            return real(outputs, batch, **kwargs)

        monkeypatch.setattr(trainer_mod, "compute_loss", _wrapped)

    @property
    def batch_sizes(self) -> list[int]:
        return [int(m.shape[0]) for m in self.markers]

    @property
    def all_markers(self) -> np.ndarray:
        return np.concatenate(self.markers) if self.markers else np.zeros((0,), dtype=np.float32)

    @property
    def all_wdl(self) -> np.ndarray:
        return np.concatenate(self.wdl) if self.wdl else np.zeros((0,), dtype=np.int64)


def _class_histogram(labels: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(np.count_nonzero(labels == c)) for c in (0, 1, 2))  # pyright: ignore[reportReturnType]


# --- 1. determinism -------------------------------------------------------


def test_the_same_weights_over_the_same_rows_give_a_bit_identical_loss(tmp_path: Path) -> None:
    """The whole point of the PR. Today's sampled eval varies with sd 0.0522."""
    trainer = _trainer(tmp_path)
    buf = _buffer()

    first = trainer.eval_full_pass(buf, batch_size=BATCH)
    second = trainer.eval_full_pass(buf, batch_size=BATCH)

    assert second.wdl_loss == first.wdl_loss
    assert second.loss == first.loss
    assert second.policy_loss == first.policy_loss
    assert second.eval_rows == first.eval_rows == ROWS


def test_the_sampled_eval_it_replaces_is_not_deterministic(tmp_path: Path) -> None:
    """The control. Without this the determinism test could pass on a set the
    sampler happens to reproduce, and would prove nothing about the change."""
    trainer = _trainer(tmp_path)
    buf = _buffer()

    losses = {
        trainer.eval_steps(buf, batch_size=BATCH, steps=LEGACY_SAMPLED_BATCHES).wdl_loss
        for _ in range(5)
    }

    assert len(losses) > 1, "the sampled eval reproduced itself; the fixture cannot detect a resample"


def test_determinism_holds_with_the_prefetch_thread_on(tmp_path: Path) -> None:
    """Production runs `prefetch_batches: true`; the prefetch path reorders
    when work happens and must not reorder which rows land in which batch."""
    serial = _trainer(tmp_path / "serial", prefetch_batches=False)
    threaded = _trainer(tmp_path / "threaded", prefetch_batches=True)
    threaded.model.load_state_dict(serial.model.state_dict())

    assert threaded.eval_full_pass(_buffer(), batch_size=BATCH).wdl_loss == (
        serial.eval_full_pass(_buffer(), batch_size=BATCH).wdl_loss
    )


# --- 2. coverage ----------------------------------------------------------


def test_the_buffer_pass_yields_every_row_exactly_once() -> None:
    buf = _buffer()
    bounds = buf.batch_row_bounds(BATCH)
    seen = np.concatenate(
        [buf.rows_slice_arrays(lo, hi)["x"][:, 0, 0, 0] for lo, hi in bounds]
    )
    every = np.asarray(buf.export_arrays()["x"])[:, 0, 0, 0]

    assert bounds == [(0, 512), (512, 1024), (1024, 1536), (1536, 2000)]

    assert seen.shape[0] == ROWS
    np.testing.assert_array_equal(np.sort(seen), np.sort(every))
  # Order too, not just the multiset: a fixed order is what makes two passes
  # over the same weights reduce in the same float order and so compare equal.
    np.testing.assert_array_equal(seen, every)


def test_the_eval_scores_every_row_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coverage measured where it matters: at the loss, not at the buffer."""
    rec = _BatchRecorder(monkeypatch)
    metrics = _trainer(tmp_path).eval_full_pass(_buffer(), batch_size=BATCH)

    markers = rec.all_markers
    assert markers.shape[0] == ROWS
    assert np.unique(markers).shape[0] == ROWS, "a row was scored twice, or one was skipped"
    assert metrics.eval_rows == ROWS


def test_the_ragged_tail_is_scored_rather_than_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping the tail is the cheap way to keep every batch the same shape.
    It silently changes WHICH rows the ruler covers -- the G14 defect in
    miniature -- so the tail is scored and weighted instead."""
    rec = _BatchRecorder(monkeypatch)
    _trainer(tmp_path).eval_full_pass(_buffer(), batch_size=BATCH)

    assert rec.batch_sizes == [BATCH, BATCH, BATCH, ROWS - 3 * BATCH]
    assert sum(rec.batch_sizes) == ROWS


# --- 3. the ragged-tail denominator --------------------------------------


def _chunk_loss(tmp_path: Path, trainer: Trainer, start: int, stop: int) -> float:
    """`wdl_loss` over rows [start, stop) alone, scored in a single batch."""
    del tmp_path
    solo = ArrayReplayBuffer(stop - start, rng=np.random.default_rng(0))
    solo.add_many_arrays(_arrays(stop - start, first_row=start))
    return float(trainer.eval_full_pass(solo, batch_size=stop - start).wdl_loss)


def test_the_reported_mean_is_row_weighted_not_batch_weighted(tmp_path: Path) -> None:
    """The trap. ``_build_metrics`` divides accumulated loss sums by the batch
    COUNT, which assumes every batch holds ``batch_size`` rows. A full pass
    ends on 464 rows, so an unweighted average of 4 per-batch means counts each
    of those rows 1.10x. Correct estimator: sum(mean_i * n_i) / sum(n_i).

    Asserting only "equals the row-weighted mean" would not be enough -- the
    two estimators have to be shown to DISAGREE here, or the test passes under
    the broken denominator too.
    """
    trainer = _trainer(tmp_path)
    bounds = [(s, min(s + BATCH, ROWS)) for s in range(0, ROWS, BATCH)]
    per_chunk = [_chunk_loss(tmp_path, trainer, lo, hi) for lo, hi in bounds]
    sizes = [hi - lo for lo, hi in bounds]

    row_weighted = sum(m * n for m, n in zip(per_chunk, sizes, strict=True)) / float(ROWS)
    batch_weighted = sum(per_chunk) / float(len(per_chunk))

    reported = float(trainer.eval_full_pass(_buffer(), batch_size=BATCH).wdl_loss)

    assert abs(row_weighted - batch_weighted) > 1e-3, (
        "fixture is not discriminating: the two estimators agree, so this test "
        "would pass against the unweighted denominator as well"
    )
    assert reported == pytest.approx(row_weighted, abs=2e-6)
    assert reported != pytest.approx(batch_weighted, abs=1e-4)


def test_a_full_pass_matches_scoring_the_whole_set_in_one_batch(tmp_path: Path) -> None:
    """Independent check of the same arithmetic: chunking must not change the
    number. One batch of 2000 has no tail to mis-weight."""
    trainer = _trainer(tmp_path)

    chunked = trainer.eval_full_pass(_buffer(), batch_size=BATCH)
    whole = trainer.eval_full_pass(_buffer(), batch_size=ROWS)

    assert chunked.eval_rows == whole.eval_rows == ROWS
    assert float(chunked.wdl_loss) == pytest.approx(float(whole.wdl_loss), abs=2e-6)
    assert float(chunked.policy_loss) == pytest.approx(float(whole.policy_loss), abs=2e-6)


# --- 4. class mix ---------------------------------------------------------


def test_the_realized_class_mix_is_the_holdouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`sample_batch_arrays` rebalances every batch toward its own targets
    (draws capped at 0.90 of the batch, wins/losses within 1.5x of each other).
    On this deliberately lopsided set that moves the mix a long way, and a
    ruler measured on a different class mix than the set it names is not a
    ruler for that set."""
    rec = _BatchRecorder(monkeypatch)
    buf = _buffer()
    holdout_mix = _class_histogram(np.asarray(buf.export_arrays()["wdl_target"]))

    _trainer(tmp_path).eval_full_pass(buf, batch_size=BATCH)

    assert _class_histogram(rec.all_wdl) == holdout_mix
    assert sum(holdout_mix) == ROWS


def test_the_sampled_eval_does_not_preserve_the_class_mix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control for the test above: the rebalance is real and large on this
    fixture, so preserving the mix is a property of the full pass and not an
    accident of the data."""
    rec = _BatchRecorder(monkeypatch)
    buf = _buffer()
    w, d, loss_n = _class_histogram(np.asarray(buf.export_arrays()["wdl_target"]))

    _trainer(tmp_path).eval_steps(buf, batch_size=BATCH, steps=LEGACY_SAMPLED_BATCHES)

    sw, sd, sl = _class_histogram(rec.all_wdl)
    drawn = sw + sd + sl
    assert abs(sl / drawn - loss_n / ROWS) > 0.05, (
        f"holdout mix {(w, d, loss_n)} vs sampled {(sw, sd, sl)}: the rebalance "
        "did not move the loss share, so this fixture cannot detect it"
    )


# --- 5. cost, and the wiring on both holdout paths ------------------------


def test_a_full_pass_costs_four_batches_where_the_resample_cost_five(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production shape: 2000 rows at batch 512. 4 batches, not 5."""
    rec_pass = _BatchRecorder(monkeypatch)
    _trainer(tmp_path).eval_full_pass(_buffer(), batch_size=BATCH)
    assert len(rec_pass.batch_sizes) == EXPECTED_FULL_PASS_BATCHES

    rec_sampled = _BatchRecorder(monkeypatch)
    _trainer(tmp_path).eval_steps(_buffer(), batch_size=BATCH, steps=LEGACY_SAMPLED_BATCHES)
    assert len(rec_sampled.batch_sizes) == LEGACY_SAMPLED_BATCHES


class _RecordingTrainer:
    """Enough of a Trainer for `_run_holdout_evaluation` to drive."""

    def __init__(self) -> None:
        self.full_pass_calls: list[int] = []
        self.sampled_calls: list[int] = []
        self.model = _MarkedNet()

    def eval_full_pass(self, buf, *, batch_size: int):
        del buf
        self.full_pass_calls.append(int(batch_size))
        return "FULL_PASS"

    def eval_steps(self, buf, *, batch_size: int, steps: int):
        del buf, batch_size
        self.sampled_calls.append(int(steps))
        return "SAMPLED"


class _RecordingAsyncEval:
    def __init__(self, *, inflight: bool = True) -> None:
        self.started: list[dict[str, Any]] = []
        self.collects = 0
        self._inflight = bool(inflight)

    def has_inflight(self) -> bool:
        return self._inflight

    def collect(self, timeout: float):
        del timeout
        self.collects += 1
        return "PRIOR", 40

    def start(self, **kwargs: Any) -> None:
        self.started.append(kwargs)


def _tc():
    from chess_anti_engine.tune.trial_config import TrialConfig

    return TrialConfig.from_dict({
        "batch_size": BATCH, "test_steps": LEGACY_SAMPLED_BATCHES,
        "input_extra_features": "v1", "policy_encoding": "az_4672",
    })


def test_the_sync_holdout_path_runs_a_full_pass() -> None:
    from chess_anti_engine.tune.trainable_phases import _run_holdout_evaluation

    trainer = _RecordingTrainer()
    metrics, source_iter = _run_holdout_evaluation(
        trainer=trainer, holdout_buf=_buffer(), tc=_tc(), model_cfg=None,
        device="cpu", config={}, iteration_idx=41, holdout_frozen=True,
    )

    assert trainer.full_pass_calls == [BATCH]
    assert trainer.sampled_calls == [], "the sync holdout path still resamples"
    assert (metrics, source_iter) == ("FULL_PASS", 41)


def test_the_async_holdout_path_runs_a_full_pass() -> None:
    """The async path calls ``_compute_metrics`` directly, so it does not
    inherit anything from ``eval_full_pass``; it has to be told. It was the
    path that had to change second, and the one a partial fix would miss."""
    from chess_anti_engine.tune.trainable_phases import _run_holdout_evaluation

    async_eval = _RecordingAsyncEval()
    metrics, source_iter = _run_holdout_evaluation(
        trainer=_RecordingTrainer(), holdout_buf=_buffer(), tc=_tc(), model_cfg=None,
        device="cpu", config={}, iteration_idx=41, holdout_frozen=True,
        async_test_eval=async_eval,
    )

    assert (metrics, source_iter) == ("PRIOR", 40)
    assert len(async_eval.started) == 1
    assert async_eval.started[0]["full_pass"] is True
    assert async_eval.started[0]["batch_size"] == BATCH
  # The cudagraph-TLS workaround at trainable_phases.py:247 must survive: the
  # eval thread cannot see cudagraph_trees TLS, so reduce-overhead has to be
  # downgraded before the snapshot is compiled on it.
    assert async_eval.started[0]["compile_mode"] == "off"


def test_the_async_path_still_strips_cudagraphs_from_the_eval_thread() -> None:
    """Removing this mapping was tried on 2026-04-29 and reverted after the
    ``_is_key_in_tls`` assertion kept firing on the eval thread."""
    from chess_anti_engine.tune.trainable_phases import _run_holdout_evaluation

    async_eval = _RecordingAsyncEval()
    _run_holdout_evaluation(
        trainer=_RecordingTrainer(), holdout_buf=_buffer(), tc=_tc(), model_cfg=None,
        device="cpu", config={"use_compile": True, "compile_mode": "reduce-overhead"},
        iteration_idx=41, holdout_frozen=True, async_test_eval=async_eval,
    )

    assert async_eval.started[0]["compile_mode"] == "default"


# ---------------------------------------------------------------------------
# audit L2: the async holdout eval must not be handed a buffer the trainer
# thread can still append to.
# ---------------------------------------------------------------------------


def _tc_growable(*, holdout_fraction: float = 0.02):
    from chess_anti_engine.tune.trial_config import TrialConfig

    return TrialConfig.from_dict({
        "batch_size": BATCH, "test_steps": LEGACY_SAMPLED_BATCHES,
        "input_extra_features": "v1", "policy_encoding": "az_4672",
        "holdout_fraction": holdout_fraction, "freeze_holdout_at": 2000,
    })


def test_the_async_eval_is_skipped_while_the_holdout_can_still_grow(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """RED on main, which starts the eval thread on a mutating buffer.

    The fresh-start shape: `holdout_frozen` is False and `holdout_fraction` is
    positive, so the NEXT iteration's `_ingest_train_arrays` will append to the
    very `ArrayReplayBuffer` the eval thread walks. `start()` passes it by
    reference, so the pass either raises `deque mutated during iteration` or --
    worse, because it is silent -- reads a short, index-shifted slice, since
    `_iter_full_pass_batches` snapshots its bounds once and `rows_slice_arrays`
    clamps.
    """
    from chess_anti_engine.tune.trainable_phases import _run_holdout_evaluation

    async_eval = _RecordingAsyncEval()
    metrics, source_iter = _run_holdout_evaluation(
        trainer=_RecordingTrainer(), holdout_buf=_buffer(), tc=_tc_growable(),
        model_cfg=None, device="cpu", config={}, iteration_idx=41,
        holdout_frozen=False, async_test_eval=async_eval,
    )

    assert async_eval.started == [], (
        "a mutating holdout was handed to the eval thread (audit L2)"
    )
  # The already-in-flight result is still collected and still returned: only
  # the row whose eval was never STARTED loses its columns.
    assert (metrics, source_iter) == ("PRIOR", 40)
  # ...and the absence is explainable from stdout, which is what train.sh
  # captures. An unexplained missing test_* column is the failure this log
  # line exists to prevent.
    out = capsys.readouterr().out
    assert "SKIPPING the async holdout eval" in out
    assert "audit L2" in out


def test_the_async_eval_runs_once_the_holdout_is_frozen() -> None:
    """The other half of the red/green pair: the guard is not a blanket off."""
    from chess_anti_engine.tune.trainable_phases import _run_holdout_evaluation

    async_eval = _RecordingAsyncEval()
    _run_holdout_evaluation(
        trainer=_RecordingTrainer(), holdout_buf=_buffer(), tc=_tc_growable(),
        model_cfg=None, device="cpu", config={}, iteration_idx=41,
        holdout_frozen=True, async_test_eval=async_eval,
    )

    assert len(async_eval.started) == 1
    assert async_eval.started[0]["full_pass"] is True


def test_a_zero_holdout_fraction_is_not_treated_as_mutable() -> None:
    """The guard shares the MUTATOR's predicate, not a proxy for it.

    `_ingest_train_arrays` appends on `holdout_frac > 0.0 and not frozen`. With
    `holdout_fraction: 0` nothing can ever be appended, so an unfrozen holdout
    is still safe -- a guard keyed on `holdout_frozen` alone would disable the
    async ruler for the whole run under that config, silently and forever.
    """
    from chess_anti_engine.tune.trainable_phases import _run_holdout_evaluation

    async_eval = _RecordingAsyncEval()
    _run_holdout_evaluation(
        trainer=_RecordingTrainer(), holdout_buf=_buffer(),
        tc=_tc_growable(holdout_fraction=0.0),
        model_cfg=None, device="cpu", config={}, iteration_idx=41,
        holdout_frozen=False, async_test_eval=async_eval,
    )

    assert len(async_eval.started) == 1, (
        "an immutable holdout was needlessly denied the async path"
    )


def test_a_skipped_start_does_not_make_the_next_collect_block() -> None:
    """`collect()` waits its FULL timeout on a cleared event.

    Without the `has_inflight()` check the iteration after a skip pays
    `distributed_async_test_eval_timeout_s` (120s in production) of dead wall
    clock for nothing -- the skip would cost more than the race it avoids.
    """
    from chess_anti_engine.tune.trainable_phases import _run_holdout_evaluation

    async_eval = _RecordingAsyncEval(inflight=False)
    metrics, source_iter = _run_holdout_evaluation(
        trainer=_RecordingTrainer(), holdout_buf=_buffer(), tc=_tc_growable(),
        model_cfg=None, device="cpu", config={}, iteration_idx=42,
        holdout_frozen=True, async_test_eval=async_eval,
    )

    assert async_eval.collects == 0, "collect() was called with nothing in flight"
    assert (metrics, source_iter) == (None, -1)
  # ...and the start still happened; skipping the collect must not skip the eval.
    assert len(async_eval.started) == 1


def test_has_inflight_tracks_the_real_start_collect_lifecycle() -> None:
    """The predicate above is only worth anything if it matches the real class.

    A stub could agree with a wrong implementation forever, so assert against
    `AsyncTestEval` itself: False before any start, True while outstanding,
    False again once collected.
    """
    from chess_anti_engine.train.async_eval import AsyncTestEval

    ev = AsyncTestEval()
    assert ev.has_inflight() is False
    ev._inflight_iter = 7
    assert ev.has_inflight() is True
  # collect() resets it; drive the reset the same way collect() does rather
  # than starting a real eval thread.
    ev._inflight_iter = -1
    assert ev.has_inflight() is False


_HOLDOUT_WRITE_METHODS = frozenset({"clear", "add_many_arrays", "add_many", "add"})


def _holdout_buffer_writers() -> dict[str, str]:
    """Every function in the package that writes a `*holdout*`-named buffer.

    Maps function name -> ``module:lineno`` of its first write, so a failure
    message can point at the offending site instead of just naming it.

    Scans the WHOLE ``chess_anti_engine`` package rather than a hand-listed
    tuple of modules. A curated tuple has the same defect one level up -- a new
    writer landing in a module nobody remembered to list escapes it -- and that
    is the failure this enumeration exists to prevent.
    """
    import ast

    import chess_anti_engine

    pkg_root = Path(chess_anti_engine.__file__).resolve().parent
    writers: dict[str, str] = {}
    for path in sorted(pkg_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
  # Innermost enclosing function for every node. ``ast.walk`` is breadth-first
  # from the root, so an outer def is seen before the defs nested in it and the
  # inner assignment wins -- a write in a closure is attributed to the closure.
        owner: dict[int, str] = {}
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    owner[id(node)] = fn.name
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in _HOLDOUT_WRITE_METHODS:
                continue
            target = func.value
  # ``ast.Attribute`` as well as ``ast.Name``: `self.holdout_buf.clear()` is a
  # write through a receiver that IS named holdout, and keying only on Name let
  # it through (#352 review, probe D4).
            if isinstance(target, ast.Name):
                receiver = target.id
            elif isinstance(target, ast.Attribute):
                receiver = target.attr
            else:
                continue
            if "holdout" not in receiver:
                continue
  # A write outside any def would be import-time and unattributable; name it so
  # the assert below fails rather than silently dropping it.
            key = owner.get(id(node), f"<module level in {path.name}>")
            writers.setdefault(key, f"{path.relative_to(pkg_root)}:{node.lineno}")
    return writers


def test_every_holdout_buffer_writer_is_ordered_against_the_eval() -> None:
    """Enumerate the writers, so a new one cannot be added unnoticed.

    The async eval reads `holdout_buf` from its own thread, so every writer of
    that buffer has to be ordered against it somewhere. Package-wide there are
    THREE, and they are ordered in three different ways for structural reasons:

      `_ingest_train_arrays`          -> the START-side guard in
                                         `_run_holdout_evaluation` (this file)
      `_maybe_reset_holdout_on_drift` -> drains at its own site, because
                                         whether it fires is unknowable at
                                         start() time
      `load_holdout_rows`             -> ordered BY CONSTRUCTION: it runs
                                         inside `_init_replay_buffers`, before
                                         `train_trial` has an eval handle at
                                         all, so there is nothing to order
                                         against. Proved, not asserted, by
                                         `test_the_init_time_holdout_writer_
                                         cannot_race_the_eval_thread` below --
                                         a claimed exemption nobody checks is
                                         the same defect as a wrong count.

    This asserts the SET of writers, not the handling -- each behaviour is
    pinned by its own test. A new `.clear()`/`add_many_arrays` on a holdout
    buffer fails here and sends the author to pick one of the three.

    LIMIT, stated rather than implied: the scan keys on a receiver NAMED
    `*holdout*`, so a write through a differently-named alias (`hb = holdout_buf;
    hb.clear()`) is invisible to it. It is a tripwire for the ordinary case, not
    a proof of absence -- which is why the real behaviours are pinned by
    behavioural tests and this only guards the enumeration.
    """
    found = _holdout_buffer_writers()

    assert set(found) == {
        "_ingest_train_arrays", "_maybe_reset_holdout_on_drift", "load_holdout_rows",
    }, (
        "the set of holdout-buffer writers changed. Every writer must be "
        "ordered against the async eval thread: the start-side guard in "
        "_run_holdout_evaluation, a drain at the writer's own site the way "
        "_maybe_reset_holdout_on_drift does, or -- for an init-time writer -- a "
        "proof that it cannot overlap the eval thread at all. Found: "
        f"{sorted(found.items())}"
    )


def test_the_init_time_holdout_writer_cannot_race_the_eval_thread() -> None:
    """`load_holdout_rows`'s exemption, proved from the code rather than claimed.

    It is the one writer with no ordering construct at its site, on the grounds
    that it runs before an eval thread exists. Two structural facts make that
    true, and both are checked here so the exemption cannot quietly rot:

    1. REACHABILITY -- `load_holdout_rows` is called only from
       `_restore_holdout_buffer`, which is called only from
       `_init_replay_buffers`. It is buffer construction, not a loop-time
       mutator.
    2. ORDER IN `train_trial` -- the `_init_replay_buffers` call precedes every
       binding of the name `async_test_eval`. At the moment the holdout rows
       are loaded the eval handle does not exist in the frame, so no thread can
       be reading the buffer; the handle is only bound to a real
       `AsyncTestEval` later, by `_lazy_construct_iter_helpers` inside the loop.

    If either stops holding, `load_holdout_rows` needs a drain like
    `_maybe_reset_holdout_on_drift`, and the enumeration test above must move
    it out of the by-construction bucket.
    """
    import ast
    import inspect

    import chess_anti_engine
    from chess_anti_engine.tune import trainable

    pkg_root = Path(chess_anti_engine.__file__).resolve().parent

    def _callers_of(name: str) -> set[str]:
        callers: set[str] = set()
        for path in sorted(pkg_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == name
                    ):
                        callers.add(fn.name)
        return callers

    assert _callers_of("load_holdout_rows") == {"_restore_holdout_buffer"}, (
        "load_holdout_rows gained a caller outside buffer construction; its "
        "by-construction exemption in the enumeration test no longer holds"
    )
    assert _callers_of("_restore_holdout_buffer") == {"_init_replay_buffers"}, (
        "_restore_holdout_buffer gained a caller outside _init_replay_buffers; "
        "the init-time holdout write may now run while an eval is in flight"
    )

    fn = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(trainable)))
        if isinstance(node, ast.FunctionDef) and node.name == "train_trial"
    )
    init_lines = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_init_replay_buffers"
    ]
    assert len(init_lines) == 1, (
        f"expected one _init_replay_buffers call in train_trial, got {init_lines}"
    )
    eval_bindings = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Name)
        and node.id == "async_test_eval"
        and isinstance(node.ctx, ast.Store)
    ]
    assert eval_bindings, "train_trial no longer binds async_test_eval at all"
    assert min(eval_bindings) > init_lines[0], (
        "train_trial binds async_test_eval at line "
        f"{min(eval_bindings)}, at or before the _init_replay_buffers call at "
        f"line {init_lines[0]}. The init-time holdout write is no longer "
        "ordered by construction and needs a drain of its own"
    )
