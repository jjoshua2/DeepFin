"""The holdout ruler must survive a restart.

`test_loss` -- and therefore best-model selection -- is measured against the
holdout buffer. It used to be reconstructed EMPTY on every process start
(`trainable_init._init_replay_buffers` built a bare `ArrayReplayBuffer`), so a
restart silently swapped the ruler: measured across two restarts of one live
trial, iter 1 held 210 rows and reached the 2000-row cap only by iter 14, and
iter 42 -- just after the next restart -- was back to 194. The first ~3
iterations of each segment had no holdout eval at all, the next ~10 were
judged against a set that was still growing, and `best_loss` carried across
the boundary defending a number earned on a set that no longer existed.

These tests pin the four properties the fix has to hold: the rows round-trip,
a frozen holdout comes back frozen, anything missing or mismatched degrades to
the old empty-and-refill behaviour instead of crashing, and a drift reset can
still throw the whole thing away and say so via `holdout_generation`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chess_anti_engine.replay.buffer import ArrayReplayBuffer
from chess_anti_engine.tune._utils import SIDECAR_HOLDOUT_ROWS, SIDECAR_TRIAL_META
from chess_anti_engine.tune.holdout_state import (
    load_holdout_rows,
    restored_holdout_scalars,
    save_holdout_rows,
)
from chess_anti_engine.tune.trainable import (
    _maybe_freeze_holdout,
    _maybe_reset_holdout_on_drift,
)
from chess_anti_engine.tune.trainable_init import (
    _apply_restored_holdout_scalars,
    _restore_holdout_buffer,
)
from chess_anti_engine.tune.trial_config import RestoreResult, TrialConfig

PLANES = 146
POLICY = 4672


def _arrays(n: int, *, first_row: int = 0, planes: int = PLANES, policy: int = POLICY) -> dict:
    """n rows whose identity is readable back out of every stored field."""
    rows = np.arange(first_row, first_row + n, dtype=np.int64)
    pol = np.zeros((n, policy), dtype=np.float32)
    pol[np.arange(n), rows % policy] = 1.0
    x = np.zeros((n, planes, 8, 8), dtype=np.float32)
    x[:, 0, 0, 0] = rows.astype(np.float32)
    return {
        "x": x,
        "policy_target": pol,
        "wdl_target": (rows % 3).astype(np.int8),
        "priority": rows.astype(np.float32) + 1.0,
        "has_policy": np.ones((n,), dtype=np.uint8),
    }


def _buffer(capacity: int, *, rows: int = 0, first_row: int = 0, **kw) -> ArrayReplayBuffer:
    buf = ArrayReplayBuffer(capacity, rng=np.random.default_rng(0))
    if rows:
        buf.add_many_arrays(_arrays(rows, first_row=first_row, **kw))
    return buf


def _row_ids(buf: ArrayReplayBuffer) -> list[int]:
    """The identity marker of every row held, in buffer order."""
    return [int(v) for v in np.asarray(buf.export_arrays()["x"])[:, 0, 0, 0]]


def _tc(**kw) -> TrialConfig:
    base = {
        "input_extra_features": "v1",
        "policy_encoding": "az_4672",
        "holdout_capacity": 64,
        "freeze_holdout_at": 2000,
    }
    return TrialConfig.from_dict({**base, **kw})


def _restore_into(
    tmp_path: Path, *, capacity: int = 64, frozen: bool = False, generation: int = 0,
    state_dir: Path | None = None, tc: TrialConfig | None = None,
) -> tuple[ArrayReplayBuffer, RestoreResult]:
    buf = ArrayReplayBuffer(capacity, rng=np.random.default_rng(1))
    restore = RestoreResult(
        holdout_state_dir=tmp_path if state_dir is None else state_dir,
        holdout_frozen=frozen,
        holdout_generation=generation,
    )
    _restore_holdout_buffer(
        tc=tc if tc is not None else _tc(holdout_capacity=capacity),
        restore=restore, holdout_buf=buf,
    )
    return buf, restore


# --- round trip -----------------------------------------------------------


def test_populated_holdout_round_trips_through_a_checkpoint(tmp_path: Path) -> None:
    """The whole point: the same rows, in the same order, after a restart."""
    saved = _buffer(64, rows=20)
    assert save_holdout_rows(ckpt_dir=tmp_path, holdout_buf=saved) == 20
    assert (tmp_path / SIDECAR_HOLDOUT_ROWS).exists()

    restored, _ = _restore_into(tmp_path)

    assert len(restored) == 20
    assert _row_ids(restored) == _row_ids(saved)
    before = saved.export_arrays()
    after = restored.export_arrays()
    for key in ("x", "policy_target", "wdl_target", "priority"):
        np.testing.assert_array_equal(np.asarray(after[key]), np.asarray(before[key]))


def test_restored_holdout_is_usable_for_evaluation(tmp_path: Path) -> None:
    """A round-tripped buffer still samples. A ruler you cannot draw from is
    not a ruler -- and a half-restored buffer would raise inside `_gather_rows`
    mid-iteration rather than at startup."""
    save_holdout_rows(ckpt_dir=tmp_path, holdout_buf=_buffer(64, rows=20))
    restored, _ = _restore_into(tmp_path)

    batch = restored.sample_batch_arrays(8)

    assert np.asarray(batch["x"]).shape == (8, PLANES, 8, 8)
    assert np.asarray(batch["policy_target"]).shape == (8, POLICY)


def test_an_empty_holdout_removes_a_stale_sidecar(tmp_path: Path) -> None:
    """`ckpt_dir` is one reused directory, so a file left over from before a
    drift reset would be restored as if it were the current ruler."""
    save_holdout_rows(ckpt_dir=tmp_path, holdout_buf=_buffer(64, rows=20))

    assert save_holdout_rows(ckpt_dir=tmp_path, holdout_buf=_buffer(64)) == 0

    assert not (tmp_path / SIDECAR_HOLDOUT_ROWS).exists()
    restored, _ = _restore_into(tmp_path)
    assert len(restored) == 0


def test_restore_honours_a_reduced_capacity(tmp_path: Path) -> None:
    """A shrunk `holdout_capacity` evicts oldest-first on the way in, exactly
    as ingest would have."""
    save_holdout_rows(ckpt_dir=tmp_path, holdout_buf=_buffer(64, rows=20))

    restored, _ = _restore_into(tmp_path, capacity=8)

    assert len(restored) == 8
    assert _row_ids(restored) == list(range(12, 20))


# --- frozen / generation --------------------------------------------------


def test_a_frozen_holdout_comes_back_frozen(tmp_path: Path) -> None:
    save_holdout_rows(ckpt_dir=tmp_path, holdout_buf=_buffer(64, rows=20))

    _, restore = _restore_into(tmp_path, frozen=True, generation=3)

    assert restore.holdout_frozen is True
    assert restore.holdout_generation == 3


def test_frozen_is_dropped_when_the_rows_are_gone(tmp_path: Path) -> None:
    """Honouring `frozen` over an empty buffer would wedge the holdout shut
    for the life of the run: ingest skips a frozen holdout, so it could never
    reach `batch_size` and never be evaluated again."""
    _, restore = _restore_into(tmp_path, frozen=True, generation=3)

    assert restore.holdout_frozen is False


def test_freeze_holdout_at_zero_unfreezes_a_restored_holdout(tmp_path: Path) -> None:
    """Freezing is one-way inside a process, so with the flag durable the
    config has to stay the authority: setting `freeze_holdout_at: 0` and
    restarting is the operator's way back, and before the holdout was
    persisted the restart itself delivered it."""
    save_holdout_rows(ckpt_dir=tmp_path, holdout_buf=_buffer(64, rows=20))

    _, restore = _restore_into(
        tmp_path, frozen=True, generation=3, tc=_tc(freeze_holdout_at=0),
    )

    assert restore.holdout_frozen is False
    assert restore.holdout_generation == 3, "the ruler did not change, only the policy"


@pytest.mark.parametrize(
    ("rows", "threshold", "expected"),
    [(20, 16, True), (20, 64, False), (20, 0, False)],
    ids=["at-threshold-freezes", "below-threshold-keeps-filling", "disabled-never-freezes"],
)
def test_a_restored_unfrozen_holdout_freezes_at_the_threshold(
    rows: int, threshold: int, expected: bool,
) -> None:
    """Before the rows were persisted this test could only ever fire against a
    fresh post-restart sample, so `freeze_holdout_at` re-froze a DIFFERENT set
    after every restart and the ruler moved anyway."""
    assert _maybe_freeze_holdout(
        holdout_buf=_buffer(64, rows=rows),
        tc=_tc(freeze_holdout_at=threshold),
        holdout_frozen=False,
    ) is expected


def test_freezing_is_not_undone_by_a_shrinking_buffer() -> None:
    """One-way: once frozen, ingest stops adding, and nothing here re-opens it
    except a drift reset."""
    assert _maybe_freeze_holdout(
        holdout_buf=_buffer(64), tc=_tc(freeze_holdout_at=16), holdout_frozen=True,
    ) is True


def test_losing_the_rows_bumps_the_generation(tmp_path: Path) -> None:
    """A restart that could not restore the set really is a different ruler,
    and the counter is how `_update_best_model` finds out."""
    _, restore = _restore_into(tmp_path, frozen=True, generation=3)

    assert restore.holdout_generation == 4


def test_a_fresh_start_stays_at_generation_zero() -> None:
    """Nothing was lost, because there was nothing to lose."""
    assert restored_holdout_scalars(
        rows_restored=0, stored_frozen=False, stored_generation=0,
        had_stored_state=False,
    ) == (False, 0)


def test_scalars_survive_a_real_trial_meta_round_trip(tmp_path: Path) -> None:
    """The two flags ride in trial_meta.json, which is the file both restore
    paths already read and salvage already copies."""
    (tmp_path / SIDECAR_TRIAL_META).write_text(
        json.dumps({"holdout_frozen": True, "holdout_generation": 7}), encoding="utf-8",
    )
    meta = json.loads((tmp_path / SIDECAR_TRIAL_META).read_text(encoding="utf-8"))
    rr = RestoreResult()

    _apply_restored_holdout_scalars(rr, meta)

    assert (rr.holdout_frozen, rr.holdout_generation) == (True, 7)


def test_trial_meta_without_the_fields_reads_as_a_fresh_holdout() -> None:
    """Backward compatibility for a checkpoint written before this existed."""
    rr = RestoreResult()

    _apply_restored_holdout_scalars(rr, {"global_iter": 41})

    assert (rr.holdout_frozen, rr.holdout_generation) == (False, 0)


# --- degradation ----------------------------------------------------------


def test_a_checkpoint_with_no_stored_holdout_starts_empty(tmp_path: Path) -> None:
    """The pre-existing checkpoints in the live tune dir. Must degrade to the
    old behaviour -- empty and refilling -- not crash the trial."""
    restored, restore = _restore_into(tmp_path)

    assert len(restored) == 0
    assert restore.holdout_frozen is False


def test_a_fresh_start_has_nothing_to_restore_from() -> None:
    buf = ArrayReplayBuffer(64, rng=np.random.default_rng(0))

    assert load_holdout_rows(
        state_dir=None, holdout_buf=buf,
        expected_planes=PLANES, expected_policy_size=POLICY,
    ) == 0


def test_a_corrupt_sidecar_degrades_instead_of_raising(tmp_path: Path) -> None:
    (tmp_path / SIDECAR_HOLDOUT_ROWS).write_bytes(b"not an npz at all")
    buf = ArrayReplayBuffer(64, rng=np.random.default_rng(0))

    assert load_holdout_rows(
        state_dir=tmp_path, holdout_buf=buf,
        expected_planes=PLANES, expected_policy_size=POLICY,
    ) == 0
    assert len(buf) == 0


@pytest.mark.parametrize(
    ("planes", "policy"),
    [(175, POLICY), (PLANES, 1858)],
    ids=["plane-count-changed", "policy-width-changed"],
)
def test_a_sidecar_from_a_different_encoding_is_discarded(
    tmp_path: Path, planes: int, policy: int,
) -> None:
    """A v1->v2_threats plane change or an az_4672->lc0_1858 policy change
    makes the stored rows unusable. Caught at startup as a logged line, not at
    eval time as a shape error a dozen frames deep."""
    stale = _buffer(64, rows=6, planes=planes, policy=policy)
    save_holdout_rows(ckpt_dir=tmp_path, holdout_buf=stale)
    buf = ArrayReplayBuffer(64, rng=np.random.default_rng(0))

    assert load_holdout_rows(
        state_dir=tmp_path, holdout_buf=buf,
        expected_planes=PLANES, expected_policy_size=POLICY,
    ) == 0
    assert len(buf) == 0


def test_a_mismatched_sidecar_still_bumps_the_generation(tmp_path: Path) -> None:
    """Discarding is the right call, but the ruler still changed."""
    save_holdout_rows(ckpt_dir=tmp_path, holdout_buf=_buffer(64, rows=6, planes=175))

    restored, restore = _restore_into(tmp_path, frozen=True, generation=2)

    assert len(restored) == 0
    assert (restore.holdout_frozen, restore.holdout_generation) == (False, 3)


# --- drift reset ----------------------------------------------------------


def _drift(l2: float) -> SimpleNamespace:
    return SimpleNamespace(drift_input_l2=l2)


def test_drift_reset_still_clears_a_restored_holdout(tmp_path: Path) -> None:
    save_holdout_rows(ckpt_dir=tmp_path, holdout_buf=_buffer(64, rows=20))
    buf, restore = _restore_into(tmp_path, frozen=True, generation=5)
    assert (len(buf), restore.holdout_frozen) == (20, True)

    frozen, generation = _maybe_reset_holdout_on_drift(
        holdout_buf=buf, drift=_drift(9.0),
        tc=_tc(reset_holdout_on_drift=True, drift_threshold=1.0),
        holdout_frozen=restore.holdout_frozen,
        holdout_generation=restore.holdout_generation,
    )

    assert len(buf) == 0
    assert (frozen, generation) == (False, 6)


def test_drift_below_threshold_leaves_a_restored_holdout_alone(tmp_path: Path) -> None:
    save_holdout_rows(ckpt_dir=tmp_path, holdout_buf=_buffer(64, rows=20))
    buf, restore = _restore_into(tmp_path, frozen=True, generation=5)

    frozen, generation = _maybe_reset_holdout_on_drift(
        holdout_buf=buf, drift=_drift(0.1),
        tc=_tc(reset_holdout_on_drift=True, drift_threshold=1.0),
        holdout_frozen=restore.holdout_frozen,
        holdout_generation=restore.holdout_generation,
    )

    assert len(buf) == 20
    assert (frozen, generation) == (True, 5)


# --- what the result row says the holdout is ------------------------------


def _test_dict(*, evaluated: bool, eval_rows: int = 0) -> dict:
    from chess_anti_engine.train.trainer import TrainMetrics
    from chess_anti_engine.tune.trainable_report import _test_and_drift_dict
    from chess_anti_engine.tune.trial_config import DriftMetrics, TrainingResult

    tr = TrainingResult()
    if evaluated:
        tr.test_metrics = TrainMetrics(
            loss=0.5, policy_loss=0.5, soft_policy_loss=0.5, future_policy_loss=0.5,
            wdl_loss=0.5, sf_move_loss=0.5, sf_move_acc=0.5, sf_eval_loss=0.5,
            categorical_loss=0.5, volatility_loss=0.5, sf_volatility_loss=0.5,
            moves_left_loss=0.5, eval_rows=eval_rows,
        )
        tr.test_metrics_source_iter = 41
    return _test_and_drift_dict(
        tr=tr, drift=DriftMetrics(),
        holdout_frozen=False, holdout_generation=0,
    )


def test_test_size_counts_the_rows_the_eval_actually_scored() -> None:
    """G6, then G14. `test_size` was the holdout BUFFER size while reading like
    an evaluated-row count; it then became ``test_steps * batch_size`` = 2560
    draws WITH REPLACEMENT from a 2000-row set. The eval is now a deterministic
    full pass, so it reports the rows it scored and nothing reconstructs that
    from config -- 2000 rows at batch 512 is 3 x 512 + a ragged 464."""
    row = _test_dict(evaluated=True, eval_rows=2000)

    assert row["test_size"] == 2000


def test_test_size_is_zero_when_no_eval_ran() -> None:
    row = _test_dict(evaluated=False)

    assert row["test_size"] == 0


def test_the_buffer_size_is_still_emitted_under_its_own_name() -> None:
    """`test_replay` is the buffer size, and `audit_realized_config.py`
    already documents it as such. Nothing was lost by making `test_size`
    honest, and no CSV column was added or removed -- Ray fixes the header on
    row 1 and a resume appends without re-heading."""
    from chess_anti_engine.tune.trainable_report import _build_report_dict

    src = Path(_build_report_dict.__code__.co_filename).read_text(encoding="utf-8")

    assert '"test_replay": int(holdout_buf_size),' in src


# --- the paths a restart actually takes -----------------------------------


def test_a_real_checkpoint_write_carries_rows_and_flags(tmp_path: Path) -> None:
    """End to end through the production writer and reader: save a frozen,
    populated holdout as `_save_trial_checkpoint` does, then bring it back the
    way a resume does -- trial_meta.json for the flags, the sidecar for the
    rows."""
    from chess_anti_engine.tune._utils import load_optional_json
    from chess_anti_engine.tune.trainable_report import _save_trial_checkpoint

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    saved = _buffer(64, rows=20)
    _save_trial_checkpoint(
        trainer=SimpleNamespace(save=lambda p: Path(p).write_bytes(b"weights")),
        buf=SimpleNamespace(flush=lambda: None),
        ckpt_dir=ckpt_dir,
        rng=np.random.default_rng(0),
        trial_id="t1",
        trial_dir=tmp_path,
        config={"optimizer": "aurora"},
        base_seed=7,
        restore=RestoreResult(),
        iteration_idx=41,
        current_window=1_500_000,
        holdout_buf=saved,
        holdout_frozen=True,
        holdout_generation=2,
        holdout_ruler="v1:full_pass:0123456789abcdef",
        opp_strength_ema=321.5,
        yaml_keys=("batch_size", "lr"),
        Checkpoint=SimpleNamespace(from_directory=lambda d: d),
    )

    meta = load_optional_json(ckpt_dir / SIDECAR_TRIAL_META)
    assert meta is not None
    rr = RestoreResult(holdout_state_dir=ckpt_dir)
    _apply_restored_holdout_scalars(rr, meta)
    buf = ArrayReplayBuffer(64, rng=np.random.default_rng(1))
    _restore_holdout_buffer(tc=_tc(), restore=rr, holdout_buf=buf)

    assert _row_ids(buf) == _row_ids(saved)
    assert (rr.holdout_frozen, rr.holdout_generation) == (True, 2)
  # The set's identity is only half the ruler's; the measurement applied to it
  # rides in the same file so a restart can see it change.
    assert rr.holdout_ruler == "v1:full_pass:0123456789abcdef"


def test_salvage_export_carries_the_holdout_sidecar() -> None:
    """A salvage restore lands in a DIFFERENT trial dir, so the sidecar only
    survives it by being in the fixed list salvage copies out of the
    checkpoint. This is the reason the rows live in the checkpoint rather than
    beside `best.json` in the durable trial dir."""
    from chess_anti_engine.tune import salvage

    src = Path(salvage.__file__).read_text(encoding="utf-8")
    copy_stmt = src[src.index('for fn in ("trainer.pt"'):src.index("src = plan.ckpt_dir / fn")]

    assert "SIDECAR_HOLDOUT_ROWS" in copy_stmt


def test_the_trial_loop_seeds_its_holdout_flags_from_the_restore() -> None:
    """The defect in one line: `holdout_frozen = False` / `holdout_generation
    = 0` at trial start, unconditionally, discarding whatever the checkpoint
    knew."""
    from chess_anti_engine.tune import trainable

    src = Path(trainable.__file__).read_text(encoding="utf-8")

    assert "holdout_frozen = bool(restore.holdout_frozen)" in src
    assert "holdout_generation = int(restore.holdout_generation)" in src
