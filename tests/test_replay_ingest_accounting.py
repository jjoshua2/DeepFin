"""Replay accounting follows submitted train rows, independently of shard metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from chess_anti_engine.replay import ArrayReplayBuffer, DiskReplayBuffer
from chess_anti_engine.moves.encode import POLICY_SIZE
from chess_anti_engine.replay.shard import (
    LOCAL_SHARD_SUFFIX,
    load_shard_arrays,
    save_local_shard_arrays,
)
from chess_anti_engine.tune import distributed_runtime as runtime
from chess_anti_engine.tune import trainable_phases as phases
from chess_anti_engine.tune import trainable_report as report
from chess_anti_engine.tune.trainable_metrics import _compute_train_step_budget
from chess_anti_engine.tune.promotion_gate import GateDecision
from chess_anti_engine.train.trainer import TrainMetrics
from chess_anti_engine.tune.trial_config import (
    DifficultyState,
    DriftMetrics,
    PidResult,
    RestoreResult,
    TrialConfig,
)
from tests.test_report_counters import _fake_trainer


def _arrays(start: int = 0, *, tagged: bool = False) -> dict[str, np.ndarray]:
    x = np.zeros((10, 175, 8, 8), dtype=np.float32)
    x[:, 0, 0, 0] = np.arange(start, start + 10)
    policy = np.zeros((10, POLICY_SIZE), dtype=np.float32)
    policy[:, 0] = 1.0
    arrs = {
        "x": x, "wdl_target": np.zeros(10, dtype=np.int8),
        "policy_target": policy, "has_policy": np.ones(10, dtype=np.uint8),
        "priority": np.ones(10, dtype=np.float32),
    }
    if tagged:
        arrs["has_is_selfplay"] = np.ones(10, dtype=np.uint8)
        arrs["is_selfplay"] = np.ones(10, dtype=np.uint8)
    return arrs


def _buffer(tmp_path: Path, *, capacity: int = 100) -> DiskReplayBuffer:
    return DiskReplayBuffer(
        capacity, shard_dir=tmp_path / "replay", rng=np.random.default_rng(3),
        read_only=False, shuffle_cap=32, shard_size=2,
    )


def _save(tmp_path: Path, name: str, arrs: dict, meta: dict) -> Path:
    path = tmp_path / "inbox" / "worker" / f"{name}{LOCAL_SHARD_SUFFIX}"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_local_shard_arrays(path, arrs=arrs, meta=meta)
    return path


@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("tagged", [False, True])
@pytest.mark.parametrize(
    ("fraction", "frozen", "train_n"),
    [(0.2, False, 8), (1.0, False, 0), (0.0, False, 10), (0.2, True, 10)],
    ids=["split", "all-holdout", "disabled", "frozen"],
)
def test_process_shard_counts_submitted_rows(
    tmp_path: Path, preloaded: bool, tagged: bool,
    fraction: float, frozen: bool, train_n: int,
) -> None:
    arrs = _arrays(tagged=tagged)
    # Missing positions metadata must fall back for game metrics, not count holdout.
    path = _save(tmp_path, "fresh", arrs, {"model_sha256": "fresh", "games": 1})
    buf = _buffer(tmp_path)
    holdout = ArrayReplayBuffer(32, rng=np.random.default_rng(1))
    summary = runtime._empty_ingest_summary()
    runtime._process_shard(
        path, inbox_dir=tmp_path / "inbox", processed_dir=tmp_path / "processed",
        buf=buf, holdout_buf=holdout, holdout_frac=fraction, holdout_frozen=frozen,
        accepted_model_shas={"fresh"}, rng=np.random.default_rng(0), summary=summary,
        preloaded=(arrs, {"model_sha256": "fresh", "games": 1}) if preloaded else None,
    )
    buf.flush()
    assert summary["positions_replay_added"] == train_n
    assert summary["matching_positions"] == 10
    assert summary["matching_games"] == summary["matching_shards"] == 1
    assert summary["stale_positions"] == 0
    assert len(buf) == train_n
    assert len(holdout) == 10 - train_n
    train_ids = {
        int(i) for shard in buf._snapshot_shards()
        for i in load_shard_arrays(shard)[0]["x"][:, 0, 0, 0]
    }
    holdout_ids = (
        set(holdout.export_arrays()["x"][:, 0, 0, 0]) if len(holdout) else set()
    )
    assert len(train_ids) == train_n
    assert train_ids.isdisjoint(holdout_ids)
    assert train_ids | holdout_ids == set(range(10))
    assert summary["ingest_is_selfplay_tagged"] == (train_n if tagged else 0)
    assert summary["ingest_is_selfplay_true"] == (train_n if tagged else 0)
    assert not path.exists()
    assert (tmp_path / "processed" / "worker" / path.name).exists()
    if train_n == 8:
        budget = _compute_train_step_budget(
            positions_ingested=summary["positions_replay_added"], imported_samples=0,
            replay_size=len(buf), batch_size=2, accum_steps=1, base_max_steps=100,
            train_window_fraction=0.0, train_views_per_ingested_position=2.5,
        )
        assert budget["target_sample_budget"] == 20
        assert budget["steps"] == 10


@pytest.mark.parametrize("failing_buffer", ["holdout", "replay"])
def test_failed_submission_is_not_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_buffer: str,
) -> None:
    path = _save(tmp_path, "fresh", _arrays(), {"model_sha256": "fresh", "games": 1})
    buf = _buffer(tmp_path)
    holdout = ArrayReplayBuffer(32, rng=np.random.default_rng(1))
    summary = runtime._empty_ingest_summary()

    def fail(_arrs: dict) -> None:
        raise RuntimeError("submission failed")

    monkeypatch.setattr(holdout if failing_buffer == "holdout" else buf, "add_many_arrays", fail)
    with pytest.raises(RuntimeError, match="submission failed"):
        runtime._process_shard(
            path, inbox_dir=tmp_path / "inbox", processed_dir=tmp_path / "processed",
            buf=buf, holdout_buf=holdout, holdout_frac=0.2, holdout_frozen=False,
            accepted_model_shas={"fresh"}, rng=np.random.default_rng(0), summary=summary,
        )
    assert summary["positions_replay_added"] == 0
    assert summary["matching_games"] == 0
    assert len(buf) == 0
    assert len(holdout) == (0 if failing_buffer == "holdout" else 2)
    assert path.exists()


def test_replay_eviction_does_not_reduce_submission_count(tmp_path: Path) -> None:
    buf = _buffer(tmp_path, capacity=4)
    buf.add_many_arrays(_arrays(20))
    buf.flush()
    before = len(buf)
    assert before == 4
    summary = runtime._empty_ingest_summary()
    path = _save(tmp_path, "fresh", _arrays(), {"model_sha256": "fresh", "games": 1})
    runtime._process_shard(
        path, inbox_dir=tmp_path / "inbox", processed_dir=tmp_path / "processed",
        buf=buf, holdout_buf=ArrayReplayBuffer(32, rng=np.random.default_rng(1)),
        holdout_frac=0.2, holdout_frozen=False,
        accepted_model_shas={"fresh"}, rng=np.random.default_rng(0), summary=summary,
    )
    buf.flush()
    assert len(buf) == before
    assert summary["positions_replay_added"] == 8


@pytest.mark.parametrize(
    ("all_holdout", "old_replay"), [(False, False), (True, False), (True, True)],
    ids=["split", "all-holdout-empty", "all-holdout-old-replay"],
)
def test_ingest_count_reaches_window_training_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, all_holdout: bool, old_replay: bool,
) -> None:
    # Keep ingest, replay, result mapping, window growth and budget/report consumers real.
    # No worker launch, model publication, sibling export or actual training is needed.
    monkeypatch.setattr(phases, "_ensure_distributed_workers", lambda **kw: [])
    monkeypatch.setattr(phases, "_ensure_inference_broker", lambda **kw: None)
    monkeypatch.setattr(phases, "_publish_iteration_model", lambda **kw: "fresh")
    monkeypatch.setattr(phases, "revive_dead_selfplay_processes", lambda **kw: False)
    monkeypatch.setattr(phases, "_export_selfplay_shards_for_siblings", lambda **kw: None)
    monkeypatch.setattr(phases, "_run_net_gating", lambda *_args, **kw: (GateDecision(), 0))
    monkeypatch.setattr(phases, "_run_holdout_evaluation", lambda **kw: (None, None))
    tc = TrialConfig(
        games_per_iter=1, batch_size=2, accum_steps=1, train_steps=3,
        train_views_per_ingested_position=2.5,
        train_window_fraction=1.0 if all_holdout else 0.0,
        holdout_fraction=1.0 if all_holdout else 0.2,
        replay_window_max=200, replay_window_growth_frac=0.5,
        distributed_wait_timeout_seconds=0.1, distributed_worker_poll_seconds=0.01,
        distributed_pause_selfplay_during_training=False,
    )
    buf = _buffer(tmp_path)
    # Old replay makes an all-held-out iteration reach the drought/cap path.
    if old_replay:
        buf.add_many_arrays(_arrays(20))
        buf.flush()
    holdout = ArrayReplayBuffer(32, rng=np.random.default_rng(1))
    for name, sha, offset, positions in (
        ("00-stale", "stale", 0, 1000), ("99-fresh", "fresh", 10, 2000),
    ):
        _save(tmp_path, name, _arrays(offset), {
            "model_sha256": sha, "games": 1, "positions": positions,
            "wins": 1, "draws": 0, "losses": 0,
            "outcome_stats": {"checkmate": 1},
        })
    ds = DifficultyState(wdl_regret=0.05, sf_nodes=1000)
    trainer = _fake_trainer()
    seen_steps = []

    def train_steps(train_buf, *, batch_size: int, steps: int) -> TrainMetrics:
        assert train_buf is buf
        assert batch_size == 2
        seen_steps.append(steps)
        return TrainMetrics(
            loss=0.0, policy_loss=0.0, soft_policy_loss=0.0, future_policy_loss=0.0,
            wdl_loss=0.0, sf_move_loss=0.0, sf_move_acc=0.0, sf_eval_loss=0.0,
            categorical_loss=0.0, volatility_loss=0.0, sf_volatility_loss=0.0,
            moves_left_loss=0.0, train_samples_seen=steps * batch_size,
        )

    trainer.train_steps = train_steps
    sp, _sha, window = phases._run_selfplay_phase(
        tc=tc, config={}, trainer=trainer, model_cfg=None, buf=buf, holdout_buf=holdout,
        holdout_frozen=False, rng=np.random.default_rng(0),
        distributed_dirs={"publish_dir": tmp_path / "publish",
                          "inbox_dir": tmp_path / "inbox", "processed_dir": tmp_path / "processed"},
        distributed_server_root=tmp_path, distributed_worker_procs=[], broker_proc_box=[None],
        prev_published_model_sha="", ds=ds, sims=1, iteration_idx=1, iteration_zero_based=1,
        trial_id="test", trial_dir=tmp_path, selfplay_shards_dir=tmp_path / "siblings",
        replay_shard_dir=tmp_path / "replay", current_window=100, in_salvage_startup_grace=False,
    )
    expected = 0 if all_holdout else 15  # seed 0: 2 held out from stale, 3 from fresh
    assert not sp.should_retry
    assert sp.replay_positions_ingested == expected
    assert sp.matching_positions == 2000
    assert sp.distributed_stale_positions == 1000
    assert sp.matching_games == sp.distributed_stale_games == 1
    assert sp.total_w == sp.gate_cur_w == 1
    assert sp.outcome_stats == {"checkmate": 1}
    assert window == 100 + (0 if all_holdout else 8)
    tr = phases._run_training_and_gating(
        tc=tc, trainer=trainer, buf=buf, holdout_buf=holdout, holdout_frozen=False,
        config={}, model_cfg=None, device="cpu", ds=ds, sims=1, sp=sp,
        positions_ingested=sp.replay_positions_ingested, imported_samples_this_iter=0,
        gate=cast(Any, None), gate_match_idx=0, gate_state_path=tmp_path / "gate.json", gate_hold=None,
        distributed_server_root=tmp_path, iteration_idx=1, iteration_zero_based=1,
        trial_id="test", restore=RestoreResult(),
    )
    assert tr.target_sample_budget == (10 if old_replay else 0 if all_holdout else 38)
    assert tr.steps == (3 if old_replay else 0 if all_holdout else 19)
    assert seen_steps == ([tr.steps] if tr.steps else [])
    row = report._build_report_dict(
        tc=tc, trainer=trainer, pr=PidResult(), sp=sp, tr=tr, drift=DriftMetrics(),
        eval_dict={}, puzzle_dict={}, wdl_regret_used=0.05, sf_nodes_used=1000,
        pause_metrics={"paused_seconds": 0.0, "paused_fraction": 0.0, "paused_percent": 0.0},
        restore=RestoreResult(), best_loss=1.0, iter_t0=0.0,
        iteration_idx=1, buf_size=len(buf), holdout_buf_size=len(holdout),
        holdout_frozen=False, holdout_generation=0,
    )
    assert row["replay_positions_ingested"] == expected
    assert row["train_views_actual"] == pytest.approx(0 if all_holdout else 38 / 15)
    assert row["replay_window_growth_positions"] == (0 if all_holdout else 8)
