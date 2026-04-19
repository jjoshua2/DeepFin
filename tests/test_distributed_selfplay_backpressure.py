from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from chess_anti_engine.replay import ArrayReplayBuffer
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import save_npz_arrays
from chess_anti_engine.tune.distributed_runtime import (
    _ingest_distributed_selfplay,
    _publish_distributed_trial_state,
    _quarantine_inbox_shards,
)
from chess_anti_engine.tune.trainable_metrics import (
    _blended_winrate_raw_or_none,
    _compute_train_step_budget,
    _iteration_pause_metrics,
    _should_retry_iteration_without_games,
)
from chess_anti_engine.worker import _manifest_poll_headers


class _FakeTrainer:
    def export_swa(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {}}, str(path))


def test_publish_distributed_trial_state_includes_pause_selfplay(tmp_path: Path) -> None:
    trainer = _FakeTrainer()
    model_cfg = SimpleNamespace(
        kind="transformer",
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_mult=2,
        use_smolgen=False,
        use_nla=False,
        use_qk_rmsnorm=False,
        use_gradient_checkpointing=False,
    )

    _publish_distributed_trial_state(
        trainer=trainer,
        config={
            "selfplay_batch": 16,
            "max_plies": 240,
            "mcts": "gumbel",
            "fast_simulations": 8,
        },
        model_cfg=model_cfg,
        server_root=tmp_path,
        trial_id="trial_00000",
        training_iteration=7,
        trainer_step=123,
        sf_nodes=1000,
        mcts_simulations=64,
        pause_selfplay=True,
        pause_reason="training",
        backpressure={"stale_games": 96, "phase": "training"},
    )

    manifest_path = tmp_path / "trials" / "trial_00000" / "publish" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["recommended_worker"]["pause_selfplay"] is True
    assert manifest["recommended_worker"]["pause_reason"] == "training"
    assert manifest["backpressure"]["pause_selfplay"] is True
    assert manifest["backpressure"]["pause_reason"] == "training"
    assert manifest["backpressure"]["stale_games"] == 96


def test_iteration_pause_metrics_reports_percent_paused() -> None:
    metrics = _iteration_pause_metrics(
        iteration_started_at=10.0,
        iteration_finished_at=20.0,
        pause_started_at=16.0,
        pause_active=True,
    )
    assert metrics["iteration_elapsed_s"] == 10.0
    assert metrics["paused_seconds"] == 4.0
    assert metrics["paused_fraction"] == 0.4
    assert metrics["paused_percent"] == 40.0


def test_iteration_pause_metrics_zero_when_not_paused() -> None:
    metrics = _iteration_pause_metrics(
        iteration_started_at=10.0,
        iteration_finished_at=20.0,
        pause_started_at=None,
        pause_active=False,
    )
    assert metrics["paused_seconds"] == 0.0
    assert metrics["paused_fraction"] == 0.0
    assert metrics["paused_percent"] == 0.0


def test_distributed_iteration_retries_without_fresh_games() -> None:
    assert _should_retry_iteration_without_games(total_games_generated=0)
    assert not _should_retry_iteration_without_games(total_games_generated=1)


def test_selfplay_winrate_raw_is_none_without_games() -> None:
    assert _blended_winrate_raw_or_none(wins=0, draws=0, losses=0) is None
    assert _blended_winrate_raw_or_none(wins=3, draws=1, losses=0) == 0.875


def test_manifest_poll_headers_include_worker_state() -> None:
    headers = _manifest_poll_headers(
        worker_id="worker-123",
        lease_id="lease-456",
        state="paused_selfplay",
        elapsed_s=1.5,
    )
    assert headers["X-CAE-Worker-ID"] == "worker-123"
    assert headers["X-CAE-Worker-Lease-ID"] == "lease-456"
    assert headers["X-CAE-Worker-State"] == "paused_selfplay"
    assert headers["X-CAE-Worker-State-Elapsed-S"] == "1.5"


def test_distributed_ingest_budget_uses_matching_positions_not_stale_backlog(tmp_path: Path) -> None:
    inbox_dir = tmp_path / "inbox"
    processed_dir = tmp_path / "processed"
    worker_dir = inbox_dir / "worker_00"
    worker_dir.mkdir(parents=True)

    stale_path = worker_dir / "00_stale.npz"
    save_npz_arrays(
        stale_path,
        arrs={
            "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
            "policy_target": np.pad(
                np.ones((2, 1), dtype=np.float32),
                ((0, 0), (0, 4671)),
            ),
            "wdl_target": np.zeros((2,), dtype=np.int8),
            "priority": np.ones((2,), dtype=np.float32),
            "has_policy": np.ones((2,), dtype=np.uint8),
        },
        meta={
            "model_sha256": "old-sha",
            "games": 1,
            "positions": 120_000,
        },
    )
    fresh_path = worker_dir / "99_fresh.npz"
    save_npz_arrays(
        fresh_path,
        arrs={
            "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
            "policy_target": np.pad(
                np.ones((2, 1), dtype=np.float32),
                ((0, 0), (0, 4671)),
            ),
            "wdl_target": np.zeros((2,), dtype=np.int8),
            "priority": np.ones((2,), dtype=np.float32),
            "has_policy": np.ones((2,), dtype=np.uint8),
        },
        meta={
            "model_sha256": "fresh-sha",
            "games": 1,
            "positions": 2_000,
        },
    )

    rng = np.random.default_rng(0)
    buf = DiskReplayBuffer(
        256,
        shard_dir=tmp_path / "replay",
        rng=rng,
        shuffle_cap=64,
        shard_size=8,
    )
    holdout = ArrayReplayBuffer(32, rng=np.random.default_rng(1))

    summary = _ingest_distributed_selfplay(
        buf=buf,
        holdout_buf=holdout,
        holdout_frac=0.0,
        holdout_frozen=False,
        inbox_dir=inbox_dir,
        processed_dir=processed_dir,
        target_games=1,
        accepted_model_shas={"fresh-sha"},
        wait_timeout_s=0.1,
        poll_seconds=0.01,
        rng=np.random.default_rng(2),
    )

    assert summary["positions_replay_added"] == 122_000
    assert summary["matching_positions"] == 2_000
    assert summary["stale_positions"] == 120_000

    budget = _compute_train_step_budget(
        positions_added=int(summary["matching_positions"]),
        imported_samples=0,
        replay_size=50_000,
        batch_size=256,
        accum_steps=4,
        base_max_steps=100,
        train_window_fraction=0.10,
    )
    assert budget["target_sample_budget"] == 5_000
    assert budget["steps"] == 5


def test_quarantine_inbox_shards_moves_preexisting_resume_backlog(tmp_path: Path) -> None:
    inbox_dir = tmp_path / "inbox" / "worker_00"
    inbox_dir.mkdir(parents=True)
    shard_path = inbox_dir / "leftover.npz"
    save_npz_arrays(
        shard_path,
        arrs={
            "x": np.zeros((1, 146, 8, 8), dtype=np.float32),
            "policy_target": np.pad(
                np.ones((1, 1), dtype=np.float32),
                ((0, 0), (0, 4671)),
            ),
            "wdl_target": np.zeros((1,), dtype=np.int8),
            "priority": np.ones((1,), dtype=np.float32),
            "has_policy": np.ones((1,), dtype=np.uint8),
        },
    )

    result = _quarantine_inbox_shards(
        inbox_dir=tmp_path / "inbox",
        processed_dir=tmp_path / "processed",
        reason="checkpoint_resume",
    )

    assert result["moved_shards"] == 1
    moved = list((tmp_path / "processed" / "_quarantine").glob("checkpoint_resume_*/*/*.npz"))
    assert len(moved) == 1
    assert not shard_path.exists()
