from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from chess_anti_engine.tune.trainable import (
    _iteration_pause_metrics,
    _publish_distributed_trial_state,
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
        random_move_prob=0.25,
        skill_level=20,
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
