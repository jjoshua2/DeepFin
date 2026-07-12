from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import torch

import chess_anti_engine.tune.distributed_runtime as distributed_runtime
from chess_anti_engine.model import ModelConfig
from chess_anti_engine.moves.encode import POLICY_SIZE
from chess_anti_engine.replay import ArrayReplayBuffer
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import LOCAL_SHARD_SUFFIX, save_local_shard_arrays
from chess_anti_engine.server.app import create_app
from chess_anti_engine.tune.distributed_runtime import (
    _ingest_distributed_selfplay,
    _publish_distributed_trial_state,
    _quarantine_inbox_shards,
)
from chess_anti_engine.tune.trainable_init import _apply_donor_config_overlay
from chess_anti_engine.tune.trainable_metrics import (
    _compute_train_step_budget,
    _curriculum_winrate_raw_or_none,
    _iteration_pause_metrics,
    _should_retry_iteration_without_games,
)
from chess_anti_engine.worker import _manifest_poll_headers


def _one_hot_policy_rows(n: int) -> np.ndarray:
    """(n, POLICY_SIZE) rows with all mass on move 0.

    Built by explicit assignment rather than np.pad: newer numpy stubs
    reject the tuple-of-tuples pad_width overload under basedpyright.
    """
    policy = np.zeros((n, POLICY_SIZE), dtype=np.float32)
    policy[:, 0] = 1.0
    return policy


class _FakeTrainer:
    def export_swa(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {}}, str(path))


class _CountingTrainer:
    def __init__(self) -> None:
        self.exports = 0

    def export_swa(self, path: Path) -> None:
        self.exports += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"export-{self.exports}".encode("ascii"))


class _WeightTrainer:
    w_policy = 0.0
    w_moves_left = 0.0
    sf_wdl_frac = 0.0
    search_wdl_frac = 0.0


def _get_asgi_json(app, path: str, *, headers: dict[str, str]) -> dict:
    async def _run() -> dict:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(path, headers=headers)
            response.raise_for_status()
            return dict(response.json())

    return asyncio.run(_run())


def _model_cfg() -> ModelConfig:
    return ModelConfig(
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


def test_publish_distributed_trial_state_includes_pause_selfplay(tmp_path: Path) -> None:
    trainer = _FakeTrainer()
    model_cfg = _model_cfg()

    _publish_distributed_trial_state(
        trainer=trainer,
        config={
            "selfplay_batch": 16,
            "max_plies": 240,
            "mcts": "gumbel",
            "fast_simulations": 8,
            "sf_move_nodes": 10000,
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
    assert manifest["recommended_worker"]["sf_move_nodes"] == 10000
    assert manifest["recommended_worker"]["slot_oversubscribe"] == 1.0
    assert manifest["encoding"]["policy_encoding"] == "lc0_1858"
    assert manifest["recommended_worker"]["pause_selfplay"] is True
    assert manifest["recommended_worker"]["pause_reason"] == "training"
    assert manifest["backpressure"]["pause_selfplay"] is True
    assert manifest["backpressure"]["pause_reason"] == "training"
    assert manifest["backpressure"]["stale_games"] == 96


def test_publish_distributed_trial_state_sets_stale_pause_target(tmp_path: Path) -> None:
    trainer = _FakeTrainer()
    model_cfg = _model_cfg()

    _publish_distributed_trial_state(
        trainer=trainer,
        config={
            "games_per_iter": 1000,
            "distributed_prev_model_max_fraction": 0.5,
        },
        model_cfg=model_cfg,
        server_root=tmp_path,
        trial_id="trial_00000",
        training_iteration=7,
        trainer_step=123,
        sf_nodes=1000,
        mcts_simulations=64,
    )

    manifest_path = tmp_path / "trials" / "trial_00000" / "publish" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["recommended_worker"]["pause_selfplay"] is False
    assert manifest["backpressure"]["stale_pause_target_games"] == 500
    assert manifest["backpressure"]["stale_pause_model_sha"] == manifest["model"]["sha256"]


def test_manifest_pauses_when_stale_backlog_target_reached(tmp_path: Path) -> None:
    trainer = _FakeTrainer()
    model_cfg = _model_cfg()
    _publish_distributed_trial_state(
        trainer=trainer,
        config={
            "games_per_iter": 1000,
            "distributed_prev_model_max_fraction": 0.5,
        },
        model_cfg=model_cfg,
        server_root=tmp_path,
        trial_id="trial_00000",
        training_iteration=7,
        trainer_step=123,
        sf_nodes=1000,
        mcts_simulations=64,
    )
    manifest_path = tmp_path / "trials" / "trial_00000" / "publish" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_sha = manifest["model"]["sha256"]

    policy = np.zeros((1, POLICY_SIZE), dtype=np.float16)
    policy[0, 0] = 1.0
    inbox_shard = tmp_path / "trials" / "trial_00000" / "inbox" / "_compacted" / f"queued{LOCAL_SHARD_SUFFIX}"
    save_local_shard_arrays(
        inbox_shard,
        arrs={
            "x": np.zeros((1, 146, 8, 8), dtype=np.float16),
            "policy_target": policy,
            "wdl_target": np.zeros((1,), dtype=np.int8),
            "priority": np.ones((1,), dtype=np.float32),
            "has_policy": np.ones((1,), dtype=np.uint8),
        },
        meta={"model_sha256": model_sha, "games": 500, "positions": 1},
    )

    served = _get_asgi_json(
        create_app(server_root=tmp_path),
        "/v1/trials/trial_00000/manifest",
        headers=_manifest_poll_headers(worker_id="test-worker"),
    )

    assert served["recommended_worker"]["pause_selfplay"] is True
    assert "stale backlog target reached" in served["recommended_worker"]["pause_reason"]
    assert served["backpressure"]["stale_pause_queued_games"] == 500
    # Dynamic pause is response-only; the persisted learner manifest remains
    # unpaused so the pause naturally clears after ingest drains the backlog.
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["recommended_worker"]["pause_selfplay"] is False


def test_manifest_pauses_when_old_model_backlog_target_reached(tmp_path: Path) -> None:
    trainer = _FakeTrainer()
    model_cfg = _model_cfg()
    _publish_distributed_trial_state(
        trainer=trainer,
        config={
            "games_per_iter": 1000,
            "distributed_prev_model_max_fraction": 0.5,
        },
        model_cfg=model_cfg,
        server_root=tmp_path,
        trial_id="trial_00000",
        training_iteration=7,
        trainer_step=123,
        sf_nodes=1000,
        mcts_simulations=64,
    )

    policy = np.zeros((1, POLICY_SIZE), dtype=np.float16)
    policy[0, 0] = 1.0
    old_sha = "a" * 64
    inbox_shard = tmp_path / "trials" / "trial_00000" / "inbox" / "_compacted" / f"old{LOCAL_SHARD_SUFFIX}"
    save_local_shard_arrays(
        inbox_shard,
        arrs={
            "x": np.zeros((1, 146, 8, 8), dtype=np.float16),
            "policy_target": policy,
            "wdl_target": np.zeros((1,), dtype=np.int8),
            "priority": np.ones((1,), dtype=np.float32),
            "has_policy": np.ones((1,), dtype=np.uint8),
        },
        meta={"model_sha256": old_sha, "games": 500, "positions": 1},
    )

    served = _get_asgi_json(
        create_app(server_root=tmp_path),
        "/v1/trials/trial_00000/manifest",
        headers=_manifest_poll_headers(worker_id="test-worker"),
    )

    assert served["recommended_worker"]["pause_selfplay"] is True
    assert "queued old-model games" in served["recommended_worker"]["pause_reason"]
    assert served["backpressure"]["stale_pause_queued_games"] == 500
    assert served["backpressure"]["stale_pause_stale_queued_games"] == 500
    assert served["backpressure"]["stale_pause_current_queued_games"] == 0


def test_donor_config_overlay_copies_sf_wdl_frac() -> None:
    trainer = _WeightTrainer()
    config: dict = {}

    _apply_donor_config_overlay(
        config,
        {
            "w_policy": 1.7,
            "w_moves_left": 0.09,
            "sf_wdl_frac": 0.37,
            "search_wdl_frac": 0.42,
        },
        trainer,
    )

    assert config["w_policy"] == 1.7
    assert config["w_moves_left"] == 0.09
    assert config["sf_wdl_frac"] == 0.37
    assert config["search_wdl_frac"] == 0.42
    assert trainer.w_policy == 1.7
    assert trainer.w_moves_left == 0.09
    assert trainer.sf_wdl_frac == 0.37
    assert trainer.search_wdl_frac == 0.42


def test_publish_reuses_existing_model_when_resume_step_matches(tmp_path: Path) -> None:
    trainer = _CountingTrainer()
    model_cfg = _model_cfg()

    first_sha = _publish_distributed_trial_state(
        trainer=trainer,
        config={},
        model_cfg=model_cfg,
        server_root=tmp_path,
        trial_id="trial_00000",
        training_iteration=7,
        trainer_step=123,
        sf_nodes=1000,
        mcts_simulations=64,
    )
    second_sha = _publish_distributed_trial_state(
        trainer=trainer,
        config={},
        model_cfg=model_cfg,
        server_root=tmp_path,
        trial_id="trial_00000",
        training_iteration=8,
        trainer_step=123,
        sf_nodes=1000,
        mcts_simulations=64,
        reuse_existing_model_for_same_step=True,
    )

    manifest_path = tmp_path / "trials" / "trial_00000" / "publish" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert trainer.exports == 1
    assert second_sha == first_sha
    assert manifest["training_iteration"] == 8
    assert manifest["model"]["sha256"] == first_sha


def test_publish_exports_new_model_when_resume_step_changes(tmp_path: Path) -> None:
    trainer = _CountingTrainer()
    model_cfg = _model_cfg()

    first_sha = _publish_distributed_trial_state(
        trainer=trainer,
        config={},
        model_cfg=model_cfg,
        server_root=tmp_path,
        trial_id="trial_00000",
        training_iteration=7,
        trainer_step=123,
        sf_nodes=1000,
        mcts_simulations=64,
    )
    second_sha = _publish_distributed_trial_state(
        trainer=trainer,
        config={},
        model_cfg=model_cfg,
        server_root=tmp_path,
        trial_id="trial_00000",
        training_iteration=8,
        trainer_step=124,
        sf_nodes=1000,
        mcts_simulations=64,
        reuse_existing_model_for_same_step=True,
    )

    assert trainer.exports == 2
    assert second_sha != first_sha


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
    assert _curriculum_winrate_raw_or_none(wins=0, draws=0, losses=0) is None
    assert _curriculum_winrate_raw_or_none(wins=3, draws=1, losses=0) == 0.875


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

    stale_path = worker_dir / f"00_stale{LOCAL_SHARD_SUFFIX}"
    save_local_shard_arrays(
        stale_path,
        arrs={
            "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
            "policy_target": _one_hot_policy_rows(2),
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
    fresh_path = worker_dir / f"99_fresh{LOCAL_SHARD_SUFFIX}"
    save_local_shard_arrays(
        fresh_path,
        arrs={
            "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
            "policy_target": _one_hot_policy_rows(2),
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


def test_train_step_budget_views_targeting_overrides_window_fraction() -> None:
    """train_views_per_position holds samples/ingest fixed instead of tracking
    the window: budget = views * fresh positions, NOT fraction * replay_size."""
    budget = _compute_train_step_budget(
        positions_added=12_000,
        imported_samples=0,
        replay_size=1_000_000,
        batch_size=512,
        accum_steps=1,
        base_max_steps=800,
        train_window_fraction=0.04,   # would give 40_000 samples — must be ignored
        train_views_per_position=2.5,
    )
    assert budget["target_sample_budget"] == 30_000  # 2.5 * 12_000
    assert budget["steps"] == 59  # ceil(30_000 / 512)
    # window fraction target still reported for metrics continuity
    assert budget["window_target_samples"] == 40_000


def test_train_step_budget_views_targeting_scales_with_ingest() -> None:
    """4x more ingest (e.g. fast-ply value rows) -> 4x the step budget at the
    same views target: the reuse ratio stays invariant."""
    def budget_for(positions_added: int) -> dict[str, int]:
        return _compute_train_step_budget(
            positions_added=positions_added,
            imported_samples=0, replay_size=1_000_000, batch_size=512,
            accum_steps=1, base_max_steps=10_000, train_window_fraction=0.04,
            train_views_per_position=2.5,
        )

    small = budget_for(12_000)
    large = budget_for(48_000)
    assert large["target_sample_budget"] == 4 * small["target_sample_budget"]


def test_train_step_budget_views_targeting_is_not_capped_by_base_max_steps() -> None:
    """Views mode owns the step budget: base_max_steps (train_steps) must not
    clamp it, or the fixed views/position contract silently breaks exactly
    when ingest grows. The budget is proportional to fresh ingest, so it is
    self-limiting."""
    budget = _compute_train_step_budget(
        positions_added=200_000,   # > 800*512/2.5 = 163_840 fresh rows
        imported_samples=0,
        replay_size=1_000_000,
        batch_size=512,
        accum_steps=1,
        base_max_steps=800,        # exp_throughput_views train_steps
        train_window_fraction=0.04,
        train_views_per_position=2.5,
    )
    assert budget["target_sample_budget"] == 500_000  # 2.5 * 200_000
    assert budget["steps"] == 977  # ceil(500_000 / 512), NOT min(977, 800)


def test_train_step_budget_views_zero_keeps_window_fraction_behavior() -> None:
    budget = _compute_train_step_budget(
        positions_added=2_000,
        imported_samples=0,
        replay_size=50_000,
        batch_size=256,
        accum_steps=4,
        base_max_steps=100,
        train_window_fraction=0.10,
        train_views_per_position=0.0,
    )
    assert budget["target_sample_budget"] == 5_000
    assert budget["steps"] == 5


def test_distributed_ingest_timeout_does_not_wait_for_empty_inbox_after_prev_cap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inbox_dir = tmp_path / "inbox"
    processed_dir = tmp_path / "processed"
    worker_dir = inbox_dir / "worker_00"
    worker_dir.mkdir(parents=True)

    arrs = {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "policy_target": _one_hot_policy_rows(2),
        "wdl_target": np.zeros((2,), dtype=np.int8),
        "priority": np.ones((2,), dtype=np.float32),
        "has_policy": np.ones((2,), dtype=np.uint8),
    }
    meta = {
        "model_sha256": "prev-sha",
        "games": 2,
        "positions": 2,
    }
    prev_path = worker_dir / f"00_prev{LOCAL_SHARD_SUFFIX}"
    save_local_shard_arrays(prev_path, arrs=arrs, meta=meta)

    class _Prefetcher:
        def drain(self):
            return [(prev_path, arrs, meta)]

    def _fail_iter(_inbox_dir: Path):
        raise AssertionError("deadline should be checked before polling for more shards")

    monkeypatch.setattr(distributed_runtime, "_iter_shard_paths_nested", _fail_iter)

    rng = np.random.default_rng(0)
    summary = _ingest_distributed_selfplay(
        buf=DiskReplayBuffer(
            256,
            shard_dir=tmp_path / "replay",
            rng=rng,
            shuffle_cap=64,
            shard_size=8,
        ),
        holdout_buf=ArrayReplayBuffer(32, rng=np.random.default_rng(1)),
        holdout_frac=0.0,
        holdout_frozen=False,
        inbox_dir=inbox_dir,
        processed_dir=processed_dir,
        target_games=10,
        accepted_model_shas={"fresh-sha", "prev-sha"},
        prev_model_sha="prev-sha",
        prev_model_max_fraction=0.2,
        wait_timeout_s=-1.0,
        poll_seconds=0.01,
        rng=np.random.default_rng(2),
        min_games_fraction=0.2,
        prefetcher=_Prefetcher(),
    )

    assert summary["matching_games"] == 2
    assert summary["stale_games"] == 0


def test_prefetched_shard_missing_from_inbox_is_not_reingested(tmp_path: Path) -> None:
    inbox_dir = tmp_path / "inbox"
    processed_dir = tmp_path / "processed"
    stale_path = inbox_dir / "worker_00" / f"stale{LOCAL_SHARD_SUFFIX}"
    arrs = {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "policy_target": _one_hot_policy_rows(2),
        "wdl_target": np.zeros((2,), dtype=np.int8),
        "priority": np.ones((2,), dtype=np.float32),
        "has_policy": np.ones((2,), dtype=np.uint8),
    }
    meta = {"model_sha256": "fresh-sha", "games": 1, "positions": 2}

    class _Prefetcher:
        def drain(self):
            return [(stale_path, arrs, meta)]

    summary = _ingest_distributed_selfplay(
        buf=DiskReplayBuffer(
            256,
            shard_dir=tmp_path / "replay",
            rng=np.random.default_rng(0),
            shuffle_cap=64,
            shard_size=8,
        ),
        holdout_buf=ArrayReplayBuffer(32, rng=np.random.default_rng(1)),
        holdout_frac=0.0,
        holdout_frozen=False,
        inbox_dir=inbox_dir,
        processed_dir=processed_dir,
        target_games=1,
        accepted_model_shas={"fresh-sha"},
        wait_timeout_s=-1.0,
        poll_seconds=0.01,
        rng=np.random.default_rng(2),
        min_games_fraction=1.0,
        prefetcher=_Prefetcher(),
    )

    assert summary["matching_games"] == 0
    assert summary["positions_replay_added"] == 0


def test_quarantine_inbox_shards_moves_preexisting_resume_backlog(tmp_path: Path) -> None:
    inbox_dir = tmp_path / "inbox" / "worker_00"
    inbox_dir.mkdir(parents=True)
    shard_path = inbox_dir / f"leftover{LOCAL_SHARD_SUFFIX}"
    save_local_shard_arrays(
        shard_path,
        arrs={
            "x": np.zeros((1, 146, 8, 8), dtype=np.float32),
            "policy_target": _one_hot_policy_rows(1),
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
    moved = list((tmp_path / "processed" / "_quarantine").glob(f"checkpoint_resume_*/*/*{LOCAL_SHARD_SUFFIX}"))
    assert len(moved) == 1
    assert not shard_path.exists()


def test_train_step_budget_views_drought_falls_back_to_window_floor() -> None:
    """Trickle ingest (< one batch of fresh data) must not collapse training to
    ~1 step: the window-fraction floor the pre-views budget guaranteed kicks
    back in."""
    budget = _compute_train_step_budget(
        positions_added=100,
        imported_samples=0,
        replay_size=1_000_000,
        batch_size=512,
        accum_steps=1,
        base_max_steps=800,
        train_window_fraction=0.04,
        train_views_per_position=2.5,
    )
    assert budget["target_sample_budget"] == 40_000  # window floor, not 250
    assert budget["steps"] == 79


def test_train_step_budget_views_counts_imported_samples_at_one_view() -> None:
    """Imported (shared/donor) samples are re-used history: they get exactly
    one pass, like the pre-views budget gave them — not views x the import."""
    budget = _compute_train_step_budget(
        positions_added=12_000,
        imported_samples=400_000,
        replay_size=1_000_000,
        batch_size=512,
        accum_steps=1,
        base_max_steps=800,
        train_window_fraction=0.04,
        train_views_per_position=2.5,
    )
    # 2.5 * 12_000 fresh + 400_000 imported (1 view), NOT 2.5 * 412_000.
    assert budget["target_sample_budget"] == 430_000
    assert budget["steps"] == 840  # uncapped on import iterations, as before


def test_train_step_budget_views_drought_fallback_respects_step_cap() -> None:
    """Codex P2 (PR #91 post-merge): the drought fallback sizes the budget by
    window fraction, so it must also take the window-fraction step cap —
    views mode's cap bypass is only for the proportional (fresh-driven) path."""
    budget = _compute_train_step_budget(
        positions_added=100,
        imported_samples=0,
        replay_size=1_000_000,
        batch_size=512,
        accum_steps=1,
        base_max_steps=50,   # cap below the 79 steps the window floor implies
        train_window_fraction=0.04,
        train_views_per_position=2.5,
    )
    assert budget["target_sample_budget"] == 40_000
    assert budget["steps"] == 50  # capped, not 79


# ── Server seed-dole gate (opening_fen_dole_per_iter) ────────────────────────

_DOLE_SEED_FEN = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2"


def _get_app_bytes(app, path: str) -> bytes:
    """GET `path` against ONE app instance and return the raw response body."""
    async def _run() -> bytes:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(path)
            response.raise_for_status()
            return response.content

    return asyncio.run(_run())


def _poll_app_n(app, path: str, *, headers: dict[str, str], n: int) -> list[dict]:
    """GET `path` n times against ONE app instance (shared dole-gate state)."""
    async def _run() -> list[dict]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            out: list[dict] = []
            for _ in range(n):
                response = await client.get(path, headers=headers)
                response.raise_for_status()
                out.append(dict(response.json()))
            return out

    return asyncio.run(_run())


def _publish_dole_trial(tmp_path: Path, *, training_iteration: int, dole: int, fen_path: Path) -> None:
    _publish_distributed_trial_state(
        trainer=_FakeTrainer(),
        config={
            "opening_fen_dole_per_iter": dole,
            "opening_fen_list_path": str(fen_path),
        },
        model_cfg=_model_cfg(),
        server_root=tmp_path,
        trial_id="trial_00000",
        training_iteration=training_iteration,
        trainer_step=training_iteration,
        sf_nodes=1000,
        mcts_simulations=64,
    )


def test_manifest_doles_fen_seeds_once_per_iteration(tmp_path: Path) -> None:
    fen_path = tmp_path / "blindspot.txt"
    fen_path.write_text(_DOLE_SEED_FEN + "\n", encoding="utf-8")
    _publish_dole_trial(tmp_path, training_iteration=7, dole=1, fen_path=fen_path)

    app = create_app(server_root=tmp_path)
    headers = _manifest_poll_headers(worker_id="test-worker")
    polls = _poll_app_n(app, "/v1/trials/trial_00000/manifest", headers=headers, n=3)

    # Plumbing sanity: dole knob rides recommended_worker; the FEN list is an asset.
    assert polls[0]["recommended_worker"]["opening_fen_dole_per_iter"] == 1
    assert isinstance(polls[0].get("opening_fen_list"), dict)
    # Exactly one poll this iteration wins the dole.
    assert [p["dole_fen_seeds"] for p in polls] == [True, False, False]

    # Next iteration re-opens the gate for exactly one poll (same app instance).
    _publish_dole_trial(tmp_path, training_iteration=8, dole=1, fen_path=fen_path)
    polls2 = _poll_app_n(app, "/v1/trials/trial_00000/manifest", headers=headers, n=2)
    assert [p["dole_fen_seeds"] for p in polls2] == [True, False]


def test_manifest_no_dole_when_disabled(tmp_path: Path) -> None:
    fen_path = tmp_path / "blindspot.txt"
    fen_path.write_text(_DOLE_SEED_FEN + "\n", encoding="utf-8")
    _publish_dole_trial(tmp_path, training_iteration=3, dole=0, fen_path=fen_path)  # dole off

    app = create_app(server_root=tmp_path)
    polls = _poll_app_n(
        app, "/v1/trials/trial_00000/manifest",
        headers=_manifest_poll_headers(worker_id="test-worker"), n=2,
    )
    assert all(p["dole_fen_seeds"] is False for p in polls)


def test_manifest_no_dole_without_published_fen_list(tmp_path: Path) -> None:
    # dole enabled but the FEN-list file is absent → no opening_fen_list asset →
    # server never doles (nothing for the worker to load).
    _publish_distributed_trial_state(
        trainer=_FakeTrainer(),
        config={"opening_fen_dole_per_iter": 1, "opening_fen_list_path": str(tmp_path / "missing.txt")},
        model_cfg=_model_cfg(),
        server_root=tmp_path,
        trial_id="trial_00000",
        training_iteration=5,
        trainer_step=5,
        sf_nodes=1000,
        mcts_simulations=64,
    )
    app = create_app(server_root=tmp_path)
    polls = _poll_app_n(
        app, "/v1/trials/trial_00000/manifest",
        headers=_manifest_poll_headers(worker_id="test-worker"), n=1,
    )
    assert polls[0]["dole_fen_seeds"] is False
    assert "opening_fen_list" not in polls[0]


def test_manifest_no_dole_while_paused(tmp_path: Path) -> None:
    # A paused poll must NOT consume the single per-iteration dole claim: the
    # worker drops a paused manifest (returns None from _poll_manifest) before it
    # can ingest the seeds, so claiming here would waste the whole batch. The gate
    # stays open so a later unpaused poll THIS iteration still wins it.
    fen_path = tmp_path / "blindspot.txt"
    fen_path.write_text(_DOLE_SEED_FEN + "\n", encoding="utf-8")

    def _publish(*, paused: bool) -> None:
        _publish_distributed_trial_state(
            trainer=_FakeTrainer(),
            config={"opening_fen_dole_per_iter": 1, "opening_fen_list_path": str(fen_path)},
            model_cfg=_model_cfg(),
            server_root=tmp_path,
            trial_id="trial_00000",
            training_iteration=11,
            trainer_step=11,
            sf_nodes=1000,
            mcts_simulations=64,
            pause_selfplay=paused,
            pause_reason="training" if paused else "",
        )

    app = create_app(server_root=tmp_path)
    headers = _manifest_poll_headers(worker_id="test-worker")

    _publish(paused=True)
    paused = _poll_app_n(app, "/v1/trials/trial_00000/manifest", headers=headers, n=3)
    assert paused[0]["recommended_worker"]["pause_selfplay"] is True
    assert all(p["dole_fen_seeds"] is False for p in paused)  # no claim burned

    # Same iteration, now unpaused: the still-unclaimed gate doles to exactly one poll.
    _publish(paused=False)
    unpaused = _poll_app_n(app, "/v1/trials/trial_00000/manifest", headers=headers, n=2)
    assert [p["dole_fen_seeds"] for p in unpaused] == [True, False]


def test_manifest_no_dole_for_non_selfplay_task(tmp_path: Path) -> None:
    # A non-selfplay (arena) task must NOT consume the single per-iteration dole:
    # the worker takes its arena path and never ingests, so claiming would drop
    # the whole batch.
    fen_path = tmp_path / "blindspot.txt"
    fen_path.write_text(_DOLE_SEED_FEN + "\n", encoding="utf-8")
    _publish_dole_trial(tmp_path, training_iteration=9, dole=1, fen_path=fen_path)
    # Flip the published task to arena (the publisher always writes selfplay).
    mf_path = tmp_path / "trials" / "trial_00000" / "publish" / "manifest.json"
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    mf["task"] = {"type": "arena"}
    mf_path.write_text(json.dumps(mf), encoding="utf-8")

    app = create_app(server_root=tmp_path)
    polls = _poll_app_n(
        app, "/v1/trials/trial_00000/manifest",
        headers=_manifest_poll_headers(worker_id="test-worker"), n=2,
    )
    assert all(p["dole_fen_seeds"] is False for p in polls)  # no claim burned on arena


def test_seed_dole_gate_single_winner_under_concurrency() -> None:
    from chess_anti_engine.server.app import _SeedDoleGate

    gate = _SeedDoleGate()

    async def _burst(trial_key: str, it: int, n: int) -> list[bool]:
        return list(await asyncio.gather(*[gate.claim(trial_key, it) for _ in range(n)]))

    # 64 simultaneous polls at the iteration boundary → exactly one winner.
    assert sum(asyncio.run(_burst("t", 5, 64))) == 1
    # Re-polling the same iteration yields no further winners.
    assert sum(asyncio.run(_burst("t", 5, 8))) == 0
    # Advancing the iteration re-opens exactly one winner.
    assert sum(asyncio.run(_burst("t", 6, 16))) == 1
    # A stale (lower) iteration never wins again.
    assert sum(asyncio.run(_burst("t", 5, 4))) == 0
    # Per-trial isolation: a different trial has its own counter.
    assert asyncio.run(gate.claim("other", 5)) is True


def test_seed_dole_gate_persists_across_restart(tmp_path: Path) -> None:
    from chess_anti_engine.server.app import _SeedDoleGate

    state = tmp_path / "seed_dole_gate.json"
    gate = _SeedDoleGate(state_path=state)
    assert asyncio.run(gate.claim("t", 7)) is True
    # A fresh gate (server restart) reloads the claimed iteration from disk and
    # must NOT re-hand iteration 7's dole (which would double the batch).
    gate2 = _SeedDoleGate(state_path=state)
    assert asyncio.run(gate2.claim("t", 7)) is False
    assert asyncio.run(gate2.claim("t", 8)) is True  # a newer iteration still wins


# ── opening_fen_list_path live reload (no restart needed) ────────────────────


def test_opening_fen_list_path_swap_takes_effect_without_restart(tmp_path: Path) -> None:
    """A yaml opening_fen_list_path change must be servable on the NEXT
    manifest publish alone — this is the whole point of excluding it from
    _LAUNCH_FIXED_ASSET_PATH_KEYS (unlike opening_book_path, which really is
    frozen to the server's launch-time value)."""
    fen_a = tmp_path / "seeds_a.txt"
    fen_a.write_text(_DOLE_SEED_FEN + "\n", encoding="utf-8")
    _publish_distributed_trial_state(
        trainer=_FakeTrainer(), config={"opening_fen_list_path": str(fen_a)},
        model_cfg=_model_cfg(), server_root=tmp_path, trial_id="trial_00000",
        training_iteration=1, trainer_step=1, sf_nodes=1000, mcts_simulations=64,
    )
    app = create_app(server_root=tmp_path)
    assert _get_app_bytes(app, "/v1/trials/trial_00000/opening_fen_list") == fen_a.read_bytes()

    # Different path, different content — no server restart, just a second publish.
    fen_b = tmp_path / "seeds_b.txt"
    other_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fen_b.write_text(other_fen + "\n", encoding="utf-8")
    _publish_distributed_trial_state(
        trainer=_FakeTrainer(), config={"opening_fen_list_path": str(fen_b)},
        model_cfg=_model_cfg(), server_root=tmp_path, trial_id="trial_00000",
        training_iteration=2, trainer_step=2, sf_nodes=1000, mcts_simulations=64,
    )
    served = _get_app_bytes(app, "/v1/trials/trial_00000/opening_fen_list")
    assert served == fen_b.read_bytes()
    assert served != fen_a.read_bytes()


def test_opening_fen_list_path_inplace_edit_detected(tmp_path: Path) -> None:
    """An in-place content edit at the SAME path must also propagate: this
    exercises the _sha256_cached fix (mtime added to the cache key) — a
    (path, size)-only cache would silently keep serving stale bytes if the
    edit happens not to change the file's byte size."""
    fen_path = tmp_path / "seeds.txt"
    fen_path.write_text(_DOLE_SEED_FEN + "\n", encoding="utf-8")
    _publish_distributed_trial_state(
        trainer=_FakeTrainer(), config={"opening_fen_list_path": str(fen_path)},
        model_cfg=_model_cfg(), server_root=tmp_path, trial_id="trial_00000",
        training_iteration=1, trainer_step=1, sf_nodes=1000, mcts_simulations=64,
    )
    app = create_app(server_root=tmp_path)
    first = _get_app_bytes(app, "/v1/trials/trial_00000/opening_fen_list")
    assert first == fen_path.read_bytes()

    # Overwrite in place with same-length content (same byte size as before —
    # the exact case a (path, size)-keyed cache would miss). Pad to match
    # _DOLE_SEED_FEN's exact length; this is a raw byte-equality check, not
    # a seed-grammar parse, so trailing padding doesn't affect the intent.
    other_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    other_fen = other_fen.ljust(len(_DOLE_SEED_FEN))
    assert len(other_fen) == len(_DOLE_SEED_FEN)
    fen_path.write_text(other_fen + "\n", encoding="utf-8")
    _publish_distributed_trial_state(
        trainer=_FakeTrainer(), config={"opening_fen_list_path": str(fen_path)},
        model_cfg=_model_cfg(), server_root=tmp_path, trial_id="trial_00000",
        training_iteration=2, trainer_step=2, sf_nodes=1000, mcts_simulations=64,
    )
    second = _get_app_bytes(app, "/v1/trials/trial_00000/opening_fen_list")
    assert second == fen_path.read_bytes()
    assert second != first
