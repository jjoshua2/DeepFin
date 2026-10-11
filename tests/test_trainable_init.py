from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.moves import policy_size_for_encoding
from chess_anti_engine.replay.shard import (
    iter_shard_paths,
    local_shard_path,
    save_local_shard_arrays,
)
from chess_anti_engine.train.trainer import Trainer
from chess_anti_engine.tune.trainable_config_ops import _apply_lr_gamma_weights
from chess_anti_engine.tune.trainable_init import (
    _init_replay_buffers,
    _restore_checkpoint_or_salvage,
    _restore_from_ray_checkpoint,
    _seed_replay_from_shared_shards,
)
from chess_anti_engine.tune.trial_config import RestoreResult, TrialConfig


def _arrays(policy_size: int, *, x_planes: int = 146) -> dict[str, np.ndarray]:
    policy = np.zeros((2, policy_size), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "x": np.zeros((2, x_planes, 8, 8), dtype=np.float32),
        "policy_target": policy,
        "wdl_target": np.zeros((2,), dtype=np.int8),
        "priority": np.ones((2,), dtype=np.float32),
        "has_policy": np.ones((2,), dtype=np.uint8),
    }


def test_seed_replay_from_shared_shards_skips_policy_width_mismatch(tmp_path) -> None:
    shared = tmp_path / "shared"
    replay = tmp_path / "trial" / "replay"
    save_local_shard_arrays(local_shard_path(shared, 0), arrs=_arrays(4672), meta={"positions": 2})
    save_local_shard_arrays(local_shard_path(shared, 1), arrs=_arrays(1858), meta={"positions": 2})
    tc = TrialConfig(shared_shards_dir=str(shared), policy_encoding="lc0_1858")

    copied = _seed_replay_from_shared_shards(
        tc=tc,
        restore=RestoreResult(),
        replay_shard_dir=replay,
    )

    paths = iter_shard_paths(replay)
    assert copied == 1
    assert [path.name for path in paths] == [local_shard_path(shared, 1).name]


@pytest.mark.parametrize(
    ("trial_meta", "rng_state"),
    [
        pytest.param(None, None, id="missing-metadata-and-rng"),
        pytest.param({"current_window": 0}, {"bit_generator": "not-a-generator"}, id="zero-window-rejected-rng"),
    ],
)
def test_missing_or_rejected_rng_sidecars_still_open_durable_replay(
    tmp_path: Path,
    trial_meta: dict | None,
    rng_state: dict | None,
) -> None:
    """Legacy checkpoints without metadata and corrupt RNG remain safe to open."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "trainer.pt").write_bytes(b"synthetic trainer checkpoint")
    if trial_meta is not None:
        (checkpoint_dir / "trial_meta.json").write_text(json.dumps(trial_meta), encoding="utf-8")
    if rng_state is not None:
        (checkpoint_dir / "rng_state.json").write_text(json.dumps(rng_state), encoding="utf-8")

    trial_dir = tmp_path / "trial-a"
    work_dir = tmp_path / "work"
    trial_dir.mkdir()
    work_dir.mkdir()
    rng = np.random.default_rng(999)
    original_rng_state = rng.bit_generator.state
    restore, rng = _restore_checkpoint_or_salvage(
        config={"optimizer": "aurora"},
        trainer=_TrainerLoadProbe(),
        device="cpu",
        trial_id="trial-a",
        trial_dir=trial_dir,
        base_seed=123,
        active_seed=999,
        rng=rng,
        ckpt=_LocalCheckpoint(checkpoint_dir),
    )
    assert restore.startup_source == "checkpoint"
    assert not restore.cross_trial_restore
    assert restore.restored_window == 0
    assert not restore.sampling_rng_restored
    assert rng.bit_generator.state == original_rng_state

    tc = TrialConfig(
        replay_window_start=2,
        replay_window_max=20,
        shard_size=2,
        policy_encoding="lc0_1858",
    )
    replay_dir = trial_dir / "replay_shards"
    arrays = _arrays(
        policy_size_for_encoding(tc.policy_encoding),
        x_planes=input_plane_count(tc.input_extra_features),
    )
    for index in range(2):
        save_local_shard_arrays(local_shard_path(replay_dir, index), arrs=arrays, meta={"positions": 2})

    buf, _, current_window, _, _ = _init_replay_buffers(
        tc=tc,
        config={"optimizer": "aurora"},
        restore=restore,
        trial_dir=trial_dir,
        work_dir=work_dir,
        rng=rng,
        ckpt=_LocalCheckpoint(checkpoint_dir),
    )
    try:
        assert len(iter_shard_paths(replay_dir)) == 2
        assert len(buf) == 4
        assert current_window == buf.capacity == 4
        assert buf.capacity <= tc.replay_window_max
    finally:
        buf.close()


def test_detected_donor_owner_does_not_install_donor_rng_into_replay(
    tmp_path: Path,
) -> None:
    """A Ray donor mismatch forks RNG before the real replay buffer opens."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "trainer.pt").write_bytes(b"synthetic trainer checkpoint")
    donor_rng = np.random.default_rng(41)
    (checkpoint_dir / "rng_state.json").write_text(
        json.dumps(donor_rng.bit_generator.state), encoding="utf-8",
    )
    (checkpoint_dir / "trial_meta.json").write_text(
        json.dumps({"owner_trial_id": "donor-trial", "optimizer": "aurora"}),
        encoding="utf-8",
    )
    trial_dir = tmp_path / "trial-a"
    work_dir = tmp_path / "work"
    trial_dir.mkdir()
    work_dir.mkdir()

    restore, rng = _restore_checkpoint_or_salvage(
        config={"optimizer": "aurora"},
        trainer=_TrainerLoadProbe(),
        device="cpu",
        trial_id="trial-a",
        trial_dir=trial_dir,
        base_seed=123,
        active_seed=999,
        rng=np.random.default_rng(999),
        ckpt=_LocalCheckpoint(checkpoint_dir),
    )
    assert restore.cross_trial_restore
    assert restore.startup_source == "exploit_restore"
    assert not restore.sampling_rng_restored
    assert rng.bit_generator.state != donor_rng.bit_generator.state

    tc = TrialConfig(replay_window_start=2, replay_window_max=20, shard_size=2)
    buf, _, current_window, _, _ = _init_replay_buffers(
        tc=tc,
        config={"optimizer": "aurora"},
        restore=restore,
        trial_dir=trial_dir,
        work_dir=work_dir,
        rng=rng,
        ckpt=_LocalCheckpoint(checkpoint_dir),
    )
    try:
        assert current_window == buf.capacity == 2
        assert buf.capacity <= tc.replay_window_max
    finally:
        buf.close()


def test_salvage_rng_sidecar_is_installed_before_replay_buffer_open(
    tmp_path: Path,
) -> None:
    """Exercise salvage's existing RNG installation through real buffer opening."""
    pool = tmp_path / "pool"
    seed_dir = pool / "seeds" / "slot_000"
    seed_dir.mkdir(parents=True)
    (seed_dir / "trainer.pt").write_bytes(b"synthetic trainer checkpoint")
    saved_rng = np.random.default_rng(41)
    saved_rng_state = saved_rng.bit_generator.state
    (seed_dir / "rng_state.json").write_text(json.dumps(saved_rng_state), encoding="utf-8")
    (seed_dir / "trial_meta.json").write_text(json.dumps({"current_window": 0}), encoding="utf-8")
    (pool / "manifest.json").write_text(
        json.dumps({"entries": [{"slot": 0, "seed_dir": "seeds/slot_000"}]}),
        encoding="utf-8",
    )

    trial_dir = tmp_path / "trial-a"
    work_dir = tmp_path / "work"
    trial_dir.mkdir()
    work_dir.mkdir()
    restore, rng = _restore_checkpoint_or_salvage(
        config={
            "optimizer": "aurora",
            "salvage_seed_pool_dir": str(pool),
            "salvage_restore_full_trainer_state": True,
        },
        trainer=_TrainerLoadProbe(),
        device="cpu",
        trial_id="trial-a",
        trial_dir=trial_dir,
        base_seed=123,
        active_seed=999,
        rng=np.random.default_rng(999),
        ckpt=None,
    )
    assert restore.startup_source == "salvage"
    assert restore.seed_warmstart_used
    assert restore.restored_window == 0
    assert restore.sampling_rng_restored
    assert rng.bit_generator.state == saved_rng_state

    tc = TrialConfig(
        replay_window_start=2,
        replay_window_max=20,
        shard_size=2,
        policy_encoding="lc0_1858",
    )
    replay_dir = seed_dir / "replay_shards"
    arrays = _arrays(
        policy_size_for_encoding(tc.policy_encoding),
        x_planes=input_plane_count(tc.input_extra_features),
    )
    for index in range(2):
        save_local_shard_arrays(local_shard_path(replay_dir, index), arrs=arrays, meta={"positions": 2})

    buf, _, current_window, _, _ = _init_replay_buffers(
        tc=tc,
        config={
            "optimizer": "aurora",
            "salvage_seed_pool_dir": str(pool),
            "salvage_restore_full_trainer_state": True,
        },
        restore=restore,
        trial_dir=trial_dir,
        work_dir=work_dir,
        rng=rng,
        ckpt=None,
    )
    try:
        assert len(iter_shard_paths(trial_dir / "replay_shards")) == 2
        assert len(buf) == 4
        assert current_window == buf.capacity == 4
        assert rng.bit_generator.state == saved_rng_state
    finally:
        buf.close()


def test_saved_window_is_capped_by_a_reduced_max_before_buffer_open(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    trial_dir = tmp_path / "trial-a"
    work_dir = tmp_path / "work"
    trial_dir.mkdir()
    work_dir.mkdir()
    tc = TrialConfig(
        replay_window_start=2,
        replay_window_max=3,
        shard_size=2,
        policy_encoding="lc0_1858",
    )
    replay_dir = trial_dir / "replay_shards"
    arrays = _arrays(
        policy_size_for_encoding(tc.policy_encoding),
        x_planes=input_plane_count(tc.input_extra_features),
    )
    for index in range(2):
        save_local_shard_arrays(local_shard_path(replay_dir, index), arrs=arrays, meta={"positions": 2})

    buf, _, current_window, _, _ = _init_replay_buffers(
        tc=tc,
        config={"optimizer": "aurora"},
        restore=RestoreResult(startup_source="checkpoint", restored_window=100),
        trial_dir=trial_dir,
        work_dir=work_dir,
        rng=np.random.default_rng(41),
        ckpt=_LocalCheckpoint(checkpoint_dir),
    )
    try:
        assert len(iter_shard_paths(replay_dir)) == 1
        assert len(buf) == 2
        assert current_window == buf.capacity == 3
        assert buf.capacity <= tc.replay_window_max
    finally:
        buf.close()


class _LocalCheckpoint:
    def __init__(self, path: Path) -> None:
        self.path = path

    def to_directory(self) -> str:
        return str(self.path)


class _TrainerLoadProbe:
    def __init__(self) -> None:
        self.loaded: Path | None = None

    def load(self, checkpoint: Path) -> None:
        self.loaded = checkpoint


def test_checkpoint_restore_reaches_replay_buffer_without_advancing_rng_or_pruning(tmp_path) -> None:
    """Exercise learner sidecar restore through the real disk-buffer constructor."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "trainer.pt").write_bytes(b"synthetic trainer checkpoint")
    saved_rng = np.random.default_rng(41)
    saved_rng_state = saved_rng.bit_generator.state
    (checkpoint_dir / "rng_state.json").write_text(json.dumps(saved_rng_state), encoding="utf-8")
    (checkpoint_dir / "trial_meta.json").write_text(
        json.dumps({"owner_trial_id": "trial-a", "optimizer": "aurora"}),
        encoding="utf-8",
    )

    trial_dir = tmp_path / "trial-a"
    work_dir = tmp_path / "work"
    trial_dir.mkdir()
    work_dir.mkdir()
    trainer = _TrainerLoadProbe()
    rng = np.random.default_rng(999)
    restore, rng = _restore_checkpoint_or_salvage(
        config={"optimizer": "aurora"},
        trainer=trainer,
        device="cpu",
        trial_id="trial-a",
        trial_dir=trial_dir,
        base_seed=123,
        active_seed=999,
        rng=rng,
        ckpt=_LocalCheckpoint(checkpoint_dir),
    )
    assert trainer.loaded == checkpoint_dir / "trainer.pt"
    assert restore.startup_source == "checkpoint"
    assert not restore.cross_trial_restore
    assert restore.restored_window == 0
    assert restore.sampling_rng_restored
    assert rng.bit_generator.state == saved_rng_state

    tc = TrialConfig(
        replay_window_start=2,
        replay_window_max=20,
        shard_size=2,
        policy_encoding="lc0_1858",
    )
    replay_dir = trial_dir / "replay_shards"
    arrays = _arrays(
        policy_size_for_encoding(tc.policy_encoding),
        x_planes=input_plane_count(tc.input_extra_features),
    )
    for index in range(2):
        save_local_shard_arrays(
            local_shard_path(replay_dir, index),
            arrs=arrays,
            meta={"positions": 2},
        )

    buf, _, current_window, _, _ = _init_replay_buffers(
        tc=tc,
        config={"optimizer": "aurora"},
        restore=restore,
        trial_dir=trial_dir,
        work_dir=work_dir,
        rng=rng,
        ckpt=_LocalCheckpoint(checkpoint_dir),
    )
    try:
        assert len(iter_shard_paths(replay_dir)) == 2
        assert len(buf) == 4
        assert current_window == buf.capacity == 4
        assert buf.capacity <= tc.replay_window_max
        assert rng.bit_generator.state == saved_rng_state
    finally:
        buf.close()


class _LRProbeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ffn = torch.nn.Linear(4, 4)
        self.out_proj = torch.nn.Linear(4, 4)


class _LRProbeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.blocks = torch.nn.ModuleList([_LRProbeBlock()])
        self.head = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {"policy": self.head.weight[:1], "wdl": self.head.bias}


_LR_OLD_PEAK = 3.0e-5


_LR_WARMUP_START = 1.0e-5


_LR_MATRIX_MULTIPLIER = 20.0


_LR_WARMUP_STEPS = 10


_LR_WINDOW_STEPS = 8


def _make_lr_trainer(log_dir: Path, lr: float, *, cycle_steps: int = 0) -> Trainer:
    torch.manual_seed(0)
    return Trainer(
        _LRProbeModel(), device="cpu", lr=lr, optimizer="aurora",
        matrix_optimizer_scope="mlp_out", matrix_lr_multiplier=_LR_MATRIX_MULTIPLIER,
        matrix_weight_decay=1.0e-3, aux_weight_decay=1.0e-4,
        warmup_steps=_LR_WARMUP_STEPS, warmup_lr_start=_LR_WARMUP_START,
        lr_eta_min=_LR_WARMUP_START, lr_schedule="sqrt_release",
        lr_release_cycle_steps=cycle_steps, lr_release_start_frac=0.8,
        lr_release_min_scale=0.1, lr_release_shape="sqrt",
        use_amp=False, use_compile=False, aurora_cuda_graphs=False,
        aurora_polar_steps=1, aurora_polar_dtype="fp32", swa_start=-1,
        log_dir=log_dir, tb_log_interval=10**9, prefetch_batches=False,
    )


def _advance_lr_window(trainer: Trainer) -> None:
    for local_step in range(_LR_WINDOW_STEPS):
        release = trainer._uses_train_window_release_cycle()
        if release:
            trainer._set_train_window_release_lr(local_step=local_step, cycle_steps=_LR_WINDOW_STEPS)
        trainer.opt.zero_grad(set_to_none=True)
        for param in trainer.model.parameters():
            if param.requires_grad:
                param.grad = torch.ones_like(param)
        trainer.opt.step()
        if not release:
            trainer._update_lr()
        trainer.step += 1


def _assert_lr_trainer_equal(left: Trainer, right: Trainer) -> None:
    left_params = dict(left.model.named_parameters())
    right_params = dict(right.model.named_parameters())
    assert left_params.keys() == right_params.keys()
    for name, left_param in left_params.items():
        right_param = right_params[name]
        torch.testing.assert_close(left_param, right_param, rtol=0.0, atol=0.0)
        left_state = left.opt.state.get(left_param, {})
        right_state = right.opt.state.get(right_param, {})
        assert left_state  # Every parameter has taken real optimizer steps.
        assert left_state.keys() == right_state.keys()
        for key, value in left_state.items():
            other = right_state[key]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other, rtol=0.0, atol=0.0)
            else:
                assert value == other
    assert len(left.opt.param_groups) == len(right.opt.param_groups) == 4
    for a, b in zip(left.opt.param_groups, right.opt.param_groups, strict=True):
        assert a["weight_decay"] == b["weight_decay"]


class _LocalRayCheckpoint:
    def __init__(self, path: Path) -> None:
        self.path = path

    def to_directory(self) -> str:
        return str(self.path)


@pytest.mark.parametrize("saved_step", [8, 16, 24])
@pytest.mark.parametrize("peak_scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("cycle_steps", [0, 20])
def test_ray_restore_preserves_active_lr_across_warmup_and_windows(
    tmp_path: Path, saved_step: int, peak_scale: float, cycle_steps: int,
) -> None:
    """Ray restore matches the live rebase for every group and first opt step."""
    source = _make_lr_trainer(tmp_path / "source", _LR_OLD_PEAK, cycle_steps=cycle_steps)
    for _ in range(saved_step // _LR_WINDOW_STEPS):
        _advance_lr_window(source)
    assert source.step == saved_step
    ckpt_dir = tmp_path / "ray-checkpoint"
    ckpt_dir.mkdir()
    source.save(ckpt_dir / "trainer.pt")
    ckpt = _LocalRayCheckpoint(ckpt_dir)
    new_peak = _LR_OLD_PEAK * peak_scale

    old_lrs = [g["lr"] for g in source.opt.param_groups]
    expected = source  # Uninterrupted PB2 rebase, without checkpoint roundtrip.
    _apply_lr_gamma_weights(expected, {"lr": new_peak}, rescale_current_lr=True)

    restored = _make_lr_trainer(tmp_path / "restored", new_peak, cycle_steps=cycle_steps)
    _restore_from_ray_checkpoint(
        ckpt=ckpt, trainer=restored,
        config={"optimizer": "aurora", "lr": new_peak},
        device="cpu", trial_id="lr-probe", rr=RestoreResult(),
    )
    # Production's first per-iteration config sync follows restore. A second
    # True call must not mask stale group LRs from a prior metadata-only rebase.
    _apply_lr_gamma_weights(restored, {"lr": new_peak}, rescale_current_lr=True)

    assert restored.step == expected.step == saved_step
    assert [g["lr"] for g in restored.opt.param_groups] == pytest.approx(
        [lr * peak_scale for lr in old_lrs], rel=1e-12, abs=1e-18,
    )
    assert restored._scheduler.base_lrs == pytest.approx(expected._scheduler.base_lrs)
    assert restored._scheduler._last_lr == pytest.approx(expected._scheduler._last_lr)
    assert restored._scheduler.last_epoch == expected._scheduler.last_epoch
    _assert_lr_trainer_equal(expected, restored)

    # After warmup the release scheduler writes the first next-window LR before
    # opt.step(); exercise that ordering, not the previous window's release floor.
    for trainer in (expected, restored):
        if trainer._uses_train_window_release_cycle():
            trainer._set_train_window_release_lr(local_step=0, cycle_steps=_LR_WINDOW_STEPS)
    used_lrs = [g["lr"] for g in restored.opt.param_groups]
    assert used_lrs == pytest.approx(
        [g["lr"] for g in expected.opt.param_groups], rel=1e-12, abs=1e-18,
    )
    for trainer in (expected, restored):
        trainer.opt.zero_grad(set_to_none=True)
        for param in trainer.model.parameters():
            if param.requires_grad:
                param.grad = torch.ones_like(param)
        trainer.opt.step()
    _assert_lr_trainer_equal(expected, restored)
