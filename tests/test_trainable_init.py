from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from chess_anti_engine.replay.shard import iter_shard_paths, local_shard_path, save_local_shard_arrays
from chess_anti_engine.train.trainer import Trainer
from chess_anti_engine.tune.trainable_config_ops import _apply_lr_gamma_weights
from chess_anti_engine.tune.trainable_init import (
    _restore_from_ray_checkpoint,
    _seed_replay_from_shared_shards,
)
from chess_anti_engine.tune.trial_config import RestoreResult, TrialConfig


def _arrays(policy_size: int) -> dict[str, np.ndarray]:
    policy = np.zeros((2, policy_size), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
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
