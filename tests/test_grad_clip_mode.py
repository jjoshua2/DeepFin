"""Disabling clipping must bypass both scaling and adaptive-state updates."""
from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from chess_anti_engine.train import trainer as trainer_module
from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config
from chess_anti_engine.tune.trainable_config_ops import (
    _reload_yaml_into_config,
    restart_required_config_keys,
)
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"policy": self.weight * x.sum()}


def _trainer(tmp_path, mode="zclip", cap=None) -> Trainer:
    return Trainer(
        _Tiny(), device="cpu", lr=1e-3, optimizer="nadamw",
        grad_clip_mode=mode, zclip_max_norm=cap, warmup_steps=0,
        use_amp=False, log_dir=tmp_path, prefetch_batches=False,
    )


def test_yaml_reaches_trainer_and_mode_is_restart_required(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  grad_clip_mode: none\n  zclip_max_norm: null\n")
    cfg = flatten_run_config_defaults(load_yaml_file(path))
    kwargs = trainer_kwargs_from_config(cfg, log_dir=tmp_path / "tb")
    kwargs.update(device="cpu", use_amp=False)
    trainer = Trainer(_Tiny(), **kwargs)
    assert trainer.grad_clip_mode == "none"
    assert trainer.grad_clip_max_norm is None
    assert "grad_clip_mode" in restart_required_config_keys()
    assert trainer_kwargs_from_config({})["grad_clip_mode"] == "zclip"
    trainer.writer.close()


@pytest.mark.parametrize("key", ["zclip_max_norm", "grad_clip"])
def test_yaml_null_cap_survives_launch_and_live_reload(tmp_path, key) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"train:\n  {key}: null\n")
    cfg = flatten_run_config_defaults(load_yaml_file(path))
    assert key in cfg
    assert cfg[key] is None
    assert trainer_kwargs_from_config(cfg)["zclip_max_norm"] is None
    live = {key: 5.0}
    _reload_yaml_into_config(live, str(path), live_reload=True)
    assert live[key] is None


@pytest.mark.parametrize(("mode", "cap"), [("off", None), (False, None), ("none", 1.0)])
def test_invalid_or_contradictory_mode_is_rejected(tmp_path, mode, cap) -> None:
    with pytest.raises(ValueError, match="grad_clip_mode"):
        _trainer(tmp_path, mode, cap)


@pytest.mark.parametrize("collect_stats", [False, True])
def test_none_keeps_gradients_and_ema_unchanged_past_warmup(tmp_path, collect_stats) -> None:
    trainer = _trainer(tmp_path, "none")
    state = trainer.zclip_state_dict()
    param = next(trainer.model.parameters())
    for scale in [1.0] * 30 + [100.0]:
        param.grad = torch.full_like(param, scale)
        before = param.grad.clone()
        norm, stats = trainer._zclip_step(collect_stats=collect_stats)
        assert norm == pytest.approx(2 * scale)
        assert torch.equal(param.grad, before)
        assert trainer.zclip_state_dict() == state
        if collect_stats:
            assert stats is not None
            assert stats["clipped"] == 0.0
            assert stats["effective_clip"] == norm
        else:
            assert stats is None
    # A live cap edit cannot pretend to take effect in the disabled mode.
    assert not trainer.set_grad_clip_max_norm(5.0)
    assert trainer.grad_clip_max_norm is None
    trainer.writer.close()


def test_default_still_adapts_when_only_hard_cap_is_disabled(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    param = next(trainer.model.parameters())
    stats = None
    for scale in [1.0] * 30 + [100.0]:
        param.grad = torch.full_like(param, scale)
        _, stats = trainer._zclip_step(collect_stats=True)
    assert trainer.zclip.initialized
    assert stats is not None
    assert stats["adaptive_bound"] == 1.0
    assert param.grad is not None
    assert torch.linalg.vector_norm(param.grad) < 200.0
    trainer.writer.close()


@pytest.mark.parametrize("scale", [1.0, float("inf"), float("nan")])
def test_none_preserves_optimizer_nonfinite_guard(tmp_path, monkeypatch, scale) -> None:
    trainer = _trainer(tmp_path, "none")
    param = next(trainer.model.parameters())
    before = param.detach().clone()

    def loss(_out, _batch, **_kwargs):
        return {"total": (param ** 2).sum() * scale}

    monkeypatch.setattr(trainer_module, "compute_loss", loss)
    monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda out, batch: {})
    stats: dict[str, float] = {}
    trainer._run_optimizer_step(
        step_sums=trainer_module._DeviceLossSums(), step_acc_sums={},
        step_opt_stats=stats, buf=cast(Any, None), batch_size=1,
        batch_iter=iter([{"x": torch.ones(1, 4, 8, 8)}]),
    )
    if scale == 1.0:
        assert not torch.equal(param.detach(), before)
        assert stats.get("nonfinite_grad", 0.0) == 0.0
    else:
        assert torch.equal(param.detach(), before)
        assert stats["nonfinite_grad"] == 1.0
        assert not trainer.opt.state
    trainer.writer.close()


def test_exact_resume_carries_mode_and_refuses_mismatch_before_loading(tmp_path) -> None:
    src = _trainer(tmp_path / "src", "none")
    path = tmp_path / "trainer.pt"
    src.save(path)
    dst = _trainer(tmp_path / "dst", "none")
    dst.load(path, exact_resume=True)
    assert dst.grad_clip_mode == "none"
    incompatible = _trainer(tmp_path / "incompatible")
    param = next(incompatible.model.parameters())
    with torch.no_grad():
        param.fill_(7)
    with pytest.raises(ValueError, match="identical grad_clip_mode"):
        incompatible.load(path, exact_resume=True)
    assert torch.all(param == 7)
    for trainer in (src, dst, incompatible):
        trainer.writer.close()


def test_legacy_checkpoint_defaults_to_zclip(tmp_path) -> None:
    src = _trainer(tmp_path / "src")
    path = tmp_path / "trainer.pt"
    src.save(path)
    payload = torch.load(path, weights_only=False)
    del payload["grad_clip_mode"]
    torch.save(payload, path)
    src.load(path, exact_resume=True)
    disabled = _trainer(tmp_path / "disabled", "none")
    with pytest.raises(ValueError, match="identical grad_clip_mode"):
        disabled.load(path, exact_resume=True)
    src.writer.close()
    disabled.writer.close()
