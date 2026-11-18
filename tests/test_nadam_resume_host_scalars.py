from __future__ import annotations

import copy
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest
import torch

from chess_anti_engine.train.soda import SODAWeightDecayWrapper
from chess_anti_engine.train.trainer import Trainer
from scripts.bootstrap_checkpoint_resume import require_equal


class _TinyModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.2, -0.3], [0.4, 0.1]], dtype=dtype))
        self.bias = torch.nn.Parameter(torch.tensor([0.1, -0.2], dtype=dtype))


def _trainer(path: Path, *, device: str, optimizer: str = "nadamw",
             dtype: torch.dtype = torch.float32) -> Trainer:
    return Trainer(
        _TinyModel(dtype), device=device, optimizer=optimizer, lr=3e-4,
        warmup_steps=0, lr_eta_min=3e-4, use_amp=False, prefetch_batches=False, log_dir=path,
        tb_log_interval=10**9,
    )


def _step(trainer: Trainer, gradient: float) -> None:
    # Exercise the real optimizer without a model forward or training-data fixture.
    trainer.opt.zero_grad(set_to_none=True)
    for parameter in trainer.model.parameters():
        parameter.grad = torch.full_like(parameter, gradient)
    trainer.opt.step()
    trainer.step += 1


def _assert_state_placement(trainer: Trainer) -> None:
    for group in trainer.opt.param_groups:
        for parameter in group["params"]:
            state = trainer.opt.state[parameter]
            assert state["exp_avg"].device == parameter.device
            assert state["exp_avg_sq"].device == parameter.device
            if isinstance(trainer.opt, torch.optim.NAdam):
                for key in ("step", "mu_product"):
                    assert state[key].device.type == "cpu"


@pytest.mark.parametrize("optimizer", ["nadamw", "adamw"])
@pytest.mark.parametrize("exact_resume", [False, True])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
])
def test_real_trainer_resume_preserves_state_and_next_update(
    tmp_path: Path, optimizer: str, exact_resume: bool, device: str,
) -> None:
    with ExitStack() as stack:
        control = _trainer(tmp_path / "control", device=device, optimizer=optimizer)
        stack.callback(control.writer.close)
        _step(control, 0.125)
        _step(control, -0.25)
        checkpoint = tmp_path / "own-synthetic.pt"
        control.save(checkpoint)
        expected_state = copy.deepcopy(control.opt.state_dict())
        resumed = _trainer(tmp_path / "resumed", device=device, optimizer=optimizer)
        stack.callback(resumed.writer.close)
        parameter_ids = [id(p) for group in resumed.opt.param_groups for p in group["params"]]
        resumed.load(checkpoint, exact_resume=exact_resume)

        assert parameter_ids == [id(p) for group in resumed.opt.param_groups for p in group["params"]]
        assert resumed._optimizer_param_names() == control._optimizer_param_names()
        require_equal(resumed.model.state_dict(), control.model.state_dict(), "loaded model")
        require_equal(resumed.opt.state_dict(), expected_state, "loaded optimizer")
        require_equal(resumed._scheduler.state_dict(), control._scheduler.state_dict(), "loaded scheduler")
        assert resumed.step == control.step
        _assert_state_placement(resumed)

        # Compare within the same device and algorithm; no CPU/CUDA trajectory
        # equivalence or noisy wall-clock performance threshold is asserted.
        _step(control, 0.375)
        _step(resumed, 0.375)
        require_equal(resumed.model.state_dict(), control.model.state_dict(), "next model update")
        require_equal(resumed.opt.state_dict(), control.opt.state_dict(), "next optimizer update")
        _assert_state_placement(resumed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(), (1,)])
def test_cpu_nadam_load_preserves_scalar_values_dtypes_and_shapes(
    tmp_path: Path, dtype: torch.dtype, shape: tuple[int, ...],
) -> None:
    trainer = _trainer(tmp_path, device="cpu", dtype=dtype)
    try:
        _step(trainer, 0.125)
        payload: dict[str, Any] = copy.deepcopy(trainer.opt.state_dict())
        for state in payload["state"].values():
            state["step"] = torch.full(shape, 7.0, dtype=dtype)
            state["mu_product"] = torch.full(shape, 0.5, dtype=dtype)
        trainer.opt.load_state_dict(payload)
        require_equal(trainer.opt.state_dict(), payload, "scalar metadata")
        _assert_state_placement(trainer)
        # Repeated and direct loads use the same registered post-hook.
        trainer.opt.load_state_dict(trainer.opt.state_dict())
        require_equal(trainer.opt.state_dict(), payload, "idempotent load")
    finally:
        trainer.writer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capturable_nadam_load_keeps_accelerator_scalars(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, device="cuda")
    try:
        _step(trainer, 0.125)
        payload = copy.deepcopy(trainer.opt.state_dict())
        for group in payload["param_groups"]:
            group["capturable"] = True
        for state in payload["state"].values():
            for key in ("step", "mu_product"):
                state[key] = state[key].to(device="cuda")
        trainer.opt.load_state_dict(payload)
        require_equal(trainer.opt.state_dict(), payload, "capturable state")
        for group in trainer.opt.param_groups:
            assert group["capturable"] is True
            for parameter in group["params"]:
                for key in ("step", "mu_product", "exp_avg", "exp_avg_sq"):
                    assert trainer.opt.state[parameter][key].device == parameter.device
    finally:
        trainer.writer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_delegated_nadam_load_restores_both_host_scalars(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, device="cuda")
    try:
        _step(trainer, 0.125)
        wrapper = SODAWeightDecayWrapper(trainer.opt)
        payload = copy.deepcopy(wrapper.state_dict())
        for state in payload["state"].values():
            for key in ("step", "mu_product"):
                state[key] = state[key].to(device="cuda")
        wrapper.load_state_dict(payload)
        require_equal(wrapper.state_dict(), payload, "delegated values")
        _assert_state_placement(trainer)
    finally:
        trainer.writer.close()
