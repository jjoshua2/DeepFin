from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import torch

from chess_anti_engine.inference_cache import EncodedEvalCache
from chess_anti_engine.inference_dispatcher import (
    BatchCoalescingDispatcher,
    ThreadSafeGPUDispatcher,
)
from chess_anti_engine.uci import __main__ as uci_main


class _TinyModule(torch.nn.Module):
    input_history_encoding = "legacy"

    def forward(self, x):
        return x


class _FakeDirectEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str,
        max_batch: int,
        n_slots: int,
    ) -> None:
        del model, device, n_slots
        self._max_batch = int(max_batch)

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = int(x.shape[0])
        return (
            np.zeros((rows, 4672), dtype=np.float32),
            np.zeros((rows, 3), dtype=np.float32),
        )


def _skip_warmup(*args: object, **kwargs: object) -> None:
    del args, kwargs


def _compile_identity(
    model: torch.nn.Module,
    *,
    mode: str | None = None,
) -> torch.nn.Module:
    del mode
    return model


def test_compiled_single_walker_uses_submitter_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uci_main, "DirectGPUEvaluator", _FakeDirectEvaluator)
    monkeypatch.setattr(uci_main, "_warmup_evaluator", _skip_warmup)
    monkeypatch.setattr(torch, "compile", _compile_identity)

    factory = uci_main._make_evaluator_factory(
        [_TinyModule()],
        ["cuda"],
        coalesce=True,
        n_walkers=1,
        walker_gather=1,
        compile_mode="max-autotune",
    )
    evaluator = factory(max_batch=64, eval_cache_entries=0)
    assert isinstance(evaluator, BatchCoalescingDispatcher)
    evaluator.close()


def test_compiled_single_walker_ignores_no_coalesce_for_thread_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uci_main, "DirectGPUEvaluator", _FakeDirectEvaluator)
    monkeypatch.setattr(uci_main, "_warmup_evaluator", _skip_warmup)
    monkeypatch.setattr(torch, "compile", _compile_identity)

    factory = uci_main._make_evaluator_factory(
        [_TinyModule()],
        ["cuda"],
        coalesce=False,
        n_walkers=1,
        walker_gather=1,
        compile_mode="max-autotune",
    )
    evaluator = factory(max_batch=64, eval_cache_entries=0)
    assert isinstance(evaluator, BatchCoalescingDispatcher)
    evaluator.close()


def test_eager_single_walker_keeps_direct_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uci_main, "DirectGPUEvaluator", _FakeDirectEvaluator)
    monkeypatch.setattr(uci_main, "_warmup_evaluator", _skip_warmup)

    factory = uci_main._make_evaluator_factory(
        [_TinyModule()],
        ["cuda"],
        coalesce=True,
        n_walkers=1,
        walker_gather=1,
        compile_mode=None,
    )
    evaluator = factory(max_batch=64, eval_cache_entries=0)
    assert isinstance(evaluator, ThreadSafeGPUDispatcher)


class _RecordingDirectEvaluator(_FakeDirectEvaluator):
    """Fake DirectGPUEvaluator that registers itself and records batch sizes,
    so tests can assert which devices the warmup traffic actually reached."""

    instances: ClassVar[list[_RecordingDirectEvaluator]] = []

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str,
        max_batch: int,
        n_slots: int,
    ) -> None:
        super().__init__(model, device=device, max_batch=max_batch, n_slots=n_slots)
        self.device = device
        self.batch_sizes: list[int] = []
        _RecordingDirectEvaluator.instances.append(self)

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.batch_sizes.append(int(x.shape[0]))
        return super().evaluate_encoded(x)


def test_warmup_covers_every_device_in_routing_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With >1 devices the warmup must reach ALL of them (each batch shape is
    issued n_devices times; MultiGPUDispatcher's round-robin tiebreaker cycles
    sequential calls). A device the warmup misses would pay its cold
    compile/capture stall on a mid-game root eval, on the clock."""
    monkeypatch.setattr(uci_main, "DirectGPUEvaluator", _RecordingDirectEvaluator)
    monkeypatch.setattr(_RecordingDirectEvaluator, "instances", [])

    factory = uci_main._make_evaluator_factory(
        [_TinyModule(), _TinyModule(), _TinyModule()],
        ["cpu", "cpu", "cpu"],
        coalesce=True,
        n_walkers=2,
        walker_gather=1,
        compile_mode=None,
    )
    evaluator = factory(max_batch=64, eval_cache_entries=0)
    assert isinstance(evaluator, BatchCoalescingDispatcher)
    try:
        instances = _RecordingDirectEvaluator.instances
        assert len(instances) == 3
        # walkers=2, gather=1 warms shapes {1, 2}; every device sees BOTH.
        for inst in instances:
            assert sorted(inst.batch_sizes) == [1, 2], (
                f"device missed warmup shapes: {inst.batch_sizes}"
            )
    finally:
        evaluator.close()


def test_warmup_runs_below_eval_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EncodedEvalCache dedupes identical rows and short-circuits full-hit
    batches, so warming THROUGH it never reaches the GPU past the first row —
    the batch=128 gumbel shape stayed cold. Warmup must target the pre-cache
    chain."""
    monkeypatch.setattr(uci_main, "DirectGPUEvaluator", _RecordingDirectEvaluator)
    monkeypatch.setattr(_RecordingDirectEvaluator, "instances", [])

    factory = uci_main._make_evaluator_factory(
        [_TinyModule()],
        ["cpu"],
        coalesce=True,
        n_walkers=1,
        walker_gather=1,
        compile_mode=None,
    )
    evaluator = factory(max_batch=256, eval_cache_entries=64)
    assert isinstance(evaluator, EncodedEvalCache)
    (inst,) = _RecordingDirectEvaluator.instances
    # Gumbel-path warmup shapes, both reaching the real evaluator.
    assert inst.batch_sizes == [1, 128]


def test_pucv_factory_pins_cuda_device_before_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each pool factory must bind its worker thread's CUDA device FIRST:
    threads inherit current-device cuda:0, and cudagraph capture on devices
    1..N-1 must not rely on stream contexts alone."""
    monkeypatch.setattr(uci_main, "DirectGPUEvaluator", _FakeDirectEvaluator)
    monkeypatch.setattr(uci_main, "_warmup_pucv_evaluator", _skip_warmup)
    pinned: list[int] = []
    monkeypatch.setattr(torch.cuda, "set_device", pinned.append)

    build = uci_main._make_multi_gpu_pucv_factory_builder(
        [_TinyModule(), _TinyModule()],
        ["cuda:0", "cuda:1"],
        compile_mode=None,
    )
    factories = build(64, 8)
    assert len(factories) == 2
    for make in factories:
        make()
    assert pinned == [0, 1]


def test_pucv_factory_skips_set_device_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cpu-device factories (the CPU-evaluator test path) must not touch
    torch.cuda at all — set_device would raise on a CUDA-less box."""
    monkeypatch.setattr(uci_main, "DirectGPUEvaluator", _FakeDirectEvaluator)
    monkeypatch.setattr(uci_main, "_warmup_pucv_evaluator", _skip_warmup)

    def _boom(_dev: object) -> None:
        raise AssertionError("set_device must not be called for cpu devices")

    monkeypatch.setattr(torch.cuda, "set_device", _boom)

    build = uci_main._make_multi_gpu_pucv_factory_builder(
        [_TinyModule(), _TinyModule()],
        ["cpu", "cpu"],
        compile_mode=None,
    )
    for make in build(64, 8):
        make()


def test_resolve_use_multi_gpu_pucv_auto_enables(capsys) -> None:
    """Unset flag + >1 devices auto-enables the pool (with a stderr notice):
    the routing chain's single submitter serializes GPU work, so multi-device
    without the pool is a silent single-GPU configuration."""
    assert uci_main._resolve_use_multi_gpu_pucv(None, 8) is True
    err = capsys.readouterr().err
    assert "auto-enabling" in err

    # Explicit opt-out is honored but warned about LOUDLY.
    assert uci_main._resolve_use_multi_gpu_pucv(False, 8) is False
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "debugging" in err

    # Explicit opt-in: no chatter needed.
    assert uci_main._resolve_use_multi_gpu_pucv(True, 8) is True
    assert capsys.readouterr().err == ""


def test_resolve_use_multi_gpu_pucv_single_device_silent(capsys) -> None:
    """Single device keeps today's behavior exactly: pool off, no output,
    regardless of how the flag was set."""
    for flag in (None, False, True):
        assert uci_main._resolve_use_multi_gpu_pucv(flag, 1) is False
    assert capsys.readouterr().err == ""
