from __future__ import annotations

import numpy as np
import pytest
import torch

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
