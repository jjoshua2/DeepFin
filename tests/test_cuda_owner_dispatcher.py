from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from chess_anti_engine.inference_dispatcher import CUDAOwnerDispatcher


class _OwnerInner:
    n_slots = 2
    _max_batch = 64
    supports_input_bf16_bits = True
    supports_legal_bf16 = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.inputs = [np.zeros((64, 4, 8, 8), dtype=np.float32) for _ in range(2)]

    def get_input_buffer(self, bsz: int, slot: int = 0) -> np.ndarray:
        return self.inputs[slot][:bsz]

    def get_input_buffer_bf16_bits(self, bsz: int, slot: int = 0) -> np.ndarray:
        return self.inputs[slot][:bsz].view(np.uint16)

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(("sync", threading.get_ident()))
        return np.zeros((x.shape[0], 8), dtype=np.float32), np.zeros((x.shape[0], 3))

    def evaluate_inplace_async(
        self, bsz: int, *, slot: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        del slot
        self.calls.append(("async", threading.get_ident()))
        return torch.zeros((bsz, 8)), torch.zeros((bsz, 3)), None


def test_cuda_owner_runs_sync_and_async_calls_on_one_thread() -> None:
    inner = _OwnerInner()
    dispatcher = CUDAOwnerDispatcher(inner)
    try:
        dispatcher.evaluate_encoded(np.zeros((2, 4, 8, 8), dtype=np.float32))
        dispatcher.get_input_buffer(3, slot=1).fill(1.0)
        policy, wdl, event = dispatcher.evaluate_inplace_async(3, slot=1)
        assert policy.shape == (3, 8)
        assert wdl.shape == (3, 3)
        assert event is None
        assert inner.calls[0][1] == inner.calls[1][1]
        assert inner.calls[0][1] != threading.get_ident()
        assert np.all(inner.inputs[1][:3] == 1.0)
    finally:
        dispatcher.close()


def test_cuda_owner_close_is_idempotent_and_rejects_work() -> None:
    dispatcher = CUDAOwnerDispatcher(_OwnerInner())
    dispatcher.close()
    dispatcher.close()
    with pytest.raises(RuntimeError, match="closed"):
        dispatcher.evaluate_encoded(np.zeros((1, 4, 8, 8), dtype=np.float32))
