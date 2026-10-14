"""CPU regressions for coalescer representation and waiter boundaries."""
from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import BatchCoalescingDispatcher, _CoalesceRequest


class InputEcho(torch.nn.Module):
    input_extra_features = "v1"

    def forward(self, x):
        value = x[:, 0, 0, 0].float()
        return {
            "policy": value[:, None].expand(-1, 4672).clone(),
            "wdl": torch.stack((value, value * 0, value * 0), dim=1),
        }


def prepared(inner, requests, *, max_batch=4, shutdown=True):
    dispatcher = BatchCoalescingDispatcher.__new__(BatchCoalescingDispatcher)
    dispatcher._inner = inner
    dispatcher._max_batch = max_batch
    dispatcher._lock = threading.Lock()
    dispatcher._wake = threading.Event()
    dispatcher._wake.set()
    dispatcher._shutdown = threading.Event()
    if shutdown:
        dispatcher._shutdown.set()
    dispatcher._pending = requests.copy()
    return dispatcher


def request(value, *, compact=False, relations=None):
    rows = value.shape[0]
    return _CoalesceRequest(
        value, relations,
        np.arange(rows, dtype=np.int32) if compact else None,
        np.ones(rows, dtype=np.int32) if compact else None,
        threading.Event(), [None],
    )


def planes(value, dtype):
    x = np.full((1, 146, 8, 8), value, dtype=np.float32)
    if dtype == np.uint16:
        return torch.from_numpy(x).to(torch.bfloat16).view(torch.uint16).numpy()
    return x


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("dtypes", [
    [np.float32, np.uint16, np.float32],
    [np.uint16, np.float32, np.uint16],
    [np.float32] * 3, [np.uint16] * 3,
])
def test_coalescer_matches_actual_direct_cpu_input_values(compact, dtypes):
    direct = DirectGPUEvaluator(InputEcho(), device="cpu", max_batch=4, use_amp=False)
    requests = [request(planes(i + 1, dtype), compact=compact) for i, dtype in enumerate(dtypes)]
    expected = []
    for item in requests:
        if compact:
            assert item.legal_flat is not None
            assert item.legal_counts is not None
            expected.append(direct.evaluate_legal_bf16(item.x, item.legal_flat, item.legal_counts))
        else:
            expected.append(direct.evaluate_encoded(item.x))
    dispatcher = prepared(direct, requests)
    dispatcher._submitter_loop()
    for item, (policy, wdl) in zip(requests, expected, strict=True):
        assert item.done.is_set()
        actual_policy, actual_wdl = item.result[0]
        np.testing.assert_array_equal(actual_policy, policy)
        np.testing.assert_array_equal(actual_wdl, wdl)
    assert not dispatcher._pending


class RecordingEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate_encoded(self, x, relations=None):
        self.calls.append((x.copy(), None if relations is None else relations.copy()))
        values = x[:, 0, 0, 0].astype(np.float32)
        return values[:, None], np.repeat(values[:, None], 3, axis=1)

    def evaluate_legal_bf16(self, x, legal_flat, legal_counts):
        self.calls.append((x.copy(), legal_flat.copy(), legal_counts.copy()))
        return legal_flat.copy(), np.repeat(x[:, 0, 0, 0, None], 3, axis=1)


def test_coalescer_preserves_fifo_row_cap_and_missing_relation_rows():
    inner = RecordingEvaluator()
    first = request(np.repeat(planes(1, np.float32), 2, axis=0))
    second = request(planes(2, np.float32), relations=np.ones((1, 5, 64, 64), dtype=np.uint8))
    third = request(planes(3, np.float32))
    dispatcher = prepared(inner, [first, second, third], max_batch=3)
    dispatcher._submitter_loop()
    assert [call[0].shape[0] for call in inner.calls] == [3, 1]
    np.testing.assert_array_equal(inner.calls[0][0][:, 0, 0, 0], [1, 1, 2])
    assert not inner.calls[0][1][:2].any()
    assert inner.calls[0][1][2].all()
    assert inner.calls[1][1] is None
    for item in [first, second, third]:
        assert item.done.is_set()
        np.testing.assert_array_equal(item.result[0][1][:, 0], item.x[:, 0, 0, 0])


@pytest.mark.parametrize("stage", ["input", "relation", "relation_copy", "legal", "counts", "slice"])
def test_coalescer_packing_failures_release_waiters_and_recover(monkeypatch, stage):
    inner = RecordingEvaluator()
    compact = stage in {"legal", "counts"}
    relation = np.ones((1, 5, 64, 64), dtype=np.uint8) if stage.startswith("relation") else None
    requests = [request(planes(i + 1, np.float32), compact=compact, relations=relation) for i in range(2)]
    # This dtype boundary leaves an accepted request pending when packing fails.
    requests.append(request(planes(3, np.uint16), compact=compact))
    dispatcher = prepared(inner, requests, shutdown=False)
    primary = MemoryError(f"{stage} packing sentinel")
    concatenate = np.concatenate
    zeros = np.zeros
    attempts = 0

    def failing_concatenate(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == {"input": 1, "legal": 2, "counts": 3}.get(stage):
            raise primary
        return concatenate(*args, **kwargs)

    def failing_zeros(*args, **kwargs):
        if stage == "relation":
            raise primary
        if stage == "relation_copy":
            return zeros(*args, **kwargs).view(FailingRelation)
        return zeros(*args, **kwargs)

    class FailingRelation(np.ndarray):
        def __setitem__(self, key, value):
            raise primary

    class FailingSlice:
        def __getitem__(self, key):
            if key.start == 0:
                return np.zeros((1, 1))
            raise primary

    original_evaluate = inner.evaluate_encoded

    def failing_evaluate(*args, **kwargs):
        if stage == "slice":
            return FailingSlice(), np.zeros((2, 3))
        return original_evaluate(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "concatenate", failing_concatenate)
        patch.setattr(np, "zeros", failing_zeros)
        patch.setattr(inner, "evaluate_encoded", failing_evaluate)
        thread = threading.Thread(target=dispatcher._submitter_loop, daemon=True)
        dispatcher._submitter = thread
        thread.start()
        completed = [item.done.wait(timeout=1) for item in requests]
    try:
        assert completed == [True, True, True]
        assert all(item.result[0] is primary for item in requests)
        assert thread.is_alive()
        later = request(planes(3, np.float32))
        with dispatcher._lock:
            dispatcher._pending.append(later)
        dispatcher._wake.set()
        assert later.done.wait(timeout=1)
        np.testing.assert_array_equal(later.result[0][1][:, 0], [3])
    finally:
        dispatcher.close()
        thread.join(timeout=1)
    assert not thread.is_alive()
