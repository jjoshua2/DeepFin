"""Actual dispatcher imports with an inert two-slot backend; no model or CUDA.

These tests are intentionally executable behavioral controls, rather than AST
copies of the dispatcher. CUDA/compile numerical qualification remains separate.
"""
from __future__ import annotations

from concurrent.futures import Future
import threading

import numpy as np
import pytest

from chess_anti_engine import inference_threaded as threaded
from chess_anti_engine.request_routing import FifoRequestQueue


class _Model:
    def __init__(self, value=0):
        self.value = value

    def eval(self):
        return self

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, values, *, strict):
        assert strict
        self.value = values["value"]


class _Event:
    def __init__(self, backend, slot):
        self.backend, self.slot = backend, slot

    def query(self):
        return True

    def synchronize(self):
        if self.backend.fail_retire:
            raise RuntimeError("inert retire failure")
        self.backend.events.append(("retire", self.slot, threading.get_ident()))
        self.backend.busy[self.slot] = False


class _Backend:
    """Pinned outputs are deliberately reused, exposing missing snapshot bugs."""

    supports_input_bf16_bits = False
    supports_legal_bf16 = True

    def __init__(self, model, *, max_batch, n_slots, **_kwargs):
        assert n_slots == 2
        self.model = model
        self.inputs = [np.empty((max_batch, 1), np.float32) for _ in range(2)]
        self.policies = [np.empty((max_batch, 2), np.float32) for _ in range(2)]
        self.compact = [np.empty(max_batch * 4, np.float32) for _ in range(2)]
        self.wdls = [np.empty((max_batch, 3), np.float32) for _ in range(2)]
        self.busy = [False, False]
        self.events = []
        self.fail_submit = False
        self.fail_retire = False

    def get_input_buffer(self, _n, *, slot):
        assert not self.busy[slot], "slot reused before retirement"
        return self.inputs[slot]

    def _launch(self, n, slot):
        self.events.append(("submit", slot, threading.get_ident(), self.model.value))
        if self.fail_submit:
            raise RuntimeError("inert launch failure")
        self.busy[slot] = True
        values = self.inputs[slot][:n, 0] + self.model.value
        self.wdls[slot][:n] = values[:, None]
        return values

    def evaluate_inplace_async(self, n, *, slot):
        values = self._launch(n, slot)
        self.policies[slot][:n] = values[:, None]
        return self.policies[slot], self.wdls[slot], _Event(self, slot)

    def evaluate_inplace_legal_bf16_async(self, n, legal_flat, legal_counts, *, slot):
        values = self._launch(n, slot)
        self.compact[slot][:len(legal_flat)] = np.repeat(values, legal_counts)
        return self.compact[slot], self.wdls[slot], _Event(self, slot)


@pytest.fixture
def held_dispatcher(monkeypatch):
    monkeypatch.setattr(threaded, "DirectGPUEvaluator", _Backend)
    release = threading.Event()
    compile_threads = []

    class Held(threaded.ThreadedDispatcher):
        def _compile_model_on_dispatcher(self, model):
            compile_threads.append(threading.get_ident())
            if not release.wait(3):
                raise TimeoutError("test did not release compile owner")
            return model.eval()

    dispatcher = Held(_Model(), device="cpu", max_batch=8, target_batch=4)
    yield dispatcher, release, compile_threads
    release.set()
    dispatcher.shutdown()
    assert not dispatcher._thread.is_alive()


def _rows(*values, dtype=np.float32):
    return np.asarray(values, dtype=dtype).reshape(-1, 1)


def _compact_request(dispatcher, values, counts):
    future = Future()
    item = threaded._EvalRequest(
        _rows(*values), future, np.arange(sum(counts), dtype=np.int32),
        np.asarray(counts, dtype=np.int32),
    )
    with dispatcher._cond:
        dispatcher._enqueue_eval(item)
        dispatcher._cond.notify()
    return future


def test_whole_fifo_dtype_compact_groups_and_owned_output(held_dispatcher):
    dispatcher, release, compile_threads = held_dispatcher
    assert isinstance(dispatcher._queue, FifoRequestQueue)
    first = dispatcher.evaluate(_rows(1, 2, 3))
    second = dispatcher.evaluate(_rows(4, 5))  # cannot split to fill target=4
    bits = dispatcher.evaluate(_rows(0, 0, dtype=np.uint16))
    compact_a = _compact_request(dispatcher, [6], [2])
    compact_b = _compact_request(dispatcher, [7, 8], [1, 2])
    release.set()
    saved_first = first.result(3)[0]
    np.testing.assert_array_equal(saved_first, [[1, 1], [2, 2], [3, 3]])
    np.testing.assert_array_equal(second.result(3)[0], [[4, 4], [5, 5]])
    np.testing.assert_array_equal(bits.result(3)[0], [[0, 0], [0, 0]])
    np.testing.assert_array_equal(compact_a.result(3)[0], [6, 6])
    np.testing.assert_array_equal(compact_b.result(3)[0], [7, 8, 8])
    # Both output slots have since been reused; the earlier result owns bytes.
    np.testing.assert_array_equal(saved_first, [[1, 1], [2, 2], [3, 3]])
    assert dispatcher.stats["lifetime_batches"] == 4
    events = dispatcher._evaluator.events
    assert [(event[0], event[1]) for event in events[:3]] == [
        ("submit", 0), ("submit", 1), ("retire", 0),
    ]
    assert {event[2] for event in events} == set(compile_threads)
    assert compile_threads[0] != threading.get_ident()


def test_model_barrier_retires_old_output_before_new_forward(held_dispatcher):
    dispatcher, release, _threads = held_dispatcher
    old = dispatcher.evaluate(_rows(1))
    done = threading.Event()
    result = {}
    with dispatcher._cond:
        dispatcher._queue.enqueue_control(threaded._ModelUpdateRequest(_Model(10), done, result))
    new = dispatcher.evaluate(_rows(2))
    release.set()
    np.testing.assert_array_equal(old.result(3)[0], [[1, 1]])
    assert done.wait(3) and not result
    np.testing.assert_array_equal(new.result(3)[0], [[12, 12]])
    events = dispatcher._evaluator.events
    assert [(event[0], event[1]) for event in events[:3]] == [
        ("submit", 0), ("retire", 0), ("submit", 1),
    ]


def test_cancelled_request_does_not_strand_its_batch_peer(held_dispatcher):
    dispatcher, release, _threads = held_dispatcher
    cancelled = dispatcher.evaluate(_rows(1))
    peer = dispatcher.evaluate(_rows(2))
    assert cancelled.cancel()
    release.set()
    np.testing.assert_array_equal(peer.result(3)[0], [[2, 2]])
    assert cancelled.cancelled()


def test_cancellation_racing_publication_does_not_poison_peer(held_dispatcher, monkeypatch):
    dispatcher, release, _threads = held_dispatcher
    cancelled = dispatcher.evaluate(_rows(1))
    peer = dispatcher.evaluate(_rows(2))
    publish = cancelled.set_result

    def cancel_then_publish(value):
        cancelled.cancel()
        publish(value)

    monkeypatch.setattr(cancelled, "set_result", cancel_then_publish)
    release.set()
    np.testing.assert_array_equal(peer.result(3)[0], [[2, 2]])
    assert cancelled.cancelled()


@pytest.mark.parametrize("phase", ["submit", "retire"])
def test_pipeline_failure_fails_queued_and_model_barrier(held_dispatcher, phase):
    dispatcher, release, _threads = held_dispatcher
    setattr(dispatcher._evaluator, "fail_" + phase, True)
    failed = dispatcher.evaluate(_rows(1, 2, 3, 4))
    queued = dispatcher.evaluate(_rows(5))
    done = threading.Event()
    result = {}
    with dispatcher._cond:
        dispatcher._queue.enqueue_control(threaded._ModelUpdateRequest(_Model(10), done, result))
    release.set()
    for future in (failed, queued):
        with pytest.raises(RuntimeError, match="inert (launch|retire) failure"):
            future.result(3)
    assert done.wait(3) and "error" in result
    dispatcher.shutdown()
    with pytest.raises(RuntimeError, match="thread died"):
        dispatcher.evaluate(_rows(6))


def test_shutdown_closes_admission_after_draining(held_dispatcher):
    dispatcher, release, _threads = held_dispatcher
    pending = dispatcher.evaluate(_rows(1))
    release.set()
    dispatcher.shutdown()
    np.testing.assert_array_equal(pending.result(3)[0], [[1, 1]])
    with pytest.raises(RuntimeError, match="shut down"):
        dispatcher.evaluate(_rows(2))


def test_whole_request_larger_than_target_remains_one_request(held_dispatcher):
    dispatcher, release, _threads = held_dispatcher
    result = dispatcher.evaluate(_rows(1, 2, 3, 4, 5, 6))
    release.set()
    np.testing.assert_array_equal(result.result(3)[0][:, 0], [1, 2, 3, 4, 5, 6])
    assert dispatcher.stats["lifetime_batches"] == 1
    assert dispatcher.stats["lifetime_requests"] == 1


@pytest.mark.parametrize("ready", [False, None])
def test_coalescing_wait_requires_known_not_ready(held_dispatcher, monkeypatch, ready):
    dispatcher, release, _threads = held_dispatcher
    observed = []
    actual_take = dispatcher._take_route

    def take(**kwargs):
        observed.append(kwargs)
        return actual_take(**kwargs)

    def query(_event):
        if ready is None:
            raise RuntimeError("unknown query")
        return ready

    monkeypatch.setattr(_Event, "query", query)
    monkeypatch.setattr(dispatcher, "_take_route", take)
    result = dispatcher.evaluate(_rows(1))
    release.set()
    result.result(3)
    nonblocking = [call for call in observed if not call["blocking"]]
    assert nonblocking and nonblocking[0]["coalesce_wait"] is (ready is False)


def test_actual_compile_and_compiled_update_stay_on_dispatcher_thread(monkeypatch):
    monkeypatch.setattr(threaded, "DirectGPUEvaluator", _Backend)
    monkeypatch.setattr(threaded, "_snapshot_dynamo_counters", lambda: None)
    compiled_threads = []

    class Compiled:
        def __init__(self, model):
            self._orig_mod = model

        @property
        def value(self):
            return self._orig_mod.value

        def eval(self):
            return self

    def compile_model(model, *, mode):
        assert mode == "reduce-overhead"
        compiled_threads.append(threading.get_ident())
        return Compiled(model)

    monkeypatch.setattr(threaded.torch, "compile", compile_model)
    dispatcher = threaded.ThreadedDispatcher(
        _Model(), device="cpu", max_batch=8, compile_mode="reduce-overhead",
    )
    try:
        np.testing.assert_array_equal(dispatcher.evaluate(_rows(1)).result(3)[0], [[1, 1]])
        compiled_model = dispatcher._evaluator.model
        dispatcher.update_model(_Model(10))
        assert dispatcher._evaluator.model is compiled_model
        np.testing.assert_array_equal(dispatcher.evaluate(_rows(2)).result(3)[0], [[12, 12]])
        assert len(compiled_threads) == 1
        assert {event[2] for event in dispatcher._evaluator.events} == set(compiled_threads)
        assert compiled_threads[0] != threading.get_ident()
    finally:
        dispatcher.shutdown()
