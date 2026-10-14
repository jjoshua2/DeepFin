"""Public evaluator replacement retains and retires the exact primary owners."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import threading
from typing import Any

import pytest
import torch

from chess_anti_engine.async_safety import note_cleanup
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import CUDAOwnerDispatcher
from chess_anti_engine.uci.__main__ import _make_evaluator_factory
from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import CmdSetOption
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci import walker_pool as walkers


class ConstantModel(torch.nn.Module):
    def forward(self, x):
        return {"policy": torch.zeros((x.shape[0], 4672), dtype=torch.float32),
                "wdl": torch.zeros((x.shape[0], 3), dtype=torch.float32)}


@dataclass
class Owner:
    evaluator: Any
    thread: threading.Thread
    close_calls: int = 0
    close_error: BaseException | None = None


def make_owner(monkeypatch, kind):
    evaluator: Any
    if kind == "async":
        evaluator = CUDAOwnerDispatcher(DirectGPUEvaluator(ConstantModel(), device="cpu", max_batch=64, n_slots=2))
        thread = evaluator._owner
    else:
        evaluator = _make_evaluator_factory([ConstantModel()], ["cpu"], coalesce=True,
                                            n_walkers=2, walker_gather=1, compile_mode=None)(64)
        thread = evaluator._submitter
    owner = Owner(evaluator, thread)
    actual_close = evaluator.close

    def close():
        owner.close_calls += 1
        actual_close()
        if owner.close_error is not None:
            raise owner.close_error

    monkeypatch.setattr(evaluator, "close", close)
    return owner


def clean(engine, *owners):
    with contextlib.suppress(BaseException):
        engine.close()
    for owner in owners:
        if owner.thread.is_alive():
            with contextlib.suppress(BaseException):
                owner.evaluator.close()


def fault_allocation(monkeypatch, error, cause):
    real_empty = walkers.np.empty

    def empty(*args, **kwargs):
        if threading.current_thread().name.startswith("walker-"):
            raise error from cause
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(walkers.np, "empty", empty)


def assert_no_walkers():
    assert not any(t.name.startswith("walker-") and t.is_alive() for t in threading.enumerate())


@pytest.mark.parametrize("kind", ["async", "factory"])
@pytest.mark.parametrize("cleanup", [None, "exception", "interrupt"])
def test_public_replacement_failure_retires_old_and_preserves_primary(monkeypatch, kind, cleanup):
    old = make_owner(monkeypatch, kind)
    new = make_owner(monkeypatch, kind)
    worker = SearchWorker(old.evaluator, device="cpu", n_walkers=2)
    previous_pool = worker._walker_pool
    assert previous_pool is not None
    engine = Engine(worker, rebuild_evaluator=lambda *_a, **_kw: new.evaluator)
    error = MemoryError("bounded walker input allocation failure")
    cause = ValueError("original allocation cause")
    note_cleanup(error, "existing cleanup note")
    if cleanup is not None:
        old.close_error = RuntimeError("old cleanup failure") if cleanup == "exception" else KeyboardInterrupt("old cleanup interrupt")
    try:
        with monkeypatch.context() as patcher:
            fault_allocation(patcher, error, cause)
            with pytest.raises(MemoryError) as caught:
                engine.dispatch(CmdSetOption(name="MaxBatch", value="64"))
        assert caught.value is error
        assert caught.value.__cause__ is cause
        assert "existing cleanup note" in getattr(caught.value, "__notes__", ())
        assert old.close_calls == 1
        assert not old.thread.is_alive()
        assert worker._evaluator is new.evaluator
        assert engine._options.max_batch == 64
        assert new.close_calls == 0
        assert new.thread.is_alive()
        assert not any(t.is_alive() for t in previous_pool._threads)
        assert_no_walkers()
        if cleanup is not None:
            assert any("outgoing evaluator close failed" in note for note in getattr(caught.value, "__notes__", ()))
        engine.close()
        assert old.close_calls == 1
        assert new.close_calls == 1
        assert not new.thread.is_alive()
    finally:
        clean(engine, old, new)


@pytest.mark.parametrize("kind", ["async", "factory"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_public_healthy_replacement_preserves_successful_transfer(monkeypatch, kind, cleanup_fails):
    old = make_owner(monkeypatch, kind)
    new = make_owner(monkeypatch, kind)
    worker = SearchWorker(old.evaluator, device="cpu", n_walkers=2)
    previous_pool = worker._walker_pool
    assert previous_pool is not None
    engine = Engine(worker, rebuild_evaluator=lambda *_a, **_kw: new.evaluator)
    if cleanup_fails:
        old.close_error = RuntimeError("existing suppressed old cleanup failure")
    try:
        engine.dispatch(CmdSetOption(name="MaxBatch", value="64"))
        assert worker._evaluator is new.evaluator
        assert old.close_calls == 1
        assert not old.thread.is_alive()
        assert new.close_calls == 0
        assert new.thread.is_alive()
        assert not any(t.is_alive() for t in previous_pool._threads)
        replacement_pool = worker._walker_pool
        assert replacement_pool is not None
        assert any(t.is_alive() for t in replacement_pool._threads)
        engine.close()
        engine.close()
        assert old.close_calls == 1
        assert new.close_calls == 1
        assert not new.thread.is_alive()
        assert not any(t.is_alive() for t in replacement_pool._threads)
    finally:
        clean(engine, old, new)


@pytest.mark.parametrize("kind", ["async", "factory"])
@pytest.mark.parametrize("fault", [False, True])
def test_public_same_owner_replacement_does_not_close_borrowed_current_owner(monkeypatch, kind, fault):
    owner = make_owner(monkeypatch, kind)
    worker = SearchWorker(owner.evaluator, device="cpu", n_walkers=2)
    engine = Engine(worker, rebuild_evaluator=lambda *_a, **_kw: owner.evaluator)
    error = MemoryError("same-owner walker allocation failure")
    try:
        with monkeypatch.context() as patcher:
            if fault:
                fault_allocation(patcher, error, ValueError("same-owner cause"))
                with pytest.raises(MemoryError) as caught:
                    engine.dispatch(CmdSetOption(name="MaxBatch", value="64"))
                assert caught.value is error
            else:
                engine.dispatch(CmdSetOption(name="MaxBatch", value="64"))
        assert worker._evaluator is owner.evaluator
        assert owner.close_calls == 0
        assert owner.thread.is_alive()
        engine.close()
        engine.close()
        assert owner.close_calls == 1
        assert not owner.thread.is_alive()
        assert_no_walkers()
    finally:
        clean(engine, owner)