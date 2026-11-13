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
from chess_anti_engine.uci.engine import Engine, EngineOptions
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


def _notes(error: BaseException) -> tuple[str, ...]:
    """Cleanup notes on any Python: `__notes__` (3.11+) or the 3.10 fallback."""
    return (*getattr(error, "__notes__", ()), *getattr(error, "_async_cleanup_notes", ()))


@pytest.mark.parametrize("kind", ["async", "factory"])
@pytest.mark.parametrize("cleanup", [None, "exception", "interrupt"])
def test_public_replacement_failure_keeps_old_primary_and_retires_new(monkeypatch, kind, cleanup):
    """BUILD, THEN SWAP: a replacement whose helpers fail to build is not adopted.

    The old evaluator and its walker pool stay installed and LIVE, the options
    keep describing them, and the replacement (whose ownership passed with the
    call) is closed exactly once. Before, the old pool was closed first and the
    worker kept pointing at it, so every later search failed.
    """
    old = make_owner(monkeypatch, kind)
    new = make_owner(monkeypatch, kind)
    worker = SearchWorker(old.evaluator, device="cpu", n_walkers=2)
    previous_pool = worker._walker_pool
    assert previous_pool is not None
    engine = Engine(worker, rebuild_evaluator=lambda *_a, **_kw: new.evaluator,
                    options=EngineOptions(search_walkers=2))  # the worker's 2 walkers
    max_batch_before = engine._options.max_batch
    error = MemoryError("bounded walker input allocation failure")
    cause = ValueError("original allocation cause")
    note_cleanup(error, "existing cleanup note")
    if cleanup is not None:
        new.close_error = RuntimeError("new cleanup failure") if cleanup == "exception" else KeyboardInterrupt("new cleanup interrupt")
    try:
        with monkeypatch.context() as patcher:
            fault_allocation(patcher, error, cause)
            # `_dispatch`, not `dispatch`: the public entry contains handler
            # failures so the engine survives them (test_uci_never_die.py);
            # this test is about ownership and the exception itself.
            with pytest.raises(MemoryError) as caught:
                engine._dispatch(CmdSetOption(name="MaxBatch", value="64"))
        assert caught.value is error
        assert caught.value.__cause__ is cause
        # Ownership facts, runnable on every Python version:
        assert worker._evaluator is old.evaluator
        assert old.close_calls == 0
        assert old.thread.is_alive()
        assert worker._walker_pool is previous_pool
        assert all(t.is_alive() for t in previous_pool._threads)
        live_walkers = {t for t in threading.enumerate() if t.name.startswith("walker-") and t.is_alive()}
        assert live_walkers == set(previous_pool._threads)  # nothing half-built left running
        assert engine._options.max_batch == max_batch_before
        assert new.close_calls == 1
        assert not new.thread.is_alive()
        assert "existing cleanup note" in _notes(caught.value)
        if cleanup is not None:
            assert any("replacement evaluator close failed" in note for note in _notes(caught.value))
        engine.close()
        assert old.close_calls == 1
        assert new.close_calls == 1
        assert not old.thread.is_alive()
        assert_no_walkers()
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
    engine = Engine(worker, rebuild_evaluator=lambda *_a, **_kw: new.evaluator,
                    options=EngineOptions(search_walkers=2))  # the worker's 2 walkers
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
    engine = Engine(worker, rebuild_evaluator=lambda *_a, **_kw: owner.evaluator,
                    options=EngineOptions(search_walkers=2))  # the worker's 2 walkers
    error = MemoryError("same-owner walker allocation failure")
    try:
        with monkeypatch.context() as patcher:
            if fault:
                fault_allocation(patcher, error, ValueError("same-owner cause"))
                # `_dispatch`: the public `dispatch` contains handler failures.
                with pytest.raises(MemoryError) as caught:
                    engine._dispatch(CmdSetOption(name="MaxBatch", value="64"))
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