"""Actual WorkerSession methods with fake callbacks, no model or engine work."""
from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from chess_anti_engine.worker import WorkerSession


@pytest.fixture
def session() -> Any:
    worker: Any = object.__new__(WorkerSession)
    worker.log = logging.getLogger(__name__)
    worker._model_watch_lock = threading.Lock()
    worker._model_watch_stop = threading.Event()
    worker._model_watch_thread = None
    worker._model_watch_started = False
    worker._selfplay_session_active = True
    worker._check_model_update_locked = lambda: None
    worker._check_model_freshness = lambda: None
    yield worker
    worker._stop_model_watch_thread()


def test_inactive_session_rejects_delayed_ply_and_watcher_callbacks(session: Any) -> None:
    calls: list[str] = []
    session._check_model_update_locked = lambda: calls.append("swap")
    session._check_model_freshness = lambda: calls.append("freshness")
    session._set_selfplay_session_active(False)
    session._check_model_update()
    session._model_watch_iteration()
    assert calls == []


def test_recovered_ply_entry_must_not_update_after_session_has_ended(session: Any) -> None:
    """This original method exists on the recovered base and lacks this gate."""
    calls: list[str] = []
    session._selfplay_session_active = False
    session._check_model_update_locked = lambda: calls.append("late-old-session-update")
    session._check_model_update()
    assert calls == []


def test_session_handoff_waits_for_inflight_swap_before_next_setup(session: Any) -> None:
    entered, release, attempted, handed_off = (threading.Event() for _ in range(4))
    calls: list[str] = []

    def swap() -> None:
        entered.set()
        assert release.wait(5), "test did not release fake swap"
        calls.append("old-swap-finished")

    def handoff() -> None:
        attempted.set()
        session._set_selfplay_session_active(False)
        calls.append("new-session-setup")
        handed_off.set()

    session._check_model_update_locked = swap
    watch = threading.Thread(target=session._model_watch_iteration)
    owner = threading.Thread(target=handoff)
    watch.start()
    try:
        assert entered.wait(5)
        owner.start()
        assert attempted.wait(5)
        assert not handed_off.wait(0.05), "session setup overlapped old swap"
    finally:
        release.set()
        watch.join(5)
        if owner.ident is not None:
            owner.join(5)
    assert not watch.is_alive() and not owner.is_alive()
    assert handed_off.is_set()
    assert calls == ["old-swap-finished", "new-session-setup"]
    session._check_model_update()
    assert calls == ["old-swap-finished", "new-session-setup"]


def test_freshness_and_update_share_one_session_boundary(session: Any) -> None:
    calls: list[str] = []

    def check(name: str) -> None:
        acquired = session._model_watch_lock.acquire(blocking=False)
        if acquired:
            session._model_watch_lock.release()
        assert not acquired, f"{name} escaped session handoff lock"
        calls.append(name)

    session._check_model_update_locked = lambda: check("update")
    session._check_model_freshness = lambda: check("freshness")
    session._model_watch_iteration()
    assert calls == ["update", "freshness"]
    session._set_selfplay_session_active(False)
    session._model_watch_iteration()
    assert len(calls) == 2
    session._set_selfplay_session_active(True)
    session._model_watch_iteration()
    assert calls == ["update", "freshness", "update", "freshness"]


def test_cleanup_wakes_and_joins_idle_watcher(session: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("CAE_WORKER_MODEL_WATCH_S", "60")
    events: list[str] = []
    session._boot_hang_watchdog = SimpleNamespace(stop=lambda: events.append("watchdog"))
    session._gil_probe = SimpleNamespace(close=lambda: events.append("probe"))

    def close_client() -> None:
        assert not session._model_watch_thread.is_alive()
        assert not session._selfplay_session_active
        events.append("client")

    session.inference_client = SimpleNamespace(close=close_client)
    session.sf = None
    session._start_model_watch_thread()
    thread = session._model_watch_thread
    assert thread.is_alive()
    session._cleanup()
    assert not thread.is_alive()
    assert events == ["watchdog", "probe", "client"]
    # Cleanup is terminal: a late start call cannot resurrect the watcher.
    session._start_model_watch_thread()
    assert session._model_watch_thread is thread


def test_cleanup_drains_callback_before_closing_clients(session: Any) -> None:
    entered, release, cleaning = (threading.Event() for _ in range(3))
    events: list[str] = []

    def update() -> None:
        entered.set()
        assert release.wait(5)
        events.append("update-finished")

    def close_client() -> None:
        assert session._model_watch_stop.is_set()
        assert not session._selfplay_session_active
        events.append("client-closed")

    session._check_model_update_locked = update
    session._boot_hang_watchdog = SimpleNamespace(stop=lambda: None)
    session._gil_probe = SimpleNamespace(close=lambda: None)
    session.inference_client = SimpleNamespace(close=close_client)
    session.sf = None

    def cleanup() -> None:
        cleaning.set()
        session._cleanup()

    callback = threading.Thread(target=session._model_watch_iteration)
    owner = threading.Thread(target=cleanup)
    callback.start()
    try:
        assert entered.wait(5)
        owner.start()
        assert cleaning.wait(5)
        assert events == []
    finally:
        release.set()
        callback.join(5)
        if owner.ident is not None:
            owner.join(5)
    assert not owner.is_alive() and not callback.is_alive()
    assert events == ["update-finished", "client-closed"]
