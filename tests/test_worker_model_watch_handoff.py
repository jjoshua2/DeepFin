"""Model-watch ownership handoff and watcher shutdown, on real WorkerSession methods.

Pre-fix: ``_run_selfplay`` cleared ``_selfplay_session_active`` without taking
``_model_watch_lock``, so a model swap the watcher thread (or a selfplay
thread's ``on_step``) had already started could still be mutating
``self.model`` / ``model_sha`` while ``run()`` set up the next session; and
``_cleanup`` never stopped or joined the watcher, closing the inference client
and Stockfish under an in-flight update.

Synchronisation here is by Events only. "Thread X is now blocked on the lock"
is observed deterministically through ``_ProbeLock``, which records a
blocking acquire attempt before delegating; the 5 s waits are failure bounds,
never the synchronisation itself.

Originally from PR #1110 (commits 2f2911111, 416a4189a); rewritten around the
probe lock, plus the freshness-outside-the-lock and bounded-cleanup cases.
"""
from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from chess_anti_engine.worker import WorkerSession

_BOUND_S = 5.0


class _ProbeLock:
    """threading.Lock that signals when a thread starts a BLOCKING acquire."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.blocking_attempt = threading.Event()
        self.attempt_threads: list[str] = []

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if blocking:
            self.attempt_threads.append(threading.current_thread().name)
            self.blocking_attempt.set()
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()


@pytest.fixture
def session() -> Any:
    worker: Any = object.__new__(WorkerSession)
    worker.log = logging.getLogger(__name__)
    worker._model_watch_lock = _ProbeLock()
    worker._model_watch_stop = threading.Event()
    worker._model_watch_thread = None
    worker._model_watch_started = False
    worker._selfplay_session_active = True
    worker._check_model_update_locked = lambda: None
    worker._check_model_freshness = lambda: None
    yield worker
    worker._stop_model_watch_thread(timeout_s=_BOUND_S)


def _cleanup_collaborators(worker: Any, close_client: Any) -> None:
    worker._boot_hang_watchdog = SimpleNamespace(stop=lambda: None)
    worker._gil_probe = SimpleNamespace(close=lambda: None)
    worker.inference_client = SimpleNamespace(close=close_client)
    worker.sf = None


def test_inactive_session_rejects_late_update_callbacks(session: Any) -> None:
    """A late on_step/watcher call after the session ended must not swap."""
    calls: list[str] = []
    session._check_model_update_locked = lambda: calls.append("late-update")
    session._set_selfplay_session_active(False)
    session._check_model_update()
    assert calls == []


def test_start_stop_restart_sequence_without_contention(session: Any) -> None:
    """No swap in flight: the handoff never blocks and gating follows the session."""
    calls: list[str] = []
    session._check_model_update_locked = lambda: calls.append("update")
    session._set_selfplay_session_active(True)
    session._check_model_update()
    session._set_selfplay_session_active(False)
    session._check_model_update()
    session._set_selfplay_session_active(True)
    session._check_model_update()
    assert calls == ["update", "update"]
    assert not session._model_watch_lock.locked()


def test_session_handoff_waits_for_inflight_swap(session: Any) -> None:
    entered, release, handed_off = (threading.Event() for _ in range(3))
    calls: list[str] = []

    def swap() -> None:
        entered.set()
        assert release.wait(_BOUND_S), "test did not release the fake swap"
        calls.append("old-swap-finished")

    def handoff() -> None:
        session._set_selfplay_session_active(False)
        calls.append("next-session-setup")
        handed_off.set()

    session._check_model_update_locked = swap
    watch = threading.Thread(target=session._check_model_update, name="watch")
    owner = threading.Thread(target=handoff, name="owner")
    watch.start()
    try:
        assert entered.wait(_BOUND_S)
        lock = session._model_watch_lock
        lock.blocking_attempt.clear()
        owner.start()
        # Deterministic: the owner has reached the lock; the swap holds it.
        assert lock.blocking_attempt.wait(_BOUND_S), "handoff never took the watch lock"
        assert lock.attempt_threads[-1] == "owner"
        assert not handed_off.is_set(), "next session setup overlapped the in-flight swap"
        assert calls == []
    finally:
        release.set()
        watch.join(_BOUND_S)
        if owner.ident is not None:
            owner.join(_BOUND_S)
    assert not watch.is_alive()
    assert not owner.is_alive()
    assert calls == ["old-swap-finished", "next-session-setup"]
    # Admitted after the handoff: no swap.
    session._check_model_update()
    assert calls == ["old-swap-finished", "next-session-setup"]


def test_freshness_alarm_still_runs_while_a_swap_holds_the_lock(
    session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stale-tag alarm must fire while another caller is wedged mid-swap.

    That is the very failure it reports, so it may not sit behind the lock.
    """
    monkeypatch.setenv("CAE_WORKER_MODEL_WATCH_S", "0.01")
    fresh = threading.Event()
    calls: list[str] = []
    session._check_model_update_locked = lambda: calls.append("update")
    session._check_model_freshness = fresh.set
    assert session._model_watch_lock.acquire(blocking=False)  # a wedged on_step
    try:
        session._start_model_watch_thread()
        assert fresh.wait(_BOUND_S), "freshness alarm blocked behind the swap lock"
        assert calls == []
    finally:
        session._model_watch_lock.release()


def test_cleanup_wakes_and_joins_idle_watcher(
    session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAE_WORKER_MODEL_WATCH_S", "600")
    events: list[str] = []

    def close_client() -> None:
        assert not session._model_watch_thread.is_alive()
        assert not session._selfplay_session_active
        events.append("client")

    _cleanup_collaborators(session, close_client)
    session._start_model_watch_thread()
    thread = session._model_watch_thread
    assert thread is not None
    assert thread.is_alive()
    # With a 600 s poll the watcher only exits this fast if the stop event
    # woke it; a sleep()-based loop would hit the cleanup bound and fail.
    assert session._stop_model_watch_thread(timeout_s=_BOUND_S)
    assert not thread.is_alive()
    session._cleanup()
    assert events == ["client"]
    # Cleanup is terminal: a late start cannot resurrect the watcher.
    session._start_model_watch_thread()
    assert session._model_watch_thread is thread


def test_cleanup_drains_inflight_update_before_closing_clients(session: Any) -> None:
    entered, release = threading.Event(), threading.Event()
    events: list[str] = []

    def update() -> None:
        entered.set()
        assert release.wait(_BOUND_S)
        events.append("update-finished")

    def close_client() -> None:
        assert session._model_watch_stop.is_set()
        assert not session._selfplay_session_active
        events.append("client-closed")

    session._check_model_update_locked = update
    _cleanup_collaborators(session, close_client)
    callback = threading.Thread(target=session._check_model_update, name="callback")
    owner = threading.Thread(target=session._cleanup, name="cleanup")
    callback.start()
    try:
        assert entered.wait(_BOUND_S)
        lock = session._model_watch_lock
        lock.blocking_attempt.clear()
        owner.start()
        assert lock.blocking_attempt.wait(_BOUND_S), "cleanup never took the watch lock"
        assert lock.attempt_threads[-1] == "cleanup"
        assert events == []
    finally:
        release.set()
        callback.join(_BOUND_S)
        if owner.ident is not None:
            owner.join(_BOUND_S)
    assert not owner.is_alive()
    assert not callback.is_alive()
    assert events == ["update-finished", "client-closed"]


def test_cleanup_wait_is_bounded_when_the_update_is_stuck(session: Any) -> None:
    """A stuck network call in the watcher must not keep the worker from exiting."""
    events: list[str] = []
    _cleanup_collaborators(session, lambda: events.append("client-closed"))
    assert session._model_watch_lock.acquire(blocking=False)  # stuck holder
    try:
        assert session._stop_model_watch_thread(timeout_s=0.01) is False
        assert session._model_watch_stop.is_set()
        assert not session._selfplay_session_active
    finally:
        session._model_watch_lock.release()
    session._cleanup()
    assert events == ["client-closed"]


def test_run_selfplay_hands_off_only_through_the_locked_setter() -> None:
    """Source-shape pin (not behavioural): ``_run_selfplay`` cannot be driven
    without a server, so this pins that both session-boundary flips go through
    ``_set_selfplay_session_active`` (the behaviour of which is tested above)
    and that nothing in the worker assigns the flag outside it except
    ``__init__`` and the shutdown path.
    """
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(WorkerSession._run_selfplay))
    assert src.count("self._set_selfplay_session_active(True)") == 1
    assert src.count("self._set_selfplay_session_active(False)") == 1
    tree = ast.parse(textwrap.dedent(inspect.getsource(WorkerSession)))
    assigners = {
        fn.name
        for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
        for node in ast.walk(fn) if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and target.attr == "_selfplay_session_active"
    }
    assert assigners == {
        "__init__", "_set_selfplay_session_active", "_stop_model_watch_thread",
    }


def test_cleanup_before_first_session_prevents_a_later_watcher_start(
    session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop is terminal even if the watcher was never started."""
    monkeypatch.setenv("CAE_WORKER_MODEL_WATCH_S", "600")
    assert session._stop_model_watch_thread(timeout_s=_BOUND_S)
    session._start_model_watch_thread()
    assert session._model_watch_thread is None
    assert not session._model_watch_started
