"""Process-wide latch: while a device thread is wedged, workers park, never exit.

A thread stuck inside a device call (a stalled GPU call the engine gave up
on, a hung evaluator build) cannot be interrupted. Measured with the real
entry point on CPU devices: while such a thread is stuck, letting ANOTHER
thread that ran evaluator / torch work exit races the native teardown of that
thread against Python running elsewhere and corrupts the process (segfault,
"terminate called", "'traceback' object is not callable"). It needs a thread
stuck at the time: the same exits with no wedged thread, or with the exiting
thread parked instead, never crashed (see PR #1333's review loops).

So the rule, applied by every evaluator-running worker the UCI engine owns
(multi-GPU pool workers, root-parallel Gumbel groups, device-call runners,
the startup builder): its thread function is wrapped by :func:`parking`, and
when it would end while any thread is marked wedged it parks forever instead.
Threads are daemons; a parked thread blocks on an Event and costs no CPU.
``quit`` with a wedged thread hard-exits without closing anything.

What stays parked, and for how long: each worker thread that would have
ended while a device thread is wedged (at most one per device per pool /
runner instance built), plus whatever its frame still references (its
evaluator, chunker and buffers), until no thread is wedged any more -- for a
genuinely hung device, until the process exits. A pool rebuilt mid-session
after a wedge leaves its old workers parked; a rebuild per wedge is the bound.
"""
from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any, TypeVar, cast

_lock = threading.Lock()
_changed = threading.Condition(_lock)  # notified when a wedge clears
_wedged: dict[threading.Thread, str] = {}
_wedged_devices: dict[str, str] = {}  # device label -> what wedged, this session
_parked: list[threading.Thread] = []
# Parked threads wait on this forever: production never sets it (only the
# test-session reset does, so one pytest process does not pile up parked
# threads across tests).
_release = threading.Event()

_F = TypeVar("_F", bound=Callable[..., Any])


def mark_wedged(thread: threading.Thread | None, what: str, *, device: str | None = None) -> None:
    """``thread`` is stuck inside a device call (``what`` says which).

    ``device`` (its label) is also remembered for the rest of the session:
    a later pool rebuild leaves that device out at once (``wedged_earlier``)
    instead of building on it and waiting out the startup timeout."""
    with _lock:
        if thread is not None:
            _wedged[thread] = what
        if device is not None:
            _wedged_devices.setdefault(device, what)


def wedged_earlier(device: str) -> str | None:
    """What wedged on ``device`` earlier this session, or None."""
    with _lock:
        return _wedged_devices.get(device)


def clear_wedged(thread: threading.Thread | None) -> None:
    """``thread``'s stuck call returned: it is no longer wedged."""
    if thread is None:
        return
    with _lock:
        if _wedged.pop(thread, None) is not None:
            _changed.notify_all()


def _live_wedged_locked() -> list[threading.Thread]:
    dead = [th for th in _wedged if not th.is_alive()]
    for th in dead:
        del _wedged[th]
    return list(_wedged)


def wedged_threads() -> list[threading.Thread]:
    """Live threads currently marked wedged."""
    with _lock:
        return _live_wedged_locked()


def any_wedged() -> bool:
    return bool(wedged_threads())


def parked_threads() -> list[threading.Thread]:
    with _lock:
        return [th for th in _parked if th.is_alive()]


def park_if_wedged() -> None:
    """Called by a worker thread that is about to end: park instead while
    any OTHER thread is wedged (a wedged thread itself never reaches here
    until its call returns, and then it is cleared first). The parked thread
    ends once no thread is wedged any more -- the crash needs a thread stuck
    inside a device call at the time."""
    me = threading.current_thread()
    clear_wedged(me)
    with _lock:
        if not _live_wedged_locked():
            return
        _parked.append(me)
        release = _release
        while _live_wedged_locked() and not release.is_set():
            _changed.wait(timeout=1.0)
        if me in _parked:
            _parked.remove(me)


def parking(fn: _F) -> _F:
    """Wrap a worker thread's target with :func:`park_if_wedged` on exit."""

    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        finally:
            park_if_wedged()

    return cast("_F", run)


def join_unless_wedged(thread: threading.Thread, timeout: float) -> None:
    """``Thread.join`` that does not wait on a thread that will park.

    With a wedged thread in the process, ending workers park instead of
    exiting, so joining them would only spend the timeout."""
    if any_wedged():
        return
    thread.join(timeout=timeout)


def reset_session_for_tests() -> None:
    """Forget wedge marks and the wedged-device registry and let parked
    threads end (tests only: one pytest process is many engine "sessions")."""
    global _release
    with _lock:
        _wedged.clear()
        _wedged_devices.clear()
        _parked.clear()
        old, _release = _release, threading.Event()
        old.set()
        _changed.notify_all()
