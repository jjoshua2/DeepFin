"""Process-wide latch: after a device fault, evaluator workers park, never exit.

Measured with the real entry point on CPU devices (PR #1333 review loops):
once a device call has ended in an exception -- a device that raised, or a
stalled call the engine gave up on -- letting threads that ran evaluator /
torch work END mid-session, or tearing the interpreter down normally at exit,
intermittently corrupts the process (segfault in the garbage collector at
``Py_Exit``, "terminate called", type-confused objects such as "'traceback'
object is not callable"). No wedged thread is needed: a non-``devices[0]``
root-parallel group that raises crashed 3-6 runs in 55 when the healthy groups
were closed, 0 in 55 when they were left running. The underlying refcount /
lifetime error was not found (see the PR for what was ruled out), so the rule
is structural:

* every evaluator-running worker the UCI engine owns (multi-GPU pool
  workers, root-parallel Gumbel groups, device-call runners, the startup
  builder) has its thread function wrapped by :func:`parking`;
* once :func:`note_device_fault` has been called this session (every place
  that drops a device, plus every :func:`mark_wedged`), a worker that would
  end parks forever instead (blocked on an Event, no CPU, daemon);
* ``quit`` after a device fault exits with ``os._exit`` (no interpreter
  teardown) with the status it would have had.

Before any fault, threads end and are joined normally.

What stays parked: every worker thread that would have ended after the
first fault, plus what its frame references (its evaluator, chunker and
buffers), until the process exits. That is one thread per device of each
pool / routing chain / root-parallel pool closed after the fault, plus one
call thread per primary-evaluator runner replaced. Each option change that
rebuilds the evaluators after a fault adds that many again (two per rebuild
on a 2-device pool); a restart frees them.
"""
from __future__ import annotations

import functools
import threading
import time
import types
from collections.abc import Callable
from typing import Any, TypeVar, cast

_lock = threading.Lock()
_changed = threading.Condition(_lock)  # notified when a wedge clears
_wedged: dict[threading.Thread, str] = {}
_wedged_devices: dict[str, str] = {}  # device label -> what wedged, this session
_parked: list[threading.Thread] = []
_fault: list[str] = []  # the first device fault this session (empty = none)
# Parked threads wait on this forever: production never sets it (only the
# test-session reset does, so one pytest process does not pile up parked
# threads across tests).
_release = threading.Event()

_F = TypeVar("_F", bound=Callable[..., Any])


def note_device_fault(what: str) -> None:
    """A device call ended in an exception, or a device was dropped (``what``
    says which). From now on ending workers park and ``quit`` hard-exits."""
    with _lock:
        if not _fault:
            _fault.append(what)


def device_fault() -> str | None:
    """The first device fault this session, or None."""
    with _lock:
        return _fault[0] if _fault else None


def mark_wedged(thread: threading.Thread | None, what: str, *, device: str | None = None) -> None:
    """``thread`` is stuck inside a device call (``what`` says which).

    ``device`` (its label) is also remembered for the rest of the session:
    a later pool rebuild leaves that device out at once (``wedged_earlier``)
    instead of building on it and waiting out the startup timeout. A wedge
    is also a device fault (:func:`note_device_fault`)."""
    with _lock:
        if not _fault:
            _fault.append(what)
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
    """Called by a worker thread that is about to end: after a device fault
    this session, park until the process exits instead (a wedged thread whose
    call returns is cleared first, then parks like any other)."""
    me = threading.current_thread()
    clear_wedged(me)
    with _lock:
        if not _fault and not _live_wedged_locked():
            return
        _parked.append(me)
        _changed.notify_all()  # ``join_unless_wedged`` waits for this
        release = _release
        while not release.is_set():
            _changed.wait(timeout=1.0)
        if me in _parked:
            _parked.remove(me)


def parking(fn: _F) -> _F:
    """Wrap a worker thread's target with :func:`park_if_wedged` on exit.

    A bound method stays a bound method on the same owner (``__self__`` is
    kept), so code that finds a thread's owner from ``thread._target`` keeps
    working."""
    func = getattr(fn, "__func__", None)
    owner = getattr(fn, "__self__", None)
    if func is not None and owner is not None:

        @functools.wraps(func)
        def run_bound(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return func(self_obj, *args, **kwargs)
            finally:
                park_if_wedged()

        return cast("_F", types.MethodType(run_bound, owner))

    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        finally:
            park_if_wedged()

    return cast("_F", run)


def join_unless_wedged(thread: threading.Thread, timeout: float) -> None:
    """``Thread.join`` that also returns once ``thread`` has parked.

    After a device fault ending workers park instead of exiting, so a plain
    join would only spend the timeout. This waits (at most ``timeout``) until
    the thread has ended or parked, so its cleanup has run when it returns."""
    if device_fault() is None and not any_wedged():
        thread.join(timeout=timeout)  # no fault: workers end normally
        return
    deadline = time.monotonic() + max(0.0, timeout)
    with _lock:
        while thread.is_alive() and thread not in _parked:
            left = deadline - time.monotonic()
            if left <= 0:
                return
            _changed.wait(timeout=min(left, 0.05))


def reset_session_for_tests() -> None:
    """Forget wedge marks and the wedged-device registry and let parked
    threads end (tests only: one pytest process is many engine "sessions")."""
    global _release
    with _lock:
        _wedged.clear()
        _wedged_devices.clear()
        _parked.clear()
        _fault.clear()
        old, _release = _release, threading.Event()
        old.set()
        _changed.notify_all()
