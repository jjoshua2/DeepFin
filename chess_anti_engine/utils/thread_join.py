"""Join a thread AND wait for its OS thread to exit.

On CPython 3.10-3.12 ``Thread.join()`` (and ``is_alive()``) report a thread
ended as soon as its Python thread state is gone, while the OS thread is still
running its C/C++ ``thread_local`` destructors -- torch's among them. Measured
on CPython 3.10.12 (CPU): a thread that ran a torch forward or backward was
still in ``/proc/self/task`` after ``join`` returned in 40 of 40 runs, for
0.2-8 ms (median 1.3 ms), with torch 2.11 and with torch 2.14 alike; 0 of 40
for a thread that ran no torch. A caller that then closes evaluators or
finalizes the interpreter races that teardown; on the UCI root-parallel raise
path (PR #1333, torch 2.11) that segfaulted in the exit-time garbage collector
in 18 of 60 runs, 0 of 60 with this wait. Torch 2.14 was not observed to crash.
From CPython 3.13 ``Thread.join`` joins the OS thread itself and the wait below
finds it gone at once.

Use :func:`join_thread` instead of ``Thread.join`` for a thread that ran torch
work when evaluator teardown or interpreter exit follows, and
:func:`os_joined` around a ``ThreadPoolExecutor`` whose workers ran torch work.
The wait is pure: it adds no ordering, raises nothing, and is bounded by
:data:`OS_EXIT_WAIT_S`; a cap hit is logged once per thread at WARNING (to the
logging handlers, i.e. stderr -- never stdout, so it cannot corrupt the UCI
protocol stream) and the caller carries on. Linux only (``/proc/self/task``);
elsewhere it returns at once.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
import weakref
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

_log = logging.getLogger(__name__)

# Cap on waiting for an ended thread's OS thread to exit. The measured linger
# is milliseconds. A caller with a deadline passes a smaller ``cap_s``.
OS_EXIT_WAIT_S = 1.0
_POLL_S = 0.0005
# A bound of at least this much that runs out is a stuck teardown (a cap hit:
# warned once, remembered). The measured linger is under 10 ms, so a tiny
# caller budget (``go movetime 1``) running out is not one; a deadline that
# allowed (about) the full cap is, even if the join itself used a little of it.
STUCK_AFTER_S = OS_EXIT_WAIT_S / 2

# Threads whose wait expired once: never waited for again (a reused tid, or a
# teardown that really is stuck, would otherwise cost the cap on every call).
_gave_up_lock = threading.Lock()
_gave_up: weakref.WeakSet[threading.Thread] = weakref.WeakSet()
# Threads whose OS thread was seen gone: a later call (a site reached on every
# command for the same ended thread) returns at once, even if the tid is reused.
_gone: weakref.WeakSet[threading.Thread] = weakref.WeakSet()

# Task start times recorded by ``note_started`` for threads that are joined
# (or found ended) later, when the caller can no longer read it.
_noted_lock = threading.Lock()
_noted: weakref.WeakKeyDictionary[threading.Thread, int] = weakref.WeakKeyDictionary()

_E = TypeVar("_E", bound=ThreadPoolExecutor)


def task_start_time(tid: int) -> int | None:
    """``starttime`` (field 22 of ``/proc/self/task/<tid>/stat``, clock ticks
    since boot) of the task with this tid, or None when there is none. Two
    tasks that share a tid one after the other have different start times."""
    try:
        with open(f"/proc/self/task/{tid}/stat", "rb") as f:
            data = f.read()
    except OSError:
        return None
    try:
        # Fields after "(comm)": state is field 3, so starttime is index 19.
        return int(data[data.rindex(b")") + 2:].split()[19])
    except (ValueError, IndexError):
        return None


def note_started(thread: threading.Thread) -> None:
    """Record ``thread``'s task start time right after ``thread.start()``, so a
    later :func:`await_os_thread_exit` on it -- even once it has ended -- stops
    at once if its tid has been reused. No-op if it already ended."""
    tid = thread.native_id
    if tid is None or not thread.is_alive():
        return
    start_time = task_start_time(tid)
    if start_time is not None and thread.is_alive():
        with _noted_lock:
            _noted[thread] = start_time


def _give_up(thread: threading.Thread, cap_s: float, *, log: bool) -> None:
    """A cap hit: never wait for ``thread`` again; warn (once per thread) if ``log``."""
    with _gave_up_lock:
        if thread in _gave_up:
            return
        _gave_up.add(thread)
    if log:
        _log.warning(
            "thread %r (native id %s) ended but its OS thread was still running %.3g s "
            "after join; carrying on without waiting for it",
            thread.name, thread.native_id, cap_s,
        )


def _await(
    thread: threading.Thread, cap: float, start_time: int | None, *,
    cap_hit_if_expired: bool, log: bool = True,
) -> bool:
    """The wait itself. ``cap_hit_if_expired`` (decided by the caller from the
    bound it set, not from the time left here): running out is a cap hit
    (remembered, and warned if ``log``)."""
    tid = getattr(thread, "native_id", None)
    if tid is None or thread.is_alive():
        return True
    if thread in _gone:
        return True
    if thread in _gave_up:
        return False
    if start_time is None:
        with _noted_lock:
            start_time = _noted.get(thread)
    stop = time.monotonic() + max(0.0, min(OS_EXIT_WAIT_S, cap))
    while True:
        seen = task_start_time(tid)
        if seen is None or (start_time is not None and seen != start_time):
            with _gave_up_lock:
                _gone.add(thread)
            return True
        if time.monotonic() >= stop:
            if cap_hit_if_expired:
                _give_up(thread, OS_EXIT_WAIT_S, log=log)
            return False
        time.sleep(_POLL_S)


def await_os_thread_exit(
    thread: threading.Thread, *, cap_s: float | None = None, start_time: int | None = None,
) -> bool:
    """After ``thread`` ended, wait (bounded) until its OS thread is gone.

    ``cap_s`` (default :data:`OS_EXIT_WAIT_S`, never more) bounds the wait, so a
    caller with a deadline passes what is left of it. ``start_time``, when the
    caller read it (:func:`task_start_time`) while the thread was alive, makes
    the wait stop as soon as the tid names a different task (a reused tid).

    Returns True when there is nothing left to wait for (the OS thread is gone,
    the thread never started, it is still running -- the caller's join timed
    out, which is the caller's business -- or not Linux). Returns False when
    the bound expired. A bound of at least :data:`STUCK_AFTER_S` (``cap_s``
    omitted, or that large) running out is a cap hit: logged once, and that
    thread is never waited for again. A smaller caller budget that runs out (a
    tiny clock) is neither logged nor remembered, so a later call with more
    budget still waits."""
    module_cap = cap_s is None or cap_s >= STUCK_AFTER_S
    cap = OS_EXIT_WAIT_S if cap_s is None else cap_s
    return _await(thread, cap, start_time, cap_hit_if_expired=module_cap)


def join_thread(
    thread: threading.Thread, timeout: float | None = None, *, deadline: float | None = None,
) -> None:
    """``thread.join(timeout)``, then :func:`await_os_thread_exit` if it ended.

    Same contract as ``Thread.join``: returns after ``timeout`` with the thread
    still alive if it did not end (check ``is_alive()``); never raises for the
    OS-thread wait. ``deadline`` (``time.monotonic()`` value) also bounds the
    OS-thread wait; when it allowed at least :data:`STUCK_AFTER_S` at the call,
    running out is a cap hit (logged once), however long the join itself took.
    A thread alive at the call has its task start time read first, so a reused
    tid ends the wait at once."""
    module_cap = deadline is None or deadline - time.monotonic() >= STUCK_AFTER_S
    tid = thread.native_id if thread.is_alive() else None
    start_time = task_start_time(tid) if tid is not None else None
    thread.join(timeout)
    cap = OS_EXIT_WAIT_S if deadline is None else deadline - time.monotonic()
    _await(thread, cap, start_time, cap_hit_if_expired=module_cap)


# Pool cap hits (``os_joined``) are logged at most once per this many seconds:
# pool threads are new on every shutdown (the selfplay pool: every batch), so
# once-per-thread would still be one line per batch for a persistent fault.
POOL_WARN_INTERVAL_S = 60.0
_pool_warn_lock = threading.Lock()
_pool_warn = {"last": float("-inf"), "suppressed": 0}


def _warn_pool_cap_hit(stuck: list[threading.Thread]) -> None:
    now = time.monotonic()
    with _pool_warn_lock:
        if now - _pool_warn["last"] < POOL_WARN_INTERVAL_S:
            _pool_warn["suppressed"] += 1
            return
        suppressed = int(_pool_warn["suppressed"])
        _pool_warn["last"], _pool_warn["suppressed"] = now, 0
    _log.warning(
        "thread pool shutdown: %d worker thread(s) (%s) ended but their OS threads were "
        "still running %.3g s after shutdown; carrying on without waiting for them "
        "(%d more pool shutdown(s) like this in the last %.0f s were not logged)",
        len(stuck), ", ".join(repr(t.name) for t in stuck), OS_EXIT_WAIT_S,
        suppressed, POOL_WARN_INTERVAL_S,
    )


def _start_times(threads: list[threading.Thread]) -> dict[threading.Thread, int]:
    out: dict[threading.Thread, int] = {}
    for thread in threads:
        tid = thread.native_id
        if tid is not None and thread.is_alive():
            start_time = task_start_time(tid)
            if start_time is not None:
                out[thread] = start_time
    return out


@contextlib.contextmanager
def os_joined(executor: _E) -> Generator[_E]:
    """``with os_joined(ThreadPoolExecutor(...)) as pool:`` -- the executor's
    ``shutdown(wait=True)`` (what its own ``with`` does), then the OS-thread
    wait for its workers, on the normal and the exception path alike. Only
    workers alive at shutdown are waited for, each checked against its task
    start time read before the shutdown, so a reused tid costs nothing, and all
    of them share one :data:`OS_EXIT_WAIT_S` cap."""
    try:
        yield executor
    finally:
        alive = _start_times(list(getattr(executor, "_threads", ())))
        executor.shutdown(wait=True)
        # One cap for the whole pool, not one per worker: a teardown that is
        # really stuck costs at most OS_EXIT_WAIT_S per pool shutdown.
        # Running out of it is a cap hit (the module cap set the bound), logged
        # once per shutdown at a bounded rate (``POOL_WARN_INTERVAL_S``).
        deadline = time.monotonic() + OS_EXIT_WAIT_S
        stuck = [
            thread for thread, start_time in alive.items()
            if not _await(thread, deadline - time.monotonic(), start_time,
                          cap_hit_if_expired=True, log=False)
        ]
        if stuck:
            _warn_pool_cap_hit(stuck)
