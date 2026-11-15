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
import os
import threading
import time
import weakref
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

_log = logging.getLogger(__name__)

# Cap on waiting for an ended thread's OS thread to exit. The measured linger
# is milliseconds; a reused tid at worst costs the cap.
OS_EXIT_WAIT_S = 1.0
_POLL_S = 0.0005

_warned_lock = threading.Lock()
_warned: weakref.WeakSet[threading.Thread] = weakref.WeakSet()

_E = TypeVar("_E", bound=ThreadPoolExecutor)


def _warn_cap_once(thread: threading.Thread, cap_s: float) -> None:
    with _warned_lock:
        if thread in _warned:
            return
        _warned.add(thread)
    _log.warning(
        "thread %r (native id %s) ended but its OS thread was still running %.3g s "
        "after join; carrying on without waiting for it",
        thread.name, thread.native_id, cap_s,
    )


def await_os_thread_exit(thread: threading.Thread, *, cap_s: float | None = None) -> bool:
    """After ``thread`` ended, wait (bounded) until its OS thread is gone.

    Returns True when there is nothing left to wait for (the OS thread is gone,
    the thread never started, it is still running -- the caller's join timed
    out, which is the caller's business -- or not Linux). Returns False when
    the cap expired, after logging that once per thread."""
    tid = getattr(thread, "native_id", None)
    if tid is None or thread.is_alive():
        return True
    task = f"/proc/self/task/{tid}"
    cap = OS_EXIT_WAIT_S if cap_s is None else max(0.0, cap_s)
    stop = time.monotonic() + cap
    while os.path.exists(task):
        if time.monotonic() >= stop:
            _warn_cap_once(thread, cap)
            return False
        time.sleep(_POLL_S)
    return True


def join_thread(thread: threading.Thread, timeout: float | None = None) -> None:
    """``thread.join(timeout)``, then :func:`await_os_thread_exit` if it ended.

    Same contract as ``Thread.join``: returns after ``timeout`` with the thread
    still alive if it did not end (check ``is_alive()``); never raises for the
    OS-thread wait."""
    thread.join(timeout)
    await_os_thread_exit(thread)


def await_executor_os_exit(executor: ThreadPoolExecutor) -> None:
    """:func:`await_os_thread_exit` for every worker thread ``executor`` started
    (call after its ``shutdown(wait=True)``)."""
    for thread in list(getattr(executor, "_threads", ())):
        await_os_thread_exit(thread)


@contextlib.contextmanager
def os_joined(executor: _E) -> Generator[_E]:
    """``with os_joined(ThreadPoolExecutor(...)) as pool:`` -- the executor's
    own ``with`` (``shutdown(wait=True)``), then the OS-thread wait for its
    workers, on the normal and the exception path alike."""
    try:
        with executor:
            yield executor
    finally:
        await_executor_os_exit(executor)
