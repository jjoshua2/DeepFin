"""Startup progress heartbeats and the revocable compile turn.

A device whose evaluator build or warm-up hangs (a wedged driver call, a
kernel that never finishes, an autotune subprocess that never returns) used to
hang the engine before ``readyok`` forever. The fix needs two things this
module provides:

- **Heartbeats.** Every completed warm-up step (one batch shape run on one
  device, a factory finishing its build) calls :func:`beat`. The startup
  timeout is "no progress for N seconds", not "N seconds in total": a cold
  ``max-autotune`` compile of every bucket on four GPUs legitimately takes many
  minutes in total, but each single step completes. A thread-local
  :class:`StartupContext` (installed by whoever watches that thread, e.g. a
  multi-GPU pool worker) records the thread's own last beat; :func:`last_beat`
  is the process-wide latest beat, read by the builder-level watchdog.
- **A revocable compile turn.** Multi-GPU factories compile and warm one at a
  time (concurrent torch.compile / CUDA-graph capture on two threads is not
  safe). A device that hangs while holding the turn would otherwise starve
  every device queued behind it, so a hung device would cost them all. The
  watcher that times a holder out calls :meth:`StartupContext.revoke_turns`,
  which hands the turn to the next waiter. A thread WAITING for the turn is not
  "making no progress": its clock is paused (``waiting``) until it holds it.
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Generator

# Default for ``--device-startup-timeout``: seconds one device's evaluator
# build / warm-up may go WITHOUT PROGRESS (no completed warm-up shape) before
# the device is dropped (multi-GPU) or startup fails (no device left). Sized
# far above one cold step: the comment on ``_MULTI_GPU_NO_CUDAGRAPH_MODE``
# records a cold max-autotune compile "can exceed 10 min on mid-range GPUs",
# and the first bucket of a device pays that whole compile. 30 minutes keeps a
# healthy cold compile from ever being dropped; a genuinely hung device costs
# at most that long once, at startup.
DEFAULT_DEVICE_STARTUP_TIMEOUT_S = 1800.0
# The builder-level watchdog (the whole startup, primary evaluator included)
# allows this much beyond the per-device timeout, so a multi-GPU pool gets to
# drop a hung device (and continue on the rest) before the builder gives up.
BUILDER_TIMEOUT_GRACE_S = 60.0

_local = threading.local()
_global_lock = threading.Lock()
_last_beat = [time.monotonic()]


class StartupContext:
    """Progress state of ONE thread building / warming a device's evaluator."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._lock = threading.Lock()
        self.last_beat = time.monotonic()
        self.waiting = False
        self.revoked = False
        self.finished = False  # its build/warm-up block has exited
        self._turns: list[StartupTurn] = []

    def beat(self) -> None:
        with self._lock:
            self.last_beat = time.monotonic()

    def idle_for(self, now: float | None = None) -> float:
        """Seconds since this thread last made progress; 0 while it waits for
        a compile turn another device holds."""
        with self._lock:
            if self.waiting:
                return 0.0
            return (time.monotonic() if now is None else now) - self.last_beat

    def revoke_turns(self) -> None:
        """This thread was timed out: give any turn it holds to the next waiter.

        Its own later ``release`` is then a no-op, so a hung thread that wakes
        up cannot release a turn another device now holds."""
        with self._lock:
            self.revoked = True
            turns, self._turns = self._turns, []
        for turn in turns:
            turn.revoke(self)


def current_context() -> StartupContext | None:
    return getattr(_local, "ctx", None)


@contextlib.contextmanager
def startup_context(ctx: StartupContext) -> Generator[StartupContext, None, None]:
    """Install ``ctx`` as this thread's startup context for the block."""
    prev = current_context()
    _local.ctx = ctx
    try:
        yield ctx
    finally:
        _local.ctx = prev
        with ctx._lock:
            ctx.finished = True


def beat() -> None:
    """Record startup progress for this thread (if watched) and the process."""
    now = time.monotonic()
    with _global_lock:
        _last_beat[0] = now
    ctx = current_context()
    if ctx is not None:
        ctx.beat()


def last_beat() -> float:
    with _global_lock:
        return _last_beat[0]


class StartupTurn:
    """A lock whose holder can be revoked by a startup watchdog.

    Used exactly like ``threading.Lock`` (``with turn:``). With no
    :class:`StartupContext` on the calling thread it IS a plain mutex. With
    one, waiting pauses that thread's no-progress clock, acquiring counts as
    progress, and a revoked holder's turn passes to the next waiter.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._holder: object | None = None
        self._revoked: list[StartupContext] = []

    def __enter__(self) -> StartupTurn:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def acquire(self) -> None:
        ctx = current_context()
        token: object = ctx if ctx is not None else threading.current_thread()
        if ctx is not None:
            with ctx._lock:
                ctx.waiting = True
        try:
            with self._cv:
                while self._holder is not None:
                    self._cv.wait()
                self._holder = token
        finally:
            if ctx is not None:
                with ctx._lock:
                    ctx.waiting = False
                    ctx.last_beat = time.monotonic()
                    revoked = ctx.revoked
                    if not revoked:
                        ctx._turns.append(self)
                if revoked:
                    # Timed out while queued: hand the turn straight on.
                    self.revoke(ctx)

    def release(self) -> None:
        ctx = current_context()
        token: object = ctx if ctx is not None else threading.current_thread()
        if ctx is not None:
            with ctx._lock:
                if self in ctx._turns:
                    ctx._turns.remove(self)
        with self._cv:
            if self._holder is token:
                self._holder = None
                self._cv.notify_all()

    def revoke(self, token: object) -> None:
        with self._cv:
            if self._holder is token:
                self._holder = None
                if isinstance(token, StartupContext):
                    self._revoked.append(token)
                self._cv.notify_all()

    def revoked_holder_running(self) -> bool:
        """A holder timed out and revoked while holding the turn is still
        inside its build (it may wake up and keep compiling at any time).

        The turn exists so two threads never compile / capture CUDA graphs at
        once; after a revocation that can no longer be promised for anything
        that compiles, so callers then build without compiling (eager)."""
        with self._cv:
            revoked = list(self._revoked)
        for ctx in revoked:
            with ctx._lock:
                if not ctx.finished:
                    return True
        return False

    def locked(self) -> bool:
        """Whether some thread holds the turn (``threading.Lock.locked``)."""
        with self._cv:
            return self._holder is not None

    def held_by(self) -> object | None:
        with self._cv:
            return self._holder
