"""Bounded cooperative producer execution shared by selfplay and teacher actors.

The controller owns admission and resource lifetime. A failed producer stops its
peers immediately; a finite join failure never pretends to kill a Python thread.
"""
from __future__ import annotations

from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
import math
import threading
import time
from typing import Any, TypeVar


_Result = TypeVar("_Result")
# Deliberately process-lifetime custody: an outer hard reaper, not Python GC,
# resolves a stuck native call. Never let exception collection release handles.
_RETAINED_OWNERSHIP: list[ProducerJoinTimeout] = []


class ProducerJoinTimeout(TimeoutError):
    """Non-quiescent ownership; callers must retain, not close, all dependencies."""

    def __init__(self, message: str, *owned: object) -> None:
        super().__init__(message)
        self.owned = list(owned)
        _RETAINED_OWNERSHIP.append(self)

    def retain(self, *owned: object) -> None:
        self.owned.extend(owned)


class ProducerExecutor:
    """Persistent bounded executor with explicit cooperative stop and join.

    ``capacity`` counts running plus queued jobs. It does not grow with elapsed
    runtime. Futures are non-cancellable once running, just like executor jobs;
    callers retain exclusive actor ownership until their future is done.
    """

    def __init__(
        self, *, max_workers: int, capacity: int,
        stop_peers: Callable[[], None] = lambda: None,
        on_failure: Callable[[BaseException], None] = lambda _exc: None,
    ) -> None:
        if (type(max_workers) is not int or type(capacity) is not int
                or not 1 <= max_workers <= capacity):
            raise ValueError("positive bounded producer workers/capacity required")
        self.stop_event = threading.Event()
        self._stop_peers = stop_peers
        self._on_failure = on_failure
        self._stop_lock = threading.Lock()
        self._lock = threading.Lock()
        self._capacity = capacity
        self._futures: set[Future[Any]] = set()
        self._failure: BaseException | None = None
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="GameProducer")

    @property
    def capacity(self) -> int:
        """Maximum admitted CPU jobs, independent of logical lane count."""
        return self._capacity

    @contextmanager
    def admission(self) -> Generator[bool]:
        """Admit one operation unless stopped; admitted work retains ownership."""
        with self._stop_lock:
            admitted = not self.stop_event.is_set()
        # The gate protects admission only, never execution of caller work.
        # STOP cannot cancel work already admitted, and must not wait for it.
        yield admitted

    def request_stop(self) -> None:
        # Never acquire the operation gate here: an already-admitted native
        # call may be stuck. Signalling STOP must not consume the join budget.
        with self._stop_lock:
            first = not self.stop_event.is_set()
            self.stop_event.set()
        if first:
            self._stop_peers()

    def raise_if_failed(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise failure

    def submit(self, operation: Callable[..., _Result], *args: Any, **kwargs: Any) -> Future[_Result]:
        def run() -> _Result:
            try:
                return operation(*args, **kwargs)
            except BaseException as exc:
                with self._lock:
                    if self._failure is None:
                        self._failure = exc
                self.request_stop()
                self._on_failure(exc)
                raise

        with self._stop_lock, self._lock:
            if self._failure is not None:
                raise self._failure
            if self._closed or self.stop_event.is_set():
                raise RuntimeError("producer admission stopped")
            self._futures = {future for future in self._futures if not future.done()}
            if len(self._futures) >= self._capacity:
                raise BufferError("producer task backpressure")
            future = self._executor.submit(run)
            self._futures.add(future)
            return future

    def finish(self, *, timeout: float | None) -> None:
        """Close admission and join all admitted work, optionally to one deadline.

        None preserves the existing selfplay session's cooperative lifetime.
        Finite callers get a typed custody exception if a producer stays alive.
        No unbounded executor context-manager shutdown is used on that path.
        """
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("finite nonnegative producer shutdown timeout required")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            self._closed = True
            futures = tuple(self._futures)
        if self.stop_event.is_set():
            for future in futures:
                future.cancel()  # only a job that never started can be cancelled
        # A cancelled queued Future has no actor ownership even before an
        # executor worker marks its cancellation notified for wait().
        active = tuple(future for future in futures if not future.done())
        _done, pending = wait(active, timeout=None if deadline is None else max(0, deadline - time.monotonic()))
        if pending:
            self.request_stop()
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise ProducerJoinTimeout("producer join budget expired; active ownership retained", self, futures)
        # All submitted bodies exited; this cannot wait for user/native work.
        self._executor.shutdown(wait=timeout is None)
        self.raise_if_failed()
