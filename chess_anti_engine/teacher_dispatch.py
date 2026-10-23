"""Bounded FIFO teacher requests, independent of feed type and head count.

The condition/deadline/gather/scatter structure follows ThreadedDispatcher.
Unlike Torch, ORT prepares the next batch off-thread, then rendezvous with the
dispatcher for execution and accepted finish. No CUDA/DMA overlap is claimed.
"""
from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from chess_anti_engine.async_safety import Readiness, completion_readiness
from chess_anti_engine.staged_execution import HostReceipt, StagedHostBackend, TeacherBackend, TwoSlotPump

from chess_anti_engine.request_routing import FifoRequestQueue, RoutedBatch

Input = TypeVar("Input")
Output = TypeVar("Output")


@dataclass
class _Request(Generic[Input, Output]):
    key: str
    rows: tuple[Input, ...]
    future: Future[tuple[Output, ...]]
    outputs: list[Output] = field(default_factory=list)
    submitted_at: float = field(default_factory=time.monotonic)


class TeacherDispatcher(Generic[Input, Output]):
    """Async submit/result interface with explicit row capacity and fill deadline.

    Capacity counts queued AND running rows. Large requests may span calls;
    their Future publishes only complete ordered output. A fill deadline always
    drains a low-volume tail; flush makes that deadline immediate. Request keys
    are unique while owned; durable consumers own longer-lived deduplication.
    """

    def __init__(
        self, backend: TeacherBackend[Input, Output], *,
        target_rows: int, max_rows: int, batch_wait_ms: float,
        control: Callable[[], None] = lambda: None,
    ) -> None:
        self.validate_geometry(target_rows, max_rows, batch_wait_ms)
        self.backend = backend
        # This callback runs on TeacherDispatcher, never a prepare executor.
        # An outer controller-affine guard must explicitly marshal to that
        # controller; accepting a callback does not make it thread-safe.
        self._control = control
        self.target_rows = target_rows
        self.max_rows = max_rows
        self.wait_s = batch_wait_ms / 1000
        self.histogram: Counter[int] = Counter()
        self.queue_wait_seconds: deque[float] = deque(maxlen=128)
        self._queue: FifoRequestQueue[_Request[Input, Output]] = FifoRequestQueue()
        self._owned: dict[str, _Request[Input, Output]] = {}
        self._rows = 0
        self._closing = False
        self._flush = False
        self._fatal: BaseException | None = None
        self._cond = threading.Condition()
        # Publish lifetime state before starting the owner; actual executors and
        # pump construction remain exclusively on that owner thread.
        self._host: StagedHostBackend[Input, Output] | None = None
        self._pump: TwoSlotPump | None = None
        self._thread = threading.Thread(target=self._run, name="TeacherDispatcher", daemon=True)
        self._thread.start()

    @staticmethod
    def validate_geometry(target_rows: int, max_rows: int, batch_wait_ms: float) -> None:
        """The same no-thread preflight for callers that own model startup."""
        if (type(target_rows) is not int or type(max_rows) is not int
                or not 1 <= target_rows <= max_rows
                or not math.isfinite(batch_wait_ms) or batch_wait_ms < 0):
            raise ValueError("explicit bounded teacher geometry and wait required")

    def submit(self, key: str, rows: Sequence[Input]) -> Future[tuple[Output, ...]]:
        admitted = self.submit_many(((key, rows),))
        if not admitted:
            raise BufferError("teacher row backpressure")
        return admitted[0]

    def submit_many(
        self, requests: Sequence[tuple[str, Sequence[Input]]],
    ) -> tuple[Future[tuple[Output, ...]], ...]:
        """Atomically admit a bounded FIFO prefix, with one Future per request.

        Validate the entire proposed list before mutating admission. Capacity
        truncates only the tail; it never skips a large head to take later rows.
        """
        candidates = tuple((key, tuple(rows)) for key, rows in requests)
        if any(len(rows) > self.max_rows for _, rows in candidates):
            raise ValueError("single teacher request exceeds row capacity")
        keys = [key for key, _ in candidates]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate owned teacher request")
        with self._cond:
            if self._fatal is not None:
                raise RuntimeError("teacher dispatcher failed") from self._fatal
            if self._closing:
                raise RuntimeError("teacher dispatcher closed")
            if any(key in self._owned for key in keys):
                raise ValueError("duplicate owned teacher request")
            admitted = []
            for key, rows in candidates:
                if self._rows + len(rows) > self.max_rows:
                    break
                future: Future[tuple[Output, ...]] = Future()
                # Admitted rows cannot be cancelled halfway through a route.
                future.set_running_or_notify_cancel()
                if not rows:
                    future.set_result(())
                else:
                    request = _Request(key, rows, future)
                    self._owned[key] = request
                    self._queue.enqueue_request(request, len(rows))
                    self._rows += len(rows)
                admitted.append(future)
            self._cond.notify_all()
            return tuple(admitted)

    def flush(self) -> None:
        """Bypass fill waits until all admitted queued rows have been drained."""
        with self._cond:
            self._flush = True
            self._cond.notify_all()

    @property
    def stage_stats(self) -> dict[str, float | int]:
        host = self._host
        return {} if host is None else host.stats

    @property
    def is_alive(self) -> bool:
        """Whether background work still owns its backend/resource state."""
        return self._thread.is_alive()

    def close(self, *, timeout: float) -> None:
        """Drain owned work and join within caller's finite shutdown budget."""
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("finite nonnegative shutdown timeout required")
        with self._cond:
            self._closing = self._flush = True
            self._cond.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("teacher backend still running; ownership retained")

    def _take_batch(self, *, blocking: bool, coalesce_wait: bool) -> Any:
        assert self._pump is not None
        with self._cond:
            if blocking:
                while not self._queue.has_items and not self._closing:
                    self._cond.wait()
            if not self._queue.has_items:
                return None
            deadline = time.monotonic() + self.wait_s
            # Only a definitely unfinished receipt allows a fill wait. Unknown
            # readiness and ready output require prompt retirement/publication.
            while (self._queue.queued_rows < self.target_rows
                   and (blocking or coalesce_wait)
                   and not self._flush and not self._closing):
                if (not blocking and self._pump.pending is not None
                        and completion_readiness(self._pump.pending[1]) is not Readiness.NOT_READY):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            route = self._queue.take_batch(
                self.target_rows, self.target_rows, split_requests=True,
            )
            if not self._queue.has_items:
                self._flush = False
            return route

    def _submit_batch(self, route: RoutedBatch, _slot: int) -> tuple[RoutedBatch, HostReceipt]:
        assert self._host is not None
        batch: list[Input] = []
        for span in route.spans:
            request = span.payload
            if span.request_row_start == 0:
                self.queue_wait_seconds.append(time.monotonic() - request.submitted_at)
            start = span.request_row_start
            batch.extend(request.rows[start:start + span.row_count])
        receipt = self._host.submit(tuple(batch))
        def wake(_future: Future) -> None:
            # Completion, not only a new request, interrupts a fill wait.
            with self._cond:
                self._cond.notify_all()
        receipt.future.add_done_callback(wake)
        return route, receipt

    def _retire(self, handle: tuple[RoutedBatch, HostReceipt]) -> None:
        assert self._host is not None
        route, receipt = handle
        outputs = tuple(self._host.release(receipt))
        if len(outputs) != route.row_count:
            raise ValueError("teacher output count differs from dispatched rows")
        self.histogram[route.row_count] += 1
        completed = []
        with self._cond:
            for span in route.spans:
                request = span.payload
                start = span.batch_row_start
                request.outputs.extend(outputs[start:start + span.row_count])
                if len(request.outputs) == len(request.rows):
                    del self._owned[request.key]
                    self._rows -= len(request.rows)
                    completed.append(request)
            self._cond.notify_all()
        for request in completed:
            request.future.set_result(tuple(request.outputs))

    def _run(self) -> None:
        self._host = StagedHostBackend(self.backend, control=self._control)
        def failed(_value: Any, error: BaseException, _stage: str) -> None:
            # Teachers retain poison/fail-all, unlike selfplay batch settlement.
            raise error
        self._pump = TwoSlotPump(
            take=self._take_batch, submit=self._submit_batch, retire=self._retire,
            readiness=lambda handle: completion_readiness(handle[1]), failed=failed,
        )
        try:
            self._pump.run()
        except BaseException as exc:
            with self._cond:
                self._fatal = exc
                stranded = tuple(self._owned.values())
                self._owned.clear()
                self._queue.drain_queued()
                self._rows = 0
                self._cond.notify_all()
            for request in stranded:
                if not request.future.done():
                    request.future.set_exception(exc)
        finally:
            # Can block while a session is still running. Public close(timeout)
            # retains this thread and every dependency rather than freeing slots.
            self._host.close()
