"""Bounded FIFO teacher requests, independent of feed type and head count.

The condition/deadline/gather/scatter structure follows ThreadedDispatcher.
Unlike its Torch buffers, each backend owns packing and output validation. One
thread calls the backend; no CUDA overlap or performance claim is implied.
"""
from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")


@dataclass
class _Request(Generic[Input, Output]):
    key: str
    rows: tuple[Input, ...]
    future: Future[tuple[Output, ...]]
    outputs: list[Output] = field(default_factory=list)
    offset: int = 0
    submitted_at: float = field(default_factory=time.monotonic)


class TeacherDispatcher(Generic[Input, Output]):
    """Async submit/result interface with explicit row capacity and fill deadline.

    Capacity counts queued AND running rows. Large requests may span calls;
    their Future publishes only complete ordered output. A fill deadline always
    drains a low-volume tail; flush makes that deadline immediate. Request keys
    are unique while owned; durable consumers own longer-lived deduplication.
    """

    def __init__(
        self, backend: Callable[[Sequence[Input]], Sequence[Output]], *,
        target_rows: int, max_rows: int, batch_wait_ms: float,
    ) -> None:
        if (type(target_rows) is not int or type(max_rows) is not int
                or not 1 <= target_rows <= max_rows
                or not math.isfinite(batch_wait_ms) or batch_wait_ms < 0):
            raise ValueError("explicit bounded teacher geometry and wait required")
        self.backend = backend
        self.target_rows = target_rows
        self.max_rows = max_rows
        self.wait_s = batch_wait_ms / 1000
        self.histogram: Counter[int] = Counter()
        self.queue_wait_seconds: deque[float] = deque(maxlen=128)
        self._queue: deque[_Request[Input, Output]] = deque()
        self._owned: dict[str, _Request[Input, Output]] = {}
        self._rows = 0
        self._closing = False
        self._flush = False
        self._fatal: BaseException | None = None
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="TeacherDispatcher", daemon=True)
        self._thread.start()

    def submit(self, key: str, rows: Sequence[Input]) -> Future[tuple[Output, ...]]:
        rows = tuple(rows)
        if len(rows) > self.max_rows:
            raise ValueError("single teacher request exceeds row capacity")
        with self._cond:
            if self._fatal is not None:
                raise RuntimeError("teacher dispatcher failed") from self._fatal
            if self._closing:
                raise RuntimeError("teacher dispatcher closed")
            if key in self._owned:
                raise ValueError("duplicate owned teacher request")
            if self._rows + len(rows) > self.max_rows:
                raise BufferError("teacher row backpressure")
            future: Future[tuple[Output, ...]] = Future()
            # Consumers cannot cancel an admitted request halfway through routing.
            future.set_running_or_notify_cancel()
            if not rows:
                future.set_result(())
                return future
            request = _Request(key, rows, future)
            self._owned[key] = request
            self._queue.append(request)
            self._rows += len(rows)
            self._cond.notify_all()
            return future

    def flush(self) -> None:
        """Bypass fill waits until all admitted queued rows have been drained."""
        with self._cond:
            self._flush = True
            self._cond.notify_all()

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

    def _run(self) -> None:
        try:
            while True:
                with self._cond:
                    while not self._queue and not self._closing:
                        self._cond.wait()
                    if not self._queue:
                        return
                    deadline = time.monotonic() + self.wait_s
                    while (sum(len(r.rows) - r.offset for r in self._queue) < self.target_rows
                           and not self._flush and not self._closing):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._cond.wait(remaining)
                    chunks: list[tuple[_Request[Input, Output], int]] = []
                    batch: list[Input] = []
                    while self._queue and len(batch) < self.target_rows:
                        request = self._queue[0]
                        if request.offset == 0:
                            self.queue_wait_seconds.append(time.monotonic() - request.submitted_at)
                        count = min(self.target_rows - len(batch), len(request.rows) - request.offset)
                        batch.extend(request.rows[request.offset:request.offset + count])
                        request.offset += count
                        chunks.append((request, count))
                        if request.offset == len(request.rows):
                            self._queue.popleft()
                    if not self._queue:
                        self._flush = False
                outputs = tuple(self.backend(batch))
                if len(outputs) != len(batch):
                    raise ValueError("teacher output count differs from dispatched rows")
                self.histogram[len(batch)] += 1
                offset = 0
                completed: list[_Request[Input, Output]] = []
                with self._cond:
                    for request, count in chunks:
                        request.outputs.extend(outputs[offset:offset + count])
                        offset += count
                        if len(request.outputs) == len(request.rows):
                            del self._owned[request.key]
                            self._rows -= len(request.rows)
                            completed.append(request)
                for request in completed:
                    request.future.set_result(tuple(request.outputs))
        except BaseException as exc:
            with self._cond:
                self._fatal = exc
                stranded = tuple(self._owned.values())
                self._owned.clear()
                self._queue.clear()
                self._rows = 0
                self._cond.notify_all()
            for request in stranded:
                request.future.set_exception(exc)
