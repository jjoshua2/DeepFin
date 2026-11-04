"""FIFO row routing shared by dispatchers with different execution backends.

The caller owns synchronization, admission capacity, execution and completion.
This queue owns only pending payloads and their partial-request row cursors.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

Payload = TypeVar("Payload")


@dataclass(frozen=True, slots=True)
class RequestSpan(Generic[Payload]):
    payload: Payload
    request_row_start: int
    batch_row_start: int
    row_count: int


@dataclass(frozen=True, slots=True)
class RoutedBatch(Generic[Payload]):
    spans: tuple[RequestSpan[Payload], ...]
    row_count: int


@dataclass(frozen=True, slots=True)
class ControlRoute(Generic[Payload]):
    payload: Payload


@dataclass(slots=True)
class _QueuedRequest(Generic[Payload]):
    payload: Payload
    row_count: int
    group_key: object
    offset: int = 0


class FifoRequestQueue(Generic[Payload]):
    """Route contiguous compatible requests without owning their completion.

    All operations require the caller's lock when used across threads. An empty
    request is consumed once as a zero-row span; it cannot stall the queue.
    Control entries are barriers and are returned separately from row batches.
    """

    def __init__(self) -> None:
        self._items: deque[_QueuedRequest[Payload] | ControlRoute[Payload]] = deque()
        self._queued_rows = 0
        self._queued_requests = 0

    def __len__(self) -> int:
        """Number of queued entries, including controls and partial requests."""
        return len(self._items)

    @property
    def queued_rows(self) -> int:
        return self._queued_rows

    @property
    def queued_request_count(self) -> int:
        return self._queued_requests

    @property
    def has_items(self) -> bool:
        return bool(self._items)

    def enqueue_request(
        self, payload: Payload, row_count: int, group_key: object = None,
    ) -> None:
        if type(row_count) is not int or row_count < 0:
            raise ValueError("request row count must be a nonnegative integer")
        self._items.append(_QueuedRequest(payload, row_count, group_key))
        self._queued_rows += row_count
        self._queued_requests += 1

    def enqueue_control(self, payload: Payload) -> None:
        self._items.append(ControlRoute(payload))

    def take_batch(
        self, target_rows: int, max_batch_rows: int, split_requests: bool,
    ) -> RoutedBatch[Payload] | ControlRoute[Payload] | None:
        """Plan then dequeue one batch, or the next control barrier.

        Split mode fills to the target. Atomic mode keeps each request whole,
        allowing the first request to exceed the target but not the hard maximum.
        Invalid geometry or an oversized atomic request encountered during
        planning leaves the entire queue unchanged.
        """
        if (type(target_rows) is not int or type(max_batch_rows) is not int
                or not 1 <= target_rows <= max_batch_rows):
            raise ValueError("positive bounded batch geometry required")
        if not self._items:
            return None
        first = self._items[0]
        if isinstance(first, ControlRoute):
            self._items.popleft()
            return first

        spans: list[RequestSpan[Payload]] = []
        total = 0
        for entry in self._items:
            if isinstance(entry, ControlRoute) or entry.group_key != first.group_key:
                break
            remaining = entry.row_count - entry.offset
            if split_requests:
                if spans and total >= target_rows:
                    break
                count = min(remaining, target_rows - total)
            else:
                if remaining > max_batch_rows:
                    raise ValueError(
                        f"queued request has {remaining} rows > max_batch_rows={max_batch_rows}"
                    )
                if spans and total + remaining > target_rows:
                    break
                count = remaining
            spans.append(RequestSpan(entry.payload, entry.offset, total, count))
            total += count
            if count < remaining:
                break

        # Commit only after planning succeeds. Each span consumes exactly the
        # current head, except a final split span may retain that head's cursor.
        for span in spans:
            entry = self._items[0]
            assert isinstance(entry, _QueuedRequest)
            entry.offset += span.row_count
            self._queued_rows -= span.row_count
            if entry.offset == entry.row_count:
                self._items.popleft()
                self._queued_requests -= 1
        return RoutedBatch(tuple(spans), total)

    def drain_queued(self) -> tuple[Payload, ...]:
        """Remove pending entries once each; already-routed ownership is external."""
        payloads = tuple(entry.payload for entry in self._items)
        self._items.clear()
        self._queued_rows = self._queued_requests = 0
        return payloads
