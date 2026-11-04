"""Shared execution lifecycle for independently owned generation domains.

A driver advances chess/label policy, never the runner. The runner owns control
polling, ready continuation and completion wakeups. Backend, writer and producer
executors are deliberately independent: a producer waiting on inference cannot
occupy the capacity required to complete that inference.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable
from concurrent.futures import Future
from dataclasses import dataclass
import math
import threading
import time
from typing import Any


@dataclass(frozen=True)
class AdvanceResult:
    progress: bool = False
    runnable: bool = False
    terminal: bool = False
    waits: tuple[Future[Any], ...] = ()
    wait_seconds: float | None = None
    wait: Callable[[float], None] | None = None


class GenerationSessionRunner:
    """Run one exclusively owned domain on its caller/owner thread.

    A driver may submit disjoint lanes to bounded producers. Its ``advance``
    must not mutate a leased lane before the corresponding Future settles.
    STOP, pause, publication acceptance and teardown policy remain explicit in
    the domain; no generic cancellation is mistaken for domain quiescence.
    """

    def __init__(self, *, idle_wait: float = 0.001, deadline: float | None = None,
                 control: Callable[[], None] = lambda: None) -> None:
        if not math.isfinite(idle_wait) or idle_wait <= 0:
            raise ValueError("positive finite generation wait required")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("finite generation deadline required")
        self.idle_wait = idle_wait
        self.deadline = deadline
        self.control = control
        self._wake = threading.Event()
        self._watched: set[Future[Any]] = set()
        self.advances = 0
        self.wait_seconds = 0.0
        self.progress_ticks = 0

    def wake(self) -> None:
        self._wake.set()

    def run(self, advance: Callable[[], AdvanceResult]) -> AdvanceResult:
        while True:
            # Clear before observing completion, never after registering a
            # callback: a fast result between observation and wait is not lost.
            self._wake.clear()
            self.control()
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise TimeoutError("finite generation session budget expired; ownership retained")
            result = advance()
            self.advances += 1
            self.progress_ticks += bool(result.progress)
            if result.terminal:
                return result
            if result.progress or result.runnable:
                continue
            delay = self.idle_wait if result.wait_seconds is None else result.wait_seconds
            if not math.isfinite(delay) or delay < 0:
                raise ValueError("finite nonnegative domain wait required")
            if self.deadline is not None:
                delay = min(delay, max(0.0, self.deadline - time.monotonic()))
            if result.wait is not None:
                started = time.monotonic()
                result.wait(delay)
                self.wait_seconds += time.monotonic() - started
                continue
            self._watched.intersection_update(future for future in self._watched if not future.done())
            for future in result.waits:
                if future.done():
                    self._wake.set()
                elif future not in self._watched:
                    self._watched.add(future)
                    future.add_done_callback(lambda _future: self.wake())
            started = time.monotonic()
            self._wake.wait(delay)
            self.wait_seconds += time.monotonic() - started


class ReservationCredits:
    """Controller-owned retained-memory reservations, released on acceptance.

    Reserve a conservative whole trajectory before admitting its game. Appends
    consume that reservation rather than requesting credits held by another
    unfinished game, so a full reservoir cannot deadlock trajectory completion.
    CPU jobs, inference rows and publication have separate bounded executors.
    """

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("positive retained byte capacity required")
        self.capacity = capacity
        self.used = 0
        self.peak = 0
        self._owned: dict[Hashable, int] = {}

    def reserve(self, key: Hashable, amount: int) -> bool:
        if key in self._owned or type(amount) is not int or amount < 1:
            raise ValueError("distinct positive memory reservation required")
        if amount > self.capacity:
            raise ValueError("one game exceeds retained trajectory byte budget")
        if self.used + amount > self.capacity:
            return False
        self._owned[key] = amount
        self.used += amount
        self.peak = max(self.peak, self.used)
        return True

    def release(self, key: Hashable) -> None:
        self.used -= self._owned.pop(key)


@dataclass(frozen=True)
class LaneLease:
    identity: Hashable
    generation: int


class ExclusiveLaneRegistry:
    """Controller-owned identity/generation leases for batched producer work."""

    def __init__(self) -> None:
        self._owned: dict[Hashable, int] = {}
        self._next: dict[Hashable, int] = {}

    def busy(self, identity: Hashable) -> bool:
        return identity in self._owned

    def admit(self, identities: tuple[Hashable, ...]) -> tuple[LaneLease, ...]:
        if len(set(identities)) != len(identities) or any(self.busy(key) for key in identities):
            raise RuntimeError("generation lane already exclusively owned")
        leases = tuple(LaneLease(key, self._next.get(key, 0) + 1) for key in identities)
        for lease in leases:
            self._owned[lease.identity] = self._next[lease.identity] = lease.generation
        return leases

    def complete(self, leases: tuple[LaneLease, ...]) -> None:
        if any(self._owned.get(lease.identity) != lease.generation for lease in leases):
            raise RuntimeError("stale or misrouted generation completion")
        for lease in leases:
            del self._owned[lease.identity]

    def forget(self, identity: Hashable) -> None:
        if self.busy(identity):
            raise RuntimeError("cannot release an owned generation lane")
        self._next.pop(identity, None)
