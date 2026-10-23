"""Bounded supervision of fixed logical domains on threads or processes.

Selfplay domains retain an entire session RNG; corpus domains retain a logical
worker's engine, cache and shard stream. Physical concurrency cannot repartition
those domains. Completion runs on the caller, independently of occupied producer
threads, and exceptions either propagate or become explicit domain receipts.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing.context import BaseContext
from typing import Any, Generic, TypeVar, cast

from chess_anti_engine.generation_runtime import AdvanceResult, GenerationSessionRunner


_Domain = TypeVar("_Domain")
_Result = TypeVar("_Result")


class FixedDomainSupervisor(Generic[_Domain, _Result]):
    """Submit and settle a fixed roster without moving work between domains.

    ``capacity`` bounds admitted, not-yet-settled tasks. Results are returned
    in roster order. ``ordered`` preserves selfplay's original result/error
    collection order; unordered settlement lets SF record a dead worker and
    continue independent workers. ``on_failure`` is never called on a worker
    thread and must return an explicit failure receipt. Without it, exceptions
    propagate to the caller's cooperative peer-stop/resource cleanup policy.
    """

    def __init__(
        self, domains: Sequence[_Domain], *,
        submit: Callable[[_Domain], Future[_Result]], capacity: int,
        ordered: bool = True,
        on_failure: Callable[[_Domain, Exception], _Result] | None = None,
    ) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("positive fixed-domain capacity required")
        self.domains = tuple(domains)
        self.submit = submit
        self.capacity = capacity
        self.ordered = ordered
        self.on_failure = on_failure
        self._next = 0
        self._active: dict[int, Future[_Result]] = {}
        self._results: dict[int, _Result] = {}
        self.failed = 0
        self.peak_admitted = 0
        self.runner = GenerationSessionRunner(idle_wait=0.25)

    @property
    def status(self) -> dict[str, int]:
        return {
            "logical_domains": len(self.domains),
            "not_admitted": len(self.domains) - self._next,
            "admitted": len(self._active),
            "settled": len(self._results),
            "failed": self.failed,
            "peak_admitted": self.peak_admitted,
        }

    def _failure(self, index: int, exc: Exception) -> _Result:
        self.failed += 1
        if self.on_failure is None:
            raise exc
        return self.on_failure(self.domains[index], exc)

    def advance(self) -> AdvanceResult:
        progress = False
        # Admit the whole bounded prefix before collecting even an immediate
        # result: selfplay must start every independent session before joining.
        while self._next < len(self.domains) and len(self._active) < self.capacity:
            index = self._next
            self._next += 1
            try:
                self._active[index] = self.submit(self.domains[index])
            except Exception as exc:
                # A broken process pool may reject later fixed workers. Record
                # each one rather than lose its identity or silently restart it.
                self._results[index] = self._failure(index, exc)
            self.peak_admitted = max(self.peak_admitted, len(self._active))
            progress = True

        for index, future in tuple(self._active.items()):
            if not future.done():
                if self.ordered:
                    break
                continue
            try:
                result = future.result()
            except Exception as exc:
                result = self._failure(index, exc)
            self._results[index] = result
            del self._active[index]
            progress = True

        terminal = len(self._results) == len(self.domains)
        # Waiting only on the oldest unresolved domain avoids spinning on an
        # already-complete later Future when ordered settlement must retain it.
        waits = tuple(self._active.values())
        if self.ordered:
            waits = waits[:1]
        return AdvanceResult(
            progress=progress, terminal=terminal, waits=waits,
            runnable=not terminal and self._next < len(self.domains)
            and len(self._active) < self.capacity,
        )

    def run(self) -> list[_Result]:
        self.runner.run(self.advance)
        return [self._results[index] for index in range(len(self.domains))]


def run_process_domains(
    domains: Sequence[_Domain], operation: Callable[[_Domain], _Result], *,
    max_workers: int, mp_context: BaseContext,
    on_failure: Callable[[_Domain, Exception], _Result],
    executor_factory: Callable[..., Any] = ProcessPoolExecutor,
) -> list[_Result]:
    """Own a bounded process pool; preserve one-domain inline execution.

    No timeout is treated as quiescence: this compatibility path joins every
    admitted process before releasing the executor. The operation's internal
    engine and persistence lifecycle is deliberately outside this adapter.
    """
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("positive fixed-domain process concurrency required")
    if len(domains) == 1:
        def inline(domain: _Domain) -> Future[_Result]:
            future: Future[_Result] = Future()
            try:
                future.set_result(operation(domain))
            except Exception as exc:
                future.set_exception(exc)
            return future

        return FixedDomainSupervisor(
            domains, submit=inline, capacity=1, ordered=False, on_failure=on_failure,
        ).run()

    with executor_factory(max_workers=max_workers, mp_context=mp_context) as executor:
        def submit(domain: _Domain) -> Future[_Result]:
            return cast(Future[_Result], executor.submit(operation, domain))

        return FixedDomainSupervisor(
            domains, submit=submit, capacity=max_workers, ordered=False,
            on_failure=on_failure,
        ).run()
