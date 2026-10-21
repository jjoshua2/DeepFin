"""One executable, two-slot submit-before-retire pump for inference owners.

Routing, row compatibility and error policy belong to the caller. This module
owns the pipeline ordering and bounded stage capacity, not a second request
queue. The ORT bridge gives preparation, execution and finishing independent
progress capacity; only its single execution owner may call a session.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, cast
import threading
import time

from chess_anti_engine.async_safety import Readiness
from chess_anti_engine.request_routing import ControlRoute, RoutedBatch


Input = TypeVar("Input")
Output = TypeVar("Output")
InputContra = TypeVar("InputContra", contravariant=True)
OutputCo = TypeVar("OutputCo", covariant=True)
Prepared = TypeVar("Prepared")
Raw = TypeVar("Raw")


class StagedBackend(Protocol[InputContra, Prepared, Raw, OutputCo]):
    """Canonical host/session/host boundary; only execute may use the session."""

    def prepare(self, rows: Sequence[InputContra], /) -> Prepared: ...
    def execute(self, prepared: Prepared, /) -> Raw: ...
    def finish(self, prepared: Prepared, raw: Raw, /) -> Sequence[OutputCo]: ...


class CallableStagedBackend(StagedBackend[InputContra, Prepared, Raw, OutputCo], Protocol):
    """Staged adapter also exposing its canonical synchronous compatibility API."""

    def __call__(self, rows: Sequence[InputContra], /) -> Sequence[OutputCo]: ...


# Prepared/raw transports differ by backend; public input and output remain
# typed while the generic pump deliberately treats those intermediates opaquely.
TeacherBackend: TypeAlias = (
    StagedBackend[Input, Any, Any, Output]
    | Callable[[Sequence[Input]], Sequence[Output]]
)


class TwoSlotPump:
    """Extracted dispatcher loop, shared by Torch and staged teacher backends.

    A lease is not reused until retirement snapshots/publishes its owned output.
    No new-input wait is permitted while an old completion is ready or unknown.
    The caller's fatal handler retains custody when quiescence is unproved.
    """

    def __init__(self, *, take: Callable[..., Any], submit: Callable[..., Any],
                 retire: Callable[[Any], None], readiness: Callable[[Any], Readiness],
                 failed: Callable[[Any, BaseException, str], None],
                 control: Callable[[Any], None] | None = None,
                 tick: Callable[[], None] = lambda: None,
                 abort: Callable[[tuple[Any, ...], BaseException], None] = lambda *_: None):
        self.take, self.submit, self.retire = take, submit, retire
        self.readiness, self.failed, self.control = readiness, failed, control
        self.tick, self.abort = tick, abort
        self.pending: Any = None
        self.next_handle: Any = None

    def run(self) -> None:
        slot = 0
        try:
            while True:
                pending = self.pending
                state = Readiness.READY if pending is None else self.readiness(pending)
                route = self.take(blocking=pending is None,
                                  coalesce_wait=state is Readiness.NOT_READY)
                self.next_handle = None
                if isinstance(route, RoutedBatch):
                    try:
                        self.next_handle = self.submit(route, slot)
                        slot = 1 - slot
                    except Exception as error:
                        self.failed(route, error, "submit")
                self.tick()
                if pending is not None:
                    try:
                        self.retire(pending)
                    except Exception as error:
                        self.failed(pending, error, "retire")
                    self.pending = None
                if isinstance(route, ControlRoute):
                    if self.control is None:
                        raise RuntimeError("unsupported execution control barrier")
                    self.control(route.payload)
                self.pending, self.next_handle = self.next_handle, None
                # Blocking take returns None only when admission is closed.
                # A nonblocking miss merely retired an older result.
                if route is None and self.pending is None and pending is None:
                    return
        except BaseException as error:
            retained = tuple(handle for handle in (self.pending, self.next_handle)
                             if handle is not None)
            self.abort(retained, error)
            raise


@dataclass
class HostReceipt:
    """Owned output Future, distinct from the execution owner's progress."""
    future: Future

    def readiness(self) -> Readiness:
        return Readiness.READY if self.future.done() else Readiness.NOT_READY

    def synchronize(self) -> None:
        self.future.result()


class StagedHostBackend(Generic[Input, Output]):
    """Bounded by the pump's two leases; no producer-pool dependency.

    prepare and finish have separate single-worker executors. execute alone has
    a dedicated session owner; it schedules finish but never waits for it. A
    Future around an opaque evaluator is supported only for legacy callables;
    production adapters implement the canonical three-stage API.
    """

    def __init__(self, backend: TeacherBackend[Input, Output]):
        self.backend = backend
        self._staged_backend = (
            cast(StagedBackend[Input, Any, Any, Output], backend)
            if all(callable(getattr(backend, name, None))
                   for name in ("prepare", "execute", "finish")) else None
        )
        self._prepare = ThreadPoolExecutor(1, thread_name_prefix="TeacherPrepare")
        self._execute = ThreadPoolExecutor(1, thread_name_prefix="TeacherSession")
        self._finish = ThreadPoolExecutor(1, thread_name_prefix="TeacherFinish")
        self._receipts: list[HostReceipt] = []
        self._closed = False
        self._fatal: BaseException | None = None
        self._stats_lock = threading.Lock()
        self._stats: dict[str, float | int] = {
            "prepare_seconds": 0.0, "execute_seconds": 0.0, "finish_seconds": 0.0,
            "prepare_calls": 0, "execute_calls": 0, "finish_calls": 0, "peak_slots": 0,
        }

    @property
    def stats(self) -> dict[str, float | int]:
        with self._stats_lock:
            return dict(self._stats)

    def _timed(self, stage: str, operation: Callable[..., Any], *args: Any) -> Any:
        started = time.perf_counter()
        try:
            return operation(*args)
        finally:
            with self._stats_lock:
                self._stats[stage + "_seconds"] += time.perf_counter() - started
                self._stats[stage + "_calls"] += 1

    def submit(self, rows: tuple[Input, ...]) -> HostReceipt:
        if self._closed or len(self._receipts) >= 2:
            raise RuntimeError("staged host slots are closed or exhausted")
        result: Future = Future()
        receipt = HostReceipt(result)
        self._receipts.append(receipt)
        with self._stats_lock:
            self._stats["peak_slots"] = max(self._stats["peak_slots"], len(self._receipts))
        staged = self._staged_backend
        prepared = self._prepare.submit(self._timed, "prepare", staged.prepare, rows) if staged is not None else None

        def settle(finished: Future) -> None:
            try:
                values = tuple(finished.result())
                if len(values) != len(rows):
                    raise ValueError("teacher output count differs from dispatched rows")
                result.set_result(values)
            except BaseException as error:
                self._fatal = error
                result.set_exception(error)

        def execute() -> None:
            try:
                if self._fatal is not None:
                    raise self._fatal
                if staged is None:
                    # Backward-compatible fixtures remain serialized on the
                    # same session owner, without claiming host-stage overlap.
                    synchronous = cast(Callable[[Sequence[Input]], Sequence[Output]], self.backend)
                    values = tuple(self._timed("execute", synchronous, rows))
                    if len(values) != len(rows):
                        raise ValueError("teacher output count differs from dispatched rows")
                    result.set_result(values)
                    return
                assert prepared is not None
                value = prepared.result()
                raw = self._timed("execute", staged.execute, value)
                finished = self._finish.submit(self._timed, "finish", staged.finish, value, raw)
                finished.add_done_callback(settle)
            except BaseException as error:
                self._fatal = error
                result.set_exception(error)

        self._execute.submit(execute)
        return receipt

    def release(self, receipt: HostReceipt) -> Any:
        value = receipt.future.result()
        self._receipts.remove(receipt)
        return value

    def close(self) -> None:
        # Called only on the pump thread after it has retired all output, or on
        # failure while it still owns the backend. close(timeout) on its caller
        # joins that thread and never destroys dependencies on timeout.
        self._closed = True
        self._prepare.shutdown(wait=True, cancel_futures=False)
        self._execute.shutdown(wait=True, cancel_futures=False)
        self._finish.shutdown(wait=True, cancel_futures=False)


def guard_staged_backend(backend: Any, before_execute: Callable[[], None]) -> Any:
    """Add a control/proof guard without erasing the canonical stage boundary."""
    if not all(callable(getattr(backend, name, None)) for name in ("prepare", "execute", "finish")):
        def synchronous(rows: Any) -> Any:
            before_execute()
            return backend(rows)
        return synchronous

    class GuardedBackend:
        def prepare(self, rows: Any) -> Any:
            return backend.prepare(rows)

        def execute(self, prepared: Any) -> Any:
            before_execute()
            return backend.execute(prepared)

        def finish(self, prepared: Any, raw: Any) -> Any:
            return backend.finish(prepared, raw)

        def __call__(self, rows: Any) -> Any:
            prepared = self.prepare(rows)
            return self.finish(prepared, self.execute(prepared))

    return GuardedBackend()
