"""One executable, two-slot submit-before-retire pump for inference owners.

Routing, row compatibility and error policy belong to the caller. This module
owns the pipeline ordering and bounded stage capacity, not a second request
queue. The ORT bridge overlaps preparation with dispatcher-owned execution and
finish acceptance; only that dispatcher owner may call a session.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, cast
import math
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


class OwnerThreadCallbacks:
    """Bounded guard rendezvous, not an executor or a second work scheduler.

    Only the constructing controller may service callbacks. Worker threads wait
    for that controller's exact check; they never invoke it themselves. The
    controller must service during normal progress AND finite shutdown joins.
    A timed-out, not-yet-serviced check is cancelled and cannot later run.
    """

    def __init__(self, operation: Callable[[], None], *, capacity: int,
                 timeout: float, deadline: float):
        if (type(capacity) is not int or capacity < 1 or not math.isfinite(timeout)
                or timeout <= 0 or not math.isfinite(deadline)):
            raise ValueError("bounded owner callback capacity/time required")
        self._operation, self._capacity = operation, capacity
        self._timeout, self._deadline = timeout, deadline
        self._owner = threading.get_ident()
        self._lock = threading.Lock()
        self._pending: deque[tuple[Future[None], float]] = deque()
        self._failure: BaseException | None = None
        self._closed = False

    def _owner_only(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("owner callbacks require original controller thread")

    def _check(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise RuntimeError("owner callback bridge closed")
        if time.monotonic() >= self._deadline:
            raise TimeoutError("owner callback deadline expired")

    def _invoke(self) -> None:
        self._owner_only()
        try:
            with self._lock:
                self._check()
            self._operation()
        except BaseException as error:
            with self._lock:
                if self._failure is None:
                    self._failure = error
            raise

    def __call__(self) -> None:
        if threading.get_ident() == self._owner:
            self._invoke()
            return
        with self._lock:
            self._check()
            if len(self._pending) >= self._capacity:
                raise BufferError("owner callback capacity exhausted")
            future: Future[None] = Future()
            expires = min(self._deadline, time.monotonic() + self._timeout)
            self._pending.append((future, expires))
        try:
            future.result(timeout=max(0, expires - time.monotonic()))
        except TimeoutError as error:
            # cancel only succeeds before the owner starts the check. If it is
            # already running, the caller still refuses to execute new work.
            future.cancel()
            with self._lock:
                if self._failure is None:
                    self._failure = error
            raise

    def service(self) -> None:
        """Deliver checks/errors without aborting the owner's shutdown join.

        The next direct owner check raises a saved failure. Background callers
        get it through their own Future; service never polls companion/SQLite.
        """
        self._owner_only()
        with self._lock:
            pending = tuple(self._pending)
            self._pending.clear()
        for future, expires in pending:
            if not future.set_running_or_notify_cancel():
                continue
            try:
                if time.monotonic() >= expires:
                    raise TimeoutError("owner callback handoff expired")
                self._invoke()
            except BaseException as error:
                future.set_exception(error)
            else:
                future.set_result(None)

    def close(self) -> None:
        self._owner_only()
        with self._lock:
            self._closed = True
            pending = tuple(self._pending)
            self._pending.clear()
        for future, _expires in pending:
            if future.set_running_or_notify_cancel():
                future.set_exception(RuntimeError("owner callback bridge closed"))


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

    Submission hands preparation to its single-worker executor. Retirement is
    the rendezvous with the constructing dispatcher thread: that thread alone
    executes the session, finishes and accepts the output. The pump may submit
    the next preparation first, but cannot execute its session before accepting
    this output. In particular, an invalid finish poisons both owned slots.
    """

    def __init__(self, backend: TeacherBackend[Input, Output], *,
                 control: Callable[[], None] = lambda: None):
        self.backend = backend
        self._control = control
        self._staged_backend = (
            cast(StagedBackend[Input, Any, Any, Output], backend)
            if all(callable(getattr(backend, name, None))
                   for name in ("prepare", "execute", "finish")) else None
        )
        self._prepare = ThreadPoolExecutor(1, thread_name_prefix="TeacherPrepare")
        self._owner_thread = threading.get_ident()
        self._receipts: list[HostReceipt] = []
        self._work: list[Callable[[], tuple[Output, ...]]] = []
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
        self._owner_only()
        if self._closed or len(self._receipts) >= 2:
            raise RuntimeError("staged host slots are closed or exhausted")
        result: Future = Future()
        receipt = HostReceipt(result)
        self._receipts.append(receipt)
        with self._stats_lock:
            self._stats["peak_slots"] = max(self._stats["peak_slots"], len(self._receipts))
        staged = self._staged_backend
        prepared = self._prepare.submit(self._timed, "prepare", staged.prepare, rows) if staged is not None else None

        def execute() -> tuple[Output, ...]:
            if self._fatal is not None:
                raise self._fatal
            self._control()
            if staged is None:
                synchronous = cast(Callable[[Sequence[Input]], Sequence[Output]], self.backend)
                values = tuple(self._timed("execute", synchronous, rows))
            else:
                assert prepared is not None
                # A preparation can be opaque/blocking. Keep the owner-side
                # stop/deadline handoff responsive while retaining its slot;
                # public close(timeout) still retains a blocked preparation.
                while not prepared.done():
                    self._control()
                    wait((prepared,), timeout=0.05)
                value = prepared.result()
                self._control()
                raw = self._timed("execute", staged.execute, value)
                values = tuple(self._timed("finish", staged.finish, value, raw))
            if len(values) != len(rows):
                raise ValueError("teacher output count differs from dispatched rows")
            self._control()
            return values

        self._work.append(execute)
        return receipt

    def _owner_only(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("staged execution requires its dispatcher owner thread")

    def release(self, receipt: HostReceipt) -> Any:
        self._owner_only()
        if not self._receipts or self._receipts[0] is not receipt:
            raise RuntimeError("staged receipts must retire in submission order")
        if not receipt.future.done():
            try:
                receipt.future.set_result(self._work[0]())
            except BaseException as error:
                self._fatal = error
                # No later session may start after a failed guard/finish. Keep
                # the owned receipts and preparations until close quiesces them.
                for retained in self._receipts:
                    if not retained.future.done():
                        retained.future.set_exception(error)
        value = receipt.future.result()
        self._receipts.pop(0)
        self._work.pop(0)
        return value

    def close(self) -> None:
        # Called only on the pump thread after it has retired all output, or on
        # failure while it still owns the backend. close(timeout) on its caller
        # joins that thread and never destroys dependencies on timeout.
        self._owner_only()
        self._closed = True
        self._prepare.shutdown(wait=True, cancel_futures=False)


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
