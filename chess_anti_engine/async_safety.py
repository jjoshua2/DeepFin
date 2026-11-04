"""Failure-only quiescence for reusable asynchronous inference buffers.

Unknown submit errors carry no proof that a pinned input is free. Such backends
are quarantined; only a successful drain of the exact launch stream authorizes
reuse. Keep failed owners and receipts alive for the lifetime of this process.
"""
from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from enum import Enum
from typing import Any, Protocol


class CompletionEvent(Protocol):
    """A completion receipt; synchronize before reading returned CPU tensors."""

    def synchronize(self) -> None: ...


class Readiness(Enum):
    READY = "ready"
    NOT_READY = "not_ready"
    UNKNOWN = "unknown"


def completion_readiness(event: Any) -> Readiness:
    """Absence/failure of a query is not evidence that work is still running."""
    if event is None:
        return Readiness.READY
    readiness = getattr(event, "readiness", None)
    if callable(readiness):
        try:
            value = readiness()
            return value if isinstance(value, Readiness) else Readiness.UNKNOWN
        except Exception:
            return Readiness.UNKNOWN
    query = getattr(event, "query", None)
    if not callable(query):
        return Readiness.UNKNOWN
    try:
        return Readiness.READY if query() else Readiness.NOT_READY
    except Exception:
        return Readiness.UNKNOWN


_quarantine: list[tuple[Any, ...]] = []
_poisoned: set[int] = set()


def _backend(owner: Any) -> Any:
    """Dispatch wrappers explicitly expose the backend whose buffers they own."""
    while isinstance(getattr(type(owner), "async_backend", None), property):
        owner = owner.async_backend
    return owner


def ensure_usable(owner: Any) -> None:
    if id(owner) in _poisoned or id(_backend(owner)) in _poisoned:
        raise RuntimeError("asynchronous inference backend is poisoned; restart required")


def poison(owner: Any, *retained: Any) -> None:
    _poisoned.add(id(owner))
    _poisoned.add(id(_backend(owner)))
    _quarantine.append((owner, *retained))


def _mark_quiesced(error: BaseException, owner: Any) -> None:
    owners = getattr(error, "_async_quiesced_owners", ())
    setattr(error, "_async_quiesced_owners", (*owners, id(owner)))


def _quiesced(error: BaseException, owner: Any) -> bool:
    return id(_backend(owner)) in getattr(error, "_async_quiesced_owners", ())


def note_cleanup(error: BaseException, message: str) -> None:
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(message)
    else:
        notes = getattr(error, "_async_cleanup_notes", ())
        setattr(error, "_async_cleanup_notes", (*notes, message))


def pending_receipt(handle: tuple[Any, ...]) -> Any:
    return handle[3] if handle[0] == "gpu_full" else handle[5]


@contextmanager
def launch_guard(owner: Any, stream: Any):
    """Drain this launch's stream on error; preserve the original exception."""
    try:
        yield
    except BaseException as error:
        try:
            if stream is not None:
                stream.synchronize()
        except BaseException as cleanup:
            poison(owner, stream, error, cleanup)
            note_cleanup(error, f"launch stream drain failed: {cleanup!r}; backend poisoned")
        else:
            _mark_quiesced(error, owner)
        raise


class _Completion:
    def __init__(self, owner: Any, event: Any, stream: Any, retained: tuple[Any, ...]):
        self._owner, self._event, self._stream = owner, event, stream
        self._retained = retained

    def readiness(self) -> Readiness:
        return completion_readiness(self._event)

    def synchronize(self) -> None:
        with launch_guard(self._owner, self._stream):
            self._event.synchronize()


def guarded_event(owner: Any, event: Any, stream: Any, *retained: Any) -> CompletionEvent:
    return _Completion(owner, event, stream, retained)


def submit_async(owner: Any, *args: Any, **kwargs: Any) -> Any:
    ensure_usable(owner)
    try:
        return owner.evaluate_inplace_async(*args, **kwargs)
    except BaseException as error:
        if not _quiesced(error, owner):
            poison(owner, error)
        raise


def drain_failed_receipts(owner: Any, helper: Any, receipts: Iterable[Any], error: BaseException) -> None:
    """Attempt every known receipt even if another drain fails."""
    for receipt in receipts:
        try:
            event = receipt[2]
            if event is not None:
                event.synchronize()
        except BaseException as cleanup:
            if not _quiesced(cleanup, owner):
                poison(owner, receipt, cleanup)
            note_cleanup(error, f"pending receipt drain failed: {cleanup!r}")
    if id(owner) in _poisoned or id(_backend(owner)) in _poisoned:
        poison(helper, owner, receipts, error)
