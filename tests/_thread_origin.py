"""Which test started a thread: so a process-wide thread scan that fails names
its offenders (and whether one leaked from an EARLIER test or was born in the
current one) instead of printing ``assert False``.

Test-side only. ``install()`` wraps ``threading.Thread.start`` once per process
to remember ``PYTEST_CURRENT_TEST`` for each thread it starts; modules that
scan ``threading.enumerate()`` call it at import, so (imports happening at
collection) every thread the session's tests start is tagged.
"""
from __future__ import annotations

import os
import threading
import weakref
from collections.abc import Callable, Iterable

_ORIGIN: weakref.WeakKeyDictionary[threading.Thread, str] = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()


def install() -> None:
    with _LOCK:
        start = threading.Thread.start
        if getattr(start, "_records_origin", False):
            return

        def start_recording_origin(self: threading.Thread) -> None:
            try:  # diagnostics only: never break Thread.start (e.g. an unhashable subclass)
                _ORIGIN[self] = os.environ.get("PYTEST_CURRENT_TEST", "<outside any test>")
            except Exception:
                pass
            start(self)

        start_recording_origin._records_origin = True  # pyright: ignore[reportFunctionMemberAccess]
        threading.Thread.start = start_recording_origin


def origin(thread: threading.Thread) -> str:
    return _ORIGIN.get(thread, "<untagged: started before tagging was installed>")


def describe(threads: Iterable[threading.Thread],
             parked: Callable[[threading.Thread], bool] = lambda _t: False) -> str:
    """One line per thread: name, ident, native id, alive, parked, origin."""
    current = os.environ.get("PYTEST_CURRENT_TEST", "<outside any test>")
    lines = []
    for th in threads:
        born = origin(th)
        here = born.rsplit(" (", 1)[0] == current.rsplit(" (", 1)[0]
        lines.append(
            f"  {th.name} ident={th.ident} native_id={th.native_id} alive={th.is_alive()} "
            f"parked={parked(th)} started_in={born!r}"
            f"{' [THIS test]' if here else ' [an EARLIER test]'}"
        )
    return "\n".join(lines) or "  (none)"
