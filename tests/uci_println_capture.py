"""Stand-in for `chess_anti_engine.uci.engine._println` in tests.

The real `_println` returns whether it wrote the line, and `Engine` records
the head of the last printed `info ... pv` line from that answer (the
alignment guard in `_publish_bestmove`). A stand-in that returns None, such
as a bare `list.append`, would silently disable that guard, so tests capture
output through this helper instead.
"""
from __future__ import annotations

from collections.abc import Callable


def capture_into(sink: list[str]) -> Callable[[str], bool]:
    """A `_println` replacement that appends to ``sink`` and reports a write."""

    def println(line: str) -> bool:
        sink.append(line)
        return True

    return println
