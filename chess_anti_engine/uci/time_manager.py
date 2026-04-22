"""Map ``go`` args to a budget the search worker can enforce.

Intentionally naive for v1: a single formula per clock type. Picking better
time-management heuristics (stability, ponder bonus, uncertainty-adaptive)
is a tuning exercise that should land after the engine plays at all.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .protocol import GoArgs


# Hard floor / ceiling guards on clock-derived deadlines.
_MIN_DEADLINE_MS = 20
# We never spend more than this fraction of remaining time on a single move,
# regardless of increment or movestogo claims.
_MAX_FRACTION_OF_REMAINING = 0.5
# Default divisor when movestogo is not specified (classic Leela/SF-lite).
_DEFAULT_MOVES_REMAINING = 30


@dataclass(frozen=True)
class SearchLimits:
    """Terminate the search as soon as ANY set bound is hit.

    ``None`` on every field means "infinite" — the search runs until an
    external ``stop`` / ``ponderhit`` arrives.
    """
    deadline_ms: int | None = None   # wall-clock budget in ms
    max_nodes: int | None = None     # total MCTS sims
    max_depth: int | None = None     # treat UCI depth as sims (coarse v1)
    infinite: bool = False
    ponder: bool = False             # no deadline until ponderhit

    def is_open_ended(self) -> bool:
        return self.infinite or self.ponder or (
            self.deadline_ms is None and self.max_nodes is None and self.max_depth is None
        )


def limits_from_go(
    args: GoArgs, *,
    side_to_move_is_white: bool,
    move_overhead_ms: int = 0,
) -> SearchLimits:
    if args.infinite:
        return SearchLimits(infinite=True)
    if args.ponder:
        # Ponder still wants a fallback budget (for `ponderhit` latency bounds)
        # but until ponderhit flips it live, the search runs open-ended.
        return SearchLimits(ponder=True, max_nodes=args.nodes, max_depth=args.depth)

    deadline_ms: int | None = None
    if args.movetime_ms is not None:
        deadline_ms = max(_MIN_DEADLINE_MS, int(args.movetime_ms))
    else:
        remaining, inc = _select_clock(args, side_to_move_is_white)
        if remaining is not None:
            moves_left = args.movestogo if args.movestogo and args.movestogo > 0 else _DEFAULT_MOVES_REMAINING
            budget = remaining / moves_left + (inc or 0)
            ceiling = remaining * _MAX_FRACTION_OF_REMAINING
            deadline_ms = max(_MIN_DEADLINE_MS, int(min(budget, ceiling)))

    # Reserve time for UCI command + GUI overhead (bestmove emission, pipe
    # latency). Without this, engines lose on time in fast games.
    if deadline_ms is not None and move_overhead_ms > 0:
        deadline_ms = max(_MIN_DEADLINE_MS, deadline_ms - int(move_overhead_ms))

    return SearchLimits(
        deadline_ms=deadline_ms,
        max_nodes=args.nodes,
        max_depth=args.depth,
    )


def _select_clock(args: GoArgs, white: bool) -> tuple[int | None, int | None]:
    if white:
        return args.wtime_ms, args.winc_ms
    return args.btime_ms, args.binc_ms


class Deadline:
    """Monotonic-clock deadline tracker.

    ``remaining_ms`` is the basis for "should we stop?" checks between sim
    chunks. When ``deadline_ms`` is None the deadline is effectively infinite.
    """

    def __init__(self, deadline_ms: int | None, *, now: float | None = None) -> None:
        self._start = now if now is not None else time.monotonic()
        self._deadline_s: float | None = (
            None if deadline_ms is None else self._start + deadline_ms / 1000.0
        )

    def elapsed_ms(self, *, now: float | None = None) -> int:
        t = now if now is not None else time.monotonic()
        return int((t - self._start) * 1000.0)

    def remaining_ms(self, *, now: float | None = None) -> int | None:
        if self._deadline_s is None:
            return None
        t = now if now is not None else time.monotonic()
        return max(0, int((self._deadline_s - t) * 1000.0))

    def expired(self, *, now: float | None = None) -> bool:
        if self._deadline_s is None:
            return False
        t = now if now is not None else time.monotonic()
        return t >= self._deadline_s
