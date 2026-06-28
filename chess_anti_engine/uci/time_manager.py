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
# Soft target as a fraction of the hard budget for clock-based searches. The
# search aims to finish around here and stops early once the best move is
# stable, banking the rest; if the move is still changing it keeps going up to
# the hard `deadline_ms`. Because the soft stop only fires *before* the hard
# bound, it can never flag where the old (always-spend-the-budget) code did not.
# This is the main time-management tuning knob — sweep with arena validation.
# A value of 0 (or negative) is the documented OFF sentinel: no `optimum_ms` is
# set, the visit-margin abort is fully inert, and clock games spend the whole
# `deadline_ms` exactly like the pre-time-management build. That makes the
# baseline reachable on one binary so an A/B gauntlet can compare old vs new.
# Values are clamped to (0, 1]; 1.0 keeps the soft target at the hard deadline.
_OPTIMUM_FRACTION = 0.7


@dataclass(frozen=True)
class SearchLimits:
    """Terminate the search as soon as ANY set bound is hit.

    ``None`` on every field means "infinite" — the search runs until an
    external ``stop`` / ``ponderhit`` arrives.
    """
    deadline_ms: int | None = None  # hard wall-clock budget in ms
    optimum_ms: int | None = None  # soft target; early-exit on a stable best move
    max_nodes: int | None = None  # total MCTS sims
    max_depth: int | None = None  # treat UCI depth as sims (coarse v1)
    infinite: bool = False
    ponder: bool = False  # no deadline until ponderhit
    searchmoves: tuple[str, ...] = ()

    def is_open_ended(self) -> bool:
        return self.infinite or self.ponder or (
            self.deadline_ms is None and self.max_nodes is None and self.max_depth is None
        )


def limits_from_go(
    args: GoArgs, *,
    side_to_move_is_white: bool,
    move_overhead_ms: int = 0,
    time_budget_scale: float = 1.0,
    optimum_fraction: float = _OPTIMUM_FRACTION,
    moves_horizon: int = _DEFAULT_MOVES_REMAINING,
) -> SearchLimits:
    if args.infinite:
        return SearchLimits(infinite=True, searchmoves=tuple(args.searchmoves))
    if args.ponder:
  # Ponder still wants a fallback budget (for `ponderhit` latency bounds)
  # but until ponderhit flips it live, the search runs open-ended.
        return SearchLimits(
            ponder=True,
            max_nodes=args.nodes,
            max_depth=args.depth,
            searchmoves=tuple(args.searchmoves),
        )

    deadline_ms: int | None = None
    movetime_ms = args.movetime_ms
    is_movetime = movetime_ms is not None
    if movetime_ms is not None:
        deadline_ms = max(_MIN_DEADLINE_MS, int(movetime_ms))
    else:
        remaining, inc = _select_clock(args, side_to_move_is_white)
        if remaining is not None:
  # `moves_horizon` is the rolling count of moves we spread the *base* reserve
  # over (a real `movestogo` from the GUI always wins). It is the front-loading
  # lever: a smaller horizon spends a larger slice of the base each move, so the
  # bulk of the base is gone by the early/middlegame and later moves coast on the
  # increment (the TCEC-style "spend it early, then ride the +inc" curve). The
  # increment is added every move, so once the base is spent the per-move budget
  # tends to `inc`.
            moves_left = args.movestogo if args.movestogo and args.movestogo > 0 else max(1, int(moves_horizon))
  # time_budget_scale lets the engine schedule more time per move on the bet
  # that the visit-margin abort banks most of it back on easy moves; the
  # 50%-of-remaining ceiling still caps any single move, so the clock cannot be
  # flagged regardless of scale.
            budget = time_budget_scale * (remaining / moves_left + (inc or 0))
            ceiling = remaining * _MAX_FRACTION_OF_REMAINING
            deadline_ms = max(_MIN_DEADLINE_MS, int(min(budget, ceiling)))

  # Reserve time for UCI command + GUI overhead (bestmove emission, pipe
  # latency). Without this, engines lose on time in fast games.
    if deadline_ms is not None and move_overhead_ms > 0:
        deadline_ms = max(_MIN_DEADLINE_MS, deadline_ms - int(move_overhead_ms))

  # Soft target the visit-margin abort aims for. Only pure clock searches get
  # one: `movetime` is an explicit "search exactly this long" instruction, and
  # an explicit `nodes`/`depth` bound means the caller (a benchmark or GUI that
  # also keeps the clocks populated) wants a fixed-work search — the soft abort
  # must not stop it early before that bound is reached. Clock games target a
  # fraction of the hard budget and extend toward `deadline_ms` on unsettled
  # positions. The hard `deadline_ms` still applies as a safety bound in all
  # cases so the engine cannot flag.
  #
  # `optimum_fraction <= 0` is the OFF sentinel: leaving `optimum_ms` None makes
  # `_abort_ready` inert (it returns False with no optimum), so the search spends
  # the full `deadline_ms` — the pre-time-management baseline, reachable without
  # a separate binary. Otherwise clamp the fraction to (0, 1].
    optimum_ms: int | None = None
    if (
        deadline_ms is not None
        and not is_movetime
        and args.nodes is None
        and args.depth is None
        and optimum_fraction > 0.0
    ):
        frac = min(1.0, optimum_fraction)
        optimum_ms = max(_MIN_DEADLINE_MS, int(deadline_ms * frac))

    return SearchLimits(
        deadline_ms=deadline_ms,
        optimum_ms=optimum_ms,
        max_nodes=args.nodes,
        max_depth=args.depth,
        searchmoves=tuple(args.searchmoves),
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
