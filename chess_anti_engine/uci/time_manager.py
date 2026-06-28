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
# Lc0-style allocation: divide the (reserved) base over an AGGRESSIVE estimate of
# the moves left, planning to deplete it by the hard middlegame, and rely on the
# visit-margin abort to bank most of that budget back on the many easy/booked
# moves — so in practice the clock lasts well past the nominal plan (no mid-search
# extension needed). `moves_left` is driven by PIECES rather than move number:
# material tracks phase + opening-book length robustly (a 10-move book leaves the
# board near its start, so it correctly reads as "early, long game ahead"), and it
# matches where this net is strong vs hard — booked 32->28, hardest middlegame down
# to ~16, then Syzygy takes over. Estimate ≈ (pieces - _ENDGAME_PIECES) * _MOVES_PER_PIECE.
_ENDGAME_PIECES = 12      # at/below this the game is ~decided by the endgame/TB; floor the estimate
_MOVES_PER_PIECE = 1.5    # planned moves remaining per piece above _ENDGAME_PIECES (32 pieces -> ~30)
# Floor on the moves estimate: never assume fewer than this many moves remain, so a
# long game can't dump the whole base into one move (the increment then carries it).
_MIN_MOVES_REMAINING = 8
# Default upper cap on the moves estimate (a conservatism guard; the pieces formula
# tops out near 30, so this normally does not bind). Exposed as `moves_horizon`.
_DEFAULT_MOVES_REMAINING = 50
# Fixed divisor used ONLY by the `optimum_fraction <= 0` baseline path, which
# reproduces the pre-time-management allocation (no reserve, no pieces estimate) so
# an A/B gauntlet can compare old vs new on one binary. This is the old
# `_DEFAULT_MOVES_REMAINING`; the new path divides by the pieces-driven estimate.
_BASELINE_MOVES_REMAINING = 30
# Safety reserve carved off `remaining` before allocating, so the late game/endgame
# can't flag: max of a few increments and a few move-overheads, but never more than
# this fraction of remaining (keeps it sane at short TCs where the inc-multiple would
# otherwise exceed the whole clock). The increment refills it every move.
_RESERVE_INC_MULT = 10
_RESERVE_OVERHEAD_MULT = 80
_MAX_RESERVE_FRACTION = 0.3
# Fraction of the increment folded into each move's base budget (the rest is left to
# refill the clock — Lc0 spends only part of the increment up front).
_INCREMENT_SPEND_FRACTION = 0.5
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
    pieces: int = 32,
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
            inc_v = inc or 0
            if optimum_fraction <= 0.0:
  # OFF sentinel: reproduce the pre-time-management allocation EXACTLY — fixed
  # divisor, full increment, NO reserve and NO pieces-driven split — so a baseline
  # binary is a true A/B against the old engine. Skipping only `optimum_ms` (below)
  # is not enough: the reserve + half-increment would still budget e.g. 2200ms vs
  # the old 4000ms at 30s+3s. A real `movestogo` from the GUI still wins.
                moves_left = (
                    args.movestogo if args.movestogo and args.movestogo > 0
                    else _BASELINE_MOVES_REMAINING
                )
                budget = remaining / moves_left + inc_v
            else:
  # Pieces-driven AGGRESSIVE allocation (a real `movestogo` from the GUI always
  # wins). Plan to spend the reserved base down by the hard middlegame; the
  # visit-margin abort banks most of it on easy moves, so the clock actually
  # lasts much longer (no extension mechanism needed). See the constants above.
                if args.movestogo and args.movestogo > 0:
                    moves_left = args.movestogo
                else:
                    est = round((max(0, int(pieces)) - _ENDGAME_PIECES) * _MOVES_PER_PIECE)
  # Clamp the horizon up to the floor first, so a small `--moves-horizon` (e.g. 1)
  # can't pull `moves_left` below `_MIN_MOVES_REMAINING` and dump the whole base
  # into one move; the floor is honored regardless of the horizon cap.
                    horizon = max(_MIN_MOVES_REMAINING, int(moves_horizon))
                    moves_left = min(horizon, max(_MIN_MOVES_REMAINING, est))
  # Carve a safety reserve off the clock first so the endgame can't flag, capped
  # at a fraction of remaining so it stays sane at short TCs. The increment refills
  # it each move. Only part of the increment is spent up front (the rest rolls over).
                reserve = min(
                    _MAX_RESERVE_FRACTION * remaining,
                    max(_RESERVE_INC_MULT * inc_v, _RESERVE_OVERHEAD_MULT * move_overhead_ms),
                )
                usable = max(0.0, remaining - reserve)
  # time_budget_scale lets the engine schedule more time per move on the bet that
  # the visit-margin abort banks most of it back; the 50%-of-remaining ceiling
  # still caps any single move, so the clock cannot be flagged regardless of scale.
                budget = time_budget_scale * (
                    usable / moves_left + _INCREMENT_SPEND_FRACTION * inc_v
                )
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
