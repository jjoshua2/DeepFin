from __future__ import annotations

from chess_anti_engine.uci.protocol import GoArgs
from chess_anti_engine.uci.time_manager import Deadline, limits_from_go


def test_infinite() -> None:
    lim = limits_from_go(GoArgs(infinite=True), side_to_move_is_white=True)
    assert lim.infinite is True
    assert lim.is_open_ended()
    assert lim.deadline_ms is None


def test_ponder_is_open_ended_until_ponderhit() -> None:
    lim = limits_from_go(GoArgs(ponder=True, wtime_ms=10000), side_to_move_is_white=True)
    assert lim.ponder is True
    assert lim.is_open_ended()
    assert lim.deadline_ms is None


def test_movetime_sets_deadline() -> None:
    lim = limits_from_go(GoArgs(movetime_ms=500), side_to_move_is_white=True)
    assert lim.deadline_ms == 500
    assert not lim.is_open_ended()


def test_movetime_floors_at_minimum() -> None:
    # Even a zero/low movetime must leave room to emit a legal move.
    lim = limits_from_go(GoArgs(movetime_ms=0), side_to_move_is_white=True)
    assert lim.deadline_ms is not None
    assert lim.deadline_ms >= 20


def test_nodes_only() -> None:
    lim = limits_from_go(GoArgs(nodes=250), side_to_move_is_white=True)
    assert lim.max_nodes == 250
    assert lim.deadline_ms is None
    assert not lim.is_open_ended()


def test_searchmoves_are_preserved_in_limits() -> None:
    lim = limits_from_go(
        GoArgs(nodes=250, searchmoves=("e2e4", "d2d4")),
        side_to_move_is_white=True,
    )
    assert lim.searchmoves == ("e2e4", "d2d4")


def test_depth_only() -> None:
    lim = limits_from_go(GoArgs(depth=8), side_to_move_is_white=True)
    assert lim.max_depth == 8


def test_clock_white_picks_wtime_winc() -> None:
    lim = limits_from_go(
        GoArgs(wtime_ms=30000, btime_ms=20000, winc_ms=500, binc_ms=300),
        side_to_move_is_white=True,
    )
    # move 1 (ply 0), default horizon 50: 30000/50 + 500 = 1100
    assert lim.deadline_ms == 1100


def test_clock_black_picks_btime_binc() -> None:
    lim = limits_from_go(
        GoArgs(wtime_ms=30000, btime_ms=20000, winc_ms=500, binc_ms=300),
        side_to_move_is_white=False,
    )
    # move 1 (ply 0), default horizon 50: 20000/50 + 300 = 700
    assert lim.deadline_ms == 700


def test_clock_movestogo_overrides_default_divisor() -> None:
    lim = limits_from_go(
        GoArgs(wtime_ms=60000, movestogo=10),
        side_to_move_is_white=True,
    )
    # 60000/10 = 6000, well below 50% ceiling of 30000
    assert lim.deadline_ms == 6000


def test_clock_ceiling_caps_half_remaining() -> None:
    # With movestogo=1 (one move to make) and no increment, naive math says
    # spend all remaining. We must cap at 50%.
    lim = limits_from_go(
        GoArgs(wtime_ms=10000, movestogo=1),
        side_to_move_is_white=True,
    )
    assert lim.deadline_ms == 5000


def test_movetime_has_no_optimum_so_it_searches_exactly() -> None:
    # movetime is "search exactly this long": no soft optimum, so the abort is
    # disabled and the search runs to the movetime deadline.
    lim = limits_from_go(GoArgs(movetime_ms=500), side_to_move_is_white=True)
    assert lim.deadline_ms == 500
    assert lim.optimum_ms is None


def test_time_budget_scale_raises_budget_below_ceiling() -> None:
    # With plenty of moves left the budget is under the 50% ceiling, so the
    # scale multiplies the hard deadline directly.
    base = limits_from_go(GoArgs(wtime_ms=60000, movestogo=40), side_to_move_is_white=True)
    scaled = limits_from_go(
        GoArgs(wtime_ms=60000, movestogo=40), side_to_move_is_white=True,
        time_budget_scale=2.0,
    )
    assert base.deadline_ms == 1500
    assert scaled.deadline_ms == 3000


def test_time_budget_scale_still_capped_at_half_remaining() -> None:
    # Even a large scale cannot exceed the 50%-of-remaining ceiling.
    lim = limits_from_go(
        GoArgs(wtime_ms=10000, movestogo=1), side_to_move_is_white=True,
        time_budget_scale=10.0,
    )
    assert lim.deadline_ms == 5000


def test_time_allocation_never_flags_over_a_game() -> None:
    # Time-control honesty: play out a long increment game where every move
    # spends its full hard deadline. The 50%-of-remaining ceiling must keep the
    # clock positive forever, even at an aggressive time_budget_scale.
    for scale in (1.0, 2.0, 8.0):
        remaining = 60_000
        inc = 1000
        for _ in range(400):
            lim = limits_from_go(
                GoArgs(wtime_ms=remaining, winc_ms=inc),
                side_to_move_is_white=True, move_overhead_ms=30,
                time_budget_scale=scale,
            )
            spend = lim.deadline_ms
            assert spend is not None and spend >= 1
            assert spend < remaining, f"would flag: spend={spend} remaining={remaining}"
            remaining = remaining - spend + inc
            assert remaining > 0


def test_clock_optimum_is_fraction_below_hard_deadline() -> None:
    lim = limits_from_go(
        GoArgs(wtime_ms=30000, winc_ms=500), side_to_move_is_white=True,
    )
    # Hard bound (ply 0, horizon 50: 30000/50 + 500 = 1100); optimum is a fraction.
    assert lim.deadline_ms == 1100
    assert lim.optimum_ms is not None
    assert lim.optimum_ms < lim.deadline_ms
    assert lim.optimum_ms == int(1100 * 0.7)


def test_optimum_none_without_time_budget() -> None:
    # Pure node/depth/infinite searches have no clock, so no soft target.
    assert limits_from_go(GoArgs(nodes=250), side_to_move_is_white=True).optimum_ms is None
    assert limits_from_go(GoArgs(depth=8), side_to_move_is_white=True).optimum_ms is None
    assert limits_from_go(GoArgs(infinite=True), side_to_move_is_white=True).optimum_ms is None
    assert limits_from_go(
        GoArgs(ponder=True, wtime_ms=10000), side_to_move_is_white=True,
    ).optimum_ms is None


def test_explicit_node_or_depth_bound_with_clock_disables_soft_abort() -> None:
    # A benchmark/GUI that sends explicit nodes/depth but also keeps the clocks
    # populated wants a fixed-work search; the soft optimum (which drives the
    # early abort) must stay off so it isn't cut short before the node/depth
    # bound. The hard deadline still applies as a safety bound.
    nodes = limits_from_go(
        GoArgs(nodes=1_000_000, wtime_ms=60000, winc_ms=500),
        side_to_move_is_white=True,
    )
    assert nodes.max_nodes == 1_000_000
    assert nodes.deadline_ms is not None  # clock still bounds it
    assert nodes.optimum_ms is None       # but no early-abort soft target

    depth = limits_from_go(
        GoArgs(depth=20, wtime_ms=60000),
        side_to_move_is_white=True,
    )
    assert depth.max_depth == 20
    assert depth.optimum_ms is None


def test_optimum_fraction_tunes_soft_target() -> None:
    # The soft target scales with optimum_fraction; deadline_ms is unchanged.
    go = GoArgs(wtime_ms=30000, winc_ms=500)
    half = limits_from_go(go, side_to_move_is_white=True, optimum_fraction=0.5)
    assert half.deadline_ms == 1100
    assert half.optimum_ms == int(1100 * 0.5)


def test_optimum_fraction_zero_disables_soft_abort() -> None:
    # 0 (or negative) is the OFF sentinel: no optimum_ms => _abort_ready inert =>
    # the search spends the whole deadline (pre-time-management baseline).
    go = GoArgs(wtime_ms=30000, winc_ms=500)
    off = limits_from_go(go, side_to_move_is_white=True, optimum_fraction=0.0)
    assert off.deadline_ms == 1100
    assert off.optimum_ms is None
    neg = limits_from_go(go, side_to_move_is_white=True, optimum_fraction=-1.0)
    assert neg.optimum_ms is None


def test_optimum_fraction_clamps_above_one_to_deadline() -> None:
    # A fraction > 1 would put the soft target past the hard bound; clamp to it
    # (soft target == deadline, so the post-optimum branch never fires early).
    go = GoArgs(wtime_ms=30000, winc_ms=500)
    lim = limits_from_go(go, side_to_move_is_white=True, optimum_fraction=2.5)
    assert lim.optimum_ms == lim.deadline_ms == 1100


def test_moves_horizon_countdown_draws_down_the_reserve() -> None:
    # As the game advances, moves-to-go shrinks, so the per-move base allocation
    # RISES — the reserve is spent down over the game rather than hoarded.
    go = GoArgs(wtime_ms=60000, winc_ms=1000)
    early = limits_from_go(go, side_to_move_is_white=True, moves_horizon=50, ply=0)
    mid = limits_from_go(go, side_to_move_is_white=True, moves_horizon=50, ply=60)
    assert early.deadline_ms is not None and mid.deadline_ms is not None
    # ply0: moves_left=50 -> 60000/50+1000=2200; ply60 (30 moves played):
    # moves_left=50-30=20 -> 60000/20+1000=4000.
    assert early.deadline_ms == 2200
    assert mid.deadline_ms == 4000
    assert mid.deadline_ms > early.deadline_ms


def test_moves_horizon_countdown_floors_for_long_games() -> None:
    # Past the expected length the countdown clamps to the floor (8), so a game
    # that outlasts the estimate can't dump the whole base into one move.
    go = GoArgs(wtime_ms=60000, winc_ms=1000)
    late = limits_from_go(go, side_to_move_is_white=True, moves_horizon=50, ply=200)
    # 100 moves played >> horizon 50 -> moves_left floored at 8 -> 60000/8+1000=8500
    # (below the 50%-of-remaining ceiling of 30000).
    assert late.deadline_ms == 8500


def test_movestogo_overrides_moves_horizon_countdown() -> None:
    # An explicit GUI movestogo always wins over the countdown (ply ignored).
    go = GoArgs(wtime_ms=60000, winc_ms=1000, movestogo=10)
    lim = limits_from_go(go, side_to_move_is_white=True, moves_horizon=50, ply=80)
    assert lim.deadline_ms == 7000  # 60000/10 + 1000


def test_deadline_tracking() -> None:
    d = Deadline(deadline_ms=500, now=100.0)
    # Floating-point subtraction at ms precision can round down by 1 ms; allow it.
    remaining_start = d.remaining_ms(now=100.0)
    remaining_mid = d.remaining_ms(now=100.4)
    assert remaining_start is not None
    assert remaining_mid is not None
    assert abs(remaining_start - 500) <= 1
    assert abs(remaining_mid - 100) <= 1
    assert d.remaining_ms(now=100.5) == 0
    assert d.expired(now=100.5) is True
    assert d.expired(now=100.2) is False


def test_deadline_none_never_expires() -> None:
    d = Deadline(deadline_ms=None, now=0.0)
    assert d.remaining_ms(now=10_000.0) is None
    assert d.expired(now=10_000.0) is False


def test_deadline_elapsed_monotonic() -> None:
    d = Deadline(deadline_ms=1000, now=50.0)
    assert d.elapsed_ms(now=50.0) == 0
    assert abs(d.elapsed_ms(now=50.123) - 123) <= 1


def test_time_capped_chunk_bounds_unbounded_first_chunk() -> None:
    """First-chunk forfeit guard: with a deadline but no nps estimate yet
    (total_nodes == 0), a scaled multi-GPU chunk must be bounded to the base
    single-GPU granularity so it can't run unbounded past the hard deadline
    (a move-1 time forfeit). Subsequent chunks use the nps-based cap."""
    from chess_anti_engine.uci.search import SearchWorker

    w = object.__new__(SearchWorker)
    w._chunk_sims = 512

    d = Deadline(deadline_ms=1000, now=0.0)
    # Scaled multi-GPU first chunk (per_device * n_devices) -> bounded to base.
    assert w._time_capped_chunk(8192, d, 0) == 512
    # Already-base first chunk (single-GPU) -> unchanged.
    assert w._time_capped_chunk(512, d, 0) == 512
    # Open-ended search (no deadline) -> never capped, even on the first chunk.
    d_open = Deadline(deadline_ms=None, now=0.0)
    assert w._time_capped_chunk(8192, d_open, 0) == 8192
