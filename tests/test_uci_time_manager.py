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
    # ~30000/30 + 500 = 1500
    assert lim.deadline_ms == 1500


def test_clock_black_picks_btime_binc() -> None:
    lim = limits_from_go(
        GoArgs(wtime_ms=30000, btime_ms=20000, winc_ms=500, binc_ms=300),
        side_to_move_is_white=False,
    )
    # ~20000/30 + 300 = 966
    assert lim.deadline_ms == 966


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
    # Hard bound unchanged (~30000/30 + 500 = 1500); optimum is a fraction of it.
    assert lim.deadline_ms == 1500
    assert lim.optimum_ms is not None
    assert lim.optimum_ms < lim.deadline_ms
    assert lim.optimum_ms == int(1500 * 0.7)


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
