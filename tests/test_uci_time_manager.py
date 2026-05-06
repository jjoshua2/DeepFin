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
