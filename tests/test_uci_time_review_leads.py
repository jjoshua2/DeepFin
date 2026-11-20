"""Time-control leads from an outside-model review (2026-11-19), each pinned by
the case that reproduced it.

- T1: ``go movetime M wtime W`` with ``W`` far below ``M`` searched the whole
  movetime: the clock was ignored by the deadline and by the watchdog.
- T2: any increment skipped the at-once rule, so ``wtime 75 winc 10`` searched
  for 20 ms although one minimal search (68 ms on CPU) cost more than the clock.
- T4: ``movestogo`` skipped it too: ``wtime 500 movestogo 40`` searched 20 ms
  moves it could not pay for 40 of.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import GoArgs, parse_command
from chess_anti_engine.uci.time_manager import (
    _MIN_DEADLINE_MS,
    limits_from_go,
    minimal_search_cost_ms,
)

from tests.test_uci_time_forfeits import _mock_worker, _Out


@pytest.fixture
def out(monkeypatch) -> _Out:
    capture = _Out()
    monkeypatch.setattr(engine_module, "sys", SimpleNamespace(stdout=capture, stderr=sys.stderr))
    return capture


def _lim(**kw: Any):
    overhead = kw.pop("overhead", 30)
    min_chunk = kw.pop("min_chunk_ms", None)
    return limits_from_go(GoArgs(**kw), side_to_move_is_white=True,
                          move_overhead_ms=overhead, min_chunk_ms=min_chunk)


# -- T1: movetime with a clock ----------------------------------------------------


def test_a_movetime_longer_than_the_clock_is_capped_by_the_clock() -> None:
    # 10 ms on the clock, 1000 ms movetime: nothing can be searched.
    lim = _lim(movetime_ms=1000, wtime_ms=10, btime_ms=10)
    assert lim.clock_exhausted
    assert lim.deadline_ms == 0
    # The clock's own deadline for the move caps the movetime: 60 s, 32 pieces
    # plans (60000 - 2400 reserve) / 30 = 1920 ms, less MoveOverheadMs.
    lim = _lim(movetime_ms=5000, wtime_ms=60_000, btime_ms=60_000)
    assert lim.deadline_ms == 1890
    assert not lim.clock_exhausted
    assert lim.fixed_time
    assert lim.optimum_ms is None
    # A clock whose own deadline is longer leaves the movetime (less overhead) alone.
    assert _lim(movetime_ms=1000, wtime_ms=60_000, btime_ms=60_000).deadline_ms == 970
    # The OTHER side's clock does not cap it, and no clock leaves it as before.
    assert _lim(movetime_ms=1000, wtime_ms=60_000, btime_ms=10).deadline_ms == 970
    assert _lim(movetime_ms=1000).deadline_ms == 970
    assert not _lim(movetime_ms=1000).clock_exhausted


def test_a_movetime_with_a_clock_obeys_the_clocks_at_once_rule() -> None:
    # One minimal search costs 68 ms against 45 ms left: at once.
    assert _lim(movetime_ms=1000, wtime_ms=75, btime_ms=75, winc_ms=10, binc_ms=10,
                min_chunk_ms=40.0).clock_exhausted
    # 470 ms owed to 40 moves: at once, not a 445 ms search.
    assert _lim(movetime_ms=1000, wtime_ms=500, btime_ms=500, movestogo=40).clock_exhausted
    # Sudden death below 20 minimal searches (900 ms): at once.
    assert _lim(movetime_ms=1000, wtime_ms=400, btime_ms=400).clock_exhausted
    assert not _lim(movetime_ms=1000, wtime_ms=2000, btime_ms=2000).clock_exhausted


def test_the_watchdog_budget_of_a_movetime_is_capped_by_the_clock() -> None:
    engine = Engine(_mock_worker(), options=EngineOptions())
    try:
        budget = engine._move_budget_ms
        assert budget(GoArgs(movetime_ms=1000, wtime_ms=400), white_to_move=True) == 370
        assert budget(GoArgs(movetime_ms=1000, wtime_ms=60_000), white_to_move=True) == 1000
        assert budget(GoArgs(movetime_ms=1000), white_to_move=True) == 1000
        assert budget(GoArgs(movetime_ms=1000, btime_ms=10), white_to_move=True) == 1000
    finally:
        engine.close()


def test_go_movetime_with_an_exhausted_clock_is_answered_at_once(out) -> None:
    worker = _mock_worker()
    worker.tree_root_bestmove.return_value = "e7e5"
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command("go movetime 1000 wtime 10 btime 10"))
        assert out.wait_bestmoves(1) == ["e7e5"], out.text()
        engine.dispatch(parse_command("go movetime 1000 wtime 500 btime 500 movestogo 40"))
        assert out.wait_bestmoves(2) == ["e7e5", "e7e5"], out.text()
        assert worker.run.call_count == 0
        assert "clock too short to search: 10 ms on the clock" in out.text()
        assert "clock too short to search: 500 ms on the clock" in out.text()
    finally:
        engine.close()


# -- T2 / T4: increments and movestogo in the at-once rule ------------------------


def test_an_increment_below_the_search_cost_does_not_skip_the_at_once_rule() -> None:
    # The review's case: one minimal search (20 + 40 + 8 = 68 ms) costs more
    # than the 45 ms the clock has after MoveOverheadMs.
    assert _lim(wtime_ms=75, btime_ms=75, winc_ms=10, binc_ms=10, min_chunk_ms=40.0).clock_exhausted
    # A small increment funds only part of each search: 20 x (45 - 10) = 700 ms.
    assert _lim(wtime_ms=729, btime_ms=729, winc_ms=10, binc_ms=10).clock_exhausted
    assert not _lim(wtime_ms=731, btime_ms=731, winc_ms=10, binc_ms=10).clock_exhausted
    # An increment that pays for a whole search leaves only the one-search floor.
    assert not _lim(wtime_ms=76, btime_ms=76, winc_ms=100, binc_ms=100,
                    min_chunk_ms=1.0).clock_exhausted
    assert _lim(wtime_ms=75, btime_ms=75, winc_ms=100, binc_ms=100,
                min_chunk_ms=40.0).clock_exhausted  # 45 < 68
    assert not _lim(wtime_ms=98, btime_ms=98, winc_ms=100, binc_ms=100,
                    min_chunk_ms=40.0).clock_exhausted  # 68 >= 68


def test_movestogo_caps_the_searches_the_clock_must_fund() -> None:
    # The review's case: 470 ms cannot pay for 20 searches of 45 ms.
    assert _lim(wtime_ms=500, btime_ms=500, movestogo=40).clock_exhausted
    # A refill sooner than 20 moves: only that many searches to fund.
    assert not _lim(wtime_ms=500, btime_ms=500, movestogo=5).clock_exhausted  # 470 >= 225
    assert _lim(wtime_ms=250, btime_ms=250, movestogo=5).clock_exhausted  # 220 < 225
    # Beyond 20 moves the rule is sudden death's: 20 x 45 = 900 ms.
    assert _lim(wtime_ms=929, btime_ms=929, movestogo=40).clock_exhausted
    assert not _lim(wtime_ms=931, btime_ms=931, movestogo=40).clock_exhausted
    # `movestogo 0` is "no repeating control": sudden death's 20 searches.
    assert _lim(wtime_ms=500, btime_ms=500, movestogo=0).clock_exhausted
    assert not _lim(wtime_ms=931, btime_ms=931, movestogo=0).clock_exhausted
    # The last move of an allotment still needs one search's worth.
    assert not _lim(wtime_ms=80, btime_ms=80, movestogo=1).clock_exhausted
    assert _lim(wtime_ms=74, btime_ms=74, movestogo=1).clock_exhausted


def test_explicit_bounds_and_the_off_baseline_keep_their_old_rule() -> None:
    for kw in ({"depth": 6}, {"nodes": 500}, {"mate": 2}):
        assert not _lim(wtime_ms=300, btime_ms=300, winc_ms=10, binc_ms=10, **kw).clock_exhausted
    off = limits_from_go(GoArgs(wtime_ms=300, btime_ms=300, movestogo=40),
                         side_to_move_is_white=True, move_overhead_ms=30, optimum_fraction=0.0)
    assert not off.clock_exhausted


def test_a_normal_increment_clock_is_never_answered_at_once() -> None:
    cost = minimal_search_cost_ms(40.0)
    for remaining in (100, 300, 1000, 3000, 60_000):
        for inc in (100, 1000):
            lim = _lim(wtime_ms=remaining, btime_ms=remaining, winc_ms=inc, binc_ms=inc,
                       min_chunk_ms=40.0)
            assert lim.clock_exhausted is (remaining - 30 < cost), (remaining, inc)
            if not lim.clock_exhausted:
                assert lim.deadline_ms is not None
                assert lim.deadline_ms >= _MIN_DEADLINE_MS
