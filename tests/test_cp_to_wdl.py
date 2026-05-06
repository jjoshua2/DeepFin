from __future__ import annotations

from typing import Any, cast

import pytest

from chess_anti_engine.stockfish.uci import StockfishUCI
from chess_anti_engine.stockfish.wdl import cp_to_wdl, mate_to_effective_cp


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _parse_stockfish_lines(lines: list[str]):
    sf = cast(Any, object.__new__(StockfishUCI))
    sf.nodes = 1
    sf.read_timeout_s = 1.0
    sf._lock = _NullLock()
    pending = list(lines)
    sf._send = lambda cmd: None
    sf._readline_with_deadline = lambda deadline: pending.pop(0)
    return StockfishUCI.search(sf, "8/8/8/8/8/8/8/8 w - - 0 1")


def test_cp_zero_is_symmetric_and_normalized():
    p = cp_to_wdl(0, None, slope=0.008, draw_width_cp=80.0)
    assert p[0] == pytest.approx(p[2], abs=1e-6)
    assert pytest.approx(p.sum(), abs=1e-6) == 1.0
    assert p[1] > 0.0


def test_wider_draw_zone_increases_draw_probability_at_zero():
    p_narrow = cp_to_wdl(0, None, slope=0.01, draw_width_cp=40.0)
    p_wide = cp_to_wdl(0, None, slope=0.01, draw_width_cp=200.0)
    assert p_wide[1] > p_narrow[1]


def test_positive_cp_makes_white_winning():
    p_small = cp_to_wdl(50, None, slope=0.008, draw_width_cp=80.0)
    p_big = cp_to_wdl(500, None, slope=0.008, draw_width_cp=80.0)
    assert p_big[0] > p_small[0]
    assert p_big[2] < p_small[2]
    assert pytest.approx(p_big.sum(), abs=1e-6) == 1.0


def test_negative_cp_is_mirror_of_positive():
    p_pos = cp_to_wdl(300, None, slope=0.008, draw_width_cp=80.0)
    p_neg = cp_to_wdl(-300, None, slope=0.008, draw_width_cp=80.0)
    assert p_pos[0] == pytest.approx(p_neg[2], abs=1e-6)
    assert p_pos[2] == pytest.approx(p_neg[0], abs=1e-6)
    assert p_pos[1] == pytest.approx(p_neg[1], abs=1e-6)


def test_mate_dominates_cp_when_present():
    p_cp = cp_to_wdl(100, None, slope=0.008, draw_width_cp=80.0)
    p_mate = cp_to_wdl(100, 5, slope=0.008, draw_width_cp=80.0)
    assert p_mate[0] > p_cp[0]
    assert p_mate[0] > 0.99


def test_mate_negative_is_loss():
    p = cp_to_wdl(None, -3, slope=0.008, draw_width_cp=80.0)
    assert p[2] > 0.99
    assert p[0] < 0.01


def test_shorter_mate_more_decisive_than_long_mate():
    p1 = cp_to_wdl(None, 1, slope=0.008, draw_width_cp=80.0)
    p_long = cp_to_wdl(None, 49, slope=0.008, draw_width_cp=80.0)
    assert p1[0] >= p_long[0]


def test_slope_zero_rejected():
    with pytest.raises(ValueError):
        cp_to_wdl(0, None, slope=0.0, draw_width_cp=80.0)


def test_t1_5_target_shape_matches_softened_one_hot():
    """At slope=0.008, draw_width_cp=80, cp=300 should give roughly the
    distribution that the loss-side temperature=1.5 produces from a near-
    one-hot SF target. This is the calibration target: replacing the
    in-loss softening with this formula should not move the regime."""
    p = cp_to_wdl(300, None, slope=0.008, draw_width_cp=80.0)
    assert 0.80 <= p[0] <= 0.92
    assert 0.05 <= p[1] <= 0.15
    assert 0.02 <= p[2] <= 0.08


def test_mate_to_effective_cp_sign():
    assert mate_to_effective_cp(5) > 0
    assert mate_to_effective_cp(-5) < 0
    assert mate_to_effective_cp(1) > mate_to_effective_cp(40)


def test_uci_parser_clears_stale_mate_when_latest_score_is_cp():
    res = _parse_stockfish_lines([
        "info depth 1 multipv 1 score mate 3 wdl 1000 0 0 pv e2e4",
        "info depth 2 multipv 1 score cp 42 wdl 600 300 100 pv e2e4",
        "bestmove e2e4",
    ])
    assert res.cp == 42
    assert res.mate is None


def test_uci_parser_clears_stale_cp_when_latest_score_is_mate():
    res = _parse_stockfish_lines([
        "info depth 1 multipv 1 score cp 42 wdl 600 300 100 pv e2e4",
        "info depth 2 multipv 1 score mate -2 wdl 0 0 1000 pv e2e4",
        "bestmove e2e4",
    ])
    assert res.cp is None
    assert res.mate == -2
