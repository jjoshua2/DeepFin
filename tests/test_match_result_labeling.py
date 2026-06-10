from __future__ import annotations

from chess_anti_engine.selfplay.match import result_from_a_pov


def test_result_from_a_pov_treats_truncated_as_draw():
    assert result_from_a_pov("*", a_is_white=True) == 0
    assert result_from_a_pov("*", a_is_white=False) == 0


def test_result_from_a_pov_standard_results():
    assert result_from_a_pov("1/2-1/2", a_is_white=True) == 0
    assert result_from_a_pov("1/2-1/2", a_is_white=False) == 0

    assert result_from_a_pov("1-0", a_is_white=True) == 1
    assert result_from_a_pov("1-0", a_is_white=False) == -1

    assert result_from_a_pov("0-1", a_is_white=True) == -1
    assert result_from_a_pov("0-1", a_is_white=False) == 1
