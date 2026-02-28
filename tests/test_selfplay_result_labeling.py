from __future__ import annotations

from chess_anti_engine.selfplay.game import _result_to_wdl


def test_result_to_wdl_star_is_draw():
    assert _result_to_wdl("*", pov_white=True) == 1
    assert _result_to_wdl("*", pov_white=False) == 1


def test_result_to_wdl_standard_results():
    assert _result_to_wdl("1/2-1/2", pov_white=True) == 1
    assert _result_to_wdl("1/2-1/2", pov_white=False) == 1

    assert _result_to_wdl("1-0", pov_white=True) == 0
    assert _result_to_wdl("1-0", pov_white=False) == 2

    assert _result_to_wdl("0-1", pov_white=True) == 2
    assert _result_to_wdl("0-1", pov_white=False) == 0
