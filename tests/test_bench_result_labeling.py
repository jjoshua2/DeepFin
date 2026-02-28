from __future__ import annotations

from chess_anti_engine.bench.play_batch_timing import _result_to_wdl


def test_bench_result_to_wdl_treats_truncated_as_draw():
    assert _result_to_wdl("*", pov_white=True) == 1
    assert _result_to_wdl("*", pov_white=False) == 1


def test_bench_result_to_wdl_standard_results():
    assert _result_to_wdl("1/2-1/2", pov_white=True) == 1
    assert _result_to_wdl("1/2-1/2", pov_white=False) == 1

    assert _result_to_wdl("1-0", pov_white=True) == 0
    assert _result_to_wdl("1-0", pov_white=False) == 2

    assert _result_to_wdl("0-1", pov_white=True) == 2
    assert _result_to_wdl("0-1", pov_white=False) == 0
