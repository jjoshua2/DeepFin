from chess_anti_engine.selfplay.game import _result_to_wdl


def test_result_to_wdl_treats_truncated_game_as_draw():
    assert _result_to_wdl("*", pov_white=True) == 1
    assert _result_to_wdl("*", pov_white=False) == 1
