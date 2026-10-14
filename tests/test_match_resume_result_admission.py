"""Durable UCI finished-game rows must preserve result/color/score identity."""
from pathlib import Path

import chess
import pytest

from chess_anti_engine.utils.game_log import GameLogWriter
from scripts import match_vs_uci as match


def _log(tmp_path: Path, index: int, result: object, score: object):
    path = tmp_path / "match.games.jsonl"
    settings = {"games": 2}
    openings = [chess.Board()]
    with GameLogWriter(path, driver="match_vs_uci", settings=settings) as writer:
        writer.write_game({"game_index": index, "a_is_white": index == 0,
                           "opening_index": 0, "opening_fen": openings[0].fen(),
                           "result": result, "score_a": score, "termination": "clock"})
    return path, settings, openings


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("result", ["1-0", "0-1", "1/2-1/2"])
def test_resume_keeps_correct_finished_result_for_both_colors(tmp_path, index, result):
    score = match._score_for_a(result, a_is_white=index == 0)
    path, settings, openings = _log(tmp_path, index, result, score)
    rows = match.load_match_resume(path, settings=settings, openings=openings).rows
    assert rows[index]["score_a"] == score


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(("result", "score"), [
    ("1/2-1/2", 1.0), ("not-a-result", 0.5), ("*", 0.5), (None, 0.5),
    ("1/2-1/2", float("nan")), ("1/2-1/2", float("inf")),
    ("1/2-1/2", 0.25), ("1/2-1/2", "0.5"), ("1/2-1/2", True),
    ("1/2-1/2", None),
])
def test_resume_refuses_inconsistent_or_unfinished_result(tmp_path, index, result, score):
    path, settings, openings = _log(tmp_path, index, result, score)
    with pytest.raises(SystemExit, match="result/score"):
        match.load_match_resume(path, settings=settings, openings=openings)
