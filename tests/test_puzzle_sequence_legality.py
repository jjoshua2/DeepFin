"""Complete continuation validation; tiny fixtures and fake policy only."""
from __future__ import annotations

import csv
from pathlib import Path

import chess
import pytest
import torch

from chess_anti_engine.eval import puzzles as p
from chess_anti_engine.moves.encode import MODEL_POLICY_SIZE, move_to_index_for_encoding


class _Policy(torch.nn.Module):
    def __init__(self, boards: list[chess.Board], moves: list[chess.Move]) -> None:
        super().__init__()
        self.input_history_encoding = "lc0_root_legacy_meta"
        self.input_extra_features = "v2_threats"
        self.indices = [move_to_index_for_encoding(m, b) for b, m in zip(boards, moves, strict=True)]
        self.calls = 0

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        policy = torch.zeros((len(x), MODEL_POLICY_SIZE))
        policy[:, self.indices[self.calls]] = 100
        self.calls += 1
        return {"policy_own": policy}


def _csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["PuzzleId", "FEN", "Moves", "Rating"])
        writer.writeheader()
        for name, moves in rows:
            writer.writerow({"PuzzleId": name, "FEN": chess.STARTING_FEN, "Moves": moves, "Rating": 1200})


@pytest.mark.parametrize("tail", ["g1f3 b8b6", "0000", "e7e6", "bad-token"])
def test_invalid_csv_continuation_is_skipped_not_shortened(tmp_path: Path, caplog, tail: str) -> None:
    path = tmp_path / "tiny.csv"
    _csv(path, [("invalid", "e2e4 e7e5 " + tail), ("valid", "e2e4 e7e5")])
    suite = p.load_lichess_csv(path, max_puzzles=1)
    assert [x.puzzle_id for x in suite.puzzles] == ["valid"]
    assert "invalid solution sequence" in caplog.text
    assert "invalid" in caplog.text
    puzzle = suite.puzzles[0]
    model = _Policy([puzzle.board], puzzle.best_moves)
    result = p.run_policy_sequence_eval(model, suite, device="cpu")
    assert (result.total, result.correct, result.accuracy) == (1, 1, 1.0)
    assert result.by_rating == [(1000, 1500, 1, 1, 1.0)]


@pytest.mark.parametrize("moves", ["e2e4 e7e4", "e2e4 0000", "e2e4 g1f3", "e2e4 e7e5 e1e8"])
def test_programmatic_illegal_sequence_refused_before_model_mutation(moves: str) -> None:
    board = chess.Board()
    before = board.fen(), list(board.move_stack)
    sequence = tuple(chess.Move.from_uci(x) for x in moves.split())
    suite = p.PuzzleSuite([p.Puzzle(board, [sequence[0]], puzzle_id="bad", solution_sequence=sequence)])

    class Untouched(torch.nn.Module):
        def eval(self):
            pytest.fail("model mutated before sequence validation")

    with pytest.raises(ValueError, match="puzzle 'bad': illegal solution move"):
        p.run_policy_sequence_eval(Untouched(), suite, device="cpu")
    assert (board.fen(), board.move_stack) == before


@pytest.mark.parametrize(("fen", "uci"), [
    (chess.STARTING_FEN, "e2e4 e7e5 g1f3"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1 e8c8 f1e1"),
    ("7k/P7/8/8/8/8/8/6K1 w - - 0 1", "a7a8q h8h7 g1f2"),
])
def test_legal_full_sequence_castling_and_promotion_preserved(fen: str, uci: str) -> None:
    board = chess.Board(fen)
    original = board.fen(), list(board.move_stack)
    sequence = tuple(chess.Move.from_uci(x) for x in uci.split())
    position = board.copy()
    boards, moves = [], []
    for step, move in enumerate(sequence):
        if step % 2 == 0:
            boards.append(position.copy())
            moves.append(move)
        position.push(move)
    model = _Policy(boards, moves)
    suite = p.PuzzleSuite([p.Puzzle(board, [sequence[0]], solution_sequence=sequence)])
    result = p.run_policy_sequence_eval(model, suite, device="cpu")
    assert (result.total, result.correct) == (1, 1)
    assert model.calls == len(moves)
    assert (board.fen(), board.move_stack) == original


def test_final_alternative_checkmate_remains_accepted() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    expected, alternative = chess.Move.from_uci("f7f8"), chess.Move.from_uci("f7g7")
    suite = p.PuzzleSuite([p.Puzzle(board, [expected], solution_sequence=(expected,))])
    result = p.run_policy_sequence_eval(_Policy([board], [alternative]), suite, device="cpu")
    assert (result.total, result.correct) == (1, 1)


def test_continuation_after_checkmate_is_refused() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    sequence = tuple(chess.Move.from_uci(x) for x in ["f7f8", "h8h7"])
    suite = p.PuzzleSuite([p.Puzzle(board, [sequence[0]], solution_sequence=sequence)])
    with pytest.raises(ValueError, match="illegal solution move h8h7 at step 2"):
        p.run_policy_sequence_eval(_Policy([board], [sequence[0]]), suite, device="cpu")


def test_csv_retains_complete_valid_continuation_and_post_setup_board(tmp_path: Path) -> None:
    path = tmp_path / "valid.csv"
    _csv(path, [("valid", "e2e4 e7e5 g1f3 b8c6")])
    suite = p.load_lichess_csv(path)
    puzzle = suite.puzzles[0]
    expected = chess.Board()
    expected.push_uci("e2e4")
    assert puzzle.board == expected
    assert [m.uci() for m in puzzle.solution_sequence] == ["e7e5", "g1f3", "b8c6"]
    assert [m.uci() for m in puzzle.best_moves] == ["e7e5"]
