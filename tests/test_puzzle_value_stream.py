"""Value evaluation keeps full-suite semantics while bounding encoded batches."""
from __future__ import annotations

import weakref
from typing import Any

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.eval import puzzles


class ValueModel(torch.nn.Module):
    def __init__(self, tuple_output: bool = False) -> None:
        super().__init__()
        self.tuple_output = tuple_output
        self.batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor) -> Any:
        self.batch_sizes.append(len(x))
        return (None, x) if self.tuple_output else {"wdl": x}


def _fixture() -> tuple[puzzles.PuzzleSuite, dict[str, np.ndarray]]:
    boards = [chess.Board(), chess.Board()]
    boards[1].push_uci("e2e4")
    entries = []
    logits = {}
    for idx, board in enumerate(boards):
        legal = list(board.legal_moves)
        entries.append(puzzles.Puzzle(board, [legal[idx]], rating=1200 + 500 * idx))
        for move_idx, move in enumerate(legal):
            board.push(move)
            logits[board.fen()] = np.array(
                [move_idx % 5 - 2, move_idx % 3 - 1, 2 - move_idx % 7], dtype=np.float32,
            )
            board.pop()
    terminal = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    entries.append(puzzles.Puzzle(terminal, [], rating=2200))
    return puzzles.PuzzleSuite(entries), logits


def _reference(suite: puzzles.PuzzleSuite, logits: dict[str, np.ndarray]) -> puzzles.PuzzleResult:
    flags = []
    for puzzle in suite.puzzles:
        legal = list(puzzle.board.legal_moves)
        rows = []
        for move in legal:
            puzzle.board.push(move)
            rows.append(logits[puzzle.board.fen()])
            puzzle.board.pop()
        if not rows:
            flags.append(False)
            continue
        probabilities = torch.softmax(torch.from_numpy(np.stack(rows)), dim=-1).numpy()
        chosen = legal[int(np.argmax(probabilities[:, 2] - probabilities[:, 0]))]
        flags.append(chosen in puzzle.best_moves)
    return puzzles.PuzzleResult(
        len(suite), sum(flags), sum(flags) / max(1, len(suite)),
        puzzles._by_rating_table(flags, suite, puzzles.DEFAULT_RATING_BUCKETS),
    )


@pytest.mark.parametrize("batch_size", [1, 7, 1000])
@pytest.mark.parametrize("tuple_output", [False, True])
@pytest.mark.parametrize("score_case", ["varied", "tied", "nan"])
def test_value_stream_preserves_full_suite_results(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, tuple_output: bool, score_case: str,
) -> None:
    suite, logits = _fixture()
    if score_case != "varied":
        for row in logits.values():
            row[:] = 0
    if score_case == "nan":
        list(logits.values())[8][:] = np.nan
    monkeypatch.setattr(puzzles, "encode_position_for_model", lambda model, board: logits[board.fen()].copy())
    before = [(p.board.fen(), list(p.board.move_stack)) for p in suite.puzzles]
    expected = _reference(suite, logits)
    model = ValueModel(tuple_output)
    assert puzzles.run_value_head_puzzle_eval(model, suite, device="cpu", batch_size=batch_size) == expected
    assert [(p.board.fen(), list(p.board.move_stack)) for p in suite.puzzles] == before
    count = len(logits)
    assert model.batch_sizes == [min(batch_size, count - n) for n in range(0, count, batch_size)]


def test_pending_encodings_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    suite, logits = _fixture()
    references: list[weakref.ReferenceType[np.ndarray]] = []
    bound = 7

    def encode(_model: Any, board: chess.Board) -> np.ndarray:
        assert sum(ref() is not None for ref in references) < bound
        result = logits[board.fen()].copy()
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(puzzles, "encode_position_for_model", encode)
    puzzles.run_value_head_puzzle_eval(ValueModel(), suite, device="cpu", batch_size=bound)
    assert all(ref() is None for ref in references)


def test_encoding_failure_restores_board(monkeypatch: pytest.MonkeyPatch) -> None:
    suite, _ = _fixture()
    before = [(p.board.fen(), list(p.board.move_stack)) for p in suite.puzzles]

    def fail(*_args: Any) -> np.ndarray:
        raise RuntimeError("encoding fixture failure")

    monkeypatch.setattr(puzzles, "encode_position_for_model", fail)
    with pytest.raises(RuntimeError, match="encoding fixture failure"):
        puzzles.run_value_head_puzzle_eval(ValueModel(), suite, device="cpu", batch_size=7)
    assert [(p.board.fen(), list(p.board.move_stack)) for p in suite.puzzles] == before


@pytest.mark.parametrize("batch_size", [0, -1])
def test_nonpositive_batch_is_rejected_before_model_mutation(batch_size: int) -> None:
    model = ValueModel()
    with pytest.raises(ValueError, match="batch_size must be positive"):
        puzzles.run_value_head_puzzle_eval(model, puzzles.PuzzleSuite(), device="cpu", batch_size=batch_size)
    assert model.training
    assert model.batch_sizes == []


def test_empty_suite_does_not_forward() -> None:
    model = ValueModel()
    result = puzzles.run_value_head_puzzle_eval(model, puzzles.PuzzleSuite(), device="cpu", batch_size=7)
    assert result == puzzles.PuzzleResult(total=0, correct=0, accuracy=0.0)
    assert model.batch_sizes == []


def test_default_cli_passes_full_suite_and_batch_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from scripts import eval_puzzles as cli

    suite, logits = _fixture()
    model = ValueModel()
    loaded: list[dict[str, Any]] = []

    def load(_path: str, **kwargs: Any) -> puzzles.PuzzleSuite:
        loaded.append(kwargs)
        return suite

    monkeypatch.setattr(cli, "load_model_from_checkpoint", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(cli, "load_lichess_csv", load)
    monkeypatch.setattr(puzzles, "encode_position_for_model", lambda model, board: logits[board.fen()].copy())
    calls: list[str] = []

    def policy(_model: Any, actual_suite: puzzles.PuzzleSuite, **kwargs: Any) -> puzzles.PuzzleResult:
        assert actual_suite is suite
        assert kwargs["batch_size"] == 7
        calls.append("policy")
        return _reference(suite, logits)

    def value(actual_model: torch.nn.Module, actual_suite: puzzles.PuzzleSuite, **kwargs: Any) -> puzzles.PuzzleResult:
        assert actual_suite is suite
        assert kwargs["batch_size"] == 7
        assert kwargs["device"] == "cpu"
        calls.append("value")
        return puzzles.run_value_head_puzzle_eval(actual_model, actual_suite, **kwargs)

    monkeypatch.setattr(cli, "run_policy_sequence_eval", policy)
    monkeypatch.setattr(cli, "run_value_head_puzzle_eval", value)
    monkeypatch.setattr(sys, "argv", ["eval_puzzles", "--checkpoint", "fixture", "--device", "cpu", "--batch-size", "7", "--log-csv", ""])
    cli.main()
    assert calls == ["policy", "value"]
    assert loaded[0]["max_puzzles"] is None
    assert sum(model.batch_sizes) == len(logits)
    assert max(model.batch_sizes) <= 7
