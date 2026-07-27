from __future__ import annotations

import chess
import pytest
import torch

from chess_anti_engine.eval.puzzles import Puzzle, PuzzleSuite, run_policy_sequence_eval
from chess_anti_engine.moves import move_to_index
from chess_anti_engine.moves.encode import MODEL_POLICY_SIZE, compact_policy_index


class _SingleMovePolicy(torch.nn.Module):
    """Stub net emitting compact lc0_1858 logits, like every real model.

    It also declares its input encoding, because puzzle eval encodes for the
    model and refuses to guess (docs/rl_loop_audit.md M11).
    """

    def __init__(self, *, board: chess.Board, move: chess.Move) -> None:
        super().__init__()
        self.move_idx = compact_policy_index(move_to_index(move, board))
        self.input_history_encoding = "lc0_root_legacy_meta"
        self.input_extra_features = "v2_threats"

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        policy = torch.zeros((x.shape[0], MODEL_POLICY_SIZE), dtype=torch.float32)
        policy[:, self.move_idx] = 100.0
        return {"policy_own": policy}


def test_policy_sequence_eval_scores_epd_best_move_puzzles() -> None:
    board = chess.Board()
    best = chess.Move.from_uci("e2e4")
    suite = PuzzleSuite(
        puzzles=[Puzzle(board=board, best_moves=[best])],
        name="epd",
    )
    model = _SingleMovePolicy(board=board, move=best)

    result = run_policy_sequence_eval(model, suite, device="cpu")

    assert result.total == 1
    assert result.correct == 1
    assert result.accuracy == 1.0


def test_policy_sequence_eval_refuses_a_model_with_no_declared_encoding() -> None:
    """Silently defaulting to legacy is what M11 was; a stub must not sneak past."""
    board = chess.Board()
    best = chess.Move.from_uci("e2e4")
    model = _SingleMovePolicy(board=board, move=best)
    del model.input_history_encoding
    suite = PuzzleSuite(puzzles=[Puzzle(board=board, best_moves=[best])], name="epd")

    with pytest.raises(ValueError, match="input_history_encoding"):
        run_policy_sequence_eval(model, suite, device="cpu")
