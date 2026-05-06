from __future__ import annotations

import chess
import torch

from chess_anti_engine.eval.puzzles import Puzzle, PuzzleSuite, run_policy_sequence_eval
from chess_anti_engine.moves import move_to_index


class _SingleMovePolicy(torch.nn.Module):
    def __init__(self, *, board: chess.Board, move: chess.Move) -> None:
        super().__init__()
        self.move_idx = move_to_index(move, board)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        policy = torch.zeros((x.shape[0], 4672), dtype=torch.float32)
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
