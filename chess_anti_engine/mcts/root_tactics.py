"""Cheap tactical root checks shared by search frontends."""
from __future__ import annotations

from collections.abc import Iterable

import chess
import numpy as np

from chess_anti_engine.encoding.cboard_encode import CBoard
from chess_anti_engine.moves import POLICY_SIZE, move_to_index


def one_hot_root_policy(action: int, *, value: float = 1.0) -> tuple[np.ndarray, int, float]:
    probs = np.zeros((POLICY_SIZE,), dtype=np.float32)
    action_i = int(action)
    probs[action_i] = 1.0
    return probs, action_i, float(value)


def immediate_mate_move(
    board: chess.Board,
    *,
    allowed_root_indices: set[int] | None = None,
) -> tuple[chess.Move, int] | None:
    """Return a legal mate-in-1 and its policy index, if one exists."""
    if board.is_game_over():
        return None
    for move in board.legal_moves:
        action = int(move_to_index(move, board))
        if allowed_root_indices is not None and action not in allowed_root_indices:
            continue
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate():
            return move, action
    return None


def immediate_mate_root_policy(
    board: chess.Board,
    *,
    allowed_root_indices: set[int] | None = None,
) -> tuple[np.ndarray, int, float] | None:
    mate = immediate_mate_move(board, allowed_root_indices=allowed_root_indices)
    if mate is None:
        return None
    return one_hot_root_policy(mate[1])


def immediate_mate_cboard_policy(
    root_cb: CBoard,
    legal_idx: Iterable[int],
) -> tuple[np.ndarray, int, float] | None:
    for action in legal_idx:
        child = root_cb.copy()
        action_i = int(action)
        child.push_index(action_i)
        if child.is_checkmate():
            return one_hot_root_policy(action_i)
    return None
