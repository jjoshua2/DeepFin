"""Cheap tactical root checks shared by search frontends."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import chess
import numpy as np

from chess_anti_engine.moves import POLICY_SIZE, move_to_index

if TYPE_CHECKING:
    from chess_anti_engine.encoding.cboard_encode import CBoard


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
        if not board.gives_check(move):
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


def immediate_terminal_draw_indices(
    board: chess.Board,
    *,
    allowed_root_indices: set[int] | None = None,
) -> set[int]:
    """Return legal root moves that immediately end the game as a draw."""
    if board.is_game_over():
        return set()
    out: set[int] = set()
    for move in board.legal_moves:
        action = int(move_to_index(move, board))
        if allowed_root_indices is not None and action not in allowed_root_indices:
            continue
        child = board.copy(stack=True)
        child.push(move)
        if child.is_game_over(claim_draw=True) and not child.is_checkmate():
            out.add(action)
    return out


def immediate_terminal_cboard_policy_or_draws(
    root_cb: CBoard,
    legal_idx: Iterable[int],
    *,
    detect_draws: bool = True,
) -> tuple[tuple[np.ndarray, int, float] | None, set[int]]:
    """Return an immediate mate policy, or terminal-draw actions when no mate exists.

    ``detect_draws=False`` computes only the mate and returns an empty draw set,
    skipping the (potentially reply-claim) draw scan. Callers that won't use the
    draws — a root that isn't winning, where draw terminals are never pruned —
    pass this to avoid the wasted work.
    """
    legal_arr = (
        np.asarray(legal_idx, dtype=np.int32)
        if isinstance(legal_idx, np.ndarray)
        else np.fromiter((int(action) for action in legal_idx), dtype=np.int32)
    )
    if hasattr(root_cb, "root_terminal_actions"):
        mate_action, draw_actions = root_cb.root_terminal_actions(legal_arr, detect_draws)
        if int(mate_action) >= 0:
            return one_hot_root_policy(int(mate_action)), set()
        return None, set(np.asarray(draw_actions, dtype=np.int32).astype(int).tolist())

    draws: set[int] = set()
    for action in legal_arr:
        child = root_cb.copy()
        action_i = int(action)
        child.push_index(action_i)
        if child.is_checkmate():
            return one_hot_root_policy(action_i), set()
        if detect_draws and child.is_game_over():
            draws.add(action_i)
    return None, draws
