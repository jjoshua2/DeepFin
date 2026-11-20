"""Cheap tactical root checks shared by search frontends."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import chess
import numpy as np

from chess_anti_engine.ext_check import refuse_fallback
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


def _ends_drawn(board: chess.Board) -> bool:
    """The game is over, drawn, on ``board``: stalemate, insufficient
    material, the 50-move clock at 100 or more, or a third occurrence."""
    return not board.is_checkmate() and (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.halfmove_clock >= 100
        or board.is_repetition(3)
    )


def _every_reply_ends_drawn(board: chess.Board) -> bool:
    """The 50-move half of the reply-forced draw: with the clock at 99, every
    legal reply ends the game drawn (each reversible reply reaches 100; a pawn
    move or capture must draw on the board some other way) and none mates.
    Only the 50-move rule is covered: a position where every reply is a third
    occurrence but the clock is not at 99 is not examined (the 99 gate is what
    keeps this off every other root)."""
    if board.halfmove_clock != 99:
        return False
    replies = list(board.legal_moves)
    if not replies:
        return False
    for reply in replies:
        board.push(reply)
        try:
            if not _ends_drawn(board):
                return False
        finally:
            board.pop()
    return True


def forced_draw_actions(board: chess.Board, actions: Iterable[int]) -> list[int]:
    """The root ``actions`` (policy indices) whose move forces a draw on the
    board: it ends the game drawn (stalemate, insufficient material, the
    50-move clock at 100 or more, a third occurrence), or it leaves the clock
    at 99 with every opponent reply ending the game drawn and none mating (an
    opponent with only reversible moves cannot decline the 50-move draw). A
    move after which the opponent merely MAY claim a draw on its reply is not
    one: an opponent with a pawn move, a capture or a mate can decline and play
    on. Returned in ``board.legal_moves`` order. The clock-99 rule (added
    2026-11-19, outside-model review lead R1) only changes roots where a move
    leaves the fifty-move clock at 99."""
    wanted = {int(a) for a in actions}
    out: list[int] = []
    for move in board.legal_moves:
        action = int(move_to_index(move, board))
        if action not in wanted:
            continue
        board.push(move)
        try:
            if _ends_drawn(board) or (
                not board.is_checkmate() and _every_reply_ends_drawn(board)
            ):
                out.append(action)
        finally:
            board.pop()
    return out


def _cboard_ends_drawn(cb: CBoard) -> bool:
    return bool(cb.is_game_over()) and not bool(cb.is_checkmate())


def _cboard_every_reply_ends_drawn(child: CBoard) -> bool:
    """`_every_reply_ends_drawn` on a ``CBoard`` (the same 50-move-only rule)."""
    if int(child.halfmove_clock) != 99:
        return False
    replies = child.legal_move_indices()
    if len(replies) == 0:
        return False
    for reply in replies.tolist():
        grandchild = child.copy()
        grandchild.push_index(int(reply))
        if not _cboard_ends_drawn(grandchild):
            return False
    return True


def forced_draw_actions_cboard(root_cb: CBoard, actions: Iterable[int]) -> list[int]:
    """``forced_draw_actions`` on a ``CBoard``, in ``actions`` order.

    ``CBoard.is_game_over`` is exactly the on-board set (50-move clock >= 100,
    a third occurrence, insufficient material, no legal move); a mate is
    excluded. The reply-claim draws ``root_terminal_actions`` also reports are
    not game-over on the child, so they are not returned, except the clock-99
    case where every reply ends the game drawn (`forced_draw_actions`)."""
    out: list[int] = []
    for action in actions:
        child = root_cb.copy()
        child.push_index(int(action))
        if _cboard_ends_drawn(child) or (
            not bool(child.is_checkmate()) and _cboard_every_reply_ends_drawn(child)
        ):
            out.append(int(action))
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

    refuse_fallback(
        "immediate_terminal_cboard_policy_or_draws", "CBoard.root_terminal_actions",
        "the Python per-child terminal scan",
    )
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
