"""History-owned automatic draw decisions, separate from neural predictions.

Only automatic endings are terminal here. Threefold and fifty-move claims are
choices; treating them as forced draws would remove winning continuations.
python-chess's insufficient-material test is not an exhaustive dead-position solver.
"""
from __future__ import annotations

from collections.abc import Sequence

import chess

from . import run_probe as sessions
from .root_protocol import board_position, packed_move
from ..legal_probe import run_probe as rules

DRAW_STATUS = 3
AUTOMATIC_DRAWS = frozenset((chess.Termination.INSUFFICIENT_MATERIAL,
                           chess.Termination.SEVENTYFIVE_MOVES,
                           chess.Termination.FIVEFOLD_REPETITION))


def automatic_draw(board: chess.Board) -> str | None:
    """Check the exact history; mate/stalemate remain native Bend decisions."""
    if board.chess960 or not board.is_valid():
        raise ValueError('draw adjudication requires valid orthodox chess')
    result = board.outcome(claim_draw=False)
    if result is not None and result.termination in AUTOMATIC_DRAWS:
        return result.termination.name.lower()
    return None


def reconstruct_leaf(root: chess.Board, path: Sequence[int], supplied: rules.Position,
                     actions: Sequence[int]) -> chess.Board:
    """Validate an entire request before allowing it to bypass model evaluation.

    Copy pre-root history, apply exact (including flags) legal keys, check the
    whole resulting board and complete legal set. No board-only cache is valid.
    The input root is never mutated, including on rejection.
    """
    if root.chess960 or not root.is_valid() or len(path) > 32:
        raise ValueError('invalid draw history root or path length')
    board = root.copy(stack=True)
    for key in path:
        if automatic_draw(board) is not None:
            raise ValueError('search path continues after an automatic draw')
        if type(key) is not int or not 0 <= key < 1 << 17:
            raise ValueError('invalid packed history key')
        move = next((m for m in board.legal_moves if packed_move(board, m) == key), None)
        if move is None:
            raise ValueError('illegal exact history move')
        board.push(move)
    if board_position(board) != supplied:
        raise ValueError('draw leaf board does not match history')
    if (not 1 <= len(actions) <= 256
            or any(type(k) is not int or not 0 <= k < 1 << 17 for k in actions)
            or len(set(actions)) != len(actions)):
        raise ValueError('invalid or duplicate draw request actions')
    if set(actions) != {packed_move(board, m) for m in board.legal_moves}:
        raise ValueError('incomplete or illegal draw request actions')
    return board


def draw_reply(epoch: int, request: int, node: int) -> str:
    if (any(type(x) is not int for x in (epoch, request, node))
            or not 1 <= epoch <= sessions.SENTINEL
            or not 1 <= request <= sessions.SENTINEL or not 0 <= node < 4096):
        raise ValueError('invalid draw reply identity')
    # Canonical terminal payload: zero policy entries, exact W/D/L = 0/1/0.
    return f'reply {epoch:x} {request:x} {node:x} {DRAW_STATUS:x} 0 3f800000 0 0\n'
