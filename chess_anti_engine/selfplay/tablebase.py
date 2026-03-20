"""Syzygy tablebase rescoring for endgame positions.

Post-hoc: after a game finishes, walk through positions and when one reaches
≤ N pieces (after capture/pawn push, no castling rights), look up the WDL in
Syzygy tables.  The tablebase-proven result is then used to relabel ALL
preceding positions in that game, providing perfect ground-truth for endgame
evaluation at zero inference cost.

Optional policy rescoring: for tablebase-eligible positions, replace the
policy target with 100% weight on the DTZ-optimal move.
"""
from __future__ import annotations

from typing import Optional

import chess
import chess.syzygy


_tablebase: Optional[chess.syzygy.Tablebase] = None
_tablebase_path: Optional[str] = None


def _get_tablebase(path: str) -> chess.syzygy.Tablebase | None:
    """Lazy-load (and cache) the Syzygy tablebase."""
    global _tablebase, _tablebase_path
    if _tablebase is not None and _tablebase_path == path:
        return _tablebase
    try:
        _tablebase = chess.syzygy.open_tablebase(path)
    except (OSError, FileNotFoundError):
        _tablebase = None
    _tablebase_path = path
    return _tablebase


def _eligible(board: chess.Board) -> bool:
    """Check if a position is eligible for tablebase probing.

    Criteria: ≤7 pieces total AND no castling rights.
    """
    return (
        chess.popcount(board.occupied) <= 7
        and not board.has_castling_rights(chess.WHITE)
        and not board.has_castling_rights(chess.BLACK)
    )


def probe_wdl(board: chess.Board, syzygy_path: str) -> int | None:
    """Probe Syzygy WDL for a position.

    Returns:
        0 = loss for side to move, 1 = draw, 2 = win for side to move.
        None if not available.
    """
    if not _eligible(board):
        return None
    try:
        tb = _get_tablebase(syzygy_path)
        if tb is None:
            return None
        wdl = tb.probe_wdl(board)
    except (KeyError, chess.syzygy.MissingTableError):
        return None

    # python-chess WDL: -2 = loss, -1 = blessed loss, 0 = draw,
    #                    1 = cursed win, 2 = win.
    # Map to our 0/1/2 encoding (from side-to-move perspective):
    #   win (1,2) -> 2, draw (0,-1,1 with DTZ) -> 1, loss (-2,-1) -> 0
    # We use a simple mapping: positive = win, 0 = draw, negative = loss.
    if wdl > 0:
        return 2  # win for side to move
    elif wdl < 0:
        return 0  # loss for side to move
    else:
        return 1  # draw


def probe_best_move(board: chess.Board, syzygy_path: str) -> chess.Move | None:
    """Probe Syzygy DTZ to find the best (DTZ-optimal) move.

    Returns the move that minimises DTZ (closest to conversion) among
    moves that preserve the current WDL outcome, or None if the position
    is not eligible or the probe fails.
    """
    if not _eligible(board):
        return None
    try:
        tb = _get_tablebase(syzygy_path)
        if tb is None:
            return None
        root_wdl = tb.probe_wdl(board)
    except (KeyError, chess.syzygy.MissingTableError):
        return None

    best_move: chess.Move | None = None
    best_dtz: int | None = None

    for move in board.legal_moves:
        board.push(move)
        try:
            child_dtz = tb.probe_dtz(board)
        except (KeyError, chess.syzygy.MissingTableError):
            board.pop()
            continue
        board.pop()

        # After our move, it's the opponent's turn, so signs flip.
        # We want to pick the move that leads to the best outcome for us
        # (most negative DTZ from opponent's perspective = fastest win for us,
        #  or least negative = slowest loss).
        #
        # For winning positions (root_wdl > 0): child_dtz should be negative
        #   (opponent is losing).  Pick most negative (abs largest).
        # For drawing positions (root_wdl == 0): child_dtz should be 0.
        # For losing positions (root_wdl < 0): child_dtz should be positive
        #   (opponent is winning).  Pick largest (longest survival).
        #
        # Simplification: negate child_dtz so "higher = better for us",
        # then pick the move with the highest score that preserves WDL.
        neg_dtz = -child_dtz

        # Filter: only consider moves that preserve the WDL category.
        # A winning move must still leave the opponent losing, etc.
        if root_wdl > 0 and child_dtz >= 0:
            continue  # This move doesn't preserve the win
        if root_wdl == 0 and child_dtz != 0:
            continue  # This move doesn't preserve the draw
        if root_wdl < 0 and child_dtz <= 0:
            continue  # Not relevant (we can't improve from a loss)

        if best_dtz is None or neg_dtz > best_dtz:
            best_dtz = neg_dtz
            best_move = move

    return best_move


def rescore_game_samples(
    boards_history: list[chess.Board],
    syzygy_path: str,
) -> str | None:
    """Find the earliest tablebase-eligible position and return the proven game result.

    Walks through the game's board history. When it finds the first position
    eligible for tablebase probing (≤7 pieces, no castling), it looks up the
    WDL and converts it to a game-level result.

    Args:
        boards_history: Board states at each ply (after the move was played).
        syzygy_path: Path to Syzygy tablebase directory.

    Returns:
        The game result string ("1-0", "0-1", "1/2-1/2") from tablebase, or
        None if no position was eligible.
    """
    for ply_idx, board in enumerate(boards_history):
        wdl = probe_wdl(board, syzygy_path)
        if wdl is None:
            continue

        # Convert side-to-move WDL to absolute result.
        if wdl == 1:
            return "1/2-1/2"
        elif wdl == 2:
            # Win for side to move
            if board.turn == chess.WHITE:
                return "1-0"
            else:
                return "0-1"
        else:
            # Loss for side to move
            if board.turn == chess.WHITE:
                return "0-1"
            else:
                return "1-0"

    return None
