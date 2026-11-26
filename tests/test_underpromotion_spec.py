"""Underpromotion policy ids against a spec written out here, not imported.

``UNDERPROMO_TO_IDX`` / ``UNDERPROMO_PIECES`` / ``_UNDERPROMO_CHAR_TO_IDX`` are
three tables in ``moves/encode.py`` that only have to agree with EACH OTHER for
every round-trip test to pass. Measured 2026-11-26 (test audit): swapping
knight and bishop consistently in all three failed one test in the suite, and
that one only through a C-versus-Python game replay. The order is a wire
format (stored shards, the C board, the compact 1858 table), so it is pinned
here by the layout itself: plane ``64 + 3 * piece + direction`` with pieces
knight, bishop, rook and directions capture-left, push, capture-right, from
the mover's point of view.
"""
from __future__ import annotations

import chess
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.moves.encode import (
    index_to_move_for_encoding,
    index_to_move_strict,
    move_to_index,
    move_to_index_for_encoding,
)

_PLANES = 73
_PIECE_SLOT = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
_CASES = [
    (colour, file, piece, df)
    for colour in (chess.WHITE, chess.BLACK)
    for file in range(8)
    for piece in _PIECE_SLOT
    for df in (-1, 0, 1)
    if 0 <= file + df <= 7
]


def _board(colour: chess.Color, file: int) -> chess.Board:
    """White pawn on the 7th with a capturable knight on each neighbouring
    file; kings on a4 / h5 (off both back ranks). Black is the vertical mirror."""
    board = chess.Board(None)
    board.set_piece_at(chess.A4, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H5, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.square(file, 6), chess.Piece(chess.PAWN, chess.WHITE))
    for df in (-1, 1):
        if 0 <= file + df <= 7:
            board.set_piece_at(chess.square(file + df, 7), chess.Piece(chess.KNIGHT, chess.BLACK))
    board.turn = chess.WHITE
    if colour == chess.BLACK:
        board = board.mirror()
    assert board.is_valid(), board.fen()
    return board


@pytest.mark.parametrize(("colour", "file", "piece", "df"), _CASES)
def test_underpromotion_index_matches_the_written_out_layout(
        colour: chess.Color, file: int, piece: chess.PieceType, df: int) -> None:
    board = _board(colour, file)
    from_rank, to_rank = (6, 7) if colour == chess.WHITE else (1, 0)
    move = chess.Move(chess.square(file, from_rank), chess.square(file + df, to_rank), promotion=piece)
    assert move in board.legal_moves, (board.fen(), move.uci())

    # The mover's view: a black pawn on rank 2 is "on the 7th" (rank index 6),
    # and files are NOT flipped, so capture-left is toward the a-file for both.
    oriented_from = chess.square(file, 6)
    expected = oriented_from * _PLANES + 64 + 3 * _PIECE_SLOT[piece] + (df + 1)

    assert move_to_index(move, board) == expected
    assert index_to_move_strict(expected, board) == move
    assert expected in set(CBoard.from_board(board).legal_move_indices().tolist())
    compact = move_to_index_for_encoding(move, board, policy_encoding="lc0_1858")
    assert index_to_move_for_encoding(compact, board, policy_encoding="lc0_1858") == move


def test_underpromotion_ids_are_distinct_from_each_other_and_from_the_queen_promotion() -> None:
    for colour in (chess.WHITE, chess.BLACK):
        board = _board(colour, 3)
        promotions = [m for m in board.legal_moves if m.promotion is not None]
        assert len(promotions) == 12  # 3 destinations x {q, r, b, n}
        assert len({move_to_index(m, board) for m in promotions}) == 12
        assert len({move_to_index_for_encoding(m, board, policy_encoding="lc0_1858") for m in promotions}) == 12
