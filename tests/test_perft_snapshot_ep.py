"""Direct native perft API validation of raw en-passant snapshots."""
from __future__ import annotations

import chess
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding._perft_ext import perft, perft_divide


def _raw_cboard(board: chess.Board) -> CBoard:
    return CBoard.from_raw(
        int(board.pawns), int(board.knights), int(board.bishops),
        int(board.rooks), int(board.queens), int(board.kings),
        int(board.occupied_co[chess.WHITE]), int(board.occupied_co[chess.BLACK]),
        int(board.turn), int(board.castling_rights),
        -1 if board.ep_square is None else int(board.ep_square),
        board.halfmove_clock,
    )


@pytest.mark.parametrize("fen", [
    "7k/8/3p4/8/8/8/8/K7 w - d5 0 1",
    "7k/8/8/8/8/3P4/8/K7 b - d4 0 1",
    "7k/8/3N4/3p4/8/8/8/K7 w - d6 0 1",
    "7k/8/8/8/3P4/3n4/8/K7 b - d3 0 1",
    "7k/8/8/4P3/8/8/8/K7 w - d6 0 1",
    "7k/8/8/8/4p3/8/8/K7 b - d3 0 1",
    "7k/3n4/8/3p4/8/8/8/K7 w - d6 0 1",
    "7k/8/8/8/3P4/8/3n4/K7 b - d3 0 1",
])
def test_native_perft_rejects_malformed_ep_snapshot(fen: str) -> None:
    board = chess.Board(fen)
    assert board.status() & chess.STATUS_INVALID_EP_SQUARE
    native = _raw_cboard(board)
    with pytest.raises(ValueError, match="en-passant snapshot"):
        perft(native, 1)
    with pytest.raises(ValueError, match="en-passant snapshot"):
        perft_divide(native, 1)


@pytest.mark.parametrize("fen", [
    "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
    "7k/8/8/3p4/8/8/8/K7 w - d6 0 1",
    "7k/8/8/8/3Pp3/8/8/K7 b - d3 0 1",
    "7k/8/8/8/3P4/8/8/K7 b - d3 0 1",
])
def test_native_perft_accepts_valid_ep_with_or_without_capturer(fen: str) -> None:
    board = chess.Board(fen)
    assert board.is_valid()
    native = _raw_cboard(board)
    expected = {move.uci(): 1 for move in board.legal_moves}
    assert perft(native, 1) == len(expected)
    assert perft_divide(native, 1) == expected
