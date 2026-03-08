from __future__ import annotations

import numpy as np
import chess

_ORIENT_FLAT_IDX = {
    chess.WHITE: tuple(range(64)),
    chess.BLACK: tuple((7 - chess.square_rank(sq)) * 8 + chess.square_file(sq) for sq in chess.SQUARES),
}


def orient_square(sq: chess.Square, turn: chess.Color) -> chess.Square:
    """Map a square into side-to-move perspective.

    Convention:
    - If it's White to move: identity.
    - If it's Black to move: flip ranks (mirror vertically).

    This matches the common LC0 convention that the board is viewed from the
    side-to-move's perspective.
    """
    if turn == chess.WHITE:
        return sq
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    return chess.square(f, 7 - r)


def write_bitboard_to_plane(dst: np.ndarray, bb: int, *, turn: chess.Color) -> np.ndarray:
    """Write an oriented bitboard mask into an existing (8,8) float32 plane."""
    flat = dst.reshape(-1)
    flat.fill(0.0)
    if not bb:
        return dst
    orient_idx = _ORIENT_FLAT_IDX[turn]
    while bb:
        lsb = bb & -bb
        sq = lsb.bit_length() - 1
        flat[orient_idx[sq]] = 1.0
        bb ^= lsb
    return dst


def bitboard_to_plane(bb: int, *, turn: chess.Color) -> np.ndarray:
    """Convert python-chess bitboard mask to an oriented 8x8 float32 plane."""
    plane = np.zeros((8, 8), dtype=np.float32)
    return write_bitboard_to_plane(plane, bb, turn=turn)


def file_to_plane(file_idx: int) -> np.ndarray:
    plane = np.zeros((8, 8), dtype=np.float32)
    plane[:, file_idx] = 1.0
    return plane
