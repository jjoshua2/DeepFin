from __future__ import annotations

import numpy as np
import chess


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


def bitboard_to_plane(bb: int, *, turn: chess.Color) -> np.ndarray:
    """Convert python-chess bitboard mask to an oriented 8x8 float32 plane."""
    plane = np.zeros((8, 8), dtype=np.float32)
    if not bb:
        return plane
    for sq in chess.scan_reversed(bb):
        osq = orient_square(sq, turn)
        plane[chess.square_rank(osq), chess.square_file(osq)] = 1.0
    return plane


def file_to_plane(file_idx: int) -> np.ndarray:
    plane = np.zeros((8, 8), dtype=np.float32)
    plane[:, file_idx] = 1.0
    return plane
