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
    if not bb:
        dst.fill(0.0)
        return dst
    # Unpack 64-bit int to 64 binary values (MSB first from big-endian bytes).
    # After reshape(8,8): row 0 = bits 63..56 (rank 7, files reversed).
    # WHITE: flip rows+cols → plane[rank][file]. BLACK: flip cols only (ranks pre-flipped).
    raw = np.unpackbits(np.frombuffer(int(bb).to_bytes(8, 'big'), dtype=np.uint8)).reshape(8, 8)
    if turn == chess.WHITE:
        dst[:] = raw[::-1, ::-1]
    else:
        dst[:] = raw[:, ::-1]
    return dst


def bitboard_to_plane(bb: int, *, turn: chess.Color) -> np.ndarray:
    """Convert python-chess bitboard mask to an oriented 8x8 float32 plane."""
    if not bb:
        return np.zeros((8, 8), dtype=np.float32)
    raw = np.unpackbits(np.frombuffer(int(bb).to_bytes(8, 'big'), dtype=np.uint8)).reshape(8, 8)
    if turn == chess.WHITE:
        return raw[::-1, ::-1].astype(np.float32)
    return raw[:, ::-1].astype(np.float32)


def bitboards_to_planes(bbs: list[int], *, turn: chess.Color) -> np.ndarray:
    """Convert multiple bitboards to oriented (N, 8, 8) float32 planes in one batch."""
    n = len(bbs)
    if n == 0:
        return np.zeros((0, 8, 8), dtype=np.float32)
    raw_bytes = b''.join(int(bb).to_bytes(8, 'big') for bb in bbs)
    raw = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8)).reshape(n, 8, 8)
    if turn == chess.WHITE:
        return raw[:, ::-1, ::-1].astype(np.float32)
    return raw[:, :, ::-1].astype(np.float32)


def file_to_plane(file_idx: int) -> np.ndarray:
    plane = np.zeros((8, 8), dtype=np.float32)
    plane[:, file_idx] = 1.0
    return plane
