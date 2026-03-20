"""Encoding helpers for CBoard objects (C-accelerated chess boards)."""
from __future__ import annotations

import numpy as np

try:
    from chess_anti_engine.encoding._lc0_ext import CBoard
    _HAS_CBOARD = True
except ImportError:
    _HAS_CBOARD = False

try:
    from chess_anti_engine.encoding._features_ext import compute_extra_features as _c_compute
    _HAS_FEATURES_C = True
except ImportError:
    _HAS_FEATURES_C = False


def encode_cboard(cb: CBoard) -> np.ndarray:
    """Encode a CBoard into (146, 8, 8) float32.

    Prefer the fused CBoard.encode_146() path when available. Fall back to
    CBoard.encode_planes() + the existing features extension otherwise.
    """
    encode_146 = getattr(cb, "encode_146", None)
    if encode_146 is not None:
        return encode_146()

    # LC0 112 planes — all in C
    lc0 = cb.encode_planes()  # (112, 8, 8) float32

    if not _HAS_FEATURES_C:
        # No features C ext — return LC0 only with zero features
        out = np.zeros((146, 8, 8), dtype=np.float32)
        out[:112] = lc0
        return out

    # Extract bitboard values for features extension
    is_white = cb.turn  # True = WHITE
    occ_w = cb.occ_white
    occ_b = cb.occ_black

    if is_white:
        us_occ, them_occ = occ_w, occ_b
    else:
        us_occ, them_occ = occ_b, occ_w

    pieces_us = np.array([
        cb.pawns & us_occ, cb.knights & us_occ, cb.bishops & us_occ,
        cb.rooks & us_occ, cb.queens & us_occ, cb.kings & us_occ,
    ], dtype=np.uint64)
    pieces_them = np.array([
        cb.pawns & them_occ, cb.knights & them_occ, cb.bishops & them_occ,
        cb.rooks & them_occ, cb.queens & them_occ, cb.kings & them_occ,
    ], dtype=np.uint64)

    occupied = int(occ_w | occ_b)
    # King squares
    us_kings = cb.kings & us_occ
    them_kings = cb.kings & them_occ
    king_sq_us = int(us_kings).bit_length() - 1 if us_kings else -1
    king_sq_them = int(them_kings).bit_length() - 1 if them_kings else -1
    ep_square = -1 if cb.ep_square is None else cb.ep_square

    feat = _c_compute(pieces_us, pieces_them, occupied,
                      king_sq_us, king_sq_them, is_white, ep_square)

    return np.concatenate([lc0, feat], axis=0)
