"""Encoding helpers for CBoard objects (C-accelerated chess boards)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import chess
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.features import extra_feature_plane_count
from chess_anti_engine.encoding.lc0 import (
    LC0_HISTORY_ROOT_LEGACY_META,
    normalize_lc0_history_encoding,
    uses_lc0_root_history,
)


def apply_lc0_root_legacy_meta_planes_cboard(out: np.ndarray, cb: CBoard) -> None:
    """Patch LC0-root planes 109/110 to legacy normalized rule50 + EP file."""
    out[109, :, :] = min(float(cb.halfmove_clock), 100.0) / 100.0
    out[110, :, :] = 0.0
    if cb.ep_square is not None:
        out[110, :, chess.square_file(int(cb.ep_square))] = 1.0


def cboard_from_board_fast(board: chess.Board) -> CBoard:
    """Fast CBoard construction from python-chess Board via from_raw.

    Extracts board state as raw integers on the Python side, then passes
    them to CBoard.from_raw() which skips all Python attribute access in C.
    History is not populated — use for MCTS hot paths where history is built
    incrementally via copy() + push_index().
    """
    cr = int(board.castling_rights)
    castling = 0
  # Column-aligned bitfield decoding; keep single-line form intentionally.
    if cr & (1 << 7):  castling |= 1  # noqa: E701  WK_CASTLE — H1
    if cr & (1 << 0):  castling |= 2  # noqa: E701  WQ_CASTLE — A1
    if cr & (1 << 63): castling |= 4  # noqa: E701  BK_CASTLE — H8
    if cr & (1 << 56): castling |= 8  # noqa: E701  BQ_CASTLE — A8
    return CBoard.from_raw(
        int(board.pawns), int(board.knights), int(board.bishops),
        int(board.rooks), int(board.queens), int(board.kings),
        int(board.occupied_co[chess.WHITE]),
        int(board.occupied_co[chess.BLACK]),
        1 if board.turn else 0,
        castling,
        -1 if board.ep_square is None else int(board.ep_square),
        int(board.halfmove_clock),
    )

try:
    from chess_anti_engine.encoding._features_ext import (
        compute_extra_features as _c_compute,
    )
    _HAS_FEATURES_C = True
except ImportError:
    _HAS_FEATURES_C = False

if TYPE_CHECKING:
    from chess_anti_engine.encoding._features_ext import (
        compute_extra_features as _c_compute,  # noqa: F401,F811
    )


def lc0_root_history_mode(input_history_encoding: str | None) -> int:
    """Map a history encoding to the C hist_mode int (0/1/2)."""
    hist_enc = normalize_lc0_history_encoding(input_history_encoding)
    if hist_enc == LC0_HISTORY_ROOT_LEGACY_META:
        return 2
    return 1 if uses_lc0_root_history(hist_enc) else 0


def encode_cboard(
    cb: CBoard,
    *,
    input_history_encoding: str | None = None,
    input_extra_features: str | None = None,
) -> np.ndarray:
    """Encode a CBoard into (C, 8, 8) float32 (C = 146 v1 / 175 v2_threats).

    Prefer the fused CBoard.encode_full() path when available. Fall back to
    CBoard.encode_planes() + the existing features extension otherwise.
    """
    hist_enc = normalize_lc0_history_encoding(input_history_encoding)
    use_lc0_root = uses_lc0_root_history(hist_enc)
    n_extra = extra_feature_plane_count(input_extra_features)
    encode_full = getattr(cb, "encode_full", None)
    if encode_full is not None:
        return encode_full(lc0_root_history_mode(hist_enc), n_extra)

  # LC0 112 planes — all in C
    lc0 = (cb.encode_planes_lc0_root() if use_lc0_root else cb.encode_planes())
    if hist_enc == LC0_HISTORY_ROOT_LEGACY_META:
        apply_lc0_root_legacy_meta_planes_cboard(lc0, cb)

    if not _HAS_FEATURES_C:
  # No features C ext — return LC0 only with zero features
        out = np.zeros((112 + n_extra, 8, 8), dtype=np.float32)
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
                      king_sq_us, king_sq_them, is_white, ep_square, n_extra)

    return np.concatenate([lc0, feat], axis=0)
