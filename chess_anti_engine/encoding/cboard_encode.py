"""Encoding helpers for CBoard objects (C-accelerated chess boards)."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import chess
import numpy as np

import chess_anti_engine.encoding._lc0_ext as _lc0_ext
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.features import extra_feature_plane_count
from chess_anti_engine.ext_check import REBUILD_HINT, CCoreIncompleteError, refuse_fallback
from chess_anti_engine.encoding.lc0 import (
    LC0_HISTORY_ROOT_LEGACY_META,
    c_input_history_mode as lc0_root_history_mode,
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
        compute_extra_features as _c_compute,
    )


def encode_cboard(
    cb: CBoard,
    *,
    input_history_encoding: str | None = None,
    input_extra_features: str | None = None,
) -> np.ndarray:
    """Encode a CBoard into (C, 8, 8) float32 (C = 146 v1 / 175 v2_threats).

    Uses the fused ``CBoard.encode_full()``. A board without it (a stale
    build) is refused; under ``ext_check.python_reference_paths`` the
    reference composition ``CBoard.encode_planes()`` + the features extension
    runs instead. A missing features extension is always an error: this used
    to return all-zero feature planes, which the net reads as a real position.
    """
    hist_enc = normalize_lc0_history_encoding(input_history_encoding)
    use_lc0_root = uses_lc0_root_history(hist_enc)
    n_extra = extra_feature_plane_count(input_extra_features)
    encode_full = getattr(cb, "encode_full", None)
    if encode_full is not None:
        return encode_full(lc0_root_history_mode(hist_enc), n_extra)
    refuse_fallback(
        "encode_cboard", "CBoard.encode_full",
        "CBoard.encode_planes() + per-board feature planes",
    )
    if not _HAS_FEATURES_C:
        raise CCoreIncompleteError(
            "encode_cboard: _features_ext.compute_extra_features is unavailable; "
            "refusing to emit all-zero feature planes. Rebuild: " + REBUILD_HINT
        )

  # LC0 112 planes — all in C
    lc0 = (cb.encode_planes_lc0_root() if use_lc0_root else cb.encode_planes())
    if hist_enc == LC0_HISTORY_ROOT_LEGACY_META:
        apply_lc0_root_legacy_meta_planes_cboard(lc0, cb)

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


def encode_cboard_batch(
    boards: Sequence[CBoard],
    *,
    input_history_encoding: str | None = None,
    input_extra_features: str | None = None,
) -> np.ndarray:
    """Encode N CBoards into (N, C, 8, 8) float32 (C = 146 v1 / 175 v2_threats).

    Uses the module-level C ``encode_full_batch`` (one Python->C call into a
    contiguous buffer). A stale build that lacks the symbol is refused; under
    ``ext_check.python_reference_paths`` per-board ``encode_cboard`` results
    are stacked instead (the reference the batch path is tested against).
    """
    n_extra = extra_feature_plane_count(input_extra_features)
    n_planes = 112 + n_extra
    n = len(boards)
    if n == 0:
        return np.empty((0, n_planes, 8, 8), dtype=np.float32)

    hist_mode = lc0_root_history_mode(input_history_encoding)
    encode_full_batch = getattr(_lc0_ext, "encode_full_batch", None)
    if encode_full_batch is not None:
        out = np.empty((n, n_planes, 8, 8), dtype=np.float32)
        encode_full_batch(boards, out, hist_mode, n_extra)
        return out
    refuse_fallback(
        "encode_cboard_batch", "_lc0_ext.encode_full_batch",
        "stacking per-board encode_cboard results",
    )
    return np.stack(
        [
            encode_cboard(
                b,
                input_history_encoding=input_history_encoding,
                input_extra_features=input_extra_features,
            )
            for b in boards
        ],
        axis=0,
    )
