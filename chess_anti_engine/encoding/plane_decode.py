"""Decode stored input planes back to bitboards; recompute extra features.

Replay shards store the encoded ``(C, 8, 8)`` input planes, not FENs. The
step-0 piece planes ``[0:12]`` are identical across all history encodings
(``legacy`` / ``lc0_root`` / ``lc0_root_legacy_meta``): "us" pieces 0-5,
"them" pieces 6-11, side-to-move POV (rank-flipped when black to move,
files never mirrored). The extra-feature block (see ``features.py``)
depends only on those 12 planes, the side to move, and — for the v1
pawn-mobility plane alone — the en-passant square; castling rights,
rule50, and deeper history never enter it.

Side-to-move POV makes the planes color-blind: rematerializing them as a
white-to-move position yields either the true position or its
color-flipped mirror, and every extra feature is invariant under that
flip (``tests/test_threat_planes.py::test_orientation_mirror_invariance``).
That is what lets v1 shards (146 planes) be upgraded to v2_threats (175
planes) without FENs: decode step-0 bitboards, recompute the full
63-plane extra block with the same kernel selfplay uses, and use the
first 34 recomputed planes as a validation oracle against the stored v1
block (``replay/threat_upgrade.py``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import chess
import numpy as np

from .features import (
    _HAS_C_EXT,
    extra_feature_plane_count,
    extra_feature_planes_fast,
)
from .lc0 import (
    LC0_HISTORY_LEGACY,
    LC0_HISTORY_ROOT,
    LC0_HISTORY_ROOT_LEGACY_META,
    normalize_lc0_history_encoding,
)

if TYPE_CHECKING or _HAS_C_EXT:
    from .features import _c_compute

# bit weight per (rank, file) cell in the white-POV plane frame, where
# plane[r, c] maps to square r*8+c (see encode_lc0 orientation: files are
# never mirrored, ranks are flipped for black so the stored frame is
# always "us moves up the board").
_SQUARE_BITS = (np.uint64(1) << np.arange(64, dtype=np.uint64)).reshape(8, 8)

# En-passant target square in the side-to-move frame is always on rank
# index 5 (the row behind "their" just-double-pushed pawn).
_EP_RANK = 5

# History slot 1 layout offsets in the lc0_root 13-plane-stride layout.
_ROOT_SLOT1_START = 13
_ROOT_SLOT1_THEM_PAWNS = 13 + 6


def decode_step0_bitboards(x: np.ndarray) -> np.ndarray:
    """Decode the current-position piece bitboards from stored planes.

    ``x`` is ``(N, C, 8, 8)`` (any float dtype, values 0/1). Returns
    ``(N, 12)`` uint64 bitboards in the stored side-to-move frame:
    columns 0-5 are "us" P/N/B/R/Q/K, columns 6-11 are "them".
    """
    planes = np.asarray(x)[:, :12] > 0.5
    return np.sum(planes.astype(np.uint64) * _SQUARE_BITS, axis=(-2, -1))


def _ep_from_file_plane(plane: np.ndarray) -> int:
    cols = np.flatnonzero(np.asarray(plane).max(axis=0) > 0.5)
    if cols.size != 1:
        return -1
    return _EP_RANK * 8 + int(cols[0])


def _ep_from_root_history(x_row: np.ndarray) -> int:
    """Infer EP for pure ``lc0_root`` rows (no EP metadata plane).

    Slot 1 is exactly one ply older than slot 0 and that ply was "their"
    move, so an EP square exists iff a "them" pawn vanished from rank
    index 6 and appeared on rank index 4 in the same file (a double
    push). Any other single move cannot set both bits for one file.
    """
    row = np.asarray(x_row)
    if not (row[_ROOT_SLOT1_START:_ROOT_SLOT1_START + 12] > 0.5).any():
        return -1  # no slot-1 history recorded
    them_p0 = row[6] > 0.5
    them_p1 = row[_ROOT_SLOT1_THEM_PAWNS] > 0.5
    cand = np.flatnonzero(them_p0[4] & ~them_p1[4] & them_p1[6] & ~them_p0[6])
    if cand.size != 1:
        return -1
    return _EP_RANK * 8 + int(cand[0])


def decode_ep_square(x_row: np.ndarray, history_encoding: str | None) -> int:
    """Recover the EP square (side-to-move frame) from one stored row, or -1.

    Only the v1 pawn-mobility plane depends on EP; the v2 threat planes
    do not (``features.py`` semantics notes), so a rare -1 fallback on
    pure ``lc0_root`` rows costs validation precision, never targets.
    """
    enc = normalize_lc0_history_encoding(history_encoding)
    if enc == LC0_HISTORY_LEGACY:
        return _ep_from_file_plane(x_row[100])
    if enc == LC0_HISTORY_ROOT_LEGACY_META:
        return _ep_from_file_plane(x_row[110])
    assert enc == LC0_HISTORY_ROOT
    return _ep_from_root_history(x_row)


def _board_from_bitboards(
    pieces_us: np.ndarray, pieces_them: np.ndarray, ep_square: int,
) -> chess.Board:
    """Rebuild a white-to-move board for the Python feature fallback."""
    board = chess.Board(None)
    for color, pieces in ((chess.WHITE, pieces_us), (chess.BLACK, pieces_them)):
        for pt_idx, bb in enumerate(pieces):
            piece = chess.Piece(pt_idx + 1, color)  # PAWN..KING == 1..6
            for sq in chess.scan_forward(int(bb)):
                board.set_piece_at(sq, piece)
    board.turn = chess.WHITE
    board.castling_rights = 0
    board.ep_square = ep_square if ep_square >= 0 else None
    return board


def recompute_extra_planes(
    x: np.ndarray,
    history_encoding: str | None = None,
    *,
    version: str = "v2_threats",
) -> np.ndarray:
    """Recompute the full extra-feature block from stored input planes.

    Returns ``(N, n_extra, 8, 8)`` float32 in the same side-to-move frame
    as ``x``. Always computed with ``turn_white=True`` on the decoded
    bitboards — for black-to-move rows that is the color-flipped mirror,
    which produces identical planes (see module docstring).
    """
    rows = np.asarray(x)
    n_extra = extra_feature_plane_count(version)
    bbs = decode_step0_bitboards(rows)
    out = np.empty((rows.shape[0], n_extra, 8, 8), dtype=np.float32)
    for i in range(rows.shape[0]):
        pieces_us = np.ascontiguousarray(bbs[i, :6])
        pieces_them = np.ascontiguousarray(bbs[i, 6:])
        ep_square = decode_ep_square(rows[i], history_encoding)
        if _HAS_C_EXT:
            occupied = int(np.bitwise_or.reduce(bbs[i]))
            us_king = int(pieces_us[5])
            them_king = int(pieces_them[5])
            out[i] = _c_compute(
                pieces_us, pieces_them, occupied,
                us_king.bit_length() - 1 if us_king else -1,
                them_king.bit_length() - 1 if them_king else -1,
                True, ep_square, n_extra,
            )
        else:
            board = _board_from_bitboards(pieces_us, pieces_them, ep_square)
            out[i] = extra_feature_planes_fast(board, version=version)
    return out
