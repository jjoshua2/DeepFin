from __future__ import annotations


import numpy as np
import chess

from .lc0 import encode_lc0_full, encode_lc0_reduced, _HAS_LC0_C_EXT
if _HAS_LC0_C_EXT:
    from .lc0 import encode_lc0_full_c
from .features import extra_feature_planes_fast, _HAS_C_EXT
if _HAS_C_EXT:
    from .features import extra_feature_planes_c

# Module-level imports for fused C path (avoid per-call import overhead)
if _HAS_LC0_C_EXT:
    from .lc0 import _c_encode_piece_planes, _check_repetitions, _write_metadata_planes
if _HAS_C_EXT:
    from .features import _c_compute


def encode_position(
    board: chess.Board,
    *,
    add_features: bool = True,
    feature_dropout_p: float = 0.0,
    rng: np.random.Generator | None = None,
    use_full_lc0: bool = True,
) -> np.ndarray:
    """Encode a position into (C, 8, 8) float32.

    By default this produces the spec target shape:
    - LC0 full 112-plane input (8 history steps)
    - plus 34 additional classical feature planes
    => total 146 planes.

    `use_full_lc0=False` keeps the earlier reduced encoder for debugging.

    Parameters
    - feature_dropout_p: probability of zeroing ALL extra feature planes.
    """
    if use_full_lc0:
        base = encode_lc0_full_c(board) if _HAS_LC0_C_EXT else encode_lc0_full(board)
    else:
        base = encode_lc0_reduced(board)

    if not add_features:
        return base

    feat_arr = extra_feature_planes_c(board) if _HAS_C_EXT else extra_feature_planes_fast(board)

    if feature_dropout_p > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        if float(rng.random()) < float(feature_dropout_p):
            feat_arr[...] = 0.0

    return np.concatenate([base, feat_arr], axis=0)


def encode_position_into(
    board: chess.Board,
    out: np.ndarray,
    *,
    add_features: bool = True,
    feature_dropout_p: float = 0.0,
    rng: np.random.Generator | None = None,
    use_full_lc0: bool = True,
) -> None:
    """Encode a position directly into a pre-allocated (146, 8, 8) buffer.

    Avoids intermediate allocations and np.concatenate.  When C extensions
    are available, extracts bitboards once and passes them to both the LC0
    encoder and the features encoder, halving Python attribute access overhead.
    """
    out[...] = 0.0

    if use_full_lc0 and _HAS_LC0_C_EXT and _HAS_C_EXT and add_features:
        # Fast fused path: extract bitboards once, pass to both C extensions.
        _encode_fused_c(board, out, feature_dropout_p=feature_dropout_p, rng=rng)
        return

    # Fallback: delegate to existing functions, copy into out.
    if use_full_lc0:
        base = encode_lc0_full_c(board) if _HAS_LC0_C_EXT else encode_lc0_full(board)
        base_planes = 112
    else:
        base = encode_lc0_reduced(board)
        base_planes = base.shape[0]
    out[:base_planes] = base

    if not add_features:
        return

    feat = extra_feature_planes_c(board) if _HAS_C_EXT else extra_feature_planes_fast(board)

    if feature_dropout_p > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        if float(rng.random()) < float(feature_dropout_p):
            feat[...] = 0.0

    out[base_planes:base_planes + feat.shape[0]] = feat


def encode_position_fused(board: chess.Board) -> np.ndarray:
    """Fused encode_position: single allocation, no concatenate."""
    out = np.zeros((146, 8, 8), dtype=np.float32)
    encode_position_into(board, out, add_features=True)
    return out


def encode_positions_batch(
    boards: list[chess.Board],
    *,
    add_features: bool = True,
    feature_dropout_p: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Encode multiple positions into a single pre-allocated (N, C, 8, 8) array.

    Avoids N per-position allocations + np.stack overhead.
    """
    n = len(boards)
    c = 146 if add_features else 112
    out = np.zeros((n, c, 8, 8), dtype=np.float32)
    for i, b in enumerate(boards):
        if add_features and c == 146:
            encode_position_into(b, out[i], add_features=True,
                                 feature_dropout_p=feature_dropout_p, rng=rng)
        else:
            out[i] = encode_position(b, add_features=add_features,
                                     feature_dropout_p=feature_dropout_p, rng=rng)
    return out


def _encode_fused_c(
    board: chess.Board,
    out: np.ndarray,
    *,
    feature_dropout_p: float = 0.0,
    rng: np.random.Generator | None = None,
) -> None:
    """Fused C-accelerated encoding: extract bitboards once for both LC0 + features.

    Writes 146 planes into out (must be (146, 8, 8) float32).
    """
    turn = board.turn
    us = turn

    # ── Extract bitboards once for current position ──
    occ_w = int(board.occupied_co[chess.WHITE])
    occ_b = int(board.occupied_co[chess.BLACK])
    pawns = board.pawns
    knights = board.knights
    bishops = board.bishops
    rooks = board.rooks
    queens = board.queens
    kings = board.kings
    ep_square = board.ep_square

    stack = board._stack
    stack_len = len(stack)

    # ── LC0 piece planes (history) ──
    bbs_list: list[int] = []
    turns_list: list[int] = []
    n_steps = 0

    for hist_idx in range(8):
        if hist_idx == 0:
            turn_h = turn
            h_occ_w, h_occ_b = occ_w, occ_b
            h_pawns, h_knights, h_bishops = pawns, knights, bishops
            h_rooks, h_queens, h_kings = rooks, queens, kings
        else:
            si = stack_len - hist_idx
            if si < 0:
                break
            s = stack[si]
            turn_h = s.turn
            h_occ_w = s.occupied_w
            h_occ_b = s.occupied_b
            h_pawns, h_knights, h_bishops = s.pawns, s.knights, s.bishops
            h_rooks, h_queens, h_kings = s.rooks, s.queens, s.kings

        turns_list.append(1 if turn_h == chess.WHITE else 0)
        us_occ = h_occ_w if turn_h == chess.WHITE else h_occ_b
        them_occ = h_occ_b if turn_h == chess.WHITE else h_occ_w
        for occ in (us_occ, them_occ):
            for bb in (h_pawns, h_knights, h_bishops, h_rooks, h_queens, h_kings):
                bbs_list.append(bb & occ)
        n_steps += 1

    bbs_arr = np.array(bbs_list, dtype=np.uint64)
    turns_arr = np.array(turns_list, dtype=np.int32)
    planes = _c_encode_piece_planes(bbs_arr, turns_arr, n_steps)
    out[:n_steps * 12] = planes

    # ── Repetition planes ──
    _check_repetitions(board, stack, stack_len, n_steps, out, 103)

    # ── Metadata planes ──
    _write_metadata_planes(out, board)

    # ── Feature planes (reuse already-extracted bitboards) ──
    us_occ_cur = occ_w if us == chess.WHITE else occ_b
    them_occ_cur = occ_b if us == chess.WHITE else occ_w

    piece_types_bb = (pawns, knights, bishops, rooks, queens, kings)
    pieces_us = np.array(
        [bb & us_occ_cur for bb in piece_types_bb], dtype=np.uint64
    )
    pieces_them = np.array(
        [bb & them_occ_cur for bb in piece_types_bb], dtype=np.uint64
    )

    occupied = occ_w | occ_b
    us_kings_bb = kings & us_occ_cur
    them_kings_bb = kings & them_occ_cur
    king_sq_us = int(us_kings_bb).bit_length() - 1 if us_kings_bb else -1
    king_sq_them = int(them_kings_bb).bit_length() - 1 if them_kings_bb else -1
    turn_white = (turn == chess.WHITE)
    ep_sq_int = ep_square if ep_square is not None else -1

    feat = _c_compute(pieces_us, pieces_them, occupied,
                      king_sq_us, king_sq_them, turn_white, ep_sq_int)

    if feature_dropout_p > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        if float(rng.random()) < float(feature_dropout_p):
            feat[...] = 0.0

    out[112:] = feat
