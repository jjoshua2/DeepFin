from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import numpy as np

from chess_anti_engine.utils.bitboards import file_to_plane

try:
    from chess_anti_engine.encoding._lc0_ext import (
        encode_piece_planes as _c_encode_piece_planes,
    )
    _HAS_LC0_C_EXT = True
except ImportError:
    _HAS_LC0_C_EXT = False

if TYPE_CHECKING:
    from chess_anti_engine.encoding._lc0_ext import (
        encode_piece_planes as _c_encode_piece_planes,  # noqa: F401,F811
    )


@dataclass(frozen=True)
class LC0ReducedPlaneSpec:
    """Plane layout for the initial reduced LC0 baseline."""

    num_piece_planes: int = 12
    num_castling_planes: int = 4
    num_ep_planes: int = 1
    num_rule50_planes: int = 1
    num_repetition_planes: int = 1
    num_ones_planes: int = 1

    @property
    def num_planes(self) -> int:
        return (
            self.num_piece_planes
            + self.num_castling_planes
            + self.num_ep_planes
            + self.num_rule50_planes
            + self.num_repetition_planes
            + self.num_ones_planes
        )


@dataclass(frozen=True)
class LC0FullPlaneSpec:
    """LC0 112-plane layout (8 history steps) as described in prompt/spec."""

    history_len: int = 8
    piece_planes_per_history: int = 12

    num_castling_planes: int = 4
    num_ep_planes: int = 1
    num_turn_planes: int = 1
    num_rule50_planes: int = 1
    num_repetition_planes: int = 8
    num_ones_planes: int = 1

    @property
    def num_piece_planes(self) -> int:
        return self.history_len * self.piece_planes_per_history

    @property
    def num_planes(self) -> int:
        return (
            self.num_piece_planes
            + self.num_castling_planes
            + self.num_ep_planes
            + self.num_turn_planes
            + self.num_rule50_planes
            + self.num_repetition_planes
            + self.num_ones_planes
        )

    @property
    def metadata_base(self) -> int:
        """First plane of the legacy per-board metadata block (right after the
        piece planes): castling, EP, turn, rule50 in that order."""
        return self.num_piece_planes  # 96

    @property
    def legacy_repetition_base(self) -> int:
        """First repetition plane in the legacy layout (after the metadata block)."""
        return (
            self.metadata_base
            + self.num_castling_planes
            + self.num_ep_planes
            + self.num_turn_planes
            + self.num_rule50_planes
        )  # 103

    @property
    def root_metadata_base(self) -> int:
        """First metadata plane in the root layout, which interleaves one
        repetition plane into each history slot (piece_planes_per_history + 1
        planes per slot) rather than appending them all at the end."""
        return self.history_len * (self.piece_planes_per_history + 1)  # 104

    @property
    def ones_plane(self) -> int:
        """Index of the trailing all-ones bias plane (shared by both layouts)."""
        return self.num_planes - self.num_ones_planes  # 111


LC0_REDUCED = LC0ReducedPlaneSpec()
LC0_FULL = LC0FullPlaneSpec()
LC0_HISTORY_LEGACY = "legacy"
LC0_HISTORY_ROOT = "lc0_root"
LC0_HISTORY_ROOT_LEGACY_META = "lc0_root_legacy_meta"
# Passed into C extension mode switches; keep these values stable.
LC0_HISTORY_MODE_LEGACY = 0
LC0_HISTORY_MODE_ROOT = 1
LC0_HISTORY_MODE_ROOT_LEGACY_META = 2


def normalize_lc0_history_encoding(input_history_encoding: str | None) -> str:
    enc = str(input_history_encoding or LC0_HISTORY_LEGACY).lower().strip()
    aliases = {
        "legacy": LC0_HISTORY_LEGACY,
        "current": LC0_HISTORY_LEGACY,
        "old": LC0_HISTORY_LEGACY,
        "lc0_root": LC0_HISTORY_ROOT,
        "lc0": LC0_HISTORY_ROOT,
        "root": LC0_HISTORY_ROOT,
        "root_pov": LC0_HISTORY_ROOT,
        "lc0_13": LC0_HISTORY_ROOT,
        "lc0_root_legacy_meta": LC0_HISTORY_ROOT_LEGACY_META,
        "root_legacy_meta": LC0_HISTORY_ROOT_LEGACY_META,
        "lc0_root_meta": LC0_HISTORY_ROOT_LEGACY_META,
        "root_meta": LC0_HISTORY_ROOT_LEGACY_META,
    }
    try:
        return aliases[enc]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported input_history_encoding {input_history_encoding!r}; "
            f"expected '{LC0_HISTORY_LEGACY}', '{LC0_HISTORY_ROOT}', "
            f"or '{LC0_HISTORY_ROOT_LEGACY_META}'"
        ) from exc


def uses_lc0_root_history(input_history_encoding: str | None) -> bool:
    """Return True for encodings that use LC0 root-side history slots."""
    return normalize_lc0_history_encoding(input_history_encoding) in {
        LC0_HISTORY_ROOT,
        LC0_HISTORY_ROOT_LEGACY_META,
    }


def uses_lc0_root_legacy_meta(input_history_encoding: str | None) -> bool:
    """Return True for LC0 root history with legacy EP/rule50 metadata."""
    return normalize_lc0_history_encoding(input_history_encoding) == LC0_HISTORY_ROOT_LEGACY_META


def c_input_history_mode(input_history_encoding: str | None) -> int:
    """Return the C fast-path mode id for an input history encoding."""
    hist_enc = normalize_lc0_history_encoding(input_history_encoding)
    if hist_enc == LC0_HISTORY_ROOT_LEGACY_META:
        return LC0_HISTORY_MODE_ROOT_LEGACY_META
    if hist_enc == LC0_HISTORY_ROOT:
        return LC0_HISTORY_MODE_ROOT
    return LC0_HISTORY_MODE_LEGACY


def apply_lc0_root_legacy_meta_planes(out: np.ndarray, board: chess.Board) -> None:
    """Patch the LC0-root rule50/movecount planes to match the legacy metadata scale."""
    rule50 = LC0_FULL.root_metadata_base + 5
    movecount = LC0_FULL.root_metadata_base + 6
    out[rule50, :, :] = min(float(board.halfmove_clock), 100.0) / 100.0
    out[movecount, :, :] = 0.0
    if board.ep_square is not None:
        out[movecount, :, chess.square_file(board.ep_square)] = 1.0


def _write_piece_planes(board: chess.Board, out: np.ndarray, start: int) -> int:
    """Write 12 piece planes for one position starting at `start`.

    Orientation and us/them are relative to this board's side-to-move.
    Batches all 12 bitboard→plane conversions into a single numpy operation.
    """
    turn = board.turn
    us = turn
    them = not turn

    raw_bytes = b''.join(
        int(board.pieces_mask(pt, color)).to_bytes(8, 'big')
        for color in (us, them)
        for pt in chess.PIECE_TYPES
    )
    raw = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8)).reshape(12, 8, 8)
    if turn == chess.WHITE:
        out[start:start + 12] = raw[:, ::-1, ::-1]
    else:
        out[start:start + 12] = raw[:, :, ::-1]
    return start + 12


def encode_lc0_reduced(board: chess.Board) -> np.ndarray:
    """Encode current position + minimal metadata into (C,8,8) float32."""
    turn = board.turn
    us = turn
    them = not turn

    out = np.zeros((LC0_REDUCED.num_planes, 8, 8), dtype=np.float32)
    idx = _write_piece_planes(board, out, 0)

  # Castling rights: us-K, us-Q, them-K, them-Q
    castling_flags = [
        board.has_kingside_castling_rights(us),
        board.has_queenside_castling_rights(us),
        board.has_kingside_castling_rights(them),
        board.has_queenside_castling_rights(them),
    ]
    for has in castling_flags:
        if has:
            out[idx, :, :] = 1.0
        idx += 1

  # En passant file
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        out[idx, :, :] = file_to_plane(ep_file)
    idx += 1

  # Rule50 (halfmove clock) normalized
    rule50 = min(float(board.halfmove_clock), 100.0) / 100.0
    out[idx, :, :] = rule50
    idx += 1

  # Repetition marker (current position repeated at least once)
    rep = 1.0 if (board.is_repetition(2) or board.is_repetition(3)) else 0.0
    if rep > 0.0:
        out[idx, :, :] = rep
    idx += 1

  # All-ones bias
    out[idx, :, :] = 1.0

    assert out.shape == (LC0_REDUCED.num_planes, 8, 8)
    return out


_BB_FIELDS = ('pawns', 'knights', 'bishops', 'rooks', 'queens', 'kings')
_STRUCT_96Q = struct.Struct('>' + 'Q' * 96)
_STRUCT_84Q = struct.Struct('>' + 'Q' * 84)
_STRUCT_72Q = struct.Struct('>' + 'Q' * 72)
_STRUCT_60Q = struct.Struct('>' + 'Q' * 60)
_STRUCT_48Q = struct.Struct('>' + 'Q' * 48)
_STRUCT_36Q = struct.Struct('>' + 'Q' * 36)
_STRUCT_24Q = struct.Struct('>' + 'Q' * 24)
_STRUCT_12Q = struct.Struct('>' + 'Q' * 12)
_STRUCT_BY_N = {
    8: _STRUCT_96Q, 7: _STRUCT_84Q, 6: _STRUCT_72Q, 5: _STRUCT_60Q,
    4: _STRUCT_48Q, 3: _STRUCT_36Q, 2: _STRUCT_24Q, 1: _STRUCT_12Q,
}


def encode_lc0_full(
    board: chess.Board,
    *,
    history_len: int = 8,
    input_history_encoding: str | None = None,
) -> np.ndarray:
    """Encode LC0 112-plane input (8 history positions × 12 + metadata planes).

    Each history position is encoded relative to THAT position's side-to-move.
    Reads bitboards directly from board._stack to avoid expensive copy/pop.
    Batches all history bitboards via struct.pack + single unpackbits call.
    """
    hist_enc = normalize_lc0_history_encoding(input_history_encoding)
    if uses_lc0_root_history(hist_enc):
        out = encode_lc0_full_root(board, history_len=history_len)
        if hist_enc == LC0_HISTORY_ROOT_LEGACY_META:
            apply_lc0_root_legacy_meta_planes(out, board)
        return out

    out = np.zeros((LC0_FULL.num_planes, 8, 8), dtype=np.float32)

    stack = board._stack
    stack_len = len(stack)

  # Collect bitboards from current board + _stack history.
    bbs: list[int] = []
    turns: list[bool] = []
    n_steps = 0

    for hist_idx in range(history_len):
        if hist_idx == 0:
            turn_h = board.turn
            occ_w = int(board.occupied_co[chess.WHITE])
            occ_b = int(board.occupied_co[chess.BLACK])
            piece_bbs = tuple(getattr(board, f) for f in _BB_FIELDS)
        else:
            si = stack_len - hist_idx
            if si < 0:
                break
            s = stack[si]
            turn_h = s.turn
            occ_w = s.occupied_w
            occ_b = s.occupied_b
            piece_bbs = tuple(getattr(s, f) for f in _BB_FIELDS)

        turns.append(turn_h)
        us_occ = occ_w if turn_h == chess.WHITE else occ_b
        them_occ = occ_b if turn_h == chess.WHITE else occ_w
        bbs.extend(
            bb & occ
            for occ in (us_occ, them_occ)
            for bb in piece_bbs
        )
        n_steps += 1

  # Batch-convert all bitboards via struct.pack (3x faster than to_bytes)
    n_bbs = n_steps * 12
    pack_struct = _STRUCT_BY_N.get(n_steps)
    if pack_struct is not None:
        raw_bytes = pack_struct.pack(*bbs)
    else:
        raw_bytes = struct.pack('>' + 'Q' * n_bbs, *bbs)
    raw = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8)).reshape(n_bbs, 8, 8)

  # Write piece planes with correct orientation per history step
    for hist_idx in range(n_steps):
        s_bb = hist_idx * 12
        s_plane = hist_idx * 12
        if turns[hist_idx] == chess.WHITE:
            out[s_plane:s_plane + 12] = raw[s_bb:s_bb + 12, ::-1, ::-1]
        else:
            out[s_plane:s_plane + 12] = raw[s_bb:s_bb + 12, :, ::-1]

  # Repetition planes — check only the 8 history positions, not entire stack.
  # Build keys only for positions within the history window + scan backward.
    rep_base = LC0_FULL.legacy_repetition_base
    _check_repetitions(board, stack, stack_len, n_steps, out, rep_base)

    _write_metadata_planes(out, board)
    return out


def encode_lc0_full_root(board: chess.Board, *, history_len: int = 8) -> np.ndarray:
    """Encode LC0-style 112 planes with 13 planes per root-POV history slot.

    Layout:
    - 0..103: 8 slots of 12 piece planes + 1 repetition plane.
    - 104..107: castling rights (us-Q, us-K, them-Q, them-K).
    - 108: side/color flag (1.0 when the root side to move is black).
    - 109: raw rule-50 counter.
    - 110: zero legacy movecount plane.
    - 111: all-ones bias.

    Unlike the legacy encoder, every history slot is encoded from the root
    side-to-move POV. That keeps "us"/"them" and board orientation stable
    across the temporal stack.
    """
    out = np.zeros((LC0_FULL.num_planes, 8, 8), dtype=np.float32)
    root_turn = board.turn
    stack = board._stack
    stack_len = len(stack)

    bbs: list[int] = []
    n_steps = 0
    for hist_idx in range(history_len):
        if hist_idx == 0:
            occ_w = int(board.occupied_co[chess.WHITE])
            occ_b = int(board.occupied_co[chess.BLACK])
            piece_bbs = tuple(getattr(board, f) for f in _BB_FIELDS)
        else:
            si = stack_len - hist_idx
            if si < 0:
                break
            s = stack[si]
            occ_w = s.occupied_w
            occ_b = s.occupied_b
            piece_bbs = tuple(getattr(s, f) for f in _BB_FIELDS)

        us_occ = occ_w if root_turn == chess.WHITE else occ_b
        them_occ = occ_b if root_turn == chess.WHITE else occ_w
        bbs.extend(bb & occ for occ in (us_occ, them_occ) for bb in piece_bbs)
        n_steps += 1

    n_bbs = n_steps * 12
    pack_struct = _STRUCT_BY_N.get(n_steps)
    if pack_struct is not None:
        raw_bytes = pack_struct.pack(*bbs)
    else:
        raw_bytes = struct.pack('>' + 'Q' * n_bbs, *bbs)
    raw = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8)).reshape(n_bbs, 8, 8)

    for hist_idx in range(n_steps):
        s_bb = hist_idx * 12
        s_plane = hist_idx * 13
        if root_turn == chess.WHITE:
            out[s_plane:s_plane + 12] = raw[s_bb:s_bb + 12, ::-1, ::-1]
        else:
            out[s_plane:s_plane + 12] = raw[s_bb:s_bb + 12, :, ::-1]

    _check_repetitions(board, stack, stack_len, n_steps, out, 12, rep_stride=13)
    _write_metadata_planes_root(out, board)
    return out


def _write_metadata_planes(out: np.ndarray, board: chess.Board) -> None:
    """Write castling, EP, turn, rule50, and ones-bias planes starting at index 96."""
    turn = board.turn
    us = turn
    them = not turn
    meta_idx = LC0_FULL.metadata_base

    for has in (
        board.has_kingside_castling_rights(us),
        board.has_queenside_castling_rights(us),
        board.has_kingside_castling_rights(them),
        board.has_queenside_castling_rights(them),
    ):
        if has:
            out[meta_idx, :, :] = 1.0
        meta_idx += 1

    if board.ep_square is not None:
        out[meta_idx, :, chess.square_file(board.ep_square)] = 1.0
    meta_idx += 1

    out[meta_idx, :, :] = 1.0  # color to move
    meta_idx += 1

    out[meta_idx, :, :] = min(float(board.halfmove_clock), 100.0) / 100.0
    meta_idx += 1

    out[LC0_FULL.ones_plane, :, :] = 1.0  # all-ones bias


def _write_metadata_planes_root(out: np.ndarray, board: chess.Board) -> None:
    """Write LC0 classical root-history metadata at planes 104..111."""
    turn = board.turn
    us = turn
    them = not turn
    meta_idx = LC0_FULL.root_metadata_base

    for has in (
        board.has_queenside_castling_rights(us),
        board.has_kingside_castling_rights(us),
        board.has_queenside_castling_rights(them),
        board.has_kingside_castling_rights(them),
    ):
        if has:
            out[meta_idx, :, :] = 1.0
        meta_idx += 1

    if turn == chess.BLACK:
        out[meta_idx, :, :] = 1.0
    meta_idx += 1

    out[meta_idx, :, :] = float(board.halfmove_clock)
    out[LC0_FULL.ones_plane, :, :] = 1.0


def encode_lc0_full_c(
    board: chess.Board,
    *,
    history_len: int = 8,
    input_history_encoding: str | None = None,
) -> np.ndarray:
    """C-accelerated version of encode_lc0_full.

    Extracts bitboard values in Python (fast attribute access), then uses C
    for the bitboard→plane conversion (pack + orient). Metadata + repetitions
    stay in Python.
    """
    hist_enc = normalize_lc0_history_encoding(input_history_encoding)
    if uses_lc0_root_history(hist_enc):
        out = _encode_lc0_full_root_c(board, history_len=history_len)
        if hist_enc == LC0_HISTORY_ROOT_LEGACY_META:
            apply_lc0_root_legacy_meta_planes(out, board)
        return out

    out = np.zeros((LC0_FULL.num_planes, 8, 8), dtype=np.float32)

    stack = board._stack
    stack_len = len(stack)

  # Collect bitboards from current board + _stack history
    bbs_list: list[int] = []
    turns_list: list[int] = []
    n_steps = 0

    for hist_idx in range(history_len):
        if hist_idx == 0:
            turn_h = board.turn
            occ_w = int(board.occupied_co[chess.WHITE])
            occ_b = int(board.occupied_co[chess.BLACK])
            pawns, knights, bishops = board.pawns, board.knights, board.bishops
            rooks, queens, kings = board.rooks, board.queens, board.kings
        else:
            si = stack_len - hist_idx
            if si < 0:
                break
            s = stack[si]
            turn_h = s.turn
            occ_w = s.occupied_w
            occ_b = s.occupied_b
            pawns, knights, bishops = s.pawns, s.knights, s.bishops
            rooks, queens, kings = s.rooks, s.queens, s.kings

        turns_list.append(1 if turn_h == chess.WHITE else 0)
        us_occ = occ_w if turn_h == chess.WHITE else occ_b
        them_occ = occ_b if turn_h == chess.WHITE else occ_w
        bbs_list.extend(
            bb & occ
            for occ in (us_occ, them_occ)
            for bb in (pawns, knights, bishops, rooks, queens, kings)
        )
        n_steps += 1

  # C-accelerated bitboard → plane conversion
    bbs_arr = np.array(bbs_list, dtype=np.uint64)
    turns_arr = np.array(turns_list, dtype=np.int32)
    planes = _c_encode_piece_planes(bbs_arr, turns_arr, n_steps)
    out[:n_steps * 12] = planes

  # Repetition planes
    rep_base = LC0_FULL.legacy_repetition_base
    _check_repetitions(board, stack, stack_len, n_steps, out, rep_base)

    _write_metadata_planes(out, board)
    return out


def _encode_lc0_full_root_c(board: chess.Board, *, history_len: int = 8) -> np.ndarray:
    """C-assisted bitboard conversion for the root-POV 13-plane history layout."""
    out = np.zeros((LC0_FULL.num_planes, 8, 8), dtype=np.float32)
    root_turn = board.turn
    stack = board._stack
    stack_len = len(stack)

    bbs_list: list[int] = []
    turns_list: list[int] = []
    n_steps = 0
    for hist_idx in range(history_len):
        if hist_idx == 0:
            occ_w = int(board.occupied_co[chess.WHITE])
            occ_b = int(board.occupied_co[chess.BLACK])
            pawns, knights, bishops = board.pawns, board.knights, board.bishops
            rooks, queens, kings = board.rooks, board.queens, board.kings
        else:
            si = stack_len - hist_idx
            if si < 0:
                break
            s = stack[si]
            occ_w = s.occupied_w
            occ_b = s.occupied_b
            pawns, knights, bishops = s.pawns, s.knights, s.bishops
            rooks, queens, kings = s.rooks, s.queens, s.kings

        turns_list.append(1 if root_turn == chess.WHITE else 0)
        us_occ = occ_w if root_turn == chess.WHITE else occ_b
        them_occ = occ_b if root_turn == chess.WHITE else occ_w
        bbs_list.extend(
            bb & occ
            for occ in (us_occ, them_occ)
            for bb in (pawns, knights, bishops, rooks, queens, kings)
        )
        n_steps += 1

    bbs_arr = np.array(bbs_list, dtype=np.uint64)
    turns_arr = np.array(turns_list, dtype=np.int32)
    planes = _c_encode_piece_planes(bbs_arr, turns_arr, n_steps)
    for hist_idx in range(n_steps):
        out[hist_idx * 13:hist_idx * 13 + 12] = planes[hist_idx * 12:hist_idx * 12 + 12]

    _check_repetitions(board, stack, stack_len, n_steps, out, 12, rep_stride=13)
    _write_metadata_planes_root(out, board)
    return out


def _check_repetitions(board, stack, stack_len, n_steps, out, rep_base, *, rep_stride: int = 1):
    """Set repetition plane flags for each history step.

    Builds a set of all position keys seen before the history window,
    then checks each history position against it (O(1) per lookup).
    """
    def _skey(s):
  # Omit ep_square: python-chess is_repetition ignores EP when no EP
  # capture is legal. Excluding it avoids false negatives and matches
  # the old behavior. False positives are extremely rare and harmless.
        return (s.pawns, s.knights, s.bishops, s.rooks, s.queens, s.kings,
                s.occupied_w, s.occupied_b, s.turn, s.castling_rights)

    def _bkey():
        return (board.pawns, board.knights, board.bishops, board.rooks,
                board.queens, board.kings, int(board.occupied_co[chess.WHITE]),
                int(board.occupied_co[chess.BLACK]), board.turn,
                board.castling_rights)

  # The history window covers stack indices [earliest_si .. stack_len-1] + current board.
  # earliest_si is the index corresponding to hist_idx = n_steps - 1.
  # For hist_idx k (k>=1), si = stack_len - k.
  # So earliest_si = stack_len - (n_steps - 1) for the last history step (if n_steps > 1).
    earliest_si = max(0, stack_len - (n_steps - 1)) if n_steps > 1 else stack_len

  # Pre-build set of all keys BEFORE the history window
    seen: set = set()
    for i in range(earliest_si):
        seen.add(_skey(stack[i]))

  # Check each history position (from oldest to newest), adding to seen as we go
    for hist_idx in range(n_steps - 1, -1, -1):
        if hist_idx == 0:
            key = _bkey()
        else:
            key = _skey(stack[stack_len - hist_idx])

        if key in seen:
            out[rep_base + hist_idx * rep_stride, :, :] = 1.0

        seen.add(key)
