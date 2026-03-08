from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import chess

from chess_anti_engine.utils.bitboards import bitboard_to_plane, file_to_plane, write_bitboard_to_plane


PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


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


LC0_REDUCED = LC0ReducedPlaneSpec()
LC0_FULL = LC0FullPlaneSpec()


def _write_piece_planes(board: chess.Board, out: np.ndarray, start: int) -> int:
    """Write 12 piece planes for one position starting at `start`.

    Orientation and us/them are relative to this board's side-to-move.
    """
    turn = board.turn
    us = turn
    them = not turn

    idx = int(start)
    for color in (us, them):
        for pt in PIECE_TYPES:
            bb = board.pieces_mask(pt, color)
            write_bitboard_to_plane(out[idx], bb, turn=turn)
            idx += 1
    return idx


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


def encode_lc0_full(board: chess.Board, *, history_len: int = 8) -> np.ndarray:
    """Encode LC0 112-plane input (8 history positions × 12 + metadata planes).

    Each history position is encoded relative to THAT position's side-to-move.
    This follows the spec statement that orientation is from side-to-move.

    If there are fewer than `history_len` plies available, remaining history
    planes are zero.
    """
    spec = LC0FullPlaneSpec(history_len=history_len)
    out = np.zeros((spec.num_planes, 8, 8), dtype=np.float32)

    # Build history boards by popping moves from a copy (keeps move stack for repetition).
    b = board.copy(stack=True)
    rep_base = spec.num_piece_planes + spec.num_castling_planes + spec.num_ep_planes + spec.num_turn_planes + spec.num_rule50_planes
    history_count = 0
    for hist_idx in range(history_len):
        _write_piece_planes(b, out, hist_idx * spec.piece_planes_per_history)
        if b.is_repetition(2) or b.is_repetition(3):
            out[rep_base + hist_idx, :, :] = 1.0
        history_count += 1
        if not b.move_stack:
            break
        b.pop()

    # Metadata planes are from the CURRENT position's perspective
    turn = board.turn
    us = turn
    them = not turn
    meta_idx = spec.num_piece_planes

    # Castling rights: us-K, us-Q, them-K, them-Q
    castling_flags = [
        board.has_kingside_castling_rights(us),
        board.has_queenside_castling_rights(us),
        board.has_kingside_castling_rights(them),
        board.has_queenside_castling_rights(them),
    ]
    for has in castling_flags:
        if has:
            out[meta_idx, :, :] = 1.0
        meta_idx += 1

    # En passant file
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        out[meta_idx, :, :] = file_to_plane(ep_file)
    meta_idx += 1

    # Color to move plane (after orientation this is always ones for side-to-move)
    out[meta_idx, :, :] = 1.0
    meta_idx += 1

    # Rule50
    rule50 = min(float(board.halfmove_clock), 100.0) / 100.0
    out[meta_idx, :, :] = rule50
    meta_idx += 1

    # All-ones bias
    out[spec.num_planes - 1, :, :] = 1.0

    assert out.shape == (spec.num_planes, 8, 8), out.shape
    return out
