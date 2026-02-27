from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import chess

from chess_anti_engine.utils.bitboards import bitboard_to_plane, file_to_plane


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


def _encode_piece_planes(board: chess.Board) -> list[np.ndarray]:
    """12 planes for a single position: 6 piece types × {us, them}.

    Orientation and us/them are relative to this board's side-to-move.
    """
    turn = board.turn
    us = turn
    them = not turn

    planes: list[np.ndarray] = []
    for color in (us, them):
        for pt in PIECE_TYPES:
            bb = board.pieces_mask(pt, color)
            planes.append(bitboard_to_plane(bb, turn=turn))
    return planes


def encode_lc0_reduced(board: chess.Board) -> np.ndarray:
    """Encode current position + minimal metadata into (C,8,8) float32."""
    turn = board.turn
    us = turn
    them = not turn

    planes: list[np.ndarray] = []
    planes.extend(_encode_piece_planes(board))

    # Castling rights: us-K, us-Q, them-K, them-Q
    castling_flags = [
        board.has_kingside_castling_rights(us),
        board.has_queenside_castling_rights(us),
        board.has_kingside_castling_rights(them),
        board.has_queenside_castling_rights(them),
    ]
    for has in castling_flags:
        planes.append(np.ones((8, 8), dtype=np.float32) if has else np.zeros((8, 8), dtype=np.float32))

    # En passant file
    if board.ep_square is None:
        planes.append(np.zeros((8, 8), dtype=np.float32))
    else:
        ep_file = chess.square_file(board.ep_square)
        planes.append(file_to_plane(ep_file).astype(np.float32))

    # Rule50 (halfmove clock) normalized
    rule50 = min(float(board.halfmove_clock), 100.0) / 100.0
    planes.append(np.full((8, 8), rule50, dtype=np.float32))

    # Repetition marker (current position repeated at least once)
    rep = 1.0 if (board.is_repetition(2) or board.is_repetition(3)) else 0.0
    planes.append(np.full((8, 8), rep, dtype=np.float32))

    # All-ones bias
    planes.append(np.ones((8, 8), dtype=np.float32))

    out = np.stack(planes, axis=0).astype(np.float32, copy=False)
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

    planes: list[np.ndarray] = []

    # Build history boards by popping moves from a copy (keeps move stack for repetition).
    b = board.copy(stack=True)
    history_boards: list[chess.Board] = [b.copy(stack=True)]
    for _ in range(history_len - 1):
        if not b.move_stack:
            break
        b.pop()
        history_boards.append(b.copy(stack=True))

    # Per-history piece planes
    for hb in history_boards:
        planes.extend(_encode_piece_planes(hb))

    # Pad missing history with zeros
    missing = history_len - len(history_boards)
    if missing > 0:
        planes.extend([np.zeros((8, 8), dtype=np.float32) for _ in range(missing * 12)])

    # Metadata planes are from the CURRENT position's perspective
    turn = board.turn
    us = turn
    them = not turn

    # Castling rights: us-K, us-Q, them-K, them-Q
    castling_flags = [
        board.has_kingside_castling_rights(us),
        board.has_queenside_castling_rights(us),
        board.has_kingside_castling_rights(them),
        board.has_queenside_castling_rights(them),
    ]
    for has in castling_flags:
        planes.append(np.ones((8, 8), dtype=np.float32) if has else np.zeros((8, 8), dtype=np.float32))

    # En passant file
    if board.ep_square is None:
        planes.append(np.zeros((8, 8), dtype=np.float32))
    else:
        ep_file = chess.square_file(board.ep_square)
        planes.append(file_to_plane(ep_file).astype(np.float32))

    # Color to move plane (after orientation this is always ones for side-to-move)
    planes.append(np.ones((8, 8), dtype=np.float32))

    # Rule50
    rule50 = min(float(board.halfmove_clock), 100.0) / 100.0
    planes.append(np.full((8, 8), rule50, dtype=np.float32))

    # Repetition planes per history step
    for hb in history_boards:
        rep = 1.0 if (hb.is_repetition(2) or hb.is_repetition(3)) else 0.0
        planes.append(np.full((8, 8), rep, dtype=np.float32))
    if missing > 0:
        planes.extend([np.zeros((8, 8), dtype=np.float32) for _ in range(missing)])

    # All-ones bias
    planes.append(np.ones((8, 8), dtype=np.float32))

    out = np.stack(planes, axis=0).astype(np.float32, copy=False)
    assert out.shape == (spec.num_planes, 8, 8), out.shape
    return out
