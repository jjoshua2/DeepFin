"""History and logits boundary for the opt-in Bend search experiment.

Bend owns chess/search. This deliberately conservative external evaluator
reconstructs history and validates requests before using the existing C encoder.
No board-only encoding cache: equal boards can have different neural inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import numpy as np

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.moves.encode import (
    COMPACT_POLICY_SIZE, FULL_TO_COMPACT_POLICY, POLICY_SIZE, move_to_index,
)
from native.bend_engine.legal_probe import run_probe as rules
from native.bend_engine.session_probe.run_probe import move_key

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class Encoding:
    input_history_encoding: str
    input_extra_features: str
    history_rep_fix: bool

    def __post_init__(self) -> None:
        # Explicit supported conventions; never silently normalize a typo/default.
        if self.input_history_encoding not in ('lc0_root', 'lc0_root_legacy_meta'):
            raise ValueError('unsupported history encoding')
        if self.input_extra_features not in ('v1', 'v2_threats'):
            raise ValueError('unsupported extra-feature encoding')
        if type(self.history_rep_fix) is not bool:
            raise ValueError('history_rep_fix must be boolean')

    @property
    def channels(self) -> int:
        return 146 if self.input_extra_features == 'v1' else 175


def board_position(b: chess.Board) -> rules.Position:
    rights = sum(1 << i for i, sq in enumerate((7, 0, 63, 56)) if b.castling_rights & 1 << sq)
    return (b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
            b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK], int(b.turn),
            rights, 64 if b.ep_square is None else b.ep_square)


def decode_key(board: chess.Board, key: int) -> chess.Move:
    if type(key) is not int or not 0 <= key < 1 << 17:
        raise ValueError('packed move exceeds private encoding')
    src, dst, promotion, flag = key & 63, (key >> 6) & 63, (key >> 12) & 7, key >> 15
    if src == dst or promotion > 4 or flag > 2:
        raise ValueError('invalid packed move fields')
    move = chess.Move(src, dst, promotion + 1 if promotion else None)
    if not board.is_legal(move):
        raise ValueError('illegal move in evaluator request')
    actual_flag = 2 if board.is_castling(move) else int(board.is_en_passant(move))
    if flag != actual_flag:
        raise ValueError('incorrect special-move flag')
    return move


class HistoryEncoder:
    """Own an immutable root with pre-root history; reconstruct each requested path."""
    def __init__(self, root: chess.Board, encoding: Encoding):
        if root.chess960 or not root.is_valid() or root.halfmove_clock > 255:
            raise ValueError('root must be valid orthodox chess with rule50 <= 255')
        self.encoding = encoding
        rep_fix.apply(encoding.history_rep_fix)  # before any CBoard is constructed
        self.root = root.copy(stack=True)
        self.croot = CBoard.from_board(self.root)

    def encode(self, path: Sequence[int], supplied: rules.Position,
               actions: Sequence[int]) -> tuple[np.ndarray, np.ndarray, chess.Board]:
        if len(path) > 32:
            raise ValueError('search path exceeds depth limit')
        board, cb = self.root.copy(stack=True), self.croot.copy()
        for key in path:
            move = decode_key(board, key)
            if board.halfmove_clock == 255 and not board.is_zeroing(move):
                raise ValueError('history rule50 exceeds CBoard storage')
            cb.push_index(move_to_index(move, board))
            board.push(move)
        if board_position(board) != supplied:
            raise ValueError('leaf board does not match root and path')
        if not 1 <= len(actions) <= 256 or len(set(actions)) != len(actions):
            raise ValueError('empty, oversized or duplicate legal actions')
        moves = [decode_key(board, key) for key in actions]
        if set(moves) != set(board.legal_moves):
            raise ValueError('incomplete legal action set')
        full = np.asarray([move_to_index(m, board) for m in moves], dtype=np.int64)
        if set(full.tolist()) != set(cb.legal_move_indices().tolist()):
            raise ValueError('CBoard and request action mapping disagree')
        compact = FULL_TO_COMPACT_POLICY[full]
        if np.any(compact < 0) or len(set(compact.tolist())) != len(actions):
            raise ValueError('invalid or duplicate compact policy mapping')
        x = np.asarray(encode_cboard(cb,
            input_history_encoding=self.encoding.input_history_encoding,
            input_extra_features=self.encoding.input_extra_features), dtype=np.float32)
        if x.shape != (self.encoding.channels, 8, 8) or not np.isfinite(x).all():
            raise ValueError('invalid encoded neural input')
        return np.ascontiguousarray(x[None]), full, board


def probabilities(policy: np.ndarray, wdl: np.ndarray, full_actions: np.ndarray) -> tuple[list[float], list[float]]:
    """Select legal logits in request order, then softmax. WDL is also logits."""
    policy, wdl = np.asarray(policy), np.asarray(wdl)
    if policy.shape not in ((1, COMPACT_POLICY_SIZE), (1, POLICY_SIZE)) or wdl.shape != (1, 3):
        raise ValueError('unexpected policy/WDL shapes')
    if not np.isfinite(policy).all() or not np.isfinite(wdl).all():
        raise ValueError('nonfinite neural output')
    if (full_actions.ndim != 1 or not 1 <= full_actions.size <= 256
            or full_actions.dtype.kind not in 'iu' or np.any(full_actions < 0)
            or np.any(full_actions >= POLICY_SIZE) or len(set(full_actions.tolist())) != full_actions.size):
        raise ValueError('invalid neural action indices')
    indices = FULL_TO_COMPACT_POLICY[full_actions] if policy.shape[1] == COMPACT_POLICY_SIZE else full_actions
    if np.any(indices < 0):
        raise ValueError('action has no compact policy slot')
    def softmax(values: np.ndarray) -> list[float]:
        # Float64 for stable host reduction; wire/Tree inputs explicitly round to F32.
        v = np.asarray(values, dtype=np.float64)
        exp = np.exp(v - v.max())
        return np.asarray(exp / exp.sum(), dtype=np.float32).tolist()
    return softmax(wdl[0]), softmax(policy[0, indices])


def legal_keys(board: chess.Board) -> list[int]:
    return [move_key((m.from_square, m.to_square, m.promotion - 1 if m.promotion else 0,
                     2 if board.is_castling(m) else int(board.is_en_passant(m)))) for m in board.legal_moves]
