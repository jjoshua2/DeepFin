"""Bind a decided Ceres actor root to its stored-float16 model input.

The caller must run its natural/rule50/Syzygy outcome gate before this function.
The stored input is the single source of the 137-byte Ceres feed; independent
game readback can still reconstruct the feed directly from the board.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b, sha256

import chess
import numpy as np

from chess_anti_engine.encoding.ceres_tpg import stored_x_to_ceres_tpg_bytes
from chess_anti_engine.eval.rvg_surgery import position_fingerprints
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import compact_index_for_move, leela_index_for_move

HISTORY = "lc0_root_legacy_meta"
EXTRA = "v2_threats"


@dataclass(frozen=True)
class PreparedCeresRootInput:
    slot_id: int
    ply_index: int
    fen: str
    move_stack: tuple[chess.Move, ...]
    legal_moves: tuple[chess.Move, ...]
    compact_indices: tuple[int, ...]
    leela_indices: tuple[int, ...]
    x_stored: np.ndarray
    x_stored_sha256: str
    input_key_float32: str
    stored_input_key: str
    position_fingerprint: bytes
    feed: np.ndarray
    feed_sha256: str
    input_history_encoding: str = HISTORY
    input_extra_features: str = EXTRA
    history_rep_fix: bool = True


def _input_key(x: np.ndarray) -> str:
    return blake2b(np.ascontiguousarray(x, dtype=np.float32).tobytes(),
                   digest_size=16).hexdigest()


def bind_undecided_ceres_root(
    board: chess.Board, x_live: np.ndarray, *, slot_id: int, ply_index: int,
    input_history_encoding: str, input_extra_features: str, history_rep_fix: bool,
) -> PreparedCeresRootInput:
    """Freeze legal maps, stored input and its feed after a strict root decision.

    ``x_live`` must have been encoded from this full-history board under the
    named profile. The caller owns that provenance; this binder checks shape,
    type, profile and live/stored position identity, but cannot prove legality
    of a caller-supplied tensor. A separate whole-game reader must do so.
    """
    if (type(slot_id) is not int or not 0 <= slot_id < 2**32
            or type(ply_index) is not int or ply_index < 0):
        raise ValueError("invalid Ceres root slot/ply identity")
    if (input_history_encoding != HISTORY or input_extra_features != EXTRA
            or history_rep_fix is not True):
        raise ValueError("unsupported Ceres root input profile")
    x = np.asarray(x_live)
    if x.shape != (175, 8, 8) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("Ceres live input shape/dtype/finite values differ")
    moves = tuple(board.legal_moves)
    if not moves:
        raise ValueError("undecided Ceres root has no legal moves")
    compact = tuple(compact_index_for_move(board, move) for move in moves)
    leela = tuple(leela_index_for_move(board, move) for move in moves)
    if (any(not 0 <= idx < COMPACT_POLICY_SIZE for idx in compact + leela)
            or len(set(compact)) != len(moves) or len(set(leela)) != len(moves)):
        raise ValueError("Ceres legal compact/Leela indices invalid or duplicated")
    x_stored = np.frombuffer(x.astype(np.float16).tobytes(order="C"),
                             dtype=np.float16).reshape(175, 8, 8)
    live_position = position_fingerprints(x[None], input_history_encoding=HISTORY)[0]
    stored_position = position_fingerprints(
        x_stored[None], input_history_encoding=HISTORY)[0]
    if live_position != stored_position:
        raise ValueError("Ceres stored position fingerprint differs from live input")
    stored_feed = stored_x_to_ceres_tpg_bytes(
        x_stored, input_history_encoding=HISTORY, history_rep_fix=True)
    if stored_feed.shape != (64, 137) or stored_feed.dtype != np.uint8:
        raise ValueError("Ceres stored feed shape/dtype differs")
    feed = np.frombuffer(stored_feed.tobytes(order="C"),
                         dtype=np.uint8).reshape(64, 137)
    return PreparedCeresRootInput(
        slot_id, ply_index, board.fen(), tuple(board.move_stack), moves,
        compact, leela, x_stored, sha256(x_stored.tobytes()).hexdigest(),
        _input_key(x), _input_key(x_stored), live_position,
        feed, sha256(feed.tobytes()).hexdigest(),
    )
