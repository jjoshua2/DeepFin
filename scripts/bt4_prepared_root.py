"""Private, owned root transfers shared by the canonical actor and evaluator.

Public views never expose a mutable alias of the backend's history, legal moves,
array metadata or storage. Underscored accessors are for the two owning adapters,
not an attestation or a replacement for independent backend re-encoding.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import chess
import numpy as np

from chess_anti_engine.moves.encode import (
    COMPACT_POLICY_SIZE,
    POLICY_SIZE,
    compact_policy_index,
    move_to_index,
)
from chess_anti_engine.moves.leela_index import leela_index_for_move
from scripts.bt4_policy_dump import _dense_legal_policy, _legal_probabilities_from_indices


def copy_history_board(board: chess.Board) -> chess.Board:
    """Copy full history, including mutable Python-chess restoration states."""
    owned = board.copy(stack=True)
    # Board.copy already copies Move objects, but shares restoration records.
    owned._stack = [copy(state) for state in board._stack]
    return owned


def board_history_identity(board: chess.Board) -> tuple[object, ...]:
    """Exact scalar history identity; FEN alone omits the restoration stack."""
    return (
        board.chess960, board.fen(en_passant="fen", promoted=True),
        tuple((move.from_square, move.to_square, move.promotion, move.drop)
              for move in board.move_stack),
        tuple(tuple(sorted(vars(state).items())) for state in board._stack),
    )


def immutable_array(array: np.ndarray) -> np.ndarray:
    """A private copy backed by immutable bytes, not a reversible write flag."""
    arr = np.asarray(array)
    return np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class BT4LegalSnapshot:
    """One Python-enumerated move order and its three policy index spaces."""

    __moves: tuple[chess.Move, ...]
    az_indices: tuple[int, ...]
    __compact_bytes: bytes
    __leela_bytes: bytes
    __mask_bytes: bytes

    def __init__(self, board: chess.Board) -> None:
        moves = tuple(board.legal_moves)
        if not moves:
            raise ValueError("BT4 root has no legal move")
        az = tuple(move_to_index(move, board) for move in moves)
        compact = np.asarray([compact_policy_index(index) for index in az], dtype=np.int64)
        leela = np.asarray([leela_index_for_move(board, move) for move in moves], dtype=np.int64)
        if (
            len(set(az)) != len(moves)
            or any(index < 0 or index >= POLICY_SIZE for index in az)
            or len(set(compact.tolist())) != len(moves)
            or bool(np.any(compact < 0)) or bool(np.any(compact >= COMPACT_POLICY_SIZE))
            or len(set(leela.tolist())) != len(moves)
            or bool(np.any(leela < 0)) or bool(np.any(leela >= COMPACT_POLICY_SIZE))
        ):
            raise ValueError("BT4 root has no distinct valid legal policy slots")
        mask = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.bool_)
        mask[compact] = True
        object.__setattr__(self, "_BT4LegalSnapshot__moves", moves)
        object.__setattr__(self, "az_indices", az)
        object.__setattr__(self, "_BT4LegalSnapshot__compact_bytes", compact.tobytes())
        object.__setattr__(self, "_BT4LegalSnapshot__leela_bytes", leela.tobytes())
        object.__setattr__(self, "_BT4LegalSnapshot__mask_bytes", mask.tobytes())

    @property
    def moves(self) -> tuple[chess.Move, ...]:
        return tuple(copy(move) for move in self.__moves)

    @property
    def compact_indices(self) -> np.ndarray:
        return np.frombuffer(self.__compact_bytes, dtype=np.int64)

    @property
    def leela_indices(self) -> np.ndarray:
        return np.frombuffer(self.__leela_bytes, dtype=np.int64)

    @property
    def legal_mask(self) -> np.ndarray:
        return np.frombuffer(self.__mask_bytes, dtype=np.bool_)

    def _move(self, index: int) -> chess.Move:
        # A validated Move, not a permissive AZ decoder, supplies the commit.
        return copy(self.__moves[index])

    def _dense_policy(self, fen: str, policy_row: np.ndarray) -> np.ndarray:
        raw = _legal_probabilities_from_indices(fen, policy_row, self.leela_indices)
        probabilities = np.asarray(raw, dtype=np.float32)
        return _dense_legal_policy(probabilities, self.compact_indices)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class BT4PreparedInput:
    """An owned internal row; construction is limited to canonical preparation.

    A public board view is a defensive full-history copy. Array views have fresh
    metadata and immutable backing, so changing shape cannot change the payload.
    Python private-member reflection is not a security boundary. The evaluator
    still reconstructs every verification CBoard from its owned Python history.
    """

    slot_id: int
    generation: int
    fen: str
    input_key: str
    source_key: bytes
    model_sha256: str
    input_history_encoding: str
    input_extra_features: str
    history_rep_fix: bool
    legal: BT4LegalSnapshot
    __board: chess.Board
    __identity: tuple[object, ...]
    __x_shape: tuple[int, ...]
    __x_bytes: bytes

    def __init__(self) -> None:
        raise TypeError("BT4 prepared inputs are issued by canonical preparation")

    @classmethod
    def _issue(
        cls, *, slot_id: int, generation: int, board: chess.Board, x: np.ndarray,
        input_key: str, source_key: bytes, model_sha256: str,
        input_history_encoding: str, input_extra_features: str, history_rep_fix: bool,
    ) -> BT4PreparedInput:
        # This snapshot is owned exclusively here, not by the live game or the
        # public PreparedRoot. No CBoard is transferred to the verifier.
        owned = copy_history_board(board)
        payload = object.__new__(cls)
        values = {
            "slot_id": slot_id, "generation": generation, "fen": owned.fen(),
            "input_key": input_key, "source_key": source_key,
            "model_sha256": model_sha256, "input_history_encoding": input_history_encoding,
            "input_extra_features": input_extra_features, "history_rep_fix": history_rep_fix,
            "legal": BT4LegalSnapshot(owned), "_BT4PreparedInput__board": owned,
            "_BT4PreparedInput__identity": board_history_identity(owned),
            "_BT4PreparedInput__x_shape": x.shape,
            "_BT4PreparedInput__x_bytes": x.tobytes(),
        }
        for name, value in values.items():
            object.__setattr__(payload, name, value)
        return payload

    @property
    def board(self) -> chess.Board:
        return copy_history_board(self.__board)

    @property
    def x(self) -> np.ndarray:
        return np.frombuffer(self.__x_bytes, dtype=np.float32).reshape(self.__x_shape)

    def _borrow_board(self) -> chess.Board:
        """Read-only adapter access. Never return this through a public view."""
        return self.__board

    def _matches_board(self, board: chess.Board) -> bool:
        return board_history_identity(board) == self.__identity

    def _input_bytes(self) -> bytes:
        return self.__x_bytes
