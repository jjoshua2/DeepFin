"""Test coordinator bindings to native encoding and CPU AOTI. No production routing."""

from __future__ import annotations

from contextlib import AbstractContextManager
import ctypes as ct
import hashlib
import json
from pathlib import Path
from typing import Self

import chess
import numpy as np

from chess_anti_engine.moves.encode import FULL_TO_COMPACT_POLICY
from native.bend_engine.legal_probe import run_probe as rules
from native.bend_engine.session_probe.run_probe import move_key

U32 = ct.POINTER(ct.c_uint32)
U64 = ct.POINTER(ct.c_uint64)
F32 = ct.POINTER(ct.c_float)
I32 = ct.POINTER(ct.c_int32)
ERROR_SIZE = 1024


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key_for(board: chess.Board, move: chess.Move) -> int:
    if move not in board.legal_moves:
        raise ValueError("illegal game/path move")
    flag = 2 if board.is_castling(move) else int(board.is_en_passant(move))
    return move_key(
        (
            move.from_square,
            move.to_square,
            move.promotion - 1 if move.promotion else 0,
            flag,
        )
    )


def decode_key(board: chess.Board, key: int) -> chess.Move:
    if not 0 <= key < 1 << 17:
        raise ValueError("packed action exceeds format")
    promotion = (key >> 12) & 7
    if promotion > 4:
        raise ValueError("invalid packed promotion")
    move = chess.Move(key & 63, (key >> 6) & 63, promotion + 1 if promotion else None)
    if key_for(board, move) != key:
        raise ValueError("packed action flag mismatch")
    return move


def position(board: chess.Board) -> rules.Position:
    return rules.fen_position(board.fen(en_passant="fen"))


def mapping_hash() -> str:
    return hashlib.sha256(
        np.asarray(FULL_TO_COMPACT_POLICY, dtype="<i4").tobytes()
    ).hexdigest()


def contract(package: Path) -> dict:
    """Require an explicit versioned CPU tuple package, not arbitrary output order."""
    data = json.loads(package.with_suffix(".json").read_text())
    if not isinstance(data, dict):
        raise ValueError("package contract must be an object")
    if (
        data.get("format") != "deepfin-bend-cpu-tuple-v1"
        or data.get("device") != "cpu"
        or data.get("batch") != 1
    ):
        raise ValueError("unsupported native package contract")
    if data.get("planes") not in (146, 175, 179) or data.get("dtype") not in (
        "float32",
        "bfloat16",
    ):
        raise ValueError("unsupported native input contract")
    if (
        type(data.get("history_mode")) is not int
        or data.get("history_mode") not in (0, 1, 2)
        or type(data.get("history_rep_fix")) is not bool
    ):
        raise ValueError("explicit history semantics required")
    if (
        data.get("package_sha256") != sha256(package)
        or data.get("policy_map_sha256") != mapping_hash()
    ):
        raise ValueError("package or policy map fingerprint mismatch")
    if data.get("outputs") != ["compact_policy_logits", "wdl_logits"] or not isinstance(
        data.get("output_spec"), str
    ):
        raise ValueError("explicit policy/WDL output contract required")
    spec = json.loads(data["output_spec"])
    leaf = {"type": None, "context": None, "children_spec": []}
    if spec != [
        1,
        {"type": "builtins.tuple", "context": "null", "children_spec": [leaf, leaf]},
    ]:
        raise ValueError("package must return policy/WDL tensor tuple")
    return data


class Features(AbstractContextManager):
    """One immutable replay root; native calls validate every leaf and action set."""

    def __init__(
        self,
        library: Path,
        seed: chess.Board,
        history: list[chess.Move],
        *,
        mode: int,
        extra: int,
        rep_fix: bool,
    ):
        self.lib = ct.CDLL(str(library))
        self.lib.df_root_new.argtypes = [
            U64,
            U32,
            U32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            I32,
            ct.c_char_p,
            ct.c_uint32,
        ]
        self.lib.df_root_new.restype = ct.c_void_p
        self.lib.df_root_free.argtypes = [ct.c_void_p]
        self.lib.df_root_free.restype = None
        self.lib.df_prepare.argtypes = [
            ct.c_void_p,
            U32,
            ct.c_uint32,
            U64,
            U32,
            U32,
            ct.c_uint32,
            F32,
            ct.c_uint32,
            U32,
            U32,
            ct.c_char_p,
            ct.c_uint32,
        ]
        self.lib.df_prepare.restype = ct.c_int
        if seed.chess960:
            raise ValueError("only orthodox chess is supported")
        if seed.move_stack:
            raise ValueError(
                "seed must not carry hidden history; pass explicit history"
            )
        if len(history) > 128 or seed.halfmove_clock > 255:
            raise ValueError("root exceeds native history/clock bounds")
        bits = np.asarray(position(seed)[:8], dtype=np.uint64)
        meta = np.asarray(
            (*position(seed)[8:], seed.halfmove_clock, seed.ply()), dtype=np.uint32
        )
        board = seed.copy(stack=False)
        game = []
        for move in history:
            game.append(key_for(board, move))
            board.push(move)
        keys = np.asarray(game, dtype=np.uint32)
        mapping = np.ascontiguousarray(FULL_TO_COMPACT_POLICY, dtype=np.int32)
        error = ct.create_string_buffer(ERROR_SIZE)
        self.handle = self.lib.df_root_new(
            bits.ctypes.data_as(U64),
            meta.ctypes.data_as(U32),
            keys.ctypes.data_as(U32),
            len(keys),
            mode,
            extra,
            int(rep_fix),
            mapping.ctypes.data_as(I32),
            error,
            ERROR_SIZE,
        )
        if not self.handle:
            raise ValueError(error.value.decode())
        self.root, self.planes = board, 112 + extra

    def __exit__(self, *_exc: object) -> None:
        if self.handle:
            self.lib.df_root_free(self.handle)
            self.handle = None

    def __enter__(self) -> Self:
        return self

    def prepare(
        self, board: rules.Position, actions: list[int], path: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.handle:
            raise ValueError("closed feature context")
        if len(actions) > 256 or len(path) > 32:
            raise ValueError("invalid native request dimensions")
        if (
            len(board) != 11
            or any(not 0 <= x < 1 << 64 for x in board[:8])
            or any(not 0 <= x < 1 << 32 for x in board[8:])
        ):
            raise ValueError("invalid board words")
        if any(not 0 <= x < 1 << 32 for x in [*actions, *path]):
            raise ValueError("invalid action/path word")
        bits = np.asarray(board[:8], dtype=np.uint64)
        meta = np.asarray(board[8:], dtype=np.uint32)
        paths = np.asarray(path, dtype=np.uint32)
        keys = np.asarray(actions, dtype=np.uint32)
        planes = np.full((self.planes, 8, 8), np.nan, dtype=np.float32)
        full = np.zeros(max(1, len(actions)), dtype=np.uint32)
        compact = np.zeros_like(full)
        error = ct.create_string_buffer(ERROR_SIZE)
        ok = self.lib.df_prepare(
            self.handle,
            paths.ctypes.data_as(U32),
            len(paths),
            bits.ctypes.data_as(U64),
            meta.ctypes.data_as(U32),
            keys.ctypes.data_as(U32),
            len(keys),
            planes.ctypes.data_as(F32),
            planes.size,
            full.ctypes.data_as(U32),
            compact.ctypes.data_as(U32),
            error,
            ERROR_SIZE,
        )
        if not ok:
            if not np.isnan(planes).all() or full.any() or compact.any():
                raise AssertionError("failed preparation partially committed outputs")
            raise ValueError(error.value.decode())
        return planes, full[: len(actions)], compact[: len(actions)]


class AOTI(AbstractContextManager):
    """Native model and legal softmax, invoked only by the test coordinator."""

    def __init__(self, library: Path, package: Path):
        self.contract = contract(package)
        self.lib = ct.CDLL(str(library))
        self.lib.df_aoti_open.argtypes = [
            ct.c_char_p,
            ct.c_char_p,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_char_p,
            ct.c_uint32,
        ]
        self.lib.df_aoti_open.restype = ct.c_void_p
        self.lib.df_aoti_close.argtypes = [ct.c_void_p]
        self.lib.df_aoti_close.restype = None
        self.lib.df_aoti_run.argtypes = [
            ct.c_void_p,
            F32,
            ct.c_uint32,
            U32,
            ct.c_uint32,
            F32,
            F32,
            F32,
            F32,
            ct.c_char_p,
            ct.c_uint32,
        ]
        self.lib.df_aoti_run.restype = ct.c_int
        error = ct.create_string_buffer(ERROR_SIZE)
        self.handle = self.lib.df_aoti_open(
            str(package).encode(),
            self.contract["output_spec"].encode(),
            self.contract["planes"],
            int(self.contract["dtype"] == "bfloat16"),
            error,
            ERROR_SIZE,
        )
        if not self.handle:
            raise ValueError(error.value.decode())

    def __exit__(self, *_exc: object) -> None:
        if self.handle:
            self.lib.df_aoti_close(self.handle)
            self.handle = None

    def __enter__(self) -> Self:
        return self

    def evaluate(
        self, planes: np.ndarray, compact: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.handle:
            raise ValueError("closed evaluator")
        if (
            compact.ndim != 1
            or compact.dtype.kind not in "iu"
            or compact.size > 256
            or np.any(compact < 0)
            or np.any(compact > 0xFFFFFFFF)
        ):
            raise ValueError("invalid compact policy words")
        inputs = np.ascontiguousarray(planes, dtype=np.float32)
        indices = np.ascontiguousarray(compact, dtype=np.uint32)
        rawp = np.full(1858, np.nan, dtype=np.float32)
        rawv = np.full(3, np.nan, dtype=np.float32)
        p = np.full(len(indices), np.nan, dtype=np.float32)
        v = np.full(3, np.nan, dtype=np.float32)
        error = ct.create_string_buffer(ERROR_SIZE)
        ok = self.lib.df_aoti_run(
            self.handle,
            inputs.ctypes.data_as(F32),
            inputs.size,
            indices.ctypes.data_as(U32),
            len(indices),
            rawp.ctypes.data_as(F32),
            rawv.ctypes.data_as(F32),
            p.ctypes.data_as(F32),
            v.ctypes.data_as(F32),
            error,
            ERROR_SIZE,
        )
        if not ok:
            if any(not np.isnan(a).all() for a in (rawp, rawv, p, v)):
                raise AssertionError("failed AOTI call partially committed outputs")
            raise ValueError(error.value.decode())
        return v, p, rawp, rawv
