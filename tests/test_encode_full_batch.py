"""Parity and validation tests for encode_full_batch / encode_cboard_batch."""
from __future__ import annotations

from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard, encode_full_batch
from chess_anti_engine.encoding.cboard_encode import encode_cboard, encode_cboard_batch
from chess_anti_engine.encoding.lc0 import (
    LC0_HISTORY_LEGACY,
    LC0_HISTORY_ROOT,
    LC0_HISTORY_ROOT_LEGACY_META,
)


def _boards_with_history() -> list[CBoard]:
    """Several positions including non-trivial move history for history planes."""
    out: list[CBoard] = []

    start = chess.Board()
    out.append(CBoard.from_board(start))

    mid = chess.Board()
    for san in ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"):
        mid.push_san(san)
    out.append(CBoard.from_board(mid))

    # Second game with different history length / castling / EP potential
    g2 = chess.Board()
    for san in ("d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O"):
        g2.push_san(san)
    out.append(CBoard.from_board(g2))

    # EP-capable position
    ep = chess.Board()
    for san in ("e4", "a6", "e5", "d5"):
        ep.push_san(san)
    assert ep.ep_square is not None
    out.append(CBoard.from_board(ep))

    return out


_HIST_ENCODINGS = (
    None,  # full/legacy history (hist_mode 0)
    LC0_HISTORY_LEGACY,
    LC0_HISTORY_ROOT,
    LC0_HISTORY_ROOT_LEGACY_META,
)
_EXTRA_FEATURES = ("v1", "v2_threats")


@pytest.mark.parametrize("input_history_encoding", _HIST_ENCODINGS)
@pytest.mark.parametrize("input_extra_features", _EXTRA_FEATURES)
def test_encode_cboard_batch_matches_per_board_stack(
    input_history_encoding: str | None,
    input_extra_features: str,
) -> None:
    boards = _boards_with_history()
    batched = encode_cboard_batch(
        boards,
        input_history_encoding=input_history_encoding,
        input_extra_features=input_extra_features,
    )
    stacked = np.stack(
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
    assert batched.shape == stacked.shape
    assert np.array_equal(batched, stacked), (
        f"batch/single mismatch hist={input_history_encoding!r} "
        f"extra={input_extra_features!r}"
    )


def test_encode_cboard_batch_empty() -> None:
    empty_v1 = encode_cboard_batch([], input_extra_features="v1")
    assert empty_v1.shape == (0, 146, 8, 8)
    assert empty_v1.dtype == np.float32

    empty_v2 = encode_cboard_batch([], input_extra_features="v2_threats")
    assert empty_v2.shape == (0, 175, 8, 8)
    assert empty_v2.dtype == np.float32


def test_encode_full_batch_raw_matches_encode_full() -> None:
    boards = _boards_with_history()
    for hist_mode in (0, 1, 2):
        for n_extra in (34, 63):
            n_planes = 112 + n_extra
            out = np.empty((len(boards), n_planes, 8, 8), dtype=np.float32)
            # Dirty buffer must still be bit-identical to fresh encode_full.
            out.fill(0.5)
            encode_full_batch(boards, out, hist_mode, n_extra)
            for i, cb in enumerate(boards):
                ref = cb.encode_full(hist_mode, n_extra)
                assert np.array_equal(out[i], ref), (
                    f"raw C batch mismatch hist_mode={hist_mode} n_extra={n_extra} i={i}"
                )


def test_encode_full_batch_accepts_tuple() -> None:
    cb = CBoard.from_board(chess.Board())
    out = np.empty((1, 146, 8, 8), dtype=np.float32)
    encode_full_batch((cb,), out, 0, 34)
    assert np.array_equal(out[0], cb.encode_full(0, 34))


def test_encode_full_batch_error_paths() -> None:
    cb = CBoard.from_board(chess.Board())
    good = np.empty((1, 146, 8, 8), dtype=np.float32)

    bad_dtype = cast(Any, np.empty((1, 146, 8, 8), dtype=np.float64))
    with pytest.raises(ValueError, match="dtype float32"):
        encode_full_batch([cb], bad_dtype, 0, 34)

    with pytest.raises(ValueError, match="ndim 4"):
        encode_full_batch([cb], cast(Any, np.empty((1, 146, 8), dtype=np.float32)), 0, 34)

    with pytest.raises(ValueError, match="shape\\[0\\]"):
        encode_full_batch([cb], np.empty((2, 146, 8, 8), dtype=np.float32), 0, 34)

    with pytest.raises(ValueError, match="shape\\[1\\]"):
        encode_full_batch([cb], np.empty((1, 175, 8, 8), dtype=np.float32), 0, 34)

    strided = np.zeros((1, 146, 8, 16), dtype=np.float32)[:, :, :, ::2]
    assert not strided.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        encode_full_batch([cb], strided, 0, 34)

    readonly = np.empty((1, 146, 8, 8), dtype=np.float32)
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        encode_full_batch([cb], readonly, 0, 34)

    fake = cast(Any, type("NotCBoard", (), {})())
    with pytest.raises(TypeError, match="CBoard"):
        encode_full_batch([fake], good, 0, 34)

    with pytest.raises(ValueError, match="hist_mode"):
        encode_full_batch([cb], good, 3, 34)

    with pytest.raises(ValueError, match="n_extra"):
        encode_full_batch([cb], good, 0, 99)
