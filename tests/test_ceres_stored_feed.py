"""Physical stored-row Ceres feeds agree with the independent board encoder."""
from __future__ import annotations

import hashlib

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.ceres_stored_feed import (
    CERES_ROW_BYTES,
    ceres_feed_bytes_from_stored_rows,
)
from chess_anti_engine.encoding.ceres_tpg import encode_ceres_tpg_bytes
from chess_anti_engine.encoding.encode import encode_position


def _raw(board: chess.Board) -> bytes:
    stored = encode_position(
        board, input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    ).astype("<f2")
    return stored.astype("<f4").tobytes()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _feed(raws: list[bytes], hashes: list[str], *, physical_slots: int | None = None) -> bytes:
    return ceres_feed_bytes_from_stored_rows(
        raws, hashes, input_history_encoding="lc0_root_legacy_meta",
        history_rep_fix=True, physical_slots=physical_slots,
    )


def test_stored_feed_preserves_history_ep_repetition_and_repeat_last_tail() -> None:
    board = chess.Board()
    boards = [board.copy()]
    for move in (
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
        "e2e4", "a7a6", "e4e5", "d7d5",
    ):
        board.push_uci(move)
        boards.append(board.copy())
    raws = [_raw(position) for position in boards]
    expected = b"".join(encode_ceres_tpg_bytes(position).tobytes() for position in boards)
    actual = _feed(raws, [_sha(raw) for raw in raws], physical_slots=16)
    assert actual == expected + expected[-CERES_ROW_BYTES:] * 3
    assert len(actual) == 16 * CERES_ROW_BYTES


def test_stored_feed_rejects_mismatched_or_nonroundtrip_rows() -> None:
    raw = _raw(chess.Board())
    with pytest.raises(ValueError, match="source hash"):
        _feed([raw], ["0" * 64])
    with pytest.raises(ValueError, match="equal nonzero length"):
        _feed([], [])
    with pytest.raises(ValueError, match="physical_slots"):
        _feed([raw], [_sha(raw)], physical_slots=0)
    with pytest.raises(ValueError, match="physical_slots"):
        _feed([raw], [_sha(raw)], physical_slots=True)

    with pytest.raises(ValueError, match="corrected lc0_root_legacy_meta"):
        ceres_feed_bytes_from_stored_rows(
            [raw], [_sha(raw)], input_history_encoding="legacy", history_rep_fix=True,
        )
    with pytest.raises(ValueError, match="corrected lc0_root_legacy_meta"):
        ceres_feed_bytes_from_stored_rows(
            [raw], [_sha(raw)], input_history_encoding="lc0_root_legacy_meta",
            history_rep_fix=False,
        )

    changed = np.frombuffer(raw, dtype="<f4").copy()
    changed[-1] = np.float32(0.1)
    nonroundtrip = changed.tobytes()
    with pytest.raises(ValueError, match="stored-f16 round trip"):
        _feed([nonroundtrip], [_sha(nonroundtrip)])
    changed[-1] = np.nan
    nonfinite = changed.tobytes()
    with pytest.raises(ValueError, match="nonfinite"):
        _feed([nonfinite], [_sha(nonfinite)])
