"""A saved-input Ceres actor root has one stored-f16 feed construction."""

from __future__ import annotations

import hashlib
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.ceres_tpg import encode_ceres_tpg_bytes
from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.source.ceres_root_input import bind_undecided_ceres_root


def _bind(board: chess.Board, **changes: Any):
    x = encode_position(
        board, input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats")
    kwargs: dict[str, Any] = {
        "slot_id": 10240, "ply_index": len(board.move_stack),
        "input_history_encoding": "lc0_root_legacy_meta",
        "input_extra_features": "v2_threats", "history_rep_fix": True}
    kwargs.update(changes)
    return bind_undecided_ceres_root(board, x, **kwargs)


@pytest.mark.parametrize("moves", [
    (),
    ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"),
    ("e2e4", "a7a6", "e4e5", "d7d5"),
    ("e2e4",),
])
def test_stored_feed_equals_full_history_board_tpg(moves: tuple[str, ...]) -> None:
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    bound = _bind(board)
    assert np.array_equal(bound.feed, encode_ceres_tpg_bytes(board))
    assert bound.feed_sha256 == hashlib.sha256(bound.feed.tobytes()).hexdigest()
    assert bound.x_stored_sha256 == hashlib.sha256(bound.x_stored.tobytes()).hexdigest()
    assert not bound.feed.flags.writeable
    assert not bound.x_stored.flags.writeable
    assert bound.move_stack == tuple(board.move_stack)
    assert {move.uci() for move in bound.legal_moves} == {
        move.uci() for move in board.legal_moves}


def test_profile_and_identity_fail_closed() -> None:
    board = chess.Board()
    for change in ({"input_history_encoding": "lc0_root"},
                   {"input_extra_features": "v1"},
                   {"history_rep_fix": False},
                   {"slot_id": True}, {"ply_index": -1}):
        with pytest.raises(ValueError, match=r"profile|identity"):
            _bind(board, **change)
    x = encode_position(
        board, input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats")
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="shape/dtype/finite"):
        bind_undecided_ceres_root(
            board, x, slot_id=0, ply_index=0,
            input_history_encoding="lc0_root_legacy_meta",
            input_extra_features="v2_threats", history_rep_fix=True)
