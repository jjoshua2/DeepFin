"""Adversarial validation at public native-extension boundaries."""

from __future__ import annotations

import sys
from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._features_ext import (
    compute_extra_features,
    compute_relation_matrices,
)
from chess_anti_engine.encoding._lc0_ext import (
    CBoard,
    encode_piece_planes,
    legal_move_policy_indices,
)
from chess_anti_engine.mcts._mcts_tree import (
    MCTSTree,
    batch_encode_146,
    batch_process_ply,
)
from chess_anti_engine.moves import move_to_index


def _unaligned_array(length: int, dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    """Construct a C-contiguous ndarray whose data starts one byte off alignment."""
    storage = bytearray(length * dtype.itemsize + 1)
    array = np.ndarray((length,), dtype=dtype, buffer=storage, offset=1)
    assert array.flags.c_contiguous
    assert not array.flags.aligned
    return array


def test_piece_plane_encoder_rejects_short_or_wrong_arrays() -> None:
    with pytest.raises(ValueError, match="bitboards"):
        encode_piece_planes(np.zeros(1, np.uint64), np.zeros(8, np.int32), 8)
    with pytest.raises(ValueError, match="turns"):
        encode_piece_planes(np.zeros(96, np.uint64), np.zeros(1, np.int32), 8)
    # Deliberately wrong dtype — runtime must reject; cast for the type checker.
    bad_bbs = cast(Any, np.zeros(96, np.int64))
    with pytest.raises(ValueError, match="bitboards"):
        encode_piece_planes(bad_bbs, np.zeros(8, np.int32), 8)


def test_direct_array_bindings_reject_unaligned_or_byteswapped_inputs() -> None:
    native_u64 = np.dtype(np.uint64)
    native_i32 = np.dtype(np.int32)
    swapped_u64 = native_u64.newbyteorder("S")

    with pytest.raises(ValueError, match="bitboards"):
        encode_piece_planes(
            _unaligned_array(96, native_u64), np.zeros(8, native_i32), 8,
        )
    with pytest.raises(ValueError, match="turns"):
        encode_piece_planes(
            np.zeros(96, native_u64), _unaligned_array(8, native_i32), 8,
        )
    with pytest.raises(ValueError, match="bitboards"):
        encode_piece_planes(
            np.zeros(96, swapped_u64), np.zeros(8, native_i32), 8,
        )

    pieces = np.zeros(6, native_u64)
    with pytest.raises(ValueError, match="pieces_us"):
        compute_extra_features(
            _unaligned_array(6, native_u64), pieces, 0, -1, -1, True, -1,
        )
    with pytest.raises(ValueError, match="pieces_us"):
        compute_extra_features(
            np.zeros(6, swapped_u64), pieces, 0, -1, -1, True, -1,
        )


def test_tree_input_wrappers_copy_unaligned_arrays_safely() -> None:
    tree = MCTSTree()
    root = tree.add_root(0, 0.0)
    actions = _unaligned_array(1, np.dtype(np.int32))
    priors = _unaligned_array(1, np.dtype(np.float64))
    actions[0] = 0
    priors[0] = 1.0

    tree.expand(root, actions, priors)
    found = tree.find_child(root, 0)
    assert found >= 0


def test_feature_bindings_reject_short_piece_arrays() -> None:
    short = np.zeros(1, np.uint64)
    pieces = np.zeros(6, np.uint64)
    with pytest.raises(ValueError, match="pieces_us"):
        compute_extra_features(short, pieces, 0, -1, -1, True, -1)
    with pytest.raises(ValueError, match="pieces_them"):
        compute_relation_matrices(pieces, short, 0, -1, -1, True)


def test_legal_move_binding_rejects_invalid_turn() -> None:
    args = (0, 0, 0, 0, 0, 1 << 4, 0, 0, 0, 0, 0, 1 << 60)
    with pytest.raises(ValueError, match="turn"):
        legal_move_policy_indices(*args, 2, 0, 0, 0, 0, -1)


def test_cboard_from_board_does_not_leak_bool_key_references() -> None:
    board = chess.Board()
    CBoard.from_board(board)  # warm any one-time python-chess/C-extension state
    before_true = sys.getrefcount(True)
    before_false = sys.getrefcount(False)
    for _ in range(100):
        CBoard.from_board(board)
    # CPython/pytest may retain one singleton reference while evaluating this
    # frame; the old C leak increased each count by exactly the loop count.
    assert sys.getrefcount(True) - before_true <= 1
    assert sys.getrefcount(False) - before_false <= 1


def test_tree_rejects_invalid_nodes_paths_and_actions() -> None:
    tree = MCTSTree()
    root = tree.add_root(0, 0.0)
    with pytest.raises(ValueError, match="node_id"):
        tree.expand(999_999, np.array([0], np.int32), np.array([1.0], np.float64))
    with pytest.raises(ValueError, match="policy"):
        tree.expand(root, np.array([-1], np.int32), np.array([1.0], np.float64))
    with pytest.raises(ValueError, match="path"):
        tree.backprop(np.array([], np.int32), 0.0)
    with pytest.raises(ValueError, match="path"):
        tree.walker_integrate_leaf(
            np.array([], np.int32), np.array([], np.int32),
            np.zeros(4672, np.float32), np.zeros(3, np.float32), 0,
        )


def test_batch_integrate_rejects_undersized_buffers() -> None:
    tree = MCTSTree()
    tree.add_root(0, 0.0)
    with pytest.raises(ValueError, match="buffers"):
        tree.batch_integrate_leaves(
            1, np.zeros(1, np.int32), np.ones(1, np.int32),
            np.zeros(1, np.int32), np.zeros(1, np.int32),
            np.zeros(1, np.int8), np.zeros((1, 4672), np.float32),
            np.zeros((1, 3), np.float32), 0,
        )


def test_batch_process_rejects_out_of_range_action() -> None:
    cb = CBoard.from_board(chess.Board())
    with pytest.raises(ValueError, match="policy"):
        batch_process_ply(
            [cb], np.zeros((1, 4672), np.float32), np.zeros((1, 3), np.float32),
            np.array([-1], np.int32), np.zeros(1, np.float64),
            np.zeros((1, 4672), np.float32), 0, 1.0, 1.0, 0.0, 1.0,
        )


def test_batch_process_ply_accepts_batch_larger_than_max_legal_moves() -> None:
    """n is slot batch size, not legal-move count — must not cap at 256.

    Production selfplay uses worker batches up to 512; reusing the per-node
    legal-list validator would raise ValueError for any network turn with
    more than CBOARD_MAX_LEGAL_MOVES live slots.
    """
    n = 257
    board = chess.Board()
    action = int(move_to_index(chess.Move.from_uci("e2e4"), board))
    cboards = [CBoard.from_board(board) for _ in range(n)]
    pol = np.zeros((n, 4672), np.float32)
    wdl = np.zeros((n, 3), np.float32)
    actions = np.full(n, action, dtype=np.int32)
    values = np.zeros(n, np.float64)
    mcts = np.zeros((n, 4672), np.float32)
    # Must not raise "legal/action count must be 0..256".
    result = batch_process_ply(
        cboards, pol, wdl, actions, values, mcts,
        0, 1.0, 1.0, 0.0, 1.0,
    )
    assert result is not None
    assert len(result) >= 12
    assert result[0].shape[0] == n


def test_batch_encoder_rejects_fake_cboard_and_strided_output() -> None:
    fake_cboard = cast(Any, type("CBoard", (), {})())
    out = np.zeros((1, 146, 8, 8), np.float32)
    with pytest.raises(TypeError, match="CBoard"):
        batch_encode_146([fake_cboard], out)

    cb = CBoard.from_board(chess.Board())
    strided = np.zeros((1, 146, 8, 16), np.float32)[:, :, :, ::2]
    assert not strided.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        batch_encode_146([cb], strided)
