"""Adversarial validation at public native-extension boundaries."""

from __future__ import annotations

import sys

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


def test_piece_plane_encoder_rejects_short_or_wrong_arrays() -> None:
    with pytest.raises(ValueError, match="bitboards"):
        encode_piece_planes(np.zeros(1, np.uint64), np.zeros(8, np.int32), 8)
    with pytest.raises(ValueError, match="turns"):
        encode_piece_planes(np.zeros(96, np.uint64), np.zeros(1, np.int32), 8)
    with pytest.raises(ValueError, match="bitboards"):
        encode_piece_planes(np.zeros(96, np.int64), np.zeros(8, np.int32), 8)


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


def test_batch_encoder_rejects_fake_cboard_and_strided_output() -> None:
    fake_cboard = type("CBoard", (), {})()
    out = np.zeros((1, 146, 8, 8), np.float32)
    with pytest.raises(TypeError, match="CBoard"):
        batch_encode_146([fake_cboard], out)

    cb = CBoard.from_board(chess.Board())
    strided = np.zeros((1, 146, 8, 16), np.float32)[:, :, :, ::2]
    assert not strided.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        batch_encode_146([cb], strided)
