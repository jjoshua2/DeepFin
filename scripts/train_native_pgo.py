#!/usr/bin/env python3
"""Exercise representative native hot paths to collect GCC PGO counters."""

from __future__ import annotations

import argparse

import chess
import numpy as np

from chess_anti_engine.encoding._features_ext import (
    compute_extra_features,
    compute_relation_matrices,
)
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import (
    MCTSTree,
    batch_encode_146,
    batch_encode_146_bf16,
    batch_process_ply,
)


def _make_boards(n: int) -> list[CBoard]:
    rng = np.random.default_rng(20260710)
    boards: list[CBoard] = []
    for index in range(n):
        board = chess.Board()
        for _ in range(index % 48):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(moves[int(rng.integers(len(moves)))])
        if board.is_game_over() or not any(board.legal_moves):
            board = chess.Board()
        boards.append(CBoard.from_board(board))
    return boards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")

    boards = _make_boards(128)
    n = len(boards)
    out_f32 = np.empty((n, 175, 8, 8), dtype=np.float32)
    out_bf16 = np.empty((n, 175, 8, 8), dtype=np.uint16)
    legal_lists = [board.legal_move_indices() for board in boards]
    actions = np.array([int(legal[0]) for legal in legal_lists], dtype=np.int32)

    pol = np.zeros((n, 4672), dtype=np.float32)
    wdl = np.tile(np.array([[2.0, 0.5, -1.0]], dtype=np.float32), (n, 1))
    values = np.zeros(n, dtype=np.float64)
    mcts_probs = np.zeros((n, 4672), dtype=np.float32)
    for row, legal in enumerate(legal_lists):
        mcts_probs[row, legal] = 1.0 / len(legal)

    pieces_us = np.array([
        0x000000000000FF00, 0x0000000000000042, 0x0000000000000024,
        0x0000000000000081, 0x0000000000000008, 0x0000000000000010,
    ], dtype=np.uint64)
    pieces_them = np.array([
        0x00FF000000000000, 0x4200000000000000, 0x2400000000000000,
        0x8100000000000000, 0x0800000000000000, 0x1000000000000000,
    ], dtype=np.uint64)

    tree = MCTSTree()
    root = tree.add_root(1, 0.0)
    root_legal = legal_lists[0].astype(np.int32, copy=False)
    priors = np.full(len(root_legal), 1.0 / len(root_legal), dtype=np.float64)
    tree.expand(root, root_legal, priors)
    first_child = tree.find_child(root, int(root_legal[0]))
    path = np.array([root, first_child], dtype=np.int32)
    roots = np.full(64, root, dtype=np.int32)
    bulk_wdl = np.tile(wdl[:1], (65_536, 1))

    for iteration in range(args.iterations):
        batch_encode_146_bf16(boards, out_bf16)
        batch_encode_146(boards, out_f32)
        for board in boards:
            board.legal_move_indices()
            child = board.copy()
            child_legal = child.legal_move_indices()
            child.push_index(int(child_legal[iteration % len(child_legal)]))
            child.is_game_over()
            child.terminal_value()

        compute_extra_features(
            pieces_us, pieces_them, 0xFFFF00000000FFFF,
            4, 60, True, -1, 63,
        )
        compute_relation_matrices(
            pieces_us, pieces_them, 0xFFFF00000000FFFF, 4, 60, True,
        )
        tree.batch_wdl_to_q(bulk_wdl)
        tree.select_leaves(roots, 1.5, 0.0, 0.25)
        tree.backprop(path, 0.25 if iteration & 1 else -0.25)
        tree.get_children_q(root, 0.0)

        if iteration % 8 == 0:
            batch_process_ply(
                boards, pol, wdl, actions, values, mcts_probs,
                0, 1.0, 1.0, 0.0, 1.0, 0, 63,
            )


if __name__ == "__main__":
    main()
