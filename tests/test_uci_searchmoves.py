from __future__ import annotations

import numpy as np
import chess

from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.moves import move_to_index
from chess_anti_engine.uci.search import (
    _allowed_root_indices,
    _best_move_and_pv,
)


def test_allowed_root_indices_ignores_invalid_searchmoves() -> None:
    board = chess.Board()

    allowed = _allowed_root_indices(board, ("e2e4", "notamove", "e7e5"))

    assert allowed == {int(move_to_index(chess.Move.from_uci("e2e4"), board))}


def test_best_move_is_restricted_to_searchmoves() -> None:
    board = chess.Board()
    e2e4 = int(move_to_index(chess.Move.from_uci("e2e4"), board))
    d2d4 = int(move_to_index(chess.Move.from_uci("d2d4"), board))
    tree = MCTSTree()
    root = tree.add_root(0, 0.0)
    tree.expand(
        root,
        np.array([e2e4, d2d4], dtype=np.int32),
        np.array([0.5, 0.5], dtype=np.float64),
    )
    tree.backprop(np.array([root, tree.find_child(root, d2d4)], dtype=np.int32), 1.0)

    best, pv = _best_move_and_pv(tree, root, allowed_root_indices={e2e4})

    assert best == e2e4
    assert pv == [e2e4]
