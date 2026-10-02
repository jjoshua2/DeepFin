"""Reset must discard pending search state before node ids are reused."""
from __future__ import annotations

import weakref

import chess
import numpy as np
import pytest
from numpy.typing import NDArray

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.moves import move_to_index


def _expanded_start(
    tree: MCTSTree,
) -> tuple[CBoard, int, NDArray[np.int32], NDArray[np.float64]]:
    cb = CBoard.from_board(chess.Board())
    actions = cb.legal_move_indices().astype(np.int32, copy=False)
    root = tree.add_root(1, 0.25)
    priors = np.zeros(4672, dtype=np.float64)
    priors[actions] = 1.0 / actions.size
    tree.expand(root, actions, priors[actions])
    return cb, root, actions, priors


def _start(
    tree: MCTSTree, cb: CBoard, root: int, actions: NDArray[np.int32],
    priors: NDArray[np.float64], enc: NDArray[np.float32],
) -> int | None:
    return tree.start_gumbel_sims(
        [cb], np.array([root], dtype=np.int32),
        [[int(actions[0]), int(actions[1])]],
        [np.zeros(4672, dtype=np.float64)], [priors],
        np.array([2], dtype=np.int32), np.array([0.25], dtype=np.float64),
        0.1, 50.0, 2.5, 1.2, True, enc, 1, 1, 0,
    )


@pytest.mark.parametrize("reset_method", ["reset", "reset_compact"])
def test_reset_reused_slots_have_no_virtual_loss_or_solved_state(reset_method: str) -> None:
    tree = MCTSTree()
    tree.set_cpuct_scaling(1.75, 12000.0)
    _cb, root, actions, _priors = _expanded_start(tree)
    path = np.array([root, tree.find_child(root, int(actions[0]))], dtype=np.int32)
    tree.backprop(path, -0.8)
    tree.mark_solved_path(path, -1)
    tree.apply_vloss_path(path)
    assert tree.get_virtual_loss(int(path[1])) == 1
    assert tree.get_solved_status(root) == 1

    getattr(tree, reset_method)()
    assert tree.node_count() == 0
    _cb, new_root, new_actions, _priors = _expanded_start(tree)
    assert new_root == root
    assert tree.get_cpuct_scaling() == (1.75, 12000.0)
    assert tree.node_q(new_root) == pytest.approx(0.25)
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))
    assert all(tree.get_solved_status(i) == 0 for i in range(tree.node_count()))
    got_actions, visits, values = tree.get_children_q(new_root, 0.125)
    assert np.array_equal(got_actions, new_actions)
    assert np.all(visits == 0)
    assert np.all(values == 0.125)

    # A new walker can acquire and release exactly its own penalty.
    new_path = np.array([new_root, tree.find_child(new_root, int(new_actions[0]))], dtype=np.int32)
    tree.apply_vloss_path(new_path)
    assert tree.get_virtual_loss(int(new_path[1])) == 1
    tree.remove_vloss_path(new_path)
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


@pytest.mark.parametrize("reset_method", ["reset", "reset_compact"])
def test_reset_cancels_pending_gumbel_batch_and_allows_new_search(reset_method: str) -> None:
    tree = MCTSTree()
    cb, root, actions, priors = _expanded_start(tree)
    enc = np.empty((4, 146, 8, 8), dtype=np.float32)
    enc_ref = weakref.ref(enc)
    pending = _start(tree, cb, root, actions, priors, enc)
    assert pending is not None
    assert pending > 0
    assert any(tree.get_virtual_loss(i) > 0 for i in range(tree.node_count()))
    del enc
    assert enc_ref() is not None  # Pending search owns the borrowed output buffer.

    getattr(tree, reset_method)()
    assert enc_ref() is None
    cb, new_root, actions, priors = _expanded_start(tree)
    initial_count = tree.node_count()
    with pytest.raises(RuntimeError, match="not in needs_eval state"):
        tree.continue_gumbel_sims(
            np.zeros((pending, 4672), dtype=np.float32),
            np.zeros((pending, 3), dtype=np.float32),
        )
    with pytest.raises(RuntimeError, match="no batch pending"):
        tree.get_pending_legal_indices()
    assert tree.node_count() == initial_count
    assert tree.node_q(new_root) == pytest.approx(0.25)
    assert not any(tree.get_virtual_loss(i) for i in range(tree.node_count()))

    # Complete a fresh search after cancellation: every accepted evaluation
    # adds a real root-child visit and all pending penalties are removed.
    result = _start(tree, cb, new_root, actions, priors,
                    np.empty((4, 146, 8, 8), dtype=np.float32))
    evaluated = 0
    while result is not None:
        evaluated += result
        result = tree.continue_gumbel_sims(
            np.zeros((result, 4672), dtype=np.float32),
            np.zeros((result, 3), dtype=np.float32),
        )
    assert evaluated == 2
    _actions, visits = tree.get_children_visits(new_root)
    assert int(visits.sum()) == evaluated
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


@pytest.mark.parametrize("reset_method", ["reset", "reset_compact"])
def test_reset_after_terminal_search_does_not_preserve_solved_state(reset_method: str) -> None:
    tree = MCTSTree()
    board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    mating_move = chess.Move.from_uci("g6g7")
    mate = move_to_index(mating_move, board)
    after_mate = board.copy()
    after_mate.push(mating_move)
    assert after_mate.is_checkmate()
    cb = CBoard.from_board(board)
    actions = cb.legal_move_indices().astype(np.int32, copy=False)
    root = tree.add_root(1, 0.25)
    priors = np.zeros(4672, dtype=np.float64)
    priors[actions] = 1.0 / actions.size
    tree.expand(root, actions, priors[actions])
    result = tree.start_gumbel_sims(
        [cb], np.array([root], dtype=np.int32), [[mate]],
        [np.zeros(4672, dtype=np.float64)], [priors],
        np.array([1], dtype=np.int32), np.array([0.25], dtype=np.float64),
        0.1, 50.0, 2.5, 1.2, True,
        np.empty((4, 146, 8, 8), dtype=np.float32), 1, 1, 0,
    )
    assert result is None  # A terminal leaf must not request an evaluation.
    child = tree.find_child(root, mate)
    assert tree.get_solved_status(child) == -1
    assert tree.get_solved_status(root) == 1
    assert tree.node_q(child) == -1.0
    assert tree.node_q(root) == pytest.approx(0.625)
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))

    getattr(tree, reset_method)()
    _cb, new_root, _actions, _priors = _expanded_start(tree)
    assert tree.get_solved_status(new_root) == 0
    assert tree.node_q(new_root) == pytest.approx(0.25)
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))
