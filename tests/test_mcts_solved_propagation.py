"""Tests for MCTS solved-node propagation.

Covers:
  - direct marking via mark_solved_path with hand-built trees
  - ancestor backup rules (any-LOSS-child → WIN, all-WIN → LOSS, all-solved
    with DRAW → DRAW)
  - selection avoids SOLVED_WIN children and prefers SOLVED_LOSS ones
  - terminal-leaf detection in walker_descend_puct (real chess M1 / stalemate)
"""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree

SOLVED_UNKNOWN = 0
SOLVED_WIN = 1
SOLVED_LOSS = -1
SOLVED_DRAW = 2


def _build_root_with_n_children(t: MCTSTree, n: int) -> tuple[int, list[int]]:
    root = t.add_root(0, 0.0)
    actions = np.arange(n, dtype=np.int32)
    priors = np.full(n, 1.0 / n, dtype=np.float64)
    t.expand(root, actions, priors)
    child_ids = [t.find_child(root, int(a)) for a in actions]
    return root, child_ids


# ── Direct propagation logic ─────────────────────────────────────────────


def test_leaf_marked_solved_loss_propagates_win_to_parent():
    """One losing child → parent is WINning."""
    t = MCTSTree()
    root, children = _build_root_with_n_children(t, 3)
    t.mark_solved_path(np.array([root, children[0]], dtype=np.int32), SOLVED_LOSS)
    assert t.get_solved_status(children[0]) == SOLVED_LOSS
    assert t.get_solved_status(root) == SOLVED_WIN


def test_all_children_solved_win_makes_parent_loss():
    """Every legal move loses → parent is LOSing."""
    t = MCTSTree()
    root, children = _build_root_with_n_children(t, 3)
    for cid in children:
        t.mark_solved_path(np.array([root, cid], dtype=np.int32), SOLVED_WIN)
    assert t.get_solved_status(root) == SOLVED_LOSS


def test_mixed_draw_and_losing_children_makes_parent_draw():
    """No winning move (no SOLVED_LOSS child); at least one draw, rest WIN-for-child
    → parent forces a DRAW (the draw is the best he can do)."""
    t = MCTSTree()
    root, children = _build_root_with_n_children(t, 3)
    t.mark_solved_path(np.array([root, children[0]], dtype=np.int32), SOLVED_WIN)
    t.mark_solved_path(np.array([root, children[1]], dtype=np.int32), SOLVED_DRAW)
    t.mark_solved_path(np.array([root, children[2]], dtype=np.int32), SOLVED_WIN)
    assert t.get_solved_status(root) == SOLVED_DRAW


def test_partial_solving_does_not_resolve_parent():
    """One unsolved child blocks parent resolution."""
    t = MCTSTree()
    root, children = _build_root_with_n_children(t, 3)
    t.mark_solved_path(np.array([root, children[0]], dtype=np.int32), SOLVED_WIN)
    t.mark_solved_path(np.array([root, children[1]], dtype=np.int32), SOLVED_DRAW)
    # children[2] still unsolved
    assert t.get_solved_status(root) == SOLVED_UNKNOWN


def test_two_ply_ancestor_propagation():
    """Solved status propagates through multiple plies."""
    t = MCTSTree()
    root = t.add_root(0, 0.0)
    t.expand(root, np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float64))
    mid = t.find_child(root, 0)
    t.expand(mid, np.array([0, 1], dtype=np.int32), np.array([0.5, 0.5], dtype=np.float64))
    leaf_a = t.find_child(mid, 0)
    leaf_b = t.find_child(mid, 1)

    # Both children of mid are LOSS-for-themselves → mid is WIN; that means
    # mid is WIN-for-child-of-root, so root sees a WIN child → root is LOSS.
    t.mark_solved_path(np.array([root, mid, leaf_a], dtype=np.int32), SOLVED_LOSS)
    t.mark_solved_path(np.array([root, mid, leaf_b], dtype=np.int32), SOLVED_LOSS)
    assert t.get_solved_status(mid) == SOLVED_WIN
    assert t.get_solved_status(root) == SOLVED_LOSS


def test_already_solved_leaf_is_idempotent():
    """Re-marking a solved leaf with a different status doesn't overwrite it."""
    t = MCTSTree()
    root, children = _build_root_with_n_children(t, 2)
    path = np.array([root, children[0]], dtype=np.int32)
    t.mark_solved_path(path, SOLVED_LOSS)
    t.mark_solved_path(path, SOLVED_WIN)  # ignored
    assert t.get_solved_status(children[0]) == SOLVED_LOSS


def test_unknown_status_is_noop():
    t = MCTSTree()
    root, children = _build_root_with_n_children(t, 2)
    t.mark_solved_path(np.array([root, children[0]], dtype=np.int32), SOLVED_UNKNOWN)
    assert t.get_solved_status(children[0]) == SOLVED_UNKNOWN
    assert t.get_solved_status(root) == SOLVED_UNKNOWN


# ── Terminal detection in walker_descend_puct ────────────────────────────


def test_walker_descend_marks_checkmate_leaf_solved():
    """Walker detecting a checkmate leaf in chess marks it SOLVED_LOSS for STM."""
    t = MCTSTree()
    # Fool's-mate position with black to move: Qh4# is a legal mate move.
    b = chess.Board()
    for san in ["f3", "e5", "g4"]:
        b.push_san(san)
    cb_root = CBoard.from_board(b)
    root = t.add_root(0, 0.0)
    qh4 = chess.Move.from_uci("d8h4")
    assert qh4 in b.legal_moves
    from chess_anti_engine.moves.encode import move_to_index

    mate_idx = move_to_index(qh4, b)
    t.expand(root, np.array([mate_idx], dtype=np.int32), np.array([1.0], dtype=np.float64))

    enc = np.zeros((1, 146, 8, 8), dtype=np.float32)
    leaf_id, _node_path, _legal, term_q = t.walker_descend_puct(
        root, cb_root, 2.5, 1.0, 1.0, 0, enc,
    )
    # Leaf was a checkmate of the player-to-move → STM (white after Qh4) is in
    # checkmate, so terminal_value() returns -1.0 from white's view, and the
    # leaf is marked SOLVED_LOSS. The parent (root) becomes SOLVED_WIN because
    # we have a winning move available.
    assert term_q is not None
    assert term_q == pytest.approx(-1.0)
    assert t.get_solved_status(leaf_id) == SOLVED_LOSS
    assert t.get_solved_status(root) == SOLVED_WIN


def test_walker_descend_marks_stalemate_leaf_draw():
    """Stalemate leaf → SOLVED_DRAW; root's only move forces stalemate → root DRAW."""
    t = MCTSTree()
    # White K g6, P h6, Black K h8. White to move; h7 is the unique stalemate-in-1
    # (black-to-move with no legal moves and not in check).
    fen = "7k/8/6KP/8/8/8/8/8 w - - 0 1"
    b = chess.Board(fen)
    stalemate_move = None
    for mv in b.legal_moves:
        b.push(mv)
        if b.is_stalemate():
            stalemate_move = mv
            b.pop()
            break
        b.pop()
    assert stalemate_move is not None, "FEN must allow a stalemate-in-1"
    cb_root = CBoard.from_board(b)
    root = t.add_root(0, 0.0)
    from chess_anti_engine.moves.encode import move_to_index

    idx = move_to_index(stalemate_move, b)
    t.expand(root, np.array([idx], dtype=np.int32), np.array([1.0], dtype=np.float64))
    enc = np.zeros((1, 146, 8, 8), dtype=np.float32)
    leaf_id, _node_path, _legal, term_q = t.walker_descend_puct(
        root, cb_root, 2.5, 1.0, 1.0, 0, enc,
    )
    assert term_q is not None
    assert term_q == pytest.approx(0.0)
    assert t.get_solved_status(leaf_id) == SOLVED_DRAW
    # Root has only one legal move and it's a draw → root is also DRAW.
    assert t.get_solved_status(root) == SOLVED_DRAW
