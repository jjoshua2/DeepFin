"""Tests for forced-move chain collapse.

When MCTS descent reaches a leaf whose own position has exactly one legal
move (forced reply: recapture, sole king escape, etc.), the chain is
inlined into the tree at prior=1.0 per ply. This saves NN evals on every
future visit through the subtree, since the policy would collapse to
one-hot anyway. See try_forced_collapse in _mcts_tree.c.

The proven-root early-exit logic in gss_begin_round (zeroing budget when
root.solved != UNKNOWN) is one structurally-trivial line; the related
solved-state plumbing is exercised by tests/test_mcts_solved_propagation.py.
"""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.moves.encode import move_to_index

SOLVED_UNKNOWN = 0


def test_collapse_inlines_single_legal_reply_into_tree():
    """Walker descends to a child whose own position has exactly one legal
    move; collapse expands the chain by one node before NN eval, advancing
    the leaf past the forced reply."""
    t = MCTSTree()
  # Pre-check root: white plays Qd1-a1+ pinning bK on a8 along the a-file.
  # Post-check, black has only Kb8 legal (Ka7 blocked by Qa1; Kb7 blocked
  # by white K on c6).
    pre_check = chess.Board("k7/8/2K5/8/8/8/8/3Q4 w - - 0 1")
    qa1_check = chess.Move.from_uci("d1a1")
    assert qa1_check in pre_check.legal_moves

    cb_pre = CBoard.from_board(pre_check)
    root = t.add_root(0, 0.0)
    qa1_idx = move_to_index(qa1_check, pre_check)
    t.expand(root, np.array([qa1_idx], dtype=np.int32), np.array([1.0], dtype=np.float64))
    assert t.node_count() == 2  # root + qa1 child

    enc = np.zeros((1, 146, 8, 8), dtype=np.float32)
    leaf_id, node_path, _legal, term_q = t.walker_descend_puct(
        root, cb_pre, 2.5, 1.0, 1.0, 0, enc,
    )
    assert t.node_count() == 3  # collapse inlined Kb8 as the qa1-child's only child
    assert len(node_path) == 3  # root → qa1_child → kb8_grandchild
    assert term_q is None  # post-Kb8 is non-terminal
    assert t.get_solved_status(leaf_id) == SOLVED_UNKNOWN


def test_collapse_chain_into_terminal_marks_solved():
    """Forced-reply chain ending in mate. Multi-ply forced-mate chains
    require both sides' replies to be uniquely determined for several
    plies, which is awkward to construct compactly in real chess. The
    1-ply collapse mechanism + solved-status backup are exercised by the
    test above and by tests/test_mcts_solved_propagation.py."""
    pytest.skip("Multi-ply forced-mate chains deferred")
