"""Virtual-loss scaffolding (Phase 2+3 prep).

Only asserts that the per-node vloss array is allocated, readable, and starts
at zero. Once descent/backprop wire vloss increments, we'll add behavioral
tests that drive it through a real search.
"""
from __future__ import annotations

import numpy as np

from chess_anti_engine.mcts._mcts_tree import MCTSTree


def test_virtual_loss_initial_zero():
    t = MCTSTree()
    rid = t.add_root(0, 0.0)
    assert t.get_virtual_loss(rid) == 0


def test_virtual_loss_zero_for_out_of_range_ids():
    t = MCTSTree()
    t.add_root(0, 0.0)
    assert t.get_virtual_loss(-1) == 0
    assert t.get_virtual_loss(1_000_000) == 0


def test_virtual_loss_survives_expansion_and_stays_zero():
    t = MCTSTree()
    rid = t.add_root(0, 0.0)
    actions = np.array([0, 1, 2], dtype=np.int32)
    priors = np.array([0.4, 0.3, 0.3], dtype=np.float64)
    t.expand(rid, actions, priors)
    # Phase 2+3 hasn't wired vloss into descent/backprop yet — count must
    # remain 0 on all nodes for now. Will change when walkers arrive.
    for node_id in range(t.node_count()):
        assert t.get_virtual_loss(node_id) == 0


def test_virtual_loss_resets_on_tree_reset():
    t = MCTSTree()
    t.add_root(0, 0.0)
    t.reset()
    # After reset, node_count is 0 and any id returns 0.
    assert t.get_virtual_loss(0) == 0
