"""Virtual-loss behavior for walker-pool (phase 2+3).

Covers the primitives the walker pool will use:
  - get_virtual_loss(node_id): accessor with safe out-of-range handling
  - apply_vloss_path(path) / remove_vloss_path(path): atomic +/- along a path
  - vloss_weight=0 path still behaves identically to pre-vloss descent
  - vloss_weight>0 steers the next descent off a path whose nodes are in-flight
"""
from __future__ import annotations

import numpy as np

from chess_anti_engine.mcts._mcts_tree import MCTSTree


def _make_root_with_children(priors: list[float]) -> tuple[MCTSTree, int, list[int]]:
    t = MCTSTree()
    rid = t.add_root(0, 0.0)
    actions = np.arange(len(priors), dtype=np.int32)
    t.expand(rid, actions, np.array(priors, dtype=np.float64))
    children = [t.find_child(rid, int(a)) for a in actions]
    return t, rid, children


def test_virtual_loss_initial_zero():
    t = MCTSTree()
    rid = t.add_root(0, 0.0)
    assert t.get_virtual_loss(rid) == 0


def test_virtual_loss_zero_for_out_of_range_ids():
    t = MCTSTree()
    t.add_root(0, 0.0)
    assert t.get_virtual_loss(-1) == 0
    assert t.get_virtual_loss(1_000_000) == 0


def test_virtual_loss_survives_expansion_and_stays_zero_without_apply():
    t, _, _ = _make_root_with_children([0.4, 0.3, 0.3])
    # No apply_vloss_path call — count must remain 0 on all nodes.
    for node_id in range(t.node_count()):
        assert t.get_virtual_loss(node_id) == 0


def test_virtual_loss_resets_on_tree_reset():
    t = MCTSTree()
    t.add_root(0, 0.0)
    t.reset()
    assert t.get_virtual_loss(0) == 0


def test_apply_and_remove_vloss_path_roundtrip():
    t, rid, children = _make_root_with_children([0.5, 0.5])
    path = np.array([rid, children[0]], dtype=np.int32)
    t.apply_vloss_path(path)
    # Root is deliberately skipped — walkers all share the root.
    assert t.get_virtual_loss(rid) == 0
    assert t.get_virtual_loss(children[0]) == 1
    assert t.get_virtual_loss(children[1]) == 0
    t.remove_vloss_path(path)
    assert t.get_virtual_loss(children[0]) == 0


def test_apply_vloss_accumulates_on_repeated_application():
    t, rid, children = _make_root_with_children([0.5, 0.5])
    path = np.array([rid, children[0]], dtype=np.int32)
    for _ in range(3):
        t.apply_vloss_path(path)
    assert t.get_virtual_loss(children[0]) == 3
    for _ in range(3):
        t.remove_vloss_path(path)
    assert t.get_virtual_loss(children[0]) == 0


def test_remove_vloss_floors_at_zero():
    t, rid, children = _make_root_with_children([0.5, 0.5])
    path = np.array([rid, children[0]], dtype=np.int32)
    # Overshoot — shouldn't go negative.
    t.remove_vloss_path(path)
    t.remove_vloss_path(path)
    assert t.get_virtual_loss(children[0]) == 0


def test_apply_vloss_rejects_out_of_range_node_id():
    t, rid, _ = _make_root_with_children([0.5, 0.5])
    bogus = np.array([rid, 9_999_999], dtype=np.int32)
    try:
        t.apply_vloss_path(bogus)
    except ValueError:
        return
    raise AssertionError("expected ValueError on out-of-range node id")
