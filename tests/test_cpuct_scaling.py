"""Lc0 classic visit-dependent Cpuct scaling on the C tree (UCI PUCT engines).

The Python transcription ``mcts.puct.effective_cpuct`` and its three formula
tests were retired with the Python PUCT search on 2026-11-14.
"""
from __future__ import annotations

from chess_anti_engine.mcts._mcts_tree import MCTSTree


def test_tree_set_cpuct_scaling_roundtrip() -> None:
    tree = MCTSTree()
    fac0, base0 = tree.get_cpuct_scaling()
    assert fac0 == 0.0
    assert base0 == 38739.0
    tree.set_cpuct_scaling(3.89, 38739.0)
    fac, base = tree.get_cpuct_scaling()
    assert fac == 3.89
    assert base == 38739.0
    tree.set_cpuct_scaling(0.0)
    assert tree.get_cpuct_scaling()[0] == 0.0


def test_tree_cpuct_scaling_survives_resets() -> None:
    # Scaling is config, not tree state: both reset paths must keep it
    # (reset_compact swaps in a tree_init'd tree, which would zero it).
    tree = MCTSTree()
    tree.set_cpuct_scaling(3.89, 20000.0)
    tree.reset()
    assert tree.get_cpuct_scaling() == (3.89, 20000.0)
    tree.reset_compact()
    assert tree.get_cpuct_scaling() == (3.89, 20000.0)
