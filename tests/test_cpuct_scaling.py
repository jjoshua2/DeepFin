"""Lc0 classic visit-dependent Cpuct scaling."""
from __future__ import annotations

import math

from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.puct import effective_cpuct

# Current Lc0 classic defaults (engine flags, Oct 2025).
_LCO_INIT = 1.75
_LCO_FACTOR = 3.89
_LCO_BASE = 38739.0


def test_effective_cpuct_fixed_when_factor_zero() -> None:
    assert effective_cpuct(2.5, 1000, cpuct_factor=0.0) == 2.5
    assert effective_cpuct(1.4, 0, cpuct_factor=0.0) == 1.4


def test_effective_cpuct_lc0_classic_formula() -> None:
    # c(N) = init + factor * log((N + base) / base)  — current Lc0 source
    n = 10_000
    expect = _LCO_INIT + _LCO_FACTOR * math.log((n + _LCO_BASE) / _LCO_BASE)
    got = effective_cpuct(
        _LCO_INIT, n, cpuct_factor=_LCO_FACTOR, cpuct_base=_LCO_BASE,
    )
    assert abs(got - expect) < 1e-9
    assert effective_cpuct(
        _LCO_INIT, 100_000, cpuct_factor=_LCO_FACTOR, cpuct_base=_LCO_BASE,
    ) > got


def test_effective_cpuct_lc0_table_values() -> None:
    """Sanity against documented effective-Cpuct table (approx)."""
    def c(n: int) -> float:
        return effective_cpuct(
            _LCO_INIT, n, cpuct_factor=_LCO_FACTOR, cpuct_base=_LCO_BASE,
        )

    assert abs(c(0) - 1.750) < 1e-3
    assert abs(c(100) - 1.760) < 1e-2
    assert abs(c(1_000) - 1.849) < 1e-2
    assert abs(c(10_000) - 2.643) < 1e-2
    assert abs(c(38_739) - 4.446) < 1e-2
    assert abs(c(100_000) - 6.713) < 1e-2
    assert abs(c(1_000_000) - 14.544) < 1e-2


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
