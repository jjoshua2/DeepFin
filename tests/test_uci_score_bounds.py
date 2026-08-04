"""`q_to_cp`'s bound is real but is NOT a clamp — pin it (audit A23).

The docstring said "clamped to a sane range" and a comment claimed to "clamp
arg just shy of ±π/2". Neither clamp existed. What bounds the output is the
combination of `q` being clamped to [0, 1] and `_CP_K = 1.5637541897` sitting
below π/2 ≈ 1.5707963, so `tan` never reaches its pole. The reachable range is
±41890 cp — about ±419 pawns.

⚑ THE CALL: the claim was deleted and the real bound documented and pinned,
rather than adding a clamp. A clamp would break two working consumers:

* `uci/search.py:2312` reports `q_to_cp(1.0)` for an immediate mate and `:2243`
  reports `_TB_WIN_CP` for a tablebase win. Both are 41890 today. Any narrower
  clamp makes a forced mate score BELOW a tablebase win, and
  `tests/test_mcts_uci_parity_gates.py:429` asserts on that constant.
* `scripts/mine_blindspot_seeds.py:_own_expected_score` inverts this map
  exactly (`atan(cp / 295) / 1.5637`) to compare our eval against deep-SF's on
  a common scale. A clamp makes the map non-injective at the extremes, so the
  inverse silently under-reports confidence for every saturated score — a
  quiet analysis error replacing a cosmetic one.

The agreement between `CP_SATURATION` and `search.py`'s `_TB_WIN_CP` was pure
coincidence of two independently written constants. These tests convert it into
a checked invariant.
"""

from __future__ import annotations

import math

import pytest

from chess_anti_engine.uci.score import _CP_A, _CP_K, CP_SATURATION, q_to_cp


def test_the_saturation_constant_matches_the_actual_extremes() -> None:
    """`CP_SATURATION` must be what the map really produces, not a guess."""
    assert q_to_cp(1.0) == CP_SATURATION
    assert q_to_cp(0.0) == -CP_SATURATION
    assert CP_SATURATION == 41890


def test_the_bound_agrees_with_search_s_tablebase_constants() -> None:
    """⚑ The invariant that was previously a coincidence.

    `uci/search.py` defines `_TB_WIN_CP`/`_TB_LOSS_CP` independently of this
    module. A forced mate (`q_to_cp(1.0)`) and a tablebase win must report the
    same score; if someone changes `_CP_K` or those constants, this fails
    instead of the two silently drifting apart.
    """
    from chess_anti_engine.uci import search as uci_search

    assert q_to_cp(1.0) == uci_search._TB_WIN_CP
    assert q_to_cp(0.0) == uci_search._TB_LOSS_CP


def test_the_calibration_constant_stays_below_the_tangent_pole() -> None:
    """The ONLY thing keeping the output finite. Nothing else enforces it.

    With `_CP_K >= π/2` the map yields inf or flips sign, and because there is
    no clamp on the centipawn value there is no second line of defence. This is
    the test the deleted docstring claim led people to believe existed.
    """
    assert math.pi / 2.0 > _CP_K
    assert math.isfinite(_CP_A * math.tan(_CP_K))


@pytest.mark.parametrize("q", [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
def test_every_q_in_range_maps_inside_the_bound(q: float) -> None:
    assert abs(q_to_cp(q)) <= CP_SATURATION


@pytest.mark.parametrize("q", [-5.0, -0.001, 1.001, 42.0, float("inf")])
def test_out_of_range_q_is_clamped_to_the_endpoints(q: float) -> None:
    """`q` IS clamped, and that is what bounds `cp`. Unchanged behaviour."""
    assert abs(q_to_cp(q)) == CP_SATURATION


def test_the_map_is_strictly_monotone_and_centred() -> None:
    """Move ordering depends on monotonicity; the sign convention on centring.

    A clamp would break strictness at the extremes, which is exactly why the
    inverse in `mine_blindspot_seeds.py` would stop being an inverse.
    """
    qs = [i / 200.0 for i in range(201)]
    cps = [q_to_cp(q) for q in qs]

    assert cps == sorted(cps)
    assert q_to_cp(0.5) == 0
    assert q_to_cp(0.75) == -q_to_cp(0.25)


def test_the_map_round_trips_through_the_analysis_inverse() -> None:
    """`mine_blindspot_seeds.py` inverts this map; prove it still inverts.

    `_own_expected_score` is `atan(cp / 295) / 1.5637`, giving expected score
    in [-1, 1], i.e. `2q - 1`. Reproduced here rather than imported so the
    check does not depend on that script's import graph, and pinned at
    non-saturated values where the integer rounding is the only loss.
    """
    for q in (0.2, 0.35, 0.5, 0.65, 0.8):
        cp = q_to_cp(q)
        recovered_q = 0.5 * (math.atan(float(cp) / _CP_A) / _CP_K + 1.0)
        assert recovered_q == pytest.approx(q, abs=1e-3)
