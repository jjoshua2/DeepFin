"""Convert WDL outputs to UCI centipawn scores.

Uses Leela's `cp = 295 * tan(1.5637 * (2Q - 1))` mapping where ``Q`` is the
expected score in [0, 1]. This keeps the sign convention consistent with
what GUIs expect (+ = better for side-to-move) and matches what Ceres/lc0
report, so we can compare analysis lines directly.
"""
from __future__ import annotations

import math

# Leela-style winrate → centipawn calibration.
_CP_A = 295.0
_CP_K = 1.5637541897

# The saturation value of the map: `q_to_cp(1.0)`, i.e. `295 * tan(_CP_K)`.
# Named so the bound is a stated constant rather than a number that falls out
# of the arithmetic. `uci/search.py` independently defines `_TB_WIN_CP` /
# `_TB_LOSS_CP` as ±41890 — the SAME value — and `test_uci_score_bounds.py`
# pins that agreement, which is what makes a tablebase win and a search-certain
# win report the same score instead of coincidentally doing so.
CP_SATURATION = 41890


def q_to_cp(q: float) -> int:
    """Map Q in [0, 1] → centipawns. Bounded to ±``CP_SATURATION`` (±41890).

    ⚑ There is no clamp ON THE CENTIPAWN VALUE, and an earlier version of this
    docstring said "clamped to a sane range" while a comment claimed to "clamp
    arg just shy of ±π/2". Neither clamp existed. What bounds the output is the
    pair of facts that ``q`` IS clamped to [0, 1] and that ``_CP_K`` happens to
    sit below π/2, so ``tan`` never reaches its pole: the reachable range is
    ±41890 cp, about ±419 pawns. Whether that counts as "sane" is a matter of
    taste; the point is that the previous text described a guard that was not
    there, so anyone changing ``_CP_K`` would have believed a bound was
    enforced independently of it. It is not — ``_CP_K >= π/2`` yields ``inf``
    or a sign flip, and only ``test_uci_score_bounds.py`` would catch it.

    A real clamp was considered and REJECTED, because two consumers depend on
    the unbounded tail:

    * ``uci/search.py:2312`` reports ``q_to_cp(1.0)`` for an immediate mate,
      and ``:2243`` reports ``_TB_WIN_CP`` (41890) for a tablebase win. They
      agree exactly today. Any narrower clamp would make a forced mate score
      BELOW a tablebase win, which is a behaviour change, and
      ``tests/test_mcts_uci_parity_gates.py:429`` asserts on that constant.
    * ``scripts/mine_blindspot_seeds.py:_own_expected_score`` inverts this map
      exactly (``atan(cp / 295) / 1.5637``) to put our eval on a common scale
      with deep-SF's. A clamp makes the map non-injective at the extremes, so
      the inverse would silently under-report confidence for every saturated
      score — a quiet analysis error in place of a cosmetic one.

    So the claim was deleted and the real bound documented and pinned, rather
    than inventing a clamp that would break two working consumers to make a
    comment true.
    """
    q = max(0.0, min(1.0, float(q)))
    arg = _CP_K * (2.0 * q - 1.0)
    cp = _CP_A * math.tan(arg)
    return round(cp)
