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


def q_to_cp(q: float) -> int:
    """Map Q in [0, 1] → centipawns, clamped to a sane range."""
    q = max(0.0, min(1.0, float(q)))
    # tan goes to infinity near q=0 or q=1; clamp arg just shy of ±π/2.
    arg = _CP_K * (2.0 * q - 1.0)
    cp = _CP_A * math.tan(arg)
    return int(round(cp))


def wdl_probs_to_cp(w: float, d: float, l: float) -> int:
    q = float(w) + 0.5 * float(d)
    # Guard against slight over-1.0 from floating softmax.
    total = max(1e-9, float(w) + float(d) + float(l))
    return q_to_cp(q / total)
