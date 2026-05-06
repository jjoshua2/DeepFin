from __future__ import annotations

import numpy as np


_MATE_BASE_CP = 1500.0
_MATE_DEPTH_BONUS_CP = 20.0


def mate_to_effective_cp(mate_in: int) -> float:
    """Map a mate-in-N score to a large effective centipawn value.

    Sign(mate_in) carries the side; magnitude grows for shorter mates so
    `cp_to_wdl` saturates them as decisive. ``mate_in=0`` is treated as
    `_MATE_BASE_CP` (positive). The returned value is bounded.
    """
    sign = 1.0 if mate_in >= 0 else -1.0
    plies = abs(int(mate_in))
    bonus = max(0.0, 50.0 - float(plies)) * _MATE_DEPTH_BONUS_CP
    return sign * (_MATE_BASE_CP + bonus)


def cp_to_wdl(
    cp: float | None,
    mate: int | None,
    *,
    slope: float,
    draw_width_cp: float,
) -> np.ndarray:
    """Convert a Stockfish cp/mate score to (W, D, L) probabilities.

    Logistic with explicit draw zone:
        p_win  = sigmoid( slope * (eff_cp - draw_width_cp) )
        p_loss = sigmoid( slope * (-eff_cp - draw_width_cp) )
        p_draw = 1 - p_win - p_loss   (clamped >= 0, then renormalised)

    ``slope`` (per-cp) controls steepness; ``draw_width_cp`` is the half
    width of the draw zone — at |cp| ~= draw_width_cp the side ahead has
    p_win ≈ 0.5. Both must be > 0; callers wanting the no-op should keep
    SF's UCI_ShowWDL output instead.

    ``mate`` takes precedence over ``cp`` when present (matches the UCI
    convention — SF emits at most one of the two per info line).
    """
    if slope <= 0.0 or draw_width_cp < 0.0:
        raise ValueError(f"cp_to_wdl requires slope>0 and draw_width_cp>=0, got {slope=} {draw_width_cp=}")
    if mate is not None:
        eff_cp = mate_to_effective_cp(int(mate))
    elif cp is not None:
        eff_cp = float(cp)
    else:
        raise ValueError("cp_to_wdl needs either cp or mate")
    p_win = 1.0 / (1.0 + np.exp(-slope * (eff_cp - draw_width_cp)))
    p_loss = 1.0 / (1.0 + np.exp(-slope * (-eff_cp - draw_width_cp)))
    p_draw = max(0.0, 1.0 - p_win - p_loss)
    total = p_win + p_loss + p_draw
    return np.array([p_win, p_draw, p_loss], dtype=np.float32) / float(total)
