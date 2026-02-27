from __future__ import annotations


def progressive_mcts_simulations(
    step: int,
    *,
    start: int = 50,
    max_sims: int = 800,
    ramp_steps: int = 10_000,
    exponent: float = 2.0,
) -> int:
    """Progressively ramp the MCTS simulation budget as training improves.

    The schedule is keyed off an increasing `step` (typically `trainer.step`).

    Args:
        step: Current training step (>=0).
        start: Initial simulation count at step=0.
        max_sims: Final simulation count at step>=ramp_steps.
        ramp_steps: Number of steps over which to ramp. If <=0, returns max_sims.
        exponent: Curve exponent. >1 ramps slowly early and faster later.

    Returns:
        An integer simulation count in [min(start,max_sims), max(start,max_sims)].
    """
    s0 = int(start)
    s1 = int(max_sims)

    if ramp_steps <= 0:
        return max(1, s1)

    st = max(0, int(step))
    t = min(1.0, float(st) / float(int(ramp_steps)))

    # Robustness: negative/NaN exponents -> linear.
    try:
        exp = float(exponent)
        if not (exp > 0.0):
            exp = 1.0
    except Exception:
        exp = 1.0

    frac = t**exp
    sims = int(round(float(s0) + (float(s1) - float(s0)) * frac))
    return max(1, sims)
