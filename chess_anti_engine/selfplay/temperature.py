from __future__ import annotations

import numpy as np


def apply_policy_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply a temperature ``T`` to a probability distribution.

    Equivalent to ``softmax(log(p)/T)`` for ``p > 0``.  Returns ``probs``
    unchanged when ``T == 1.0`` or the distribution is degenerate.
    Used for the soft policy target in ``finalize_game`` and for the
    post-search resampling in ``run_network_turn``.
    """
    t = float(temperature)
    if t == 1.0:
        return probs.astype(np.float32, copy=False)
    p = probs.astype(np.float64, copy=True)
    p = np.maximum(p, 0.0)
    if float(p.sum()) <= 0:
        return probs.astype(np.float32, copy=False)
    p = p / float(p.sum())
    p = np.power(p, 1.0 / t)
    s = float(p.sum())
    if s > 0:
        p /= s
    return p.astype(np.float32, copy=False)


def temperature_for_ply(
    *,
    ply: int,
    temperature: float,
    drop_plies: int,
    after: float,
    decay_start_move: int = 0,
    decay_moves: int = 0,
    endgame: float = 0.0,
) -> float:
    """Return the temperature to use at a given ply/move.

    In our codebase, the selfplay loop passes a 1-based *move_number* here (not a raw half-move ply),
    but the function is intentionally simple and treats the input as an integer time-step.

    Precedence:
    1) Linear decay schedule (LC0-like), if decay_moves > 0:
       - use `temperature` until decay_start_move
       - then linearly decay to `endgame` over `decay_moves`
       - then clamp at `endgame`
    2) Step schedule, if drop_plies > 0:
       - use `temperature` for t < drop_plies else `after`
    3) Constant: `temperature`
    """
    t = int(ply)

    dm = int(decay_moves)
    if dm > 0:
        start = int(decay_start_move)
        if t < start:
            return float(temperature)
        if t >= start + dm:
            return float(endgame)
  # linear interpolation from temperature -> endgame
        frac = float(t - start) / float(dm)
        return float(temperature + (float(endgame) - float(temperature)) * frac)

    dp = int(drop_plies)
    if dp > 0:
        return float(temperature) if t < dp else float(after)

    return float(temperature)
