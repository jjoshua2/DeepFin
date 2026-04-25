"""Shared MCTS action-sampling primitives."""
from __future__ import annotations

import numpy as np


def sample_action_with_temperature(
    rng: np.random.Generator,
    actions: np.ndarray,
    weights: np.ndarray,
    temperature: float,
    *,
    argmax_idx: int,
) -> int:
    """Sample an action by raising ``weights`` to ``1/temperature`` and choosing.

    ``actions`` is the action-index array and ``weights[i]`` is the
    nonnegative weight (visit count, prior, importance) for ``actions[i]``.
    ``argmax_idx`` is the fallback index *into actions* used when
    ``temperature <= 0``, ``weights.sum()`` is zero or non-finite, or the
    distribution is otherwise degenerate. Callers pre-compute it as the
    natural argmax (``np.argmax(weights)`` for PUCT; ``0`` for Gumbel,
    where the survivor is always at index 0 after sequential halving).
    """
    if actions.size == 0:
        return 0
    if temperature <= 0:
        return int(actions[argmax_idx])
    p = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if temperature != 1.0:
        p = np.power(p, 1.0 / float(temperature))
    ps = float(p.sum())
    if not np.isfinite(ps) or ps <= 0:
        return int(actions[argmax_idx])
    p /= ps
    return int(actions[rng.choice(actions.size, p=p)])
