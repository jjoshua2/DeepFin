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
    natural argmax: ``np.argmax(weights)`` for PUCT, and for Gumbel
    ``np.searchsorted(legal, best_a)`` where ``best_a = remaining[0]`` is the
    sequential-halving survivor. It is NOT 0 for Gumbel — index 0 of ``legal``
    is the lowest legal action id, not the survivor. (This docstring claimed
    otherwise until the 2026-08-03 play-path audit, F10; both Gumbel call
    sites always passed the searchsorted index.)
    """
    if actions.size == 0:
        return 0
    if temperature <= 0:
        return int(actions[argmax_idx])
    p = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    ps = float(p.sum())
    if not np.isfinite(ps) or ps <= 0:
        return int(actions[argmax_idx])
    p /= ps
    if temperature != 1.0:
        logits = np.log(np.maximum(p, np.finfo(np.float64).tiny)) / float(temperature)
        logits -= float(np.max(logits))
        p = np.exp(logits)
        ps = float(p.sum())
        if not np.isfinite(ps) or ps <= 0:
            return int(actions[argmax_idx])
        p /= ps
    return int(actions[rng.choice(actions.size, p=p)])
