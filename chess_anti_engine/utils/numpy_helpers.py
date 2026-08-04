"""Tiny numpy helpers shared across mcts/, selfplay/, and similar.

Centralizes utilities that were previously duplicated in 2-3 modules
(softmax with the uniform fallback, etc.). Keep this module deliberately
thin — high-churn vector ops belong in the consumer modules where their
shape contracts are obvious.
"""
from __future__ import annotations

import math

import numpy as np


def softmax_1d(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over a 1D array, with uniform fallback.

    Computes in fp64 for stability and casts the result back to fp32.
    Returns a uniform distribution when the input sums to zero after
    exponentiation, or when it is NOT FINITE -- both avoid a corrupt value
    reaching the downstream MCTS prior (``mcts/puct.py``, ``mcts/gumbel.py``)
    or the SF policy target (``selfplay/stockfish_turn.py``).

    ⚑ The non-finite half of that guard is the one that matters, and it used
    to be missing. The check was ``if s <= 0``, and ``nan <= 0`` is False, so
    a SINGLE nan logit produced an all-nan prior rather than the fallback:
    ``np.max`` over a vector containing one nan is nan, so subtracting it made
    EVERY entry nan, and nan/nan sailed through the guard. Not a crash -- a
    silent nan distribution handed to search, from one bad logit. A diverged
    net is the realistic source and this repo has that failure on record
    (GPBT: lr > 0.003 destroys the model; the warm-start 3e-4 regime cost
    -494 Elo). The guard is documented as protecting the prior, and it did not
    fire on the pathology most likely to reach it.

    Testing the MAX rather than the sum is what covers all three non-finite
    shapes at once, before any arithmetic can launder them:

    * one ``nan`` logit           -> max is nan;
    * all ``-inf`` (fully masked) -> max is -inf, and ``-inf - -inf`` is a nan
      the old code produced together with a RuntimeWarning;
    * any ``+inf`` logit          -> max is +inf, same subtraction.

    A ``+inf`` logit is treated as corruption rather than as certainty on
    purpose: nothing in this codebase encodes a certain move that way. The
    legitimate use of infinity here is ``-inf`` for MASKED entries, and a
    vector like ``[-inf, -inf, 0.0]`` has a finite max, so it takes the normal
    path and still returns ``[0, 0, 1]`` exactly as before.
    """
    z = x.astype(np.float64, copy=False)
    m = float(np.max(z))
    if not math.isfinite(m):
        return np.full_like(z, 1.0 / z.size, dtype=np.float32)
    z = z - m
    e = np.exp(z)
    s = float(e.sum())
  # Retained as the DOCUMENTED contract, not as live defence: once the max is
  # finite the top entry contributes exp(0) == 1, so s is in [1, size] and
  # neither branch can fire. Deleting it would be an undocumented behaviour
  # change for a caller relying on the promise.
    if s <= 0 or not math.isfinite(s):
        return np.full_like(e, 1.0 / e.size, dtype=np.float32)
    return (e / s).astype(np.float32, copy=False)
