"""Tiny numpy helpers shared across mcts/, selfplay/, and similar.

Centralizes utilities that were previously duplicated in 2-3 modules
(softmax with the uniform fallback, etc.). Keep this module deliberately
thin — high-churn vector ops belong in the consumer modules where their
shape contracts are obvious.
"""
from __future__ import annotations

import numpy as np


def softmax_1d(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over a 1D array, with uniform fallback.

    Computes in fp64 for stability and casts the result back to fp32.
    Returns a uniform distribution when the input is all-zero or sums
    to zero after exponentiation (avoids divide-by-zero corrupting the
    downstream MCTS prior or SF policy target).
    """
    z = x.astype(np.float64, copy=False)
    z = z - float(np.max(z))
    e = np.exp(z)
    s = float(e.sum())
    if s <= 0:
        return np.full_like(e, 1.0 / e.size, dtype=np.float32)
    return (e / s).astype(np.float32, copy=False)
