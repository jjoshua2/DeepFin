from __future__ import annotations

import math

import numpy as np

DEFAULT_CATEGORICAL_BINS = 32


def hlgauss_target(
    value: float,
    *,
    num_bins: int = DEFAULT_CATEGORICAL_BINS,
    vmin: float = -1.0,
    vmax: float = 1.0,
    sigma: float = 0.04,
) -> np.ndarray:
    """HL-Gauss categorical target distribution.

    Computes bin masses via Gaussian CDF differences (Imani et al. 2018), using
    `erf` to avoid SciPy.
    """
    value = float(np.clip(value, vmin, vmax))
    sigma = float(max(1e-6, sigma))

    edges = np.linspace(vmin, vmax, num_bins + 1, dtype=np.float64)
    z = (edges - value) / (sigma * math.sqrt(2.0))
  # Vectorized erf via numpy (np.frompyfunc is ~3x faster than list comprehension).
  # frompyfunc returns an object-dtype ndarray; pyright's stubs narrow it to
  # `float` so the downstream .astype call needs the suppression.
    erf_vec: np.ndarray = np.asarray(np.frompyfunc(math.erf, 1, 1)(z))
    cdfs = 0.5 * (1.0 + erf_vec.astype(np.float64))
    probs = cdfs[1:] - cdfs[:-1]
    s = float(probs.sum())
    if s <= 0:
        out = np.zeros((num_bins,), dtype=np.float32)
        out[int(np.clip(int((value - vmin) / (vmax - vmin) * num_bins), 0, num_bins - 1))] = 1.0
        return out

    probs = probs / s
    return probs.astype(np.float32, copy=False)
