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


def _wdl_expected_score(row: np.ndarray | None) -> float | None:
    """Side-to-move expected score ``(W - L) / (W + D + L)`` for a 3-vector WDL
    row, or ``None`` when the row is missing / ill-shaped / empty. Scale-invariant
    (works for normalized probs or SF's raw permille ``UCI_ShowWDL``)."""
    if row is None:
        return None
    arr = np.asarray(row, dtype=np.float32)
    if arr.shape != (3,):
        return None
    total = float(arr[0]) + float(arr[1]) + float(arr[2])
    if total <= 0.0:
        return None
    return (float(arr[0]) - float(arr[2])) / total


def categorical_target_value(
    scalar_v: float,
    sf_wdl: np.ndarray | None,
    *,
    blend_frac: float,
    search_wdl: np.ndarray | None = None,
    search_blend_frac: float = 0.0,
) -> float:
    """Continuous value for the categorical (HL-Gauss) value head.

    Default (both fracs ``== 0``) returns the ternary game outcome unchanged so
    the stored target is byte-identical to the legacy behaviour. Otherwise the
    value is a convex blend of up to three side-to-move expected scores, mirroring
    the main WDL head's target (see ``train/losses.py``)::

        game_frac * scalar_v + sf_frac * E[sf_wdl] + search_frac * E[search_wdl]

    Each expected score is ``(W - L) / (W + D + L)`` (scale-invariant, so it works
    for normalized probs or raw permille). ``sf_frac`` / ``search_frac`` come from
    ``blend_frac`` / ``search_blend_frac``; if their sum exceeds 1 they are
    renormalized and ``game_frac`` drops to 0 (same rule as the WDL head). A
    component whose row is missing/empty is dropped — its weight is NOT
    redistributed, so the blend simply leans on the remaining sources / the
    outcome (matching the WDL head's per-sample availability handling).

    Shared by the live finalize path (``selfplay/finalize.py``) and the offline
    rebuild (``train/target_builder.py``) so the two produce identical targets.
    """
    sf_frac = max(0.0, float(blend_frac))
    search_frac = max(0.0, float(search_blend_frac))
    sf_e = _wdl_expected_score(sf_wdl) if sf_frac > 0.0 else None
    search_e = _wdl_expected_score(search_wdl) if search_frac > 0.0 else None
    if sf_e is None:
        sf_frac = 0.0
    if search_e is None:
        search_frac = 0.0
    blend_sum = sf_frac + search_frac
    if blend_sum <= 0.0:
        return float(scalar_v)
    if blend_sum > 1.0:
        sf_frac /= blend_sum
        search_frac /= blend_sum
        game_frac = 0.0
    else:
        game_frac = 1.0 - blend_sum
    return (
        game_frac * float(scalar_v)
        + sf_frac * float(sf_e or 0.0)
        + search_frac * float(search_e or 0.0)
    )
