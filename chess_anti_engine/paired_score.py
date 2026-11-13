"""Fixed-count intervals for color-paired game-score means.

A pair is the mean of two game scores in {0, 0.5, 1}. Positive sample variance
uses a normal approximation; a singleton or constant sample uses the stated
bounded Hoeffding radius. This mixed rule is not an exact coverage or global
error guarantee. Independence, opening-book scope, selection and multiplicity
remain the caller's responsibility. No Elo or promotion decision is computed.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import fsum, isfinite, log, sqrt
from typing import Literal, TypedDict

_PAIR_MEANS = (0.0, 0.25, 0.5, 0.75, 1.0)


class PairedScoreInterval(TypedDict):
    method: Literal['paired_normal_approx_95', 'hoeffding_95_fallback']
    mean: float
    variance: float | None
    count: int
    lower: float
    upper: float
    radius: float
    confidence: float
    fallback_reason: Literal['single_pair', 'zero_variance'] | None


def paired_score_interval(pair_means: Sequence[object], *, expected_pairs: object) -> PairedScoreInterval:
    """Validate complete pair means and return full-precision serializable stats.

    ``variance`` is the unbiased sample variance (n-1), undefined/None at n=1.
    The normal radius is 1.96*sqrt(variance/n). The fallback radius is
    sqrt(log(40)/(2*n)); both intervals are clamped to [0,1]. Booleans,
    nonfinite values, non-quarter means, empty input and count mismatch refuse.
    No rounded summary, pair total, missing pair or effective-N substitution is
    accepted. Builtin int/float values are the supported JSON numeric inputs.
    """
    if isinstance(expected_pairs, bool) or not isinstance(expected_pairs, int) or expected_pairs < 1:
        raise ValueError('expected_pairs must be a positive nonboolean integer')
    n = len(pair_means)
    if n == 0 or n != expected_pairs:
        raise ValueError('complete nonempty pair count must equal expected_pairs')
    values: list[float] = []
    for value in pair_means:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('pair means must be finite nonboolean numeric quarter scores')
        # Membership rejects arbitrarily large integers before any float cast.
        if value not in _PAIR_MEANS or not isfinite(value):
            raise ValueError('pair means must be finite quarter scores in [0,1]')
        values.append(float(value))
    mean = fsum(values) / n
    variance = fsum((value - mean) ** 2 for value in values) / (n - 1) if n >= 2 else None
    reason: Literal['single_pair', 'zero_variance'] | None
    method: Literal['paired_normal_approx_95', 'hoeffding_95_fallback']
    if variance is not None and variance > 0:
        method = 'paired_normal_approx_95'
        radius = 1.96 * sqrt(variance / n)
        reason = None
    else:
        method = 'hoeffding_95_fallback'
        radius = sqrt(log(40) / (2 * n))
        reason = 'single_pair' if n == 1 else 'zero_variance'
    return {'method': method, 'mean': mean, 'variance': variance, 'count': n,
            'lower': max(0.0, mean - radius), 'upper': min(1.0, mean + radius),
            'radius': radius, 'confidence': 0.95, 'fallback_reason': reason}
