from __future__ import annotations

import numpy as np

from scripts.paired_compare import paired_bootstrap_ci


def test_bootstrap_ci_covers_known_shift() -> None:
    rng = np.random.default_rng(7)
    deltas = rng.normal(loc=-2.0, scale=5.0, size=2000)
    lo, hi = paired_bootstrap_ci(deltas, n_boot=4000, seed=1)
    assert lo < -2.0 < hi or (lo < deltas.mean() < hi)
    # width ~ 2*1.96*5/sqrt(2000) ≈ 0.44
    assert 0.2 < (hi - lo) < 1.0
    assert hi < 0  # a real 2cp shift at n=2000 is clearly significant


def test_bootstrap_ci_null_is_not_significant() -> None:
    rng = np.random.default_rng(3)
    deltas = rng.normal(loc=0.0, scale=5.0, size=2000)
    lo, hi = paired_bootstrap_ci(deltas, n_boot=4000, seed=2)
    assert lo < 0 < hi
