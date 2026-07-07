"""Pure resolution classification (scripts/blindspot_resolution.py)."""
from __future__ import annotations

import numpy as np

from scripts.blindspot_resolution import ResolutionSummary, classify


def test_classify_splits_blind_and_resolved() -> None:
    # net_q: two still-blind (net thinks fine), two resolved (net reads losing).
    q = np.array([0.6, 0.3, -0.1, -0.7])
    s = classify(q, resolved_below=0.0)
    assert s.n == 4
    assert s.blind == 2         # 0.6, 0.3 still > 0
    assert s.resolved == 2      # -0.1, -0.7 <= 0
    assert abs(s.resolved_frac - 0.5) < 1e-9


def test_classify_threshold_is_inclusive() -> None:
    assert classify(np.array([0.0]), resolved_below=0.0).resolved == 1     # exactly 0.0 counts
    assert classify(np.array([-0.4]), resolved_below=-0.4).resolved == 1   # -0.4 counts
    assert classify(np.array([-0.39]), resolved_below=-0.4).resolved == 0  # -0.39 does not


def test_classify_empty() -> None:
    s = classify(np.array([]), resolved_below=0.0)
    assert s == ResolutionSummary(0, 0, 0)
    assert s.resolved_frac == 0.0
