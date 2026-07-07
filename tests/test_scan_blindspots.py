"""Pure band statistics for the blind-spot scanner (scripts/scan_blindspots.py)."""
from __future__ import annotations

import numpy as np

from scripts.scan_blindspots import band_stat


def test_band_counts_only_value_blind() -> None:
    #            fine+lost   fine+ok   losing+lost   fine+lost
    net_q = np.array([0.60, 0.60, -0.40, 0.30])
    sf_q = np.array([-0.60, 0.10, -0.60, -0.55])
    isp = np.array([True, True, True, False])  # last is curriculum
    s = band_stat(net_q, sf_q, isp, pos_key=None, net_ok=0.2, sf_lost=-0.5)
    assert s.count == 2                       # rows 0 and 3 (net_q>0.2 & sf_q<-0.5)
    assert abs(s.curriculum_frac - 0.5) < 1e-9  # one of the two is curriculum
    assert abs(s.severity_median - ((1.2 + 0.85) / 2)) < 1e-6


def test_band_severity_is_gap() -> None:
    net_q = np.array([0.8])
    sf_q = np.array([-0.7])
    s = band_stat(net_q, sf_q, np.array([True]), None, net_ok=0.5, sf_lost=-0.5)
    assert s.count == 1
    assert abs(s.severity_median - 1.5) < 1e-6  # 0.8 - (-0.7)


def test_band_empty_when_threshold_excludes_all() -> None:
    net_q = np.array([0.60, 0.30])
    sf_q = np.array([-0.20, -0.10])  # neither below -0.5
    s = band_stat(net_q, sf_q, np.array([True, True]), None, net_ok=0.2, sf_lost=-0.5)
    assert s.count == 0
    assert s.frac_of_evald == 0.0
    assert s.curriculum_frac == 0.0


def test_band_unique_fraction() -> None:
    # 3 blind rows, 2 sharing a placement key -> 2 unique / 3.
    net_q = np.array([0.6, 0.6, 0.6])
    sf_q = np.array([-0.6, -0.6, -0.6])
    keys = np.array([10, 10, 20], dtype=np.uint64)
    s = band_stat(net_q, sf_q, np.array([True, True, True]), keys, net_ok=0.2, sf_lost=-0.5)
    assert s.count == 3
    assert abs(s.unique_frac - 2 / 3) < 1e-9
