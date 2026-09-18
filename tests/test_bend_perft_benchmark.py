"""Cheap benchmark-report contracts only: no compiler, perft, or timing loops."""
from __future__ import annotations

import pytest

from native.bend_engine.legal_probe.benchmark import parse_sample, summary


def test_sample_requires_exact_warmup_and_measured_counts() -> None:
    assert parse_sample("warmup 0 400\nbench bend-pext 2 400 100000\n", 2, 400, bend=True) == {
        "backend": "bend-pext", "nodes": 400, "nanoseconds": 100000,
    }
    assert parse_sample("warmup 1 4\nbench pext 5 4294967300 100\n", 5, 2**32 + 4, bend=False)["nodes"] == 2**32 + 4


@pytest.mark.parametrize(("text", "reason"), [
    ("", "warmup"),
    ("warmup 0 399\nbench bend-pext 2 400 100\n", "warmup count"),
    ("warmup 4294967296 400\nbench bend-pext 2 400 100\n", "warmup count"),
    ("warmup 0 400\nbench bend-pext 3 400 100\n", "depth/count"),
    ("warmup 0 400\nbench bend-pext 2 399 100\n", "depth/count"),
    ("warmup 0 400\nbench bend-pext 2 400 0\n", "interval"),
    ("warmup 0 400\nbench bend-pext 2 400 120000000000\n", "interval"),
    ("warmup 0 400\nbench bend-pext 2 400 -1\n", "timing row"),
    ("warmup 0 400\nbench rays 2 400 100\n", "timing row"),
    ("warmup 0 400\nbench pext 2 400 100\n", "backend"),
    ("warmup 0 400\nbench bend-pext 2 400 100\nextra\n", "warmup"),
])
def test_bad_sample_rejected(text: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_sample(text, 2, 400, bend=True)


def test_timing_summary_keeps_spread_and_uses_median() -> None:
    result = summary([1000000, 9000000, 2000000], 1000)
    assert result["median_seconds"] == 0.002
    assert result["median_leaf_nodes_per_second"] == 500000
    assert result["min_seconds"] == 0.001
    assert result["max_seconds"] == 0.009
    with pytest.raises(ValueError, match="positive"):
        summary([], 1000)
    with pytest.raises(ValueError, match="positive"):
        summary([0], 1000)
