"""Bounded-memory bootstrap controls against the original vectorized estimator."""
from __future__ import annotations

import numpy as np
import pytest

from scripts import paired_compare as pc


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("seed", [0, 19])
@pytest.mark.parametrize("alpha", [0.05, 0.2])
def test_row_chunks_preserve_vectorized_interval(monkeypatch, dtype, seed, alpha):
    values = np.array([0, 1, -8, 3, 11, 2, -2] * 7, dtype=dtype)[::2]
    original = values.copy()
    n_boot = 29
    indices = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    expected = np.percentile(values[indices].mean(axis=1), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    monkeypatch.setattr(pc, "_BOOTSTRAP_MAX_ELEMENTS", len(values) * 3, raising=False)
    actual = pc.paired_bootstrap_ci(values, n_boot=n_boot, seed=seed, alpha=alpha)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("seed", [0, 19])
@pytest.mark.parametrize("alpha", [0.05, 0.2])
def test_cluster_chunks_preserve_weighted_row_mean(monkeypatch, seed, alpha):
    clusters = np.array(["g-b"] * 3 + ["g-a"] + ["g-c"] * 7 + ["g-d"] * 2)
    values = np.array([1, 2, 7, -3, 8, 9, -6, 1, 4, 8, 2, 4, 3], dtype=np.float32)
    unique, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    sampled = np.random.default_rng(seed).integers(0, len(unique), size=(29, len(unique)))
    expected = np.percentile(sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1),
                             [100 * alpha / 2, 100 * (1 - alpha / 2)])
    monkeypatch.setattr(pc, "_BOOTSTRAP_MAX_ELEMENTS", len(unique) * 3, raising=False)
    actual = pc.paired_cluster_bootstrap_ci(values, clusters, n_boot=29, seed=seed, alpha=alpha)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize(("sample_count", "budget"), [(257, 1024), (257, 100)])
def test_rng_request_is_bounded_to_chunks_or_one_draw(monkeypatch, clustered, sample_count, budget):
    factory = np.random.default_rng
    calls = []

    class BoundedRNG:
        def __init__(self, seed):
            self.rng = factory(seed)

        def integers(self, low, high, *, size):
            assert len(size) == 2
            assert size[1] == sample_count
            assert np.prod(size) <= max(budget, sample_count)
            calls.append(size)
            return self.rng.integers(low, high, size=size)

    monkeypatch.setattr(pc, "_BOOTSTRAP_MAX_ELEMENTS", budget, raising=False)
    monkeypatch.setattr(pc.np.random, "default_rng", BoundedRNG)
    values = np.linspace(-3, 5, sample_count)
    if clustered:
        pc.paired_cluster_bootstrap_ci(values, np.arange(sample_count), n_boot=19)
    else:
        pc.paired_bootstrap_ci(values, n_boot=19)
    assert sum(shape[0] for shape in calls) == 19
    assert len(calls) > 1


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("n_boot", [0, -1])
def test_nonpositive_draw_count_is_refused(monkeypatch, clustered, n_boot):
    def unexpected_rng(*_args, **_kwargs):
        pytest.fail("invalid draw count reached random allocation")

    monkeypatch.setattr(pc.np.random, "default_rng", unexpected_rng)

    def invoke():
        if clustered:
            return pc.paired_cluster_bootstrap_ci(np.array([1., 2.]), np.array([0, 1]), n_boot=n_boot)
        return pc.paired_bootstrap_ci(np.array([1., 2.]), n_boot=n_boot)

    with pytest.raises(ValueError, match="n_boot must be positive"):
        invoke()
