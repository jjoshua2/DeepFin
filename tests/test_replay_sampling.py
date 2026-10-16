from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.replay.buffer import ArrayReplayBuffer
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.sampling import (
    priority_probabilities,
    sample_pool_indices,
    wdl_batch_counts,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_priority_normalization_preserves_input(dtype) -> None:
    pri = np.array([np.nan, np.inf, -np.inf, -2, 0, 2, 6], dtype=dtype)
    original = pri.copy()
    pri.flags.writeable = False
    p = priority_probabilities(pri)
    np.testing.assert_array_equal(p, [0, 0, 0, 0, 0, .25, .75])
    np.testing.assert_array_equal(pri, original)
    assert p is not None
    assert p.dtype == np.float64
    assert not np.shares_memory(pri, p)


@pytest.mark.parametrize("values", [[0, 0], [-1, np.nan], [np.inf, -np.inf], [np.finfo(float).max]*2])
def test_unusable_priorities_request_uniform_sampling(values) -> None:
    with np.errstate(over="ignore"):
        assert priority_probabilities(np.asarray(values)) is None


@pytest.mark.parametrize(("batch_size", "pool", "mix"), [(0, [2], .5), (3, [], .5), (7, [2, 5, 2], 0)])
def test_uniform_or_empty_pool_does_not_read_priorities(batch_size, pool, mix) -> None:
    def forbidden() -> np.ndarray:
        raise AssertionError("priorities read without a weighted draw")

    rng = np.random.default_rng(7)
    expected_rng = np.random.default_rng(7)
    values = np.asarray(pool, dtype=np.int64)
    got = sample_pool_indices(values, batch_size, rng=rng, surprise_mix=mix, priority=forbidden)
    expected = values[expected_rng.choice(len(values), size=batch_size)] if batch_size and len(values) else np.empty(0, dtype=np.int64)
    np.testing.assert_array_equal(got, expected)
    assert rng.bit_generator.state == expected_rng.bit_generator.state


@pytest.mark.parametrize(("batch", "size", "wins", "draws", "losses", "cap", "expected"), [
    (100, 10_000, 100, 9_000, 900, .9, (90, 4, 6)),
    (3, 3, 1, 1, 1, .9, (1, 1, 1)),
    (1, 2, 1, 0, 1, .9, (0, 1, 0)),
    (2, 4, 1, 2, 1, .9, (1, 1, 0)),
    (3, 8, 1, 6, 1, .25, (0, 1, 2)),
    (7, 8, 1, 1, 1, .9, (1, 3, 3)),
    (-1, 2, 1, 0, 1, .9, (0, 0, 0)),
])
def test_wdl_quotas_preserve_caps_rounding_and_unknown_labels(batch, size, wins, draws, losses, cap, expected) -> None:
    assert wdl_batch_counts(batch, size=size, wins=wins, draws=draws, losses=losses, draw_cap_frac=cap, wl_max_ratio=1.5) == expected


@pytest.mark.parametrize("backend", ["array", "disk"])
def test_buffer_consumes_wdl_quotas(tmp_path: Path, backend: str) -> None:
    rows = {
        "x": np.zeros((8, 146, 8, 8), dtype=np.float16),
        "policy_target": np.ones((8, 4672), dtype=np.float16) / 4672,
        "wdl_target": np.array([0]+[1]*6+[2], dtype=np.int8),
        "priority": np.arange(1, 9, dtype=np.float32),
        "has_policy": np.ones(8, dtype=np.uint8),
    }
    rng = np.random.default_rng(7)
    if backend == "disk":
        buf = DiskReplayBuffer(8, shard_dir=tmp_path, rng=rng, read_only=False,
                               shuffle_cap=8, shard_size=32, draw_cap_frac=.25,
                               wl_max_ratio=1.0, refresh_interval=0)
        expected = [3, 1, 3]
    else:
        buf = ArrayReplayBuffer(8, rng=rng)
        expected = [1, 5, 1]
    try:
        buf.add_many_arrays(rows)
        batch = buf.sample_batch_arrays(7)
        np.testing.assert_array_equal(np.bincount(batch["wdl_target"], minlength=3), expected)
    finally:
        if isinstance(buf, DiskReplayBuffer):
            buf.close()
