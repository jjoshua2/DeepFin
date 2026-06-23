"""The P0 own-move SF teacher (sf_p0_policy_target) must survive the shard
serialize + batch-collate round-trip, and be absent/masked on legacy rows."""

from __future__ import annotations

import numpy as np

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.dataset import collate as samples_to_batch
from chess_anti_engine.replay.shard import samples_to_arrays

_W = 4672


def _sample(sf_p0: np.ndarray | None) -> ReplaySample:
    return ReplaySample(
        x=np.zeros((1, 8, 8), dtype=np.float32),
        policy_target=np.zeros((_W,), dtype=np.float32),
        wdl_target=0,
        sf_p0_policy_target=sf_p0,
        has_sf_p0=(sf_p0 is not None),
    )


def test_sf_p0_policy_roundtrips_through_shard() -> None:
    tgt = np.zeros((_W,), dtype=np.float32)
    tgt[5] = 1.0
    arrs = samples_to_arrays([_sample(tgt), _sample(None)])
    assert arrs["has_sf_p0"].tolist() == [1, 0]
    assert float(arrs["sf_p0_policy_target"][0][5]) == 1.0
    assert float(arrs["sf_p0_policy_target"][1].sum()) == 0.0


def test_sf_p0_policy_in_batch_dict() -> None:
    tgt = np.zeros((_W,), dtype=np.float32)
    tgt[7] = 1.0
    batch = samples_to_batch([_sample(tgt), _sample(None)], device="cpu")
    assert "sf_p0_policy_t" in batch
    assert "has_sf_p0" in batch
    assert batch["has_sf_p0"].tolist() == [1.0, 0.0]
    assert float(batch["sf_p0_policy_t"][0][7]) == 1.0


def test_legacy_sample_without_sf_p0_has_flag_zero() -> None:
    s = ReplaySample(
        x=np.zeros((1, 8, 8), dtype=np.float32),
        policy_target=np.zeros((_W,), dtype=np.float32),
        wdl_target=0,
    )
    arrs = samples_to_arrays([s])
    assert arrs["has_sf_p0"].tolist() == [0]


def test_absent_sf_p0_allocates_no_dense_array() -> None:
    """Default-off (no sample carries sf_p0): samples_to_arrays must NOT allocate
    the policy-sized dense zero array (only the cheap flag), so the experiment
    imposes no per-batch memory cost when disabled."""
    from chess_anti_engine.replay.shard import samples_to_arrays

    samples = [
        ReplaySample(
            x=np.zeros((1, 8, 8), dtype=np.float32),
            policy_target=np.zeros((_W,), dtype=np.float32),
            wdl_target=0,
        )
        for _ in range(4)
    ]
    arrs = samples_to_arrays(samples)
    assert "sf_p0_policy_target" not in arrs  # dense array skipped
    assert "has_sf_p0" in arrs  # cheap flag still present
    assert int(arrs["has_sf_p0"].sum()) == 0


def _valid_sample(sf_p0: np.ndarray | None) -> ReplaySample:
    policy = np.zeros((_W,), dtype=np.float32)
    policy[0] = 1.0  # valid distribution (passes prune validation)
    return ReplaySample(
        x=np.zeros((1, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=0,
        sf_p0_policy_target=sf_p0,
        has_sf_p0=(sf_p0 is not None),
    )


def test_sf_p0_full_write_prune_read_roundtrip() -> None:
    from chess_anti_engine.replay.shard import (
        arrays_to_samples,
        prune_storage_arrays,
        samples_to_arrays,
    )

    tgt = np.zeros((_W,), dtype=np.float32)
    tgt[9] = 1.0
    arrs = prune_storage_arrays(samples_to_arrays([_valid_sample(tgt)]))
    out = arrays_to_samples(arrs)
    assert out[0].sf_p0_policy_target is not None
    assert float(out[0].sf_p0_policy_target[9]) == 1.0


def test_sf_p0_absent_reader_no_alloc_and_none() -> None:
    """Reading an experiment-off shard: no dense sf_p0 array on disk, and
    arrays_to_samples returns sf_p0=None without allocating a policy-sized zero."""
    from chess_anti_engine.replay.shard import (
        arrays_to_samples,
        prune_storage_arrays,
        samples_to_arrays,
    )

    arrs = prune_storage_arrays(samples_to_arrays([_valid_sample(None)]))
    assert "sf_p0_policy_target" not in arrs  # not persisted
    out = arrays_to_samples(arrs)
    assert out[0].sf_p0_policy_target is None
