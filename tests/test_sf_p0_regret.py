"""The P0 own-move SF cp-regret vector (sf_p0_regret) must survive the shard
serialize + batch-collate round-trip, stay a value vector (NOT renormalized to
a distribution), and be absent/masked on legacy rows.

Mirrors tests/test_sf_p0_policy.py; the regret vector is the value-vector twin
of the sf_p0_policy_target distribution field.
"""

from __future__ import annotations

import numpy as np

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.dataset import collate as samples_to_batch
from chess_anti_engine.replay.shard import samples_to_arrays

_W = 4672


def _sample(reg: np.ndarray | None) -> ReplaySample:
    return ReplaySample(
        x=np.zeros((1, 8, 8), dtype=np.float32),
        policy_target=np.zeros((_W,), dtype=np.float32),
        wdl_target=0,
        sf_p0_regret=reg,
        has_sf_p0_regret=(reg is not None),
    )


def test_sf_p0_regret_roundtrips_through_shard() -> None:
    reg = np.ones((_W,), dtype=np.float32)
    reg[5] = 0.0  # best move -> regret 0
    reg[9] = 0.25
    arrs = samples_to_arrays([_sample(reg), _sample(None)])
    assert arrs["has_sf_p0_regret"].tolist() == [1, 0]
    # Best move's 0.0 is dropped by the sparse store and densifies back to 0.0.
    assert float(arrs["sf_p0_regret"][0][5]) == 0.0
    assert float(arrs["sf_p0_regret"][0][9]) == 0.25
    assert float(arrs["sf_p0_regret"][0][7]) == 1.0  # untouched index stays 1.0
    assert float(arrs["sf_p0_regret"][1].sum()) == 0.0  # legacy/absent row stays zero


def test_sf_p0_regret_in_batch_dict() -> None:
    reg = np.ones((_W,), dtype=np.float32)
    reg[7] = 0.0
    batch = samples_to_batch([_sample(reg), _sample(None)], device="cpu")
    assert "sf_p0_regret_t" in batch
    assert "has_sf_p0_regret" in batch
    assert batch["has_sf_p0_regret"].tolist() == [1.0, 0.0]
    assert float(batch["sf_p0_regret_t"][0][7]) == 0.0
    assert float(batch["sf_p0_regret_t"][0][3]) == 1.0


def test_sf_p0_regret_is_not_renormalized() -> None:
    """A regret vector with two 1.0 entries must stay [1.0, 1.0, ...] through the
    shard round-trip — it is a per-move value vector, not a sum-to-1
    distribution (so it must NOT be in _OPTIONAL_DISTRIBUTION_FIELDS)."""
    reg = np.zeros((_W,), dtype=np.float32)
    reg[1] = 1.0
    reg[2] = 1.0
    arrs = samples_to_arrays([_sample(reg)])
    assert float(arrs["sf_p0_regret"][0][1]) == 1.0
    assert float(arrs["sf_p0_regret"][0][2]) == 1.0
    batch = samples_to_batch([_sample(reg)], device="cpu")
    assert float(batch["sf_p0_regret_t"][0][1]) == 1.0
    assert float(batch["sf_p0_regret_t"][0][2]) == 1.0


def test_legacy_sample_without_sf_p0_regret_has_flag_zero() -> None:
    s = ReplaySample(
        x=np.zeros((1, 8, 8), dtype=np.float32),
        policy_target=np.zeros((_W,), dtype=np.float32),
        wdl_target=0,
    )
    arrs = samples_to_arrays([s])
    assert arrs["has_sf_p0_regret"].tolist() == [0]


def test_absent_sf_p0_regret_allocates_no_dense_array() -> None:
    """Default-off (no sample carries sf_p0_regret): samples_to_arrays must NOT
    allocate the policy-sized dense zero array (only the cheap flag), so the
    experiment imposes no per-batch memory cost when disabled."""
    samples = [
        ReplaySample(
            x=np.zeros((1, 8, 8), dtype=np.float32),
            policy_target=np.zeros((_W,), dtype=np.float32),
            wdl_target=0,
        )
        for _ in range(4)
    ]
    arrs = samples_to_arrays(samples)
    assert "sf_p0_regret" not in arrs  # dense array skipped
    assert "has_sf_p0_regret" in arrs  # cheap flag still present
    assert int(arrs["has_sf_p0_regret"].sum()) == 0


def _valid_sample(reg: np.ndarray | None) -> ReplaySample:
    policy = np.zeros((_W,), dtype=np.float32)
    policy[0] = 1.0  # valid distribution (passes prune validation)
    return ReplaySample(
        x=np.zeros((1, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=0,
        sf_p0_regret=reg,
        has_sf_p0_regret=(reg is not None),
    )


def test_sf_p0_regret_full_write_prune_read_roundtrip() -> None:
    from chess_anti_engine.replay.shard import (
        arrays_to_samples,
        prune_storage_arrays,
        samples_to_arrays,
    )

    reg = np.ones((_W,), dtype=np.float32)
    reg[9] = 0.0  # SF best move
    reg[3] = 0.5
    arrs = prune_storage_arrays(samples_to_arrays([_valid_sample(reg)]))
    out = arrays_to_samples(arrs)
    assert out[0].sf_p0_regret is not None
    assert float(out[0].sf_p0_regret[9]) == 0.0
    assert float(out[0].sf_p0_regret[3]) == 0.5
    assert float(out[0].sf_p0_regret[7]) == 1.0


def test_sf_p0_regret_absent_reader_no_alloc_and_none() -> None:
    """Reading an experiment-off shard: no dense sf_p0_regret array on disk, and
    arrays_to_samples returns sf_p0_regret=None without allocating a
    policy-sized zero."""
    from chess_anti_engine.replay.shard import (
        arrays_to_samples,
        prune_storage_arrays,
        samples_to_arrays,
    )

    arrs = prune_storage_arrays(samples_to_arrays([_valid_sample(None)]))
    assert "sf_p0_regret" not in arrs  # not persisted
    out = arrays_to_samples(arrs)
    assert out[0].sf_p0_regret is None
