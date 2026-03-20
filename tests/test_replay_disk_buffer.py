from __future__ import annotations

import numpy as np
import pytest

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay import shard as shard_mod
from chess_anti_engine.replay.shard import LEGACY_SHARD_SUFFIX, iter_shard_paths


def _sample() -> ReplaySample:
    policy = np.zeros((4672,), dtype=np.float32)
    policy[0] = 1.0
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=0,
        priority=1.0,
        has_policy=True,
    )


def test_shuffle_buffer_capped_by_capacity(tmp_path) -> None:
    rng = np.random.default_rng(0)
    buf = DiskReplayBuffer(
        20,
        shard_dir=tmp_path / "replay",
        rng=rng,
        shuffle_cap=70,
        shard_size=10,
    )

    buf.add_many([_sample() for _ in range(35)])

    assert buf._shuffle_len() == 20


def test_shuffle_buffer_retrimmed_after_capacity_shrink(tmp_path) -> None:
    rng = np.random.default_rng(0)
    buf = DiskReplayBuffer(
        50,
        shard_dir=tmp_path / "replay",
        rng=rng,
        shuffle_cap=70,
        shard_size=10,
    )

    buf.add_many([_sample() for _ in range(35)])
    assert buf._shuffle_len() == 35

    buf.capacity = 15
    batch = buf.sample_batch(4, wdl_balance=False)

    assert len(batch) == 4
    assert buf._shuffle_len() == 15


def test_sample_batch_arrays_shapes(tmp_path) -> None:
    rng = np.random.default_rng(0)
    buf = DiskReplayBuffer(
        50,
        shard_dir=tmp_path / "replay",
        rng=rng,
        shuffle_cap=20,
        shard_size=10,
    )

    buf.add_many([_sample() for _ in range(12)])
    arrs = buf.sample_batch_arrays(6, wdl_balance=False)

    assert arrs["x"].shape == (6, 146, 8, 8)
    assert arrs["policy_target"].shape == (6, 4672)
    assert arrs["wdl_target"].shape == (6,)
    assert arrs["priority"].shape == (6,)


def test_resumed_buffer_samples_from_pruned_optional_shards(tmp_path) -> None:
    rng = np.random.default_rng(0)
    shard_dir = tmp_path / "replay"
    buf = DiskReplayBuffer(
        50,
        shard_dir=shard_dir,
        rng=rng,
        shuffle_cap=20,
        shard_size=4,
    )

    buf.add_many([_sample() for _ in range(6)])
    buf.flush()

    resumed = DiskReplayBuffer(
        50,
        shard_dir=shard_dir,
        rng=np.random.default_rng(1),
        shuffle_cap=20,
        shard_size=4,
    )
    arrs = resumed.sample_batch_arrays(4, wdl_balance=False)

    assert arrs["x"].shape == (4, 146, 8, 8)
    assert arrs["policy_target"].shape == (4, 4672)
    assert arrs["wdl_target"].shape == (4,)
    assert "sf_wdl" not in arrs
    assert "has_sf_wdl" not in arrs
    assert "future_policy_target" not in arrs
    assert "has_future" not in arrs


def test_window_enforcement_deletes_directory_backed_local_shards(tmp_path) -> None:
    rng = np.random.default_rng(0)
    shard_dir = tmp_path / "replay"
    buf = DiskReplayBuffer(
        2,
        shard_dir=shard_dir,
        rng=rng,
        shuffle_cap=2,
        shard_size=1,
    )

    buf.add_many([_sample() for _ in range(3)])
    buf.flush()

    shard_paths = iter_shard_paths(shard_dir)
    if shard_paths and not shard_paths[0].is_dir():
        pytest.skip("local shards are file-backed in this environment")

    assert len(buf._shard_paths) == 2
    assert len(shard_paths) == 2
    assert [p.name for p in shard_paths] == [p.name for p in buf._shard_paths]


def test_flush_tracks_npz_fallback_path_when_zarr_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shard_mod, "zarr", None)
    monkeypatch.setattr(shard_mod, "Blosc", None)

    rng = np.random.default_rng(0)
    buf = DiskReplayBuffer(
        10,
        shard_dir=tmp_path / "replay",
        rng=rng,
        shuffle_cap=10,
        shard_size=4,
    )

    buf.add_many([_sample() for _ in range(4)])
    buf.flush()

    assert len(buf._shard_paths) == 1
    saved_path = buf._shard_paths[0]
    assert saved_path.suffix == LEGACY_SHARD_SUFFIX
    assert saved_path.exists()
    assert iter_shard_paths(tmp_path / "replay") == [saved_path]
