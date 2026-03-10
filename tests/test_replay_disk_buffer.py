from __future__ import annotations

import numpy as np

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer


def _sample() -> ReplaySample:
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=np.zeros((4672,), dtype=np.float32),
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

    assert len(buf._shuffle_buf) == 20


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
    assert len(buf._shuffle_buf) == 35

    buf.capacity = 15
    batch = buf.sample_batch(4, wdl_balance=False)

    assert len(batch) == 4
    assert len(buf._shuffle_buf) == 15
