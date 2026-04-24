import numpy as np

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.dataset import collate as samples_to_batch
from chess_anti_engine.replay.shard import samples_to_arrays


def _make_sample(is_selfplay: bool | None) -> ReplaySample:
    return ReplaySample(
        x=np.zeros((1, 8, 8), dtype=np.float32),
        policy_target=np.zeros((4672,), dtype=np.float32),
        wdl_target=0,
        is_selfplay=is_selfplay,
    )


def test_is_selfplay_roundtrips_through_shard():
    samples = [_make_sample(True), _make_sample(False), _make_sample(None)]
    arrs = samples_to_arrays(samples)
    assert arrs["is_selfplay"].tolist() == [1, 0, 0]
    assert arrs["has_is_selfplay"].tolist() == [1, 1, 0]


def test_is_selfplay_in_batch_dict():
    samples = [_make_sample(True), _make_sample(False)]
    batch = samples_to_batch(samples, device="cpu")
    assert "is_selfplay" in batch
    assert "has_is_selfplay" in batch
    assert batch["is_selfplay"].tolist() == [True, False]
    assert batch["has_is_selfplay"].tolist() == [1.0, 1.0]


def test_legacy_sample_without_is_selfplay():
    """Legacy shards that don't set is_selfplay should produce has_is_selfplay=0."""
    s = ReplaySample(
        x=np.zeros((1, 8, 8), dtype=np.float32),
        policy_target=np.zeros((4672,), dtype=np.float32),
        wdl_target=0,
    )
    arrs = samples_to_arrays([s])
    assert arrs["has_is_selfplay"].tolist() == [0]
