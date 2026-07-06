import numpy as np
import pytest

from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.replay.buffer import (
    ArrayReplayBuffer,
    ReplayBuffer,
    ReplaySample,
)


def test_surprise_sampling_batch_size():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(1000, rng=rng)

    x = np.zeros((146, 8, 8), dtype=np.float32)
    p = np.zeros((64 * 73,), dtype=np.float32)
    p[0] = 1.0

    for i in range(200):
        buf.add(ReplaySample(x=x, policy_target=p, wdl_target=1, priority=float(i + 1)))

    batch = buf.sample_batch(32)
    assert len(batch) == 32

    arrs = buf.sample_batch_arrays(16)
    assert arrs["x"].shape == (16, 146, 8, 8)
    assert arrs["policy_target"].shape == (16, 64 * 73)
    assert arrs["wdl_target"].shape == (16,)


def test_wdl_balance_draw_cap_and_wl_ratio():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(10_000, rng=rng)

    x = np.zeros((146, 8, 8), dtype=np.float32)
    p = np.zeros((64 * 73,), dtype=np.float32)
    p[0] = 1.0

    # Very draw-heavy buffer with some decisive outcomes.
    for i in range(9_000):
        buf.add(ReplaySample(x=x, policy_target=p, wdl_target=1, priority=float(i + 1)))
    for i in range(900):
        buf.add(ReplaySample(x=x, policy_target=p, wdl_target=2, priority=float(i + 1)))  # win
    for i in range(100):
        buf.add(ReplaySample(x=x, policy_target=p, wdl_target=0, priority=float(i + 1)))  # loss

    bs = 100
    batch = buf.sample_batch(bs)
    assert len(batch) == bs

    n_draw = sum(1 for s in batch if int(s.wdl_target) == 1)
    n_win = sum(1 for s in batch if int(s.wdl_target) == 2)
    n_loss = sum(1 for s in batch if int(s.wdl_target) == 0)

    # Draw cap is 90%.
    assert n_draw <= int(np.floor(0.90 * bs))

    # Among decisive slots, enforce max ratio of 1.5x when both classes exist.
    if (n_win > 0) and (n_loss > 0):
        assert max(n_win, n_loss) <= int(np.floor(1.5 * min(n_win, n_loss)))


def test_wdl_balance_falls_back_without_both_decisive_classes():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(10_000, rng=rng)

    x = np.zeros((146, 8, 8), dtype=np.float32)
    p = np.zeros((64 * 73,), dtype=np.float32)
    p[0] = 1.0

    # Draws + wins only (no losses). We should fall back to _sample_raw behavior,
    # meaning draws are NOT artificially capped.
    for i in range(9_500):
        buf.add(ReplaySample(x=x, policy_target=p, wdl_target=1, priority=float(i + 1)))
    for i in range(500):
        buf.add(ReplaySample(x=x, policy_target=p, wdl_target=2, priority=float(i + 1)))

    bs = 100
    batch = buf.sample_batch(bs)
    assert len(batch) == bs
    n_draw = sum(1 for s in batch if int(s.wdl_target) == 1)

    # With p(draw)=0.95, we'd expect ~95 draws under raw sampling; allow some noise.
    assert n_draw >= 85


def test_array_replay_buffer_arrays_and_capacity():
    rng = np.random.default_rng(0)
    buf = ArrayReplayBuffer(50, rng=rng)

    x = np.zeros((146, 8, 8), dtype=np.float32)
    p = np.zeros((64 * 73,), dtype=np.float32)
    p[0] = 1.0

    for i in range(80):
        buf.add(ReplaySample(x=x, policy_target=p, wdl_target=i % 3, priority=float(i + 1)))

    assert len(buf) == 50
    arrs = buf.sample_batch_arrays(20)
    assert arrs["x"].shape == (20, 146, 8, 8)
    assert arrs["policy_target"].shape == (20, 64 * 73)
    assert arrs["wdl_target"].shape == (20,)


def test_array_replay_buffer_capacity_preserves_scalar_metadata():
    rng = np.random.default_rng(0)
    buf = ArrayReplayBuffer(3, rng=rng)

    x = np.zeros((5, 146, 8, 8), dtype=np.float16)
    policy = np.zeros((5, 4672), dtype=np.float16)
    policy[:, 0] = 1.0
    buf.add_many_arrays({
        "x": x,
        "policy_target": policy,
        "wdl_target": np.arange(5, dtype=np.int8) % 3,
        "priority": np.arange(5, dtype=np.float32) + 1.0,
        "has_policy": np.ones((5,), dtype=np.uint8),
    })

    assert len(buf) == 3
    chunk = buf._chunks[0]
    assert np.asarray(chunk["_policy_size"]).shape == ()
    assert int(np.asarray(chunk["_policy_size"]).item()) == 4672

    arrs = buf.sample_batch_arrays(2, wdl_balance=False)
    assert arrs["policy_target"].shape == (2, 4672)


def test_array_replay_buffer_rejects_out_of_range_gather_indices():
    rng = np.random.default_rng(0)
    buf = ArrayReplayBuffer(3, rng=rng)

    x = np.zeros((1, 146, 8, 8), dtype=np.float16)
    policy = np.zeros((1, 4672), dtype=np.float16)
    policy[:, 0] = 1.0
    buf.add_many_arrays({
        "x": x,
        "policy_target": policy,
        "wdl_target": np.zeros((1,), dtype=np.int8),
        "priority": np.ones((1,), dtype=np.float32),
        "has_policy": np.ones((1,), dtype=np.uint8),
    })

    with pytest.raises(ValueError, match="did not match any replay chunk"):
        buf._gather_rows(np.array([10], dtype=np.int64))


def test_array_replay_buffer_accepts_compact_policy_after_training_supports_it():
    rng = np.random.default_rng(0)
    buf = ArrayReplayBuffer(10, rng=rng)
    x = np.zeros((1, 146, 8, 8), dtype=np.float16)
    compact = np.zeros((1, COMPACT_POLICY_SIZE), dtype=np.float16)
    compact[:, 0] = 1.0

    common = {
        "x": x,
        "wdl_target": np.zeros((1,), dtype=np.int8),
        "priority": np.ones((1,), dtype=np.float32),
        "has_policy": np.ones((1,), dtype=np.uint8),
    }
    buf.add_many_arrays({**common, "policy_target": compact})

    batch = buf.sample_batch_arrays(1, wdl_balance=False)
    assert batch["policy_target"].shape == (1, COMPACT_POLICY_SIZE)


def _signed_gap_buffer(signed: bool):
    """Bare DiskReplayBuffer to exercise _shape_shuffle_priority in isolation."""
    from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer

    b = DiskReplayBuffer.__new__(DiskReplayBuffer)
    b.sf_gap_priority_weight = 30.0
    b.sf_gap_priority_signed = signed
    b.fast_low_surprise_priority = 1.0
    return b


def test_signed_gap_priority_boosts_only_over_optimism():
    # WDL = [W, D, L]; signed value = W - L.
    #  row0 over-optimistic (search +0.8 vs sf -0.6) -> boosted in BOTH modes
    #  row1 pessimistic     (search -0.6 vs sf +0.8) -> boosted only by abs()
    #  row2 agree           (both 0.0)               -> boosted by neither
    search_wdl = np.array([[0.9, 0.0, 0.1], [0.1, 0.3, 0.7], [0.3, 0.4, 0.3]], np.float32)
    sf_wdl = np.array([[0.1, 0.3, 0.7], [0.9, 0.0, 0.1], [0.3, 0.4, 0.3]], np.float32)
    abs_gap = np.abs(
        (search_wdl[:, 0] - search_wdl[:, 2]) - (sf_wdl[:, 0] - sf_wdl[:, 2])
    ).astype(np.float32)
    arrs = {
        "priority": np.ones(3, np.float32),
        "priority_sf_search_gap": abs_gap,
        "has_priority_sf_search_gap": np.ones(3, bool),
        "sf_wdl": sf_wdl, "has_sf_wdl": np.ones(3, bool),
        "search_wdl": search_wdl, "has_search_wdl": np.ones(3, bool),
        "wdl_target": np.array([1, 1, 1], np.int8), "has_policy": np.ones(3, bool),
    }

    abs_pri = _signed_gap_buffer(False)._shape_shuffle_priority(arrs, np.ones(3, np.float32))
    signed_pri = _signed_gap_buffer(True)._shape_shuffle_priority(arrs, np.ones(3, np.float32))

    # abs() boosts BOTH the over-optimistic and pessimistic rows (equal |gap|).
    assert abs_pri[0] > 1.0
    assert abs_pri[1] > 1.0
    # signed boosts ONLY over-optimism; pessimistic + agree rows stay at 1.0.
    assert signed_pri[0] > 1.0
    assert signed_pri[1] == pytest.approx(1.0)
    assert signed_pri[2] == pytest.approx(1.0)
    expected = 1.0 + 30.0 * np.maximum(
        (search_wdl[:, 0] - search_wdl[:, 2]) - (sf_wdl[:, 0] - sf_wdl[:, 2]), 0.0
    )
    np.testing.assert_allclose(signed_pri, expected, rtol=1e-5)


def test_signed_gap_priority_requires_wdl_arrays():
    # Signed mode must fail loud, not silently fall back to the abs() gap, when
    # the WDL arrays needed to recompute the signed gap are absent.
    arrs = {
        "priority": np.ones(2, np.float32),
        "priority_sf_search_gap": np.array([1.0, 1.0], np.float32),
        "has_priority_sf_search_gap": np.ones(2, bool),
        "wdl_target": np.array([1, 1], np.int8), "has_policy": np.ones(2, bool),
    }
    with pytest.raises(RuntimeError, match="sf_wdl/search_wdl"):
        _signed_gap_buffer(True)._shape_shuffle_priority(arrs, np.ones(2, np.float32))
