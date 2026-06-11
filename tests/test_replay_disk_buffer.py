from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer, _concat_sparse_batches
from chess_anti_engine.replay.shard import (
    INPUT_HISTORY_ENCODING_ARRAY_KEY,
    POLICY_ENCODING_ARRAY_KEY,
    delete_shard_path,
    iter_shard_paths,
    load_shard_arrays,
)


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


def _arrays(policy_size: int, n: int = 1) -> dict[str, np.ndarray]:
    policy = np.zeros((n, policy_size), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "x": np.zeros((n, 146, 8, 8), dtype=np.float32),
        "policy_target": policy,
        "wdl_target": np.zeros((n,), dtype=np.int8),
        "priority": np.ones((n,), dtype=np.float32),
        "has_policy": np.ones((n,), dtype=np.uint8),
    }


def test_take_write_prefix_preserves_scalar_chunk_fields(tmp_path) -> None:
    rng = np.random.default_rng(0)
    buf = DiskReplayBuffer(
        10,
        shard_dir=tmp_path / "replay",
        rng=rng,
        shuffle_cap=10,
        shard_size=10,
    )
    chunk = _arrays(4672, n=5)
    chunk["_policy_size"] = np.array(4672, dtype=np.int32)
    buf._write_buf = [chunk]
    buf._write_buf_sizes = [5]
    buf._write_buf_rows = 5

    taken = buf._take_write_prefix(2)

    assert int(np.asarray(taken["_policy_size"]).item()) == 4672
    assert int(np.asarray(buf._write_buf[0]["_policy_size"]).item()) == 4672
    assert buf._write_buf[0]["x"].shape[0] == 3


def test_concat_sparse_batches_rejects_missing_and_concrete_history_metadata() -> None:
    legacy = _arrays(4672, n=1)
    root = _arrays(4672, n=1)
    root[INPUT_HISTORY_ENCODING_ARRAY_KEY] = np.asarray("lc0_root")

    with pytest.raises(ValueError, match="mixed replay metadata"):
        _concat_sparse_batches([legacy, root])


def test_concat_sparse_batches_keeps_matching_history_metadata() -> None:
    first = _arrays(4672, n=1)
    second = _arrays(4672, n=1)
    first[INPUT_HISTORY_ENCODING_ARRAY_KEY] = np.asarray("lc0_root")
    second[INPUT_HISTORY_ENCODING_ARRAY_KEY] = np.asarray("lc0_root")

    out = _concat_sparse_batches([first, second])

    assert out["x"].shape[0] == 2
    assert str(out[INPUT_HISTORY_ENCODING_ARRAY_KEY].item()) == "lc0_root"


def test_concat_sparse_batches_rejects_missing_and_concrete_policy_metadata() -> None:
    legacy = _arrays(4672, n=1)
    tagged = _arrays(4672, n=1)
    tagged[POLICY_ENCODING_ARRAY_KEY] = np.asarray("az_4672")

    with pytest.raises(ValueError, match="mixed replay metadata"):
        _concat_sparse_batches([legacy, tagged])


def test_concat_sparse_batches_keeps_matching_policy_metadata() -> None:
    first = _arrays(4672, n=1)
    second = _arrays(4672, n=1)
    first[POLICY_ENCODING_ARRAY_KEY] = np.asarray("az_4672")
    second[POLICY_ENCODING_ARRAY_KEY] = np.asarray("az_4672")

    out = _concat_sparse_batches([first, second])

    assert out["x"].shape[0] == 2
    assert str(out[POLICY_ENCODING_ARRAY_KEY].item()) == "az_4672"


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


def test_sample_batch_arrays_accepts_compact_policy(tmp_path) -> None:
    rng = np.random.default_rng(0)
    buf = DiskReplayBuffer(
        50,
        shard_dir=tmp_path / "replay",
        rng=rng,
        shuffle_cap=20,
        shard_size=10,
    )

    buf.add_many_arrays(_arrays(COMPACT_POLICY_SIZE, n=4))
    arrs = buf.sample_batch_arrays(2, wdl_balance=False)

    assert arrs["policy_target"].shape == (2, COMPACT_POLICY_SIZE)


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


def test_resumed_shuffle_cache_survives_deleted_shard_directories(tmp_path) -> None:
    rng = np.random.default_rng(0)
    shard_dir = tmp_path / "replay"
    buf = DiskReplayBuffer(
        12,
        shard_dir=shard_dir,
        rng=rng,
        shuffle_cap=12,
        shard_size=2,
    )

    buf.add_many([_sample() for _ in range(6)])
    buf.flush()

    resumed = DiskReplayBuffer(
        12,
        shard_dir=shard_dir,
        rng=np.random.default_rng(1),
        shuffle_cap=12,
        shard_size=2,
        refresh_shards=3,
    )
    assert resumed._shuffle_len() > 0

    for sp in iter_shard_paths(shard_dir):
        if sp.is_dir():
            for child in sp.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(sp.rglob("*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
            sp.rmdir()
        else:
            sp.unlink()

    arrs = resumed.sample_batch_arrays(2, wdl_balance=False)

    assert arrs["x"].shape == (2, 146, 8, 8)
    assert arrs["policy_target"].shape == (2, 4672)


def test_resume_enforces_capacity_before_seeding_shuffle(tmp_path) -> None:
    rng = np.random.default_rng(0)
    shard_dir = tmp_path / "replay"
    buf = DiskReplayBuffer(
        6,
        shard_dir=shard_dir,
        rng=rng,
        shuffle_cap=6,
        shard_size=2,
        refresh_shards=1,
    )

    buf.add_many([_sample() for _ in range(12)])
    buf.flush()
    buf.close()

    resumed = DiskReplayBuffer(
        6,
        shard_dir=shard_dir,
        rng=np.random.default_rng(1),
        shuffle_cap=6,
        shard_size=2,
        refresh_shards=1,
    )

    assert resumed._tracked_shard_positions() <= 6
    assert len(iter_shard_paths(shard_dir)) == 3
    assert resumed._shuffle_len() <= 6
    resumed.close()


def test_delete_shard_path_unlinks_symlinked_directory(tmp_path) -> None:
    src = tmp_path / "src_shard.zarr"
    src.mkdir()
    (src / "x").write_text("data", encoding="utf-8")
    dst = tmp_path / "linked_shard.zarr"
    dst.symlink_to(src, target_is_directory=True)

    delete_shard_path(dst)

    assert not dst.exists()
    assert src.exists()


def test_refresh_interval_controls_shuffle_refresh(monkeypatch, tmp_path) -> None:
    rng = np.random.default_rng(0)
    buf = DiskReplayBuffer(
        12,
        shard_dir=tmp_path / "replay",
        rng=rng,
        shuffle_cap=12,
        shard_size=2,
        refresh_interval=1,
        refresh_shards=1,
    )
    buf.add_many([_sample() for _ in range(6)])
    buf.flush()

    calls = {"count": 0}

    def _fake_schedule() -> None:
        calls["count"] += 1

    monkeypatch.setattr(buf, "_schedule_refresh_prefetch", _fake_schedule)
    arrs = buf.sample_batch_arrays(2, wdl_balance=False)

    assert arrs["x"].shape == (2, 146, 8, 8)
    assert calls["count"] == 1


def test_prefetched_refresh_is_consumed_before_sync_refresh(tmp_path) -> None:
    rng = np.random.default_rng(0)
    shard_dir = tmp_path / "replay"
    buf = DiskReplayBuffer(
        12,
        shard_dir=shard_dir,
        rng=rng,
        shuffle_cap=12,
        shard_size=2,
        refresh_interval=1,
        refresh_shards=1,
    )
    buf.add_many([_sample() for _ in range(6)])
    buf.flush()

    # The background prefetch loop autonomously refills _prefetched_refresh
    # whenever it is empty (within ~0.1s), so the final is-None assertion
    # would race it under cross-test CPU contention. This test is about the
    # CONSUME path only: stop the thread and keep it stopped so the injected
    # chunk and the post-sample state are fully deterministic.
    buf.close()
    buf._ensure_prefetch_thread = lambda: None  # type: ignore[method-assign]

    first_shard = iter_shard_paths(shard_dir)[0]
    arrs, _ = load_shard_arrays(first_shard, lazy=False)
    buf._prefetched_refresh = [arrs]

    def _fail_refresh() -> None:
        raise AssertionError("sync refresh should not be used when a prefetched chunk is ready")

    buf._refresh_shuffle_buf = _fail_refresh  # type: ignore[method-assign]
    sampled = buf.sample_batch_arrays(2, wdl_balance=False)

    assert sampled["x"].shape == (2, 146, 8, 8)
    assert buf._prefetched_refresh is None


def test_background_prefetch_populates_ready_chunk(tmp_path) -> None:
    rng = np.random.default_rng(0)
    shard_dir = tmp_path / "replay"
    buf = DiskReplayBuffer(
        12,
        shard_dir=shard_dir,
        rng=rng,
        shuffle_cap=12,
        shard_size=2,
        refresh_interval=2,
        refresh_shards=1,
    )
    buf.add_many([_sample() for _ in range(6)])
    buf.flush()
    buf._schedule_refresh_prefetch()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if buf._prefetched_refresh is not None:
            break
        time.sleep(0.01)

    assert buf._prefetched_refresh is not None
    buf.close()


def test_close_allows_prefetch_thread_restart(tmp_path) -> None:
    rng = np.random.default_rng(0)
    shard_dir = tmp_path / "replay"
    buf = DiskReplayBuffer(
        12,
        shard_dir=shard_dir,
        rng=rng,
        shuffle_cap=12,
        shard_size=2,
        refresh_interval=2,
        refresh_shards=1,
    )
    buf.add_many([_sample() for _ in range(6)])
    buf.flush()

    buf.close()
    buf._schedule_refresh_prefetch()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if buf._prefetched_refresh is not None:
            break
        time.sleep(0.01)

    assert buf._prefetched_refresh is not None
    buf.close()


def test_close_discards_late_prefetch_results(tmp_path) -> None:
    rng = np.random.default_rng(0)
    shard_dir = tmp_path / "replay"
    buf = DiskReplayBuffer(
        12,
        shard_dir=shard_dir,
        rng=rng,
        shuffle_cap=12,
        shard_size=2,
        refresh_interval=2,
        refresh_shards=1,
    )
    buf.add_many([_sample() for _ in range(6)])
    buf.flush()

    first_shard = iter_shard_paths(shard_dir)[0]
    arrs, _ = load_shard_arrays(first_shard, lazy=False)
    started = threading.Event()
    release = threading.Event()

    def _slow_load_refresh_chunks(*, shard_paths, refresh_shards, rng):  # pylint: disable=unused-argument  # mock matches real signature
        del shard_paths, refresh_shards, rng
        started.set()
        release.wait(timeout=2.0)
        return [arrs]

    buf._load_refresh_chunks = _slow_load_refresh_chunks  # type: ignore[method-assign]
    buf._schedule_refresh_prefetch()
    assert started.wait(timeout=1.0)

    buf.close()
    release.set()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if buf._prefetched_refresh is not None:
            break
        time.sleep(0.01)

    assert buf._prefetched_refresh is None
    buf.close()
