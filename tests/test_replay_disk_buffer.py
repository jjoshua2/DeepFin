from __future__ import annotations

import fcntl
import logging
import os
import threading
import time
from pathlib import Path

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
    local_shard_path,
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
        rng=rng, read_only=False,
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
        rng=rng, read_only=False,
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
        rng=rng, read_only=False,
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
        rng=rng, read_only=False,
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
        rng=rng, read_only=False,
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
        rng=rng, read_only=False,
        shuffle_cap=20,
        shard_size=4,
    )

    buf.add_many([_sample() for _ in range(6)])
    buf.flush()

    resumed = DiskReplayBuffer(
        50,
        shard_dir=shard_dir,
        rng=np.random.default_rng(1), read_only=False,
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
        rng=rng, read_only=False,
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
        rng=rng, read_only=False,
        shuffle_cap=12,
        shard_size=2,
    )

    buf.add_many([_sample() for _ in range(6)])
    buf.flush()

    resumed = DiskReplayBuffer(
        12,
        shard_dir=shard_dir,
        rng=np.random.default_rng(1), read_only=False,
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
        rng=rng, read_only=False,
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
        rng=np.random.default_rng(1), read_only=False,
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
        rng=rng, read_only=False,
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
        rng=rng, read_only=False,
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
    buf._ensure_prefetch_thread = lambda: None

    first_shard = iter_shard_paths(shard_dir)[0]
    arrs, _ = load_shard_arrays(first_shard, lazy=False)
    buf._prefetched_refresh = [arrs]

    def _fail_refresh() -> None:
        raise AssertionError("sync refresh should not be used when a prefetched chunk is ready")

    buf._refresh_shuffle_buf = _fail_refresh
    sampled = buf.sample_batch_arrays(2, wdl_balance=False)

    assert sampled["x"].shape == (2, 146, 8, 8)
    assert buf._prefetched_refresh is None


def test_background_prefetch_populates_ready_chunk(tmp_path) -> None:
    rng = np.random.default_rng(0)
    shard_dir = tmp_path / "replay"
    buf = DiskReplayBuffer(
        12,
        shard_dir=shard_dir,
        rng=rng, read_only=False,
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
        rng=rng, read_only=False,
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
        rng=rng, read_only=False,
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

    # Names only the three arguments this test relies on, typed as the real
    # method types them, so a RENAME or a type change OF ONE OF THOSE THREE on the
    # real `_load_refresh_chunks` still fails the type gate here. `**_added_later`
    # absorbs every other keyword argument, so an additive signature change cannot
    # turn a test that never reads them red.
    #
    # Scope that guarantee honestly: it covers ONLY the three named arguments.
    # `context` and `chosen_out` exist today (they arrived with the open-time pool
    # seed) and land in `**_added_later`, so renaming or retyping either one is
    # invisible here.
    #
    # Note `chosen_out` is an OUT-parameter: `_load_refresh_chunks` appends the
    # drawn shard ranks to it (disk_buffer.py:955-958) and `_seed_shuffle_pool`
    # reads them back for `mean_recency`. `**_added_later` would accept it and
    # silently drop the writes. Harmless today because the only call this test
    # reaches is `_prefetch_loop` (disk_buffer.py:895-899), which passes exactly
    # the three named kwargs — but one refactor away from this repo's signature
    # defect, a value accepted and then silently ignored. If a call this test
    # reaches ever starts passing `chosen_out`, name it here.
    def _slow_load_refresh_chunks(
        *,
        shard_paths: list[Path],
        refresh_shards: int,
        rng: np.random.Generator,
        **_added_later: object,
    ) -> list[dict[str, np.ndarray]]:
        del shard_paths, refresh_shards, rng, _added_later
        started.set()
        release.wait(timeout=2.0)
        return [arrs]

    buf._load_refresh_chunks = _slow_load_refresh_chunks
    # The prefetch loop refills whenever the slot is empty, including on plain
    # 0.1s timeout ticks — so during the add_many/flush setup above the REAL
    # loader may already have filled the slot, in which case a single
    # _schedule_refresh_prefetch() is a no-op and the stub never runs (the
    # pre-fix 1-in-4 flake on a loaded box). Drain the slot and re-arm until
    # the stub is the load that actually happens.
    deadline = time.monotonic() + 15.0
    while not started.is_set() and time.monotonic() < deadline:
        buf._take_prefetched_refresh()
        buf._schedule_refresh_prefetch()
        started.wait(timeout=0.1)
    assert started.is_set()

    buf.close()
    release.set()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if buf._prefetched_refresh is not None:
            break
        time.sleep(0.01)

    assert buf._prefetched_refresh is None
    buf.close()


def _shaping_arrays(n_full: int, n_fast: int) -> dict[str, np.ndarray]:
    """Rows: n_full full plies with sf-gap fields, then n_fast value-only rows.

    Full rows carry priority 5.0 and sf gaps [0.0, 0.5, 1.0, ...]; fast rows
    carry the neutral priority 1.0 and alternate low/high outcome surprise
    (search agreed with z / search sign-flipped vs z).
    """
    n = n_full + n_fast
    arrs = _arrays(4672, n=n)
    arrs["has_policy"][:] = 0
    arrs["has_policy"][:n_full] = 1
    arrs["priority"][:n_full] = 5.0

    gap = np.zeros((n,), dtype=np.float16)
    has_gap = np.zeros((n,), dtype=np.uint8)
    for i in range(n_full):
        gap[i] = 0.5 * i
        has_gap[i] = 1
    arrs["priority_sf_search_gap"] = gap
    arrs["has_priority_sf_search_gap"] = has_gap

    search_wdl = np.zeros((n, 3), dtype=np.float16)
    has_search = np.zeros((n,), dtype=np.uint8)
    arrs["wdl_target"][:] = 0                      # every game a win (z=+1)
    for j in range(n_fast):
        i = n_full + j
        has_search[i] = 1
        if j % 2 == 0:
            search_wdl[i] = (0.9, 0.1, 0.0)        # agrees with z: |1-0.9|=0.1
        else:
            search_wdl[i] = (0.0, 0.1, 0.9)        # sign-flip: |1-(-0.9)|=1.9
    arrs["search_wdl"] = search_wdl
    arrs["has_search_wdl"] = has_search
    return arrs


def test_shape_shuffle_priority_defaults_are_noop(tmp_path) -> None:
    buf = DiskReplayBuffer(
        100, shard_dir=tmp_path / "replay", rng=np.random.default_rng(0), read_only=False,
        shuffle_cap=100, shard_size=100,
    )
    buf.add_many_arrays(_shaping_arrays(n_full=3, n_fast=4))
    pri = buf._active_shuffle_priority()
    np.testing.assert_allclose(pri[:3], 5.0)
    np.testing.assert_allclose(pri[3:], 1.0)


def test_shape_shuffle_priority_boosts_full_rows_by_sf_gap(tmp_path) -> None:
    buf = DiskReplayBuffer(
        100, shard_dir=tmp_path / "replay", rng=np.random.default_rng(0), read_only=False,
        shuffle_cap=100, shard_size=100,
    )
    buf.sf_gap_priority_weight = 10.0
    buf.add_many_arrays(_shaping_arrays(n_full=3, n_fast=2))
    pri = buf._active_shuffle_priority()
    # gaps 0.0 / 0.5 / 1.0 on stored priority 5.0
    np.testing.assert_allclose(pri[:3], [5.0, 10.0, 15.0])
    # fast rows untouched (demotion off)
    np.testing.assert_allclose(pri[3:], 1.0)


def test_shape_shuffle_priority_demotes_low_surprise_fast_rows(tmp_path) -> None:
    buf = DiskReplayBuffer(
        100, shard_dir=tmp_path / "replay", rng=np.random.default_rng(0), read_only=False,
        shuffle_cap=100, shard_size=100,
    )
    buf.fast_low_surprise_priority = 0.2
    buf.add_many_arrays(_shaping_arrays(n_full=2, n_fast=4))
    pri = buf._active_shuffle_priority()
    # full rows untouched (gap boost off)
    np.testing.assert_allclose(pri[:2], 5.0)
    # fast rows alternate agree/sign-flip: demoted / kept
    np.testing.assert_allclose(pri[2:], [0.2, 1.0, 0.2, 1.0])


def test_shape_shuffle_priority_skips_chunks_missing_fields(tmp_path) -> None:
    buf = DiskReplayBuffer(
        100, shard_dir=tmp_path / "replay", rng=np.random.default_rng(0), read_only=False,
        shuffle_cap=100, shard_size=100,
    )
    buf.sf_gap_priority_weight = 10.0
    buf.fast_low_surprise_priority = 0.2
    buf.add_many_arrays(_arrays(4672, n=3))       # no gap / search_wdl columns
    pri = buf._active_shuffle_priority()
    np.testing.assert_allclose(pri, 1.0)


def test_constructor_shaping_applies_to_resume_seeded_shuffle(tmp_path) -> None:
    """Codex P2 on PR #104: a resumed buffer seeds its hot pool from existing
    shards inside __init__, before any live-knob push — the constructor args
    must already shape those seeded priorities."""
    d = tmp_path / "replay"
    buf1 = DiskReplayBuffer(
        100, shard_dir=d, rng=np.random.default_rng(0), read_only=False,
        shuffle_cap=100, shard_size=100,
    )
    buf1.add_many_arrays(_shaping_arrays(n_full=3, n_fast=4))
    buf1.flush()
    buf1.close()

    buf2 = DiskReplayBuffer(
        100, shard_dir=d, rng=np.random.default_rng(1), read_only=False,
        shuffle_cap=100, shard_size=100,
        sf_gap_priority_weight=10.0,
        fast_low_surprise_priority=0.2,
    )
    pri = np.sort(buf2._active_shuffle_priority())
    # full rows 5.0 + 10*gap(0/0.5/1.0) -> 5/10/15; fast rows agree/flip
    # alternating -> 0.2/1.0/0.2/1.0.
    np.testing.assert_allclose(pri, [0.2, 0.2, 1.0, 1.0, 5.0, 10.0, 15.0])
    buf2.close()


def test_priority_mass_stats_decompose_and_reset(tmp_path) -> None:
    buf = DiskReplayBuffer(
        100, shard_dir=tmp_path / "replay", rng=np.random.default_rng(0), read_only=False,
        shuffle_cap=100, shard_size=100,
        sf_gap_priority_weight=10.0, fast_low_surprise_priority=0.2,
        diff_focus_pol_scale=2.0, diff_focus_q_weight=4.0,
    )
    arrs = _shaping_arrays(n_full=3, n_fast=4)
    kl = np.zeros((7,), dtype=np.float16)
    kl[:3] = 1.0
    qd = np.zeros((7,), dtype=np.float16)
    qd[:3] = 0.5
    has_full = np.zeros((7,), dtype=np.uint8)
    has_full[:3] = 1
    arrs["priority_policy_kl"] = kl
    arrs["has_priority_policy_kl"] = has_full
    arrs["priority_q_delta"] = qd
    arrs["has_priority_q_delta"] = has_full
    buf.add_many_arrays(arrs)

    st = buf.pop_priority_mass_stats()
    # effective priorities: full [5+0, 5+5, 5+10] = 30; fast [0.2,1,0.2,1] = 2.4
    total = 30.0 + 2.4
    assert st["replay_pmass_rows_full"] == 3
    assert st["replay_pmass_rows_fast"] == 4
    np.testing.assert_allclose(st["replay_pmass_kl_share"], 3 * 1.0 * 2.0 / total, rtol=1e-3)
    np.testing.assert_allclose(st["replay_pmass_qd_share"], 3 * 0.5 * 4.0 / total, rtol=1e-3)
    np.testing.assert_allclose(st["replay_pmass_gap_share"], (0.0 + 5.0 + 10.0) / total, rtol=1e-3)
    np.testing.assert_allclose(st["replay_pmass_fast_share"], 2.4 / total, rtol=1e-3)
    np.testing.assert_allclose(st["replay_fast_demoted_frac"], 0.5)

    # pop resets
    st2 = buf.pop_priority_mass_stats()
    assert st2["replay_pmass_rows_full"] == 0
    assert st2["replay_pmass_kl_share"] == 0.0


def test_priority_mass_stats_tolerate_missing_fields(tmp_path) -> None:
    buf = DiskReplayBuffer(
        100, shard_dir=tmp_path / "replay", rng=np.random.default_rng(0), read_only=False,
        shuffle_cap=100, shard_size=100,
        diff_focus_pol_scale=2.0, diff_focus_q_weight=4.0,
    )
    buf.add_many_arrays(_arrays(4672, n=3))    # bare chunk, no optional fields
    st = buf.pop_priority_mass_stats()
    assert st["replay_pmass_rows_full"] == 3
    assert st["replay_pmass_kl_share"] == 0.0
    assert st["replay_pmass_gap_share"] == 0.0


def test_priority_mass_excludes_fast_row_kl_and_seed_scan(tmp_path) -> None:
    d = tmp_path / "replay"
    buf = DiskReplayBuffer(
        100, shard_dir=d, rng=np.random.default_rng(0), read_only=False,
        shuffle_cap=100, shard_size=7,
        diff_focus_pol_scale=2.0, diff_focus_q_weight=4.0,
    )
    arrs = _shaping_arrays(n_full=3, n_fast=4)
    # kl present on ALL rows (finalize stores it on fast rows too) — only the
    # 3 full rows may contribute term mass.
    arrs["priority_policy_kl"] = np.ones((7,), dtype=np.float16)
    arrs["has_priority_policy_kl"] = np.ones((7,), dtype=np.uint8)
    buf.add_many_arrays(arrs)
    st = buf.pop_priority_mass_stats()
    total = 3 * 5.0 + 4 * 1.0
    np.testing.assert_allclose(st["replay_pmass_kl_share"], 3 * 2.0 / total, rtol=1e-3)
    buf.flush()
    buf.close()

    # Resume over the same shards: the constructor seed scan must NOT leak
    # into the first pop.
    buf2 = DiskReplayBuffer(
        100, shard_dir=d, rng=np.random.default_rng(1), read_only=False,
        shuffle_cap=100, shard_size=7,
        diff_focus_pol_scale=2.0, diff_focus_q_weight=4.0,
    )
    st2 = buf2.pop_priority_mass_stats()
    assert st2["replay_pmass_rows_full"] == 0
    assert st2["replay_pmass_kl_share"] == 0.0
    buf2.close()


def test_second_writer_into_one_shard_dir_is_reported(tmp_path, caplog) -> None:
    """Audit G7: two writers into one window dir duplicate every row silently.

    On 2026-07-11 that ran for 87 minutes and left 20.5k byte-identical
    duplicate rows in the window (and in a salvage revert point). Nothing
    detected it. The lock is advisory: the second writer still writes, so the
    only thing this asserts is that it SAYS so.
    """
    shard_dir = tmp_path / "replay"
    shard_dir.mkdir()
    fd = os.open(str(shard_dir / ".writer.lock"), os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.pwrite(fd, b"424242\n", 0)
    try:
        buf = DiskReplayBuffer(
            100, shard_dir=shard_dir, rng=np.random.default_rng(0), read_only=False,
            shuffle_cap=100, shard_size=2,
        )
        with caplog.at_level(logging.ERROR, logger="chess_anti_engine.replay.disk_buffer"):
            buf.add_many([_sample() for _ in range(2)])
            buf.flush()
        assert "CONCURRENT WRITER" in caplog.text
        assert "424242" in caplog.text
        # Reported once, not once per shard.
        buf.add_many([_sample() for _ in range(2)])
        buf.flush()
        assert caplog.text.count("CONCURRENT WRITER") == 1
        assert len(iter_shard_paths(shard_dir)) == 2
        buf.close()
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_sole_writer_is_silent_and_releases_the_lock_on_close(tmp_path, caplog) -> None:
    shard_dir = tmp_path / "replay"
    with caplog.at_level(logging.ERROR, logger="chess_anti_engine.replay.disk_buffer"):
        buf = DiskReplayBuffer(
            100, shard_dir=shard_dir, rng=np.random.default_rng(0), read_only=False,
            shuffle_cap=100, shard_size=2,
        )
        buf.add_many([_sample() for _ in range(2)])
        buf.flush()
        buf.close()
        # A resume in the same process must not inherit a stale complaint.
        resumed = DiskReplayBuffer(
            100, shard_dir=shard_dir, rng=np.random.default_rng(1), read_only=False,
            shuffle_cap=100, shard_size=2,
        )
        resumed.add_many([_sample() for _ in range(2)])
        resumed.flush()
        resumed.close()
    assert "CONCURRENT WRITER" not in caplog.text
    assert len(iter_shard_paths(shard_dir)) == 2


def test_shard_write_steps_over_an_index_another_writer_owns(tmp_path, caplog) -> None:
    """The overwrite variant of G7: same index, one shard silently destroyed."""
    shard_dir = tmp_path / "replay"
    buf = DiskReplayBuffer(
        100, shard_dir=shard_dir, rng=np.random.default_rng(0), read_only=False,
        shuffle_cap=100, shard_size=2,
    )
    # Somebody else already owns index 0.
    squatter = local_shard_path(shard_dir, 0)
    squatter.mkdir(parents=True)
    (squatter / "marker").write_text("not ours", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="chess_anti_engine.replay.disk_buffer"):
        buf.add_many([_sample() for _ in range(2)])
        buf.flush()
    assert "shard index collision" in caplog.text
    assert (squatter / "marker").read_text(encoding="utf-8") == "not ours"
    assert local_shard_path(shard_dir, 1).exists()
    assert buf._tracked_shard_positions() == 2
    buf.close()
