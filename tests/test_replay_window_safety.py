"""Window-eviction safety pins for audit invariants G10 and G12.

Both are latent: today's code is correct, and both are one plausible refactor
away from destroying live data. These tests exist so the refactor fails here
instead of in production.

G10 -- ``delete_shard_path`` unlinks symlinked shards instead of ``rmtree``-ing
through them. Live windows are seeded with relative symlinks into salvage
pools, so an ordinary FIFO eviction routinely targets a link into a designated
revert point. Measured 2026-07-26: 280 links already evicted, and
``swap_512x16_20260711`` still held all 815 shards.

G12 -- ``DiskReplayBuffer.__init__`` scans the shard dir and enforces the
window, i.e. deletes shards, before the caller gets the object back. Opening a
live window "to read it" therefore truncates it. ``read_only=True`` is the safe
opener.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import (
    copy_or_link_shard,
    delete_shard_path,
    iter_shard_paths,
    local_shard_path,
    save_local_shard_arrays,
)

ROWS_PER_SHARD = 4


def _shard_arrays(n: int = ROWS_PER_SHARD) -> dict[str, np.ndarray]:
    policy = np.zeros((n, 4672), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "x": np.zeros((n, 146, 8, 8), dtype=np.float32),
        "policy_target": policy,
        "wdl_target": np.zeros((n,), dtype=np.int8),
        "priority": np.ones((n,), dtype=np.float32),
        "has_policy": np.ones((n,), dtype=np.uint8),
    }


def _make_pool(pool_dir: Path, n_shards: int) -> list[Path]:
    """Write ``n_shards`` real shards into a stand-in salvage pool."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    return [
        save_local_shard_arrays(local_shard_path(pool_dir, i), arrs=_shard_arrays())
        for i in range(n_shards)
    ]


def _seed_window_with_links(window_dir: Path, pool_shards: list[Path]) -> list[Path]:
    """Symlink a whole pool into a window dir, the way salvage seeding does."""
    window_dir.mkdir(parents=True, exist_ok=True)
    links = [
        copy_or_link_shard(src, local_shard_path(window_dir, i))
        for i, src in enumerate(pool_shards)
    ]
    assert all(p.is_symlink() for p in links), (
        "this test only means something if seeding produced symlinks"
    )
    return links


def test_delete_shard_path_unlinks_symlink_without_touching_target(tmp_path: Path) -> None:
    """G10: the symlink branch is load-bearing, not cosmetic."""
    [pool_shard] = _make_pool(tmp_path / "pool", 1)
    pool_contents = sorted(p.name for p in pool_shard.iterdir())
    assert pool_contents, "pool shard must have contents for the test to detect loss"

    link = copy_or_link_shard(pool_shard, tmp_path / "window" / "shard_000000.zarr")
    assert link.is_symlink()

    delete_shard_path(link)

    assert not link.exists()
    assert not link.is_symlink()
    assert pool_shard.is_dir()
    assert sorted(p.name for p in pool_shard.iterdir()) == pool_contents


def test_delete_shard_path_removes_real_shard_directories(tmp_path: Path) -> None:
    """The non-symlink branch must still delete the directory tree.

    Without this the G10 pin above could be satisfied by a helper that simply
    never deletes anything.
    """
    [shard] = _make_pool(tmp_path / "window", 1)
    assert shard.is_dir()
    assert not shard.is_symlink()

    delete_shard_path(shard)

    assert not shard.exists()


def test_delete_shard_path_removes_plain_files_and_tolerates_missing(tmp_path: Path) -> None:
    legacy = tmp_path / "shard_000000.npz"
    legacy.write_bytes(b"not really a shard")

    delete_shard_path(legacy)
    assert not legacy.exists()

    delete_shard_path(legacy)  # second call must not raise


def test_window_eviction_of_symlinks_leaves_the_salvage_pool_intact(tmp_path: Path) -> None:
    """G10 end to end: FIFO eviction through the buffer, not just the helper.

    A plain ``shutil.rmtree`` in ``delete_shard_path`` would follow every link
    and empty the pool here -- destroying a revert point while the run holds it
    open.
    """
    pool_shards = _make_pool(tmp_path / "pool", 4)
    window = tmp_path / "window"
    _seed_window_with_links(window, pool_shards)

    # Capacity fits one shard, so the scan in __init__ evicts the other three.
    buf = DiskReplayBuffer(
        ROWS_PER_SHARD,
        shard_dir=window,
        rng=np.random.default_rng(0), read_only=False,
        shuffle_cap=ROWS_PER_SHARD,
        shard_size=ROWS_PER_SHARD,
    )
    try:
        assert len(iter_shard_paths(window)) == 1
        assert len(iter_shard_paths(tmp_path / "pool")) == 4
        for src in pool_shards:
            assert src.is_dir()
            assert any(src.iterdir())
    finally:
        buf.close()


def test_constructing_a_writable_buffer_evicts_shards(tmp_path: Path) -> None:
    """G12, the hazard itself: ``__init__`` deletes before you can read.

    Pinned deliberately. This is what makes ``read_only=True`` necessary, and a
    change that quietly stopped the constructor from enforcing the window would
    be changing training's window semantics, not just probe ergonomics.
    """
    window = tmp_path / "window"
    _make_pool(window, 5)

    buf = DiskReplayBuffer(
        2 * ROWS_PER_SHARD,
        shard_dir=window,
        rng=np.random.default_rng(0), read_only=False,
        shuffle_cap=ROWS_PER_SHARD,
        shard_size=ROWS_PER_SHARD,
    )
    try:
        assert len(iter_shard_paths(window)) == 2
    finally:
        buf.close()


def test_read_only_buffer_never_deletes_or_writes(tmp_path: Path) -> None:
    """G12, the fix: the same open, with ``read_only=True``, mutates nothing."""
    window = tmp_path / "window"
    before = _make_pool(window, 5)

    buf = DiskReplayBuffer(
        2 * ROWS_PER_SHARD,
        shard_dir=window,
        rng=np.random.default_rng(0),
        shuffle_cap=ROWS_PER_SHARD,
        shard_size=ROWS_PER_SHARD,
        read_only=True,
    )
    try:
        assert iter_shard_paths(window) == before
        assert len(buf) == 5 * ROWS_PER_SHARD

        # Reads keep working -- read-only is not "crippled".
        batch = buf.sample_batch_arrays(2)
        assert batch["x"].shape[0] == 2

        for call in (
            lambda: buf.add_many_arrays(_shard_arrays()),
            buf.flush,
            buf.enforce_window,
            buf.clear,
        ):
            with pytest.raises(RuntimeError, match="read_only=True"):
                call()

        assert iter_shard_paths(window) == before
    finally:
        buf.close()

    assert iter_shard_paths(window) == before


def test_read_only_buffer_does_not_claim_the_writer_lock(tmp_path: Path) -> None:
    """The reject must precede ``_claim_writer``, which writes into the dir.

    ``_flush_shard_arrays`` takes an advisory lock by creating ``.writer.lock``
    inside the shard directory. If the read-only check ran after it, a probe
    would leave a file behind in the live window and, worse, could take the
    lock away from a writer that legitimately wants it.
    """
    window = tmp_path / "window"
    _make_pool(window, 2)

    buf = DiskReplayBuffer(
        10 ** 9,
        shard_dir=window,
        rng=np.random.default_rng(0),
        shuffle_cap=ROWS_PER_SHARD,
        shard_size=ROWS_PER_SHARD,
        read_only=True,
    )
    try:
        with pytest.raises(RuntimeError, match="read_only=True"):
            buf._flush_shard_arrays(_shard_arrays())
        assert buf._writer_lock_fd is None
        assert not (window / ".writer.lock").exists()
    finally:
        buf.close()


def test_read_only_open_does_not_create_a_missing_directory(tmp_path: Path) -> None:
    """A mistyped path must read as an empty window, not appear to exist."""
    missing = tmp_path / "not_a_window"

    buf = DiskReplayBuffer(
        10 ** 9,
        shard_dir=missing,
        rng=np.random.default_rng(0),
        shuffle_cap=ROWS_PER_SHARD,
        shard_size=ROWS_PER_SHARD,
        read_only=True,
    )
    try:
        assert not missing.exists()
        assert len(buf) == 0
    finally:
        buf.close()


def test_read_only_has_no_default_so_a_new_caller_cannot_forget_it(tmp_path: Path) -> None:
    """G12 is closed by the SIGNATURE, not by the docstring.

    ``read_only: bool = False`` would leave the shard-deleting behaviour as
    what you get for forgetting a kwarg -- exactly the hazard G12 records. With
    no default, a new construction site is a type error (basedpyright:
    ``Argument missing for parameter "read_only"``) and a TypeError at runtime,
    so it cannot reach production as a silently-deleting probe. This test pins
    the absence of the default: restoring one makes it fail.
    """
    with pytest.raises(TypeError, match="read_only"):
        DiskReplayBuffer(  # pyright: ignore[reportCallIssue]
            10,
            shard_dir=tmp_path / "window",
            rng=np.random.default_rng(0),
        )
