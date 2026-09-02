"""Regression tests for ``_prune_processed_shards`` aging processed shards.

A processed directory can contain shards of different ages. Adding a new shard
refreshes the parent directory mtime without touching older children, so the
parent mtime must never be used as a shortcut for child retention.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from chess_anti_engine.tune.distributed_runtime import _prune_processed_shards


def _make_shard(parent: Path, name: str, mtime: float) -> Path:
    """Create a minimal zarr-dir shard and stamp it (and its dir) at *mtime*."""
    shard = parent / name
    shard.mkdir(parents=True)
    (shard / ".zarray").write_text("{}")
    for f in shard.rglob("*"):
        os.utime(f, (mtime, mtime))
    os.utime(shard, (mtime, mtime))
    return shard


def test_compacted_old_shards_aged_out_despite_fresh_dir_mtime(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    comp = proc / "_compacted"
    comp.mkdir(parents=True)

    now = time.time()
    old = now - 3 * 86400.0
    fresh = now - 60.0
    old_shard = _make_shard(comp, "shard_old.zarr", old)
    fresh_shard = _make_shard(comp, "shard_fresh.zarr", fresh)

    os.utime(comp, None)
    assert comp.stat().st_mtime >= now - 5

    deleted = _prune_processed_shards(processed_dir=proc, max_age_seconds=86400.0)

    assert deleted == 1
    assert not old_shard.exists()
    assert fresh_shard.exists()


def test_active_user_dir_prunes_old_child_despite_fresh_parent_mtime(tmp_path: Path) -> None:
    """A newly-arrived shard must not hide older shards from retention."""
    proc = tmp_path / "processed"
    user = proc / "alice"
    user.mkdir(parents=True)

    now = time.time()
    old = now - 3 * 86400.0
    fresh = now - 60.0
    old_shard = _make_shard(user, "shard_old.zarr", old)
    fresh_shard = _make_shard(user, "shard_fresh.zarr", fresh)

    os.utime(user, None)
    assert user.stat().st_mtime >= now - 5

    deleted = _prune_processed_shards(processed_dir=proc, max_age_seconds=86400.0)

    assert deleted == 1
    assert not old_shard.exists()
    assert fresh_shard.exists()


def test_per_user_dir_stale_shards_deleted_when_dir_is_stale(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    user = proc / "alice"
    user.mkdir(parents=True)

    old = time.time() - 3 * 86400.0
    old_shard = _make_shard(user, "shard_old.zarr", old)
    os.utime(user, (old, old))

    deleted = _prune_processed_shards(processed_dir=proc, max_age_seconds=86400.0)

    assert deleted == 1
    assert not old_shard.exists()
