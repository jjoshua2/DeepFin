"""Regression tests for _iter_shard_paths_nested: tmp-dir skipping and
FileNotFoundError tolerance under concurrent rename."""
from __future__ import annotations

from pathlib import Path

import pytest

from chess_anti_engine.tune.distributed_runtime import _iter_shard_paths_nested


def _touch_shard_dir(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "chunk_0").touch()


def test_finds_normal_shards(tmp_path: Path) -> None:
    _touch_shard_dir(tmp_path / "user1" / "shard_000001.zarr")
    _touch_shard_dir(tmp_path / "user2" / "shard_000002.zarr")
    (tmp_path / "user2" / "legacy_000001.npz").touch()

    paths = _iter_shard_paths_nested(tmp_path)

    names = sorted(p.name for p in paths)
    assert names == ["legacy_000001.npz", "shard_000001.zarr", "shard_000002.zarr"]


def test_skips_tmp_user_dir(tmp_path: Path) -> None:
    """tmp_* at the user level is a mid-upload staging dir; never descend."""
    _touch_shard_dir(tmp_path / "user1" / "shard_000001.zarr")
    _touch_shard_dir(tmp_path / "tmp_21266_aaa.zarr" / "chunk_junk")
    _touch_shard_dir(tmp_path / "._tmp_compactor" / "shard_junk.zarr")

    paths = _iter_shard_paths_nested(tmp_path)

    assert [p.name for p in paths] == ["shard_000001.zarr"]


def test_skips_server_pending_and_in_flight_staging_dirs(tmp_path: Path) -> None:
    """Server staging dirs are not worker/user dirs and must not be ingested."""
    _touch_shard_dir(tmp_path / "worker_00" / "shard_000001.zarr")
    _touch_shard_dir(tmp_path / "_compacted" / "server_compacted.zarr")
    _touch_shard_dir(tmp_path / "_pending" / "pending_upload.zarr")
    _touch_shard_dir(tmp_path / "_in_flight" / "flush_token" / "staged_upload.zarr")

    paths = _iter_shard_paths_nested(tmp_path)

    assert sorted(p.name for p in paths) == ["server_compacted.zarr", "shard_000001.zarr"]


def test_skips_tmp_shard_inside_user_dir(tmp_path: Path) -> None:
    """tmp_* at the shard level is a mid-upload file; skip but keep siblings."""
    _touch_shard_dir(tmp_path / "user1" / "shard_000001.zarr")
    _touch_shard_dir(tmp_path / "user1" / "tmp_99_xyz.zarr")
    (tmp_path / "user1" / "._tmp_upload.npz").touch()

    paths = _iter_shard_paths_nested(tmp_path)

    assert [p.name for p in paths] == ["shard_000001.zarr"]


def test_missing_root_returns_empty(tmp_path: Path) -> None:
    """Ingest runs before any uploads land; missing root is not an error."""
    assert _iter_shard_paths_nested(tmp_path / "does-not-exist") == []


def test_empty_root_returns_empty(tmp_path: Path) -> None:
    assert _iter_shard_paths_nested(tmp_path) == []


def test_user_dir_vanishing_midscan_is_tolerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate: user dir was listed, then deleted before inner iterdir runs.

    This reproduces the production race (tmp upload dir renamed mid-scan)
    without needing a filesystem racer: patch Path.iterdir on the vanishing
    dir to raise FileNotFoundError. The helper must return the surviving
    shard, not crash.
    """
    user_a = tmp_path / "user_a"
    user_b = tmp_path / "user_b_racing"
    _touch_shard_dir(user_a / "shard_000001.zarr")
    user_b.mkdir()

    original_iterdir = Path.iterdir

    def racing_iterdir(self: Path):
        if self == user_b:
            raise FileNotFoundError(f"simulated rename race: {self}")
        yield from original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", racing_iterdir)

    paths = _iter_shard_paths_nested(tmp_path)

    assert [p.name for p in paths] == ["shard_000001.zarr"]


def test_non_shard_files_ignored(tmp_path: Path) -> None:
    (tmp_path / "user1").mkdir()
    (tmp_path / "user1" / "README").touch()
    (tmp_path / "user1" / "notes.txt").touch()
    _touch_shard_dir(tmp_path / "user1" / "shard_000001.zarr")

    paths = _iter_shard_paths_nested(tmp_path)

    assert [p.name for p in paths] == ["shard_000001.zarr"]
