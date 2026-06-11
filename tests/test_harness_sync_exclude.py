"""Regression: Ray's experiment-state syncer must skip atomic-rename tmp dirs.

Ray's `_upload_to_fs_path` enumerates the trial dir and copies each file. While
`save_local_shard_arrays` writes to `._tmp_<pid>_*.zarr` then atomic-renames,
the enumerator can capture the tmp path then race the rename, raising
`FileNotFoundError`. `_patch_ray_artifact_sync_excludes` wraps Ray's uploader
to inject exclude patterns matching our tmp prefixes.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.fs
import pytest

from chess_anti_engine.tune.harness import (
    _TMP_SHARD_EXCLUDE_PATTERNS,
    _patch_ray_artifact_sync_excludes,
)


def _stage_shards(tmp_path: Path) -> Path:
    """Fake selfplay_shards dir + a sibling file that should still sync.

    selfplay_shards/ is excluded wholesale (race condition with window
    enforcement deletes), so only the sibling file should land at dst.
    """
    src = tmp_path / "src"
    real = src / "selfplay_shards" / "shard_000001.zarr"
    real.mkdir(parents=True)
    (real / "x").write_bytes(b"real")

    (src / "selfplay_shards" / "._tmp_12345_shard_000002.zarr").mkdir()
    (src / "selfplay_shards" / "._tmp_12345_shard_000002.zarr" / "x").write_bytes(b"in-progress")

    (src / "selfplay_shards" / "tmp_compaction_abc").mkdir()
    (src / "selfplay_shards" / "tmp_compaction_abc" / "x").write_bytes(b"in-progress")

    # Sibling that must still sync (proves we didn't blanket-exclude everything).
    (src / "checkpoint_000001").mkdir()
    (src / "checkpoint_000001" / "trainer.pt").write_bytes(b"trainer-state")
    return src


def test_patch_injects_tmp_excludes_into_upload(tmp_path: Path) -> None:
    src = _stage_shards(tmp_path)
    dst = tmp_path / "dst"
    dst.mkdir()

    _patch_ray_artifact_sync_excludes()
    from ray.train._internal import storage as _ray_storage

    upload = _ray_storage._upload_to_fs_path
    # The patch should have re-bound the module attribute already.
    assert getattr(upload, "_chess_tmp_exclude_patched", False)

    fs = pyarrow.fs.LocalFileSystem()
    upload(str(src), fs, str(dst), exclude=None)

    copied = {p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()}
    # Sibling file is required: this proves we excluded selfplay_shards/ specifically,
    # not the entire trial dir.
    assert "checkpoint_000001/trainer.pt" in copied
    # Whole selfplay_shards/ tree must be skipped — both tmp and final paths.
    assert not any(p.startswith("selfplay_shards/") for p in copied), copied
    assert not any("._tmp_" in p for p in copied), copied


def test_patch_is_idempotent() -> None:
    _patch_ray_artifact_sync_excludes()
    from ray.train._internal import storage as _ray_storage
    first = _ray_storage._upload_to_fs_path
    _patch_ray_artifact_sync_excludes()
    second = _ray_storage._upload_to_fs_path
    assert first is second


def test_patterns_cover_both_tmp_prefixes() -> None:
    # Both prefixes used by `is_tmp_shard_name` must be covered, anchored and
    # nested. If anyone changes the patterns, this catches it.
    import fnmatch
    candidates = [
        ("._tmp_99_shard.zarr", True),
        ("tmp_compaction_xyz", True),
        ("selfplay_shards/._tmp_99_shard.zarr", True),
        ("selfplay_shards/tmp_compaction_xyz", True),
        ("a/b/c/._tmp_99", True),
        # Whole selfplay_shards tree is excluded — completed shards are local-only
        # and Ray's enumeration races against window-enforcement deletes.
        ("selfplay_shards/shard_000001.zarr", True),
        ("selfplay_shards/shard_000005.zarr/.zattrs", True),
        # atomic_write tmp pattern: <final>.tmp.<pid>.<uuid>
        ("best/best_model.pt.tmp.2290.8e141129e0fe", True),
        ("checkpoint_000010/trainer.pt.tmp.1131.abc123", True),
        # Other paths must still be syncable.
        ("shard_000001.zarr", False),
        ("checkpoint_000123/trainer.pt", False),
        ("best/best_model.pt", False),
    ]
    for path, expect_match in candidates:
        matched = any(fnmatch.fnmatch(path, pat) for pat in _TMP_SHARD_EXCLUDE_PATTERNS)
        assert matched is expect_match, (path, expect_match, matched)


@pytest.mark.parametrize(
    "extra",
    [None, [], ["kept/*.junk"]],
    ids=["none", "empty", "preexisting"],
)
def test_patched_upload_preserves_caller_excludes(tmp_path: Path, extra: list | None) -> None:
    src = tmp_path / "src"
    (src / "kept").mkdir(parents=True)
    (src / "kept" / "f").write_bytes(b"k")
    (src / "kept" / "j.junk").write_bytes(b"j")
    (src / "._tmp_1_x").mkdir()
    (src / "._tmp_1_x" / "f").write_bytes(b"t")
    dst = tmp_path / "dst"
    dst.mkdir()

    _patch_ray_artifact_sync_excludes()
    from ray.train._internal import storage as _ray_storage

    fs = pyarrow.fs.LocalFileSystem()
    _ray_storage._upload_to_fs_path(str(src), fs, str(dst), exclude=extra)

    copied = {p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()}
    assert "kept/f" in copied
    # Our patch always filters tmp prefixes regardless of caller-supplied excludes.
    assert not any("._tmp_" in p for p in copied)
    # Caller-supplied excludes must still apply.
    if extra and "kept/*.junk" in extra:
        assert not any(p.endswith(".junk") for p in copied)
