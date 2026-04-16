from __future__ import annotations

from pathlib import Path

import pytest

from chess_anti_engine.utils.atomic import (
    atomic_copy2,
    atomic_write,
    atomic_write_bytes,
    atomic_write_text,
)


def test_atomic_write_bytes_creates_file_and_no_tmp_remains(tmp_path: Path) -> None:
    dst = tmp_path / "sub" / "data.bin"
    atomic_write_bytes(dst, b"hello")
    assert dst.read_bytes() == b"hello"
    assert list(dst.parent.iterdir()) == [dst]


def test_atomic_write_text_creates_file(tmp_path: Path) -> None:
    dst = tmp_path / "note.txt"
    atomic_write_text(dst, "line\n")
    assert dst.read_text() == "line\n"


def test_atomic_write_default_tmp_is_invisible_to_source_suffix_glob(tmp_path: Path) -> None:
    # Regression: suffix-preserving tmp names leak into *.json / *.npz
    # globs used by prune_expired_leases and _upload_pending_shards.
    dst = tmp_path / "lease.json"

    def writer(tmp: Path) -> None:
        # Concurrent scanner simulation — during the write, no .json file
        # (neither the final nor the tmp) must match *.json yet.
        assert list(tmp.parent.glob("*.json")) == []
        tmp.write_text("{}")

    atomic_write(dst, writer)
    assert dst.read_text() == "{}"


def test_atomic_write_preserve_suffix_keeps_source_extension(tmp_path: Path) -> None:
    dst = tmp_path / "shard.npz"
    seen: dict[str, str] = {}

    def writer(tmp: Path) -> None:
        seen["suffix"] = tmp.suffix
        tmp.write_bytes(b"x")

    atomic_write(dst, writer, preserve_suffix=True)
    assert seen["suffix"] == ".npz"


def test_atomic_write_cleans_up_tmp_on_writer_failure(tmp_path: Path) -> None:
    dst = tmp_path / "fail.bin"

    def writer(tmp: Path) -> None:
        tmp.write_bytes(b"partial")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        atomic_write(dst, writer)
    assert not dst.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_copy2_preserves_metadata(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    dst = tmp_path / "dst.bin"
    src.write_bytes(b"payload")
    src_mtime = src.stat().st_mtime
    atomic_copy2(src, dst)
    assert dst.read_bytes() == b"payload"
    assert dst.stat().st_mtime == src_mtime
