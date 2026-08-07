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


def test_mode_is_applied_before_the_writer_ever_sees_the_file(tmp_path: Path) -> None:
    """⚑ THE WINDOW IS THE WHOLE WRITE, NOT THE RENAME.

    `mode` used to be a `chmod` *after* `writer(tmp)` returned, so the tmp
    holding the content — for `users.json`, the PBKDF2 material of every
    account — sat at the umask default (0644, or 0666 at `umask 000`) for the
    entire write and fsync. A users.json big enough to be worth fsyncing is
    long enough to read.

    The writer here reports the mode it observes on its own file, which is the
    only vantage point that can tell "created 0600" from "chmod'ed to 0600
    afterwards". Both end with a 0600 destination; only one is safe.
    """
    import os
    import stat

    seen: list[int] = []

    def writer(tmp: Path) -> None:
        seen.append(stat.S_IMODE(tmp.stat().st_mode))
        tmp.write_text("secret", encoding="utf-8")

    dst = tmp_path / "users.json"
    saved = os.umask(0)
    try:
        atomic_write(dst, writer, mode=0o600)
    finally:
        os.umask(saved)

    assert seen == [0o600], f"the writer saw the tmp at {seen[0]:04o}, not 0600"
    assert stat.S_IMODE(dst.stat().st_mode) == 0o600
    assert dst.read_text(encoding="utf-8") == "secret"


def test_mode_is_reapplied_for_writers_that_replace_the_tmp(tmp_path: Path) -> None:
    """`shutil.copy2` copies the SOURCE's permission bits over the tmp.

    Pre-creating the tmp cannot help there, so the post-writer chmod is the
    backstop that makes `mode` mean the same thing for every writer. Pinned
    because deleting it would leave a rule that holds for some callers only.
    """
    import os
    import stat

    src = tmp_path / "src.txt"
    src.write_text("x", encoding="utf-8")
    os.chmod(src, 0o644)

    dst = tmp_path / "dst.txt"
    atomic_write(dst, lambda tmp: __import__("shutil").copy2(src, tmp), mode=0o600)
    assert stat.S_IMODE(dst.stat().st_mode) == 0o600
