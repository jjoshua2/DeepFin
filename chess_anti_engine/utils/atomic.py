"""Atomic file-write helpers.

All functions write to a tmp file in the destination directory, then rename
it into place. Parent directories are created if needed. Tmp names include
the pid and a uuid to prevent collisions across concurrent writers, and are
cleaned up on failure.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Callable


def _tmp_path_for(path: Path, *, preserve_suffix: bool = False) -> Path:
    if preserve_suffix:
        # e.g. foo.npz -> foo.tmp.<pid>.<uuid>.npz
        return path.with_name(f"{path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex}{path.suffix}")
    # e.g. foo.json -> foo.json.tmp.<pid>.<uuid>  (does not match "*.json" glob)
    return path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")


def atomic_write(
    path: Path,
    writer: Callable[[Path], object],
    *,
    preserve_suffix: bool = False,
) -> None:
    """Invoke ``writer(tmp)`` then atomically rename tmp to ``path``.

    Use this when the content source requires a path (``torch.save``,
    ``save_npz``, ``shutil.copy2``). For in-memory bytes or text, prefer
    :func:`atomic_write_bytes` or :func:`atomic_write_text`.

    By default the tmp name is ``<name>.tmp.<pid>.<uuid>`` so globs like
    ``*.json`` / ``*.npz`` in the same directory do not match it. Pass
    ``preserve_suffix=True`` only when the writer dispatches on the file
    extension — e.g. ``numpy.savez`` auto-appends ``.npz`` when the path
    does not already end in it. (``torch.save`` does NOT dispatch on
    extension, so it does not need this flag.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(path, preserve_suffix=preserve_suffix)
    try:
        writer(tmp)
        os.replace(str(tmp), str(path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def atomic_write_bytes(path: Path, data: bytes) -> None:
    atomic_write(path, lambda p: p.write_bytes(data))


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write(path, lambda p: p.write_text(text, encoding=encoding))


def atomic_copy2(src: Path, dst: Path) -> None:
    """``shutil.copy2`` into place atomically (preserves file metadata)."""
    atomic_write(dst, lambda tmp: shutil.copy2(src, tmp))
