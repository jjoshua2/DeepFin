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


def _tmp_path_for(path: Path) -> Path:
    # Preserve the suffix so callers that dispatch on extension (numpy.savez
    # appends .npz when absent; torch uses zip if given .pt, etc.) still see
    # the expected type.
    return path.with_name(f"{path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex}{path.suffix}")


def atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    """Invoke ``writer(tmp)`` then atomically rename tmp to ``path``.

    Use this when the content source requires a path (``torch.save``,
    ``save_npz``, ``shutil.copy2``). For in-memory bytes or text, prefer
    :func:`atomic_write_bytes` or :func:`atomic_write_text`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(path)
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
