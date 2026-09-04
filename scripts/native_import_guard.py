"""Snapshot in-place native extensions before project imports can load them.

Keep this module free of ``chess_anti_engine`` imports.  Importing it is the
ordering barrier used by decision-grade producers: the extension binaries are
hashed here before a later project import can ask the dynamic loader to map
them into the process.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import errno
import os
import stat
from pathlib import Path
from typing import Any


EARLY_NATIVE_MODULES = (
    "chess_anti_engine.encoding._features_ext",
    "chess_anti_engine.encoding._lc0_ext",
    "chess_anti_engine.mcts._mcts_tree",
)


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_mode), int(value.st_size), int(value.st_mtime_ns),
        int(value.st_ctime_ns), int(value.st_dev), int(value.st_ino),
    )


def _stable_regular_bytes(path: Path) -> tuple[bytes, os.stat_result]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise OSError(errno.ENOTSUP, "safe reads require O_NOFOLLOW", str(path))
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | nofollow,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError(errno.EINVAL, "not a regular file", str(path))
        chunks: list[bytes] = []
        offset = 0
        while offset < int(before.st_size):
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, int(before.st_size) - offset),
                offset,
            )
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        if _stable_identity(before) != _stable_identity(after) or len(
            content
        ) != int(after.st_size):
            raise OSError(errno.EIO, "file changed during stable read", str(path))
        return content, after
    finally:
        os.close(descriptor)


def sha256(path: Path) -> str:
    content, _observed = _stable_regular_bytes(path)
    return hashlib.sha256(content).hexdigest()


def artifact(path: Path, *, require_file: bool) -> dict[str, Any]:
    lexical = Path(os.path.abspath(path.expanduser()))
    resolved = lexical
    try:
        before = lexical.lstat()
        if stat.S_ISLNK(before.st_mode):
            raise SystemExit(
                f"decision-grade provenance rejects symlink artifacts: {lexical}"
        )
        resolved = lexical.resolve(strict=True)
        if stat.S_ISREG(before.st_mode):
            content, after = _stable_regular_bytes(lexical)
        else:
            content = None
            after = lexical.lstat()
    except OSError as exc:
        raise SystemExit(f"artifact cannot be read: {resolved}: {exc}") from exc
    before_identity = (
        int(before.st_mode), int(before.st_size), int(before.st_mtime_ns),
        int(before.st_ctime_ns), int(before.st_dev), int(before.st_ino),
    )
    after_identity = (
        int(after.st_mode), int(after.st_size), int(after.st_mtime_ns),
        int(after.st_ctime_ns), int(after.st_dev), int(after.st_ino),
    )
    stable = before_identity == after_identity
    if require_file and (not stat.S_ISREG(after.st_mode) or not stable):
        raise SystemExit(
            f"decision-grade provenance requires a stable regular file: {resolved}"
        )
    result: dict[str, Any] = {
        "path": str(resolved),
        "lexical_path": str(lexical),
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "stable_read": stable,
    }
    if content is not None:
        result["sha256"] = hashlib.sha256(content).hexdigest()
    return result


def _repo_extension_output(repo_root: Path, module: str) -> Path:
    base = repo_root.joinpath(*module.split("."))
    candidates = [
        base.with_name(base.name + suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ]
    output = next((
        candidate
        for candidate in candidates
        if candidate.exists()
        and stat.S_ISREG(candidate.lstat().st_mode)
        and not stat.S_ISLNK(candidate.lstat().st_mode)
    ), None)
    if output is None:
        raise SystemExit(f"decision-grade native extension is missing: {module}")
    return output


def snapshot_repo_native_extensions(repo_root: Path) -> dict[str, dict[str, Any]]:
    return {
        module: artifact(_repo_extension_output(repo_root, module), require_file=True)
        for module in EARLY_NATIVE_MODULES
    }


PREIMPORT_NATIVE_ARTIFACTS = snapshot_repo_native_extensions(
    Path(__file__).resolve().parents[1],
)
