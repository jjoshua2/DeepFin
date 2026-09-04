"""Snapshot in-place native extensions before project imports can load them.

Keep this module free of ``chess_anti_engine`` imports.  Importing it is the
ordering barrier used by decision-grade producers: the extension binaries are
hashed here before a later project import can ask the dynamic loader to map
them into the process.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import os
import stat
from pathlib import Path
from typing import Any


EARLY_NATIVE_MODULES = (
    "chess_anti_engine.encoding._features_ext",
    "chess_anti_engine.encoding._lc0_ext",
    "chess_anti_engine.mcts._mcts_tree",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        content = lexical.read_bytes() if stat.S_ISREG(before.st_mode) else None
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
