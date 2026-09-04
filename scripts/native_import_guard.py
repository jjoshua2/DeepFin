"""Snapshot in-place native extensions before project imports can load them.

Keep this module free of ``chess_anti_engine`` imports.  Importing it is the
ordering barrier used by decision-grade producers: the extension binaries are
hashed here before a later project import can ask the dynamic loader to map
them into the process.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
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
    resolved = path.expanduser().resolve()
    if require_file and not resolved.is_file():
        raise SystemExit(f"decision-grade provenance requires a regular file: {resolved}")
    if not resolved.exists():
        raise SystemExit(f"artifact does not exist: {resolved}")
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }
    if resolved.is_file():
        result["sha256"] = sha256(resolved)
    return result


def _repo_extension_output(repo_root: Path, module: str) -> Path:
    base = repo_root.joinpath(*module.split("."))
    candidates = [
        base.with_name(base.name + suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ]
    output = next((candidate for candidate in candidates if candidate.is_file()), None)
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
