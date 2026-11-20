"""Manifests, hashing and stage bookkeeping shared by every simple-RL stage."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
STOP_FILE = "STOP"
PAUSE_FILE = "PAUSE"
REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(path: Path) -> str:
    """Digest of every regular file under ``path`` (relative names + bytes)."""
    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    h = hashlib.sha256()
    for child in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(child.relative_to(root)).encode())
        h.update(b"\0")
        h.update(sha256_file(child).encode())
        h.update(b"\n")
    return h.hexdigest()


def input_ref(path: Path) -> dict[str, Any]:
    p = Path(path)
    return {"path": str(p), "sha256": sha256_tree(p)}


def git_identity() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", "-c", "core.fsmonitor=false", *args], cwd=REPO_ROOT,
                capture_output=True, text=True, check=True, timeout=30,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return out.stdout.strip()

    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def runtime_identity() -> dict[str, Any]:
    import numpy as np
    import torch

    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "argv": list(sys.argv)}


def write_json_atomic(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with tmp.open("w") as fh:
        json.dump(value, fh, indent=1, sort_keys=True, default=_json_default)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(value, sort_keys=True, default=_json_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def read_manifest(stage_dir: Path) -> dict[str, Any] | None:
    path = Path(stage_dir) / MANIFEST
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def stage_done(stage_dir: Path) -> bool:
    return (Path(stage_dir) / MANIFEST).is_file()


def write_manifest(stage_dir: Path, *, stage: str, config: dict[str, Any], inputs: dict[str, Any],
                   outputs: dict[str, Any], counts: dict[str, Any], seeds: dict[str, Any],
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the stage's completion record. Always the LAST write of a stage."""
    manifest = {
        "schema": "SIMPLE_RL_STAGE_V1", "stage": stage, "written_unix": time.time(),
        "config": config, "inputs": inputs, "outputs": outputs, "counts": counts,
        "seeds": seeds, "git": git_identity(), "runtime": runtime_identity(),
        **(extra or {}),
    }
    write_json_atomic(Path(stage_dir) / MANIFEST, manifest)
    return manifest


def blocking_markers(run_dir: Path, extra_markers: list[Path] | tuple[Path, ...] = ()) -> list[str]:
    """Markers that forbid STARTING another stage, each with a reason.

    ``<run>/STOP`` stops the loop at the next boundary; ``<run>/PAUSE`` and any
    operator-named marker (``--pause-marker``, e.g. a GPU-window marker) refuse
    to start a stage while present. A dangling symlink counts as present.
    """
    found: list[str] = []
    for name, reason in ((STOP_FILE, "run STOP file"), (PAUSE_FILE, "run PAUSE file")):
        p = Path(run_dir) / name
        if p.exists() or p.is_symlink():
            found.append(f"{reason}: {p}")
    for marker in extra_markers:
        p = Path(marker).expanduser()
        if p.exists() or p.is_symlink():
            found.append(f"operator pause marker: {p}")
    return found


def seed_for(base: int, *parts: object) -> int:
    """Deterministic 63-bit seed derived from a base seed and stage labels."""
    h = hashlib.sha256(repr((int(base), *parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)
