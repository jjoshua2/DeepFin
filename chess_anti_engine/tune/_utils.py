from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from chess_anti_engine.replay.shard import (
    _REQUIRED_STORAGE_FIELDS,
    _SHARD_FIELDS,
    zeros_for_storage_field,
)


  # Checkpoint sidecar filenames — read by trainable_init, written by
  # trainable_report + salvage. Keeping these in one place avoids
  # reader/writer drift on rename.
SIDECAR_PID_STATE = "pid_state.json"
SIDECAR_RNG_STATE = "rng_state.json"
SIDECAR_TRIAL_META = "trial_meta.json"


def load_optional_json(path: Path) -> dict | None:
    """Read ``path`` as JSON. Returns the parsed dict, or ``None`` for any
    miss (file absent, unreadable, malformed, or non-dict root). Used for
    optional checkpoint sidecar files (pid_state.json, rng_state.json,
    trial_meta.json, etc.) where the absence of the file is not an error.
    """
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def stable_seed_u32(*parts: object) -> int:
    """Deterministic 32-bit seed from arbitrary parts."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "little") & 0xFFFFFFFF


def slice_array_batch(arrs: dict[str, np.ndarray], idxs: np.ndarray) -> dict[str, np.ndarray]:
    ii = np.asarray(idxs, dtype=np.int64).reshape(-1)
    return {k: np.array(np.asarray(v)[ii], copy=True, order="C") for k, v in arrs.items()}


def concat_array_batches(batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not batches:
        raise ValueError("cannot concatenate empty replay shard list")

    def _zeros_for_missing(name: str, batch: dict[str, np.ndarray]) -> np.ndarray:
        return zeros_for_storage_field(
            name,
            n=int(np.asarray(batch["x"]).shape[0]),
            policy_size=int(np.asarray(batch["policy_target"]).shape[1]),
            x_planes=int(np.asarray(batch["x"]).shape[1]),
        )

    present_keys = set().union(*(batch.keys() for batch in batches))
    keys = [
        name for name in _SHARD_FIELDS
        if name in _REQUIRED_STORAGE_FIELDS or name in present_keys
    ]
    return {
        k: np.concatenate(
            [
                np.asarray(batch[k]) if k in batch else _zeros_for_missing(k, batch)
                for batch in batches
            ],
            axis=0,
        )
        for k in keys
    }


def to_nonnegative_int(v: object, default: int = 0) -> int:
    try:
        iv = int(v)  # type: ignore[arg-type] # object input validated by try/except
        return iv if iv >= 0 else default
    except Exception:
        return default


def terminate_process(proc: subprocess.Popen[bytes] | None, *, timeout_s: float = 5.0) -> None:
    """Gracefully terminate a subprocess (SIGTERM → wait → SIGKILL fallback)."""
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=float(timeout_s))
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            pass  # stuck or gone — caller is on a cleanup path either way


def resolve_local_override_root(
    *,
    raw_root: object,
    tune_work_dir: object,
    suffix: str,
) -> Path:
    root = Path(str(raw_root or "")).expanduser()
    tune_dir = Path(str(tune_work_dir)).expanduser()
    if not tune_dir.is_absolute():
        tune_dir = Path(__file__).resolve().parents[2] / tune_dir
    tune_dir = tune_dir.resolve()
    run_root = tune_dir.parent
    if root.as_posix().startswith("/mnt/c/chess_active/"):
        return run_root.with_name(f"{run_root.name}_{suffix}")
    return root
