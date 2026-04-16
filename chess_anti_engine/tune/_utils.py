from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np

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
    # Use intersection so shards missing optional keys (e.g. future_policy_target
    # absent from bootstrap data) don't crash the concat.
    keys = set(batches[0].keys())
    for batch in batches[1:]:
        keys &= set(batch.keys())
    return {
        k: np.concatenate([np.asarray(batch[k]) for batch in batches], axis=0)
        for k in sorted(keys)
    }


def to_nonnegative_int(v: object, default: int = 0) -> int:
    try:
        iv = int(v)  # type: ignore[arg-type]
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
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception:
            pass


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
