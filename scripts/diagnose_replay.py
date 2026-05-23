"""Read-only replay sampling helpers for diagnostic scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays, shard_positions


def _take_rows(array, rows: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(array[rows])
    except Exception:
        return np.asarray(array)[rows]


def sample_replay_arrays(
    shard_dir: Path,
    n: int,
    *,
    rng: np.random.Generator,
    fields: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], int, int]:
    """Sample replay rows without constructing ``DiskReplayBuffer``.

    ``DiskReplayBuffer`` enforces its capacity during construction. Diagnostics
    must be read-only, so they sample directly from existing shards instead.
    """
    paths = iter_shard_paths(shard_dir)
    if not paths:
        sys.exit(f"No replay shards found in {shard_dir}")

    sizes = np.asarray([shard_positions(path) for path in paths], dtype=np.int64)
    keep = sizes > 0
    paths = [path for path, ok in zip(paths, keep) if bool(ok)]
    sizes = sizes[keep]
    total = int(sizes.sum())
    if total <= 0:
        sys.exit(f"No replay positions found in {shard_dir}")

    sample_n = min(int(n), total)
    picks = np.sort(rng.choice(total, size=sample_n, replace=False).astype(np.int64))
    cumulative = np.cumsum(sizes)
    shard_idxs = np.searchsorted(cumulative, picks, side="right")
    shard_starts = cumulative - sizes

    parts: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    for shard_idx in np.unique(shard_idxs):
        mask = shard_idxs == shard_idx
        rows = picks[mask] - int(shard_starts[int(shard_idx)])
        arrs, _ = load_shard_arrays(paths[int(shard_idx)], lazy=True)
        for field in fields:
            if field in arrs:
                parts[field].append(_take_rows(arrs[field], rows))
            elif field == "has_policy":
                parts[field].append(np.ones((rows.shape[0],), dtype=np.uint8))
            else:
                sys.exit(f"Replay shard {paths[int(shard_idx)]} is missing required field {field!r}")

    return {field: np.concatenate(chunks, axis=0) for field, chunks in parts.items()}, total, len(paths)
