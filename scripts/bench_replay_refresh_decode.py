"""Measure ordered zarr-field decode parallelism on a real replay refresh.

This is a CPU/decode screen, not the deployment yardstick.  It loads the same
fixed shard panel through the same four-shard outer scheduling shape used by
``DiskReplayBuffer`` and varies only ``load_shard_arrays(eager_workers=...)``.
Raw per-round observations are written as JSON so medians can be recomputed
without rerunning the decode.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays


def _parse_workers(raw: str) -> list[int]:
    workers = [int(piece) for piece in raw.split(",") if piece.strip()]
    if not workers or any(value <= 0 for value in workers):
        raise argparse.ArgumentTypeError("workers must be comma-separated positive integers")
    if len(set(workers)) != len(workers):
        raise argparse.ArgumentTypeError("workers must not contain duplicates")
    return workers


def _fixed_panel(paths: list[Path], count: int) -> list[Path]:
    if count <= 0:
        raise ValueError("refresh shard count must be positive")
    if len(paths) < count:
        raise ValueError(f"need at least {count} shards, found {len(paths)}")
    indices = np.linspace(0, len(paths) - 1, num=count, dtype=np.int64)
    return [paths[int(index)] for index in indices]


def _load_panel(
    panel: list[Path], *, array_workers: int, outer_workers: int,
) -> tuple[float, int, list[list[str]]]:
    started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=min(int(outer_workers), len(panel)),
        thread_name_prefix="bench-refresh-shard",
    ) as pool:
        futures = [
            pool.submit(
                load_shard_arrays,
                path,
                lazy=False,
                validate=False,
                eager_workers=int(array_workers),
            )
            for path in panel
        ]
        loaded = [future.result() for future in futures]
    wall_s = time.perf_counter() - started
    rows = sum(int(arrays["x"].shape[0]) for arrays, _ in loaded)
    keys = [list(arrays) for arrays, _ in loaded]
    return wall_s, rows, keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--workers", type=_parse_workers, default=_parse_workers("1,2,4,8"))
    parser.add_argument("--outer-workers", type=int, default=4)
    parser.add_argument("--refresh-shards", type=int, default=5)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.outer_workers <= 0 or args.warmup_rounds < 0 or args.rounds <= 0:
        raise SystemExit("outer-workers and rounds must be positive; warmup-rounds must be nonnegative")

    shard_dir = args.shard_dir.resolve()
    paths = list(iter_shard_paths(shard_dir))
    panel = _fixed_panel(paths, int(args.refresh_shards))
    workers = list(args.workers)

    expected_rows: int | None = None
    expected_keys: list[list[str]] | None = None
    observations: list[dict[str, Any]] = []
    total_rounds = int(args.warmup_rounds) + int(args.rounds)
    for round_index in range(total_rounds):
        offset = round_index % len(workers)
        order = workers[offset:] + workers[:offset]
        for array_workers in order:
            wall_s, rows, keys = _load_panel(
                panel,
                array_workers=array_workers,
                outer_workers=int(args.outer_workers),
            )
            if expected_rows is None:
                expected_rows, expected_keys = rows, keys
            if rows != expected_rows or keys != expected_keys:
                raise SystemExit(
                    f"decoded identity moved at workers={array_workers}: "
                    f"rows={rows}/{expected_rows} key_order_equal={keys == expected_keys}",
                )
            observations.append({
                "round": round_index,
                "warmup": round_index < int(args.warmup_rounds),
                "order": order,
                "array_workers": array_workers,
                "peak_decode_tasks": min(int(args.outer_workers), len(panel)) * array_workers,
                "wall_s": wall_s,
                "rows": rows,
            })
            print(
                f"round={round_index} warmup={round_index < int(args.warmup_rounds)} "
                f"array_workers={array_workers} wall_s={wall_s:.6f} rows={rows}",
                flush=True,
            )

    medians = {
        str(array_workers): statistics.median(
            row["wall_s"] for row in observations
            if not row["warmup"] and row["array_workers"] == array_workers
        )
        for array_workers in workers
    }
    baseline = medians[str(workers[0])]
    payload = {
        "schema": 1,
        "shard_dir": str(shard_dir),
        "shards_available": len(paths),
        "panel": [str(path.resolve()) for path in panel],
        "outer_workers": int(args.outer_workers),
        "array_workers": workers,
        "refresh_shards": int(args.refresh_shards),
        "warmup_rounds": int(args.warmup_rounds),
        "measured_rounds": int(args.rounds),
        "rows_per_refresh": expected_rows,
        "median_wall_s": medians,
        "wall_ratio_to_first_arm": {
            key: value / baseline for key, value in medians.items()
        },
        "observations": observations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["median_wall_s"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
