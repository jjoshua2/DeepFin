"""Read-only directory/packed-Zarr exact-sampler comparison on frozen inputs.

This does not migrate a corpus, configure training, or prove cold-cache speed.
Choose enough shards for the sampler's game-frequency constraint at the requested
batch size (e.g. at least 32 of the bootstrap pilot shards for batch512).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def tensor_digest(arrays: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def measure(root: Path, *, packed: bool, options: dict[str, Any]) -> dict[str, Any]:
    from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer

    start = time.monotonic()
    buffer = GameAwareEpochBuffer(shard_dir=root, allow_packed_zarr=packed, **options)
    planning = time.monotonic() - start
    start = time.monotonic()
    sequence = hashlib.sha256()
    rows = 0
    digest_seconds = 0.0
    batch_wait_seconds = 0.0
    try:
        for _ in range(buffer.num_batches):
            tick = time.monotonic()
            arrays = buffer.sample_batch_arrays(options["batch_size"])
            batch_wait_seconds += time.monotonic() - tick
            rows += len(arrays["x"])
            tick = time.monotonic()
            sequence.update(tensor_digest(arrays).encode())
            digest_seconds += time.monotonic() - tick
        return {
            "root": str(root.resolve()),
            "packed_opt_in": packed,
            "plan": buffer.plan.as_dict(),
            "rows": rows,
            "planning_seconds": planning,
            "consumer_wall_seconds_including_digest": time.monotonic() - start,
            "digest_seconds": digest_seconds,
            "batch_wait_seconds": batch_wait_seconds,
            "sequence_sha256": sequence.hexdigest(),
        }
    finally:
        buffer.close()


def qualify(directory: Path, packed: Path, options: dict[str, Any]) -> dict[str, Any]:
    from chess_anti_engine.replay.packed_zarr import shard_paths
    from chess_anti_engine.replay.shard import iter_shard_paths

    ordinary = iter_shard_paths(directory)
    archives = shard_paths(packed)
    if (
        not ordinary
        or [p.name for p in ordinary] != [p.name.removesuffix(".zip") for p in archives]
        or any(not p.name.endswith(".zarr.zip") for p in archives)
    ):
        raise ValueError("qualification requires identical directory/ZIP shard rosters")
    results = [
        measure(directory, packed=False, options=options),
        measure(packed, packed=True, options=options),
    ]
    if (
        results[0]["rows"] != results[1]["rows"]
        or results[0]["sequence_sha256"] != results[1]["sequence_sha256"]
    ):
        raise ValueError(
            "packed sampler rows/order/tensors differ from directory control"
        )
    return {
        "status": "PASS_MATCHED_PACKED_ZARR_SAMPLER",
        "runs": results,
        "scope": "Exact sampler only; fixed order, uncontrolled caches; no trainer/GPU or live adoption",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--packed", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--input-planes", type=int, required=True)
    p.add_argument("--input-history-encoding", required=True)
    p.add_argument(
        "--history-rep-fix", action=argparse.BooleanOptionalAction, required=True
    )
    p.add_argument("--mirror-augmentation", action="store_true")
    p.add_argument("--working-set-gib", type=float, default=12)
    p.add_argument("--seconds", type=int, default=1800)
    a = p.parse_args()
    if not (
        1 <= a.batch_size <= 4096
        and 0 < a.working_set_gib <= 12
        and 1 <= a.seconds <= 7200
    ):
        p.error("invalid batch/memory/time bounds")
    if a.result.exists() or not a.result.parent.is_dir():
        p.error("result must be a fresh file in an existing artifact directory")
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLOSC_NTHREADS",
    ]:
        os.environ[key] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    os.nice(19)
    import psutil
    from numcodecs.blosc import set_nthreads

    set_nthreads(2)
    start = time.monotonic()
    done = threading.Event()

    def check():
        if (
            time.monotonic() - start > a.seconds
            or psutil.Process().memory_info().rss > 16 * 2**30
            or psutil.virtual_memory().available < 32 * 2**30
            or (a.result.parent / "STOP").exists()
        ):
            raise RuntimeError("qualification time/memory/STOP limit")

    def monitor():
        while not done.wait(0.5):
            try:
                check()
            except BaseException as exc:
                try:
                    a.result.write_text(
                        json.dumps({"status": "FAILED", "error": repr(exc)})
                    )
                finally:
                    os._exit(9)

    check()
    threading.Thread(target=monitor, daemon=True).start()
    result: dict[str, Any] = {"status": "FAILED"}
    try:
        result = qualify(
            a.directory,
            a.packed,
            {
                "batch_size": a.batch_size,
                "seed": 121,
                "input_planes": a.input_planes,
                "input_history_encoding": a.input_history_encoding,
                "history_rep_fix": a.history_rep_fix,
                "mirror_augmentation": a.mirror_augmentation,
                "plan_workers": 1,
                "load_workers": 1,
                "max_working_set_bytes": int(a.working_set_gib * 2**30),
            },
        )
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        done.set()
        result["elapsed_seconds"] = time.monotonic() - start
        a.result.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
