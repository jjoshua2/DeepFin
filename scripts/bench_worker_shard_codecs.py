#!/usr/bin/env python3
"""Compare lossless codecs for production-shaped local worker zarr shards."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from chess_anti_engine.replay.shard import (
    _local_chunks,
    prune_storage_arrays,
    samples_to_arrays,
)
from scripts.bench_worker_shard_pipeline import _samples


def _tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _core_hash(group) -> str:
    digest = hashlib.sha256()
    for name in ("x", "policy_target", "wdl_target", "legal_mask"):
        value = np.asarray(group[name][:])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    arrays = prune_storage_arrays(samples_to_arrays(_samples(int(args.positions))))
    codecs = {
        "zstd3": Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
        "zstd2": Blosc(cname="zstd", clevel=2, shuffle=Blosc.SHUFFLE),
        "zstd1": Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE),
        "lz4_1": Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE),
        "lz4_3": Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
    }
    writes: dict[str, list[float]] = {name: [] for name in codecs}
    reads: dict[str, list[float]] = {name: [] for name in codecs}
    sizes: dict[str, list[int]] = {name: [] for name in codecs}
    hashes: dict[str, list[str]] = {name: [] for name in codecs}
    names = list(codecs)

    with tempfile.TemporaryDirectory(prefix="cae-codec-bench-") as td:
        root = Path(td)
        for round_idx in range(int(args.rounds) + 1):
            order = names[round_idx % len(names):] + names[:round_idx % len(names)]
            for name in order:
                path = root / f"{round_idx}_{name}.zarr"
                t0 = time.perf_counter()
                group = zarr.open_group(str(path), mode="w")
                for field, value in arrays.items():
                    if field.startswith("_"):
                        continue
                    arr = np.asarray(value)
                    group.create_dataset(
                        field,
                        data=arr,
                        chunks=_local_chunks(arr),
                        compressor=codecs[name],
                        overwrite=True,
                    )
                t1 = time.perf_counter()
                reopened = zarr.open_group(str(path), mode="r")
                decoded_hash = _core_hash(reopened)
                t2 = time.perf_counter()
                if round_idx > 0:
                    writes[name].append(1000.0 * (t1 - t0))
                    reads[name].append(1000.0 * (t2 - t1))
                    sizes[name].append(_tree_bytes(path))
                    hashes[name].append(decoded_hash)
                shutil.rmtree(path)

    baseline_write = statistics.median(writes["zstd3"])
    baseline_read = statistics.median(reads["zstd3"])
    baseline_size = statistics.median(sizes["zstd3"])
    print(f"positions={args.positions} rounds={args.rounds}")
    for name in names:
        write = statistics.median(writes[name])
        read = statistics.median(reads[name])
        size = statistics.median(sizes[name])
        print(
            f"{name} write_ms={write:.3f} write_ratio={write / baseline_write:.6f} "
            f"read_ms={read:.3f} read_ratio={read / baseline_read:.6f} "
            f"bytes={int(size)} size_ratio={size / baseline_size:.6f} "
            f"hash={hashes[name][0]} stable={len(set(hashes[name])) == 1}"
        )


if __name__ == "__main__":
    main()
