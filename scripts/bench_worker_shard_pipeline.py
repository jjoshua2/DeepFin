#!/usr/bin/env python3
"""Profile the worker's completion-to-upload shard stages on synthetic v2 rows."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from chess_anti_engine.replay import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    load_shard_arrays,
    pack_shard_for_upload,
    save_local_shard_arrays,
    samples_to_arrays,
)


def _samples(count: int) -> list[ReplaySample]:
    rng = np.random.default_rng(20260712)
    xs = rng.integers(
        0, 2, size=(count, 175, 8, 8), dtype=np.uint8,
    ).astype(np.float32)
    policy = np.zeros((1858,), dtype=np.float32)
    selected = rng.choice(1858, size=40, replace=False)
    policy[selected] = np.float32(1.0 / len(selected))
    legal = np.zeros((1858,), dtype=np.uint8)
    legal[selected] = 1
    categorical = np.full((32,), np.float32(1.0 / 32.0), dtype=np.float32)
    wdl = np.asarray([0.45, 0.25, 0.30], dtype=np.float32)
    raw = np.full((48, 5), (-1, -32768, 0, -1, -1), dtype=np.int16)
    raw[:40, 0] = selected[:40].astype(np.int16)
    raw[:40, 1] = np.arange(200, 160, -1, dtype=np.int16)
    raw[:40, 3] = np.arange(600, 560, -1, dtype=np.int16)
    raw[:40, 4] = 200
    samples: list[ReplaySample] = []
    for i in range(count):
        sample = ReplaySample(
            x=xs[i],
            policy_target=policy,
            wdl_target=i % 3,
            x_lc0_root=xs[i],
            input_history_encoding="lc0_8ply_v2_threats",
            history_rep_fix=True,
            sf_wdl=wdl,
            sf_multipv_raw=raw,
            sf_label_meta=np.asarray([700_000, 25, 100, 0, 600, 200], dtype=np.int32),
            search_wdl=wdl,
            categorical_target=categorical,
            policy_soft_target=policy,
            future_policy_target=policy,
            has_future=True,
            sf_p0_regret=policy,
            has_sf_p0_regret=True,
            legal_mask=legal,
            sf_legal_mask=legal,
            future_legal_mask=legal,
        )
        sample.game_id = i // 80
        sample.ply_index = i % 80
        sample.has_policy = True
        sample.is_selfplay = True
        samples.append(sample)
    return samples


def _hash_arrays(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
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
    samples = _samples(int(args.positions))
    convert_ms: list[float] = []
    zarr_ms: list[float] = []
    tar_ms: list[float] = []
    hashes: list[str] = []
    tar_sizes: list[int] = []

    with tempfile.TemporaryDirectory(prefix="cae-shard-bench-") as td:
        root = Path(td)
        for round_idx in range(int(args.rounds) + 1):
            shard = root / f"round_{round_idx}.zarr"
            t0 = time.perf_counter()
            arrays = samples_to_arrays(samples)
            t1 = time.perf_counter()
            save_local_shard_arrays(
                shard,
                arrs=arrays,
                meta=ShardMeta(username="bench", positions=len(samples)),
            )
            t2 = time.perf_counter()
            _name, payload = pack_shard_for_upload(shard)
            t3 = time.perf_counter()
            loaded, meta = load_shard_arrays(shard)
            assert int(meta["positions"]) == len(samples)
            assert np.array_equal(loaded["wdl_target"], arrays["wdl_target"])
            if round_idx > 0:
                convert_ms.append(1000.0 * (t1 - t0))
                zarr_ms.append(1000.0 * (t2 - t1))
                tar_ms.append(1000.0 * (t3 - t2))
                hashes.append(_hash_arrays(arrays))
                tar_sizes.append(payload.getbuffer().nbytes)
            payload.close()
            shutil.rmtree(shard)

    convert = statistics.median(convert_ms)
    zarr = statistics.median(zarr_ms)
    tar = statistics.median(tar_ms)
    materialize = convert + zarr
    total = materialize + tar
    print(f"positions={len(samples)} rounds={args.rounds}")
    print(f"samples_to_arrays_ms={convert:.3f}")
    print(f"zarr_write_ms={zarr:.3f}")
    print(f"tar_pack_ms={tar:.3f}")
    print(f"locked_materialize_ms={materialize:.3f}")
    print(f"locked_pipeline_share={materialize / max(total, 1e-9):.6f}")
    print(f"array_hash={hashes[0]} stable={len(set(hashes)) == 1}")
    print(f"tar_bytes={tar_sizes[0]} stable={len(set(tar_sizes)) == 1}")


if __name__ == "__main__":
    main()
