#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import numpy as np

from scripts.bench_worker_shard_pipeline import _samples


def _hash(arrays: tuple[np.ndarray, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(value.tobytes())
    return digest.hexdigest()[:16]


def _reference(samples) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([sample.x for sample in samples], axis=0).astype(np.float16, copy=False),
        np.stack([sample.policy_target for sample in samples], axis=0).astype(
            np.float16, copy=False,
        ),
    )


def _candidate(samples) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([sample.x for sample in samples], axis=0, dtype=np.float16),
        np.stack(
            [sample.policy_target for sample in samples], axis=0, dtype=np.float16,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    samples = _samples(int(args.positions))
    timings = {"reference": [], "candidate": []}
    hashes = {"reference": [], "candidate": []}
    functions = {"reference": _reference, "candidate": _candidate}
    for round_idx in range(int(args.rounds) + 1):
        order = ("reference", "candidate") if round_idx % 2 == 0 else ("candidate", "reference")
        for name in order:
            t0 = time.perf_counter()
            arrays = functions[name](samples)
            elapsed = time.perf_counter() - t0
            if round_idx > 0:
                timings[name].append(1000.0 * elapsed)
                hashes[name].append(_hash(arrays))
    reference = statistics.median(timings["reference"])
    candidate = statistics.median(timings["candidate"])
    print(f"reference_ms={reference:.3f} hash={hashes['reference'][0]}")
    print(
        f"candidate_ms={candidate:.3f} ratio={candidate / reference:.6f} "
        f"hash={hashes['candidate'][0]} stable={len(set(hashes['candidate'])) == 1}"
    )


if __name__ == "__main__":
    main()
