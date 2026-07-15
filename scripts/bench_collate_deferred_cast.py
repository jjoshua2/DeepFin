"""Benchmark host preparation for replay float16-to-float32 transfers."""
from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import numpy as np
import torch


def _reference(arrays: list[np.ndarray]) -> list[np.ndarray]:
    # Current path: NumPy widens, then pin_memory copies that widened source.
    return [np.asarray(arr, dtype=np.float32).copy(order="C") for arr in arrays]


def _candidate(arrays: list[np.ndarray]) -> list[np.ndarray]:
    # Candidate path: pin_memory copies the compact source; device transfer casts.
    return [np.asarray(arr).copy(order="C") for arr in arrays]


def _checksum(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for arr in arrays:
        digest.update(np.asarray(arr, dtype=np.float32).tobytes(order="C"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--planes", type=int, default=175)
    parser.add_argument("--policy-size", type=int, default=1858)
    parser.add_argument("--policy-fields", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()

    torch.set_num_threads(1)
    rng = np.random.default_rng(20260715)
    arrays = [
        rng.standard_normal(
            (args.batch_size, args.planes, 8, 8),
            dtype=np.float32,
        ).astype(np.float16),
        *[
            rng.random((args.batch_size, args.policy_size), dtype=np.float32).astype(np.float16)
            for _ in range(args.policy_fields)
        ],
    ]
    samples: dict[str, list[float]] = {"reference": [], "candidate": []}
    checksums: set[str] = set()
    source_bytes: dict[str, int] = {}
    for round_idx in range(args.rounds + 2):
        order = (
            ("reference", _reference),
            ("candidate", _candidate),
        ) if round_idx % 2 == 0 else (
            ("candidate", _candidate),
            ("reference", _reference),
        )
        for name, fn in order:
            start = time.perf_counter()
            result: list[np.ndarray] = []
            for _ in range(args.iterations):
                result = fn(arrays)
            elapsed = time.perf_counter() - start
            checksums.add(_checksum(result))
            source_bytes[name] = sum(arr.nbytes for arr in result)
            if round_idx >= 2:
                samples[name].append(elapsed)

    if len(checksums) != 1:
        raise AssertionError(f"cast mismatch: {checksums}")
    reference = statistics.median(samples["reference"])
    candidate = statistics.median(samples["candidate"])
    print(f"reference_seconds={reference:.9f}")
    print(f"candidate_seconds={candidate:.9f}")
    print(f"ratio={candidate / reference:.6f}")
    print(f"reference_source_bytes={source_bytes['reference']}")
    print(f"candidate_source_bytes={source_bytes['candidate']}")
    print(f"source_byte_ratio={source_bytes['candidate'] / source_bytes['reference']:.6f}")
    print(f"checksum={next(iter(checksums))}")


if __name__ == "__main__":
    main()
