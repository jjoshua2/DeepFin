#!/usr/bin/env python3
"""Benchmark compact-policy length bookkeeping in ThreadedDispatcher."""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np


def _reference(
    counts: list[np.ndarray], flats: list[np.ndarray], iterations: int,
) -> int:
    del flats
    combined_counts = np.concatenate(counts)
    checksum = 0
    for _ in range(iterations):
        checksum += int(combined_counts.sum())
        offset = 0
        for request_counts in counts:
            offset += int(np.asarray(request_counts, dtype=np.int32).sum())
        checksum += offset
    return checksum


def _candidate(
    counts: list[np.ndarray], flats: list[np.ndarray], iterations: int,
) -> int:
    del counts
    combined_flat = np.concatenate(flats)
    checksum = 0
    for _ in range(iterations):
        checksum += int(combined_flat.size)
        offset = 0
        for request_flat in flats:
            offset += int(request_flat.size)
        checksum += offset
    return checksum


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--legal-per", type=int, default=32)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch % args.requests:
        raise ValueError("batch must be divisible by requests")
    rows_per_request = args.batch // args.requests
    rng = np.random.default_rng(20260712)
    counts = [
        rng.integers(
            max(1, args.legal_per - 8),
            args.legal_per + 9,
            size=rows_per_request,
            dtype=np.int32,
        )
        for _ in range(args.requests)
    ]
    flats = [
        rng.integers(0, 4672, size=int(request_counts.sum()), dtype=np.int32)
        for request_counts in counts
    ]

    expected_offsets = np.cumsum([int(item.sum()) for item in counts])
    candidate_offsets = np.cumsum([int(item.size) for item in flats])
    np.testing.assert_array_equal(candidate_offsets, expected_offsets)

    timings: dict[str, list[float]] = {"reference": [], "candidate": []}
    checksums: dict[str, int] = {}
    arms = (("reference", _reference), ("candidate", _candidate))
    for round_index in range(args.rounds):
        for name, function in (arms if round_index % 2 == 0 else reversed(arms)):
            start = time.perf_counter()
            checksums[name] = function(counts, flats, args.iterations)
            timings[name].append(time.perf_counter() - start)

    if checksums["reference"] != checksums["candidate"]:
        raise AssertionError(f"checksum mismatch: {checksums}")
    reference_s = statistics.median(timings["reference"])
    candidate_s = statistics.median(timings["candidate"])
    print(f"reference_s={reference_s:.6f}")
    print(f"candidate_s={candidate_s:.6f}")
    print(f"speedup={reference_s / candidate_s:.6f}x")
    print(f"requests={args.requests} total_legal={candidate_offsets[-1]}")
    print(f"checksum={checksums['candidate']}")


if __name__ == "__main__":
    main()
