#!/usr/bin/env python3
"""Benchmark singleton compact-legal concatenation against array aliasing."""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np


def _copy_pack(legal: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([np.asarray(legal, dtype=np.int32)]),
        np.concatenate([np.asarray(counts, dtype=np.int32)]),
    )


def _alias_pack(legal: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(legal, dtype=np.int32), np.asarray(counts, dtype=np.int32)


def _time_path(
    name: str, legal: np.ndarray, counts: np.ndarray, iterations: int,
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    function = _copy_pack if name == "copy" else _alias_pack
    packed = (legal, counts)
    started = time.perf_counter()
    for _ in range(iterations):
        packed = function(legal, counts)
    return iterations / (time.perf_counter() - started), packed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--legal-per", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=500_000)
    args = parser.parse_args()
    if min(args.batch, args.legal_per, args.rounds, args.iterations) <= 0:
        raise SystemExit("all arguments must be positive")

    rng = np.random.default_rng(20260712)
    counts = np.full((args.batch,), args.legal_per, dtype=np.int32)
    legal = rng.integers(0, 1_858, size=args.batch * args.legal_per, dtype=np.int32)
    samples: dict[str, list[float]] = {"copy": [], "alias": []}
    for round_index in range(args.rounds):
        order = ("copy", "alias") if round_index % 2 == 0 else ("alias", "copy")
        row: list[str] = []
        for name in order:
            throughput, packed = _time_path(name, legal, counts, args.iterations)
            np.testing.assert_array_equal(packed[0], legal)
            np.testing.assert_array_equal(packed[1], counts)
            if name == "alias":
                assert np.shares_memory(packed[0], legal)
                assert np.shares_memory(packed[1], counts)
            samples[name].append(throughput)
            row.append(f"{name}={throughput:,.0f} packs/s")
        print(f"round {round_index + 1}: " + "  ".join(row))

    copy_median = statistics.median(samples["copy"])
    alias_median = statistics.median(samples["alias"])
    print(f"copy median:  {copy_median:,.0f} packs/s")
    print(f"alias median: {alias_median:,.0f} packs/s")
    print(f"alias/copy:   {alias_median / copy_median:.6f}x")
    print(f"bytes avoided: {legal.nbytes + counts.nbytes:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
