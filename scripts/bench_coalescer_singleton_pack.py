#!/usr/bin/env python3
"""Measure singleton coalescer packing: unconditional copies vs aliases."""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np


def _copy_pack(
    x: np.ndarray, legal: np.ndarray, counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([x], axis=0),
        np.concatenate([legal]),
        np.concatenate([counts]),
    )


def _alias_pack(
    x: np.ndarray, legal: np.ndarray, counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return x, legal, counts


def _time_path(
    name: str,
    *,
    x: np.ndarray,
    legal: np.ndarray,
    counts: np.ndarray,
    iterations: int,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    pack = _copy_pack if name == "copy" else _alias_pack
    started = time.perf_counter()
    result = (x, legal, counts)
    for _ in range(iterations):
        result = pack(x, legal, counts)
    elapsed = time.perf_counter() - started
    return iterations / elapsed, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--planes", type=int, default=175)
    parser.add_argument("--legal-per", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if min(args.batch, args.planes, args.legal_per, args.iterations, args.rounds) <= 0:
        raise SystemExit("all numeric arguments must be positive")

    rng = np.random.default_rng(7)
    x = rng.integers(
        0, 65_536, size=(args.batch, args.planes, 8, 8), dtype=np.uint16,
    )
    counts = np.full((args.batch,), args.legal_per, dtype=np.int32)
    legal = rng.integers(
        0, 1_858, size=(args.batch * args.legal_per,), dtype=np.int32,
    )
    samples: dict[str, list[float]] = {"copy": [], "alias": []}

    for round_idx in range(args.rounds):
        order = ("copy", "alias") if round_idx % 2 == 0 else ("alias", "copy")
        row: list[str] = []
        for name in order:
            throughput, packed = _time_path(
                name,
                x=x,
                legal=legal,
                counts=counts,
                iterations=args.iterations,
            )
            np.testing.assert_array_equal(packed[0], x)
            np.testing.assert_array_equal(packed[1], legal)
            np.testing.assert_array_equal(packed[2], counts)
            if name == "alias":
                assert all(np.shares_memory(got, source) for got, source in zip(packed, (x, legal, counts)))
            samples[name].append(throughput)
            row.append(f"{name}={throughput:,.0f} packs/s")
        print(f"round {round_idx + 1}: " + "  ".join(row))

    copy_median = statistics.median(samples["copy"])
    alias_median = statistics.median(samples["alias"])
    print(f"copy median:  {copy_median:,.0f} packs/s")
    print(f"alias median: {alias_median:,.0f} packs/s")
    print(f"alias/copy:   {alias_median / copy_median:.6f}x")
    print(f"input bytes avoided per pack: {x.nbytes + legal.nbytes + counts.nbytes:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
