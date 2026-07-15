"""Benchmark synchronous per-step sampling versus phase-scoped prefetch."""
from __future__ import annotations

import argparse
import hashlib
import statistics
import time
from concurrent.futures import ThreadPoolExecutor


def _sample(index: int, delay: float) -> int:
    time.sleep(delay)
    return index


def _reference(steps: int, sample_delay: float, compute_delay: float) -> list[int]:
    out = []
    for index in range(steps):
        out.append(_sample(index, sample_delay))
        time.sleep(compute_delay)
    return out


def _prefetched(steps: int, sample_delay: float, compute_delay: float) -> list[int]:
    out = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_sample, 0, sample_delay)
        for index in range(steps):
            out.append(future.result())
            if index + 1 < steps:
                future = pool.submit(_sample, index + 1, sample_delay)
            time.sleep(compute_delay)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sample-ms", type=float, default=2.0)
    parser.add_argument("--compute-ms", type=float, default=8.0)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    sample_delay = args.sample_ms / 1000.0
    compute_delay = args.compute_ms / 1000.0
    samples = {"reference": [], "prefetched": []}
    hashes: set[str] = set()
    for round_idx in range(args.rounds + 2):
        order = (
            ("reference", _reference), ("prefetched", _prefetched)
        ) if round_idx % 2 == 0 else (
            ("prefetched", _prefetched), ("reference", _reference)
        )
        for name, fn in order:
            start = time.perf_counter()
            result = fn(args.steps, sample_delay, compute_delay)
            elapsed = time.perf_counter() - start
            hashes.add(hashlib.sha256(repr(result).encode()).hexdigest())
            if round_idx >= 2:
                samples[name].append(elapsed)
    if len(hashes) != 1:
        raise AssertionError(f"batch sequence mismatch: {hashes}")
    reference = statistics.median(samples["reference"])
    prefetched = statistics.median(samples["prefetched"])
    print(f"reference_seconds={reference:.9f}")
    print(f"prefetched_seconds={prefetched:.9f}")
    print(f"ratio={prefetched / reference:.6f}")
    print(f"checksum={next(iter(hashes))}")


if __name__ == "__main__":
    main()
