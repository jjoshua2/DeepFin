#!/usr/bin/env python3
"""Benchmark direct dict membership against copying pending keys to a set."""
from __future__ import annotations

import argparse
import random
import statistics
import time


def _filter_with_set(
    groups: tuple[list[int], list[int], list[int]], pending: dict[int, None],
) -> tuple[list[int], list[int], list[int]]:
    pending_set = set(pending)
    net, selfplay, curriculum = groups
    return (
        [idx for idx in net if idx not in pending_set],
        [idx for idx in selfplay if idx not in pending_set],
        [idx for idx in curriculum if idx not in pending_set],
    )


def _filter_with_dict(
    groups: tuple[list[int], list[int], list[int]], pending: dict[int, None],
) -> tuple[list[int], list[int], list[int]]:
    net, selfplay, curriculum = groups
    return (
        [idx for idx in net if idx not in pending],
        [idx for idx in selfplay if idx not in pending],
        [idx for idx in curriculum if idx not in pending],
    )


def _time_path(
    name: str,
    groups: tuple[list[int], list[int], list[int]],
    pending: dict[int, None],
    iterations: int,
) -> float:
    function = _filter_with_set if name == "set" else _filter_with_dict
    started = time.perf_counter()
    for _ in range(iterations):
        function(groups, pending)
    return iterations / (time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pending", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=1_000_000)
    args = parser.parse_args()
    if min(args.batch_size, args.rounds, args.iterations) <= 0:
        raise SystemExit("batch size, rounds, and iterations must be positive")
    if not 0 <= args.pending <= args.batch_size:
        raise SystemExit("pending must be between zero and batch size")

    groups = (
        [idx for idx in range(args.batch_size) if idx % 3 == 0],
        [idx for idx in range(args.batch_size) if idx % 3 == 1],
        [idx for idx in range(args.batch_size) if idx % 3 == 2],
    )
    pending = dict.fromkeys(range(args.pending))
    assert _filter_with_dict(groups, pending) == _filter_with_set(groups, pending)
    rng = random.Random(20260712)
    for _ in range(1000):
        keys = rng.sample(range(args.batch_size), rng.randrange(args.batch_size + 1))
        randomized = dict.fromkeys(keys)
        assert _filter_with_dict(groups, randomized) == _filter_with_set(groups, randomized)

    samples: dict[str, list[float]] = {"set": [], "dict": []}
    for round_index in range(args.rounds):
        order = ("set", "dict") if round_index % 2 == 0 else ("dict", "set")
        row: list[str] = []
        for name in order:
            throughput = _time_path(name, groups, pending, args.iterations)
            samples[name].append(throughput)
            row.append(f"{name}={throughput:,.0f} filters/s")
        print(f"round {round_index + 1}: " + "  ".join(row))

    set_median = statistics.median(samples["set"])
    dict_median = statistics.median(samples["dict"])
    print(f"set median:  {set_median:,.0f} filters/s")
    print(f"dict median: {dict_median:,.0f} filters/s")
    print(f"dict/set:    {dict_median / set_median:.6f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
