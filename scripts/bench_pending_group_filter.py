#!/usr/bin/env python3
"""Benchmark filtering pending curriculum slots from one versus three groups."""
from __future__ import annotations

import argparse
import random
import statistics
import time

_Groups = tuple[list[int], list[int], list[int]]


def _filter_all(groups: _Groups, pending: dict[int, None]) -> _Groups:
    net, selfplay, curriculum = groups
    return (
        [idx for idx in net if idx not in pending],
        [idx for idx in selfplay if idx not in pending],
        [idx for idx in curriculum if idx not in pending],
    )


def _filter_curriculum(groups: _Groups, pending: dict[int, None]) -> _Groups:
    net, selfplay, curriculum = groups
    return net, selfplay, [idx for idx in curriculum if idx not in pending]


def _time_path(name: str, groups: _Groups, pending: dict[int, None], iterations: int) -> float:
    function = _filter_all if name == "all" else _filter_curriculum
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

    curriculum = list(range(args.pending))
    remaining = list(range(args.pending, args.batch_size))
    groups = (remaining[::2], remaining[1::2], curriculum)
    pending = dict.fromkeys(curriculum)
    assert _filter_curriculum(groups, pending) == _filter_all(groups, pending)

    rng = random.Random(20260712)
    for _ in range(1000):
        shuffled = list(range(args.batch_size))
        rng.shuffle(shuffled)
        net_size = rng.randrange(args.batch_size + 1)
        selfplay_size = rng.randrange(args.batch_size - net_size + 1)
        randomized_groups = (
            shuffled[:net_size],
            shuffled[net_size:net_size + selfplay_size],
            shuffled[net_size + selfplay_size:],
        )
        randomized_cur = randomized_groups[2]
        count = rng.randrange(len(randomized_cur) + 1)
        randomized_pending = dict.fromkeys(rng.sample(randomized_cur, count))
        assert _filter_curriculum(randomized_groups, randomized_pending) == _filter_all(
            randomized_groups, randomized_pending,
        )

    samples: dict[str, list[float]] = {"all": [], "curriculum": []}
    for round_index in range(args.rounds):
        order = ("all", "curriculum") if round_index % 2 == 0 else ("curriculum", "all")
        row: list[str] = []
        for name in order:
            throughput = _time_path(name, groups, pending, args.iterations)
            samples[name].append(throughput)
            row.append(f"{name}={throughput:,.0f} filters/s")
        print(f"round {round_index + 1}: " + "  ".join(row))

    all_median = statistics.median(samples["all"])
    curriculum_median = statistics.median(samples["curriculum"])
    print(f"all median:        {all_median:,.0f} filters/s")
    print(f"curriculum median: {curriculum_median:,.0f} filters/s")
    print(f"curriculum/all:    {curriculum_median / all_median:.6f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
