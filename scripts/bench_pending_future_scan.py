#!/usr/bin/env python3
"""Benchmark snapshot-all versus ready-only pending-future collection."""
from __future__ import annotations

import argparse
import random
import statistics
import time


class _Future:
    def __init__(self, value: int, *, ready: bool) -> None:
        self.value = value
        self.ready = ready

    def done(self) -> bool:
        return self.ready

    def result(self) -> int:
        return self.value


def _snapshot_all(pending: dict[int, _Future]) -> list[tuple[int, _Future]]:
    return [(idx, future) for idx, future in list(pending.items()) if future.done()]


def _ready_only(pending: dict[int, _Future]) -> list[tuple[int, _Future]]:
    return [(idx, future) for idx, future in pending.items() if future.done()]


def _finish(
    pending: dict[int, _Future], collector,
) -> tuple[list[tuple[int, int]], dict[int, _Future]]:
    completed: list[tuple[int, int]] = []
    for idx, future in collector(pending):
        completed.append((idx, future.result()))
        del pending[idx]
    return completed, pending


def _time_path(name: str, pending: dict[int, _Future], iterations: int) -> float:
    collector = _snapshot_all if name == "snapshot" else _ready_only
    started = time.perf_counter()
    for _ in range(iterations):
        collector(pending)
    return iterations / (time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending", type=int, default=24)
    parser.add_argument("--ready", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=1_000_000)
    args = parser.parse_args()
    if min(args.pending, args.rounds, args.iterations) <= 0:
        raise SystemExit("pending, rounds, and iterations must be positive")
    if not 0 <= args.ready <= args.pending:
        raise SystemExit("ready must be between zero and pending")

    pending = {
        idx: _Future(idx * 7, ready=idx < args.ready)
        for idx in range(args.pending)
    }
    assert _snapshot_all(pending) == _ready_only(pending)
    rng = random.Random(20260712)
    for _ in range(1000):
        reference = {
            idx: _Future(idx * 7, ready=bool(rng.randrange(2)))
            for idx in range(rng.randrange(1, args.pending + 1))
        }
        candidate = {
            idx: _Future(future.value, ready=future.ready)
            for idx, future in reference.items()
        }
        ref_completed, ref_remaining = _finish(reference, _snapshot_all)
        got_completed, got_remaining = _finish(candidate, _ready_only)
        assert got_completed == ref_completed
        assert tuple(got_remaining) == tuple(ref_remaining)

    samples: dict[str, list[float]] = {"snapshot": [], "ready": []}
    for round_index in range(args.rounds):
        order = ("snapshot", "ready") if round_index % 2 == 0 else ("ready", "snapshot")
        row: list[str] = []
        for name in order:
            throughput = _time_path(name, pending, args.iterations)
            samples[name].append(throughput)
            row.append(f"{name}={throughput:,.0f} scans/s")
        print(f"round {round_index + 1}: " + "  ".join(row))

    snapshot_median = statistics.median(samples["snapshot"])
    ready_median = statistics.median(samples["ready"])
    print(f"snapshot median: {snapshot_median:,.0f} scans/s")
    print(f"ready median:    {ready_median:,.0f} scans/s")
    print(f"ready/snapshot:  {ready_median / snapshot_median:.6f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
