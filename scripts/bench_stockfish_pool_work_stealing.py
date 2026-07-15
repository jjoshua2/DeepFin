#!/usr/bin/env python3
"""Compare fixed per-engine queues with engine-owning shared workers."""

from __future__ import annotations

import argparse
import queue
import random
import statistics
import threading
import time

from concurrent.futures import Future, ThreadPoolExecutor


class _Engine:
    def run(self, request: tuple[int, int]) -> int:
        request_id, cost = request
        time.sleep(cost * 0.0005)
        return request_id * 17 + cost


class _FixedScheduler:
    def __init__(self, workers: int) -> None:
        self._engines = [_Engine() for _ in range(workers)]
        self._executors = [ThreadPoolExecutor(max_workers=1) for _ in range(workers)]
        self._next = 0

    def submit(self, request: tuple[int, int]) -> Future[int]:
        idx = self._next
        self._next = (idx + 1) % len(self._engines)
        return self._executors[idx].submit(self._engines[idx].run, request)

    def close(self) -> None:
        for executor in self._executors:
            executor.shutdown()


class _StealingScheduler:
    def __init__(self, workers: int) -> None:
        engines: queue.SimpleQueue[_Engine] = queue.SimpleQueue()
        for _ in range(workers):
            engines.put(_Engine())
        self._engines = engines
        self._local = threading.local()
        self._executor = ThreadPoolExecutor(
            max_workers=workers, initializer=self._initialize_worker,
        )

    def _initialize_worker(self) -> None:
        self._local.engine = self._engines.get()

    def _run(self, request: tuple[int, int]) -> int:
        engine: _Engine = self._local.engine
        return engine.run(request)

    def submit(self, request: tuple[int, int]) -> Future[int]:
        return self._executor.submit(self._run, request)

    def close(self) -> None:
        self._executor.shutdown()


def _run(requests: list[tuple[int, int]], *, workers: int, stealing: bool) -> tuple[float, int]:
    scheduler = _StealingScheduler(workers) if stealing else _FixedScheduler(workers)
    try:
        started = time.perf_counter()
        futures = [scheduler.submit(request) for request in requests]
        checksum = sum(future.result() for future in futures)
        return time.perf_counter() - started, checksum
    finally:
        scheduler.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests", type=int, default=96)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fixed_times: list[float] = []
    stealing_times: list[float] = []
    ratios: list[float] = []
    checksum: int | None = None
    for round_idx in range(args.rounds):
        requests = [
            (idx, 1 if rng.random() < 0.25 else 4)
            for idx in range(args.requests)
        ]
        results: dict[bool, tuple[float, int]] = {}
        order = (False, True) if round_idx % 2 == 0 else (True, False)
        for stealing in order:
            results[stealing] = _run(requests, workers=args.workers, stealing=stealing)
        fixed, fixed_checksum = results[False]
        stealing, stealing_checksum = results[True]
        if fixed_checksum != stealing_checksum:
            raise RuntimeError("scheduler checksum mismatch")
        if checksum is None:
            checksum = fixed_checksum
        fixed_times.append(fixed)
        stealing_times.append(stealing)
        ratios.append(stealing / fixed)

    fixed_median = statistics.median(fixed_times)
    stealing_median = statistics.median(stealing_times)
    print(f"fixed_median_s={fixed_median:.9f}")
    print(f"stealing_median_s={stealing_median:.9f}")
    print(f"stealing_fixed_ratio={stealing_median / fixed_median:.6f}")
    print(f"worst_round_ratio={max(ratios):.6f}")
    print(f"checksum={checksum}")


if __name__ == "__main__":
    main()
