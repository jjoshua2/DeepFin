#!/usr/bin/env python3
"""Compare shared versus per-engine executor head-of-line latency."""

from __future__ import annotations

import argparse
import statistics
import threading
import time

from concurrent.futures import Future, ThreadPoolExecutor


class _Engine:
    def __init__(self, index: int, releases: list[threading.Event]) -> None:
        self.index = index
        self.release = releases[index]
        self.started = threading.Event()
        self.lock = threading.Lock()

    def run(self, command: str) -> int:
        with self.lock:
            if command == "block":
                self.started.set()
                self.release.wait()
            elif command == "probe":
                time.sleep(0.0001)
            return self.index + len(command)


class _Scheduler:
    def __init__(self, workers: int, *, sharded: bool) -> None:
        self.releases = [threading.Event() for _ in range(workers)]
        self.engines = [_Engine(i, self.releases) for i in range(workers)]
        self.executors = (
            [ThreadPoolExecutor(max_workers=1) for _ in range(workers)]
            if sharded
            else [ThreadPoolExecutor(max_workers=workers)]
        )
        self.sharded = sharded
        self.next_engine = 0

    def submit(self, command: str) -> Future[int]:
        idx = self.next_engine
        self.next_engine = (idx + 1) % len(self.engines)
        executor = self.executors[idx] if self.sharded else self.executors[0]
        return executor.submit(self.engines[idx].run, command)

    def close(self) -> None:
        for release in self.releases:
            release.set()
        for executor in self.executors:
            executor.shutdown(wait=True)


def _run_once(workers: int, requests: int, *, sharded: bool) -> tuple[float, int]:
    scheduler = _Scheduler(workers, sharded=sharded)
    futures = [scheduler.submit("block") for _ in range(workers)]
    for engine in scheduler.engines:
        if not engine.started.wait(timeout=1.0):
            raise RuntimeError("blocker failed to start")

    # These bind to engine 0 and engine 1 respectively. Releasing engine 1 in
    # the shared design lets its executor thread take engine 0's queued work
    # and block on that lock, so the engine-1 probe cannot run despite engine 1
    # being available. The sharded design leaves the probe on engine 1's queue.
    futures.append(scheduler.submit("queued"))
    probe = scheduler.submit("probe")
    futures.append(probe)
    futures.extend(
        scheduler.submit("filler") for _ in range(requests - workers - 2)
    )

    scheduler.releases[1].set()
    start = time.perf_counter()
    release_timer = threading.Timer(0.01, scheduler.releases[0].set)
    release_timer.start()
    try:
        probe_result = probe.result(timeout=1.0)
        elapsed = time.perf_counter() - start
    finally:
        scheduler.close()
        release_timer.cancel()
    checksum = probe_result + sum(future.result() for future in futures)
    return elapsed, checksum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    if args.workers < 2 or args.requests < args.workers + 2 or args.rounds < 1:
        parser.error("need workers >= 2, requests >= workers + 2, and rounds >= 1")

    reference: list[float] = []
    candidate: list[float] = []
    checksum: int | None = None
    for round_idx in range(args.rounds):
        order = (False, True) if round_idx % 2 == 0 else (True, False)
        for sharded in order:
            elapsed, value = _run_once(
                args.workers, args.requests, sharded=sharded,
            )
            (candidate if sharded else reference).append(elapsed)
            if checksum is None:
                checksum = value
            elif value != checksum:
                raise RuntimeError(f"checksum mismatch: {value} != {checksum}")

    ref_median = statistics.median(reference)
    candidate_median = statistics.median(candidate)
    print(f"reference_median_s={ref_median:.9f}")
    print(f"candidate_median_s={candidate_median:.9f}")
    print(f"candidate_reference_ratio={candidate_median / ref_median:.6f}")
    print(f"checksum={checksum}")


if __name__ == "__main__":
    main()
