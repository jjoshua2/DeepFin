#!/usr/bin/env python3
"""Measure idle wakeups while many selfplay states await Stockfish."""

from __future__ import annotations

import argparse
import statistics
import threading
import time

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait


def _run(*, threads: int, delay_s: float, background_wait_s: float) -> tuple[int, float]:
    future: Future[None] = Future()
    barrier = threading.Barrier(threads + 1)

    def _worker(index: int) -> tuple[int, float]:
        timeout_s = 0.05 if index == 0 else background_wait_s
        polls = 0
        barrier.wait()
        while not future.done():
            wait((future,), timeout=timeout_s, return_when=FIRST_COMPLETED)
            polls += 1
        return polls, time.perf_counter()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        workers = [executor.submit(_worker, index) for index in range(threads)]
        barrier.wait()
        time.sleep(delay_s)
        completed_at = time.perf_counter()
        future.set_result(None)
        results = [worker.result() for worker in workers]
    return sum(polls for polls, _ in results), max(done for _, done in results) - completed_at


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()

    reference_polls: list[int] = []
    candidate_polls: list[int] = []
    reference_latency: list[float] = []
    candidate_latency: list[float] = []
    for round_idx in range(args.rounds):
        order = (0.05, 0.25) if round_idx % 2 == 0 else (0.25, 0.05)
        for background_wait_s in order:
            polls, latency = _run(
                threads=args.threads,
                delay_s=args.delay,
                background_wait_s=background_wait_s,
            )
            if background_wait_s == 0.05:
                reference_polls.append(polls)
                reference_latency.append(latency)
            else:
                candidate_polls.append(polls)
                candidate_latency.append(latency)

    ref_polls = statistics.median(reference_polls)
    cand_polls = statistics.median(candidate_polls)
    ref_latency = statistics.median(reference_latency)
    cand_latency = statistics.median(candidate_latency)
    print(f"reference_wakeups={ref_polls:.0f}")
    print(f"candidate_wakeups={cand_polls:.0f}")
    print(f"candidate_reference_wakeup_ratio={cand_polls / ref_polls:.6f}")
    print(f"reference_completion_latency_s={ref_latency:.9f}")
    print(f"candidate_completion_latency_s={cand_latency:.9f}")
    print(f"completion_latency_delta_s={cand_latency - ref_latency:.9f}")


if __name__ == "__main__":
    main()
