#!/usr/bin/env python3
"""Measure useful-work overlap with an outstanding finalization label."""

from __future__ import annotations

import argparse
import statistics
import time

from concurrent.futures import ThreadPoolExecutor


def _run_once(
    *, label_delay_s: float, work_items: int, work_s: float, overlap: bool,
) -> tuple[float, int]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        label = executor.submit(_finish_label, label_delay_s)
        started = time.perf_counter()
        checksum = 0
        if not overlap:
            checksum += label.result()
        for item in range(work_items):
            time.sleep(work_s)
            checksum += item + 1
        if overlap:
            checksum += label.result()
        return time.perf_counter() - started, checksum


def _finish_label(delay_s: float) -> int:
    time.sleep(delay_s)
    return 17


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-delay-ms", type=float, default=20.0)
    parser.add_argument("--work-items", type=int, default=20)
    parser.add_argument("--work-ms", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    if (
        args.label_delay_ms < 0
        or args.work_items < 1
        or args.work_ms < 0
        or args.rounds < 1
    ):
        parser.error("delays must be nonnegative; work-items and rounds must be positive")

    reference: list[float] = []
    candidate: list[float] = []
    checksum: int | None = None
    for round_idx in range(args.rounds):
        order = (False, True) if round_idx % 2 == 0 else (True, False)
        for overlap in order:
            elapsed, value = _run_once(
                label_delay_s=args.label_delay_ms / 1000.0,
                work_items=args.work_items,
                work_s=args.work_ms / 1000.0,
                overlap=overlap,
            )
            (candidate if overlap else reference).append(elapsed)
            if checksum is None:
                checksum = value
            elif checksum != value:
                raise RuntimeError(f"checksum mismatch: {value} != {checksum}")

    reference_median = statistics.median(reference)
    candidate_median = statistics.median(candidate)
    print(f"reference_median_s={reference_median:.9f}")
    print(f"candidate_median_s={candidate_median:.9f}")
    print(f"candidate_reference_ratio={candidate_median / reference_median:.6f}")
    print(f"checksum={checksum}")


if __name__ == "__main__":
    main()
