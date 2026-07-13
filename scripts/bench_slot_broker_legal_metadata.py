#!/usr/bin/env python3
"""Benchmark compact SlotBroker shared-memory metadata preparation."""
from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import numpy as np


Prepared = tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]


def _reference(metas: list[np.ndarray], rows_per_slot: int) -> Prepared:
    counts_by_slot: list[np.ndarray] = []
    flat_by_slot: list[np.ndarray] = []
    for meta in metas:
        n_legal = int(meta[0])
        counts_by_slot.append(
            np.asarray(meta[1:1 + rows_per_slot], dtype=np.int32).copy()
        )
        flat_by_slot.append(
            np.asarray(
                meta[1 + rows_per_slot:1 + rows_per_slot + n_legal],
                dtype=np.int32,
            ).copy()
        )
    return _combine(counts_by_slot, flat_by_slot, rows_per_slot, singleton_alias=False)


def _candidate(metas: list[np.ndarray], rows_per_slot: int) -> Prepared:
    counts_by_slot: list[np.ndarray] = []
    flat_by_slot: list[np.ndarray] = []
    for meta in metas:
        n_legal = int(meta[0])
        counts_by_slot.append(np.asarray(meta[1:1 + rows_per_slot], dtype=np.int32))
        flat_by_slot.append(np.asarray(
            meta[1 + rows_per_slot:1 + rows_per_slot + n_legal], dtype=np.int32,
        ))
    return _combine(counts_by_slot, flat_by_slot, rows_per_slot, singleton_alias=True)


def _combine(
    counts_by_slot: list[np.ndarray],
    flat_by_slot: list[np.ndarray],
    rows_per_slot: int,
    *,
    singleton_alias: bool,
) -> Prepared:
    rows_parts: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    row_base = 0
    pol_base = 0
    for counts, flat in zip(counts_by_slot, flat_by_slot, strict=True):
        n_legal = int(flat.size)
        offsets.append((pol_base, pol_base + n_legal))
        rows_parts.append(np.repeat(
            np.arange(row_base, row_base + rows_per_slot, dtype=np.int64),
            counts.astype(np.int64, copy=False),
        ))
        row_base += rows_per_slot
        pol_base += n_legal
    if singleton_alias and len(counts_by_slot) == 1:
        counts_all = counts_by_slot[0].astype(np.int64, copy=False)
        flat_all = flat_by_slot[0].astype(np.int64, copy=False)
        rows_all = rows_parts[0]
    else:
        counts_all = np.concatenate(counts_by_slot).astype(np.int64, copy=False)
        flat_all = np.concatenate(flat_by_slot).astype(np.int64, copy=False)
        rows_all = np.concatenate(rows_parts).astype(np.int64, copy=False)
    return counts_all, flat_all, rows_all, tuple(offsets)


def _time_arm(
    function: Callable[[list[np.ndarray], int], Prepared],
    metas: list[np.ndarray],
    rows_per_slot: int,
    iterations: int,
) -> tuple[float, Prepared]:
    start = time.perf_counter()
    result = function(metas, rows_per_slot)
    for _ in range(iterations - 1):
        result = function(metas, rows_per_slot)
    return time.perf_counter() - start, result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-slot", type=int, default=32)
    parser.add_argument("--legal-per", type=int, default=32)
    parser.add_argument("--slots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(20260712)
    for num_slots in args.slots:
        metas: list[np.ndarray] = []
        for _ in range(num_slots):
            counts = rng.integers(
                max(1, args.legal_per - 8), args.legal_per + 9,
                size=args.rows_per_slot, dtype=np.int32,
            )
            flat = rng.integers(0, 4672, size=int(counts.sum()), dtype=np.int32)
            metas.append(np.concatenate((np.array([flat.size], dtype=np.int32), counts, flat)))

        timings: dict[str, list[float]] = {"reference": [], "candidate": []}
        results: dict[str, Prepared] = {}
        arms = (("reference", _reference), ("candidate", _candidate))
        for round_index in range(args.rounds):
            for name, function in (arms if round_index % 2 == 0 else reversed(arms)):
                elapsed, result = _time_arm(
                    function, metas, args.rows_per_slot, args.iterations,
                )
                timings[name].append(elapsed)
                results[name] = result

        for actual, expected in zip(
            results["candidate"][:3], results["reference"][:3], strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        if results["candidate"][3] != results["reference"][3]:
            raise AssertionError("compact offsets differ")
        reference_s = statistics.median(timings["reference"])
        candidate_s = statistics.median(timings["candidate"])
        print(
            f"slots={num_slots} reference_s={reference_s:.6f} "
            f"candidate_s={candidate_s:.6f} speedup={reference_s / candidate_s:.6f}x"
        )


if __name__ == "__main__":
    main()
