#!/usr/bin/env python3
"""Benchmark dense-policy fallback legal gather metadata preparation."""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--rows-per-slot", type=int, default=32)
    parser.add_argument("--legal-per", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args = _parse_args()
    rng = np.random.default_rng(20260712)
    counts_by_slot: list[np.ndarray] = []
    flat_by_slot: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    row_base = 0
    for _ in range(args.slots):
        counts = rng.integers(
            max(1, args.legal_per - 8), args.legal_per + 9,
            size=args.rows_per_slot, dtype=np.int32,
        )
        flat = rng.integers(0, 4672, size=int(counts.sum()), dtype=np.int32)
        counts_by_slot.append(counts)
        flat_by_slot.append(flat)
        row_parts.append(np.repeat(
            np.arange(row_base, row_base + args.rows_per_slot, dtype=np.int64),
            counts.astype(np.int64, copy=False),
        ))
        row_base += args.rows_per_slot
    rows_all = np.concatenate(row_parts).astype(np.int64, copy=False)
    flat_all = np.concatenate(flat_by_slot).astype(np.int64, copy=False)
    rows_pin = torch.empty(rows_all.size, dtype=torch.long, pin_memory=True)
    flat_pin = torch.empty(flat_all.size, dtype=torch.long, pin_memory=True)
    rows_pin_np = rows_pin.numpy(force=True)
    flat_pin_np = flat_pin.numpy(force=True)

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        rebuilt_rows: list[np.ndarray] = []
        rebuilt_cols: list[np.ndarray] = []
        base = 0
        for counts, flat in zip(counts_by_slot, flat_by_slot, strict=True):
            rebuilt_rows.append(np.repeat(
                np.arange(base, base + args.rows_per_slot, dtype=np.int64),
                counts.astype(np.int64, copy=False),
            ))
            rebuilt_cols.append(flat.astype(np.int64, copy=False))
            base += args.rows_per_slot
        rows_gpu = torch.as_tensor(np.concatenate(rebuilt_rows), device="cuda")
        cols_gpu = torch.as_tensor(np.concatenate(rebuilt_cols), device="cuda")
        return rows_gpu, cols_gpu

    def candidate() -> tuple[torch.Tensor, torch.Tensor]:
        rows_pin_np[:] = rows_all
        flat_pin_np[:] = flat_all
        return (
            rows_pin.to("cuda", non_blocking=True),
            flat_pin.to("cuda", non_blocking=True),
        )

    timings: dict[str, list[float]] = {"reference": [], "candidate": []}
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for round_index in range(args.rounds):
        arms = (("reference", reference), ("candidate", candidate))
        for name, function in (arms if round_index % 2 == 0 else reversed(arms)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = function()
            for _ in range(args.iterations - 1):
                torch.cuda.synchronize()
                result = function()
            torch.cuda.synchronize()
            timings[name].append(time.perf_counter() - start)
            outputs[name] = (result[0].cpu(), result[1].cpu())
    for candidate_out, reference_out in zip(
        outputs["candidate"], outputs["reference"], strict=True,
    ):
        torch.testing.assert_close(candidate_out, reference_out, rtol=0, atol=0)
    reference_s = statistics.median(timings["reference"])
    candidate_s = statistics.median(timings["candidate"])
    print(f"reference_s={reference_s:.6f}")
    print(f"candidate_s={candidate_s:.6f}")
    print(f"ratio={candidate_s / reference_s:.6f}")
    print(f"legal={flat_all.size}")


if __name__ == "__main__":
    main()
