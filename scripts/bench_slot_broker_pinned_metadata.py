#!/usr/bin/env python3
"""Benchmark pageable versus reusable-pinned legal metadata H2D copies."""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch


def _pageable(
    flat: np.ndarray, rows: np.ndarray, iterations: int,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    flat_gpu = torch.empty(0, device="cuda", dtype=torch.long)
    rows_gpu = torch.empty(0, device="cuda", dtype=torch.long)
    for _ in range(iterations):
        flat_gpu = torch.as_tensor(flat, dtype=torch.long, device="cuda")
        rows_gpu = torch.as_tensor(rows, dtype=torch.long, device="cuda")
        torch.cuda.synchronize()
    return time.perf_counter() - start, flat_gpu, rows_gpu


def _pinned(
    flat: np.ndarray,
    rows: np.ndarray,
    iterations: int,
    flat_pin: torch.Tensor,
    rows_pin: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    flat_np = flat_pin.numpy(force=True)
    rows_np = rows_pin.numpy(force=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    flat_gpu = torch.empty(0, device="cuda", dtype=torch.long)
    rows_gpu = torch.empty(0, device="cuda", dtype=torch.long)
    for _ in range(iterations):
        flat_np[:] = flat
        rows_np[:] = rows
        flat_gpu = flat_pin.to("cuda", non_blocking=True)
        rows_gpu = rows_pin.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
    return time.perf_counter() - start, flat_gpu, rows_gpu


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legal", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args = _parse_args()
    rng = np.random.default_rng(20260712)
    for size in args.legal:
        flat = rng.integers(0, 4672, size=size, dtype=np.int64)
        rows = rng.integers(0, max(1, size // 24), size=size, dtype=np.int64)
        flat_pin = torch.empty(size, dtype=torch.long, pin_memory=True)
        rows_pin = torch.empty(size, dtype=torch.long, pin_memory=True)
        timings: dict[str, list[float]] = {"pageable": [], "pinned": []}
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for round_index in range(args.rounds):
            arms = ("pageable", "pinned")
            for arm in (arms if round_index % 2 == 0 else reversed(arms)):
                if arm == "pageable":
                    elapsed, flat_gpu, rows_gpu = _pageable(flat, rows, args.iterations)
                else:
                    elapsed, flat_gpu, rows_gpu = _pinned(
                        flat, rows, args.iterations, flat_pin, rows_pin,
                    )
                timings[arm].append(elapsed)
                outputs[arm] = (flat_gpu.cpu(), rows_gpu.cpu())
        for candidate, reference in zip(outputs["pinned"], outputs["pageable"], strict=True):
            torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
        pageable_s = statistics.median(timings["pageable"])
        pinned_s = statistics.median(timings["pinned"])
        print(
            f"legal={size} pageable_s={pageable_s:.6f} pinned_s={pinned_s:.6f} "
            f"ratio={pinned_s / pageable_s:.6f}"
        )


if __name__ == "__main__":
    main()
