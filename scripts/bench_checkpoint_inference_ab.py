#!/usr/bin/env python3
"""Compare forward latency for two model checkpoints.

This is a small A/B wrapper around worker-shaped inference: fixed random
boards (plane count read from the checkpoint's model config), optional
torch.compile, CUDA event timing, and side-by-side
relative speedups. It is useful for checking whether a pruned checkpoint
actually buys enough inference speed to matter.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from chess_anti_engine.uci.model_loader import load_model_from_checkpoint
from chess_anti_engine.utils.amp import inference_autocast


@dataclass(frozen=True)
class BatchTiming:
    batch: int
    wall_median_ms: float
    wall_mean_ms: float
    wall_p95_ms: float
    gpu_median_ms: float
    gpu_mean_ms: float
    gpu_p95_ms: float
    boards_per_s: float


@dataclass(frozen=True)
class CheckpointTiming:
    label: str
    checkpoint: str
    params: int
    batches: tuple[BatchTiming, ...]


def parse_batches(batch: int, batches: str) -> tuple[int, ...]:
    raw = str(batches).strip()
    if not raw:
        values = [int(batch)]
    else:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("no batch sizes were provided")
    if any(value <= 0 for value in values):
        raise ValueError("all batch sizes must be > 0")
    return tuple(values)


def speedup_pct(before_ms: float, after_ms: float) -> float:
    if before_ms <= 0.0 or after_ms <= 0.0:
        return 0.0
    return 100.0 * ((before_ms / after_ms) - 1.0)


def timing_to_json(timing: CheckpointTiming) -> dict[str, Any]:
    return {
        "label": timing.label,
        "checkpoint": timing.checkpoint,
        "params": timing.params,
        "batches": [
            {
                "batch": batch.batch,
                "wall_median_ms": batch.wall_median_ms,
                "wall_mean_ms": batch.wall_mean_ms,
                "wall_p95_ms": batch.wall_p95_ms,
                "gpu_median_ms": batch.gpu_median_ms,
                "gpu_mean_ms": batch.gpu_mean_ms,
                "gpu_p95_ms": batch.gpu_p95_ms,
                "boards_per_s": batch.boards_per_s,
            }
            for batch in timing.batches
        ],
    }


def normalize_cuda_device(raw: str) -> torch.device:
    try:
        device = torch.device(str(raw))
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    if device.type != "cuda":
        raise ValueError("bench_checkpoint_inference_ab.py requires a CUDA device")
    # torch stubs type device.index as int, but a bare "cuda" device carries
    # index None at runtime — the cast keeps this check honest under any stubs.
    if cast("int | None", device.index) is None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def cuda_synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _profile_batch(
    model: torch.nn.Module,
    *,
    batch: int,
    warmup: int,
    iters: int,
    device: torch.device,
    amp_dtype: str,
    planes: int,
) -> BatchTiming:
    rng = np.random.default_rng(12345 + int(batch))
    x_cpu = rng.standard_normal((int(batch), int(planes), 8, 8), dtype=np.float32)
    with torch.cuda.device(device):
        x = torch.from_numpy(x_cpu).to(device, non_blocking=True)

        with torch.inference_mode(), inference_autocast(
            device=str(device),
            enabled=amp_dtype != "off",
            dtype=amp_dtype,
        ):
            for _ in range(int(warmup)):
                _ = model(x)
                cuda_synchronize(device)

        g_start = torch.cuda.Event(enable_timing=True)
        g_end = torch.cuda.Event(enable_timing=True)
        wall_times: list[float] = []
        gpu_times: list[float] = []
        with torch.inference_mode(), inference_autocast(
            device=str(device),
            enabled=amp_dtype != "off",
            dtype=amp_dtype,
        ):
            for _ in range(int(iters)):
                cuda_synchronize(device)
                t0 = time.perf_counter()
                g_start.record()
                _ = model(x)
                g_end.record()
                cuda_synchronize(device)
                wall_times.append((time.perf_counter() - t0) * 1000.0)
                gpu_times.append(float(g_start.elapsed_time(g_end)))

    wall = np.asarray(wall_times, dtype=np.float64)
    gpu = np.asarray(gpu_times, dtype=np.float64)
    wall_median = float(np.median(wall))
    return BatchTiming(
        batch=int(batch),
        wall_median_ms=wall_median,
        wall_mean_ms=float(np.mean(wall)),
        wall_p95_ms=float(np.percentile(wall, 95)),
        gpu_median_ms=float(np.median(gpu)),
        gpu_mean_ms=float(np.mean(gpu)),
        gpu_p95_ms=float(np.percentile(gpu, 95)),
        boards_per_s=(1000.0 * float(batch) / wall_median) if wall_median > 0.0 else 0.0,
    )


def profile_checkpoint(
    *,
    checkpoint: str,
    label: str,
    batches: Sequence[int],
    warmup: int,
    iters: int,
    device: torch.device,
    compile_mode: str,
    amp_dtype: str,
) -> CheckpointTiming:
    print(f"[load] {label}: {checkpoint}")
    with torch.cuda.device(device):
        model = load_model_from_checkpoint(checkpoint, device=str(device)).eval()
        params = sum(param.numel() for param in model.parameters())
        cfg = getattr(model, "cfg", None)
        planes = int(cfg.in_planes) if cfg is not None else 146

        if compile_mode != "off":
            print(f"[compile] {label}: mode={compile_mode}")
            model = cast(torch.nn.Module, torch.compile(model, mode=compile_mode))

        timings: list[BatchTiming] = []
        for batch in batches:
            print(f"[bench] {label}: batch={batch} warmup={warmup} iters={iters}")
            timing = _profile_batch(
                model,
                batch=int(batch),
                warmup=warmup,
                iters=iters,
                device=device,
                amp_dtype=amp_dtype,
                planes=planes,
            )
            timings.append(timing)
            print(
                f"  wall={timing.wall_median_ms:.2f}ms "
                f"gpu={timing.gpu_median_ms:.2f}ms "
                f"boards/s={timing.boards_per_s:.0f}"
            )

    return CheckpointTiming(
        label=label,
        checkpoint=checkpoint,
        params=int(params),
        batches=tuple(timings),
    )


def print_comparison(a: CheckpointTiming, b: CheckpointTiming) -> None:
    if len(a.batches) != len(b.batches):
        raise ValueError("checkpoint timing batch lists differ")

    param_delta = b.params - a.params
    param_delta_pct = 100.0 * param_delta / a.params if a.params > 0 else 0.0
    print()
    print(f"params: {a.label}={a.params:,} {b.label}={b.params:,} ({param_delta_pct:+.2f}%)")
    print(
        f"{'batch':>7} "
        f"{a.label + '_ms':>14} "
        f"{b.label + '_ms':>14} "
        f"{'speedup':>9} "
        f"{a.label + '_bps':>14} "
        f"{b.label + '_bps':>14}"
    )
    for row_a, row_b in zip(a.batches, b.batches):
        if row_a.batch != row_b.batch:
            raise ValueError(f"batch mismatch: {row_a.batch} != {row_b.batch}")
        gain = speedup_pct(row_a.wall_median_ms, row_b.wall_median_ms)
        print(
            f"{row_a.batch:7d} "
            f"{row_a.wall_median_ms:14.2f} "
            f"{row_b.wall_median_ms:14.2f} "
            f"{gain:8.2f}% "
            f"{row_a.boards_per_s:14.0f} "
            f"{row_b.boards_per_s:14.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="baseline checkpoint")
    parser.add_argument("--b", required=True, help="candidate checkpoint")
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--batch", type=int, default=680)
    parser.add_argument("--batches", default="", help="comma-separated batch sizes; overrides --batch")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile-mode",
        default="off",
        choices=("off", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
    )
    parser.add_argument("--amp-dtype", default="auto", choices=("auto", "bf16", "fp16", "off"))
    parser.add_argument("--out", default="", help="optional JSON output path")
    args = parser.parse_args()

    try:
        device = normalize_cuda_device(str(args.device))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.iters <= 0:
        raise SystemExit("--iters must be > 0")
    try:
        batches = parse_batches(args.batch, args.batches)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    timing_a = profile_checkpoint(
        checkpoint=str(args.a),
        label=str(args.label_a),
        batches=batches,
        warmup=int(args.warmup),
        iters=int(args.iters),
        device=device,
        compile_mode=str(args.compile_mode),
        amp_dtype=str(args.amp_dtype),
    )
    timing_b = profile_checkpoint(
        checkpoint=str(args.b),
        label=str(args.label_b),
        batches=batches,
        warmup=int(args.warmup),
        iters=int(args.iters),
        device=device,
        compile_mode=str(args.compile_mode),
        amp_dtype=str(args.amp_dtype),
    )
    print_comparison(timing_a, timing_b)

    if args.out:
        payload = {
            "args": vars(args),
            "a": timing_to_json(timing_a),
            "b": timing_to_json(timing_b),
        }
        out = Path(str(args.out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
