#!/usr/bin/env python3
"""Diagnostic profile of worker-style compiled inference.

Mirrors the worker's inference path (max-autotune compile, batch ~700-positions)
on a real checkpoint, then reports:

  * Graph-break count from torch._dynamo counters (zero = full graph captured)
  * Wall-clock per forward call
  * GPU compute fraction (CUDA event timing of the actual kernel work)
  * Theoretical compute given model FLOPs and 5090 peak — useful headroom signal

Run while training is up (you'll fight for GPU briefly), or stop training first
for clean numbers. Usage:

    PYTHONPATH=. python3 scripts/profile_worker_inference.py \\
        --checkpoint runs/.../best/best_model.pt \\
        --batch 700 --warmup 5 --iters 50

Environment:
    TORCH_LOGS=graph_breaks,recompiles  ← run with this to log every graph break
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, cast

import numpy as np
import torch

from chess_anti_engine.uci.model_loader import load_model_from_checkpoint
from chess_anti_engine.utils.amp import inference_autocast
from chess_anti_engine.model import infer_input_planes


def _parse_batches(batch: int, batches: str) -> list[int]:
    if not str(batches).strip():
        return [int(batch)]
    out = [int(x) for x in str(batches).split(",") if x.strip()]
    if not out:
        raise SystemExit("--batches did not contain any batch sizes")
    if any(x <= 0 for x in out):
        raise SystemExit("all batch sizes must be > 0")
    return out


def _profile_batch(
    compiled: torch.nn.Module,
    model: torch.nn.Module,
    *,
    batch: int,
    warmup: int,
    iters: int,
    device: str,
    use_amp: bool,
    amp_dtype: str,
    input_planes: int,
) -> dict[str, Any]:
    # Match the checkpoint's declared encoding. Production v2_threats uses
    # 175 planes; older v1 checkpoints use 146.
    rng = np.random.default_rng(42 + batch)
    x_cpu = rng.standard_normal((batch, input_planes, 8, 8), dtype=np.float32)
    x = torch.from_numpy(x_cpu).to(device, non_blocking=True)

    print(f"[profile] warmup batch={batch} ({warmup} iters)")
    t0 = time.perf_counter()
    with torch.inference_mode(), inference_autocast(
        device=device, enabled=use_amp, dtype=amp_dtype,
    ):
        for _ in range(warmup):
            _ = compiled(x)
            torch.cuda.synchronize()
    warmup_secs = time.perf_counter() - t0
    print(f"[profile] warmup batch={batch} done in {warmup_secs:.1f}s")

    print(f"[profile] timed run batch={batch} ({iters} iters)")
    g_start = torch.cuda.Event(enable_timing=True)
    g_end = torch.cuda.Event(enable_timing=True)
    wall_times: list[float] = []
    gpu_times: list[float] = []
    with torch.inference_mode(), inference_autocast(
        device=device, enabled=use_amp, dtype=amp_dtype,
    ):
        for _ in range(iters):
            torch.cuda.synchronize()
            wt0 = time.perf_counter()
            g_start.record()
            _ = compiled(x)
            g_end.record()
            torch.cuda.synchronize()
            gpu_times.append(g_start.elapsed_time(g_end))  # ms
            wall_times.append((time.perf_counter() - wt0) * 1000.0)

    wall_arr = np.array(wall_times)
    gpu_arr = np.array(gpu_times)

    # Estimate theoretical compute given the model's ~15M params and 64 spatial
    # positions per board. Each linear layer is roughly 2 x params x tokens flops.
    n_params = sum(p.numel() for p in model.parameters())
    n_tokens = batch * 64  # 8x8 spatial flattened
    rough_flops = 2.0 * n_params * n_tokens  # forward only, dense path
    peak_tflops = 100.0
    theoretical_ms = (rough_flops / (peak_tflops * 1e12)) * 1000.0

    median_wall = float(np.median(wall_arr))
    median_gpu = float(np.median(gpu_arr))
    p95_wall = float(np.percentile(wall_arr, 95))
    p95_gpu = float(np.percentile(gpu_arr, 95))
    mean_wall = float(wall_arr.mean())
    mean_gpu = float(gpu_arr.mean())
    boards_per_s = 1000.0 * batch / median_wall if median_wall > 0 else 0.0
    realized_peak_pct = 100.0 * theoretical_ms / median_gpu if median_gpu > 0 else 0.0
    wall_overhead_pct = (
        100.0 * (median_wall - median_gpu) / median_wall
        if median_wall > 0 else 0.0
    )

    print()
    print(f"=== batch={batch}, params={n_params/1e6:.1f}M ===")
    print(f"  wall   median={median_wall:6.2f} ms  p95={p95_wall:6.2f}  mean={mean_wall:6.2f}")
    print(f"  gpu    median={median_gpu:6.2f} ms  p95={p95_gpu:6.2f}  mean={mean_gpu:6.2f}")
    print(f"  boards/s median-wall: {boards_per_s:,.0f}")
    print(f"  theoretical-compute @ 100 TFLOPs bf16: {theoretical_ms:.2f} ms")
    if median_gpu > 0:
        print(f"  realized SM throughput   ~= {realized_peak_pct:5.1f}% of peak (gpu_time vs flops)")
        print(f"  wall-launch overhead     ~= {wall_overhead_pct:5.1f}% of wall (wall - gpu_time)")

    return {
        "batch": batch,
        "warmup_s": warmup_secs,
        "wall_median_ms": median_wall,
        "wall_p95_ms": p95_wall,
        "wall_mean_ms": mean_wall,
        "gpu_median_ms": median_gpu,
        "gpu_p95_ms": p95_gpu,
        "gpu_mean_ms": mean_gpu,
        "boards_per_s_median_wall": boards_per_s,
        "theoretical_ms_at_100tflops": theoretical_ms,
        "realized_peak_pct": realized_peak_pct,
        "wall_overhead_pct": wall_overhead_pct,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch", type=int, default=700)
    p.add_argument(
        "--batches",
        default="",
        help="Comma-separated batch sizes to sweep; overrides --batch.",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--mode", default="max-autotune", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs", "off"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp-dtype", default="auto", choices=["auto", "bf16", "fp16", "off"])
    p.add_argument("--out", default="", help="Optional JSON output path.")
    args = p.parse_args()

    if not str(args.device).startswith("cuda"):
        raise SystemExit("profile_worker_inference.py requires a CUDA device")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    batches = _parse_batches(args.batch, args.batches)
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.iters <= 0:
        raise SystemExit("--iters must be > 0")

    print(f"[profile] loading {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device=args.device).eval()
    input_extra_features = getattr(model, "input_extra_features", None)
    input_planes = infer_input_planes(input_extra_features)
    print(
        f"[profile] input encoding={input_extra_features!r} "
        f"planes={input_planes}"
    )

    if args.mode != "off":
        print(f"[profile] compiling with mode={args.mode}")
        from torch._dynamo.utils import counters
        counters.clear()
        compiled = cast(torch.nn.Module, torch.compile(model, mode=args.mode))
    else:
        compiled = model

    results = [
        _profile_batch(
            compiled,
            model,
            batch=batch,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
            use_amp=args.amp_dtype != "off",
            amp_dtype=args.amp_dtype,
            input_planes=input_planes,
        )
        for batch in batches
    ]

    # torch.compile is lazy, so counters are meaningful only after warmup and
    # timing have executed the graph. Reading them immediately after wrapping
    # the model incorrectly reports zero captures.
    if args.mode != "off":
        try:
            from torch._dynamo.utils import counters
            captures = counters.get("frames", {}).get("ok", 0)
            recompiles = counters.get("recompiles", {}).get("recompile", 0)
            breaks = sum(counters.get("graph_break", {}).values())
            print(f"[profile] dynamo counters: frames_ok={captures} recompiles={recompiles} graph_breaks={breaks}")
            if breaks > 0:
                print("[profile] WARNING: graph breaks detected — torch.compile is not capturing the full forward")
                print("[profile] Re-run with TORCH_LOGS=graph_breaks env var to see exactly where")
            elif captures == 0:
                print("[profile] WARNING: zero frames captured — running eager")
            else:
                print("[profile] OK: full graph captured, no breaks")
        except Exception as exc:
            print(f"[profile] dynamo counters unavailable: {exc}")
    print()
    print("Read these:")
    print("  * wall ≈ gpu  →  CUDA graphs effective, no host slack to hide")
    print("  * wall >> gpu →  big launch/sync overhead per call (megakernel territory)")
    print("  * SM throughput < 30% → small-batch tensor-core inefficiency, would")
    print("    benefit from fused MHA/FFN Triton kernels OR larger batch")
    print()
    print(f"{'batch':>7} {'wall_ms':>9} {'gpu_ms':>9} {'boards/s':>11} {'peak%':>7} {'over%':>7}")
    for r in results:
        print(
            f"{r['batch']:>7} {r['wall_median_ms']:>9.2f} "
            f"{r['gpu_median_ms']:>9.2f} {r['boards_per_s_median_wall']:>11.0f} "
            f"{r['realized_peak_pct']:>7.1f} {r['wall_overhead_pct']:>7.1f}"
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"\n[profile] wrote {args.out}")


if __name__ == "__main__":
    main()
