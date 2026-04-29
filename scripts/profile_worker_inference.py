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
import time

import numpy as np
import torch

from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch", type=int, default=700)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--mode", default="max-autotune", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs", "off"])
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print(f"[profile] loading {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device=args.device).eval()

    if args.mode != "off":
        print(f"[profile] compiling with mode={args.mode}")
        from torch._dynamo.utils import counters
        counters.clear()
        compiled = torch.compile(model, mode=args.mode)
    else:
        compiled = model

    # Random board-shaped input. 146 planes × 8 × 8 matches our encoding.
    rng = np.random.default_rng(42)
    x_cpu = rng.standard_normal((args.batch, 146, 8, 8), dtype=np.float32)
    x = torch.from_numpy(x_cpu).to(args.device, non_blocking=True)

    print(f"[profile] warmup ({args.warmup} iters) — compile + autotune fires here")
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = compiled(x)
            torch.cuda.synchronize()
    warmup_secs = time.perf_counter() - t0
    print(f"[profile] warmup done in {warmup_secs:.1f}s")

    # Graph break + recompile diagnostics
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

    # Wall + GPU timing
    print(f"[profile] timed run ({args.iters} iters, batch {args.batch})")
    g_start = torch.cuda.Event(enable_timing=True)
    g_end = torch.cuda.Event(enable_timing=True)
    wall_times: list[float] = []
    gpu_times: list[float] = []
    with torch.inference_mode():
        for _ in range(args.iters):
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
    # positions per board. Each linear layer is ~ 2 × params × tokens flops.
    # Rough bf16 peak of RTX 5090 (sm_120, 32 GB): ~100 TFLOPs.
    n_params = sum(p.numel() for p in model.parameters())
    n_tokens = args.batch * 64  # 8x8 spatial flattened
    rough_flops = 2.0 * n_params * n_tokens  # forward only, dense path
    peak_tflops = 100.0
    theoretical_ms = (rough_flops / (peak_tflops * 1e12)) * 1000.0

    print()
    print(f"=== batch={args.batch}, params={n_params/1e6:.1f}M, mode={args.mode} ===")
    print(f"  wall   median={np.median(wall_arr):6.2f} ms  p95={np.percentile(wall_arr, 95):6.2f}  mean={wall_arr.mean():6.2f}")
    print(f"  gpu    median={np.median(gpu_arr):6.2f} ms  p95={np.percentile(gpu_arr, 95):6.2f}  mean={gpu_arr.mean():6.2f}")
    print(f"  theoretical-compute @ 100 TFLOPs bf16: {theoretical_ms:.2f} ms")
    median_gpu = float(np.median(gpu_arr))
    if median_gpu > 0:
        utilization_vs_peak = 100.0 * theoretical_ms / median_gpu
        wall_overhead_pct = 100.0 * (np.median(wall_arr) - median_gpu) / np.median(wall_arr)
        print(f"  realized SM throughput   ≈ {utilization_vs_peak:5.1f}% of peak (gpu_time vs flops)")
        print(f"  wall-launch overhead     ≈ {wall_overhead_pct:5.1f}% of wall (wall - gpu_time)")
    print()
    print("Read these:")
    print("  * wall ≈ gpu  →  CUDA graphs effective, no host slack to hide")
    print("  * wall >> gpu →  big launch/sync overhead per call (megakernel territory)")
    print("  * SM throughput < 30% → small-batch tensor-core inefficiency, would")
    print("    benefit from fused MHA/FFN Triton kernels OR larger batch")


if __name__ == "__main__":
    main()
