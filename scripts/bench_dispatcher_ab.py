#!/usr/bin/env python3
"""Stage 3 A/B benchmark for ThreadedDispatcher.

Compares three evaluator paths under identical concurrent-producer GPU load:
  - DirectGPUEvaluator (current production: 1 thread, 1 subprocess)
  - ThreadedBatchEvaluator (existing multi-thread path)
  - ThreadedDispatcher (Stage 2 candidate)

Each path runs in its own subprocess so torch.compile + cudagraph TLS is set up
on the same thread that does the timed forwards (mixing them in one process
contaminates cudagraph_trees thread-locals and the runs crash).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import threading
import time
from typing import Any


def _build_model(ckpt_path: str | None, compile_mode: str | None):
    import torch

    from chess_anti_engine.model import ModelConfig, build_model, load_state_dict_tolerant

    cfg = ModelConfig(
        kind="transformer", embed_dim=384, num_layers=9, num_heads=12,
        ffn_mult=1.5, use_smolgen=True, use_nla=False,
    )
    model = build_model(cfg).cuda().eval()
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=True)
        load_state_dict_tolerant(model, ckpt.get("model", ckpt))
    if compile_mode:
        model = torch.compile(model, mode=compile_mode)
    return model


def _producer_loop(evaluator: Any, encoded, stop_at: float, counter: list[int], idx: int) -> None:
    n = 0
    while time.perf_counter() < stop_at:
        evaluator.evaluate_encoded(encoded)
        n += 1
    counter[idx] = n


def _run_in_subprocess(
    path: str,
    n_threads: int,
    batch_size: int,
    duration_s: float,
    warmup_s: float,
    ckpt: str | None,
    compile_mode: str | None,
    result_q: mp.Queue,
) -> None:
    """Build the named evaluator, run warmup + timed window, push results."""
    import numpy as np
    import torch

    from chess_anti_engine.inference import DirectGPUEvaluator, ThreadedBatchEvaluator
    from chess_anti_engine.inference_threaded import ThreadedDispatcher

    try:
        model: Any = _build_model(ckpt, compile_mode)

        if path == "DirectGPU":
            evaluator: Any = DirectGPUEvaluator(model, device="cuda", max_batch=4096)
            shutdown = lambda: None  # noqa: E731
        elif path == "ThreadedBatchEvaluator":
            ev = ThreadedBatchEvaluator(model, device="cuda", max_batch=4096, min_batch=256)
            evaluator, shutdown = ev, ev.shutdown
        elif path == "ThreadedDispatcher":
            ev = ThreadedDispatcher(model, device="cuda", max_batch=4096, batch_wait_ms=1.0)
            evaluator, shutdown = ev, ev.shutdown
        else:
            raise ValueError(f"unknown path: {path}")

        rng = np.random.default_rng(0)
        enc = rng.standard_normal((batch_size, 146, 8, 8), dtype=np.float32)

        # Warmup triggers compile + cudagraph capture before timing.
        warmup_stop = time.perf_counter() + warmup_s
        counts = [0] * n_threads
        threads = [
            threading.Thread(target=_producer_loop, args=(evaluator, enc, warmup_stop, counts, i))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        counters = torch._dynamo.utils.counters
        pre_frames_ok = int(counters.get("frames", {}).get("ok", 0))
        pre_skips = int(counters.get("inductor", {}).get("cudagraph_skips", 0))

        t0 = time.perf_counter()
        stop_at = t0 + duration_s
        counts = [0] * n_threads
        threads = [
            threading.Thread(target=_producer_loop, args=(evaluator, enc, stop_at, counts, i))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - t0
        total_calls = sum(counts)
        total_positions = total_calls * batch_size

        post_frames_ok = int(counters.get("frames", {}).get("ok", 0))
        post_skips = int(counters.get("inductor", {}).get("cudagraph_skips", 0))

        out: dict[str, Any] = {
            "name": path,
            "n_threads": n_threads,
            "batch_size": batch_size,
            "elapsed_s": elapsed,
            "total_calls": total_calls,
            "positions_per_sec": total_positions / elapsed,
            "calls_per_sec": total_calls / elapsed,
            "frames_ok_delta": post_frames_ok - pre_frames_ok,
            "cudagraph_skips_delta": post_skips - pre_skips,
        }
        if isinstance(evaluator, ThreadedDispatcher):
            s = evaluator.stats
            out["dispatcher_avg_batch"] = s["avg_batch_size"]
            out["dispatcher_avg_forward_ms"] = s["avg_forward_ms"]
            out["dispatcher_full_drains"] = s["lifetime_full_drains"]
        shutdown()
        result_q.put(out)
    except Exception as exc:  # noqa: BLE001
        import traceback
        result_q.put({"name": path, "error": str(exc), "traceback": traceback.format_exc()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--compile-mode", type=str, default="reduce-overhead")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=44,
                    help="Per-thread submission batch (≈ leaves per gumbel step).")
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--warmup-s", type=float, default=10.0)
    ap.add_argument("--out", type=str, default="docs/threaded_dispatcher_results.json")
    args = ap.parse_args()

    mp.set_start_method("spawn", force=True)
    compile_mode = args.compile_mode or None

    paths = [
        # name, n_threads, per-call batch
        ("DirectGPU", 1, args.batch_size * args.threads),
        ("ThreadedBatchEvaluator", args.threads, args.batch_size),
        ("ThreadedDispatcher", args.threads, args.batch_size),
    ]
    results: list[dict[str, Any]] = []
    for name, n_threads, batch in paths:
        print(f"\n=== running {name} (threads={n_threads}, batch={batch}, compile={compile_mode}) ===", flush=True)
        q: mp.Queue = mp.Queue()
        p = mp.Process(target=_run_in_subprocess, args=(
            name, n_threads, batch, args.duration_s, args.warmup_s,
            args.ckpt, compile_mode, q,
        ))
        p.start()
        p.join(timeout=600)
        if p.is_alive():
            p.terminate()
            p.join()
            results.append({"name": name, "error": "subprocess timed out"})
            continue
        try:
            r = q.get(timeout=5)
        except Exception as exc:  # noqa: BLE001
            r = {"name": name, "error": f"no result from subprocess: {exc}"}
        if "error" in r:
            print(f"  FAILED: {r['error']}")
            if "traceback" in r:
                print(r["traceback"])
        else:
            print(f"  pos/s={r['positions_per_sec']:.0f} calls/s={r['calls_per_sec']:.1f} "
                  f"frames_ok={r['frames_ok_delta']} cudagraph_skips={r['cudagraph_skips_delta']}")
        results.append(r)

    print()
    print(f"{'path':<24} {'pos/s':>10} {'calls/s':>10} {'frames_ok':>10} {'skips':>8} {'avg_batch':>10} {'fwd_ms':>8}")
    for r in results:
        if "error" in r:
            print(f"{r['name']:<24} ERROR: {r['error']}")
            continue
        avg_batch = r.get("dispatcher_avg_batch") or r["batch_size"] * r["n_threads"]
        fwd_ms = r.get("dispatcher_avg_forward_ms", float("nan"))
        print(
            f"{r['name']:<24} {r['positions_per_sec']:>10.0f} {r['calls_per_sec']:>10.1f} "
            f"{r['frames_ok_delta']:>10} {r['cudagraph_skips_delta']:>8} "
            f"{avg_batch:>10.1f} {fwd_ms:>8.2f}"
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
