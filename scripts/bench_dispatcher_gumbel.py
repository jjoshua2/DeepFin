#!/usr/bin/env python3
"""A/B benchmark using real gumbel MCTS on N games at once.

Calls run_gumbel_root_many_c directly with a fixed game count + sim budget,
under each evaluator path. This is the production inference shape (small
batches per gumbel-step forward, lots of CPU between forwards), so it tells
us whether ThreadedDispatcher's cross-thread batching pays off where the
synthetic bench (constant tight-loop submissions) said it didn't.

Each path runs in its own subprocess so torch.compile + cudagraph TLS lives
on the same thread that does the timed forwards.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import threading
import time
from typing import Any


def _build_model(compile_mode: str | None):
    import torch

    from chess_anti_engine.model import ModelConfig, build_model

    cfg = ModelConfig(
        kind="transformer", embed_dim=384, num_layers=9, num_heads=12,
        ffn_mult=1.5, use_smolgen=True, use_nla=False,
    )
    model = build_model(cfg).cuda().eval()
    if compile_mode:
        model = torch.compile(model, mode=compile_mode)
    return model


def _starting_boards(n: int):
    """N random-ish opening positions for timing."""
    import chess

    boards = []
    for seed in range(n):
        rng = random.Random(seed)
        b = chess.Board()
        for _ in range(4):
            moves = list(b.legal_moves)
            if not moves:
                break
            mv = moves[rng.randrange(len(moves))]
            b.push(mv)
        boards.append(b)
    return boards


def _run_one_thread(evaluator: Any, boards: list, simulations: int, topk: int) -> int:
    import numpy as np
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c

    rng = np.random.default_rng(0)
    cfg = GumbelConfig(simulations=simulations, topk=topk)
    _, actions, _, _, _, _ = run_gumbel_root_many_c(
        None, boards, device="cuda", rng=rng, cfg=cfg, evaluator=evaluator,
    )
    return len(actions)


def _run_in_subprocess(
    path: str,
    n_threads: int,
    games_per_thread: int,
    simulations: int,
    topk: int,
    compile_mode: str | None,
    dispatcher_batch_wait_ms: float,
    dispatcher_target_batch: int,
    iters: int,
    warmup_iters: int,
    result_q: mp.Queue,
) -> None:
    try:
        import torch

        from chess_anti_engine.inference import DirectGPUEvaluator, ThreadedBatchEvaluator
        from chess_anti_engine.inference_threaded import ThreadedDispatcher

        # ThreadedDispatcher compiles on its own thread (cudagraph TLS lives
        # on the thread that does the forward). All other paths compile on
        # the main thread of this subprocess.
        compile_on_main = compile_mode if path != "ThreadedDispatcher" else None
        model: Any = _build_model(compile_on_main)

        if path == "DirectGPU":
            evaluator: Any = DirectGPUEvaluator(model, device="cuda", max_batch=4096)
            shutdown = lambda: None  # noqa: E731
        elif path == "ThreadedBatchEvaluator":
            ev = ThreadedBatchEvaluator(model, device="cuda", max_batch=4096, min_batch=64)
            evaluator, shutdown = ev, ev.shutdown
        elif path == "ThreadedDispatcher":
            ev = ThreadedDispatcher(
                model, device="cuda", max_batch=4096, batch_wait_ms=dispatcher_batch_wait_ms,
                target_batch=dispatcher_target_batch,
                compile_mode=compile_mode,
            )
            evaluator, shutdown = ev, ev.shutdown
        else:
            raise ValueError(f"unknown path: {path}")

        def run_one_iter() -> int:
            boards_per_thread = [_starting_boards(games_per_thread) for _ in range(n_threads)]
            # Single-thread runs on the main thread directly so torch.compile +
            # cudagraph_trees TLS lives where the forwards happen. Spawning even
            # one worker thread tripped CUDAGraph TLS asserts on cold compile.
            if n_threads == 1:
                return _run_one_thread(evaluator, boards_per_thread[0], simulations, topk)
            results = [0] * n_threads

            def worker(i: int) -> None:
                results[i] = _run_one_thread(evaluator, boards_per_thread[i], simulations, topk)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            return sum(results)

        # Warmup (compile + cudagraph capture).
        for _ in range(warmup_iters):
            run_one_iter()

        counters = torch._dynamo.utils.counters
        pre_frames_ok = int(counters.get("frames", {}).get("ok", 0))
        pre_skips = int(counters.get("inductor", {}).get("cudagraph_skips", 0))
        dispatcher_stats0 = (
            dict(evaluator.stats) if isinstance(evaluator, ThreadedDispatcher) else None
        )

        t0 = time.perf_counter()
        total_actions = 0
        for _ in range(iters):
            total_actions += run_one_iter()
        elapsed = time.perf_counter() - t0
        dispatcher_stats1 = (
            dict(evaluator.stats) if isinstance(evaluator, ThreadedDispatcher) else None
        )

        post_frames_ok = int(counters.get("frames", {}).get("ok", 0))
        post_skips = int(counters.get("inductor", {}).get("cudagraph_skips", 0))

        # Each "action" = one gumbel root call = `simulations` forward leaves
        # (approximately — gumbel halves so it's ≤ simulations). Use sims as
        # the position count for nps.
        total_games = n_threads * games_per_thread * iters
        total_positions_approx = total_actions * simulations

        out: dict[str, Any] = {
            "name": path,
            "n_threads": n_threads,
            "games_per_thread": games_per_thread,
            "simulations": simulations,
            "dispatcher_batch_wait_ms": dispatcher_batch_wait_ms if path == "ThreadedDispatcher" else None,
            "dispatcher_target_batch": dispatcher_target_batch if path == "ThreadedDispatcher" else None,
            "iters": iters,
            "elapsed_s": elapsed,
            "iters_per_sec": iters / elapsed,
            "games_per_sec": total_games / elapsed,
            "approx_nps": total_positions_approx / elapsed,
            "frames_ok_delta": post_frames_ok - pre_frames_ok,
            "cudagraph_skips_delta": post_skips - pre_skips,
        }
        if isinstance(evaluator, ThreadedDispatcher):
            s = dispatcher_stats1 or evaluator.stats
            s0 = dispatcher_stats0 or {}
            timed_positions = int(s.get("lifetime_positions", 0)) - int(s0.get("lifetime_positions", 0))
            timed_forward_rows = int(s.get("lifetime_forward_rows", 0)) - int(s0.get("lifetime_forward_rows", 0))
            timed_batches = int(s.get("lifetime_batches", 0)) - int(s0.get("lifetime_batches", 0))
            out["dispatcher_avg_batch"] = s["avg_batch_size"]
            out["dispatcher_avg_requests_per_batch"] = s.get("avg_requests_per_batch", 0.0)
            out["dispatcher_avg_rows_per_request"] = s.get("avg_rows_per_request", 0.0)
            out["dispatcher_avg_queue_depth"] = s.get("avg_queue_depth", 0.0)
            out["dispatcher_queue_depth_max"] = s.get("queue_depth_max", 0)
            out["dispatcher_avg_forward_ms"] = s["avg_forward_ms"]
            out["dispatcher_avg_drain_ms"] = s.get("avg_drain_ms", 0.0)
            out["dispatcher_avg_pack_ms"] = s.get("avg_pack_ms", 0.0)
            out["dispatcher_avg_submit_ms"] = s.get("avg_submit_ms", s["avg_forward_ms"])
            out["dispatcher_avg_wait_ms"] = s.get("avg_wait_ms", 0.0)
            out["dispatcher_avg_scatter_ms"] = s.get("avg_scatter_ms", 0.0)
            out["dispatcher_full_drains"] = s["lifetime_full_drains"]
            out["dispatcher_lifetime_batches"] = s["lifetime_batches"]
            out["timed_dispatcher_batches"] = timed_batches
            out["timed_dispatcher_positions"] = timed_positions
            out["timed_dispatcher_forward_rows"] = timed_forward_rows
            out["timed_dispatcher_leaf_per_s"] = timed_positions / elapsed
            out["timed_dispatcher_forward_rows_per_s"] = timed_forward_rows / elapsed
            out["timed_dispatcher_pad_ratio"] = timed_forward_rows / max(1, timed_positions)
        shutdown()
        result_q.put(out)
    except Exception as exc:
        import traceback
        result_q.put({"name": path, "error": str(exc), "traceback": traceback.format_exc()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-mode", type=str, default="reduce-overhead")
    ap.add_argument("--total-games", type=int, default=400)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument(
        "--thread-counts",
        type=str,
        default="",
        help="Comma-separated threaded producer counts to sweep; overrides --threads for threaded paths.",
    )
    ap.add_argument("--simulations", type=int, default=50)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup-iters", type=int, default=2)
    ap.add_argument(
        "--dispatcher-batch-wait-ms",
        type=float,
        default=1.0,
        help="ThreadedDispatcher batch wait to benchmark.",
    )
    ap.add_argument(
        "--dispatcher-batch-waits-ms",
        type=str,
        default="",
        help="Comma-separated ThreadedDispatcher batch waits to sweep; overrides --dispatcher-batch-wait-ms.",
    )
    ap.add_argument(
        "--dispatcher-target-batch",
        type=int,
        default=0,
        help="ThreadedDispatcher target batch to benchmark; 0 means max-batch.",
    )
    ap.add_argument(
        "--dispatcher-target-batches",
        type=str,
        default="",
        help="Comma-separated ThreadedDispatcher target batches to sweep; overrides --dispatcher-target-batch.",
    )
    ap.add_argument(
        "--paths",
        type=str,
        default="DirectGPU,ThreadedBatchEvaluator,ThreadedDispatcher",
        help="Comma-separated paths to run.",
    )
    ap.add_argument("--out", type=str, default="docs/threaded_dispatcher_gumbel_results.json")
    args = ap.parse_args()

    mp.set_start_method("spawn", force=True)
    compile_mode = args.compile_mode or None

    requested_paths = [p.strip() for p in str(args.paths).split(",") if p.strip()]
    valid_paths = {"DirectGPU", "ThreadedBatchEvaluator", "ThreadedDispatcher"}
    unknown = sorted(set(requested_paths) - valid_paths)
    if unknown:
        raise SystemExit(f"unknown --paths entries: {', '.join(unknown)}")
    thread_counts = (
        [int(x) for x in str(args.thread_counts).split(",") if x.strip()]
        if str(args.thread_counts).strip()
        else [int(args.threads)]
    )
    if any(t <= 0 for t in thread_counts):
        raise SystemExit("--threads/--thread-counts must be positive")
    wait_values = (
        [float(x) for x in str(args.dispatcher_batch_waits_ms).split(",") if x.strip()]
        if str(args.dispatcher_batch_waits_ms).strip()
        else [float(args.dispatcher_batch_wait_ms)]
    )
    target_values = (
        [int(x) for x in str(args.dispatcher_target_batches).split(",") if x.strip()]
        if str(args.dispatcher_target_batches).strip()
        else [int(args.dispatcher_target_batch)]
    )

    paths: list[tuple[str, int, int, float, int]] = []
    if "DirectGPU" in requested_paths:
        # DirectGPU's best case: one producer, one large independent-game batch.
        paths.append(("DirectGPU", 1, int(args.total_games), 0.0, 0))
    for threads in thread_counts:
        games_per_thread = max(1, int(args.total_games) // int(threads))
        if "ThreadedBatchEvaluator" in requested_paths:
            paths.append(("ThreadedBatchEvaluator", int(threads), games_per_thread, 0.0, 0))
        if "ThreadedDispatcher" in requested_paths:
            for wait_ms in wait_values:
                for target_batch in target_values:
                    paths.append((
                        "ThreadedDispatcher", int(threads), games_per_thread,
                        float(wait_ms), int(target_batch),
                    ))
    results: list[dict[str, Any]] = []
    for name, n_threads, games_per_thread, wait_ms, target_batch in paths:
        print(f"\n=== {name} (threads={n_threads}, games/thread={games_per_thread}, "
              f"sims={args.simulations}, compile={compile_mode}, "
              f"wait_ms={wait_ms:g}, target={target_batch}) ===", flush=True)
        q: mp.Queue = mp.Queue()
        p = mp.Process(target=_run_in_subprocess, args=(
            name, n_threads, games_per_thread, args.simulations, args.topk,
            compile_mode, float(wait_ms), int(target_batch), args.iters, args.warmup_iters, q,
        ))
        p.start()
        p.join(timeout=900)
        if p.is_alive():
            p.terminate()
            p.join()
            results.append({"name": name, "error": "subprocess timed out"})
            continue
        try:
            r = q.get(timeout=5)
        except Exception as exc:
            r = {"name": name, "error": f"no result: {exc}"}
        if "error" in r:
            print(f"  FAILED: {r['error']}")
            if "traceback" in r:
                print(r["traceback"])
        else:
            print(f"  iters/s={r['iters_per_sec']:.2f}  games/s={r['games_per_sec']:.0f}  "
                  f"approx_nps={r['approx_nps']:.0f}  frames_ok={r['frames_ok_delta']}  "
                  f"skips={r['cudagraph_skips_delta']}")
            if "dispatcher_avg_batch" in r:
                print(f"  dispatcher: avg_batch={r['dispatcher_avg_batch']:.1f}  "
                      f"leaf/s={r['timed_dispatcher_leaf_per_s']:.0f}  "
                      f"fwd_rows/s={r['timed_dispatcher_forward_rows_per_s']:.0f}  "
                      f"pad={r['timed_dispatcher_pad_ratio']:.2f}  "
                      f"req/batch={r['dispatcher_avg_requests_per_batch']:.1f}  "
                      f"avg_forward_ms={r['dispatcher_avg_forward_ms']:.2f}  "
                      f"batches={r['dispatcher_lifetime_batches']}  "
                      f"full_drains={r['dispatcher_full_drains']}")
        results.append(r)

    print()
    print(
        f"{'path':<24} {'thr':>4} {'wait':>6} {'target':>6} {'iters/s':>8} "
        f"{'games/s':>10} {'nps':>10} {'leaf/s':>10} {'fwd/s':>10} "
        f"{'pad':>5} {'frames_ok':>10} {'skips':>8}"
    )
    for r in results:
        if "error" in r:
            print(f"{r['name']:<24} ERROR: {r['error']}")
            continue
        leaf_s = r.get("timed_dispatcher_leaf_per_s", 0.0)
        fwd_s = r.get("timed_dispatcher_forward_rows_per_s", 0.0)
        pad_ratio = r.get("timed_dispatcher_pad_ratio", 0.0)
        print(
            f"{r['name']:<24} {r['n_threads']:>4} "
            f"{r.get('dispatcher_batch_wait_ms') or ''!s:>6} "
            f"{r.get('dispatcher_target_batch') or ''!s:>6} "
            f"{r['iters_per_sec']:>8.2f} {r['games_per_sec']:>10.0f} "
            f"{r['approx_nps']:>10.0f} {leaf_s:>10.0f} {fwd_s:>10.0f} "
            f"{pad_ratio:>5.2f} {r['frames_ok_delta']:>10} "
            f"{r['cudagraph_skips_delta']:>8}"
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
