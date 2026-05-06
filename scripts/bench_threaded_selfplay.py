#!/usr/bin/env python3
"""Benchmark threaded selfplay throughput with different thread counts.

Each config runs in a separate subprocess to avoid CUDA graph contamination.
Uses target_games for replenishment (matching production behavior).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from typing import cast


def _run_config(
    n_threads: int,
    sf_workers: int,
    batch_size: int,
    mcts_sims: int,
    sf_nodes: int,
    compile_model: bool,
    stockfish_path: str,
    bootstrap_path: str,
    result_dict: dict,
) -> None:
    """Run one config in a subprocess (clean CUDA context)."""
    import numpy as np
    import torch

    from chess_anti_engine.inference import DirectGPUEvaluator, ThreadedBatchEvaluator
    from chess_anti_engine.model import (
        ModelConfig,
        build_model,
        load_state_dict_tolerant,
    )
    from chess_anti_engine.selfplay import play_batch
    from chess_anti_engine.selfplay.config import (
        GameConfig,
        OpponentConfig,
        SearchConfig,
        TemperatureConfig,
    )
    from chess_anti_engine.selfplay.opening import OpeningConfig
    from chess_anti_engine.stockfish import StockfishPool

    cfg = ModelConfig(
        kind="transformer", embed_dim=384, num_layers=9, num_heads=12,
        ffn_mult=1.5, use_smolgen=True, use_nla=False,
    )
    model = build_model(cfg).cuda().eval()
    if bootstrap_path:
        ckpt = torch.load(bootstrap_path, map_location="cuda", weights_only=True)
        load_state_dict_tolerant(model, ckpt.get("model", ckpt))
    if compile_model:
        model = cast(torch.nn.Module, torch.compile(model, mode="reduce-overhead"))

    if n_threads > 1:
        evaluator = ThreadedBatchEvaluator(model, device="cuda", max_batch=4096)
    else:
        evaluator = DirectGPUEvaluator(model, device="cuda", max_batch=4096)

    sf = StockfishPool(path=stockfish_path, num_workers=sf_workers, nodes=sf_nodes, multipv=1)
    rng = np.random.default_rng(42)

    opponent = OpponentConfig()
    temp = TemperatureConfig()
    search = SearchConfig(simulations=mcts_sims, fast_simulations=mcts_sims // 4)
    opening = OpeningConfig()
    game = GameConfig(max_plies=200)

    # Per-thread: split batch across threads (matches production worker.py)
    thread_batch = max(1, batch_size // max(1, n_threads))

    # Warmup (small batch, no replenishment)
    print(f"  Warming up (threads={n_threads}, sf={sf_workers}, "
          f"batch={thread_batch}/thread)...", flush=True)
    if n_threads > 1:
        from concurrent.futures import ThreadPoolExecutor

        def _run_warmup(tid):
            return play_batch(
                None, device="cuda", rng=np.random.default_rng(tid),
                stockfish=sf, evaluator=evaluator,
                games=min(thread_batch, 4),
                opponent=opponent, temp=temp, search=search, opening=opening, game=game,
            )

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(lambda i: _run_warmup(i), range(n_threads)))
    else:
        play_batch(
            None, device="cuda", rng=rng,
            stockfish=sf, evaluator=evaluator, games=min(batch_size, 4),
            opponent=opponent, temp=temp, search=search, opening=opening, game=game,
        )

    # Benchmark — no replenishment, matches production worker.py
    print(f"  Running {batch_size} games (batch={batch_size})...", flush=True)
    t0 = time.perf_counter()

    if n_threads > 1:
        from concurrent.futures import ThreadPoolExecutor

        seeds = [int(rng.integers(2**63)) for _ in range(n_threads)]

        def _run_bench(tid):
            return play_batch(
                None, device="cuda", rng=np.random.default_rng(seeds[tid]),
                stockfish=sf, evaluator=evaluator,
                games=thread_batch,
                opponent=opponent, temp=temp, search=search, opening=opening, game=game,
            )

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            results = list(pool.map(lambda i: _run_bench(i), range(n_threads)))
        total_positions = sum(stats.positions for _, stats in results)
        total_games = sum(stats.games for _, stats in results)
    else:
        _, stats = play_batch(
            None, device="cuda", rng=rng,
            stockfish=sf, evaluator=evaluator,
            games=batch_size,
            opponent=opponent, temp=temp, search=search, opening=opening, game=game,
        )
        total_positions = stats.positions
        total_games = stats.games

    elapsed = time.perf_counter() - t0

    shutdown = getattr(evaluator, "shutdown", None)
    if callable(shutdown):
        shutdown()
    sf.close()

    free_vram, total_vram = torch.cuda.mem_get_info()

    result_dict["threads"] = n_threads
    result_dict["sf_workers"] = sf_workers
    result_dict["games"] = total_games
    result_dict["positions"] = total_positions
    result_dict["elapsed"] = elapsed
    result_dict["games_per_sec"] = total_games / elapsed
    result_dict["pos_per_sec"] = total_positions / elapsed
    result_dict["vram_used_gb"] = (total_vram - free_vram) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=256,
                    help="Concurrent games (split across threads)")
    ap.add_argument("--mcts-sims", type=int, default=128)
    ap.add_argument("--sf-nodes", type=int, default=5000)
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--stockfish-path", default="stockfish")
    ap.add_argument("--bootstrap", default="")
    ap.add_argument(
        "--configs", type=str, default="1:8,4:8,8:8,16:16",
        help="Comma-separated thread:sf_workers pairs to test"
    )
    args = ap.parse_args()

    configs = []
    for spec in args.configs.split(","):
        parts = spec.strip().split(":")
        threads = int(parts[0])
        sf = int(parts[1]) if len(parts) > 1 else threads
        configs.append((threads, sf))

    print(f"Benchmark: batch={args.batch_size} games, "
          f"{args.mcts_sims} sims, {args.sf_nodes} SF nodes, compile={args.compile}")
    print("=" * 80)

    mp.set_start_method("spawn", force=True)

    results = []
    for threads, sf_workers in configs:
        print(f"\n--- threads={threads}, sf_workers={sf_workers} ---")
        mgr = mp.Manager()
        result_dict = mgr.dict()
        p = mp.Process(
            target=_run_config,
            args=(threads, sf_workers, args.batch_size,
                  args.mcts_sims, args.sf_nodes, args.compile,
                  args.stockfish_path, args.bootstrap, result_dict),
        )
        p.start()
        p.join(timeout=600)
        if p.exitcode != 0:
            print(f"  => FAILED (exit code {p.exitcode})")
            continue
        r = dict(result_dict)
        results.append(r)
        print(
            f"  => {r['games_per_sec']:.2f} games/s, "
            f"{r['pos_per_sec']:.0f} pos/s, "
            f"{r['elapsed']:.1f}s, "
            f"VRAM={r['vram_used_gb']:.1f}GB"
        )

    if results:
        print("\n" + "=" * 80)
        print(f"{'threads':>7} {'sf_wk':>5} {'games/s':>8} {'pos/s':>8} {'time':>6} {'VRAM':>6}")
        print("-" * 50)
        for r in results:
            print(
                f"{r['threads']:>7} {r['sf_workers']:>5} "
                f"{r['games_per_sec']:>8.2f} {r['pos_per_sec']:>8.0f} "
                f"{r['elapsed']:>6.1f} {r['vram_used_gb']:>5.1f}G"
            )
        best = max(results, key=lambda r: r["pos_per_sec"])
        print(f"\nBest: threads={best['threads']}, sf_workers={best['sf_workers']} "
              f"({best['pos_per_sec']:.0f} pos/s)")


if __name__ == "__main__":
    main()
