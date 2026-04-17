#!/usr/bin/env python3
"""Benchmark multi-worker × multi-thread selfplay throughput.

Each worker is a separate subprocess (clean CUDA context).
Measures total throughput across all workers.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time


def _worker_fn(
    worker_id: int,
    n_threads: int,
    sf_workers: int,
    games: int,
    mcts_sims: int,
    sf_nodes: int,
    compile_model: bool,
    stockfish_path: str,
    bootstrap_path: str,
    result_dict: dict,
    barrier,
) -> None:
    """One worker subprocess: build model, warmup, then benchmark."""
    import numpy as np
    import torch

    from chess_anti_engine.model import ModelConfig, build_model, load_state_dict_tolerant
    from chess_anti_engine.inference import DirectGPUEvaluator, ThreadedBatchEvaluator
    from chess_anti_engine.selfplay import play_batch
    from chess_anti_engine.selfplay.config import (
        OpponentConfig, TemperatureConfig, SearchConfig, GameConfig,
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
        model = torch.compile(model, mode="reduce-overhead")

    if n_threads > 1:
        evaluator = ThreadedBatchEvaluator(model, device="cuda", max_batch=4096)
    else:
        evaluator = DirectGPUEvaluator(model, device="cuda", max_batch=512)

    sf = StockfishPool(path=stockfish_path, num_workers=sf_workers, nodes=sf_nodes, multipv=1)
    rng = np.random.default_rng(42 + worker_id)

    opponent = OpponentConfig()
    temp = TemperatureConfig()
    search = SearchConfig(simulations=mcts_sims, fast_simulations=mcts_sims // 4)
    opening = OpeningConfig()
    game = GameConfig(max_plies=200)

    # Warmup
    warmup_games = min(max(n_threads, 2), 8)
    if n_threads > 1:
        from concurrent.futures import ThreadPoolExecutor

        base, rem = divmod(warmup_games, n_threads)
        tg = [base + (1 if i < rem else 0) for i in range(n_threads)]
        tg = [g for g in tg if g > 0]

        def _run(tid):
            return play_batch(
                None, device="cuda", rng=np.random.default_rng(tid + worker_id * 100),
                stockfish=sf, evaluator=evaluator, games=tg[tid],
                opponent=opponent, temp=temp, search=search, opening=opening, game=game,
            )

        with ThreadPoolExecutor(max_workers=len(tg)) as pool:
            list(pool.map(lambda i: _run(i), range(len(tg))))
    else:
        play_batch(
            None, device="cuda", rng=rng,
            stockfish=sf, evaluator=evaluator, games=warmup_games,
            opponent=opponent, temp=temp, search=search, opening=opening, game=game,
        )

    # Synchronize all workers before benchmark
    barrier.wait()

    # Benchmark
    t0 = time.perf_counter()

    if n_threads > 1:
        from concurrent.futures import ThreadPoolExecutor

        base, rem = divmod(games, n_threads)
        tg = [base + (1 if i < rem else 0) for i in range(n_threads)]
        tg = [g for g in tg if g > 0]
        seeds = [int(rng.integers(2**63)) for _ in range(len(tg))]

        def _run_bench(tid):
            return play_batch(
                None, device="cuda", rng=np.random.default_rng(seeds[tid]),
                stockfish=sf, evaluator=evaluator, games=tg[tid],
                opponent=opponent, temp=temp, search=search, opening=opening, game=game,
            )

        with ThreadPoolExecutor(max_workers=len(tg)) as pool:
            results = list(pool.map(lambda i: _run_bench(i), range(len(tg))))
        total_positions = sum(stats.positions for _, stats in results)
        total_games = sum(stats.games for _, stats in results)
    else:
        _, stats = play_batch(
            None, device="cuda", rng=rng,
            stockfish=sf, evaluator=evaluator, games=games,
            opponent=opponent, temp=temp, search=search, opening=opening, game=game,
        )
        total_positions = stats.positions
        total_games = stats.games

    elapsed = time.perf_counter() - t0

    if hasattr(evaluator, 'shutdown'):
        evaluator.shutdown()
    sf.close()

    import torch
    free_vram, total_vram = torch.cuda.mem_get_info()

    result_dict[worker_id] = {
        "games": total_games,
        "positions": total_positions,
        "elapsed": elapsed,
        "vram_used_gb": (total_vram - free_vram) / 1e9,
    }


def run_config(n_workers, n_threads, sf_per_worker, games_per_worker, args):
    """Run one config with n_workers subprocesses, return summary."""
    mgr = mp.Manager()
    result_dict = mgr.dict()
    barrier = mp.Barrier(n_workers)

    procs = []
    for i in range(n_workers):
        p = mp.Process(
            target=_worker_fn,
            args=(i, n_threads, sf_per_worker, games_per_worker,
                  args.mcts_sims, args.sf_nodes, args.compile,
                  args.stockfish_path, args.bootstrap, result_dict, barrier),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=600)

    failed = sum(1 for p in procs if p.exitcode != 0)
    if failed:
        return None

    results = dict(result_dict)
    total_games = sum(r["games"] for r in results.values())
    total_positions = sum(r["positions"] for r in results.values())
    max_elapsed = max(r["elapsed"] for r in results.values())
    max_vram = max(r["vram_used_gb"] for r in results.values())

    return {
        "workers": n_workers,
        "threads": n_threads,
        "sf_per_worker": sf_per_worker,
        "total_threads": n_workers * n_threads,
        "total_sf": n_workers * sf_per_worker,
        "total_games": total_games,
        "total_positions": total_positions,
        "wall_time": max_elapsed,
        "games_per_sec": total_games / max_elapsed,
        "pos_per_sec": total_positions / max_elapsed,
        "vram_gb": max_vram,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-per-worker", type=int, default=16)
    ap.add_argument("--mcts-sims", type=int, default=128)
    ap.add_argument("--sf-nodes", type=int, default=5000)
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--stockfish-path", default="/home/josh/local_stockfish/extract/usr/games/stockfish")
    ap.add_argument("--bootstrap", default="")
    args = ap.parse_args()

    mp.set_start_method("spawn", force=True)

    # Configs: (workers, threads_per_worker, sf_per_worker, compile_override)
    # Total CPU threads = workers * (threads + sf_per_worker)
    # Constraint: 16 CPU cores, 1 GPU
    configs = [
        # Single worker baselines
        (1,  1,  8, None),    # 1w × 1t — single-threaded baseline
        (1,  8,  8, None),    # 1w × 8t — moderate threading
        # 2 workers compiled
        (2,  1,  4, None),    # 2w × 1t — previous proven best
        (2,  8,  8, None),    # 2w × 8t — current production config
        # 4 workers compiled (may work on 5090 with 32GB VRAM)
        (4,  4,  4, None),    # 4w × 4t compiled
        (4,  8,  4, None),    # 4w × 8t compiled
        # 4 workers eager fallback
        (4,  8,  4, False),   # 4w × 8t eager
    ]

    print(f"Benchmark: {args.games_per_worker} games/worker, {args.mcts_sims} sims, {args.sf_nodes} SF nodes")
    print(f"Default compile: {args.compile}")
    print("=" * 90)

    results = []
    for workers, threads, sf_per, compile_override in configs:
        label_compile = compile_override if compile_override is not None else args.compile
        print(f"\n--- {workers}w × {threads}t, sf={sf_per}/w, compile={label_compile} ---")

        saved = args.compile
        if compile_override is not None:
            args.compile = compile_override

        r = run_config(workers, threads, sf_per, args.games_per_worker, args)

        args.compile = saved

        if r is None:
            print("  => FAILED")
            continue

        results.append(r)
        print(
            f"  => {r['games_per_sec']:.2f} games/s, "
            f"{r['pos_per_sec']:.0f} pos/s, "
            f"wall={r['wall_time']:.1f}s, "
            f"VRAM={r['vram_gb']:.1f}GB"
        )

    if results:
        print("\n" + "=" * 90)
        print(f"{'config':>16} {'compile':>7} {'games/s':>8} {'pos/s':>8} {'wall':>6} {'VRAM':>6}")
        print("-" * 60)
        for r in results:
            label = f"{r['workers']}w×{r['threads']}t"
            # Infer compile from VRAM (rough)
            print(
                f"{label:>16} "
                f"{'':>7} "
                f"{r['games_per_sec']:>8.2f} {r['pos_per_sec']:>8.0f} "
                f"{r['wall_time']:>6.1f} {r['vram_gb']:>5.1f}G"
            )
        best = max(results, key=lambda r: r["pos_per_sec"])
        print(f"\nBest: {best['workers']}w × {best['threads']}t "
              f"({best['games_per_sec']:.2f} games/s, {best['pos_per_sec']:.0f} pos/s)")


if __name__ == "__main__":
    main()
