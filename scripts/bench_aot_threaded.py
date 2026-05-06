#!/usr/bin/env python3
"""Benchmark AOT evaluator with 1 vs 2 threads (StreamPerThread).

AOT kernels are thread-safe across CUDA streams — no CUDA graph conflicts.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from typing import cast


def _run_config(
    n_threads: int,
    batch_size: int,
    sf_workers: int,
    mcts_sims: int,
    sf_nodes: int,
    aot_dir: str,
    stockfish_path: str,
    bootstrap_path: str,
    result_dict: dict,
) -> None:
    import numpy as np
    import torch

    from chess_anti_engine.inference import DirectGPUEvaluator
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

    # Load weights for AOT
    cfg = ModelConfig(
        kind="transformer", embed_dim=384, num_layers=9, num_heads=12,
        ffn_mult=1.5, use_smolgen=True, use_nla=False,
    )
    model = build_model(cfg).cuda().eval()
    if bootstrap_path:
        ckpt = torch.load(bootstrap_path, map_location="cuda", weights_only=True)
        load_state_dict_tolerant(model, ckpt.get("model", ckpt))

    if n_threads > 1:
        # Multi-threaded: use AOT with per-thread CUDA streams
        from chess_anti_engine.inference import AOTEvaluator
        evaluator = AOTEvaluator(aot_dir, device="cuda", max_batch=4096)
        evaluator.load_weights(model.state_dict())
    else:
        # Single thread baseline: use compiled DirectGPU (same as production)
        model = cast(torch.nn.Module, torch.compile(model, mode="reduce-overhead"))
        evaluator = DirectGPUEvaluator(model, device="cuda", max_batch=512)

    sf = StockfishPool(path=stockfish_path, num_workers=sf_workers, nodes=sf_nodes, multipv=1)
    rng = np.random.default_rng(42)

    opponent = OpponentConfig()
    temp = TemperatureConfig()
    search = SearchConfig(simulations=mcts_sims, fast_simulations=mcts_sims // 4)
    opening = OpeningConfig()
    game = GameConfig(max_plies=200)

    thread_batch = max(1, batch_size // max(1, n_threads))

    # Warmup
    print(f"  Warming up (threads={n_threads}, {'AOT' if n_threads > 1 else 'compiled'})...", flush=True)
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

    # Benchmark
    print(f"  Running {batch_size} games...", flush=True)
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
            stockfish=sf, evaluator=evaluator, games=batch_size,
            opponent=opponent, temp=temp, search=search, opening=opening, game=game,
        )
        total_positions = stats.positions
        total_games = stats.games

    elapsed = time.perf_counter() - t0
    sf.close()

    free_vram, total_vram = torch.cuda.mem_get_info()

    result_dict["threads"] = n_threads
    result_dict["games"] = total_games
    result_dict["positions"] = total_positions
    result_dict["elapsed"] = elapsed
    result_dict["games_per_sec"] = total_games / elapsed
    result_dict["pos_per_sec"] = total_positions / elapsed
    result_dict["vram_used_gb"] = (total_vram - free_vram) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--mcts-sims", type=int, default=128)
    ap.add_argument("--sf-nodes", type=int, default=5000)
    ap.add_argument("--sf-workers", type=int, default=8)
    ap.add_argument("--stockfish-path", default="stockfish")
    ap.add_argument("--bootstrap", default="")
    ap.add_argument("--aot-dir", default="data/aot_models")
    ap.add_argument("--configs", type=str, default="1,2,4",
                    help="Comma-separated thread counts")
    args = ap.parse_args()

    mp.set_start_method("spawn", force=True)

    configs = [int(x) for x in args.configs.split(",")]

    print(f"Benchmark: batch={args.batch_size}, {args.mcts_sims} sims, {args.sf_nodes} SF nodes")
    print(f"AOT dir: {args.aot_dir}")
    print("=" * 80)

    results = []
    for n_threads in configs:
        label = f"{'compiled' if n_threads == 1 else 'AOT'} {n_threads}t"
        print(f"\n--- {label} ---")
        mgr = mp.Manager()
        result_dict = mgr.dict()
        p = mp.Process(
            target=_run_config,
            args=(n_threads, args.batch_size, args.sf_workers,
                  args.mcts_sims, args.sf_nodes, args.aot_dir,
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
        print(f"{'config':>15} {'games/s':>8} {'pos/s':>8} {'time':>6} {'VRAM':>6}")
        print("-" * 50)
        for r in results:
            label = f"{'compiled' if r['threads'] == 1 else 'AOT'} {r['threads']}t"
            print(f"{label:>15} {r['games_per_sec']:>8.2f} {r['pos_per_sec']:>8.0f} "
                  f"{r['elapsed']:>6.1f} {r['vram_used_gb']:>5.1f}G")


if __name__ == "__main__":
    main()
