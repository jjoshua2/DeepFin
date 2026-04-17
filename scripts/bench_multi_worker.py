#!/usr/bin/env python3
"""Benchmark multi-worker selfplay throughput (1 thread per worker, DirectGPU).

Each worker is a separate subprocess with its own compiled model.
Measures total throughput across all workers running concurrently.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time


def _worker_fn(
    worker_id: int,
    batch_size: int,
    sf_workers: int,
    mcts_sims: int,
    sf_nodes: int,
    compile_model: bool,
    stockfish_path: str,
    bootstrap_path: str,
    result_dict: dict,
    barrier,
) -> None:
    import numpy as np
    import torch

    from chess_anti_engine.model import ModelConfig, build_model, load_state_dict_tolerant
    from chess_anti_engine.inference import DirectGPUEvaluator
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

    evaluator = DirectGPUEvaluator(model, device="cuda", max_batch=512)
    sf = StockfishPool(path=stockfish_path, num_workers=sf_workers, nodes=sf_nodes, multipv=1)
    rng = np.random.default_rng(42 + worker_id)

    opponent = OpponentConfig()
    temp = TemperatureConfig()
    search = SearchConfig(simulations=mcts_sims, fast_simulations=mcts_sims // 4)
    opening = OpeningConfig()
    game = GameConfig(max_plies=200)

    # Warmup
    play_batch(
        None, device="cuda", rng=rng,
        stockfish=sf, evaluator=evaluator, games=min(batch_size, 4),
        opponent=opponent, temp=temp, search=search, opening=opening, game=game,
    )

    # Sync all workers
    barrier.wait()

    # Benchmark
    t0 = time.perf_counter()
    _, stats = play_batch(
        None, device="cuda", rng=rng,
        stockfish=sf, evaluator=evaluator, games=batch_size,
        opponent=opponent, temp=temp, search=search, opening=opening, game=game,
    )
    elapsed = time.perf_counter() - t0

    sf.close()
    free_vram, total_vram = torch.cuda.mem_get_info()

    result_dict[worker_id] = {
        "games": stats.games,
        "positions": stats.positions,
        "elapsed": elapsed,
        "vram_used_gb": (total_vram - free_vram) / 1e9,
    }


def run_config(n_workers, batch_per_worker, sf_per_worker, compile_model, args):
    mgr = mp.Manager()
    result_dict = mgr.dict()
    barrier = mp.Barrier(n_workers)

    procs = []
    for i in range(n_workers):
        p = mp.Process(
            target=_worker_fn,
            args=(i, batch_per_worker, sf_per_worker,
                  args.mcts_sims, args.sf_nodes, compile_model,
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
        "batch_per_worker": batch_per_worker,
        "sf_per_worker": sf_per_worker,
        "compile": compile_model,
        "total_games": total_games,
        "total_positions": total_positions,
        "wall_time": max_elapsed,
        "games_per_sec": total_games / max_elapsed,
        "pos_per_sec": total_positions / max_elapsed,
        "vram_gb": max_vram,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--mcts-sims", type=int, default=128)
    ap.add_argument("--sf-nodes", type=int, default=5000)
    ap.add_argument("--stockfish-path", default="/home/josh/local_stockfish/extract/usr/games/stockfish")
    ap.add_argument("--bootstrap", default="")
    args = ap.parse_args()

    mp.set_start_method("spawn", force=True)

    # (workers, batch_per_worker, sf_per_worker, compile)
    configs = [
        (6, 256, 3, True),    # 6 compiled workers
        (8, 256, 2, True),    # 8 compiled workers
    ]

    print(f"Benchmark: {args.mcts_sims} sims, {args.sf_nodes} SF nodes")
    print("=" * 80)

    results = []
    for workers, batch, sf, compile in configs:
        label = f"{workers}w×{batch}g"
        comp_label = "compiled" if compile else "eager"
        print(f"\n--- {label} sf={sf}/w {comp_label} ---")

        r = run_config(workers, batch, sf, compile, args)
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
        print("\n" + "=" * 80)
        print(f"{'config':>14} {'mode':>8} {'games/s':>8} {'pos/s':>8} {'wall':>6} {'VRAM':>6}")
        print("-" * 60)
        for r in results:
            label = f"{r['workers']}w×{r['batch_per_worker']}g"
            mode = "compiled" if r['compile'] else "eager"
            print(
                f"{label:>14} {mode:>8} "
                f"{r['games_per_sec']:>8.2f} {r['pos_per_sec']:>8.0f} "
                f"{r['wall_time']:>6.1f} {r['vram_gb']:>5.1f}G"
            )
        best = max(results, key=lambda r: r["pos_per_sec"])
        print(f"\nBest: {best['workers']}w×{best['batch_per_worker']}g "
              f"{'compiled' if best['compile'] else 'eager'} "
              f"({best['games_per_sec']:.2f} games/s, {best['pos_per_sec']:.0f} pos/s)")


if __name__ == "__main__":
    main()
