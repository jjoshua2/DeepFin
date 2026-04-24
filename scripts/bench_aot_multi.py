#!/usr/bin/env python3
"""Benchmark multi-process × multi-thread AOT selfplay."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time


def _worker_fn(worker_id, n_threads, batch_per_thread, sf_workers,
               mcts_sims, sf_nodes, aot_dir, stockfish_path, bootstrap_path,
               result_dict, barrier):
    import numpy as np
    import torch

    from chess_anti_engine.inference import ThreadedAOTEvaluator
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

    cfg = ModelConfig(kind="transformer", embed_dim=384, num_layers=9, num_heads=12,
                      ffn_mult=1.5, use_smolgen=True, use_nla=False)
    model = build_model(cfg).cuda().eval()
    if bootstrap_path:
        ckpt = torch.load(bootstrap_path, map_location="cuda", weights_only=True)
        load_state_dict_tolerant(model, ckpt.get("model", ckpt))

    evaluator = ThreadedAOTEvaluator(aot_dir, device="cuda", max_batch=4096)
    evaluator.load_weights(model.state_dict())
    del model

    sf = StockfishPool(path=stockfish_path, num_workers=sf_workers, nodes=sf_nodes, multipv=1)
    rng = np.random.default_rng(42 + worker_id)

    opponent = OpponentConfig()
    temp = TemperatureConfig()
    search = SearchConfig(simulations=mcts_sims, fast_simulations=mcts_sims // 4)
    opening = OpeningConfig()
    game = GameConfig(max_plies=200)

    total_batch = batch_per_thread * n_threads

    # Warmup
    if n_threads > 1:
        from concurrent.futures import ThreadPoolExecutor
        def _warmup(tid):
            return play_batch(None, device="cuda", rng=np.random.default_rng(tid + worker_id*100),
                              stockfish=sf, evaluator=evaluator, games=min(batch_per_thread, 4),
                              opponent=opponent, temp=temp, search=search, opening=opening, game=game)
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(lambda i: _warmup(i), range(n_threads)))
    else:
        play_batch(None, device="cuda", rng=rng, stockfish=sf, evaluator=evaluator,
                   games=min(total_batch, 4), opponent=opponent, temp=temp,
                   search=search, opening=opening, game=game)

    barrier.wait()

    t0 = time.perf_counter()
    if n_threads > 1:
        from concurrent.futures import ThreadPoolExecutor
        seeds = [int(rng.integers(2**63)) for _ in range(n_threads)]
        def _bench(tid):
            return play_batch(None, device="cuda", rng=np.random.default_rng(seeds[tid]),
                              stockfish=sf, evaluator=evaluator, games=batch_per_thread,
                              opponent=opponent, temp=temp, search=search, opening=opening, game=game)
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            results = list(pool.map(lambda i: _bench(i), range(n_threads)))
        total_positions = sum(s.positions for _, s in results)
        total_games = sum(s.games for _, s in results)
    else:
        _, stats = play_batch(None, device="cuda", rng=rng, stockfish=sf, evaluator=evaluator,
                              games=total_batch, opponent=opponent, temp=temp,
                              search=search, opening=opening, game=game)
        total_positions = stats.positions
        total_games = stats.games

    elapsed = time.perf_counter() - t0
    evaluator.shutdown()
    sf.close()
    free, total = torch.cuda.mem_get_info()
    result_dict[worker_id] = {"games": total_games, "positions": total_positions,
                               "elapsed": elapsed, "vram_used_gb": (total - free) / 1e9}


def run_config(n_workers, n_threads, batch_per_thread, sf_per_worker, args):
    mgr = mp.Manager()
    result_dict = mgr.dict()
    barrier = mp.Barrier(n_workers)
    procs = []
    for i in range(n_workers):
        p = mp.Process(target=_worker_fn,
                       args=(i, n_threads, batch_per_thread, sf_per_worker,
                             args.mcts_sims, args.sf_nodes, args.aot_dir,
                             args.stockfish_path, args.bootstrap, result_dict, barrier))
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
    return {"workers": n_workers, "threads": n_threads, "batch_per_thread": batch_per_thread,
            "total_games": total_games, "total_positions": total_positions,
            "wall_time": max_elapsed, "games_per_sec": total_games / max_elapsed,
            "pos_per_sec": total_positions / max_elapsed, "vram_gb": max_vram}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcts-sims", type=int, default=128)
    ap.add_argument("--sf-nodes", type=int, default=5000)
    ap.add_argument("--stockfish-path", default="/home/josh/local_stockfish/extract/usr/games/stockfish")
    ap.add_argument("--bootstrap", default="")
    ap.add_argument("--aot-dir", default="/home/josh/projects/chess/data/aot_models")
    args = ap.parse_args()

    mp.set_start_method("spawn", force=True)

    # (workers, threads, batch_per_thread, sf_per_worker)
    configs = [
        (4, 2, 170, 2),   # 4 proc × 2 thread × 170 games — matches 5090 170 SMs
    ]

    print(f"AOT Multi-Worker Benchmark: {args.mcts_sims} sims, {args.sf_nodes} SF nodes")
    print("=" * 80)

    results = []
    for workers, threads, bpt, sf in configs:
        label = f"{workers}w×{threads}t×{bpt}g"
        print(f"\n--- {label} (sf={sf}/w) ---")
        r = run_config(workers, threads, bpt, sf, args)
        if r is None:
            print("  => FAILED")
            continue
        results.append(r)
        print(f"  => {r['games_per_sec']:.2f} games/s, {r['pos_per_sec']:.0f} pos/s, "
              f"wall={r['wall_time']:.1f}s, VRAM={r['vram_gb']:.1f}GB")

    if results:
        print("\n" + "=" * 80)
        print(f"{'config':>20} {'games/s':>8} {'pos/s':>8} {'wall':>6} {'VRAM':>6}")
        print("-" * 55)
        for r in results:
            label = f"{r['workers']}w×{r['threads']}t×{r['batch_per_thread']}g"
            print(f"{label:>20} {r['games_per_sec']:>8.2f} {r['pos_per_sec']:>8.0f} "
                  f"{r['wall_time']:>6.1f} {r['vram_gb']:>5.1f}G")


if __name__ == "__main__":
    main()
