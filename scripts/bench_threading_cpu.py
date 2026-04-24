#!/usr/bin/env python3
"""Benchmark selfplay threading overhead using TinyNet on CPU.

Measures games/sec at 1, 2, 4, 8 threads to isolate GIL contention.
No GPU involved — pure CPU threading test.

Usage:
    PYTHONPATH=. python3 scripts/bench_threading_cpu.py [--threads 1,2,4,8] [--games 16] [--sims 8]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from chess_anti_engine.inference import (
    BatchEvaluator,
    LocalModelEvaluator,
    ThreadedBatchEvaluator,
)
from chess_anti_engine.model.tiny import TinyNet
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.manager import play_batch
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.stockfish.uci import StockfishUCI


class InstantEvaluator(BatchEvaluator):
    """Returns random policy/WDL logits instantly. No model, no GIL pressure."""

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = x.shape[0]
        pol = np.random.randn(n, 4672).astype(np.float32) * 0.1
        wdl = np.random.randn(n, 3).astype(np.float32) * 0.1
        return pol, wdl


def run_single_thread(
    evaluator, sf_path: str, games: int, sims: int, seed: int, device: str = "cpu",
) -> tuple[int, float]:
    """Run play_batch in a single thread, return (positions, elapsed)."""
    rng = np.random.default_rng(seed)
    sf = StockfishUCI(sf_path, nodes=100)
    search = SearchConfig(simulations=sims, fast_simulations=max(2, sims // 4), mcts_type="gumbel")
    game_cfg = GameConfig(max_plies=60, selfplay_fraction=1.0)  # all selfplay, no SF turns
    t0 = time.perf_counter()
    samples, stats = play_batch(
        None, device=device, rng=rng, stockfish=sf,
        evaluator=evaluator, games=games,
        search=search, game=game_cfg,
        opponent=OpponentConfig(), temp=TemperatureConfig(),
        opening=OpeningConfig(), diff_focus=DiffFocusConfig(),
    )
    elapsed = time.perf_counter() - t0
    sf.close()
    return stats.positions, elapsed


def bench_threads(n_threads: int, total_games: int, sims: int, sf_path: str, instant: bool = False) -> dict:
    torch.set_num_threads(1)  # Don't let PyTorch use intra-op parallelism

    device = "cuda" if torch.cuda.is_available() and not instant else "cpu"

    if instant:
        evaluator = InstantEvaluator()
    elif n_threads <= 1:
        model = TinyNet(in_planes=146).eval()
        if device == "cuda":
            model = model.cuda()
        evaluator = LocalModelEvaluator(model, device=device)
    else:
        model = TinyNet(in_planes=146).eval()
        if device == "cuda":
            model = model.cuda()
        evaluator = ThreadedBatchEvaluator(
            model, device=device, max_batch=4096, min_batch=32,
            accumulation_timeout_s=0.001,
        )

    if n_threads <= 1:
        positions, elapsed = run_single_thread(evaluator, sf_path, total_games, sims, seed=42, device=device)
        if hasattr(evaluator, "shutdown"):
            evaluator.shutdown()
        return {"threads": 1, "games": total_games, "positions": positions,
                "elapsed": elapsed, "games_per_sec": total_games / elapsed,
                "pos_per_sec": positions / elapsed}

    base, rem = divmod(total_games, n_threads)
    thread_games = [base + (1 if i < rem else 0) for i in range(n_threads)]

    def _worker(tid: int) -> tuple[int, float]:
        return run_single_thread(evaluator, sf_path, thread_games[tid], sims, seed=42 + tid, device=device)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futs = [pool.submit(_worker, i) for i in range(n_threads)]
        results = [f.result() for f in futs]
    elapsed = time.perf_counter() - t0

    if hasattr(evaluator, "shutdown"):
        evaluator.shutdown()

    total_positions = sum(r[0] for r in results)
    return {"threads": n_threads, "games": total_games, "positions": total_positions,
            "elapsed": elapsed, "games_per_sec": total_games / elapsed,
            "pos_per_sec": total_positions / elapsed}


def main():
    parser = argparse.ArgumentParser(description="CPU threading benchmark")
    parser.add_argument("--threads", default="1,2,4,8", help="Comma-separated thread counts")
    parser.add_argument("--games", type=int, default=16, help="Total games per config")
    parser.add_argument("--sims", type=int, default=8, help="MCTS simulations per move")
    parser.add_argument("--sf-path", default="stockfish", help="Stockfish binary path")
    parser.add_argument("--instant", action="store_true", help="Use instant evaluator (no model)")
    args = parser.parse_args()

    thread_counts = [int(x) for x in args.threads.split(",")]
    mode = "instant evaluator" if args.instant else "TinyNet CPU"

    print(f"Benchmarking: {args.games} games, {args.sims} sims, selfplay-only, {mode}")
    print(f"{'Threads':>8} {'Games':>6} {'Positions':>10} {'Time(s)':>8} {'Games/s':>8} {'Pos/s':>8} {'Speedup':>8}")
    print("-" * 70)

    base_gps = None
    for n in thread_counts:
        result = bench_threads(n, args.games, args.sims, args.sf_path, instant=args.instant)
        if base_gps is None:
            base_gps = result["games_per_sec"]
        speedup = result["games_per_sec"] / base_gps
        print(f"{result['threads']:>8} {result['games']:>6} {result['positions']:>10} "
              f"{result['elapsed']:>8.2f} {result['games_per_sec']:>8.2f} "
              f"{result['pos_per_sec']:>8.0f} {speedup:>8.2f}x")


if __name__ == "__main__":
    main()
