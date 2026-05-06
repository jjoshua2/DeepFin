"""Calibrate opponent difficulty by measuring winrate at current settings.

Usage:
    PYTHONPATH=. python3 scripts/calibrate_difficulty.py \
        --checkpoint runs/pbt2_small/tune/train_trial_44fa7_00000_.../checkpoint_000036/trainer.pt \
        --games 50
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.selfplay.config import (
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.manager import play_batch
from chess_anti_engine.stockfish import StockfishPool


def run_games(
    model: torch.nn.Module,
    sf_pool: StockfishPool,
    *,
    device: str,
    games: int,
    regret: float,
    nodes: int,
) -> dict:
    rng = np.random.default_rng(seed=42)
    sf_pool.set_nodes(nodes)

    opponent = OpponentConfig(wdl_regret_limit=regret)
    search = SearchConfig(simulations=16, mcts_type="gumbel")
    temp = TemperatureConfig(temperature=1.0)
    game_cfg = GameConfig(max_plies=240)

    _, stats = play_batch(
        model,
        device=device,
        rng=rng,
        stockfish=sf_pool,
        games=games,
        opponent=opponent,
        search=search,
        temp=temp,
        game=game_cfg,
    )
    total = stats.w + stats.d + stats.l
    wr = stats.w / max(1, total)
    dr = stats.d / max(1, total)
    return {"w": stats.w, "d": stats.d, "l": stats.l, "wr": wr, "dr": dr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--sf-path", type=str, default="stockfish")
    ap.add_argument("--sf-workers", type=int, default=2)
    args = ap.parse_args()

    # Load model
    print("Loading model...")
    model_cfg = ModelConfig(
        kind="transformer", embed_dim=384, num_layers=8, num_heads=12,
        ffn_mult=2, use_smolgen=True,
    )
    model = build_model(model_cfg)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()

    # Create SF pool
    sf_pool = StockfishPool(
        path=args.sf_path,
        nodes=1000,
        num_workers=args.sf_workers,
        multipv=12,
        hash_mb=4,
    )

    results = []

    # Test 1: Vary WDL regret at fixed nodes.
    print("\n=== Vary regret (nodes=1000) ===")
    for regret in [0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.00]:
        r = run_games(model, sf_pool, device=args.device, games=args.games,
                      regret=regret, nodes=1000)
        print(f"  regret={regret:.2f}  W={r['w']:3d} D={r['d']:3d} L={r['l']:3d}  wr={r['wr']:.2f}")
        results.append({"test": "regret", "regret": regret, "nodes": 1000, **r})

    # Test 2: Vary nodes at a moderate regret limit.
    print("\n=== Vary nodes (regret=0.15) ===")
    for nodes in [500, 1000, 2000, 5000, 10000]:
        r = run_games(model, sf_pool, device=args.device, games=args.games,
                      regret=0.15, nodes=nodes)
        print(f"  nodes={nodes:5d}  W={r['w']:3d} D={r['d']:3d} L={r['l']:3d}  wr={r['wr']:.2f}")
        results.append({"test": "nodes", "regret": 0.15, "nodes": nodes, **r})

    sf_pool.close()

    # Summary
    print("\n=== Summary ===")
    print(f"{'test':<8} {'regret':>7} {'nodes':>6} {'wr':>6}")
    for r in results:
        print(f"{r['test']:<8} {r['regret']:7.2f} {r['nodes']:6d} {r['wr']:6.2f}")


if __name__ == "__main__":
    main()
