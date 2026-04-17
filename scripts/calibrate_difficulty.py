"""Calibrate opponent_strength weights by measuring winrate at various settings.

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
from chess_anti_engine.selfplay.manager import play_batch
from chess_anti_engine.selfplay.config import (
    OpponentConfig, SearchConfig, TemperatureConfig, GameConfig,
)
from chess_anti_engine.stockfish import StockfishPool


def run_games(
    model: torch.nn.Module,
    sf_pool: StockfishPool,
    *,
    device: str,
    games: int,
    rmp: float,
    topk: int,
    nodes: int,
) -> dict:
    rng = np.random.default_rng(seed=42)
    sf_pool.set_nodes(nodes)

    # Force topk by setting topk_min=topk and topk_stage_end=999
    # so _effective_curriculum_topk always returns topk_min
    opponent = OpponentConfig(
        random_move_prob=rmp,
        topk_stage_end=999.0,
        topk_min=topk,
    )
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
    ap.add_argument("--sf-path", type=str,
                    default="/home/josh/projects/chess/e2e_server/publish/stockfish")
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

    # Test 1: Vary rmp at low range (topk=3, nodes=1000)
    print("\n=== Vary rmp (topk=3, nodes=1000) ===")
    for rmp in [0.00, 0.02, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20]:
        r = run_games(model, sf_pool, device=args.device, games=args.games,
                      rmp=rmp, topk=3, nodes=1000)
        print(f"  rmp={rmp:.2f}  W={r['w']:3d} D={r['d']:3d} L={r['l']:3d}  wr={r['wr']:.2f}")
        results.append({"test": "rmp", "rmp": rmp, "topk": 3, "nodes": 1000, **r})

    # Test 2: Vary topk at rmp=0.05 (nodes=1000)
    print("\n=== Vary topk (rmp=0.05, nodes=1000) ===")
    for topk in [2, 3, 4, 6, 8, 12]:
        r = run_games(model, sf_pool, device=args.device, games=args.games,
                      rmp=0.05, topk=topk, nodes=1000)
        print(f"  topk={topk:2d}   W={r['w']:3d} D={r['d']:3d} L={r['l']:3d}  wr={r['wr']:.2f}")
        results.append({"test": "topk", "rmp": 0.05, "topk": topk, "nodes": 1000, **r})

    # Test 3: Vary nodes at rmp=0.05 (topk=3)
    print("\n=== Vary nodes (rmp=0.05, topk=3) ===")
    for nodes in [500, 1000, 2000, 5000, 10000]:
        r = run_games(model, sf_pool, device=args.device, games=args.games,
                      rmp=0.05, topk=3, nodes=nodes)
        print(f"  nodes={nodes:5d}  W={r['w']:3d} D={r['d']:3d} L={r['l']:3d}  wr={r['wr']:.2f}")
        results.append({"test": "nodes", "rmp": 0.05, "topk": 3, "nodes": nodes, **r})

    sf_pool.close()

    # Summary
    print("\n=== Summary ===")
    print(f"{'test':<6} {'rmp':>5} {'topk':>5} {'nodes':>6} {'wr':>6}")
    for r in results:
        print(f"{r['test']:<6} {r['rmp']:5.2f} {r['topk']:5d} {r['nodes']:6d} {r['wr']:6.2f}")


if __name__ == "__main__":
    main()
