#!/usr/bin/env python3
"""Head-to-head match between two checkpoints."""
from __future__ import annotations

import argparse
import time

import numpy as np

from chess_anti_engine.selfplay.match import play_match_batch
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="model A checkpoint path")
    p.add_argument("--b", required=True, help="model B checkpoint path")
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--mcts", default="gumbel", choices=["gumbel", "puct"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    args = p.parse_args()

    t0 = time.time()
    print(f"[match] loading A: {args.a}")
    model_a = load_model_from_checkpoint(args.a, device=args.device)
    print(f"[match] loading B: {args.b}")
    model_b = load_model_from_checkpoint(args.b, device=args.device)
    print(f"[match] loaded both in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    a_plays_white = [i % 2 == 0 for i in range(args.games)]
    opening_cfg = OpeningConfig(random_start_plies=4)

    print(f"[match] playing {args.games} games, {args.sims} sims/move, temp={args.temperature}, {args.mcts} MCTS")
    t0 = time.time()
    stats = play_match_batch(
        model_a, model_b,
        device=args.device, rng=rng,
        games=args.games, max_plies=args.max_plies,
        a_plays_white=a_plays_white,
        mcts_type=args.mcts,
        mcts_simulations=args.sims,
        temperature=args.temperature,
        opening_cfg=opening_cfg,
    )
    dt = time.time() - t0

    total = stats.a_win + stats.a_draw + stats.a_loss
    wr_a = (stats.a_win + 0.5 * stats.a_draw) / max(1, total)
    print()
    print(f"[match] {total} games in {dt:.0f}s ({dt/max(1,total):.1f}s/game)")
    print(f"[match] {args.label_a} (A) vs {args.label_b} (B):")
    print(f"  A wins: {stats.a_win}  draws: {stats.a_draw}  A losses: {stats.a_loss}")
    print(f"  A winrate (incl half for draws): {wr_a:.3f}")
    print(f"  A plays white: {stats.a_as_white}  black: {stats.a_as_black}")
    # Rough Elo estimate; saturates near 0/1
    if 0.01 < wr_a < 0.99:
        import math
        elo = -400.0 * math.log10(1.0 / wr_a - 1.0)
        print(f"  Elo (A - B) ≈ {elo:+.0f} (from winrate only)")


if __name__ == "__main__":
    main()
