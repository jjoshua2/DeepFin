#!/usr/bin/env python3
"""Head-to-head match between two checkpoints."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from chess_anti_engine.selfplay.match import play_match_batch
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

_LOG_DIR = Path("runs/matches")


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
    p.add_argument("--log-dir", default=str(_LOG_DIR),
                   help="Directory for match result logs (default: runs/matches)")
    p.add_argument("--no-log", action="store_true", help="Skip writing log file")
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
    elo = -400.0 * math.log10(1.0 / wr_a - 1.0) if 0.01 < wr_a < 0.99 else None

    lines = [
        f"[match] {total} games in {dt:.0f}s ({dt/max(1,total):.1f}s/game)",
        f"[match] {args.label_a} (A) vs {args.label_b} (B):",
        f"  A wins: {stats.a_win}  draws: {stats.a_draw}  A losses: {stats.a_loss}",
        f"  A winrate (incl half for draws): {wr_a:.3f}",
        f"  A plays white: {stats.a_as_white}  black: {stats.a_as_black}",
    ]
    if elo is not None:
        lines.append(f"  Elo (A - B) ≈ {elo:+.0f} (from winrate only)")

    print()
    for line in lines:
        print(line)

    if not args.no_log:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{ts}_{args.label_a}_vs_{args.label_b}.json"
        record = {
            "timestamp": ts,
            "label_a": args.label_a,
            "label_b": args.label_b,
            "checkpoint_a": args.a,
            "checkpoint_b": args.b,
            "games": total,
            "sims": args.sims,
            "temperature": args.temperature,
            "mcts": args.mcts,
            "seed": args.seed,
            "a_wins": stats.a_win,
            "draws": stats.a_draw,
            "b_wins": stats.a_loss,
            "wr_a": round(wr_a, 4),
            "elo_diff": round(elo, 1) if elo is not None else None,
            "duration_s": round(dt, 1),
            "argv": sys.argv,
        }
        log_path.write_text(json.dumps(record, indent=2))
        print(f"\n[match] result logged to {log_path}")


if __name__ == "__main__":
    main()
