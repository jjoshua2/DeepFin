#!/usr/bin/env python3
"""Benchmark one checkpoint against Stockfish at a fixed regret setting.

Use this to measure how much per-iter winrate variance is due to Stockfish
game sampling noise vs genuine model strength changes. Run the same checkpoint
twice at the same regret to get the noise floor, or compare two checkpoints.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from chess_anti_engine.selfplay import manager as selfplay_manager
from chess_anti_engine.selfplay.config import (
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.state import CompletedGameBatch
from chess_anti_engine.stockfish import StockfishPool
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

_LOG_DIR = Path("runs/bench_vs_sf")
_DEFAULT_MAX_PLIES = 450  # match configs/pbt2_small.yaml max_plies


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _resolve_stockfish_path(path_arg: str | None) -> str:
    candidates = [path_arg] if path_arg else []
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        candidates.append(env)
    candidates += [
        "stockfish",
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/home/josh/projects/chess/e2e_server/publish/stockfish",
    ]
    seen: set[str] = set()
    for raw in candidates:
        if not raw or raw in seen:
            continue
        seen.add(raw)
        found = shutil.which(raw)
        if found:
            return str(found)
        p = Path(raw).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    raise FileNotFoundError(
        "Stockfish binary not found. Install stockfish, add to PATH, "
        "set STOCKFISH_PATH, or pass --stockfish-path."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="path to trainer.pt")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--regret", type=float, default=0.37,
                   help="SF wdl_regret_limit (fraction of best WDL; higher=weaker SF)")
    p.add_argument("--sf-nodes", type=int, default=5000)
    p.add_argument("--sf-multipv", type=int, default=40,
                   help="SF MultiPV (must be >1 for regret to have effect; default=40 matches production)")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--mcts", default="gumbel", choices=["gumbel", "puct"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default="model")
    p.add_argument("--log-dir", default=str(_LOG_DIR))
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--stockfish-path", default=None)
    args = p.parse_args()

    if args.games <= 0:
        p.error("--games must be > 0")
    if args.games % 2 != 0:
        p.error("--games must be even so model colors are exactly balanced")
    if args.sf_multipv < 2:
        print("[bench] WARNING: --sf-multipv < 2 means regret has no effect; SF always plays its best move",
              file=sys.stderr)

    try:
        sf_path = _resolve_stockfish_path(args.stockfish_path)
    except FileNotFoundError as exc:
        print(f"[bench] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    print(f"[bench] loading {args.label}: {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    print(f"[bench] loaded in {time.time() - t0:.1f}s")

    try:
        sf = StockfishPool(
            path=sf_path,
            nodes=args.sf_nodes,
            num_workers=1,
            multipv=args.sf_multipv,
        )
    except Exception as exc:
        print(f"[bench] ERROR: failed to start StockfishPool at {sf_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    completed = 0
    def _on_game(batch: CompletedGameBatch) -> None:
        nonlocal completed
        completed += batch.games
        print(f"[bench] {completed}/{args.games} games completed", flush=True)

    print(
        f"[bench] playing {args.games} games vs Stockfish "
        f"(nodes={args.sf_nodes}, multipv={args.sf_multipv}, "
        f"regret={args.regret}, {args.sims} sims/move, "
        f"temp={args.temperature}, {args.mcts} MCTS)"
    )
    t0 = time.time()
    _, stats = selfplay_manager.play_batch(
        model,
        device=args.device,
        rng=rng,
        stockfish=sf,
        games=args.games,
        target_games=args.games,
        on_game_complete=_on_game,
        opponent=OpponentConfig(wdl_regret_limit=args.regret),
        temp=TemperatureConfig(
            temperature=args.temperature,
            drop_plies=0,
            after=args.temperature,
            decay_moves=0,
            endgame=args.temperature,
        ),
        search=SearchConfig(
            simulations=args.sims,
            mcts_type=args.mcts,
            playout_cap_fraction=1.0,
            fast_simulations=args.sims,
        ),
        opening=OpeningConfig(random_start_plies=4),
        game=GameConfig(max_plies=_DEFAULT_MAX_PLIES, selfplay_fraction=0.0),
    )
    dt = time.time() - t0
    sf.close()

    total = stats.w + stats.d + stats.l
    wr = (stats.w + 0.5 * stats.d) / max(1, total)
    se = math.sqrt(wr * (1.0 - wr) / max(1, total))
    ci_lo, ci_hi = wr - 1.96 * se, wr + 1.96 * se
    elo = -400.0 * math.log10(1.0 / wr - 1.0) if 0.01 < wr < 0.99 else None

    print()
    print(f"[bench] {total} games in {dt:.0f}s ({dt / max(1, total):.1f}s/game)")
    print(f"[bench] {args.label} vs Stockfish (regret={args.regret}, nodes={args.sf_nodes}):")
    print(f"  W={stats.w}  D={stats.d}  L={stats.l}")
    print(f"  winrate: {wr:.4f}")
    print(f"  binomial SE: {se:.4f}   95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    if elo is not None:
        print(f"  implied Elo vs SF-at-this-regret: {elo:+.0f}")

    if not args.no_log:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"bench_{ts}_{args.label}.json"
        record = {
            "timestamp": ts,
            "label": args.label,
            "checkpoint": args.checkpoint,
            "stockfish_path": sf_path,
            "regret": args.regret,
            "sf_nodes": args.sf_nodes,
            "sf_multipv": args.sf_multipv,
            "games": total,
            "sims": args.sims,
            "temperature": args.temperature,
            "mcts": args.mcts,
            "seed": args.seed,
            "wins": stats.w,
            "draws": stats.d,
            "losses": stats.l,
            "winrate": round(wr, 6),
            "se": round(se, 6),
            "ci95": [round(ci_lo, 6), round(ci_hi, 6)],
            "elo_vs_sf": round(elo, 1) if elo is not None else None,
            "duration_s": round(dt, 1),
            "git_hash": _git_hash(),
            "argv": sys.argv,
        }
        log_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\n[bench] result logged to {log_path}")


if __name__ == "__main__":
    main()
