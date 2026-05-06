#!/usr/bin/env python3
"""Profile one current-API play_batch call."""
from __future__ import annotations

import argparse
import os
import shutil
import time


def _resolve_stockfish_path(path_arg: str | None) -> str:
    candidates = [path_arg] if path_arg else []
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        candidates.append(env)
    candidates += ["stockfish", "/usr/games/stockfish", "/usr/local/bin/stockfish"]
    for raw in candidates:
        if not raw:
            continue
        found = shutil.which(raw)
        if found:
            return found
        if os.path.isfile(os.path.expanduser(raw)):
            return os.path.expanduser(raw)
    raise FileNotFoundError("Stockfish not found; pass --stockfish-path or set STOCKFISH_PATH")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stockfish-path", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--max-plies", type=int, default=40)
    p.add_argument("--sf-nodes", type=int, default=1000)
    p.add_argument("--sf-workers", type=int, default=2)
    p.add_argument("--mcts-simulations", type=int, default=16)
    p.add_argument("--fast-simulations", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import numpy as np
    import torch

    from chess_anti_engine.model.tiny import TinyNet
    from chess_anti_engine.selfplay.config import GameConfig, SearchConfig
    from chess_anti_engine.selfplay.manager import play_batch
    from chess_anti_engine.stockfish.pool import StockfishPool

    sf_path = _resolve_stockfish_path(args.stockfish_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyNet(in_planes=146).to(device).eval()
    sf = StockfishPool(
        path=sf_path,
        nodes=int(args.sf_nodes),
        multipv=3,
        num_workers=int(args.sf_workers),
    )
    try:
        t0 = time.perf_counter()
        _samples, stats = play_batch(
            model,
            device=device,
            rng=np.random.default_rng(int(args.seed)),
            stockfish=sf,
            games=int(args.games),
            target_games=int(args.games),
            search=SearchConfig(
                simulations=int(args.mcts_simulations),
                fast_simulations=int(args.fast_simulations),
            ),
            game=GameConfig(max_plies=int(args.max_plies)),
        )
        elapsed = time.perf_counter() - t0
    finally:
        sf.close()

    print(
        f"play_batch: {elapsed:.1f}s, games={stats.games}, "
        f"positions={stats.positions}, W/D/L={stats.w}/{stats.d}/{stats.l}"
    )


if __name__ == "__main__":
    main()
