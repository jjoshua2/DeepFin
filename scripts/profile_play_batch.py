"""Profile time breakdown within a single play_batch call."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.getcwd())

import torch

from chess_anti_engine.model.tiny import TinyNet

# Minimal setup
from chess_anti_engine.selfplay.manager import play_batch
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.utils.config_yaml import load_yaml_config

cfg = load_yaml_config("configs/pbt2_small.yaml")
cfg["selfplay.batch_size"] = 8
cfg["game.max_plies"] = 40
cfg["search.simulations"] = 16
cfg["search.fast_simulations"] = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyNet().to(device).eval()

sf = StockfishPool(
    path="/usr/games/stockfish",
    nodes=1000,
    multipv=3,
    num_workers=2,
)

t0 = time.time()
result = play_batch(
    model=model,
    stockfish=sf,
    config=cfg,
    device=device,
    target=8,
)
elapsed = time.time() - t0
print(f"play_batch: {elapsed:.1f}s, {result.games_completed} games")
sf.close()
