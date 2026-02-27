import numpy as np
import chess
import torch

from chess_anti_engine.model.transformer import ChessNet, TransformerConfig
from chess_anti_engine.mcts.gumbel import run_gumbel_root, GumbelConfig


def test_gumbel_returns_distribution():
    m = ChessNet(TransformerConfig(in_planes=146, embed_dim=64, num_layers=1, num_heads=4, use_smolgen=False))
    b = chess.Board()
    rng = np.random.default_rng(0)
    probs, a, v = run_gumbel_root(m, b, device="cpu", rng=rng, cfg=GumbelConfig(simulations=10, topk=8, child_sims=2))
    assert probs.shape[0] > 1000
    s = float(probs.sum())
    assert 0.99 <= s <= 1.01
    assert probs[a] > 0
    assert -1.01 <= float(v) <= 1.01
