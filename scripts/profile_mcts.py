"""Profile MCTS loop to find where time goes."""
from __future__ import annotations

import time

import chess
import numpy as np
import torch

from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.model.tiny import TinyNet
from chess_anti_engine.moves.encode import index_to_move

model = TinyNet(in_planes=146).eval()

# Simulate a realistic batch: 256 games from starting position
N_GAMES = 256
SIMS = 256
N_PLIES = 10  # enough to profile

boards = [chess.Board() for _ in range(N_GAMES)]
cboards = [cboard_from_board_fast(b) for b in boards]
rng = np.random.default_rng(42)
cfg = GumbelConfig(simulations=SIMS, temperature=1.0, add_noise=True)

# Pre-compute root eval (like manager.py does)
xs = np.empty((N_GAMES, 146, 8, 8), dtype=np.float32)
for i, cb in enumerate(cboards):
    xs[i] = cb.encode_146()

with torch.no_grad():
    out = model(torch.from_numpy(xs))
    pol_logits = out["policy"].numpy()
    wdl_logits = out["wdl"].numpy()

tree = MCTSTree()

# Warm up
_ = run_gumbel_root_many_c(
    model, boards[:4], device='cpu', rng=rng, cfg=GumbelConfig(simulations=8),
    cboards=cboards[:4],
)

# Profile
times = []
for ply in range(N_PLIES):
    t0 = time.perf_counter()
    result = run_gumbel_root_many_c(
        None, boards, device='cpu', rng=rng, cfg=cfg,
        evaluator=type('E', (), {
            'evaluate_encoded': lambda self, x: (
                (lambda o: (o["policy"].numpy(), o["wdl"].numpy()))(model(torch.from_numpy(x)))
            )
        })(),
        pre_pol_logits=pol_logits,
        pre_wdl_logits=wdl_logits,
        cboards=cboards,
        tree=tree,
        root_node_ids=[-1] * N_GAMES,
    )
    elapsed = time.perf_counter() - t0
    times.append(elapsed)

    # Advance all games
    actions = result[1]
    root_ids = result[5]
    for i in range(N_GAMES):
        if actions[i] is not None:
            try:
                m = index_to_move(actions[i], boards[i])
                boards[i].push(m)
                cboards[i] = cboard_from_board_fast(boards[i])
            except Exception:
                pass

    # Re-eval for next ply
    with torch.no_grad():
        for i, cb in enumerate(cboards):
            xs[i] = cb.encode_146()
        out = model(torch.from_numpy(xs))
        pol_logits = out["policy"].numpy()
        wdl_logits = out["wdl"].numpy()

avg = np.mean(times[1:])  # skip first (cold)
print(f"Avg time per ply ({N_GAMES} games, {SIMS} sims): {avg:.3f}s")
print(f"Estimated evals: ~{N_GAMES * SIMS:.0f}")
print(f"Effective NPS: {N_GAMES * SIMS / avg:.0f}")
print(f"Per-ply times: {[f'{t:.3f}' for t in times]}")
