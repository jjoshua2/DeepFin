from __future__ import annotations

import numpy as np
import chess
import torch

from chess_anti_engine.mcts.gumbel import GumbelConfig, run_gumbel_root
from chess_anti_engine.moves import POLICY_SIZE


class CountingNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        batch = x.shape[0]
        return {
            "policy_own": torch.zeros((batch, POLICY_SIZE), dtype=torch.float32, device=x.device),
            "wdl": torch.zeros((batch, 3), dtype=torch.float32, device=x.device),
        }


def test_gumbel_uses_extra_forward_passes_when_simulations_increase() -> None:
    model = CountingNet().eval()
    board = chess.Board()
    rng = np.random.default_rng(0)

    probs, action, value = run_gumbel_root(
        model,
        board,
        device="cpu",
        rng=rng,
        cfg=GumbelConfig(simulations=8, topk=8, temperature=1.0),
    )

    assert probs.shape == (POLICY_SIZE,)
    assert probs[action] > 0.0
    assert -1.01 <= float(value) <= 1.01
    # Root eval + at least two deeper batched leaf evaluations. The old
    # shallow implementation typically used only root + one child batch here.
    assert model.forward_calls >= 3
