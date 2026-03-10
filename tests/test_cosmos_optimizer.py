from __future__ import annotations

import torch

from chess_anti_engine.train.cosmos import COSMOS


def test_cosmos_handles_matrices_narrower_than_default_rank() -> None:
    param = torch.nn.Parameter(torch.randn(3, 64))
    opt = COSMOS([param], lr=3e-4, rank=64, gamma=0.2)

    before = param.detach().clone()
    param.grad = torch.randn_like(param)
    opt.step()

    state = opt.state[param]
    assert not torch.equal(param, before)
    assert state["exp_avg_GG"].shape == (3, 3)
    assert state["exp_avg_P"].shape == (64, 3)
    assert state["exp_avg_sq"].shape == (3, 3)
