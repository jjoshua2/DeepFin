from __future__ import annotations

import torch

from chess_anti_engine.train.cosmos_fast import COSMOSFast


def test_cosmos_fast_updates_cosmos_and_fallback_groups() -> None:
    hidden = torch.nn.Parameter(torch.randn(256, 256))
    bias = torch.nn.Parameter(torch.randn(256))

    opt = COSMOSFast(
        [
            {"params": [hidden], "lr": 3e-4, "weight_decay": 1e-4, "use_cosmos_fast": True},
            {"params": [bias], "lr": 3e-4, "weight_decay": 0.0, "use_cosmos_fast": False},
        ],
        rank=64,
        gamma=0.2,
        residual_work_dtype=torch.float32,
    )

    hidden_before = hidden.detach().clone()
    bias_before = bias.detach().clone()

    hidden.grad = torch.randn_like(hidden)
    bias.grad = torch.randn_like(bias)
    opt.step()

    hidden_state = opt.state[hidden]
    bias_state = opt.state[bias]

    assert not torch.equal(hidden, hidden_before)
    assert not torch.equal(bias, bias_before)

    assert hidden_state["step"] == 1
    assert hidden_state["initialized"] is True
    assert hidden_state["P"].shape == (256, 64)
    assert hidden_state["GG"].shape == (64, 64)
    assert hidden_state["exp_avg_sq"].shape == (256, 64)

    assert bias_state["step"] == 1
    assert bias_state["exp_avg"].shape == bias.shape
    assert bias_state["exp_avg_sq"].shape == bias.shape
