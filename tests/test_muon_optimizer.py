from __future__ import annotations

import torch

from chess_anti_engine.train.muon import MuonWithAuxAdam


def test_muon_with_aux_adam_updates_both_parameter_types():
    mat = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32))
    vec = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))

    opt = MuonWithAuxAdam(
        [
            {"params": [mat], "lr": 0.02, "weight_decay": 0.01, "use_muon": True},
            {"params": [vec], "lr": 1e-3, "weight_decay": 0.01, "use_muon": False},
        ]
    )

    mat_before = mat.detach().clone()
    vec_before = vec.detach().clone()

    mat.grad = torch.tensor([[0.2, -0.1], [0.05, 0.3]], dtype=torch.float32)
    vec.grad = torch.tensor([0.1, -0.2], dtype=torch.float32)
    opt.step()

    assert not torch.allclose(mat, mat_before)
    assert not torch.allclose(vec, vec_before)
    assert "momentum_buffer" in opt.state[mat]
    assert "exp_avg" in opt.state[vec]
    assert "exp_avg_sq" in opt.state[vec]
