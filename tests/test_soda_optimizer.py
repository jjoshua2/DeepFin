from __future__ import annotations

import pytest
import torch

from chess_anti_engine.train.soda import (
    SODA_REGULARIZE_KEY,
    SODA_START_STEP_KEY,
    SODAWeightDecayWrapper,
    mark_soda_weight_decay_groups,
)


def test_soda_delay_uses_initial_anchor_and_global_step_beta() -> None:
    param = torch.nn.Parameter(torch.tensor([2.0]))
    groups = [{"params": [param], "weight_decay": 0.0}]
    assert mark_soda_weight_decay_groups(groups, start_step=2, force=True)
    assert groups[0][SODA_REGULARIZE_KEY] is True
    assert groups[0][SODA_START_STEP_KEY] == 2

    opt = SODAWeightDecayWrapper(torch.optim.SGD(groups, lr=1.0))

    for _ in range(2):
        param.grad = torch.tensor([1.0])
        opt.step()
    assert torch.allclose(param.detach(), torch.tensor([0.0]))

    param.grad = torch.tensor([1.0])
    opt.step()

    # Third update: base SGD gives -1.0, then SODA uses beta=1/(global_k+2)=1/4
    # and the construction-time anchor 2.0: -1 + (2 - 0) / 4 = -0.5.
    assert torch.allclose(param.detach(), torch.tensor([-0.5]))


def test_soda_replaces_weight_decay_with_zero_base_decay() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    groups = [{"params": [param], "weight_decay": 0.01}]
    assert mark_soda_weight_decay_groups(groups)
    assert groups[0]["weight_decay"] == 0.0

    opt = SODAWeightDecayWrapper(torch.optim.AdamW(groups, lr=0.0))
    param.grad = torch.tensor([0.0])
    opt.step()

    # With lr=0 and base weight_decay replaced, AdamW itself cannot move the
    # parameter; SODA's anchor equals the initial value, so it also leaves it.
    assert torch.allclose(param.detach(), torch.tensor([1.0]))


def test_soda_reset_anchors_uses_loaded_weights() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    groups = [{"params": [param], "weight_decay": 0.0}]
    assert mark_soda_weight_decay_groups(groups, force=True)
    opt = SODAWeightDecayWrapper(torch.optim.SGD(groups, lr=1.0))

    with torch.no_grad():
        param.fill_(5.0)
    opt.reset_anchors()

    param.grad = torch.tensor([1.0])
    opt.step()

    # Base SGD gives 4.0. With the reset anchor at 5.0 and beta=1/2:
    # 4.0 + (5.0 - 5.0) / 2 = 4.0. A stale construction anchor at 1.0 would
    # incorrectly pull this to 2.0.
    assert torch.allclose(param.detach(), torch.tensor([4.0]))


def test_soda_load_state_requires_anchors_for_marked_params() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    groups = [{"params": [param], "weight_decay": 0.0}]
    assert mark_soda_weight_decay_groups(groups, force=True)
    opt = SODAWeightDecayWrapper(torch.optim.SGD(groups, lr=1.0))
    state = opt.state_dict()
    state["soda_anchors"] = {}

    with pytest.raises(ValueError, match="missing anchors"):
        opt.load_state_dict(state)
