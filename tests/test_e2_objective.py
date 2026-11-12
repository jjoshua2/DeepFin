import dataclasses
import json

import pytest
import torch

from chess_anti_engine.train.e2_batch_adapter import PairedBatch, tensor_hash
from chess_anti_engine.train.e2_objective import freeze_objective, objective_losses


def contract(**overrides):
    return freeze_objective(
        {"w_wdl": 1.0, "w_categorical": 1.0, **overrides},
        donor_sha256="1" * 64,
        architecture_sha256="2" * 64,
        optimizer_plan_sha256="3" * 64,
        batch_trace_sha256="4" * 64,
    )


def pair():
    from chess_anti_engine.train.targets import hlgauss_target

    common = {
        "x": torch.zeros((1, 175, 8, 8)),
        "policy_t": torch.tensor([[1.0, 0.0, 0.0]]),
        "has_policy": torch.ones(1),
        "wdl_t": torch.tensor([0]),
        "is_network_turn": torch.ones(1),
        "search_wdl": torch.tensor([[0.25, 0.5, 0.25]]),
        "has_search_wdl": torch.ones(1),
        "has_categorical": torch.ones(1),
    }
    a = {k: v.clone() for k, v in common.items()}
    b = {k: v.clone() for k, v in common.items()}
    a["categorical_t"] = torch.tensor(hlgauss_target(0.0)[None, :])
    b["categorical_t"] = torch.tensor(hlgauss_target(0.2)[None, :])
    return PairedBatch(
        a, b, {}, tuple((k, tensor_hash(a[k]), tensor_hash(b[k])) for k in sorted(a))
    )


def outputs():
    return {
        "policy": torch.zeros((1, 3), requires_grad=True),
        "wdl": torch.zeros((1, 3), requires_grad=True),
        "categorical": torch.zeros((1, 32), requires_grad=True),
    }


def test_full_defaults_and_immutable_overrides():
    source = {"w_wdl": 1.0, "w_categorical": 1.0}
    c = freeze_objective(
        source,
        donor_sha256="1" * 64,
        architecture_sha256="2" * 64,
        optimizer_plan_sha256="3" * 64,
        batch_trace_sha256="4" * 64,
    )
    source["w_wdl"] = 0
    effective = json.loads(c.payload_json)["effective_shared_kwargs"]
    assert effective["w_soft"] == 0.5
    assert effective["w_future"] == 0.15
    assert effective["w_sf_volatility"] == effective["w_volatility"]
    assert effective["w_wdl"] == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"w_wdl": 0},
        {"w_categorical": 0},
        {"w_wdl": True},
        {"sf_wdl_frac": 0.2},
        {"w_categorical": float("nan")},
        {"sf_sparse_params": {}},
        {"search_wdl_frac": 0.8},
    ],
)
def test_invalid_objective_rejected(overrides):
    with pytest.raises(ValueError, match=r".+"):
        contract(**overrides)


def test_contract_tamper_and_external_pin():
    c = contract()
    p = pair()
    a = outputs()
    b = outputs()
    changed = dataclasses.replace(
        c, payload_json=c.payload_json.replace('"w_wdl":1.0', '"w_wdl":2.0')
    )
    with pytest.raises(ValueError, match="identity"):
        objective_losses(changed, c.sha256, p, a, b)
    with pytest.raises(ValueError, match="identity"):
        objective_losses(c, "0" * 64, p, a, b)


def test_actual_production_backward_matches_single_blend_target():
    c = contract(
        w_policy=0.0,
        w_soft=0.0,
        w_future=0.0,
        w_sf_move=0.0,
        w_sf_eval=0.0,
        w_volatility=0.0,
        w_moves_left=0.0,
    )
    p = pair()
    a = outputs()
    b = outputs()
    before = {name: t.detach().clone() for name, t in a.items()}
    la, lb = objective_losses(c, c.sha256, p, a, b)
    la["total"].backward()
    lb["total"].backward()
    assert a["wdl"].grad is not None
    assert b["wdl"].grad is not None
    search = p.search_only["search_wdl"]
    natural = torch.tensor([[1.0, 0.0, 0.0]])
    torch.testing.assert_close(a["wdl"].grad, torch.full((1, 3), 1 / 3) - search)
    torch.testing.assert_close(
        b["wdl"].grad, torch.full((1, 3), 1 / 3) - (0.8 * search + 0.2 * natural)
    )
    torch.testing.assert_close(b["wdl"].grad - a["wdl"].grad, -0.2 * (natural - search))
    torch.testing.assert_close(
        b["categorical"].grad,
        torch.full((1, 32), 1 / 32) - p.outcome20["categorical_t"],
        atol=1e-7,
        rtol=1e-5,
    )
    for name in a:
        torch.testing.assert_close(a[name], before[name])
    assert torch.isfinite(b["wdl"].grad).all()
