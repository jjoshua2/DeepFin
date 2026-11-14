"""Synthetic tiny production transformer only; no real checkpoints/data."""

import pytest
import torch
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.train.losses import (
    soft_cross_entropy,
    masked_mean,
    _normalize_sf_wdl_probs,
)
from scripts.stored_wdl_gradient import gradient_probe, gradient_geometry, SUBSET


def fixture():
    torch.manual_seed(2026)
    model = build_model(
        ModelConfig(
            kind="transformer",
            embed_dim=16,
            num_layers=1,
            num_heads=2,
            use_smolgen=False,
            categorical_head_coupled=True,
        )
    ).eval()
    x = torch.randn(3, 175, 8, 8)
    w = torch.softmax(torch.randn(3, 3), 1)
    c = torch.softmax(torch.randn(3, 32), 1)
    masks = torch.ones(3)
    return model, x, w, c, masks, masks.clone()


def test_geometry_controls():
    a = torch.tensor([1.0, 2.0, -3.0])
    assert gradient_geometry(a, a)["cosine"] == pytest.approx(1)
    assert gradient_geometry(a, -a)["cosine"] == pytest.approx(-1)
    assert gradient_geometry(a, torch.zeros_like(a))["cosine"] is None
    assert gradient_geometry(a, 2 * a)["dot"] == pytest.approx(
        2 * gradient_geometry(a, a)["dot"]
    )


def test_actual_graph_chunk_repeat_and_state():
    args = fixture()
    model = args[0]
    original = {k: p.detach().clone() for k, p in model.named_parameters()}
    flags = {k: p.requires_grad for k, p in model.named_parameters()}
    a = gradient_probe(*args, batch_size=3)
    b = gradient_probe(*args, batch_size=1)
    repeat = gradient_probe(*args, batch_size=3)
    for key in ("wdl_gradient", "categorical_gradient"):
        assert torch.allclose(a[key], b[key], atol=1e-6, rtol=1e-4)
        assert torch.equal(a[key], repeat[key])
    for k, p in model.named_parameters():
        assert torch.equal(p, original[k])
        assert p.requires_grad == flags[k]
        assert p.grad is None
    assert model.training is False


def test_matches_actual_production_forward_losses_and_subset_gradients():
    args = fixture()
    model, x, w, c, net, has = args
    net[1] = 0
    has[2] = 0
    probe = gradient_probe(*args, batch_size=2)
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    params = tuple(named[k] for k in SUBSET)
    for p in params:
        p.requires_grad_(True)
    out = model(x)
    target = _normalize_sf_wdl_probs(w)
    assert target is not None
    lw = masked_mean(soft_cross_entropy(out["wdl"], target), net)
    lc = masked_mean(soft_cross_entropy(out["categorical"], c), net * has)
    a = torch.cat(
        [
            g.flatten().double()
            for g in torch.autograd.grad(lw, params, retain_graph=True)
        ]
    )
    b = torch.cat([g.flatten().double() for g in torch.autograd.grad(lc, params)])
    assert torch.allclose(a, probe["wdl_gradient"], atol=1e-6, rtol=1e-4)
    assert torch.allclose(b, probe["categorical_gradient"], atol=1e-6, rtol=1e-4)
    assert float(lw.detach()) == pytest.approx(probe["weighted_wdl_loss"], abs=1e-6)
    assert float(lc.detach()) == pytest.approx(
        probe["weighted_categorical_loss"], abs=1e-6
    )


def test_weight_and_empty_mask_controls():
    args = fixture()
    a = gradient_probe(*args)
    b = gradient_probe(*args, w_categorical=-2)
    zero = gradient_probe(*args, w_categorical=0)
    assert torch.allclose(b["categorical_gradient"], -2 * a["categorical_gradient"])
    assert zero["geometry"]["cosine"] is None
    args[-1].zero_()
    empty = gradient_probe(*args)
    assert empty["categorical_rows"] == 0
    assert empty["geometry"]["cosine"] is None


@pytest.mark.parametrize("bad", [True, 0, 65])
def test_reject_bad_batch(bad):
    with pytest.raises(ValueError, match="batch"):
        gradient_probe(*fixture(), batch_size=bad)


def test_restore_on_exception(monkeypatch):
    args = fixture()
    model = args[0]
    model.train()
    flags = [p.requires_grad for p in model.parameters()]

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(model, "_encode_tokens", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        gradient_probe(*args)
    assert model.training
    assert [p.requires_grad for p in model.parameters()] == flags


def test_missing_categorical_zero_row_kept():
    args = fixture()
    args[3][1].zero_()
    args[5][1] = 0
    out = gradient_probe(*args)
    assert out["rows"] == 3
    assert out["categorical_rows"] == 2
