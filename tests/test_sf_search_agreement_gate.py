"""Verify the SF/search disagreement gate handles each direction independently."""
from __future__ import annotations

import torch

from chess_anti_engine.train.losses import compute_loss


def _batch(*, sf_says: str, search_says: str, outcome: int) -> tuple[dict, dict]:
    """Build a 1-sample batch with controllable SF / search / outcome.

    `sf_says` and `search_says` ∈ {"win","loss"}; `outcome` is class index
    (0=W, 1=D, 2=L) for ``wdl_t``.
    """
    sf = torch.tensor([[0.95, 0.04, 0.01]] if sf_says == "win" else [[0.01, 0.04, 0.95]])
    sr = torch.tensor([[0.90, 0.05, 0.05]] if search_says == "win" else [[0.05, 0.05, 0.90]])
    outputs = {
        "policy": torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True),
        "wdl": torch.tensor([[2.0, -1.0, -2.0]], requires_grad=True),
    }
    batch = {
        "x": torch.zeros((1, 1, 1, 1)),
        "policy_t": torch.tensor([[1.0, 0.0, 0.0]]),
        "wdl_t": torch.tensor([outcome], dtype=torch.long),
        "has_policy": torch.ones((1,)),
        "is_network_turn": torch.ones((1,)),
        "sf_wdl": sf,
        "has_sf_wdl": torch.ones((1,)),
        "search_wdl": sr,
        "has_search_wdl": torch.ones((1,)),
    }
    return outputs, batch


def _loss(out, batch, **knobs):
    return compute_loss(
        out, batch,
        sf_wdl_frac=0.5, search_wdl_frac=0.5,
        w_policy=0.0, w_wdl=1.0, w_sf_move=0.0, w_sf_eval=0.0,
        w_categorical=0.0, w_volatility=0.0, w_moves_left=0.0,
        **knobs,
    )


def test_sf_low_dampening_only_affects_sf_low_disagreement():
    """SF says LOSS, search says WIN — sf_low side. Dampen knob should cut SF target."""
    out, batch = _batch(sf_says="loss", search_says="win", outcome=0)
    losses_off = _loss(out, batch)
    losses_low = _loss(out, batch, sf_search_dampen_sf_low=1.0)
    losses_high = _loss(out, batch, sf_search_dampen_sf_high=1.0)
    assert losses_off["sf_search_disagree_sf_low_frac"].item() == 1.0
    assert losses_off["sf_search_disagree_sf_high_frac"].item() == 0.0
  # sf_low knob lowers blended_wdl_ce (target moves toward outcome=W).
    assert losses_low["blended_wdl_ce"].item() < losses_off["blended_wdl_ce"].item()
  # sf_high knob does nothing here because it gates the OTHER disagreement direction.
    assert abs(losses_high["blended_wdl_ce"].item() - losses_off["blended_wdl_ce"].item()) < 1e-6


def test_sf_high_dampening_only_affects_sf_high_disagreement():
    """SF says WIN, search says LOSS — sf_high side. The sf_high knob should
    change the blended target; the sf_low knob should not (different bucket)."""
    out, batch = _batch(sf_says="win", search_says="loss", outcome=2)
    losses_off = _loss(out, batch)
    losses_low = _loss(out, batch, sf_search_dampen_sf_low=1.0)
    losses_high = _loss(out, batch, sf_search_dampen_sf_high=1.0)
    assert losses_off["sf_search_disagree_sf_high_frac"].item() == 1.0
    assert losses_off["sf_search_disagree_sf_low_frac"].item() == 0.0
    assert abs(losses_high["blended_wdl_ce"].item() - losses_off["blended_wdl_ce"].item()) > 0.1
    assert abs(losses_low["blended_wdl_ce"].item() - losses_off["blended_wdl_ce"].item()) < 1e-6


def test_partial_dampening_is_between_off_and_full():
    out, batch = _batch(sf_says="loss", search_says="win", outcome=0)
    off = _loss(out, batch)["blended_wdl_ce"].item()
    half = _loss(out, batch, sf_search_dampen_sf_low=0.5)["blended_wdl_ce"].item()
    full = _loss(out, batch, sf_search_dampen_sf_low=1.0)["blended_wdl_ce"].item()
    assert full < half < off


def test_no_dampening_when_signals_agree():
    out, batch = _batch(sf_says="win", search_says="win", outcome=0)
    losses_off = _loss(out, batch)
    losses_on = _loss(out, batch, sf_search_dampen_sf_low=1.0, sf_search_dampen_sf_high=1.0)
    assert losses_off["sf_search_agree_frac"].item() == 1.0
    assert abs(losses_on["blended_wdl_ce"].item() - losses_off["blended_wdl_ce"].item()) < 1e-6


def test_diagnostics_zero_when_search_missing():
    out = {
        "policy": torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True),
        "wdl": torch.tensor([[2.0, -1.0, -2.0]], requires_grad=True),
    }
    batch = {
        "x": torch.zeros((1, 1, 1, 1)),
        "policy_t": torch.tensor([[1.0, 0.0, 0.0]]),
        "wdl_t": torch.tensor([0], dtype=torch.long),
        "has_policy": torch.ones((1,)),
        "is_network_turn": torch.ones((1,)),
        "sf_wdl": torch.tensor([[0.95, 0.04, 0.01]]),
        "has_sf_wdl": torch.ones((1,)),
    }
    losses = _loss(out, batch)
    assert losses["sf_search_agree_frac"].item() == 0.0
    assert losses["sf_search_disagree_sf_low_frac"].item() == 0.0
    assert losses["sf_search_disagree_sf_high_frac"].item() == 0.0
