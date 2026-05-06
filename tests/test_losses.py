from __future__ import annotations

import torch

from chess_anti_engine.train.losses import compute_loss


def _base_batch(*, batch_size: int = 2, actions: int = 3) -> dict[str, torch.Tensor]:
    policy_t = torch.zeros((batch_size, actions), dtype=torch.float32)
    policy_t[0, 0] = 1.0
    if batch_size > 1:
        policy_t[1, 1] = 1.0
        wdl_t = torch.tensor([0, 2], dtype=torch.long)
    else:
        wdl_t = torch.tensor([0], dtype=torch.long)
    return {
        "x": torch.zeros((batch_size, 1, 1, 1), dtype=torch.float32),
        "policy_t": policy_t,
        "wdl_t": wdl_t,
        "has_policy": torch.ones((batch_size,), dtype=torch.float32),
        "is_network_turn": torch.ones((batch_size,), dtype=torch.float32),
    }


def test_compute_loss_absent_optional_heads_return_zero_losses_and_main_gradients() -> None:
    outputs = {
        "policy": torch.tensor([[3.0, -1.0, -2.0], [-2.0, 4.0, -1.0]], requires_grad=True),
        "wdl": torch.tensor([[2.0, -1.0, -2.0], [-1.0, -2.0, 3.0]], requires_grad=True),
    }
    batch = _base_batch()

    losses = compute_loss(outputs, batch)
    losses["total"].backward()

    assert losses["soft_policy_ce"].detach().item() == 0.0
    assert losses["future_policy_ce"].detach().item() == 0.0
    assert losses["sf_move_ce"].detach().item() == 0.0
    assert losses["sf_eval_ce"].detach().item() == 0.0
    assert losses["categorical_ce"].detach().item() == 0.0
    assert losses["volatility"].detach().item() == 0.0
    assert losses["sf_volatility"].detach().item() == 0.0
    assert losses["moves_left"].detach().item() == 0.0
    assert outputs["policy"].grad is not None
    assert outputs["wdl"].grad is not None
    assert torch.count_nonzero(outputs["policy"].grad).item() > 0
    assert torch.count_nonzero(outputs["wdl"].grad).item() > 0


def test_compute_loss_masks_current_policy_only_when_has_legal_mask_is_set() -> None:
    outputs = {
        "policy": torch.tensor([[0.0, 9.0, -2.0]], requires_grad=True),
        "wdl": torch.tensor([[2.0, -1.0, -2.0]], requires_grad=True),
    }
    batch = _base_batch(batch_size=1)
    batch["policy_t"] = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    batch["wdl_t"] = torch.tensor([0], dtype=torch.long)
    batch["legal_mask"] = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    batch["has_legal_mask"] = torch.tensor([1.0], dtype=torch.float32)

    masked = compute_loss(outputs, batch, w_wdl=0.0)

    batch["has_legal_mask"] = torch.tensor([0.0], dtype=torch.float32)
    unmasked = compute_loss(outputs, batch, w_wdl=0.0)

    assert masked["policy_ce"].detach().item() < 1e-4
    assert unmasked["policy_ce"].detach().item() > 5.0


def test_compute_loss_future_and_sf_policy_ignore_current_legal_mask() -> None:
    outputs = {
        "policy": torch.tensor([[4.0, -1.0, -2.0]], requires_grad=True),
        "policy_future": torch.tensor([[-5.0, -4.0, 10.0]], requires_grad=True),
        "policy_sf": torch.tensor([[-6.0, -3.0, 11.0]], requires_grad=True),
        "wdl": torch.tensor([[2.0, -1.0, -2.0]], requires_grad=True),
    }
    batch = _base_batch(batch_size=1)
    batch["policy_t"] = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    batch["future_policy_t"] = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    batch["has_future"] = torch.tensor([1.0], dtype=torch.float32)
    batch["sf_policy_t"] = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    batch["has_sf_policy"] = torch.tensor([1.0], dtype=torch.float32)
    batch["has_sf_move"] = torch.tensor([1.0], dtype=torch.float32)
    batch["wdl_t"] = torch.tensor([0], dtype=torch.long)
    batch["legal_mask"] = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    batch["has_legal_mask"] = torch.tensor([1.0], dtype=torch.float32)

    losses = compute_loss(outputs, batch, w_wdl=0.0, w_soft=0.0)

    assert losses["policy_ce"].detach().item() < 1e-4
    assert losses["future_policy_ce"].detach().item() < 1e-4
    assert losses["sf_move_ce"].detach().item() < 1e-4


def test_compute_loss_optional_heads_produce_gradients_when_targets_present() -> None:
    outputs = {
        "policy": torch.tensor([[2.0, -1.0, 0.0]], requires_grad=True),
        "policy_soft": torch.tensor([[0.5, 1.0, -0.5]], requires_grad=True),
        "policy_future": torch.tensor([[0.0, -1.0, 1.5]], requires_grad=True),
        "policy_sf": torch.tensor([[1.0, -1.0, 0.5]], requires_grad=True),
        "wdl": torch.tensor([[0.1, -0.2, 0.3]], requires_grad=True),
        "sf_eval": torch.tensor([[0.2, 0.1, -0.4]], requires_grad=True),
        "categorical": torch.tensor([[0.3, -0.1, 0.2, 0.0]], requires_grad=True),
        "volatility": torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True),
        "sf_volatility": torch.tensor([[0.3, 0.2, 0.1]], requires_grad=True),
        "moves_left": torch.tensor([[0.5]], requires_grad=True),
    }
    batch = _base_batch(batch_size=1)
    batch["policy_t"] = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    batch["wdl_t"] = torch.tensor([1], dtype=torch.long)
    batch["policy_soft_t"] = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
    batch["has_policy_soft"] = torch.tensor([1.0], dtype=torch.float32)
    batch["future_policy_t"] = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    batch["has_future"] = torch.tensor([1.0], dtype=torch.float32)
    batch["sf_policy_t"] = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    batch["has_sf_policy"] = torch.tensor([1.0], dtype=torch.float32)
    batch["has_sf_move"] = torch.tensor([1.0], dtype=torch.float32)
    batch["sf_wdl"] = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float32)
    batch["has_sf_wdl"] = torch.tensor([1.0], dtype=torch.float32)
    batch["categorical_t"] = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    batch["has_categorical"] = torch.tensor([1.0], dtype=torch.float32)
    batch["volatility_t"] = torch.tensor([[0.2, 0.1, 0.0]], dtype=torch.float32)
    batch["has_volatility"] = torch.tensor([1.0], dtype=torch.float32)
    batch["sf_volatility_t"] = torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32)
    batch["has_sf_volatility"] = torch.tensor([1.0], dtype=torch.float32)
    batch["moves_left"] = torch.tensor([0.25], dtype=torch.float32)
    batch["has_moves_left"] = torch.tensor([1.0], dtype=torch.float32)

    losses = compute_loss(outputs, batch, sf_wdl_frac=0.25)
    losses["total"].backward()

    for key in (
        "policy",
        "policy_soft",
        "policy_future",
        "policy_sf",
        "wdl",
        "sf_eval",
        "categorical",
        "volatility",
        "sf_volatility",
        "moves_left",
    ):
        grad = outputs[key].grad
        assert grad is not None, key
        assert torch.count_nonzero(grad).item() > 0, key
