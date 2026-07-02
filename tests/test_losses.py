from __future__ import annotations

import numpy as np
import torch

from chess_anti_engine.train.losses import compute_loss, soft_cross_entropy


def test_soft_cross_entropy_renormalizes_target_rows() -> None:
    """Soft CE must be invariant to a positive rescaling of the target row.

    The loss is linear in the target, so an un-normalized row would scale the
    sample's loss/gradient. After renormalization a row and twice that row
    must give the same loss.
    """
    torch.manual_seed(0)
    logits = torch.randn(4, 16)
    target = torch.rand(4, 16)
    target = target / target.sum(dim=-1, keepdim=True)

    base = soft_cross_entropy(logits, target)
    scaled = soft_cross_entropy(logits, target * 2.0)
    torch.testing.assert_close(base, scaled)


def test_soft_cross_entropy_corrects_rowsum_drift() -> None:
    """Soft CE recovers the normalized loss when target rows do not sum to
    exactly 1 (as happens after the float16 round-trip in replay shards)."""
    torch.manual_seed(1)
    width = 1858  # production lc0_1858 policy width
    logits = torch.randn(8, width)
    target = torch.rand(8, width).pow(4.0)
    target = target / target.sum(dim=-1, keepdim=True)

    ce_norm = soft_cross_entropy(logits, target)

    # A target whose rows sum to ~1.01 must give the same loss (a 1% scaling
    # of the loss/gradient without the renorm).
    drifted = target * 1.01
    torch.testing.assert_close(soft_cross_entropy(logits, drifted), ce_norm)

    # A realistic float16 round-trip recovers the float32 loss within f16 error.
    target_f16 = target.to(torch.float16).to(torch.float32)
    torch.testing.assert_close(
        soft_cross_entropy(logits, target_f16), ce_norm, atol=2e-3, rtol=2e-3,
    )


def test_soft_cross_entropy_all_zero_rows_are_finite_zero() -> None:
    """Missing/masked targets are all-zero rows; they must contribute 0, not NaN."""
    logits = torch.randn(3, 8)
    target = torch.zeros(3, 8)
    target[1] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ce = soft_cross_entropy(logits, target)
    assert torch.isfinite(ce).all()
    assert float(ce[0]) == 0.0
    assert float(ce[2]) == 0.0


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


def test_compute_loss_sf_own_teaches_policy_own_head_masked() -> None:
    """w_sf_own adds a P0 SF-recommendation CE on the SAME head as policy_own
    (the search prior), masked to has_sf_p0 rows, in the current legal space."""
    outputs = {
        "policy": torch.tensor([[4.0, -1.0, -2.0]], requires_grad=True),
        "wdl": torch.tensor([[2.0, -1.0, -2.0]], requires_grad=True),
    }
    batch = _base_batch(batch_size=1)
    batch["policy_t"] = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    # SF recommends a DIFFERENT move than the net's own visits (index 2).
    batch["sf_p0_policy_t"] = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    batch["has_sf_p0"] = torch.tensor([1.0], dtype=torch.float32)

    # Off by default (w_sf_own=0): no contribution.
    off = compute_loss(outputs, batch, w_wdl=0.0)
    assert off["sf_own_ce"].detach().item() > 0.0  # ce is computed...
    base_total = compute_loss(outputs, batch, w_wdl=0.0, w_sf_own=0.0)["total"].detach().item()

    # On: the term enters the total and pushes the policy_own head toward index 2.
    on = compute_loss(outputs, batch, w_wdl=0.0, w_sf_own=0.5)
    on["total"].backward()
    assert on["total"].detach().item() > base_total
    # gradient on the shared "policy" head should pull mass toward index 2.
    assert outputs["policy"].grad is not None
    assert outputs["policy"].grad[0, 2].item() < 0.0  # increasing logit[2] lowers loss


def test_compute_loss_sf_own_zero_when_unmasked() -> None:
    outputs = {
        "policy": torch.tensor([[4.0, -1.0, -2.0]], requires_grad=True),
        "wdl": torch.tensor([[2.0, -1.0, -2.0]], requires_grad=True),
    }
    batch = _base_batch(batch_size=1)
    batch["sf_p0_policy_t"] = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    batch["has_sf_p0"] = torch.tensor([0.0], dtype=torch.float32)  # masked out
    m = compute_loss(outputs, batch, w_wdl=0.0, w_sf_own=0.5)
    assert m["sf_own_ce"].detach().item() == 0.0


def test_compute_loss_sf_own_regret_minimizes_expected_regret_on_policy_own() -> None:
    """w_sf_own_regret adds expected SF cp-regret = sum_m p_own(m)*regret(m) on
    the policy_own head, masked to has_sf_p0_regret rows. Move 2 has zero regret
    (SF best), moves 0/1 are maximally bad; the loss must drop as policy_own mass
    moves toward move 2."""
    # regret(m): index 0 and 1 are maximally bad (1.0), index 2 is SF's best (0.0).
    reg = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)

    def _metric(logits: list[float], *, masked: bool) -> float:
        outputs = {
            "policy": torch.tensor([logits], requires_grad=True),
            "wdl": torch.tensor([[2.0, -1.0, -2.0]], requires_grad=True),
        }
        batch = _base_batch(batch_size=1)
        batch["sf_p0_regret_t"] = reg
        batch["has_sf_p0_regret"] = torch.tensor(
            [1.0 if masked else 0.0], dtype=torch.float32,
        )
        m = compute_loss(outputs, batch, w_wdl=0.0, w_sf_own_regret=0.5)
        return m["sf_own_regret"].detach().item()

    # policy_own favoring the high-regret move 0 vs the low-regret move 2.
    bad = _metric([4.0, -1.0, -2.0], masked=True)   # mass on move 0 (regret 1.0)
    good = _metric([-2.0, -1.0, 4.0], masked=True)  # mass on move 2 (regret 0.0)
    assert np.isfinite(bad)
    assert np.isfinite(good)
    assert good < bad  # shifting mass to the low-regret move lowers the loss

    # Masked out: contributes ~0 regardless of the logits.
    assert _metric([4.0, -1.0, -2.0], masked=False) == 0.0

    # The term flows into the total and its gradient pulls mass toward move 2.
    outputs = {
        "policy": torch.tensor([[4.0, -1.0, -2.0]], requires_grad=True),
        "wdl": torch.tensor([[2.0, -1.0, -2.0]], requires_grad=True),
    }
    batch = _base_batch(batch_size=1)
    batch["sf_p0_regret_t"] = reg
    batch["has_sf_p0_regret"] = torch.tensor([1.0], dtype=torch.float32)
    base_total = compute_loss(
        outputs, batch, w_policy=0.0, w_wdl=0.0, w_sf_own_regret=0.0,
    )["total"].detach().item()
    # Isolate the regret head (w_policy=0) so its gradient sign is unambiguous —
    # the default policy CE pulls toward the net's own move (index 0) and would
    # otherwise dominate the shared policy_own logits.
    on = compute_loss(outputs, batch, w_policy=0.0, w_wdl=0.0, w_sf_own_regret=0.5)
    on["total"].backward()
    assert on["total"].detach().item() > base_total
    assert outputs["policy"].grad is not None
    assert outputs["policy"].grad[0, 2].item() < 0.0  # raising logit[2] lowers expected regret
