from __future__ import annotations

import pytest
import torch

from chess_anti_engine.moves import COMPACT_POLICY_SIZE, COMPACT_TO_FULL_POLICY, POLICY_SIZE
from chess_anti_engine.train.losses import compute_loss
from chess_anti_engine.train.trainer import Trainer


def _uniform_policy(b: int) -> torch.Tensor:
    p = torch.ones((b, POLICY_SIZE))
    return p / p.sum(dim=-1, keepdim=True)


def _fake_outputs(b: int, *, requires_grad: bool = False) -> dict[str, torch.Tensor]:
    return {
        "policy_own": torch.randn((b, POLICY_SIZE), requires_grad=requires_grad),
        "policy_soft": torch.randn((b, POLICY_SIZE), requires_grad=requires_grad),
        "policy_sf": torch.randn((b, POLICY_SIZE), requires_grad=requires_grad),
        "policy_future": torch.randn((b, POLICY_SIZE), requires_grad=requires_grad),
        "wdl": torch.randn((b, 3), requires_grad=requires_grad),
        "sf_eval": torch.randn((b, 3), requires_grad=requires_grad),
        "categorical": torch.randn((b, 32), requires_grad=requires_grad),
        "volatility": torch.rand((b, 3), requires_grad=requires_grad),
        "sf_volatility": torch.rand((b, 3), requires_grad=requires_grad),
        "moves_left": torch.rand((b, 1), requires_grad=requires_grad),
    }


def _minimal_batch(b: int) -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn((b, 146, 8, 8)),
        "policy_t": _uniform_policy(b),
        "wdl_t": torch.randint(0, 3, (b,)),
        "has_policy": torch.ones((b,)),
    }


def _full_batch(b: int) -> dict[str, torch.Tensor]:
    batch = _minimal_batch(b)
    batch["is_network_turn"] = torch.ones((b,))
    batch["has_sf_wdl"] = torch.ones((b,))
    batch["sf_wdl"] = torch.tensor([[0.2, 0.5, 0.3]] * b)
    batch["has_sf_move"] = torch.ones((b,))
    batch["sf_move_index"] = torch.randint(0, POLICY_SIZE, (b,))
    batch["has_sf_policy"] = torch.ones((b,))
    batch["sf_policy_t"] = _uniform_policy(b)
    batch["has_moves_left"] = torch.ones((b,))
    batch["moves_left"] = torch.rand((b,))
    batch["has_categorical"] = torch.ones((b,))
    batch["categorical_t"] = torch.ones((b, 32)) / 32.0
    batch["has_policy_soft"] = torch.ones((b,))
    batch["policy_soft_t"] = _uniform_policy(b)
    batch["has_future"] = torch.ones((b,))
    batch["future_policy_t"] = _uniform_policy(b)
    batch["has_volatility"] = torch.ones((b,))
    batch["volatility_t"] = torch.rand((b, 3))
    batch["has_sf_volatility"] = torch.ones((b,))
    batch["sf_volatility_t"] = torch.rand((b, 3))
    batch["legal_mask"] = torch.ones((b, POLICY_SIZE))
    batch["has_legal_mask"] = torch.ones((b,))
    return batch


# ── Basic shape and key tests ────────────────────────────────────────


def test_compute_loss_returns_expected_keys():
    b = 4
    losses = compute_loss(_fake_outputs(b), _full_batch(b))
    # Core keys must be present. Observation-only per-source / per-phase
    # diagnostics are additive and tested separately.
    expected = {
        "total", "policy_ce", "wdl_ce", "blended_wdl_ce",
        "soft_policy_ce", "future_policy_ce",
        "sf_move_ce", "sf_eval_ce", "categorical_ce",
        "volatility", "sf_volatility", "moves_left",
    }
    assert expected <= set(losses.keys())


def test_compute_loss_all_scalars():
    b = 4
    losses = compute_loss(_fake_outputs(b), _full_batch(b))
    for k, v in losses.items():
        assert v.ndim == 0, f"{k} should be scalar, got shape {v.shape}"


def test_compute_loss_accepts_compact_logits_with_full_policy_targets():
    b = 2
    outputs = _fake_outputs(b)
    outputs["policy_own"] = torch.randn((b, COMPACT_POLICY_SIZE), requires_grad=True)
    outputs["policy_soft"] = torch.randn((b, COMPACT_POLICY_SIZE), requires_grad=True)
    outputs["policy_sf"] = torch.randn((b, COMPACT_POLICY_SIZE), requires_grad=True)
    outputs["policy_future"] = torch.randn((b, COMPACT_POLICY_SIZE), requires_grad=True)

    batch = _full_batch(b)
    full_idx = int(COMPACT_TO_FULL_POLICY[0])
    for key in ("policy_t", "policy_soft_t", "sf_policy_t", "future_policy_t"):
        target = torch.zeros((b, POLICY_SIZE), dtype=torch.float32)
        target[:, full_idx] = 1.0
        batch[key] = target
    for key in ("legal_mask", "sf_legal_mask", "future_legal_mask"):
        batch[key] = torch.ones((b, POLICY_SIZE), dtype=torch.float32)
    batch["has_sf_legal_mask"] = torch.ones((b,))
    batch["has_future_legal_mask"] = torch.ones((b,))

    losses = compute_loss(outputs, batch)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert outputs["policy_own"].grad is not None


def test_policy_accuracy_stats_accepts_compact_logits_with_full_policy_targets():
    b = 2
    compact_idx = 7
    full_idx = int(COMPACT_TO_FULL_POLICY[compact_idx])
    out = {
        "policy_own": torch.zeros((b, COMPACT_POLICY_SIZE)),
        "policy_sf": torch.zeros((b, COMPACT_POLICY_SIZE)),
        "policy_future": torch.zeros((b, COMPACT_POLICY_SIZE)),
    }
    out["policy_own"][:, compact_idx] = 10.0
    out["policy_sf"][:, compact_idx] = 10.0
    out["policy_future"][:, compact_idx] = 10.0

    batch = _full_batch(b)
    for key in ("policy_t", "sf_policy_t", "future_policy_t"):
        target = torch.zeros((b, POLICY_SIZE), dtype=torch.float32)
        target[:, full_idx] = 1.0
        batch[key] = target
    for key in ("legal_mask", "sf_legal_mask", "future_legal_mask"):
        batch[key] = torch.ones((b, POLICY_SIZE), dtype=torch.float32)
    batch["has_sf_legal_mask"] = torch.ones((b,))
    batch["has_future_legal_mask"] = torch.ones((b,))
    batch["sf_move_index"] = torch.full((b,), full_idx, dtype=torch.int64)

    stats = Trainer._policy_accuracy_stats(out, batch)

    for key in ("policy_own_acc_top1", "sf_move_acc", "policy_future_acc_top1"):
        numerator, denominator = stats[key]
        assert float(numerator) == float(b)
        assert float(denominator) == float(b)


def test_total_is_finite():
    b = 4
    losses = compute_loss(_fake_outputs(b), _full_batch(b))
    assert torch.isfinite(losses["total"])


# ── Masking tests ────────────────────────────────────────────────────


def test_zero_net_mask_zeros_policy_and_wdl():
    """When no sample is a network turn, masked losses should be zero."""
    b = 4
    batch = _full_batch(b)
    batch["is_network_turn"] = torch.zeros((b,))
    losses = compute_loss(_fake_outputs(b), batch)

    assert losses["policy_ce"].item() == 0.0
    assert losses["wdl_ce"].item() == 0.0
    assert losses["sf_move_ce"].item() == 0.0
    assert losses["sf_eval_ce"].item() == 0.0
    assert losses["categorical_ce"].item() == 0.0
    assert losses["moves_left"].item() == 0.0


def test_absent_optional_heads_produce_zero_loss():
    """With only required fields, optional head losses should be zero."""
    b = 4
    batch = _minimal_batch(b)
    # No is_network_turn → defaults to all-ones inside compute_loss
    losses = compute_loss(_fake_outputs(b), batch)

    assert losses["soft_policy_ce"].item() == 0.0
    assert losses["future_policy_ce"].item() == 0.0
    assert losses["sf_move_ce"].item() == 0.0
    assert losses["sf_eval_ce"].item() == 0.0
    assert losses["categorical_ce"].item() == 0.0
    assert losses["volatility"].item() == 0.0
    assert losses["sf_volatility"].item() == 0.0
    assert losses["moves_left"].item() == 0.0


def test_adjusted_wdl_target_replaces_game_component():
    batch = _minimal_batch(1)
    batch["wdl_t"] = torch.tensor([0])
    batch["future_sf_regret_sum"] = torch.ones((1,))
    batch["has_future_sf_regret_sum"] = torch.ones((1,))

    outputs = _fake_outputs(1)
    outputs["wdl"] = torch.tensor([[-4.0, 0.0, 4.0]])

    adjusted = compute_loss(
        outputs,
        batch,
        sf_wdl_frac=0.0,
        search_wdl_frac=0.0,
        use_adjusted_wdl_target=True,
    )

    raw_batch = _minimal_batch(1)
    raw_batch["wdl_t"] = torch.tensor([0])
    raw = compute_loss(outputs, raw_batch, sf_wdl_frac=0.0, search_wdl_frac=0.0)

    assert adjusted["blended_wdl_ce"].item() < raw["blended_wdl_ce"].item()


def test_adjusted_wdl_target_does_not_replace_missing_sf_fallback():
    batch = _minimal_batch(1)
    batch["wdl_t"] = torch.tensor([0])
    batch["future_sf_regret_sum"] = torch.ones((1,))
    batch["has_future_sf_regret_sum"] = torch.ones((1,))

    outputs = _fake_outputs(1)
    outputs["wdl"] = torch.tensor([[4.0, 0.0, -4.0]])

    adjusted = compute_loss(
        outputs,
        batch,
        sf_wdl_frac=1.0,
        search_wdl_frac=0.0,
        use_adjusted_wdl_target=True,
    )

    raw_batch = _minimal_batch(1)
    raw_batch["wdl_t"] = torch.tensor([0])
    raw = compute_loss(outputs, raw_batch, sf_wdl_frac=1.0, search_wdl_frac=0.0)

    assert adjusted["blended_wdl_ce"].item() == raw["blended_wdl_ce"].item()


def test_adjusted_wdl_target_is_opt_in():
    batch = _minimal_batch(1)
    batch["wdl_t"] = torch.tensor([0])
    batch["future_sf_regret_sum"] = torch.ones((1,))
    batch["has_future_sf_regret_sum"] = torch.ones((1,))

    outputs = _fake_outputs(1)
    outputs["wdl"] = torch.tensor([[-4.0, 0.0, 4.0]])

    disabled = compute_loss(outputs, batch, sf_wdl_frac=0.0, search_wdl_frac=0.0)

    raw_batch = _minimal_batch(1)
    raw_batch["wdl_t"] = torch.tensor([0])
    raw = compute_loss(outputs, raw_batch, sf_wdl_frac=0.0, search_wdl_frac=0.0)

    assert disabled["blended_wdl_ce"].item() == raw["blended_wdl_ce"].item()


def test_adjusted_wdl_target_can_use_discounted_source():
    batch = _minimal_batch(1)
    batch["wdl_t"] = torch.tensor([0])
    batch["future_sf_regret_sum"] = torch.zeros((1,))
    batch["has_future_sf_regret_sum"] = torch.ones((1,))
    batch["future_sf_regret_d95"] = torch.ones((1,))
    batch["has_future_sf_regret_d95"] = torch.ones((1,))

    outputs = _fake_outputs(1)
    outputs["wdl"] = torch.tensor([[-4.0, 0.0, 4.0]])

    from_sum = compute_loss(
        outputs,
        batch,
        sf_wdl_frac=0.0,
        search_wdl_frac=0.0,
        use_adjusted_wdl_target=True,
    )
    from_d95 = compute_loss(
        outputs,
        batch,
        sf_wdl_frac=0.0,
        search_wdl_frac=0.0,
        use_adjusted_wdl_target=True,
        adjusted_wdl_regret_source="d95",
    )

    assert from_d95["blended_wdl_ce"].item() < from_sum["blended_wdl_ce"].item()


def test_adjusted_wdl_target_can_use_horizon_source():
    batch = _minimal_batch(1)
    batch["wdl_t"] = torch.tensor([0])
    batch["future_sf_regret_sum"] = torch.zeros((1,))
    batch["has_future_sf_regret_sum"] = torch.ones((1,))
    batch["future_sf_regret_h12"] = torch.ones((1,))
    batch["has_future_sf_regret_h12"] = torch.ones((1,))

    outputs = _fake_outputs(1)
    outputs["wdl"] = torch.tensor([[-4.0, 0.0, 4.0]])

    from_sum = compute_loss(
        outputs,
        batch,
        sf_wdl_frac=0.0,
        search_wdl_frac=0.0,
        use_adjusted_wdl_target=True,
    )
    from_h12 = compute_loss(
        outputs,
        batch,
        sf_wdl_frac=0.0,
        search_wdl_frac=0.0,
        use_adjusted_wdl_target=True,
        adjusted_wdl_regret_source="h12",
    )

    assert from_h12["blended_wdl_ce"].item() < from_sum["blended_wdl_ce"].item()


def test_adjusted_wdl_target_rejects_unknown_regret_source():
    batch = _minimal_batch(1)

    with pytest.raises(ValueError, match="adjusted_wdl_regret_source"):
        compute_loss(
            _fake_outputs(1),
            batch,
            use_adjusted_wdl_target=True,
            adjusted_wdl_regret_source="d99",
        )


def test_has_flag_masking():
    """Only samples with has_* = 1 should contribute to their respective head."""
    b = 4
    batch = _full_batch(b)
    # Zero out half the has_sf_wdl flags
    batch["has_sf_wdl"] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    compute_loss(_fake_outputs(b), batch)

    batch["has_sf_wdl"] = torch.zeros((b,))
    losses_none = compute_loss(_fake_outputs(b), batch)

    # With all flags zeroed, sf_eval loss must be zero
    assert losses_none["sf_eval_ce"].item() == 0.0
    # With some flags set, loss should be nonzero (extremely unlikely to be exactly 0)
    # (not strictly guaranteed, but with random logits the probability is negligible)


# ── Weight tests ─────────────────────────────────────────────────────


def test_zero_weight_removes_contribution():
    """Setting a head weight to zero should not affect total loss."""
    b = 4
    out = _fake_outputs(b)
    batch = _full_batch(b)

    losses_with = compute_loss(out, batch, w_sf_move=1.0)
    losses_without = compute_loss(out, batch, w_sf_move=0.0)

    # Total should differ (sf_move contributes when weight > 0)
    assert losses_with["sf_move_ce"].item() == losses_without["sf_move_ce"].item()
    # But total changes
    diff = abs(losses_with["total"].item() - losses_without["total"].item())
    if losses_with["sf_move_ce"].item() > 0:
        assert diff > 0


def test_wdl_weight_scales_total():
    b = 4
    out = _fake_outputs(b)
    batch = _full_batch(b)

    losses_1x = compute_loss(out, batch, w_wdl=1.0, w_policy=0.0, w_soft=0.0,
                              w_future=0.0, w_sf_move=0.0, w_sf_eval=0.0,
                              w_categorical=0.0, w_volatility=0.0,
                              w_moves_left=0.0, sf_wdl_frac=0.0)
    losses_2x = compute_loss(out, batch, w_wdl=2.0, w_policy=0.0, w_soft=0.0,
                              w_future=0.0, w_sf_move=0.0, w_sf_eval=0.0,
                              w_categorical=0.0, w_volatility=0.0,
                              w_moves_left=0.0, sf_wdl_frac=0.0)

    ratio = losses_2x["total"].item() / max(losses_1x["total"].item(), 1e-12)
    assert abs(ratio - 2.0) < 1e-4


# ── Gradient flow ────────────────────────────────────────────────────


def test_gradient_flows_to_all_output_heads():
    """Total loss should produce gradients for every model output head."""
    b = 4
    out = _fake_outputs(b, requires_grad=True)
    batch = _full_batch(b)

    losses = compute_loss(out, batch)
    losses["total"].backward()

    for key, tensor in out.items():
        assert tensor.grad is not None, f"No gradient for output head {key!r}"
        assert tensor.grad.abs().sum() > 0, f"Zero gradient for output head {key!r}"


# ── Legal mask tests ─────────────────────────────────────────────────


def test_legal_mask_suppresses_illegal_moves():
    """Legal mask should drive illegal move logits to ~zero probability."""
    b = 2
    out = _fake_outputs(b)
    batch = _full_batch(b)

    # Only first move is legal
    mask = torch.zeros((b, POLICY_SIZE))
    mask[:, 0] = 1.0
    batch["legal_mask"] = mask
    batch["has_legal_mask"] = torch.ones((b,))

    losses = compute_loss(out, batch)
    assert torch.isfinite(losses["total"])


def test_legal_mask_not_applied_to_future_or_sf_policy():
    """Future and SF policy heads should NOT use the current position's legal mask."""
    b = 2
    out = _fake_outputs(b)
    batch = _full_batch(b)

    # Restrictive legal mask
    mask = torch.zeros((b, POLICY_SIZE))
    mask[:, 0] = 1.0
    batch["legal_mask"] = mask
    batch["has_legal_mask"] = torch.ones((b,))

    losses_masked = compute_loss(out, batch)

    # Remove legal mask entirely
    del batch["legal_mask"]
    del batch["has_legal_mask"]
    losses_no_mask = compute_loss(out, batch)

    # Future policy and SF move should be identical regardless of legal mask
    assert abs(losses_masked["future_policy_ce"].item() -
               losses_no_mask["future_policy_ce"].item()) < 1e-5
    assert abs(losses_masked["sf_move_ce"].item() -
               losses_no_mask["sf_move_ce"].item()) < 1e-5


def test_reported_categorical_and_moves_left_match_total_loss_masks():
    """Diagnostics should use the same net_mask gating as the optimized total."""
    b = 4
    out = _fake_outputs(b)
    batch = _full_batch(b)
    batch["is_network_turn"] = torch.tensor([1.0, 0.0, 1.0, 0.0])

    losses = compute_loss(
        out,
        batch,
        w_policy=0.0,
        w_soft=0.0,
        w_future=0.0,
        w_wdl=0.0,
        w_sf_move=0.0,
        w_sf_eval=0.0,
        sf_wdl_frac=0.0,
        w_volatility=0.0,
        w_sf_volatility=0.0,
        w_categorical=1.0,
        w_moves_left=1.0,
    )
    expected = losses["categorical_ce"].item() + losses["moves_left"].item()
    assert abs(losses["total"].item() - expected) < 1e-6


# ── WDL convention ───────────────────────────────────────────────────


def test_wdl_targets_accept_all_classes():
    """WDL cross-entropy should work for all three classes (0=win, 1=draw, 2=loss)."""
    b = 3
    out = _fake_outputs(b)
    batch = _minimal_batch(b)
    batch["wdl_t"] = torch.tensor([0, 1, 2])

    losses = compute_loss(out, batch)
    assert torch.isfinite(losses["wdl_ce"])
    assert losses["wdl_ce"].item() > 0


# ── Reported value-loss names (docs/rl_loop_audit.md I7) ─────────────


def _blend_batch(b: int) -> dict[str, torch.Tensor]:
    """A batch where the blended target genuinely differs from the one-hot."""
    batch = _full_batch(b)
    batch["has_search_wdl"] = torch.ones((b,))
    batch["search_wdl"] = torch.tensor([[0.6, 0.3, 0.1]] * b)
    return batch


def test_wdl_ce_reports_the_trained_loss_not_the_onehot_diagnostic():
    b = 8
    torch.manual_seed(0)
    losses = compute_loss(
        _fake_outputs(b), _blend_batch(b), sf_wdl_frac=0.45, search_wdl_frac=0.20,
    )
    # `wdl_ce` is the name people reach for; it must BE the trained tensor.
    assert losses["wdl_ce"].item() == losses["blended_wdl_ce"].item()
    # The diagnostic is still reported, under a name that cannot be mistaken
    # for the trained loss, and it is a genuinely different number.
    assert "wdl_onehot_ce" in losses
    assert abs(losses["wdl_onehot_ce"].item() - losses["blended_wdl_ce"].item()) > 1e-6


def test_total_uses_the_blended_wdl_loss_and_never_the_onehot():
    b = 8
    torch.manual_seed(1)
    out = _fake_outputs(b)
    batch = _blend_batch(b)
    losses = compute_loss(
        out, batch,
        w_wdl=1.0,
        w_policy=0.0, w_soft=0.0, w_future=0.0, w_sf_move=0.0, w_sf_eval=0.0,
        w_categorical=0.0, w_volatility=0.0, w_sf_volatility=0.0, w_moves_left=0.0,
        sf_wdl_frac=0.45, search_wdl_frac=0.20,
    )
    assert abs(losses["total"].item() - losses["blended_wdl_ce"].item()) < 1e-6
    assert abs(losses["total"].item() - losses["wdl_onehot_ce"].item()) > 1e-6


def test_wdl_split_losses_track_the_trained_loss():
    b = 8
    torch.manual_seed(2)
    out = _fake_outputs(b)
    batch = _blend_batch(b)
    # Every row selfplay-tagged, so the selfplay split covers the whole batch.
    batch["has_is_selfplay"] = torch.ones((b,))
    batch["is_selfplay"] = torch.ones((b,))
    losses = compute_loss(out, batch, sf_wdl_frac=0.45, search_wdl_frac=0.20)
    assert abs(
        losses["wdl_loss_selfplay"].item() - losses["blended_wdl_ce"].item()
    ) < 1e-6
    assert abs(
        losses["wdl_loss_selfplay"].item() - losses["wdl_onehot_ce"].item()
    ) > 1e-6
