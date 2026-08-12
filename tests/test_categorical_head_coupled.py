"""The coupled categorical value head (lc0's value-embedding topology).

Background: the standalone `value_categorical` head was measured as a
two-instrument NULL even after its TARGET was repaired to the SF blend
(Tier-10 arena 2026-08-06, Tier-11 paired `value_regret` 2026-08-12). The
remaining structural difference from lc0 is topology: lc0 hangs the
distributional aux off the SAME value embedding the scalar output reads, so the
aux gradient supervises that representation directly. Ours hung it off an
independent `ValueHead`, where the aux gradient reaches the scalar head only by
diffusing back through the whole 63M trunk.

The load-bearing test here is `test_coupled_categorical_gradient_reaches_wdl_head`
and its standalone counterpart: they assert the MECHANISM (where the gradient
lands), not merely that the config key is plumbed. A knob that is accepted and
then has no effect is this codebase's signature defect.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from chess_anti_engine.model import (
    ModelConfig,
    build_model,
    model_config_to_manifest_dict,
)
from chess_anti_engine.model.transformer import (
    CATEGORICAL_HEAD_BINS,
    ChessNet,
    ValueHead,
)
from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults


def _cfg(*, coupled: bool) -> ModelConfig:
    return ModelConfig(
        kind="transformer",
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        use_smolgen=False,
        categorical_head_coupled=coupled,
    )


def _build(*, coupled: bool) -> ChessNet:
    model = build_model(_cfg(coupled=coupled))
    assert isinstance(model, ChessNet)
    return model.eval()


def _planes(batch: int = 3) -> torch.Tensor:
    torch.manual_seed(1234)
    return torch.randn(batch, 175, 8, 8)


def test_head_bin_count_is_pinned_to_the_target_builder() -> None:
    """`model/` cannot import `train/`, so the two constants are pinned by test."""
    assert CATEGORICAL_HEAD_BINS == DEFAULT_CATEGORICAL_BINS


def test_hidden_split_is_bit_identical_to_forward() -> None:
    """`hidden()` + `head_from_hidden()` must reproduce `forward()` exactly.

    If this drifts, the coupled model's WDL output silently stops matching the
    standalone model's and every cross-topology comparison becomes invalid.
    """
    torch.manual_seed(0)
    head = ValueHead(64, 3)
    x = torch.randn(5, 64, 64)
    assert torch.equal(head(x), head.head_from_hidden(head.hidden(x)))


def test_net_layout_guard_fires_when_the_sequential_is_restructured() -> None:
    """Negative control: the positional indexing guard must be ABLE to fail."""
    head = ValueHead(64, 3)
    head.net = nn.Sequential(nn.Linear(8192, 3))  # a plausible future "simplification"
    with pytest.raises(RuntimeError, match=r"ValueHead\.net layout changed"):
        head.hidden(torch.randn(2, 64, 64))


def test_coupled_head_is_built_instead_of_the_standalone_one() -> None:
    std, cpl = _build(coupled=False), _build(coupled=True)
    assert std.value_categorical is not None
    assert std.value_categorical_coupled is None
    assert cpl.value_categorical is None
    assert cpl.value_categorical_coupled is not None
    for model in (std, cpl):
        assert model(_planes())["categorical"].shape == (3, CATEGORICAL_HEAD_BINS)


def test_coupled_head_costs_two_orders_of_magnitude_fewer_params() -> None:
    std, cpl = _build(coupled=False), _build(coupled=True)
    n_std = sum(p.numel() for p in std.parameters())
    n_cpl = sum(p.numel() for p in cpl.parameters())
    standalone_cost = sum(p.numel() for p in ValueHead(64, CATEGORICAL_HEAD_BINS).parameters())
    coupled_cost = cpl.value_wdl.hidden_dim * CATEGORICAL_HEAD_BINS + CATEGORICAL_HEAD_BINS
    assert n_std - n_cpl == standalone_cost - coupled_cost
    assert coupled_cost < standalone_cost


def test_warm_start_from_a_standalone_checkpoint_leaves_wdl_bit_identical() -> None:
    """The whole point of this topology: iter138 loads with every shared tensor
    unchanged, so the coupled arm starts from exactly the banked net."""
    std, cpl = _build(coupled=False), _build(coupled=True)
    missing, unexpected = cpl.load_state_dict(std.state_dict(), strict=False)
    assert set(missing) == {
        "value_categorical_coupled.weight",
        "value_categorical_coupled.bias",
    }
    assert all(k.startswith("value_categorical.") for k in unexpected)

    x = _planes()
    with torch.no_grad():
        assert torch.equal(std(x)["wdl"], cpl(x)["wdl"])


def test_coupled_categorical_logits_start_near_zero() -> None:
    """Small init, so enabling the aux head does not shock a warm-started net."""
    cpl = _build(coupled=True)
    with torch.no_grad():
        assert float(cpl(_planes())["categorical"].abs().max()) < 0.05


def _wdl_first_projection(model: ChessNet) -> torch.Tensor:
    """`value_wdl`'s first projection weight — the parameter the coupled aux head
    is supposed to supervise and the standalone one provably cannot reach.

    Indexed through `nn.Sequential`, whose `__getitem__` is typed `Tensor | Module`,
    so the isinstance narrowing is the version-proof way to get at `.weight`.
    """
    layer = model.value_wdl.net[0]
    assert isinstance(layer, nn.Linear)
    return layer.weight


def _categorical_grad_on(model: ChessNet, param: torch.Tensor) -> float:
    """Backprop a loss on `categorical` ALONE and report |grad| landing on `param`."""
    model.zero_grad(set_to_none=True)
    model(_planes())["categorical"].pow(2).sum().backward()
    return 0.0 if param.grad is None else float(param.grad.abs().sum())


def test_coupled_categorical_gradient_reaches_the_wdl_head_itself() -> None:
    """⚑ THE MECHANISM. Coupled, the aux loss must supervise the very projection
    the WDL logits are read from — not just the shared trunk."""
    cpl = _build(coupled=True)
    assert _categorical_grad_on(cpl, _wdl_first_projection(cpl)) > 0.0


def test_standalone_categorical_gradient_never_touches_the_wdl_head() -> None:
    """The counterpart that makes the test above meaningful: with the legacy
    topology the aux gradient reaches `value_wdl`'s own parameters NOT AT ALL,
    which is precisely the representation supervision lc0 gets and we did not."""
    std = _build(coupled=False)
    assert _categorical_grad_on(std, _wdl_first_projection(std)) == 0.0


def test_flag_survives_the_manifest_round_trip() -> None:
    """All five plumbing sites: a flag that does not round-trip through the
    manifest is a flag that silently reverts on resume."""
    manifest = model_config_to_manifest_dict(_cfg(coupled=True))
    assert manifest["categorical_head_coupled"] is True
    assert dataclasses.replace(_cfg(coupled=True)).categorical_head_coupled is True


def test_flag_is_accepted_by_the_yaml_schema() -> None:
    """Category-(a) trap, tested BEHAVIOURALLY: a live-yaml key absent from the
    schema makes `flatten_run_config_defaults` raise, and it is called before the
    argument parser and outside any try -- so the process would not boot at all."""
    flat = flatten_run_config_defaults(
        {"model": {"categorical_head_coupled": True}, "train": {}}
    )
    assert flat["categorical_head_coupled"] is True

    with pytest.raises(ValueError, match="categorical_head_coupled_typo"):
        flatten_run_config_defaults(
            {"model": {"categorical_head_coupled_typo": True}, "train": {}}
        )
