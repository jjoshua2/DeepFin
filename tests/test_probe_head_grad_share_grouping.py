"""The grad-share probe's groups must partition the TRAINED loss components.

docs/rl_loop_audit.md I3: the first measurement reported POLICY 65.09% /
VALUE 15.51% / OTHER 0.02% — a sum of 80.62%, because the gradient-free
`wdl_ce` diagnostic sat in the weighted denominator but in no group.
"""
from __future__ import annotations

import inspect

import torch

from chess_anti_engine.train.losses import compute_loss
from scripts.probe_head_grad_share import (
    COMPONENT_WEIGHT_KEY,
    DIAGNOSTIC_COMPONENTS,
    GROUPS,
    check_grouping_partitions_components,
)


def test_groups_partition_the_trained_components() -> None:
    check_grouping_partitions_components()
    grouped = [comp for members in GROUPS.values() for comp in members]
    assert sorted(grouped) == sorted(COMPONENT_WEIGHT_KEY)
    assert len(grouped) == len(set(grouped))


def test_the_diagnostic_is_not_a_weighted_component() -> None:
    for comp in DIAGNOSTIC_COMPONENTS:
        assert comp not in COMPONENT_WEIGHT_KEY
        assert all(comp not in members for members in GROUPS.values())


def test_the_alias_key_is_not_double_counted() -> None:
    """`wdl_ce` and `blended_wdl_ce` are the same tensor; only one may count."""
    assert "blended_wdl_ce" in COMPONENT_WEIGHT_KEY
    assert "wdl_ce" not in COMPONENT_WEIGHT_KEY


def test_weight_keys_are_real_compute_loss_parameters() -> None:
    params = inspect.signature(compute_loss).parameters
    for comp, weight_key in COMPONENT_WEIGHT_KEY.items():
        assert weight_key in params, f"{comp} maps to unknown weight {weight_key}"


def test_components_and_diagnostics_are_keys_compute_loss_returns() -> None:
    b = 2
    batch = {
        "x": torch.randn((b, 146, 8, 8)),
        "policy_t": torch.full((b, 1858), 1.0 / 1858.0),
        "wdl_t": torch.randint(0, 3, (b,)),
        "has_policy": torch.ones((b,)),
    }
    outputs = {
        "policy_own": torch.randn((b, 1858)),
        "policy_soft": torch.randn((b, 1858)),
        "policy_sf": torch.randn((b, 1858)),
        "policy_future": torch.randn((b, 1858)),
        "wdl": torch.randn((b, 3)),
        "sf_eval": torch.randn((b, 3)),
        "categorical": torch.randn((b, 32)),
        "volatility": torch.rand((b, 3)),
        "sf_volatility": torch.rand((b, 3)),
        "moves_left": torch.rand((b, 1)),
    }
    keys = set(compute_loss(outputs, batch))
    missing = sorted((set(COMPONENT_WEIGHT_KEY) | set(DIAGNOSTIC_COMPONENTS)) - keys)
    assert not missing, f"probe measures keys compute_loss does not return: {missing}"


def test_the_coupled_categorical_head_is_not_counted_as_trunk() -> None:
    """⚑ `_classify_params` splits on the FIRST dotted component and tests set
    membership, so `value_categorical_coupled` is NOT covered by the
    `value_categorical` entry — it is a distinct top-level attribute.

    Left out of HEAD_PREFIXES it lands in the TRUNK, and the probe then reports
    the value head's own 32-way Linear as trunk gradient, inflating the very
    value-vs-policy trunk share this instrument exists to measure.
    """
    from chess_anti_engine.model import ModelConfig, build_model
    from scripts.probe_head_grad_share import _classify_params

    for coupled in (True, False):
        model = build_model(ModelConfig(
            kind="transformer", embed_dim=64, num_layers=2, num_heads=4,
            use_smolgen=False, categorical_head_coupled=coupled,
        ))
        _trunk, trunk_names, head_names, _pol_shared = _classify_params(model)
        categorical = [n for n in trunk_names + head_names if n.startswith("value_categorical")]
        assert categorical, "the categorical head has no parameters at all"
        assert not [n for n in trunk_names if n.startswith("value_categorical")], (
            f"coupled={coupled}: categorical params classified as TRUNK: "
            f"{[n for n in trunk_names if n.startswith('value_categorical')]}"
        )


def test_the_shared_policy_adapter_is_NOT_counted_as_trunk() -> None:
    """⚑ The instrument this experiment is READ OUT ON must not fold a policy-only
    module into the trunk.

    `_classify_params` treats anything unlisted as shared trunk, so `policy_embedding`
    would land there and inflate the very "trunk share" the probe reports -- the same
    defect `value_categorical_coupled` had one PR earlier. It is its own bucket rather
    than a head, because lumping it with private head params biases trunk share DOWN
    and lumping it with the trunk biases it UP, and how much lands on it IS the
    question.

    ⚑ Also pins that pre-adapter numbers (the 2026-08-12 soft 94.1% / own 94.7% pair)
    are NOT the same measurement as post-adapter ones.
    """
    from chess_anti_engine.model import ModelConfig, build_model
    from scripts.probe_head_grad_share import _classify_params

    model = build_model(ModelConfig(
        kind="transformer", embed_dim=32, num_layers=2, num_heads=4,
        use_smolgen=False, policy_embedding_mode="residual_mish",
    ))
    _trunk, trunk_names, head_names, pol_shared = _classify_params(model)

    assert sorted(pol_shared) == ["policy_embedding.bias", "policy_embedding.weight"]
    assert not [n for n in trunk_names if n.startswith("policy_embedding")]
    assert not [n for n in head_names if n.startswith("policy_embedding")]


def test_the_shared_adapter_grad_norm_is_MEASURED_not_just_classified() -> None:
    """⚑ Classification alone is not a measurement.

    Keeping `policy_embedding` out of the trunk bucket stops it inflating
    `trunk_share`, but the shared-adapter experiment's central question is how much
    supervision lands ON that representation. A probe that classifies it correctly
    and then reports nothing about it cannot read the experiment out.

    Asserted behaviourally on real gradients: a loss that depends on the adapter
    must give it a POSITIVE norm, and one that does not must give exactly 0.0.
    """
    import torch

    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.model.transformer import ChessNet
    from scripts.probe_head_grad_share import _classify_params, component_grad_norms

    model = build_model(ModelConfig(
        kind="transformer", embed_dim=32, num_layers=2, num_heads=4,
        use_smolgen=False, policy_embedding_mode="residual_mish",
    ))
    assert isinstance(model, ChessNet)
    with torch.no_grad():  # off the zero init, or every norm is trivially 0
        emb = model.policy_embedding
        assert emb is not None
        emb.weight.normal_(std=0.05)

    trunk, _tn, _hn, shared_names = _classify_params(model)
    trunk_params = [p for _, p in trunk]
    by_name = dict(model.named_parameters())
    shared_params = [by_name[n] for n in shared_names]
    assert shared_params

    out = model(torch.randn(2, 175, 8, 8))
    trunk_g, shared_g = component_grad_norms(
        out["policy_soft"].square().sum(), trunk_params, shared_params,
    )
    assert trunk_g > 0.0
    assert shared_g > 0.0, "the adapter's gradient is not being measured"

    # Negative control: a value head does not read the adapter, so 0.0 is a
    # MEASURED zero rather than a missing measurement.
    trunk_v, shared_v = component_grad_norms(
        model(torch.randn(2, 175, 8, 8))["wdl"].square().sum(),
        trunk_params, shared_params,
    )
    assert trunk_v > 0.0
    assert shared_v == 0.0


def test_component_grad_norms_returns_zero_when_the_model_has_no_adapter() -> None:
    """Callers must not need a branch: `mode: off` builds no layer at all."""
    import torch

    from chess_anti_engine.model import ModelConfig, build_model
    from scripts.probe_head_grad_share import _classify_params, component_grad_norms

    model = build_model(ModelConfig(
        kind="transformer", embed_dim=32, num_layers=2, num_heads=4, use_smolgen=False,
    ))
    trunk, _tn, _hn, shared_names = _classify_params(model)
    assert not shared_names
    trunk_g, shared_g = component_grad_norms(
        model(torch.randn(2, 175, 8, 8))["policy_own"].square().sum(),
        [p for _, p in trunk], [],
    )
    assert trunk_g > 0.0
    assert shared_g == 0.0
