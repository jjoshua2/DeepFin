"""The shared broker's architecture-identity guard.

`SlotBroker` refuses to load a published model whose architecture differs from
the one it already cached ("different model config, skipping"). That guard was a
hand-enumerated tuple of 35 of ModelConfig's 41 fields, and it had silently
drifted: the five omissions all change tensor shapes, so the guard could not
fail for exactly the mismatches it exists to catch.

These tests pin the derived replacement. The completeness test is the important
one — it makes a newly added ModelConfig field default to BEING identity, and
turns "someone forgot to update the tuple" from a silent hole into a red test.
"""
from __future__ import annotations

import dataclasses

import pytest

from chess_anti_engine.inference import (
    _NON_IDENTITY_MODEL_CONFIG_FIELDS,
    model_config_identity_key,
)
from chess_anti_engine.model import ModelConfig


def _cfg(**kwargs: object) -> ModelConfig:
    base = ModelConfig(kind="transformer", embed_dim=64, num_layers=2, num_heads=4)
    return dataclasses.replace(base, **kwargs)


def test_every_model_config_field_is_identity_or_explicitly_excluded() -> None:
    """⚑ THE GATE. A new ModelConfig field must either appear in the key or be
    named as a deliberate exclusion — never fall through unnoticed."""
    names = {f.name for f in dataclasses.fields(ModelConfig)}
    assert names >= _NON_IDENTITY_MODEL_CONFIG_FIELDS, "exclusion names a field that no longer exists"
    assert len(model_config_identity_key(_cfg())) == len(names - _NON_IDENTITY_MODEL_CONFIG_FIELDS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # the field this PR adds
        ("categorical_head_coupled", True),
        # narrows policy_soft/policy_sf q/k — a broker that treated two aux
        # widths as "the same config" would load one into the other
        ("aux_policy_head_dim", 128),
        # the five the hand-written tuple had silently dropped, all shape-changing
        ("input_extra_features", "v1"),
        ("history_rep_fix", True),
        ("use_dynamic_relations", True),
        ("policy_dynamic_relations", True),
        ("dynamic_relation_count", 7),
        # a control from the fields the tuple always had
        ("num_layers", 3),
    ],
)
def test_key_separates_architectures_that_differ_in_one_field(field: str, value: object) -> None:
    """Failure scenario if it does not: a broker holding a v1 (146-plane) model
    accepts a v2_threats (175-plane) publish as "the same config" and loads the
    state dict into the wrong architecture."""
    assert model_config_identity_key(_cfg()) != model_config_identity_key(_cfg(**{field: value}))


def test_gradient_checkpointing_is_deliberately_not_identity() -> None:
    """It is normalized to False on both sides before the key is built and
    changes no tensor shape, so it must NOT split otherwise-identical models."""
    a = model_config_identity_key(_cfg(use_gradient_checkpointing=False))
    b = model_config_identity_key(_cfg(use_gradient_checkpointing=True))
    assert a == b


def test_identical_configs_agree() -> None:
    assert model_config_identity_key(_cfg()) == model_config_identity_key(_cfg())
