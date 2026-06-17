"""Resume rebuilds the model at the checkpoint's topology, not the trial config's.

Regression for the variable-width-FFN resume crash: trainable.py used to build
the model purely from the trial config and then load the checkpoint tolerantly.
When the saved checkpoint carried a non-uniform ffn_mult_by_layer (wide FFN
blocks) but the config had ffn_mult_by_layer=None, the rebuilt model was uniform
width, the tolerant loader silently dropped the wide blocks, and the optimizer
moments crashed at the first step() ("size of tensor a (608) must match (576)").

The fix selects topology from the checkpoint ``arch`` dict while keeping the
four encoding keys config-driven so deliberate warm-start migrations (v1 ->
v2_threats input planes) still work.

Imports ONLY chess_anti_engine.model (the pure ModelConfig helpers) so the test
runs without the compiled _lc0_ext / _mcts_tree C extensions or a GPU.
"""
from __future__ import annotations

import dataclasses

from chess_anti_engine.model import (
    ModelConfig,
    model_config_from_flat_config,
    resume_model_config_from_arch,
)

# The real crashing schedule: blocks 4/5/6/8/9/10 widened.
_WIDE_FFN_SCHEDULE = (
    1.5,
    1.5,
    1.5,
    1.5,
    1.5833333333333333,
    1.75,
    1.9166666666666667,
    1.5,
    1.9166666666666667,
    1.75,
    1.8333333333333333,
)


def _checkpoint_arch() -> dict:
    """A saved ``arch`` dict shaped like Trainer.save writes it.

    Topology is a variable-width FFN model; encoding is v1 / lc0_1858 — i.e. the
    pre-migration checkpoint identity.
    """
    base = ModelConfig(
        embed_dim=384,
        num_layers=11,
        num_heads=8,
        ffn_mult=1.5,
        ffn_mult_by_layer=_WIDE_FFN_SCHEDULE,
        input_extra_features="v1",
        policy_encoding="lc0_1858",
        input_history_encoding="legacy",
        history_rep_fix=False,
    )
    return {"_schema_version": 17, **dataclasses.asdict(base)}


def test_resume_recovers_ffn_topology_keeps_config_encoding() -> None:
    arch = _checkpoint_arch()
    # The trial config dropped the per-layer schedule (uniform 1.5) AND requests
    # a deliberate v1 -> v2_threats input-encoding migration.
    config_cfg = model_config_from_flat_config(
        {
            "embed_dim": 384,
            "num_layers": 11,
            "num_heads": 8,
            "ffn_mult": 1.5,
            "ffn_mult_by_layer": None,
            "input_extra_features": "v2_threats",
            "policy_encoding": "lc0_1858",
        }
    )
    assert config_cfg.ffn_mult_by_layer is None
    assert config_cfg.input_extra_features == "v2_threats"

    out = resume_model_config_from_arch(arch, config_cfg)

    # (a) Topology recovered from the checkpoint arch (the crash root cause).
    assert out.ffn_mult_by_layer == _WIDE_FFN_SCHEDULE
    assert out.embed_dim == 384
    assert out.num_layers == 11

    # (b) Encoding keys preserved from CONFIG so the warm-start migration runs.
    assert out.input_extra_features == "v2_threats"
    assert out.policy_encoding == "lc0_1858"
    assert out.input_history_encoding == "legacy"
    assert out.history_rep_fix is False


def test_resume_recovers_embed_dim_by_layer() -> None:
    schedule = (384, 384, 416, 448, 448, 416, 384, 384, 384)
    arch_cfg = ModelConfig(
        embed_dim=384,
        num_layers=9,
        embed_dim_by_layer=schedule,
        input_extra_features="v1",
    )
    arch = {"_schema_version": 17, **dataclasses.asdict(arch_cfg)}
    config_cfg = model_config_from_flat_config(
        {
            "embed_dim": 384,
            "num_layers": 9,
            "embed_dim_by_layer": None,
            "input_extra_features": "v2_threats",
        }
    )

    out = resume_model_config_from_arch(arch, config_cfg)

    assert out.embed_dim_by_layer == schedule  # topology recovered
    assert out.input_extra_features == "v2_threats"  # warm-start preserved


def test_resume_with_matching_config_is_identical() -> None:
    """No drift, no migration: the result equals the config-built ModelConfig."""
    cfg = ModelConfig(
        embed_dim=384,
        num_layers=11,
        ffn_mult=1.5,
        ffn_mult_by_layer=_WIDE_FFN_SCHEDULE,
        input_extra_features="v1",
        policy_encoding="lc0_1858",
    )
    arch = {"_schema_version": 17, **dataclasses.asdict(cfg)}

    out = resume_model_config_from_arch(arch, cfg)

    assert out == cfg
