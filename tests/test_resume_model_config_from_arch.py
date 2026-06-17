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

import pytest

from chess_anti_engine.encoding.features import RELATION_COUNT
from chess_anti_engine.model import (
    ARCH_SCHEMA_VERSION,
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


def test_resume_preserves_gradient_checkpointing_from_arch() -> None:
    """Trainer.save writes dataclasses.asdict(), so the arch key is
    ``use_gradient_checkpointing``; the manifest loader historically only read
    ``gradient_checkpointing``. The arch value must survive resume (else large
    runs rebuild with checkpointing OFF and can OOM)."""
    arch_cfg = ModelConfig(
        embed_dim=384,
        num_layers=9,
        use_gradient_checkpointing=True,
        input_extra_features="v1",
    )
    arch = {"_schema_version": 17, **dataclasses.asdict(arch_cfg)}
    assert "use_gradient_checkpointing" in arch
    assert "gradient_checkpointing" not in arch  # asdict spelling only

    # Config does not request checkpointing; topology (incl. this memory knob)
    # must still follow the checkpoint.
    config_cfg = model_config_from_flat_config(
        {"embed_dim": 384, "num_layers": 9, "input_extra_features": "v2_threats"}
    )
    assert config_cfg.use_gradient_checkpointing is False

    out = resume_model_config_from_arch(arch, config_cfg)

    assert out.use_gradient_checkpointing is True


def test_resume_enables_dynamic_relations_from_config() -> None:
    """A deliberate restart that turns ON a zero-init dynamic-relations
    experiment must build the NEW topology (params exist for the optimizer
    splice to warm-start), not silently rebuild the donor's relations-off arch."""
    # Donor checkpoint predates the experiment: relations OFF. Its stored count
    # is the dataclass default, which differs from RELATION_COUNT only if the
    # default ever drifts — pin it explicitly so the assertion below proves the
    # count came from CONFIG, not the arch.
    arch_cfg = ModelConfig(
        embed_dim=384,
        num_layers=9,
        use_dynamic_relations=False,
        policy_dynamic_relations=False,
        dynamic_relation_count=RELATION_COUNT,
        input_extra_features="v1",
    )
    arch = {"_schema_version": 17, **dataclasses.asdict(arch_cfg)}
    # Restart config enables the experiment. RELATION_COUNT is the only value
    # build_model() accepts when relations are on (the count is baked into the C
    # extension / shard schema), so it is the only outcome this resume path can
    # legitimately produce — assert it rather than a count that can never build.
    config_cfg = model_config_from_flat_config(
        {
            "embed_dim": 384,
            "num_layers": 9,
            "use_dynamic_relations": True,
            "policy_dynamic_relations": True,
            "dynamic_relation_count": RELATION_COUNT,
            "input_extra_features": "v1",
        }
    )

    out = resume_model_config_from_arch(arch, config_cfg)

    # Experiment flags + their coupled shape follow CONFIG so the params exist.
    assert out.use_dynamic_relations is True
    assert out.policy_dynamic_relations is True
    assert out.dynamic_relation_count == RELATION_COUNT
    # But unrelated topology still follows the checkpoint.
    assert out.num_layers == 9
    assert out.embed_dim == 384


def test_resume_rejects_newer_arch_schema() -> None:
    """A checkpoint written by a newer build must fail loudly rather than have
    its unknown shape fields silently defaulted (mirrors the UCI loader)."""
    cfg = ModelConfig(embed_dim=384, num_layers=9, input_extra_features="v1")
    arch = {"_schema_version": ARCH_SCHEMA_VERSION + 1, **dataclasses.asdict(cfg)}

    with pytest.raises(ValueError, match="newer than this build"):
        resume_model_config_from_arch(arch, cfg)


def test_resume_rejects_unknown_arch_keys() -> None:
    """An unrecognised ModelConfig key (a future topology field) must fail loudly
    so the tolerant loader can't quietly skip the matching tensors."""
    cfg = ModelConfig(embed_dim=384, num_layers=9, input_extra_features="v1")
    arch = {"_schema_version": ARCH_SCHEMA_VERSION, **dataclasses.asdict(cfg)}
    arch["some_future_topology_field"] = 42

    with pytest.raises(ValueError, match="unknown ModelConfig keys"):
        resume_model_config_from_arch(arch, cfg)


def test_resume_tolerates_missing_schema_version() -> None:
    """Older checkpoints predate the ``_schema_version`` tag; resume must still
    work for them (config-driven fallback), so a missing tag is not an error."""
    cfg = ModelConfig(embed_dim=384, num_layers=9, input_extra_features="v1")
    arch = dataclasses.asdict(cfg)  # no _schema_version
    assert "_schema_version" not in arch

    out = resume_model_config_from_arch(arch, cfg)

    assert out == cfg
