from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from chess_anti_engine.encoding import encode_position, input_plane_count
from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding.features import (
    EXTRA_FEATURES_V1,
    EXTRA_FEATURES_V2_THREATS,
    RELATION_COUNT,
    normalize_extra_features_encoding,
)
from chess_anti_engine.encoding.lc0 import LC0_HISTORY_LEGACY, normalize_lc0_history_encoding
from chess_anti_engine.moves import MODEL_POLICY_ENCODING
from chess_anti_engine.utils.architecture import (
    DEFAULT_PHASE_PIECE_THRESHOLDS,
    normalize_embed_dim_by_layer,
    normalize_ffn_mult_by_layer,
    normalize_phase_piece_thresholds,
)

from .tiny import TinyNet
from .transformer import ChessNet, TransformerConfig

# Bump this when ModelConfig gains a field that a defaulted value would
# misrepresent. Trainer embeds this version when saving; the UCI loader
# rejects checkpoints with a higher version AND rejects unknown keys at
# the same version — both prevent silent architecture mismatch on skew.
# v17: history_rep_fix (defaulting it would silently feed a rep-fix-trained
# model legacy repetition planes at eval time).
ARCH_SCHEMA_VERSION = 17


@dataclass
class ModelConfig:
    kind: str = "transformer"  # tiny|transformer
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    # When set, this schedule defines each trunk layer's width. ``embed_dim``
    # remains the uniform-width fallback and manifest compatibility field.
    embed_dim_by_layer: tuple[int, ...] | None = None
    ffn_mult: float = 2.0
    ffn_mult_by_layer: tuple[float, ...] | None = None
    use_smolgen: bool = True
    use_nla: bool = False
    use_qk_rmsnorm: bool = False
    use_gradient_checkpointing: bool = False
    input_pos_encoding: str = "none"
    qkv_projection: str = "fused"
    use_deepnorm: bool = False
    policy_encoding: str = MODEL_POLICY_ENCODING
    input_history_encoding: str = LC0_HISTORY_LEGACY
    # Gated candidate (see encoding/rep_fix.py): the model was trained on
    # corrected lc0-root repetition planes, so eval-time encoding must apply
    # the same fix. Part of checkpoint/manifest identity — a legacy default
    # would silently feed such a model legacy planes.
    history_rep_fix: bool = False
    input_extra_features: str = EXTRA_FEATURES_V2_THREATS
    use_dynamic_relations: bool = False
    dynamic_relation_count: int = 5
    policy_dynamic_relations: bool = False
    input_global_embedding: str = "none"
    input_global_embedding_channels: int = 0
    input_square_embedding: str = "none"
    smolgen_mode: str = "shared"
    smolgen_pooling: str = "flatten"
    smolgen_hidden_channels: int = 32
    smolgen_hidden_sz: int = 256
    smolgen_gen_sz: int = 256
    smolgen_bias_scale: str = "none"
    smolgen_bias_norm: str = "none"
    arc_attention_bias: str = "none"
    smolgen_relation_basis: bool = False
    smolgen_relation_norm: str = "none"
    smolgen_relation_coeff_norm: str = "none"
    smolgen_relation_scale: str = "none"
    phase_output_adapter: bool = False
    phase_output_adapter_dim: int = 64
    phase_smolgen: bool = False
  # Attach the categorical value output to value_wdl's hidden activation
  # (lc0 value-embedding topology) instead of a standalone ValueHead. Part of
  # checkpoint identity: the two topologies have different state_dict keys.
    categorical_head_coupled: bool = False
    phase_piece_thresholds: tuple[int, int] = DEFAULT_PHASE_PIECE_THRESHOLDS


def model_config_from_manifest_dict(mc: dict) -> ModelConfig:
    """Build a ModelConfig from the ``model_config`` block of a publish manifest.

    Manifest field name is ``gradient_checkpointing`` (not ``use_*``) for
    historical reasons; everything else maps 1:1. The resume path also feeds
    this with ``dataclasses.asdict(ModelConfig)`` (what ``Trainer.save`` embeds
    as ``arch``), where the field is spelled ``use_gradient_checkpointing`` — so
    accept either spelling, preferring the dataclass key, else the model would
    silently rebuild with checkpointing OFF on resume and large runs could OOM.
    """
    num_layers = int(mc.get("num_layers", 6))
    return ModelConfig(
        kind=str(mc.get("kind", "transformer")),
        embed_dim=int(mc.get("embed_dim", 256)),
        num_layers=num_layers,
        num_heads=int(mc.get("num_heads", 8)),
        embed_dim_by_layer=normalize_embed_dim_by_layer(
            mc.get("embed_dim_by_layer"),
            num_layers=num_layers,
        ),
        ffn_mult=float(mc.get("ffn_mult", 2)),
        ffn_mult_by_layer=normalize_ffn_mult_by_layer(
            mc.get("ffn_mult_by_layer"),
            num_layers=num_layers,
        ),
        use_smolgen=bool(mc.get("use_smolgen", True)),
        use_nla=bool(mc.get("use_nla", False)),
        use_qk_rmsnorm=bool(mc.get("use_qk_rmsnorm", False)),
        use_gradient_checkpointing=bool(
            mc.get("use_gradient_checkpointing", mc.get("gradient_checkpointing", False))
        ),
        input_pos_encoding=str(mc.get("input_pos_encoding", "none")),
        qkv_projection=str(mc.get("qkv_projection", "fused")),
        use_deepnorm=bool(mc.get("use_deepnorm", False)),
        # Always compact; model heads ignore any other declared width.
        policy_encoding=MODEL_POLICY_ENCODING,
        input_history_encoding=normalize_lc0_history_encoding(mc.get("input_history_encoding", LC0_HISTORY_LEGACY)),
        history_rep_fix=bool(mc.get("history_rep_fix", False)),
        input_extra_features=normalize_extra_features_encoding(mc.get("input_extra_features", EXTRA_FEATURES_V1)),
        use_dynamic_relations=bool(mc.get("use_dynamic_relations", False)),
        dynamic_relation_count=int(mc.get("dynamic_relation_count", 5)),
        policy_dynamic_relations=bool(mc.get("policy_dynamic_relations", False)),
        input_global_embedding=str(mc.get("input_global_embedding", "none")),
        input_global_embedding_channels=int(mc.get("input_global_embedding_channels", 0)),
        input_square_embedding=str(mc.get("input_square_embedding", "none")),
        smolgen_mode=str(mc.get("smolgen_mode", "shared")),
        smolgen_pooling=str(mc.get("smolgen_pooling", "flatten")),
        smolgen_hidden_channels=int(mc.get("smolgen_hidden_channels", 32)),
        smolgen_hidden_sz=int(mc.get("smolgen_hidden_sz", 256)),
        smolgen_gen_sz=int(mc.get("smolgen_gen_sz", 256)),
        smolgen_bias_scale=str(mc.get("smolgen_bias_scale", "none")),
        smolgen_bias_norm=str(mc.get("smolgen_bias_norm", "none")),
        arc_attention_bias=str(mc.get("arc_attention_bias", "none")),
        smolgen_relation_basis=bool(mc.get("smolgen_relation_basis", False)),
        smolgen_relation_norm=str(mc.get("smolgen_relation_norm", "none")),
        smolgen_relation_coeff_norm=str(mc.get("smolgen_relation_coeff_norm", "none")),
        smolgen_relation_scale=str(mc.get("smolgen_relation_scale", "none")),
        phase_output_adapter=bool(mc.get("phase_output_adapter", False)),
        phase_output_adapter_dim=int(mc.get("phase_output_adapter_dim", 64)),
        phase_smolgen=bool(mc.get("phase_smolgen", False)),
        categorical_head_coupled=bool(mc.get("categorical_head_coupled", False)),
        phase_piece_thresholds=normalize_phase_piece_thresholds(mc.get("phase_piece_thresholds")),
    )


# Fields a deliberate warm-start migration changes on resume. These MUST follow
# the run config, not the donor checkpoint's saved arch, so the model is built
# with the NEW layout and the tolerant loader / optimizer remap can zero-init
# the added tensors. Everything else in ModelConfig is SHAPE/topology and must
# follow the checkpoint, so the rebuilt model matches the saved tensors and the
# optimizer moments load.
#
# Classes:
#   * Input encoding identity (e.g. v1 -> v2_threats input planes): the tolerant
#     loader keeps the rebuilt model's zero-init columns for the widened input
#     projection, and Trainer.load -> migrate_optimizer_input_plane_state
#     zero-pads the matching optimizer moments. This IS a warm start.
#   * policy_encoding is fixed to compact lc0_1858 on every build (manifest
#     identity only). Search expands compact→full 4672 action ids at MCTS.
#   * Zero-init experiment flags whose freshly added parameters are handled by
#     Trainer._remap_optimizer_state_for_new_params (the dynamic_relation_weight
#     / policy_relation_weight splice). If these followed the donor arch instead,
#     enabling the experiment on resume would silently rebuild the OLD topology
#     and drop the requested experiment.
# input_pos_encoding=arc_adapter is deliberately NOT here: its pos_adapter param
# has no remap machinery, so it is treated as a topology change (checkpoint-owned).
_RESUME_CONFIG_OWNED_ENCODING_KEYS = (
    "input_extra_features",
    "policy_encoding",
    "input_history_encoding",
    "history_rep_fix",
    "use_dynamic_relations",
    "policy_dynamic_relations",
    # Shape of the (zero-init) relation weights, so a freshly enabled
    # use_dynamic_relations builds them at the config's count rather than the
    # donor's stale default.
    "dynamic_relation_count",
)


def resume_model_config_from_arch(arch: dict, config_cfg: ModelConfig) -> ModelConfig:
    """Pick a resume ModelConfig: topology from the checkpoint ``arch`` dict,
    encoding identity from the run config.

    On resume, the rebuilt model must match the checkpoint's saved tensor
    shapes (else the tolerant loader silently drops the mismatched blocks and
    the optimizer moments crash at ``step()`` — the variable-width-FFN resume
    bug). So every shape/topology field comes from ``arch``. The keys in
    ``_RESUME_CONFIG_OWNED_ENCODING_KEYS`` stay config-driven so a deliberate
    warm-start migration (v1 -> v2_threats input planes, or enabling a
    dynamic-relations experiment) still works: the model is built with the NEW
    layout and the tolerant loader / optimizer remap zero-inits the added
    tensors.

    ``arch`` is the dict written by ``Trainer.save`` (``dataclasses.asdict(
    ModelConfig)`` plus a ``_schema_version`` tag), which
    ``model_config_from_manifest_dict`` round-trips — including the
    ``use_gradient_checkpointing`` field that the manifest spells
    ``gradient_checkpointing``.

    Like the UCI loader (``uci.model_loader.model_config_from_arch``), reject a
    checkpoint whose ``_schema_version`` is newer than this build or that carries
    unknown topology keys, instead of letting ``model_config_from_manifest_dict``
    silently default them and hide the drift behind the tolerant state-dict
    loader — the exact silent partial-load this arch-peek path exists to prevent.
    """
    _reject_incompatible_arch_schema(arch)
    topo = model_config_from_manifest_dict(arch)
    overrides = {k: getattr(config_cfg, k) for k in _RESUME_CONFIG_OWNED_ENCODING_KEYS}
    return dataclasses.replace(topo, **overrides)


def _reject_incompatible_arch_schema(arch: dict) -> None:
    """Fail loudly on a forward-incompatible resume ``arch`` (mirrors the UCI loader).

    A missing/non-int ``_schema_version`` is tolerated (older checkpoints predate
    the tag and ``peek_checkpoint_arch`` returns them for the config-driven
    fallback); a NEWER version or an unknown ModelConfig key is not, because
    ``model_config_from_manifest_dict`` would otherwise default the unknown shape
    fields and the tolerant loader would skip the new tensors — reintroducing the
    silent topology/optimizer-state mismatch this path prevents.
    """
    version = arch.get("_schema_version")
    if isinstance(version, int) and version > ARCH_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint arch schema v{version} is newer than this build "
            f"(v{ARCH_SCHEMA_VERSION}); upgrade chess_anti_engine before resuming"
        )
    valid = {f.name for f in dataclasses.fields(ModelConfig)}
    unknown = {k for k in arch if k != "_schema_version"} - valid
    if unknown:
        raise ValueError(
            f"checkpoint arch has unknown ModelConfig keys {sorted(unknown)}; this "
            "build would silently default them away on resume — upgrade the package "
            "or re-save the checkpoint"
        )


def model_config_from_flat_config(
    cfg: dict,
    *,
    embed_dim_default: int = 384,
    num_layers_default: int = 9,
    num_heads_default: int = 8,
) -> ModelConfig:
    """Build a ModelConfig from a flattened run-config dict (``configs/*.yaml``).

    Distinct from ``model_config_from_manifest_dict``: the flat run config keys
    the model kind under ``model`` (not ``kind``) and may carry the legacy
    ``no_smolgen`` negation. Used by the offline diagnostic / bootstrap / replay
    scripts that rebuild a model from a YAML config rather than a publish
    manifest. ``policy_encoding`` / ``input_history_encoding`` are normalized
    to canonical names, matching ``model_config_from_manifest_dict``. The
    dimension defaults are explicit because a few scripts historically used
    smaller fallback models when those keys were absent.
    """
    num_layers = int(cfg.get("num_layers", num_layers_default))
    return ModelConfig(
        kind=str(cfg.get("model", "transformer")),
        embed_dim=int(cfg.get("embed_dim", embed_dim_default)),
        num_layers=num_layers,
        num_heads=int(cfg.get("num_heads", num_heads_default)),
        embed_dim_by_layer=normalize_embed_dim_by_layer(
            cfg.get("embed_dim_by_layer"),
            num_layers=num_layers,
        ),
        ffn_mult=float(cfg.get("ffn_mult", 2.0)),
        ffn_mult_by_layer=normalize_ffn_mult_by_layer(
            cfg.get("ffn_mult_by_layer"),
            num_layers=num_layers,
        ),
        use_smolgen=bool(cfg.get("use_smolgen", not bool(cfg.get("no_smolgen", False)))),
        use_nla=bool(cfg.get("use_nla", False)),
        use_qk_rmsnorm=bool(cfg.get("use_qk_rmsnorm", False)),
        use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        input_pos_encoding=str(cfg.get("input_pos_encoding", "none")),
        qkv_projection=str(cfg.get("qkv_projection", "fused")),
        use_deepnorm=bool(cfg.get("use_deepnorm", False)),
        policy_encoding=MODEL_POLICY_ENCODING,
        input_history_encoding=normalize_lc0_history_encoding(
            cfg.get("input_history_encoding", LC0_HISTORY_LEGACY)
        ),
        history_rep_fix=bool(cfg.get("history_rep_fix", False)),
        input_extra_features=normalize_extra_features_encoding(
            cfg.get("input_extra_features", EXTRA_FEATURES_V2_THREATS)
        ),
        use_dynamic_relations=bool(cfg.get("use_dynamic_relations", False)),
        dynamic_relation_count=int(cfg.get("dynamic_relation_count", 5)),
        policy_dynamic_relations=bool(cfg.get("policy_dynamic_relations", False)),
        input_global_embedding=str(cfg.get("input_global_embedding", "none")),
        input_global_embedding_channels=int(cfg.get("input_global_embedding_channels", 0)),
        input_square_embedding=str(cfg.get("input_square_embedding", "none")),
        smolgen_mode=str(cfg.get("smolgen_mode", "shared")),
        smolgen_pooling=str(cfg.get("smolgen_pooling", "flatten")),
        smolgen_hidden_channels=int(cfg.get("smolgen_hidden_channels", 32)),
        smolgen_hidden_sz=int(cfg.get("smolgen_hidden_sz", 256)),
        smolgen_gen_sz=int(cfg.get("smolgen_gen_sz", 256)),
        smolgen_bias_scale=str(cfg.get("smolgen_bias_scale", "none")),
        smolgen_bias_norm=str(cfg.get("smolgen_bias_norm", "none")),
        arc_attention_bias=str(cfg.get("arc_attention_bias", "none")),
        smolgen_relation_basis=bool(cfg.get("smolgen_relation_basis", False)),
        smolgen_relation_norm=str(cfg.get("smolgen_relation_norm", "none")),
        smolgen_relation_coeff_norm=str(cfg.get("smolgen_relation_coeff_norm", "none")),
        smolgen_relation_scale=str(cfg.get("smolgen_relation_scale", "none")),
        phase_output_adapter=bool(cfg.get("phase_output_adapter", False)),
        phase_output_adapter_dim=int(cfg.get("phase_output_adapter_dim", 64)),
        phase_smolgen=bool(cfg.get("phase_smolgen", False)),
        categorical_head_coupled=bool(cfg.get("categorical_head_coupled", False)),
        phase_piece_thresholds=normalize_phase_piece_thresholds(
            cfg.get("phase_piece_thresholds")
        ),
    )


def model_config_to_manifest_dict(cfg: ModelConfig) -> dict:
    """Inverse of ``model_config_from_manifest_dict``.

    Use when writing the manifest's ``model_config`` block, so encode and
    decode stay in sync as ModelConfig fields evolve.
    """
    ffn_mult_by_layer = normalize_ffn_mult_by_layer(
        cfg.ffn_mult_by_layer,
        num_layers=int(cfg.num_layers),
    )
    embed_dim_by_layer = normalize_embed_dim_by_layer(
        cfg.embed_dim_by_layer,
        num_layers=int(cfg.num_layers),
    )
    return {
        "kind": str(cfg.kind),
        "embed_dim": int(cfg.embed_dim),
        "num_layers": int(cfg.num_layers),
        "num_heads": int(cfg.num_heads),
        "embed_dim_by_layer": list(embed_dim_by_layer) if embed_dim_by_layer is not None else None,
        "ffn_mult": float(cfg.ffn_mult),
        "ffn_mult_by_layer": list(ffn_mult_by_layer) if ffn_mult_by_layer is not None else None,
        "use_smolgen": bool(cfg.use_smolgen),
        "use_nla": bool(cfg.use_nla),
        "use_qk_rmsnorm": bool(cfg.use_qk_rmsnorm),
        "gradient_checkpointing": bool(cfg.use_gradient_checkpointing),
        "input_pos_encoding": str(cfg.input_pos_encoding),
        "qkv_projection": str(cfg.qkv_projection),
        "use_deepnorm": bool(cfg.use_deepnorm),
        "policy_encoding": MODEL_POLICY_ENCODING,
        "input_history_encoding": normalize_lc0_history_encoding(cfg.input_history_encoding),
        "history_rep_fix": bool(cfg.history_rep_fix),
        "input_extra_features": normalize_extra_features_encoding(cfg.input_extra_features),
        "use_dynamic_relations": bool(cfg.use_dynamic_relations),
        "dynamic_relation_count": int(cfg.dynamic_relation_count),
        "policy_dynamic_relations": bool(cfg.policy_dynamic_relations),
        "input_global_embedding": str(cfg.input_global_embedding),
        "input_global_embedding_channels": int(cfg.input_global_embedding_channels),
        "input_square_embedding": str(cfg.input_square_embedding),
        "smolgen_mode": str(cfg.smolgen_mode),
        "smolgen_pooling": str(cfg.smolgen_pooling),
        "smolgen_hidden_channels": int(cfg.smolgen_hidden_channels),
        "smolgen_hidden_sz": int(cfg.smolgen_hidden_sz),
        "smolgen_gen_sz": int(cfg.smolgen_gen_sz),
        "smolgen_bias_scale": str(cfg.smolgen_bias_scale),
        "smolgen_bias_norm": str(cfg.smolgen_bias_norm),
        "arc_attention_bias": str(cfg.arc_attention_bias),
        "smolgen_relation_basis": bool(cfg.smolgen_relation_basis),
        "smolgen_relation_norm": str(cfg.smolgen_relation_norm),
        "smolgen_relation_coeff_norm": str(cfg.smolgen_relation_coeff_norm),
        "smolgen_relation_scale": str(cfg.smolgen_relation_scale),
        "phase_output_adapter": bool(cfg.phase_output_adapter),
        "phase_output_adapter_dim": int(cfg.phase_output_adapter_dim),
        "phase_smolgen": bool(cfg.phase_smolgen),
        "categorical_head_coupled": bool(cfg.categorical_head_coupled),
        "phase_piece_thresholds": list(normalize_phase_piece_thresholds(cfg.phase_piece_thresholds)),
    }


def infer_input_planes(input_extra_features: str | None = None) -> int:
  # Use startpos to infer plane count.
    import chess

    b = chess.Board()
    x = encode_position(b, add_features=True, input_extra_features=input_extra_features)
    return int(x.shape[0])


def _attach_runtime_model_metadata(model: torch.nn.Module, cfg: ModelConfig) -> torch.nn.Module:
    setattr(model, "policy_encoding", MODEL_POLICY_ENCODING)
    setattr(model, "input_history_encoding", normalize_lc0_history_encoding(cfg.input_history_encoding))
    setattr(model, "history_rep_fix", bool(cfg.history_rep_fix))
    setattr(model, "input_extra_features", normalize_extra_features_encoding(cfg.input_extra_features))
    setattr(model, "use_dynamic_relations", bool(cfg.use_dynamic_relations))
    # The repetition-plane fix is process-global in the C encoders, so apply
    # it here — the one chokepoint every loader (trainer, worker, UCI, arena,
    # inference) passes through. Selfplay re-applies per batch from
    # GameConfig with the same trial value. Two models with different flag
    # values cannot share a process; cross-encoding arenas already run each
    # side as its own engine subprocess.
    # boards_discarded: a model is built before it evaluates anything, and the
    # only in-process cross-flag case (an arena loading two checkpoints) rebuilds
    # its CBoards per search call — see selfplay/match.py.
    rep_fix.apply(bool(cfg.history_rep_fix), boards_discarded=True)
    return model


def build_model(cfg: ModelConfig) -> torch.nn.Module:
    if (cfg.use_dynamic_relations or cfg.policy_dynamic_relations) and (
        int(cfg.dynamic_relation_count) != RELATION_COUNT
    ):
  # The relation count is baked into the C computation, the shard field
  # shape, and every transport buffer — reject mismatches at build time
  # instead of failing at the first einsum.
        raise ValueError(
            f"dynamic_relation_count must be {RELATION_COUNT} "
            f"(got {cfg.dynamic_relation_count}); the relation set is fixed "
            "by the C extension / shard schema"
        )
    in_planes = infer_input_planes(cfg.input_extra_features)
    if cfg.kind == "tiny":
        return _attach_runtime_model_metadata(TinyNet(in_planes=in_planes), cfg)
    if cfg.kind == "transformer":
        tcfg = TransformerConfig(
            in_planes=in_planes,
            embed_dim=int(cfg.embed_dim),
            num_layers=int(cfg.num_layers),
            num_heads=int(cfg.num_heads),
            embed_dim_by_layer=normalize_embed_dim_by_layer(
                cfg.embed_dim_by_layer,
                num_layers=int(cfg.num_layers),
            ),
            ffn_mult=float(cfg.ffn_mult),
            ffn_mult_by_layer=normalize_ffn_mult_by_layer(
                cfg.ffn_mult_by_layer,
                num_layers=int(cfg.num_layers),
            ),
            use_smolgen=bool(cfg.use_smolgen),
            use_nla=bool(cfg.use_nla),
            use_qk_rmsnorm=bool(cfg.use_qk_rmsnorm),
            use_gradient_checkpointing=bool(cfg.use_gradient_checkpointing),
            input_pos_encoding=str(cfg.input_pos_encoding),
            qkv_projection=str(cfg.qkv_projection),
            use_deepnorm=bool(cfg.use_deepnorm),
            policy_encoding=MODEL_POLICY_ENCODING,
            use_dynamic_relations=bool(cfg.use_dynamic_relations),
            dynamic_relation_count=int(cfg.dynamic_relation_count),
            policy_dynamic_relations=bool(cfg.policy_dynamic_relations),
            input_global_embedding=str(cfg.input_global_embedding),
            input_global_embedding_channels=int(cfg.input_global_embedding_channels),
            input_square_embedding=str(cfg.input_square_embedding),
            smolgen_mode=str(cfg.smolgen_mode),
            smolgen_pooling=str(cfg.smolgen_pooling),
            smolgen_hidden_channels=int(cfg.smolgen_hidden_channels),
            smolgen_hidden_sz=int(cfg.smolgen_hidden_sz),
            smolgen_gen_sz=int(cfg.smolgen_gen_sz),
            smolgen_bias_scale=str(cfg.smolgen_bias_scale),
            smolgen_bias_norm=str(cfg.smolgen_bias_norm),
            arc_attention_bias=str(cfg.arc_attention_bias),
            smolgen_relation_basis=bool(cfg.smolgen_relation_basis),
            smolgen_relation_norm=str(cfg.smolgen_relation_norm),
            smolgen_relation_coeff_norm=str(cfg.smolgen_relation_coeff_norm),
            smolgen_relation_scale=str(cfg.smolgen_relation_scale),
            phase_output_adapter=bool(cfg.phase_output_adapter),
            phase_output_adapter_dim=int(cfg.phase_output_adapter_dim),
            phase_smolgen=bool(cfg.phase_smolgen),
            categorical_head_coupled=bool(cfg.categorical_head_coupled),
            phase_piece_thresholds=normalize_phase_piece_thresholds(cfg.phase_piece_thresholds),
        )
        return _attach_runtime_model_metadata(ChessNet(tcfg), cfg)
    raise ValueError(f"Unknown model kind: {cfg.kind}")


def _reinit_heads(model: torch.nn.Module, head_names: tuple[str, ...]) -> list[str]:
    """Re-init named heads with small Xavier-uniform weights and zero biases.

    Uses Xavier(gain=0.1) instead of zeros to avoid multiplicative dead
    gradients in attention-based policy heads (logits = Q @ K^T — both zero
    means d_logits/d_Q = K^T = 0).
    """
    reinit: list[str] = []
    for name in head_names:
        head = getattr(model, name, None)
        if not isinstance(head, torch.nn.Module):
            continue
        for param in head.parameters():
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param, gain=0.1)
            else:
                torch.nn.init.zeros_(param)
        reset_neutral_bias = getattr(head, "reset_neutral_output_bias_", None)
        if callable(reset_neutral_bias):
            reset_neutral_bias()
        reinit.append(name)
    return reinit


_POLICY_HEADS = ("policy", "policy_own", "policy_soft", "policy_sf", "policy_future")
_VOLATILITY_HEADS = ("volatility", "sf_volatility")


def zero_policy_head_parameters_(model: torch.nn.Module) -> list[str]:
    """Re-init policy-head parameters to small random values."""
    return _reinit_heads(model, _POLICY_HEADS)


def reinit_volatility_head_parameters_(model: torch.nn.Module) -> list[str]:
    """Re-init volatility heads while leaving trunk/policy/value intact."""
    return _reinit_heads(model, _VOLATILITY_HEADS)


def _migrate_qkv_keys(ckpt_state: dict, *, model_state: dict, label: str) -> dict:
    """Translate between fused and split QKV keys when topology changes."""
    original_state = dict(ckpt_state)

    split_prefixes: set[str] = set()
    for k in original_state:
        for proj in ("q_proj", "k_proj", "v_proj"):
            if k.endswith((f".{proj}.weight", f".{proj}.bias")):
                prefix = k.rsplit(f".{proj}.", 1)[0]
                suffix = k.rsplit(f".{proj}.", 1)[1]
                if (
                    f"{prefix}.qkv_proj.{suffix}" in model_state
                    and f"{prefix}.qkv_proj.{suffix}" not in original_state
                ):
                    split_prefixes.add(prefix)
                break

    extra: dict = {}
    migrated_count = 0
    for prefix in split_prefixes:
        for suffix in ("weight", "bias"):
            q_k = f"{prefix}.q_proj.{suffix}"
            k_k = f"{prefix}.k_proj.{suffix}"
            v_k = f"{prefix}.v_proj.{suffix}"
            fused_k = f"{prefix}.qkv_proj.{suffix}"
            if (
                q_k in original_state
                and k_k in original_state
                and v_k in original_state
                and fused_k not in original_state
            ):
                extra[fused_k] = torch.cat(
                    [original_state[q_k], original_state[k_k], original_state[v_k]],
                    dim=0,
                )
                migrated_count += 3
    if migrated_count:
        print(f"[{label}] Migrated {migrated_count} separate q/k/v keys -> fused qkv_proj")

    fused_extra: dict = {}
    migrated_count = 0
    fused_prefixes: set[str] = set()
    for k in original_state:
        if k.endswith((".qkv_proj.weight", ".qkv_proj.bias")):
            prefix = k.rsplit(".qkv_proj.", 1)[0]
            suffix = k.rsplit(".qkv_proj.", 1)[1]
            if (
                f"{prefix}.q_proj.{suffix}" in model_state
                and f"{prefix}.q_proj.{suffix}" not in original_state
            ):
                fused_prefixes.add(prefix)

    for prefix in fused_prefixes:
        for suffix in ("weight", "bias"):
            fused_k = f"{prefix}.qkv_proj.{suffix}"
            if fused_k not in original_state:
                continue
            tensor = original_state[fused_k]
            if tensor.shape[0] % 3 != 0:
                continue
            q, k, v = tensor.chunk(3, dim=0)
            for proj, part in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
                split_k = f"{prefix}.{proj}.{suffix}"
                if split_k in model_state and split_k not in original_state:
                    fused_extra[split_k] = part
                    migrated_count += 1
    if migrated_count:
        print(f"[{label}] Migrated {migrated_count} fused qkv_proj keys -> separate q/k/v")
    return {**original_state, **extra, **fused_extra}


def _normalize_orig_mod_prefix(ckpt_state: dict, *, model_state: dict) -> dict:
    """Add/remove torch.compile's ``_orig_mod.`` prefix so a checkpoint saved
    under one wrap-state loads under either."""
    ckpt_has_prefix = any(k.startswith("_orig_mod.") for k in ckpt_state)
    model_has_prefix = any(k.startswith("_orig_mod.") for k in model_state)
    if ckpt_has_prefix and not model_has_prefix:
        return {k.removeprefix("_orig_mod."): v for k, v in ckpt_state.items()}
    if model_has_prefix and not ckpt_has_prefix:
        return {f"_orig_mod.{k}": v for k, v in ckpt_state.items()}
    return ckpt_state


# Derived from the canonical plane counts so a future extra-feature version
# can't leave these literals stale (see encoding/features.py).
_V1_INPUT_PLANES = input_plane_count(EXTRA_FEATURES_V1)            # 146 = 112 LC0 + 34 v1 extra
_V2_EXTRA_DELTA = input_plane_count(EXTRA_FEATURES_V2_THREATS) - _V1_INPUT_PLANES  # 29


def _pad_v1_input_columns(value: torch.Tensor, target_shape: torch.Size) -> torch.Tensor | None:
    """Insert 29 zero input columns at index 146 if the shapes match the
    v1 -> v2_threats migration pattern; return None otherwise.

    The new planes sit at indices [146:175] (any appended channels like arc
    position encodings come after the plane block), so the tensor is split
    there. Zeros are exact: zero weights make the new planes no-ops, and zero
    AdamW moments are the exact continuation for zero-initialized weights.
    """
    if value.shape == target_shape or value.ndim != len(target_shape):
        return None
    diff_dims = [d for d in range(value.ndim) if value.shape[d] != target_shape[d]]
    if len(diff_dims) != 1:
        return None
    d = diff_dims[0]
    if target_shape[d] - value.shape[d] != _V2_EXTRA_DELTA or value.shape[d] < _V1_INPUT_PLANES:
        return None
    pad_shape = list(value.shape)
    pad_shape[d] = _V2_EXTRA_DELTA
    head = value.narrow(d, 0, _V1_INPUT_PLANES)
    tail = value.narrow(d, _V1_INPUT_PLANES, value.shape[d] - _V1_INPUT_PLANES)
    return torch.cat(
        [head, torch.zeros(pad_shape, dtype=value.dtype, device=value.device), tail],
        dim=d,
    )


def _migrate_input_plane_columns(
    ckpt_state: dict, *, model_state: dict, label: str,
) -> dict:
    """Zero-init the v2_threats input columns when warm-starting from v1.

    A v1 checkpoint's input projection has 146 input channels; a v2_threats
    model has 175 — see _pad_v1_input_columns. The warm-started model is
    bit-identical to the v1 model at step 0.
    """
    out = dict(ckpt_state)
    migrated = 0
    for k, mv in model_state.items():
        cv = out.get(k)
        if cv is None or not torch.is_tensor(cv):
            continue
        padded = _pad_v1_input_columns(cv, mv.shape)
        if padded is not None:
            out[k] = padded
            migrated += 1
    if migrated:
        print(
            f"[{label}] Zero-initialized {_V2_EXTRA_DELTA} v2_threats input "
            f"columns on {migrated} tensor(s) (v1 -> v2 warm start)"
        )
    return out


def migrate_optimizer_input_plane_state(optimizer: torch.optim.Optimizer) -> int:
    """Zero-pad per-parameter optimizer state after a v1 -> v2 warm start.

    ``Optimizer.load_state_dict`` restores moment tensors by parameter order
    without shape validation, so a v1 checkpoint's exp_avg/exp_avg_sq for the
    input projection silently keeps 146 columns under a 175-column parameter
    and crashes at the first ``step()``. Zero moments are the exact AdamW
    continuation for the zero-initialized new weight columns.

    Only tensors matching the v1 -> v2 pattern are touched — optimizers with
    legitimately param-shape-mismatched state (factored preconditioners)
    pass through untouched. Returns the number of migrated tensors.
    """
    migrated = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if not state:
                continue
            for key, value in list(state.items()):
                if not torch.is_tensor(value):
                    continue
                padded = _pad_v1_input_columns(value, param.shape)
                if padded is not None:
                    state[key] = padded
                    migrated += 1
    return migrated


def _filter_shape_mismatches(ckpt_state: dict, model_state: dict) -> tuple[dict, list[str]]:
    """Drop keys whose checkpoint shape differs from the model. Returns (filtered, skipped)."""
    filtered: dict = {}
    skipped: list[str] = []
    for k, v in ckpt_state.items():
        if k in model_state and v.shape != model_state[k].shape:
            skipped.append(k)
        else:
            filtered[k] = v
    return filtered, skipped


def load_state_dict_tolerant(
    model: torch.nn.Module,
    ckpt_state: dict,
    *,
    label: str = "checkpoint",
    require_complete: bool = False,
) -> None:
    """Load checkpoint into *model*, tolerating shape and key mismatches.

    Any key whose shape differs between checkpoint and model is silently
    dropped (model keeps its freshly-initialised weights for that layer).
    Missing and unexpected keys are logged but not fatal, allowing
    architecture changes (new layers, renamed modules) to load gracefully.

    With ``require_complete=True`` any drop at all (shape skip, missing or
    unexpected key, after the compile-prefix/qkv/plane migrations) raises
    instead. Use this when the model was reconstructed from the checkpoint's
    own ``arch`` payload, where the load must be exact and a partially
    random-init model would silently corrupt whatever consumes it.
    """
    model_state = model.state_dict()
    ckpt_state = _normalize_orig_mod_prefix(ckpt_state, model_state=model_state)
    ckpt_state = _migrate_qkv_keys(ckpt_state, model_state=model_state, label=label)
    ckpt_state = _migrate_input_plane_columns(ckpt_state, model_state=model_state, label=label)
    filtered, skipped = _filter_shape_mismatches(ckpt_state, model_state)

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if require_complete and (skipped or missing or unexpected):
        raise RuntimeError(
            f"[{label}] Incomplete state-dict load with require_complete=True: "
            f"shape_skipped={skipped}, missing={missing}, unexpected={unexpected}. "
            f"The model config claims to match this checkpoint exactly, so any "
            f"drop means a partially fresh-initialized model."
        )

    # Catastrophic-load detector: if essentially nothing loaded, bail loudly.
    # Trainer would otherwise silently fall back to fresh-init weights and
    # then publish them to selfplay workers, destroying the model. Threshold
    # is generous (50%): partial loads with arch drift are still allowed,
    # but "0/192 keys loaded" gets caught.
    n_expected = len(model_state)
    n_loaded = n_expected - len(missing)
    if n_expected > 0 and n_loaded < max(1, n_expected // 2):
        raise RuntimeError(
            f"[{label}] Catastrophic state-dict load: only {n_loaded}/{n_expected} "
            f"parameters loaded from checkpoint. This usually indicates a key-prefix "
            f"mismatch (e.g. saving under torch.compile then loading without it). "
            f"Refusing to continue with a fresh-initialized model. "
            f"Sample missing keys: {missing[:5]}, sample unexpected: {unexpected[:5]}"
        )
    if skipped or missing or unexpected:
        print(f"[{label}] Tolerant load — shape_skipped={skipped}, "
              f"missing={missing}, unexpected={unexpected}")
