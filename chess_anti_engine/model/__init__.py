from __future__ import annotations

from dataclasses import dataclass

import torch

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import LC0_HISTORY_LEGACY, normalize_lc0_history_encoding
from chess_anti_engine.moves import POLICY_ENCODING_AZ_4672, normalize_policy_encoding
from chess_anti_engine.utils.architecture import normalize_ffn_mult_by_layer

from .tiny import TinyNet
from .transformer import ChessNet, TransformerConfig

# Bump this when ModelConfig gains a field that a defaulted value would
# misrepresent. Trainer embeds this version when saving; the UCI loader
# rejects checkpoints with a higher version AND rejects unknown keys at
# the same version — both prevent silent architecture mismatch on skew.
ARCH_SCHEMA_VERSION = 12


@dataclass
class ModelConfig:
    kind: str = "transformer"  # tiny|transformer
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_mult: float = 2.0
    ffn_mult_by_layer: tuple[float, ...] | None = None
    use_smolgen: bool = True
    use_nla: bool = False
    use_qk_rmsnorm: bool = False
    use_gradient_checkpointing: bool = False
    input_pos_encoding: str = "none"
    qkv_projection: str = "fused"
    use_deepnorm: bool = False
    policy_encoding: str = POLICY_ENCODING_AZ_4672
    input_history_encoding: str = LC0_HISTORY_LEGACY
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


def model_config_from_manifest_dict(mc: dict) -> ModelConfig:
    """Build a ModelConfig from the ``model_config`` block of a publish manifest.

    Manifest field name is ``gradient_checkpointing`` (not ``use_*``) for
    historical reasons; everything else maps 1:1.
    """
    num_layers = int(mc.get("num_layers", 6))
    return ModelConfig(
        kind=str(mc.get("kind", "transformer")),
        embed_dim=int(mc.get("embed_dim", 256)),
        num_layers=num_layers,
        num_heads=int(mc.get("num_heads", 8)),
        ffn_mult=float(mc.get("ffn_mult", 2)),
        ffn_mult_by_layer=normalize_ffn_mult_by_layer(
            mc.get("ffn_mult_by_layer"),
            num_layers=num_layers,
        ),
        use_smolgen=bool(mc.get("use_smolgen", True)),
        use_nla=bool(mc.get("use_nla", False)),
        use_qk_rmsnorm=bool(mc.get("use_qk_rmsnorm", False)),
        use_gradient_checkpointing=bool(mc.get("gradient_checkpointing", False)),
        input_pos_encoding=str(mc.get("input_pos_encoding", "none")),
        qkv_projection=str(mc.get("qkv_projection", "fused")),
        use_deepnorm=bool(mc.get("use_deepnorm", False)),
        policy_encoding=normalize_policy_encoding(mc.get("policy_encoding", POLICY_ENCODING_AZ_4672)),
        input_history_encoding=normalize_lc0_history_encoding(mc.get("input_history_encoding", LC0_HISTORY_LEGACY)),
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
    return {
        "kind": str(cfg.kind),
        "embed_dim": int(cfg.embed_dim),
        "num_layers": int(cfg.num_layers),
        "num_heads": int(cfg.num_heads),
        "ffn_mult": float(cfg.ffn_mult),
        "ffn_mult_by_layer": list(ffn_mult_by_layer) if ffn_mult_by_layer is not None else None,
        "use_smolgen": bool(cfg.use_smolgen),
        "use_nla": bool(cfg.use_nla),
        "use_qk_rmsnorm": bool(cfg.use_qk_rmsnorm),
        "gradient_checkpointing": bool(cfg.use_gradient_checkpointing),
        "input_pos_encoding": str(cfg.input_pos_encoding),
        "qkv_projection": str(cfg.qkv_projection),
        "use_deepnorm": bool(cfg.use_deepnorm),
        "policy_encoding": normalize_policy_encoding(cfg.policy_encoding),
        "input_history_encoding": normalize_lc0_history_encoding(cfg.input_history_encoding),
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
    }


def infer_input_planes() -> int:
  # Use startpos to infer plane count.
    import chess

    b = chess.Board()
    x = encode_position(b, add_features=True)
    return int(x.shape[0])


def _attach_runtime_model_metadata(model: torch.nn.Module, cfg: ModelConfig) -> torch.nn.Module:
    setattr(model, "policy_encoding", normalize_policy_encoding(cfg.policy_encoding))
    setattr(model, "input_history_encoding", normalize_lc0_history_encoding(cfg.input_history_encoding))
    return model


def build_model(cfg: ModelConfig) -> torch.nn.Module:
    policy_encoding = normalize_policy_encoding(cfg.policy_encoding)
    in_planes = infer_input_planes()
    if cfg.kind == "tiny":
        if policy_encoding != POLICY_ENCODING_AZ_4672:
            raise ValueError("tiny model only supports policy_encoding='az_4672'")
        return _attach_runtime_model_metadata(TinyNet(in_planes=in_planes), cfg)
    if cfg.kind == "transformer":
        tcfg = TransformerConfig(
            in_planes=in_planes,
            embed_dim=int(cfg.embed_dim),
            num_layers=int(cfg.num_layers),
            num_heads=int(cfg.num_heads),
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
            policy_encoding=policy_encoding,
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
            if k.endswith(f".{proj}.weight") or k.endswith(f".{proj}.bias"):
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
        if k.endswith(".qkv_proj.weight") or k.endswith(".qkv_proj.bias"):
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
) -> None:
    """Load checkpoint into *model*, tolerating shape and key mismatches.

    Any key whose shape differs between checkpoint and model is silently
    dropped (model keeps its freshly-initialised weights for that layer).
    Missing and unexpected keys are logged but not fatal, allowing
    architecture changes (new layers, renamed modules) to load gracefully.
    """
    model_state = model.state_dict()
    ckpt_state = _normalize_orig_mod_prefix(ckpt_state, model_state=model_state)
    ckpt_state = _migrate_qkv_keys(ckpt_state, model_state=model_state, label=label)
    filtered, skipped = _filter_shape_mismatches(ckpt_state, model_state)

    missing, unexpected = model.load_state_dict(filtered, strict=False)

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
