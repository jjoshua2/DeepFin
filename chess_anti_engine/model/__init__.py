from __future__ import annotations

from dataclasses import dataclass

import torch

from chess_anti_engine.encoding import encode_position

from .tiny import TinyNet
from .transformer import ChessNet, TransformerConfig

# Bump this when ModelConfig gains a field that a defaulted value would
# misrepresent. Trainer embeds this version when saving; the UCI loader
# rejects checkpoints with a higher version AND rejects unknown keys at
# the same version — both prevent silent architecture mismatch on skew.
ARCH_SCHEMA_VERSION = 1


@dataclass
class ModelConfig:
    kind: str = "transformer"  # tiny|transformer
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_mult: float = 2.0
    use_smolgen: bool = True
    use_nla: bool = False
    use_qk_rmsnorm: bool = False
    use_gradient_checkpointing: bool = False


def model_config_from_manifest_dict(mc: dict) -> ModelConfig:
    """Build a ModelConfig from the ``model_config`` block of a publish manifest.

    Manifest field name is ``gradient_checkpointing`` (not ``use_*``) for
    historical reasons; everything else maps 1:1.
    """
    return ModelConfig(
        kind=str(mc.get("kind", "transformer")),
        embed_dim=int(mc.get("embed_dim", 256)),
        num_layers=int(mc.get("num_layers", 6)),
        num_heads=int(mc.get("num_heads", 8)),
        ffn_mult=float(mc.get("ffn_mult", 2)),
        use_smolgen=bool(mc.get("use_smolgen", True)),
        use_nla=bool(mc.get("use_nla", False)),
        use_qk_rmsnorm=bool(mc.get("use_qk_rmsnorm", False)),
        use_gradient_checkpointing=bool(mc.get("gradient_checkpointing", False)),
    )


def infer_input_planes() -> int:
  # Use startpos to infer plane count.
    import chess

    b = chess.Board()
    x = encode_position(b, add_features=True)
    return int(x.shape[0])


def build_model(cfg: ModelConfig) -> torch.nn.Module:
    in_planes = infer_input_planes()
    if cfg.kind == "tiny":
        return TinyNet(in_planes=in_planes)
    if cfg.kind == "transformer":
        tcfg = TransformerConfig(
            in_planes=in_planes,
            embed_dim=int(cfg.embed_dim),
            num_layers=int(cfg.num_layers),
            num_heads=int(cfg.num_heads),
            ffn_mult=float(cfg.ffn_mult),
            use_smolgen=bool(cfg.use_smolgen),
            use_nla=bool(cfg.use_nla),
            use_qk_rmsnorm=bool(cfg.use_qk_rmsnorm),
            use_gradient_checkpointing=bool(cfg.use_gradient_checkpointing),
        )
        return ChessNet(tcfg)
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
  # Migrate separate q_proj/k_proj/v_proj -> fused qkv_proj
    migrated_keys: list[str] = []
    extra = {}
    prefixes_seen: set[str] = set()
    for k in list(ckpt_state.keys()):
        for proj in ("q_proj", "k_proj", "v_proj"):
            if k.endswith(f".{proj}.weight") or k.endswith(f".{proj}.bias"):
                prefix = k.rsplit(f".{proj}.", 1)[0]
                suffix = k.rsplit(f".{proj}.", 1)[1]  # "weight" or "bias"
                fused_key = f"{prefix}.qkv_proj.{suffix}"
                if fused_key not in ckpt_state and prefix not in prefixes_seen:
                    prefixes_seen.add(prefix)
                break
    for prefix in prefixes_seen:
        for suffix in ("weight", "bias"):
            q_k = f"{prefix}.q_proj.{suffix}"
            k_k = f"{prefix}.k_proj.{suffix}"
            v_k = f"{prefix}.v_proj.{suffix}"
            fused_k = f"{prefix}.qkv_proj.{suffix}"
            if q_k in ckpt_state and k_k in ckpt_state and v_k in ckpt_state and fused_k not in ckpt_state:
                extra[fused_k] = torch.cat([ckpt_state[q_k], ckpt_state[k_k], ckpt_state[v_k]], dim=0)
                migrated_keys.extend([q_k, k_k, v_k])
    if migrated_keys:
        print(f"[{label}] Migrated {len(migrated_keys)} separate q/k/v keys -> fused qkv_proj")
    ckpt_state = {**ckpt_state, **extra}

    model_state = model.state_dict()
    filtered = {}
    skipped: list[str] = []
    for k, v in ckpt_state.items():
        if k in model_state and v.shape != model_state[k].shape:
            skipped.append(k)
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped or missing or unexpected:
        print(f"[{label}] Tolerant load — shape_skipped={skipped}, "
              f"missing={missing}, unexpected={unexpected}")
