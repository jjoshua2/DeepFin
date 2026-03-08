from __future__ import annotations

from dataclasses import dataclass

import torch

from chess_anti_engine.encoding import encode_position
from .tiny import TinyNet
from .transformer import ChessNet, TransformerConfig


@dataclass
class ModelConfig:
    kind: str = "transformer"  # tiny|transformer
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_mult: int = 2
    use_smolgen: bool = True
    use_nla: bool = False
    use_qk_rmsnorm: bool = False
    use_gradient_checkpointing: bool = False


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
            ffn_mult=int(cfg.ffn_mult),
            use_smolgen=bool(cfg.use_smolgen),
            use_nla=bool(cfg.use_nla),
            use_qk_rmsnorm=bool(cfg.use_qk_rmsnorm),
            use_gradient_checkpointing=bool(cfg.use_gradient_checkpointing),
        )
        return ChessNet(tcfg)
    raise ValueError(f"Unknown model kind: {cfg.kind}")


def zero_policy_head_parameters_(model: torch.nn.Module) -> list[str]:
    """Zero policy-head parameters so their masked softmax starts near-uniform."""

    zeroed: list[str] = []
    for name in ("policy", "policy_own", "policy_soft", "policy_sf", "policy_future"):
        head = getattr(model, name, None)
        if not isinstance(head, torch.nn.Module):
            continue
        for param in head.parameters():
            torch.nn.init.zeros_(param)
        zeroed.append(name)
    return zeroed
