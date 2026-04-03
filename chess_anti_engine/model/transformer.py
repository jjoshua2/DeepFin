from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from chess_anti_engine.moves import POLICY_SIZE, build_policy_gather_tables


_VOLATILITY_HEAD_NEUTRAL_OUTPUT = 0.01


def _softplus_inverse(y: float) -> float:
    y = float(max(1e-6, y))
    return float(math.log(math.expm1(y)))


def _rmsnorm(normalized_shape: int, *, eps: float = 1e-6) -> nn.Module:
    """Return an RMSNorm module.

    Uses torch.nn.RMSNorm when available; otherwise falls back to a minimal custom impl.
    """

    if hasattr(nn, "RMSNorm"):
        # PyTorch 2.0+
        return nn.RMSNorm(int(normalized_shape), eps=float(eps))

    class _FallbackRMSNorm(nn.Module):
        def __init__(self, d: int, *, eps: float):
            super().__init__()
            self.eps = float(eps)
            self.weight = nn.Parameter(torch.ones((int(d),)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (..., d)
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (x / rms) * self.weight

    return _FallbackRMSNorm(int(normalized_shape), eps=float(eps))


class AttentionPolicyHead(nn.Module):
    """Attention-based policy head that emits LC0-style 8x8x73 = 4672 logits.

    We compute from->to logits via scaled dot-product between from-square queries
    and to-square keys, then gather those logits into the LC0 73-plane encoding:
    - planes 0..55: queen-like moves (8 directions x 7 distances)
    - planes 56..63: knight moves
    - planes 64..72: underpromotions (separate linear head)

    Output is a flat `(B, 4672)` tensor, indexed as `from_sq * 73 + plane`.
    """

    def __init__(self, embed_dim: int, policy_dim: int | None = None):
        super().__init__()
        if policy_dim is None:
            policy_dim = embed_dim
        self.q = nn.Linear(embed_dim, policy_dim)
        self.k = nn.Linear(embed_dim, policy_dim)
        self.scale = policy_dim**-0.5

        # Underpromotion planes: 9 logits per from-square (N,B,R) x (left,forward,right).
        self.underpromo = nn.Linear(embed_dim, 9)

        tables = build_policy_gather_tables()
        self.register_buffer("to_sq", torch.from_numpy(tables.to_sq).long(), persistent=False)  # (64,64)
        self.register_buffer("to_valid", torch.from_numpy(tables.valid).bool(), persistent=False)  # (64,64)

        promo_from = torch.zeros((64,), dtype=torch.bool)
        for sq in range(64):
            if (sq // 8) == 6:  # oriented rank 7
                promo_from[sq] = True
        self.register_buffer("promo_from", promo_from, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        q = self.q(x)
        k = self.k(x)
        logits_ft = (q @ k.transpose(-1, -2)) * self.scale  # (B,64,64)

        gather_idx = self.to_sq.unsqueeze(0).expand(b, -1, -1)
        logits_64 = torch.gather(logits_ft, dim=-1, index=gather_idx)  # (B,64,64)

        neg_inf = logits_64.new_full((), -1e9)
        logits_64 = torch.where(self.to_valid.unsqueeze(0), logits_64, neg_inf)

        up = self.underpromo(x)  # (B,64,9)
        up = torch.where(self.promo_from.unsqueeze(0).unsqueeze(-1), up, neg_inf)

        logits = torch.cat([logits_64, up], dim=-1)  # (B,64,73)
        out = logits.reshape(b, 64 * 73)
        assert out.shape[-1] == POLICY_SIZE
        return out


class ValueHead(nn.Module):
    """LC0-style value head: per-square projection → flatten → MLP.

    Instead of mean-pooling (which discards spatial information), each of the
    64 square tokens is projected to ``token_dim`` features and then
    concatenated into a single vector.  This preserves which-square-has-what
    information that is critical for position evaluation.
    """

    def __init__(self, embed_dim: int, out_dim: int, *, token_dim: int = 128):
        super().__init__()
        self.token_proj = nn.Linear(embed_dim, token_dim)
        flat_dim = 64 * token_dim  # 64 squares
        self.net = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.Mish(),
            nn.Linear(128, out_dim),
        )
        # Small-scale init on the output projection so initial logits are ~0
        # (softmax → ~uniform distribution, win_p ≈ 1/3 instead of random ±1.4σ).
        nn.init.normal_(self.net[2].weight, std=0.01)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64, embed_dim)
        projected = F.mish(self.token_proj(x))  # (B, 64, token_dim)
        flat = projected.flatten(1)              # (B, 64 * token_dim)
        return self.net(flat)


class VolatilityHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Mish(),
            nn.Linear(64, 3),
            # Keep outputs non-negative without the hard-zero dead zone of ReLU.
            nn.Softplus(),
        )
        self.reset_neutral_output_bias_()

    def reset_neutral_output_bias_(self) -> None:
        with torch.no_grad():
            self.net[2].bias.fill_(_softplus_inverse(_VOLATILITY_HEAD_NEUTRAL_OUTPUT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        return self.net(pooled)


class ScalarHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Mish(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        return self.net(pooled)


class Smolgen(nn.Module):
    """LC0 BT4 smolgen: per-square compress → flatten → bottleneck MLP → gen weight.

    BT4 reference: compress to 64×hidden_channels, flatten, project down to
    hidden_sz bottleneck, then up to H×gen_sz, reshape, and final gen_sz→4096
    projection per head.  hidden_channels=32, hidden_sz=256, gen_sz=256 in BT4.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_channels: int = 32,
        hidden_sz: int = 256,
        gen_sz: int = 256,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.gen_sz = int(gen_sz)
        # Per-square compression (no bias, matching LC0).
        self.compress = nn.Linear(embed_dim, hidden_channels, bias=False)
        flat_dim = 64 * hidden_channels
        # Bottleneck MLP: flatten → hidden_sz → H*gen_sz (LC0 BT4 topology).
        self.dense1 = nn.Linear(flat_dim, hidden_sz)
        self.ln1 = nn.LayerNorm(hidden_sz)
        self.dense2 = nn.Linear(hidden_sz, num_heads * gen_sz)
        self.ln2 = nn.LayerNorm(num_heads * gen_sz)
        # Per-head weight generation: gen_sz → 64*64 (no bias, matching LC0).
        self.gen_weight = nn.Linear(gen_sz, 64 * 64, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,64,D) -> bias: (B,H,64,64)
        b = x.shape[0]
        h = self.compress(x)               # (B, 64, hidden_channels)
        h = h.flatten(1)                    # (B, 64 * hidden_channels)
        h = self.ln1(F.silu(self.dense1(h)))  # (B, hidden_sz)
        h = self.ln2(F.silu(self.dense2(h)))  # (B, H * gen_sz)
        h = h.view(b, self.num_heads, self.gen_sz)  # (B, H, gen_sz)
        bias = self.gen_weight(h)           # (B, H, 64*64)
        return bias.view(b, self.num_heads, 64, 64)


class _NLAProjection(nn.Module):
    """NonLinear Attention projection: Linear -> Mish -> Linear."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Mish(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Post-norm transformer block with optional Smolgen and NLA.

    We implement attention explicitly so we can add a per-head bias.

    Optional: QK RMSNorm ("QK-norm"): normalize queries and keys per-head before
    the attention dot product to keep attention logits in a stable range.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        ffn_mult: float = 2,
        dropout: float = 0.0,
        use_nla: bool = False,
        use_qk_rmsnorm: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(dropout)

        # QKV projections
        if use_nla:
            self.q_proj = _NLAProjection(self.embed_dim, self.embed_dim)
            self.k_proj = _NLAProjection(self.embed_dim, self.embed_dim)
            self.v_proj = _NLAProjection(self.embed_dim, self.embed_dim)
        else:
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.use_qk_rmsnorm = bool(use_qk_rmsnorm)
        self.q_norm = _rmsnorm(self.head_dim) if self.use_qk_rmsnorm else None
        self.k_norm = _rmsnorm(self.head_dim) if self.use_qk_rmsnorm else None

        self.ln1 = nn.LayerNorm(self.embed_dim)

        hidden = int(self.embed_dim * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, self.embed_dim),
        )
        self.ln2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor, smolgen_bias: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B,64,D), smolgen_bias: (B,H,64,64) or None
        b, t, d = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (B,H,T,hd)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_logits = (q @ k.transpose(-1, -2)) * (self.head_dim**-0.5)  # (B,H,T,T)
        if smolgen_bias is not None:
            attn_logits = attn_logits + smolgen_bias

        attn = F.softmax(attn_logits, dim=-1)
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = attn @ v  # (B,H,T,hd)
        out = out.transpose(1, 2).contiguous().view(b, t, d)  # (B,T,D)
        out = self.out_proj(out)

        x = self.ln1(x + out)
        x = self.ln2(x + self.ffn(x))
        return x


@dataclass
class TransformerConfig:
    in_planes: int
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_mult: float = 2
    dropout: float = 0.0
    use_smolgen: bool = True
    use_nla: bool = False
    use_qk_rmsnorm: bool = False
    use_gradient_checkpointing: bool = False


class ChessNet(nn.Module):
    """Spec-aligned multi-head network (initial implementation).

    Input: (B, C, 8, 8)
    Tokenization: 64 square tokens with C features each.

    Heads:
    - 4 policy heads: own, soft, sf, future
    - 3 value heads: wdl, sf_eval, categorical
    - volatility head
    - moves_left head

    Note: training targets for most heads are added later when MCTS + SF targets
    are integrated; but the forward pass and shapes are fixed now.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self._use_grad_ckpt = bool(cfg.use_gradient_checkpointing)

        # LC0 BT4 input block: Dense(activation) → mult_gate → add_gate
        self.embed = nn.Linear(cfg.in_planes, cfg.embed_dim)
        # ma_gating: multiplicative gate (init 1, non-negative) then additive gate (init 0)
        # Store raw param; apply softplus in forward to enforce non-negativity (LC0 uses NonNeg constraint).
        self._embed_gate_mul_raw = nn.Parameter(torch.full((cfg.embed_dim,), _softplus_inverse(1.0)))
        self.embed_gate_add = nn.Parameter(torch.zeros(cfg.embed_dim))
        # Shared smolgen: one instance, bias reused by every encoder layer (LC0 BT4).
        self.smolgen = Smolgen(cfg.embed_dim, cfg.num_heads) if cfg.use_smolgen else None
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.embed_dim,
                    cfg.num_heads,
                    ffn_mult=cfg.ffn_mult,
                    dropout=cfg.dropout,
                    use_nla=cfg.use_nla,
                    use_qk_rmsnorm=cfg.use_qk_rmsnorm,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        self.policy_own = AttentionPolicyHead(cfg.embed_dim)
        self.policy_soft = AttentionPolicyHead(cfg.embed_dim)
        self.policy_sf = AttentionPolicyHead(cfg.embed_dim)
        self.policy_future = AttentionPolicyHead(cfg.embed_dim)

        self.value_wdl = ValueHead(cfg.embed_dim, 3)
        self.value_sf_eval = ValueHead(cfg.embed_dim, 3)
        self.value_categorical = ValueHead(cfg.embed_dim, 32)

        self.volatility = VolatilityHead(cfg.embed_dim)
        self.sf_volatility = VolatilityHead(cfg.embed_dim)
        self.moves_left = ScalarHead(cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B,C,8,8) -> (B,64,C)
        b, c, h, w = x.shape
        assert (h, w) == (8, 8)
        tokens = x.reshape(b, c, 64).transpose(1, 2)  # (B,64,C)

        # BT4 input: Dense(mish) → mult_gate(non-neg) → add_gate
        t = F.mish(self.embed(tokens))
        t = t * F.softplus(self._embed_gate_mul_raw) + self.embed_gate_add
        # Compute smolgen bias once, shared across all layers (LC0 BT4).
        smolgen_bias = self.smolgen(t) if self.smolgen is not None else None
        for blk in self.blocks:
            if self._use_grad_ckpt and self.training:
                t = grad_checkpoint(blk, t, smolgen_bias, use_reentrant=False)
            else:
                t = blk(t, smolgen_bias)

        return {
            "policy_own": self.policy_own(t),
            "policy_soft": self.policy_soft(t),
            "policy_sf": self.policy_sf(t),
            "policy_future": self.policy_future(t),
            "wdl": self.value_wdl(t),
            "sf_eval": self.value_sf_eval(t),
            "categorical": self.value_categorical(t),
            "volatility": self.volatility(t),
            "sf_volatility": self.sf_volatility(t),
            "moves_left": self.moves_left(t),
        }
