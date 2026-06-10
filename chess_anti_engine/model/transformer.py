from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from chess_anti_engine.moves import (
    COMPACT_POLICY_SIZE,
    COMPACT_TO_FULL_POLICY,
    POLICY_ENCODING_AZ_4672,
    POLICY_ENCODING_LC0_1858,
    POLICY_SIZE,
    build_policy_gather_tables,
    normalize_policy_encoding,
    policy_size_for_encoding,
)
from chess_anti_engine.utils.architecture import normalize_phase_piece_thresholds

_VOLATILITY_HEAD_NEUTRAL_OUTPUT = 0.01
_ARC_POS_CHANNELS = 64
_ARC_RELATION_CHANNELS = 5
_NUM_PHASE_BUCKETS = 3


def _softplus_inverse(y: float) -> float:
    y = float(max(1e-6, y))
    return float(math.log(math.expm1(y)))


class _DtypeMatchedRMSNorm(nn.Module):
    """RMSNorm with an affine scale cast to the activation dtype.

    Q/K norm runs inside bf16 autocast, while parameters are kept in fp32. PyTorch's
    RMSNorm warns and misses its fused kernel for that dtype mismatch.
    """

    def __init__(self, normalized_shape: int, *, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = (int(normalized_shape),)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # skylos: ignore (nn.Module dispatch via __call__)
        weight = self.weight
        if weight.dtype != x.dtype or weight.device != x.device:
            weight = weight.to(device=x.device, dtype=x.dtype)
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, self.normalized_shape, weight, self.eps)
        y = x * x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return y * weight


def _rmsnorm(normalized_shape: int, *, eps: float = 1e-6) -> nn.Module:
    return _DtypeMatchedRMSNorm(int(normalized_shape), eps=float(eps))


def _make_arc_pos_encoding() -> torch.Tensor:
    """LC0-style fixed square-relation channels, shape ``(1, 64, 64)``."""
    pos = torch.zeros((1, 64, _ARC_POS_CHANNELS), dtype=torch.float32)
    queen_dirs = (
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    )
    knight_dirs = (
        (1, 2), (2, 1), (2, -1), (1, -2),
        (-1, -2), (-2, -1), (-2, 1), (-1, 2),
    )
    for rank in range(8):
        for file in range(8):
            sq = rank * 8 + file
            pos[0, sq, sq] = -1.0
            for dr, df in queen_dirs:
                rr = rank + dr
                ff = file + df
                while 0 <= rr < 8 and 0 <= ff < 8:
                    pos[0, sq, rr * 8 + ff] = 1.0
                    rr += dr
                    ff += df
            for dr, df in knight_dirs:
                rr = rank + dr
                ff = file + df
                if 0 <= rr < 8 and 0 <= ff < 8:
                    pos[0, sq, rr * 8 + ff] = 1.0
    return pos


def _make_arc_relation_basis() -> torch.Tensor:
    """Fixed chess square-pair relation masks, shape ``(R, 64, 64)``."""
    basis = torch.zeros((_ARC_RELATION_CHANNELS, 64, 64), dtype=torch.float32)
    for src_rank in range(8):
        for src_file in range(8):
            src = src_rank * 8 + src_file
            for dst_rank in range(8):
                for dst_file in range(8):
                    dst = dst_rank * 8 + dst_file
                    dr = dst_rank - src_rank
                    df = dst_file - src_file
                    adr = abs(dr)
                    adf = abs(df)
                    if dst == src:
                        basis[0, src, dst] = 1.0
                    elif dr == 0 or df == 0:
                        basis[1, src, dst] = 1.0
                    elif adr == adf:
                        basis[2, src, dst] = 1.0
                    elif (adr, adf) in ((1, 2), (2, 1)):
                        basis[3, src, dst] = 1.0
                    elif max(adr, adf) == 1:
                        basis[4, src, dst] = 1.0
    return basis


def _normalize_relation_basis(basis: torch.Tensor, mode: str) -> torch.Tensor:
    mode = str(mode).lower().strip()
    if mode == "basis_center":
        return _normalize_attention_bias(basis, "center")
    if mode == "basis_center_rms":
        return _normalize_attention_bias(basis, "center_rms")
    return basis


def _normalize_attention_bias(bias: torch.Tensor, mode: str, *, eps: float = 1e-6) -> torch.Tensor:
    mode = str(mode).lower().strip()
    if mode == "none":
        return bias
    if mode == "center":
        return bias - bias.mean(dim=-1, keepdim=True)
    if mode == "center_rms":
        var, mean = torch.var_mean(bias, dim=-1, correction=0, keepdim=True)
        return (bias - mean) * torch.rsqrt(var + float(eps))
    raise ValueError(f"Unsupported smolgen_bias_norm: {mode!r}")


class AttentionPolicyHead(nn.Module):
    """Attention-based policy head that emits LC0-style policy logits.

    We compute from->to logits via scaled dot-product between from-square queries
    and to-square keys, then gather those logits into the LC0 73-plane encoding:
    - planes 0..55: queen-like moves (8 directions x 7 distances)
    - planes 56..63: knight moves
    - planes 64..72: underpromotions (separate linear head)

    By default output is the legacy padded `(B, 4672)` tensor, indexed as
    `from_sq * 73 + plane`. With `policy_encoding="lc0_1858"` dense forward
    gathers only geometrically valid LC0 move classes, while `forward_legal*`
    still accepts full 4672 action ids for the MCTS/CBoard action space.
    """

  # Registered as buffers in __init__ via register_buffer; these declarations are
  # for type checkers (pyright does not recognize register_buffer as initialization).
    to_sq: torch.Tensor
    to_valid: torch.Tensor
    promo_from: torch.Tensor
    compact_to_full: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        policy_dim: int | None = None,
        *,
        policy_encoding: str | None = None,
    ):
        super().__init__()
        self.policy_encoding = normalize_policy_encoding(policy_encoding)
        self.policy_size = policy_size_for_encoding(self.policy_encoding)
        if policy_dim is None:
            policy_dim = embed_dim
        self.q = nn.Linear(embed_dim, policy_dim)
        self.k = nn.Linear(embed_dim, policy_dim)
        self.scale = policy_dim**-0.5
  # Learnable sharpness multiplier: 1/√d squashes terminal-head logits below sharp MCTS targets.
        self.log_temp = nn.Parameter(torch.zeros(1))

  # Underpromotion planes: 9 logits per from-square (N,B,R) x (left,forward,right).
        self.underpromo = nn.Linear(embed_dim, 9)

        tables = build_policy_gather_tables()
        self.register_buffer("to_sq", torch.from_numpy(tables.to_sq).long(), persistent=False)  # (64,64)
        self.to_sq = cast(torch.Tensor, self._buffers["to_sq"])
        self.register_buffer("to_valid", torch.from_numpy(tables.valid).bool(), persistent=False)  # (64,64)
        self.to_valid = cast(torch.Tensor, self._buffers["to_valid"])
        self.register_buffer(
            "compact_to_full",
            torch.from_numpy(COMPACT_TO_FULL_POLICY).long(),
            persistent=False,
        )
        self.compact_to_full = cast(torch.Tensor, self._buffers["compact_to_full"])

        promo_from = torch.zeros((64,), dtype=torch.bool)
        for sq in range(64):
            if (sq // 8) == 6:  # oriented rank 7
                promo_from[sq] = True
        self.register_buffer("promo_from", promo_from, persistent=False)
        self.promo_from = cast(torch.Tensor, self._buffers["promo_from"])

    def forward(self, x: torch.Tensor, ft_bias: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        q = self.q(x)
        k = self.k(x)
        logits_ft = (q @ k.transpose(-1, -2)) * self.scale * torch.exp(self.log_temp)  # (B,64,64)
        if ft_bias is not None:
  # Dynamic-relation bias on the raw from->to logits (zero-init weights
  # upstream keep warm starts bit-identical).
            logits_ft = logits_ft + ft_bias.to(dtype=logits_ft.dtype)

        gather_idx = self.to_sq.unsqueeze(0).expand(b, -1, -1)
        logits_64 = torch.gather(logits_ft, dim=-1, index=gather_idx)  # (B,64,64)

        neg_inf = logits_64.new_full((), -1e9)
        logits_64 = torch.where(self.to_valid.unsqueeze(0), logits_64, neg_inf)

        up = self.underpromo(x)  # (B,64,9)
        up = torch.where(self.promo_from.unsqueeze(0).unsqueeze(-1), up, neg_inf)

        logits = torch.cat([logits_64, up], dim=-1)  # (B,64,73)
        out = logits.reshape(b, 64 * 73)
        if self.policy_encoding == POLICY_ENCODING_LC0_1858:
            out = out.index_select(dim=-1, index=self.compact_to_full)
            assert self.policy_size == COMPACT_POLICY_SIZE
            assert out.shape[-1] == COMPACT_POLICY_SIZE
        else:
            assert self.policy_size == POLICY_SIZE
            assert out.shape[-1] == POLICY_SIZE
        return out

    def forward_legal(
        self,
        x: torch.Tensor,
        legal_flat: torch.Tensor,
        legal_counts: torch.Tensor,
        ft_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if legal_flat.ndim != 1:
            raise ValueError(f"legal_flat must be 1D, got {legal_flat.ndim}D")
        if legal_counts.ndim != 1:
            raise ValueError(f"legal_counts must be 1D, got {legal_counts.ndim}D")
        bsz = x.shape[0]
        if legal_counts.shape[0] > bsz:
            raise ValueError(f"legal_counts len {legal_counts.shape[0]} exceeds batch {bsz}")
        n_legal = int(legal_flat.shape[0])
        if n_legal == 0:
            return x.new_empty((0,))

        legal_flat = legal_flat.to(device=x.device, dtype=torch.long)
        legal_counts = legal_counts.to(device=x.device, dtype=torch.long)
        rows = torch.repeat_interleave(
            torch.arange(legal_counts.shape[0], device=x.device, dtype=torch.long),
            legal_counts,
        )
        if rows.shape[0] != legal_flat.shape[0]:
            raise ValueError("legal_flat len must equal sum(legal_counts)")
        return self.forward_legal_rows(x, legal_flat, rows, ft_bias=ft_bias)

    def forward_legal_rows(
        self,
        x: torch.Tensor,
        legal_flat: torch.Tensor,
        legal_rows: torch.Tensor,
        ft_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if legal_flat.ndim != 1:
            raise ValueError(f"legal_flat must be 1D, got {legal_flat.ndim}D")
        if legal_rows.ndim != 1:
            raise ValueError(f"legal_rows must be 1D, got {legal_rows.ndim}D")
        n_legal = int(legal_flat.shape[0])
        if legal_rows.shape[0] != n_legal:
            raise ValueError("legal_rows shape must match legal_flat")
        if n_legal == 0:
            return x.new_empty((0,))

        legal_flat = legal_flat.to(device=x.device, dtype=torch.long)
        rows = legal_rows.to(device=x.device, dtype=torch.long)
        from_sq = torch.div(legal_flat, 73, rounding_mode="floor")
        plane = legal_flat.remainder(73)
        normal_mask = plane < 64
        promo_mask = ~normal_mask
        neg_inf = x.new_full((n_legal,), -1e9)

        q = self.q(x)
        k = self.k(x)
        normal_plane = plane.clamp(max=63)
        to_sq = self.to_sq[from_sq, normal_plane]
        normal_valid = normal_mask & self.to_valid[from_sq, normal_plane]
        normal_logits = (q[rows, from_sq] * k[rows, to_sq]).sum(dim=-1)
        normal_logits = normal_logits * self.scale * torch.exp(self.log_temp)
        if ft_bias is not None:
            normal_logits = normal_logits + ft_bias.to(dtype=normal_logits.dtype)[rows, from_sq, to_sq]
        normal_logits = torch.where(normal_valid, normal_logits, neg_inf)

        up = self.underpromo(x)
        promo_plane = (plane - 64).clamp(min=0, max=8)
        promo_valid = promo_mask & self.promo_from[from_sq]
        promo_logits = up[rows, from_sq, promo_plane]
        promo_logits = torch.where(promo_valid, promo_logits, neg_inf)

        return torch.where(normal_mask, normal_logits, promo_logits)


class PhaseTokenAdapter(nn.Module):
    """Small phase-conditioned adapter applied only at the output heads."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        hidden = int(hidden_dim)
        if hidden <= 0:
            raise ValueError(f"phase_output_adapter_dim must be positive, got {hidden_dim!r}")
        self.down = nn.Linear(embed_dim, hidden)
        self.phase = nn.Embedding(_NUM_PHASE_BUCKETS, hidden)
        self.up = nn.Linear(hidden, embed_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, phase_idx: torch.Tensor) -> torch.Tensor:
        phase = self.phase(phase_idx.to(device=x.device, dtype=torch.long)).unsqueeze(1)
        h = F.mish(self.down(x) + phase)
        return x + self.up(h)


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
        out = cast("nn.Linear", self.net[2])
        nn.init.normal_(out.weight, std=0.01)
        nn.init.zeros_(out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
  # x: (B, 64, embed_dim)
        projected = F.mish(self.token_proj(x))  # (B, 64, token_dim)
        flat = projected.flatten(1)  # (B, 64 * token_dim)
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
            out = cast("nn.Linear", self.net[2])
            out.bias.fill_(_softplus_inverse(_VOLATILITY_HEAD_NEUTRAL_OUTPUT))

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
    """LC0 BT4 smolgen / GAB-style dynamic attention bias.

    ``pooling="flatten"`` is the existing BT4 path: compress to
    64×hidden_channels, flatten, project down to hidden_sz, then up to
    H×gen_sz.

    ``pooling="mean"`` is a cheaper pooled-GAB variant: average the 64 token
    embeddings and feed that pooled board vector directly into the generator
    bottleneck. This preserves the generated per-head 64×64 bias while making
    the root smolgen input independent of the token count.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_channels: int = 32,
        hidden_sz: int = 256,
        gen_sz: int = 256,
        gen_weight: nn.Linear | None = None,
        use_relation_basis: bool = False,
        relation_norm: str = "none",
        relation_coeff_norm: str = "none",
        relation_scale: str = "none",
        pooling: str = "flatten",
        phase_conditioning: bool = False,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        hidden_channels = int(hidden_channels)
        hidden_sz = int(hidden_sz)
        self.gen_sz = int(gen_sz)
        if hidden_channels <= 0:
            raise ValueError("smolgen hidden_channels must be > 0")
        if hidden_sz <= 0:
            raise ValueError("smolgen hidden_sz must be > 0")
        if self.gen_sz <= 0:
            raise ValueError("smolgen gen_sz must be > 0")
        self.pooling = str(pooling).lower().strip()
        if self.pooling not in ("flatten", "mean"):
            raise ValueError(f"Unsupported smolgen_pooling: {pooling!r}")
        self.relation_norm = str(relation_norm).lower().strip()
        if self.relation_norm not in (
            "none",
            "branch_center",
            "branch_center_rms",
            "basis_center",
            "basis_center_rms",
        ):
            raise ValueError(f"Unsupported smolgen_relation_norm: {relation_norm!r}")
        self.relation_coeff_norm = str(relation_coeff_norm).lower().strip()
        if self.relation_coeff_norm not in ("none", "rms"):
            raise ValueError(f"Unsupported smolgen_relation_coeff_norm: {relation_coeff_norm!r}")
        self.relation_scale = str(relation_scale).lower().strip()
        if self.relation_scale not in ("none", "layer", "layer_head"):
            raise ValueError(f"Unsupported smolgen_relation_scale: {relation_scale!r}")
        self.compress: nn.Linear | None
        if self.pooling == "flatten":
            # Per-square compression (no bias, matching LC0).
            self.compress = nn.Linear(embed_dim, hidden_channels, bias=False)
            first_dim = 64 * hidden_channels
        else:
            self.compress = None
            first_dim = int(embed_dim)
        self.phase_conditioning = bool(phase_conditioning)
        self.phase_embedding: nn.Embedding | None = None
        if self.phase_conditioning:
            self.phase_embedding = nn.Embedding(_NUM_PHASE_BUCKETS, first_dim)
            nn.init.zeros_(self.phase_embedding.weight)
        self.dense1 = nn.Linear(first_dim, hidden_sz)
        self.ln1 = nn.LayerNorm(hidden_sz)
        self.dense2 = nn.Linear(hidden_sz, num_heads * gen_sz)
        self.ln2 = nn.LayerNorm(num_heads * gen_sz)
  # Per-head weight generation: gen_sz → 64*64 (no bias, matching LC0).
        self.gen_weight = (
            gen_weight if gen_weight is not None else nn.Linear(gen_sz, 64 * 64, bias=False)
        )
        self.relation_weight: nn.Linear | None = None
        if use_relation_basis:
            self.relation_weight = nn.Linear(gen_sz, _ARC_RELATION_CHANNELS, bias=False)
            nn.init.zeros_(self.relation_weight.weight)
        self.relation_scale_param: nn.Parameter | None = None
        if use_relation_basis and self.relation_scale == "layer":
            self.relation_scale_param = nn.Parameter(torch.ones(()))
        elif use_relation_basis and self.relation_scale == "layer_head":
            self.relation_scale_param = nn.Parameter(torch.ones(num_heads))
        self.register_buffer(
            "relation_basis_flat",
            _normalize_relation_basis(_make_arc_relation_basis(), self.relation_norm).flatten(1)
            if use_relation_basis
            else torch.empty(0, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, phase_idx: torch.Tensor | None = None) -> torch.Tensor:
  # x: (B,64,D) -> bias: (B,H,64,64)
        b = x.shape[0]
        if self.pooling == "flatten":
            compress = self.compress
            assert compress is not None
            h = compress(x)  # (B, 64, hidden_channels)
            h = h.flatten(1)  # (B, 64 * hidden_channels)
        else:
            h = x.mean(dim=1)  # (B, D)
        phase_embedding = self.phase_embedding
        if phase_embedding is not None:
            if phase_idx is None:
                raise ValueError("phase_idx is required when phase-conditioned smolgen is enabled")
            h = h + phase_embedding(phase_idx.to(device=x.device, dtype=torch.long))
        h = self.ln1(F.silu(self.dense1(h)))  # (B, hidden_sz)
        h = self.ln2(F.silu(self.dense2(h)))  # (B, H * gen_sz)
        h = h.view(b, self.num_heads, self.gen_sz)  # (B, H, gen_sz)
        bias = self.gen_weight(h)  # (B, H, 64*64)
        bias = bias.view(b, self.num_heads, 64, 64)
        relation_weight = self.relation_weight
        if relation_weight is not None:
            coeff = relation_weight(h)
            if self.relation_coeff_norm == "rms":
                coeff = coeff * torch.rsqrt(coeff.square().mean(dim=-1, keepdim=True) + 1e-6)
            basis = cast(torch.Tensor, self._buffers["relation_basis_flat"]).to(dtype=bias.dtype, device=bias.device)
            relation_bias = torch.matmul(coeff, basis).view(b, self.num_heads, 64, 64)
            if self.relation_norm == "branch_center":
                relation_bias = _normalize_attention_bias(relation_bias, "center")
            elif self.relation_norm == "branch_center_rms":
                relation_bias = _normalize_attention_bias(relation_bias, "center_rms")
            scale = self.relation_scale_param
            if scale is not None:
                if scale.ndim == 0:
                    relation_bias = relation_bias * scale.to(dtype=bias.dtype, device=bias.device)
                else:
                    relation_bias = relation_bias * scale.to(
                        dtype=bias.dtype,
                        device=bias.device,
                    ).view(1, -1, 1, 1)
            bias = bias + relation_bias
        return bias


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

    q_proj: _NLAProjection | nn.Linear
    k_proj: _NLAProjection | nn.Linear
    v_proj: _NLAProjection | nn.Linear
    qkv_proj: nn.Linear
    out_proj: nn.Linear
    ffn: nn.Sequential

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        ffn_mult: float = 2,
        dropout: float = 0.0,
        use_nla: bool = False,
        qkv_projection: str = "fused",
        use_qk_rmsnorm: bool = False,
        residual_branch_scale: float = 1.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(dropout)
        self.residual_branch_scale = float(residual_branch_scale)

        self.qkv_projection = str(qkv_projection).lower().strip()
        if self.qkv_projection not in ("fused", "split"):
            raise ValueError(f"Unsupported qkv_projection: {qkv_projection!r}")

  # QKV projections (fused by default; split is useful for optimizer ablations).
        self.use_nla = bool(use_nla)
        if use_nla:
            self.q_proj = _NLAProjection(self.embed_dim, self.embed_dim)
            self.k_proj = _NLAProjection(self.embed_dim, self.embed_dim)
            self.v_proj = _NLAProjection(self.embed_dim, self.embed_dim)
        elif self.qkv_projection == "split":
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)

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
        if self.use_nla or self.qkv_projection == "split":
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            qkv = self.qkv_proj(x).view(b, t, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)

  # (B,H,T,hd)
        if self.use_nla or self.qkv_projection == "split":
            q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            q = q.transpose(1, 2)  # already (B,T,H,hd) from unbind
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=smolgen_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B,H,T,hd)
        out = out.transpose(1, 2).contiguous().view(b, t, d)  # (B,T,D)
        out = self.out_proj(out)

        branch_scale = self.residual_branch_scale
        x = self.ln1(x + out * branch_scale)
        x = self.ln2(x + self.ffn(x) * branch_scale)
        return x


@dataclass
class TransformerConfig:
    in_planes: int
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    embed_dim_by_layer: tuple[int, ...] | list[int] | None = None
    ffn_mult: float = 2
    ffn_mult_by_layer: tuple[float, ...] | list[float] | None = None
    dropout: float = 0.0
    use_smolgen: bool = True
    use_nla: bool = False
    use_qk_rmsnorm: bool = False
    use_gradient_checkpointing: bool = False
    input_pos_encoding: str = "none"
    qkv_projection: str = "fused"
    use_deepnorm: bool = False
    policy_encoding: str = POLICY_ENCODING_AZ_4672
  # Dynamic board-relation attention bias (exact board relations computed in
  # C alongside the feature planes; see features.relation_matrices).
    use_dynamic_relations: bool = False
    dynamic_relation_count: int = 5
    policy_dynamic_relations: bool = False
    input_global_embedding: str = "none"
    input_global_embedding_channels: int = 0
    input_square_embedding: str = "none"
    smolgen_mode: str = "shared"  # shared | per_layer
    smolgen_pooling: str = "flatten"  # flatten | mean
    smolgen_hidden_channels: int = 32
    smolgen_hidden_sz: int = 256
    smolgen_gen_sz: int = 256
    smolgen_bias_scale: str = "none"  # none | layer | layer_head
    smolgen_bias_norm: str = "none"  # none | center | center_rms
    arc_attention_bias: str = "none"  # none | basic
    smolgen_relation_basis: bool = False
    # none | branch_center | branch_center_rms | basis_center | basis_center_rms
    smolgen_relation_norm: str = "none"
    smolgen_relation_coeff_norm: str = "none"  # none | rms
    smolgen_relation_scale: str = "none"  # none | layer | layer_head
    phase_output_adapter: bool = False
    phase_output_adapter_dim: int = 64
    phase_smolgen: bool = False
    phase_piece_thresholds: tuple[int, int] | list[int] | str | None = (13, 22)


def _resolve_ffn_mults(
    *,
    scalar: float,
    by_layer: tuple[float, ...] | list[float] | None,
    num_layers: int,
) -> tuple[float, ...]:
    if by_layer is None:
        vals = tuple(float(scalar) for _ in range(int(num_layers)))
    else:
        vals = tuple(float(v) for v in by_layer)
    if len(vals) != int(num_layers):
        raise ValueError(f"ffn_mult_by_layer length {len(vals)} must match num_layers {int(num_layers)}")
    for val in vals:
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError(f"FFN multipliers must be positive finite floats, got {val!r}")
    return vals


def _resolve_embed_dims(
    *,
    scalar: int,
    by_layer: tuple[int, ...] | list[int] | None,
    num_layers: int,
    num_heads: int,
) -> tuple[int, ...]:
    def coerce(value: int | float) -> int:
        if isinstance(value, bool):
            raise ValueError(f"embed_dim_by_layer values must be positive integers, got {value!r}")
        if isinstance(value, int):
            return value
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise ValueError(f"embed_dim_by_layer values must be positive integers, got {value!r}")

    if by_layer is None:
        vals = tuple(int(scalar) for _ in range(int(num_layers)))
    else:
        vals = tuple(coerce(v) for v in by_layer)
    if len(vals) != int(num_layers):
        raise ValueError(f"embed_dim_by_layer length {len(vals)} must match num_layers {int(num_layers)}")
    for val in vals:
        if val <= 0:
            raise ValueError(f"embed_dim_by_layer values must be positive integers, got {val!r}")
        if val % int(num_heads) != 0:
            raise ValueError(
                "each embed_dim_by_layer value must be divisible by num_heads "
                f"({val} % {int(num_heads)} != 0)"
            )
    return vals


def _make_width_adapter(in_dim: int, out_dim: int) -> nn.Module:
    if int(in_dim) == int(out_dim):
        return nn.Identity()
    adapter = nn.Linear(int(in_dim), int(out_dim), bias=False)
    with torch.no_grad():
        adapter.weight.zero_()
        diag = min(int(in_dim), int(out_dim))
        adapter.weight[:diag, :diag].fill_diagonal_(1.0)
    return adapter


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

    embed: nn.Linear

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self._use_grad_ckpt = bool(cfg.use_gradient_checkpointing)
        self._inference_only = False
        self.resid_channel_dropout = 0.0
        self.resid_channel_balance_weight = 0.0
        self._last_channel_balance_loss: torch.Tensor | None = None
        self.policy_encoding = normalize_policy_encoding(cfg.policy_encoding)
        self.policy_size = policy_size_for_encoding(self.policy_encoding)
        self.input_global_embedding = str(cfg.input_global_embedding).lower().strip()
        if self.input_global_embedding not in ("none", "bt4_board", "bt4_board_adapter"):
            raise ValueError(f"Unsupported input_global_embedding: {cfg.input_global_embedding!r}")
        self.input_global_embedding_channels = (
            int(cfg.input_global_embedding_channels)
            if int(cfg.input_global_embedding_channels) > 0
            else int(cfg.embed_dim) // 2
        )
        self.input_square_embedding = str(cfg.input_square_embedding).lower().strip()
        if self.input_square_embedding not in ("none", "add", "ma_gate"):
            raise ValueError(f"Unsupported input_square_embedding: {cfg.input_square_embedding!r}")
        self.input_pos_encoding = str(cfg.input_pos_encoding)
        if self.input_pos_encoding not in ("none", "arc", "arc_adapter"):
            raise ValueError(f"Unsupported input_pos_encoding: {self.input_pos_encoding!r}")
        self.qkv_projection = str(cfg.qkv_projection).lower().strip()
        if self.qkv_projection not in ("fused", "split"):
            raise ValueError(f"Unsupported qkv_projection: {cfg.qkv_projection!r}")
        self.smolgen_mode = str(cfg.smolgen_mode).lower().strip()
        if self.smolgen_mode not in ("shared", "per_layer"):
            raise ValueError(f"Unsupported smolgen_mode: {cfg.smolgen_mode!r}")
        self.smolgen_pooling = str(cfg.smolgen_pooling).lower().strip()
        if self.smolgen_pooling not in ("flatten", "mean"):
            raise ValueError(f"Unsupported smolgen_pooling: {cfg.smolgen_pooling!r}")
        self.smolgen_hidden_channels = int(cfg.smolgen_hidden_channels)
        self.smolgen_hidden_sz = int(cfg.smolgen_hidden_sz)
        self.smolgen_gen_sz = int(cfg.smolgen_gen_sz)
        if self.smolgen_hidden_channels <= 0:
            raise ValueError("smolgen_hidden_channels must be > 0")
        if self.smolgen_hidden_sz <= 0:
            raise ValueError("smolgen_hidden_sz must be > 0")
        if self.smolgen_gen_sz <= 0:
            raise ValueError("smolgen_gen_sz must be > 0")
        self.smolgen_bias_scale = str(cfg.smolgen_bias_scale).lower().strip()
        if self.smolgen_bias_scale not in ("none", "layer", "layer_head"):
            raise ValueError(f"Unsupported smolgen_bias_scale: {cfg.smolgen_bias_scale!r}")
        self.smolgen_bias_norm = str(cfg.smolgen_bias_norm).lower().strip()
        if self.smolgen_bias_norm not in ("none", "center", "center_rms"):
            raise ValueError(f"Unsupported smolgen_bias_norm: {cfg.smolgen_bias_norm!r}")
        self.arc_attention_bias = str(cfg.arc_attention_bias).lower().strip()
        if self.arc_attention_bias not in ("none", "basic"):
            raise ValueError(f"Unsupported arc_attention_bias: {cfg.arc_attention_bias!r}")
        self.smolgen_relation_norm = str(cfg.smolgen_relation_norm).lower().strip()
        if self.smolgen_relation_norm not in (
            "none",
            "branch_center",
            "branch_center_rms",
            "basis_center",
            "basis_center_rms",
        ):
            raise ValueError(f"Unsupported smolgen_relation_norm: {cfg.smolgen_relation_norm!r}")
        self.smolgen_relation_coeff_norm = str(cfg.smolgen_relation_coeff_norm).lower().strip()
        if self.smolgen_relation_coeff_norm not in ("none", "rms"):
            raise ValueError(
                f"Unsupported smolgen_relation_coeff_norm: {cfg.smolgen_relation_coeff_norm!r}"
            )
        self.smolgen_relation_scale = str(cfg.smolgen_relation_scale).lower().strip()
        if self.smolgen_relation_scale not in ("none", "layer", "layer_head"):
            raise ValueError(f"Unsupported smolgen_relation_scale: {cfg.smolgen_relation_scale!r}")
        self.phase_output_adapter_enabled = bool(cfg.phase_output_adapter)
        self.phase_smolgen_enabled = bool(cfg.phase_smolgen)
        self.phase_piece_thresholds = normalize_phase_piece_thresholds(cfg.phase_piece_thresholds)
        self.embed_dim_by_layer = _resolve_embed_dims(
            scalar=int(cfg.embed_dim),
            by_layer=cfg.embed_dim_by_layer,
            num_layers=int(cfg.num_layers),
            num_heads=int(cfg.num_heads),
        )
        input_embed_dim = self.embed_dim_by_layer[0]
        output_embed_dim = self.embed_dim_by_layer[-1]
        self.ffn_mult_by_layer = _resolve_ffn_mults(
            scalar=float(cfg.ffn_mult),
            by_layer=cfg.ffn_mult_by_layer,
            num_layers=int(cfg.num_layers),
        )

  # LC0 BT4 input block: Dense(activation) → mult_gate → add_gate
        embed_in_planes = int(cfg.in_planes) + (
            _ARC_POS_CHANNELS if self.input_pos_encoding == "arc" else 0
        )
        if self.input_global_embedding == "bt4_board":
            embed_in_planes += self.input_global_embedding_channels
        self.global_board_preprocess: nn.Linear | None = None
        self.global_board_adapter: nn.Linear | None = None
        if self.input_global_embedding == "bt4_board":
            self.global_board_preprocess = nn.Linear(12 * 64, 64 * self.input_global_embedding_channels)
        elif self.input_global_embedding == "bt4_board_adapter":
            with torch.random.fork_rng(devices=[]):
                self.global_board_preprocess = nn.Linear(12 * 64, 64 * self.input_global_embedding_channels)
                self.global_board_adapter = nn.Linear(
                    self.input_global_embedding_channels,
                    input_embed_dim,
                    bias=False,
                )
            nn.init.zeros_(self.global_board_adapter.weight)
        self.embed = nn.Linear(embed_in_planes, input_embed_dim)
        self.register_buffer(
            "arc_pos_encoding",
            _make_arc_pos_encoding()
            if self.input_pos_encoding in ("arc", "arc_adapter")
            else torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "arc_relation_basis_flat",
            _make_arc_relation_basis().flatten(1)
            if self.arc_attention_bias == "basic"
            else torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.arc_attention_weight: nn.Parameter | None = None
        if self.arc_attention_bias == "basic":
            self.arc_attention_weight = nn.Parameter(
                torch.zeros(cfg.num_layers, cfg.num_heads, _ARC_RELATION_CHANNELS)
            )
  # Dynamic board relations: per-layer, per-head scalar weight per relation,
  # zero-init so warm-started checkpoints are bit-identical at step 0. The
  # bias term is skipped entirely when no relations tensor is supplied
  # (old shards / paths that don't transport relations).
        self.dynamic_relation_count = int(cfg.dynamic_relation_count)
        self.dynamic_relation_weight: nn.Parameter | None = None
        if bool(cfg.use_dynamic_relations):
            self.dynamic_relation_weight = nn.Parameter(
                torch.zeros(cfg.num_layers, cfg.num_heads, self.dynamic_relation_count)
            )
  # Policy relation bias applies to policy_own ONLY (the head MCTS uses);
  # the soft/sf/future heads train unbiased. Intentional: the experiment
  # measures search-relevant marginal value first.
        self.policy_relation_weight: nn.Parameter | None = None
        if bool(cfg.policy_dynamic_relations):
            self.policy_relation_weight = nn.Parameter(
                torch.zeros(self.dynamic_relation_count)
            )
        self.pos_adapter: nn.Linear | None = None
        if self.input_pos_encoding == "arc_adapter":
            self.pos_adapter = nn.Linear(_ARC_POS_CHANNELS, input_embed_dim, bias=False)
            nn.init.zeros_(self.pos_adapter.weight)
  # ma_gating: multiplicative gate (init 1, non-negative) then additive gate (init 0)
  # Store raw param; apply softplus in forward to enforce non-negativity (LC0 uses NonNeg constraint).
        self._embed_gate_mul_raw = nn.Parameter(torch.full((input_embed_dim,), _softplus_inverse(1.0)))
        self.embed_gate_add = nn.Parameter(torch.zeros(input_embed_dim))
        self.square_embed_gate_mul_raw: nn.Parameter | None = None
        self.square_embed_gate_add: nn.Parameter | None = None
        if self.input_square_embedding in ("add", "ma_gate"):
            self.square_embed_gate_add = nn.Parameter(torch.zeros(64, input_embed_dim))
        if self.input_square_embedding == "ma_gate":
            self.square_embed_gate_mul_raw = nn.Parameter(
                torch.full((64, input_embed_dim), _softplus_inverse(1.0))
            )
        self.smolgen: Smolgen | None = None
        self.layer_smolgens: nn.ModuleList | None = None
        self.smolgen_layer_scale: nn.Parameter | None = None
        self.smolgen_layer_head_scale: nn.Parameter | None = None
        if self.smolgen_bias_scale == "layer":
            self.smolgen_layer_scale = nn.Parameter(torch.ones(cfg.num_layers))
        elif self.smolgen_bias_scale == "layer_head":
            self.smolgen_layer_head_scale = nn.Parameter(torch.ones(cfg.num_layers, cfg.num_heads))
        if cfg.use_smolgen and self.smolgen_mode == "per_layer":
            shared_gen = nn.Linear(self.smolgen_gen_sz, 64 * 64, bias=False)
            self.layer_smolgens = nn.ModuleList(
                [
                    Smolgen(
                        self.embed_dim_by_layer[layer_idx],
                        cfg.num_heads,
                        hidden_channels=self.smolgen_hidden_channels,
                        hidden_sz=self.smolgen_hidden_sz,
                        gen_sz=self.smolgen_gen_sz,
                        gen_weight=shared_gen,
                        use_relation_basis=bool(cfg.smolgen_relation_basis),
                        relation_norm=self.smolgen_relation_norm,
                        relation_coeff_norm=self.smolgen_relation_coeff_norm,
                        relation_scale=self.smolgen_relation_scale,
                        pooling=self.smolgen_pooling,
                        phase_conditioning=self.phase_smolgen_enabled,
                    )
                    for layer_idx in range(cfg.num_layers)
                ]
            )
        elif cfg.use_smolgen:
  # Legacy repo mode: one smolgen bias computed once and reused by all layers.
            self.smolgen = Smolgen(
                input_embed_dim,
                cfg.num_heads,
                hidden_channels=self.smolgen_hidden_channels,
                hidden_sz=self.smolgen_hidden_sz,
                gen_sz=self.smolgen_gen_sz,
                use_relation_basis=bool(cfg.smolgen_relation_basis),
                relation_norm=self.smolgen_relation_norm,
                relation_coeff_norm=self.smolgen_relation_coeff_norm,
                relation_scale=self.smolgen_relation_scale,
                pooling=self.smolgen_pooling,
                phase_conditioning=self.phase_smolgen_enabled,
            )
        residual_branch_scale = (2.0 * float(cfg.num_layers)) ** -0.25 if cfg.use_deepnorm else 1.0
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.embed_dim_by_layer[layer_idx],
                    cfg.num_heads,
                    ffn_mult=self.ffn_mult_by_layer[layer_idx],
                    dropout=cfg.dropout,
                    use_nla=cfg.use_nla,
                    qkv_projection=self.qkv_projection,
                    use_qk_rmsnorm=cfg.use_qk_rmsnorm,
                    residual_branch_scale=residual_branch_scale,
                )
                for layer_idx in range(cfg.num_layers)
            ]
        )
        self.width_adapters = nn.ModuleList(
            [
                _make_width_adapter(self.embed_dim_by_layer[idx], self.embed_dim_by_layer[idx + 1])
                for idx in range(int(cfg.num_layers) - 1)
            ]
        )
        if cfg.use_deepnorm:
            self._apply_deepnorm_init()

        self.policy_own = AttentionPolicyHead(output_embed_dim, policy_encoding=self.policy_encoding)
        self.policy_soft = AttentionPolicyHead(output_embed_dim, policy_encoding=self.policy_encoding)
        self.policy_sf = AttentionPolicyHead(output_embed_dim, policy_encoding=self.policy_encoding)
        self.policy_future = AttentionPolicyHead(output_embed_dim, policy_encoding=self.policy_encoding)

        self.value_wdl = ValueHead(output_embed_dim, 3)
        self.value_sf_eval = ValueHead(output_embed_dim, 3)
        self.value_categorical = ValueHead(output_embed_dim, 32)

        self.volatility = VolatilityHead(output_embed_dim)
        self.sf_volatility = VolatilityHead(output_embed_dim)
        self.moves_left = ScalarHead(output_embed_dim)
        self.phase_output_adapter: PhaseTokenAdapter | None = None
        if self.phase_output_adapter_enabled:
            self.phase_output_adapter = PhaseTokenAdapter(
                output_embed_dim,
                int(cfg.phase_output_adapter_dim),
            )

    def _apply_deepnorm_init(self) -> None:
        """Scale newly initialized branch weights for DeepNorm-style post-norm runs."""
        beta = (8.0 * float(self.cfg.num_layers)) ** -0.25
        with torch.no_grad():
            for module in self.blocks:
                if not isinstance(module, TransformerBlock):
                    continue
                block = module
                modules: list[nn.Linear] = [block.out_proj]
                if block.use_nla:
                    for proj in (block.q_proj, block.k_proj, block.v_proj):
                        if isinstance(proj, _NLAProjection):
                            modules.extend(m for m in proj.net if isinstance(m, nn.Linear))
                elif block.qkv_projection == "split":
                    for proj in (block.q_proj, block.k_proj, block.v_proj):
                        if isinstance(proj, nn.Linear):
                            modules.append(proj)
                else:
                    modules.append(block.qkv_proj)
                modules.extend(m for m in block.ffn if isinstance(m, nn.Linear))
                for module in modules:
                    module.weight.mul_(beta)

    def _smolgen_scale_for_layer(self, idx: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        if self.smolgen_layer_scale is not None:
            return self.smolgen_layer_scale[idx].to(dtype=dtype, device=device).view(1, 1, 1, 1)
        if self.smolgen_layer_head_scale is not None:
            return self.smolgen_layer_head_scale[idx].to(dtype=dtype, device=device).view(1, -1, 1, 1)
        return None

    def _arc_attention_bias_for_layer(
        self,
        idx: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        weight = self.arc_attention_weight
        if weight is None:
            return None
        basis = cast(torch.Tensor, self._buffers["arc_relation_basis_flat"]).to(dtype=dtype, device=device)
        flat = torch.matmul(weight[idx].to(dtype=dtype, device=device), basis)
        return flat.view(1, self.cfg.num_heads, 64, 64)

    def _attention_bias_for_layer(
        self,
        idx: int,
        smolgen_bias: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        bias = smolgen_bias
        if bias is not None:
            bias = _normalize_attention_bias(bias, self.smolgen_bias_norm)
            scale = self._smolgen_scale_for_layer(idx, dtype=bias.dtype, device=bias.device)
            if scale is not None:
                bias = bias * scale
        arc_bias = self._arc_attention_bias_for_layer(idx, dtype=dtype, device=device)
        if arc_bias is not None:
            bias = arc_bias if bias is None else bias + arc_bias
        return bias

    def _phase_indices_from_input(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.phase_output_adapter_enabled and not self.phase_smolgen_enabled:
            return None
        low, high = self.phase_piece_thresholds
        # Planes 0:12 are the current position's 12 piece-occupancy planes in
        # every shipping LC0 encoding (legacy and root-history), so summing them
        # yields the exact root piece count. Assumes that input plane layout.
        piece_counts = torch.round(x[:, :12].float().sum(dim=(1, 2, 3))).to(torch.long)
        phase = torch.ones_like(piece_counts)
        phase = torch.where(piece_counts <= int(low), torch.zeros_like(phase), phase)
        phase = torch.where(piece_counts > int(high), torch.full_like(phase, 2), phase)
        return phase

    def _relations_to_float(self, relations: torch.Tensor | None, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        """(B, K, 64, 64) uint8/bool/float -> float on the compute device."""
        if relations is None:
            return None
        if self.dynamic_relation_weight is None and self.policy_relation_weight is None:
            return None
        rel = relations.to(device=device)
        if rel.dtype != dtype:
            rel = rel.to(dtype=dtype)
        return rel

    def _encode_tokens(
        self, x: torch.Tensor, relations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, c, h, w = x.shape
        assert (h, w) == (8, 8)
        phase_idx = self._phase_indices_from_input(x)
        tokens = x.reshape(b, c, 64).permute(0, 2, 1).contiguous()  # (B,64,C)
        pos: torch.Tensor | None = None
        if self.input_pos_encoding in ("arc", "arc_adapter"):
            pos_encoding = self._buffers["arc_pos_encoding"]
            assert pos_encoding is not None
            pos = pos_encoding.to(dtype=tokens.dtype, device=tokens.device)
        if self.input_pos_encoding == "arc":
            assert pos is not None
            tokens = torch.cat((tokens, pos.expand(b, -1, -1)), dim=-1)
        global_board_preprocess = self.global_board_preprocess
        global_tokens: torch.Tensor | None = None
        if global_board_preprocess is not None:
            board_summary = x[:, :12].reshape(b, 12 * 64)
            global_tokens = global_board_preprocess(board_summary).view(
                b,
                64,
                self.input_global_embedding_channels,
            )
            if self.input_global_embedding == "bt4_board":
                assert global_tokens is not None
                tokens = torch.cat((tokens, global_tokens), dim=-1)

  # BT4 input: Dense(mish) → mult_gate(non-neg) → add_gate
        t = F.mish(self.embed(tokens))
        t = t * F.softplus(self._embed_gate_mul_raw) + self.embed_gate_add
        if self.square_embed_gate_mul_raw is not None:
            t = t * F.softplus(self.square_embed_gate_mul_raw).unsqueeze(0)
        if self.square_embed_gate_add is not None:
            t = t + self.square_embed_gate_add.unsqueeze(0)
        if self.global_board_adapter is not None:
            assert global_tokens is not None
            t = t + self.global_board_adapter(global_tokens)
        if self.input_pos_encoding == "arc_adapter":
            assert pos is not None
            pos_adapter = self.pos_adapter
            assert pos_adapter is not None
            t = t + pos_adapter(pos.expand(b, -1, -1))
        relations_f = self._relations_to_float(relations, dtype=t.dtype, device=t.device)
        shared_smolgen_bias = self.smolgen(t, phase_idx) if self.smolgen is not None else None
        balance_terms: list[torch.Tensor] = []
        balance_weight = float(getattr(self, "resid_channel_balance_weight", 0.0) or 0.0)
        for idx, blk in enumerate(self.blocks):
            smolgen_bias = shared_smolgen_bias
            if self.layer_smolgens is not None:
                smolgen_bias = self.layer_smolgens[idx](t, phase_idx)
            smolgen_bias = self._attention_bias_for_layer(
                idx,
                smolgen_bias,
                dtype=t.dtype,
                device=t.device,
            )
            if self.dynamic_relation_weight is not None and relations_f is not None:
                dyn = torch.einsum(
                    "hk,bkft->bhft",
                    self.dynamic_relation_weight[idx].to(dtype=t.dtype),
                    relations_f,
                )
                smolgen_bias = dyn if smolgen_bias is None else smolgen_bias + dyn
            t_in = t
            if self._use_grad_ckpt and self.training:
                t = cast(torch.Tensor, grad_checkpoint(blk, t_in, smolgen_bias, use_reentrant=False))
            else:
                t = cast(torch.Tensor, blk(t_in, smolgen_bias))
            if self.training and balance_weight > 0.0:
                delta = (t - t_in).float()
                energy = delta.square().mean(dim=(0, 1)).clamp_min(1e-12)
                log_energy = energy.log()
                balance_terms.append((log_energy - log_energy.mean()).square().mean())
            drop_p = float(getattr(self, "resid_channel_dropout", 0.0) or 0.0)
            if self.training and drop_p > 0.0:
                keep_p = max(1e-6, 1.0 - drop_p)
                mask = (torch.rand((1, 1, t.shape[-1]), device=t.device) < keep_p).to(t.dtype)
                t = t * (mask / keep_p)
            if idx < len(self.width_adapters):
                t = self.width_adapters[idx](t)
        self._last_channel_balance_loss = (
            torch.stack(balance_terms).mean().to(dtype=t.dtype)
            if balance_terms
            else None
        )
        phase_output_adapter = self.phase_output_adapter
        if phase_output_adapter is not None:
            assert phase_idx is not None
            t = phase_output_adapter(t, phase_idx)
        return cast(torch.Tensor, t), relations_f

    def forward_legal_policy(
        self,
        x: torch.Tensor,
        legal_flat: torch.Tensor,
        legal_counts: torch.Tensor,
        relations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        t, relations_f = self._encode_tokens(x, relations)
        pol_ft_bias: torch.Tensor | None = None
        if self.policy_relation_weight is not None and relations_f is not None:
            pol_ft_bias = torch.einsum(
                "k,bkft->bft",
                self.policy_relation_weight.to(dtype=relations_f.dtype),
                relations_f,
            )
        return {
            "policy_own": self.policy_own.forward_legal(t, legal_flat, legal_counts, ft_bias=pol_ft_bias),
            "wdl": self.value_wdl(t),
        }

    def forward_legal_policy_rows(
        self,
        x: torch.Tensor,
        legal_flat: torch.Tensor,
        legal_rows: torch.Tensor,
        relations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        t, relations_f = self._encode_tokens(x, relations)
        pol_ft_bias: torch.Tensor | None = None
        if self.policy_relation_weight is not None and relations_f is not None:
            pol_ft_bias = torch.einsum(
                "k,bkft->bft",
                self.policy_relation_weight.to(dtype=relations_f.dtype),
                relations_f,
            )
        return {
            "policy_own": self.policy_own.forward_legal_rows(t, legal_flat, legal_rows, ft_bias=pol_ft_bias),
            "wdl": self.value_wdl(t),
        }

    def forward(
        self,
        x: torch.Tensor,
        legal_flat: torch.Tensor | None = None,
        legal_counts: torch.Tensor | None = None,
        legal_rows: torch.Tensor | None = None,
        relations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if legal_flat is not None or legal_counts is not None or legal_rows is not None:
            if legal_flat is None:
                raise ValueError("legal_flat must be provided for legal policy forward")
            if legal_rows is not None:
                return self.forward_legal_policy_rows(x, legal_flat, legal_rows, relations=relations)
            if legal_counts is None:
                raise ValueError("legal_counts must be provided with legal_flat")
            return self.forward_legal_policy(x, legal_flat, legal_counts, relations=relations)

        t, relations_f = self._encode_tokens(x, relations)
        pol_ft_bias: torch.Tensor | None = None
        if self.policy_relation_weight is not None and relations_f is not None:
            pol_ft_bias = torch.einsum(
                "k,bkft->bft",
                self.policy_relation_weight.to(dtype=relations_f.dtype),
                relations_f,
            )

        if self._inference_only:
            return {
                "policy_own": self.policy_own(t, ft_bias=pol_ft_bias),
                "wdl": self.value_wdl(t),
            }

        return {
            "policy_own": self.policy_own(t, ft_bias=pol_ft_bias),
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
