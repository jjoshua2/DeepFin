"""Sparse SF-policy cross-entropy over gathered log-probs.

Computes the ``policy_sf`` training loss directly from the stored sparse
MultiPV rows (``sf_multipv_raw``) instead of a dense ``sf_policy_target``
vector — the follow-up sketched in future_ideas.md after shard schema v2.

Equality with the dense path (exact, not approximate): the live dense target
is built as

    p_top = softmax(scores / T)          over scoreable candidates
    p_sf  = scatter_add(p_top) * (1-s);  p_sf[legal] += s / L

which sums to exactly 1, so the final renormalize in
``stockfish_turn._build_sf_policy_target`` is a no-op and the soft CE
factorizes into two analytic terms over the SAME masked log-softmax the
dense path uses:

    CE = -(1-s) * sum_k p_k * lsm[idx_k]  -  (s / L) * sum_legal lsm

Row scoring mirrors ``target_builder._row_score`` exactly, including the
native-permille no-rescale quirk (live candidate scores are raw permille
when SF WDL is present; logistic scores are fractions).

Empty-candidate fallback: when a row carries sparse labels but none of its
candidates is scoreable, the live builder degraded to a one-hot at the
(legalized) bestmove — which is exactly ``sf_move_index`` — so the sparse CE
uses p = 1 at that index.

Encoding widths: candidate indices are stored in the SHARD's policy
encoding. When the model emits compact lc0_1858 logits over full-4672
shards the indices are remapped through ``FULL_TO_COMPACT_POLICY`` (all
candidates are real moves, so the dense path's compact renormalization is a
no-op for this target and equality survives the projection). The reverse
(full logits over compact shards) widens through ``COMPACT_TO_FULL_POLICY``.
"""
from __future__ import annotations

import torch

from chess_anti_engine.moves import (
    COMPACT_POLICY_SIZE,
    COMPACT_TO_FULL_POLICY,
    FULL_TO_COMPACT_POLICY,
    POLICY_SIZE,
)
from chess_anti_engine.replay.shard import SF_CP_SENTINEL
from chess_anti_engine.train.target_builder import SfTargetParams

# Mirrors stockfish/wdl.py::mate_to_effective_cp.
_MATE_BASE_CP = 1500.0
_MATE_DEPTH_BONUS_CP = 20.0


def _index_remap_table(
    src_width: int, dst_width: int, device: torch.device,
) -> torch.Tensor | None:
    """Lookup table mapping shard-space indices to logits-space, or None
    when the widths already agree. Raises on unknown width pairs."""
    if src_width == dst_width:
        return None
    if src_width == POLICY_SIZE and dst_width == COMPACT_POLICY_SIZE:
        return torch.as_tensor(FULL_TO_COMPACT_POLICY, dtype=torch.long, device=device)
    if src_width == COMPACT_POLICY_SIZE and dst_width == POLICY_SIZE:
        return torch.as_tensor(COMPACT_TO_FULL_POLICY, dtype=torch.long, device=device)
    raise ValueError(
        f"shard policy width {src_width} is incompatible with logits width {dst_width}"
    )


def _row_scores(
    raw: torch.Tensor, *, params: SfTargetParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(scores, scoreable) per candidate row, both (B, K).

    Replicates ``target_builder._row_score``: logistic path takes precedence
    when enabled and cp/mate is present; otherwise native SF WDL permille
    (deliberately NOT rescaled — matching the live path is the point).
    """
    cp = raw[..., 1].to(torch.float32)
    mate = raw[..., 2]
    w = raw[..., 3].to(torch.float32)
    d = raw[..., 4].to(torch.float32)
    present = raw[..., 0] >= 0

    native_ok = (raw[..., 3] >= 0) & (raw[..., 4] >= 0)
    native_score = w + 0.5 * d

    if not params.sf_wdl_use_cp_logistic:
        return native_score, present & native_ok

    has_mate = mate != 0
    has_cp = raw[..., 1] != SF_CP_SENTINEL
    logistic_ok = has_mate | has_cp
    sign = torch.where(mate >= 0, 1.0, -1.0)
    bonus = (50.0 - mate.abs().to(torch.float32)).clamp_min(0.0) * _MATE_DEPTH_BONUS_CP
    eff = torch.where(has_mate, sign * (_MATE_BASE_CP + bonus), cp)
    slope = float(params.sf_wdl_cp_slope)
    width = float(params.sf_wdl_cp_draw_width)
    p_win = torch.sigmoid(slope * (eff - width))
    p_loss = torch.sigmoid(slope * (-eff - width))
    p_draw = (1.0 - p_win - p_loss).clamp_min(0.0)
    logistic_score = (p_win + 0.5 * p_draw) / (p_win + p_loss + p_draw)
    score = torch.where(logistic_ok, logistic_score, native_score)
    return score, present & (logistic_ok | native_ok)


def sparse_sf_policy_ce(
    masked_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    params: SfTargetParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample sparse SF-policy CE and its availability mask, both (B,).

    ``masked_logits`` must be the SAME legal-masked logits the dense soft CE
    consumes (the -1e9 mask shifts the log-softmax denominator). Rows are
    eligible when they carry sparse labels AND the SF legal mask (needed for
    the analytic smoothing term). Ineligible rows return 0 with mask 0 —
    callers keep the dense CE there.
    """
    raw = batch.get("sf_multipv_raw")
    has_raw = batch.get("has_sf_multipv_raw")
    legal = batch.get("sf_legal_mask")
    has_legal = batch.get("has_sf_legal_mask")
    zeros = masked_logits.new_zeros(masked_logits.shape[0])
    if raw is None or has_raw is None or legal is None or has_legal is None:
        return zeros, zeros

    lsm = torch.log_softmax(masked_logits.float(), dim=-1)
    dst_width = int(masked_logits.shape[-1])
    src_width = int(legal.shape[-1])
    remap = _index_remap_table(src_width, dst_width, masked_logits.device)

    raw = raw.to(torch.long)
    idx = raw[..., 0]
    scores, ok = _row_scores(raw, params=params)
    if remap is not None:
        mapped = remap[idx.clamp_min(0)]
        ok = ok & (mapped >= 0)
        idx = mapped

    # Candidate term: softmax(scores/T) over scoreable rows, gathered
    # log-probs. float64 like the live _softmax_np — at T ~= 0.012 the
    # z-magnitudes reach ~1e5 and fp32 rounding there moves the probs by
    # ~1%, which would break the 1e-5 dense-parity contract.
    temp = max(1e-6, float(params.sf_policy_temp))
    z = torch.where(ok, scores.double() / temp, scores.new_full((), -torch.inf, dtype=torch.float64))
    z = z - z.amax(dim=-1, keepdim=True).clamp_min(-1e30)  # stable even for all-masked rows
    expz = torch.where(ok, z.exp(), z.new_zeros(()))
    denom = expz.sum(dim=-1)
    has_cand = denom > 0
    p = (expz / denom.clamp_min(1e-300).unsqueeze(-1)).float()
    g = lsm.gather(1, idx.clamp(0, dst_width - 1))
    cand_term = (p * g).masked_fill(~ok, 0.0).sum(dim=-1)

    # Empty-candidate fallback: live built a one-hot at sf_move_index.
    sf_move_index = batch.get("sf_move_index")
    has_sf_move = batch.get("has_sf_move")
    fallback_ok = zeros.bool()
    if sf_move_index is not None and has_sf_move is not None:
        fb_idx = sf_move_index.to(torch.long)
        fb_valid = (fb_idx >= 0) & (has_sf_move > 0)
        if remap is not None:
            fb_mapped = remap[fb_idx.clamp_min(0)]
            fb_valid = fb_valid & (fb_mapped >= 0)
            fb_idx = fb_mapped
        fb_term = lsm.gather(1, fb_idx.clamp(0, dst_width - 1).unsqueeze(1)).squeeze(1)
        fallback_ok = ~has_cand & fb_valid
        cand_term = torch.where(fallback_ok, fb_term, cand_term)

    # Analytic smoothing term over the legal set.
    smooth = float(params.sf_policy_label_smooth)
    legal_aligned = legal.float()
    if remap is not None:
        if src_width == POLICY_SIZE:
            full_idx = torch.as_tensor(
                COMPACT_TO_FULL_POLICY, dtype=torch.long, device=masked_logits.device,
            )
            legal_aligned = legal_aligned.index_select(-1, full_idx)
        else:
            widened = legal_aligned.new_zeros((legal_aligned.shape[0], dst_width))
            compact_to_full = torch.as_tensor(
                COMPACT_TO_FULL_POLICY, dtype=torch.long, device=masked_logits.device,
            )
            widened.index_copy_(
                -1, compact_to_full, legal_aligned,
            )
            legal_aligned = widened
    legal_count = legal_aligned.sum(dim=-1)
    if smooth > 0.0:
        smooth_sum = (legal_aligned * lsm).sum(dim=-1)
        smooth_term = smooth * smooth_sum / legal_count.clamp_min(1.0)
        ce = -(1.0 - smooth) * cand_term - smooth_term
    else:
        ce = -cand_term

    computed = (
        (has_raw > 0)
        & (has_legal > 0)
        & (legal_count > 0)
        & (has_cand | fallback_ok)
    ).to(masked_logits.dtype)
    return ce * computed, computed
