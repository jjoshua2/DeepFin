from __future__ import annotations

import torch
import torch.nn.functional as F

# Phase buckets for per-phase loss reporting. `moves_left` is plies-remaining /
# max_plies so 1.0 = opening, 0.0 = endgame. Thresholds calibrated from
# empirical P33/P67 of recent selfplay shards (data is skewed toward shorter
# games due to adjudication, so a naive 0.33/0.66 split puts ~11% in open
# and ~51% in mid). Re-derive periodically — `scripts/eval_phase_thresholds`
# (or the inline grep in trainable_phases) when the distribution drifts.
_PHASE_OPEN_THRESHOLD = 0.45
_PHASE_END_THRESHOLD = 0.31


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x over mask==1 entries. mask is broadcastable to x."""
    mask = mask.to(x.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with a soft target distribution.

    ``target_probs`` must already be normalized along the last axis (rows
    summing to 1 — true for ``policy_t``, ``policy_soft_t``,
    ``future_policy_t``, ``categorical_t``, and the post-clamp
    ``sf_wdl_probs``). Callers working from raw counts must normalize
    first.
    """
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def apply_mask_to_logits(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    mask_key: str,
    has_key: str,
) -> torch.Tensor:
    """LC0-style illegal-move masking: `(1 - mask) * -1e9` added to logits,
    gated by `has_key` so rows without a mask pass through unchanged.
    """
    mask = batch.get(mask_key)
    if mask is None:
        return logits
    has = batch.get(has_key)
    active = has.unsqueeze(-1) if has is not None else 1.0
    return logits + (1.0 - mask) * -1e9 * active


def _huber_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=0.1, reduction="none").mean(dim=-1)


def _get_mask(batch: dict[str, torch.Tensor], key: str, *, default: float = 0.0) -> torch.Tensor:
    """Get a per-sample mask from batch, defaulting to a constant tensor."""
    v = batch.get(key)
    if v is not None:
        return v
    return torch.full((batch["x"].shape[0],), default, device=batch["x"].device)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    w_policy: float = 1.0,
    w_soft: float = 0.5,
    w_future: float = 0.15,
    w_wdl: float = 1.0,
    w_sf_move: float = 0.15,
    w_sf_eval: float = 0.15,
    w_categorical: float = 0.10,
    w_volatility: float = 0.05,
    w_sf_volatility: float | None = None,
    w_moves_left: float = 0.02,
    w_sf_wdl: float = 0.0,
    sf_wdl_conf_power: float = 0.0,
    sf_wdl_draw_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute multi-head training loss."""
    net_mask = _get_mask(batch, "is_network_turn", default=1.0).to(torch.float32)

    def _apply_legal_mask(logits: torch.Tensor) -> torch.Tensor:
        return apply_mask_to_logits(logits, batch, "legal_mask", "has_legal_mask")

    base_policy_logits = outputs["policy"] if "policy" in outputs else outputs.get("policy_own")
    if base_policy_logits is None:
        raise KeyError("Model outputs must include either 'policy' or 'policy_own'.")

    pol_ce = soft_cross_entropy(_apply_legal_mask(base_policy_logits), batch["policy_t"])
    zero_loss = torch.zeros_like(pol_ce)
    has_policy = _get_mask(batch, "has_policy", default=1.0)

    has_soft = _get_mask(batch, "has_policy_soft")
    soft_logits = outputs.get("policy_soft", base_policy_logits)
    soft_target = batch.get("policy_soft_t")
    soft_ce = soft_cross_entropy(_apply_legal_mask(soft_logits), soft_target) if soft_target is not None else zero_loss

  # Future policy (t+2): target and legal mask are in the t+2 move space.
    has_future = _get_mask(batch, "has_future")
    future_logits = outputs.get("policy_future", base_policy_logits)
    future_target = batch.get("future_policy_t")
    if future_target is not None:
        future_ce = soft_cross_entropy(
            apply_mask_to_logits(future_logits, batch, "future_legal_mask", "has_future_legal_mask"),
            future_target,
        )
    else:
        future_ce = zero_loss

    wdl_ce = F.cross_entropy(outputs["wdl"], batch["wdl_t"], reduction="none")

  # SF move prediction: target and legal mask are in the t+1 move space (opp POV).
    has_sf_move = _get_mask(batch, "has_sf_move")
    has_sf_policy = _get_mask(batch, "has_sf_policy") if "has_sf_policy" in batch else has_sf_move

    sf_pol_logits = outputs.get("policy_sf")
    sf_policy_target = batch.get("sf_policy_t")
    if sf_pol_logits is None or sf_policy_target is None:
        sf_move_ce = zero_loss
    else:
        sf_move_ce = soft_cross_entropy(
            apply_mask_to_logits(sf_pol_logits, batch, "sf_legal_mask", "has_sf_legal_mask"),
            sf_policy_target,
        )

    has_sf_wdl = _get_mask(batch, "has_sf_wdl")
    sf_wdl_raw = batch.get("sf_wdl")
    sf_wdl_probs = None
    if sf_wdl_raw is not None:
        sf_wdl_probs = sf_wdl_raw.clamp_min(0.0)
        sf_wdl_probs = sf_wdl_probs / sf_wdl_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    sf_eval_logits = outputs.get("sf_eval")
    sf_eval_ce = soft_cross_entropy(sf_eval_logits, sf_wdl_probs) if sf_eval_logits is not None and sf_wdl_probs is not None else zero_loss

    has_moves_left = _get_mask(batch, "has_moves_left")
    ml_pred = outputs.get("moves_left")
    moves_left_t = batch.get("moves_left")
    if ml_pred is None or moves_left_t is None:
        ml_loss = zero_loss
    else:
        ml_loss = F.smooth_l1_loss(ml_pred.squeeze(-1), moves_left_t, reduction="none")

    has_cat = _get_mask(batch, "has_categorical")
    cat_logits = outputs.get("categorical")
    categorical_t = batch.get("categorical_t")
    cat_ce = soft_cross_entropy(cat_logits, categorical_t) if cat_logits is not None and categorical_t is not None else zero_loss

    has_vol = _get_mask(batch, "has_volatility")
    vol_pred = outputs.get("volatility")
    volatility_t = batch.get("volatility_t")
    vol_loss = _huber_per_sample(vol_pred, volatility_t) if vol_pred is not None and volatility_t is not None else zero_loss

    has_sf_vol = _get_mask(batch, "has_sf_volatility")
    sf_vol_pred = outputs.get("sf_volatility")
    sf_volatility_t = batch.get("sf_volatility_t")
    sf_vol_loss = _huber_per_sample(sf_vol_pred, sf_volatility_t) if sf_vol_pred is not None and sf_volatility_t is not None else zero_loss

  # Loss weights — float() casts defend against numpy scalars from Ray Tune config mutation
    w_sf_volatility = float(w_sf_volatility) if w_sf_volatility is not None else float(w_volatility)
    sf_wdl_conf_power = max(0.0, float(sf_wdl_conf_power))
    sf_wdl_draw_scale = max(0.0, float(sf_wdl_draw_scale))

  # SF-WDL confidence damping: (1 - draw_prob)^power, with optional draw_scale
    sf_wdl_mask = net_mask * has_sf_wdl
    if sf_wdl_probs is not None:
        if sf_wdl_conf_power > 0.0:
            sf_conf = (1.0 - sf_wdl_probs[:, 1]).clamp(0.0, 1.0).pow(sf_wdl_conf_power)
            sf_wdl_mask = sf_wdl_mask * sf_conf
        if sf_wdl_draw_scale != 1.0:
            draw_mask = (batch["wdl_t"] == 1).to(torch.float32)
            sf_wdl_mask = sf_wdl_mask * (1.0 - draw_mask + draw_mask * sf_wdl_draw_scale)

    if sf_wdl_probs is None:
        sf_wdl_soft_ce = zero_loss
    else:
        sf_wdl_soft_ce = soft_cross_entropy(outputs["wdl"], sf_wdl_probs)

  # Precompute the per-sample base mask for each head so the downstream
  # split reductions don't recompute `net_mask * has_X` once per bucket.
    pol_base = net_mask * has_policy
    m_policy = masked_mean(pol_ce, pol_base)
    m_soft = masked_mean(soft_ce, net_mask * has_soft)
    m_future = masked_mean(future_ce, net_mask * has_future)
    m_wdl = masked_mean(wdl_ce, net_mask)
    m_sf_wdl = masked_mean(sf_wdl_soft_ce, sf_wdl_mask)
    m_sf_move = masked_mean(sf_move_ce, net_mask * has_sf_policy)
    m_sf_eval = masked_mean(sf_eval_ce, net_mask * has_sf_wdl)
    m_cat = masked_mean(cat_ce, net_mask * has_cat)
    m_vol = masked_mean(vol_loss, net_mask * has_vol)
    m_sf_vol = masked_mean(sf_vol_loss, net_mask * has_sf_vol)
    m_ml = masked_mean(ml_loss, net_mask * has_moves_left)

  # Gated on `has_is_selfplay` so legacy shards without the tag are excluded
  # from the split (they won't contribute to either selfplay_ or curriculum_ keys).
    has_is_sp = _get_mask(batch, "has_is_selfplay").to(torch.float32)
    is_sp_bool = _get_mask(batch, "is_selfplay", default=0.0).to(torch.float32)
    sp_mask = has_is_sp * is_sp_bool
    cur_mask = has_is_sp - sp_mask

    ml_val = _get_mask(batch, "moves_left", default=1.0).to(torch.float32)
    open_mask = has_moves_left * (ml_val > _PHASE_OPEN_THRESHOLD).to(torch.float32)
    end_mask = has_moves_left * (ml_val < _PHASE_END_THRESHOLD).to(torch.float32)
    mid_mask = has_moves_left - open_mask - end_mask

    split_masks = (
        ("selfplay", sp_mask),
        ("curriculum", cur_mask),
        ("open", open_mask),
        ("mid", mid_mask),
        ("end", end_mask),
    )
    split_losses: dict[str, torch.Tensor] = {}
    for suffix, m in split_masks:
        split_losses[f"policy_loss_{suffix}"] = masked_mean(pol_ce, pol_base * m)
        split_losses[f"wdl_loss_{suffix}"] = masked_mean(wdl_ce, net_mask * m)

    total = (
        float(w_policy) * m_policy
        + float(w_soft) * m_soft
        + float(w_future) * m_future
        + float(w_wdl) * m_wdl
        + float(w_sf_wdl) * m_sf_wdl
        + float(w_sf_move) * m_sf_move
        + float(w_sf_eval) * m_sf_eval
        + float(w_categorical) * m_cat
        + float(w_volatility) * m_vol
        + float(w_sf_volatility) * m_sf_vol
        + float(w_moves_left) * m_ml
    )

    return {
        "total": total,
        "policy_ce": m_policy,
        "wdl_ce": m_wdl,
        "sf_wdl_ce": m_sf_wdl,
        "soft_policy_ce": m_soft,
        "future_policy_ce": m_future,
        "sf_move_ce": m_sf_move,
        "sf_eval_ce": m_sf_eval,
        "categorical_ce": m_cat,
        "volatility": m_vol,
        "sf_volatility": m_sf_vol,
        "moves_left": m_ml,
        **split_losses,
        "frac_is_selfplay": masked_mean(is_sp_bool, has_is_sp),
        "frac_tagged": masked_mean(has_is_sp, net_mask),
    }


def wdl_calibration_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    n_bins: int = 10,
) -> dict[str, torch.Tensor]:
    """Per-sample Brier + per-bin calibration aggregates (all on-device, no syncs).

    Returns accumulator-friendly sums: callers sum these across eval batches and
    derive global Brier/ECE at the end. Doing per-batch ECE and then averaging
    weights small batches the same as large ones, which is wrong.

    Keys in the returned dict:
      - ``brier_sum``: scalar, sum of per-sample Brier over the batch
      - ``n``: scalar, number of samples in the batch
      - ``bin_conf_sum``: (n_bins,) sum of max-prob confidence per bin
      - ``bin_correct_sum``: (n_bins,) sum of correctness (0/1) per bin
      - ``bin_n``: (n_bins,) count per bin
    """
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(target.to(torch.int64), num_classes=3).to(probs.dtype)
    brier_per_sample = ((probs - one_hot) ** 2).sum(dim=-1)

    conf, pred = probs.max(dim=-1)
    correct = (pred == target).to(probs.dtype)
  # bucketize boundaries: n_bins-1 inner edges so bins span [0,1/n), ... ,[(n-1)/n, 1].
  # Clamp to [0, n_bins-1] so the topmost conf==1.0 doesn't land in an n_bins slot.
    inner_edges = torch.linspace(
        1.0 / n_bins, 1.0 - 1.0 / n_bins, n_bins - 1, device=logits.device, dtype=probs.dtype
    )
    bin_idx = torch.bucketize(conf.detach(), inner_edges).clamp_max(n_bins - 1)
    bin_n = torch.zeros(n_bins, device=logits.device, dtype=probs.dtype).scatter_add_(
        0, bin_idx, torch.ones_like(conf)
    )
    bin_conf_sum = torch.zeros(n_bins, device=logits.device, dtype=probs.dtype).scatter_add_(
        0, bin_idx, conf
    )
    bin_correct_sum = torch.zeros(n_bins, device=logits.device, dtype=probs.dtype).scatter_add_(
        0, bin_idx, correct
    )
    return {
        "brier_sum": brier_per_sample.sum(),
        "n": torch.tensor(float(target.numel()), device=logits.device, dtype=probs.dtype),
        "bin_conf_sum": bin_conf_sum,
        "bin_correct_sum": bin_correct_sum,
        "bin_n": bin_n,
    }


def wdl_brier_ece_from_stats(stats: dict[str, torch.Tensor]) -> tuple[float, float]:
    """Combine accumulated calibration stats into (mean_brier, global_ece).

    ECE = sum_b |correct_sum[b] - conf_sum[b]| / n_total — algebraically identical
    to the standard definition, since bin_acc[b]*bin_n[b] = correct_sum[b] and
    bin_conf[b]*bin_n[b] = conf_sum[b], and the per-bin weight is bin_n[b]/n_total.
    """
    n = float(stats["n"].item())
    if n <= 0:
        return 0.0, 0.0
    brier = float(stats["brier_sum"].item()) / n
    diff = (stats["bin_correct_sum"] - stats["bin_conf_sum"]).abs().sum()
    ece = float(diff.item()) / n
    return brier, ece
