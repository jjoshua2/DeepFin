from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x over mask==1 entries. mask is broadcastable to x."""
    mask = mask.to(x.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def policy_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with a soft target distribution.

    logits: (B,A)
    target_probs: (B,A) with rows summing to 1 over legal moves
    """
    logp = F.log_softmax(logits, dim=-1)
    return -(target_probs * logp).sum(dim=-1)


def wdl_cross_entropy(logits: torch.Tensor, target_wdl: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target_wdl, reduction="none")


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Cross-entropy with a soft target distribution.

    logits: (B,K)
    target_probs: (B,K) non-negative (does not need to be perfectly normalized)
    """
    p = target_probs.to(torch.float32).clamp_min(0.0)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1)


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

    # LC0-style illegal move masking: set illegal move logits to -1e9 before softmax.
    _legal_mask = batch.get("legal_mask")
    _has_legal = batch.get("has_legal_mask")
    def _apply_legal_mask(logits: torch.Tensor) -> torch.Tensor:
        if _legal_mask is None:
            return logits
        mask_active = _has_legal.unsqueeze(-1) if _has_legal is not None else 1.0
        penalty = (1.0 - _legal_mask) * -1e9
        return logits + penalty * mask_active

    # policy
    base_policy_logits = outputs["policy"] if "policy" in outputs else outputs.get("policy_own")
    if base_policy_logits is None:
        raise KeyError("Model outputs must include either 'policy' or 'policy_own'.")

    pol_ce = policy_cross_entropy(_apply_legal_mask(base_policy_logits), batch["policy_t"])
    zero_loss = torch.zeros_like(pol_ce)
    has_policy = _get_mask(batch, "has_policy", default=1.0)

    # soft policy (temp2)
    has_soft = _get_mask(batch, "has_policy_soft")
    soft_logits = outputs.get("policy_soft", base_policy_logits)
    soft_target = batch.get("policy_soft_t")
    soft_ce = policy_cross_entropy(_apply_legal_mask(soft_logits), soft_target) if soft_target is not None else zero_loss

    # future policy (t+2) — do NOT apply legal_mask: future_policy_t uses move indices
    # from position t+2, so t's legal mask would incorrectly mask most moves.
    has_future = _get_mask(batch, "has_future")
    future_logits = outputs.get("policy_future", base_policy_logits)
    future_target = batch.get("future_policy_t")
    future_ce = policy_cross_entropy(future_logits, future_target) if future_target is not None else zero_loss

    # value
    wdl_ce = wdl_cross_entropy(outputs["wdl"], batch["wdl_t"])

    # SF move prediction — do NOT apply legal_mask: sf_policy_t uses move indices from the
    # next board position (after the network moves).
    has_sf_move = _get_mask(batch, "has_sf_move")
    has_sf_policy = batch.get("has_sf_policy")
    if has_sf_policy is None:
        has_sf_policy = has_sf_move

    sf_pol_logits = outputs.get("policy_sf")
    sf_policy_target = batch.get("sf_policy_t")
    if sf_pol_logits is None or sf_policy_target is None:
        sf_move_ce = zero_loss
    else:
        sf_move_ce = soft_cross_entropy(sf_pol_logits, sf_policy_target)

    # sf_eval
    has_sf_wdl = _get_mask(batch, "has_sf_wdl")
    sf_wdl_raw = batch.get("sf_wdl")
    sf_wdl_probs = None
    if sf_wdl_raw is not None:
        sf_wdl_probs = sf_wdl_raw.clamp_min(0.0)
        sf_wdl_probs = sf_wdl_probs / sf_wdl_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    sf_eval_logits = outputs.get("sf_eval")
    if sf_eval_logits is None or sf_wdl_probs is None:
        sf_eval_ce = zero_loss
    else:
        sf_eval_ce = soft_cross_entropy(sf_eval_logits, sf_wdl_probs)

    # moves_left
    has_moves_left = _get_mask(batch, "has_moves_left")
    ml_pred = outputs.get("moves_left")
    moves_left_t = batch.get("moves_left")
    if ml_pred is None or moves_left_t is None:
        ml_loss = zero_loss
    else:
        ml_loss = F.smooth_l1_loss(ml_pred.squeeze(-1), moves_left_t, reduction="none")

    # categorical value
    has_cat = _get_mask(batch, "has_categorical")
    cat_logits = outputs.get("categorical")
    categorical_t = batch.get("categorical_t")
    if cat_logits is None or categorical_t is None:
        cat_ce = zero_loss
    else:
        cat_ce = policy_cross_entropy(cat_logits, categorical_t)

    # volatility (network)
    has_vol = _get_mask(batch, "has_volatility")
    vol_pred = outputs.get("volatility")
    volatility_t = batch.get("volatility_t")
    if vol_pred is None or volatility_t is None:
        vol_loss = zero_loss
    else:
        vol_loss = F.huber_loss(vol_pred, volatility_t, delta=0.1, reduction="none").mean(dim=-1)

    # volatility (Stockfish)
    has_sf_vol = _get_mask(batch, "has_sf_volatility")
    sf_vol_pred = outputs.get("sf_volatility")
    sf_volatility_t = batch.get("sf_volatility_t")
    if sf_vol_pred is None or sf_volatility_t is None:
        sf_vol_loss = zero_loss
    else:
        sf_vol_loss = F.huber_loss(sf_vol_pred, sf_volatility_t, delta=0.1, reduction="none").mean(dim=-1)

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

    # Compute each masked_mean once, reuse for both total and return dict.
    m_policy = masked_mean(pol_ce, net_mask * has_policy)
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
    }
