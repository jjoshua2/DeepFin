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
    """Compute a spec-inspired loss using whatever targets we currently record.

    Currently trained:
    - policy_own (always)
    - wdl (always)
    - sf_eval (only when sf_wdl available; only on SF turns)
    - moves_left (if available)

    This is an incremental step toward the full 9-head spec loss.
    """
    is_net = batch.get("is_network_turn")
    if is_net is None:
        is_net = torch.ones((batch["x"].shape[0],), device=batch["x"].device, dtype=torch.bool)

    net_mask = is_net.to(torch.float32)

    # LC0-style illegal move masking: for positions that have a legal_mask stored,
    # set illegal move logits to -1e9 before softmax so probability doesn't leak to illegal moves.
    _legal_mask = batch.get("legal_mask")  # (B, POLICY_SIZE) float, 1=legal 0=illegal, or None
    _has_legal = batch.get("has_legal_mask")  # (B,) float
    def _apply_legal_mask(logits: torch.Tensor) -> torch.Tensor:
        if _legal_mask is None:
            return logits
        # Where has_legal_mask=1: mask illegal moves. Where has_legal_mask=0: leave unchanged.
        penalty = (1.0 - _legal_mask) * -1e9  # (B, POLICY_SIZE)
        mask_active = _has_legal.unsqueeze(-1)  # (B, 1)
        return logits + penalty * mask_active

    # policy
    base_policy_logits = outputs["policy"] if "policy" in outputs else outputs.get("policy_own")
    if base_policy_logits is None:
        raise KeyError("Model outputs must include either 'policy' or 'policy_own'.")

    pol_ce = policy_cross_entropy(_apply_legal_mask(base_policy_logits), batch["policy_t"])  # (B,)
    has_policy = batch.get("has_policy")
    if has_policy is None:
        has_policy = torch.ones((batch["x"].shape[0],), device=batch["x"].device)

    # soft policy (temp2)
    has_soft = batch.get("has_policy_soft")
    if has_soft is None:
        has_soft = torch.zeros((batch["x"].shape[0],), device=batch["x"].device)
    soft_logits = outputs["policy_soft"] if "policy_soft" in outputs else base_policy_logits
    soft_ce = policy_cross_entropy(_apply_legal_mask(soft_logits), batch.get("policy_soft_t", batch["policy_t"]))

    # future policy (t+2)
    has_future = batch.get("has_future")
    if has_future is None:
        has_future = torch.zeros((batch["x"].shape[0],), device=batch["x"].device)
    future_logits = outputs["policy_future"] if "policy_future" in outputs else base_policy_logits
    # Do NOT apply legal_mask here: future_policy_t uses move indices from position t+2
    # (two plies later: network move + SF reply), so t's legal mask would incorrectly
    # treat most future moves as illegal, inflating loss to ~100M.
    future_ce = policy_cross_entropy(future_logits, batch.get("future_policy_t", batch["policy_t"]))

    # value
    wdl_ce = wdl_cross_entropy(outputs["wdl"], batch["wdl_t"])  # (B,)

    # SF move prediction head.
    # With the "train on network turns only" scheme, SF targets live on network-turn samples
    # and represent SF's reply distribution after the network plays its move.
    has_sf_move = batch.get("has_sf_move")
    if has_sf_move is None:
        has_sf_move = torch.zeros((batch["x"].shape[0],), device=batch["x"].device)
    has_sf_policy = batch.get("has_sf_policy")
    if has_sf_policy is None:
        has_sf_policy = has_sf_move

    sf_pol_logits = outputs.get("policy_sf")
    if sf_pol_logits is None:
        sf_move_ce = torch.zeros_like(wdl_ce)
    else:
        # Do NOT apply legal_mask here: sf_policy_t uses move indices from the next board
        # position (after the network moves), so the network-turn legal mask would incorrectly
        # mark most of SF's moves as "illegal", inflating loss to ~1e9.
        sf_move_ce = soft_cross_entropy(sf_pol_logits, batch.get("sf_policy_t", batch["policy_t"]))

    # sf_eval
    has_sf_wdl = batch.get("has_sf_wdl")
    if has_sf_wdl is None:
        has_sf_wdl = torch.zeros((batch["x"].shape[0],), device=batch["x"].device)
    # SF eval: soft-target CE vs Stockfish's WDL distribution (no argmax/hardening)
    sf_wdl_probs = batch["sf_wdl"].clamp_min(0.0)
    sf_wdl_probs = sf_wdl_probs / sf_wdl_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    sf_eval_logits = outputs.get("sf_eval")
    if sf_eval_logits is None:
        sf_eval_ce = torch.zeros_like(wdl_ce)
    else:
        sf_eval_ce = soft_cross_entropy(sf_eval_logits, sf_wdl_probs)

    # moves_left
    has_moves_left = batch.get("has_moves_left")
    if has_moves_left is None:
        has_moves_left = torch.zeros((batch["x"].shape[0],), device=batch["x"].device)
    ml_pred = outputs.get("moves_left")
    if ml_pred is None:
        ml_loss = torch.zeros_like(wdl_ce)
    else:
        ml_loss = F.smooth_l1_loss(ml_pred.squeeze(-1), batch["moves_left"], reduction="none")

    # categorical value
    has_cat = batch.get("has_categorical")
    if has_cat is None:
        has_cat = torch.zeros((batch["x"].shape[0],), device=batch["x"].device)
    cat_logits = outputs.get("categorical")
    if cat_logits is None:
        cat_ce = torch.zeros_like(wdl_ce)
    else:
        logp_cat = F.log_softmax(cat_logits, dim=-1)
        cat_ce = -(batch["categorical_t"] * logp_cat).sum(dim=-1)

    # volatility (network)
    has_vol = batch.get("has_volatility")
    if has_vol is None:
        has_vol = torch.zeros((batch["x"].shape[0],), device=batch["x"].device)
    vol_pred = outputs.get("volatility")
    if vol_pred is None:
        vol_loss = torch.zeros_like(wdl_ce)
    else:
        vol_loss = F.huber_loss(vol_pred, batch["volatility_t"], delta=0.1, reduction="none").mean(dim=-1)

    # volatility (Stockfish)
    has_sf_vol = batch.get("has_sf_volatility")
    if has_sf_vol is None:
        has_sf_vol = torch.zeros((batch["x"].shape[0],), device=batch["x"].device)
    sf_vol_pred = outputs.get("sf_volatility")
    if sf_vol_pred is None:
        sf_vol_loss = torch.zeros_like(wdl_ce)
    else:
        sf_vol_loss = F.huber_loss(sf_vol_pred, batch.get("sf_volatility_t", batch["volatility_t"]), delta=0.1, reduction="none").mean(dim=-1)

    # Loss weights (all configurable for Ray Tune ablations)
    w_policy = float(w_policy)
    w_soft = float(w_soft)
    w_future = float(w_future)
    w_wdl = float(w_wdl)
    w_sf_move = float(w_sf_move)
    w_sf_eval = float(w_sf_eval)
    w_categorical = float(w_categorical)
    w_volatility = float(w_volatility)
    # sf_volatility defaults to same as volatility if not explicitly specified
    w_sf_volatility = float(w_sf_volatility) if w_sf_volatility is not None else float(w_volatility)
    w_moves_left = float(w_moves_left)
    w_sf_wdl = float(w_sf_wdl)
    sf_wdl_conf_power = max(0.0, float(sf_wdl_conf_power))
    sf_wdl_draw_scale = max(0.0, float(sf_wdl_draw_scale))

    # Optional confidence damping for SF-WDL auxiliary loss on the main WDL head.
    # - confidence: (1 - draw_prob)^power (power=0 disables)
    # - draw_scale: additional multiplier for game-outcome draws (1.0 disables)
    sf_wdl_mask = net_mask * has_sf_wdl
    if sf_wdl_conf_power > 0.0:
        sf_conf = (1.0 - sf_wdl_probs[:, 1]).clamp(0.0, 1.0).pow(sf_wdl_conf_power)
        sf_wdl_mask = sf_wdl_mask * sf_conf
    if sf_wdl_draw_scale != 1.0:
        draw_mask = (batch["wdl_t"] == 1).to(torch.float32)
        sf_wdl_mask = sf_wdl_mask * (1.0 - draw_mask + draw_mask * sf_wdl_draw_scale)

    # Soft WDL target from SF eval: train the main WDL head with SF's position-local
    # value estimate as a soft cross-entropy target. This provides dense gradient signal
    # even when game outcomes are draws (e.g. max_plies reached).
    sf_wdl_soft_ce = soft_cross_entropy(outputs["wdl"], sf_wdl_probs)

    # Train only on network turns for the main value/policy targets.
    # SF heads are also trained on network turns, but only when their targets are present.
    total = (
        w_policy * masked_mean(pol_ce, net_mask * has_policy)
        + w_soft * masked_mean(soft_ce, net_mask * has_soft)
        + w_future * masked_mean(future_ce, net_mask * has_future)
        + w_wdl * masked_mean(wdl_ce, net_mask)
        + w_sf_wdl * masked_mean(sf_wdl_soft_ce, sf_wdl_mask)
        + w_sf_move * masked_mean(sf_move_ce, net_mask * has_sf_policy)
        + w_sf_eval * masked_mean(sf_eval_ce, net_mask * has_sf_wdl)
        + w_categorical * masked_mean(cat_ce, net_mask * has_cat)
        + w_volatility * masked_mean(vol_loss, net_mask * has_vol)
        + w_sf_volatility * masked_mean(sf_vol_loss, net_mask * has_sf_vol)
        + w_moves_left * masked_mean(ml_loss, net_mask * has_moves_left)
    )

    return {
        "total": total,
        "policy_ce": masked_mean(pol_ce, net_mask * has_policy),
        "wdl_ce": masked_mean(wdl_ce, net_mask),
        "sf_wdl_ce": masked_mean(sf_wdl_soft_ce, sf_wdl_mask),
        "soft_policy_ce": masked_mean(soft_ce, net_mask * has_soft),
        "future_policy_ce": masked_mean(future_ce, net_mask * has_future),
        "sf_move_ce": masked_mean(sf_move_ce, net_mask * has_sf_policy),
        "sf_eval_ce": masked_mean(sf_eval_ce, net_mask * has_sf_wdl),
        "categorical_ce": masked_mean(cat_ce, has_cat),
        "volatility": masked_mean(vol_loss, net_mask * has_vol),
        "sf_volatility": masked_mean(sf_vol_loss, net_mask * has_sf_vol),
        "moves_left": masked_mean(ml_loss, has_moves_left),
    }
