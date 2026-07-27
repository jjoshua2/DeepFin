from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from chess_anti_engine.train.sparse_sf_ce import sparse_sf_policy_ce

if TYPE_CHECKING:
    from chess_anti_engine.train.target_builder import SfTargetParams

from chess_anti_engine.moves import COMPACT_POLICY_SIZE, POLICY_SIZE
from chess_anti_engine.moves.torch_maps import compact_to_full_index_for as _compact_to_full_index_for
from chess_anti_engine.train.constants import REGRET_TO_Q_SCALE, future_regret_field_names

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


def normalize_distribution(probs: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Renormalize a distribution along the last axis so each row sums to 1.

    ``clamp_min(eps)`` keeps all-zero rows finite (they stay ~0 rather than
    producing NaN), so callers can pass missing/masked rows through safely.
    """
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with a soft target distribution.

    Soft targets are *meant* to be normalized along the last axis, but the
    replay shards persist them as float16 (``policy_t``, ``policy_soft_t``,
    ``future_policy_t``, ``categorical_t``, ``sf_policy_t``). Round-tripping a
    distribution through f16 leaves each row summing to ``1 ± O(width * 2**-11)``
    instead of exactly 1 (smaller on peaked visit distributions, up to ~1e-3 on
    flatter targets like ``categorical_t`` / smoothed ``sf_policy_t``). Because
    soft CE is *linear* in the target, that residual row-sum scales the whole
    sample's loss and gradient, most impactfully on the main policy head
    (``w_policy = 1.0``). Renormalizing here removes that f16 bias and makes the
    documented invariant true for every soft-CE head.

    It is a no-op (within fp error) for already-normalized targets such as the
    post-clamp ``sf_wdl_probs`` and the convex WDL blend, and all-zero rows
    (missing/masked targets) stay zero — ``clamp_min`` keeps the division
    finite so they contribute 0 loss rather than NaN.
    """
    target_probs = normalize_distribution(target_probs)
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def align_policy_target(target: torch.Tensor, width: int) -> torch.Tensor:
    """Return a policy target in the same action encoding width as logits.

    Replay shards are allowed to remain in the full 4672 action space while a
    model emits compact LC0-1858 logits. Compact projection gathers the valid
    full actions and renormalizes because invalid 4672 padding can carry tiny
    mass from smoothing or legacy producers.
    """
    src_width = int(target.shape[-1])
    dst_width = int(width)
    if src_width == dst_width:
        return target
    if src_width == POLICY_SIZE and dst_width == COMPACT_POLICY_SIZE:
        out = target.index_select(-1, _compact_to_full_index_for(target))
        return normalize_distribution(out)
    if src_width == COMPACT_POLICY_SIZE and dst_width == POLICY_SIZE:
        out = target.new_zeros((*target.shape[:-1], POLICY_SIZE))
        out.index_copy_(-1, _compact_to_full_index_for(target), target)
        return out
    raise ValueError(f"policy target width {src_width} is incompatible with logits width {dst_width}")


def align_policy_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    """Return a legal-move mask in the same action encoding width as logits."""
    src_width = int(mask.shape[-1])
    dst_width = int(width)
    if src_width == dst_width:
        return mask
    if src_width == POLICY_SIZE and dst_width == COMPACT_POLICY_SIZE:
        return mask.index_select(-1, _compact_to_full_index_for(mask))
    if src_width == COMPACT_POLICY_SIZE and dst_width == POLICY_SIZE:
        out = mask.new_zeros((*mask.shape[:-1], POLICY_SIZE))
        out.index_copy_(-1, _compact_to_full_index_for(mask), mask)
        return out
    raise ValueError(f"policy mask width {src_width} is incompatible with logits width {dst_width}")


def align_action_values(values: torch.Tensor, width: int) -> torch.Tensor:
    """Reindex a per-action VALUE vector (e.g. cp-regret) to the logits' width.

    Unlike :func:`align_policy_target`, this does NOT renormalize: the vector is
    not a probability distribution, so a 4672->1858 compact projection must
    gather the valid actions and leave their magnitudes untouched. The reindex
    is identical to a legal-mask reindex, so we delegate to that path.
    """
    return align_policy_mask(values, width)


def apply_policy_mask_to_logits(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    mask_key: str,
    has_key: str,
) -> torch.Tensor:
    """Policy-specific mask application that supports full/compact encodings."""
    mask = batch.get(mask_key)
    if mask is None:
        return logits
    mask = align_policy_mask(mask, int(logits.shape[-1]))
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


def _compute_sf_wdl_mask(
    *,
    net_mask: torch.Tensor,
    has_sf_wdl: torch.Tensor,
    sf_wdl_probs: torch.Tensor | None,
    wdl_target: torch.Tensor,
    conf_power: float,
    draw_scale: float,
) -> torch.Tensor:
    """SF-WDL per-sample mask with optional confidence damping + draw rescale.

    Damping: ``(1 - draw_prob)^power`` zeros out high-draw rows where SF's
    label barely disagrees with a fresh-init model. Draw rescale boosts/cuts
    the contribution of game-decided-as-draw outcomes.
    """
    mask = net_mask * has_sf_wdl
    if sf_wdl_probs is None:
        return mask
    if conf_power > 0.0:
        sf_conf = (1.0 - sf_wdl_probs[:, 1]).clamp(0.0, 1.0).pow(conf_power)
        mask = mask * sf_conf
    if draw_scale != 1.0:
        draw_mask = (wdl_target == 1).to(torch.float32)
        mask = mask * (1.0 - draw_mask + draw_mask * draw_scale)
    return mask


def _normalize_sf_wdl_probs(
    sf_wdl_raw: torch.Tensor | None, *, temperature: float = 1.0
) -> torch.Tensor | None:
    """Clamp negatives to 0, optionally soften via ``p^(1/T)``, renormalize."""
    if sf_wdl_raw is None:
        return None
    p = sf_wdl_raw.clamp_min(0.0)
    if temperature != 1.0 and temperature > 0.0:
        p = p.clamp_min(1e-6).pow(1.0 / float(temperature))
    return normalize_distribution(p)


def _q_to_wdl_probs(q: torch.Tensor) -> torch.Tensor:
    q_clamped = q.clamp(-1.0, 1.0)
    win = q_clamped.clamp_min(0.0)
    loss = (-q_clamped).clamp_min(0.0)
    draw = (1.0 - win - loss).clamp_min(0.0)
    return torch.stack((win, draw, loss), dim=1)


def _future_regret_tensor(batch: dict[str, torch.Tensor], source: str) -> tuple[torch.Tensor, torch.Tensor]:
    key, has_key = future_regret_field_names(source)
    return (
        _get_mask(batch, key).to(torch.float32),
        _get_mask(batch, has_key).to(torch.float32),
    )


def _phase_split_masks(
    *,
    has_is_selfplay: torch.Tensor,
    is_selfplay: torch.Tensor,
    has_moves_left: torch.Tensor,
    moves_left_val: torch.Tensor,
) -> tuple[tuple[str, torch.Tensor], ...]:
    """selfplay/curriculum + opening/midgame/endgame masks for split loss reporting."""
    sp_mask = has_is_selfplay * is_selfplay
    cur_mask = has_is_selfplay - sp_mask
    open_mask = has_moves_left * (moves_left_val > _PHASE_OPEN_THRESHOLD).to(torch.float32)
    end_mask = has_moves_left * (moves_left_val < _PHASE_END_THRESHOLD).to(torch.float32)
    mid_mask = has_moves_left - open_mask - end_mask
    return (
        ("selfplay", sp_mask),
        ("curriculum", cur_mask),
        ("open", open_mask),
        ("mid", mid_mask),
        ("end", end_mask),
    )


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    w_policy: float = 1.0,
    w_soft: float = 0.5,
    w_future: float = 0.15,
    w_sf_own: float = 0.0,
    w_sf_own_regret: float = 0.0,
    w_wdl: float = 1.0,
    w_sf_move: float = 0.15,
    w_sf_eval: float = 0.15,
    w_categorical: float = 0.10,
    w_volatility: float = 0.05,
    w_sf_volatility: float | None = None,
    w_moves_left: float = 0.02,
    sf_wdl_frac: float = 0.0,
    search_wdl_frac: float = 0.0,
    sf_wdl_conf_power: float = 0.0,
    sf_wdl_draw_scale: float = 1.0,
    sf_wdl_temperature: float = 1.0,
    sf_search_dampen_sf_low: float = 0.0,
    sf_search_dampen_sf_high: float = 0.0,
    use_adjusted_wdl_target: bool = False,
    adjusted_wdl_regret_source: str = "sum",
    adjusted_wdl_regret_scale: float = 1.0,
    adjusted_wdl_regret_cap: float = 0.0,
    soft_policy_min_tv: float = 0.0,
    sf_sparse_params: SfTargetParams | None = None,
) -> dict[str, torch.Tensor]:
    """Compute multi-head training loss.

    ``soft_policy_min_tv`` masks the soft-policy loss to zero on samples
    whose soft target is within that total-variation distance of the hard
    target (they're a deterministic retempering of the same distribution —
    see scripts/probe_policy_targets.py). 0.0 keeps current behavior exactly.

    ``sf_sparse_params`` switches the ``policy_sf`` loss to sparse CE over
    gathered log-probs (train/sparse_sf_ce.py) for rows carrying sparse
    MultiPV labels; rows without them keep the dense soft CE. None (the
    default) leaves the dense path untouched.

    Value-loss reporting: ``wdl_ce`` and ``blended_wdl_ce`` are the same
    tensor — the blended soft CE that ``total`` (and therefore every
    gradient) is built from. The hard one-hot CE against the recorded game
    result is a diagnostic and is reported separately as ``wdl_onehot_ce``.
    """
    net_mask = _get_mask(batch, "is_network_turn", default=1.0).to(torch.float32)

    def _apply_legal_mask(logits: torch.Tensor) -> torch.Tensor:
        return apply_policy_mask_to_logits(logits, batch, "legal_mask", "has_legal_mask")

    base_policy_logits = outputs["policy"] if "policy" in outputs else outputs.get("policy_own")
    if base_policy_logits is None:
        raise KeyError("Model outputs must include either 'policy' or 'policy_own'.")

    # Legal-masked base logits, computed once and reused by the main policy CE,
    # the sf_p0 CE, and the sf_own_regret softmax below (all train policy_own /
    # policy in the same legal space) — the mask is a full-width align+fill, so
    # recomputing it per term wastes work every training step.
    masked_base = _apply_legal_mask(base_policy_logits)

    pol_target = align_policy_target(batch["policy_t"], int(base_policy_logits.shape[-1]))
    pol_ce = soft_cross_entropy(masked_base, pol_target)
    zero_loss = torch.zeros_like(pol_ce)
    has_policy = _get_mask(batch, "has_policy", default=1.0)

    has_soft = _get_mask(batch, "has_policy_soft")
    soft_logits = outputs.get("policy_soft", base_policy_logits)
    soft_target = batch.get("policy_soft_t")
    soft_ce = (
        soft_cross_entropy(
            _apply_legal_mask(soft_logits),
            align_policy_target(soft_target, int(soft_logits.shape[-1])),
        )
        if soft_target is not None else zero_loss
    )
    soft_mask_kept_frac = torch.ones((), device=pol_ce.device, dtype=torch.float32)
    if float(soft_policy_min_tv) > 0.0 and soft_target is not None:
        # Drop the soft loss where TV(hard, soft) is below the threshold: there
        # the soft target is a retempering of (nearly) the same distribution, so
        # for the trunk the gradient largely duplicates the main policy CE (the
        # policy_soft head itself still loses its only signal on masked rows).
        # Strictly off at the default 0.0.
        _soft_aligned = align_policy_target(soft_target, int(base_policy_logits.shape[-1]))
        _tv = 0.5 * (pol_target.float() - _soft_aligned.float()).abs().sum(dim=-1)
        _keep = (_tv >= float(soft_policy_min_tv)).to(has_soft.dtype)
        soft_mask_kept_frac = masked_mean(_keep, net_mask * has_soft)
        has_soft = has_soft * _keep

  # Future policy (t+2): target and legal mask are in the t+2 move space.
    has_future = _get_mask(batch, "has_future")
    future_logits = outputs.get("policy_future", base_policy_logits)
    future_target = batch.get("future_policy_t")
    if future_target is not None:
        future_ce = soft_cross_entropy(
            apply_policy_mask_to_logits(future_logits, batch, "future_legal_mask", "has_future_legal_mask"),
            align_policy_target(future_target, int(future_logits.shape[-1])),
        )
    else:
        future_ce = zero_loss

    # P0 own-move SF teacher on policy_own (the head MCTS reads as the search
    # prior). Unlike policy_sf (a separate, search-invisible head trained on the
    # opponent reply), this blends SF's recommended move for THIS position into
    # the prior itself. Same legal space as the main policy. Masked to the ~15%
    # eligible selfplay rows (has_sf_p0).
    has_sf_p0 = _get_mask(batch, "has_sf_p0")
    sf_p0_target = batch.get("sf_p0_policy_t")
    if sf_p0_target is not None:
        sf_p0_ce = soft_cross_entropy(
            masked_base,
            align_policy_target(sf_p0_target, int(base_policy_logits.shape[-1])),
        )
    else:
        sf_p0_ce = zero_loss

    # Regret-weighted SF teacher on policy_own: minimize the net's EXPECTED SF
    # cp-regret = sum_m p_own(m) * regret(m), where regret(m) in [0,1] is the
    # normalized cp loss vs SF's best move at THIS (P0) position. Pushes mass
    # toward low-regret moves; auto-weighted by how much each position matters
    # (flat positions have tiny regrets -> tiny gradient). Same legal-masked
    # policy_own head as sf_p0_ce, masked to eligible selfplay rows.
    has_sf_p0_regret = _get_mask(batch, "has_sf_p0_regret")
    sf_p0_regret_t = batch.get("sf_p0_regret_t")
    if sf_p0_regret_t is not None:
        po_probs = torch.softmax(masked_base, dim=-1)
        reg_vec = align_action_values(sf_p0_regret_t, int(base_policy_logits.shape[-1]))
        sf_own_regret = (po_probs * reg_vec).sum(-1)
    else:
        sf_own_regret = zero_loss

  # DIAGNOSTIC ONLY — hard one-hot CE against the recorded game result. The
  # optimizer never sees this term (see ``blended_wdl_ce`` below, which is the
  # value loss in ``total``). Reported as ``wdl_onehot_ce`` so nothing can
  # mistake it for the trained loss again; see docs/rl_loop_audit.md I7.
    wdl_onehot_ce = F.cross_entropy(outputs["wdl"], batch["wdl_t"], reduction="none")

  # SF move prediction: target and legal mask are in the t+1 move space (opp POV).
    has_sf_move = _get_mask(batch, "has_sf_move")
    has_sf_policy = _get_mask(batch, "has_sf_policy") if "has_sf_policy" in batch else has_sf_move

    sf_pol_logits = outputs.get("policy_sf")
    sf_policy_target = batch.get("sf_policy_t")
    if sf_pol_logits is None:
        sf_move_ce = zero_loss
    else:
        masked_sf_logits = apply_policy_mask_to_logits(
            sf_pol_logits, batch, "sf_legal_mask", "has_sf_legal_mask",
        )
        sf_move_ce = (
            soft_cross_entropy(
                masked_sf_logits,
                align_policy_target(sf_policy_target, int(sf_pol_logits.shape[-1])),
            )
            if sf_policy_target is not None else zero_loss
        )
        sf_legal = batch.get("sf_legal_mask")
        if sf_sparse_params is not None and "sf_multipv_raw" in batch and sf_legal is not None:
            sparse_ce, sparse_ok = sparse_sf_policy_ce(
                masked_sf_logits, batch, params=sf_sparse_params,
                legal_aligned=align_policy_mask(sf_legal, int(sf_pol_logits.shape[-1])),
            )
            keep_sparse = sparse_ok > 0
            sf_move_ce = torch.where(keep_sparse, sparse_ce, sf_move_ce)
            # Rows whose shards no longer carry a dense target (the
            # record_dense_sf_policy=false transition) still train via the
            # sparse path; widen the head mask to include them.
            has_sf_policy = torch.maximum(
                has_sf_policy.to(torch.float32), sparse_ok.to(torch.float32),
            )

    has_sf_wdl = _get_mask(batch, "has_sf_wdl")
    sf_wdl_probs = _normalize_sf_wdl_probs(batch.get("sf_wdl"), temperature=sf_wdl_temperature)
    sf_eval_logits = outputs.get("sf_eval")
    sf_eval_ce = (
        soft_cross_entropy(sf_eval_logits, sf_wdl_probs)
        if sf_eval_logits is not None and sf_wdl_probs is not None else zero_loss
    )

    has_search_wdl = _get_mask(batch, "has_search_wdl")
    search_wdl_probs = _normalize_sf_wdl_probs(batch.get("search_wdl"))

    wdl_t = batch["wdl_t"].to(torch.int64)
    game_oh = F.one_hot(wdl_t, 3).float()
    if bool(use_adjusted_wdl_target):
        future_regret, has_future_regret = _future_regret_tensor(batch, adjusted_wdl_regret_source)
        # Individual regrets are stored in winrate units [0, 1], while Q spans [-1, 1].
        # The 2x scale maps a single regret onto Q units; cumulative sources still need
        # explicit source/scale/cap choices. The one-sided shift intentionally discounts
        # W->D and D->L; a drawn game with future SF mistakes is treated as
        # under-converted anti-engine value.
        # Clamp the scalar config and per-sample tensor separately; they guard different inputs.
        correction = REGRET_TO_Q_SCALE * max(0.0, float(adjusted_wdl_regret_scale)) * future_regret.clamp_min(0.0)
        if float(adjusted_wdl_regret_cap) > 0.0:
            correction = correction.clamp_max(float(adjusted_wdl_regret_cap))
        game_q = game_oh[:, 0] - game_oh[:, 2]
        adjusted_wdl_probs = _q_to_wdl_probs(game_q - correction)
        has_adjusted_wdl = has_future_regret.unsqueeze(1)
        game_target = has_adjusted_wdl * adjusted_wdl_probs + (1.0 - has_adjusted_wdl) * game_oh
    else:
        game_target = game_oh
    # Regret adjustment is a heuristic on the game-outcome component only.
    # Missing/dampened SF/search WDL labels fall back to the raw result so the
    # adjusted target does not silently expand beyond ``game_frac``.
    blend_fallback_target = game_oh
    sf_wdl_frac_f = max(0.0, float(sf_wdl_frac))
    search_wdl_frac_f = max(0.0, float(search_wdl_frac))
    blend_sum = sf_wdl_frac_f + search_wdl_frac_f
    if blend_sum > 1.0:
        sf_wdl_frac_f /= blend_sum
        search_wdl_frac_f /= blend_sum
        game_frac = 0.0
    else:
        game_frac = 1.0 - blend_sum
    target = game_frac * game_target
    sf_available = has_sf_wdl.float()
    search_available = has_search_wdl.float()

  # Per-sample SF/search disagreement is split by direction so each side of
  # the lever can be dampened independently:
  #   sf_low : SF says STM losing but search says STM winning. Real-data
  #            calibration (handicapped-opponent regime) shows SF is often
  #            wrong about *outcomes* here (the handicapped opponent fails
  #            to convert), but SF is still right about *objective* eval —
  #            partial dampening keeps that signal alive.
  #   sf_high: SF says STM winning but search says STM losing. Less
  #            common; SF over-confident, search hedging. Usually best to
  #            dampen lightly or not at all (SF is the search-horizon
  #            reference).
    if sf_wdl_probs is not None and search_wdl_probs is not None:
        sf_sig = sf_wdl_probs[:, 0] - sf_wdl_probs[:, 2]
        sr_sig = search_wdl_probs[:, 0] - search_wdl_probs[:, 2]
        joint = sf_available * search_available
        agree = ((sf_sig * sr_sig) > 0).float() * joint
        dis_sf_low = ((sf_sig < 0) & (sr_sig > 0)).float() * joint
        dis_sf_high = ((sf_sig > 0) & (sr_sig < 0)).float() * joint
        joint_count = joint.sum().clamp_min(1.0)
        sf_search_agree_frac = (agree.sum() / joint_count).detach()
        sf_search_disagree_sf_low_frac = (dis_sf_low.sum() / joint_count).detach()
        sf_search_disagree_sf_high_frac = (dis_sf_high.sum() / joint_count).detach()
    else:
        dis_sf_low = torch.zeros_like(sf_available)
        dis_sf_high = torch.zeros_like(sf_available)
        zero_scalar = torch.zeros((), device=sf_available.device)
        sf_search_agree_frac = zero_scalar
        sf_search_disagree_sf_low_frac = zero_scalar
        sf_search_disagree_sf_high_frac = zero_scalar

    keep = 1.0 - (
        float(sf_search_dampen_sf_low) * dis_sf_low
        + float(sf_search_dampen_sf_high) * dis_sf_high
    )
    sf_effective = sf_available * keep
    sf_effective_b = sf_effective.unsqueeze(1)
    if sf_wdl_probs is not None:
        target += sf_wdl_frac_f * (
            sf_effective_b * sf_wdl_probs + (1.0 - sf_effective_b) * blend_fallback_target
        )
    else:
        target += sf_wdl_frac_f * blend_fallback_target
    search_available_b = search_available.unsqueeze(1)
    if search_wdl_probs is not None:
        target += search_wdl_frac_f * (
            search_available_b * search_wdl_probs + (1.0 - search_available_b) * blend_fallback_target
        )
    else:
        target += search_wdl_frac_f * blend_fallback_target
    blended_wdl_ce = soft_cross_entropy(outputs["wdl"], target.detach())

    has_moves_left = _get_mask(batch, "has_moves_left")
    ml_pred = outputs.get("moves_left")
    moves_left_t = batch.get("moves_left")
    ml_loss = (
        F.smooth_l1_loss(ml_pred.squeeze(-1), moves_left_t, reduction="none")
        if ml_pred is not None and moves_left_t is not None else zero_loss
    )

    has_cat = _get_mask(batch, "has_categorical")
    cat_logits = outputs.get("categorical")
    categorical_t = batch.get("categorical_t")
    cat_ce = (
        soft_cross_entropy(cat_logits, categorical_t)
        if cat_logits is not None and categorical_t is not None else zero_loss
    )

    has_vol = _get_mask(batch, "has_volatility")
    vol_pred = outputs.get("volatility")
    volatility_t = batch.get("volatility_t")
    vol_loss = (
        _huber_per_sample(vol_pred, volatility_t)
        if vol_pred is not None and volatility_t is not None else zero_loss
    )

    has_sf_vol = _get_mask(batch, "has_sf_volatility")
    sf_vol_pred = outputs.get("sf_volatility")
    sf_volatility_t = batch.get("sf_volatility_t")
    sf_vol_loss = (
        _huber_per_sample(sf_vol_pred, sf_volatility_t)
        if sf_vol_pred is not None and sf_volatility_t is not None else zero_loss
    )

  # Loss weights — float() casts defend against numpy scalars from Ray Tune config mutation
    w_sf_volatility = float(w_sf_volatility) if w_sf_volatility is not None else float(w_volatility)
    m_sf_wdl_mask = _compute_sf_wdl_mask(
        net_mask=net_mask, has_sf_wdl=has_sf_wdl, sf_wdl_probs=sf_wdl_probs,
        wdl_target=batch["wdl_t"],
        conf_power=max(0.0, float(sf_wdl_conf_power)),
        draw_scale=max(0.0, float(sf_wdl_draw_scale)),
    )

  # Precompute the per-sample base mask for each head so the downstream
  # split reductions don't recompute `net_mask * has_X` once per bucket.
    pol_base = net_mask * has_policy
    m_policy = masked_mean(pol_ce, pol_base)
    m_soft = masked_mean(soft_ce, net_mask * has_soft)
    m_future = masked_mean(future_ce, net_mask * has_future)
    m_sf_own = masked_mean(sf_p0_ce, net_mask * has_sf_p0)
    m_sf_own_regret = masked_mean(sf_own_regret, net_mask * has_sf_p0_regret)
    m_wdl_onehot = masked_mean(wdl_onehot_ce, net_mask)
    m_blended_wdl = masked_mean(blended_wdl_ce, net_mask)
    m_sf_move = masked_mean(sf_move_ce, net_mask * has_sf_policy)
    m_sf_eval = masked_mean(sf_eval_ce, m_sf_wdl_mask)
    m_cat = masked_mean(cat_ce, net_mask * has_cat)
    m_vol = masked_mean(vol_loss, net_mask * has_vol)
    m_sf_vol = masked_mean(sf_vol_loss, net_mask * has_sf_vol)
    m_ml = masked_mean(ml_loss, net_mask * has_moves_left)

  # Gated on `has_is_selfplay` so legacy shards without the tag are excluded
  # from the split (they won't contribute to either selfplay_ or curriculum_ keys).
    has_is_sp = _get_mask(batch, "has_is_selfplay").to(torch.float32)
    is_sp_bool = _get_mask(batch, "is_selfplay", default=0.0).to(torch.float32)
    split_masks = _phase_split_masks(
        has_is_selfplay=has_is_sp, is_selfplay=is_sp_bool,
        has_moves_left=has_moves_left,
        moves_left_val=_get_mask(batch, "moves_left", default=1.0).to(torch.float32),
    )
  # Split reductions use the TRAINED per-sample value loss (blended soft CE),
  # not the one-hot diagnostic — before 2026-07-26 these were the diagnostic,
  # which made `wdl_loss_selfplay` / `_open` / ... track a term no gradient
  # ever came from.
    split_losses: dict[str, torch.Tensor] = {}
    for suffix, m in split_masks:
        split_losses[f"policy_loss_{suffix}"] = masked_mean(pol_ce, pol_base * m)
        split_losses[f"wdl_loss_{suffix}"] = masked_mean(blended_wdl_ce, net_mask * m)

    total = (
        float(w_policy) * m_policy
        + float(w_soft) * m_soft
        + float(w_future) * m_future
        + float(w_sf_own) * m_sf_own
        + float(w_sf_own_regret) * m_sf_own_regret
        + float(w_wdl) * m_blended_wdl
        + float(w_sf_move) * m_sf_move
        + float(w_sf_eval) * m_sf_eval
        + float(w_categorical) * m_cat
        + float(w_volatility) * m_vol
        + float(w_sf_volatility) * m_sf_vol
        + float(w_moves_left) * m_ml
    )

  # Reported value-loss names (docs/rl_loop_audit.md I7):
  #   wdl_ce / blended_wdl_ce -> the SAME tensor, the loss the optimizer sees.
  #     `wdl_ce` is the name people reach for (it becomes the `wdl_loss`
  #     column), `blended_wdl_ce` is kept because existing readers use it.
  #   wdl_onehot_ce -> the hard one-hot diagnostic, never in `total`.
    return {
        "total": total,
        "policy_ce": m_policy,
        "wdl_ce": m_blended_wdl,
        "blended_wdl_ce": m_blended_wdl,
        "wdl_onehot_ce": m_wdl_onehot,
        "soft_policy_ce": m_soft,
        "soft_mask_kept_frac": soft_mask_kept_frac,
        "future_policy_ce": m_future,
        "sf_own_ce": m_sf_own,
        "sf_own_regret": m_sf_own_regret,
        "sf_move_ce": m_sf_move,
        "sf_eval_ce": m_sf_eval,
        "categorical_ce": m_cat,
        "volatility": m_vol,
        "sf_volatility": m_sf_vol,
        "moves_left": m_ml,
        **split_losses,
        "frac_is_selfplay": masked_mean(is_sp_bool, has_is_sp),
        "frac_tagged": masked_mean(has_is_sp, net_mask),
        "sf_search_agree_frac": sf_search_agree_frac,
        "sf_search_disagree_sf_low_frac": sf_search_disagree_sf_low_frac,
        "sf_search_disagree_sf_high_frac": sf_search_disagree_sf_high_frac,
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
