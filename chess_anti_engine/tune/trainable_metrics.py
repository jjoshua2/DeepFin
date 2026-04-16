"""Pure numeric/metric helpers for the Ray Tune trainable.

All functions here are pure: no filesystem mutation, no Ray, no torch
model mutation. Extracted from trainable.py to keep the orchestration
module focused on lifecycle.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from chess_anti_engine.replay import ArrayReplayBuffer, DiskReplayBuffer
from chess_anti_engine.tune.trial_config import DriftMetrics, TrialConfig


def _count_jsonl_rows(path: Path) -> int:
    """Count non-empty rows in a JSONL file (best effort)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return 0


def _iteration_pause_metrics(
    *,
    iteration_started_at: float,
    iteration_finished_at: float,
    pause_started_at: float | None,
    pause_active: bool,
) -> dict[str, float]:
    """Compute how much of an iteration was spent with selfplay paused."""
    elapsed_s = max(0.0, float(iteration_finished_at) - float(iteration_started_at))
    paused_s = 0.0
    if pause_active and pause_started_at is not None:
        paused_s = max(0.0, float(iteration_finished_at) - max(float(iteration_started_at), float(pause_started_at)))
    paused_fraction = 0.0
    if elapsed_s > 0.0:
        paused_fraction = min(1.0, max(0.0, paused_s / elapsed_s))
    return {
        "iteration_elapsed_s": float(elapsed_s),
        "paused_seconds": float(paused_s),
        "paused_fraction": float(paused_fraction),
        "paused_percent": float(paused_fraction * 100.0),
    }


def _compute_train_step_budget(
    *,
    positions_added: int,
    imported_samples: int,
    replay_size: int,
    batch_size: int,
    accum_steps: int,
    base_max_steps: int,
    train_window_fraction: float,
) -> dict[str, int]:
    effective_batch_size = max(1, int(batch_size) * max(1, int(accum_steps)))
    window_target_samples = int(math.ceil(float(train_window_fraction) * max(0, int(replay_size))))
    target_sample_budget = max(
        int(positions_added) + int(imported_samples),
        int(window_target_samples),
    )
    target_steps = max(1, int(math.ceil(float(target_sample_budget) / float(effective_batch_size))))
    if int(imported_samples) > 0:
        steps = int(target_steps)
    else:
        steps = min(int(target_steps), max(1, int(base_max_steps)))
    return {
        "steps": int(steps),
        "target_sample_budget": int(target_sample_budget),
        "window_target_samples": int(window_target_samples),
    }


# Default weights for regret-only mode (rmp/topk pinned).
# When regret is disabled, rmp/topk weights are restored automatically.
_W_RMP = 0.0
_W_TOPK = 0.0
_W_REGRET = 350.0
_W_NODES = 100.0
_W_SKILL = 50.0


def _opponent_strength(
    *,
    random_move_prob: float,
    sf_nodes: int,
    skill_level: int,
    ema_winrate: float,
    min_nodes: int,
    max_nodes: int,
    pid_target_winrate: float = 0.60,
    wdl_regret: float = -1.0,
    wdl_regret_max: float = 1.0,
    topk: int = -1,
    topk_max: int = 12,
    topk_min: int = 2,
) -> float:
    """Composite metric: weighted sum of normalised difficulty factors.

    Each factor maps to 0.0 (easiest) → 1.0 (hardest), then scaled by its
    weight.  All factors contribute independently — no sequential stages.

    Winrate scaling with floor: penalises below-target winrate by at most 50%
    to avoid death spirals after exploit, while still ranking trials that are
    losing worse than those that are winning at equal difficulty.

    Factors & weights (total max ≈ 500):
      wdl_regret        ×350   — primary difficulty lever (regret-only mode)
      sf_nodes          ×100   — search depth (log-scaled, secondary)
      skill_level        ×50   — usually pinned at 20
      random_move_prob    ×0   — pinned at 0.01, not a lever
      topk                ×0   — pinned at multipv, not a lever

    Final score multiplied by min(1, ema_winrate / target) to penalise
    PID overshoot without rewarding above-target winrate.
    """
    rand_prob = float(random_move_prob)
    nodes = int(sf_nodes)
    skill = int(skill_level)
    min_nodes = int(min_nodes)
    max_nodes = int(max_nodes)

    # rmp: 1.0→0.0 maps to 0→1
    rmp_score = 1.0 - max(0.0, min(1.0, rand_prob))

    # topk: max→min maps to 0→1
    k = int(topk)
    k_max = max(1, int(topk_max))
    k_min = max(1, int(topk_min))
    if k < 0 or k_max <= k_min:
        topk_score = 1.0  # not available — assume full difficulty
    else:
        k = max(k_min, min(k_max, k))
        topk_score = 1.0 - (k - k_min) / max(1, k_max - k_min)

    # regret: max→0 maps to 0→1 (negative = disabled = no contribution)
    regret_enabled = float(wdl_regret) >= 0.0
    if not regret_enabled:
        regret_score = 0.0
    else:
        r_max = max(0.001, float(wdl_regret_max))
        regret_score = 1.0 - max(0.0, min(1.0, float(wdl_regret) / r_max))

    # nodes: log-scaled, min→max maps to 0→1
    if min_nodes < max_nodes and nodes > 0:
        log_frac = (math.log(max(nodes, min_nodes)) - math.log(max(1, min_nodes))) / (
            math.log(max(1, max_nodes)) - math.log(max(1, min_nodes))
        )
        nodes_score = max(0.0, min(1.0, log_frac))
    else:
        nodes_score = 0.0

    # skill: 0→20 maps to 0→1
    skill_score = max(0.0, min(1.0, float(skill) / 20.0))

    # When regret is disabled, restore rmp/topk weights so the old
    # multi-stage curriculum is reflected in opponent_strength.
    w_rmp = 200.0 if not regret_enabled else _W_RMP
    w_topk = 150.0 if not regret_enabled else _W_TOPK
    w_regret = _W_REGRET if regret_enabled else 0.0

    difficulty = (
        w_rmp * rmp_score
        + w_topk * topk_score
        + w_regret * regret_score
        + _W_NODES * nodes_score
        + _W_SKILL * skill_score
    )

    # Winrate scaling with floor: cap penalty at 50% so a single bad batch
    # can't crater the metric (old death spiral), but still penalise trials
    # that are consistently losing at their difficulty level.
    target = max(0.01, float(pid_target_winrate))
    winrate_factor = max(0.5, min(1.0, max(0.0, float(ema_winrate)) / target))

    return difficulty * winrate_factor


def _should_retry_distributed_iteration_without_games(
    *,
    use_distributed_selfplay: bool,
    total_games_generated: int,
) -> bool:
    """Return True when a distributed iteration should wait for fresh selfplay."""
    return bool(use_distributed_selfplay) and int(total_games_generated) <= 0


def _blended_winrate_raw_or_none(
    *,
    wins: int,
    draws: int,
    losses: int,
) -> float | None:
    total_games_played = int(wins) + int(draws) + int(losses)
    if total_games_played <= 0:
        return None
    return (float(wins) + 0.5 * float(draws)) / float(total_games_played)


def _games_per_iter_for_iteration(tc: TrialConfig, iteration_idx: int) -> int:
    target = max(1, tc.games_per_iter)
    start = tc.games_per_iter_start
    ramp_iters = max(0, tc.games_per_iter_ramp_iters)

    if ramp_iters <= 0 or iteration_idx >= ramp_iters:
        return int(target)

    frac = float(max(0, iteration_idx - 1)) / float(ramp_iters)
    value = float(start) + (float(target) - float(start)) * frac
    return max(1, int(round(value)))


def _sample_drift_arrays(src_buf: object, n: int) -> dict[str, np.ndarray]:
    """Sample x, wdl_target, policy_target arrays for drift computation."""
    if hasattr(src_buf, "sample_batch_arrays"):
        arrs = getattr(src_buf, "sample_batch_arrays")(n, wdl_balance=False)
        return {
            "x": np.asarray(arrs["x"], dtype=np.float32),
            "wdl_target": np.asarray(arrs["wdl_target"], dtype=np.int64),
            "policy_target": np.asarray(arrs["policy_target"], dtype=np.float32),
        }
    samples = getattr(src_buf, "sample_batch")(n, wdl_balance=False)
    return {
        "x": np.stack([s.x for s in samples], axis=0).astype(np.float32, copy=False),
        "wdl_target": np.array([int(getattr(s, "wdl_target", 1)) for s in samples], dtype=np.int64),
        "policy_target": np.stack([s.policy_target for s in samples], axis=0).astype(np.float32, copy=False),
    }


def _mean_entropy(arrs: dict[str, np.ndarray], eps: float = 1e-12) -> float:
    """Mean per-sample policy entropy across the batch."""
    p = np.asarray(arrs["policy_target"], dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return 0.0
    ps = p.sum(axis=1, keepdims=True)
    valid = ps[:, 0] > 0.0
    if not np.any(valid):
        return 0.0
    p = p[valid] / ps[valid]
    ent = -np.sum(p * np.log(p + eps), axis=1)
    return float(np.mean(ent))


def _wdl_hist(arrs: dict[str, np.ndarray]) -> np.ndarray:
    """Normalised WDL histogram from sample arrays."""
    arr = np.asarray(arrs["wdl_target"], dtype=np.int64)
    valid = arr[(arr >= 0) & (arr <= 2)]
    hst = np.bincount(valid, minlength=3).astype(np.float64)
    hst /= max(1.0, float(hst.sum()))
    return hst


def _dynamic_sf_wdl_weight(
    *,
    sf_wdl_start: float,
    sf_wdl_floor: float,
    sf_wdl_floor_at_regret: float,
    sf_wdl_floor_at_rmp: float,
    regret_max: float,
    wdl_regret_used: float,
    current_rand: float,
) -> float | None:
    """Compute the dynamic sf_wdl weight based on difficulty proxy.

    Returns the interpolated weight, or None if sf_wdl_start <= 0 (disabled).
    """
    if sf_wdl_start <= 0:
        return None
    if float(wdl_regret_used) >= 0.0:
        regret = float(wdl_regret_used)
        if regret >= regret_max:
            return sf_wdl_start
        elif regret <= sf_wdl_floor_at_regret:
            return sf_wdl_floor
        else:
            t = (regret - sf_wdl_floor_at_regret) / (regret_max - sf_wdl_floor_at_regret)
            return sf_wdl_floor + t * (sf_wdl_start - sf_wdl_floor)
    else:
        rmp = float(current_rand)
        if rmp >= 1.0:
            return sf_wdl_start
        elif rmp <= sf_wdl_floor_at_rmp:
            return sf_wdl_floor
        else:
            t = (rmp - sf_wdl_floor_at_rmp) / (1.0 - sf_wdl_floor_at_rmp)
            return sf_wdl_floor + t * (sf_wdl_start - sf_wdl_floor)


def _compute_drift_metrics(
    *,
    buf: DiskReplayBuffer,
    holdout_buf: ArrayReplayBuffer,
    drift_sample_size: int,
) -> DriftMetrics:
    """Compute drift and data diversity metrics from training and holdout buffers."""
    dm = DriftMetrics()
    eps = 1e-12

    if len(buf) < drift_sample_size:
        return dm

    train_batch = _sample_drift_arrays(buf, drift_sample_size)

    dm.data_policy_entropy = _mean_entropy(train_batch)

    # Unique positions (approximate via row-hash).
    train_x = train_batch["x"]
    x_flat = train_x.reshape(train_x.shape[0], -1)
    row_sums = x_flat.view(np.uint32).reshape(x_flat.shape[0], -1).sum(axis=1)
    dm.data_unique_positions = float(np.unique(row_sums).shape[0]) / float(max(1, train_x.shape[0]))

    # WDL balance: entropy of WDL distribution.
    wdl_arr = np.asarray(train_batch["wdl_target"], dtype=np.int64)
    wdl_valid = wdl_arr[(wdl_arr >= 0) & (wdl_arr <= 2)]
    if wdl_valid.size > 0:
        h = np.bincount(wdl_valid, minlength=3).astype(np.float64)
        h /= max(1.0, float(h.sum()))
        dm.data_wdl_balance = float(-np.sum(h * np.log(h + eps)))

    # Drift metrics (train vs holdout).
    if len(holdout_buf) >= drift_sample_size:
        hold_batch = _sample_drift_arrays(holdout_buf, drift_sample_size)
        hold_x = hold_batch["x"]
        dm.drift_input_l2 = float(np.linalg.norm(train_x.mean(axis=0) - hold_x.mean(axis=0)))

        p = _wdl_hist(train_batch)
        q = _wdl_hist(hold_batch)
        m = 0.5 * (p + q)
        dm.drift_wdl_js = float(
            0.5 * np.sum(p * (np.log(p + eps) - np.log(m + eps)))
            + 0.5 * np.sum(q * (np.log(q + eps) - np.log(m + eps)))
        )

        dm.drift_policy_entropy_train = dm.data_policy_entropy
        dm.drift_policy_entropy_holdout = _mean_entropy(hold_batch)
        dm.drift_policy_entropy_diff = float(dm.drift_policy_entropy_train - dm.drift_policy_entropy_holdout)

    return dm
