from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# Floor on per-observation SE to prevent saturated (all-win/all-loss/all-draw)
# batches from dominating the weighted fit with 1/SE^2 weights.
_OBSERVATION_SE_FLOOR = 0.01

# Max ±fractional change to nodes per observe() step once the nodes stage is
# active. Hardcoded — this is a safety rail, not a tuning knob.
_NODES_STEP_CAP = 0.10


def _observation_se(wins: int, draws: int, losses: int) -> float:
    """Standard error of the mean score for a batch of wins/draws/losses.

    Score per game x_i in {1, 0.5, 0}. Floored at _OBSERVATION_SE_FLOOR.
    """
    n = int(wins) + int(draws) + int(losses)
    if n <= 1:
        return 0.5
    mu = (float(wins) + 0.5 * float(draws)) / float(n)
    var_sum = (
        float(wins) * (1.0 - mu) ** 2
        + float(draws) * (0.5 - mu) ** 2
        + float(losses) * (0.0 - mu) ** 2
    )
    sample_var = var_sum / float(n - 1)
    raw_se = math.sqrt(max(sample_var, 0.0) / float(n))
    return max(raw_se, _OBSERVATION_SE_FLOOR)


def _fit_inverse_regret(
    history: list[tuple[float, float, float]],
    *,
    target_wr: float,
    recency_half_life: float = 0.0,
) -> float | None:
    """Weighted least-squares fit ``winrate = a + b*regret``, solve for r* at target.

    History is ordered oldest-first. When ``recency_half_life > 0`` each point
    is additionally weighted by ``0.5 ** (age / half_life)`` so recent points
    dominate — intended to handle training drift that lifts the regret→winrate
    curve upward over iterations.

    Returns the predicted regret, or None if the fit is degenerate (fewer than
    3 points, zero x variance, or physically implausible negative slope).
    Callers handle None by falling back to an exploration step.
    """
    if len(history) < 3:
        return None
    r_vals = [float(h[0]) for h in history]
    w_vals = [float(h[1]) for h in history]
    se_vals = [max(float(h[2]), _OBSERVATION_SE_FLOOR) for h in history]
    weights = [1.0 / (s * s) for s in se_vals]
    if recency_half_life > 0.0:
        n = len(history)
        for i in range(n):
            age = (n - 1) - i
            weights[i] *= 0.5 ** (age / recency_half_life)
    sw = sum(weights)
    swr = sum(w * r for w, r in zip(weights, r_vals))
    sww = sum(w * v for w, v in zip(weights, w_vals))
    swrr = sum(w * r * r for w, r in zip(weights, r_vals))
    swrw = sum(w * r * v for w, r, v in zip(weights, r_vals, w_vals))
    det = sw * swrr - swr * swr
    if abs(det) < 1e-12:
        return None
    a = (swrr * sww - swr * swrw) / det
    b = (sw * swrw - swr * sww) / det
  # Physics sanity: winrate should increase with regret (easier SF → more wins).
    if b <= 1e-4:
        return None
    return (float(target_wr) - a) / b


@dataclass
class PIDUpdate:
    nodes_before: int
    nodes_after: int
    ema_winrate: float
    err: float
    adjusted: bool

    wdl_regret_before: float = -1.0
    wdl_regret_after: float = -1.0
    wdl_regret_changed: bool = False


class DifficultyPID:
    """Adaptive difficulty controller — regret + nodes, inverse-model only.

    Two knobs in series:
      1. Regret stage: inverse-model controller fits winrate = a + b*regret
         from a rolling history of observations and moves regret toward r*
         predicted to hit the target winrate. Capped per-iter by
         inverse_regret_max_step scaled by prediction confidence. Safety
         floor forces emergency ease when raw winrate drops below floor.
      2. Nodes stage: only active once regret has dropped to
         wdl_regret_stage_end (the gate). Proportional multiplicative step
         against the EMA-winrate error, capped at ±_NODES_STEP_CAP per iter.
    """

    def __init__(
        self,
        *,
        initial_nodes: int,
        target_winrate: float = 0.60,
        ema_alpha: float = 0.03,
        min_games_between_adjust: int = 30,
        min_nodes: int = 1_000,
        max_nodes: int = 1_000_000,
  # --- WDL regret ---
        initial_wdl_regret: float = -1.0,
        wdl_regret_min: float = 0.01,
        wdl_regret_max: float = 1.0,
        wdl_regret_stage_end: float = -1.0,
        wdl_regret_stage_reenter: float | None = None,
  # --- Inverse-model regret controller ---
        inverse_regret_window: int = 20,
        inverse_regret_max_step: float = 0.01,
        inverse_regret_max_step_frac: float = 0.0,
        inverse_regret_safety_floor: float = 0.50,
        inverse_regret_emergency_ease_step: float = 0.01,
        inverse_regret_recency_half_life: float = 0.0,
        inverse_regret_target_deadband_sigma: float = 1.0,
    ):
        init = int(initial_nodes)
        self.nodes = int(_clamp(init, int(min_nodes), int(max_nodes)))
        self.target = float(target_winrate)
        self.alpha = float(ema_alpha)
        self.min_games_between_adjust = int(min_games_between_adjust)
        self.min_nodes = int(min_nodes)
        self.max_nodes = int(max_nodes)

        self._regret_enabled = float(initial_wdl_regret) >= 0.0
        self._regret_gate_enabled = self._regret_enabled and float(wdl_regret_stage_end) >= 0.0
        self.wdl_regret_min = float(wdl_regret_min)
        self.wdl_regret_max = float(wdl_regret_max)
        if self._regret_enabled:
            self.wdl_regret = _clamp(float(initial_wdl_regret), self.wdl_regret_min, self.wdl_regret_max)
        else:
            self.wdl_regret = float(initial_wdl_regret)

        if self._regret_gate_enabled:
            self.wdl_regret_stage_end = _clamp(
                float(wdl_regret_stage_end), self.wdl_regret_min, self.wdl_regret_max
            )
            regret_reenter_default = min(self.wdl_regret_max, self.wdl_regret_stage_end + 0.05)
            regret_reenter_val = (
                regret_reenter_default if wdl_regret_stage_reenter is None else float(wdl_regret_stage_reenter)
            )
            self.wdl_regret_stage_reenter = _clamp(
                regret_reenter_val, self.wdl_regret_stage_end, self.wdl_regret_max
            )
            self._regret_stage_complete = float(self.wdl_regret) <= float(self.wdl_regret_stage_end)
        else:
            self.wdl_regret_stage_end = float(wdl_regret_stage_end)
            self.wdl_regret_stage_reenter = float(wdl_regret_stage_end)
            self._regret_stage_complete = True

        self.ema_winrate: float = float(target_winrate)
        self._games_since_adjust = 0

        self.inverse_regret_window = int(inverse_regret_window)
        self.inverse_regret_max_step = float(inverse_regret_max_step)
        self.inverse_regret_max_step_frac = float(inverse_regret_max_step_frac)
        self.inverse_regret_safety_floor = float(inverse_regret_safety_floor)
        self.inverse_regret_emergency_ease_step = float(inverse_regret_emergency_ease_step)
        self.inverse_regret_recency_half_life = float(inverse_regret_recency_half_life)
        self.inverse_regret_target_deadband_sigma = float(inverse_regret_target_deadband_sigma)
        self._inverse_history: deque[tuple[float, float, float]] = deque(
            maxlen=max(3, self.inverse_regret_window)
        )

    def refresh_live_params(self, config: dict) -> None:
        """Re-read live-reloadable knobs from a flat config dict.

        Construction-time fields (``min_nodes``, ``max_nodes``,
        ``inverse_regret_window``, regret enable/gate flags) stay pinned — the
        deque ``maxlen`` and the ``_regret_enabled``/``_regret_gate_enabled``
        flags are decided once in ``__init__``. Everything else is a scalar
        attr that can be safely overwritten each iteration so live-reloaded
        yaml actually takes effect.
        """
        if "sf_pid_target_winrate" in config:
            self.target = float(config["sf_pid_target_winrate"])
        if "sf_pid_ema_alpha" in config:
            self.alpha = float(config["sf_pid_ema_alpha"])
        if "sf_pid_min_games_between_adjust" in config:
            self.min_games_between_adjust = int(config["sf_pid_min_games_between_adjust"])
        if "sf_pid_wdl_regret_min" in config:
            self.wdl_regret_min = float(config["sf_pid_wdl_regret_min"])
        if "sf_pid_wdl_regret_max" in config:
            self.wdl_regret_max = float(config["sf_pid_wdl_regret_max"])
        if self._regret_gate_enabled and "sf_pid_wdl_regret_stage_end" in config:
            stage_end = _clamp(
                float(config["sf_pid_wdl_regret_stage_end"]),
                self.wdl_regret_min, self.wdl_regret_max,
            )
            self.wdl_regret_stage_end = stage_end
            self.wdl_regret_stage_reenter = _clamp(
                self.wdl_regret_stage_reenter, stage_end, self.wdl_regret_max,
            )
        if "sf_pid_inverse_regret_max_step" in config:
            self.inverse_regret_max_step = float(config["sf_pid_inverse_regret_max_step"])
        if "sf_pid_inverse_regret_max_step_frac" in config:
            self.inverse_regret_max_step_frac = float(
                config["sf_pid_inverse_regret_max_step_frac"]
            )
        if "sf_pid_inverse_regret_safety_floor" in config:
            self.inverse_regret_safety_floor = float(
                config["sf_pid_inverse_regret_safety_floor"]
            )
        if "sf_pid_inverse_regret_emergency_ease_step" in config:
            self.inverse_regret_emergency_ease_step = float(
                config["sf_pid_inverse_regret_emergency_ease_step"]
            )
        if "sf_pid_inverse_regret_recency_half_life" in config:
            self.inverse_regret_recency_half_life = float(
                config["sf_pid_inverse_regret_recency_half_life"]
            )
        if "sf_pid_inverse_regret_target_deadband_sigma" in config:
            self.inverse_regret_target_deadband_sigma = float(
                config["sf_pid_inverse_regret_target_deadband_sigma"]
            )

    def state_dict(self) -> dict:
        return {
            "nodes": int(self.nodes),
            "wdl_regret": float(self.wdl_regret),
            "ema_winrate": float(self.ema_winrate),
            "games_since_adjust": int(self._games_since_adjust),
            "regret_stage_complete": bool(self._regret_stage_complete),
            "inverse_history": [
                [float(r), float(w), float(s)]
                for (r, w, s) in self._inverse_history
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore controller state. Silently ignores retired fields from legacy checkpoints."""
        self.nodes = int(_clamp(int(state.get("nodes", self.nodes)), self.min_nodes, self.max_nodes))

        if self._regret_enabled:
            self.wdl_regret = _clamp(
                float(state.get("wdl_regret", self.wdl_regret)),
                self.wdl_regret_min, self.wdl_regret_max,
            )

        self.ema_winrate = float(state.get("ema_winrate", self.ema_winrate))

        saved_hist = state.get("inverse_history") or []
        if saved_hist:
            self._inverse_history.clear()
            for entry in saved_hist:
                try:
                    r, w, s = float(entry[0]), float(entry[1]), float(entry[2])
                except (TypeError, ValueError, IndexError):
                    continue
                self._inverse_history.append((r, w, s))

        self._games_since_adjust = int(state.get("games_since_adjust", self._games_since_adjust))

        rgc = state.get("regret_stage_complete")
        if rgc is not None and self._regret_gate_enabled:
            self._regret_stage_complete = bool(rgc)

    def _no_change_update(self, err: float) -> PIDUpdate:
        return PIDUpdate(
            nodes_before=int(self.nodes),
            nodes_after=int(self.nodes),
            ema_winrate=float(self.ema_winrate),
            err=err,
            adjusted=False,
            wdl_regret_before=float(self.wdl_regret),
            wdl_regret_after=float(self.wdl_regret),
            wdl_regret_changed=False,
        )

    def observe(self, *, wins: int, draws: int, losses: int, force: bool = False) -> PIDUpdate:
        games = int(wins) + int(draws) + int(losses)
        if games <= 0:
            return self._no_change_update(0.0)

        wr = (float(wins) + 0.5 * float(draws)) / float(games)
        self.ema_winrate = (1.0 - self.alpha) * float(self.ema_winrate) + self.alpha * float(wr)

        self._games_since_adjust += games

        err = float(self.ema_winrate) - float(self.target)
        nodes_before = int(self.nodes)
        raw_wr_this_batch = float(wr)
        se_this_batch = _observation_se(int(wins), int(draws), int(losses))

        if not force and self._games_since_adjust < int(self.min_games_between_adjust):
            return self._no_change_update(err)

  # --- Stage 1: inverse-model regret controller ---
        regret_before = float(self.wdl_regret)
        regret_changed = False

        if self._regret_enabled:
            self._inverse_history.append(
                (float(regret_before), raw_wr_this_batch, se_this_batch)
            )

            regret_after = float(regret_before)
            floor = self.inverse_regret_safety_floor
  # Dual z-gate: act if EITHER the single iter's raw err is
  # beyond sigma × SE, OR the EMA of err (which accumulates
  # same-direction evidence) is beyond sigma × SE(ema_err).
  #   - Raw gate catches one clear spike (|z|>1.5 is ~6.5:1 real).
  #   - EMA gate catches a slow drift of small same-direction
  #     deviations that the raw gate would miss each iter.
  # For iid per-iter err with variance σ², the EMA's steady-state
  # variance is α/(2-α) · σ², so SE(ema_err) = σ · √(α/(2-α)).
            err = self.target - raw_wr_this_batch
            ema_err = self.target - float(self.ema_winrate)
            ema_se_factor = math.sqrt(self.alpha / max(2.0 - self.alpha, 1e-9))
            sigma = max(0.0, self.inverse_regret_target_deadband_sigma)
            raw_deadband = sigma * se_this_batch
            ema_deadband = sigma * se_this_batch * ema_se_factor
            in_deadband = abs(err) <= raw_deadband and abs(ema_err) <= ema_deadband
  # Effective absolute step cap for this iter. When max_step_frac > 0
  # it scales with current regret (2% of 0.07 ≈ 0.0014; 2% of 0.02 ≈
  # 0.0004). Constant-regret steps become non-constant-wr steps as
  # SF approaches optimal, so scaling by current regret keeps the
  # wr perturbation per step roughly constant.
            if self.inverse_regret_max_step_frac > 0.0:
                abs_max_step = self.inverse_regret_max_step_frac * regret_before
            else:
                abs_max_step = self.inverse_regret_max_step

            if raw_wr_this_batch < floor:
  # Safety airbag — raw_wr dropped below floor. Ease regret by
  # emergency_ease_step, bypassing the max_step_frac cap so the
  # airbag can loosen faster than normal PID tightens. Still
  # clamped by the absolute max_step as a runaway guard. Fires
  # strictly below floor so the deadband around target stays
  # reachable.
                ease = min(self.inverse_regret_emergency_ease_step, self.inverse_regret_max_step)
                regret_after = _clamp(
                    regret_before + ease, self.wdl_regret_min, self.wdl_regret_max
                )
            elif in_deadband:
  # Neither raw nor ema err crossed its sigma-gated threshold
  # — hold regret steady.
                pass
            else:
                predicted_regret = _fit_inverse_regret(
                    list(self._inverse_history),
                    target_wr=self.target,
                    recency_half_life=self.inverse_regret_recency_half_life,
                )
                if predicted_regret is not None:
  # Step toward fit's predicted r*, bounded by abs_max_step. Prior
  # versions scaled by sigma_pred/sigma_tolerance, but backtest on
  # 168 iters showed sigma_pred has ~0 correlation with actual
  # prediction error (Pearson 0.009) — the iid-Gaussian residual
  # assumption doesn't hold for drifting training data. max_step_frac
  # (cap) is the only bound that actually works.
                    delta = _clamp(
                        predicted_regret - regret_before,
                        -abs_max_step,
                        abs_max_step,
                    )
                else:
  # Fit degenerate (history span < min_span, or other).
  # We're outside the deadband (checked above) so we know the
  # direction to move. Step sign(err) * abs_max_step to
  # explore and widen history for the next iteration's fit.
                    step_sign = 1.0 if err > 0 else -1.0
                    delta = step_sign * abs_max_step

  # Raw-vs-fit sign disagreement: raw is the fresh single-iter
  # signal; the fit is averaged over ``inverse_regret_window`` iters
  # and lags reality. When they disagree (and raw is outside its own
  # deadband — within the deadband, raw is statistical noise around
  # target and shouldn't override the fit), step in raw's direction
  # at half ``abs_max_step``. Full step would defeat the fit's
  # magnitude calibration when signs agree; zero (the prior behavior)
  # leaves us holding while raw genuinely underperforms.
                if err > raw_deadband and delta < 0:
                    delta = 0.5 * abs_max_step
                elif err < -raw_deadband and delta > 0:
                    delta = -0.5 * abs_max_step

                regret_after = _clamp(
                    regret_before + delta,
                    self.wdl_regret_min,
                    self.wdl_regret_max,
                )
            self.wdl_regret = float(regret_after)
            regret_changed = abs(regret_after - regret_before) > 1e-12

            if self._regret_gate_enabled:
                regret_end = _clamp(self.wdl_regret_stage_end, self.wdl_regret_min, self.wdl_regret_max)
                regret_reenter = _clamp(self.wdl_regret_stage_reenter, regret_end, self.wdl_regret_max)
                if self._regret_stage_complete:
                    if self.wdl_regret >= regret_reenter:
                        self._regret_stage_complete = False
                else:
                    if self.wdl_regret <= regret_end:
                        self._regret_stage_complete = True

  # --- Stage 2: nodes (only once regret stage is complete) ---
  # Use ema − target here (NOT the regret-stage-local err, which is
  # target − raw and has the opposite sign). Positive ema_err = model
  # winning too much → raise nodes. Shadowing caused a positive-feedback
  # loop where nodes climbed while wr dropped.
        nodes_after = int(self.nodes)
        if self._regret_stage_complete:
            ema_err = float(self.ema_winrate) - float(self.target)
            delta_frac = _clamp(ema_err, -_NODES_STEP_CAP, _NODES_STEP_CAP)
            new_nodes = int(round(float(self.nodes) * (1.0 + delta_frac)))
            self.nodes = int(_clamp(new_nodes, self.min_nodes, self.max_nodes))
            nodes_after = int(self.nodes)

        self._games_since_adjust = 0

        adjusted = (nodes_after != nodes_before) or bool(regret_changed)

        return PIDUpdate(
            nodes_before=nodes_before,
            nodes_after=int(nodes_after),
            ema_winrate=float(self.ema_winrate),
            err=err,
            adjusted=bool(adjusted),
            wdl_regret_before=float(regret_before),
            wdl_regret_after=float(self.wdl_regret),
            wdl_regret_changed=bool(regret_changed),
        )


def pid_from_config(config: dict) -> DifficultyPID:
    """Construct a DifficultyPID from a flat config dict."""
    return DifficultyPID(
        initial_nodes=int(config.get("sf_nodes", 500)),
        target_winrate=float(config.get("sf_pid_target_winrate", 0.60)),
        ema_alpha=float(config.get("sf_pid_ema_alpha", 0.03)),
        min_games_between_adjust=int(config.get("sf_pid_min_games_between_adjust", 10)),
        min_nodes=int(config.get("sf_pid_min_nodes", 100)),
        max_nodes=int(config.get("sf_pid_max_nodes", 50000)),
        initial_wdl_regret=float(config.get("sf_pid_wdl_regret_start", -1.0)),
        wdl_regret_min=float(config.get("sf_pid_wdl_regret_min", 0.01)),
        wdl_regret_max=float(config.get("sf_pid_wdl_regret_max", 1.0)),
        wdl_regret_stage_end=float(config.get("sf_pid_wdl_regret_stage_end", -1.0)),
        inverse_regret_window=int(config.get("sf_pid_inverse_regret_window", 20)),
        inverse_regret_max_step=float(config.get("sf_pid_inverse_regret_max_step", 0.01)),
        inverse_regret_max_step_frac=float(
            config.get("sf_pid_inverse_regret_max_step_frac", 0.0)
        ),
        inverse_regret_safety_floor=float(
            config.get("sf_pid_inverse_regret_safety_floor", 0.50)
        ),
        inverse_regret_emergency_ease_step=float(
            config.get("sf_pid_inverse_regret_emergency_ease_step", 0.01)
        ),
        inverse_regret_recency_half_life=float(
            config.get("sf_pid_inverse_regret_recency_half_life", 0.0)
        ),
        inverse_regret_target_deadband_sigma=float(
            config.get("sf_pid_inverse_regret_target_deadband_sigma", 1.0)
        ),
    )
