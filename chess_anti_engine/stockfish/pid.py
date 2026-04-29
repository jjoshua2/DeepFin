from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# Floor on per-observation SE to prevent saturated (all-win/all-loss/all-draw)
# batches from dominating the weighted fit with 1/SE^2 weights.
_OBSERVATION_SE_FLOOR = 0.01


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


def _fit_inverse_lever(
    history: list[tuple[float, float, float]],
    *,
    target_wr: float,
    expected_slope_sign: int,
    recency_half_life: float = 0.0,
) -> float | None:
    """Weighted least-squares fit ``winrate = a + b*x``, solve for x* at target.

    History is ``(lever_value, raw_winrate, observation_se)`` entries ordered
    oldest-first. When ``recency_half_life > 0`` each point is additionally
    weighted by ``0.5 ** (age / half_life)`` so recent points dominate.

    ``expected_slope_sign`` ±1 is a physics check on b: regret-style levers
    expect b > 0 (more regret → more wins), nodes-style levers expect b < 0
    (more nodes → fewer wins). A wrong-sign fit is treated as degenerate.

    Returns the predicted x* at ``target_wr``, or None if degenerate (fewer
    than 3 points, zero x variance, or wrong-sign slope).
    """
    if len(history) < 3:
        return None
    x_vals = [float(h[0]) for h in history]
    w_vals = [float(h[1]) for h in history]
    se_vals = [max(float(h[2]), _OBSERVATION_SE_FLOOR) for h in history]
    weights = [1.0 / (s * s) for s in se_vals]
    if recency_half_life > 0.0:
        n = len(history)
        for i in range(n):
            age = (n - 1) - i
            weights[i] *= 0.5 ** (age / recency_half_life)
    sw = sum(weights)
    swx = sum(w * x for w, x in zip(weights, x_vals))
    swr = sum(w * v for w, v in zip(weights, w_vals))
    swxx = sum(w * x * x for w, x in zip(weights, x_vals))
    swxr = sum(w * x * v for w, x, v in zip(weights, x_vals, w_vals))
    det = sw * swxx - swx * swx
    if abs(det) < 1e-12:
        return None
    a = (swxx * swr - swx * swxr) / det
    b = (sw * swxr - swx * swr) / det
    # Reject only floating-point zeros and wrong-sign slopes — nodes-side
    # b is O(1e-6/node), so a 1e-4 cutoff would reject every real fit.
    if b * expected_slope_sign <= 1e-12:
        return None
    return (target_wr - a) / b


@dataclass
class _Lever:
    """A difficulty knob with inverse-fit + dual z-gate update logic.

    ``direction``:
      +1 → raising ``value`` makes things HARDER (lowers winrate). Used for
           sf_nodes: more nodes = stronger Stockfish.
      -1 → lowering ``value`` makes things HARDER. Used for wdl_regret:
           less regret = stronger Stockfish (closer to optimal play).

    The fit ``wr = a + b·value`` has expected slope ``b`` of sign
    ``-direction``: regret slope is positive, nodes slope is negative.
    Stored under ``history`` with ``deque(maxlen=window)`` for replay.

    All knobs except ``window`` (deque maxlen) and ``min_value/max_value``
    are intended to be live-reloadable.
    """

    name: str
    value: float
    min_value: float
    max_value: float
    direction: int
    max_step: float
    max_step_frac: float = 0.0
    safety_floor: float = 0.50
    emergency_ease_step: float = 0.01
    recency_half_life: float = 0.0
    deadband_sigma: float = 1.0
    window: int = 20
    history: deque[tuple[float, float, float]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.direction = 1 if self.direction >= 0 else -1
        self.value = _clamp(float(self.value), self.min_value, self.max_value)
        self.history = deque(self.history, maxlen=max(3, self.window))


def _step_lever(
    lever: _Lever,
    *,
    target_wr: float,
    raw_wr: float,
    ema_wr: float,
    ema_alpha: float,
    se: float,
) -> bool:
    """Apply one observation to ``lever``; return True if value changed.

    Branches via ``lever.direction``:
      - airbag (raw_wr < safety_floor): step easier by emergency_ease_step
        (capped by max_step). Records the data point — extreme low wr is
        informative for future fits.
      - dual z-gate deadband: hold AND skip the history append. Appending
        same-x entries during steady state collapses the next fit's slope
        (det → 0 → degenerate), forcing blind exploration steps the next
        time a real disturbance arrives.
      - else: fit ``wr = a + b·value``, step toward predicted x* capped by
        abs_max_step (= max_step_frac·value if frac > 0 else max_step).
        Fit-degenerate → exploration step in raw's direction. Raw vs fit
        sign disagreement → half-step in raw's direction.
    """
    value_before = lever.value
    ease_sign = -lever.direction  # +1 for regret (up=easier), -1 for nodes

    err = target_wr - raw_wr
    ema_err = target_wr - ema_wr
    # EMA's steady-state variance is α/(2-α)·σ²; SE(ema_err) = σ·√(α/(2-α)).
    ema_se_factor = math.sqrt(ema_alpha / max(2.0 - ema_alpha, 1e-9))
    sigma = max(0.0, lever.deadband_sigma)
    raw_deadband = sigma * se
    ema_deadband = raw_deadband * ema_se_factor
    in_deadband = abs(err) <= raw_deadband and abs(ema_err) <= ema_deadband

    # Airbag fires only when raw_wr is statistically distinguishable from the
    # floor (1.5σ). Without this, a 77-game iter at 0.435 (SE≈0.057, well within
    # noise of 0.50) fires the same response as a 665-game iter at 0.413
    # (SE≈0.019, 4σ below floor). The first is noise; the second is a real
    # crash. Same trigger → spurious overshoots near equilibrium.
    if raw_wr + 1.5 * se < lever.safety_floor:
        lever.history.append((value_before, raw_wr, se))
        # Airbag capped by max_step so a runaway ease branch can't break the
        # user's per-iter movement promise.
        ease_mag = min(lever.emergency_ease_step, lever.max_step)
        delta = ease_sign * ease_mag
        new_value = _clamp(value_before + delta, lever.min_value, lever.max_value)
    elif in_deadband:
        return False  # hold; skip history append (see docstring).
    else:
        lever.history.append((value_before, raw_wr, se))
        # frac scales with value so wr-perturbation per step stays roughly
        # constant as the lever approaches its target.
        abs_max_step = (
            lever.max_step_frac * value_before if lever.max_step_frac > 0.0
            else lever.max_step
        )
        predicted = _fit_inverse_lever(
            list(lever.history),
            target_wr=target_wr,
            expected_slope_sign=ease_sign,
            recency_half_life=lever.recency_half_life,
        )
        if predicted is not None:
            delta = _clamp(predicted - value_before, -abs_max_step, abs_max_step)
        else:
            # Fit degenerate, but direction is known (outside deadband) — step
            # abs_max_step to widen history for the next fit.
            delta = (1.0 if err > 0 else -1.0) * ease_sign * abs_max_step

        # Raw is fresh; the fit lags by ~window iters. On sign disagreement
        # with raw outside its deadband, override delta to half-step in raw's
        # direction (full step would defeat the fit's magnitude calibration
        # when signs agree; zero would hold while raw genuinely underperforms).
        if err > raw_deadband and delta * ease_sign < -1e-12:
            delta = ease_sign * 0.5 * abs_max_step
        elif err < -raw_deadband and delta * ease_sign > 1e-12:
            delta = -ease_sign * 0.5 * abs_max_step

        new_value = _clamp(value_before + delta, lever.min_value, lever.max_value)

    lever.value = new_value
    return abs(new_value - value_before) > 1e-12


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


# Sentinel "no absolute cap" for the nodes lever. The nodes-side
# max_step_frac (default 10%) is the real per-iter cap; the absolute
# max_step bound only matters for the airbag. Set high so the airbag's
# min(emergency_ease_step, max_step) cap is governed by emergency_ease_step.
_NODES_MAX_STEP_DEFAULT = 1_000_000_000.0


_LEVER_LIVE_FIELDS = (
    "max_step",
    "max_step_frac",
    "safety_floor",
    "emergency_ease_step",
    "recency_half_life",
    "deadband_sigma",
)
# Fields where negative values are physically meaningless; clamp at 0.
_LEVER_NONNEG_FIELDS: frozenset[str] = frozenset({
    "max_step", "max_step_frac",
    "emergency_ease_step", "recency_half_life", "deadband_sigma",
})


class DifficultyPID:
    """Adaptive difficulty controller — regret + nodes via shared lever logic.

    Two ``_Lever`` instances share the same inverse-fit + dual z-gate code
    via ``_step_lever``; the only difference is ``direction``:
      - regret_lever (direction=-1): lowering regret makes SF stronger.
      - nodes_lever (direction=+1):  raising nodes makes SF stronger.

    Series gate (one-way): regret moves first. Once ``self.wdl_regret`` ≤
    stage_end, the regret lever freezes and the nodes lever takes over for
    the life of this PID. The nodes lever has its own airbag for stage-2
    wr-crash response — keeping the controller a single-knob system in
    each regime so a change has one observable effect.
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
        # WDL regret bounds + stage gate
        initial_wdl_regret: float = -1.0,
        wdl_regret_min: float = 0.01,
        wdl_regret_max: float = 1.0,
        wdl_regret_stage_end: float = -1.0,
        # Regret lever
        regret_window: int = 20,
        regret_max_step: float = 0.01,
        regret_max_step_frac: float = 0.0,
        regret_safety_floor: float = 0.50,
        regret_emergency_ease_step: float = 0.01,
        regret_recency_half_life: float = 0.0,
        regret_deadband_sigma: float = 1.0,
        # Nodes lever — defaults match the prior proportional controller
        nodes_window: int = 20,
        nodes_max_step: float | None = None,
        nodes_max_step_frac: float = 0.10,
        nodes_safety_floor: float = 0.0,
        nodes_emergency_ease_step: float = 0.0,
        nodes_recency_half_life: float = 0.0,
        nodes_deadband_sigma: float = 0.0,
    ):
        init = int(initial_nodes)
        nodes_clamped = int(_clamp(init, int(min_nodes), int(max_nodes)))
        self.target = float(target_winrate)
        self.alpha = float(ema_alpha)
        self.min_games_between_adjust = int(min_games_between_adjust)
        self.min_nodes = int(min_nodes)
        self.max_nodes = int(max_nodes)

        self._regret_enabled = float(initial_wdl_regret) >= 0.0
        self._regret_gate_enabled = self._regret_enabled and float(wdl_regret_stage_end) >= 0.0
        self.wdl_regret_min = float(wdl_regret_min)
        self.wdl_regret_max = float(wdl_regret_max)
        regret_init = (
            _clamp(float(initial_wdl_regret), self.wdl_regret_min, self.wdl_regret_max)
            if self._regret_enabled else float(initial_wdl_regret)
        )

        if self._regret_gate_enabled:
            self.wdl_regret_stage_end = _clamp(
                float(wdl_regret_stage_end), self.wdl_regret_min, self.wdl_regret_max
            )
            self._regret_stage_complete = float(regret_init) <= float(self.wdl_regret_stage_end)
        else:
            self.wdl_regret_stage_end = float(wdl_regret_stage_end)
            self._regret_stage_complete = True

        self.ema_winrate: float = float(target_winrate)
        self._games_since_adjust = 0

        self.regret_lever = _Lever(
            name="regret",
            value=regret_init,
            min_value=self.wdl_regret_min,
            max_value=self.wdl_regret_max,
            direction=-1,  # lowering regret makes harder
            max_step=float(regret_max_step),
            max_step_frac=float(regret_max_step_frac),
            safety_floor=float(regret_safety_floor),
            emergency_ease_step=float(regret_emergency_ease_step),
            recency_half_life=float(regret_recency_half_life),
            deadband_sigma=float(regret_deadband_sigma),
            window=int(regret_window),
        )
        self.nodes_lever = _Lever(
            name="nodes",
            value=float(nodes_clamped),
            min_value=float(self.min_nodes),
            max_value=float(self.max_nodes),
            direction=+1,  # raising nodes makes harder
            max_step=float(nodes_max_step) if nodes_max_step is not None
                else _NODES_MAX_STEP_DEFAULT,
            max_step_frac=float(nodes_max_step_frac),
            safety_floor=float(nodes_safety_floor),
            emergency_ease_step=float(nodes_emergency_ease_step),
            recency_half_life=float(nodes_recency_half_life),
            deadband_sigma=float(nodes_deadband_sigma),
            window=int(nodes_window),
        )
        _seed_lever_history(self.regret_lever, target_wr=self.target)
        _seed_lever_history(self.nodes_lever, target_wr=self.target)

    # Compatibility shims — call sites in worker.py / tune/* still read
    # pid.wdl_regret and pid.nodes directly; renaming them is a separate
    # refactor (~48 sites across production + tests).
    @property
    def wdl_regret(self) -> float:
        return float(self.regret_lever.value)

    @wdl_regret.setter
    def wdl_regret(self, value: float) -> None:
        self.regret_lever.value = _clamp(
            float(value), float(self.regret_lever.min_value), float(self.regret_lever.max_value)
        )

    @property
    def nodes(self) -> int:
        # Ceil so a sub-1.0 step in the float lever still produces a visible
        # node change. With round-down, frac=10% at value≈1 (post-airbag floor)
        # ate every step until cumulative drift cleared 1.0 — recovery from
        # the floor took ~10 winning iters before SF visibly strengthened.
        return int(math.ceil(float(self.nodes_lever.value)))

    @nodes.setter
    def nodes(self, value: int) -> None:
        self.nodes_lever.value = float(_clamp(int(value), self.min_nodes, self.max_nodes))

    def refresh_live_params(self, config: dict) -> None:
        """Re-read live-reloadable knobs from a flat config dict.

        Construction-time fields (``min_nodes``, ``max_nodes``, lever
        ``window``, regret enable/gate flags) stay pinned. Everything else is
        a scalar attr that can be safely overwritten each iteration so
        live-reloaded yaml takes effect.
        """
        if "sf_pid_target_winrate" in config:
            self.target = float(config["sf_pid_target_winrate"])
        if "sf_pid_ema_alpha" in config:
            self.alpha = float(config["sf_pid_ema_alpha"])
        if "sf_pid_min_games_between_adjust" in config:
            self.min_games_between_adjust = int(config["sf_pid_min_games_between_adjust"])
        if "sf_pid_wdl_regret_min" in config:
            self.wdl_regret_min = float(config["sf_pid_wdl_regret_min"])
            self.regret_lever.min_value = self.wdl_regret_min
        if "sf_pid_wdl_regret_max" in config:
            self.wdl_regret_max = float(config["sf_pid_wdl_regret_max"])
            self.regret_lever.max_value = self.wdl_regret_max
        if self._regret_gate_enabled and "sf_pid_wdl_regret_stage_end" in config:
            self.wdl_regret_stage_end = _clamp(
                float(config["sf_pid_wdl_regret_stage_end"]),
                self.wdl_regret_min, self.wdl_regret_max,
            )

        _refresh_lever_from_config(self.regret_lever, config, "sf_pid_regret_")
        _refresh_lever_from_config(self.nodes_lever, config, "sf_pid_nodes_")

    def state_dict(self) -> dict:
        return {
            "nodes": int(self.nodes),
            "wdl_regret": float(self.wdl_regret),
            "ema_winrate": float(self.ema_winrate),
            "games_since_adjust": int(self._games_since_adjust),
            "regret_stage_complete": bool(self._regret_stage_complete),
            "regret_history": [
                [float(x), float(w), float(s)]
                for (x, w, s) in self.regret_lever.history
            ],
            "nodes_history": [
                [float(x), float(w), float(s)]
                for (x, w, s) in self.nodes_lever.history
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore controller state. Tolerates legacy ``inverse_history``."""
        self.nodes = int(_clamp(int(state.get("nodes", self.nodes)), self.min_nodes, self.max_nodes))

        if self._regret_enabled:
            self.wdl_regret = _clamp(
                float(state.get("wdl_regret", self.wdl_regret)),
                self.wdl_regret_min, self.wdl_regret_max,
            )

        self.ema_winrate = float(state.get("ema_winrate", self.ema_winrate))

        # `inverse_history` is the pre-2026-04-26-rename key. Drop the
        # fallback once no live training run still has a checkpoint with it
        # (i.e., when every active trial has saved at least once on the new
        # code — first checkpoint after restart suffices).
        regret_hist = state.get("regret_history") or state.get("inverse_history") or []
        _restore_history(self.regret_lever, regret_hist)
        nodes_hist = state.get("nodes_history") or []
        _restore_history(self.nodes_lever, nodes_hist)

        # If saved history was empty (legacy / pre-seed checkpoints), the
        # construction-time seed entry is now stale because we just restored
        # `nodes` / `wdl_regret` to potentially different values. Re-seed at
        # the restored values so the inverse fit's first anchor point matches
        # current state instead of the configured initial_nodes / initial_regret.
        if not regret_hist:
            _seed_lever_history(self.regret_lever, target_wr=self.target)
        if not nodes_hist:
            _seed_lever_history(self.nodes_lever, target_wr=self.target)

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
        games = wins + draws + losses
        if games <= 0:
            return self._no_change_update(0.0)

        raw_wr = (wins + 0.5 * draws) / games
        self.ema_winrate = (1.0 - self.alpha) * self.ema_winrate + self.alpha * raw_wr

        self._games_since_adjust += games
        err = self.ema_winrate - self.target
        if not force and self._games_since_adjust < self.min_games_between_adjust:
            return self._no_change_update(err)

        nodes_before = self.nodes
        regret_before = self.wdl_regret
        se = _observation_se(wins, draws, losses)

        # Regret lever freezes one-way once the stage gate fires; otherwise
        # a stage-2 wr dip would trip its airbag and double-mutate alongside
        # the nodes lever, hiding cause from effect. Exception: if the nodes
        # lever has no ease-direction headroom (already at min_nodes), there's
        # no stage-2 channel to handle a winrate drop — unfreeze regret so the
        # controller still has an ease lever instead of wedging.
        regret_changed = False
        regret_frozen = (
            self._regret_gate_enabled
            and self._regret_stage_complete
            and self.nodes > self.min_nodes
        )
        if self._regret_enabled and not regret_frozen:
            regret_changed = _step_lever(
                self.regret_lever,
                target_wr=self.target, raw_wr=raw_wr,
                ema_wr=self.ema_winrate, ema_alpha=self.alpha, se=se,
            )
            if self._regret_gate_enabled:
                regret_end = _clamp(
                    self.wdl_regret_stage_end, self.wdl_regret_min, self.wdl_regret_max
                )
                if self.wdl_regret <= regret_end:
                    self._regret_stage_complete = True

        nodes_changed = False
        # Nodes lever runs when nodes is the active controller — either regret
        # is disabled entirely (initial_wdl_regret<0 → nodes is the only knob)
        # or the stage gate is wired up AND has fired (regret descended to the
        # floor and is intentionally handing off). With regret enabled but
        # gate disabled (stage_end=-1, "no second stage"), stage_complete is
        # set True unconditionally at construction; without this guard that
        # silently gave a dual-knob system — regret AND nodes both moving each
        # iter, with nodes ratcheting up as the model won, making SF steadily
        # harder behind our backs.
        nodes_active = (
            not self._regret_enabled
            or (self._regret_gate_enabled and self._regret_stage_complete)
        )
        if nodes_active:
            nodes_changed = _step_lever(
                self.nodes_lever,
                target_wr=self.target, raw_wr=raw_wr,
                ema_wr=self.ema_winrate, ema_alpha=self.alpha, se=se,
            )

        self._games_since_adjust = 0
        nodes_after = int(self.nodes)
        adjusted = bool(regret_changed) or bool(nodes_changed) or (nodes_after != nodes_before)

        return PIDUpdate(
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            ema_winrate=float(self.ema_winrate),
            err=err,
            adjusted=bool(adjusted),
            wdl_regret_before=regret_before,
            wdl_regret_after=float(self.wdl_regret),
            wdl_regret_changed=bool(regret_changed),
        )


def _refresh_lever_from_config(lever: _Lever, config: dict, prefix: str) -> None:
    """Update ``lever``'s live-reloadable knobs from ``config[f"{prefix}{field}"]``."""
    for field_name in _LEVER_LIVE_FIELDS:
        key = f"{prefix}{field_name}"
        if key not in config:
            continue
        value = float(config[key])
        if field_name in _LEVER_NONNEG_FIELDS:
            value = max(0.0, value)
        setattr(lever, field_name, value)


def _restore_history(lever: _Lever, saved: list) -> None:
    if not saved:
        return
    lever.history.clear()
    for entry in saved:
        try:
            x, w, s = float(entry[0]), float(entry[1]), float(entry[2])
        except (TypeError, ValueError, IndexError):
            continue
        lever.history.append((x, w, s))


def _seed_lever_history(lever: _Lever, *, target_wr: float) -> None:
    """Replace history with a single anchor entry at the lever's current value.

    The inverse fit needs ≥3 points to activate, so seeding gives it a head
    start; without it the first 2 real observations produce no fit and the
    controller blindly explores. The seed encodes "until proven otherwise,
    assume the current value yields target_wr" — biases the fit toward
    holding still rather than ramping. recency_half_life ages it out within
    a window once real data arrives.
    """
    lever.history.clear()
    lever.history.append((float(lever.value), float(target_wr), _OBSERVATION_SE_FLOOR))


def pid_from_config(config: dict) -> DifficultyPID:
    """Construct a DifficultyPID from a flat config dict (yaml-flattened)."""
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
        regret_window=int(config.get("sf_pid_regret_window", 20)),
        regret_max_step=float(config.get("sf_pid_regret_max_step", 0.01)),
        regret_max_step_frac=float(config.get("sf_pid_regret_max_step_frac", 0.0)),
        regret_safety_floor=float(config.get("sf_pid_regret_safety_floor", 0.50)),
        regret_emergency_ease_step=float(config.get("sf_pid_regret_emergency_ease_step", 0.01)),
        regret_recency_half_life=float(config.get("sf_pid_regret_recency_half_life", 0.0)),
        regret_deadband_sigma=float(config.get("sf_pid_regret_deadband_sigma", 1.0)),
        nodes_window=int(config.get("sf_pid_nodes_window", 20)),
        nodes_max_step=float(config["sf_pid_nodes_max_step"])
            if "sf_pid_nodes_max_step" in config else None,
        nodes_max_step_frac=float(config.get("sf_pid_nodes_max_step_frac", 0.10)),
        nodes_safety_floor=float(config.get("sf_pid_nodes_safety_floor", 0.0)),
        nodes_emergency_ease_step=float(config.get("sf_pid_nodes_emergency_ease_step", 0.0)),
        nodes_recency_half_life=float(config.get("sf_pid_nodes_recency_half_life", 0.0)),
        nodes_deadband_sigma=float(config.get("sf_pid_nodes_deadband_sigma", 0.0)),
    )
