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
  # Threshold filters out floating-point zero slopes without rejecting
  # small-but-real ones (nodes-side slopes are O(1e-6/node) — a 1e-4 cutoff
  # would reject every real fit). Wrong-sign slopes still get rejected.
    if b * float(expected_slope_sign) <= 1e-12:
        return None
    return (float(target_wr) - a) / b


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
        self.direction = 1 if int(self.direction) >= 0 else -1
        self.value = _clamp(float(self.value), float(self.min_value), float(self.max_value))
        target_max = max(3, int(self.window))
        if self.history.maxlen != target_max:
            self.history = deque(self.history, maxlen=target_max)


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

    Mirrors the structure used previously for the regret-only inverse-fit
    controller, generalized via ``lever.direction``:
      - airbag: ``raw_wr < safety_floor`` → step in the easier direction
        by ``emergency_ease_step`` (capped by ``max_step``).
      - dual z-gate deadband (raw and EMA both within ``sigma·SE``) → hold.
      - else fit ``wr = a + b·value`` and step toward predicted x*, capped
        by ``abs_max_step`` (``max_step_frac·value`` if frac > 0 else
        ``max_step``). Fit-degenerate → exploration step in raw's direction.
      - raw vs fit sign disagreement → half-step in raw's direction.
    """
    value_before = float(lever.value)
    lever.history.append((value_before, float(raw_wr), float(se)))

    # ease_sign is the lever delta sign that decreases difficulty.
    # direction=+1 (nodes): easier = decrease → ease_sign = -1.
    # direction=-1 (regret): easier = increase → ease_sign = +1.
    ease_sign = -lever.direction

    err = float(target_wr) - float(raw_wr)
    ema_err = float(target_wr) - float(ema_wr)
  # Dual z-gate: act if EITHER the single iter's raw err is beyond
  # sigma × SE, OR the EMA of err is beyond sigma × SE(ema_err).
  # For iid per-iter err with variance σ², EMA's steady-state variance
  # is α/(2-α) · σ², so SE(ema_err) = σ · √(α/(2-α)).
    ema_se_factor = math.sqrt(float(ema_alpha) / max(2.0 - float(ema_alpha), 1e-9))
    sigma = max(0.0, float(lever.deadband_sigma))
    raw_deadband = sigma * float(se)
    ema_deadband = sigma * float(se) * ema_se_factor
    in_deadband = abs(err) <= raw_deadband and abs(ema_err) <= ema_deadband

  # max_step_frac > 0 scales the cap with current value, so wr-perturbation
  # per step stays roughly constant as the lever approaches its target.
    if lever.max_step_frac > 0.0:
        abs_max_step = float(lever.max_step_frac) * value_before
    else:
        abs_max_step = float(lever.max_step)

    if float(raw_wr) < float(lever.safety_floor):
  # Airbag: emergency ease, capped by max_step (absolute) so a runaway
  # ease branch can't break the user's per-iter movement promise.
        ease_mag = min(float(lever.emergency_ease_step), float(lever.max_step))
        delta = float(ease_sign) * ease_mag
        new_value = _clamp(value_before + delta, float(lever.min_value), float(lever.max_value))
    elif in_deadband:
        new_value = value_before
    else:
        predicted = _fit_inverse_lever(
            list(lever.history),
            target_wr=float(target_wr),
            expected_slope_sign=int(ease_sign),
            recency_half_life=float(lever.recency_half_life),
        )
        if predicted is not None:
            delta = _clamp(predicted - value_before, -abs_max_step, abs_max_step)
        else:
  # Fit degenerate. We're outside the deadband so the direction is
  # known; step by abs_max_step in the err's "ease/tighten" direction
  # to widen history for the next iter's fit.
            step_sign = 1.0 if err > 0 else -1.0
            delta = step_sign * float(ease_sign) * abs_max_step

  # Raw-vs-fit sign disagreement: raw is the fresh single-iter
  # signal; the fit is averaged over the window and lags reality.
  # When raw is outside its deadband AND its sign disagrees with
  # delta's, step in raw's direction at half abs_max_step.
        if err > raw_deadband and delta * float(ease_sign) < -1e-12:
            delta = float(ease_sign) * 0.5 * abs_max_step
        elif err < -raw_deadband and delta * float(ease_sign) > 1e-12:
            delta = -float(ease_sign) * 0.5 * abs_max_step

        new_value = _clamp(value_before + delta, float(lever.min_value), float(lever.max_value))

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


# Default knobs for the nodes lever — chosen so older configs that don't
# specify them still get the proportional-style stage-2 behavior the
# previous controller used (±10%/iter, no airbag, no deadband).
_NODES_DEFAULTS = dict(
    max_step_frac=0.10,
    max_step=1_000_000_000.0,  # effectively unbounded; frac is the real cap
    safety_floor=0.0,           # off (regret-lever airbag handles wr crashes)
    emergency_ease_step=0.0,
    recency_half_life=0.0,
    deadband_sigma=0.0,         # off — react to every iter; matches old proportional
    window=20,
)


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
  # --- WDL regret bounds + stage gate ---
        initial_wdl_regret: float = -1.0,
        wdl_regret_min: float = 0.01,
        wdl_regret_max: float = 1.0,
        wdl_regret_stage_end: float = -1.0,
  # --- Regret lever ---
        regret_window: int = 20,
        regret_max_step: float = 0.01,
        regret_max_step_frac: float = 0.0,
        regret_safety_floor: float = 0.50,
        regret_emergency_ease_step: float = 0.01,
        regret_recency_half_life: float = 0.0,
        regret_deadband_sigma: float = 1.0,
  # --- Nodes lever (defaults chosen to match prior proportional behavior) ---
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
                else float(_NODES_DEFAULTS["max_step"]),
            max_step_frac=float(nodes_max_step_frac),
            safety_floor=float(nodes_safety_floor),
            emergency_ease_step=float(nodes_emergency_ease_step),
            recency_half_life=float(nodes_recency_half_life),
            deadband_sigma=float(nodes_deadband_sigma),
            window=int(nodes_window),
        )

  # --- Compatibility shims so callers keep using pid.wdl_regret / pid.nodes ---

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
        return int(round(float(self.nodes_lever.value)))

    @nodes.setter
    def nodes(self, value: int) -> None:
        self.nodes_lever.value = float(_clamp(int(value), self.min_nodes, self.max_nodes))

  # --- Live reload ---

    def refresh_live_params(self, config: dict) -> None:
        """Re-read live-reloadable knobs from a flat config dict.

        Construction-time fields (``min_nodes``, ``max_nodes``, lever
        ``window``, regret enable/gate flags) stay pinned. Everything else is
        a scalar attr that can be safely overwritten each iteration so
        live-reloaded yaml takes effect.

        Accepts both new ``sf_pid_regret_*`` / ``sf_pid_nodes_*`` keys and
        the legacy ``sf_pid_inverse_regret_*`` / ``sf_pid_node_step_cap``
        names, with new keys taking precedence.
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

        _refresh_lever_from_config(
            self.regret_lever, config,
            primary_prefix="sf_pid_regret_",
            legacy_prefix="sf_pid_inverse_regret_",
        )
        _refresh_lever_from_config(
            self.nodes_lever, config,
            primary_prefix="sf_pid_nodes_",
            legacy_prefix=None,
            legacy_keys={"max_step_frac": "sf_pid_node_step_cap"},
        )

  # --- Persistence ---

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

        regret_hist = state.get("regret_history") or state.get("inverse_history") or []
        _restore_history(self.regret_lever, regret_hist)
        nodes_hist = state.get("nodes_history") or []
        _restore_history(self.nodes_lever, nodes_hist)

        self._games_since_adjust = int(state.get("games_since_adjust", self._games_since_adjust))

        rgc = state.get("regret_stage_complete")
        if rgc is not None and self._regret_gate_enabled:
            self._regret_stage_complete = bool(rgc)

  # --- Observe ---

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
        regret_before = float(self.wdl_regret)
        raw_wr = float(wr)
        se = _observation_se(int(wins), int(draws), int(losses))

        if not force and self._games_since_adjust < int(self.min_games_between_adjust):
            return self._no_change_update(err)

  # Stage 1: regret lever. Frozen after the gate has fired (one-way) so a
  # stage-2 wr dip can't trip its airbag and double-mutate the controller.
  # Without a gate, regret keeps running every iter as before.
        regret_changed = False
        regret_frozen = self._regret_gate_enabled and self._regret_stage_complete
        if self._regret_enabled and not regret_frozen:
            regret_changed = _step_lever(
                self.regret_lever,
                target_wr=float(self.target),
                raw_wr=raw_wr,
                ema_wr=float(self.ema_winrate),
                ema_alpha=float(self.alpha),
                se=se,
            )
            if self._regret_gate_enabled:
                regret_end = _clamp(
                    self.wdl_regret_stage_end, self.wdl_regret_min, self.wdl_regret_max
                )
                if self.wdl_regret <= regret_end:
                    self._regret_stage_complete = True

  # Stage 2: nodes lever (only after regret has settled at floor).
  # One-way gate — once entered, stays entered for the life of this PID.
        nodes_changed = False
        if self._regret_stage_complete:
            nodes_changed = _step_lever(
                self.nodes_lever,
                target_wr=float(self.target),
                raw_wr=raw_wr,
                ema_wr=float(self.ema_winrate),
                ema_alpha=float(self.alpha),
                se=se,
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


_LEVER_LIVE_FIELDS = (
    "max_step",
    "max_step_frac",
    "safety_floor",
    "emergency_ease_step",
    "recency_half_life",
    "deadband_sigma",
)


def _refresh_lever_from_config(
    lever: _Lever,
    config: dict,
    *,
    primary_prefix: str,
    legacy_prefix: str | None = None,
    legacy_keys: dict[str, str] | None = None,
) -> None:
    """Update ``lever`` knobs from a flat config dict.

    Looks up ``f"{primary_prefix}{field}"`` first, falls back to
    ``f"{legacy_prefix}{field}"`` (if given) or the per-field override in
    ``legacy_keys`` (if given). Allows both ``sf_pid_regret_*`` (new) and
    ``sf_pid_inverse_regret_*`` (old) to coexist during transition.
    """
    legacy_keys = legacy_keys or {}
    for field_name in _LEVER_LIVE_FIELDS:
        primary_key = f"{primary_prefix}{field_name}"
        legacy_key = (
            f"{legacy_prefix}{field_name}" if legacy_prefix is not None else None
        )
        if primary_key in config:
            value = config[primary_key]
        elif legacy_key is not None and legacy_key in config:
            value = config[legacy_key]
        elif field_name in legacy_keys and legacy_keys[field_name] in config:
            value = config[legacy_keys[field_name]]
        else:
            continue
        setattr(lever, field_name, max(0.0, float(value)) if field_name in {
            "max_step", "max_step_frac", "emergency_ease_step",
            "recency_half_life", "deadband_sigma",
        } else float(value))


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


def pid_from_config(config: dict) -> DifficultyPID:
    """Construct a DifficultyPID from a flat config dict.

    Accepts new ``sf_pid_regret_*`` / ``sf_pid_nodes_*`` keys and falls back
    to legacy ``sf_pid_inverse_regret_*`` / ``sf_pid_node_step_cap`` so a
    yaml mid-rename still constructs the same controller.
    """
    def cfg(*keys, default):
        for k in keys:
            if k in config:
                return config[k]
        return default

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
        regret_window=int(cfg(
            "sf_pid_regret_window", "sf_pid_inverse_regret_window", default=20)),
        regret_max_step=float(cfg(
            "sf_pid_regret_max_step", "sf_pid_inverse_regret_max_step", default=0.01)),
        regret_max_step_frac=float(cfg(
            "sf_pid_regret_max_step_frac", "sf_pid_inverse_regret_max_step_frac", default=0.0)),
        regret_safety_floor=float(cfg(
            "sf_pid_regret_safety_floor", "sf_pid_inverse_regret_safety_floor", default=0.50)),
        regret_emergency_ease_step=float(cfg(
            "sf_pid_regret_emergency_ease_step",
            "sf_pid_inverse_regret_emergency_ease_step", default=0.01)),
        regret_recency_half_life=float(cfg(
            "sf_pid_regret_recency_half_life",
            "sf_pid_inverse_regret_recency_half_life", default=0.0)),
        regret_deadband_sigma=float(cfg(
            "sf_pid_regret_deadband_sigma",
            "sf_pid_inverse_regret_target_deadband_sigma", default=1.0)),
        nodes_window=int(config.get("sf_pid_nodes_window", 20)),
        nodes_max_step=float(config["sf_pid_nodes_max_step"])
            if "sf_pid_nodes_max_step" in config else None,
        nodes_max_step_frac=float(cfg(
            "sf_pid_nodes_max_step_frac", "sf_pid_node_step_cap", default=0.10)),
        nodes_safety_floor=float(config.get("sf_pid_nodes_safety_floor", 0.0)),
        nodes_emergency_ease_step=float(config.get("sf_pid_nodes_emergency_ease_step", 0.0)),
        nodes_recency_half_life=float(config.get("sf_pid_nodes_recency_half_life", 0.0)),
        nodes_deadband_sigma=float(config.get("sf_pid_nodes_deadband_sigma", 0.0)),
    )
