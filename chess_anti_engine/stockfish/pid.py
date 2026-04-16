from __future__ import annotations

from dataclasses import dataclass


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


@dataclass
class PIDUpdate:
    nodes_before: int
    nodes_after: int
    ema_winrate: float
    err: float
    adjusted: bool

    # Skill-level ladder
    skill_level: int = 0
    skill_changed: bool = False

    # Opponent-move corruption: probability the opponent plays a random legal move
    # instead of Stockfish's best move. (We still query Stockfish and record its
    # stats/targets; we just don't always execute its move.)
    random_move_prob_before: float = 0.0
    random_move_prob_after: float = 0.0
    random_move_prob_changed: bool = False

    # WDL regret limit: max allowed WDL-score drop when picking a suboptimal move.
    # High regret = easy (any top-k move is fine). Low regret = hard (must be
    # nearly best). Negative means disabled (no regret filter).
    wdl_regret_before: float = -1.0
    wdl_regret_after: float = -1.0
    wdl_regret_changed: bool = False


class DifficultyPID:
    """Adaptive difficulty controller for Stockfish node count.

    Tracks an exponential moving average (EMA) of the network win rate and applies a
    PID controller to adjust Stockfish nodes to maintain a target win rate.

    Difficulty stages (each gates the next):
      1. rmp stage: lower random_move_prob to stage_end
      2. regret stage: lower wdl_regret to stage_end  (only active after rmp stage)
      3. nodes stage: increase SF nodes                (only active after regret stage)

    Spec-aligned behaviors:
    - EMA win rate tracking (alpha default 0.03)
    - dead zone (±4% around target): no adjustment within the band
    - rate limiter: <=10% node change per adjustment
    - anti-windup: clamp integral term; also stop integrating when saturated
    - adjustment period: require a minimum number of games between adjustments

    Notes:
    - We assume the network is White and Stockfish is Black, so a "network win" is a
      game result of 1-0.
    """

    def __init__(
        self,
        *,
        initial_nodes: int,
        target_winrate: float = 0.60,
        ema_alpha: float = 0.03,
        deadzone: float = 0.04,
        rate_limit: float = 0.10,
        min_games_between_adjust: int = 30,
        kp: float = 1.0,
        ki: float = 0.10,
        kd: float = 0.0,
        integral_clamp: float = 1.0,
        min_nodes: int = 1_000,
        max_nodes: int = 1_000_000,
        # Opponent random-move probability.
        # Higher => easier opponent (more random moves). Lower => closer to pure SF.
        initial_random_move_prob: float = 0.0,
        random_move_prob_min: float = 0.0,
        random_move_prob_max: float = 1.0,
        # "Random-first" gating: while random_move_prob is above this threshold,
        # we ONLY adjust random_move_prob (and keep regret/nodes/skill fixed). This
        # lets training bootstrap against a weak opponent before ramping SF strength.
        random_move_stage_end: float = 0.5,
        # Maximum absolute change to random_move_prob per adjustment step.
        # Prevents large jumps (e.g. 1.0→0.90) when the PID fires with a big error.
        # Default 0.01 = 1% per step.
        max_rand_step: float = 0.01,
        # If set, random-first gating re-enters only when random_move_prob rises to
        # this level or higher. This provides hysteresis: temporary dips/recoveries
        # around random_move_stage_end do not rapidly freeze/unfreeze regret updates.
        random_move_stage_reenter: float | None = None,
        # WDL regret: maximum allowed WDL-score drop for suboptimal moves.
        # High => easy (any top-k move OK), low => hard (near-best only).
        # Negative initial value disables the regret stage entirely.
        initial_wdl_regret: float = -1.0,
        wdl_regret_min: float = 0.01,
        wdl_regret_max: float = 1.0,
        # Regret stage gating: nodes unlock only when regret drops to this level.
        # Negative disables the gate (nodes unlock as soon as rmp stage completes).
        wdl_regret_stage_end: float = -1.0,
        max_regret_step: float = 0.001,
        max_regret_ease_step: float | None = None,  # Ease rate; defaults to 5x tighten
        wdl_regret_stage_reenter: float | None = None,
        # Skill-level ladder: PID manages both node count and SF Skill Level.
        # When nodes climb past skill_promote_nodes the difficulty tier increases;
        # when they fall below skill_demote_nodes the tier decreases (safety only).
        # On each transition nodes are reset to give hysteresis.
        initial_skill_level: int = 1,
        skill_min: int = 0,
        skill_max: int = 20,
        skill_promote_nodes: int = 200,
        skill_demote_nodes: int = 100,
        skill_nodes_on_promote: int = 100,
        skill_nodes_on_demote: int = 150,
    ):
        init = int(initial_nodes)
        self.nodes = _clamp(init, int(min_nodes), int(max_nodes))
        self.target = float(target_winrate)
        self.alpha = float(ema_alpha)
        self.deadzone = float(deadzone)
        self.rate_limit = float(rate_limit)
        self.min_games_between_adjust = int(min_games_between_adjust)

        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_clamp = float(integral_clamp)

        self.min_nodes = int(min_nodes)
        self.max_nodes = int(max_nodes)

        # Opponent random-move probability
        self.random_move_prob = float(initial_random_move_prob)
        self.random_move_prob_min = float(random_move_prob_min)
        self.random_move_prob_max = float(random_move_prob_max)
        self.random_move_stage_end = float(random_move_stage_end)
        self.max_rand_step = float(max_rand_step)
        stage_end = _clamp(float(self.random_move_stage_end), self.random_move_prob_min, self.random_move_prob_max)
        reenter_default = min(self.random_move_prob_max, stage_end + 0.05)
        reenter_val = reenter_default if random_move_stage_reenter is None else float(random_move_stage_reenter)
        self.random_move_stage_reenter = _clamp(reenter_val, stage_end, self.random_move_prob_max)
        self._random_stage_complete = float(self.random_move_prob) <= float(stage_end)

        # WDL regret
        self._regret_enabled = float(initial_wdl_regret) >= 0.0
        self._regret_gate_enabled = self._regret_enabled and float(wdl_regret_stage_end) >= 0.0
        if self._regret_enabled:
            self.wdl_regret = _clamp(float(initial_wdl_regret), float(wdl_regret_min), float(wdl_regret_max))
            self.wdl_regret_min = float(wdl_regret_min)
            self.wdl_regret_max = float(wdl_regret_max)
            self.max_regret_step = float(max_regret_step)
            self.max_regret_ease_step = float(max_regret_ease_step) if max_regret_ease_step is not None else float(max_regret_step) * 5
            if self._regret_gate_enabled:
                self.wdl_regret_stage_end = _clamp(
                    float(wdl_regret_stage_end), float(wdl_regret_min), float(wdl_regret_max)
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
        else:
            self.wdl_regret = float(initial_wdl_regret)
            self.wdl_regret_min = float(wdl_regret_min)
            self.wdl_regret_max = float(wdl_regret_max)
            self.wdl_regret_stage_end = float(wdl_regret_stage_end)
            self.max_regret_step = float(max_regret_step)
            self.max_regret_ease_step = float(max_regret_ease_step) if max_regret_ease_step is not None else float(max_regret_step) * 5
            self.wdl_regret_stage_reenter = float(wdl_regret_stage_end)
            self._regret_stage_complete = True  # disabled = always complete

        # Skill level ladder
        self.skill_level = _clamp(int(initial_skill_level), int(skill_min), int(skill_max))
        self.skill_min = int(skill_min)
        self.skill_max = int(skill_max)
        self.skill_promote_nodes = int(skill_promote_nodes)
        self.skill_demote_nodes = int(skill_demote_nodes)
        self.skill_nodes_on_promote = int(skill_nodes_on_promote)
        self.skill_nodes_on_demote = int(skill_nodes_on_demote)

        # Start EMA at target so the first few noisy batches don't misfire the controller.
        self.ema_winrate: float = float(target_winrate)
        self._integral = 0.0
        self._prev_err: float | None = None
        self._games_since_adjust = 0

    def state_dict(self) -> dict:
        """Serialize controller state for checkpointing.

        This intentionally captures only the minimal internal state needed to
        resume without discontinuities.
        """
        return {
            "nodes": int(self.nodes),
            "skill_level": int(self.skill_level),
            "random_move_prob": float(self.random_move_prob),
            "wdl_regret": float(self.wdl_regret),
            "ema_winrate": float(self.ema_winrate),
            "integral": float(self._integral),
            "prev_err": None if self._prev_err is None else float(self._prev_err),
            "games_since_adjust": int(self._games_since_adjust),
            "random_stage_complete": bool(self._random_stage_complete),
            "regret_stage_complete": bool(self._regret_stage_complete),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore controller state from `state_dict()`."""
        if not isinstance(state, dict):
            return

        # Clamp to configured bounds.
        self.nodes = _clamp(int(state.get("nodes", self.nodes)), self.min_nodes, self.max_nodes)
        self.skill_level = _clamp(int(state.get("skill_level", self.skill_level)), self.skill_min, self.skill_max)
        self.random_move_prob = _clamp(
            float(state.get("random_move_prob", self.random_move_prob)),
            self.random_move_prob_min, self.random_move_prob_max,
        )

        if self._regret_enabled:
            self.wdl_regret = _clamp(
                float(state.get("wdl_regret", self.wdl_regret)),
                self.wdl_regret_min, self.wdl_regret_max,
            )

        ew = float(state.get("ema_winrate", self.ema_winrate))
        self.ema_winrate = float(ew)

        self._integral = _clamp(
            float(state.get("integral", self._integral)),
            -self.integral_clamp, self.integral_clamp,
        )

        pe = state.get("prev_err", self._prev_err)
        self._prev_err = None if pe is None else float(pe)

        self._games_since_adjust = int(state.get("games_since_adjust", self._games_since_adjust))

        rsc = state.get("random_stage_complete")
        if rsc is not None:
            self._random_stage_complete = bool(rsc)

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
            skill_level=int(self.skill_level),
            skill_changed=False,
            random_move_prob_before=float(self.random_move_prob),
            random_move_prob_after=float(self.random_move_prob),
            random_move_prob_changed=False,
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

        # Wait for enough games before changing difficulty,
        # unless force=True (iteration boundary).
        if not force and self._games_since_adjust < int(self.min_games_between_adjust):
            return self._no_change_update(err)

        # Deadzone: do not adjust within +/- deadzone of target.
        if abs(err) <= float(self.deadzone):
            self._games_since_adjust = 0
            self._prev_err = err
            return self._no_change_update(err)

        # Derivative term (per adjustment period).
        derr = 0.0 if self._prev_err is None else (err - float(self._prev_err))

        # Anti-windup: if we're already saturated and error pushes further into saturation, stop integrating.
        # Check both node limits and random-move-prob limits (rand=1.0 means "already at max easy").
        saturated_low = self.nodes <= self.min_nodes + 1
        saturated_high = self.nodes >= self.max_nodes - 1
        saturated_rand_max = self.random_move_prob >= self.random_move_prob_max - 1e-6
        saturated_rand_min = self.random_move_prob <= self.random_move_prob_min + 1e-6
        saturated_regret_max = self._regret_enabled and self.wdl_regret >= self.wdl_regret_max - 1e-6
        saturated_regret_min = self._regret_enabled and self.wdl_regret <= self.wdl_regret_min + 1e-6
        if not (
            (saturated_low and err < 0.0)
            or (saturated_high and err > 0.0)
            or (saturated_rand_max and err < 0.0)
            or (saturated_rand_min and err > 0.0)
            or (saturated_regret_max and err < 0.0)
            or (saturated_regret_min and err > 0.0)
        ):
            self._integral = _clamp(self._integral + err, -self.integral_clamp, self.integral_clamp)

        u = self.kp * err + self.ki * self._integral + self.kd * derr

        # Rate limit is interpreted as a fractional multiplicative change.
        u = _clamp(u, -self.rate_limit, self.rate_limit)

        # --- Stage 1: always adjust opponent random-move probability ---
        rand_before = float(self.random_move_prob)
        rand_delta = _clamp(-u, -self.max_rand_step, self.max_rand_step)
        rand_after = _clamp(rand_before + rand_delta, self.random_move_prob_min, self.random_move_prob_max)
        self.random_move_prob = float(rand_after)

        # Random-first gating with hysteresis:
        # - enable regret when random_move_prob drops to stage_end or below
        # - re-freeze only after substantial regression to stage_reenter or above
        # When rmp is pinned (min == max), the gate is always complete — skip
        # hysteresis to avoid flapping when stage_end == stage_reenter.
        if self.random_move_prob_min >= self.random_move_prob_max - 1e-9:
            self._random_stage_complete = True
        else:
            stage_end = _clamp(self.random_move_stage_end, self.random_move_prob_min, self.random_move_prob_max)
            stage_reenter = _clamp(self.random_move_stage_reenter, stage_end, self.random_move_prob_max)
            if self._random_stage_complete:
                if self.random_move_prob >= stage_reenter:
                    self._random_stage_complete = False
            else:
                if self.random_move_prob <= stage_end:
                    self._random_stage_complete = True

        rand_changed = abs(rand_after - rand_before) > 1e-12

        # --- Stage 2: adjust WDL regret (only after rmp stage complete) ---
        regret_before = float(self.wdl_regret)
        regret_changed = False
        allow_regret = bool(self._random_stage_complete) and self._regret_enabled

        if allow_regret:
            # Winning too much (u>0) → make harder → decrease regret (tighter).
            # Losing too much (u<0) → make easier → increase regret (looser).
            # Asymmetric: tighten slow, ease fast. Model improves slowly
            # but can collapse quickly — PID should mirror that.
            ease_step = self.max_regret_ease_step
            tighten_step = self.max_regret_step
            regret_delta = _clamp(-u, -tighten_step, ease_step)
            regret_after = _clamp(regret_before + regret_delta, self.wdl_regret_min, self.wdl_regret_max)
            self.wdl_regret = float(regret_after)
            regret_changed = abs(regret_after - regret_before) > 1e-12

            # Regret stage gating with hysteresis:
            if self._regret_gate_enabled:
                regret_end = _clamp(self.wdl_regret_stage_end, self.wdl_regret_min, self.wdl_regret_max)
                regret_reenter = _clamp(self.wdl_regret_stage_reenter, regret_end, self.wdl_regret_max)
                if self._regret_stage_complete:
                    if self.wdl_regret >= regret_reenter:
                        self._regret_stage_complete = False
                else:
                    if self.wdl_regret <= regret_end:
                        self._regret_stage_complete = True

        # --- Stage 3: adjust nodes (only after both rmp + regret stages complete) ---
        allow_nodes = bool(self._random_stage_complete) and bool(self._regret_stage_complete)

        nodes_after = int(self.nodes)
        skill_before = int(self.skill_level)
        skill_changed = False

        if allow_nodes:
            # Apply multiplicative update.
            new_nodes = int(round(float(self.nodes) * (1.0 + u)))
            self.nodes = _clamp(new_nodes, self.min_nodes, self.max_nodes)

            # Skill-level ladder:
            # - only promote when the controller is actively making the opponent harder (u > 0)
            # - only demote when the controller is actively making the opponent easier (u < 0)
            # This avoids pathological ratcheting where a still-high node count can trigger
            # promotion even while the model is underperforming.
            #
            # Demotion threshold is inclusive (<=) so it still works when min_nodes equals
            # skill_demote_nodes.
            if self.nodes >= self.skill_promote_nodes and self.skill_level < self.skill_max:
                if u > 0.0:
                    self.skill_level += 1
                    self.nodes = _clamp(self.skill_nodes_on_promote, self.min_nodes, self.max_nodes)
                    self._integral = 0.0  # fresh start at new difficulty tier
            elif self.nodes <= self.skill_demote_nodes and self.skill_level > self.skill_min:
                if u < 0.0:
                    self.skill_level -= 1
                    self.nodes = _clamp(self.skill_nodes_on_demote, self.min_nodes, self.max_nodes)
                    self._integral = 0.0  # fresh start at easier tier

            nodes_after = int(self.nodes)
            skill_changed = self.skill_level != skill_before

        self._games_since_adjust = 0
        self._prev_err = err

        adjusted = (nodes_after != nodes_before) or bool(skill_changed) or bool(rand_changed) or bool(regret_changed)

        return PIDUpdate(
            nodes_before=nodes_before,
            nodes_after=int(nodes_after),
            ema_winrate=float(self.ema_winrate),
            err=err,
            adjusted=bool(adjusted),
            skill_level=int(self.skill_level),
            skill_changed=bool(skill_changed),
            random_move_prob_before=float(rand_before),
            random_move_prob_after=float(rand_after),
            random_move_prob_changed=bool(rand_changed),
            wdl_regret_before=float(regret_before),
            wdl_regret_after=float(self.wdl_regret),
            wdl_regret_changed=bool(regret_changed),
        )


def pid_from_config(config: dict) -> DifficultyPID:
    """Construct a DifficultyPID from a flat config dict.

    Works with both the ``vars(args)`` namespace produced by ``run.py`` and the
    per-trial ``config`` dict consumed by ``trainable.py``.  All keys fall back
    to the same defaults used by the ``DifficultyPID`` constructor or the
    argparse definitions in ``run.py``.
    """
    return DifficultyPID(
        initial_nodes=int(config.get("sf_nodes", 500)),
        target_winrate=float(config.get("sf_pid_target_winrate", 0.60)),
        ema_alpha=float(config.get("sf_pid_ema_alpha", 0.03)),
        deadzone=float(config.get("sf_pid_deadzone", 0.0)),
        rate_limit=float(config.get("sf_pid_rate_limit", 0)),
        min_games_between_adjust=int(config.get("sf_pid_min_games_between_adjust", 10)),
        kp=float(config.get("sf_pid_kp", 1.0)),
        ki=float(config.get("sf_pid_ki", 0.1)),
        kd=float(config.get("sf_pid_kd", 0.05)),
        integral_clamp=float(config.get("sf_pid_integral_clamp", 10.0)),
        min_nodes=int(config.get("sf_pid_min_nodes", 100)),
        max_nodes=int(config.get("sf_pid_max_nodes", 50000)),
        initial_skill_level=int(config.get("sf_pid_initial_skill_level", 0)),
        skill_min=int(config.get("sf_pid_skill_min", 0)),
        skill_max=int(config.get("sf_pid_skill_max", 20)),
        skill_promote_nodes=int(config.get("sf_pid_skill_promote_nodes", 200)),
        skill_demote_nodes=int(config.get("sf_pid_skill_demote_nodes", 100)),
        skill_nodes_on_promote=int(config.get("sf_pid_skill_nodes_on_promote", 100)),
        skill_nodes_on_demote=int(config.get("sf_pid_skill_nodes_on_demote", 150)),
        initial_random_move_prob=float(config.get("sf_pid_random_move_prob_start", 1.0)),
        random_move_prob_min=float(config.get("sf_pid_random_move_prob_min", 0.0)),
        random_move_prob_max=float(config.get("sf_pid_random_move_prob_max", 1.0)),
        max_rand_step=float(config.get("sf_pid_max_rand_step", 0.01)),
        random_move_stage_end=float(config.get("sf_pid_random_move_stage_end", 0.5)),
        initial_wdl_regret=float(config.get("sf_pid_wdl_regret_start", -1.0)),
        wdl_regret_min=float(config.get("sf_pid_wdl_regret_min", 0.01)),
        wdl_regret_max=float(config.get("sf_pid_wdl_regret_max", 1.0)),
        wdl_regret_stage_end=float(config.get("sf_pid_wdl_regret_stage_end", -1.0)),
        max_regret_step=float(config.get("sf_pid_max_regret_step", 0.001)),
    )
