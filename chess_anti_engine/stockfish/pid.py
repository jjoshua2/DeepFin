from __future__ import annotations

from dataclasses import dataclass


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


class DifficultyPID:
    """Adaptive difficulty controller for Stockfish node count.

    Tracks an exponential moving average (EMA) of the network win rate and applies a
    PID controller to adjust Stockfish nodes to maintain a target win rate.

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
        target_winrate: float = 0.53,
        ema_alpha: float = 0.03,
        deadzone: float = 0.04,
        rate_limit: float = 0.10,
        min_games_between_adjust: int = 30,
        kp: float = 1.5,
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
        # we ONLY adjust random_move_prob (and keep nodes/skill fixed). This lets
        # training bootstrap against a weak opponent before ramping SF strength.
        random_move_stage_end: float = 0.5,
        # Maximum absolute change to random_move_prob per adjustment step.
        # Prevents large jumps (e.g. 1.0→0.90) when the PID fires with a big error.
        # Default 0.01 = 1% per step.
        max_rand_step: float = 0.01,
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
        init = max(int(min_nodes), min(int(max_nodes), init))
        self.nodes = int(init)
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

        # Skill level ladder
        self.skill_level = max(int(skill_min), min(int(skill_max), int(initial_skill_level)))
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

    def observe(self, *, wins: int, draws: int, losses: int) -> PIDUpdate:
        games = int(wins) + int(draws) + int(losses)
        if games <= 0:
            return PIDUpdate(
                nodes_before=int(self.nodes),
                nodes_after=int(self.nodes),
                ema_winrate=float(self.ema_winrate or self.target),
                err=0.0,
                adjusted=False,
                skill_level=int(self.skill_level),
                skill_changed=False,
                random_move_prob_before=float(self.random_move_prob),
                random_move_prob_after=float(self.random_move_prob),
                random_move_prob_changed=False,
            )

        wr = (float(wins) + 0.5 * float(draws)) / float(games)
        self.ema_winrate = (1.0 - self.alpha) * float(self.ema_winrate) + self.alpha * float(wr)

        self._games_since_adjust += games

        err = float(self.ema_winrate) - float(self.target)
        nodes_before = int(self.nodes)

        # Wait for enough games before changing difficulty.
        if self._games_since_adjust < int(self.min_games_between_adjust):
            return PIDUpdate(
                nodes_before=nodes_before,
                nodes_after=int(self.nodes),
                ema_winrate=float(self.ema_winrate),
                err=err,
                adjusted=False,
                skill_level=int(self.skill_level),
                skill_changed=False,
                random_move_prob_before=float(self.random_move_prob),
                random_move_prob_after=float(self.random_move_prob),
                random_move_prob_changed=False,
            )

        # Deadzone: do not adjust within +/- deadzone of target.
        if abs(err) <= float(self.deadzone):
            self._games_since_adjust = 0
            self._prev_err = err
            # Do not integrate in deadzone.
            return PIDUpdate(
                nodes_before=nodes_before,
                nodes_after=int(self.nodes),
                ema_winrate=float(self.ema_winrate),
                err=err,
                adjusted=False,
                skill_level=int(self.skill_level),
                skill_changed=False,
                random_move_prob_before=float(self.random_move_prob),
                random_move_prob_after=float(self.random_move_prob),
                random_move_prob_changed=False,
            )

        # Derivative term (per adjustment period).
        derr = 0.0 if self._prev_err is None else (err - float(self._prev_err))

        # Anti-windup: if we're already saturated and error pushes further into saturation, stop integrating.
        # Check both node limits and random-move-prob limits (rand=1.0 means "already at max easy").
        saturated_low = self.nodes <= self.min_nodes + 1
        saturated_high = self.nodes >= self.max_nodes - 1
        saturated_rand_max = self.random_move_prob >= self.random_move_prob_max - 1e-6
        saturated_rand_min = self.random_move_prob <= self.random_move_prob_min + 1e-6
        if not (
            (saturated_low and err < 0.0)
            or (saturated_high and err > 0.0)
            or (saturated_rand_max and err < 0.0)
            or (saturated_rand_min and err > 0.0)
        ):
            self._integral += err
            if self._integral > self.integral_clamp:
                self._integral = self.integral_clamp
            elif self._integral < -self.integral_clamp:
                self._integral = -self.integral_clamp

        u = self.kp * err + self.ki * self._integral + self.kd * derr

        # Rate limit is interpreted as a fractional multiplicative change.
        u = max(-self.rate_limit, min(self.rate_limit, float(u)))

        # Always adjust opponent random-move probability using the same control signal.
        rand_before = float(self.random_move_prob)
        rand_delta = rand_before * (-float(u))
        rand_delta = max(-self.max_rand_step, min(self.max_rand_step, rand_delta))
        rand_after = rand_before + rand_delta
        rand_after = max(self.random_move_prob_min, min(self.random_move_prob_max, float(rand_after)))
        self.random_move_prob = float(rand_after)
        rand_changed = abs(rand_after - rand_before) > 1e-12

        # Random-first gating: while random_move_prob is still high, keep nodes/skill fixed.
        stage_end = max(self.random_move_prob_min, min(self.random_move_prob_max, float(self.random_move_stage_end)))
        allow_nodes = float(self.random_move_prob) <= float(stage_end)

        nodes_after = int(self.nodes)
        skill_before = int(self.skill_level)
        skill_changed = False

        if allow_nodes:
            # Apply multiplicative update.
            new_nodes = int(round(float(self.nodes) * (1.0 + u)))
            new_nodes = max(self.min_nodes, min(self.max_nodes, new_nodes))
            self.nodes = int(new_nodes)

            # Skill-level ladder: promote when nodes exceed the promote threshold;
            # demote (rarely) when they would floor below the demote threshold.
            if self.nodes >= self.skill_promote_nodes and self.skill_level < self.skill_max:
                self.skill_level += 1
                self.nodes = max(self.min_nodes, min(self.max_nodes, self.skill_nodes_on_promote))
                self._integral = 0.0  # fresh start at new difficulty tier
            elif self.nodes < self.skill_demote_nodes and self.skill_level > self.skill_min:
                self.skill_level -= 1
                self.nodes = max(self.min_nodes, min(self.max_nodes, self.skill_nodes_on_demote))
                self._integral = 0.0  # fresh start at easier tier

            nodes_after = int(self.nodes)
            skill_changed = self.skill_level != skill_before

        self._games_since_adjust = 0
        self._prev_err = err

        adjusted = (nodes_after != nodes_before) or bool(skill_changed) or bool(rand_changed)

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
        )
