"""Anchored promotion gate for the RL loop.

WHAT THIS REPLACES, AND WHY THE OLD ONE COULD NOT WORK
------------------------------------------------------
The historical in-loop gate (``gate_games`` / ``gate_threshold`` /
``gate_mcts_sims``, ``trainable_phases._run_net_gating``, removed by the same
change that added this module) played ``gate_games`` fresh games of the
post-training net **against Stockfish** at ``gate_mcts_sims: 1``, and restored
the pre-training weights whenever ``(W + 0.5 D) / N < 0.50``.

It had four independent defects, only two of which the config comment named:

1. **Search mismatch.** 1 simulation is raw policy argmax; production selfplay
   runs 256 sims on 25% of plies and 32 on the rest. The gated quantity was
   not the quantity the loop produces.
2. **Wrong reference.** It scored against Stockfish, so it could only measure
   the net against a *moving* opponent, not against its own predecessor.
3. **A controller pinned the gate's own statistic.** The PID exists precisely
   to drive the net's score against handicapped SF to a setpoint. Any statistic
   measured against that opponent is therefore held at the setpoint no matter
   how the net changes, so it carries no information about the training step --
   *whatever the setpoint is*. Where the threshold happens to EQUAL it (today:
   ``sf_pid_target_winrate: 0.50`` and ``gate_threshold: 0.50``) the verdict
   degenerates to a coin flip on the controller's tracking error; where it does
   not, the gate degenerates to a rubber stamp or a permanent reject. Equality
   only picks which. **Historical note:** ``sf_pid_target_winrate`` did not
   exist until 2026-04-15, so this was NOT one of the defects that froze Run 3
   in 2026-03 -- (1) alone explains that -- but it is a property of the config
   the gate would be re-enabled into today, and it survives anyone re-tuning
   the setpoint.
4. **Rejection threw away the optimizer.** It restored ``model.state_dict()``
   and nothing else -- step counter, optimizer moments and LR schedule marched
   on. Repeated rejection therefore did not "hold the model", it walked the
   optimizer forward against a frozen parameter vector.

Defects 2 and 3 are one defect seen twice: **the ruler moved with the model.**
The PID exists to keep the net's score against handicapped Stockfish at its
setpoint, so any statistic measured against that opponent is held at the
setpoint no matter how the net changes. Scoring a net against a ruler that
tracks it can never show improvement -- which is the same shape as the OTHER
ruler this loop was using, below.

WHY THIS IS NOT A LOSS-BASED GATE
---------------------------------
The obvious cheap gate is the deterministic holdout that already runs every
iteration for free (noise floor 0.00736 nats, ~7x tighter than the old
resampling ruler). It would be wrong, and audit row L16 now proves it:

    ``policy_target`` is SELF-GENERATED MCTS visit counts, so a frozen holdout
    is a DECAYING RULER BY CONSTRUCTION. As the policy improves it necessarily
    moves away from a visit distribution frozen many iterations ago, and
    ``test_policy_loss`` rises on a net that is getting better.

Measured paired, per-row, game-clustered bootstrap (2026-07-28): ckpt157 ->
ckpt192 scores **+0.0392 WORSE** on the frozen holdout and **-0.0129 BETTER**
on fresh unseen post-dated targets; ckpt192 -> ckpt215 is **+0.0045** frozen
and **-0.0357 BETTER** fresh, improving 4.2x faster per iteration than the
first span. A gate on that ruler rejects every model forever -- the 2026-03
freeze reached by a second, independent road. ``test_wdl_loss`` alone is
exempt, because the game result does not age; it is also the flattest and
least sensitive leg, so it would buy a gate almost nothing.

Note the same defect already has a live casualty: ``_update_best_model`` ranks
on holdout loss, which is why ``best_loss`` has not moved in 50+ iterations.
This gate deliberately shares no state with best-model selection -- it must not
be built on a ruler that is known to decay.

A head-to-head between two nets PLAYING GAMES has no aging problem at all: the
comparison is regenerated from scratch every iteration and nothing about it is
frozen. That, and not cost, is the first reason this gate plays games rather
than reading a loss.

WHAT THIS DOES INSTEAD
----------------------
Three changes of kind, not of tuning:

* **The reference is the previous published model, not Stockfish.**
* **The games are free.** Up to ``distributed_prev_model_max_fraction`` of every
  iteration's ingest is already played by the *previous* published net against
  the *same* handicapped Stockfish as the current one. That is an anchored A/B
  that the loop pays for anyway. This module reads it; it never plays a game.
  Because both sides face the same opponent *within one iteration*, the PID's
  tracking cancels in the difference -- the moving-ruler defect above does not
  survive the subtraction. What survives is a one-iteration difficulty lag,
  documented at the bottom of this docstring.
* **The action is on the PUBLISHED model, not on the training weights.**
  A demote holds the selfplay fleet on the last promoted export while training
  continues uninterrupted. A false demote costs a little data freshness. It
  cannot freeze learning, which is the failure mode that killed the old gate.

WHAT IT CAN AND CANNOT RESOLVE -- READ THIS BEFORE TRUSTING A VERDICT
---------------------------------------------------------------------
These numbers are MEASURED, not derived from the config. Binning 418 processed
shard ``.zattrs`` by ``generated_at_unix`` against ``progress.csv`` timestamps
over live iterations 164-219 (every window contains exactly two shas, which is
what the publish cadence predicts):

    n_cur ~= **197** games/iteration, n_prev ~= **38** (prev share **16.3%**)
    per-iteration anchored delta: mean **-4.3 Elo**, sd **45.6 Elo** (n=53)

Note ``distributed_prev_model_max_fraction: 0.60`` is a CEILING that never
binds -- ``distributed_stale_games`` is 0 on every recent iteration, and a
binding cap would have to discard prev shards into ``stale_*``. Sizing this
gate off 0.60 gives 95/143 and a standard error of 31 Elo; the realized 197/38
gives **41.5 Elo** predicted from binomial noise alone and **45.6 Elo**
observed. Always take the realized split from shard data, never the cap.

The loop's measured improvement is +0.21 Elo per 1000 optimizer steps, roughly
**0.02 Elo per iteration**. Against a per-iteration se of 45.6 Elo, detecting
that at 80% power needs ~4e7 iterations. It is therefore *arithmetically
impossible* to gate on per-iteration improvement, here or with any affordable
arena; a 200-game sims-32 arena, the entire ~30 min/day GPU budget, carries se
~24.9 Elo -- worse than one free iteration. **The correction makes the negative
STRONGER, not weaker:** the noise is 47% larger than first estimated, so "no
gate can ratchet at 0.02 Elo/iteration" holds a fortiori.

So this gate is a **regression alarm and a publish brake**, not a ratchet. At
the default 24-iteration window (~4.5 h) the window se is **9.3 Elo**, which
catches a sustained **-50 Elo/iteration** break at 93% power with the demote
line at -25, and a **-200** break instantly. It does NOT catch the 2026-06
warm-start LR crash (-494 Elo over 74 iterations = -6.7 Elo/iteration) at any
window this design can reach -- power there is ~0%. That class of slow bleed
needs the cumulative vs-frozen-anchor series in
``scripts/daily_gate_ratchet.sh``, which is why this gate does not replace it.

THE KNOWN BIAS, AND WHY THE DEFAULT MODE IS ``shadow``
------------------------------------------------------
Model and difficulty are published in ONE manifest, so a game tagged with the
previous model's sha was also played at the previous iteration's
``wdl_regret`` / ``sf_nodes``. While the PID is moving difficulty in one
direction, the anchored delta carries a systematic offset of unknown size that
is plausibly the same order as the effects above. Nothing has measured it,
because the per-sha split did not exist until this change.

That offset has since been measured offline from the same 418 shards: mean
**-4.3 Elo**, with cur/prev draw rates 0.5147 vs 0.5149 -- so it is small and
shows no length-bias signature. ``gate_mode`` still defaults to ``shadow``, but
the shadow window's job is now narrower and falsifiable: confirm that the
IN-LOOP instrument reproduces that offline reconstruction, which it can fail.

Note what the metrics must therefore report. A rolling-window mean is not a
readable series -- consecutive rows share ~95% of their samples, so the sd of
the reported column understates the per-iteration sd by ~10x, and any rule
keyed to it cannot fail. ``gate_metrics`` emits ``gate_sample_*`` (THIS
iteration's independent anchored delta) alongside the window aggregates, and
the sample columns are the ones a decision rule must be written against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

# Decision codes. Deliberately NOT booleans: the whole point of this module is
# that "did not run" must be distinguishable from "passed" in the metrics, and
# a bool cannot carry that. See ``gate_metrics``.
DECISION_NOT_RUN = -1
DECISION_DEMOTE = 0
DECISION_PROMOTE = 1

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ENFORCE = "enforce"
_MODE_CODES = {MODE_OFF: 0, MODE_SHADOW: 1, MODE_ENFORCE: 2}

# Reason codes are stable integers so a dashboard can chart them; the string
# form travels alongside for humans.
REASONS = (
    "disabled",          # 0  gate_mode: off
    "insufficient_iters",  # 1  window not full yet
    "insufficient_games",  # 2  too few anchored games in the window
    "degenerate_variance",  # 3  zero spread across iterations -- refuse to divide
    "promote_no_regression",  # 4  upper bound not below the demote line
    "demote_regression",  # 5  upper bound below the demote line
    "shadow_would_demote",  # 6  demote condition met, but mode is shadow
    "hold_expired",      # 7  demote condition still met past gate_max_hold_iters
)
_REASON_CODES = {name: i for i, name in enumerate(REASONS)}

# 400 / ln(10) / 0.25 -- d(Elo)/d(score) at score 0.5. The anchored delta is a
# DIFFERENCE of two scores both sitting near 0.5 by construction (the PID pins
# them there), so the local linearisation is the honest conversion; running
# each score through the logistic separately would inject the PID's setpoint
# error into the delta.
ELO_PER_SCORE_AT_HALF = 400.0 / (math.log(10.0) * 0.25)

# One-sided normal quantiles. A t-table would be more correct at small window
# sizes; ``_t_quantile`` widens these for small df instead of adding a scipy
# dependency to the training loop.
_Z_ONE_SIDED = {0.10: 1.2816, 0.05: 1.6449, 0.025: 1.9600, 0.01: 2.3263, 0.005: 2.5758}


def elo_from_score_delta(delta: float) -> float:
    """Local-linear score-delta -> Elo, valid for small deltas around 0.5."""
    return float(delta) * ELO_PER_SCORE_AT_HALF


def _t_quantile(alpha: float, df: int) -> float:
    """One-sided t quantile, table-free.

    Uses the normal quantile inflated by the Cornish-Fisher correction
    ``z + (z**3 + z) / (4 df)``, which is within ~1% of the exact t for
    df >= 8 and conservative (wider) below it. The gate's minimum window is 8
    iterations, so this is the regime it runs in.
    """
    z = _Z_ONE_SIDED.get(round(float(alpha), 4))
    if z is None:
        raise ValueError(
            f"unsupported alpha {alpha!r}; supported: {sorted(_Z_ONE_SIDED)}"
        )
    if df <= 0:
        return float("inf")
    return float(z + (z ** 3 + z) / (4.0 * df))


@dataclass(frozen=True)
class AnchoredSample:
    """One iteration's anchored A/B: current vs previous published model.

    Both sides are vs-Stockfish (curriculum) games only. Selfplay games are
    model-vs-itself and carry no information about relative strength, so they
    are excluded upstream -- ``matching_w/d/l`` already excludes them.
    """

    iteration: int
    cur_w: int = 0
    cur_d: int = 0
    cur_l: int = 0
    prev_w: int = 0
    prev_d: int = 0
    prev_l: int = 0

    @property
    def cur_games(self) -> int:
        return int(self.cur_w) + int(self.cur_d) + int(self.cur_l)

    @property
    def prev_games(self) -> int:
        return int(self.prev_w) + int(self.prev_d) + int(self.prev_l)

    @property
    def cur_score(self) -> float:
        n = self.cur_games
        return (self.cur_w + 0.5 * self.cur_d) / n if n else float("nan")

    @property
    def prev_score(self) -> float:
        n = self.prev_games
        return (self.prev_w + 0.5 * self.prev_d) / n if n else float("nan")

    @property
    def delta(self) -> float:
        """Current-model score minus previous-model score, this iteration."""
        return self.cur_score - self.prev_score

    def usable(self, *, min_games_per_side: int) -> bool:
        m = max(1, int(min_games_per_side))
        return self.cur_games >= m and self.prev_games >= m


@dataclass
class GateDecision:
    """The gate's verdict plus every number it was computed from.

    Every field here is emitted as a metric. A verdict whose inputs are not
    also reported is unauditable, and this repository's signature defect is a
    value that is accepted and then silently ignored -- so the games actually
    counted travel with the decision, always.
    """

    decision: int = DECISION_NOT_RUN
    reason: str = "disabled"
    mode: str = MODE_OFF
    iters: int = 0
    games_cur: int = 0
    games_prev: int = 0
    delta_score: float = float("nan")
    delta_elo: float = float("nan")
    elo_lo: float = float("nan")
    elo_hi: float = float("nan")
    holds: int = 0
    # THIS iteration's independent anchored sample, carried separately from the
    # window aggregates above. Consecutive windows overlap by ~95%, so the sd of
    # a reported window-mean column understates the per-iteration sd ~10x; any
    # readout or kill rule written against the window column silently cannot
    # fail. These four are the only columns a decision rule may be keyed to.
    sample_delta_score: float = float("nan")
    sample_delta_elo: float = float("nan")
    sample_games_cur: int = 0
    sample_games_prev: int = 0

    @property
    def reason_code(self) -> int:
        return _REASON_CODES[self.reason]

    @property
    def mode_code(self) -> int:
        return _MODE_CODES[self.mode]

    @property
    def acted(self) -> bool:
        """True iff this decision must actually hold the published model back."""
        return self.decision == DECISION_DEMOTE


@dataclass
class GateConfig:
    """Gate knobs. Defaults are OFF and deliberately unusable as-is.

    Every default below is either measured or derived from a measurement over
    live iterations 164-219; see the module docstring. None is a round guess.
    """

    mode: str = MODE_OFF
    window_iters: int = 24
    min_iters: int = 8
    # 15, not 40. MEASURED over live iters 164-219: the prev arm realizes ~38
    # games/iteration (median 33), not the 143 the 0.60 cap suggests, so a
    # floor of 40 disqualifies 70% of iterations, pins the effective window at
    # K=9-12, and makes ``gate_delta_elo`` repeat verbatim across consecutive
    # rows -- an inert series that reads exactly like a live one. 15 keeps 96%.
    min_games_per_side: int = 15
    # -45 Elo is DERIVED, not chosen: the measured null is mean -4.3 Elo with
    # per-iteration sd 45.6, so at the default 24-iteration window the window se
    # is 9.3 Elo and this line sits ~4.4 sigma below the null. Tightening to -25
    # buys 93% power on a -50 Elo/iteration break at the same window; do that
    # only after the shadow readout confirms the null in-loop.
    demote_delta_elo: float = -45.0
    alpha: float = 0.05
    max_hold_iters: int = 12

    def validate(self) -> None:
        if self.mode not in _MODE_CODES:
            raise ValueError(
                f"gate_mode must be one of {sorted(_MODE_CODES)}, got {self.mode!r}"
            )
        # 4 / 5 rather than 2 / 1: a spread needs 2 points and a score needs 1
        # game, but a verdict off two iterations of one game a side is not a
        # verdict, and validate() is the only thing standing between an
        # operator and a gate that holds the fleet on four coin flips.
        if self.min_iters < 4:
            raise ValueError("gate_min_iters must be >= 4")
        if self.window_iters < self.min_iters:
            raise ValueError("gate_window_iters must be >= gate_min_iters")
        if self.min_games_per_side < 5:
            raise ValueError("gate_min_games_per_side must be >= 5")
        if self.demote_delta_elo >= 0.0:
            raise ValueError(
                "gate_demote_delta_elo must be negative: the gate demotes on "
                "evidence of REGRESSION, never on absence of improvement -- a "
                "loop improving 0.02 Elo/iteration cannot prove improvement at "
                "any affordable sample size"
            )
        _t_quantile(self.alpha, 8)  # raises on an unsupported alpha


@dataclass
class PromotionGate:
    """Rolling anchored-delta gate over the games the loop already plays."""

    cfg: GateConfig = field(default_factory=GateConfig)
    samples: list[AnchoredSample] = field(default_factory=list)
    holds: int = 0
    # Whether the fleet is CURRENTLY held back. Persisted: without it a restart
    # mid-hold silently publishes the held-back weights AND overwrites the
    # promoted anchor with them, while ``holds`` still reads "N deep" -- so an
    # enforce-mode hold would be bounded by restart cadence, not max_hold_iters.
    hold_active: bool = False

    def __post_init__(self) -> None:
        self.cfg.validate()

    def observe(self, sample: AnchoredSample) -> None:
        """Record one iteration's anchored counts (cheap; always safe to call)."""
        self.samples.append(sample)
        keep = max(1, int(self.cfg.window_iters))
        if len(self.samples) > keep:
            del self.samples[:-keep]

    def _base_decision(self) -> GateDecision:
        """A blank verdict pre-filled with mode, holds, and THIS iteration's
        independent anchored sample.

        The sample is carried on every return path, including the ones that
        decline to decide: the whole point of shadow mode is to read the
        per-iteration null, and a readout that only exists on iterations the
        gate happened to judge is a biased sample of its own inputs.
        """
        base = GateDecision(mode=self.cfg.mode, holds=self.holds)
        if not self.samples:
            return base
        s = self.samples[-1]
        nc, npv = s.cur_games, s.prev_games
        d = s.delta if (nc > 0 and npv > 0) else float("nan")
        return replace(
            base,
            sample_delta_score=float(d),
            sample_delta_elo=elo_from_score_delta(d),
            sample_games_cur=int(nc),
            sample_games_prev=int(npv),
        )

    def decide(self) -> GateDecision:
        """Judge the current window. Pure: never mutates ``samples``."""
        cfg = self.cfg
        base = self._base_decision()
        if cfg.mode == MODE_OFF:
            return replace(base, decision=DECISION_NOT_RUN, reason="disabled")

        usable = [
            s for s in self.samples
            if s.usable(min_games_per_side=cfg.min_games_per_side)
        ]
        games_cur = sum(s.cur_games for s in usable)
        games_prev = sum(s.prev_games for s in usable)

        if len(usable) < cfg.min_iters:
            reason = "insufficient_games" if self.samples else "insufficient_iters"
            if len(self.samples) < cfg.min_iters:
                reason = "insufficient_iters"
            return replace(
                base, decision=DECISION_NOT_RUN, reason=reason,
                iters=len(usable), games_cur=games_cur, games_prev=games_prev,
            )

        deltas = [s.delta for s in usable]
        n = len(deltas)
        mean = sum(deltas) / n
        var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        se = math.sqrt(var / n)

        if se <= 0.0:
            # Every iteration produced an identical delta. Real data cannot do
            # this; a stuck counter can. Refusing beats emitting a zero-width
            # interval that claims certainty (the L11 lesson, in a new place).
            return replace(
                base, decision=DECISION_NOT_RUN, reason="degenerate_variance",
                iters=n, games_cur=games_cur, games_prev=games_prev,
                delta_score=mean, delta_elo=elo_from_score_delta(mean),
            )

        t = _t_quantile(cfg.alpha, n - 1)
        lo_score, hi_score = mean - t * se, mean + t * se
        delta_elo = elo_from_score_delta(mean)
        elo_lo = elo_from_score_delta(lo_score)
        elo_hi = elo_from_score_delta(hi_score)

        # The rule. Demote only when the UPPER bound is below the demote line:
        # we must be confident the regression is real, not merely unable to
        # prove it is absent. Symmetrically, "promote" here means "no proven
        # regression" and explicitly does NOT claim the step was an improvement.
        regressed = elo_hi < cfg.demote_delta_elo

        measured = replace(
            base, iters=n,
            games_cur=games_cur, games_prev=games_prev,
            delta_score=mean, delta_elo=delta_elo,
            elo_lo=elo_lo, elo_hi=elo_hi,
        )
        if not regressed:
            return replace(measured, decision=DECISION_PROMOTE,
                           reason="promote_no_regression")
        if cfg.mode == MODE_SHADOW:
            return replace(measured, decision=DECISION_PROMOTE,
                           reason="shadow_would_demote")
        if self.holds >= cfg.max_hold_iters:
            # A brake that can never release is a new way to freeze the fleet
            # on stale weights -- the exact 2026-03 failure, one level up. Past
            # the cap the gate yields and says so.
            return replace(measured, decision=DECISION_PROMOTE,
                           reason="hold_expired")
        return replace(measured, decision=DECISION_DEMOTE,
                       reason="demote_regression")

    def apply(self, decision: GateDecision) -> GateDecision:
        """Commit ``decision``'s effect on the hold latch and counter."""
        self.holds = self.holds + 1 if decision.acted else 0
        self.hold_active = bool(decision.acted)
        return replace(decision, holds=self.holds)

    def advance_hold_without_decision(self) -> bool:
        """Age an active hold on an iteration that produced no verdict.

        ``sp.should_retry`` aborts an iteration before the gate ever observes
        it, so without this the release counter freezes while the fleet stays
        held: a run stuck in retry during a hold holds forever. A hold is
        therefore bounded by iterations ATTEMPTED, not iterations judged.

        Returns whether the fleet is still held after ageing.
        """
        if not self.hold_active:
            return False
        self.holds += 1
        if self.holds >= self.cfg.max_hold_iters:
            self.holds = 0
            self.hold_active = False
            return False
        return True

    def state_dict(self) -> dict[str, object]:
        return {
            "holds": int(self.holds),
            "hold_active": bool(self.hold_active),
            "samples": [
                {
                    "iteration": int(s.iteration),
                    "cur_w": int(s.cur_w), "cur_d": int(s.cur_d), "cur_l": int(s.cur_l),
                    "prev_w": int(s.prev_w), "prev_d": int(s.prev_d), "prev_l": int(s.prev_l),
                }
                for s in self.samples
            ],
        }

    def load_state_dict(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        holds_raw = state.get("holds", 0)
        self.holds = int(holds_raw) if isinstance(holds_raw, (int, float)) else 0
        self.hold_active = bool(state.get("hold_active", False))
        raw = state.get("samples")
        if not isinstance(raw, list):
            return

        def _i(d: dict[str, object], key: str, default: int = 0) -> int:
            v = d.get(key, default)
            return int(v) if isinstance(v, (int, float)) else default

        self.samples = [
            AnchoredSample(
                iteration=_i(d, "iteration", -1),
                cur_w=_i(d, "cur_w"), cur_d=_i(d, "cur_d"), cur_l=_i(d, "cur_l"),
                prev_w=_i(d, "prev_w"), prev_d=_i(d, "prev_d"), prev_l=_i(d, "prev_l"),
            )
            for d in raw if isinstance(d, dict)
        ]
        keep = max(1, int(self.cfg.window_iters))
        if len(self.samples) > keep:
            del self.samples[:-keep]


def gate_metrics(decision: GateDecision) -> dict[str, float]:
    """Flatten a decision into reported metrics.

    THE INVARIANT THIS FUNCTION EXISTS TO ENFORCE: a pass is never reported
    without the games that produced it. The old loop emitted a constant
    ``gate_passed: 1`` for 52+ iterations while ``gate_games: 0`` -- a gate
    that could not fail, indistinguishable in the metrics from one that ran and
    passed 216 times. ``gate_decision`` carries its own "did not run" code, and
    the assertion below makes the old shape unrepresentable rather than merely
    discouraged.
    """
    if decision.decision == DECISION_PROMOTE and min(
        decision.games_cur, decision.games_prev,
    ) <= 0:
        raise AssertionError(
            "promotion reported with zero anchored games "
            f"(cur={decision.games_cur}, prev={decision.games_prev}, "
            f"reason={decision.reason}): a gate that cannot fail must not "
            "report a pass"
        )
    return {
        "gate_decision": float(decision.decision),
        "gate_reason_code": float(decision.reason_code),
        "gate_mode_code": float(decision.mode_code),
        "gate_iters": float(decision.iters),
        "gate_games_cur": float(decision.games_cur),
        "gate_games_prev": float(decision.games_prev),
        "gate_delta_score": float(decision.delta_score),
        "gate_delta_elo": float(decision.delta_elo),
        "gate_elo_lo": float(decision.elo_lo),
        "gate_elo_hi": float(decision.elo_hi),
        "gate_holds": float(decision.holds),
  # The per-iteration sample. Independent across rows, unlike everything
  # above it -- write kill rules and shadow readouts against THESE.
        "gate_sample_delta_score": float(decision.sample_delta_score),
        "gate_sample_delta_elo": float(decision.sample_delta_elo),
        "gate_sample_games_cur": float(decision.sample_games_cur),
        "gate_sample_games_prev": float(decision.sample_games_prev),
    }


def gate_config_from_dict(config: dict) -> GateConfig:
    """Build a :class:`GateConfig` from the live yaml dict.

    Rejects the removed 1-sim-vs-Stockfish knobs outright. Silently ignoring
    ``gate_games: 100`` would let an operator "just turn the gate back on" and
    get the 2026-03 freeze; the whole point of this module is that the old gate
    is not reachable by a config edit.
    """
    # ``gate_games`` at anything but 0 is the forbidden shortcut: it used to
    # mean "play N games vs Stockfish at gate_mcts_sims and roll the weights
    # back". Nothing implements that any more, so accepting the key silently
    # would be a knob that never reaches the code -- refuse loudly instead.
    # ``gate_threshold`` and ``gate_mcts_sims`` stay tolerated at any value:
    # they are inert scalars, and DELETING a key from a live yaml is itself a
    # reload risk, so operators must be able to leave them in place.
    dead_games = config.get("gate_games")
    if dead_games is not None and int(dead_games) != 0:
        raise ValueError(
            f"gate_games={dead_games!r} is no longer implemented. The 1-sim "
            "vs-Stockfish gate was removed, not disabled: it measured raw "
            "policy against a PID-controlled opponent whose setpoint equals "
            "the threshold, and rejection restored the weights while the "
            "optimizer marched on. Use gate_mode / gate_demote_delta_elo."
        )
    return GateConfig(
        mode=str(config.get("gate_mode", MODE_OFF)),
        window_iters=int(config.get("gate_window_iters", 24)),
        min_iters=int(config.get("gate_min_iters", 8)),
        min_games_per_side=int(config.get("gate_min_games_per_side", 15)),
        demote_delta_elo=float(config.get("gate_demote_delta_elo", -45.0)),
        alpha=float(config.get("gate_alpha", 0.05)),
        max_hold_iters=int(config.get("gate_max_hold_iters", 12)),
    )
