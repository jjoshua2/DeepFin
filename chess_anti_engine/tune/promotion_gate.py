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
    that mean is **BOUNDED, NOT MEASURED**: se 6.26, 95% CI **[-16.6, +7.9]**,
    t = -0.69 -- indistinguishable from zero and spanning 25 Elo

Note ``distributed_prev_model_max_fraction: 0.60`` is a CEILING that never
binds -- ``distributed_stale_games`` is 0 on every recent iteration, and a
binding cap would have to discard prev shards into ``stale_*``. Sizing this
gate off 0.60 gives 95/143 and a standard error of 31 Elo. Always take the
realized split from shard data, never the cap.

**The observed spread is pure binomial noise -- there is NO detectable
anchor-drift variance**, and an earlier revision of this module claimed the
opposite. Standardizing each window's delta by its binomial se, using the
POOLED per-game score variance (sd 0.3447) rather than a per-window estimate,
gives a residual sd of **1.011**, where 1.0 is "pure independent binomial".
Under a simulated pure-binomial null at the realized window shape that
statistic is 1.007 +/- 0.095 over 200 draws, so 1.011 is dead centre. Using
per-window variance estimates instead gives 1.262 and looks like 26%
overdispersion -- that is an artefact: the same simulation returns
1.365 +/- 0.598 under the null, because at n_prev ~ 12-38 the denominator is
noisy and correlated with the numerator. Use the pooled estimate.

Two consequences. First, the empirical between-iteration sd absorbs no drift
variance, because there is none to absorb. Second, the raw sd of 45.6 is
*below* the independent-binomial value for this shape (RMS of the per-window
se, 61.1 Elo) -- consistent with the two arms being positively correlated
within an iteration, which they should be: they share the opening book draw,
the same PID difficulty and the same fleet. That correlation is a benefit of
anchoring inside one iteration, and 45.6 -- the empirical dispersion of the
statistic this class actually computes -- is the number to size power with.

The loop's measured improvement is +0.21 Elo per 1000 optimizer steps, roughly
**0.02 Elo per iteration**. Against a per-iteration se of 45.6 Elo, detecting
that at 80% power needs ~4e7 iterations. It is therefore *arithmetically
impossible* to gate on per-iteration improvement, here or with any affordable
arena; a 200-game sims-32 arena, the entire ~30 min/day GPU budget, carries se
~24.9 Elo -- worse than one free iteration. **The correction makes the negative
STRONGER, not weaker:** the noise is 47% larger than first estimated, so "no
gate can ratchet at 0.02 Elo/iteration" holds a fortiori.

So this gate is a **regression alarm and a publish brake**, not a ratchet. At
the default 24-iteration window (~4.5 h) the window se is **9.3 Elo**. Analytic
one-sided power against a sustained **-50 Elo/iteration** break, recomputed
from sd 45.56 by ``test_documented_power_at_the_shipped_line_reproduces``:

    demote line **-25** (shipped):  K=8 -> **47.1%**   K=24 -> **92.5%**
    demote line -45 (earlier draft): K=8 ->  9.4%      K=24 -> 23.9%

An earlier revision of this PR published **14% / 37%** for the -45 rows. Those
do not reproduce. A later one published 10.0% / 48.3%, computed with the OLD
``_t_quantile``, which was anti-conservative and therefore OVERSTATED power;
against the corrected quantile the numbers are 9.4% / 47.1% and now agree with
``scipy.stats.t`` to two decimals. Both corrections move the retired -45 line
further from the headline it was quoted with, not closer.

The line ships at -25 because 0 spurious holds occurred in 8000 simulated null
iterations at EITHER line (95% upper bound 0.04%), so the tighter line costs
nothing measurable. K=48 rows are UNREACHABLE at ``window_iters: 24`` and are
not quoted.

WHAT AN ENFORCE-MODE HOLD ACTUALLY DOES -- IT ERASES ITS OWN EVIDENCE
---------------------------------------------------------------------
During a hold the fleet is on ONE sha, so every accepted shard buckets to prev
and ``cur_games == 0``. ``_run_net_gating`` still observes a row -- a row of
ZEROS -- and those zero rows occupy slots in the ``window_iters``-long window.
A 12-iteration hold therefore evicts half the real samples, ``len(usable)``
falls under ``min_iters``, and the verdict goes ``NOT_RUN`` (``acted=False``),
which RELEASES the brake. Measured through the real ``GateHoldController`` and
``PromotionGate``, 200 iterations, ``max_hold_iters: 12``, 5 seeds:

    line -25, break  -50 Elo/iter: held 54.6%  longest 12  109 zero rows  45 NOT_RUN
    line -25, break -100 Elo/iter: held 66.2%  longest 12  131 zero rows  66 NOT_RUN
    line -45, break -100 Elo/iter: held 65.8%
    line -25 or -45, no break:     held  0.0%

and ``delta_elo`` is FROZEN at its pre-hold value for the whole hold (-94.2 for
all 12 iterations in one run), because no new sample can form. So the brake is
**partial by construction, at 55-66%**, and the cause is the zero-row eviction
rather than where the line sits. This is fail-open, so nothing is unsafe, but
it is the thing to decide before turning ``enforce`` on -- see the PR's open
questions. ``test_a_hold_erases_its_own_evidence`` pins the mechanism.

A -100 or -200 Elo/iteration break -- a bad merge, a broken loss term, a
mis-set LR -- trips within the ``min_iters`` floor of 8 iterations (~1.5 h).
The gate does NOT catch the 2026-06 warm-start LR crash (-494 Elo over 74
iterations = -6.7 Elo/iteration) at any window this design can reach -- the
worst-case power over K in {8, 12, 16, 24} and lines {-25, -45} is **0.32%**,
at K=8 / -25. That class of slow bleed
needs the cumulative vs-frozen-anchor series in
``scripts/daily_gate_ratchet.sh``, which is why this gate does not replace it.

THE KNOWN BIAS, AND WHY THE SHADOW READOUT NEEDS NO CONFIG CHANGE
-----------------------------------------------------------------
Model and difficulty are published in ONE manifest, so a game tagged with the
previous model's sha was also played at the previous iteration's
``wdl_regret`` / ``sf_nodes``. While the PID is moving difficulty in one
direction, the anchored delta carries a systematic offset of unknown size that
is plausibly the same order as the effects above. Nothing has measured it,
because the per-sha split did not exist until this change.

That offset has since been bounded offline from the same 418 shards: mean
**-4.3 Elo, 95% CI [-16.6, +7.9]** -- small, and not distinguishable from
zero. On length bias: pooled over games (the estimator with the power to see
one) the draw rates are cur **0.5192** vs prev **0.5384**, a **1.9 pp** gap at
z = **-1.6**, so not significant -- but NOT the "identical to four decimals" an
earlier revision claimed, which was an artefact of averaging per-iteration
rates and so weighting a 12-game arm like an 80-game one.

``gate_mode`` defaults to ``off``, and the shadow window does NOT need
``shadow``: ``decide()`` fills the per-iteration sample before the ``MODE_OFF``
check, so all four ``gate_sample_*`` columns are populated at the shipped
default. Running at ``shadow`` adds only the window aggregates -- which the
readout must not read -- and re-arms the ~252 MB anchor copy. The readout's job
is narrow and falsifiable: confirm the IN-LOOP split reproduces the offline
reconstruction, decided on the per-iteration COUNTS rather than on the delta.
See ``shadow_readout_verdict`` for why the delta cannot carry that decision and
what the rule still cannot detect.

Note what the metrics must therefore report. A rolling-window mean is not a
readable series -- consecutive rows share ~95% of their samples, so the sd of
the reported column understates the per-iteration sd by ~10x, and any rule
keyed to it cannot fail. ``gate_metrics`` emits ``gate_sample_*`` (THIS
iteration's independent anchored delta) alongside the window aggregates, and
the sample columns are the ones a decision rule must be written against.
"""
from __future__ import annotations

import csv
import logging
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

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
_T_ALPHA_ORDER = (0.10, 0.05, 0.025, 0.01, 0.005)
# Exact one-sided t quantiles for the df the gate can actually reach, rounded
# UP at the 4th decimal so every entry is >= the true value. See _t_quantile:
# the asymptotic series this replaces was anti-conservative by 8.5% at df=3,
# which `validate()`'s `min_iters >= 4` floor makes reachable.
_T_EXACT_SMALL_DF = {
    # df: (alpha 0.10, 0.05, 0.025, 0.01, 0.005)
    1: (3.0777, 6.3138, 12.7063, 31.8206, 63.6568),
    2: (1.8857, 2.9200, 4.3027, 6.9646, 9.9249),
    3: (1.6378, 2.3534, 3.1825, 4.5408, 5.8410),
    4: (1.5333, 2.1319, 2.7765, 3.7470, 4.6041),
    5: (1.4759, 2.0151, 2.5706, 3.3650, 4.0322),
    6: (1.4398, 1.9432, 2.4470, 3.1427, 3.7075),
    7: (1.4150, 1.8946, 2.3647, 2.9980, 3.4995),
    8: (1.3969, 1.8596, 2.3061, 2.8965, 3.3554),
    9: (1.3831, 1.8332, 2.2622, 2.8215, 3.2499),
    10: (1.3722, 1.8125, 2.2282, 2.7638, 3.1693),
    11: (1.3635, 1.7959, 2.2010, 2.7181, 3.1059),
    12: (1.3563, 1.7823, 2.1789, 2.6810, 3.0546),
    13: (1.3502, 1.7710, 2.1604, 2.6504, 3.0123),
    14: (1.3451, 1.7614, 2.1448, 2.6245, 2.9769),
    15: (1.3407, 1.7531, 2.1315, 2.6025, 2.9468),
}

log = logging.getLogger(__name__)


def elo_from_score_delta(delta: float) -> float:
    """Local-linear score-delta -> Elo, valid for small deltas around 0.5."""
    return float(delta) * ELO_PER_SCORE_AT_HALF


def _t_quantile(alpha: float, df: int) -> float:
    """One-sided t quantile, scipy-free and NEVER narrower than the exact t.

    An earlier revision used the one-term Cornish-Fisher correction
    ``z + (z**3 + z) / (4 df)`` and documented it as "within ~1% for df >= 8
    and conservative (wider) below it". Both halves were wrong: it is
    ANTI-conservative at every df, by 1.7% at df=7 and **8.5% at df=3** -- and
    df=3 is reachable, because ``validate()`` admits ``min_iters >= 4``. A CI
    that is narrower than it claims demotes more readily than its stated alpha,
    which is the cheap direction, but "documented as the opposite of what it
    does" is the defect class this module exists to remove.

    Two branches, both conservative by construction:

    * ``df <= 15`` -- the exact quantile from the table below, rounded UP at
      the 4th decimal. This is the regime the gate can actually reach.
    * ``df >= 16`` -- the five-term Cornish-Fisher expansion, whose worst
      shortfall against the exact t over all supported alphas and every
      ``df >= 16`` is 0.0026%, scaled by ``1 + 1e-4`` so the margin dominates
      the truncation error.

    ``test_t_quantile_is_never_narrower_than_the_exact_t`` pins the direction
    against exact literals, so this cannot silently regress.
    """
    z = _Z_ONE_SIDED.get(round(float(alpha), 4))
    if z is None:
        raise ValueError(
            f"unsupported alpha {alpha!r}; supported: {sorted(_Z_ONE_SIDED)}"
        )
    if df <= 0:
        return float("inf")
    exact = _T_EXACT_SMALL_DF.get(int(df))
    if exact is not None:
        return float(exact[_T_ALPHA_ORDER.index(round(float(alpha), 4))])
    v = float(df)
    series = (
        z
        + (z ** 3 + z) / (4.0 * v)
        + (5.0 * z ** 5 + 16.0 * z ** 3 + 3.0 * z) / (96.0 * v ** 2)
        + (3.0 * z ** 7 + 19.0 * z ** 5 + 17.0 * z ** 3 - 15.0 * z) / (384.0 * v ** 3)
        + (79.0 * z ** 9 + 776.0 * z ** 7 + 1482.0 * z ** 5
           - 1920.0 * z ** 3 - 945.0 * z) / (92160.0 * v ** 4)
    )
    return float(series * (1.0 + 1.0e-4))


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
    # -25, and the derivation is a FALSE-BRAKE BUDGET, not a sigma count. A
    # 4-sigma line (-45) was the first choice and it does not deliver what the
    # PR claimed for it. Analytic one-sided power against a sustained -50
    # Elo/iteration break at sd 45.56 (recomputed by
    # ``test_documented_power_at_the_shipped_line_reproduces``):
    #
    #     line -45:  K=8 -> 10.0%   K=24 -> 23.9%   (PR round 1 said 14% / 37%)
    #     line -25:  K=8 -> 48.3%   K=24 -> 92.5%
    #
    # The cost of buying that is a false brake, and it is not measurable at
    # this window: 0 spurious holds in 8000 simulated null iterations at BOTH
    # -45 and -25 (95% upper bound 0.04%). An earlier revision of this comment
    # quoted "0.01%" as a point estimate; it is an upper bound, not a
    # measurement. Each spurious hold costs at most ``max_hold_iters`` of
    # slightly stale selfplay and never a training step, which is the right
    # trade for an alarm whose action is cheap and whose misses are expensive.
    demote_delta_elo: float = -25.0
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


def gate_metrics(decision: GateDecision, *, strict: bool = True) -> dict[str, float]:
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
        if not strict:
  # The reporting path passes strict=False. This invariant guards a
  # state no shipped code path can reach, so a raise from inside
  # ``_build_report_dict`` would take the whole trial down to protect
  # against something that cannot happen. Degrade to "did not run",
  # loudly, and let the loop keep training.
            log.error(
                "promotion gate reported PROMOTE with zero anchored games "
                "(cur=%d, prev=%d, reason=%s) -- downgrading to NOT_RUN in the "
                "metrics. This should be unreachable; investigate.",
                decision.games_cur, decision.games_prev, decision.reason,
            )
            return gate_metrics(
                replace(decision, decision=DECISION_NOT_RUN, reason="disabled"),
            )
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
        demote_delta_elo=float(config.get("gate_demote_delta_elo", -25.0)),
        alpha=float(config.get("gate_alpha", 0.05)),
        max_hold_iters=int(config.get("gate_max_hold_iters", 12)),
    )


# --------------------------------------------------------------------------
# The shadow readout: the promote-to-enforce decision, as CODE.
# --------------------------------------------------------------------------
# It lives here rather than in the ledger's shell one-liner because the two
# disagreed: the ledger's worked example computed ``usable_frac`` over windows
# with a non-empty split (51/53 = 0.96) while its command computed it over all
# progress rows (51/57 = 0.89), against a KILL line of 0.85. One implementation
# means the worked example and the shipped command CANNOT diverge, and it means
# the rule is something a test can drive.

READOUT_PROMOTE = "promote_to_enforce"
READOUT_HOLD = "hold_in_shadow"
READOUT_KILL = "kill"


@dataclass(frozen=True)
class OfflineReference:
    """What the offline reconstruction measured, BEFORE the shadow window ran.

    Binning 418 processed shard ``.zattrs`` by ``generated_at_unix`` against
    ``progress.csv`` over live iterations 163-219, splitting each window's
    curriculum games by ``model_sha256``. Reproduced independently three times
    (author, review round 2, this change) to the digit.

    ``tests/test_promotion_gate.py`` carries the 57-row reconstruction itself
    and recomputes every field here from it, so these cannot drift into being
    decoration.

    ``mean_iter_seconds`` is the cadence the other fields were measured at, and
    it is load-bearing rather than decorative: see ``refresh_lag_seconds``.
    """

    mean_games_cur: float = 196.8
    mean_games_prev: float = 38.3
    prev_share: float = 0.1629
    mean_delta_elo: float = -4.33
    sd_delta_elo: float = 45.56
    n_usable: int = 53
    # Mean ``time_this_iter_s`` over the same 51 rows the counts come from.
    mean_iter_seconds: float = 721.0
    # Mean anchored games per iteration-SECOND over those rows. Empirical and
    # aggregate -- it includes the training phase, during which
    # `distributed_pause_selfplay_during_training` stops selfplay -- so it is a
    # cadence-normalised count, not a physical selfplay rate. Per-iteration it
    # spans 0.59x-1.22x of this and the rolling-40 window mean spans
    # 1.00x-1.05x, which is why its leg is a factor-of-2 band and not tight.
    games_per_second: float = 0.3411

    @property
    def refresh_lag_seconds(self) -> float:
        """The prev arm's implied duration, in SECONDS -- the cadence-free form.

        ``prev`` games are not a fixed fraction of ingest. They are the fleet's
        model-refresh lag: shards still tagged with the previous sha because the
        worker had not picked up the new manifest yet. That lag is roughly
        constant in wall-clock seconds, so a longer iteration shrinks its SHARE
        without anything being wrong. Measured on the reference window:

            corr(time_this_iter_s, prev_share) = **-0.332** (n=51)
            slow half (815 s/iter): prev_share 0.1438
            fast half (629 s/iter): prev_share 0.1820
            iters <=175 (1012 s/iter): 0.1218 | iters >175 (675 s/iter): 0.1728
              -> a 0.0509 excursion, 85% of the leg's 0.06 tolerance, from a
                 1.5x cadence change INSIDE the calibration window

        ``prev_share * iter_seconds`` removes most of that: the same early/late
        split moves this quantity by **7.5%** where it moves ``prev_share`` by
        **31.1%**. So the leg is evaluated against ``refresh_lag / cadence``
        rather than against a fixed share.
        """
        return self.prev_share * self.mean_iter_seconds


OFFLINE = OfflineReference()

# The band of cadence ratios over which the 1/cadence model is trusted.
# Below the floor the model predicts shares approaching
# ``distributed_prev_model_max_fraction: 0.60``, where the CAP starts binding
# and the model stops holding (it predicts 0.60 at ~0.27x); it also erodes the
# margin against a coin-shuffle's ~0.50, which is the negative control the leg
# exists to pass. Outside the band the readout reports a cadence leg by name
# rather than extrapolating -- an operator must not read "your cadence moved"
# as "your attribution is broken".
_CADENCE_RATIO_MIN = 0.4
_CADENCE_RATIO_MAX = 3.0


@dataclass(frozen=True)
class ShadowReadout:
    """The shadow window's verdict plus every number it was computed from."""

    verdict: str
    n_rows: int
    n_usable: int
    usable_frac: float
    mean_games_cur: float
    mean_games_prev: float
    prev_share: float
    mean_delta_elo: float
    sd_delta_elo: float
    # Cadence of the read window, and the prev_share it implies. NaN when the
    # rows carry no `time_this_iter_s`, in which case the raw reference share
    # is used and `failed_legs` says so on a failure.
    mean_iter_seconds: float = float("nan")
    expected_prev_share: float = float("nan")
    failed_legs: tuple[str, ...] = ()

    def __str__(self) -> str:
        cad = (f"  cadence={self.mean_iter_seconds:.0f}s "
               f"expected_prev_share={self.expected_prev_share:.4f}"
               if not math.isnan(self.mean_iter_seconds) else "  cadence=unknown")
        return (
            f"{self.verdict}  rows={self.n_rows} usable={self.n_usable} "
            f"({self.usable_frac:.3f})  games_cur={self.mean_games_cur:.1f} "
            f"games_prev={self.mean_games_prev:.1f} "
            f"prev_share={self.prev_share:.4f}{cad}  "
            f"delta mean={self.mean_delta_elo:.2f} sd={self.sd_delta_elo:.2f}"
            + (f"  FAILED: {', '.join(self.failed_legs)}" if self.failed_legs else "")
        )


def shadow_readout_verdict(
    rows: Sequence[Sequence[float]],
    *,
    min_games_per_side: int = 15,
    last_n: int = 40,
    ref: OfflineReference = OFFLINE,
) -> ShadowReadout:
    """Decide whether the in-loop instrument reproduces the offline split.

    ``rows`` are ``(gate_sample_games_cur, gate_sample_games_prev,
    gate_sample_delta_elo)`` -- the PER-ITERATION columns. Never the window
    aggregates: consecutive windows overlap ~95%, so the sd of a window column
    understates the per-iteration sd ~10x and a rule keyed to it cannot fail.

    WHY THE DECIDING LEGS ARE COUNTS AND NOT THE DELTA
    --------------------------------------------------
    The window exists to catch a mis-attributed ingest split. Round 2 of review
    tested exactly that: pool each iteration's cur+prev games and redeal them
    at random into the SAME realized arm sizes -- destroying the
    ``model_sha256`` attribution completely -- and all three delta-based legs
    still passed on 176 of 200 reshuffles. That is not a tuning miss, it is
    structural. The true anchored signal is -4.33 Elo with 95% CI [-16.6, +7.9]
    -- indistinguishable from zero -- so any splitter that keeps the arms
    balanced lands inside any honest band around it, and the spread leg is
    satisfied by pure binomial noise (standardized residual sd 1.011), which is
    exactly what a broken split also produces.

    The per-iteration game COUNTS are the opposite: attribution moves them by
    tens of games per row with essentially no sampling noise. A random
    shard-level relabelling drives ``prev_share`` from 0.1629 to ~0.50; a
    cur/prev swap drives it to ~0.83; an unrecognised prev sha drives it to 0
    and empties the window. All three are caught every time
    (``test_negative_control_reshuffled_attribution_is_killed``).

    ``prev_share`` IS COUPLED TO CADENCE, and an earlier revision of this
    docstring claimed the opposite ("a ratio, so cadence drift cannot trip
    it"). It is measurably false. ``prev`` games are the fleet's model-refresh
    lag, roughly constant in SECONDS, so a longer iteration shrinks their share
    with nothing wrong: ``corr(time_this_iter_s, prev_share) = -0.332`` over
    the 51 reference rows, and a 1.5x cadence change *inside the calibration
    window* moves the leg by **0.0509** against a tolerance of 0.06. The
    documented cadence excursions in this repo are larger than that (11 -> 22
    min under side-job CPU contention alone), so the claim would have produced
    a `kill` -- read as "your attribution is broken" -- on a benign restart.

    The leg is therefore evaluated against ``refresh_lag_seconds / cadence``
    rather than a fixed share, which cuts that excursion from 0.0509 to 0.0058,
    and a separate CADENCE leg fires by name outside a 0.4x-3.0x band instead
    of extrapolating a model that stops holding there.

    THE SENSITIVITY FLOOR, because "0.34-0.67" is true only of TOTAL failures.
    Leaking a fraction f of one arm into the other:

        prev -> cur:  f = 0.10 and 0.20 PROMOTE (invisible to every leg);
                      f = 0.30 kills, but on ``usable_frac``, not ``prev_share``
        cur  -> prev: f <= 0.06 PROMOTE;  f = 0.08 kills on ``prev_share``

    So a 10-20% one-sided partial leak is NOT detected. That is consistent with
    the whole-shard argument -- shards are attributed entire, so a partial leak
    has no mechanism -- but it is a blind spot, not an absence of one, and
    ``test_partial_attribution_leak_sensitivity_floor`` pins the numbers.

    THE OTHER COUNT LEG IS A RATE, for the same reason. An absolute band on
    ``mean_games_cur`` carries exactly the false alarm ``prev_share`` used to:
    at 1.5x cadence a healthy loop plays 1.5x the games and a +/-60 band fires.
    The leg is ``(cur + prev) / iter_seconds`` against a factor-2 band, which
    covers the 1.4x throughput difference between the reference window's own
    two regimes (0.2545 vs 0.3549 games/s) with margin. ``mean_games_cur`` and
    ``mean_games_prev`` are still REPORTED, and carry no invariance claim.

    Verified across the cadence band: benign 0.5x-2.5x cadence changes all
    return ``promote_to_enforce`` with no leg firing, 3.5x fires the CADENCE
    leg by name, and the coin/swap destructions are still caught 50/50 at
    0.4x, 0.5x, 1.0x, 2.0x and 3.0x.

    WHAT THIS RULE CANNOT DO, STATED SO NOBODY DISCOVERS IT LATER
    -------------------------------------------------------------
    It cannot detect a destruction that CONDITIONS on the realized arm sizes --
    review round 2's exact control. Nothing can: conditional on the counts, the
    only remaining channel is the delta, and the true effect is not
    distinguishable from zero at n=53. That control is also unphysical -- shards
    are attributed whole, by sha, so no real bug moves games between arms
    without moving the counts -- but the honest statement is "no rule passes
    that control", not "this rule does".
    """
    window = [_readout_row(r) for r in list(rows)[-max(1, int(last_n)):]]
    floor = max(1, int(min_games_per_side))
    usable = [r for r in window
              if r[0] >= floor and r[1] >= floor and not math.isnan(r[2])]
    n_rows, n_usable = len(window), len(usable)
    frac = n_usable / n_rows if n_rows else 0.0

    if n_usable < 2:
        # The most important failure mode must have a defined answer. An
        # earlier draft of this rule raised StatisticsError here instead.
        return ShadowReadout(
            verdict=READOUT_KILL, n_rows=n_rows, n_usable=n_usable,
            usable_frac=frac, mean_games_cur=0.0, mean_games_prev=0.0,
            prev_share=0.0, mean_delta_elo=float("nan"),
            sd_delta_elo=float("nan"),
            failed_legs=("usable_rows>=2",),
        )

    mean_cur = statistics.mean([r[0] for r in usable])
    mean_prev = statistics.mean([r[1] for r in usable])
    prev_share = mean_prev / (mean_cur + mean_prev)
    deltas = [r[2] for r in usable]
    mean_d, sd_d = statistics.mean(deltas), statistics.stdev(deltas)

    # -- the cadence adjustment, and its own leg ---------------------------
    secs = [r[3] for r in usable if not math.isnan(r[3]) and r[3] > 0.0]
    mean_secs = statistics.mean(secs) if secs else float("nan")
    failed: list[str] = []
    if math.isnan(mean_secs):
        # No `time_this_iter_s` in the rows. Fall back to the raw reference
        # share and SAY SO on a failure, rather than pretending to a
        # cadence-corrected comparison that was never made.
        expected_share, share_tol, share_note = ref.prev_share, 0.06, " (cadence unknown)"
    else:
        ratio = mean_secs / ref.mean_iter_seconds
        if not (_CADENCE_RATIO_MIN <= ratio <= _CADENCE_RATIO_MAX):
            failed.append(
                f"cadence {mean_secs:.0f}s is {ratio:.2f}x the reference "
                f"{ref.mean_iter_seconds:.0f}s, outside "
                f"[{_CADENCE_RATIO_MIN}, {_CADENCE_RATIO_MAX}] -- the "
                "prev-share model is not trusted here; this is a CADENCE "
                "finding, NOT an attribution finding"
            )
        expected_share = ref.refresh_lag_seconds / mean_secs
        share_tol, share_note = 0.06, ""

    # -- deciding legs: attribution-sensitive, near-noiseless --------------
    if frac < 0.85:
        failed.append(f"usable_frac {frac:.3f} < 0.85")
    if not math.isnan(mean_secs):
        # Absolute counts scale with cadence, so an absolute band on them is
        # the same false alarm prev_share used to carry -- at 1.5x cadence a
        # `mean_games_cur` band of +/-60 fires on a perfectly healthy loop.
        # Compare the cadence-normalised rate instead, with a factor-2 band:
        # the reference window's own throughput regimes differ by 1.4x.
        rate = (mean_cur + mean_prev) / mean_secs
        if not (0.5 <= rate / ref.games_per_second <= 2.0):
            failed.append(
                f"games_per_second {rate:.4f} is "
                f"{rate / ref.games_per_second:.2f}x the reference "
                f"{ref.games_per_second} (band 0.5-2.0)"
            )
    elif not (0.25 <= (mean_cur + mean_prev) / (
            ref.mean_games_cur + ref.mean_games_prev) <= 4.0):
        # No cadence column: fall back to a deliberately loose absolute band,
        # because without cadence the count carries no information about
        # anything except gross breakage.
        failed.append(
            f"anchored games/iteration {mean_cur + mean_prev:.1f} vs reference "
            f"{ref.mean_games_cur + ref.mean_games_prev:.1f} (band 0.25-4.0x,"
            " cadence unknown)"
        )
    if abs(prev_share - expected_share) > share_tol:
        failed.append(
            f"prev_share {prev_share:.4f} vs expected {expected_share:.4f}"
            f" +/-{share_tol}{share_note}"
        )
    # -- deciding leg: instrument-sensitive --------------------------------
    # 4.56 is the sd of the 95%-overlapping WINDOW column. If the readout is
    # ever wired to that column again this leg is what says so.
    if not (20.0 < sd_d < 70.0):
        failed.append(f"sd_delta_elo {sd_d:.2f} outside (20, 70)")
    # -- offset leg: the PID-lag bias must not dominate ---------------------
    if abs(mean_d) > 25.0:
        failed.append(f"|mean_delta_elo| {abs(mean_d):.2f} > 25")

    if failed:
        verdict = READOUT_KILL
    elif abs(mean_d) > 15.0:
        verdict = READOUT_HOLD  # extend the window rather than promote or kill
    else:
        verdict = READOUT_PROMOTE
    return ShadowReadout(
        verdict=verdict, n_rows=n_rows, n_usable=n_usable, usable_frac=frac,
        mean_games_cur=mean_cur, mean_games_prev=mean_prev,
        prev_share=prev_share, mean_delta_elo=mean_d, sd_delta_elo=sd_d,
        mean_iter_seconds=mean_secs, expected_prev_share=expected_share,
        failed_legs=tuple(failed),
    )


def _readout_row(row: Sequence[float]) -> tuple[int, int, float, float]:
    """Normalise a readout row to (cur, prev, delta_elo, iter_seconds).

    Three-element rows are accepted so a caller with no cadence column still
    gets a verdict -- with the cadence adjustment disabled and named.
    """
    cur, prev, delta = int(row[0]), int(row[1]), float(row[2])
    secs = float(row[3]) if len(row) > 3 else float("nan")
    return cur, prev, delta, secs


def shadow_readout_rows_from_csv(
    rows: Iterable[dict[str, str]],
) -> list[tuple[int, int, float, float]]:
    """Pull the gate sample columns, plus cadence, out of ``progress.csv`` rows.

    ``time_this_iter_s`` comes along because the ``prev_share`` leg is
    evaluated against a CADENCE-ADJUSTED expectation -- see
    ``OfflineReference.refresh_lag_seconds``. A row missing it still counts;
    the adjustment is then disabled for the whole window and says so.

    A row whose gate columns are BLANK never ran the gate and is not an
    iteration of the shadow window, so it is dropped. A row with
    ``games_cur == 0`` (the shape during a hold) DID run and is kept, with its
    NaN delta: it is unusable, and dropping it would quietly inflate
    ``usable_frac`` -- which is the denominator confusion that made the
    ledger's worked example and its command disagree by 7 points against a
    kill line 4 points away.
    """
    out: list[tuple[int, int, float, float]] = []
    for r in rows:
        raw_c = r.get("gate_sample_games_cur")
        raw_p = r.get("gate_sample_games_prev")
        if raw_c is None or raw_p is None or raw_c == "" or raw_p == "":
            continue
        try:
            c, p = int(float(raw_c)), int(float(raw_p))
            d = float(r.get("gate_sample_delta_elo") or "nan")
            secs = float(r.get("time_this_iter_s") or "nan")
        except ValueError:
            continue
        out.append((c, p, d, secs))
    return out


def shadow_readout_from_csv(
    path: str | Path, *, min_games_per_side: int = 15, last_n: int = 40,
) -> ShadowReadout:
    """The ledger's ONE deciding command, as a function. See the ledger entry."""
    with Path(path).open(newline="") as fh:
        rows = shadow_readout_rows_from_csv(csv.DictReader(fh))
    return shadow_readout_verdict(
        rows, min_games_per_side=min_games_per_side, last_n=last_n,
    )


@dataclass
class GateHoldController:
    """Owns every piece of gate state the trial loop used to keep in locals.

    Six loose local variables in ``trainable.py`` -- the anchor path, the hold
    path, the startup restore, the retry ageing, the release, and the
    sign-validity history -- were individually mutable and individually
    untested: an independent reviewer mutated each of them and all six escaped
    35 tests. Collapsing them into one object means the loop makes three calls
    instead of six assignments, and the state machine is drivable by a test.

    Nothing here plays a game or touches a weight. It decides which FILE the
    next publish serves.
    """

    gate: PromotionGate
    promoted_model_path: Path | None = None
    hold_path: Path | None = None
    _this_publish_held: bool = False
    _prev_publish_held: bool = False

    @classmethod
    def create(cls, gate: PromotionGate, *, durable_dir: Path) -> GateHoldController:
        """Build at trial startup, restoring a hold that survived a restart.

        The anchor path is None while the gate is off, which is what keeps a
        disabled feature from copying a ~252 MB export every iteration. A
        restored hold with no anchor on disk is released rather than trusted.
        """
        promoted = (
            durable_dir / "gate_promoted_model.pt" if gate.cfg.mode != MODE_OFF else None
        )
        hold = (
            promoted
            if (gate.hold_active and promoted is not None and promoted.is_file())
            else None
        )
        if gate.hold_active:
            print(
                f"[gate] resuming with the fleet HELD: holds={gate.holds} "
                f"fallback={'present' if hold else 'MISSING (releasing)'}",
                flush=True,
            )
        if gate.hold_active and hold is None:
            log.warning(
                "promotion gate resumed with hold_active but no anchor on disk; "
                "releasing the hold",
            )
            gate.hold_active = False
        return cls(gate=gate, promoted_model_path=promoted, hold_path=hold)

    def note_published(self) -> None:
        """Record whether the publish that just happened served the anchor.

        Called once per publish. The one-iteration history is what makes
        ``sample_is_valid`` able to see the transition.
        """
        self._prev_publish_held = self._this_publish_held
        self._this_publish_held = self.hold_path is not None

    @property
    def sample_is_valid(self) -> bool:
        """Whether this iteration's anchored counts mean what their names say.

        THE SIGN INVERTS ON A HOLD TRANSITION. ``_process_shard`` labels a
        shard "prev" by comparing against ``prev_published_model_sha``, which
        still names the DEMOTED net on the iteration that first serves the
        anchor -- so the older net is labelled "cur" and the newer one "prev",
        and a -139 Elo regression is recorded as +139 at exactly the moment the
        gate acts on it. The release iteration is wrong differently: "prev" is
        then the anchor, so the delta spans however many iterations the hold
        lasted rather than one.

        Neither is a sample of "one training iteration", so neither is
        observed. The gate's own action must not contaminate its own evidence
        -- that is the "ruler moved with the model" defect this module exists
        to remove, and it would be the worst possible place to reintroduce it.
        """
        return not (self._this_publish_held or self._prev_publish_held)

    def on_decision(self, decision: GateDecision) -> None:
        """Apply a verdict to the NEXT publish."""
        self.hold_path = resolve_gate_hold_path(
            decision, gate_promoted_model_path=self.promoted_model_path,
        )

    def on_aborted_iteration(self) -> None:
        """Age an active hold on an iteration that produced no verdict."""
        if not self.gate.advance_hold_without_decision():
            self.hold_path = None


def resolve_gate_hold_path(
    decision: GateDecision, *, gate_promoted_model_path: Path | None,
) -> Path | None:
    """Turn a verdict into the path the next publish must serve, or None.

    A demote with no fallback on disk (the first iterations after a fresh
    start) publishes normally and says so rather than crashing the trial.
    """
    if not decision.acted:
        return None
    hold = (
        gate_promoted_model_path
        if gate_promoted_model_path is not None and gate_promoted_model_path.is_file()
        else None
    )
    print(
        "[gate] HOLD publish on the promoted export: "
        f"reason={decision.reason} delta_elo={decision.delta_elo:.1f} "
        f"ci=[{decision.elo_lo:.1f},{decision.elo_hi:.1f}] "
        f"iters={decision.iters} "
        f"games={decision.games_cur}/{decision.games_prev} "
        f"holds={decision.holds} "
        f"fallback={'present' if hold else 'MISSING (publishing anyway)'}",
        flush=True,
    )
    return hold
