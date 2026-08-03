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
  a handicapped Stockfish. That is an anchored A/B that the loop pays for
  anyway. This module reads it; it never plays a game.

  **The two arms do NOT face the same opponent, and the PID's tracking does
  NOT cancel in the difference.** An earlier revision of this docstring and of
  the PR body claimed both, and the claim is false by construction -- see
  "THE PID LAG DOES NOT CANCEL" below. What the anchoring DOES remove is the
  aging defect: the comparison is regenerated from scratch every iteration and
  nothing about it is frozen. That is the property this design buys, and it is
  a different property from "the controller cancels".
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
against the corrected quantile the numbers are 9.4% / 47.1% and agree with
``scipy.stats.t`` to within 0.01 of a percentage point. Both corrections move
the retired -45 line further from the headline it was quoted with, not closer.

Every copy of these four numbers -- this table, the prose above, the comment
next to the shipped ``demote_delta_elo`` and the production yaml -- is pinned
against the source text by
``test_the_documented_power_numbers_are_quoted_consistently_everywhere``.

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

TIME TO TRIP, AND WHY "8 ITERATIONS" WAS THE WRONG NUMBER
---------------------------------------------------------
An earlier revision said "a -100 or -200 Elo/iteration break trips within the
``min_iters`` floor of 8 iterations (~1.5 h)". **Eight is the COLD-window
number** -- the latency measured from an EMPTY window, where the first verdict
is possible at iteration 8 and the window contains nothing but broken rows.
Production never has that window; it has 24 healthy rows, and the 23 pre-break
rows sit in the mean dragging it toward zero until they age out.

Measured, 400 seeds per row, deltas ~ N(break, 45.56) driven through the
shipped rule (``test_documented_time_to_trip_is_the_steady_state_number``):

    break        steady-state window        cold window
    -50/iter     median 20   p90 26         median 10   p90 25
    -100/iter    median 12   p90 15         median  8   p90  8
    -200/iter    median  8   p90 10         median  8   p90  8
    -300/iter    median  6   p90  8         median  8   p90  8

So a -100 Elo/iteration break costs a median of **12 iterations (~2.4 h)** to
trip in production and up to 15 at p90, not 8. Every one of these is a
SUSTAINED per-iteration rate; a one-off STEP of any size is a different object
and the mean-CI rule cannot see it at all -- see the next section.

The gate does NOT catch the 2026-06 warm-start LR crash (-494 Elo over 74
iterations = -6.7 Elo/iteration) at any window this design can reach -- the
worst-case power over K in {8, 12, 16, 24} and lines {-25, -45} is **0.287%**,
at K=8 / -25 (``test_documented_power_at_the_shipped_line_reproduces`` asserts
0.0029; the yaml comment rounds it to 0.29%). That class of slow bleed
needs the cumulative vs-frozen-anchor series in
``scripts/daily_gate_ratchet.sh``, which is why this gate does not replace it.

A SINGLE-ITERATION STEP OF **ANY** MAGNITUDE CANNOT MOVE THE MEAN-CI RULE
-------------------------------------------------------------------------
The mean-CI leg (``elo_hi < demote_delta_elo``) tests a per-iteration RATE. A
bad merge, a broken loss term or a mis-set LR is a **STEP**: the model is worse
from one iteration onward. The anchored delta is a *first difference* -- cur
model minus prev model, both re-measured every iteration -- so a level shift of
-M Elo appears in EXACTLY ONE sample and the series returns to its old level
immediately after. That single sample cannot demote, and the arithmetic says
so without any reference to M:

    window of K deltas, one equal to -M and the rest 0
        mean = -M/K
        var  = sum((d - mean)^2)/(K-1) = (M^2 (K-1)/K) / (K-1) = M^2/K
        se   = sqrt(var/K) = M/K                      <- EXACTLY the mean
        elo_hi = mean + t*se = (M/K)(t - 1)           <- POSITIVE for every t>1

``t = _t_quantile(alpha, K-1) > 1`` at every alpha this module supports, so
**elo_hi is strictly positive and M cancels**. Verified numerically: at K=24,
alpha=0.05 a -1,000,000 Elo one-shot gives ``elo_hi = +29,754``. Raising K,
lowering alpha or moving ``demote_delta_elo`` cannot help -- M is not in the
answer. ``test_a_one_iteration_step_cannot_move_the_mean_ci_rule`` pins it.

So the gate carries a SECOND, independent leg keyed to THIS iteration's sample
alone: ``sample_elo_hi < demote_step_elo``, where ``sample_elo_hi`` is the
one-sided upper bound of the single anchored sample from its own binomial
counts (``AnchoredSample.elo_hi``). It is evaluated BEFORE the ``min_iters``
window check, because a bad merge three iterations after a restart is exactly
when a step leg has to work, and it is OR-ed into the demote condition only --
it can never turn a demote into a promote, and it is not consulted at all on
the promote path.

WHAT THE STEP LEG COSTS AND WHAT IT BUYS, at the realized shape (n_cur ~197,
n_prev ~38, pooled per-game score sd 0.3447 -> sample se **42.4 Elo**), and at
the worst shape ``min_games_per_side`` admits (197/15 -> se **64.2 Elo**):

    line   shape     fires at        spurious per 8000     50% power   90% power
    -125   197/38    delta < -195    0.02 null iters       -195 Elo    -249 Elo
    -125   197/15    delta < -231    1.31 null iters       -231 Elo    -313 Elo

``demote_step_elo`` ships at **-125**, chosen on the same FALSE-BRAKE BUDGET as
``demote_delta_elo``: 1.31 spurious holds per 8000 null iterations at the worst
admissible shape is 0.016%, inside the 0.04% upper bound the window leg was
sized against. **It does NOT catch a -100 step** and nothing at this sample
size can: one anchored sample carries 42-64 Elo of binomial noise, so a
-100 step is under 2.5 sigma and buying it would cost a false brake every few
hundred iterations. The honest claim is "a one-iteration step of about -200 Elo
or worse, at 50% power; -250 or worse at 90%", and that is the claim the tests
assert.

THE PID LAG DOES NOT CANCEL -- THE ANCHORED DELTA CARRIES A CONTROLLER TERM
----------------------------------------------------------------------------
**This section replaces a claim that was wrong.** Earlier revisions of this
module and of the PR said the two arms face the same opponent within one
iteration, so that the controller's movement would subtract out of the
anchored difference. The arms do not face the same opponent, and nothing
subtracts out.

Model and difficulty ship in the SAME ``recommended_worker`` manifest and are
applied together (``opponent_wdl_regret_limit`` and ``sf_nodes`` are both in
``worker._RECO_LIVE_KEYS``; ``_publish_distributed_trial_state`` writes the
model and both levers in one publish). A shard still tagged with the previous
model's sha is therefore a shard from a worker that had not picked up the new
manifest, so it was played at the PREVIOUS iteration's difficulty as well.
**Old-model-at-new-difficulty is not an observable state**, so there is nothing
for the subtraction to cancel: the prev arm is one PID step behind on
difficulty, always, by construction.

Worse, the offset is not noise -- it is a controller output, so its SIGN
OPPOSES the model change and its MAGNITUDE SCALES WITH IT. The PID lowers
``wdl_regret`` (harder) when the net wins too much and raises it (easier) when
the net wins too little, so a net that just got weaker is met with an EASIER
opponent on the next publish, and the cur arm scores better than the weakening
warrants. That is the masking direction. Measured on the live trial's
``progress.csv``:

    corr(pid_raw_winrate_t, pid_regret_delta_{t+1}) = **-0.303** (n=42)

and the size of the induced offset, per iteration, as
``d(wdl_regret) x pid_regret_fit_slope x ELO_PER_SCORE_AT_HALF``:

    median |confound| **4.96 Elo**, mean 8.13, p90 20.9, max 36.4 (n=34)
    **5.9% of healthy iterations exceed the whole of ``demote_delta_elo``**

An independent review measured the same quantities over 259 live iterations
and got median |confound| 7.0 Elo and 9.4% over the line, and found that under
a closed-loop simulation with this repo's real ``DifficultyPID`` a -50
Elo/iteration break reads **+3.2 Elo instead of -30** -- the sign flips.

**WHAT THIS DOES TO THE -4.3 Elo BOUND.** The offline reconstruction's mean of
-4.33 Elo (95% CI [-16.6, +7.9]) was measured UNCONDITIONALLY over 53 healthy
iterations, where the PID is doing small corrective steps in both directions
and the confound averages out. That bound says nothing about the regime the
gate ACTS in: a real regression is precisely the regime where the PID makes a
large, one-signed, model-correlated step, and there the term does not average
out -- it opposes the signal the gate is looking for. **Do not quote -4.3 Elo
as a bound on the bias during a break.**

WHAT THIS CHANGE DOES ABOUT IT: nothing, deliberately. The gate ships
``off``; the defect was the false claim, not the unsolved statistics. What it
adds is the means to FALSIFY the claim with production data instead of
simulation:

* ``ShardMeta`` now records ``opponent_wdl_regret_limit`` and ``sf_nodes``, so
  every shard says which difficulty it was played at. Before this, neither the
  loop nor an offline reconstruction could check it at all. The ingest split
  carries the games-weighted mean per arm into ``gate_cur_wdl_regret`` /
  ``gate_prev_wdl_regret``, so the arms' difficulty gap is a reported number
  rather than an inference.
* ``gate_metrics`` emits ``gate_sample_confound_elo`` next to
  ``gate_sample_delta_elo``: the MEASURED per-arm regret gap times the PID's
  own ``pid_regret_fit_slope``, in the same Elo units as the delta. If the two
  columns track each other, the gate is measuring the controller.
  ``shadow_readout_verdict`` regresses one on the other and reports the slope,
  and holds (never promotes) when the confound is proven to be carrying the
  delta. **At 40 rows that regression cannot decide anything** -- se(slope) is
  ~0.6 against a hypothesis of 1.0 -- so the readout reports the rows needed
  rather than pretending; see ``ShadowReadout.confound_slope``.

On length bias: pooled over games (the estimator with the power to see one)
the draw rates are cur **0.5192** vs prev **0.5384**, a **1.9 pp** gap at
z = **-1.6**, so not significant -- but NOT the "identical to four decimals" an
earlier revision claimed, which was an artefact of averaging per-iteration
rates and so weighting a 12-game arm like an 80-game one.

WHY THE SHADOW READOUT NEEDS NO CONFIG CHANGE
---------------------------------------------

``gate_mode`` defaults to ``off``, and the shadow window does NOT need
``shadow``: ``decide()`` fills the per-iteration sample before the ``MODE_OFF``
check, so all four ``gate_sample_*`` columns are populated at the shipped
default. Running at ``shadow`` adds only the window aggregates -- which the
readout must not read -- and re-arms the ~252 MB anchor copy. The readout's job
is narrow and falsifiable: confirm the IN-LOOP split reproduces the offline
reconstruction, decided on the per-iteration COUNTS rather than on the delta.
See ``shadow_readout_verdict`` for why the delta cannot carry that decision and
what the rule still cannot detect. The rule ships as ONE implementation and as
a command::

    PYTHONPATH=. python3 scripts/gate_shadow_readout.py <trial>/progress.csv

which exits 0 promote / 1 hold / 2 kill / 3 the gate never ran.

Note what the metrics must therefore report. A rolling-window mean is not a
readable series -- consecutive rows share ~95% of their samples, so the sd of
the reported column understates the per-iteration sd by ~10x, and any rule
keyed to it cannot fail. ``gate_metrics`` emits ``gate_sample_*`` (THIS
iteration's independent anchored delta) alongside the window aggregates, and
the sample columns are the ones a decision rule must be written against.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
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
    "demote_step",       # 8  THIS iteration's own sample cleared demote_step_elo
    "shadow_would_demote_step",  # 9  as 8, but mode is shadow
)
# The reasons that mean "the demote rule fired", whatever the mode then did
# with it. ``gate_would_demote`` is emitted from this set -- see gate_metrics.
_WOULD_DEMOTE_REASONS = frozenset({
    "demote_regression", "shadow_would_demote", "hold_expired",
    "demote_step", "shadow_would_demote_step",
})
_REASON_CODES = {name: i for i, name in enumerate(REASONS)}

# 400 / ln(10) / 0.25 -- d(Elo)/d(score) at score 0.5. The anchored delta is a
# DIFFERENCE of two scores both sitting near 0.5 by construction (the PID pins
# them there), so the local linearisation is the honest conversion; running
# each score through the logistic separately would inject the PID's setpoint
# error into the delta.
ELO_PER_SCORE_AT_HALF = 400.0 / (math.log(10.0) * 0.25)

# Per-GAME score standard deviation, pooled over the 418-shard reconstruction.
# Used as a FLOOR on each arm's own observed variance when sizing a single
# iteration's standard error (``AnchoredSample.score_se``). Two reasons it is a
# floor and not a replacement: an arm that happened to draw every game has an
# observed variance of ZERO and would otherwise claim certainty (the L11
# lesson), and at n_prev ~ 15-38 a per-arm estimate is noisy and correlated
# with the numerator -- the same artefact the overdispersion discussion in the
# module docstring works through. max(observed, pooled) is conservative in both
# directions: it can only WIDEN the interval, so it can only make the step leg
# harder to fire, never easier.
_POOLED_GAME_SCORE_SD = 0.3447

# How many ulps of the window's own scale still count as "no spread at all".
# 4096 is ~5,000x the largest float round-trip residue observed over 40,000
# fuzzed identical-delta windows and ~1e-11 of the smallest spread two distinct
# W/D/L splits can produce, so the two populations it separates are ten orders
# of magnitude apart. See ``_window_is_degenerate``.
_DEGENERATE_SE_ULPS = 4096.0

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


def _arm_score_var(w: int, d: int, l: int) -> float:
    """Per-GAME score variance of one arm from its own W/D/L counts.

    Scores are 1 / 0.5 / 0, so this is exact rather than an approximation:
    ``E[x^2] - E[x]^2`` over the realized outcome distribution.
    """
    n = int(w) + int(d) + int(l)
    if n <= 0:
        return float("nan")
    mean = (int(w) + 0.5 * int(d)) / n
    mean_sq = (int(w) * 1.0 + int(d) * 0.25) / n
    return max(0.0, mean_sq - mean * mean)


def _window_is_degenerate(deltas: Sequence[float], se: float) -> bool:
    """Whether this window's spread is a float artefact rather than a measurement.

    THE TEST USED TO BE ``se <= 0.0`` AND IT MISSED 51.2% OF THE WINDOWS IT WAS
    WRITTEN FOR (audit G3-3). ``se`` is ``sqrt(var/n)`` with
    ``var = sum((d - mean)**2)/(n-1)``, and for K copies of one float ``d`` the
    computed ``mean = fl(K*d/K)`` is only SOMETIMES exactly ``d``: when the
    summation does not round-trip, every residual is a fraction of an ulp, the
    variance is ~1e-34 and ``se`` comes out at ~1e-17 -- strictly positive, so
    the guard passed and the gate then demoted on a CI of width zero
    (``elo_lo == elo_hi == -192.3590``), which is precisely the certainty claim
    the guard exists to refuse. Fuzzed over 40,000 realistic ``(w,d,l)`` window
    shapes, 20,466 of them produced ``se > 0`` from identical deltas.

    Two ways in, because either alone is a rule that can be fooled:

    * ``len(set(deltas)) == 1`` -- the direct statement of "a stuck counter",
      exact and independent of any tolerance;
    * ``se`` at or below a SCALE-AWARE floor. The floor is derived from the
      deltas themselves (``math.ulp`` of the largest magnitude in the window)
      rather than being a fixed literal, because a literal that is right at
      score-delta 0.14 is wrong by orders of magnitude at another scale, and a
      threshold that does not track its data is how a guard silently stops
      guarding.

    ``_DEGENERATE_SE_ULPS`` is the headroom, and
    ``test_the_degenerate_variance_floor_separates_float_noise_from_real_spread``
    pins BOTH margins: the observed round-trip residue sits ~3 orders of
    magnitude BELOW this floor, and the smallest spread REAL data can produce
    (two distinct win/draw/loss splits differ by at least ~1/(n_cur*n_prev),
    i.e. ~2e-5 in score, ~4e-6 in ``se``) sits ~10 orders of magnitude ABOVE it.
    Nothing in between is reachable, which is why a floor is safe here at all.
    """
    if len({float(d) for d in deltas}) <= 1:
        return True
    scale = max((abs(float(d)) for d in deltas), default=0.0)
    floor = _DEGENERATE_SE_ULPS * math.ulp(scale) if scale > 0.0 else 0.0
    return se <= floor


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
    # The difficulty each arm actually played at, games-weighted, straight off
    # the shards' own ``ShardMeta.opponent_wdl_regret_limit``. NaN when the
    # shards predate the field -- NEVER back-filled from the PID's bookkeeping,
    # because the whole point of recording it per shard is to be an INDEPENDENT
    # check on that bookkeeping. See "THE PID LAG DOES NOT CANCEL".
    cur_wdl_regret: float = float("nan")
    prev_wdl_regret: float = float("nan")
    # d(winrate)/d(wdl_regret) from the PID's own inverse fit
    # (``pid_regret_fit_slope``). NaN when the PID had no usable fit this
    # iteration (deadband, airbag, or fewer than 3 history points).
    regret_fit_slope: float = float("nan")

    @property
    def confound_elo(self) -> float:
        """The PID-lag offset this sample is predicted to carry, in Elo.

        ``(cur_regret - prev_regret) * dWR/dregret * dElo/dscore``. Positive
        means the difficulty gap flatters the current model. Both inputs are
        measured -- the regret gap from the shards, the slope from the PID --
        so this is a prediction that can be wrong, and comparing it against
        ``delta`` is the point.
        """
        gap = float(self.cur_wdl_regret) - float(self.prev_wdl_regret)
        return gap * float(self.regret_fit_slope) * ELO_PER_SCORE_AT_HALF

    @property
    def score_se(self) -> float:
        """One-sample standard error of ``delta``, in SCORE units.

        Independent-binomial across the two arms. That overstates the true se
        (the arms are positively correlated within an iteration -- shared
        opening book, same fleet, same PID setpoint -- which is why the
        empirical between-iteration sd of 45.6 sits BELOW the 61.1 RMS this
        form predicts). Overstating is the fail-safe direction for a leg whose
        only action is to demote.
        """
        nc, npv = self.cur_games, self.prev_games
        if nc <= 0 or npv <= 0:
            return float("nan")
        floor = _POOLED_GAME_SCORE_SD ** 2
        return math.sqrt(
            max(_arm_score_var(self.cur_w, self.cur_d, self.cur_l), floor) / nc
            + max(_arm_score_var(self.prev_w, self.prev_d, self.prev_l), floor) / npv,
        )

    def elo_hi(self, *, alpha: float) -> float:
        """One-sided upper confidence bound on THIS sample's delta, in Elo."""
        se = self.score_se
        if math.isnan(se):
            return float("nan")
        z = _Z_ONE_SIDED[round(float(alpha), 4)]
        return elo_from_score_delta(self.delta + z * se)

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
    # The single-sample upper bound the STEP leg is keyed to, carried so the
    # leg's own input is auditable rather than recomputable-in-principle.
    sample_elo_hi: float = float("nan")
    # The PID-lag offset this sample is predicted to carry, same Elo units as
    # ``sample_delta_elo`` so the two columns are directly comparable. See
    # "THE PID LAG DOES NOT CANCEL" in the module docstring.
    sample_confound_elo: float = float("nan")
    # Consecutive failed refreshes of the promoted anchor, filled in by
    # ``GateHoldController.on_decision``. Nonzero means the fallback export the
    # next hold would publish is that many iterations stale.
    anchor_refresh_failures: int = 0
    # -- DID THE BRAKE ACTUALLY ENGAGE (audit G3-4) -------------------------
    # ``acted`` says the gate DECIDED to hold. It does not say the fleet was
    # held: ``resolve_gate_hold_path`` degrades to None when the anchor is not
    # on disk and publishes normally, and before this field the two cases were
    # byte-identical in every metric row (``gate_decision=0``,
    # ``gate_reason_code=5``, ``gate_holds=1`` in BOTH) with the only witness a
    # stdout line in the Ray actor log that no csv consumer reads. A hold that
    # never reached the publish path still burns the ``max_hold_iters`` budget
    # and reported as a successful brake -- the module's own founding defect
    # (``gate_passed: 1`` at ``gate_games: 0``) one level down, moved from the
    # VERDICT to the ACTUATION.
    hold_effective: bool = False
    # Cumulative count of iterations where the gate wanted to hold and had no
    # anchor to hold on. Cumulative rather than per-iteration so a single
    # sampled row answers "has this ever happened", which is the question an
    # operator has after the fact.
    fallback_missing: int = 0
    # Iterations since the anchor was last refreshed from a promoted export,
    # NaN when unknown. Load-bearing since the refresh was restricted to
    # genuine promotes (audit G3-8): the anchor now legitimately stops moving
    # during a hold or a NOT_RUN stretch, so its AGE is the channel by which it
    # can become older than the longest hold the gate is allowed to impose.
    anchor_age_iters: float = float("nan")
    # Cumulative failed writes of ``gate_state.json`` (audit G3-6). A failed
    # write leaves the persisted window one iteration behind and the restart
    # that follows silently resumes on it.
    state_write_failures: int = 0

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

    @property
    def would_demote(self) -> bool:
        """True iff the demote rule FIRED, whatever the mode then did with it.

        ``decision`` cannot carry this: ``DECISION_PROMOTE`` is emitted for
        ``promote_no_regression`` (nothing fired), ``shadow_would_demote`` and
        ``shadow_would_demote_step`` (fired, suppressed by shadow mode) and
        ``hold_expired`` (fired, suppressed by the hold cap). In shadow mode --
        the only mode this ships anywhere near -- the interesting event is
        exactly "the gate wanted to fire", and before this property it was
        visible only as ``gate_reason_code == 6``, a number neither the yaml
        nor ``scripts/gate_shadow_readout.py`` mentioned.
        """
        return self.reason in _WOULD_DEMOTE_REASONS


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
    #     line -45:  K=8 ->  9.4%   K=24 -> 23.9%   (PR round 1 said 14% / 37%)
    #     line -25:  K=8 -> 47.1%   K=24 -> 92.5%
    #
    # Exactly, against ``scipy.stats.t``: 9.42 / 23.87 / 47.06 / 92.51. The
    # shipped ``_t_quantile`` rounds its exact small-df table UP and widens the
    # large-df asymptote, so it is deliberately conservative and returns
    # 9.42 / 23.86 / 47.06 / 92.50 -- within 0.01 point, and never on the
    # optimistic side. An earlier revision of THIS comment carried
    # 10.0% / 48.3% for the K=8 rows -- those came from the RETIRED
    # anti-conservative ``_t_quantile`` and OVERSTATE power. The module
    # docstring said so while this copy, the one sitting next to the shipped
    # constant, still quoted them.
    #
    # The cost of buying that is a false brake, and it is not measurable at
    # this window: 0 spurious holds in 8000 simulated null iterations at BOTH
    # -45 and -25 (95% upper bound 0.04%). An earlier revision of this comment
    # quoted "0.01%" as a point estimate; it is an upper bound, not a
    # measurement. Each spurious hold costs at most ``max_hold_iters`` of
    # slightly stale selfplay and never a training step, which is the right
    # trade for an alarm whose action is cheap and whose misses are expensive.
    demote_delta_elo: float = -25.0
    # The SECOND, independent line: THIS iteration's own sample, not the window
    # mean. It exists because the mean-CI rule provably cannot fire on a
    # single-iteration STEP of any magnitude -- ``elo_hi = (M/K)(t-1) > 0``,
    # with M cancelling -- and "a bad merge, a broken loss term, a mis-set LR",
    # the three things the docs name as what this gate catches, are all steps.
    #
    # -125, on the SAME false-brake budget that sized ``demote_delta_elo``.
    # One anchored sample carries 42.4 Elo of binomial se at the realized shape
    # (n_cur 197 / n_prev 38) and 64.2 at the worst shape ``min_games_per_side``
    # admits (197/15). At -125 the leg fires when the sample delta is below
    # -195 (realized) / -231 (worst), which under a null of N(0, se) is
    # 0.02 / 1.31 spurious holds per 8000 iterations -- 0.016%, inside the
    # 0.04% upper bound the window line was sized against.
    #
    # WHAT IT DOES NOT BUY: a -100 step. 50% power lands at -195 Elo and 90% at
    # -249 (realized shape). Nothing at this sample size can do better -- a
    # -100 step is under 2.5 sigma of ONE sample -- and buying it would cost a
    # false brake every few hundred iterations. Quote the leg as "a step of
    # about -200 or worse", never as "any bad merge".
    demote_step_elo: float = -125.0
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
        if self.demote_step_elo >= 0.0:
            raise ValueError(
                "gate_demote_step_elo must be negative: the step leg demotes "
                "on evidence of a one-iteration REGRESSION, never on absence "
                "of improvement"
            )
        if self.demote_step_elo > self.demote_delta_elo:
            raise ValueError(
                f"gate_demote_step_elo ({self.demote_step_elo}) must be at "
                f"least as strict as gate_demote_delta_elo "
                f"({self.demote_delta_elo}), i.e. more negative. A single "
                "sample carries 42-64 Elo of binomial noise where the window "
                "mean carries ~9, so a step line looser than the sustained "
                "line would fire on noise every few iterations"
            )
        _t_quantile(self.alpha, 8)  # raises on an unsupported alpha


def _json_float(v: float) -> float | None:
    """A float for ``json.dumps``, with the non-finite ones as ``null``.

    ``json.dumps`` writes NaN/Infinity as bare ``NaN``/``Infinity`` tokens,
    which no JSON parser outside Python accepts. ``gate_state.json`` is meant
    to be readable by hand and by other tools, so the non-finite cases become
    ``null`` and ``load_state_dict`` maps them back to NaN.
    """
    f = float(v)
    return f if math.isfinite(f) else None


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
    # Failed writes of ``gate_state.json`` this process (audit G3-6). NOT
    # persisted, for the obvious reason that the file it would be persisted in
    # is the one that failed to write -- the log line and the
    # ``gate_state_write_failures`` metric are the record.
    state_write_failures: int = 0

    def __post_init__(self) -> None:
        self.cfg.validate()

    def note_state_write_failed(self, exc: BaseException, *, path: Path) -> None:
        """The gate-state write raised. Record it LOUDLY; never re-raise.

        Same judgement as ``GateHoldController.note_anchor_refresh_failed``:
        an optional alarm must not take a training run down, but it must not be
        silent either. A failed write leaves the persisted window and hold
        latch one iteration behind, and the restart that follows resumes on
        that -- silently, until this counter existed.
        """
        self.state_write_failures += 1
        log.error(
            "promotion-gate state write FAILED %d time(s) this process (%s): "
            "%s: %s. gate_state.json is now stale, so a restart would restore "
            "an older window and hold latch than the gate is actually in.",
            self.state_write_failures, path, type(exc).__name__, exc,
        )

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
            sample_elo_hi=s.elo_hi(alpha=self.cfg.alpha),
            sample_confound_elo=s.confound_elo,
        )

    def _step_regressed(self) -> bool:
        """Did THIS iteration's own sample clear ``demote_step_elo``?

        Independent of the window in every way: it uses one sample, its own
        binomial counts and its own one-sided bound. It exists because a
        single-iteration STEP cannot move the mean-CI rule at ANY magnitude
        (``elo_hi = (M/K)(t-1) > 0``, M cancels) and a bad merge is a step.

        Fail-safe by construction: the caller only ever OR-s this into the
        demote condition, so it can add a demote and can never remove one --
        there is no path by which a True here produces a PROMOTE that a False
        would not have produced.
        """
        if not self.samples:
            return False
        s = self.samples[-1]
        if not s.usable(min_games_per_side=self.cfg.min_games_per_side):
            return False
        hi = s.elo_hi(alpha=self.cfg.alpha)
        return not math.isnan(hi) and hi < self.cfg.demote_step_elo

    def _resolve(
        self, measured: GateDecision, *, regressed: bool, step: bool,
    ) -> GateDecision:
        """Map (did the rule fire, which leg) onto a decision, honouring mode.

        One place, so shadow-mode suppression and the ``max_hold_iters`` yield
        cannot be implemented twice and drift.
        """
        if not regressed:
            return replace(measured, decision=DECISION_PROMOTE,
                           reason="promote_no_regression")
        if self.cfg.mode == MODE_SHADOW:
            return replace(
                measured, decision=DECISION_PROMOTE,
                reason="shadow_would_demote_step" if step else "shadow_would_demote",
            )
        if self.holds >= self.cfg.max_hold_iters:
            # A brake that can never release is a new way to freeze the fleet
            # on stale weights -- the exact 2026-03 failure, one level up. Past
            # the cap the gate yields and says so.
            return replace(measured, decision=DECISION_PROMOTE,
                           reason="hold_expired")
        return replace(
            measured, decision=DECISION_DEMOTE,
            reason="demote_step" if step else "demote_regression",
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
        # The step leg is evaluated BEFORE the window checks, not after. A bad
        # merge three iterations after a restart is exactly when it has to
        # work, and every window path below (short window, thin iterations,
        # degenerate variance) returns NOT_RUN -- which RELEASES the brake.
        step = self._step_regressed()

        if len(usable) < cfg.min_iters:
            if step:
                return self._resolve(
                    replace(base, iters=len(usable),
                            games_cur=games_cur, games_prev=games_prev),
                    regressed=True, step=True,
                )
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

        if _window_is_degenerate(deltas, se):
            # Every iteration produced an identical delta. Real data cannot do
            # this; a stuck counter can. Refusing beats emitting a zero-width
            # interval that claims certainty (the L11 lesson, in a new place).
            # The STEP leg still stands: it never divides by a window spread.
            degenerate = replace(
                base, iters=n, games_cur=games_cur, games_prev=games_prev,
                delta_score=mean, delta_elo=elo_from_score_delta(mean),
            )
            if step:
                return self._resolve(degenerate, regressed=True, step=True)
            return replace(
                degenerate, decision=DECISION_NOT_RUN,
                reason="degenerate_variance",
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
        window_regressed = elo_hi < cfg.demote_delta_elo
        # OR, never AND: the step leg can only ADD a demote. The window leg
        # keeps its name when both fire, because a sustained break is the more
        # informative diagnosis.
        measured = replace(
            base, iters=n,
            games_cur=games_cur, games_prev=games_prev,
            delta_score=mean, delta_elo=delta_elo,
            elo_lo=elo_lo, elo_hi=elo_hi,
        )
        return self._resolve(
            measured,
            regressed=window_regressed or step,
            step=step and not window_regressed,
        )

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
        """The gate's full state, JSON-round-trippable.

        EVERY field of ``AnchoredSample`` is written, and
        ``test_gate_state_dict_covers_every_anchored_sample_field`` enumerates
        the dataclass to prove it. Before that rule existed the three confound
        fields (``cur_wdl_regret``, ``prev_wdl_regret``, ``regret_fit_slope``)
        were recorded at runtime and dropped here, so a restart silently
        replaced measured difficulty gaps with NaN in every restored row while
        ``state_dict`` still claimed to be the gate's state -- the "accepted
        and then silently ignored" shape, applied to persistence.

        NaN is written as ``null``, not as the ``NaN`` token ``json.dumps``
        emits by default: that token is not JSON, and ``gate_state.json`` is a
        file offline readers are invited to open. ``load_state_dict`` maps
        ``null``/missing back to NaN, which is the same "no measurement" the
        producer meant -- nothing is ever back-filled from a default.
        """
        return {
            "holds": int(self.holds),
            "hold_active": bool(self.hold_active),
            "samples": [
                {
                    "iteration": int(s.iteration),
                    "cur_w": int(s.cur_w), "cur_d": int(s.cur_d), "cur_l": int(s.cur_l),
                    "prev_w": int(s.prev_w), "prev_d": int(s.prev_d), "prev_l": int(s.prev_l),
                    "cur_wdl_regret": _json_float(s.cur_wdl_regret),
                    "prev_wdl_regret": _json_float(s.prev_wdl_regret),
                    "regret_fit_slope": _json_float(s.regret_fit_slope),
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

        def _f(d: dict[str, object], key: str) -> float:
            # Missing, null, or non-numeric all mean "not measured" -> NaN.
            # A state file written before the confound fields existed lands
            # here, and NaN is exactly what the producer would have recorded.
            v = d.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return float("nan")
            return float(v)

        self.samples = [
            AnchoredSample(
                iteration=_i(d, "iteration", -1),
                cur_w=_i(d, "cur_w"), cur_d=_i(d, "cur_d"), cur_l=_i(d, "cur_l"),
                prev_w=_i(d, "prev_w"), prev_d=_i(d, "prev_d"), prev_l=_i(d, "prev_l"),
                cur_wdl_regret=_f(d, "cur_wdl_regret"),
                prev_wdl_regret=_f(d, "prev_wdl_regret"),
                regret_fit_slope=_f(d, "regret_fit_slope"),
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
  # DID THE DEMOTE RULE FIRE, whatever the mode then did with it. Distinct
  # from ``gate_decision``, which is 1 for four different situations:
  # nothing fired, shadow suppressed a window demote, shadow suppressed a
  # step demote, and the hold cap yielded. In shadow mode -- the only mode
  # this ships near -- "the gate wanted to fire" is THE event, and it used to
  # be legible only as ``gate_reason_code == 6``, a number documented
  # nowhere outside this module. Chart this column, not the reason code.
        "gate_would_demote": float(1.0 if decision.would_demote else 0.0),
  # The per-iteration sample. Independent across rows, unlike everything
  # above it -- write kill rules and shadow readouts against THESE.
        "gate_sample_delta_score": float(decision.sample_delta_score),
        "gate_sample_delta_elo": float(decision.sample_delta_elo),
        "gate_sample_games_cur": float(decision.sample_games_cur),
        "gate_sample_games_prev": float(decision.sample_games_prev),
  # The step leg's own input, and the PID-lag offset the delta beside it is
  # predicted to carry. See "THE PID LAG DOES NOT CANCEL".
        "gate_sample_elo_hi": float(decision.sample_elo_hi),
        "gate_sample_confound_elo": float(decision.sample_confound_elo),
  # Nonzero means the ~252 MB anchor copy has been failing and the export a
  # hold would publish is that many iterations stale. Used to be a silently
  # suppressed OSError.
        "gate_anchor_refresh_failures": float(decision.anchor_refresh_failures),
  # THE ACTUATION, not the verdict. 1.0 only when this decision both held AND
  # resolved to an anchor the next publish will serve; 0.0 covers "did not
  # hold" and "held with no fallback, so the fleet got the demoted net
  # anyway". ``gate_fallback_missing`` counts the second case, so the two are
  # distinguishable from the csv alone -- they used to be identical in it.
        "gate_hold_effective": float(1.0 if decision.hold_effective else 0.0),
        "gate_fallback_missing": float(decision.fallback_missing),
  # Iterations since the anchor last tracked a promoted export. The anchor is
  # deliberately NOT refreshed on a hold, a release or a NOT_RUN (G3-8), so
  # this is the number that says how far back a hold would roll the fleet.
        "gate_anchor_age_iters": float(decision.anchor_age_iters),
  # Failed ``gate_state.json`` writes. Used to be a bare
  # ``contextlib.suppress(Exception)`` two functions from the site where
  # exactly that pattern had already been fixed.
        "gate_state_write_failures": float(decision.state_write_failures),
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
        demote_step_elo=float(config.get("gate_demote_step_elo", -125.0)),
        alpha=float(config.get("gate_alpha", 0.05)),
        max_hold_iters=int(config.get("gate_max_hold_iters", 12)),
    )


# --------------------------------------------------------------------------
# The shadow readout: the promote-to-enforce decision, as CODE.
# --------------------------------------------------------------------------
# It lives here rather than in a ledger shell one-liner because a rule stated
# twice is a rule stated inconsistently, and ``usable_frac`` is where that bites:
# on the SAME reconstruction, counting only windows with a non-empty split gives
# 51/53 = 0.962 while counting every progress row gives 51/57 = 0.895, against a
# KILL line of 0.85. Seven points apart with four points of margin -- the two
# denominators do not straddle the line here, but a slightly worse window would
# have them disagree on the verdict, and neither number announces which one it
# is. ``test_usable_frac_denominator_counts_every_iteration_that_ran`` computes
# BOTH from the committed rows and pins which one ships (all rows: an iteration
# that produced no anchored sample is exactly what the leg is meant to count).
# One implementation means the ledger's number and the shipped command cannot
# diverge, and it means the rule is something a test can drive.
# ``scripts/gate_shadow_readout.py`` is that one implementation as a command.

READOUT_PROMOTE = "promote_to_enforce"
READOUT_HOLD = "hold_in_shadow"
READOUT_KILL = "kill"

# Per-leg states. The readout used to report only the legs that FIRED, so a
# reader could tell which legs failed and could not tell which ones were
# evaluated at all -- and "not evaluated" is the state the confound leg has
# been in for 109 of 109 live rows (audit G3-1). An exit code is an OR over
# axes; the axis states are what the pre-registered rule is actually about.
LEG_PASS = "PASS"
LEG_FAIL = "FAIL"
LEG_HOLD = "HOLD"
LEG_SKIPPED = "SKIPPED"
# The axis that HAS no measurement, distinct from one that was skipped by a
# guard: skipped means "this window cannot evaluate it", unmeasured means "the
# column carries no data at all", which is a producer defect, not a window
# property.
LEG_UNMEASURED = "UNMEASURED"

# Exit codes of ``scripts/gate_shadow_readout.py``, here rather than in the
# script so the ledger's pre-committed command, the CLI and the tests cannot
# drift. 3 (never ran) and 4 (no such file) belong to the script's own
# preconditions and are documented there.
READOUT_EXIT_PROMOTE = 0
READOUT_EXIT_HOLD = 1
READOUT_EXIT_KILL = 2
READOUT_EXIT_CONFOUND_UNMEASURED = 5

# Rows the confound fit needs before it says anything at all.
_CONFOUND_MIN_ROWS = 3


@dataclass(frozen=True)
class ReadoutLeg:
    """One axis of the readout, with the state it is actually in."""

    name: str
    state: str
    detail: str

    def __str__(self) -> str:
        return f"  {self.state:<10} {self.name:<24} {self.detail}"


def readout_exit_code(r: ShadowReadout) -> int:
    """The command's exit code -- per AXIS STATE, never a bare verdict.

    ⚑ AN EXIT CODE IS AN *OR* OVER AXES, and this one carried a specific trap
    (ledger, audit wave 3, K1 x G3-2). The ``prev_share`` leg false-killed on
    the production fleet; fixing that leg ALONE would make the same command
    exit 0 -- ``promote_to_enforce``, the pre-registered signal to set
    ``gate_mode: enforce`` -- while the PID-confound leg, the deciding KILL
    rule the ledger registered for exactly this promotion, was measuring
    nothing at all, because the server compactor drops
    ``ShardMeta.opponent_wdl_regret_limit`` before it ever reaches the gate.
    Every one of the 109 live rows carrying the column has
    ``gate_sample_confound_elo = NaN``.

    So a window whose confound axis has fewer than ``_CONFOUND_MIN_ROWS``
    measurements cannot exit 0 or 1. It gets its own code and its own message,
    because "promote" and "the instrument's other half is not plumbed" must not
    be the same observation. A KILL still outranks it: a failing leg is a
    plumbing fact about THIS window and is read first.
    """
    if r.verdict == READOUT_KILL:
        return READOUT_EXIT_KILL
    if not r.confound_is_measured:
        return READOUT_EXIT_CONFOUND_UNMEASURED
    return READOUT_EXIT_HOLD if r.verdict == READOUT_HOLD else READOUT_EXIT_PROMOTE


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
    it is load-bearing rather than decorative: see ``refresh_lag_seconds``. It
    was re-derived independently from the rotated ``progress.1785322501.csv``
    (iters 163-219, 51 usable rows) at **721.1 s**, and
    ``corr(time_this_iter_s, prev_share) = -0.332`` reproduces exactly.

    ONE CAVEAT ON THE PER-ROW NUMBERS, because the aggregates are much better
    than the rows they are made of. The offline reconstruction does NOT satisfy
    the pooled-count identity row by row: cross-joined against the same
    iterations' ``pid_curriculum_*``, only **6 of 51** rows have
    ``cur + prev == w + d + l`` (the residuals range -45..+49), while the
    aggregate agrees to **0.13%** (240.3 reconstructed vs 240.0 pooled). That
    is a timestamp-binning artefact of the RECONSTRUCTION -- shards are binned
    by ``generated_at_unix`` against iteration boundaries, so a shard straddling
    a boundary lands in the neighbouring row -- and NOT a loop defect; the
    in-loop split increments an arm and the pool in the same branch of
    ``_process_shard`` and cannot disagree. But it means ``mean_games_cur =
    196.8`` is a 0.1%-accurate aggregate of per-row values that are +/-45 off,
    so nothing here licenses a per-row claim about the reconstruction. The
    ``anchored_games_vs_pooled`` leg is checked against LIVE rows, where the
    identity is exact, never against this table.
    """

    mean_games_cur: float = 196.8
    mean_games_prev: float = 38.3
    prev_share: float = 0.1629
    mean_delta_elo: float = -4.33
    sd_delta_elo: float = 45.56
    n_usable: int = 53
    # Mean ``time_this_iter_s`` over the same 51 rows the counts come from.
    mean_iter_seconds: float = 721.0

    @property
    def games_per_second(self) -> float:
        """Anchored games per iteration-SECOND, DERIVED so it cannot disagree.

        This was a stored field, ``0.3411``, and it made the reference
        internally inconsistent by 4.6%::

            mean_games_cur + mean_games_prev   = 235.1
            games_per_second * mean_iter_secs  = 245.9   (4.6% apart)
            mean_games_prev                    =  38.3
            refresh_lag_seconds * games_per_s  =  40.1   (4.6% apart)

        because 0.3411 was the mean of the per-row RATES while ``mean_games_*``
        are means of per-row COUNTS, and ``E[g/s] != E[g]/E[s]``. Two legs then
        measured the same window against two different reference loops. Derived
        here it is 0.32607, and both identities close exactly --
        ``test_offline_reference_is_internally_consistent`` asserts them.

        It is still an aggregate, cadence-normalised count and NOT a physical
        selfplay rate: it includes the training phase, during which
        ``distributed_pause_selfplay_during_training`` stops selfplay. That is
        why its leg is a factor-of-2 band and not tight.
        """
        return (self.mean_games_cur + self.mean_games_prev) / self.mean_iter_seconds

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
#
# THE FLOOR IS 0.6 AND IT USED TO BE 0.4, WHICH WAS DECLARED TRUSTED AND WAS
# NOT. Review round 4 swept a healthy loop across the band and found `kill` on
# the ATTRIBUTION leg -- "your split is mis-attributing shards", the exact false
# alarm this rework exists to remove -- for every ratio in [0.40, ~0.46]:
#
#     x0.40  cur=79 prev=38 -> kill  ('prev_share 0.3248 vs expected 0.4072')
#     x0.45  cur=89 prev=38 -> kill
#     x0.47  cur=92 prev=38 -> promote_to_enforce
#
# The cause is NOT the reference's 4.6% inconsistency (fixing that leaves the
# band unchanged -- the failing leg is `prev_share`, which never read
# `games_per_second`). It is that below ~0.5x the defensible pictures of "a
# healthy loop at k x cadence" stop agreeing, by more than the leg's ENTIRE
# 0.06 tolerance. THERE ARE THREE OF THEM, not the two an earlier revision of
# this comment named as "the two defensible pictures":
#
#   (A) `pinned_rate`: total ingest scales with cadence and the refresh lag is
#       constant in seconds, so cur = rate*T - rate*L. Its SHARE prediction,
#       expected = refresh_lag/T, is exact at every k by construction, and it
#       survives `rate` itself moving with cadence, because the rate cancels
#       in the ratio.
#   (B) `pinned_prev`: cur alone scales with cadence and prev is pinned at its
#       absolute count (the shipped sweep's construction). expected is then
#       prev/(prev + cur*k), which is 1/k only to first order in prev_share.
#   (C) `pinned_counts`: cadence moves because the TRAINING phase changes
#       length, and `distributed_pause_selfplay_during_training` stops selfplay
#       for the duration, so NEITHER anchored count moves. prev_share is then
#       flat at 0.1629 while the leg expects 0.1629/k, and a healthy loop is
#       killed on the ATTRIBUTION leg at 0.60x-0.70x and again at 1.65x-3.00x
#       -- INSIDE the declared band, at both ends of it.
#
# WHICH ONE THE DATA PICKS: (A). (C) predicts corr(time_this_iter_s,
# prev_share) = 0, and the 51 reference rows give **-0.332**; that correlation
# is the entire reason the leg is keyed to refresh_lag/cadence rather than to a
# fixed share. (C)'s mechanism is also OFF in production --
# `distributed_pause_selfplay_during_training: false` in pbt2_small.yaml, so
# selfplay overlaps the training phase and the counts are not frozen by it. If
# that key is ever flipped to true, re-read this: (C) becomes the live picture
# and the band stops being safe at BOTH ends. (C) is fail-safe as it stands --
# it can only produce a kill, never a false promote -- so it is named here and
# pinned by the sweep rather than widened for.
#
# WHAT THE DATA DOES NOT SUPPORT is the COUNT half of (A) and (B): total ingest
# is nowhere near proportional to cadence. corr(time_this_iter_s, cur + prev)
# over the reference rows is **-0.053** on the 53 rows the delta leg uses and
# **+0.321** on the 51 the cadence leg uses -- the sign is not even stable, let
# alone the ~+1 that proportionality would need. So the floor must NOT be
# justified by "the A/B gap is under half the tolerance at 0.6x": that is an
# argument about count constructions the counts themselves refute. The gap
# numbers are kept (0.0000 at 1.0x, 0.0079 at 0.8x, 0.0265 at 0.6x, 0.0455 at
# 0.5x, 0.0800 at 0.4x, and never above 0.0072 out to 3.0x, which is why only
# the floor moved) because the sweep still runs both, but the reason 0.6 is
# safe does not depend on which picture is right:
#
#     raising the floor can only convert a kill-with-the-wrong-name
#     (`prev_share`, which an operator reads as "your split is broken") into a
#     kill-with-the-right-name (`cadence`, "your cadence moved"). It can never
#     convert a kill into a promote, because outside the band the share leg is
#     not evaluated AND the cadence leg fires unconditionally. The floor is a
#     LABELLING choice on the failing side, so the conservative direction is up.
#
# HOW MUCH HEADROOM 0.6x LEAVES, stated PER ROW because an earlier revision of
# this comment stated it wrong. "Production runs 620-750 s/iter, so a 0.6 floor
# (433 s/iter) excludes nothing that has ever been observed" is FALSE row by
# row: the live progress.csv holds a 355 s iteration (0.49x) and a 2364 s one
# (3.28x), and rotated files reach 266 s (0.37x) and 3234 s (4.49x). What the
# leg reads is the MEAN over the window's usable rows, and THAT stays in band:
# rolling-8 means span 0.893x-1.377x on the current file and 0.747x-2.909x on
# the oldest rotated one. The band covers the statistic the leg computes, not
# every iteration feeding it -- and a window short enough for one 355 s row to
# dominate its mean is one the `window_too_short` hold leg will not promote on
# anyway. A 0.9 floor, which the suite used to permit, would already false-kill
# an observed live window at 0.893x.
# ``test_benign_cadence_change_is_not_reported_as_an_attribution_bug`` sweeps
# (A) and (B) across the whole band and pins (C)'s false-kill range.
_CADENCE_RATIO_MIN = 0.6
_CADENCE_RATIO_MAX = 3.0

# The shadow window's pre-registered length, IN CODE rather than in prose.
# The ledger pre-registers this readout as "run it after >=40 iterations
# carrying the ``gate_sample_*`` columns", and until this constant existed that
# precondition lived ONLY in the ledger: ``shadow_readout_verdict`` happily
# returned ``promote_to_enforce`` off two rows, and every leg it checks passes
# trivially on a window that short.
#
# The failure is not hypothetical. ``harness._rotate_progress_csv_if_schema_
# changed`` starts a FRESH ``progress.csv`` whenever the reported key set
# changes -- three rotations in four days in this repo -- so an operator who
# runs the pre-committed command a few iterations after a rotation reads a
# 3-row window. Without this leg that reads ``promote_to_enforce``, and the
# deciding action is to set ``gate_mode: enforce``.
#
# A short window is NOT a kill: nothing is broken, the window is simply not
# finished. It reports ``hold_in_shadow`` with a named hold leg, so the
# operator sees WHY rather than an unexplained non-promotion.
_READOUT_MIN_ROWS = 40


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
    # -- the PID-lag confound, REPORTED and (one-sidedly) acted on ----------
    # ``confound_slope`` is the OLS slope of ``gate_sample_delta_elo`` on
    # ``gate_sample_confound_elo`` over the usable rows. 0 means the anchored
    # delta ignores the controller; 1 means it passes the controller term
    # through untouched, i.e. the gate is measuring the PID.
    #
    # ``confound_slope_se`` is what stops this from being a rule that cannot
    # fail in EITHER direction. The residual sd is ~45.6 Elo and the confound's
    # own sd is ~10-12, so se(slope) ~ 45.6/(12*sqrt(n)): about **0.60 at
    # n=40** and 0.24 at n=259. At the pre-registered 40-row window the
    # estimate cannot separate 0 from 1 and NOTHING should be concluded from
    # it; ``confound_rows_needed`` reports the n at which se falls to 0.25, so
    # the readout says how much more window it needs instead of pretending.
    n_confound: int = 0
    mean_confound_elo: float = float("nan")
    confound_slope: float = float("nan")
    confound_slope_se: float = float("nan")
    confound_rows_needed: int = 0
    failed_legs: tuple[str, ...] = ()
    # Why a window that broke no leg still did not promote. Named for the same
    # reason ``failed_legs`` is: "not promoted" with no reason attached is how
    # an operator ends up re-running the command until it says what they want.
    hold_legs: tuple[str, ...] = ()
    # EVERY axis with the state it is in, including the ones that passed and
    # the ones that were never evaluated. ``failed_legs`` / ``hold_legs`` are
    # views of this and stay for the callers that read them.
    legs: tuple[ReadoutLeg, ...] = ()
    # The reference refresh lag the attribution axis was measured against, and
    # the lag this window actually shows. Both in seconds, which is the
    # cadence-free form -- see ``OfflineReference.refresh_lag_seconds``.
    refresh_lag_seconds: float = float("nan")
    ref_refresh_lag_seconds: float = float("nan")

    @property
    def confound_is_measured(self) -> bool:
        """Whether the PID-confound axis has any measurement behind it at all.

        NOT a statement about the slope: it is a statement about whether a
        number exists to fit. Zero is the value it has had on every live row
        ever written (audit G3-1).
        """
        return self.n_confound >= _CONFOUND_MIN_ROWS

    def per_leg_report(self) -> str:
        """Every axis and its state, one per line. The thing to actually read."""
        conf_state = LEG_PASS if self.confound_is_measured else LEG_UNMEASURED
        conf_detail = (
            f"n={self.n_confound} mean={self.mean_confound_elo:.2f} "
            f"slope={self.confound_slope:.3f}+/-{self.confound_slope_se:.3f} "
            f"(needs ~{self.confound_rows_needed} rows to decide)"
            if self.confound_is_measured else
            f"n={self.n_confound} of {self.n_usable} usable rows carry "
            "gate_sample_confound_elo -- the PID-confound leg, which the ledger "
            "pre-registered as the deciding KILL rule for enabling this gate, "
            "has NO measurement. Do not read a promote off this window"
        )
        legs = [*self.legs, ReadoutLeg("confound", conf_state, conf_detail)]
        return "\n".join(str(leg) for leg in legs)

    def __str__(self) -> str:
        cad = (f"  cadence={self.mean_iter_seconds:.0f}s "
               f"expected_prev_share={self.expected_prev_share:.4f} "
               f"refresh_lag={self.refresh_lag_seconds:.0f}s"
               f"/ref{self.ref_refresh_lag_seconds:.0f}s"
               if not math.isnan(self.mean_iter_seconds) else "  cadence=unknown")
        why = (f"  FAILED: {', '.join(self.failed_legs)}" if self.failed_legs
               else f"  HOLD: {', '.join(self.hold_legs)}" if self.hold_legs
               else "")
        conf = (
            f"  confound n={self.n_confound} mean={self.mean_confound_elo:.2f} "
            f"slope={self.confound_slope:.3f}+/-{self.confound_slope_se:.3f} "
            f"(needs ~{self.confound_rows_needed} rows to decide)"
            if self.n_confound >= 3 else "  confound=unreported"
        )
        return (
            f"{self.verdict}  rows={self.n_rows} usable={self.n_usable} "
            f"({self.usable_frac:.3f})  games_cur={self.mean_games_cur:.1f} "
            f"games_prev={self.mean_games_prev:.1f} "
            f"prev_share={self.prev_share:.4f}{cad}  "
            f"delta mean={self.mean_delta_elo:.2f} sd={self.sd_delta_elo:.2f}"
            + conf + why
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

    THE WINDOW LENGTH IS A LEG, NOT A PRECONDITION IN PROSE. Fewer than
    ``_READOUT_MIN_ROWS`` rows returns ``hold_in_shadow`` with a named
    ``window_too_short`` hold leg and can never return ``promote_to_enforce``:
    every leg below passes trivially on a 3-row window, and a fresh
    ``progress.csv`` after a report-schema rotation is exactly how an operator
    gets one. See ``_READOUT_MIN_ROWS``.

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
    and a separate CADENCE leg fires by name outside a 0.6x-3.0x band instead
    of extrapolating a model that stops holding there. That floor was 0.4x and
    the band below ~0.46x false-killed a healthy loop on the ATTRIBUTION leg --
    see ``_CADENCE_RATIO_MIN`` for the two constructions that stop agreeing
    there.

    THE POOLED-COUNT IDENTITY IS THE ONE LEG WITH NO STATISTICS IN IT. By
    construction in ``_process_shard``, an accepted shard increments the cur or
    the prev bucket AND the pooled ``matching_w/d/l`` in the same branch, so::

        gate_sample_games_cur + gate_sample_games_prev
            == pid_curriculum_w + pid_curriculum_d + pid_curriculum_l

    holds exactly on every iteration that formed a sample -- both sides are
    already ``progress.csv`` columns. Shard loss between the split and the pool,
    or a game bucketed to neither arm because its sha went unrecognised, breaks
    it by an exact integer with no noise to hide in. Rows that carry no
    ``pid_curriculum_*`` skip the leg rather than passing it.

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

    Verified across the cadence band: benign 0.6x-3.0x cadence changes all
    return ``promote_to_enforce`` with no leg firing under EITHER construction
    of "healthy at k x cadence", 3.5x fires the CADENCE leg by name, and the
    coin/swap destructions are still caught 50/50 at 0.6x, 1.0x, 2.0x and 3.0x.

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
    # The identity leg, first because it involves no estimation at all: a
    # mismatch is a plumbing fact, and reading a share or a spread off a window
    # whose two arms do not add up to the pooled count it was split from is
    # reading an artefact.
    pooled_mismatch = [
        (r[0], r[1], int(r[4])) for r in usable
        if not math.isnan(r[4]) and r[0] + r[1] != int(r[4])
    ]
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
            legs=(ReadoutLeg(
                "usable_rows>=2", LEG_FAIL,
                f"{n_usable} of {n_rows} rows are usable; nothing else was "
                "evaluated",
            ),),
        )

    conf = _confound_fit([(r[2], r[5]) for r in usable])

    mean_cur = statistics.mean([r[0] for r in usable])
    mean_prev = statistics.mean([r[1] for r in usable])
    prev_share = mean_prev / (mean_cur + mean_prev)
    deltas = [r[2] for r in usable]
    mean_d, sd_d = statistics.mean(deltas), statistics.stdev(deltas)

    # -- the cadence adjustment, and its own leg ---------------------------
    secs = [r[3] for r in usable if not math.isnan(r[3]) and r[3] > 0.0]
    mean_secs = statistics.mean(secs) if secs else float("nan")
    legs: list[ReadoutLeg] = []

    def _leg(name: str, state: str, detail: str) -> None:
        legs.append(ReadoutLeg(name, state, detail))

    share_leg_valid = True
    if math.isnan(mean_secs):
        # No `time_this_iter_s` in the rows. Fall back to the raw reference
        # share and SAY SO on a failure, rather than pretending to a
        # cadence-corrected comparison that was never made.
        expected_share, share_tol, share_note = ref.prev_share, 0.06, " (cadence unknown)"
        _leg("cadence", LEG_SKIPPED, "no time_this_iter_s in the rows")
    else:
        ratio = mean_secs / ref.mean_iter_seconds
        # Outside the band the share model is not trusted, so the share leg is
        # NOT EVALUATED rather than evaluated against an extrapolation. The
        # verdict is already `kill` on the cadence leg, so nothing is missed --
        # what this buys is that the operator reads one unambiguous finding
        # ("your cadence moved") instead of that finding plus a `refresh_lag`
        # failure that means "the model you were told not to trust here says
        # your attribution is broken".
        share_leg_valid = _CADENCE_RATIO_MIN <= ratio <= _CADENCE_RATIO_MAX
        _leg(
            "cadence", LEG_PASS if share_leg_valid else LEG_FAIL,
            f"cadence {mean_secs:.0f}s is {ratio:.2f}x the reference "
            f"{ref.mean_iter_seconds:.0f}s, outside "
            f"[{_CADENCE_RATIO_MIN}, {_CADENCE_RATIO_MAX}] -- the "
            "prev-share model is not trusted here, so the refresh_lag leg "
            "was NOT evaluated; this is a CADENCE finding, NOT an "
            "attribution finding"
            if not share_leg_valid else
            f"{mean_secs:.0f}s = {ratio:.2f}x the reference "
            f"{ref.mean_iter_seconds:.0f}s, inside "
            f"[{_CADENCE_RATIO_MIN}, {_CADENCE_RATIO_MAX}]",
        )
        expected_share = ref.refresh_lag_seconds / mean_secs
        share_tol, share_note = 0.06, ""
    # The cadence-free form of what the attribution leg compares, REPORTED
    # rather than left implicit: `prev` games are the fleet's model-refresh
    # lag, and `prev_share` is that lag divided by the cadence.
    measured_lag = prev_share * mean_secs

    # -- deciding legs: attribution-sensitive, near-noiseless --------------
    if pooled_mismatch:
        c, p, pooled = pooled_mismatch[0]
        _leg(
            "anchored_games_vs_pooled", LEG_FAIL,
            f"anchored_games_vs_pooled: {len(pooled_mismatch)} of {n_usable} "
            f"rows have gate_sample_games_cur+prev != pid_curriculum_w+d+l "
            f"(first: {c}+{p}={c + p} vs {pooled}) -- the split and the pool it "
            "was split from disagree, so games were lost or bucketed to neither "
            "arm. This is an exact integer identity, not a statistic",
        )
    elif any(math.isnan(r[4]) for r in usable):
        _leg("anchored_games_vs_pooled", LEG_SKIPPED,
             "some rows carry no pid_curriculum_w/d/l")
    else:
        _leg("anchored_games_vs_pooled", LEG_PASS,
             f"cur+prev == pid_curriculum_w+d+l on {n_usable}/{n_usable} rows")
    _leg(
        "usable_frac", LEG_FAIL if frac < 0.85 else LEG_PASS,
        f"usable_frac {frac:.3f} < 0.85" if frac < 0.85
        else f"{frac:.3f} >= 0.85 ({n_usable}/{n_rows} rows)",
    )
    if not math.isnan(mean_secs):
        # Absolute counts scale with cadence, so an absolute band on them is
        # the same false alarm prev_share used to carry -- at 1.5x cadence a
        # `mean_games_cur` band of +/-60 fires on a perfectly healthy loop.
        # Compare the cadence-normalised rate instead, with a factor-2 band:
        # the reference window's own throughput regimes differ by 1.4x.
        rate = (mean_cur + mean_prev) / mean_secs
        rate_ok = 0.5 <= rate / ref.games_per_second <= 2.0
        _leg(
            "games_per_second", LEG_PASS if rate_ok else LEG_FAIL,
            f"games_per_second {rate:.4f} is "
            f"{rate / ref.games_per_second:.2f}x the reference "
            f"{ref.games_per_second} (band 0.5-2.0)",
        )
    else:
        # No cadence column: fall back to a deliberately loose absolute band,
        # because without cadence the count carries no information about
        # anything except gross breakage.
        ratio_abs = (mean_cur + mean_prev) / (
            ref.mean_games_cur + ref.mean_games_prev)
        _leg(
            "anchored games/iteration",
            LEG_PASS if 0.25 <= ratio_abs <= 4.0 else LEG_FAIL,
            f"anchored games/iteration {mean_cur + mean_prev:.1f} vs reference "
            f"{ref.mean_games_cur + ref.mean_games_prev:.1f} (band 0.25-4.0x,"
            " cadence unknown)",
        )
    # -- THE ATTRIBUTION AXIS, NAMED FOR WHAT IT MEASURES (audit G3-2) -----
    # It used to be called `prev_share` and its failure message named only the
    # share, which the module's own docstring tells an operator to read as
    # "your split is mis-attributing shards". It fires on today's production
    # csv, and that reading is NOT established:
    #
    #   * the pooled-count identity -- the one leg with no statistics in it --
    #     passes 109 of 109 live rows. That proves no game was LOST or bucketed
    #     to neither arm. It does NOT prove the attribution is right: a
    #     coin-shuffle of the labels preserves the pooled sum exactly, which is
    #     why the reshuffle control moves the SHARE and not this identity;
    #   * the audit read the mover as the fleet's own model-refresh lag,
    #     117.5 s at calibration against 166 s over the live last-40 window, at
    #     0.88x the reference cadence -- inside the trusted band, so the
    #     cadence leg stays silent and the whole discrepancy lands on a leg
    #     whose name blames the split;
    #   * an independent shard re-derivation over the same trial
    #     (`rederive_reference_from_shards`, 435 compacted shards binned
    #     against 68 iterations) gives a refresh lag of 112.7 s and a
    #     prev_share of 0.1802, i.e. essentially the calibration value. The two
    #     sources DISAGREE, and which one is right is not settled here.
    #
    # THE TWO CAUSES ARE NOT SEPARABLE FROM progress.csv, and pretending
    # otherwise would be the worse defect. `refresh_lag = prev_share *
    # cadence` is an identity, not an independent measurement, so a fleet
    # whose lag moved and a splitter that mislabels shards produce the same
    # number here. The discriminator has to come from outside the gate's own
    # split -- the shard `.zattrs` re-derivation this module ships as
    # `rederive_reference_from_shards`, which is what `OfflineReference` was
    # built from in the first place, and which must never be re-based on
    # `gate_sample_*` (a control conditioned on its own outcome cannot fail).
    #
    # So the leg keeps its threshold and its fail-closed direction EXACTLY as
    # pre-registered (the comparison below is algebraically the one it always
    # made) and changes only what it is called and what it reports: both lags
    # in seconds, and an explicit instruction not to read it as an attribution
    # finding until the reference has been re-derived at the current lag.
    share_gap = abs(prev_share - expected_share)
    if not share_leg_valid:
        _leg("refresh_lag", LEG_SKIPPED,
             "cadence outside the trusted band; not evaluated")
    else:
        _leg(
            "refresh_lag", LEG_FAIL if share_gap > share_tol else LEG_PASS,
            f"refresh_lag {measured_lag:.1f}s vs reference "
            f"{ref.refresh_lag_seconds:.1f}s "
            f"({measured_lag / ref.refresh_lag_seconds:.2f}x): prev_share "
            f"{prev_share:.4f} vs expected {expected_share:.4f} "
            f"+/-{share_tol}{share_note}"
            + (
                " -- a fleet whose model-refresh lag MOVED and a split that "
                "mis-attributes shards are the same number here; check the "
                "anchored_games_vs_pooled leg above and re-derive the "
                "reference from shards (gate_shadow_readout.py "
                "--rederive-reference) before reading this as an attribution "
                "failure"
                if share_gap > share_tol else ""
            ),
        )
    # -- deciding leg: instrument-sensitive --------------------------------
    # 4.56 is the sd of the 95%-overlapping WINDOW column. If the readout is
    # ever wired to that column again this leg is what says so.
    _leg(
        "sd_delta_elo", LEG_PASS if 20.0 < sd_d < 70.0 else LEG_FAIL,
        f"sd_delta_elo {sd_d:.2f} outside (20, 70)" if not 20.0 < sd_d < 70.0
        else f"{sd_d:.2f} inside (20, 70)",
    )
    # -- offset leg: the PID-lag bias must not dominate ---------------------
    if abs(mean_d) > 25.0:
        _leg("|mean_delta_elo|", LEG_FAIL,
             f"|mean_delta_elo| {abs(mean_d):.2f} > 25")
    elif abs(mean_d) > 15.0:
        # -- hold legs: nothing is broken, but this window cannot promote --
        _leg("|mean_delta_elo|", LEG_HOLD,
             f"|mean_delta_elo| {abs(mean_d):.2f} > 15 -- the anchored offset is "
             "larger than expected; extend the window")
    else:
        _leg("|mean_delta_elo|", LEG_PASS, f"{abs(mean_d):.2f} <= 15")

    if n_rows < _READOUT_MIN_ROWS:
        _leg(
            "window_length", LEG_HOLD,
            f"window_too_short: {n_rows} rows < the pre-registered "
            f"{_READOUT_MIN_ROWS}. Every leg above passes trivially on a "
            "window this short, and progress.csv is rotated whenever the "
            "report schema changes, so a short window means 'keep watching', "
            "never 'promote'",
        )
    else:
        _leg("window_length", LEG_PASS,
             f"{n_rows} rows >= the pre-registered {_READOUT_MIN_ROWS}")
    # The PID-lag leg. HOLD, never kill, and one-sided: it fires only when the
    # regression of the anchored delta on the predicted confound is
    # SIGNIFICANTLY positive, which is the direction that means "this gate is
    # reading the controller". It cannot manufacture a promote -- a hold leg
    # only ever subtracts one -- and it cannot fire on noise, because the
    # significance test uses the slope's own se, which is ~0.60 at the
    # pre-registered 40 rows. That also means it will essentially never fire at
    # 40 rows even if the confound IS passing through at slope 1; that is a
    # stated limitation, not a tuned threshold. Pass --last-n to read a longer
    # window once one exists.
    if (
        conf.n >= _CONFOUND_MIN_ROWS
        and not math.isnan(conf.slope_se)
        and conf.slope - CONFOUND_Z * conf.slope_se > CONFOUND_SLOPE_MAX
    ):
        _leg(
            "confound_slope", LEG_HOLD,
            f"confound_slope {conf.slope:.3f} +/- {conf.slope_se:.3f} over "
            f"{conf.n} rows is significantly above {CONFOUND_SLOPE_MAX} -- the "
            "anchored delta is tracking the PID's difficulty step, not the "
            "model. The gate must not be promoted to enforce while its own "
            "statistic is a controller output; see 'THE PID LAG DOES NOT "
            "CANCEL'",
        )

    failed = [leg.detail for leg in legs if leg.state == LEG_FAIL]
    holds = [leg.detail for leg in legs if leg.state == LEG_HOLD]
    if failed:
        verdict = READOUT_KILL
    elif holds:
        verdict = READOUT_HOLD  # extend the window rather than promote or kill
    else:
        verdict = READOUT_PROMOTE
    return ShadowReadout(
        verdict=verdict, n_rows=n_rows, n_usable=n_usable, usable_frac=frac,
        mean_games_cur=mean_cur, mean_games_prev=mean_prev,
        prev_share=prev_share, mean_delta_elo=mean_d, sd_delta_elo=sd_d,
        mean_iter_seconds=mean_secs, expected_prev_share=expected_share,
        n_confound=conf.n, mean_confound_elo=conf.mean_x,
        confound_slope=conf.slope, confound_slope_se=conf.slope_se,
        confound_rows_needed=conf.rows_needed,
        failed_legs=tuple(failed), hold_legs=tuple(holds), legs=tuple(legs),
        refresh_lag_seconds=measured_lag,
        ref_refresh_lag_seconds=ref.refresh_lag_seconds,
    )


# The confound leg's two pre-registered constants.
#
# A slope of 1.0 means the predicted PID-lag term passes into the anchored
# delta untouched. 0.5 is "half of it does", which is already enough that a
# demote verdict is as much a statement about the controller as about the
# model. The one-sided z is 1.6449 (alpha 0.05), matching the gate's own
# ``alpha`` default -- the leg fires only when the LOWER bound of the slope
# clears 0.5.
CONFOUND_SLOPE_MAX = 0.5
CONFOUND_Z = 1.6449
# The se the leg needs before the estimate can separate slope 0 from slope 1
# at ~2 sigma. ``confound_rows_needed`` is reported against this, so an
# operator reading a wide interval is told how much window would fix it.
_CONFOUND_TARGET_SE = 0.25


@dataclass(frozen=True)
class _ConfoundFit:
    n: int = 0
    mean_x: float = float("nan")
    slope: float = float("nan")
    slope_se: float = float("nan")
    rows_needed: int = 0


def _confound_fit(pairs: Sequence[tuple[float, float]]) -> _ConfoundFit:
    """OLS of the anchored delta (y) on the predicted PID-lag confound (x).

    ``pairs`` are ``(delta_elo, confound_elo)``; rows where either is NaN are
    dropped, which is what happens for every row whose shards predate
    ``ShardMeta.opponent_wdl_regret_limit`` or whose PID had no usable fit.
    A degenerate x (no spread) returns NaNs rather than dividing -- the same
    refusal ``decide()`` makes on a degenerate window.
    """
    pts = [(y, x) for y, x in pairs if not math.isnan(y) and not math.isnan(x)]
    n = len(pts)
    if n < 3:
        return _ConfoundFit(n=n)
    mx = sum(x for _, x in pts) / n
    my = sum(y for y, _ in pts) / n
    sxx = sum((x - mx) ** 2 for _, x in pts)
    if sxx <= 0.0:
        return _ConfoundFit(n=n, mean_x=mx)
    slope = sum((x - mx) * (y - my) for y, x in pts) / sxx
    resid = sum((y - my - slope * (x - mx)) ** 2 for y, x in pts)
    slope_se = math.sqrt(resid / (n - 2) / sxx) if n > 2 else float("nan")
    # se scales as 1/sqrt(n) at fixed spread, so the n that reaches the target.
    needed = (
        math.ceil(n * (slope_se / _CONFOUND_TARGET_SE) ** 2)
        if slope_se > _CONFOUND_TARGET_SE else n
    )
    return _ConfoundFit(
        n=n, mean_x=mx, slope=slope, slope_se=slope_se, rows_needed=needed,
    )


def _readout_row(
    row: Sequence[float],
) -> tuple[int, int, float, float, float, float]:
    """Normalise to (cur, prev, delta_elo, iter_seconds, pooled, confound_elo).

    Three- to five-element rows are accepted so a caller with no cadence
    column, no pooled-count column or no confound column still gets a verdict
    -- with the corresponding leg disabled and named rather than silently
    passing.
    """
    cur, prev, delta = int(row[0]), int(row[1]), float(row[2])
    secs = float(row[3]) if len(row) > 3 else float("nan")
    pooled = float(row[4]) if len(row) > 4 else float("nan")
    confound = float(row[5]) if len(row) > 5 else float("nan")
    return cur, prev, delta, secs, pooled, confound


def shadow_readout_rows_from_csv(
    rows: Iterable[dict[str, str]],
) -> list[tuple[int, int, float, float, float, float]]:
    """Pull the gate sample columns, cadence and the pooled curriculum count.

    ``time_this_iter_s`` comes along because the ``prev_share`` leg is
    evaluated against a CADENCE-ADJUSTED expectation -- see
    ``OfflineReference.refresh_lag_seconds``. A row missing it still counts;
    the adjustment is then disabled for the whole window and says so.

    ``pid_curriculum_w/d/l`` comes along because their sum is the pooled
    quantity the gate split, so ``cur + prev == w + d + l`` is an exact
    identity on every iteration that formed a sample. It is the only leg with
    no statistics in it, and it catches shard loss and unrecognised-sha
    bucketing outright. A row missing any of the three skips the leg.

    A row whose gate columns are BLANK never ran the gate and is not an
    iteration of the shadow window, so it is dropped. A row with
    ``games_cur == 0`` (the shape during a hold) DID run and is kept, with its
    NaN delta: it is unusable, and dropping it would quietly inflate
    ``usable_frac`` -- the denominator confusion worked through in
    ``test_usable_frac_denominator_counts_every_iteration_that_ran``, where the
    two readings land 7 points apart with 4 points of margin to the kill line.
    """
    out: list[tuple[int, int, float, float, float, float]] = []
    for r in rows:
        raw_c = r.get("gate_sample_games_cur")
        raw_p = r.get("gate_sample_games_prev")
        if raw_c is None or raw_p is None or raw_c == "" or raw_p == "":
            continue
        try:
            c, p = int(float(raw_c)), int(float(raw_p))
            d = float(r.get("gate_sample_delta_elo") or "nan")
            secs = float(r.get("time_this_iter_s") or "nan")
            pid = [r.get(f"pid_curriculum_{k}") for k in ("w", "d", "l")]
            pooled = (
                float(sum(int(float(v)) for v in pid if v is not None))
                if all(v not in (None, "") for v in pid) else float("nan")
            )
  # Absent on every csv written before this column existed, and NaN on
  # any row whose shards predate ShardMeta.opponent_wdl_regret_limit or
  # whose PID had no usable inverse fit. The confound leg drops those
  # rows and reports its own n, rather than treating "unknown" as 0 --
  # a zero would read as "the controller contributed nothing".
            confound = float(r.get("gate_sample_confound_elo") or "nan")
        except ValueError:
            continue
        out.append((c, p, d, secs, pooled, confound))
    return out


def shadow_readout_from_csv(
    path: str | Path, *, min_games_per_side: int = 15, last_n: int = 40,
    ref: OfflineReference = OFFLINE,
) -> ShadowReadout:
    """The shadow window's ONE deciding rule, applied to a ``progress.csv``.

    ``scripts/gate_shadow_readout.py`` is the command form of exactly this
    call -- one implementation, so the ledger's pre-committed command, the CLI
    an operator actually runs and the tests cannot drift apart. Before it
    existed this function was documented as "the ledger's ONE deciding command,
    as a function" while nothing invoked it: a yardstick pre-committed as an
    exact command that could not be run as one.
    """
    with Path(path).open(newline="") as fh:
        rows = shadow_readout_rows_from_csv(csv.DictReader(fh))
    return shadow_readout_verdict(
        rows, min_games_per_side=min_games_per_side, last_n=last_n, ref=ref,
    )


# --------------------------------------------------------------------------
# Re-deriving the reference the attribution axis is measured against.
# --------------------------------------------------------------------------
# WHY THIS EXISTS, AND WHY IT MAY NOT READ THE GATE'S OWN COLUMNS.
#
# ``OfflineReference`` was built by binning processed shard ``.zattrs`` against
# ``progress.csv`` and splitting each iteration's curriculum games by
# ``model_sha256``. That is an INDEPENDENT reconstruction: it never touches the
# in-loop splitter the readout then checks. Re-deriving the reference from
# ``gate_sample_games_cur/prev`` instead would be much easier and would destroy
# the leg -- the control would be conditioned on its own outcome, and a
# splitter defect present in both windows would cancel exactly. So the
# re-derivation reads shards, like the original, and nothing here consumes a
# ``gate_*`` column.
#
# It is a REPORT, never an input to a verdict: it prints constants for an
# operator to record in the ledger and paste into ``OfflineReference`` at
# restart prep. A reference that moves silently under a running rule is a ruler
# that moved with the model.
@dataclass(frozen=True)
class ShardArm:
    """One shard's curriculum result, as the compactor writes it to ``.zattrs``."""

    generated_at_unix: float
    model_step: int
    model_sha256: str
    wins: int
    draws: int
    losses: int


@dataclass(frozen=True)
class RederivedReference:
    """What a shard window says the reference constants are TODAY."""

    n_iterations: int
    n_usable: int
    n_shards: int
    mean_games_cur: float
    mean_games_prev: float
    prev_share: float
    mean_iter_seconds: float
    refresh_lag_seconds: float
    mean_delta_elo: float
    sd_delta_elo: float

    def as_offline_reference_source(self) -> str:
        """The constants, formatted as the dataclass body they replace."""
        return (
            "    mean_games_cur: float = "
            f"{self.mean_games_cur:.1f}\n"
            f"    mean_games_prev: float = {self.mean_games_prev:.1f}\n"
            f"    prev_share: float = {self.prev_share:.4f}\n"
            f"    mean_delta_elo: float = {self.mean_delta_elo:.2f}\n"
            f"    sd_delta_elo: float = {self.sd_delta_elo:.2f}\n"
            f"    n_usable: int = {self.n_usable}\n"
            f"    mean_iter_seconds: float = {self.mean_iter_seconds:.1f}\n"
        )


def rederive_reference_from_shards(
    iterations: Sequence[tuple[float, float]],
    shards: Sequence[ShardArm],
) -> RederivedReference:
    """Rebuild the reference constants from shard metadata and iteration bins.

    ``iterations`` are ``(end_unix, iter_seconds)`` pairs -- ``timestamp`` and
    ``time_this_iter_s`` from ``progress.csv``, which is used ONLY for the bin
    edges and the cadence, never for a gate column. ``shards`` are the
    ``.zattrs`` of the processed/compacted shards.

    Within each bin the arms are split the way the loop splits them: the
    HIGHEST ``model_step`` present is the current model, the next-highest is
    the previous one, and anything older is dropped -- the in-loop
    ``accepted_model_shas`` set has exactly two elements. The split is by
    ``model_step`` rather than by publish order because the step is stamped on
    the shard itself, so the reconstruction needs nothing from the publisher.

    A bin with only one model present contributes no anchored sample, which is
    the same rule ``AnchoredSample.usable`` applies.
    """
    by_bin: list[list[ShardArm]] = [[] for _ in iterations]
    edges = [(end - max(0.0, secs), end) for end, secs in iterations]
    for sh in shards:
        for i, (lo, hi) in enumerate(edges):
            if lo <= sh.generated_at_unix < hi:
                by_bin[i].append(sh)
                break

    samples: list[tuple[AnchoredSample, float]] = []
    for (_, secs), bucket in zip(iterations, by_bin, strict=True):
        if not bucket:
            continue
        steps = sorted({s.model_step for s in bucket}, reverse=True)
        cur_step = steps[0]
        prev_step = steps[1] if len(steps) > 1 else None
        cur = [s for s in bucket if s.model_step == cur_step]
        prev = [s for s in bucket if s.model_step == prev_step]
        samples.append((
            AnchoredSample(
                iteration=-1,
                cur_w=sum(s.wins for s in cur), cur_d=sum(s.draws for s in cur),
                cur_l=sum(s.losses for s in cur),
                prev_w=sum(s.wins for s in prev), prev_d=sum(s.draws for s in prev),
                prev_l=sum(s.losses for s in prev),
            ),
            float(secs),
        ))

    usable = [(s, t) for s, t in samples if s.cur_games and s.prev_games]
    if len(usable) < 2:
        nan = float("nan")
        return RederivedReference(
            n_iterations=len(iterations), n_usable=len(usable),
            n_shards=len(shards), mean_games_cur=nan, mean_games_prev=nan,
            prev_share=nan, mean_iter_seconds=nan, refresh_lag_seconds=nan,
            mean_delta_elo=nan, sd_delta_elo=nan,
        )
    mean_cur = statistics.mean([s.cur_games for s, _ in usable])
    mean_prev = statistics.mean([s.prev_games for s, _ in usable])
    share = (
        sum(s.prev_games for s, _ in usable)
        / sum(s.cur_games + s.prev_games for s, _ in usable)
    )
    cadence = statistics.mean([t for _, t in usable])
    deltas = [elo_from_score_delta(s.delta) for s, _ in usable]
    return RederivedReference(
        n_iterations=len(iterations), n_usable=len(usable), n_shards=len(shards),
        mean_games_cur=mean_cur, mean_games_prev=mean_prev, prev_share=share,
        mean_iter_seconds=cadence, refresh_lag_seconds=share * cadence,
        mean_delta_elo=statistics.mean(deltas),
        sd_delta_elo=statistics.stdev(deltas),
    )


def read_shard_arms(shard_root: Path) -> list[ShardArm]:
    """Every ``.zattrs`` under *shard_root*, as :class:`ShardArm` records.

    ``wins`` / ``draws`` / ``losses`` on a shard are the CURRICULUM (vs
    Stockfish) results -- they sum to ``curriculum_games`` -- which is the
    population the anchored A/B is drawn from. Shards with no curriculum game
    are dropped rather than counted as zeros.
    """
    out: list[ShardArm] = []
    for attrs_path in sorted(shard_root.rglob(".zattrs")):
        try:
            raw = json.loads(attrs_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        w, d, l = (int(raw.get(k, 0) or 0) for k in ("wins", "draws", "losses"))
        if w + d + l <= 0:
            continue
        gen, step = raw.get("generated_at_unix"), raw.get("model_step")
        if not isinstance(gen, (int, float)) or not isinstance(step, (int, float)):
            continue
        out.append(ShardArm(
            generated_at_unix=float(gen), model_step=int(step),
            model_sha256=str(raw.get("model_sha256", "")),
            wins=w, draws=d, losses=l,
        ))
    return out


def read_iteration_bins(path: str | Path) -> list[tuple[float, float]]:
    """``(end_unix, iter_seconds)`` per row of a ``progress.csv``."""
    out: list[tuple[float, float]] = []
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                end = float(row.get("timestamp") or "nan")
                secs = float(row.get("time_this_iter_s") or "nan")
            except ValueError:
                continue
            if math.isnan(end) or math.isnan(secs) or secs <= 0.0:
                continue
            out.append((end, secs))
    return out


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
    # Consecutive failed anchor refreshes. The refresh copies ~252 MB into
    # ``durable_dir`` every iteration and used to fail SILENTLY inside a bare
    # ``contextlib.suppress(OSError)``, so a full disk froze the fallback at an
    # arbitrarily old export with nothing to read. Counted rather than raised
    # (an optional alarm must not kill a training run), reported as
    # ``gate_anchor_refresh_failures``, and load-bearing via
    # ``anchor_is_trustworthy``.
    #
    # IT IS PERSISTED (audit G3-5). It used to live only in this object, so the
    # restart that ``train.sh`` auto-resume performs after an ENOSPC crash reset
    # it to 0 and re-armed an arbitrarily old anchor as trustworthy -- the one
    # thing standing between a stale ~252 MB export and the whole selfplay
    # fleet, reconstructed from nothing.
    anchor_refresh_failures: int = 0
    # Cumulative iterations where the gate wanted to hold and had no anchor to
    # hold on, so it published normally. Reported as ``gate_fallback_missing``.
    fallback_missing: int = 0
    # The verdict the last completed iteration produced, or None before the
    # gate has issued one. The publish at the TOP of iteration N is governed by
    # iteration N-1's verdict, so this is exactly the decision that says
    # whether this publish is held -- and, since G3-8, whether the anchor may
    # be refreshed from it.
    last_decision: GateDecision | None = None
    # The iteration the anchor on disk was written at, read from its stamp
    # sidecar. None when there is no anchor or no stamp.
    anchor_iteration: int | None = None
    # Whether the anchor on disk carries a stamp this run is willing to serve.
    # False after a stamp that is missing (a file from before stamping, or from
    # a previous era left behind by an off -> on cycle) or older than the
    # longest hold the gate is allowed to impose.
    anchor_stamp_ok: bool = True
    # Iterations since the anchor was last refreshed, filled by ``on_decision``
    # when it is told the current iteration. NaN when unknown.
    anchor_age_iters: float = float("nan")

    @classmethod
    def create(
        cls,
        gate: PromotionGate,
        *,
        durable_dir: Path,
        state: Mapping[str, object] | None = None,
        current_iteration: int | None = None,
    ) -> GateHoldController:
        """Build at trial startup, restoring a hold that survived a restart.

        The anchor path is None while the gate is off, which is what keeps a
        disabled feature from copying a ~252 MB export every iteration. A
        restored hold with no anchor on disk is released rather than trusted.

        ``state`` is the ``hold`` sub-object of ``gate_state.json``; its
        counters are restored rather than reset, because a counter that dies
        with the process cannot make a persistent failure legible across the
        restart that failure causes.

        THE ANCHOR MUST PROVE ITS AGE (audit G3-5). ``.is_file()`` was the
        whole admission test, so any ``gate_promoted_model.pt`` left in
        ``durable_dir`` -- by an ENOSPC episode, or by an earlier era of the
        run, since nothing deletes the file when ``gate_mode`` goes back to
        ``off`` -- was re-armed as the model a hold would serve to the entire
        fleet. The stamp written beside it names the iteration, step, trial and
        source sha it was copied from, and an anchor whose stamp is absent or
        older than ``gate_max_hold_iters`` is refused here. It costs nothing:
        the next genuine promote rewrites both file and stamp.
        """
        promoted = (
            durable_dir / "gate_promoted_model.pt" if gate.cfg.mode != MODE_OFF else None
        )
        stamp = read_anchor_stamp(promoted)
        anchor_iteration = None if stamp is None else stamp.iteration
        age = (
            float(current_iteration - anchor_iteration)
            if (current_iteration is not None and anchor_iteration is not None)
            else float("nan")
        )
        on_disk = promoted is not None and promoted.is_file()
        # No anchor on disk is not a distrusted anchor: there is nothing to
        # distrust, and the first promoted export creates both file and stamp.
        stamp_ok = not on_disk or (
            stamp is not None
            and not (not math.isnan(age) and abs(age) > gate.cfg.max_hold_iters)
        )
        if on_disk and not stamp_ok:
            log.error(
                "promotion-gate anchor %s is present but its stamp is %s -- "
                "refusing to serve it. An unstamped or stale anchor is exactly "
                "what an off -> on cycle leaves behind, and a hold would "
                "publish it to the whole fleet. The next promoted export "
                "re-creates it.",
                promoted,
                "MISSING" if stamp is None else f"{age:.0f} iterations old",
            )
        hold = promoted if (gate.hold_active and on_disk and stamp_ok) else None
        if gate.hold_active:
            print(
                f"[gate] resuming with the fleet HELD: holds={gate.holds} "
                f"fallback={'present' if hold else 'MISSING (releasing)'}",
                flush=True,
            )
        if gate.hold_active and hold is None:
            log.warning(
                "promotion gate resumed with hold_active but no usable anchor "
                "on disk; releasing the hold",
            )
            gate.hold_active = False
        ctrl = cls(
            gate=gate, promoted_model_path=promoted, hold_path=hold,
            anchor_iteration=anchor_iteration, anchor_stamp_ok=stamp_ok,
            anchor_age_iters=age,
        )
        ctrl.load_state_dict(state)
        return ctrl

    def state_dict(self) -> dict[str, object]:
        """The controller's own persistent counters.

        Separate from ``PromotionGate.state_dict`` because they belong to the
        actuator, not to the window. Both go into ``gate_state.json``.
        """
        return {
            "anchor_refresh_failures": int(self.anchor_refresh_failures),
            "fallback_missing": int(self.fallback_missing),
        }

    def load_state_dict(self, state: Mapping[str, object] | None) -> None:
        if not state:
            return
        for key in ("anchor_refresh_failures", "fallback_missing"):
            raw = state.get(key)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                setattr(self, key, max(0, int(raw)))
        if self.anchor_refresh_failures:
            log.warning(
                "promotion-gate anchor refresh was failing when this trial last "
                "stopped: %d consecutive failures restored from gate_state.json "
                "(gate_max_hold_iters=%d). The counter is NOT reset by a "
                "restart.",
                self.anchor_refresh_failures, self.gate.cfg.max_hold_iters,
            )

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

    def note_anchor_refreshed(self, *, iteration: int | None = None) -> None:
        """The promoted anchor was rewritten from this iteration's export."""
        if self.anchor_refresh_failures:
            log.warning(
                "promotion-gate anchor refresh recovered after %d consecutive "
                "failures", self.anchor_refresh_failures,
            )
        self.anchor_refresh_failures = 0
        # A freshly written anchor carries a freshly written stamp, so whatever
        # was wrong with the old one is gone.
        self.anchor_stamp_ok = True
        if iteration is not None:
            self.anchor_iteration = int(iteration)
            self.anchor_age_iters = 0.0

    def anchor_refresh_is_due(self) -> bool:
        """Whether this publish may overwrite the promoted anchor.

        THE ANCHOR IS "LAST PROMOTED", NOT "LAST PUBLISHED" (audit G3-8). The
        condition used to be "we are not holding this iteration", which is true
        on a ``DECISION_NOT_RUN`` (short window, thin iterations, degenerate
        variance), true on a shadow-suppressed demote, and true on the RELEASE
        iteration -- where the fallback was overwritten with the very weights
        the preceding hold existed to keep off the fleet. "Holds the selfplay
        fleet on the last promoted export" then meant "on last iteration's
        export", and the documented partial braking was a one-iteration-lag
        filter that re-poisoned itself at every release.

        So the refresh needs a POSITIVE verdict: the gate's most recent
        decision must be a promote that no demote leg fired on. Two exceptions,
        both bounded:

        * no anchor on disk yet -- there is nothing to protect and a gate with
          no fallback cannot brake at all, so the first publish creates it;
        * no verdict yet (startup, before the first ``on_decision``) -- the
          controller has no verdict to withhold on. This lasts exactly one
          iteration per process.

        The cost is that the anchor legitimately stops moving for as long as
        the gate declines to promote, which is why ``anchor_is_trustworthy``
        now also refuses on AGE: an anchor older than ``gate_max_hold_iters``
        would roll the fleet back further than the mechanism is designed to,
        whether it got old by failing to copy or by never being promoted.
        """
        if self.promoted_model_path is None:
            return False
        if not self.promoted_model_path.is_file():
            return True
        d = self.last_decision
        if d is None:
            return True
        return d.decision == DECISION_PROMOTE and not d.would_demote

    def note_anchor_refresh_failed(self, exc: BaseException) -> None:
        """The refresh raised. Record it LOUDLY; never re-raise.

        A raise here would take a training run down to protect an optional
        alarm. A silent pass is what this replaces: the counter is what makes a
        persistent failure legible, and ``anchor_is_trustworthy`` is what makes
        it consequential rather than merely reported.
        """
        self.anchor_refresh_failures += 1
        log.error(
            "promotion-gate anchor refresh FAILED %d consecutive time(s) "
            "(%s): %s. The fallback export is now stale by that many "
            "iterations; past gate_max_hold_iters (%d) the gate will stop "
            "braking rather than publish it.",
            self.anchor_refresh_failures, self.promoted_model_path, exc,
            self.gate.cfg.max_hold_iters,
        )

    @property
    def anchor_is_trustworthy(self) -> bool:
        """Whether the promoted export is recent enough to publish to the fleet.

        A hold serves the anchor to every worker. An anchor that has not been
        refreshed for more than ``gate_max_hold_iters`` iterations is older
        than the longest hold the gate is allowed to impose, so serving it is
        strictly worse than not braking at all -- it would roll the fleet back
        further than the mechanism is designed to. Refusing is the same
        judgement ``_publish_distributed_trial_state``'s ``FileNotFoundError``
        makes about a MISSING anchor, taken on the fail-open side because a
        stale anchor, unlike a missing one, cannot be distinguished from a good
        one at publish time.

        THREE WAYS AN ANCHOR GOES BAD, and all three are the same judgement:
        the copy keeps failing (``anchor_refresh_failures``), the file on disk
        cannot prove when it was written (``anchor_stamp_ok``), or it was
        written too long ago (``anchor_age_iters``). The last one only became
        reachable when the refresh was restricted to genuine promotes (G3-8) --
        before that the anchor could not be old, only wrong.
        """
        if not self.anchor_stamp_ok:
            return False
        if (
            not math.isnan(self.anchor_age_iters)
            and abs(self.anchor_age_iters) > self.gate.cfg.max_hold_iters
        ):
            return False
        return self.anchor_refresh_failures <= self.gate.cfg.max_hold_iters

    def on_decision(
        self, decision: GateDecision, *, iteration: int | None = None,
    ) -> GateDecision:
        """Apply a verdict to the NEXT publish, and report the anchor's health.

        Returns the decision with the actuator's own state filled in --
        ``anchor_refresh_failures``, ``hold_effective``, ``fallback_missing``
        and ``anchor_age_iters`` -- so those numbers reach ``progress.csv``
        instead of living only in this object. ``hold_effective`` is the one
        that answers the question the verdict cannot: a hold that resolved to
        no fallback published the demoted net anyway, and used to be
        byte-identical in the metrics to one that braked (G3-4).
        """
        if iteration is not None and self.anchor_iteration is not None:
            self.anchor_age_iters = float(int(iteration) - self.anchor_iteration)
        if decision.acted and not self.anchor_is_trustworthy:
            log.error(
                "promotion gate wanted to HOLD but the promoted anchor is not "
                "trustworthy (%d consecutive refresh failures, stamp_ok=%s, "
                "age=%.0f iterations, gate_max_hold_iters=%d); publishing "
                "normally rather than rolling the fleet back to an export that "
                "old",
                self.anchor_refresh_failures, self.anchor_stamp_ok,
                self.anchor_age_iters, self.gate.cfg.max_hold_iters,
            )
            self.hold_path = None
        else:
            self.hold_path = resolve_gate_hold_path(
                decision, gate_promoted_model_path=self.promoted_model_path,
            )
        if decision.acted and self.hold_path is None:
            self.fallback_missing += 1
        self.last_decision = decision
        return replace(
            decision,
            anchor_refresh_failures=int(self.anchor_refresh_failures),
            hold_effective=bool(decision.acted and self.hold_path is not None),
            fallback_missing=int(self.fallback_missing),
            anchor_age_iters=float(self.anchor_age_iters),
        )

    def on_aborted_iteration(self) -> None:
        """Age an active hold on an iteration that produced no verdict."""
        if not self.gate.advance_hold_without_decision():
            self.hold_path = None


@dataclass(frozen=True)
class AnchorStamp:
    """Provenance of the promoted anchor on disk.

    The anchor is a ~252 MB copy of an export, and before this stamp existed
    the only thing recorded about it was that it was a file. ``.is_file()`` was
    the whole admission test, so an anchor from a previous era of the run --
    left behind because nothing deletes it when ``gate_mode`` returns to
    ``off`` -- was re-armed as trustworthy by the next ``off -> on`` cycle
    (audit G3-5). What a hold serves to the fleet is now something the
    controller can name: which iteration and step it came from, which trial
    wrote it, and the sha of the bytes it was copied from.
    """

    iteration: int
    trainer_step: int
    model_sha256: str
    trial_id: str
    written_at_unix: float


def anchor_stamp_path(anchor: Path) -> Path:
    """The stamp sidecar for an anchor file."""
    return anchor.with_suffix(anchor.suffix + ".stamp.json")


def write_anchor_stamp(
    anchor: Path,
    *,
    iteration: int,
    trainer_step: int,
    model_sha256: str,
    trial_id: str,
) -> None:
    """Record what the anchor at *anchor* is. Raises like any other write.

    Deliberately NOT suppressed: it is written by the same ``try`` block as the
    copy it describes, so a stamp that fails to write counts as a failed
    refresh. A stamped anchor whose stamp is a lie would be worse than an
    unstamped one, and an unstamped one is refused.
    """
    anchor_stamp_path(anchor).write_text(
        json.dumps(
            {
                "iteration": int(iteration),
                "trainer_step": int(trainer_step),
                "model_sha256": str(model_sha256),
                "trial_id": str(trial_id),
                "written_at_unix": float(time.time()),
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )


def read_anchor_stamp(anchor: Path | None) -> AnchorStamp | None:
    """The anchor's stamp, or None when it is absent or unreadable.

    Unreadable and absent are the same answer on purpose: both mean "this file
    cannot say when it was written", and the consequence is the same refusal.
    """
    if anchor is None:
        return None
    path = anchor_stamp_path(anchor)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    it, step = raw.get("iteration"), raw.get("trainer_step")
    if not isinstance(it, (int, float)) or isinstance(it, bool):
        return None
    return AnchorStamp(
        iteration=int(it),
        trainer_step=int(step) if isinstance(step, (int, float)) else -1,
        model_sha256=str(raw.get("model_sha256", "")),
        trial_id=str(raw.get("trial_id", "")),
        written_at_unix=float(raw.get("written_at_unix", 0.0) or 0.0),
    )


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
