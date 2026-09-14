# PFS-inspired frontier advancement: cheap-first experiment plan

Status: **Analytical examples completed; no DeepFin search, training or deployment
experiment run.** Updated September 14, 2026 against main
`c7b88eaf9da6f2c8a23675b52cac60275f726e36`.

This amendment replaces the initial planning order in PR #755. The
[original proposal](https://github.com/jjoshua2/DeepFin/blob/2651018bbf0c45cf4883c0e6099f583c7ded60a4/docs/experiments/2026-09-14-pfs-frontier-advancement.md)
remains immutable. No real chess treatment outcomes were collected before this
revision. The original 1,024-position confirmation and arena are optional later
stages, not prerequisites for finding out whether a small idea works.

## Decision

**Do not start by implementing the full U/V selector or a general trace system.**
First run a root-admission census, inspect actual first-cut evidence, and choose
one existing-knob control or tiny intervention. Prefer experiments that answer
whether the failure is **admission, early elimination, missing interior evidence,
or failure to use evidence once found**. These are not interchangeable.

[Probabilistic Focal Search](https://arxiv.org/html/2609.10584v1) motivates allocating
work to a boundary that blocks useful progress. Its minimum-f lower-bound
certificate and bounded-suboptimality guarantees do not carry over to DeepFin.
A shallow visit floor is not a minimax bound or a statistical confidence interval.
Neither the reported node savings nor an optimal chess dose is assumed here.

## What the follow-up arithmetic changed

Runnable evidence: [analytic_screen.py](evidence/pfs-frontier/analytic_screen.py),
[analytic_screen.json](evidence/pfs-frontier/analytic_screen.json).
Eight standard-library arithmetic tests passed locally, including exhaustive
finite-population subset checks through population size eight. The script does
**not** import or execute DeepFin. Its schedule formulas are transcribed from the
pinned helpers below; its other examples are constructed counterexamples, not
chess positions, prevalence estimates, native parity tests or strength results.

### 1. Interior probes cannot rescue every early elimination

For an ordinary fresh root with 40 searched legal moves, no shortcuts or reused
subtrees, the existing candidate-width and first-round formulas give:

| Simulations | Requested topk | Realized width | First-round visits per candidate | Maximum interior opportunities per candidate before first cut |
| --- | --- | --- | --- | --- |
| 25 | 32 | 13 | 1 | 0 |
| 100 | 32 | 32 | 1 | 0 |
| 100 | 16 | 16 | 1 | 0 |
| 100 | 8 | 8 | 4 | 3 |
| 400 | 32 | 32 | 2 | 1 |
| 400 | 16 | 16 | 6 | 5 |

The first visit evaluates the root child; it cannot also descend through that
child's unevaluated replies. Thus at 100/topk32, 16 candidates are eliminated
before interior probing can investigate their replies. Even the upper bounds
above require fresh feedback: batching duplicate unevaluated leaves can reduce
them further. Carry, transpositions and terminal shortcuts can change the case;
measure those separately rather than asserting this is the live population.

A 5% interior null at this budget therefore says nothing about rescuing those
first-cut casualties. **topk8 versus topk32 at B=100 is a cheap breadth/depth
package contrast worth testing when admission coverage permits it.** topk16
still has only one first-round visit at B=100; it is not a first-cut reply test.
At B=400, topk16 versus topk32 is another possible contrast. Pick one, not a grid.

### 2. Nominal dose is not useful exposure

For five uniformly selected marks among 100 simulation indices and a *fixed*
set of k eligible opportunities, the probability of hitting that set is
`1 - C(100-k,5)/C(100,5)`: 5% for k=1, 14.4001% for k=3, and 41.6248% for k=10.
This is a combinatorial illustration, not a counterfactual search model; treatment
can change future opportunities. A marked first evaluation has no interior choice.
Report scheduled marks, eligible opportunities, actual diversions, distinct new
evaluations and affected root candidates separately.

The original "first eligible node in the next two levels" rule can repeatedly
choose level one while level two gets nothing. Use a preselected depth stratum
and record depth-specific no-ops instead. Do not silently fall back to the
shallower level. Avoid a fixed every-20th schedule that can alias with candidate
ordering; a position-seeded sample without replacement gives an exact quota.

### 3. Discovery can be diluted by prior backups

Illustration in a single root POV: nine observations of +0.6 followed by one
non-proven -0.9 observation give a mean of +0.45. Against a competitor at +0.2,
three such negative observations still give +0.225; four give +0.13846.
This is arithmetic, not a claim about the halving winner: priors, normalization,
solved propagation and subsequent ordinary descent also matter. It motivates
logging whether a new refutation is followed up and changes the root evidence,
not merely whether the search visited it once.

### 4. A bad third move can change the leading pair through normalization

With all moves observed, fixed Q scale 48, log-priors up to a common constant
`[0,-1,-20]` and Q `[.20,.21,.19]`, normalized scores are `[24,47,-20]`: B wins.
Changing only C's Q to -.90 gives approximately `[47.568,47,-20]`: A wins.
A and B's Q estimates did not change. Exploring C changed their ranking by
changing the shared min-max range.

This is not automatically a bug or a bad decision. It is an alternative causal
explanation for an apparent frontier benefit. On a handful of saved snapshots,
recompute the score with the old range and, separately, the old imputation/scale
statistics held fixed. Treat these as **local score attribution**, not a replay
of the search that would have happened under another algorithm. Off-trace leaf
values are unknown; a baseline trace cannot evaluate arbitrary new search paths.

## Cheapest useful tests, in order

### A. Admission ceiling: zero additional leaf search

Use existing per-root logits and reference scores if available; otherwise budget
one root forward per selected history, not a search per knob. Reconstruct the
actual searched-legal support, candidate cap, realized K and original Gumbels.
For noise-free search, any positive temperature preserves strict logit ranks;
root warming cannot admit an excluded action except through numerical ties or
floors. Noisy admission is different: changing temperature changes the relative
weight of the fixed Gumbel draws. Keep root draws fixed for comparisons.

For zero-temperature final play with a fixed candidate set A and complete frozen
reference scores over searched support L, compute
`R_admission = max(v_ref over L) - max(v_ref over A)`.
This is a lower bound on that search's *reference regret*, not an optimal-chess
bound. Compare it with actual regret when already banked. Little excluded regret
means a root-admission rescue has little headroom on that panel; a large floor
means interior-only modifications cannot remove it. Count tied reference best
sets, not an arbitrary one of their moves. Missing scores or a changed/late-entry
candidate set invalidate this bound; positive-temperature policy sampling is
also outside this fixed-candidate played-action claim.

On the same root bank, compare K=8/16/32 coverage at B=100/400 with the actual cap.
This chooses the *next contrast*, not the best engine. Root logits alone cannot
measure the benefit of deeper replies, elimination rescue or an unseen scout.

### B. Existing-knob pilot before writing a new selector

Freeze one checkpoint and a development panel of at most 64 replayable histories:
32 random positions and up to 32 separately reported baseline-selected difficult
cases. Prefer one position per source-qualified game and exclude future holdout
games. Keep solved/tablebase cases as separate controls. The challenge slice
must not estimate population prevalence or determine population strength.

Choose the one contrast supported by A and the schedule:

| Evidence | Next cheap contrast |
| --- | --- |
| Good moves usually in top8; first-cut reply evidence absent at B=100 | Current shape versus topk8 at equal 100 simulations |
| B=400 first-cut evidence thin; top16 coverage adequate | Current shape versus topk16 at equal 400 simulations |
| Real duplicate evaluations or stale batches dominate | Existing supported batching/virtual-visit alternative at fixed search parameters |
| Important reference moves outside current K | Feasible wider-K control or a separate costed admission scout, not interior repair |

A K change also changes halving and realized Q scale; label it a breadth/depth
package, not an isolated exploration rule. Fix all other settings and count real
work. Changing batch handling is likewise not proof of the PFS hypothesis.

Collect just root initialization, each completed halving boundary and the final
state: candidate identities, Q/visits/pending counts, score components, chosen
move, and cost. Trace detailed paths on at most 16 selected failures. Do not add
all-node trajectory storage to the hot path before it is needed; perform verbose
trace and throughput measurements separately.

### C. A small causal probe, only for the diagnosed bottleneck

On at most 16 explained failures, ask whether the *right* intervention could
help under the proposed budget. An offline oracle may nominate a known missed
continuation from a separate baseline/deeper-reference analysis, but the actual
frozen evaluator must evaluate it, preserve history/POV and pay every visit.
Do not inject reference Q, silently change legal support or use it as deployable
policy. Keep oracle-assisted rows out of the performance result.

If even this targeted exposure cannot affect the decision before its deadline,
a blind 5% heuristic is a poor next bet at that depth and budget. Inspect whether
the problem is a spent elimination deadline, inaccurate evaluation, score range,
backup dilution or insufficient follow-up. An oracle probe can still fail for
reasons other than impossibility; this is a diagnostic, not a theorem or global
kill rule. If ordinary allocation already follows the new evidence promptly,
there is no reason to add a follow-up mechanism.

## Candidate changes worth keeping small

Select **one** after A-C; do not combine these into the initial treatment.

**Depth-stratified, one-deviation exploration.** Retain the SH-forced root action;
change at most one interior choice per marked simulation. Preselect whether the
choice is at the opponent-reply node or the next mover node. Keep the existing
solved-result priority and ignore pending edges as independent evidence. Unknown
children stay unknown; observed plausible children can use the original proposed
N<4 / Q-gap<=.10 heuristic, but this is not a confidence interval. A wrong early
Q can exclude the real winner, so do not describe that gate as safe pruning.

**A policy-rank challenger.** A strong policy may make one wrong turn rather than
be wrong everywhere. Instead of choosing uniformly from all unknown children,
rotate probes between a small next-priority band and the remaining tail, with
no-repeat reservations. Compare against uniform unknown-child selection at the
same depth and dose. Freeze the bands before confirmation. This is a proposed
one-deviation search heuristic, not established superiority or a new PFS theorem;
never make tail moves permanently unreachable.

**Probe-and-follow-up packets.** If actual traces show isolated discovery followed
by dilution, compare isolated probes with short packets under the same reserve
(e.g. one scout plus at most two follow-ups). Trigger follow-ups only on a
predeclared observable surprise in the local mover's Q, not SF labels. Consume
the next already-scheduled visits for that root candidate before its elimination
deadline; do not add visits, revive an eliminated action or change root width.
When the schedule cannot supply them, report an unavailable packet. Compare
selection rules at equal packet shape so depth alone is not called targeting.

**Pre-cut audit / root scouting.** Choose these instead when the boundary is at
the root. A pre-cut audit needs actual completed evaluations before rescoring;
an admission scout needs an explicit route into the surviving roster. Preserve
original priors/Gumbels unless their change is the named intervention. Charge
scouting and rechecks to B; define feasible width/replacement/budget arithmetic
before collecting outcomes. Do not simply pass B-minus-reserve and accidentally
change automatic K. At zero move temperature, extra all-legal policy mass does
not by itself rescue an action outside the halving survivor set.

## Controls, decisions and budget

For a newly implemented selector compare unchanged search, an untargeted control,
and the proposed treatment on the **same backend and checkpoint**. Match scheduled
marks, selected depth strata, solved/pending safeguards and packet shape. Log
realized eligibility because states diverge. Random within the same U/V eligible
set tests *ranking within that set*; random over all safe alternatives tests the
larger eligibility-plus-ranking package. Do not conflate those two controls.

Start with one dose, 5% of B, and at most 64 development positions at one primary
budget. Three arms at B=100 require at most 19,200 scheduled simulations, plus
root evaluations and separately counted reference/diagnostic work. At most 16
positions may receive a secondary B=400 check. No training is needed. Before
launch, use a 16-position timing canary to freeze a feasible size with a cap of
30 CPU-minutes and one GPU-hour total for this pilot, including failed attempts,
compilation, warm-up and diagnostic searches. Do not appropriate a running job's
resources. Budgets are proposals, not launched or scheduled work.

Report paired regret changes, corrections/regressions, severe mistakes, exact
outcomes, reference coverage and actual cost. For ordinary complete non-mate
reference rows use `min(1000,max(0,best_cp-played_cp))`, in root mover POV, alongside
uncapped tails and errors >300cp. Categorical mate/tablebase cases stay separate.
Incomplete score coverage is not zero regret. A reference used to select challenge
cases is not an independent confirmation source. SF agreement alone cannot
establish a proposed SF blind-spot improvement.

A promising but uncertain pilot can justify a **bounded** next experiment; it is
not a strength claim. More exposed moves alone is not success. A treatment that
beats baseline but not random has no demonstrated structural advantage; neither
has it proved that the mechanisms are equal. An unchanged output with few actual
eligible probes is a dose/deadline null, not a general falsification. Check costs
and regression cases before choosing further work, not only p-values.

Only after a useful pilot, freeze the selector and a fresh game-disjoint holdout.
The original proposal's 1,024 random positions, paired seeds and 100/400 budgets
remain an optional confirmation design; size it for observed paired variance and
cost before seeing confirmation outcomes. Retain the original stricter
"supported mechanism" label only if both primary regret-contrast 95% intervals
favor treatment, the baseline mean reduction is at least 5%, and the specified
400-simulation mean/tail safeguards hold. Those safeguards were +2 capped cp and
+1 percentage point severe-error noninferiority margins. Uncertainty should be
clustered by source-qualified game (opening pair for arenas), not seed/row.

A separately budgeted fixed-pair native arena supports playing-strength claims;
matched time tests deployment cost. A 512-game arena is a possible follow-up,
not an automatic job or universal evidence requirement. A Python prototype's
latency does not predict a future native port. Preserve raw observations, declare
read points, and do not expand until the preferred sign appears. See
[evaluation](../eval_protocol.md) and [development](../development.md).

## Source map and implementation invariants

At the reviewed revision, use these actual consumers:

- `mcts/gumbel_c.py` under `chess_anti_engine/`: `_realized_candidate_width`,
  root preparation/top-m, final `remaining[0]`, `leaf_buffer_rows`, duplicate
  diagnostics and `_start_gumbel_trailing_args` (sequential and two-group paths).
- `mcts/gumbel.py`: `halving_*`, `_completed_q_transform`,
  `_select_full_gumbel_child`, `apply_policy_temp` and backend guards.
- `mcts/_mcts_tree.c`: `tree_gumbel_select_child`, `tree_gumbel_collect_leaf`,
  `gss_begin_round` and `gss_score_and_halve`; update `_mcts_tree.pyi` for API changes.
- `scripts/search_gain_probe.py`: reusable score/headroom ideas, not authority for
  current settings or history completeness. Its dated shapes are not live reads.

Default full-tree Gumbel is improved-policy visit-deficit selection, not ordinary
PUCT. Preserve the normal `wdl` head and parent/child POV; `policy_sf` is a P1
opponent-reply head, not a current-root teacher. Do not change targets or training
in this test. Gumbel's target is not simply visit proportions, so a future RL
study must examine actual rebuilt targets rather than blindly subtract probe
visits as though that removed their training effects.

Disabled treatment and opt-in counters must preserve visits, survivor, policy,
value and RNG state. Split probe/root-noise streams. Test deadlines, exact quota,
no-op/empty eligibility, legal/searchmoves support, solved precedence, repetition,
pending reservations and exception cleanup; do not call twofold repetition an
exact chess draw. Preserve complete input history and source/game identity.

Reject unsupported backends/settings rather than silently dropping a flag. Check
actual loaded extension hash/ABI/`GSS_HALVING_REV`, candidate cap, batch sizing and
virtual-visit mode. Buffer exhaustion may absorb work rather than just slow it.
Count real NN rows, distinct expansions, duplicates, padding, terminal/TT work
and wall time separately. Tree carry and real batching need later explicit tests.
The UCI matched-time path may use different walkers; verify the modified consumer
actually runs. Merging code neither deploys a native extension nor changes a
running process. No live YAML, active checkout, corpus or worker is changed here.

## Completed readout and limits

The committed JSON records the script SHA256, source revision, local Python
version and eight passing arithmetic tests. The results establish the finite
examples above, not their frequency or importance in DeepFin. No model checkpoint,
real reference bank, native extension or chess engine ran. No whole-repository
validation or independent review is claimed; author self-review only.

**Next concrete task:** on the owning host, reuse a qualified root-logit/reference
bank for A, then run the one small B contrast its admission/evidence result
supports. Build a new selector only when those cheaper tests leave useful headroom.
