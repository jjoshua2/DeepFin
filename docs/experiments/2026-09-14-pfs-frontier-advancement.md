# PFS-inspired frontier advancement in DeepFin

Status: **PROPOSED; no implementation, benchmark, training, or deployment run.**

Prepared 2026-09-14 against main commit
`c7b88eaf9da6f2c8a23675b52cac60275f726e36`. This is a search experiment plan,
not permission to replace an active bootstrap run, change live YAML, or rebuild
an extension used by another process. Numerical doses and decision thresholds
below are prospective choices, not measured optima. Before execution, freeze the
checkpoint, input panel, effective search settings, implementation and resource
allocation in a run manifest. Keep amendments and readouts in this record.

## Recommendation

First establish **which boundary is withholding useful evidence**. Then test one
small, frozen-checkpoint intervention against both unchanged search and a
same-budget random-exploration control. Prefer a shallow interior intervention
that leaves the root candidate list and sequential-halving schedule unchanged.
Only implement root admission or elimination rescue if the diagnostic identifies
those as the relevant bottleneck. Do not build three search engines or train a
new network before finding a search-quality signal.

A useful positive result is not simply more breadth or a changed move. It is:

> A targeted diversion exposes information the normal search misses, that
> information changes the eventual decision beneficially, and the targeted
> diversion outperforms equally budgeted untargeted exploration.

### What transfers from the paper

[Probabilistic Focal Search, arXiv:2609.10584v1](https://arxiv.org/html/2609.10584v1)
alternates guided FOCAL expansion with minimum-f OPEN expansion. The latter can
advance a certified lower bound and admit useful nodes. The reported large gains
are conditional on this admission bottleneck; they are not a generic MCTS result.
Its main probability settings include p = 0.6, 0.7, 0.8: 20–40% secondary choices,
not evidence that a 1–5% chess intervention is sufficient.

DeepFin has neither the paper's admissible f-min frontier nor its bounded-cost
certificate. Here, a visit floor or plausible-Q gate is a scheduling heuristic,
not a confidence interval, minimax proof or monotone global lower bound. Use the
name **PFS-inspired frontier advancement**, not an implementation of PFS. No
90% reduction or theoretical guarantee is forecast.

## Current search: where an intervention can actually work

Source anchors at the reviewed revision:

| Boundary / existing behavior | Source and consequence |
| --- | --- |
| Root admission | `chess_anti_engine/mcts/gumbel_c.py`, root preparation and top-m selection in `run_gumbel_root_many_c`: rank `log(prior) + gumbel`; width is bounded by topk, searched legal count and the budget-dependent cap. Merely creating all legal root edges does not admit them to halving. |
| Root elimination | Native `gss_begin_round`, `gss_score_and_halve` and associated state in `chess_anti_engine/mcts/_mcts_tree.c`. Surviving candidates receive scheduled work; a generic least-visited-root rule can duplicate work halving already does. |
| Interior allocation | Native `tree_gumbel_collect_leaf` and `tree_gumbel_select_child`. With default `full_tree=True`, descent matches visits to the improved policy, approximately `argmax(pi_improved - N_eff/(1 + total_N_eff))`; this is not ordinary PUCT. A completely unvisited child frontier initially follows highest prior. |
| Deterministic played move | The final `gumbel_c.py` path takes the halving survivor `remaining[0]` at zero move temperature. The returned improved policy is a separate object; changing a noncandidate's Q or target mass alone need not change the move played. |
| Score headroom | `scripts/search_gain_probe.py` already measures log-prior gaps, root/descent sigma scale, changes from the prior and their reference quality. Insufficient value-score range is different from insufficient information. |
| Realized search shape | `GumbelConfig`, `PLAY_SEARCH_DEFAULTS`, backend guards and root-scale helpers in `chess_anti_engine/mcts/gumbel.py`. Play and selfplay intentionally differ. PUCT-only `c_puct`/FPU knobs are not useful positive controls for default Gumbel descent. |

Use these files as implementation starting points, not a promise that an existing
CLI already exposes the proposed controls. `scripts/search_gain_probe.py` offers
reusable measurement ideas, but its dated built-in shapes and any FEN/plane-only
inputs do not establish current, full-history production equivalence.

### Four diagnoses, with different remedies

1. **Admission starvation:** an independently preferred root move never entered
   the active candidate set. Interior probing cannot repair this.
2. **Premature elimination:** it entered, but was discarded before enough useful
   evidence arrived. Work after final elimination cannot help unless re-entry is
   explicitly implemented.
3. **Interior starvation:** an admitted branch repeatedly follows the same
   replies and misses a low-prior continuation or refutation. This is the first
   intervention proposed below.
4. **Not an evidence bottleneck:** the value estimates remain wrong despite
   added exposure; a prior/Gumbel gap is beyond the available score range; or
   nominal simulations are repeated/stale evaluations. Classify these separately
   rather than claiming all search failures support frontier advancement.

These labels can overlap. Log the earliest observable blocker and later blockers,
and retain an unknown category. A high-budget winner is diagnostic evidence, not
proof of the objectively best move.

## Stage 0: a bounded, nonintervening diagnosis

### Panel and immutable baseline

Select one completed, recoverable checkpoint from the current bootstrap work,
then pin its bytes and architecture. Both arms must use exactly that checkpoint;
do not compare a search change plus V50/V100/Downside or other teacher changes
against a different network. Select a second existing checkpoint only for a later
transfer check, not another training run.

Prefer banked, held-out positions with replayable histories and a usable reference
score bank. Qualify original source, shard, game and ply identities; exclude
cross-split games and duplicate input histories. A FEN reconstructed from planes
is insufficient when the evaluator consumes history. If no suitable bank exists,
report a data-readiness blocker and propose a bounded collection amendment rather
than silently launching a full corpus relabel.

Proposed sampling:

- A 64-position timing and instrumentation canary, never counted as holdout.
- 256 development positions for debugging and choosing the bottleneck.
- 1,024 randomly selected, game-disjoint confirmation positions, frozen before
  examining treatment results.
- An optional, separate challenge panel of up to 256 baseline-selected low-prior
  or late-discovery cases. Select it without treatment results. It estimates
  mechanism effects in that slice, not population strength or prevalence.

Baseline diagnosis uses native budgets 25, 100 and 400. A 1,600-budget baseline
on at most 128 development positions can expose delayed discoveries. Do not
confuse separately initialized budget runs with checkpoints of one continuous
search: halving schedules and initial widths can change with the total budget.
Within-search traces must come from a single fixed-total-budget invocation.

First run cold trees to isolate the allocation mechanism. The later confirmation
must separately exercise actual tree carry and batching before deployment claims.
Use noise-free diagnostics for intelligible traces, then repeat the registered
comparison with the intended noisy shape and paired position-level seeds. Do not
average play and training shapes into one result.

### Bank the missing evidence

At root initialization, every halving boundary, probe opportunity and final
selection, record the minimum useful raw fields:

- Complete searched legal support, action mapping, priors and original per-action
  Gumbel draws; admitted/remaining/eliminated sets and elimination round.
- Actual visited-child Q in the parent's point of view; distinguish it from
  imputed completed-Q/FPU. Record completed visits, pending reservations and
  solved status separately. Bank both root and local descent score scales.
- Chosen paths, first real child evaluation, subsequent value updates and the
  root candidates affected by each update. Record the actual played action,
  halving survivor, improved-policy vector and returned value separately.
- Attempted/completed simulations, fresh expansions, actual neural rows,
  history-aware duplicate rows, padded rows, cache/TT reuse, terminal backups,
  buffer exhaustion, batch sizes, latency and peak memory.

Instrumentation must preserve outputs and RNG state when disabled and when
recording baseline traces. A trace may measure first exposure or when the
current survivor joins a fixed reference-best set; it cannot observe the future
or retrospectively call an unsampled alternative bad. Record failures to discover
within the cap as censored, not missing successes.

### Reference quality and coverage

Reuse a frozen Stockfish reference only with actual per-move coverage. Score
played actions against the common observed reference best set; missing narrowed
roster scores remain unavailable, never zero regret. Report coverage jointly
for all arms and the excluded fraction. If an arm selects unscored moves more
frequently, a complete-case ranking alone cannot decide it; obtain a separately
budgeted common adjudication or mark the comparison incomplete.

Use the existing calibrated score domain rather than inventing an alternate WDL
conversion. For the prospective cp screen, primary regret is the non-mate score
shortfall from the best covered reference move: `min(1000, max(0, best_cp -
played_cp))`, with both scores in the root mover's perspective. Also report
uncapped regret and errors above 300 cp. Handle mate scores and exact tablebase
outcomes categorically. Confirmed tactical wins/refutations and eventual paired
games are independent checks against merely copying the SF ruler. A same-network
deeper search is a second diagnostic, not ground truth. Do not claim to resolve
an SF blind spot using SF agreement alone.

## Experiment I: shallow interior frontier repair

This is the default first candidate **only if Stage 0 finds interior headroom**.
Its advantage is a small causal surface: root admission, root halving, priors,
value transform, final selection rule and network remain unchanged.

### Prospective rule

Let B be the ordinary simulation budget. Mark at most
`R = floor(0.05 * B)` existing simulation opportunities using a separate,
position-keyed schedule. Those opportunities replace normal descent choices;
they are not extra simulations. Spread them through the fixed budget rather
than triggering with probability 0.05 independently at every depth.

On a marked simulation, retain the root action forced by sequential halving.
Permit **at most one** alternative edge choice at the first eligible expanded
node in the next two interior levels, then resume ordinary descent. If no
eligible node exists, perform ordinary search and log a no-op; do not bank an
unbounded probe debt for later.

At that node, derive Q only from actual completed observations, in that node's
mover perspective. Exclude solved alternatives according to the existing search
rules and avoid edges already reserved for pending work. Let:

- U be unsolved children with no completed visit: their quality is **unknown**.
- V be observed, unsolved children with fewer than four completed visits and
  Q no more than 0.10 below the best observed eligible child Q.

The 0.10 threshold is in `Q = P(win) - P(loss)` units, not centipawns.
Alternate marked opportunities between discovery from U and refinement from V;
fall back to the other set when the requested set is empty. Within V choose the
lowest completed visit count, breaking ties with a stable, prior-independent
position/action hash. Within U use that hash order. Exclude the ordinary choice
when assessing whether the operation is a real diversion. If no alternative
remains, it is a no-op. All-empty or all-solved cases retain existing behavior.

This floor is a cheap allocation heuristic, **not four independent samples**.
Repeated backups, correlated leaves, virtual visits and duplicate evaluator rows
must not be described as independent confidence. Keep the visit-based primary
rule simple and use the trace to determine whether the diverted work actually
adds distinct information. A unique-expansion-based selector would be a later,
separate ablation, not silently substituted during the holdout.

The unknown set is essential: requiring every candidate to look good under an
imputed Q would prevent the experiment from discovering precisely the moves it
is intended to test. Likewise, a hard shallow-Q gate may reject a genuinely good
move with a pessimistic early evaluation; any such failures belong in the readout.

### Controls and what they isolate

| Arm | Behavior |
| --- | --- |
| B: unchanged | Frozen checkpoint and current explicitly pinned search shape. |
| R: untargeted diversion | Same marked simulation opportunities and maximum depth/one-edge limit, but choose a random safe alternative at the eligible node, without the low-visit/plausible-Q priority. |
| S: structural diversion | The U/V frontier rule above. |

All arms retain total B, root candidate width, halving schedule, root noise draws,
move temperature, model and evaluator settings. Root-noise and probe RNG streams
must be independent so adding a probe cannot shift future Gumbel draws. Runtime
states naturally diverge, so report realized diversions and evaluator cost rather
than asserting identical useful work. Run order should be interleaved in matched
blocks, with the same warm-up and cache policy.

S versus B measures the package. S versus R asks whether targeted frontier work
beats generic exploration; it does not independently identify the contribution
of every U/V component. Only after a positive development signal add one simple
challenger, such as 20% softer search-prior temperature, or an eligible-set-random
ablation. Avoid a full temperature × dose × depth × value-scale grid.

Start with the single 5% dose. If it produces a development signal, compare 1%
and 10% on development data only and freeze one dose before confirmation. A tiny
dose can fail even if a larger intervention would help; a negative result is
conditional on the tested dose, checkpoint and horizon.

## Conditional alternatives: do not combine in the first test

### Experiment A: root admission scouting

Select this instead if Stage 0 finds a material population of useful root moves
outside the admitted set. Test a bounded scout of excluded **searched-legal**
actions, not merely more descent inside admitted candidates.

Reserve a fixed number of evaluations within B. Pick scout actions using a
precommitted rank-band/permutation rule independent of the SF reference. Actually
evaluate their successors with the same frozen evaluator and correct history/POV;
an unvisited move has no free shallow evaluation. Compare a value-selected scout
challenger with a random challenger under the same scouting cost. A simple wider
`topk` baseline at equal B is mandatory here.

The implementation must give the challenger a real route into the active halving
roster, with current priors, its original Gumbel draw and an explicit replacement
or width rule. At zero temperature, changing only the all-legal improved-policy
vector cannot rescue a move excluded from `remaining[0]`. Define the replacement
and remaining-budget schedule in a small implementation preregistration before
collecting outcomes; this paragraph is not an executable scheduler.

Do not assume `topk=all_legal` is supported. Check loaded `GSS_MAX_CANDS`, the
budget-dependent width and the evaluator buffer cap. All-legal child evaluation
is an informative costed control, not free information for the treatment.

### Experiment H: a pre-elimination audit

Select this instead if helpful moves are admitted but discarded with insufficient
useful evidence. At the first halving boundary, wait for pending evaluations,
then use a reserved part of B to recheck one at-risk, plausible candidate before
recomputing the ordinary halving scores. Compare selection by observed-Q/visit
criteria against a random at-risk candidate with the same audit budget. Avoid
resurrection after final selection in the first version.

Preserve the baseline initial width explicitly. Simply running halving with
`B - R` can change its automatic width at low B and confound the comparison.
Require a budget-feasible fixed-width schedule or declare that case ineligible;
never silently change K or overspend. A Q-score gap that cannot overtake the
prior/Gumbel gap at this budget is a different problem, not evidence that another
visit will repair the elimination.

The open AVI proposal [PR #635](https://github.com/jjoshua2/DeepFin/pull/635)
contains adjacent all-legal successor/POV ideas. It was not on main at this review;
do not assume its helper exists or include its target rewrite/training changes.
The archived budget-controller work [PR #505](https://github.com/jjoshua2/DeepFin/pull/505)
also is not a dependency. This plan reallocates work *within one position* and
does not rank all future holdout positions to enforce a transductive quota.

## Confirmation, metrics and stop rules

The proposed primary confirmation is B/R/S on the same frozen checkpoint at
**100 simulations**. A matched **400-simulation** comparison checks whether a
low-budget gain reverses as ordinary search becomes more capable. Use two paired,
position-keyed stochastic seeds for confirmation; aggregate seeds per position
and cluster uncertainty by source-qualified game, not by individual row or seed.
This count is a chosen cost/variance compromise, not a universal evidence gate.

On the random confirmation panel, precommit these deciding statistics:

1. Mean capped reference-regret differences S minus B and S minus R at B=100,
   with paired game-cluster bootstrap 95% intervals (5,000 resamples).
2. The same differences at B=400, reference coverage, error rate above 300 cp,
   categorical mate/tablebase outcomes, and uncapped regret tails.
3. On the separate challenge panel: exposure and survival of the fixed reference
   best set, capped first-discovery work, final success at the cap and corrections
   versus regressions. Include all eligible cases, not only successful solves.
4. Actual evaluator rows, fresh expanded nodes, duplicate and padding shares,
   completed work, wall time and peak memory. A node-count win alone is not a
   compute win. Discovery that does not change the eventual decision is not a
   playing-quality win.

A **supported mechanism screen** requires both B=100 mean-regret contrast
intervals to favor S (upper bound below zero), at least a 5% point reduction
against B when B's regret is nonzero, and trace evidence of exposure → new
information → improved final choice. For extension to an arena, also require
that the B=400 upper confidence bound on mean degradation is at most +2 capped
cp and the upper bound on the >300cp error-rate increase is at most +1 percentage
point. These are prospective practical margins, not universal chess thresholds.
No transfer claim follows from this one checkpoint.

A directionally favorable but uncertain result is **INCONCLUSIVE**, not a pass
or a forced rejection of the general idea. S beating B but not R supports generic
exploration, not the claimed structural advantage. More exposure without quality
improvement points toward value error or waste. Worse regret/tails rejects the
specific candidate at this horizon. Missing reference coverage, changed budgets,
silent backend fallbacks, buffer absorption, corrupt mappings or failed jobs are
**INVALID/INCOMPLETE**, not clean nulls. Do not increase the panel until a preferred
sign appears. Any larger study needs a prospective amendment and fresh holdout.

### Strength and real-cost follow-up

Only a promising mechanism justifies native implementation/qualification and a
paired playing screen. Proposed first arena: the same checkpoint on both sides,
400 simulations, 256 color-swapped opening pairs (512 games), independent of the
position-panel selection. Use the repository's pentanomial/pair analysis and bank
complete raw games. Keep the intended root noise and move-temperature settings
explicit; an identical-engine deterministic self-reference can be uninformative.

Report Elo and its interval without treating a 512-game unresolved result as
proof of no small benefit. Predeclare the fixed pair cap and read point. Then,
only if warranted, use native matched-time evaluation at the intended deployment
latency, with a separately pinned pair cap. A Python prototype's overhead does
not predict native performance, and native fixed-simulation gains need not survive
padding, dispatch or concurrency costs.

Reuse `scripts/arena_standard.py` where its real consumer supports the new
per-side settings. Its matched-time path uses UCI subprocesses; an in-process
Gumbel override does not configure those subprocesses. Verify that the chosen
UCI mode actually executes the modified search path. If it uses PUCT walkers
instead, select a supported matching mode or register a separate port/contrast;
a silently inactive Gumbel flag is not a negative result. Add/test the appropriate
surface before giving executable launch commands. Follow [the evaluation
protocol](../eval_protocol.md), preserve complete pairs and resume banked games
rather than rerolling completed results.

## Compute proposal and execution boundaries

These are upper bounds to propose at launch, not reservations or work started:

| Stage | Suggested cap | Decision |
| --- | --- | --- |
| Manifest/bank audit and 64-position canary | 30 CPU-minutes plus at most 1 GPU-hour | Establish coverage, headroom, throughput and a safe panel size. |
| Diagnosis and one candidate's development screen | At most 2 additional GPU-hours | Choose I, A or H; stop if no useful headroom or signal. |
| Frozen confirmation | At most 2 additional GPU-hours | Read the precommitted holdout once; stop at the cap even if uncertain. |
| Conditional native playing screen | Separate budget of at most 4 GPU-hours | No launch until native semantics and preceding evidence justify it. |

Use one GPU only after checking the owning host's actual active jobs, available
VRAM and CPU/disk load; a percentage allocator limit is not a reservation. Retain
the ordinary two-thread cap for shared CPU analysis/tests. Compilation, warm-up,
failed attempts and any new reference evaluation count against the relevant
budget. These caps may not fit the suggested panel on the actual host: use the
canary to select and freeze a feasible panel before reading holdout outcomes, or
record a resource blocker. Do not borrow a running bootstrap job's resources.

Keep artifacts in a fresh run namespace with a manifest, raw trace bank, reference
coverage report, realized-cost counters and readout. This experiment changes no
checkpoint, optimizer, replay or live config; recovery is stopping its isolated
process and retaining the immutable observations. Resume only matching partial
artifacts, never overwrite a completed bank. No automated future run is requested.

## Small implementation sequence and required behavior tests

**PR 1: instrument and diagnose.** Add opt-in trace hooks around root admission,
halving and interior selection, reusing `search_gain_probe.py` scoring/mapping
where valid. A possible new `scripts/frontier_advancement_probe.py` is a proposed
name, not an existing command. Do not introduce a generalized learned controller.

**PR 2: one research selector.** Implement the selected mechanism with all defaults
off. For a Python prototype, compare B/R/S on that same backend and explicitly
reject unsupported C dispatch; do not compare Python treatment with native
baseline as if only the selector changed. Port only a promising rule to native
code. Update `_mcts_tree.pyi` and the shared `_start_gumbel_trailing_args` contract
when the native signature changes, including both sequential and two-group paths.

**PR 3: qualified evaluation/adoption, only with signal.** Wire the actual per-side
arena/UCI consumer and later the worker path. Native parity, carried-tree tests,
real-cost measurements and an independent review precede any live proposal.
Merging source is not adopting a loaded native extension. Train against changed
selfplay targets only under a later, separate target-generation/training study.

Minimum behavioral checks, not source-spelling tests:

- Disabled intervention preserves visits, survivor, policy, value and RNG state;
  recording diagnostics does not change those outputs. Marked opportunities are
  no more than floor(epsilon * B), with at most one alternative edge per marked
  simulation and no extra budget. Invalid settings fail before state mutation.
- A small deterministic evaluator exposes a low-prior refutation only after the
  structural probe, and the refutation changes the final root decision. A second
  fixture contains an attractive but bad shallow alternative and checks that it
  cannot monopolize the budget. Test an all-empty eligibility no-op as well.
- Preserve mate/solved-win priority, losing-move exclusions, legal/searchmoves
  support, draw semantics and parent/child WDL sign. Never promote a twofold
  heuristic draw to an exact chess result. Do not use the P1 `policy_sf` head as
  current-root guidance; MCTS still consumes the normal `wdl` head.
- Exercise small budgets and width boundaries, pending work, duplicate leaves,
  virtual-loss cleanup on exceptions, batch/pipeline paths and tree carry. An
  admission arm must prove that the rescued action can actually become the
  zero-temperature halving survivor, not just gain target probability.
- Record loaded extension path/hash, ABI and `GSS_HALVING_REV`, compiled candidate
  cap, target_batch, virtual-loss mode, active board count and evaluator capacity.
  `leaf_buffer_rows` must fit over the actual board-count/pipeline range: an
  undersized buffer can absorb work rather than merely reduce throughput.
- Validate real config → dispatcher → native/worker behavior, with nonzero
  intervention counters in a controlled fixture. Unsupported backends fail
  loudly. Run the relevant focused tests and
  `tests/test_e2e_smoke.py -k gumbel_selfplay_smoke` when worker wiring changes;
  follow [development validation](../development.md) for the actual change scope.

## Readout template

Append, without rewriting the preregistration: exact code/checkpoint/panel hashes;
loaded native/evaluator identity; effective search shape; completed scope and
resource cost; coverage/missingness; raw B/R/S paired contrasts and uncertainty;
which predicted boundary advanced; corrections versus regressions; tail/cost
results; classification against the precommitted rule; independent review or
explicit self-review limitation; and one next decision.

Current readout: **not run**. This documentation was prepared from repository
source and the paper. It contains no engine correctness, speed, Elo, or training
result. The archived experiments and open proposals cited above remain separate.
