# Fresh whole-game initial-policy anchor pilot

Status: **scientific design selected; final roles/configurations and execution
bindings pending. No new scientific games, training or playing results.** This
record is prospective and grants no phase execution authority. The first-week
strength program remains active, and A39 remains the donor.

## Question and prior evidence

On one newly collected frozen-A39 selfplay bank, does an initial-policy anchor
produce conditional playing utility beyond matched unanchored learning and a
quarter-rate control? Report independent heldout WDL health and apply a fixed
severe-damage veto. This does not identify an optimal RL formula or establish
strength against unhandicapped multicore Rofchade, Stockfish or Lc0.

Earlier [outcome-policy continuation](2026-11-13-core-outcome-policy-baseline.md),
[low-rate continuation](2026-11-14-low-lr-on-policy-continuation.md) and
[retained-search WDL](2026-11-15-retained-search-mainwdl-512-pilot.md) studies did
not establish donor utility. Their banks, endpoints and unused roles remain
protected. This new bank, batch and dose define a different conditional regime;
old studies are not pooled or reopened.

The [software readout](2026-11-16-frozen-policy-anchor-software-qualification.md)
established a synthetic full494 causal gradient/save check, not a research
coefficient. The strict HOLD family is implemented in
[PR #1352](https://github.com/jjoshua2/DeepFin/pull/1352); its local 276-case and
whole-lint qualification is synthetic. Actual A39/B32 public-learner engineering,
new-game generation and the deterministic scoring consumer remain separate gates.

## Fixed arms and horizon

All three start from original immutable A39 model weights, SHA256
`a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`:
175 planes, 494 state entries and 61,444,448 unique parameters. Original embedded
step 680276 is provenance, not the fresh optimizer counter. Restore no original
optimizer state. Keep the complete coupled architecture and initial heads.

| Arm | Constant LR | Initial-policy forward-KL coefficient |
| --- | ---: | ---: |
| A | 3e-6 | 0, null reference and exact bypass |
| B | 7.5e-7 | 0, null reference and exact bypass |
| C | 3e-6 | 0.1, unchanged original A39 reference |

Use the explicit anchored public bank family for all three arms. Main policy and
genuine-outcome main WDL have coefficient 1 each; all 12 auxiliary objectives and
SF/search WDL shares are zero. Preserve existing finite CORE NadamW/groups/masks.
Use eager execution, batch 32, mirror probability .5, no added clipping, SWA,
compile, prefetch, residual dropout, loss rescaling or transient retry.

There is one logical generation and one segment per arm. With N qualified TRAIN
rows, freeze K=min(ceil(N/32),1024) and apply exactly K updates / 32K views. The
minimum row gate makes K at least 250. This is approximately one expected row
exposure below the 32,768-view cap, with replacement: it can repeat or omit rows.
Report N, K, views/N, unique drawn physical rows and per-game journal coverage.
Numeric contracts are instantiated after data audit and before learning without
any discretionary extension. Fresh model/optimizer/scheduler/ZClip start at step
zero; peak=floor=warmup-start=arm LR, warmup 0 and first period 2048, avoiding a
restart within K. Sampling journals, backend and five RNG streams must match.
Default CUDA bit repeatability is not assumed.

These rates are prospective choices informed by earlier controls, not a
validated safer or equivalent effective-step regime relative to B256/warmup256.
For common normalized targets/support, CE(T,p)+.1 KL(q||p) changes effective
policy weighting relative to WDL as well as pulling toward q. The three arms test
that objective package; they do not isolate a pure trust-region mechanism or
protect WDL. No fourth arm, coefficient grid, checkpoint selection or best-arm
replacement is authorized.

## New cohorts and exclusions

Plan exactly 192 TRAIN and 64 HOLD attempts, with separate collection UUIDs and
strict public schemas. Assign exactly one physical game per distinct complete
source-parent group. Require at least 160 resolved nonempty TRAIN games / 8,000
rows and 48 resolved nonempty HOLD games / 2,000 rows. Preserve and audit every
attempt/class, including censored or empty games. No outcome quota, topup,
replacement, reseed, row split or old TRAIN relabel is allowed. Inference is
conditional on resolved retained games; a censored '*' is never imputed as draw.

Publish ordered rosters and all role identities before outcomes, using one fixed
label-blind cached-rank/protection rule. Complete parent-member sets, full-history
and raw/mirror input identities must be protected across TRAIN, HOLD and all six
playing roles, including known old used, unused, reserved and cancelled roles,
public starts and retained trajectories. Broader pretraining, unretained and
family exposure remains UNKNOWN. After TRAIN audit and before HOLD execution,
derive only the predeclared label-blind trajectory/input/mirror exclusion union.
Any known cross-split collision rejects the whole study; no selective row removal
or replacement follows.

Both cohorts use frozen A39, full-simulation Gumbel64, pure selfplay, retention of
all eligible policy rows, eager BF16 serving, full opening history and 450
additional plies. Strict 3–6-man rule50 tablebases and policy rescoring apply.
Diagnostic SF requests are 64 nodes, Threads 1, Hash 16 MiB and MultiPV 1;
record actual nodes and shutdown behavior. Those observations are not a new
5000-node label study, a hard node cap or an active auxiliary training objective.
Exact full realized configurations, source/asset identities and role hashes must
be appended before collection; this draft does not silently adopt defaults.

## Fixed heldout ruler and health veto

Score unchanged A39 and final A/B/C on every qualified HOLD row exactly once, in
fixed physical-ID/row order. Use the same immutable arrays and actual strict
HOLD proof/input/flag/legal/encoding/integer-outcome predicates. No sampler,
mirror, augmentation, filtering, rebuild, relabel, SF/search blend or anchor
penalty belongs in this ruler.

WDL order is win0/draw1/loss2. Genuine-outcome CE uses stable FP32 log-softmax.
The primary estimate is the all-row CE mean per game, equal games within each
complete parent, then equal parent means. One game per parent makes the 48-game
minimum a 48-parent minimum. Also report row-weighted CE, unhalved three-class
Brier sum, expected-score MSE (pWin+.5pDraw against 1/.5/0), policy CE and legal
forward KL(A39||candidate) descriptively. Retain source-qualified compact row
observations and game/parent sums/counts for checking and estimator corrections.

Use paired complete-parent resampling, preserving all corresponding models,
games and rows together: 10,000 local NumPy 2.2.6 PCG64 default_rng replicates,
seed 2026111680, two-sided percentile 95% at 2.5/97.5 via quantile method linear.
Apply this to equal-parent C-minus-A39 CE differences; never bootstrap rows.
If all differences equal, report the constant empirical interval and label it
degenerate. Fewer than 48 parents, empty denominators, nonfinite/missing values or
incomplete identities make health UNREAD and prevent playing advancement.

The severe-damage rule is a **veto only**: halt playing/confirmation if the
C-minus-A39 primary CE point is greater than .20 nats AND the paired bootstrap
95% lower bound is greater than zero. A non-triggered veto establishes neither
noninferiority, safety nor absence of harm. A/B health and other metrics are
descriptive and cannot choose a different candidate, LR, coefficient or horizon.

Scoring uses honest public model/Trainer construction, eval/no-grad forwards and
no optimizer restoration or update. Disclose any optimizer allocation or
construction-time initialization RNG and restore it before scoring; do not use
an uninitialized/fake Trainer. Scored forwards must preserve state/RNG. This
specifies deterministic membership/reduction order, not cross-hardware CUDA
bit identity. Source-specific actual B32/ruler evidence remains necessary.

## Mandatory playing decisions

Use the same qualified internal Gumbel64 matched-simulation training profile,
root noise ON, temperature .1, eager BF16/F32 transport, full 16-ply history,
strict rule50 tablebases and 450 additional plies. Freeze the complete realized
profile and served-state checks before games. This is not an external timed
multicore-engine benchmark.

Preallocate six mutually disjoint/protected panels of 400 distinct complete
parent opening pairs, 800 reciprocal-color games per panel. Mandatory screens
are C/A, C/B and C/A39. After technical/health acceptance, run all three regardless
of an earlier screen's score. Do not peek for early efficacy, expand or replace.
The statistical unit is a complete opening pair, not two independent games.

For each edge, estimate mean reciprocal-color pair score and the approximate
marginal 95% normal interval mean +/-1.96 sqrt(sample variance with n-1 / n),
clamped to [0,1]. Fewer than two pairs, nonfinite values or zero variance yields
an UNAVAILABLE interval and cannot pass. Require complete 400-pair technical
acceptance, point >= .55 AND available lower > .50 for every edge.

The generic bounded-pair sample-SE bound is 1/(2sqrt399)=.0250313, giving about
.0490614 halfwidth. This describes precision; it is not 80% power for a true
five-percentage-point edge. Marginal intervals are not simultaneous or
project-wide coverage.

Only if all three complete screens and all technical/health gates pass may the
same fixed C/A/B/A39 hashes repeat all three edges on the respective fresh
400-pair confirmation panels with the same gates. Consume the attempt at the
first confirmation execution admission, including later failure/cancellation;
no retry, substitute or pooling. One collection/training seed remains one seed.
Even full PASS gives conditional candidate utility, not an optimal recipe,
external-engine milestone or automatic adoption. G2 requires a later decision.

## Seeds, costs and phase order

| Purpose | Seed |
| --- | ---: |
| TRAIN collection | 2026111671 |
| HOLD collection | 2026111672 |
| Common training and scoring setup | 2026111673 |
| C/A, C/B, C/A39 screens | 2026111674 / 1675 / 1676 |
| C/A, C/B, C/A39 confirmations | 2026111677 / 1678 / 1679 |
| HOLD parent bootstrap | 2026111680 |
| Separate software engineering | 2026111681 |

GPU science caps are TRAIN collection 3600, HOLD collection 1800, three learners
900 each, four-model HOLD scoring 1200 total, three screens 3600 each and three
conditional confirmations 3600 each: **30,900 seconds maximum**. CPU science
caps are 1500 total data audits, 900 HOLD reader and 7200 total six arena readers:
**9600 seconds maximum**. Separate actual-A39/B32 software engineering is capped
at 600 GPU / 600 CPU seconds; unit and role-selector controls have their own
bounded CPU admissions. No historical unused credit is inherited. Every failure
and cleanup wall second counts toward its scope and the 172,800-second all-GPU
ceiling. Stop new compute at 2026-11-19 16:09:20 UTC unless the user extends it.

Readiness is phase-specific, avoiding a collection/output dependency cycle:
source/unit checks can precede research; roles/config/assets/main prereg and
producer readiness precede collection; closed corresponding producers precede
data audits; accepted TRAIN and public B32 engineering precede scientific
learning; accepted HOLD, model deliveries and a qualified ruler precede scoring;
closed scoring precedes its CPU reader; health/model/profile acceptance precedes
screens; all three accepted passing screens precede confirmation. Each actual
phase still needs fresh controls, known-birth custody, three leases, resource and
deadline checks and a root execution admission.

Bank all faults and partial observations. Technical failure, timeout, control or
UNREAD health stops the affected study without automatic replay, resume, reseed,
topup, alternative checkpoint or budget extension. Lost transport alone proves
neither failure nor closure; reconcile the owner first. Old retained queues and
THREE remain stopped; cancelled 5k/successors/150k work stays cancelled.

## Prospective authority and remaining freeze

Study UUID: `8045699e-4088-4a41-ae9c-4013d4a55469`.
External evidence root: `~/chess-artifacts/operations/rl-week-one-20261112-v1/`.

| Record | Relative path | SHA256 |
| --- | --- | --- |
| Root selected design V2 | `prospective-whole-game-anchor-root-design-selection-v2/design.json` | `37c4e9cf078e794f853cddfd0a2ee6f7fba339701bc68178ae5804286be298c6` |
| Independent selected-design review | `prospective-whole-game-anchor-root-selection-independent-review-v1/review.json` | `8dd550ab2911d6fd7871b2302f045ba289894a89b2e74cb3a3fde823671d7b94` |
| HOLD CPU software root | `prospective-holdout-CPU-bindings-v2/root-acceptance.json` | `b550b725b651d4ddb716cb14bcd94e5decfa9f52dff40d1190d5571c4e973b85` |
| HOLD CPU independent actual | `prospective-holdout-CPU-independent-actual-review-v1/review.json` | `df82c32d8b97af2ac3e1386b2b545891580671132471dc932472077ac9f3fb7f` |
| Current-H genuine CPU assets | `collector-current-H-asset-qualification-bindings-v1/root-acceptance.json` | `a94200325d7f7829f837a93019ab67ce2f029fea3ba9a2a15efcf6653512689f` |
| Current-H independent assets | `collector-current-H-asset-qualification-independent-actual-review-v1/review.json` | `efa0ac077928109c9fb0dc98b221eb77b17771bb14c53744ff1ac40077972ef4` |

The qualified source runtime remains pinned to `31ba591bbda95e1530ce995a07bdbce24f00fb73`.
New main UCI/tablebase changes are not implicitly deployed into that runtime.
Local synthetic software and CPU asset checks do not qualify collection, public
A39/B32 training, GPU resume, holdout scoring, playing or scientific execution.
Before first affected compute, append exact source/native/runtime and realized
configuration hashes, all TRAIN/HOLD/playing rosters and protected exposures,
closed engineering/role/readiness receipts and immutable phase contracts.
These bindings must implement the fixed choices above. Any substantive change
requires a prospective amendment before the affected observations, never a
post-result rescue. No new science owner is admitted by this draft.
