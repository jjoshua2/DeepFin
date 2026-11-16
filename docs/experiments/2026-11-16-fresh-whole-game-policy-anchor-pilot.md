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


## 2026-11-16 pre-science source and role freeze

The original prospective record above is preserved verbatim. This addendum
replaces its pending-source/roles/configuration description for this study;
its question, three arms, losses, rates, dose formula, seeds, health veto,
playing/confirmation gates and science ceilings are unchanged. **Scientific
collection, training, scoring and playing have not started.** Metadata preparation
is complete; neither publication nor a prepared descriptor admits execution.

Root selected the final profile and role identities in
`prospective-whole-game-final-profile-role-root-selection-v1/selection.json`,
SHA256 `e5faa0b1652146b8bbf0185e868d25a64d1d8c64848b68e5f849c6e138cefa9a`,
before any scientific outcomes. The [preparation readout](2026-11-16-fresh-whole-game-preparation-readout.md)
and [compact evidence](2026-11-16-fresh-whole-game-preparation-evidence.json)
pin the completed cache, role and starts authorities and their independent reviews.
The entire new 16,384-candidate cache is reserved to this study and protected
against future studies, including its rejected and unused entries. It is not the
current TRAIN prior and must not create an own-role self-veto.

### Current source, assets and realized-profile expectations

The scientific runtime is frozen I/T revision
`bbb4b725a2d5218ffa4630841dd93e8b2884b58d`; the earlier H/31ba paragraph is
historical. I7 has actual 394-case/full-lint/closed-owner root and independent
acceptance (`9c5df1f6…` / `e4e6934e…`). Source-map domains are distinct:
unit map (218 files) `6f9cd298…`; current asset map (219 files) `574c855d…` adds the Lee metadata consumer;
role map (219 files) `a26a82c3…` instead adds `arena_standard.py` for selected metadata
functions. Counts do not make these maps interchangeable, nor establish whole
current-main equivalence or live adoption. Current-I genuine assets have source
manifest `01b4335e…`, root `6949de24…`, independent actual `43a4c892…` and consumer
envelopes `5f23c6cd…`. Original A39 remains SHA `a39f900c…`, embedded step 680276;
no scientific model/optimizer boundary or accepted TRAIN/HOLD bank exists yet.

The [full 109-field profile](2026-11-16-fresh-whole-game-collector-profile.json)
records all six constructor groups, selected serving fields and exact cohort
starts substitutions. It preserves selected profile `ab5cf42f…`; the six explicit
prospective overrides are noise-after .75 and all five temperature endpoints .1.
There is no omitted-default adoption or additional tuning. Pure selfplay uses
full/fast simulations 64, playout-cap fraction 1, Gumbel scale .75 at both schedule
ends, root noise ON and constant temperature .1 on searched/resampled plies.
Curriculum/refute/diff-focus routes and fast-value/P0-policy label recording
remain inactive. Diagnostic SF64 requests are observations, not a hard node cap
or an auxiliary objective.

Both exact starts files preserve 16 legal opening plies. `game.max_plies=450`
bounds **additional** plies after that history, potentially 466 absolute plies;
natural or strict-rule50 termination can occur earlier, and an unresolved cap
remains censored. `concurrent_slots=16` counts simultaneous games;
`serving.n_slots=1` is the reusable evaluator buffer. Eager BF16 autocast retains
F32 input/legal/host policy-WDL transport, no padding, configured `max_batch=512`.
That configuration is not an observed scientific batch size, allocator bound,
all-operator BF16 claim or demonstrated450-ply runtime fit.

| Cohort | Attempts / minimum resolved nonempty / minimum rows | Source-derived expected six-group SHA256 |
| --- | --- | --- |
| TRAIN | 192 / 160 / 8000 | `960e1dc5dc9dcdf09d5ceee6adaba4ae7f1b858fb49add5fc79360dc82b12c21` |
| HOLD | 64 / 48 / 2000 | `d63a104aa030c1adf4293524fef431eef8f8beb413ebf5225b06add7c5d39b5b` |

Both public JSON companions are display records, not runtime inputs. Private
home paths use `~/...`. Canonical reconstruction must first expand those paths
using the original private-home binding in the sealed absolute descriptors,
rather than a reader's home directory. The `public_path_rendering` metadata is
outside the six-group profile. The accepted binder computed the expected hashes
from the original absolute descriptors and starts-path substitutions; the
home-relative public JSON bytes do not have those canonical profile hashes.
**They are expectations, not observed resolved profiles.** The real
public `build_configs`/six-asdict equality check remains mandatory inside the
owned collection. Source-only metadata candidates `13fcd976…` / projection
`997950ec…` have independent source/metadata review `3b602629…`; this is neither
collection execution nor positive bank evidence. TRAIN descriptor is `e0511c70…`;
HOLD template `55785638…` retains only its final prior path/SHA as late leaves.
That future prior and resulting final HOLD descriptor hash are not invented here.

### Fixed prospective identities and protection wiring

The run UUID is `70d23dfa-c1d8-4f1e-86a0-ee29ae777a8a`; logical generation1 uses
TRAIN collection `e1d943b5-83da-4e85-a2df-f086c9e680fc` and HOLD collection
`cdbaedef-d47e-4276-b873-fb17f1f3ccbc`. Accepted pre-outcome role root/independent
`14c2d63c…` / `88eca945…` pin 192 TRAIN, 64 HOLD and all six panels of 400 pairs with
complete-parent maps. Accepted starts root/independent `5f6d9847…` / `1671e361…`
pin the exact ordered full-history bytes. No role is selected or replaced after
an outcome; all confirmation panels are already frozen, but their execution
attempt remains conditional and is consumed only at actual confirmation admission.

Root wiring `4d454322…` explicitly binds the real legacy `playing_starts=[]`,
count 0, SHA `37517e5f…`. **This is not a protection omission or a chunk-as-flat
alias.** The effective strict chunked TRAIN prior `4b84d19a…` already includes
OLD+SMOKE, all 64 HOLD starts and all 2,400 starts across six panels. The separate six-panel index
`a0e7c0a8…` authenticates full-history and raw/mirror identities. HOLDbase
`3cfdc859…` includes OLD+SMOKE plus panels, excluding HOLD's own starts; copying
TRAINprior into HOLD would incorrectly self-veto HOLD and is forbidden.

The same planned TRAIN audit (900-second outer / 870-second inner) must run the unchanged genuine
positive TRAIN data auditor and the reviewed all-trajectory wrapper. It must
cover every one of 192 attempts and every played root at offsets `0..N`, including final,
noneligible/no-policy and unresolved positions, at most 86,592 roots. Source
reconstruction uses actual source encoded 175-plane FP16 and production mirror without a model
or additional GPU forward. Before deduplication or bank acceptance, any known
HOLD/panel full-history, rawF16, mirror or literal-EPD collision rejects the whole
program; there is no row removal, reroll or replacement. A zero-row return from
the public auditor cannot stand in for complete trajectory protection.

Wrapper V2 source `04b42069…`, helper `3b4a42d7…` and independent source review
`60c254e1…` are accepted as source only. Its twelve same-owner controls are
**UNRUN**. It preserves real positive candidate statuses and the existing
root/independent bank schema, original strict-TB verification handle and streaming
actor SHA checks. Separate postclosure protection root/independent acceptance is
also required. Canonically rewrite/validate HOLDbase union the complete actual
TRAIN index within the same 900 / 870-second budget; do not append unsorted chunks or merge
TRAINprior wholesale. Until then, final HOLDprior stays UNBOUND and HOLD collection
cannot be admitted. Reconstructed trajectory roots do not prove coverage of
unretained internal MCTS leaves or every actual GPU-served input; these and donor
pretraining/broader exposure remain UNKNOWN.

### Phase authority still required

The unchanged science ceilings are 30,900 GPU and 9600 CPU seconds, all unstarted;
cache/role/starts CPU below is separate preparation cost. Before each phase,
root must bind its exact current helper/public suffix, protocol, source/runtime,
accepted prerequisites, independent binding review, fresh U3/known-birth closure,
three leases, deadline and actual admission. The finite collection startup guard
must be source-reviewed and bound before launch: observe at least 8 GiB free on GPU 0 without
initializing CUDA first, at least 64 GiB host memory at entry/final and 20 GiB free disk, while R7 retains
its continuous 8 GiB host floor. This is no allocator cap/reservation, continuous
64 GiB guarantee or 512-row fit claim. Actual phases stop on their selected caps;
there is no resource, precision, batch, seed or horizon fallback.

Future actual producer readout/inventory/terminal/full child command, positive
TRAIN bank/protection receipts, final HOLDprior and HOLD descriptor, numeric
K derived from qualified N, A/B/C contracts/stages/model deliveries, scoring and
playing phase admissions remain late evidence. The source profile and all
statistical/recipe choices are already fixed; late binding is not permission to
retune after observations. The final public MAIN record and descriptor/helper
binding checks must precede affected compute. Old cancelled 5k/successors/150k,
retained queues and THREE remain cancelled/stopped.


## 2026-11-16: technical stop and prospective recovery R1

The original comparison is technically UNREAD. TRAIN collected and independently
audited 192 eligible games / 17,568 rows; HOLD collected and independently audited
64 eligible games / 5,943 rows. The first A owner failed after 6.08346829097718 GPU
owner-seconds because its bank consumer required a closed CPU-audit terminal to be
read-only. Source order places this failure before output creation, RNG seeding,
replay sampling, model/optimizer construction, forwards and updates. Preserve A as
FAILED; B and C were not started. This is not a negative learning result.

An independently reviewed custody correction changed only the two closed CPU-audit
terminal modes from0644 to0444. Paths, JSON bytes, hashes, inode identities and
modification times stayed unchanged; bank/role/protection/contract identities remain
unchanged. A subsequent engineering owner failed after4.359743445063941 CPU seconds
while serializing a frozen metadata mapping, after its four genuine bank-API calls
and native freshness check. It earned no153-case/full-lint/final-qualification credit.
Both failures and their costs remain in the record.

The corrected engineering qualification recursively serializes plain metadata,
verifies the existing exact native files, and explicitly sets/verifies/records both
Torch thread counts as2. It must pass153 reviewed synthetic cases, four separate
genuine bank-metadata controls, full static checks and final source/native/runtime
custody in a fresh CPU550/520-second owner. Prior engineering CPU cost is
42.775648968177855 seconds: adding its entire550-second cap stays below the existing
600-second ceiling. Synthetic tiny-model/shard fixtures are allowed in the153 tests;
the four genuine-bank controls remain metadata-only. ROOT observes final host64GiB
and disk20GiB at acceptance; R7's continuous host-memory floor remains8GiB.

This is an explicit prospective technical amendment under the standing instruction
to repair the RL pipeline and run bounded comparisons. It selects recovery R1 only
after this amendment is published and reviewed, corrected qualification has ROOT
and independent acceptance, and exact new attempts/scopes/admissions are issued.
Until then R1 is unadmitted. This does not resume or reuse a failed command/scope.

R1 retains the same accepted TRAIN/HOLD data, original A39 donor, complete-parent
roles, reserved playing panels and frozen learner implementation. N=17,568 and K=549
remain fixed, with batch32, sampling with replacement, mirror0.5 and common
seed2026111673. A remains LR3e-6/anchor0; B remains LR7.5e-7/anchor0; C remains
LR3e-6/original-A39 reference-to-student forward policy-KL coefficient0.1. All loss
weights and zero auxiliary objectives remain unchanged. Each arm starts afresh from
A39 with a fresh optimizer/scheduler/Zclip state. No partial checkpoint is resumed.
Use distinct recovery owner/output/observation/scope identities for all three arms,
each still bounded900/870 seconds with full saved-state/RNG/membership proof.

No replacement collection, reseeding, role/quota adjustment, precision or dose
change, metric/threshold change, efficacy peek or checkpoint selection is authorized.
The original four-model HOLD scoring/reader, equal-parent estimator, paired
bootstrap, catastrophic veto, three400-pair screens and three conditional fresh
confirmations remain unchanged. A technical failure stops R1; no further automatic
recovery or G2 is selected here.

Failed scientific wall time stays charged. Before R1, scientific GPU owner time is
1,455.5775687580463 seconds; all remaining selected learner/scorer/screen/confirmation
caps total25,500 seconds, giving26,955.5775687580463 within the original30,900-second
study ceiling. The weekly172,800-second GPU ceiling and2026-11-19 16:09:20 UTC cutoff
for starting new compute remain unchanged. Every new phase requires fresh exact
custody/resource checks, scope and admission.

Accepted data, software qualification and HOLD metrics do not establish Elo gain,
a win over multicore Rofchade/Stockfish/Leela, or an optimum RL recipe.

Compact supporting identities: [technical recovery evidence](2026-11-16-fresh-whole-game-technical-recovery-evidence.json).


## 2026-11-16 D1: user-directed coefficient sweep and development-first gates

This prospective amendment supersedes the unused R1 A/B/C launch plan for
future learning and evaluation. R1 was published and its controller reviewed,
but no R1 execution authority, bindings, admissions or compute were issued.
Original A remains a closed technical failure before model construction;
original B/C remain unstarted. No new scientific learner update or playing
result has been obtained. The prior record and failed artifacts remain intact.

The user requested several KL strengths, a matched no-KL baseline, a meaningful
training dose, reuse of eligible selfplay, and cheap rejection of poor settings
before reserved holdout and larger final matches. Root selected D1 before any
new trained-model outcome. This is a finite comparison, not an optimum-RL claim
or a multicore Rofchade, Stockfish or Leela strength milestone.

### Fixed learning comparison

Reuse only the accepted 192-game, 17,568-row TRAIN bank. No new collection,
topup, old-bank substitution or HOLD input is authorized. All four fresh runs
use unchanged frozen T `bbb4b725a2d5218ffa4630841dd93e8b2884b58d`, original
immutable A39 model-only initialization, the complete coupled 494-state /
175-plane architecture, and fresh optimizer/scheduler/ZClip counters at zero.
Original embedded step 680276 remains provenance.

| D1 arm | Constant LR | Forward KL(original A39 || student) |
| --- | ---: | ---: |
| K0 | 3e-6 | 0, null reference and exact bypass |
| K001 | 3e-6 | .01 |
| K01 | 3e-6 | .1 |
| K1 | 3e-6 | 1 |

The decade grid is a coarse prospective range, not a calibrated or optimal
coefficient. Main policy and genuine-outcome WDL coefficients remain one;
all 12 auxiliary objectives and SF/search WDL shares remain zero. Positive
arms use unchanged original A39 as the frozen reference. This objective package
changes policy weighting relative to WDL as well as the policy target; K1 does
not isolate preservation at fixed objective balance or imply doubled gradient
magnitude.

Fix batch32, mirror .5, common seed2026111673, and exactly4096 updates /
131,072 replacement-sampled views per arm: approximately7.461 views per TRAIN
row, not complete epochs. Freeze peak=floor=warmup-start=3e-6, warmup0,
first-period4096. This satisfies the public no-restart horizon guard while
keeping LR constant. Eager execution, optimizer/groups/masks and architecture
remain unchanged; no added clipping, loss rescaling, SWA, compile, prefetch,
dropout or transient retry. Match complete bank membership, draw/mirror journals,
five RNG streams, backend and initial states across arms. CUDA bit repeatability
is not assumed.

Each arm has one complete final checkpoint and a3600-second GPU owner /
3570-second absolute child bound, including passive live-to-saved proof.
Checkpoint cap is8GiB and journal cap64MiB. Two-update engineering timings
do not establish sustained throughput or completion within this cap. Timeout
or technical failure is UNREAD/STOP; never shorten dose, select a partial
checkpoint, keep only surviving arms, retry or substitute another endpoint.

### Cheap development selection before HOLD

Use exact first80 complete-parent opening pairs from existing `SCREEN_C_A`
as a common DEV panel for K0/K001/K01/K1 versus A39. Use exact first80 from
existing `SCREEN_C_B` as a second common DEV panel for K001/K01/K1 versus K0.
All seven edges are mandatory and fixed before outcomes. Reusing a DEV panel
across coefficients is deliberate; these results are for screening and selection.

A bounded owned CPU phase must authenticate each original400-pair book and
complete parent map, materialize the exact80-pair prefix, and prove the actual
public arena loader consumes all80 in order without skips, deduplication or
shrinkage. Passing a full400 book with a160-game arena request would instead
invoke seeded subsampling and is not this protocol. Preserve known TRAIN/HOLD/
final history, input, mirror and parent protection; no role replacement follows
an outcome. Unused book entries remain reserved and protected.

Use common seed2026111674 for all four versus-A39 edges and2026111675 for
the three versus-K0 edges. Each edge requires exactly80 complete reciprocal-color
pairs /160 games, with GPU900/870 and CPU-reader300-second bounds. Complete and
technically accept all seven before selection. Report all pair-score points and
paired normal95 intervals descriptively, without interpreting selected DEV
scores as significance or pooling them with final matches.

An arm is eligible when its versus-A39 pair-score point is at least.55.
A positive-KL arm additionally needs its versus-K0 point at least.55. Among
eligible arms select the highest versus-A39 point; break an exact tie by lower
KL coefficient. K0 may win. If no arm is eligible, STOP without HOLD. This noisy,
conservative80-pair screen can reject useful settings; no threshold relaxation,
extension, checkpoint selection or replacement is allowed.

### Reserved health check and final playing test

Seal the selected winner before HOLD. The existing64-game /5943-row HOLD bank
was generated by original frozen A39 and audited; its aggregates and metadata
have been consumed. No newly trained candidate has been scored on it. It is
reserved for that use, not globally untouched.

Only after DEV advancement score A39, K0 and the winner when distinct, on all
HOLD rows exactly once with the original deterministic membership and genuine
outcome ruler. Do not load or score unselected positive-KL models. Retain
equal-parent FP32 WDL-CE, paired complete-parent PCG64 bootstrap10000 /
seed2026111680 /percentile95 linear, the48-parent/2000-row minimum and the
original descriptive metrics. Veto when selected-minus-A39 CE point>.20 nats
AND bootstrap lower95>0. Missing/nonfinite/invalid evidence is UNREAD/STOP.
Nonveto establishes neither safety nor noninferiority. K0 health is descriptive
when another arm wins. No HOLD ranking, fallback or second candidate follows.

If health is readable and nonvetoed, use the same immutable winner on the full
reserved `CONFIRM_C_A39`400-pair panel versus A39, seed2026111679. A positive-KL
winner must also face K0 on the full reserved `CONFIRM_C_A`400-pair panel,
seed2026111677. A K0 winner has only the A39 comparison; omit redundant
self-play and retire that unused budget. These books have no playing or
candidate-scoring results; prior metadata qualification and unknown broader
pretraining/family exposure remain disclosed.

Each required final edge needs all400 complete reciprocal-color pairs,
point>=.55 AND available paired normal95 lower>.50. Use sample variance n-1,
mean +/-1.96 SE, clamped[0,1]; n<2, nonfinite or zero variance is UNAVAILABLE
and cannot pass. Complete all required final edges before efficacy readout.
Never pool DEV/final games, replace a failed winner, reseed, replay or extend.
All edges retain Gumbel64 matched-simulation training shape, root noise ON,
temperature.1, eager BF16/F32 explicit-legal transport, full16-ply opening
history, strict original3-6-man rule50 TB behavior and450 additional plies.
Each final GPU owner is3600/3570 seconds and each CPU reader1200 seconds.

Even a full pass supports only conditional single-seed fixed-bank utility versus
A39 and, for a KL winner, matched no-KL. The quarter-rate comparator is absent
from D1. There is no general KL optimum, lower-LR comparison or external-engine
strength conclusion, and no automatic generation2 or adoption.

### Finite costs, readiness and failure preservation

Explicitly retire unused collection caps now that TRAIN/HOLD are closed and
accepted; no collection reruns are permitted. Reallocate within the existing
30,900-second study GPU ceiling before new outcomes:

| GPU item | Maximum owner wall seconds |
| --- | ---: |
| Closed TRAIN + HOLD + original failed A, actual charged |1455.5775687580463|
| Four D1 learners |14400|
| Seven DEV edges |6300|
| Conditional HOLD scoring |1200|
| At most two final edges |7200|
| Total |30555.577568758046|

Leave344.422431241954 seconds unallocated, with no automatic retry. CPU science
is104.29835316492245 already charged +300 role preparation +900 HOLD reader
+2100 seven DEV readers +2400 two final readers =5804.298353164922, below9600.
Separate engineering remains capped600GPU/600CPU, with54.84933027205989GPU /
164.3428476222325CPU actual charged; a future400-second consumer qualification
must include the new D1 interfaces and cannot borrow an older qualification's
claim. All failures and cleanup count in their declared ledgers. Preserve the
172,800-second all-GPU ceiling and stop-new-compute deadline
2026-11-19T16:09:20Z. No credit from cancelled or historical studies is inherited.

Publication alone admits no execution. Before learning require this amendment
on main, independent selected-design and source review, exact finite D1
contracts/prover/controller, existing actual T/I7 + A39/B32 engineering and
accepted TRAIN/protection, then fresh per-arm binding review, U3/admission,
known-birth custody, all three leases and resource/deadline/budget gates.
New scored-consumer qualification is required before its affected DEV/HOLD
consumer phase; it is not an invented predecessor to otherwise qualified
training. Merged main changes do not implicitly alter frozen runtimes.

Any technical failure, timeout, control mismatch or UNREAD stops D1 and preserves
partial observations/costs. It is not a clean scientific negative. No automatic
resume, replay, retry, dose shortening, topup, survivor selection or fallback is
authorized. Cancelled THREE/5k/150k work stays stopped. The long-term strength
objective remains unfinished regardless of D1's result.

Canonical identities and display-only external references accompany this
amendment in [D1 evidence](2026-11-16-policy-anchor-development-first-evidence.json).

## D1-T1: prospective recovery of the K0 verification failure

This amendment addresses a mechanical verification failure. The original D1 technical stop remains in force until this amendment is published and the separate implementation, qualification, recovery and admission gates below pass. Publication alone authorizes no job.

### Recorded failure

The K0 GPU owner ended FAILED after 1,514.692136213067 seconds. Its immutable public boundary records 4,096 applied updates and 131,072 sampled views, with zero nonfinite skips or CUDA retries. The public learner saved its checkpoint. No candidate match, HOLD scoring or strength measurement followed.

The private verifier then failed its saved counters/architecture/reference assertion. Independent source review identified a deterministic defect: `ModelConfig` normalizes `ffn_mult_by_layer` and `phase_piece_thresholds` to tuples, and `Trainer.save` preserves those tuples through `dataclasses.asdict`; the JSON contract contains lists. Direct dictionary equality therefore rejects this genuine representation. The correction compares the complete architectures using the existing strict canonical serialization. It drops no keys, inserts no defaults and preserves scalar values. Donor-provenance and policy-anchor checks retain their existing semantics.

The traceback establishes that the preceding saved-versus-live fingerprint and alias assertions were reached and passed. This is source-order evidence; the numerical live fingerprint was not persisted. Later optimizer, journal and final source/native/runtime/RNG checks did not finish. The failed terminal and private failure marker remain immutable and fully charged. This amendment neither converts that owner to PASS nor invents its missing final witnesses.

### Fixed technical checks before continuation

1. Independently review a versioned corrected prover. Preserve the original production dose, data, seed, mathematical recipe and 3,600/3,570-second limits for the three unstarted KL arms.
2. Qualify its complete path in one separate engineering GPU owner, capped at 400 seconds with a 370-second inner deadline: full A39 architecture, coefficient 0.1, batch 32 and two actual public-learner updates. Exercise frozen-reference handling and every late saved-state/custody guard. Its contract, schema and status are explicitly engineering-only. It supplies no D1 model-delivery or efficacy credit. No shortened production contract or K0 retraining is permitted.
3. Audit the existing K0 checkpoint in one CPU owner capped at 600/570 seconds. Allow only the exact failed GPU terminal, prover, traceback, completed public boundary and checkpoint identities in the supporting evidence. Decode with `weights_only=True` on CPU, without restoring an optimizer, constructing a model, forwarding, training, sampling or reseeding. Authenticate complete saved architecture, tensors and aliases, named optimizer ownership and active-slot steps, scheduler, clipping state, provenance, boundary RNG and membership, and the full original draw journal. Require 4,096 updates, 131,072 views, 17,568 population rows and 192 games.

The CPU audit may use authenticated historical full-model layout metadata for key/shape/dtype/alias comparison. That metadata supplies no K0 live-state or strength evidence. Its new source/native/runtime brackets describe the CPU audit's current custody, not a retroactive final bracket for the failed GPU process. It cannot reconstruct an unpersisted live fingerprint. File limits are 8 GiB per checkpoint and 64 MiB per journal; total explicitly serialized reads are capped at 32 GiB, aggregate output at 4 MiB. Host and disk guards require 64 GiB and 20 GiB at the declared boundaries; they are not continuous memory or fit guarantees.

Any failed qualification or recovery stops continuation. There is no automatic retry, survivor selection, horizon reduction, checkpoint repair or fallback.

### Explicit recovery family and unchanged scientific comparison

A new K0-only failed-GPU-plus-closed-CPU proof family must carry distinct recovery statuses, both real owner graphs, the exact allowlisted trigger and a newly issued Root recovery authority. A source selection, source review or publication alone is insufficient. The ordinary complete same-GPU delivery route remains unchanged. Independent acceptance must state the recovery limitations rather than label K0 as an ordinary same-GPU PASS.

Only after actual engineering qualification, actual CPU recovery, independent acceptance, published compatible consumer code and its bounded local qualification may a new authority admit the three previously unstarted KL arms. Use new D1-T1 stage names. Preserve the existing K0 weights without rerunning its training.

The four coefficients, A39 initialization, training bank, seed, batch, learning rate, mirror probability and 4,096-update/131,072-view dose remain fixed. The seven development comparisons, fixed opening panels, eligibility and tie rule, selected-model-only HOLD health check, veto and final confirmation requirements remain unchanged. No effectiveness data are read to select this recovery policy. Recovery does not authorize adoption, additional opponents or any strength claim.

### Charged costs and remaining bounds

All closed scientific GPU work, including failed K0, totals 2,970.2697049711132 seconds. Adding the three remaining learner caps (10,800), development (6,300), HOLD scoring (1,200) and final confirmations (7,200) gives 28,470.269704971113 seconds, below the unchanged 30,900-second study cap. Retire K0's unused reservation; it does not authorize another run.

The new engineering GPU cap gives at most 454.8493302720599 seconds against its 600-second aggregate limit. Existing engineering CPU plus the future 400-second consumer qualification remains below 600 seconds. The recovery CPU cap and remaining science readers project to 6,106.593127332861 seconds, below the 9,600-second science CPU cap. Global resource limits, three-lease custody and the November 19, 2026 16:09:20 UTC compute cutoff remain unchanged.

Supporting identities and the independent diagnosis are in [the D1-T1 evidence record](evidence/2026-11-16-policy-anchor-k0-technical-recovery.json). Source artifacts, actual qualification/recovery receipts and a fresh Root authority must be authenticated before execution. This prospective amendment is not an implementation or recovery result.
