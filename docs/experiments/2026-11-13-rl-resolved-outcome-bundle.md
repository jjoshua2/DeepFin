# Resolved-outcome value targets at a fixed 40-update dose

Status: preregistration frozen before the 96 research games and independently
accepted. Main publication and individual stage admissions remain required. No research
updates, playing result or candidate adoption have occurred.

The current RL transition has not demonstrated a repeatable strength gain.
Earlier endpoint results and the interrupted 53-game collection remain preserved
in the [week-one record](2026-11-12-rl-week-one.md). The donor's measured external
baseline was 1 win, 24 draws and 39 losses against Rofchade 3.1 with eight threads
at 30 seconds plus 0.1 seconds per move. That profile scored 20.3125%; it does not
establish the project's Rofchade milestone. The long-term objective remains
convincingly beating unhandicapped multicore Rofchade, then approaching strong
Stockfish and Lc0 deployments and investigating improvements beyond them.

This comparison asks whether adding 20% verified resolved-game outcome to both
value objectives improves playing strength over search-only targets at one fixed
dose. It tests a target bundle, not an isolated head mechanism or an optimal RL
formula. Teacher policy supervision and the rest of the original full-network
training configuration remain common to both arms.

## Frozen design

The machine-readable [preregistration](evidence/rl-e2tb-week-one/preregistration.json)
is authoritative. Its SHA256 is
`ee24d3b7aa575c3dd2b85061f376a4920d5f9034a2cc358e03ed8312cd1476c0`.
The [artifact manifest](evidence/rl-e2tb-week-one/artifact-manifest.json) records
the published evidence and external inputs. External artifacts live under
`~/chess-artifacts/operations/rl-week-one-20261112-v1`.

Both arms start from the exact epoch-two donor
`a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`,
with fresh optimizers and no SWA. The runtime remains frozen at
`ba9d6de1987902e1759d984532a4c63f091b9269`. The strict selfplay library is
published through [PR 1249](https://github.com/jjoshua2/DeepFin/pull/1249);
the final main version adds only two test metadata corrections to that runtime.
Other UCI changes on main are not deployed into this measured checkout.

| Property | Fixed choice |
| --- | --- |
| New study identity | `f1dadbe4-43e2-4cf3-9a93-70a288533b48` |
| Collection | 96 fixed physical attempts, 16 slots, 32 TRAIN and 64 HOLDOUT |
| Generation | One donor, pure selfplay, native Gumbel 64 simulations, 450 additional plies |
| Tablebases | Indexed three-through-six-piece WDL and DTZ; strict rule-50 and full-history semantics |
| FULLA | Main WDL search 1.0, outcome 0, SF 0; original categorical search target |
| FULLB | Main WDL search 0.8, resolved outcome 0.2, SF 0; matching categorical scalar blend |
| Delivery | 40 updates per arm, batch 256, one common uniform physical-row draw plan |
| Pairing | Common draw, mirror and forward/dropout RNG; original optimizer, losses and schedule |
| Snapshots | 0, 1, 8 and 40; only FULLB at step 40 is the candidate |

Categorical targets use signed Q in [-1, 1]. FULLB blends the original producer
float32 search Q with the verified outcome Q before HL-Gauss construction and the
production float16 cast. Every eligible FULLA row must reproduce the original
float32 search control and stored float16 categorical target exactly. This is
not a mixture of cross-entropy losses or gradients.

Natural outcomes, strict tablebase adjudications and unresolved games retain
distinct provenance classes. An independent owned tablebase verifier reconstructs
the legal full history, clocks, side to move and outcome witness. A positive-clock
decisive tablebase result can remain unresolved and play on. Censored games never
become draws or receive Stockfish outcome substitutes. Policy-only tablebase
rescoring remains enabled; theoretical whole-game outcome rescoring is disabled.

Exactly all 96 attempts must pass source, history, inventory and custody checks.
All 32 TRAIN games must resolve and have usable retained rows with original raw
controls. All 628 fixed playing starts and their production mirrors must be absent
from the retained raw/mirror inputs of all 96 games before learning. A collision,
missing attempt or unusable TRAIN game prevents learning. There are no favorable
subsets or replacements after collection starts.

All HOLDOUT games remain outside learning. Valid censoring, zero rows or failure
to provide an eligible held-out root makes the entire offline diagnostic
inconclusive without independently vetoing the primary playing endpoint. Missing
or invalid provenance is an all-96 integrity failure. Complete offline metrics
require all 64 games and one score-blind NET root per game at offsets 8 through
80, excluding known prior exposure and all new TRAIN inputs and histories.
The frozen hash and chronological tie rule choose roots before predictions.
Expected-score error, draw and three-class Brier scores, and legal-policy drift
use 10,000 whole-game bootstrap draws. They are diagnostics, never endpoint selectors.

## Playing decision and stopping rules

Run three fixed 256-game, 128-opening-pair comparisons: FULLB40 versus FULLA40,
FULLB40 versus donor, and descriptive FULLA40 versus donor. Run all three even
when a performance screen fails, unless a global control or technical failure
stops the protocol. FULLA cannot be promoted from its context result.

Both FULLB screens must score at least 0.53 with a paired normal 95% lower bound
strictly above 0.5. The sampling unit is the mean of the two color-swapped games
in an opening pair, with sample variance using n-1 and z=1.96. A zero-variance
interval is unavailable and cannot pass. Complete legal games, full histories,
counts, cap handling and process custody are required; a successful launcher
alone does not qualify a partial arena.

Only when both FULLB screens pass and all three screens are technically qualified
may the same FULLB40 checkpoint play one 1,000-game, 500-pair confirmation against
the donor on the preselected disjoint book. Apply the same point and interval
rule. There is no extension, retry, reseeding, alternate endpoint, FULLA fallback
or pooling of screening and confirmation games. A complete zero-variance context
result remains descriptive; a context technical failure prevents confirmation.

A primary upper confidence bound below 0.5 indicates harm relative to FULLA at
this dose. An eligibility upper bound below 0.5 indicates donor regression.
Other nonpasses are unread at this dose and training realization. Even a confirmed
gain would be conditional on this internal six-piece/64-simulation profile;
external wall-clock comparisons, training replication and adoption need their
own evidence. None of these outcomes completes the months-long research goal.

## Qualification, resources and recovery

The accepted engineering producer exercised a retained strict-tablebase row,
a natural zero-row draw and a censored zero-row game. It passed the raw-value
controls and the public shard format's exact canonical storage comparison.
Its final attempt took 17.352 seconds. Both earlier failed attempts remain failed;
all three attempts cost 41.874 seconds, charged against the original 180-second
producer reservation. Engineering-only retention conditioning is explicitly
different from research thinning, which remains enabled. These games are excluded
from research. The accepted known-exposure supplement preserves six deduplicated
raw/mirror identities and four history hashes, including failed-attempt lineage
and its remaining unknowns.

Actual CPU checks covered current native imports and all 756 selected encodings,
the production builder and final collector configuration, 29 TRAIN controls and
94 normalized Trainer arguments. Historical 4+4-update A/A and 32-game arena
evidence is reused only within the documented source compatibility scope;
neither is represented as a new runtime experiment or proof of 40-step parity.
The offline reader's 44 controlled wiring cases are synthetic evidence, not an
actual held-out or model-inference result.

The complete study has a hard 21,060-second GPU-intended owner-wall ceiling,
including failed and interrupted qualifications. Stage ceilings are 1,200 seconds
for collection, 1,200 for both learning arms, 300 for offline inference, 3,600 for
each screen and 7,200 for conditional confirmation. Qualification reservations
remain fixed and cannot be replenished from failed attempts. These charges also
count toward the week's 172,800-second ceiling and hard deadline
`2026-11-19T16:09:20Z`; the earlier 5,859.448 seconds remain charged to the week.

Use one GPU stage at a time, two CPU threads on physical cores 20 and 22, and all
three existing leases. The user explicitly permits bounded GPU experiments beside
Call of Duty. Leave that application untouched and record contention; latency and
cost are conditional on the resource window. Clear inherited math overrides in
the fixed commands, then verify the actual declared math context without changing
Torch settings to force agreement.

Every stage checks controls, remaining budgets, deadline, memory, disk and process
closure before admission. Honor STOP, PAUSE, CANCEL, YIELD and intentional-stop
markers; never clear them. Preserve the donor, both failed qualifications and the
stopped 53-game study. Immutable component templates and explicit identity-only
binding contracts prevent scientific changes during late binding of cohort,
draw, checkpoint and publication identities. Technical failure yields an
unqualified or inconclusive study and a separate diagnosis. This preregistration
allows no second research collection, repeated training, playing or inference
attempt, or replenished allowance. A finite CPU correction may reuse the same
banked observations without new games, updates or predictions and without changing
the estimator or decision rules. A new research execution requires a separate
prospective study.

The prepublication review corrected offline phase directories so their unchanged
metadata lookup remains valid, replaced a stale root-bank example with the fixed
current output, pinned the exact CPU draw-plan binder, and clarified the no-reroll
rule. The [scientific review](evidence/rl-e2tb-week-one/review-final-science.json) and
[integration review](evidence/rl-e2tb-week-one/review-final-integration-and-draw-binder.json)
accepted the final bytes. The superseded candidate is preserved in the evidence
bundle; no research
games or updates occurred between candidates. Scientific parameters are unchanged.

## Readout

Research collection, updates, offline scores and playing results are unobserved
at preregistration. Append their actual provenance, costs and decision against
these rules here without rewriting the frozen preregistration.

### Accepted collection and common batches

The fixed collection completed all 96 attempts in 703.663 seconds of GPU-intended
owner wall time. It retained 8,348 positions: 3,012 from all 32 TRAIN games and
5,336 from all 64 HOLDOUT games. TRAIN had 15 natural and 17 strict-tablebase
endings; HOLDOUT had 33 natural and 31 strict-tablebase endings. There were no
censored games or replacement attempts. These outcome counts measure data
delivery, not strength against an external engine.

An independent CPU audit reissued all 96 outcome proofs, streamed all 24,096
inventory files and reconciled the 628-start raw/mirror exclusion gate. Its
21.423-second stage closed successfully. It did not decode arrays. A separate
12.329-second CPU stage ran the qualified TRAIN loader on all 32 games, reissued
their proofs, validated their actual arrays and saved the one common 40-by-256
draw plan. No HOLDOUT arrays, models or optimizers were loaded in that stage.
The draw contains 3,012 unique source row identities and samples 2,914 distinct
rows across 10,240 draws per arm. Both arms must consume that exact saved plan.

The [cohort acceptance](evidence/rl-e2tb-week-one-results/cohort-root-acceptance.json),
[independent audit](evidence/rl-e2tb-week-one-results/cohort-independent-audit.json)
and [draw acceptance](evidence/rl-e2tb-week-one-results/draw-root-acceptance.json)
record the actual identities and limitations. Bulk artifacts remain under
`~/chess-artifacts/operations/rl-week-one-20261112-v1/`. The frozen scientific
protocol remains unchanged. Offline diagnostics and playing decisions require
their own completed readouts below.

### Accepted paired learning delivery

Both FULLA and FULLB completed the fixed 40 updates and 10,240 sampled rows per
arm. The single GPU stage closed successfully in 64.865 seconds, with peak
allocated CUDA memory of 9,778,347,520 bytes. There were no skipped updates,
transient CUDA retries, extra batches or training retries. Across all 40 steps,
the recorded raw, host and device inputs match except for the declared
categorical targets; the only differing loss argument is the predeclared search
share of 1.0 versus 0.8. Recorded random states, optimizer ownership and actual
learning rates match between arms.

The independent CPU checkpoint audit completed in 15.359 seconds. It checked the
donor and both arms' snapshots at steps 0, 1, 8 and 40, including all 494 model
tensors, full architecture, finiteness, optimizer ownership and trained state,
scheduler and ZClip state, final hashes and immutable checkpoint custody. Both
step-0 models equal the donor, with empty optimizer state and no SWA. The final
arms differ in 476 tensor signatures; each trained snapshot changes 477 tensors
relative to the donor. Tensor changes establish delivery, not useful learning.
Each `step-40.pt` is a hardlink to that arm's final `trainer.pt`; intermediate
snapshots remain diagnostic only.

The auxiliary AdamW hard cap bound 39 of 40 updates in each arm. Its unclipped
gradient-norm medians were 8.109 and 8.374, above the 6.5 cap. The adaptive branch
did not bind, and Aurora matrices are outside this clipping scope. This is a
material limitation of the realized optimizer window; no setting was changed
during the comparison. The eight selected tensors do not measure whole-model
gradient interference.

The [learning acceptance](evidence/rl-e2tb-week-one-results/learning-root-acceptance.json),
[CPU checkpoint readout](evidence/rl-e2tb-week-one-results/learning-cpu-readout.json)
and [independent review](evidence/rl-e2tb-week-one-results/learning-cpu-independent-review.json)
record the technical qualification. Live process environment and native mappings
were not independently captured before the fast job closed; actual forward and
math-context claims come from the qualified instrumentation. No playing result,
Elo gain, checkpoint adoption or optimal recipe is established by this delivery.

### Whole-cohort offline diagnostic

All 64 HOLDOUT games provided an eligible, verified root under the frozen
score-blind selector and known-exposure exclusions. The CPU freezer took 14.313
seconds; the seven-model GPU scoring stage took 21.408 seconds; the CPU metric
readout took 7.206 seconds. Each model used one recorded batch of 64 identical
inputs, with BF16 autocast and inference mode. The actual policy head outputs
were float32 and WDL outputs BF16; exported logits use the declared CPU float64
softmax ruler. Peak allocated scoring memory was 940,549,120 bytes.

| Fixed model | Expected-score MSE | Three-class WDL Brier sum | Policy TV from donor |
| --- | ---: | ---: | ---: |
| Donor | 0.077660 | 0.463696 | 0 |
| FULLA40 | 0.072713 | 0.442234 | 0.052244 |
| FULLB40 | 0.071505 | 0.441048 | 0.051345 |

Both final arms show lower MSE and WDL Brier error than the donor on these games.
FULLB40 minus donor MSE is -0.006155, with the predeclared conditional bootstrap
95% interval [-0.010750, -0.002145]. FULLB40 minus FULLA40 MSE is -0.001209,
with interval [-0.003035, 0.000497]. The incremental outcome-bundle advantage is
unresolved: its MSE, WDL Brier and draw Brier intervals all include zero. Both
arms' draw Brier comparisons with donor also remain unresolved. Policy TV is
drift magnitude, not policy quality.

The [accepted diagnostic](evidence/rl-e2tb-week-one-results/offline-root-acceptance.json)
and [independent readout review](evidence/rl-e2tb-week-one-results/offline-independent-review.json)
preserve all seven snapshots, nine comparisons and 36 intervals. The bootstrap
uses 64 physical games, one root each, and 10,000 shared-seed resamples; the 448
model-root predictions are not independent samples. These diagnostic intervals
have no multiplicity adjustment. This is one training realization and cohort;
opening-family correlations and unknown donor pretraining exposure remain.
Repeat-forward stability and batch-size or compiled/eager numerical parity were
not qualified. No checkpoint selection or playing gate changes follow from this
diagnostic; the three fixed playing tests remain required.

### Primary playing screen: FULLB40 versus FULLA40

The primary screen completed all 256 games and 128 opening pairs. FULLB40 scored
81 wins, 88 draws and 87 losses: 0.48828125, with the predeclared paired-normal
95% interval [0.4465835543, 0.5299789457]. Its sample variance over opening-pair
means is 0.0579324557 and score standard error 0.0212743345. These are computed
from complete color-swapped pairs, without using the public summary's rounding.

The primary improvement gate did not pass. The interval includes 0.5, so this
also does not meet the registered harm definition. The outcome bundle has no
resolved playing advantage over the search-only control at this dose, profile
and training realization. Confirmation is unavailable under the frozen protocol;
both remaining donor comparisons are still required, and FULLA cannot replace
FULLB as a promotion candidate.

Technical qualification passed: 135 games ended through strict tablebase
adjudication and 121 through rules, comprising 92 checkmates and 29 next-move
threefold claims. There were no capped, incomplete or faulted games; the longest
added 241 plies against the fixed 450-ply ceiling. The GPU owner closed in 672.764
seconds and the CPU game/history/tablebase/statistics reader in 16.291 seconds.
Model, source, native, opening-book, configuration and registered tablebase
metadata identities matched before and after the run.

The [primary classification](evidence/rl-e2tb-week-one-results/primary-root-classification.json),
[complete game readout](evidence/rl-e2tb-week-one-results/primary-readout.json)
and [independent review](evidence/rl-e2tb-week-one-results/primary-independent-review.json)
preserve the result and custody. A live snapshot verified the actual child
command, two-core affinity, GPU visibility and declared environment, plus three
mapped native modules; two other native modules were not mapped in that snapshot.
There is no per-game served-state hash or per-forward precision/finite-logit
instrumentation in this arena. This is an internal fixed-simulation comparison,
not an Elo estimate or external-engine strength result.

### Donor eligibility screen: FULLB40 versus A39

All 256 games and 128 pairs qualified. FULLB40 scored 88 wins, 91 draws and 77
losses: 0.521484375, paired 95% interval [0.4763284427, 0.5666403073]. The paired
sample variance is 0.0679402990 and score standard error 0.0230387410. Neither
the point threshold nor the lower-bound threshold passed. The interval includes
0.5, so improvement remains unresolved and the donor-regression rule does not
trigger. Together with the primary nonpass, this precludes confirmation of the
preselected B40 in this study.

There were 125 strict-tablebase endings and 131 rules endings: 93 checkmates and
38 next-move threefold claims. No game was capped, incomplete or faulted; the
longest added 183 plies. The GPU owner closed in 647.340 seconds and CPU readout
in 15.283 seconds, with matching pre/post model and runtime-asset identities.
The [eligibility classification](evidence/rl-e2tb-week-one-results/eligibility-root-classification.json),
[full readout](evidence/rl-e2tb-week-one-results/eligibility-readout.json) and
[independent review](evidence/rl-e2tb-week-one-results/eligibility-independent-review.json)
bank the comparison. The descriptive FULLA40-versus-donor control remains
required; it cannot promote FULLA within this protocol.

### Descriptive control: FULLA40 versus A39

The final screen completed and qualified all 256 games and 128 pairs. FULLA40
scored 72 wins, 102 draws and 82 losses: 0.48046875, with paired 95% interval
[0.4357790239, 0.5251584761]. Pair sample variance is 0.0665446604 and score
standard error 0.0228008807. This descriptive interval includes 0.5; it establishes
neither a gain nor a regression of the common search-only training recipe.

There were 127 strict-tablebase endings and 129 rules endings: 93 checkmates and
36 next-move threefold claims. No game was capped, incomplete or faulted; the
longest added 228 plies. The GPU stage closed in 662.115 seconds and CPU readout
in 16.300 seconds. Pre/post asset identities match. The
[classification](evidence/rl-e2tb-week-one-results/context-root-classification.json),
[game readout](evidence/rl-e2tb-week-one-results/context-readout.json) and
[independent review](evidence/rl-e2tb-week-one-results/context-independent-review.json)
preserve the actual result. FULLA remains ineligible for promotion under this
study regardless of its descriptive score.

### Completed decision and research implications

| Candidate versus reference | W / D / L | Candidate score | Paired 95% interval |
| --- | ---: | ---: | ---: |
| FULLB40 versus FULLA40 | 81 / 88 / 87 | 48.828% | 44.658–52.998% |
| FULLB40 versus A39 | 88 / 91 / 77 | 52.148% | 47.633–56.664% |
| FULLA40 versus A39, descriptive | 72 / 102 / 82 | 48.047% | 43.578–52.516% |

All three screens technically qualified. Neither preselected FULLB40 improvement
criterion passed, and neither registered harm nor donor regression was
established. Retain A39. There is no confirmation, FULLA fallback, checkpoint
adoption or registered playing-gain claim. The
[official decision](evidence/rl-e2tb-week-one-results/screens-decision.json),
[independent decision review](evidence/rl-e2tb-week-one-results/decision-independent-review.json)
and [root acceptance](evidence/rl-e2tb-week-one-results/study-root-acceptance.json)
close the study. The machine label
`UNREAD_FOR_REGISTERED_GAIN_AT40_UPDATE_DOSE` means insufficient evidence of the
registered gain; all 768 games were read and qualified. No additional collection,
updates, inference or playing may be appended to this preregistration.

The study generated 96 research games and ran 80 optimizer updates, followed by
the fixed 768 playing games. All study GPU-intended owner stages, including
excluded producer qualifications and their preserved failures, consumed
2,814.029 seconds against the 21,060-second ceiling. Total week-one GPU-intended
owner time at this decision was 8,673.477 seconds against 172,800 seconds. These
are elapsed owner times, not exclusive GPU kernel time; concurrent user activity
was explicitly authorized. All 83 recorded owner stages and 333 recorded process
births were closed at the decision snapshot. Publication validation is additional
CPU work and is recorded separately.

Lower held-out value errors did not establish playing improvement. These results
are conditional on one donor, one cohort and training realization, 40 updates per
arm, and this fixed-simulation arena. The intervals are not equivalence tests and
do not establish that longer training or other outcome shares cannot help. The
three screens cannot be pooled into a common score or Elo estimate. The existing
external baseline remains 1 win, 24 draws and 39 losses against eight-thread
Rofchade 3.1 at its recorded 30+0.1 profile; this study supplies no new external
result. Beating multicore Rofchade and approaching Stockfish or Lc0 remain unmet.

The source and metadata teacher audit raises a separate question for a new
prospective study. The current Stockfish value annotation is queried after the
network move and flipped into record POV; its reply-policy target describes the
opponent's reply. These are not interchangeable with an optimal current-position
teacher value or a current-move teacher policy. Actual Stockfish gradient weights
were zero in both arms. Availability ratios and diagnostic positive-floor values
are not effective loss weights. Zero gradient weight alone does not establish
that annotations have no effect on data selection.

The donor's bounded supervised lineage trace identifies native root-inference
teacher outputs for 66,245,022 raw rows, but does not establish those teachers'
human/public model names or the complete lineage of the original supervised corpus. A replay
of the original supervised objective on new positions therefore needs further
qualification. These audits support a focused teacher-information or gradient
diagnostic on banked TRAIN data before selecting another recipe. They do not
show that teacher removal, auxiliary losses or clipping caused this inconclusive
result. A new study needs its own hypothesis, control, deciding metric, budget
and recovery rules; the week-one and longer-term strength program remains active.

The [results artifact manifest](evidence/rl-e2tb-week-one-results/artifact-manifest.json)
maps compact copies to the immutable external receipts. Bulk game logs, arrays
and checkpoints remain outside checkouts. The original scientific protocol and
its supporting evidence remain byte-identical to their preregistration.
