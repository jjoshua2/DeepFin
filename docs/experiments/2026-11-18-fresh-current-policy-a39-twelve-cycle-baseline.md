# Fresh current-policy A39: fixed twelve-cycle baseline (prospective)

**Current readout (M3):** [Accepted descriptive 400-game interim](#m3-descriptive-interim-readout): 49.875% versus fixed A39, paired 95% CI 46.01–53.74%. No gain or equivalence established; A39 retained. Four training cycles are technically complete, but this evaluates the saved cycle-3 snapshot. The prospective status below is preserved as historical text.

**Status:** proposed method/source preparation; no owner, bank, learner, model delivery or strength result. The exact independently reviewed Root-selected method must be published before this study's first compute. This document does not issue execution authority. Retained A39 remains the control.

The question is whether a modest repeated current-policy selfplay loop can produce a fixed-horizon development gain. F1's historical once pass and U2's KL1 followup were negative against A39; V10's search.1 followup did not establish gain. They do not identify a causal head failure or prove that fresh on-policy learning cannot work. Earlier loops did refresh actors, but under different high-dose/FIFO/replacement schedules. This package changes freshness, dose and batch together; it is a baseline/control comparison, not a sampling-only ablation.

## Frozen proposed recipe

Start only from retained A39 checkpoint `a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0` (donor step680276), with fresh student step0/empty optimizer. Run12 adjacent cycles:192 current-actor selfplay attempts each (2304 total), then one finite pass through that cycle's newly accepted bank. Both sides use the same current neural actor's Gumbel64 profile; Stockfish labels positions but is not the playing opponent. Exact bank actor SHA/step must equal the preceding genuine delivered model; cycle1 uses A39. No previous-bank replay/FIFO, replacement attempts, or alternative actor selection.

C3 architecture/objective/NoClip/NAdamW unchanged except the explicit target mixture: outcome.9 +retained-search.1, SF0, policy KL0/no teacher. B256, flat LR3e-6 (peak=floor=warmup-start, warmup0, period16384), mirror.5, one total presentation per stored identity including its mirror. Each fresh bank gets an independent standard uniform permutation; final partial batch is emitted once without drop/pad/wrap, with existing actual/B objective scaling. Require192 complete eligible games, zero unresolved, a nonempty bank of at most86400 retained rows; otherwise stop without replacement. Every selected retained-search vector must be usable; availability1/fallback0 and actual fraction.1 must be observed, using existing normalization rather than an exact stored-unit-sum test.

Model/core seed2026111890. For cycles1–12, collection seeds2026111901–2026111912 and order seeds2026112001–2026112012; final evaluation seed2026111999. Study `39e5d436-9930-5016-9030-d80671d8ac98`; persistent run/quota lineage `cb765533-b4ea-537a-b996-c09044ac3b81`. Collection UUIDs are fixed in the design; physical attempt IDs are that cycle's collection UUID +`:`+ordinal0–191. Preserve complete original source/game/row identity and actor genealogy. No semantic board deduplication or independent-game claim.

Updates are ceil(actual rows/256), not a chosen short prefix. The banked192-game cohort had17568 rows:69 updates, including160 final rows. At that illustrative density12 cycles would be210816 new views/828 updates; future counts are unknown. Max450 plies/game bounds each cycle at86400 rows/338 updates and cumulative4056 updates. B256 has real prior full-model execution evidence; it is not a promise of current throughput or sufficient learning.

Carry actual full model/optimizer/scheduler/ZClip/five RNG and the same locked cumulative ledger across cycles. Reserve quota before mirror/forward; failed tails cannot be rewound/retried, and no optimizer/ledger reset or failed-lineage substitution is authorized. Complete the first2 genuine cycles as an engineering condition: two current-actor banks, exact adjacent state/RNG restoration, source/typed hash/target/tail/quota checks and real CPU delivery. These are not two strength gains. There are no interim efficacy looks or arenas. A technical/data/resource failure stops the study and preserves its costs; no efficacy-based horizon or recipe revision.

## Openings and the single deciding test

Reuse exactly the existing192 TRAIN starts/roster under the accepted role pair; those starts were previously used. Different seeds and current actors generate new games, not unseen openings or guaranteed independent positions. Preserve original TRAIN_prior refusal rules. A separate pre-study owned CPU60/50 phase must reconcile current controlled use and explicit named-role provenance for the complete UNION of unchanged authenticated TRAIN_prior/refuse and a concrete playing_starts list of genuine known input/mirror pairs. The collector checks both current raw and mirror hashes against the prior flat identity set; no mirror relationship is invented for unpaired inherited hashes. Actual final DEV, HOLD and all six real playing-role coverage across the union needs genuine Root/ind acceptance. Missing coverage stops before collection; an empty fallback or untraced flat-index assumption is not sufficient. Never use protected panels for training or silently repurpose reserved roles.

The cancelled “5k” work means the **5,000-node Stockfish-label study**, not an opening panel; THREE is also a stopped study. Preserve both stop/cancellation rules, costs and unread partials. No cancelled-study data, models or protocols may be inspected/reused/audited/reencoded, and no successor tests/readers or replacement150k training/relabeling are authorized. The authenticated cancellation receipt explicitly permits broader independent DeepFin work while keeping THREE stopped. Existing general TRAIN_prior exclusion membership remains protective metadata, not a reopening of those studies. F12 starts retained A39 and is separate.


The final cycle12 endpoint alone plays retained A39 for800 games/400 color pairs on the exact existing **consumed CONFIRM_C_A39 panel**, explicitly reused DEV (book SHA `7e99de3a1033f1c6d3e3ef89966bda221c3f4f858a7358bc8ee6baa55110898e`). Matched Gumbel64, noise.75/c_scale.1/topk16/policy temperature1/game temperature.1, full16 history, strict rule50/TB3–6/net_raw, float32 transport/BF16 autocast, compile off, rolling32/eval-max-batch1024/two evaluator slots/no padding, Torch2 intra+2 interop/CPU20,22; max450 additional plies. Full active search/runner/serving projection and source223/Tbbb4/native/current-TB conditions are frozen in the design; only candidate identity, labels, paths, seed and new authority change from the accepted mathematical playing instrument. Frozen role separation/exposure metadata must be reconciled before compute. No intermediate checkpoint selection or pooling with old160/800 results.

Primary success: paired-normal95% lower score bound strictly>.5. Pair score is its two games' total/2; sample variance uses all400 pair scores with divisor399, SE=sqrt(variance/400), z=1.96. Zero variance is unavailable, not automatic success. Complete GPU technical Root/ind acceptance then closed CPU reader Root/ind must precede the final verdict. If the rule is unmet, no development gain is established and A39 stays retained; no generic harm/equivalence conclusion. Even a positive reused-DEV result does not automatically promote: genuinely fresh permitted-panel confirmation must be separately selected. Another separately frozen advancement against the accepted predecessor is needed for two successive gains.

## Resource and orchestration commitment

| Item | GPU seconds | CPU seconds |
| --- | ---: | ---: |
|12 collections,1800/1770 each |21600 |— |
|12 learners,600/570 each |7200 |— |
|12 audits180/170 +compilers60/50 +deliveries120/110 |— |4320 |
|Final800 arena3600/3570 +reader120/110 |3600 |120 |
|Pre-study protection projection60/50 |— |60 |
|**Study totals** |**32400** |**4500** |

Each failure is charged once and retained. The whole week GPU ceiling remains172800 seconds/cutoff2026-11-19T16:09:20Z; Root's reported pre-study actual was125734.75660899421 GPU/7646.391129419964 CPU, never reset to an older method snapshot. A possible7200 GPU future reserve is unselected and issues no owner/extension. Actual current census, complete remaining caps and deadline govern each admission.

The real loop needs five sequential owners/cycle: GPU collection →CPU recipe-free bank audit →CPU single-bank population/order compiler →GPU public capped learner/proof →CPU owned cumulative-journal delivery. Genuine Root/ind bank and model receipts then feed the next actor. Including pre-study protection and final arena/reader,63 numerical owners are planned. Root may orchestrate metadata autonomously under the fixed method, but independent receipts are authored at actual closures, not automatically invented. No new standalone independent compiler qualification is required.

Observed cost context: accepted192-game collection1064.712409 GPU seconds and audit78.116571 CPU seconds; old actual B256512-update learners used239.451863 (G1) and431.697826 (G4) owner seconds. These source/objective/replay-specific measurements support the proposed caps, not a guaranteed current-path ETA. Do not substitute arena wall or hot-loop time for collection/whole learner cost.

## Source readiness and limits

Reviewed capped sourceV3 retains V2 production bytes; prospective combined code-only target is`e633212dafc0d06507730114270d86fa7a22e4da`. It is not an adopted runtime; hosted CI/current source/native/profile/actual import custody remain concrete execution conditions. NAdam463 has genuine Root and independent14-case PASS (8CPU/6CUDA,42 reports, zero failures/errors/skips); this synthetic optimizer qualification is not a full-model speedup, strength gain or runtime adoption. Thin capped owned-launcher, compiler/protection projection, cumulative-journal delivery and playing adapters remain to be completed/reviewed. OldU1/U2 statuses cannot be relabeled. Reuse unchanged qualified playing/TB mathematics where source-bound; no duplicate blanket qualification is implied. All applicable CI and actual coherent source/native/freshness/current TB/SF checks still apply; merged/reviewed source is not adoption.

Terminal z estimates value under actual selfplay policies/exploration, unlike an SL strong-teacher estimate. Retained-search WDL is endogenous, including bootstrapped network information. Historical census4032 recorded collection-qualified groups/405013 rows/5–297 rows per group does not prove statistical independence; row-once is not game-balanced. No causal target/head/optimality/Elo claim follows. New game-balanced value weighting or SL-value anchoring would be separate unselected source/interventions; policy-only KL does not anchor WDL, and the shared trunk couples effects. A game-unseen value holdout would require new reserved whole groups, not a post-hoc row split of consumed history.

Exact scientific design/evidence refs are in [the compact record](evidence/2026-11-18-fresh-current-policy-a39-twelve-cycle-baseline.json). Actual source adoption, pre-study protection result, per-cycle banks/contracts/owners, final delivery and deciding result remain **UNISSUED**. No public-before-compute claim is made until Root actually publishes the exact selected record.


## Prospective engineering handoff addendum, November 18

The first collection was already running when source inspection exposed a missing
handoff artifact: `core_finite_collect.seal_readout` computes the complete sealed
output inventory but discards the return value. Its readout embeds every retained
row's validation metadata. Root cannot open that bulk readout or hash the roster
payload. The unchanged data auditor requires a hash-pinned producer inventory in
its protocol before launch, and the accepted-bank consumer authenticates that
same protocol in the recorded owner command. Writing it late inside the auditor
would violate that contract.

Before any extra handoff owner, Root selects one bounded engineering projection
per successfully closed collection, at most 12 owners of CPU 60 seconds each
(inner 50 seconds), with no automatic retry. It reads the existing sealed producer
metadata inside that owner, derives game inventory entries from the authenticated
producer descriptors, hashes the named root JSON files, and reconciles canonical
regular read-only file paths and recorded sizes against the complete output tree.
It writes an external producer inventory and a compact summary with identities,
counts and references, without row, history or roster contents. It does not open
raw shard/NPZ/model payloads or construct arrays, models or encodings. Descriptor
derivation is not independent payload verification: the original CPU 180-second
data audit must still hash and reconcile the full actual output tree.

These are necessary artifact-producing operations, not additional model tests or
replacement collections. Reviewed source and actual closed-producer bindings,
separate Root admission, current custody and a fresh resource check are required.
A failed handoff stops this study under its existing engineering stop rule. All
partial artifacts and costs remain recorded; no source or terminal bytes are
changed and no extra game attempts are introduced.

The original 63 scientific owners retain every selected bound, including CPU
4,500 seconds and GPU 32,400 seconds. This addendum adds an explicit ceiling of
720 CPU owner-wall seconds for up to 12 engineering handoffs: the combined F12
ceiling is CPU 5,220 seconds and GPU 32,400 seconds, across at most 75 owners.
Charge actual handoff costs separately and include them in all-in F12 and global
costs. The week cutoff and global GPU ceiling remain unchanged. No unused CPU
allowance authorizes GPU work or a recipe/horizon change.

The original V4 method identity, publication-before-collection evidence, fixed
2,304 attempts, actor progression, one presentation per original row, value blend,
learning rate, seeds and final paired 800-game DEV endpoint remain unchanged.
This operational addendum was selected while cycle 1 was running, before any F12
training, arena, handoff owner or efficacy result. It does not establish an RL gain.


### Subsequent audit-summary handoff correction

The first unchanged CPU data audit subsequently closed PASS (session 66710,
62.26493861700874 owner-wall seconds). Its sealed readout is 5,863,920 bytes and
its candidate manifest is 5,606,141 bytes; Root observed their sizes without
opening or hashing either file. The public CLI reports the candidate reference
but omits the readout reference and compact deciding metadata. This exposes the
same reporting gap at the audit boundary.

Before any additional summary owner or F12 training/arena, Root selects one
CPU 60-second operation (inner 50 seconds) to extract compact metadata from this
already completed audit. It reads only the sealed audit protocol/specification,
readout and candidate JSON, computes genuine references, and emits identities and
counts without game inventories, histories, row details or payload contents. It
preserves the existing audit and requires its actual closed owner; it does not
repeat the data audit or promote the bank. Root and independent bank acceptance
still follow the original consumer contract.

For subsequent cycles, a separately reviewed private wrapper emits that compact
summary after the unchanged public data audit runs once, within its existing
CPU 180/170-second owner. Its actual command keeps the required public module and
protocol suffix. The companion summary is written at a declared fresh external
path after the audit seals its own output; public runtime/source, data and audit
results remain unchanged. No additional per-cycle summary owner is selected.

This explicitly adds one engineering owner and 60 CPU seconds to the earlier
allowance: maximum engineering CPU 780 seconds, combined F12 CPU 5,280 seconds,
and at most 76 owners. The prior 720/5,220/75 figures above record the earlier
selection and are prospectively superseded by these combined ceilings. The
original 63 scientific stage bounds and GPU 32,400-second ceiling remain fixed,
as do all games, seeds, recipe, row quotas, endpoint and week cutoff. No automatic
retry, extra collection or efficacy-based change is authorized. This correction
was selected after a successful data audit and before any F12 training, arena or
efficacy result; it is implementation readiness, not a failed or repeated audit.


### Prospective manual exception for the cycle 2 pre-execution admission failure

The first cycle completed technically, and cycle 2's unchanged 192-game bank was
collected, audited and compiled. Its first learner owner (session 59758) then
failed before creating output or entering source/native/runtime checks or the
learner route. Root had set an admission deadline only 180 seconds ahead for a
600-second outer owner. The private gate correctly rejected that envelope. The
failed owner used 1.2667814940214157 GPU owner-wall seconds, which remain charged.
There were zero new training updates and no model/bootstrap, RNG, draw or ledger
operations. The original failed receipt and its engineering-stop disposition are
preserved; the original plan did not pass.

A second command-envelope metadata error was also found: the failed command's
top-level contract and learner-stage references still named cycle 1, while its
actual child and public proof suffix correctly named cycle 2. It did not cause
the deadline refusal or execute a different learner. A corrected envelope must
bind those top-level references to the actual plan as well as the child command.

The V4 rule that any technical failure stops this study was triggered. The
following is a proposed, separately reviewed prospective manual operational
exception, not continuation under that failed admission and not an automatic
retry. After independent amendment review and public recording, Root may select
exactly one new named owner for this proven pre-route admission error. It has
outer/inner bounds of 600/570 seconds, a genuinely new admission and fresh
resource/custody check, and a deadline computed from issue time plus the outer
bound plus 180 seconds, within the unchanged week cutoff. Any further failure
stops the study; no additional recovery is selected here.

The same genuine M1 parent, original A39 origin, cycle 2 method/contract/stage,
19,295-row bank, compiled order, training seed, full-state continuation and locked
lineage ledger are retained. No numerical state is rewound or redrawn: the failed
attempt never entered that route. The corrected learner still has 76 updates,
step 69 to 145, batch 256 and final batch 95. The original wrapper/learner output
paths were never created and remain the selected fresh paths. There is no new
lineage, checkpoint selection, replacement collection or scientific intervention.

The versioned GPU-scope amendment retains all 25 original named stages, including
the failed cycle 2 learner, and adds one explicit recovery stage (26 total).
Maximum all-in owner count becomes 77. Aggregate F12 GPU remains 32,400 seconds,
global GPU remains 172,800 seconds, all-in CPU remains 5,280 seconds and the week
cutoff is unchanged. Every actual success and failure is counted once; each next
full admitted cap must fit the remaining aggregate budget. Adding a stage ceiling
does not add an aggregate GPU allowance.

The scope retains its genuine family UUID and original creation timestamp only
to reconcile the already counted original owners. This amendment is dated
separately after the failure. That historical creation time does not authorize
the added stage retroactively: manual Root selection, independent binding review,
fresh U3 and actual admission must follow this public record. The original
engineering stop is superseded only by that explicit reviewed manual selection.

Future phase bindings must carry the amended scope and include all actual costs.
The final-playing metadata bridge will receive a narrow reviewed scope update
before M12; no extra numerical qualification or arena is added. The twelve-cycle
scientific recipe, horizon, seeds, exact-once row quota and sole final paired
800-game reused-DEVELOPMENT endpoint remain unchanged. No interim strength result
or efficacy-based decision motivated this exception, and A39 is not promoted or
replaced by this operational record.


## User-requested M3 and M6 descriptive evaluations — prospective amendment

After M3 was saved and before any interim evaluation, the user requested two
400-game views against the same original A39, while retaining the final
800-game M12 endpoint. This changes the prospective evaluation schedule,
not the fixed12-cycle training recipe, data/order/dose, optimizer, source or
horizon. The original prohibition on pre-M12 efficacy reads is superseded
only for these registered descriptive endpoints; the original record remains
visible above. Healthy jobs continue to their natural boundaries.

|Checkpoint|Games / color pairs|Opening selection|Role|GPU cap / inner|Reader cap / inner|
|---|---:|---|---|---:|---:|
|M3|400 /200|First200 of the original400 pairs|Descriptive only|1800 /1770s|120 /110s|
|M6|400 /200|The same first200 pairs|Descriptive only|1800 /1770s|120 /110s|
|M12|800 /400|Original full400 pairs|Original final screen|3600 /3570s|120 /110s|

All edges use fixed A39, seed2026111999 and the same frozen T playing profile:
Gumbel64, noise, scale0.75/c0.1/topk16/policyT1/gameT0.1, full16 history,
strict3–6man/rule50 TB, eager BF16/F32, rolling32/eval1024/slots2/no padding
and CPU2. The panel has already been consumed as DEVELOPMENT. The owned
wrapper must authenticate/load the full400-role order and take exactly
`[:200]`; requesting200 directly from the oversize public loader can sample
instead. The final full400-pair path remains unchanged.

M3 is the immutable cycle3 model at step223 (accepted Root57240/inde1897);
its checkpoint is `cc6bc2cf…`, typed model state `f2011177…`. M6 and M12
require their genuine future cycle6/cycle12 deliveries; no future hash,
checkpoint choice or step is invented. The intended M3 view is after healthy
C4 closes, before C5 if source/readiness permits. M6 follows its accepted
delivery. No interim score changes the recipe, horizon or training sequence.

Interims report paired point/normal95 with n200, variance divisor199 and
z1.96, with `descriptive_only` and criterion false. Final inference keeps
n400/divisor399/z1.96 and the original available-lower95>0.5 screen; a
zero-variance interval stays unavailable. These nested repeated-panel views
are correlated and pointwise, without pooling or multiplicity claims. Each
GPU must close with genuine technical Root/independent acceptance before
its owned reader; actual M6/M12 receipts are not prerequisites for M3.

Existing readers also report the conditional Elo-equivalent difference
against fixed A39: `400 log10(score/(1-score))`, with the same monotone
paired-CI transform. This is not an external rating. Boundary/unavailable
values use null plus explicit flags, never NaN/Infinity. No daily-rate
calculation or forecast is selected from short/noisy correlated snapshots.

The user's goal is rapid, sustained clear improvement toward Stockfish.
The approximately +100 Elo/day figure was a personal made-up estimate, not
a measured AlphaZero result or a hard numerical gate. It is not an automatic
pass/fail, stop or adaptation rule, nor an observed or guaranteed rate.
Technical completion, an interim score, or even a positive reused-DEV final
screen does not establish achievement of that broader goal.

The versioned GPU scope retains the same family UUID, original creation time
and all26 old names/costs, including failed477 and its manual recovery, then
adds exactly the two interim1800-second names. New endpoint authority is
prospective; the old scope timestamp does not backdate it. GPU32400,
global172800 and the Nov19 16:09:20UTC cutoff remain. Two reader owners add
240scientificCPU seconds: science4500→4740, all-in5280→5520, maxowners77→81.
Engineering780 is unchanged. No extra training, collection, protection,
model-delivery or qualification owner is added; no retry is selected.

Actual485-owner snapshot charged3583.644023771165F12GPU seconds. Including
C4's full unclosed cap, all remaining9collect/9learn/final/interim full caps
would total32383.644023771165F12GPU seconds, within32400; actual costs are
never reset. F12 chronology starts with its first scientific protection
owner465 at14:03:11.497911UTC; M3's saved-owner end is17:44:43.132243UTC.
These are current-study timestamps only: Nov12week-one prior negative
results, costs, engineering and calendar history remain, and cannot be
hidden to claim a daily gain. M6/M12 timestamps remain actual future facts.

The amendment requires independent review, public record and actual Root
selection/source/binding checks before any new interim clock. Healthy C4's
previously admitted scope is retained. Existing technical/data/resource
stops and A39 retention remain; no automatic Goal-complete or promotion is
authorized by a descriptive or positive DEV result. Full named receipts
and limits are in the [compact evidence](evidence/2026-11-18-f12-user-requested-interim-evaluation-amendment.json).


## M3 descriptive interim readout

The registered saved M3 checkpoint (cycle3, step223, checkpoint `cc6bc2cf…`,
typed model state `f2011177…`) completed its fixed 400-game /200-pair
DEVELOPMENT comparison against original A39. Four training cycles were
technically complete before this evaluation; M4 at step303 was not the
evaluated model. The candidate was fixed before play, without checkpoint
selection from the result.

|Observation|Accepted result|
|---|---:|
|M3 wins / draws / losses|123 /153 /124|
|Paired match score|49.875%|
|Pointwise paired normal 95% interval|46.0149–53.7351%|
|Conditional Elo-equivalent versus fixed A39|−0.87 (95% interval −27.75 to +26.00)|
|Actual GPU owner wall / reader CPU owner wall|1067.5090s /21.2816s|

The owned wrapper authenticated the original consumed 400-pair DEV order
and used exactly its first200 pairs, with reciprocal colors, seed2026111999
and the same frozen playing profile. All400 games and200 pairs reconciled;
the GPU and reader each closed with genuine Root and independent technical
acceptance before the frozen descriptive verdict. Inference uses n200,
sample-variance divisor199 and z1.96. The five pair-score bins were
`[18, 50, 67, 45, 20]`; sample variance was0.07757380653266331.

This screen establishes neither a gain nor equivalence to A39. The interval
allows deterioration and improvement under these settings. It is a
descriptive, pointwise view of an already consumed and repeatedly reused
panel; the M3/M6/final views are correlated. There is no pooling, fresh
confirmation or multiplicity-control claim. The Elo-equivalent is only the
monotone conversion of this conditional match score and its paired interval,
not an absolute external rating or a daily forecast.

The prescribed interim criterion is nonselecting: `descriptive_only=true`,
`criterion_pass=false`, and no result changes training, the twelve-cycle
horizon, data/order/dose, recipe, checkpoint choice or stopping rules.
Training continues unchanged toward the M6 400-game descriptive check and
the M12 800-game final screen. A39 remains retained; this result does not
establish rapid sustained improvement, goal completion, adoption, a causal
head diagnosis or a verdict about RL generally.

M3's saved learner owner ended at17:44:43.132243UTC; it was evaluated after
M4's accepted delivery. Current-study chronology begins at14:03:11.497911UTC,
but does not reset the Nov12 week-one negative results, engineering/calendar
history, costs or stops. The user's rough personal +100 Elo/day estimate
was not measured AlphaZero evidence and is not a hard numerical gate. No
new arena, numerical qualification, retry or training change is selected
by this readout.

The accepted frozen verdict is `3746973fc8bcff41ec58c5b48072e1eb8ca14079099838d3a92c99e5d7faaae9`;
the reconciled summary is `ab28a61a416319a1a724a0906fe8949acee58870e3644c33e450f061f890e058`.
Exact model, profile, amendment/public chronology, technical receipts and
actual owner costs are in the [compact M3 evidence](evidence/2026-11-18-f12-M3-interim-readout.json).
