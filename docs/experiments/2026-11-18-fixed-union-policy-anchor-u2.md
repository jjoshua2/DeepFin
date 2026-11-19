# U2: frozen-A39 policy anchoring on the fixed self-play union

The local method and independent review were frozen before the first U2 GPU
admission. This public record is being prepared **after the first segment closed**,
after the completed V10 decision and first U2 launch. It is not a claim that public
preregistration appeared on Main before launch. The original local method
requested public preregistration before compute; that publication did not occur
before this start. This record makes that chronology explicit and preserves
the earlier choices without backdating them.

Root selected the unchanged U2 method after V10 selected no mitigation:
V10 scored50.8125% against frozen F1 and47.4375% against A39, with neither
predeclared lower95>.5 criterion passing. U2 is a separate KL-only comparison,
not a combination with V10's10% retained-search intervention. Original A39,
the SL donor, remains retained. No final U2 model or playing outcome is reported here.

## Frozen question and intervention

Does an additive frozen-A39 reference-policy anchor limit harmful recipe-level
drift while learning the same historical population, and does the resulting
endpoint show development advantage over the donor? Prior playing scores do
not establish policy-head, WDL-head or shared-trunk causality. Coefficient1 is
one prospectively fixed choice, not a known optimum.

Use the existing legal forward KL in the reference-to-student direction:
`L = L_CORE + 1.0 * mean_eligible KL(q_A39 || p_student)`. The reference and
student policy distributions use the explicit legal mask; eligible rows are
original network turns with a policy target. This is additive regularization,
not a normalized objective mixture or policy-target replacement. The A39
reference must remain frozen, in eval mode, with no gradients or optimizer
ownership. Required own-live evidence records actual teacher custody,
eligible-row/KL diagnostics and reference forwards; source acceptance alone
does not assert those facts for the running full-model trajectory.

Keep the base main-policy and WDL weights1, all auxiliary weights0, and
outcome-only WDL: retained-search share0 and SF share0. Keep C3 architecture
schema20,175 input planes and61,444,448 unique parameters. Bootstrap a genuinely
fresh A39 model-only student from donor recorded step680276; student step0,
fresh optimizer/scheduler/scaler/RNG, model seed2026111780. No failed, F1 or
V10 learned state transfers. Use NAdamW, flat LR3e-6, no warmup or clipping,
first period16384, B32, AMP, two threads and mirror probability.5.

Reuse the authenticated4,032 **source assets** containing405,013 stored rows.
These are not asserted to be4,032 independent games. Keep the same compiled
membership/origin map and global uniform order, seed2026111777, without
replacement; no recompile, filtering, priority, bounded shuffle, prefetch or
extra pass. Recorded copied assets are counted once; natural repeated boards
and openings remain allowed. Original and mirror share one row presentation.

Two adjacent closed segments use[6336,6321] optimizer updates. Segment1 is
202,752 views; segment2 is202,261 views, including one real21-row tail. Total
is12,657 updates/405,013 views. Segment2 resumes the same full model,
optimizer, scheduler, scaler, five RNG streams, permutation/cursor, journal
chain and ledger; it does not bootstrap another student or reset quota. The
whole final objective, including the anchor, is scaled21/32 and performs one
real final optimizer/scheduler/weight-decay step. Only the complete segment2
endpoint is eligible for playing; segment1 is an operational handoff, never
a model selected by efficacy.

## Two fixed800-game development endpoints

Compare that same final U2 endpoint first against immutable F1 (seed2026111801),
then against immutable A39 (seed2026111802). Each match is800 games/400
reciprocal pairs on the already consumed CONFIRM_C_A39 panel, reused explicitly
as DEVELOPMENT, not unused/reserved data or fresh confirmation.

Keep64 matched simulations, Gumbel.75, c-scale.1, top-k16, policy temperature1,
game temperature.1, full16 history,450 additional plies, rule50 TB3–6, eager
BF16 with float32 transport, rolling32 games/evaluation batch1024/two slots,
no padding and two threads. Existing qualified playing implementation/source
is reused. No new whole-tablebase qualification is implied.

BOTH full GPU endpoints must close and receive Root and independent technical
acceptance before ANY efficacy reader/look. Both accepted reader pairs must
then exist before the joint verdict. No interim scores, score-based ordering,
count changes, extensions, rerolls, pooling or intermediate-model selection.
Each endpoint requires800 legal complete games/400 reciprocal pairs, no
crashes, and exact model/source/profile/role/TB/row/custody reconciliation.

Primary mitigation requires an available paired-normal95 lower bound>.5
against F1. Donor advantage requires that primary pass AND an available lower
bound>.5 against A39. Use reciprocal-pair score as the sampling unit and
sample-variance divisor399; zero variance is unavailable and cannot pass.
The fixed deciding operation is the accepted Root controller's `verdict`
command after both actual reader pairs, against this unchanged method. Report
both separate endpoints; no pooled estimate or retrospectively changed rule.
The exact accepted deciding CLI shape is fixed below. The actual final-reader
authority path/hash must be filled and the command frozen before either arena;
this display template is not an issued command:

```text
/usr/bin/python3 -I -B <ARTIFACT_ROOT>/historical-selfplay-policy-anchor-U2-playing-root-admission-source-preparation-v1-corrected-v1/root_control.py verdict --authority <ARTIFACT_ROOT>/historical-selfplay-policy-anchor-U2-vs-A39-800DEV-bindings-v1/reader.root-authority.json --authority-sha256 <ACTUAL_FINAL_READER_AUTHORITY_SHA256>
```

Any apparent gain needs a separately selected fresh confirmation. No
familywise-control, Elo, external-engine strength or adoption claim follows.

## Recorded start, existing qualification and future work

Local method047ea2d5 and independent review9276906e preceded Root's post-V10
selection at2026-11-18 07:33:09 UTC and first admission at07:39:17 UTC. The
actual first plan binds contract7238f741, parent-null stage1, original finite
driverbe4583d6/49b0f342,6000 outer/5970 inner seconds, and method/scope/source
identities. Admission33ea2b98 and commandb25ac65a are issued. Supported session81141 subsequently closed with exit0; its independently
accepted partial technical progress is recorded below. This author read only
named compact immutable launch/source/authority/closure metadata, not live
logs, ledger, journal, models, shards or process state. No complete-pass endpoint,
isolated teacher throughput or playing gain is asserted.

The frozen Cea runtime and its1,400 source/config pins reuse closed CPU443:
206 cases/12 modules passed, native profile0143e75d and source composition
accepted. Explicit Root receipt51efff67 resolved finalized terminal/process
metadata sealing with bytes unchanged. This does not claim full-model GPU
trajectory equivalence to F1's older runtime. The source revision and
two-segment execution also differ from F1's single-process run, so matching
seeds/recipe does not establish GPU bit identity or isolate a head mechanism.
No new build, compiler, blanket qualification or neural smoke is selected.

Accepted final-delivery CPU sourcee291/aee12, playing sourced32ee/ba68/dde9
and corrected Root controller8fc103/293022 are future dependencies, not
results or admissions. Future CPU120/110 calls the genuine public anchor
closed-delivery consumer once and runs14 synthetic adapter cases; those14
cases are UNRUN in this U2 preparation snapshot. Actual Root9/ind11 delivery
requires genuine closed full-pass model evidence. The c453 second-segment
parameter handoff is inert source-only material with no separate independent
review or actual segment2 authority; it does not manufacture a parent result.
Documentation creates no additional CI/NN gate and grants no execution permission.

## First segment: accepted partial technical progress

Root closed supported session81141/exit0 at2026-11-18 08:37:20 UTC, before
this public record was prepared. Root and independent receipts accept only
segment1's partial pass for adjacent same-family continuation:6,336 updates,
202,752 views, cursor202,752/batches6,336 and last batch32. `pass_complete`
is false. This is neither a final delivery nor a playing-eligible endpoint.

The passive own-live/saved certificate records494 finite state entries,
61,444,448 unique parameters and175 planes. There were6,336 student forwards
and optimizer pre/post calls, one save, zero extra student forwards and one
final verification decode inside the learner owner. The saved checkpoint
identity is banked; this publication did not reopen it. Save observations
preserved model/optimizer and five RNG streams.

The partial trajectory actually exercised coefficient1:202,752 eligible rows,
6,336 reference forwards and frozen/eval/no-gradient/no-optimizer teacher
custody. Its recorded trajectory mean KL was0.023801901881299017. That is an
on-trajectory training diagnostic, not a held-out comparison, calibrated
trust region, isolated teacher GPU-throughput measurement or playing result.
It confirms this first segment's intervention was exercised; source acceptance
alone did not establish it. Full segment2/final custody still require proof.

Charged GPU owner-wall time was3412.132668819977 seconds; process CPU3118.29
seconds is telemetry inside that GPU owner, not another CPU-owner charge.
All453 owners then closed. U2 GPU actual was3412.132668819977, global GPU
116922.86435613304 and global CPU7552.043652844906 seconds. The original
19,200 GPU/360 CPU ceilings, per-segment limits, cutoff and every earlier
cost/stop remain unchanged. No unused capacity grants an extension.

Second-segment parameter binding was prepared from the genuine closed parent;
this snapshot issues no segment2 authority, admission, clock or result. It
permits no reset, new donor/optimizer, extra quota or failed-parent resume.
Its actual execution and subsequent final/playing receipts belong to later
dated updates, independent of when this documentation is published.

## Budget, stops and limits

Each learner segment is6000/5970 GPU seconds; each arena3600/3570. U2's new
GPU ceiling is19,200 (12,000 training plus7,200 playing). CPU ceiling360
seconds comprises final delivery120/110 and two readers120/110 each. One
numerical owner at a time, affinity20/22, two threads, no paid compute and
the2026-11-19 16:09:20 UTC cutoff remain. Caps include loading, teacher,
checkpoint/proof overhead and are not completion or throughput guarantees.

At Root's post-V10 selection, global charged GPU/CPU baselines were
113510.73168731306/7552.043652844906 seconds. The full new GPU cap would reach
132710.73168731306, below172800. The method's older103834.38945806609 GPU
snapshot is historical and is not restored. All earlier failed and successful
costs, stops, source/ledger/model histories and negative observations remain.

Source/target/nonfinite/forward/backward/optimizer/quota/integrity/resource/
wall faults stop with evidence, costs and any tainted reservations retained.
Only a genuinely closed valid same-family parent may continue. No failed-tail
rewind, quota reset, silent teacher replacement, filtering, automatic retry
or cap extension. A technical failure leaves efficacy unread; donor retained.

Row-once is not game balancing, independent outcomes, semantic deduplication
or globally unseen training data. F1 and V10 already used this population;
failed437 used a prefix, and historical training adds further exposure. U2
intentionally makes a separate fresh-student pass. Earlier pre-V10 exposure
bounds in the local method are dated snapshots, not a current global-use cap.
Positions in a game share outcomes and long games retain more row influence.
Reference anchoring does not itself fix correlated outcome labels or establish
a conventional online/current-policy refreshed self-play baseline.

**Final outcome: pending.** Segment1 has the accepted partial technical
result below. Segment2, final model delivery, both GPU endpoints, both readers
and the fixed joint verdict remain unreported in this closed453 snapshot and
require their own subsequent actual closed receipts. No U2 strength, successful mitigation, donor advantage, causal
mechanism, optimum coefficient, adoption or goal achievement is claimed.
Append the completed readout to this record when genuine endpoints exist.

[Compact method and in-progress evidence](evidence/2026-11-18-fixed-union-policy-anchor-u2.json)
pins the frozen choices, chronology, actual first admission and scoped source reuse.

## Training complete and final CPU closed through owner455 (2026-11-18)

This unpublished append records the closed455 technical snapshot at
2026-11-18 10:10:52 UTC. The published record above, including the missed
public-before-start chronology and earlier partial453 snapshot, remains
unchanged. Both800-game GPU endpoints, both efficacy readers and the joint
verdict are **PENDING/unread in this snapshot**. No playing score is asserted.

| Closed learner segment | Updates | Views | Final cursor | GPU owner-wall seconds |
|---|---:|---:|---:|---:|
|453, segment1 |6,336 |202,752 |202,752 |3412.132668819977 |
|454, segment2 |6,321 |202,261 |405,013 |4708.78190848208 |

Root and independent technical receipts accept both segments. The second
started from the genuine first boundary with full model/optimizer/scheduler/
scaler, five RNG streams and the same ledger/quota/journal state. Its initial
full-state fingerprint equals the accepted parent's final state. The complete
pass reached12,657 updates/405,013 views, including the real21-row tail;
`pass_complete` is true. The4,032 units are source assets, not an assumed
independent-game count. No additional learning, replay pass or quota reset is
selected by this completion.

The final saved certificate records494 finite state entries,61,444,448 unique
parameters and175 input planes on float32 cuda:0. Its checkpoint identity is
`611d82ab7048f7cbe497e1569b385b3dc17f8c71fdfda9b2565bb1479e3d4213`
and saved-model state hash is
`25e9034c0fdc323fea0c75f3f0d6a9c631ada39d1c7238722b575ca38f97414c`.
These are metadata descriptors; the publication author reopened no checkpoint
or model payload. Each learner owner recorded one save and one final
verification decode, with zero extra student forwards and passive saved/RNG
observations. Final CPU work is separate from those in-owner decodes.

Coefficient1 remained active with frozen/eval/no-gradient/no-optimizer teacher
custody. The two segments recorded202,752/202,261 eligible rows and
6,336/6,321 reference forwards. Mean trajectory KL was0.023801901881299017
and0.029690278866548004 respectively. These are training-trajectory diagnostics,
not held-out policy comparisons, calibrated trust regions, isolated GPU
teacher-throughput measurements or evidence of playing improvement. Value
search/SF shares remained0; U2 was not mixed with V10.

Final CPU session93344 closed with supported EOF/exit0, charging
20.156058119027875 CPU owner-wall seconds. It invoked the genuine public
closed-model-delivery consumer once and passed all14 synthetic adapter cases
(4 functions/one module). It recorded zero real checkpoint decodes, raw journal
reads, row draws, real-model constructions or forwards. The1,400 source pins
and existing qualified native profile remained unchanged; no native rebuild,
compiler rerun or new broad qualification occurred. Root issued the compact
nine-field final model-delivery receipt. Independent eleven-field delivery
acceptance was **PENDING at this Root closed455 snapshot**, so it is not
preaccepted or inferred from the tests here.

Actual total U2 training GPU time is8120.914577302057 seconds; U2 CPU-owner
time through455 is20.156058119027875. The shared historical CPU443 engineering
qualification is not charged as a new U2 CPU phase. Process CPU3118.29/4210.93
seconds inside the two GPU owners is separate telemetry, not another pair of
CPU-owner charges. Global GPU after training is121631.64626461512 seconds.
Original19,200 GPU/360 CPU study ceilings, two3600 GPU arena caps, two120 CPU
reader caps, global ceiling/cutoff and all older failures/costs/stops remain.
Unused learning capacity grants no extra arena, variant or automatic retry.

The future completed readout will append actual independent model delivery,
both accepted full GPU endpoints, both reader pairs, the frozen joint verdict
and final Root costs. It must report each fixed800/400 endpoint separately,
its pair bins and prespecified n399 interval/availability and thresholds. Both
GPU endpoints precede any efficacy reader; both accepted reader pairs precede
the joint verdict. Nothing in this technical snapshot establishes mitigation,
donor advantage, causal mechanism, optimal lambda, Elo, external strength or
adoption. A39 remains retained. No PR is selected until actual complete results
and independent readout review.

Row-once still does not make positions within a game independent or equalize
per-game influence. A separate source-only game-weighting feasibility report
has independent review, but exact authenticated grouping is unknown and no
third experiment or code is selected. Keeping a direct policy term unchanged
would not prevent shared-trunk updates from changing learned policy. This
feasibility work is not part of U2's intervention or evidence of its effect.

## Completed fixed development readout (closed459, 2026-11-18)

The final independent model-delivery receipt subsequently accepted the closed455
full-pass model. Both fixed GPU endpoints then closed with Root and independent
technical acceptance before either efficacy reader. Both reader pairs were
accepted before the genuinely frozen joint deciding command was applied.
All459 owners were closed at this snapshot. The earlier method-publication
chronology and dated453/455 pending statements above are preserved as history;
they are not current result claims or a retroactive public preregistration.

| Fixed development comparison | W / D / L | Score | Paired normal95 interval | Lower>.5 |
|---|---:|---:|---:|---|
| U2 vs frozen F1 |237 /322 /241 |49.75% |[47.23049468793101%,52.26950531206899%] |False |
| U2 vs original A39 |202 /322 /276 |45.375% |[42.818889931123755%,47.93111006887624%] |False |

Each endpoint contains exactly800 games/400 reciprocal pairs. Pair-total bins
for[0,.5,1,1.5,2] are[30,96,149,98,27] against F1 and[42,113,144,79,22]
against A39. The observed pair-score sample variances are0.06609649122807017
and0.06803101503759398, using the predeclared divisor399. Both intervals are
available; zero-variance-unavailable handling was unchanged. The two results
remain separate, with no pooling, count extension, efficacy-selected checkpoint
or retrospectively changed threshold.

The frozen verdict is **NO_PRIMARY_MITIGATION_SELECTION**. U2 did not
demonstrate mitigation relative to frozen F1: that interval includes50%.
Against original A39, the interval lies below50% for this exact endpoint.
Neither the primary criterion nor the combined donor-advantage criterion passed.
Original A39 remains retained. This is a conditional negative result for the
selected KL1 recipe/checkpoint/horizon and reused panel, not proof that KL
generally fails or identification of a policy-head, value-head or shared-trunk
cause. It does not establish an optimal coefficient, equivalence, Elo or
external-engine gain, adoption, retry or another experiment.

Both closed readers reconciled all game rows, reciprocal colours and pair
order, with zero incomplete pairs or illegal/crashed games. Each passed its16
controls and loaded no model or CUDA context. Root/ind receipts bind the same
final494-state model, sources, playing profile and consumed-panel DEVELOPMENT
classification. These engineering controls establish measurement integrity;
they do not turn the consumed panel into fresh confirmation or establish
familywise inference across prior/adaptive experiments.

| Charged owner scope | GPU seconds | CPU seconds |
|---|---:|---:|
| Two learner segments |8120.914577302057 |0 |
| Final model delivery |0 |20.156058119027875 |
| U2 vs F1 playing / reader |2060.3332983560394 |36.990443750983104 |
| U2 vs A39 playing / reader |2025.308576279087 |34.86886475607753 |
| U2 total |12206.556451937184 |92.01536662608851 |

Totals remain within19,200 GPU/360 CPU seconds. The closed459 global charged
snapshot is125717.28813925025 GPU and7644.059019470995 CPU seconds. Shared
historical CPU443 qualification stays global-only; process CPU inside GPU
owners is telemetry, not another CPU-owner charge. All earlier successful and
failed costs, scopes, stops and protected-data limits remain. Unused capacity
does not authorize an extension.

Training still used4032 source assets/405013 stored rows, not an asserted
independent-game count. Once per stored row is not per-game balancing; positions
within a game share outcomes and search targets are endogenous. The source
revision and two-segment execution differ from F1, and matching recipe/seeds
does not prove GPU bit identity or causal isolation. Frozen-teacher trajectory
KL demonstrates the anchor was exercised; it is not held-out playing evidence
or isolated teacher GPU-throughput measurement. The method remains beta0/SF0,
unmixed with V10. Separate diagnostic/capability preparation contributes no
U2 effect credit or authority. Actual Root and independent endpoint, delivery,
joint-verdict and custody refs are in the compact evidence.

## Post-method recorded-game census (separate closed CPU460)

After the frozen U2 verdict, a separate metadata-only CPU audit closed with
Root and independent acceptance. The authenticated fixed union contains4,032
recorded collection-qualified game groups, exactly one canonical source asset
per group, and405,013 retained positions. Retained-row mass ranges from5 to297
positions per recorded game group. The recorded grouping key is
`(run_uuid, collection_uuid, physical_game_id)`; there were no copied-asset
aliases, numeric game-ID conflicts or groups with multiple origin records.

These are recorded game groups, not a proven count of statistically independent
games. Statistical independence and cross-collection same-original-game
equivalence remain **UNKNOWN**; no semantic board/history deduplication was
performed. The count neither establishes an effective sample size nor proves
per-game gradient influence or a cause of the U2 result. Once per stored row
still gives unequal retained-row mass to games and does not remove shared
outcome correlation.

This post-method audit used no model, shard or raw-game payload and performed
no row draws or project imports. It charged2.332109948969446 CPU owner-wall
seconds separately; all460 owners then closed. The frozen closed459 U2 totals
remain12206.556451937184 GPU/92.01536662608851 CPU seconds, and its outcomes,
criteria and verdict are unchanged. The audit selects no training recipe,
reweighting, model adoption or new experiment. Its accepted compact readout
and Root/independent/custody references are appended in the evidence.
