# 58M policy/value teacher factorial

## Decision and status

User-authorized September 19: test the Ceres policy addition and equal-third value
recipe together and separately before choosing the package to scale toward 500M.
Preparation is underway. No factorial model has trained or produced playing results.
The existing 58M continuation is a separate trajectory and remains running; it is
not arm A, because these four arms start from the same fresh initialization.

## Frozen scientific design

All arms use the same ordered 35-cohort, 7,108-shard corpus containing 58,090,688
eligible physical rows. This is not a claim of deduplicated unique chess positions.
The corpus is the completed downstream58_retry_v1 union used by the current V50 run.

| Arm | Policy | Main supervised WDL |
| --- | --- | --- |
| A | Existing BT4 T=0.5 | Existing V50: 50% SF + 50% BT4 |
| B | 50% BT4 T=0.5 + 50% Ceres T=0.5 | Existing V50 |
| C | Existing BT4 T=0.5 | Equal thirds SF, BT4, Ceres |
| D | Same policy mixture as B | Same value mixture as C |

Policy mixing normalizes the already sharpened BT4 distribution and averages it
with independently legal-normalized Ceres logits at T=0.5. Do not sharpen BT4 twice
or sharpen the resulting mixture again. The latter is a different experiment.

For economical and consistent storage, the equal-third recipe is defined as
(2/3) * normalize(stored V50 WDL) + (1/3) * Ceres WDL. This inherits the V50
float16 rounding before mixing; it is not claimed bit-identical to a mixture of
three unrounded original teachers. Every admitted base must prove V50 lineage.
Ceres WDL retains the tested calibration: 60% primary softmax at T=0.55 plus 40%
secondary softmax at T=1.5. Equal thirds refers to teachers, not the two Ceres heads.
All arithmetic uses normalized probabilities and the same side-to-move WDL order.

All other targets, masks, inputs, loss weights, optimizer and architecture stay
identical. In particular the prepared supervised WDL must actually reach the main
value loss; parser acceptance or a changed auxiliary target is not evidence of that.

Train one exact game epoch per arm, seed 121, batch 512. Use identical initialization
and canonical row order, verified by actual initial state and schedule identities,
not merely identical CLI seeds or differently named directories. No warm start from
any teacher-specific trained checkpoint. Preserve hourly full-state recovery bundles
and final checkpoints. A resumed partial run must record what sampling was repeated;
it is not automatically a completed matched epoch.

## Matches and interpretation

Common settings: 400 simulations, search-prior temperature 1.0, frozen engine build
and search shape, no per-arm calibration in the first screen. Use 128 swapped
opening pairs (256 games) per contrast and bank every completed game. Freeze opening
bank identities and random seeds before examining outcomes.

The factorial questions are policy effects B-A and D-C, and value effects C-A and
D-B. These are conditional contrasts, not independent training-seed replications.
Use direct matches for these four contrasts; an additional D-A package comparison
is allowed if it changes selection. Fit no position-specific weighting rule from
posthoc winning pockets. Report paired intervals with point estimates; an interval
crossing zero is not a veto on provisional adoption. Report interactions rather
than treating two favorable conditional contrasts as independent confirmations.

Prefer the best-supported package using estimated strength, uncertainty, teacher
cost and behavior across both contrasts. Comparable packages may remain alternatives
for scale transfer. At most two additional 256-game comparisons resolve consequential
close calls; do not extend games until significance appears. Search-temperature
calibration is a separately recorded equal-budget follow-up, not folded into these
training-recipe results. One training seed limits generalization.

## Execution and resource plan

First map Ceres policy plus both value heads to every exact base row. Reuse existing
qualified labels; collect only missing labels with the already qualified GPU batching
path. Do not silently restrict the four arms to the Ceres-covered subset.

Use immutable target overlays to share x, history and legal masks. A uses the base;
B replaces policy_target, C replaces search_wdl, D replaces both. New overlay consumer
support and row/teacher provenance checks require a focused test and independent
review before adopting the successor runtime. Do not mutate the active trainer.

Planning budget: up to 18 hours Ceres collection, 16 hours CPU target preparation,
20 hours per training arm inclusive of startup, and 1 hour per initial match.
These are bounded allocations, not promises of runtime. The previous 50M pass took
6.96 hours; current 58M input loading is slower, so confirm its stable throughput
before advertising an 8-hour epoch. Pause on storage pressure, retain at least 32 GiB
available RAM, and serialize GPU work through the existing lease. Do not launch a
competing collector alongside the active training job.

Queue order: preserve current 58M pass; prioritize missing-label collection and the
four-arm experiment over the two queued identical continuation passes once executable
preparation is reviewed. Existing continuation remains useful fallback until then.
Complete A/B to obtain the first policy contrast while preparing C/D if dependencies
permit. Collection, target readiness and training completion are distinct receipts.

## Prior evidence

The earlier CeresB50 experiment used 18,910,484 rows and 36,935 updates in one epoch.
Its two 256-game comparisons with B100 yielded +12.22 Elo [-22.88,47.57] and -5.43
[-44.54,33.55], approximately +3.4 pooled. These are different arena samples of the
same checkpoints, not two trained seeds. CeresV25 used 50% SF/25% BT4/25% Ceres
values and BT4-only policy; it does not test equal-third values or this joint package.

Bulk artifacts: scratchpad/bt4_joint20/factorial58_20260919/. Current base plan:
scratchpad/bt4_joint20/takeover_20260919/expanded58m_pressure_v1/plan.json.
Current queue revalidation: takeover_20260919/queue_revalidation_20260919.json.
Completion evidence and exact runtime adoption will be added here as they occur.

## September 19: reviewed pipeline adopted

The existing supervisor now holds 23 new stages: 14 missing-Ceres collections,
one CPU overlay preparation, four fresh training arms, and four direct arenas.
The active earlier 58M continuation is preserved. Its two additional queued passes
were cancelled in favor of this factorial. Order after collection/preparation is
A, B, B-A, C, C-A, D, D-C, D-B. The bounded queue deadline is September 26 at
17:46:46 UTC; this is an execution allowance, not an estimate of required time.

Coverage is 35,314,577 existing Ceres policy/dual-value rows plus 22,776,111 missing
rows in 2,787 shards. Fourteen actual audited-source admissions passed. The latter
collection uses batch512/group4, two CPU threads and a 16GiB ONNX arena; previous
production throughput suggests approximately6.3h, while registered caps total13.9h.

Training/preparation runtime is frozen in a detached checkout at
`502cd02e072471c901255f3fdb580d6ea7b826d0`. Collection runtime is
`28f9eab293a445db21c77c8452b8c23f451e4a15`; arena runtime remains
`82298a5d4010e7097f473712e3720de229522427`. Publication changes do not mutate these
execution checkouts. Each fresh training has20h active allowance and2h bounded disk
pauses, hourly full-state recovery and retention of the latest two bundles. Actual
pre-optimizer model tensor fingerprints must match across arms.

Independent review covered protocol, collector integration, target formulas,
immutable overlays, preparation, training and arena runners. Final readback checked
1,195 distinct source/plan pins and all9 preparation/train/arena descriptors through
the actual scheduler validator. The14 collection descriptors were separately reviewed
and validated. The oversized opening-book pin is checked inside pinned arena plans,
not passed to the scheduler's small-file pin list.

Validation includes real tiny cohort construction, changed-teacher and missing-chunk
rejection, policy/value-only inheritance, multi-root sampler parity, real CPU training
parity with physically materialized targets, initialization RNG preservation and hourly
recovery. The integrated21-test run passed, followed by18 affected tests after the
missing-fill-chunk hardening. Scoped repository lint passed. Owned-child runner tests
cover incorrect initialization, incomplete epochs, memory/deadline/STOP interruption;
both-checkpoint arena binding has seven passing cases.

[Adoption receipt](evidence/factorial58-20260919/pipeline_adoption.json),
[final independent readback](evidence/factorial58-20260919/review_frozen.json),
[coverage](evidence/factorial58-20260919/coverage/coverage.json), and the accompanying
frozen plans and operating-script snapshots preserve the operational handoff. Bulk
labels, model files, optimizer states and game banks remain in host storage.

No factorial model or match has completed at this adoption snapshot. CPU target
preparation remains a real stage, not implied by completed label collection. Broader
500M generation/label throughput work remains separate; this experiment does not
require repeating these58M rows indefinitely before scaling.
