# BT4 target temperature, training horizon and search placement

**T1 materialization started; no completed T1 corpus, training or matches claimed.** Compare
pure BT4 policy targets at T=1 and T=0.5 on the original corpus, with unchanged SF
value supervision. [B100V50 remains first](2026-09-08-bootstrap-next-value-and-policy-contrasts.md).
This experiment asks whether the relative benefit of target sharpening changes
with training exposure, and whether sharpening at search time can substitute for
sharpening during training.

## Materialization started — September 9

The reviewed pure-T1 materialization has launched after the E0T05 archive copy
finished, during V50 GPU training. The actual mixer command uses the original SF
corpus and unsharpened float32 BT4 sidecars with **global alpha 1, teacher T=1**;
its target inventory is **18,910,484 rows / 2,309 shards**, with SF values and all
other non-policy arrays preserved. These are requested output counts, not completed
or qualified rows. The four-hour inclusive preparation bound remains unchanged.
[Compact launch identities](artifacts/storage-t1-wdl-update-20260909/status.json)
bind the reviewed plan and actual operator/mixer starts. No teacher inference,
two-epoch training or match launch is claimed by this preparation snapshot.

## Treatments and uninterrupted training

| Arm | Policy supervision | Value supervision |
| --- | --- | --- |
| T1 | Pure global BT4 distribution, teacher temperature 1 | Original SF WDL |
| T0.5 | Pure global BT4 distribution, teacher temperature 0.5 (B100) | Identical original SF WDL |

Construct T1 from the original raw-legal float32 teacher sidecars. Do not invert
already sharpened float16 targets, substitute the 20% BT4 mixture called G20T1,
or use the stored top-tie recipe. Preserve all 16 non-policy arrays, including
SF's cp-derived WDL and game outcomes. Stored rounding and tail support remain
part of the realized recipe and must be recorded.

Train **two fresh, same-runtime trajectories**, each from initialization seed zero
for two uninterrupted epochs. Keep architecture, optimizer, learning rate,
augmentation, resource settings and original row order matched. Use each
trajectory's own epoch-one and epoch-two checkpoints; do not continue historical
B100 weights or use its older-runtime checkpoint as the epoch-one control.
Optimizer, scheduler and augmentation RNG state continue across the boundary.
The existing global 1,000-update warmup and per-window release schedule remain.

Each epoch covers 18,910,484 rows and 97,968 games in 36,935 updates / 420 windows:
36,699 batches of 512 and 236 of 511. Sampler seeds are 0 and 1. Each arm therefore
gets 73,870 updates and 37,820,968 row presentations. Logical schedule identity
must match across recipes; policy-dependent physical content hashes may differ.

## Three protected comparisons

All three cells use **400 simulations and 128 color-swapped opening pairs**
(256 games), fixed N with no SPRT, optional extension or result-dependent omission.
Use the same registered 128 opening pairs, including 16-ply histories, across
cells. Freeze their identities and the actual checkpoint/setting manifests before
play. Other settings retain the qualified fixed-deep recipe profile: move
temperature 0.1, flat root noise, 300-ply limit, no tablebases, rolling pool 128 and
evaluator batch 4096. Search-prior temperature is distinct from move temperature.

| Cell | Candidate | Reference | Question |
| --- | --- | --- | --- |
| Epoch one | T1 epoch one, prior 1 | T0.5 epoch one, prior 1 | Early common-search recipe contrast |
| Epoch two | T1 epoch two, prior 1 | T0.5 epoch two, prior 1 | Later contrast and relative horizon response |
| Final placement | T1 epoch two, prior 0.5 | T0.5 epoch two, prior 1 | Compensated sharpening package |

For each cell report candidate score, paired uncertainty and Elo. The primary
horizon contrast is the change in T1's relative score from epoch one to epoch
two. For opening pair i, calculate its candidate score difference between epochs;
bootstrap these aligned pair differences with 10,000 resamples, seed 20260909,
and report a percentile 95% interval. An interval wholly above zero supports
relative recovery by T1 over this horizon; wholly below zero supports the reverse;
otherwise the direction is unresolved. Do not infer an interaction from one
significant cell and one nonsignificant cell, or subtract Elo intervals.
This does not measure either family's absolute improvement.

For ideal exact fitting, teacher probabilities q become q^(1/T), while search
uses logits divided by prior temperature τ. Thus (T=1, τ=0.5) and (T=0.5, τ=1) have
the same ideal exponent, 1/(Tτ)=2. Actual fitting, stored zeros, rank errors and
shared policy/value representation can break equivalence. The protected placement
cell tests the resulting packages; it does not replace common-search recipe ranking.
Its nominal paired 95% score interval above/below 0.5 supports the corresponding
package direction; an interval crossing 0.5 leaves placement unresolved. Failure
to detect a difference is not proof of equivalence. These related development
intervals are not multiplicity-adjusted promotion tests.

The nondefault prior disables the compact legal BF16 leaf path where available.
The historical direct-evaluator slowdown did not reproduce on the broker path;
current whole-match cost is unmeasured. This is a matched-simulation comparison,
not an equal-wall-clock deployment test. The [completed 0.7 calibration](2026-09-09-b100-prior-calibration.md)
did not compare training targets. No generic calibration grid is queued here;
later equal calibration or fresh-panel confirmation requires its own scientific
question and bounded registration.

## Preparation, limits and allocation

The [existing horizon evidence](2026-09-08-value-collection-and-horizon-readiness.md#two-epoch-runtime-qualification)
covers the 61,444,448-parameter runtime, a compiled CUDA fixture with four updates,
and full SF/B100 CPU plans. It does not measure full-corpus two-epoch throughput,
loader memory or the first 511-row CUDA batch. Genuine T1 admission is implemented
and merged in [PR #604](https://github.com/jjoshua2/DeepFin/pull/604). Materialization
has started; completed output qualification, full physical plans/objective census
and logical-order comparison remain required preparation. The separate explicit
checkpoint/prior package reader is merged in [PR #606](https://github.com/jjoshua2/DeepFin/pull/606);
actual checkpoint admission and match preparation remain pending. Existing
same-checkpoint calibration and equal-search recipe contracts remain unchanged.

Materialization has a **four-hour inclusive ceiling, 32 GiB sampled output cap and
150 GiB free-space reserve**, with two CPU threads and no teacher inference.
Historical B100 preparation took 7,725 seconds and 13.59 GiB allocated; T1 cost and
compression are unmeasured. Its real planner reads the corpus and is not a
metadata-only operation. Reuse unchanged counterpart plans. Schedule CPU work
after V50 rewrite/qualification, potentially during V50 GPU training, accounting
for shared I/O and avoiding concurrent archive copying.

Training is capped at **nine hours per arm**, including startup/compilation;
three match cells each retain the existing 90-minute inclusive stage ceiling.
The maximum training-plus-match allocation is therefore **22.5 hours**, excluding
CPU materialization/planning. Historical four-epoch equivalents suggest about
10.3–11 GPU hours of training, not measured throughput for this runtime. Resource
failure, timeout or incomplete sampling is an operationally incomplete comparison;
preserve partials and do not automatically retry or extend it.

One seed and reused development openings limit generalization. Two passes over
18.91M rows are repeated exposure, not 100M distinct data or asymptotic training.
Separately, independently reviewed native BT4 WDL coverage totals **1,574,952
selected G10 rows**: 526,376 from two common-increment sources plus two disjoint
524,288-row common-large slices. This is value-data readiness, not a new trained
control or full G10 coverage. [Current coverage identities](artifacts/storage-t1-wdl-update-20260909/status.json)
bind the completed review; the [original registration](artifacts/bt4-target-temperature-horizon-20260909/registration.json)
retains its earlier coverage snapshot. Operational paths and raw logs remain
retained separately.

## Future raw-WDL reuse

[PR #609](https://github.com/jjoshua2/DeepFin/pull/609) adds an explicit CPU-only
path to reuse already retained native WDL through the existing verified raw-to-derived
row/history join. It checks canonical feed equality, preserves native values and
publishes explicit adapter provenance; policy-only defaults and direct-inference
admission remain intact. The value consumer requires a pinned adapter manifest.
No collection or frozen-runtime adoption follows from this merge.

The [metadata coverage review](artifacts/storage-t1-wdl-update-20260909/status.json)
found **zero raw-WDL overlap with the current 9,298,514 frozen common-derived
rows**. Their raw labels remain policy-only. A separate observed **3,137,424 newer
raw rows** have native WDL; these are not yet qualified common-derived survivors.
Reuse can avoid another teacher evaluation for future selected covered rows, but
adds **zero rows** to the current G10 value bank. It neither supplies the missing
historical values nor establishes a measured speedup or full 100M capacity.
