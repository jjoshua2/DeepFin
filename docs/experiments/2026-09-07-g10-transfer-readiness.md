# G10 transfer readiness and target alignment

September 7, 2026. This preparation work supports the larger-data stage of the
[adaptive bootstrap research](2026-09-07-bt4-hybrid-endpoints.md). It does not choose
another recipe before H20's completed comparison or establish playing strength.

## Available data

A receipt/metadata inventory observed from 22:07 to 22:10 UTC found the following
closed physical rows. These are source-qualified row occurrences, not a count of
globally deduplicated positions or an atomic snapshot of the still-running generators.

| Source | Closed raw rows | Receipt-backed BT4 rows | BT4 backlog |
| --- | ---: | ---: | ---: |
| run06 G10 | 24,476,040 | 20,186,464 | 4,289,576 |
| run07 G10 companion4 | 6,155,799 | 4,738,228 | 1,417,571 |
| Total | 30,631,839 | 24,924,692 | 5,707,147 |

The 3,003 completed BT4 shard receipts matched source metadata, sidecar attributes
and array layouts. Active labeling advances these intended G10 sources. At that
inventory snapshot, no G10 matched-recipe preparation pipeline had been qualified.
The completed pilot below now qualifies 16,429 fixture rows; the larger training
corpus and its actual training schedule remain unqualified. Complete SF observation
coverage across the inventory is still unknown. Merely reaching 35M or 100M labeled
rows does not resolve those semantic requirements.

## Bounded alignment diagnostic

The preregistered sample used the first 64 physical rows of one closed shard from
each source: 128 rows across three source-qualified games. The same saved raw bank
was retained when correcting the estimator. This is a construction diagnostic,
not a random sample, prevalence estimate, held-out test or strength comparison.

All 128 sampled rows had complete legal-move d9 observations in phase zero. The
legacy deriver can instead select later-phase d9 observations. Switching between
those choices changed centipawn scores on 126 rows, stored SF maximum sets on
40 rows, and consistently constructed near-close candidate sets on 53 rows.
For 36 rows, the phase-zero rank-one move was outside the legacy stored maximum.
Thus mixing a phase-zero rank sidecar with the legacy target can violate the
intended shared SF observation semantics.

Value selection is a separate intervention. Counterfactual phase-zero searched WDL
differed on 114 rows, with maximum component difference 0.139746. Both actual
writer diagnostics retained the current latest-phase searched value unchanged;
these observations do not show which value choice plays better.

The input join also needs more than a board or game/ply key. All 128 full-history
input keys changed when the current writer quantized float32 inputs to float16,
even though every quantization-aware writer comparison passed. There were 35 bare
game/ply key collisions across the two sources. Recomputing an original BT4 input
key from stored float16 inputs therefore cannot establish the intended join.

## Preparation decision

Expose policy/rank observation selection independently from value observation
selection, retaining historical defaults. For the G10 transfer pilot, align policy
and near-close ranks on explicitly selected complete phase-zero d9 observations.
Keep the existing searched-value convention identical across recipe arms unless a
separate value intervention is deliberately registered.

Carry source-qualified raw row references and original full-history keys through
the exact filtering, grouping and shuffle used by derivation. Verify the stored
quantized input separately, then publish a source-bound derived BT4 sidecar.
Use compact indexed provenance at 100M scale rather than repeating full paths and
configuration metadata for every row. The implementation and small real pipeline
pilot must qualify this contract before any full G10 derivation or recipe training.
H20 continues to use its already frozen, single-phase development corpus.

## Registered bounded pipeline pilot

[Machine-readable registration](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_pipeline_pilot_v1/preregistration.json)
and the two original closed receipt snapshots are included in the evidence manifest.
Registered before operational derivation or adaptation. This tests whether the
new tools can produce aligned inputs for a later recipe comparison. It does not
train a network or choose the next target family.

Use the complete first closed `w00-00000.jsonl.zst` shard from each source, selected
by filename before inspecting further rows. The existing closed receipts identify
8,236 run06 rows and 8,243 run07 rows (16,479 physical rows total). Keep the sources
in separate derived directories. Freeze a copy of each selected completed BT4
receipt and record the source manifest, raw-shard, teacher, implementation and
command identities before execution. Source hashes are:

- run06: `3c93d2d127bc6d11db9eb31425e0c795c96b417106d737632ef4b689569ae17d`
- run07: `a499d2921043065b388778435efa2e2e63d3e206ab1310264b75dafa39d9dea0`

Derive once per source with `uniform-d9`, policy observation `phase0`, value
observation `latest-phase`, value scheme `search`, temperature `0.0005`, floor `0`,
seed `0`, 8,192 rows per output shard, one worker and `--row-provenance`.
`--limit` is the selected source's complete physical row count, including any
rows dropped by derivation. Before launch, verify that this is exactly the first
shard in the tool's resolved input inventory. Do not substitute a different shard
if a qualification fails.

On those exact surviving rows, adapt the existing raw BT4 labels and construct
phase-zero d9 ranks with rank cap three. Use the ordinary mixer admission path to
materialize C20T05 and its H20T05 hybrid, with identical non-policy arrays. These
small recipe artifacts are pipeline fixtures, not additional training arms.
Reuse compatible frozen descriptive audit receipts; if admission requires a new
receipt for the registered context, reanalyze the same saved observations and
identify it as reanalysis, without rerunning a teacher or interpreting its score
as new evidence.

Success requires complete source-qualified row joins, original and quantized
history-key checks, legal rank/target coverage, ordinary sidecar admission,
unchanged non-policy arrays, and identical source-qualified game/ply schedules
between the two recipes. Report all actual exclusions and counts. Every emitted
row must meet these invariants; a partial output or metadata-only match fails.
This qualifies only the selected inputs and paths, not all G10 rows or 100M-scale
throughput. The existing loader may combine the separate source parents; do not
flatten colliding game IDs into one physical source directory.

Use a frozen reviewed checkout with the qualified CPU development interpreter,
GPU hidden, at most two numerical threads and two CPU cores, nice 19 and idle I/O
priority. Stages run sequentially with a **30-minute total wall-clock cap**,
including termination grace, **8 GiB of new output/cache allowance**, and the
existing **150 GiB free-space reserve**. The adapter's identity-cache allowance
is 16 MiB for these two raw shards. Preserve failures and logs, stop dependent
stages on failure, and do not automatically retry or expand the input. Active H20
preparation and GPU labeling retain priority. After this pilot, decide whether a
larger preparation batch is justified from its actual failures, throughput and
coverage; a full transfer training is a separate result-dependent selection.

## Pilot readout: completed

The registered pilot completed at 23:46:10 UTC on September 7, with exit zero in
**204.08 seconds**, using merged implementation `ac4a246ba025ca54329d6e4dc380473f1ea4b728`.
It read the exact two registered raw shards. Both recipe fixtures passed their
ordinary admission and all-row comparison checks.

| Source | Physical rows read | Rows emitted | Excluded |
| --- | ---: | ---: | ---: |
| run06 G10 | 8,236 | 8,236 | 0 |
| run07 G10 companion4 | 8,243 | 8,193 | 50 |
| Total | 16,479 | 16,429 | 50 |

The 50 exclusions were rows without game results. Neither source dropped rows for
observation-envelope failures. Original raw hashes remained unchanged throughout.
For every emitted row, the pipeline qualified the original/quantized history join,
legal BT4 policy, phase-zero d9 rank alignment, and ordinary C20T05/H20T05 mixing.
The final comparisons checked exact shard inventories, array sets, shapes, dtypes,
all non-policy values and source-qualified row-reference order.

An initial status message inferred that all physical rows survived from the
successful terminal status. Reading the per-source receipts showed the 50 exclusions;
the count was corrected immediately. No observation or pipeline was rerun.

The following wall times include subprocess startup and the sampled guard overhead,
with two low-priority CPU cores. They are measurements of these small shards, not
an estimate of full-scale throughput.

| Stage | run06 seconds | run07 seconds |
| --- | ---: | ---: |
| Derivation | 24.10 | 25.28 |
| Raw BT4 adaptation | 23.44 | 23.33 |
| d9 ranks | 14.30 | 13.98 |
| C fixture | 7.26 | 7.61 |
| H fixture | 9.34 | 9.87 |
| Final array/reference comparison | 7.40 | 7.71 |

The two descriptive audit reanalyses took about 11 seconds each on the already
saved observations; neither invoked a teacher. No GPU work, network training or
arena was performed. The pilot proves input-schedule identity after mapping output
parents to original sources; it does not execute the future training planner.

This establishes a working preparation path on **16,429 emitted fixture rows**.
It does not qualify the complete G10 corpus, prove which observation selector or
recipe plays better, or establish 100M-scale costs. Before expanding preparation,
use these stage costs and the existing parallel paths to select a bounded common-data
batch. The next target-family training remains conditional on completed H20 results.

[Actual readout](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_pipeline_pilot_v1/readout.json)
and the [pilot evidence manifest](evidence/bt4-bootstrap/g10-pipeline-pilot-manifest.json)
retain the fixed runner, launch/runtime pins, original stage logs and receipts,
source-qualified row references, and completed derivation/sidecar/mixer summaries.
Bulk Zarr payloads stay host-local with their recorded identities.

## Selected next common-data preparation

The [machine-readable registration](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_common_batch_v1/preregistration.json)
and both closed-receipt snapshots are included in the evidence manifest.

The next preparation batch is **531,412 raw rows**: the first 32 closed,
receipt-backed shards in each source's sorted inventory (`w00-00000` through
`w00-00031`). The recorded counts are 265,528 for run06 and 265,884 for run07.
Keep separate output parents and freeze all 64 original closed receipts and raw
payload identities before execution. This balanced preparation slice does not
choose the eventual source mixture for training.

Run the existing two-worker derivation once with the pilot's phase-zero policy,
latest-phase value, uniform-d9, temperature 0.0005, floor zero, seed zero,
8,192-row output shards and compact provenance. Then adapt existing BT4 labels
and construct top-three phase-zero d9 ranks on those exact surviving rows.
There is no C/H materialization, teacher inference, training or arena in this batch.
The common inputs remain usable by multiple later target families.

Use a **two-hour total wall cap including termination grace**, an aggregate two-core
CPU affinity, two numerical threads, nice 19/idle I/O, hidden GPU, **8 GiB new
output/cache allowance** and **150 GiB free-space reserve**. Each source's private
adapter/rank cache has a 64 MiB cap. Record the actual frozen implementation,
interpreter and commands before launch. Measure each stage's wall/user/system
time, peak RSS and I/O counters during this single useful preparation pass;
there are no timing reruns.

Report actual emitted rows, every exclusion, completed shared sidecar identities
and any failure. Stop on an identity, legal-coverage, publication or resource
failure; retain evidence without replacing the source set or automatically retrying.
Successful completion qualifies only this frozen common slice. Use the observed
resource costs to decide whether to expand preparation or address a measured
bottleneck. The pilot suggests that the adapter's serial history reconstruction
may matter, but it has not separated CPU from I/O cost or shown that a new
implementation is necessary. H20 and useful GPU labeling retain priority.

The [read-only assessment](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_readiness_v1/post_pilot_preparation_assessment_v1.json)
records the exact prefix counts and distinguishes the labeler's ONNX thread setting
from Python encoding parallelism. This batch is selected but has not launched at
this publication snapshot.

## Evidence

The [publication manifest](evidence/bt4-bootstrap/g10-readiness-manifest.json) binds
the original inventory, 3,003-shard membership record, 128 raw rows, exact diagnostic
arrays and row-level joins/readouts, estimator corrections and producing script.
[Readiness receipt](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_readiness_v1/readiness.json)
and [diagnostic/recommendation](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_readiness_v1/sample_diagnostic_v1/diagnostic_and_recommendation.json)
retain complete hashes, observed settings and limitations. Host commands and PIDs
are archival provenance. Bulk source corpora and temporary writer directories remain
host-local; this evidence publication is not a portable launch bundle.

[Tool validation and independent reviews](evidence/bt4-bootstrap/g10-tools-validation-manifest.json)
bind the final derivation, adapter and rank changes. The affected derivation suite
passed 317 cases, adapter tests passed eight, rank tests passed 13, and the combined
repository lint passed with no findings. Independent checks covered actual mixer
admission, shuffled source joins and rejection of changes during publication.
Those checks preceded the real-data pilot; its completed readout is recorded above.
