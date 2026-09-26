# BT4 labeling throughput toward100M and1B positions

Status: read-only reconciliation of saved producer code, commands, logs and receipts.
No inference, corpus scan, benchmark or running-job change was performed. This
updates planning costs; it is not a new training or playing-strength result.

## Verified BT4 cost

The September13 bounded allocation completed **1,527,153 fresh policy+native-WDL
rows in184 raw shards**. Every group's logged `label_shard` source/shard roster and
cumulative row count exactly match its new receipts. Thus this particular allocation
establishes fresh inference coverage, beyond the weaker fact that new receipts exist
(the collector can also recover previously published sidecars with missing receipts).

| Measured scope | Seconds | Rows/second |
| --- | ---: | ---: |
| Entire12-group allocation, including pauses and surrounding work |2,638.682|578.76|
| Sum of owned collector invocations |2,458.070|621.28|
| Approximate sum of labeling loops, reconstructed from rounded logs |2,204.995|692.59|

Individual loop rates were678.4–711.5 rows/s. The loop includes raw decoding,
history reconstruction, encoding, key verification, policy+WDL inference, legal
projection, hashing and publication. It excludes model/session startup, leased
preflight and final inventory refresh. These logs do not separate CUDA time from
CPU work or prove neural-kernel utilization.

The actual September13 command used **batch128, two ORT threads,8GiB CUDA arena,
at most16 shards per session**, CPU affinity2,3 and15-second inter-group yields.
Runtime `/tmp/deepfin-bt4-label-runtime-pr580-overlay`, commit
`8fd3940e60530aebdd7bd7f398cd04eb2679af1d`, used the saved qualified Python3.10,
NumPy1.26.2 / ORT1.23.2 environment and BT4 model
`1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0`.

An earlier actual driver used **batch1024,16 ORT threads,24GiB arena** and reported
446.9–490.9 loop rows/s (approximately382–396 rows/s between publication cycles).
Those are different dates, inputs, thread/resource settings and timing scopes.
The later higher rate does **not** establish that batch128 beats1024. Neither
configuration should be inferred from CLI defaults or described as an active job.

## Meaning of the labels and reuse

B100 means100% BT4 policy supervision at its specified target temperature. It does
not mean100 search nodes. The raw producer performs one batched network evaluation
per position and requests `/output/policy` plus `/output/wdl` together. Native WDL
retention is a second output of that call, not a second search or model evaluation.

Fresh raw labeling is different from adapting saved labels into shuffled derived
rows, writing sharpened B100 policies, or writing SF/BT4 V50 targets. Those CPU stages
reuse banked teacher outputs and should not be priced as new GPU inference.

The pinned September14 joint-receipt snapshot contains59,264,839 policy-covered
raw rows, of which23,827,971 also have native WDL and35,436,868 are policy-only.
This is dated physical-source coverage, not unique-FEN counts, current inventory or
training admission. New joint collection does not automatically backfill old
policy-only rows. Already-available joint rows should be reused before relabeling.

## Scale estimates and limits

| Observed path |100M fresh rows |1B fresh rows |
| --- | ---: | ---: |
| BT4 allocation at578.76 rows/s |48.0h|20.0days|
| Ceres fixed32/group4 pilot at376.58 rows/s |73.8h|30.7days|

These are linear planning extrapolations, not runtime guarantees. The Ceres pilot
covered65,536 derived rows in174.028s and used a different model, corpus and timing
scope. It is not a direct teacher-speed comparison. It showed2.54× versus its own
saved baseline, with approximately25% mean sampled GPU utilization; utilization
alone is not the objective. The two labeling costs add when sharing one GPU and
both teachers are required. CPU derivation/adaptation, auditing, target writing,
training, storage transfer and future availability remain additional costs.

The BT4 bank stores a dense float32 legal-policy array of width1858 before Zstd
compression:743.2GB for100M rows or7.432TB for1B, excluding other arrays. These are
uncompressed logical sizes, not predicted disk use; illegal zeros compress well.
A sparse legal representation may eventually reduce decoding and copy costs, but
requires an explicit compatible storage/consumer design and measured compressed
size. It is not a reason to delete existing inputs.

## Highest-value next measurements

The producer is serial across raw decode/history preparation, synchronous ORT,
legal projection and publication. `--threads` sets ORT intra-op threads; it does
not create Python preprocessing workers. The previously measured duplicate legal
mapping work was already removed; do not implement that optimization again.

Two reusable opportunities deserve measurement:

1. **Bounded inference-batch tuning under the current8GiB allowance.** Compare128,
   then256, then512 on the same small closed-shard input, holding two threads,
   model, outputs, provider options and row order fixed. Stop larger shapes after
   OOM or numerical/provider failure. Compare complete legal policies and native
   WDL, not only argmax. Separate precomputed-feed inference throughput from one
   actual end-to-end pass so CPU costs are visible. No return to24GiB is implied.
2. **Avoid repeated whole-bank metadata work while keeping its guarantees.** The
   parent hashes the model and validates existing sidecars; the leased child hashes
   it again, creates its session, validates the bank again, labels up to16 shards,
   then refreshes the full bank before release. With fixed-size groups, work over a
   growing bank can become superlinear. First time these stages. A validated
   incremental receipt/index scheme or moving final CPU status work after lease
   release may help; neither warrants dropping identity/history checks.

If stage timers show preparation dominating, test a single bounded prepared-batch
prefetch worker, with backpressure and unchanged physical row order. Do not add
many workers or C++ rewrites before identifying the actual expensive stage.
Ceres already has a source-block read cache and stage timers on main; its frozen
collector runtime lacks that change. Adopt and qualify the existing implementation
separately rather than duplicating it or assuming its CPU-component gain transfers
to BT4 raw JSON processing.

Proposed next benchmark envelope: one frozen approximately8K-row raw shard, two CPU
workers/threads, at most15min exclusive GPU including cleanup,48GiB available RAM
at startup/32GiB running,8GiB ORT arena and150GiB disk reserve. Precompute and verify
inputs on CPU before the GPU pass; capture first-call provider proof for each
shape, session startup, all per-stage times and actual memory samples. Save raw and
normalized differences even on failure. Freeze numerical thresholds and a concrete
command before execution. A single ordered screen gives descriptive rates; it does
not justify a universal optimum or silently change production sidecar contracts.

## Evidence

[Compact receipt reconciliation](evidence/2026-09-17-bt4-label-throughput.json)
contains every12-group command/log/completion hash, exact timing arithmetic,
teacher identity and source-code pins. Host records are
`scratchpad/bt4_joint20/raw_scale_fill_v1/`,
`raw_wdl_throughput_readiness_v1/`, and
`takeover_20260916/ceres_throughput/pilot_v1/`.

The [bounded raw-label record](2026-09-13-bounded-raw-scale-labels.md) explains the
allocation, resource guards and literal `MAX_NEW_SHARDS` stopping-reason caveat.
The [legal-move reuse record](2026-09-07-bt4-label-legal-reuse.md) separates its
existing CPU-component parity result from full-pipeline throughput.
