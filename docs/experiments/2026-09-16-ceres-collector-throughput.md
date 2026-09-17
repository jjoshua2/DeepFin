# Ceres collection throughput: zero pauses and grouped sessions

Status: the bounded fixed32 pilot completed with bitwise-equivalent outputs. Group4 was adopted for the next two cohorts; the first production group completed successfully. Batch512 has since been characterized and selected for a fresh streaming check; its numerical differences are recorded below.

## Why change orchestration

Completed evening block09 contained 113 shards and took 6753.61 seconds. Its prescribed 112 thirty-second pauses consumed 3360 seconds, almost half the wall time. Chunk receipts total 3389.07 seconds; synchronous `session.run` timings total 2035.41 seconds. Removing only the fixed sleeps projects 1.99× throughput, before accounting for session reuse. That subtraction is a planning estimate, not a benchmark.

The collector already opens one ONNX session before iterating selected shards. Grouping several shards therefore reuses existing session support. The scheduler change permits an explicit zero pause; existing 30-second plans keep their behavior. Process cleanup, resource checks and source/output proofs remain active. A sleep is not a memory bound.

## Registered comparison

On the same eight qualified block09 shards, compare two four-shard sessions with one eight-shard session. Both retain fixed32 neural batches, the same teacher and runtime, both raw value heads, gathered legal policy and zero pause. Reference outputs are the already qualified single-shard banks. Recomputed rows serve numerical and performance qualification only.

The complete pilot has a 900-second bound, collection allocations of 380 and 300 seconds, and a bounded CPU comparator. Resource limits remain an 8 GiB GPU arena, 48/32 GiB startup/running available RAM, 150 GiB free-disk floor and two numeric CPU threads. Fresh outputs and owned-process cleanup preserve prior work.

Require bitwise equality of every payload array and matching source hashes. Record wall rows/sec, synchronous inference time, first-call CUDA provider proof and sampled NVML utilization. The four-shard arm runs first; this is a bounded engineering comparison, not a randomized estimate of the causal effect of grouping.

Descriptor: `/home/josh/projects/chess/scratchpad/bt4_joint20/takeover_20260916/ceres_throughput/pilot_v1/registered_command.json`, SHA256 `c8d4b979d56854f4b5ede827ff525efd678a58201548edfed9de88242ead81d1`.

## Completed results

Each arm processed 65,536 rows. The saved single-shard reference spanned 442.562849 seconds including seven pauses, with 232.301631 seconds of active chunk time.

| Configuration | Wall seconds | Rows/sec | Speedup over saved wall baseline | Mean sampled GPU utilization |
|---|---:|---:|---:|---:|
| Four shards per session | 174.028 | 376.584 | 2.54× | 25.1% |
| Eight shards per session | 196.362 | 333.752 | 2.25× | 20.6% |

All 144 payload-array comparisons across the 16 output shards were bitwise equal to the saved reference. Every session recorded 403 CUDA neural kernel events and four CPU integer shape operations. Those events establish GPU execution, not high utilization.

Four-shard grouping was faster in this ordered pilot and is the practical choice. Eight-shard grouping is not established as generally slower: different device clocks or other time-varying conditions could affect this single ordered comparison. The approximately 80% utilization objective remains unmet. Synchronous `session.run` fractions of 70.5% and 78.3% are also not GPU utilization percentages.

Compact evidence: [pilot readout](evidence/2026-09-16-ceres-grouped-pilot.json) and [saved timing analysis](evidence/2026-09-16-ceres-collector-throughput.json). Actual comparison receipt SHA256: `9b55094a803c26b123bb1ef7361b4caff18013469b8f610bbf4575ea17ea700e`.

## Adoption and next probe

Fresh group4 continuation was adopted for cohorts11–12 only: 2,111,278 rows and 33 invocations per cohort, with final groups of one and two shards. Full consumer qualification retains all source, payload and provider checks while validating the exact grouped shard roster. The two cohort caps sum 10,200 seconds; the outer limit is three hours. Existing qualified cohorts and interrupted banks remain intact. The parent adopted the pinned descriptor `9a3df1eb3b03e0b8ffafc4801e7adb4745fa2b3cd3466db6f1d55211a93cd3bc`. Its first production group completed 32,768 rows in 97.272454 seconds with exit code 0 and all1,024 fixed32 calls accounted for. This establishes a working production invocation, not completion or qualification of either whole cohort. See the [adoption receipt](evidence/2026-09-16-ceres-group4-adoption.json).

The ONNX input has a dynamic batch dimension. Fixed32 is the qualified collector contract, not a static model shape. A separate bounded batch32/64/128 probe will test whether larger neural batches improve throughput under the same 8 GiB arena, with fresh per-shape provider proofs and prespecified numerical tolerances. Precomputed input tests measure an inference ceiling; successful candidates still need end-to-end streaming validation. Input prefetch is a subsequent option if CPU conversion and gathering remain the bottleneck.

Validation: 28 focused scheduler tests passed, including real two-chunk runs at zero and thirty-second pauses, resource checking without a sleep, malformed plans and failure/timeout behavior. Ruff and diff checks passed. Independent reviews passed the scheduler change and the frozen pilot plan.

## Batch512 characterization and useful-row continuation

A completed 8,192-row precomputed-feed probe measured batch512 at **1,968.31 rows/sec** (4.16195 seconds for the inference pass, 10.86434 seconds including session setup/profiling). This is an inference measurement, not streaming collection throughput. Provider profiling again found 403 CUDA neural events and only four CPU integer shape operations. The selected candidate uses a 16 GiB arena, with a separate 24 GiB total-device ceiling in fresh collection plans.

Batch512 is **not bitwise equivalent** to fixed32 and did not meet the initially recorded numerical-equivalence thresholds. Legal-policy top1 agreement was 99.707%; at teacher temperature 0.5, maximum/p99 total variation were 0.03674/0.01042. Maximum raw-logit differences were 0.19775 (policy), 0.12231 (value), and 0.21411 (value2). Primary-value T1 maximum total variation was 0.02441. These are measured differences, not evidence that either batch shape is a ground-truth accuracy reference. The operational choice accepts those reported differences for a same-teacher batch change; it does not claim numerical equivalence or playing-strength validation. Full metrics and receipt identities are in the [batch512 evidence](evidence/2026-09-16-ceres-batch512.json).

The earlier group4 run was stopped before source shard 36 after nine successful groups: 294,912 rows are retained. Together with the completed prefixes of two earlier interrupted cohorts, 507,904 rows across 62 shards remain reusable. Fresh plans prepare only the remaining 3,056,289 rows across four cohorts. They preserve each saved shard's original batch/backend provenance and qualify the complete union. The first useful batch512 group is the missing old01 source shards 14–17 (32,768 rows); its streaming result is pending. Source-block caching already on main accompanies the new batch profile.

## First useful batch512 streaming result

The fresh missing-row group completed **32,768 rows in 30.63385 seconds: 1,069.67 rows/sec**, with 64 physical inference calls and exit code 0. Peak sampled device memory was 7,445 MiB. Recorded shard stage totals were 0.508 seconds source reading, 4.620 CPU preparation/postprocessing, 17.065 synchronous inference, and 1.481 output/verification; setup and guards account for additional wall time. This establishes working end-to-end collection, about 2.84× the earlier 376.58 rows/sec group4 fixed32 pilot, with an ordered cross-run comparison limitation. It is distinct from the 1,968.31 rows/sec precomputed-feed measurement.

The output is useful new coverage: source shards 14–17 of partial01, preserving all prior completed rows. The sequential collector can continue from successful chunk receipts; this first receipt does not claim completion of the remaining full cohorts. See the [actual streaming receipt](evidence/2026-09-16-ceres-batch512-streaming.json).

## Completed first cohort and qualification recovery

All 13 batch512 groups of partial01 completed, adding 412,916 rows. The first CPU qualification then failed because the historical module-pin roster did not include newly imported `scripts/adaptive_sf_value.py`. This was a qualification bookkeeping failure, not failed inference or lost outputs.

A fresh CPU-only qualifier bound all 101 actually observed Python module hashes to the frozen runtime and passed the full 527,604-row, 65-shard union: 51 new shards plus 14 retained old shards. No inference was repeated. The failed lane and original qualifier are preserved. The remaining 2,643,373 rows in cohorts 08/11/12 have fresh prepared plans with the expanded module pins; they follow the independent 35M training continuation. See the [qualification recovery evidence](evidence/2026-09-17-ceres-partial01-recovery.json).
