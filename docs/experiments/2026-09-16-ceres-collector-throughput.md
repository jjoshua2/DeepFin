# Ceres collection throughput: zero pauses and grouped sessions

Status: the bounded fixed32 pilot completed with bitwise-equivalent outputs. Group4 is selected provisionally for the next two cohorts. Dynamic batch sizing remains a separate investigation.

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

Fresh group4 continuation is prepared for cohorts11–12 only: 2,111,278 rows and 33 invocations per cohort, with final groups of one and two shards. Full consumer qualification retains all source, payload and provider checks while validating the exact grouped shard roster. The two cohort caps sum 10,200 seconds; the outer limit is three hours. Existing qualified cohorts and interrupted banks remain intact. Queue adoption is recorded separately by the parent operator.

The ONNX input has a dynamic batch dimension. Fixed32 is the qualified collector contract, not a static model shape. A separate bounded batch32/64/128 probe will test whether larger neural batches improve throughput under the same 8 GiB arena, with fresh per-shape provider proofs and prespecified numerical tolerances. Precomputed input tests measure an inference ceiling; successful candidates still need end-to-end streaming validation. Input prefetch is a subsequent option if CPU conversion and gathering remain the bottleneck.

Validation: 28 focused scheduler tests passed, including real two-chunk runs at zero and thirty-second pauses, resource checking without a sleep, malformed plans and failure/timeout behavior. Ruff and diff checks passed. Independent reviews passed the scheduler change and the frozen pilot plan.
