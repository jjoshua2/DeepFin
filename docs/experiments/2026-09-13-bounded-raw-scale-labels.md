# Bounded raw policy and native-WDL scale labels

Status: the bounded collector completed with exit 0 on **2026-09-13 at 18:34:10 UTC**, recording **184 raw shards / 1,527,153 rows** in **2638.682 seconds** (43 minutes 58.682 seconds). The 12-group allocation ended; these labels are not yet derived or training-admitted. The original launch snapshot is retained below.

## Allocation and purpose

Prepare useful labels toward the larger corpus while the [registered V100 target rewrite](2026-09-13-native-bt4-value-endpoint.md) runs. This is an operational use of the existing qualified BT4 collector, not another teacher recipe or strength experiment. A metadata snapshot found **1,477,290 closed raw rows  / 178 shards** without sidecar receipts: 979,788 / 118 in run06 and 497,502 / 60 in run07. Their exact raw-name intersections with the accepted 20 G10 cohort rosters are empty within each source namespace. These are raw counts, not retained derived rows.

The allocation is **7200 seconds total, at most 192 newly recorded shard receipts, at most 16 per group**, one group at a time. Newly closed shards may enter the selection after the snapshot; actual group receipts establish coverage. There is no automatic extension. The first actual group has 1800 seconds including cleanup:1740 seconds for the collector and the existing cleanup margin. Later group entry uses its measured elapsed time per newly recorded shard, multiplied by the next count and 1.5, plus 60 seconds; teacher/batch settings remain fixed. The outer timeout is 7160 seconds TERM plus 40 seconds KILL, covering the inherited 30+5-second cleanup.

## Unchanged collector and resource controls

The runtime remains `/tmp/deepfin-bt4-label-runtime-pr580-overlay` at `8fd3940e60530aebdd7bd7f398cd04eb2679af1d`. One inference collects the original legal BT4 policy and native `/output/wdl` probabilities, batch 128, two threads and an 8 GiB ORT GPU allowance. CPU affinity is 2,3; available host RAM must be 48 GiB at startup and 32 GiB during collection, with 150 GiB free SSD. Sampled headroom and the ORT allowance are not hard total-RSS/device-memory limits.

The saved old bank contains 35,436,868 policy-only rows; this operation does **not** backfill them. Existing receipts must remain identical. Newly recorded coverage can include the unchanged collector's recovery of a published sidecar whose append receipt was missing, so receipt counts do not necessarily mean fresh inference. No `--verify-all` payload sweep is requested.

The fresh operator holds the old driver's coordination lock and checks actual same-user processes before collection; the recorded ownership check found no matching old driver or producer. The collector still enforces its writer and GPU leases. Old pause markers and PID files remain intact. A fresh parent-owned `STOP.request` yields at the next group boundary, with 15 seconds between groups to give waiting training priority. Hard faults clean the entire owned collector process group. A `.writing` remainder is preserved and refused on a later restart, requiring separate review.

## Actual launch evidence

The outer launch recorded 88,090,669,056 bytes available RAM and 214,444,892,160 bytes free disk. The operator started at 1789321811.9766617; the first collector group started at 1789321812.2052217, PID/PGID 411322, with the reviewed exact argument vector and maximum 16 shards. These are launch samples, not peak-memory or completed-throughput measurements. No active log, game bank, model or corpus payload was read for this publication.

Parent source/command review and independent review passed. Independent review `74ce1320987521e09924488519e646eaf6ebccdf54d52bb9a3a32128e9df67a1` includes the corrected outer cleanup grace. The plan is `bdca27264c2ca8d86f56b4b28a18c4a20504b30a16356a7ae024debdf72ef287`; command is `2e9964cc4a00151758a8a055cdf40f9e959f35156df14e8dc67a8ed1c83c3eca`.

[Compact evidence](evidence/2026-09-13-bounded-raw-scale-label-launch.json) preserves actual launch, ownership, first-group argv and original snapshot references. Later derivation and raw-to-derived identity qualification remain necessary before these labels can join a training corpus. Existing accepted G10 native coverage and frozen V100 implementation are unchanged.

## Actual bounded completion

Execution 55179 and parent observer 609 are closed with exit 0. All 12 collector groups completed successfully: the first eleven recorded 16 shards each and the last recorded 8, despite requesting up to 16. Every group remained below its 1740-second cap; the first took 232.444 seconds. The whole outer invocation took 2638.682 seconds of the allocated 7200.

| Source namespace | Newly recorded shards | Raw rows |
| --- | ---: | ---: |
| run06_g10 | 122 | 1,013,011 |
| run07_g10_companion4 | 62 | 514,142 |
| Total | 184 | 1,527,153 |

The saved completion reason is literally `MAX_NEW_SHARDS`, but **the 192-shard cap was not reached**. The unchanged operator initializes that reason before its fixed 12-iteration loop and retains it when the loop ends naturally. The observed stopping condition was exhaustion of those 12 group slots, with a nonempty final group of 8. The original receipt remains immutable; this correction does not establish that current generation is caught up or show another collection run is needed.

The independent compact review reconciled every group receipt with the final list: all 184 `(source_id, source_shard)` identities are unique, and group/source/whole row sums match. Each receipt binds the qualified BT4 model and remapping revision, source and input identity digests, legal policy output, and native float32 WDL probabilities in win/draw/loss order from the side-to-move perspective. Recorded publication times fall within this allocation. Existing receipt stability checks remained enabled. No sidecar arrays, raw payloads, models or inference were reread for the review.

Successful completion passed the frozen operator's resource guards; the startup samples above remain the available resource evidence, not measured peak RAM or GPU usage. This adds raw label coverage only: `training_admitted` and `backfill` are both false. The earlier snapshot's disjointness observation applies to that snapshot, not automatically to all later source closures. No further collection is queued by this record, and source generation and unrelated work remain unchanged.

[Completion evidence](evidence/2026-09-13-bounded-raw-scale-label-completed.json) retains exact terminal, final/group receipt and independent-review pins, the saved reason string and its corrected interpretation. Parent and independent compact checks agree on the counts; neither repeated payload validation.

## Actual baseline eligibility audit launched

At **18:45:38.569 UTC**, the parent launched the receipt-selected CPU audit of
the exact **184 shards / 1,527,153 rows** completed above. The tool merged in
[PR #728](https://github.com/jjoshua2/DeepFin/pull/728) checks source hashes, row
identity and the existing phase-zero uniform-d9 policy and consumed composite-d9
value eligibility rules. It does not repair rows, select adaptive labels or
produce training targets. See the [tool contract](../toolchains.md#receipt-selected-raw-baseline-eligibility).

The frozen runtime is `1cacea59dff150673e41cfe0dd9b6cfb66edf6da`. The command
uses CPUs 4,5, two numeric threads and a hidden GPU, with an **1800-second whole
budget** (1740-second internal deadline; 1770-second TERM plus 30-second KILL).
It requires 48 GiB available RAM at startup, 32 GiB during sampled checks,
150 GiB free disk and at most 512 MiB of rejection diagnostics; no virtual-memory
cap is imposed. The start sample recorded 88,646,012,928 bytes available RAM.

Parent review and independent source/execution review passed before launch.
[Compact launch evidence](evidence/bounded-ceres-and-raw-audit-launched-20260913.json)
binds the actual start, exact selected manifest, command and reviews. This is a
launch snapshot: preliminary counters are intentionally not a final eligibility
result. Failure or a completed progress prefix would not admit a training corpus.

## Completed baseline eligibility audit

The CPU audit completed with **exit 0 in 1300.442 seconds**, within its
1800-second budget. All 184 selected raw shards and 1,527,153 physical rows were
accounted for under the frozen collection and source identities.

| Classification | Rows |
| --- | ---: |
| Eligible under the existing baseline rules | 1,517,925 |
| No result, following the existing drop rule | 9,198 |
| Additional required baseline exclusions | 30 |
| Physical total | 1,527,153 |

All 30 result-bearing failures have invalid **phase-zero legal-move support**;
the composite-value validator also rejects those same rows because it requires
valid phase-zero support. Policy and value counts therefore overlap rather than
identify 60 failures. These compact diagnostics do not determine whether the
underlying move-list defect was duplication, an illegal move or missing/replaced
support. They do not establish a later-phase-only failure.

The 30 rows span 13 shards (22 rows from run06 and 8 from run07). Dropping entire
affected shards would discard **105,839 otherwise eligible rows**, leaving
1,412,086 instead of 1,517,925. The next implementation therefore targets exact
reviewed row exclusions, separately preserving the existing no-result rule. Its
identities must include source namespace, shard and physical row, with actual
derivation identity checks; any unlisted defect remains fatal. No repair,
derivation or training admission has occurred in this result.

Independent compact review reconciled all shard counts and 9,228 unique
diagnostic identities. Saved diagnostics occupy 5,178,177 bytes, below the
512-MiB cap. [Completion evidence](evidence/ceres-prefix-and-raw-audit-completed-20260913.json)
preserves the exact report and review identities, the 30 required exclusions and
their limits. Publication did not repeat raw payload checks or the audit.

## Exact filtered derivation launched

At **19:27:03.448 UTC**, the parent launched the reviewed row-filtered derivation
of the same 184 audited raw shards. The implementation merged in
[PR #731](https://github.com/jjoshua2/DeepFin/pull/731) binds the 30 exact
additional exclusions to audited source/shard/physical-row identities and retains
the existing 9,198 no-result drops. Expected output is **1,517,925 rows**:
1,006,190 from run06 and 511,735 from run07. These are expectations, not completed
derivation counts. Policy and value selectors remain unchanged; no scores are
repaired and no unlisted defect is silently skipped.

The frozen runtime `a22893d1434d05365a33a95c846d3b8bf47b1eed` processes run06
then run07 sequentially, one worker per source, on CPUs 8,9 with two numeric
threads, nice 19 and a hidden GPU. The **three-hour inclusive allocation** uses
one shared deadline and owned process-group cleanup. Available RAM must be
48 GiB at startup and 32 GiB during sampled checks, with 150 GiB disk reserve
and an 8-GiB aggregate allocated-output threshold sampled every 30 seconds.
Sampling thresholds are not hard quotas. The start recorded 88,641,429,504 bytes
available RAM and 189,193,846,784 bytes free disk.

Before launch, review caught and fixed a restart-freshness issue: existing or
symlinked execution receipts and stage directories are refused before writing
the start record. Parent and independent final preparation reviews passed.
Partial evidence is retained on failure; completing only the first source does
not establish complete two-source derivation. Actual output qualification and
exact retained-row alignment in downstream teacher joins remain necessary.

[Launch evidence](evidence/filtered-raw-derivation-launched-20260913.json) retains
the immutable start, command, plan, runtime and review pins. No completed output,
adapter qualification or training admission is claimed here.

## Completed exact filtered derivation

Both sequential sources completed with **exit 0**. The outer execution took
**2016.500 seconds** (33 minutes 37 seconds), within its three-hour allocation.
Actual retained rows equal the audited expectation: **1,517,925 rows**, written
to **186 derived shards**. Raw and derived shard counts differ because output
is repacked at 8192 rows per shard.

| Source | Physical raw rows | No-result drops | Exact audited exclusions | Derived rows | Derived shards |
| --- | ---: | ---: | ---: | ---: | ---: |
| run06 | 1,013,011 | 6,799 | 22 | 1,006,190 | 123 |
| run07 | 514,142 | 2,399 | 8 | 511,735 | 63 |
| Total | 1,527,153 | 9,198 | 30 | 1,517,925 | 186 |

The two source stages took 1348.194 and 667.376 seconds. This is actual elapsed
execution, not a controlled speedup or peak-memory comparison. The recorded
selectors remain uniform-d9 phase-zero policy and the existing latest-phase
composite value rule, with temperature 0.0005 and zero requested policy floor.
All retained rows have verified input keys; the outputs retain source directory
and configuration, raw shard and physical row, plus original/stored history
input keys in per-shard provenance. The 105,839 eligible rows that a whole-shard
exclusion would have discarded are preserved by the exact row filter.

The inherited sharp-SF target storage reports 21,617,693 positive move entries
underflowing to zero in float16. This counts move entries, not excluded rows;
no new clamp or target repair was applied by the row filter.

[Completion evidence](evidence/filtered-raw-derivation-completed-20260913.json)
records the actual terminal and both summary pins. This completes derived
baseline eligibility and provenance only. Matching these filtered/permuted rows
to the existing raw BT4 policy/native-WDL banks still requires an exact adapter
and consumer qualification; no BT4 adapter or training admission is claimed.
