# Full-width Stockfish d8 versus d6 generation screen — September 22

Depth 6 yielded **65.07 eligible rows/s**, versus **33.50 rows/s** at depth 8 (1.943×) in one short fixed-order CPU screen. This measures newly generated self-play using the existing full-width generator. Both arms score every legal move; root-only or teacher-neutral generation was not measured. The result does not establish equal target quality or playing strength.

## Preregistered procedure

Before launch the parent independently reviewed and rehashed the [plan](evidence/sf-shallow-generation-20260922/plan.json) and harness, then approved exactly two arms: `all:8` followed by `all:6`, each with a 175-second generation cutoff. Both use four Stockfish workers, one engine thread each, CPU cores 20–23, nice 19, and two numerical-library threads. CUDA is hidden. The original engine, opening book, seed 20260922, temperature, 400-ply limit, 256-row shard setting and other configuration are identical apart from depth and fresh output/run names. No existing experiment or corpus is resumed.

The primary metric is eligible rows in progress-listed closed-game shards at the generation cutoff divided by actual cutoff wall time, including startup. The fixed cutoff excludes games/shards that finish only during termination. The existing production decoder and target derivation checks validate row identity, result, stored input, policy support and value support; this screen does not write neural teacher labels. Closed games include capped games, whose result-less rows are separately excluded from eligible throughput. No-result rows are not silently accepted.

Bounds were 700 seconds aggregate plus at most 10 seconds owned-process cleanup (under 12 minutes), 2,200 observed CPU seconds, 40 GiB available RAM, 150 GiB free disk and 1 GiB output, with a 75% output stop margin. STOP, resource and time guards remain active. All owned worker/engine descendants are terminated and reaped between arms. The retained-tablebase engine arm was omitted before launch because this generator exposes no corresponding UCI option passthrough.

## Results

| Measure | Full-width d8 | Full-width d6 |
| --- | ---: | ---: |
| Cutoff wall seconds | 175.004 | 175.010 |
| Banked rows at cutoff | 5,862 | 11,788 |
| Eligible rows | 5,862 | 11,388 |
| Rows without result, excluded | 0 | 400 |
| Invalid rows | 0 | 0 |
| Closed games / shards | 29 / 16 | 56 / 32 |
| Unique source identities / input keys | 5,862 / 5,862 | 11,788 / 11,788 |
| Eligible rows per wall second | 33.496 | 65.070 |
| First manifest observed, seconds | 2.252 | 2.079 |
| First progress file observed, seconds | 85.734 | 70.438 |
| Observed child CPU seconds | 591.10 | 574.02 |

The 400 excluded depth-6 rows are exactly worker 3, game 51, plies 0–399, matching the configured 400-ply cap. There are zero duplicate source identities within either arm. Unfinished in-memory game rows are uncounted. The first-progress metric is a polled file-appearance latency, not an engine-initialization timer; lifetime CPU totals cannot identify a steady-state hotspot.

Both arms stopped at their registered cutoff; child return code −15 is the expected controlled stop. Overall status is `COMPLETE_SCREEN`, outer return code 0, elapsed 362.385 seconds, observed child CPU 1165.12 seconds and controller CPU 18.25 seconds. Banked files totaled 23.41 MiB at completion. Post-run runtime HEAD and all input/code pins still match, and no screen supervisor or Stockfish children remain.

## Implications and limits

The measured depth reduction is a useful cost lever to test with downstream training. Depth changes also change moves, game lengths, results and target labels. One fixed-order 175-second observation per arm supplies no statistical uncertainty estimate, strength evidence, cold-cache comparison or fleet scaling guarantee. The earlier 600-second depth-8 confirmation measured 64.23 rows/s, versus 33.50 here over 175 seconds. That difference illustrates startup, opening and full-game buffering sensitivity; this is not a controlled regression comparison. Startup and full-game buffering materially affect these rates; input uniqueness here is within each arm, not deduplication against the existing corpus.

A purely arithmetic extrapolation of these startup-inclusive rates would take 172.8 days at d8 or 88.9 days at d6 to generate 500 million eligible rows on this four-worker configuration. For a one-third-SF share of 167 million rows, the depth-6 rate would imply about 29.7 days on four workers if sustained, leaving little margin in a 30-day plan. These are illustrations, not forecasts or compute commitments; labeling, deduplication and training costs are additional. This does not establish the cheapest route to 500M.

A separate design opportunity is teacher-neutral generation that avoids scoring every legal move with SF. If BT4 or Ceres drives cheap-policy generation, the same legal-policy/value outputs could be retained as labels where the target contract matches, avoiding redundant inference by that generating teacher. Neither root-only generation nor this reuse is implemented or measured here. Any adoption needs its own provenance and target-equivalence checks.

## Evidence and validation

[Compact readout and exact file hashes](evidence/sf-shallow-generation-20260922/readout.json); [prior launch approval](evidence/sf-shallow-generation-20260922/parent-approval.json). Raw corpora, cutoff progress records, full samples and the exact harness remain under `~/chess-artifacts/operations/sf-shallow-generation-screen-20260922`. Plan SHA256 `8eea58cde112942ef1ed6c94d0778e3d33e31c66204331d4766c6fb97a99f7f6`; executed harness SHA256 `25c3ac0be09f3473f3d9e86af1fbe85bf1b88c71776bca405359263f035d02fd`. The runtime is clean commit `3d9802eab4d98433e97856058216b1cbb3a57dc5`; its change from the historical throughput runtime affects only documentation and a separate scalar-value benchmark, with generation/derive/UCI source hashes unchanged.

The production eligibility readout audited every counted row. The [independent parent bank review](evidence/sf-shallow-generation-20260922/parent-bank-review.json) checked both arms' raw/eligible/missing counts, games, shards, uniqueness and identity hashes. Post-run preflight passed again. Publication changes contain documentation and compact evidence only; no runtime changes or additional generation are included.

## Longer depth-6 confirmation: completed c4, recovered c8 bank, failed cleanup gate

A separately preregistered confirmation compared four then eight single-threaded full-width d6 workers for 600 seconds each. It preserves the short-screen result above and tests longer-run capacity. The [independently reviewed plan](evidence/sf-d6-capacity-confirmation-20260922/plan.json) retained startup-inclusive eligible rows/s as primary, with ~30-second closed-shard increments and the final-five-minute rate as secondary. It set 1.6× c8/c4 as useful scaling (80% of ideal) and 128.6 eligible rows/s as a capacity reference for roughly 167M rows in 30 days at 50% effective availability. Neither reference is a target-quality gate.

**The overall run failed its cleanup gate.** C4 completed and was independently audited. C8 reached its generation cutoff and saved its exact progress-listed shard roster, but the harness raised `owned process survived cleanup` before serializing its result. The cleanup helper checks descendants only 0.1 seconds after SIGKILL; the later read-only ownership audit found no remaining supervisor, generator or Stockfish process. This does not retroactively pass the original check. The original `FAILED_BOUNDED_SCREEN` receipt is preserved (elapsed 1248.562 seconds, SHA256 `93e70074308ce17dcc1c7206166a9df4859b50c01ad170a23eb6dbd110ee75f1`). A versioned v2 harness with a 10-second bounded poll/reap and a pre-cleanup cutoff receipt is prepared and parent-reviewed, with five independent cleanup tests passing; its review receipt is under `~/chess-artifacts/operations/sf-retained-generation-plan-20260923-v2/parent-review.json` (SHA256 `4056183eb7a89576f41d4a656a62d56af9b37b3fdec819c1977029335d1d453b`). It has not launched, and the executed harness and failed receipt remain unchanged. No engines or additional games were launched to recover the bank.

| Measure | Four workers: saved result | Eight workers: recovered bank |
| --- | ---: | ---: |
| Banked rows | 68,073 | 140,940 |
| Eligible rows | 67,673 | 139,340 |
| Result-less rows excluded | 400 | 1,600 |
| Closed games / shards | 350 / 182 | 734 / 379 |
| Missing shard rows | 0 | 0 |
| Invalid rows / duplicate source identities | 0 / 0 | 0 / 0 |
| Unique input keys | 68,073 | 140,939 |
| Exact generation cutoff seconds | 600.006 | unavailable |
| Eligible rows/s | 112.787, exact saved cutoff | 232.233, nominal 600 seconds only |
| Final-five-minute closure rate | 120.893 over 299.008 seconds | unavailable |
| First manifest / progress observed, seconds | 1.843 / 69.395 | unavailable |

The c8 cutoff roster was saved at `2026-09-23T00:12:09.340019+00:00` (file mtime). Its exact monotonic cutoff elapsed, startup latency, sample series, closure windows and c8 CPU accumulator were not serialized. File mtime is not substituted for elapsed time. Therefore **232.233 rows/s is a conditional nominal-budget calculation**, not a passed exact primary measurement. The corresponding ratio is 2.059× (103.0% of ideal), numerically above both preregistered capacity references, but the registered complete-run gate failed and the c8 secondary timing metric cannot be recovered from these receipts.

The unchanged production eligibility checker subsequently audited all c8 cutoff-listed rows in 94.26 seconds on cores 16–17 at nice 19, under a 150-second recovery bound. An independent stdlib JSON audit then verified raw/nonmissing counts, uniqueness, per-shard counts and the identity hash. Four 400-row games lacked results and were excluded: worker/game `(1,481)`, `(3,51)`, `(4,124)`, `(6,198)`. There is one repeated input key across distinct source identities in c8, so these counts are not globally deduplicated production yield. All original failed receipts and raw shards remain intact.

The longer plan's caps were 1,500 seconds plus at most 10 seconds cleanup, 8,200 observed CPU seconds, 40 GiB available RAM, 150 GiB free disk and 1 GiB output with a 75% stop margin. Four workers used cores 20–23; eight used cores 20–27, all at nice 19 with one engine thread each. Both arms retained six-man Syzygy adjudication, using the configured `data/syzygy_3-4-5` and `data/syzygy_6` tablebase paths. A 10,000-game quota avoided a small-quota early stop. The failed receipt retains only 2028.97 observed child CPU seconds from completed c4; it is not total c4+c8 compute. No GPU job or existing corpus was changed.

**Concurrent-training cost matters.** E's windows 820–829 averaged 27.991 seconds, compared with 40.327 for 830–839 and 44.236 for 840–849. Prefetch waits rose from 11.803 to 24.455 and 28.604 seconds, while non-prefetch work stayed near 16 seconds. This was not one isolated 95-second outlier. E windows 864–869, observed after the generator stopped, returned to a mean 24.321 seconds with 7.787 seconds waiting. That pattern supports shared-resource interference; it does not identify the causal share because data/cache conditions vary and the source log has no per-window timestamps. Host memory/disk reserves stayed above the guards, and E had all 32 logical CPUs available while generators used overlapping subsets. Treating this CPU generation throughput as free alongside training would be unwarranted.

Pure arithmetic at the saved c4 rate implies 17.1 days for a 167M SF share at full availability; the recovered c8 nominal rate implies 8.3 days, or 16.6 at 50% effective availability. These are conditional illustrations, not forecasts: the c8 cleanup gate failed, neural labeling/deduplication costs are additional, concurrent training slowed, and no target-quality equivalence was measured. Root-only/teacher-neutral generation and generating-teacher output reuse remain separate, unmeasured opportunities.

[Compact readout and exact external file pins](evidence/sf-d6-capacity-confirmation-20260922/readout.json), [c4 independent audit](evidence/sf-d6-capacity-confirmation-20260922/d6_c4.independent-review.json), [recovered c8 independent audit](evidence/sf-d6-capacity-confirmation-20260922/d6_c8.independent-review.json), and [E timing evidence](evidence/sf-d6-capacity-confirmation-20260922/concurrent-E-timing-snapshot.json). Full raw data, original failed receipt, launch inputs, recovery code and receipts remain in `~/chess-artifacts/operations/sf-d6-capacity-confirmation-20260922`. No rerun or follow-on generation is included in this record.
