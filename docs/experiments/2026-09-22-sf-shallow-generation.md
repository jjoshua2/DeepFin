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
