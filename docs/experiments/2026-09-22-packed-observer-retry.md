# Storage observer recovery and completed BT4 screen

The original storage pair failed after 125.23 seconds on its first sampled batch,
before any optimizer update. Its observer requested `ply`, but persisted shards
and the actual sampler return `ply_index`. The original synthetic observer test
invented the wrong field. The original failure, logs and outputs are retained;
there is no valid storage throughput result.

The correction observes `game_id` plus `ply_index` and rejects missing identity
flags. It leaves the returned tensors and trainer unchanged. A regression now
writes real shards and consumes a complete `GameAwareEpochBuffer` epoch through
the observer, checks recorded order against returned tensors, and verifies hook
restoration. Four focused CPU tests and scoped Ruff checks pass.

The fresh retry keeps the original qualified 256 shards, 35 sources and 1,963,948
rows, runtime, model/config, seed121, external-then-NVMe order, 45-minute arm and
90-minute pair limits, and preregistered 0.90 throughput threshold. Only observer
schema and output paths change. Prior failed output bytes remain in the combined
disk budget. Independent review approved queued admission after 20 passing observer/pair tests,
including the real-sampler regression, and verification of all 2,862 runtime
members and 10 source pins. The retry is prepared; queue insertion belongs to
the parent scheduler. Plan, review and descriptor are banked in the evidence directory.

The completed BT4 pipeline screen passed exact input and output parity. Median
producer-plus-verification time was 19.3321s original, 13.4608s optimized serial,
and 12.4678s with prefetch. Optimized serial throughput was 1.4362 times original;
prefetch was 1.07965 times optimized serial and passed the preregistered 5% screen.
These are two observations per variant on one 8,236-row banked shard with warm
sessions, not a fleet or 500M projection. Source hashes and medians are banked in
`evidence/packed-observer-retry-20260922/readout.json`.
