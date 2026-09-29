# External trainer throughput inventory for 500M planning

Read-only publication of the September 26 archived inventory and its independently
reviewed sizing addendum. No new benchmark, data staging, trainer run or GPU work
was performed for this record. The earlier [500M continuation](2026-09-21-500m-continuation.md)
registered the packed trainer comparison; this record reports its completed
receipts and the next measurement proposed in that historical inventory.
The subsequent [September 29 same-ZIP screen](2026-09-29-zip128-external-trainer-screen.md)
reports the completed successor separately; it does not retrospectively change
these archived observations or qualify a sustained 500M-epoch rate.

## What the completed runs measure

| Run | External packed ZIP | Local directory | External/local | Scope |
| --- | ---: | ---: | ---: | --- |
| Cold post-driver pair | 1,387.1648 rows/s | 1,363.9366 rows/s | **1.01703** | Same 256-shard bank, 1,963,948 rows and 3,836 updates; 1,873,836 row visits after the first two 88-step windows |
| Descriptive warm recovery | 2,118.3455 rows/s | 2,191.0487 rows/s | **0.96682** | Same measured row count and batch parity, but separated by a failed supervisor and another GPU workload |

The cold pair's frozen plan and final source audit passed exact packed-trainer
parity. Initial model, raw sampled batches, prepared host batches, and row/game
order matched. Both ratios exceed the previously registered external/local
**0.90** short-screen threshold. The cold arms took 2,030.727 and 2,046.606
seconds in full, including startup and observer work. Their first batches took
46.675 and 34.688 seconds; registered-window prefetch waits were 222.967 and
214.939 seconds. The measured rate includes a late final-batch shape compile
and batch-hash observer overhead. The warm recovery is a separate descriptive
observation, not a controlled cache-warmth effect.

The completed original-E local-directory epoch processed **58,090,688 rows in
47,871.4264 active seconds**, or **1,213.47 run-level rows/s**. Holding that
rate fixed over 500M rows gives **4.769 ideal training days**. This is a
sensitivity calculation; it excludes preparation, stalls, contention, failures
and possible rate changes with a larger working set. The local filesystem is
ext4 and the external path is a 9p mount. The name `nvme_directory` in the
historical comparison does not prove the local mount's physical backing.

The short pairs do not isolate storage medium: external used ZIP Zarr and local
used directory Zarr, in fixed order with possible cache effects. They do not
measure a sustained external 500M epoch. The independent source-note reviews
passed arithmetic and receipt checks within their stated read-only scopes;
the [compact evidence](evidence/2026-09-28-500m-external-trainer-inventory.json)
pins the archive, review and upstream receipt hashes without publishing private
host paths or bulk logs.

## Historical proposal for one same-format screen

The archived inventory proposed copying sealed source ZIPs to local ext4, then
comparing ZIP against ZIP with identical trainer settings in a quiet I/O window.
Its addendum corrected the selection: the **lowest 64 shard IDs without a
full-shard filter** contain partial shards and end in a **201-row batch**.
The completed cold pair's own final 511-row batch caused windows of 542.418
seconds externally and 554.215 seconds locally. A new tail shape could dominate
a short post-warm rate. Do not use the unfiltered 64-shard selection for the
0.90 decision.

The proposed quantitative successor selects the **lowest 128 full 8,192-row
ZIP shards**, through global ID 142: **1,048,576 rows, 2,048 full 512-row
updates, 24 reporting windows (up to 88 updates each; 24 in the final window),
958,464 measured row visits in 22 post-warm windows, and 714,978,885 source ZIP
bytes by archived file-size metadata**. The 64-full-shard option is only a smoke
check. These were prospective sizing facts in the September 26 archive, not a
registered or completed benchmark at that time; future registration must
revalidate source bytes and seals. The linked September 29 successor retains
its own decision rule and execution evidence.

Before a GPU run, a separate reviewed registration must pin the exact roster
and source/local ZIP hashes, runtime, driver, model, loader, batch observer,
randomized arm order, serial sole-GPU lease, quiet-disk condition, deadlines
and resource bounds. Require matching consumed game/ply order, initial weights,
raw/prepared batch digests, source preflight and posthash. Report full wall,
first-batch, per-window, prefetch and observer times. The archived proposal
retained the prior nominal 0.90 rate ratio and required preregistration of
borderline-result handling; one close pair is inconclusive as a stable storage
decision. A nonborderline pass would qualify only the short storage path. A
sustained external trainer tranche or full run at the intended source mix and
operational contention is still needed for a 500M external-wall forecast.
This record launches none of it.
