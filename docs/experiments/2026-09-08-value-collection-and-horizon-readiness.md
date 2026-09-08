# Value collection and two-epoch runtime readiness

The matched value-head sample supports trying one modest SF/BT4 value mixture
before a nearly redundant Ceres arm. It does not establish better calibration,
reduced bias or playing strength. This record tracks the collection and runtime
work needed to run that test and the separate training-horizon comparison.

Related science: [value-head readout](2026-09-08-sf-anchored-value-bootstrap.md)
and [completed G50/B100 comparison](2026-09-08-bt4-g50-b100-dose-comparison.md).
[Historical launch evidence](evidence/value-bootstrap/collection-readiness.json) and
[new completed evidence](evidence/value-bootstrap/collection-completed.json) retain
the receipt identities. Bulk labels and runtime logs remain external.

Snapshot: 2026-09-08, 19:36 UTC. Later completions are recorded separately.

## Completed value collection checks

[PR #580](https://github.com/jjoshua2/DeepFin/pull/580) added optional native WDL
retention to future BT4 policy labeling. The old labeler finished its queued group
and paused cleanly; the replacement consumed the matching handoff and retained the
same shared GPU lease and original labeling settings. The deployment preserves the
qualified Python 3.10 / ORT 1.23.2 native stack, with only the reviewed Python changes.
It is not a full-main native-runtime upgrade.

At the banked observation, **99,305 new rows across 12 shards** had native WDL;
**35,436,868 older policy-only rows** remained unchanged. One completed 8,243-row
shard was reread: native float32 W/D/L probabilities in side-to-move order were
finite and normalized within 1.41e-7, and their decoded-array digest matched the
receipt. The actual GPU provider was recorded. This checks output retention,
not a second raw-history reconstruction or complete historical value coverage.
[Adoption registration and readout](https://github.com/jjoshua2/DeepFin/pull/580#issuecomment-5589848848).

[PR #582](https://github.com/jjoshua2/DeepFin/pull/582) added a separate WDL-only
producer for completed original SF training shards. It uses the existing derived
inputs; the qualified conversion preserves the consumed 112-plane LC0 feed.
It does not reconstruct the original full float32 input key. The producer validates
source history metadata, stored input domains, row/game/ply identity, native WDL
and source stability, and keeps per-row feed digests.

The first canonical **8,192-row shard completed in 10.653 seconds** total wall time,
including startup, session work and teardown; producer time was 8.431 seconds.
All five small output arrays were independently hashed, the source identity stayed
unchanged, and parent completion followed successful child exit. The output bank
occupied 388,122 bytes. No inference was repeated for this verification.
[Prospective pilot and completed readout](https://github.com/jjoshua2/DeepFin/pull/582#issuecomment-5590014506).

The original one-shard extrapolation was about 6.83 hours for the whole corpus.
The completed larger prefix below now provides an amortized planning observation.

## Completed larger value prefix

Exactly the next **128 shards / 1,048,576 rows completed**, bringing the reusable
bank to **1,056,768 rows across 129 shards**. The existing pilot was excluded from
relabeling and its metadata, saved hashes and storage identity remain unchanged.
All 128 completion records, native float32 W/D/L contracts, five array layouts and
saved digest sets match the registered source selection. This readout reused the
producer's native validation and readback; it did not repeat inference or reread
source features or output-array payloads.

Parent wall time was **1,795.94 seconds**, including GPU-lease wait; the outer
operator-to-completion interval was 1,799.11 seconds, within the registered hour.
The following 127 shards published 1,040,384 rows over **602.12 seconds**, about
**1,728 rows/s**. No lease-acquisition timestamp was banked: the 1,189.47 seconds
before first publication includes waiting, preflight, session startup and the first
shard. The publication span is an amortized cost proxy, not a pure GPU timer or a
controlled speedup measurement.

At that proxy rate, the remaining 17,853,716 rows would take about **2.87 hours**,
before startup, waiting and teardown. A reviewed **four-hour prospective plan**
covers the remaining 2,180 shards, preserving the completed prefix. It retains the
shared lease and 150 GiB reserve, with a sampled 2 GiB logical-file-size bank cap;
physical allocation is tracked separately. It is prepared for a later GPU gap,
**not launched or queued**.
The same model, output, batch 128 and two-thread settings were used; the shared
lease, sampled 128 MiB bank cap, 8 GiB ORT allocator budget and 150 GiB free-disk
reserve were retained. Completed labels remain reusable, with no automatic retry.
[Prefix registration](https://github.com/jjoshua2/DeepFin/pull/582#issuecomment-5590119219).

After full coverage and target qualification, the proposed value candidate is
`0.9 * normalize(SF WDL) + 0.1 * normalize(BT4 winner WDL)`, stored as float16
and normalized by the existing trainer. B100 policy supervision, other labels,
initialization and the original one-epoch schedule remain fixed. This changes only
value targets; shared-trunk learning can still change the learned policy.
The implementation uses ordinary immutable copies and changes only `search_wdl`.
The unchanged historical consumer has passed a tiny target/gradient wiring proof;
that is not a full-corpus admission or completed training result.

## Two-epoch runtime qualification

The first compiled-CUDA probe **timed out and did not qualify**. Its stage lasted
572.571 seconds, with total operator time 850.189 seconds including 277.437 seconds
waiting for the GPU. It ended during first-graph cold max-autotune. No completed
optimizer update, objective table, epoch receipt or checkpoint is certified.
Resource-infeasible kernel candidates were discarded by autotuning; this does not
establish an uncaught training numerical failure.
[Independent timeout readout](https://github.com/jjoshua2/DeepFin/pull/530#issuecomment-5589805736).

The separately registered second attempt **passed** with fresh output and compiler
caches. The actual 61,444,448-parameter model completed two epochs of the
1,024-row fixture: **four batch-512 optimizer updates**, finite losses, zero retries
and zero skipped updates. The observer verified the actual compiled model wrapper
and one captured graph, with compiler-error suppression disabled. This establishes
compiled graph capture, not a separate CUDA-graph-replay claim.

The CUDA stage took **1,697.67 seconds**, and the inclusive operator took
**1,739.37 seconds**, within its 1,830/3,000-second bounds. Peak CUDA allocation was
11,118,537,728 bytes; peak reservation was 13,314,818,048 bytes. No eager fallback,
smaller-model substitution or automatic retry was used. Original failed receipts
remain unchanged. [Completed CUDA review](https://github.com/jjoshua2/DeepFin/pull/530#issuecomment-5590637482).

These four warmup updates qualify narrow runtime plumbing. They do not measure
steady-state throughput, full-corpus training memory, the complete learning-rate
schedule, ragged batch-511 CUDA behavior or playing strength. Historical control
limitations remain. They also do not migrate a live job to full-main native code.

The separately registered CPU planner also **passed** for SF and B100 with seeds
0 and 1. Each epoch contains **18,910,484 rows, 2,309 shards and 36,935 batches**:
36,699 batches of 512 and 236 of 511. Policy and WDL mask weights each equal the
row count; the other 12 objective mask sums are zero. The two-epoch total per arm is
37,820,968 rows and 73,870 batches. Source-qualified logical schedule witnesses
match between arms for each seed; different physical policy-content hashes remain.
SF is the planner's source and schedule anchor, not a mandatory second horizon
finalist.

The real planner performed its objective-mask census and compressed-content reads;
this was not a metadata-only job. It finished in **1,480.56 seconds** process wall
time (1,478.88 seconds body), with maximum process RSS 1,105,128 KiB. Its two-worker,
8 GiB prospective loader budget is retained; the planned 6.82/6.32 GiB working-set
estimates are not observed full-training memory. Logical witnesses commit to the
schedule-driving inputs under pinned code, not an independently emitted row stream.
No full two-epoch training comparison has completed, and the selected registration
and full preparation bindings still need assembly.
[Completed planner review](https://github.com/jjoshua2/DeepFin/pull/578#issuecomment-5590639848).

## SoftSF10 materialization complete

The raw-effective-cp policy alternative is now fully materialized and qualified:
**18,910,484 rows across 2,309 shards**, preserving the original 20M-row source
prefix and its 1,089,516 missing-result exclusions. At temperature 10 cp,
16,696,682 stored policy rows changed; maximum stored probability-mass error was
0.000457763671875. The producer retains the original cp/mate score semantics,
checks the old policy before rewriting it, and copies the **16 non-policy columns**
unchanged, including SF search value and history inputs.

The CPU process completed in **17,351.98 seconds (4.82 hours)**, with 666,064 KiB
maximum RSS and 12,548,145,152 allocated output bytes. Qualification reused that
completed full-row and compressed-copy proof, then refreshed all shard attrs and
array layouts without another payload scan. It preserves the original history
lineage and historical-control limitations; a genuine SF rewrite receipt is used.

At the earlier materialization snapshot, the prospective schedule was running.
The September 8 follow-up now records [completed original-epoch SoftSF10 training](2026-09-08-soft-sf-qualified-training-sample.md#september-8-update-original-epoch-training-complete):
18,910,484 rows, 36,935 updates and 420 windows, followed by matched realized-schedule
verification. Arena preparation passed and its coordinator started; no GPU stage
or playing outcome is established by the saved launch snapshot.
This retains the original B100/G50 training runtime and historical-control caveats.
[SoftSF training preregistration](https://github.com/jjoshua2/DeepFin/pull/569#issuecomment-5590602401).
