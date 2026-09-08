# Value collection and two-epoch runtime readiness

The matched value-head sample supports trying one modest SF/BT4 value mixture
before a nearly redundant Ceres arm. It does not establish better calibration,
reduced bias or playing strength. This record tracks the collection and runtime
work needed to run that test and the separate training-horizon comparison.

Related science: [value-head readout](2026-09-08-sf-anchored-value-bootstrap.md)
and [completed G50/B100 comparison](2026-09-08-bt4-g50-b100-dose-comparison.md).
[Compact measured evidence and launch identities](evidence/value-bootstrap/collection-readiness.json)
contain the exact receipt hashes; bulk labels and runtime logs remain external.

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

Charging every source shard this same total cost projects about **6.83 hours** for
the whole corpus. This is a planning extrapolation from one fixed shard. Sharing
session startup may reduce cost, while other shards may cost more.

## Next collection and value test

Exactly the next **128 shards / 1,048,576 rows** are registered and queued behind
the active horizon GPU check. The existing pilot is pinned and excluded from
relabeling. The same model, output, batch 128 and two-thread settings apply.
The one-hour inclusive bound covers lease wait and cleanup; the bank has a sampled
128 MiB output cap, an 8 GiB ORT allocator budget and a 150 GiB free-disk reserve.
Completed labels remain reusable. There is no automatic expansion or retry.
[Prefix registration](https://github.com/jjoshua2/DeepFin/pull/582#issuecomment-5590119219).

The measured amortized cost will determine the remaining collection budget. After
full coverage and target qualification, the first candidate is
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

One new attempt is registered and running with fresh output and compiler caches.
The actual 61,444,448-parameter model, 1,024-row fixture, batch 512, two epochs,
max-autotune and two numeric/compiler threads are unchanged. The inclusive operator
cap is 3,000 seconds. It refuses to begin unless at least 1,890 seconds remain,
then grants a fixed 1,830-second stage including 30 seconds of termination grace.
No automatic retry, eager fallback or smaller-model substitution is allowed.
[Revised registration](https://github.com/jjoshua2/DeepFin/pull/530#issuecomment-5590016165).

Success still requires actual compiled-graph evidence, four finite loss calls,
zero retried/skipped updates, two completed seeded schedules, checkpoints and final
source pins. Four warmup updates qualify narrow runtime plumbing; they do not
qualify full-corpus memory, the release schedule, throughput or playing strength.

In parallel, a separately registered CPU job computes actual SF/B100 plans for
seeds 0 and 1 and compares source-qualified schedule inputs. It runs the real
trainer objective-mask census and reads all compressed bytes twice per corpus;
seed 1 reuses verified records. Logical parity is a code-backed commitment to
sufficient scheduling inputs, not an emitted-row-stream hash. Different physical
policy-content hashes are retained.

The planning job has a one-hour bound, two separate CPU cores, 12 GiB address-space
cap, 32 MiB sampled output cap and 150 GiB reserve. Its prospective training loader
budget is two workers / 8 GiB and must carry into any later manifest. Neither a
partial plan nor a CPU success substitutes for completed CUDA qualification or
realized training evidence.
[Full-corpus planner registration](https://github.com/jjoshua2/DeepFin/pull/578#issuecomment-5590118274).
