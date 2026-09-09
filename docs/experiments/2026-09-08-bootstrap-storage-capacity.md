# Storage capacity changes the 100M bootstrap plan

The September 8 read-only inventory does **not** support retaining 100 million G10 raw rows, their common-input data and multiple full recipe copies together on the current local disk. The archive has ample capacity; the local working set is the constraint. A larger experiment needs explicit staging and recipe retention, or suitable added working capacity. Neither design is qualified yet.

The **earlier inventory snapshot** in the [capacity evidence](../../scratchpad/storage_scaling_20260908/publication/capacity.json) records 397.49 GB free on local ext4, leaving **236.43 GB above the existing 150-GiB reserve**. The archive mount is a 10.0007-TB volume with **8.1855 TB free**, exposed through 9p. GB/TB here are decimal; GiB is binary. Free space is an observation, not an allocation reserved for this experiment.

## Continuing storage objective — September 8 update

Storage is part of completing the approximately **100M-position bootstrap**.
Continue transferring completed cold data from SSD to the external archive
throughout research, and reclaim the exact local copies after verification and
current dependency checks. This is ongoing work, not an optional cleanup pass
at the end of the experiments. Cold reproducible derived recipes come first;
preserve live generators, original bootstrap sources, selected controls,
checkpoints and required sidecars.

The latest recorded observation is **359 GiB free on SSD and 7.5 TiB free on the
external drive**, rounded figures from the continuing-objective note. Refresh
these observations before substantial allocations. The **150 GiB operating
reserve remains a floor, not proof that the full workflow fits**. Maintain a
working-set budget for raw and common corpora, policy/value labels, concurrent
rewrite and transfer temporaries, selected training datasets/checkpoints and
continued generation. Track each archive's destination and verified identity,
pending copy/verification/removal work, and bytes actually reclaimed. A successful
copy alone recovers no source space; archive capacity does not qualify training
directly from the external filesystem.

Use bounded low-priority transfers alongside GPU research. Before removing an
exact local copy, verify archive contents and external readback, recheck source
stability and current consumers, and retain the recipe/source/checkpoint evidence
and restore location in the experiment record. Restored data must be admitted
for its next consumer because inode/ctime identities can change. No broad new
scan, transfer or deletion was performed for this publication.

### G20T1 copy completed; source retained

The first cold-recipe copy completed in **2,977.57 seconds (49.63 minutes)**,
with outer exit code zero. Its external archive contains **713,331 verified
members** and occupies **13,529,845,760 bytes**. The local and external readback
SHA-256 agree: `c742245979a3878ba8c8d212703f7a95ebae4e7df740a7a8f7ca2de4f607aa2d`.
The original dataset remains local: **source bytes reclaimed = 0**. Removal of
the verified staging tar is distinct from recovering space occupied by the
original dataset. The complete archive retains the source and recipe evidence.

The [independent completion review](../../scratchpad/bt4_joint20/publication_20260908_storage_objective_v1/g20t1/independent_review.curated.json)
passed the copy/content chain. Before reclamation, compare current source-entry
stamps and exact shard membership with the saved manifest, check current
consumers/dependencies and external identity, and retain the two top-level
recipe summaries at their original paths. The review checked the current root
and those summaries, not every current source entry. Its **12,171,591,409 logical
source bytes** are not a measurement of allocated space that removal will recover. File contents and recorded member metadata were
verified; ACL/xattrs were captured but not independently compared, and no restore
test is claimed. The original process visibility limitations remain in the
[copy receipt](../../scratchpad/bt4_joint20/publication_20260908_storage_objective_v1/g20t1/verified.json).
The [byte-exact full review](../../scratchpad/bt4_joint20/publication_20260908_storage_objective_v1/g20t1/independent_completion_review.json),
[outer completion](../../scratchpad/bt4_joint20/publication_20260908_storage_objective_v1/g20t1/operator.actual_complete.json)
and [snapshot manifest](evidence/bt4-bootstrap/storage-objective-v50-manifest.json)
retain exact source hashes. The earlier preparation below remains historical.

### Account for the next value rewrite

The [B100V50 preparation](2026-09-08-bootstrap-next-value-and-policy-contrasts.md#first-b100v50-with-policy-supervision-fixed)
is frozen and independently reviewed, but unlaunched. It retains B100 policy and
rewrites only the value target at 50% SF / 50% BT4. Budget its ordinary new copy
and partial output explicitly: **32 GiB sampled allocated-output ceiling** and
**150 GiB free reserve**, not an output-size estimate or capacity reservation.
The six-hour ceiling covers the producer command including cleanup; supervisor
metadata prelude and final admission are outside that timeout. No six-hour
end-to-end bound is claimed. Full WDL-bank completion and a final bound launch
remain pending; the original V10 plan stays untouched.

The [preparation review](../../scratchpad/bt4_joint20/publication_20260908_storage_objective_v1/v50/parent_preparation_review.json),
[author receipt](../../scratchpad/bt4_joint20/publication_20260908_storage_objective_v1/v50/preparation_receipt.json)
and [lossless prepared plan](../../scratchpad/bt4_joint20/publication_20260908_storage_objective_v1/v50/plan.prepared.json.gz)
are banked with the snapshot manifest. This preparation does not establish that
V50 plus the growing 100M workflow can coexist without further reclamation.

## What the measured sizes imply

[Complete worker-progress snapshots](../../scratchpad/storage_scaling_20260908/publication/raw_density.json) cover 36,025,871 G10 raw rows: 28,526,032 from run06 and 7,499,839 from run07. Their closed compressed files occupy 92.05 GB logically, approximately 2,555 bytes per raw row. Open tails and subsequent generation are excluded; no raw payload was read.

| Component | Observed basis | Projection or consequence |
| --- | --- | --- |
| Additional raw data to reach 100M G10 rows | Current closed-file compressed density | About 163.47 GB more; leaves 72.96 GB above the reserve |
| Common-input working output | Completed 1,057,326-row batch, sampled peak 1.125 GB logical output | About 106.41 GB for 100M common rows |
| One separate final recipe copy | Completed original B100: 14,593,728,512 allocated bytes for 18,910,484 rows | About 77.17 GB allocated per 100M rows |

These are size proxies from different workloads, not interchangeable measurements or guaranteed future costs. The common-bundle projection is **gross**, including a subset already prepared locally; it is not all additional storage. Logical bytes understate allocated blocks, while the recipe estimate uses actual allocation. Different histories and policy entropy can change compression. Raw teacher sidecars, additional value banks, checkpoints, temporary copies and continued generation consume further space and are omitted from this comparison. One hundred million raw rows also does not mean exactly one hundred million eligible training rows.

The original capacity snapshot used 5,276,543 prepared common rows; the [current G10 record](2026-09-07-g10-transfer-readiness.md) reports 9,298,514. Combined training qualification remains separate. The existing subset does not close the much larger gap created by new raw data and multiple ordinary recipe copies. Smaller, explicitly budgeted increments remain plausible; blind 100M multi-recipe coexistence does not fit the current reserve policy.

## Archive evidence and the next storage decision

The [38-receipt refresh](../../scratchpad/storage_scaling_20260908/publication/archive_refresh.json) matches every previously recorded receipt hash and every tar's size and mtime. The tars total 85.32 GB. This reuses the [earlier full-content verification](2026-09-07-storage-archive.md); it is **not** a new payload checksum or restore test. Twenty-six archived source pools remain local, with prior dependency exclusions still relevant. No leftover staged tar duplicates were found in the two known staging locations.

This record recommends no deletion of historical checkpoint or holdout candidates and no relocation of identity-pinned inputs. The small candidates inspected would not materially solve the scale constraint. The inventory left raw sources, derived corpora, policy/value sidecars, checkpoints, openings and frozen runtimes untouched. The later retention section refines which completed recipe payloads need remain on fast storage. Archive free space does not qualify direct training over 9p, and moving existing inputs can invalidate their path and storage-identity receipts.

Before committing the larger scale experiment, account for the actual retained raw, common and recipe working set, including temporary publication space and concurrent growth. Then qualify a staged-data/recipe-retention approach or suitable additional working storage. This is a change to the scale plan, not an approved storage implementation. The inventory performed no move, deletion, payload hashing, full-disk recursive scan or I/O benchmark. The [publication manifest](../../scratchpad/storage_scaling_20260908/publication/manifest.json) binds the compact projections and original evidence identities.


## First storage measure: archive completed unselected recipe payloads

The [consumer-source review](../../scratchpad/storage_scaling_20260908/publication/retention_feasibility_v1/receipt.json), bound to main `d53c749`, identifies a practical first measure before changing loaders: archive completed unselected recipe shards while keeping their checkpoint and lineage evidence locally. The reviewed checkpoint/arena consumers read checkpoints, completion receipts, run summaries and saved realized schedules; they do not reopen the training shard payloads. Existing role and protocol restrictions still apply.

Completed G20T1, G20T05, G50T05 and top-tie payloads can become cold once no active job or selected producer needs them. C20 and H20 payloads are likewise optional for checkpoint evaluation, conditional on no further training or materialization needing them. **Keep the full original SF and B100 corpora for current value/horizon work, and active SoftSF10 through training and realized-schedule verification.**

Keep each policy recipe's `derive_targets_summary.json` and `bt4_policy_mix_summary.json` byte-exact at its original corpus path: H20 admission specifically reads both C20 summaries, and B100V10 reads both B100 summaries. Also retain the separate C20 **run** `summary.json`, which current training admission pins. Preserve selected checkpoints, training completion and run summaries, saved schedules, registration/qualification receipts, and arena banks/runtime/book evidence. Retaining these files preserves the reviewed metadata dependencies; it does not leave a trainable corpus after shards become cold.

The complete archive must retain shard metadata and payloads for reconstruction. A later byte-exact restore at original paths can preserve the current horizon content/path digest, but changed inode/ctime can invalidate separate storage snapshots. Re-admit restored data for its next consuming job without rewriting historical receipts or repeating completed matches.

At this earlier retention-review snapshot, archive copy, content verification and local removal had **not** been executed; the completed G20T1 copy is recorded above. A [six-directory metadata-size attempt](../../scratchpad/storage_scaling_20260908/publication/retention_feasibility_v1/candidate_size_inventory_attempt.json) reached its 55-second cap without totals, so aggregate savings remain unmeasured. The existing 13.59-GiB B100 measurement is not a measured size for every recipe and does not establish that the 100M pipeline fits.

The separate [sharing review](../../scratchpad/storage_scaling_20260908/publication/retention_feasibility_v1/sharing_limit.json) rules out ordinary hardlinks as a drop-in change to pinned data: creating or removing a link changes the shared inode's ctime, which existing storage receipts bind. A new family linked before qualification could be investigated, but is unqualified. Its 74.2% common-array observation is from **one B100 shard**, not a corpus-wide savings estimate; archival remains the first measure.


## Original G20T1 preparation snapshot — before launch

At the original preparation snapshot, the first concrete archive step was **PREPARED, NOT LAUNCHED**: a [frozen plan](../../scratchpad/bt4_joint20/G20T1_cold_archive_v1/plan.json) for the complete G20T1 corpus (18,910,484 rows, 2,309 shards), including top-level summaries and shard provenance. The [independent review](../../scratchpad/bt4_joint20/G20T1_cold_archive_v1/independent_review.json) passed. This is a snapshot of one operational use of the existing archive helper, not a new shared archive framework or a change to the historical recipe verdict.

The [operator command](../../scratchpad/bt4_joint20/G20T1_cold_archive_v1/operator_command.json) is for a later G10-free window, using the existing nonblocking shared preparation lock. It is not queued. The strict four-hour deadline includes prelude and cleanup; CPU affinity is 0,1 with two numeric threads, low priority and GPU hidden. The 24-GiB tar ceiling and sampled aggregate staging limit are bounds, not measured corpus size; sampling may transiently overshoot. The local reserve is 150 GiB, with additional admission headroom and destination-space checks specified in the plan. Four hours is a failure ceiling, not a completion estimate.

Execution would inventory the source progressively, verify every tar member and its contents against that inventory, compare external readback and local archive digests, and recheck source stability. **The full source remains in place even after success.** Existing attempts refuse, with no automatic resume; failures retain source and partial archive state. Only the verified local staging tar can be removed after successful external readback. ACL/xattrs are captured but not independently compared, and no restore test is claimed.

The [publication manifest](../../scratchpad/bt4_joint20/G20T1_cold_archive_v1/publication_manifest.json) binds the exact helper, wrapper, command, plan and review evidence. Six focused disposable checks passed before a separately reviewed absolute-tool-path correction; their original source/result and correction supplement are preserved. This publication ran only link, hash and hygiene checks. No dataset inventory, copy, transfer, deletion or launch has occurred in this preparation.
