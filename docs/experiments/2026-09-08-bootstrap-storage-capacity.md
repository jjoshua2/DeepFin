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

The earlier continuing-objective observation was **359 GiB free on SSD and 7.5 TiB free on the
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

### E0T05 copy complete; source retained — September 9

E0T05's copy and content verification completed with exit zero in **2,484.38
seconds**. The archive contains **713,331 verified members** and occupies
**11,021,598,720 bytes**. Full external readback matched archive SHA-256
`a1ad137e9f65829f1dc386b5cfb2322617ff9789c60f127be6b653ac538d8473`.
The original dataset remains local: **source bytes reclaimed = 0**. Summaries,
checkpoints and restore evidence remain retained; any later source removal still
needs exact current source and consumer checks.

[Compact completion identities](artifacts/storage-t1-wdl-update-20260909/status.json)
bind the actual operator, content-verification and root-review receipts. They are
selected public fields, not copies of the operational logs or a new archive rehash.
The [earlier startup snapshot](artifacts/v50-training-g10-status-20260909/status.json)
remains historical. After this copy finished, [T1 materialization started](2026-09-09-bt4-target-temperature-horizon.md#materialization-started--september-9)
under its separate preparation plan. Copy completion provides no new reclaimed
capacity and does not establish that the full 100M workflow fits.

### G50T05 copy and reclamation complete — September 9 update

G50T05's external copy completed with outer exit zero in **2,944.11 seconds**.
The [copy review](../../scratchpad/bt4_joint20/publication_20260909_full_wdl_v50_storage_v1/g50t05/independent_completion_review.json)
records **713,331 verified members**, **11,936,466,856 logical source-file bytes**
and a **13,294,694,400-byte archive** with matching local/external SHA-256
`a50bebaac1b58074787ff9d1e9e4ce1db230ef94ca016722bb3a5b4788a9b208`.
Its source-retained status describes that completed-copy stage, before reclamation.

The separate removal completed in **79.90 seconds**, outer exit zero. The
[independent terminal review](../../scratchpad/bt4_joint20/publication_20260909_full_wdl_v50_storage_v1/g50t05/independent_reclamation_completion_review.json)
verified **2,309 unique removed shard directories and 4,618 ordered journal
records**, the exact bound list, unchanged hashes/stamps of both original local
summaries, and the external archive identity. The [lossless evidence bundle](../../scratchpad/bt4_joint20/publication_20260909_full_wdl_v50_storage_v1/g50t05-copy-reclamation.tar.gz)
retains the completion, source checks, command and removal journal/list. No transfer,
delete, payload hash replay or restore test was performed for this publication.

The removed shards had **14,495,428,608 allocated bytes (13.500 GiB)**. Concurrent
whole-filesystem free space rose by **14,464,651,264 bytes (13.471 GiB)**; this is
not isolated attribution. G20T1, G20T05 and G50T05 are now cold local corpora with
original summaries retained; checkpoints and archive restore evidence remain.
Later row consumers must restore the shards and satisfy their identity admission.
Do not delete original SF/B100 data, live inputs, needed labels, selected controls,
checkpoints or holdout material as part of this cold-recipe workflow.

At the parent observation after this reclamation and at V50 rewrite launch, free
space was **422,991,724,544 bytes on SSD (393.942 GiB)** and
**8,145,479,417,856 bytes externally (7.408 TiB)**. These are current-snapshot
observations, not reserved allocations; the earlier 359/372/383 GiB figures below
are historical. Do not add any of the three reclamations again to this baseline.
The active V50 rewrite and raw label queue can consume space afterward.

[The updated storage objective and exact manifest](evidence/bt4-bootstrap/full-wdl-v50-storage-manifest.json)
keep continual transfers **and verified reclamation** part of bootstrap completion.
A 150 GiB reserve is still only an operating floor. Account for continued raw/common
data, policy/value labels, selected recipes, rewrite/transfer temporaries and
training artifacts before claiming capacity for 100M. The [V50-first sequence](2026-09-08-bootstrap-next-value-and-policy-contrasts.md#v50-preparation-complete-training-started--september-9)
now has an actual CPU rewrite launch; tactical work remains prepared, not launched.
The archive lock release did not make a GPU window: useful raw-WDL labeling remains
active and must be preserved. Existing archived restart/checkpoint pools retain
their exclusions; their tar totals are not a wholesale reclamation plan.
At that earlier snapshot, the next candidate was the cold E0T05
`qtemp_0.0005_hist_20m_bt4_toptie_t050` corpus; its completed copy is now recorded
above. Retain its two
summaries and both seed runs, checkpoints, schedules and arena evidence.

### G20T05 reclaimed; G50T05 copy launched — September 8, 23:24 snapshot

G20T05's complete copy and external readback finished in **2,917.96 seconds
(48.63 minutes)** with outer exit zero. The archive contains **713,331 verified
members**, occupies **13,155,164,160 bytes**, and has matching local/external
SHA-256 `bec3436aea9b7da55e4f3832a080bd9dca929658f2a3e8c02c1c8e732df07e28`.
The [copy review](../../scratchpad/bt4_joint20/publication_20260908_storage_g20t05_g50_v1/g20t05/independent_copy_review.json)
records **11,796,945,040 logical source-file bytes**. Its source-retained status
belongs to the copy-completion stage, before the separate reclamation below.

After fresh exact source-membership/stamp and consumer checks, reclamation
removed **2,309 shard directories in 86.35 seconds**, outer exit zero. The
[independent terminal review](../../scratchpad/bt4_joint20/publication_20260908_storage_g20t05_g50_v1/g20t05/independent_reclamation_completion_review.json)
verified all **4,618 ordered intent/removal journal records**, the exact shard
list, both retained original summary hashes/stamps and unchanged external archive
identity. The [completion receipt](../../scratchpad/bt4_joint20/publication_20260908_storage_g20t05_g50_v1/g20t05/reclamation.completed.json)
and [lossless copy/removal bundle](../../scratchpad/bt4_joint20/publication_20260908_storage_g20t05_g50_v1/g20t05-copy-reclamation.tar.gz)
preserve the command, source checks, journal and earlier binding evidence. The
[operator result](https://github.com/jjoshua2/DeepFin/pull/597#issuecomment-5595278607)
was recorded before this publication; no removal or payload rehash was repeated here.

The removed shards' prior allocation was **14,356,144,128 bytes (13.370 GiB)**.
Observed whole-filesystem free space rose from **396,554,809,344** to
**410,869,010,432 bytes**, a **14,314,201,088-byte (13.331 GiB)** change. Concurrent
activity prevents attributing that entire difference to this removal. These are
terminal observations, not currently reserved capacity. Do not add the earlier
G20T1 recovery again to a post-reclamation free-space baseline.

G50T05's [reviewed copy preparation](../../scratchpad/bt4_joint20/publication_20260908_storage_g20t05_g50_v1/g50t05/independent_preparation_review.json)
then [launched at 23:24 local time](../../scratchpad/bt4_joint20/publication_20260908_storage_g20t05_g50_v1/g50t05/operator.actual_start.json).
This is **launch evidence only: its source is retained, with no completed copy
or reclamation claim**. The [frozen launch bundle](../../scratchpad/bt4_joint20/publication_20260908_storage_g20t05_g50_v1/g50t05-copy-launch.tar.gz)
reuses the verified copy path with a four-hour inclusive deadline, CPUs 0,1,
two numeric threads, low priority, GPU hidden, a 24 GiB sampled staging cap and
150 GiB reserve. The cap is not an expected output size or an allocation already
reserved. Earlier G20 copy durations are references, not a G50 completion promise.

Continue these transfers **and verified reclamation** throughout the 100M work.
Keep original raw/SF/B100 inputs and required sidecars, live jobs, checkpoint and
holdout exclusions, both corpus summaries, and all training/schedule/arena/archive
proof. G20T05 is now a cold, summary-only local corpus: later row consumers need
restored shards and relevant identity admission; retained checkpoints can still
use the reviewed arena path. Current checks reuse the complete archive-content
proof and external stat chain; they are not a new full rehash, restore test or
independent ACL/xattr comparison.

The G50 copy uses the shared CPU preparation lock. [V50 remains first](2026-09-08-b100-tactical-policy-readiness.md#compute-order-and-storage-headroom)
after full WDL completion and archive release; disjoint affinities do not allow
simultaneous lock-owning recipe rewrites. Continued archive staging and label
growth require current headroom at the next allocation. The 150 GiB floor is
not proof that all raw/common/recipe/temporary data for 100M fit. Exact original
snapshot/member hashes are in the [publication manifest](evidence/bt4-bootstrap/storage-g20t05-g50-manifest.json).
This storage update changes no scientific verdict or training selection.

### Historical copy-completion snapshot — G20T1 source retained

The first cold-recipe copy completed in **2,977.57 seconds (49.63 minutes)**,
with outer exit code zero. Its external archive contains **713,331 verified
members** and occupies **13,529,845,760 bytes**. The local and external readback
SHA-256 agree: `c742245979a3878ba8c8d212703f7a95ebae4e7df740a7a8f7ca2de4f607aa2d`.
At that copy-completion snapshot the original dataset remained local: **source
bytes reclaimed = 0**. Removal of
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

### Completed G20T1 reclamation and next copy — September 8, 22:21 snapshot

After the archive content review, current source stamps/membership and consumer
checks, G20T1 reclamation completed with **2,309 shard directories removed** and
both original top-level summaries retained. Its [completion receipt](../../scratchpad/bt4_joint20/publication_20260908_v50_storage_v1/g20t1/reclamation.completed.json)
and [outer completion](../../scratchpad/bt4_joint20/publication_20260908_v50_storage_v1/g20t1/operator.actual_complete.json)
record exit zero and **86.49 seconds**. The exact removal list, metadata/consumer
checks, removal journal and operator source are preserved in the [compact evidence bundle](../../scratchpad/bt4_joint20/publication_20260908_v50_storage_v1/g20t1-reclamation.tar.gz).
The parent counted 2,309 unique journal removals and checked retained summaries;
the separate independent terminal review is pending at this draft snapshot.

The prior allocated size of those shards was **14,730,653,696 bytes (13.719 GiB)**.
Whole-filesystem free space changed from **384,425,050,112** to
**399,113,687,040 bytes**, a **14,688,636,928-byte** increase. Concurrent filesystem
activity prevents assigning that entire difference to this operation. Allocation,
logical archive size and observed free-space change are distinct measurements.
The verified external archive and checkpoint/training/schedule/arena evidence
remain; restoring shards later requires admission for the next consumer.

G20T05's [copy operator started](../../scratchpad/bt4_joint20/publication_20260908_v50_storage_v1/g20t05/operator.actual_start.json)
at 22:21 local time after [preparation review](../../scratchpad/bt4_joint20/publication_20260908_v50_storage_v1/g20t05/parent_preparation_review.json).
This snapshot claims **launch only**, with source retained and no completed
transfer or reclamation result. The [frozen launch bundle](../../scratchpad/bt4_joint20/publication_20260908_v50_storage_v1/g20t05-copy-launch.tar.gz)
retains the exact helper, wrapper, plan and start records: four hours inclusive,
CPUs 0,1, low priority, GPU hidden, a 24 GiB sampled staging ceiling and 150 GiB
reserve. It reuses the reviewed copy/verification path without changing loaders.

Continue eligible cold policy-corpus copies while assessing removable payload
components within already-verified archived pools, avoiding redundant transfers.
The earlier inventory's 26 still-local pools are mostly checkpoint/restart/rollback
pools, including anchor material; they are not a wholesale deletion list. Preserve
checkpoint, holdout and live dependency exclusions. Its **85.32 GB total tar size
is not a measure of recoverable allocated source space**.
This publication adds no inventory, transfer, removal or payload recheck. Exact
snapshot and member hashes are in the [publication manifest](evidence/bt4-bootstrap/v50-training-storage-progress-manifest.json).

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

At this earlier retention-review snapshot, archive copy, content verification and local removal had **not** been executed; the later G20T1 copy and reclamation are recorded above. A [six-directory metadata-size attempt](../../scratchpad/storage_scaling_20260908/publication/retention_feasibility_v1/candidate_size_inventory_attempt.json) reached its 55-second cap without totals, so aggregate savings remain unmeasured. The existing 13.59-GiB B100 measurement is not a measured size for every recipe and does not establish that the 100M pipeline fits.

The separate [sharing review](../../scratchpad/storage_scaling_20260908/publication/retention_feasibility_v1/sharing_limit.json) rules out ordinary hardlinks as a drop-in change to pinned data: creating or removing a link changes the shared inode's ctime, which existing storage receipts bind. A new family linked before qualification could be investigated, but is unqualified. Its 74.2% common-array observation is from **one B100 shard**, not a corpus-wide savings estimate; archival remains the first measure.


## Original G20T1 preparation snapshot — before launch

At the original preparation snapshot, the first concrete archive step was **PREPARED, NOT LAUNCHED**: a [frozen plan](../../scratchpad/bt4_joint20/G20T1_cold_archive_v1/plan.json) for the complete G20T1 corpus (18,910,484 rows, 2,309 shards), including top-level summaries and shard provenance. The [independent review](../../scratchpad/bt4_joint20/G20T1_cold_archive_v1/independent_review.json) passed. This is a snapshot of one operational use of the existing archive helper, not a new shared archive framework or a change to the historical recipe verdict.

The [operator command](../../scratchpad/bt4_joint20/G20T1_cold_archive_v1/operator_command.json) is for a later G10-free window, using the existing nonblocking shared preparation lock. It is not queued. The strict four-hour deadline includes prelude and cleanup; CPU affinity is 0,1 with two numeric threads, low priority and GPU hidden. The 24-GiB tar ceiling and sampled aggregate staging limit are bounds, not measured corpus size; sampling may transiently overshoot. The local reserve is 150 GiB, with additional admission headroom and destination-space checks specified in the plan. Four hours is a failure ceiling, not a completion estimate.

Execution would inventory the source progressively, verify every tar member and its contents against that inventory, compare external readback and local archive digests, and recheck source stability. **The full source remains in place even after success.** Existing attempts refuse, with no automatic resume; failures retain source and partial archive state. Only the verified local staging tar can be removed after successful external readback. ACL/xattrs are captured but not independently compared, and no restore test is claimed.

The [publication manifest](../../scratchpad/bt4_joint20/G20T1_cold_archive_v1/publication_manifest.json) binds the exact helper, wrapper, command, plan and review evidence. Six focused disposable checks passed before a separately reviewed absolute-tool-path correction; their original source/result and correction supplement are preserved. This publication ran only link, hash and hygiene checks. No dataset inventory, copy, transfer, deletion or launch has occurred in this preparation.

## 2026-09-11: legacy run04/run05 external copies verified

The bounded two-pool archive operation finished successfully in 2,778.74 seconds.
Both pools were inventoried, checked against every tar member, copied to the
external drive and verified by full archive readback. A later parent check confirmed
the external sizes and mtimes still match those receipts. Verified local staging
tars were removed; the original SSD source directories remain present. **This copy
operation reclaimed zero source bytes.** Source reclamation remains a separate
dependency and stability decision.

| Pool | Verified members | Archive bytes | External tar SHA256 |
| --- | --- | --- | --- |
| run04 | 452 | 6,547,896,320 | `52a5338cf999708e39b4721a715ef33d7a1fbb3850fe0d35f12ab1c9971ca72f` |
| run05 | 458 | 6,509,905,920 | `cfe7806d9b893b87852465a34d5fd157d4d9632a5c59aad43a33763b6d39027f` |

External location: `/mnt/e/chess_raw_archive_20260911/legacy_run04_run05_v1/`,
with `run04.tar` and `run05.tar`. Source locations are
`data/nnue_bootstrap/run04` and `data/nnue_bootstrap/run05`. The source manifests
and archive verification receipts remain under
`scratchpad/bt4_joint20/legacy_run04_run05_cold_archive_v1/staging/{run04,run05}/`.
Restore into a fresh staging directory and compare against the retained source
manifest before replacing a research input. No restore rehearsal is claimed.

The plan SHA256 is
`42671cd1650afa6fd88b7aad86971894afe63c0b1003f82f7f2e72f7fbdf15d3`;
the run04 and run05 verification receipt SHA256 values are respectively
`43f3d4caafa9027e3049d6aace859ad0e0a24a499951dc2f99a7e0a084a38924`
and `c0b50d19d773faa9f5ad96c38be143c5dcbdda4af9b600949e47538bf0f56ee7`.
File contents and mode/uid/gid/mtime were checked. ACL/xattrs were captured but
not independently compared. The live Ceres and BT4 labelers were preserved.

### Verified closed-shard reclamation

After independent review and fresh source/consumer/mount checks, the operator
removed exactly 870 archived shards explicitly listed as closed in the 18 worker
progress files: 438 from run04 and 432 from run05. Their prior allocated size was
12,935,413,760 bytes (12.05 GiB). The operation finished with exit 0 in 25.91 seconds.
Both original directories, 20 metadata/progress files and 18 unlisted tails remain.
All 38 retained files and both external archives kept their verified identities.

The parent postcheck reconciled all 870 intent/removal journal pairs, exact final
membership and unchanged retained/archive metadata. SSD free space measured
359,738,429,440 bytes afterward; concurrent generation means this is not an exclusive
filesystem-delta measurement attributable solely to reclamation. The earlier copy
operation still correctly reports zero source reclamation; this later operation
performed the removals.

Archived progress records now describe historical closed files that are no longer
local. Do not resume generation or silently repair an old derived lineage in these
directories. For historical reconstruction, restore the required archived files into
a fresh directory and verify their manifest identities before requalifying inputs.
The active G10/BT4 generators use run06/run07, and the Ceres experiments use the
original run03_s3-derived corpus; these inputs were preserved.

Local evidence: `legacy_run04_run05_cold_archive_v1/reclamation_preparation_v1/`
and `reclamation_execution_v1/` beneath `scratchpad/bt4_joint20/`. The independent
review SHA256 is
`813b01cf044820cd24f94db891d3fc1afafdfe62946a24cc204d688830cb53b7`;
the completion receipt SHA256 is
`a813f9ed47f151e876a4eb228c0032ecae846414553fe32010eccdee2eb7ddf3`.
The full source manifests and external archives remain available.

## September 12: older E0 corpus archived and reclaimed

The distinct legacy `qtemp_0.0005_hist_20m_bt4_toptie_a100` corpus completed
copy/content verification and exact local shard reclamation. This is the older
top-max-ties alpha1 recipe, not sharpened E0T05 or current SF/B100/Ceres inputs.
The archive preserves 713,331 members and 9,658,007,914 logical source bytes in an
11,016,949,760-byte tar. Local and full external readback SHA256 both equal
`88008399652e742796eca2ede6eac4e4db3c4de111529b173e0754a7713c2bae`.
The copy finished in 2,386.83 seconds (39m47s); the recorded maximum RSS was
99,844 KiB with zero swaps. The verified staging tar was removed.

A separate metadata comparison checked the saved manifest digest, all current
source stamps and exact membership in 38.76 seconds without reading payloads
again. Host consumer inspection found no source references. The inaccessible
same-user process was identified as `ssh-agent`; other disclosed processes were
WSL support services. This is practical scoped visibility, not a claim that every
process descriptor was readable. Current generation, downside and WDL batch plans
do not bind this legacy corpus.

After independent and parent review, the remover deleted exactly 2,309 shard
directories in 37.76 seconds. The durable journal has 2,309 matched intent/removal
pairs. Only the original `derive_targets_summary.json` and
`bt4_policy_mix_summary.json` remain at the source root, with unchanged hashes;
all runs, checkpoints, arena records and lineage were retained. The independently
observed external archive identity also remained unchanged. Prior allocated shard
blocks totaled **12,217,753,600 bytes**; the concurrent filesystem free-space
increase was **12,163,784,704 bytes**. These are different measurements, not an
exact attribution of every free byte. The completed operation observed
265,960,140,800 bytes free; this is a historical snapshot, not reserved capacity.

The copy retained its four-hour inclusive bound, CPUs0–1, two threads, no GPU,
2GiB per-process address-space limit, 24GiB staging ceiling, 16MiB/s transfer
pacing and 150GiB local reserve plus staging reservation. Exact reclamation used
a separate 30-minute inclusive bound, 512MiB address space, the same two CPUs,
shared nonblocking preparation lock, STOP and fd-based deletion with a durable
per-shard journal. No active input or preserved interrupted output was removed.

Restore location:
`/mnt/e/chess_derived_archive_20260908/E0Legacy_v1/qtemp_0.0005_hist_20m_bt4_toptie_a100.tar`.
The source manifest remains at
`scratchpad/bt4_joint20/E0Legacy_cold_archive_v1/staging/qtemp_0.0005_hist_20m_bt4_toptie_a100/source.jsonl`,
SHA256 `4331ace78c550937a6a77fe70360ba6030cbf447cbe7528eebc3fb5dc2dc0305`.
Restore into a fresh directory and verify the archived identities before renewed
source admission. Retained summaries alone are insufficient for training.
[Compact completed receipts and review pins](evidence/e0legacy-archive-reclaimed-20260912.json)
record both operations; this storage recovery establishes no new playing result
or proof that the full 100M working set fits.


## September 13: Tactical100 cold copy launched

The parent launched the completed, unselected Tactical100 policy corpus copy at
**00:56:00 UTC**, with one absolute **04:56:00 UTC** deadline. This continues
SSD offloading during training. Tactical100's completed registered match was
unresolved; B100 remained the incumbent. Archiving its derived corpus does not
change that scientific interpretation or remove its trained checkpoints.

Source:
`data/nnue_derived/armB/qtemp_0.0005_hist_20m_bt4_global_B100T05_tactical100`.
The intended archive is
`/mnt/e/chess_derived_archive_20260908/Tactical100_v1/qtemp_0.0005_hist_20m_bt4_global_B100T05_tactical100.tar`.
A single bounded metadata census found **2,309 shards**, **671,768 files** and
**41,563 directories**, with no hardlinked regular files. Logical file size is
**11,896,485,651 bytes**; allocated blocks sum to **14,447,185,920 bytes**.
Neither number is an observed archive size or a promise of space reclaimed.
The census took about 80 seconds and did not read shard contents.

The reviewed E0Legacy helper/wrapper are reused with literal path bindings and
one resource change: the available-memory floor rises from 16 to **48 GiB**.
This leaves margin above training's 32 GiB guard, without guaranteeing reaction
ordering on sudden allocation. The copy keeps its four-hour inclusive bound,
CPUs 0–1, idle I/O and low CPU priority, two numeric threads, hidden GPU,
2 GiB per-process address-space limit, 24 GiB staging ceiling, 16 MiB/s transfer
pacing, 150 GiB SSD reserve plus staging reservation and 48 GiB external reserve.
The parent observed approximately 77 GiB available RAM, zero swap use, 210 GiB
SSD free and 7,514 GiB external free on the writable E: 9p mount before launch.
These are startup samples, not aggregate memory or disk guarantees.

The actual command uses plan `a5c822c6…` and preserves the shared nonblocking
preparation lock, STOP markers and one absolute deadline. Parent root session
**71709**, sole observer **233**, owns completion. All source files remain until
copy/content verification, full external readback and a later separately reviewed
reclamation step. Fresh consumer and source-stability checks precede any removal.
All runs, checkpoints, optimizer state and receipts remain; the current Downside,
original SF/B100/C and active G10 inputs are outside this operation.

**At the launch snapshot, copy completion was unread and source bytes reclaimed were 0.** The
[compact actual-launch evidence](evidence/tactical100-archive-launched-20260913.json)
contains exact command, source/destination and review pins. Preparation's initial
sandbox mount check failed before the census because its view was read-only;
the retained host attempt confirmed writable E: and completed the single census.
This publication read only small saved preparation/launch records; it did not
poll the archive or repeat the inventory.


## September 13: Tactical100 archive verified and exact shards reclaimed

The copy finished successfully in **2,879.775 seconds (47m59.775s)**. Parent root
71709 / sole observer 233 closed with exit 0. The **13,254,707,200-byte** archive
passed complete member/content verification and external readback against SHA-256
`b1765a5674d1dd561718c09be34e7b940aad6d92952659f82e6bb8f98fbbe96e`.
Its **713,331 members** preserve the original corpus files and directories;
source manifest SHA-256 is
`f3156922c3b8dedcb437f2dd752b761836e22eeba62aac6ae1965f4a22753fd0`.
Only the verified local staging tar was removed at copy completion.

A separate metadata comparison then checked exact membership and all recorded
source stamps without rereading payloads. It confirmed **2,309 shard directories**
with **14,384,427,008 allocated bytes**, excluding the two summaries retained
locally. Fresh scoped observation found no exact-source references among 52
same-user processes; permission-limited PID 30 was identified as ssh-agent and
seven other-user processes as WSL roles. Current Downside training and SF recovery
plans excluded the candidate. These are scoped observations, not a claim of full
process visibility or absence of future dependencies.

After independent review and parent authorization, the unchanged removal method
with exact Tactical100 bindings finished in **49.168 seconds**, ending at
**2026-09-13T02:08:00.076416+00:00**. Parent root **17721 / sole observer 243**
closed with exit 0. Its durable journal contains exactly **2,309 removal intents
and 2,309 successful removals**. Final top-level inspection found only
`derive_targets_summary.json` and `bt4_sf_tactical_policy_summary.json`, with
unchanged recorded metadata; the external archive identity also remained unchanged.
All runs, checkpoints, optimizer state and receipts remain. Original SF/B100/C,
current Downside and active G10 inputs were outside the removal scope.

Filesystem free space changed from **215,922,315,264** to **230,295,519,232 bytes**,
an observed increase of **14,373,203,968 bytes**. This is a concurrent filesystem
measurement, not isolated attribution; the prior allocated shard sum is a
separate measurement. The removal retained the 1,800-second inclusive bound,
512 MiB address-space cap, CPUs 0–1, owned lock, STOP handling and partial-failure
journal. No automatic retry, new archive or additional reclamation is launched.

Restore from the exact external archive:
`/mnt/e/chess_derived_archive_20260908/Tactical100_v1/qtemp_0.0005_hist_20m_bt4_global_B100T05_tactical100.tar`.
Before restoration, verify its full SHA-256 above and extract with GNU tar into a
fresh empty directory with sufficient space. The archive contains the original
corpus directory name. Do not overwrite the retained source path or an active
experiment; validate restored contents and the intended consumer before reuse.
ACL/xattrs were captured but not independently compared. Later metadata stability
is not a new content-integrity proof against undetected storage corruption.

[Compact completed copy and reclamation evidence](evidence/tactical100-archive-reclaimed-20260913.json)
pins the copy, metadata eligibility, exact removal list, independent reviews,
completion and journal. This publication reviewed those saved records, a single
final directory listing and archive/summary metadata only; it repeated no payload
hashes, source census, model reads or active-job polling. Scientific results and
the separately running Downside training are unchanged.
