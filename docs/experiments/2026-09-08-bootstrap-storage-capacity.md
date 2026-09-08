# Storage capacity changes the 100M bootstrap plan

The September 8 read-only inventory does **not** support retaining 100 million G10 raw rows, their common-input data and multiple full recipe copies together on the current local disk. The archive has ample capacity; the local working set is the constraint. A larger experiment needs explicit staging and recipe retention, or suitable added working capacity. Neither design is qualified yet.

The [capacity evidence](../../scratchpad/storage_scaling_20260908/publication/capacity.json) records 397.49 GB free on local ext4, leaving **236.43 GB above the existing 150-GiB reserve**. The archive mount is a 10.0007-TB volume with **8.1855 TB free**, exposed through 9p. GB/TB here are decimal; GiB is binary. Free space is an observation, not an allocation reserved for this experiment.

## What the measured sizes imply

[Complete worker-progress snapshots](../../scratchpad/storage_scaling_20260908/publication/raw_density.json) cover 36,025,871 G10 raw rows: 28,526,032 from run06 and 7,499,839 from run07. Their closed compressed files occupy 92.05 GB logically, approximately 2,555 bytes per raw row. Open tails and subsequent generation are excluded; no raw payload was read.

| Component | Observed basis | Projection or consequence |
| --- | --- | --- |
| Additional raw data to reach 100M G10 rows | Current closed-file compressed density | About 163.47 GB more; leaves 72.96 GB above the reserve |
| Common-input working output | Completed 1,057,326-row batch, sampled peak 1.125 GB logical output | About 106.41 GB for 100M common rows |
| One separate final recipe copy | Completed original B100: 14,593,728,512 allocated bytes for 18,910,484 rows | About 77.17 GB allocated per 100M rows |

These are size proxies from different workloads, not interchangeable measurements or guaranteed future costs. The common-bundle projection is **gross**, including a subset already prepared locally; it is not all additional storage. Logical bytes understate allocated blocks, while the recipe estimate uses actual allocation. Different histories and policy entropy can change compression. Raw teacher sidecars, additional value banks, checkpoints, temporary copies and continued generation consume further space and are omitted from this comparison. One hundred million raw rows also does not mean exactly one hundred million eligible training rows.

The [G10 record](2026-09-07-g10-transfer-readiness.md) reports 5,276,543 common rows already prepared, with combined training qualification still separate. The existing subset does not close the much larger gap created by new raw data and multiple ordinary recipe copies. Smaller, explicitly budgeted increments remain plausible; blind 100M multi-recipe coexistence does not fit the current reserve policy.

## Archive evidence and the next storage decision

The [38-receipt refresh](../../scratchpad/storage_scaling_20260908/publication/archive_refresh.json) matches every previously recorded receipt hash and every tar's size and mtime. The tars total 85.32 GB. This reuses the [earlier full-content verification](2026-09-07-storage-archive.md); it is **not** a new payload checksum or restore test. Twenty-six archived source pools remain local, with prior dependency exclusions still relevant. No leftover staged tar duplicates were found in the two known staging locations.

This record recommends no deletion of historical checkpoint or holdout candidates and no relocation of identity-pinned inputs. The small candidates inspected would not materially solve the scale constraint. Current raw sources, derived corpora, policy/value sidecars, checkpoints, openings and frozen runtimes remain protected. Archive free space does not qualify direct training over 9p, and moving existing inputs can invalidate their path and storage-identity receipts.

Before committing the larger scale experiment, account for the actual retained raw, common and recipe working set, including temporary publication space and concurrent growth. Then qualify a staged-data/recipe-retention approach or suitable additional working storage. This is a change to the scale plan, not an approved storage implementation. The inventory performed no move, deletion, payload hashing, full-disk recursive scan or I/O benchmark. The [publication manifest](../../scratchpad/storage_scaling_20260908/publication/manifest.json) binds the compact projections and original evidence identities.
