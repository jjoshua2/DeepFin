# Storage archive and qualified local retirement

The September 6–7 storage operation completed **38 queued salvage archives,
85,320,079,360 bytes (79.4605 GiB)**, at
`/mnt/e/chess_archive_20260906/salvage/`. Every completed archive has a full-content
verification receipt, including external SHA256 readback. This count excludes the
separate manual Zarr pilot and the DRIFT directory-copy pilot. Copy completion did
not authorize removing every original: only separately qualified local retirements
were performed.

The [completion snapshot](../../scratchpad/storage_audit_20260906/archive_completion.json)
and [transfer record](../../scratchpad/storage_audit_20260906/transfer_status.json)
record the final queue state. The [publication manifest](../../scratchpad/storage_audit_20260906/publication_manifest.json)
links 69 byte-identical compact artifacts with their original paths, sizes and
SHA256 values. They include all 38 successful archive receipts, completed cleanup
receipts, independent reviews, dependency preservation, and the external drive's
[original restore instructions](../../scratchpad/storage_audit_20260906/publication_originals/RESTORE.md).
These are dated provenance, not live process status or runnable continuation plans.
The publication copied compact evidence; it did not repeat the tar payload readback.

## Verification and archive layout

Uncompressed tar files avoid transferring millions of individual Zarr files over
the Windows E drive's 9p mount. Each tar includes its named pool root, checkpoint,
replay files and recovery metadata. The archive driver streamed source/member
SHA256 inventories, independently compared tar contents and ordinary metadata,
checked source stability, transferred with resumable rsync, then hashed the full
external tar. Only its verified local staging tar was removed by that driver.
External successful receipts are mirrored under
`/mnt/e/chess_archive_20260906/receipts/POOL/verified.json`.

The original driver refused hardlinked source files. Those failed attempts were
preserved. Three separately reviewed archives—`recover_ckpt751_20260711`,
`scaleup_512x16_window_20260707`, and `swap_512x16_20260711`—subsequently stored each
regular path independently. Their contents and ordinary metadata were verified,
but restoration does **not** recreate inode sharing. All three originals remain
local. See their [review](../../scratchpad/storage_audit_20260906/hardlink_copy_v1/independent_review.json),
[qualification](../../scratchpad/storage_audit_20260906/hardlink_copy_v1/qualification.json),
and successful receipts listed in the publication manifest.

GNU tar captured ACLs and xattrs; those attributes were not independently compared.
The older `/mnt/e/chess_backup_20260827` directory backup is separate and must not be
assumed complete. The explicitly verified DRIFT pilot is the documented exception.

## What was removed and what stays local

| Operation | Completed evidence and interpretation |
| --- | --- |
| DRIFT pilot | [`DRIFT_20260728_postC17_iter192`](../../scratchpad/storage_audit_20260906/pilot_retired.json) was verified against its directory copy under `/mnt/e/chess_backup_20260827/salvage/`; its original path now links to that external copy. |
| Manual Zarr pilot | [`pre_audit_deploy_20260726`](../../scratchpad/storage_audit_20260906/zarr_pilot/cleanup_completed.json) was separately verified and removed; its tar is under `chess_archive_20260906/manual_pilot/`. |
| Three checkpoint pools | [`apf_endpoint_checkpoint_000297_20260818`](../../scratchpad/storage_audit_20260906/small_checkpoint_cleanup_v1/apf_endpoint_checkpoint_000297_20260818.completed.json), [`f_only_readout_iter487_20260819`](../../scratchpad/storage_audit_20260906/small_checkpoint_cleanup_v1/f_only_readout_iter487_20260819.completed.json), and [`iter118_pre_value_easy_drop`](../../scratchpad/storage_audit_20260906/small_checkpoint_cleanup_v1/iter118_pre_value_easy_drop.completed.json) were separately qualified and removed. |
| Nine large pools | The [fixed allowlist](../../scratchpad/storage_audit_20260906/large_salvage_cleanup_v1/plan.json), [independent review](../../scratchpad/storage_audit_20260906/large_salvage_cleanup_v1/independent_review.json), and [parent completion observation](../../scratchpad/storage_audit_20260906/large_salvage_cleanup_v1/parent_completion_observation.json) cover all nine completed removals. Their per-pool completion receipts are in the publication manifest. |
| Superseded derived partial | [`qtemp_0.0005_hist_20m_bt4_global_G20T1.interrupted_reboot_20260905`](../../scratchpad/storage_audit_20260906/abandoned_G20T1_cleanup_v1/completed.json) was discarded after separate dependency and exact layout qualification. Metadata and lineage survive; **there is no payload archive of this partial**. Its complete replacement and original inputs remain local. |

The nine large pools were `bt4heads_iter112_20260817`, `f_only_midpoint_20260818`,
`post_quarantine_20260801`, `pre_abandonfix_20260724`, `pre_aot_deploy_20260714`,
`pre_audit_window_20260726`, `pre_durable_deploy_20260726`,
`pre_lc0_control_20260819`, and `pre_mainmerge_20260810`. Historical ledger paths into
these pools now require restoration; the ledger itself remains frozen evidence.

This batch qualified 3,412,711 entries using a disk-backed unique-path index rather
than assuming traversal order. It checked exact historical metadata and membership,
regular-file link counts, source stability, and a fresh external archive SHA256 at
16 MiB/s. Source payload hashes were not repeated: the contract relied on historical
content verification plus unchanged inode/ctime/mtime/size and other metadata.
This does not independently detect local storage corruption without a metadata
change. Process, config and shallow-alias checks were refreshed before removal;
eleven explicitly recorded inaccessible process identities remained uncertified.
No claim was made about arbitrary external aliases or future consumers.

Each owned quarantine was removed only after a durable external restore receipt
and a local `.ARCHIVED.json` pointer existed. A pre-removal proof is not completion;
separate completion receipts establish the latter. The batch's measured peak RSS
was 32,660 KiB. Its gross removed allocation was **32.5340 GiB**. Earlier operations
reported **18.3167 GiB of observed free-space change**, which includes concurrent
work. These are different measurements and must not be added into an exact net
reclaimed-space claim.

The configured `bt4heads_iter100_20260815` restart seed, mutable `rolling`, ANCHOR,
and the linked `armB_launch_base_20260814` / `pre_policy_adapter_20260812` pair remain
local. The [dependency record](../../scratchpad/storage_audit_20260906/linked_pool_dependencies.json)
shows that `armB_launch_base` points into `pre_policy_adapter` replay shards.
`pre_pairing_20260819` was also retained because production YAML names it as a
historical revert point. Other copied originals have not been implicitly cleared
for deletion. Training, generation, completed models and original source corpora
were preserved.

## Restoration

1. Verify that `/mnt/e` is the intended E drive and obtain the successful receipt
   for the required pool. Check the tar's complete `sha256sum` against its recorded
   `external_tar_sha256`/`tar_sha256` before extracting; launch/preparation receipts
   are insufficient.
2. Provide sufficient free space and a new empty restore directory. With GNU tar:

   ```bash
   mkdir /path/to/new-restore
   tar --acls --xattrs -xf /mnt/e/chess_archive_20260906/salvage/POOL.tar \
     -C /path/to/new-restore
   ```

   The archive creates `POOL/` beneath that directory. The manual pilot uses its
   separate `manual_pilot/` tar path. DRIFT uses the verified directory copy described
   above. The discarded derived partial cannot be restored from metadata alone.
3. Validate the restored pool manifest, checkpoint and intended configuration
   before an RL restart. Resolve linked-pool dependencies explicitly. Never extract
   over an existing pool or a running experiment, and do not assume hardlink-aware
   archives restore inode sharing.

This record publishes the completed storage operation. It does not request another
transfer, cleanup, restoration, or training restart.
