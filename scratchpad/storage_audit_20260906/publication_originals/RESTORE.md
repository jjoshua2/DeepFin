# Salvage archives, September 6, 2026

Each uncompressed tar contains one complete named salvage snapshot, including its checkpoint, replay files and recovery metadata. The old incomplete August 27 directory backup is separate and must not be assumed complete.

Verification receipts for completed queued archives are mirrored here under `receipts/SNAPSHOT/verified.json`; source inventories and operation records remain at `/home/josh/projects/chess/scratchpad/storage_audit_20260906/`. A successful receipt records the full tar SHA-256 after external readback. The manual pilot has a receipt alongside its tar. Compare `sha256sum` of the archive with the receipt before restoration.

Restore into an empty staging directory, using GNU tar, for example:

```bash
mkdir -p /path/to/empty-restore
tar --acls --xattrs -xf /mnt/e/chess_archive_20260906/salvage/SNAPSHOT.tar -C /path/to/empty-restore
```

The archive creates its named SNAPSHOT directory beneath that location. Do not extract over a running experiment or an existing salvage pool. Validate the restored manifest/checkpoint and its intended configuration before using it for an RL restart.

The copy queue itself retains originals. Separate verified cleanup has removed some archived local pools; completed cleanup leaves a `SNAPSHOT.ARCHIVED.json` pointer beside the former source. The configured `bt4heads_iter100_20260815` restart pool, mutable `rolling` pool, ANCHOR, and linked `armB_launch_base_20260814` / `pre_policy_adapter_20260812` dependency stay local. GNU tar captured ACLs and xattrs, but those were not independently compared during verification.

`derived_evidence/G20T1_interrupted_20260905/` contains metadata and lineage for a discarded, superseded partial corpus. It is **not** a payload backup and cannot restore that partial corpus. Its complete replacement corpus and raw inputs remain local.

Hardlink-aware archives for recover_ckpt751_20260711, scaleup_512x16_window_20260707 and swap_512x16_20260711 store each regular path independently. Restoring preserves verified contents and ordinary metadata but does not recreate inode sharing. A completed verification receipt is required; preparation or launch records are not proof of a finished archive. Their operation evidence is under `operations/hardlink_copy_v1/`; all three original sources remain local.
