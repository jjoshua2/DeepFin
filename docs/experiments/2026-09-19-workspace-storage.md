# Workspace storage migration — 2026-09-19

Keep workspaces freely copyable by externalizing bulk data. This is an operational
storage change; it does not alter bootstrap training recipes. Migration is partial:
active dataset roots stay fixed until the running epoch completes.


- `data/desync_quarantine_20260801` moved to
  `/home/josh/chess-artifacts/corpora/desync_quarantine_20260801` (499 MiB).
  It is inactive quarantine data; no compatibility link was needed.
- `scratchpad/preserved_corpora_20260802` moved to
  `/home/josh/chess-artifacts/corpora/preserved_corpora_20260802` (65 GiB).
  Its old location is an absolute directory link. Active/queued references and
  open handles were checked before relocation.

Both were same-filesystem renames preserving the original directory inode, without
copying or deleting payloads. This reduces what workspace creation can duplicate;
it does not itself reclaim disk space. The complete relocation journal is
`/home/josh/chess-artifacts/operations/relocations-20260919.jsonl`.

A disposable model-free test of the installed Grok workspace creation endpoint
confirmed that its copy strategy preserves a nonignored absolute directory symlink
to external data. That evidence applies to this copier/version; recheck before
using a different copy implementation or options that dereference links.

Legacy model stores `data/best_regret_checkpoints`, `data/salvage_pre_v2layer`,
`data/salvage_ba920_iter475` and `data/salvage` now link to the corresponding
`/home/josh/chess-artifacts/models/` directories. `data/salvage/rolling` separately
links to `models/salvage_rolling`. Internal links were checked before migration.

The future factorial target `factorial58_20260919/outputs` parent now links to
`/home/josh/chess-artifacts/labels/factorial58_20260919/outputs`. A real small
producer fixture validated creation, resume checks, qualification and loader access
through this link. The destination remains on the same filesystem, preserving
the existing disk-space guard's applicability. Frozen plans were not edited.

The 35 active training input roots must stay at their current canonical paths
until the running epoch completes. The sampler hashes resolved shard paths into
the schedule and checks them again while loading; a mid-epoch relocation could
fail the final schedule check despite identical data. Likewise, future training
output leaves cannot simply be precreated as links: the runner requires a fresh
nonexistent output. Those output paths need explicit plan and queue repinning.

Nine inactive clean review worktrees were retired after rechecking live references
and dirty files. Their commits remain under `refs/archive/workspace-cleanup-20260919/`;
`operations/retired-workspaces-20260919.jsonl` in the shared artifact root records
each former path and retained commit. Dirty and unpublished worktrees remain intact.

All 14 still-empty Ceres collection bank parents now link to
`/home/josh/chess-artifacts/labels/factorial58_20260919/ceres_banks/<cohort>`.
Adoption rechecked empty directories, unchanged driver-plan hashes, same filesystem
and all 14 queued statuses under the scheduler lock. The actual collector/cache and
driver completion paths passed a small fixture beforehand; no plan hashes changed.
See `operations/factorial58-storage-audit-20260919/` for fixture evidence and
`operations/factorial58-ceres-bank-storage.jsonl` for adoption receipts.

The four future factorial training outputs have now been explicitly rebound to
`/home/josh/chess-artifacts/runs/factorial58_20260919_<arm>`. Their four matches
reference those external checkpoints. The independent review verified all 30 file
changes and all eight queued records; only output paths and dependent hashes changed.
Adoption rechecked every hash/status under the scheduler lock, validated all eight
actual operator descriptors, and preserved the active job.

[Published migration evidence](evidence/workspace-storage-20260919/)
contains the exact diff, before/after hashes, independent reviews, fixtures and
adoption receipts. This amends the host storage paths of PR #786, not its experiment
recipe, training horizon or evaluation settings.
