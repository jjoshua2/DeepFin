# Workspace storage

Development checkouts contain code, documentation and small test fixtures. Bulk
artifacts belong in shared storage, addressed by explicit absolute paths in job
manifests. `.gitignore` prevents commits; it does not prevent a tool from copying
ignored files.

## Creating workspaces

Workspace copying, Git worktrees and tracked-file review snapshots are all valid.
Keep bulk data outside the source workspace so creating another workspace cannot
duplicate it. Disabling agent delegation is not a storage-layout solution.

The temporary September 19 Grok subagent restriction was reverted to the exact
previous host configuration. Automatic workspace creation remains available.

## Runtime artifacts

The host shared artifact root is `/home/josh/chess-artifacts/`, with `corpora/`,
`labels/`, `models/`, `runs/`, `cache/` and `operations/` subdirectories.

Give corpus, teacher labels, checkpoints, caches and run output explicit paths
outside the development checkout. Keep artifact identities and compact experiment
receipts in Git; reference the bulk data by path and content identity. Point tools
to shared inputs rather than copying them into each worktree. Keep output directories
separate by run so multiple workspaces cannot overwrite each other's results.

The pinned live checkout currently owns legacy `scratchpad/` and `runs/` paths.
These are existing shared artifact locations, not templates to copy. Moving them
requires checking active and queued job manifests, overlay dependencies and recovery
paths. Do not relocate them underneath a running experiment just to satisfy the
new layout. Migrate inactive artifacts first. Any temporary compatibility links
must be checked against the actual workspace copier: a copier that follows them
would reproduce the same duplication. New jobs should reference the external paths
directly.

## Existing workspace cleanup

Inventory running processes, open files and queued runtime/source dependencies
before retiring a workspace. Preserve dirty source and unpublished commits. For
redundant bulk files, verify the retained copy's content and stability, then remove
only the verified duplicates and record their identities. Size and modification
time alone do not establish identical content. Preserve unmatched files for review.

The September 19 incident involved an interactive Grok workspace recursively copying
experiment payloads. Its process recorded roughly 1.19 TB of cumulative writes;
this is process I/O, not a measurement of unique duplicated data. The cleanup receipt
is `scratchpad/bt4_joint20/efficiency500m_20260919/workspace_duplicate_cleanup.jsonl`
on the pinned host. Active training and frozen queued runtimes remain in place.

## Migration receipt, 2026-09-19

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

[Published migration evidence](experiments/evidence/workspace-storage-20260919/)
contains the exact diff, before/after hashes, independent reviews, fixtures and
adoption receipts. This amends the host storage paths of PR #786, not its experiment
recipe, training horizon or evaluation settings.
