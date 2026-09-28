# Workspace storage migration — 2026-09-19

Keep workspaces freely copyable by externalizing bulk data. This is an operational
storage change; it does not alter bootstrap training recipes. Migration is partial:
the 35 pinned dataset roots remain at their existing paths pending qualified rebinding.


- `data/desync_quarantine_20260801` moved to
  `~/chess-artifacts/corpora/desync_quarantine_20260801` (499 MiB).
  It is inactive quarantine data; no compatibility link was needed.
- `scratchpad/preserved_corpora_20260802` moved to
  `~/chess-artifacts/corpora/preserved_corpora_20260802` (65 GiB).
  Its old location is an absolute directory link. Active/queued references and
  open handles were checked before relocation.

Both were same-filesystem renames preserving the original directory inode, without
copying or deleting payloads. This reduces what workspace creation can duplicate;
it does not itself reclaim disk space. The complete relocation journal is
`~/chess-artifacts/operations/relocations-20260919.jsonl`.

A disposable model-free test of the installed Grok workspace creation endpoint
confirmed that its copy strategy preserves a nonignored absolute directory symlink
to external data. That evidence applies to this copier/version; recheck before
using a different copy implementation or options that dereference links.

Legacy model stores `data/best_regret_checkpoints`, `data/salvage_pre_v2layer`,
`data/salvage_ba920_iter475` and `data/salvage` now link to the corresponding
`~/chess-artifacts/models/` directories. `data/salvage/rolling` separately
links to `models/salvage_rolling`. Internal links were checked before migration.

The future factorial target `factorial58_20260919/outputs` parent now links to
`~/chess-artifacts/labels/factorial58_20260919/outputs`. A real small
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
each former path and retained commit. Dirty worktrees remain intact. Retired heads, including unpublished commits, are retained by archive refs.

All 14 still-empty Ceres collection bank parents now link to
`~/chess-artifacts/labels/factorial58_20260919/ceres_banks/<cohort>`.
Adoption rechecked empty directories, unchanged driver-plan hashes, same filesystem
and all 14 queued statuses under the scheduler lock. The actual collector/cache and
driver completion paths passed a small fixture beforehand; no plan hashes changed.
See `operations/factorial58-storage-audit-20260919/` for fixture evidence and
`operations/factorial58-ceres-bank-storage.jsonl` for adoption receipts.

The four future factorial training outputs have now been explicitly rebound to
`~/chess-artifacts/runs/factorial58_20260919_<arm>`. Their four matches
reference those external checkpoints. The independent review verified all 30 file
changes and all eight queued records; only output paths and dependent hashes changed.
Adoption rechecked every hash/status under the scheduler lock, validated all eight
actual operator descriptors, and preserved the active job.

[Published migration evidence](evidence/workspace-storage-20260919/)
contains the exact diff, before/after hashes, independent reviews, fixtures and
adoption receipts. This amends the host storage paths of PR #786, not its experiment
recipe, training horizon or evaluation settings.

The four Syzygy stores now live under `chess-artifacts/tablebases/`, still on local
NVMe, with their former paths retained as absolute links. The two large stores
contain 150.76 GiB and 68.75 GiB. The combined relative alias remains valid. Post-move
Python-chess probes through the old and external 3-man paths returned identical
table counts, WDL and DTZ. This moves payloads out of the workspace without changing
their storage device or duplicating them.

The two bounded duplicate-only cleanup passes completed: 1,328 files / 75.41 GiB
and 9,050,614 files / 364.33 GiB, respectively. Each removed payload was verified
against a retained canonical copy. About 439.74 GiB of duplicate payload was removed;
this is separate from same-filesystem relocations, which do not reclaim space.
The second pass journal is `operations/workspace-duplicate-payloads-20260919.jsonl`.
These were scoped passes, not a claim that every workspace is now data-free.

The process-exit trigger attempted the CPU-only SF generation throughput screen,
but its readout failed because a foreign installed `scripts` package shadowed the
repository package. The failed run remains preserved. A corrected v2 runtime and
plan are staged and independently reviewed; no completed benchmark is claimed.
See `operations/sf_trigger_20260919/` and
`operations/sf_generation_throughput_v2_20260919/` for the receipts.

## Sibling and nested workspace retirement

There were 167 `~/projects/chess-*` sibling folders: 161 registered worktrees and
six other folders, including two containers of nested worktrees. We retired 132
clean direct sibling worktrees and 34 clean nested worktrees. The top-level count
is now 35. Across all locations, 550 registered worktrees remain at this snapshot;
this cleanup does not establish that the global workspace inventory is clean.

Before each retirement, the operation rechecked HEAD, full tracked/untracked status,
hidden index flags, Git metadata backreferences, ignored artifacts and live process
references. It held the scheduler lock and required the queue hash to match the
independent dependency audit. Removal used normal `git worktree remove`, without
force. All 166 removals were subsequently checked against their retained refs.

Branches were retained. Each head additionally lives under
`refs/archive/workspace-cleanup-20260919/`. Ignored local outputs were moved to
`~/chess-artifacts/retired-workspaces/<name>/local-artifacts/`, with a
`retirement.json` recording provenance. Nested names include the former parent to
avoid collisions. These artifact moves preserve data and do not themselves free space.
The compact per-worktree receipts are published alongside this record:

- [132 sibling retirements](evidence/workspace-storage-20260919/retired-siblings.jsonl)
- [34 nested retirements](evidence/workspace-storage-20260919/retired-nested.jsonl)

To restore source, create a worktree from the receipt's retained ref into a new
path. Consult its archive before restoring any local output. Dirty checkouts,
three differing non-Git snapshots, the ambiguous `chess-instr-436` /
`chess-mergetest-436` metadata pair, and historically referenced `chess-armf` /
`chess-sfgate` remain preserved. `chess-wt/eval-race` retains its 18 staged changes.

Independent review passed the retirement safeguards and the nested-scope adaptation.
Future manual worktrees use `~/projects/chess-worktrees/<task>`; independent writers
still receive separate worktrees. Workspace copying and Grok delegation remain enabled.
