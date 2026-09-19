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
