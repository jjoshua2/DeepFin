# Workspace storage

Development checkouts contain code, documentation and small test fixtures. Bulk
artifacts belong in shared storage, addressed by explicit absolute paths in job
manifests. `.gitignore` prevents commits; it does not prevent a tool from copying
ignored files.

## Creating workspaces

Use `git worktree add` from the intended revision. For review snapshots, use
`git archive` plus the specific source changes being reviewed. Do not use recursive
checkout copies, including tools that fall back to copying when Git setup fails.
If isolation cannot be created without copying runtime data, fail and fix the setup.

The repository Grok implementation wrapper creates a Git worktree and disables
nested subagents. The review wrapper starts from a Git archive and includes
nonignored untracked changes for worktree reviews; inspect that file set before
launch. Direct interactive Grok sessions have their own workspace implementation
and require separate configuration.

The host now sets the documented `~/.grok/config.toml` option:

```toml
[subagents]
enabled = false
```

This disables automatic Grok child sessions by default; explicit Grok implementer
and reviewer jobs in code-only workspaces remain available. Do not override it
with `--subagents` or `GROK_SUBAGENTS=1` until child workspace creation is verified
to exclude runtime data. Already-running sessions may retain their loaded settings.
The original host config is preserved as `config.toml.before-code-only-20260919`.

## Runtime artifacts

Give corpus, teacher labels, checkpoints, caches and run output explicit paths
outside the development checkout. Keep artifact identities and compact experiment
receipts in Git; reference the bulk data by path and content identity. Point tools
to shared inputs rather than copying them into each worktree. Keep output directories
separate by run so multiple workspaces cannot overwrite each other's results.

The pinned live checkout currently owns legacy `scratchpad/` and `runs/` paths.
These are existing shared artifact locations, not templates to copy. Moving them
requires checking active and queued job manifests, overlay dependencies and recovery
paths. Do not relocate them underneath a running experiment just to satisfy the
new layout. New standalone artifact stores should live outside all code checkouts.

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
