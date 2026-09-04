---
name: deepfin-grok-review
description: Run this repository's portable Grok CLI wrapper for an independently authored review of a PR, commit range or uncommitted worktree. Use when Grok or another model-family review is requested.
---

# Grok review

Use `scripts/grok_review.sh` from this Skill's repository root. This repository-local
entry point is named separately from any user-installed `grok-review` Skill; it needs
no files under another developer's home directory.

```bash
scripts/grok_review.sh --pr <number> [--focus "..."]
scripts/grok_review.sh --diff <base>...<head> [--focus "..."]
scripts/grok_review.sh --worktree <path> [--focus "..."]
```

The wrapper needs Bash, Git, Unix archive tools, GNU `timeout`, and an
authenticated `grok` CLI. PR mode also uses authenticated `gh`. `--repo <path>` selects
the source repository for PR/range modes; worktree mode uses the supplied worktree.
The model and effort inherit Grok's own configuration. An explicit request to run this
review authorizes the invocation; do not introduce another approval gate.

Grok runs against a disposable snapshot. CLI permission flags alone have not reliably
prevented edits, so never replace the snapshot with direct operation in the live tree.
The snapshot is working-copy isolation, not an OS security sandbox. Worktree mode
includes tracked edits and non-ignored untracked files; inspect scope before using it
on an artifact-heavy checkout. Keep credentials and run outputs ignored.

The default time limit is 30 minutes; `GROK_REVIEW_TIMEOUT_SECONDS` can set the agreed
budget, with a five-second termination grace for the process group. Start it as an ongoing process and continue independent work. Each invocation
preserves the prompt and complete CLI output under a private directory in
`${GROK_REVIEW_OUT:-${TMPDIR:-/tmp}/grok_review}`. Read the printed `raw_output` file.

A nonzero exit means no valid review verdict: unavailable CLI, failed snapshot,
changed source, timeout or malformed output. A valid `NONE` findings block is an
explicit no-finding result. Never substitute another model under Grok's name.
Verify findings against the actual code before changing it or posting a reviewed
summary. Report the raw-output path, verdict and verification; source changes detected
during a run are not automatically attributable to Grok or safe to revert.
