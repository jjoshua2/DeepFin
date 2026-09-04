---
name: grok-review
description: Get an independent code review from Grok (a different model family) on a PR, a diff range, or a worktree's uncommitted work. Returns Grok's complete raw output with no Claude in the path. Use alongside a Claude-family review on anything that matters — the two lanes find largely different defects.
---

# Grok review — no Claude in the path

Run the script. Read the output. That is the whole skill.

```bash
scripts/grok_review.sh --pr <N>              [--focus "..."]
scripts/grok_review.sh --diff <base>...<head> [--focus "..."]
scripts/grok_review.sh --worktree <path>      [--focus "..."]   # includes uncommitted work
```

It takes **10–20 minutes**. Launch it with `run_in_background: true` and keep working;
you will be notified when it exits. Then read the printed `raw_output:` path.

## Why this is a script and not a subagent

It used to be a `grok-reviewer` subagent. Measuring 36 of its runs (2026-08-26) found two
failures that are *impossible* here rather than merely forbidden:

- It rendered Grok's findings as its own triage list — you got the wrapper's summary of
  Grok, not Grok. A pipe cannot summarise.
- On PR #453 it reviewed the diff **itself** and returned zero Grok output under a
  `GROK REVIEW` header, so the caller believed it had a second model family when it had a
  second Claude. A shell script cannot decide to write the review itself.

It is also ~25k output tokens and ~6.4M cache-reads cheaper per review, and most of the
old ~18 min median was wrapper turns rather than Grok.

## Reading the result

**Findings are LEADS, not verdicts. Verify each one before acting.** Measured on this
repo: the outside lane contributes ~27% of all distinct defects found — real ones the
Claude lane missed entirely — at a rejection rate around 6%. Both halves of that matter:
do not skip the lane, and do not merge on its say-so.

When a finding turns out to be wrong, **say so and record it as a false positive**. Do not
"fix it defensively" — a defensive fix makes the error invisible and quietly corrupts any
later measurement of how good this lane is.

## Non-negotiables

- **If the script exits non-zero, the review FAILED. Report that, with zero findings.**
  Never write a review yourself and present it as this lane's output — that silently
  collapses two model families into one and destroys the only thing the lane is for.
- The script snapshots to a temp dir and runs Grok against the copy. Grok is **not**
  read-only even without `--always-approve`: on 2026-08-22 it rewrote four files in a live
  worktree mid-review. Never point it at the live tree, and never remove the snapshot step.
- It reports, and does not repair, any dirt it finds in the real tree afterwards.
- Never post its output to a PR unedited — it is unverified by construction.
