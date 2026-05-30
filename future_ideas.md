# Future ideas
This file is for longer-term roadmap items that are intentionally out-of-scope for the minimal “working end-to-end” pipeline.

## Opponent mixing / league training
Goal: avoid overspecializing to Stockfish by mixing opponents.

Ideas:
- Play a fraction of games against older checkpoints (self-league), not just the current “latest” model.
- Maintain a small pool of opponents (e.g. last K champions, plus a few diversity picks) and sample opponents by a schedule.
- Add simple promotion rules for opponents (e.g. keep models that are distinct and/or that beat some baseline).
- Track generalization canaries (puzzle suite, fixed-strength Stockfish eval set) and adjust mixing ratio if they regress.

## Test/lint hygiene follow-ups (from PR #11 review)
Two non-blocking items surfaced while verifying the compact-policy work; recorded here rather than blocking the merge.

- **Flaky `test_prefetched_refresh_is_consumed_before_sync_refresh`** (`tests/test_replay_disk_buffer.py`): passes in isolation (6/6) but intermittently fails when run alongside other `test_replay_disk_buffer` tests, leaving `buf._prefetched_refresh` non-None. Root cause is a prefetch-thread timing race (introduced in `f5ab8e2`, unrelated to compact policy). Make the prefetch consumption deterministic in the test (join/await the prefetch thread, or assert via a synchronization hook) so it doesn't depend on cross-test scheduling.
- **basedpyright `reportUnnecessaryTypeIgnoreComment` at `inference.py:574`**: a `# pyright: ignore[reportArgumentType]` flagged as unnecessary. Likely a baseline artifact (the worktree run reported "venv .venv subdirectory not found", so the baseline wasn't applied). Confirm in a full checkout and drop the now-redundant inline ignore, or refresh the baseline with `basedpyright --writebaseline`.
