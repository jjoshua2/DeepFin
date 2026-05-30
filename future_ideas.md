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

## PR #12 cleanup / simplification follow-ups
Non-blocking cleanup ideas from the Gumbel/selfplay-runtime review, recorded here rather than expanding the stabilization PR.

- **Compact policy tensor helper**: `inference.py` and `train/losses.py` each cache `COMPACT_TO_FULL_POLICY` as a device tensor. Consider one shared torch-policy utility if it can avoid import cycles and keep the device-cache behavior explicit.
- **Input-history mode mapping**: `_input_history_mode` in `mcts/gumbel_c.py`, `_c_input_history_mode` in `selfplay/network_turn.py`, and the C extension constants all spell out the same mode IDs. Move the Python mapping to one home and pin it against the C binding in tests.
- **Compact-everywhere replay storage**: once PR #12/#13 settle, decide whether replay shards should stay canonical 4672 or move fully to compact `lc0_1858`. If moving compact everywhere, do it in one PR with manifest semantics, shard validation, augmentation, and old-shard migration all updated together.
- **Search policy adapter**: consolidate compact-to-full widening and `policy_encoding` plumbing across Gumbel/PUCT/Python/C paths through one helper. Keep the current explicit config fields, but remove shape-sniffing and duplicate conversion sites.
- **Upload state machine split**: only refactor `server/app.py` upload handling after tests cover pre-extraction size limits, quarantine, pending/in-flight recovery, duplicate retry, and compaction. A small `UploadState`/`UploadPaths` object would make the flow easier to audit later.
