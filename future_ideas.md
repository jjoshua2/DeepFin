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

## Training target calibration backlog
Goal: keep policy/WDL supervision tunable without throwing away replay.

Ideas:
- Store sparse Stockfish policy labels as MultiPV move indices + raw scores + candidate count, then build dense or sparse CE targets at train time. Main benefit is retuning `sf_policy_temp` and `sf_policy_label_smooth` over old replay; disk savings are secondary because dense `sf_policy_target` compresses well.
- For the sparse SF-policy path, keep dense `sf_policy_target` for a transition period and add parity tests for current temp/smoothing. Preferred final loss is sparse CE over gathered log-probs, with smoothing handled analytically.
- Probe BT4 raw policy sharpness on the same replay/start positions and compare top move probability, top-5 mass, entropy, and effective legal moves against our main search target, soft policy target, and SF-policy target.
- Recheck whether `sf_policy_label_smooth` should remain legal-set smoothing or become uncovered-legal smoothing after `multipv=40`; current live setting is intentionally small at 1%.

## Offline architecture sweep backlog
Goal: test architecture changes on fixed replay snapshots before risking live selfplay.

Ideas:
- Directional relation-basis Smolgen: split coarse rank/file/diagonal masks into N/S/E/W, diagonal directions, knight offsets, pawn attacks, king adjacency, and self. Compare against the current relation-basis center/RMS setup with the same seed and replay.
- Occupancy-conditioned attention relations: add zero-initialized dynamic masks for clear rook rays, clear bishop rays, blocked rays, first visible own/opponent piece, and own/opponent attackers. Judge policy loss, regret-tail buckets, and speed, not only total loss.
- Layer/head restriction for relation branches: train variants with relation bias only in early, mid, late, or early+mid layers. If only a subset matters, use this to reduce inference cost.
- BT4-style global board embedding adapter: retest global board embedding as a gated/zero-init adapter, separately from square embedding and arc adapter, from fresh init on the same replay snapshot.
- Square embedding audit only if needed: square-add and multiplicative/additive gate variants have already been run in both older fixed-snapshot tests and the later `lc0_root` sidecar chain. Do not queue another generic square-embedding sweep unless it is a tightly matched-seed/current-replay audit to resolve a specific conflict.
- Smolgen/relation normalization variants: only revisit normalization around the generated relation branch if diagnostics show scale drift. Do not rerun Q/K RMS normalization as a priority; the prior QK-norm sidecar was negative.
- Channel-utilization interventions: if dormant channels persist, compare small residual channel dropout, channel-balance regularization, and architecture-side routing changes from fresh init. Track channel spread at epochs 1/3/6 plus policy/WDL loss.
- Extra-depth Pareto checks: rerun best 11-layer candidate against a 12-layer control and a 12-layer version of the same relation setup to see whether relation gains persist with more capacity.

## PR #13 tooling cleanup follow-ups
Non-blocking cleanup from the optimizer/offline-sweep review. These are worth doing, but they are not correctness blockers for the optimizer/config PR.

- **Offline sweep shell driver consolidation**: the `scripts/arch_sweep_*.sh` files repeat the same launch/poll/log/result scaffolding. Replace them with one shared driver plus small per-sweep config files that define only `experiments=()` and common args.
- **`offline_replay_epoch.py` pyright blanket cleanup**: replace the module-wide pyright disables with either targeted suppressions or a refreshed basedpyright baseline, so future edits still get optional-member/type feedback.
- **SODA per-step clone optimization**: the current SODA implementation preserves the paper formula exactly by saving pre-base-step weights. Investigate a clone-free formulation only if it can be proven equivalent for Aurora/Muon-style optimizers that may inspect current parameter norms during `base.step()`.
