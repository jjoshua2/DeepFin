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
Both items resolved by the hygiene PR (2026-06):

- DONE: **Flaky `test_prefetched_refresh_is_consumed_before_sync_refresh`** — the background prefetch loop autonomously refills `_prefetched_refresh` within ~0.1s of it emptying, so the test's final is-None assertion raced the refill under cross-test CPU load. The test now stops the prefetch thread (`buf.close()` + no-op `_ensure_prefetch_thread`) before injecting the chunk, making the consume path fully deterministic.
- DONE (stale): **basedpyright unnecessary-ignore at `inference.py:574`** — the ignore no longer exists; it was removed by interim refactors.

## PR #12 cleanup / simplification follow-ups
Non-blocking cleanup ideas from the Gumbel/selfplay-runtime review, recorded here rather than expanding the stabilization PR.

- DONE: **Compact policy tensor helper** — `moves/torch_maps.py` now holds the single lru-cached device-tensor home (`compact_to_full_index[_for]`, `full_to_compact_index`, `policy_index_remap_table`); the private copies in `inference.py`, `train/losses.py`, and `train/sparse_sf_ce.py` were removed.
- DONE: **Input-history mode mapping** — the mapping was already consolidated into `encoding/lc0.py::c_input_history_mode` (both `gumbel_c` and `network_turn` import it); the remaining gap was the test pin, now covered by `test_encoding_basic.py::test_batch_encoder_selection_matches_python_per_mode` (name → C-encoder selection vs `encode_cboard` per mode) alongside the existing mode-id stability test.
- **Compact-everywhere replay storage**: once PR #12/#13 settle, decide whether replay shards should stay canonical 4672 or move fully to compact `lc0_1858`. If moving compact everywhere, do it in one PR with manifest semantics, shard validation, augmentation, and old-shard migration all updated together.
- **Search policy adapter**: consolidate compact-to-full widening and `policy_encoding` plumbing across Gumbel/PUCT/Python/C paths through one helper. Keep the current explicit config fields, but remove shape-sniffing and duplicate conversion sites.
- **Upload state machine split**: only refactor `server/app.py` upload handling after tests cover pre-extraction size limits, quarantine, pending/in-flight recovery, duplicate retry, and compaction. A small `UploadState`/`UploadPaths` object would make the flow easier to audit later.

## Training target calibration backlog
Goal: keep policy/WDL supervision tunable without throwing away replay.

Ideas:
- DONE (shard schema v2): sparse Stockfish labels are stored per SF-labeled sample as `sf_multipv_raw` (move idx + cp/mate + native wdl per MultiPV line, padded int16 rows) and `sf_label_meta` (nodes/depth/record-level eval). `train/target_builder.py` rebuilds `sf_policy_target`/`sf_wdl` under arbitrary params (`train.rebuild_sf_targets`, default off), `scripts/retarget_retrain.py` drives offline retuning runs. Dense `sf_policy_target` is still written alongside (the transition-period plan below); parity is test-enforced.
- DONE (follow-up): `train.sf_policy_sparse_ce` switches the `policy_sf` loss to sparse CE over gathered log-probs with analytic smoothing (`train/sparse_sf_ce.py`, exact-parity-tested against the dense soft CE incl. the compact-logits-over-full-shard projection and the one-hot fallback), and `selfplay.record_dense_sf_policy: false` stops writing the dense `sf_policy_target`. Operational sequencing still applies: flip the writer flag only after a full replay window carries sparse labels AND training runs with the sparse CE on — rows with neither dense nor sparse labels drop out of the `policy_sf` loss.
- Consolidate optional replay metadata declarations after the adjusted-WDL target path settles. PR #14 added matching field lists in shard storage, dataset collation, and loss-source selection; keep the explicit behavior for now, but eventually move the future-regret field names/dtypes/has-flags to one small shared spec so future target metadata does not require hand-updating three places.
- Before live-enabling `use_adjusted_wdl_target`, explicitly decide whether the adjusted outcome should remain the fallback anchor for missing SF/search WDL blend components, or whether adjustment should apply only to the pure game-outcome fraction. Current PR #14 behavior uses the adjusted target consistently as the anchor everywhere once the feature is enabled.
- If start-position metadata ever matters for memory or API clarity, replace `SelfplayState.starting_boards` with a smaller start-FEN string array plus a syzygy-only replay-board copy path. PR #14 keeps full board copies because existing syzygy replay code already consumes them and batch-size memory cost is negligible.
- Probe BT4 raw policy sharpness on the same replay/start positions and compare top move probability, top-5 mass, entropy, and effective legal moves against our main search target, soft policy target, and SF-policy target.
- Recheck whether `sf_policy_label_smooth` should remain legal-set smoothing or become uncovered-legal smoothing after `multipv=40`; current live setting is intentionally small at 1%.

## Compact policy simplification backlog
Goal: make `lc0_1858` the normal live-training policy path now that compact output tested well, while keeping old `az_4672` data/checkpoints loadable.

Ideas:
- Make `policy_encoding: lc0_1858` the production/default config path once the stacked replay/model/runtime PRs are merged.
- Keep `az_4672` as a legacy import, conversion, and checkpoint-migration path rather than a first-class live training mode.
- Require each shard to declare `policy_encoding`, validate every policy-space array against that encoding, and reject mixed encodings inside one replay buffer unless an explicit conversion step runs.
- Store `sf_move_index` in the shard's own policy encoding, or store a move identity and derive the index at load time. Avoid implicit "full index with compact target" conventions.
- Remove the misleading `lc0_4672` name from manifests and helpers. Full 4672 should be `az_4672`; LC0 should mean the compact 1858 move encoding.
- Keep conversion helpers for old replay/checkpoints, but keep the hot training path branch-light once a run has chosen its encoding.

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

## Search/training audit bets (2026-06)
From the search+training correctness audit (`docs/AUDIT_2026-06.md`). Both follow the eval protocol: fixed-budget train, then `scripts/arena_standard.py --matched_sims` vs the control.

- **DONE (flag-gated, default off): continuous categorical value target** — `selfplay.categorical_blend_frac` blends the `value_categorical` HL-Gauss target toward SF's objective eval expected score `(W - L)` instead of the ternary game outcome, so the 32-bin distributional head uses its bins and carries a search-horizon signal distinct from the `wdl` hard target. The head is auxiliary (does not feed MCTS), so it cannot regress search. Sweep `{0.3, 0.5, 0.7}` via `configs/exp_categorical_continuous.yaml`; KEEP iff arena ≥ +5 Elo, else KILL (a neutral auxiliary just wastes the `w_categorical = 0.30` gradient). Follow-up variants if it clears: blend toward the search-averaged value, or toward the full `wdl`-style blend.
- **gumbel_c_scale sweep** — `search.gumbel_c_scale` default `0.1` is 10× below the mctx/Gumbel-MuZero reference `1.0`, so the completed-Q value term contributes ~10× less than the prior term during sequential-halving selection (halving is much more prior-driven than canonical Gumbel). Already wired end-to-end. Sweep `{0.1, 0.3, 1.0}` at the production 256-sim budget with `scripts/arena_standard.py --matched_sims`; this is a pure search-time knob (no retrain needed), so it's cheap to evaluate. Document the chosen value's rationale once validated.
- **Sequential-halving budget micro-overspend** — when `budget < len(rem) · rounds_left`, `vpa` floors to 1 and a round can spend ~1-2 sims beyond the requested budget (`gumbel.py` / `_mcts_tree.c`). ~0 Elo and bounded by `topk`; only worth fixing if both the Python and C hot loops are refactored together behind a shared budget-split helper.
- **`moves_left` target scale** — normalized by `max_plies` (`finalize.py`), compressing the target into a small fraction of [0, 1]; raw plies (smooth-L1 handles the scale) would be a cleaner target. Low priority (`w_moves_left = 0.02`).
