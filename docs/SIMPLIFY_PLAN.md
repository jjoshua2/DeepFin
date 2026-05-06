# Simplify Plan

This is the durable tracker for the post-bug-hunt `/simplify` pass.
`docs/REVIEW_BUG_HUNT.md` remains the completed adversarial bug-review record;
this file tracks efficiency, code quality, code reuse, and refactoring work that
should be done only when it reduces real complexity or operational risk.

## Working Rules

- Preserve behavior unless a simplification intentionally changes behavior and
  has explicit test/metric evidence.
- Prefer deleting branches, consolidating duplicate paths, or clarifying
  ownership over adding new abstraction.
- Keep commits scoped by subsystem. Do not mix aesthetic cleanup with runtime
  behavior changes.
- Every accepted item should name the current complexity, why it matters, the
  proposed smaller shape, and the verification command.
- Defer changes that need architecture-level tradeoff decisions to the
  "Heavy Thinking Prompts" section before implementation.

## Status

| Order | Track | Scope | Status | Exit Criteria |
|-------|-------|-------|--------|---------------|
| 1 | Inventory | Hot paths, largest modules, duplicate orchestration | active | Top simplify candidates are listed with owner files and risk. |
| 2 | Inference/worker | `inference.py`, `inference_threaded.py`, `worker.py`, `worker_pool.py`, `worker_assets.py` | pending | Model load/swap, compile/cache, queueing, and worker lifecycle paths have one clear contract each. |
| 3 | Server/distributed ops | `server/*`, `tune/distributed_runtime.py`, `tune/process_cleanup.py` | pending | Upload/lease/process cleanup logic has fewer duplicated state machines and clearer failure ownership. |
| 4 | Replay/shards | `replay/*`, `worker_buffer.py`, `tune/replay_exchange.py` | pending | Schema, optional arrays, staging, and sampling paths avoid duplicate compatibility logic. |
| 5 | Trainer/losses | `train/*`, `tune/trainable_*`, `trial_config.py` | pending | Loss-weight/config plumbing has one source of truth and fewer repeated metric/report fields. |
| 6 | Selfplay/finalization | `selfplay/*`, `arena.py`, `bench/play_batch_timing.py` | pending | Result labels, target construction, timeout/TB handling, and benchmark mirrors are not duplicated unnecessarily. |
| 7 | MCTS/UCI | `mcts/*`, `uci/*`, `tablebase.py` | pending | Python/C search paths and UCI runtime ownership are easier to reason about without weakening hot paths. |
| 8 | Scripts/docs/config | `scripts/*`, `configs/*`, docs | pending | Operational scripts share path/config helpers where useful and docs match the current workflow. |
| 9 | Remaining coverage | Everything not naturally owned by S001-S008 | active | Every tracked source/config/script/test/doc file is assigned to a simplify track. |

## Candidate Inventory

| ID | Area | Files | Complexity Signal | Decision | Verification |
|----|------|-------|-------------------|----------|--------------|
| S001 | Inference queue typing and model swap | `chess_anti_engine/inference_threaded.py`, `chess_anti_engine/inference.py` | Queue items, model updates, compile fallback, and cudagraph probing share one dispatcher loop. Recent lint fix needed casts to describe the implicit union. | Implemented small internal typed request/handle objects for `ThreadedDispatcher`; no queue ordering or async pipeline behavior changes intended. | `python3 -m pytest tests/test_threaded_dispatcher.py tests/test_gpu_dispatcher.py -q`; `ruff check chess_anti_engine/inference_threaded.py tests/test_threaded_dispatcher.py`; `basedpyright chess_anti_engine/inference_threaded.py` |
| S002 | Worker lifecycle size | `chess_anti_engine/worker.py`, `worker_config.py`, `worker_assets.py`, `worker_inference.py` | `worker.py` is the largest Python file and owns config, manifest sync, assets, Stockfish, selfplay, uploads, metrics, and cleanup. | First boundary implemented: worker-side inference compile/sync helpers live in `worker_inference.py`; defer larger lifecycle splits until a state-owner boundary is clearer. | `python3 -m pytest tests/test_worker_model_update.py tests/test_threaded_dispatcher.py tests/test_gpu_dispatcher.py -q`; `ruff check chess_anti_engine/worker.py chess_anti_engine/worker_inference.py`; `basedpyright chess_anti_engine/worker.py chess_anti_engine/worker_inference.py` |
| S003 | Server upload state machine | `chess_anti_engine/server/app.py` | Upload handling now preserves durability, but staging, quarantine, dedupe, recovery, and compaction still live in one large module. | Deferred. Keep upload/recovery ordering visible until there is a dedicated upload-state object; extraction now would hide atomicity and dedupe invariants. | `python3 -m pytest tests/test_server_upload_* tests/test_server_trial_lease.py -q` |
| S004 | Replay schema compatibility | `chess_anti_engine/replay/shard.py`, `disk_buffer.py`, `tune/_utils.py`, `tune/replay_exchange.py` | Optional-array validation, union/intersection behavior, old NPZ support, and server zarr paths are spread across several modules. | Implemented shared `zeros_for_storage_field()` in `replay/shard.py` and reused it in disk/tune mixed-schema concatenation. | `python3 -m pytest tests/test_replay_* tests/test_collation.py tests/test_tune_utils.py -q`; `ruff check chess_anti_engine/replay/shard.py chess_anti_engine/replay/disk_buffer.py chess_anti_engine/tune/_utils.py`; `basedpyright chess_anti_engine/replay/shard.py chess_anti_engine/replay/disk_buffer.py chess_anti_engine/tune/_utils.py` |
| S005 | Trial config/report plumbing | `chess_anti_engine/tune/trial_config.py`, `trainable_config_ops.py`, `trainable_report.py`, `trainable_phases.py`, `utils/config_yaml.py`, `config_keys.py` | Runtime-mutable config keys and report fields are repeated across parser, sync, report, and status output. | Implemented shared `TRAINER_WEIGHT_KEYS` for YAML allowlisting, live trainer sync, and salvage overlay. Report-field consolidation deferred. | `python3 -m pytest tests/test_trial_config.py tests/test_trainable_config_ops.py tests/test_trainable_rng_checkpoint.py tests/test_worker_config_yaml.py -q`; `ruff check chess_anti_engine/config_keys.py chess_anti_engine/tune/trainable_config_ops.py chess_anti_engine/utils/config_yaml.py`; `basedpyright chess_anti_engine/config_keys.py chess_anti_engine/tune/trainable_config_ops.py chess_anti_engine/utils/config_yaml.py` |
| S006 | Selfplay target assembly | `chess_anti_engine/selfplay/finalize.py`, `state.py`, `manager.py`, `bench/play_batch_timing.py` | Benchmark mirrors and production target assembly can drift. | Routed production WDL label mapping through shared `_result_to_wdl`; larger sample-construction sharing deferred because benchmark records are intentionally lighter than production records. | `python3 -m pytest tests/test_selfplay_* tests/test_batch_process_ply.py tests/test_bench_result_labeling.py tests/test_selfplay_result_labeling.py tests/test_is_selfplay_tagging.py -q`; `ruff check chess_anti_engine/selfplay/finalize.py chess_anti_engine/selfplay/game.py chess_anti_engine/bench/play_batch_timing.py`; `basedpyright chess_anti_engine/selfplay/finalize.py chess_anti_engine/selfplay/game.py chess_anti_engine/bench/play_batch_timing.py` |
| S007 | Search implementation parity | `chess_anti_engine/mcts/gumbel.py`, `gumbel_c.py`, `puct.py`, `puct_c.py`, `uci/search.py`, `uci/*`, `tablebase.py` | Python and C paths have overlapping terminal/root/legal-mask behavior. | Deferred. This needs parity-test-first work before hot-loop refactors; current tests cover terminal roots, C/Python root masks, searchmoves, UCI timing, and walker smoke paths. | `python3 -m pytest tests/test_gumbel_* tests/test_mcts_* tests/test_uci_searchmoves.py tests/test_uci_* -q` |
| S008 | Script config/path helpers | `scripts/*`, `configs/*`, `AGENTS.md`, `CLAUDE.md` | Many scripts now have safer defaults but still parse roots, run dirs, and Stockfish paths independently. | Deferred. Inventory found many one-off benchmark/profile scripts, but no helper that would delete nontrivial duplicated logic from at least three scripts without forcing a new abstraction. | `python3 -m py_compile $(find scripts -name '*.py' -type f)` |
| S009 | Remaining file coverage | see "Coverage Matrix" | Some source areas were not changed by S001-S008 but still need explicit ownership. | Active. Use the coverage matrix to drive the rest of the repo pass. | Track-specific tests plus full lint/test sweep at the end. |
| S010 | Encoding and move indexing | `chess_anti_engine/encoding/*`, `chess_anti_engine/moves/*` | Encoding entrypoints repeated LC0 backend selection, extra-feature backend selection, and feature-dropout handling; move/CBoard parity code is hot and correctness-sensitive. | Implemented small shared helpers for LC0 plane selection, feature plane selection, and feature dropout in `encoding/encode.py`; deferred move-index and CBoard changes because existing parity coverage should be expanded before touching those paths. | `python3 -m pytest tests/test_encoding_basic.py tests/test_encode_optimization.py tests/test_move_encoding_lc0_4672.py tests/test_cboard_terminal.py tests/test_feature_dropout.py tests/test_mirror_augmentation.py -q`; `ruff check chess_anti_engine/encoding chess_anti_engine/moves`; `basedpyright chess_anti_engine/encoding chess_anti_engine/moves` |
| S011 | Model, train, ONNX, eval | `chess_anti_engine/model/*`, `onnx/*`, `train/*`, `eval/*` | Trainer and loss paths are large but behavior-sensitive; puzzle rating bucket aggregation was duplicated across evaluators while a helper already existed. | Reused `_by_rating_table()` in all puzzle evaluators; deferred trainer/loss/model checkpoint refactors because recent runtime semantics depend on exact key, metric, and state-dict behavior. | `python3 -m pytest tests/test_puzzle_eval.py tests/test_compute_loss.py tests/test_losses.py tests/test_trainer_warmup.py tests/test_warmup_lr_schedule.py tests/test_training_e2e.py tests/test_swa_export.py tests/test_async_test_eval.py tests/test_onnx_export_int8_smoke.py tests/test_cosmos_fast_optimizer.py -q`; `ruff check chess_anti_engine/model chess_anti_engine/onnx chess_anti_engine/train chess_anti_engine/eval`; `basedpyright chess_anti_engine/model chess_anti_engine/onnx chess_anti_engine/train chess_anti_engine/eval` |
| S012 | Stockfish process and PID helpers | `chess_anti_engine/stockfish/*` | `StockfishUCI.search()` mixed UCI protocol flow with nested parsing for MultiPV, score, WDL, and PV moves; PID code is intentionally dense but heavily tested. | Extracted small UCI info-line parse helpers for integer fields, score, WDL, and PV move; left PID lever math unchanged. | `python3 -m pytest tests/test_cp_to_wdl.py tests/test_stockfish_uci_timeout.py tests/test_pid_inverse_regret.py tests/test_selfplay_sf_wdl_pov.py -q`; `ruff check chess_anti_engine/stockfish`; `basedpyright chess_anti_engine/stockfish` |

## Current Slice

Inventory and rank the first three simplify candidates:

- [x] S001: read `inference_threaded.py` and dispatcher tests for a no-behavior-change typed queue cleanup.
- [x] S002: map `worker.py` responsibilities and identify extraction boundaries that would reduce risk.
- [x] S003: map `server/app.py` upload helpers and decide whether upload staging should be extracted.

Next slice:

- [x] S004: map replay shard schema compatibility and duplicate optional-field handling.
- [x] S005: map trial config/report plumbing duplication.
- [x] S006: map selfplay target/result assembly and benchmark mirrors.

Next slice:

- [x] S007: map search implementation parity across MCTS/UCI hot paths.
- [x] S008: map script/config path helper duplication.
- [ ] S009: inventory remaining package files not covered by S001-S008.

## Coverage Matrix

This matrix covers the 300 tracked repo files in scope for this pass
(`git ls-files 'chess_anti_engine/**' 'scripts/**' 'tests/**' 'configs/**'
'*.md' 'docs/**' 'pyproject.toml'`).

| Track | Globs | Files | Simplify Status |
|-------|-------|-------|-----------------|
| S010 | `chess_anti_engine/encoding/**`, `chess_anti_engine/moves/**` | 13 | processed; move/CBoard parity refactors deferred |
| S011 | `chess_anti_engine/model/**`, `chess_anti_engine/onnx/**`, `chess_anti_engine/train/**`, `chess_anti_engine/eval/**` | 17 | processed; trainer/loss state refactors deferred |
| S001/S002 | `chess_anti_engine/inference*.py`, `chess_anti_engine/worker*.py` | 9 | processed; revisit worker lifecycle later |
| S007 | `chess_anti_engine/mcts/**`, `chess_anti_engine/uci/**`, `chess_anti_engine/tablebase.py` | 21 | processed; parity-test-first |
| S006 | `chess_anti_engine/selfplay/**`, `chess_anti_engine/arena.py`, `chess_anti_engine/bench/**` | 15 | processed; benchmark sharing mostly deferred |
| S004 | `chess_anti_engine/replay/**` | 6 | processed |
| S003/S005 | `chess_anti_engine/server/**`, `chess_anti_engine/tune/**` | 23 | partially processed; more tune/server review later |
| S012 | `chess_anti_engine/stockfish/**` | 5 | processed |
| S013 | `chess_anti_engine/utils/**`, `chess_anti_engine/run.py`, `chess_anti_engine/config_keys.py`, `chess_anti_engine/version.py`, `chess_anti_engine/__init__.py` | 11 | pending |
| S008 | `scripts/**`, `configs/**`, `*.md`, `docs/**`, `pyproject.toml` | 76 | processed; broad helper deferred |
| S014 | `tests/**` | 104 | pending; simplify only where test helpers are meaningfully duplicated |

## Decisions

Record accepted/rejected simplification decisions here.

| Date | ID | Decision | Reason | Follow-up |
|------|----|----------|--------|-----------|
| 2026-05-06 | S009 | Add explicit coverage matrix for the whole tracked repo scope. | The simplify pass should not depend on memory of which directories were already examined. The matrix assigns all 300 tracked source/config/script/test/doc files to a track and shows what remains. | Continue with S010, S011, S012, S013, S014, then revisit partially deferred S002/S003/S005/S007 if tests justify it. |
| 2026-05-06 | S011 | Share puzzle rating-bucket aggregation across all puzzle evaluators. | `run_puzzle_eval`, `run_policy_sequence_eval`, and `run_value_head_puzzle_eval` report the same rating bucket shape, but only one path used the existing helper. The trainer/loss/model checkpoint surfaces were reviewed and left unchanged because simplification there would touch live metric names, optimizer state, compile wrappers, and tolerant load behavior. | Revisit trainer/loss extraction only with golden metric-key tests for `TrainMetrics`, `result.json`, `status.csv`, checkpoint resume, SWA export, and live loss-weight sync. |
| 2026-05-06 | S012 | Extract Stockfish UCI info-line parsing helpers. | The timeout/process lifecycle should stay inline in `StockfishUCI`, but score/WDL/PV parsing is pure token handling and was obscuring the search loop. PID lever logic was reviewed and left unchanged because the existing tests encode subtle controller behavior. | Add dedicated parser fuzz tests before changing accepted malformed-UCI behavior or PID math. |
| 2026-05-06 | S010 | Centralize encoding feature-plane/dropout plumbing, but leave move-index and CBoard logic unchanged. | The repeated LC0/feature/dropout branches were low-risk Python duplication. The move-policy LUTs, C legal move path, CBoard push/index behavior, and C headers are high-blast-radius hot paths where a cleanup would need stronger C/Python parity tests first. | Revisit move/CBoard simplification only after parity tests cover all legal moves across representative FENs, promotions, en passant, castling, terminal boards, and mirrored policies. |
| 2026-05-06 | S008 | Defer shared script helper extraction. | The scripts directory is mostly benchmark, profile, and operational entrypoints with different argument contracts. A common helper would add an import dependency without deleting enough duplicated path/config logic yet. Compile coverage is clean. | Revisit if three or more scripts converge on the same run-dir/config/Stockfish resolution logic after future edits. |
| 2026-05-06 | S007 | Defer MCTS/UCI hot-loop simplification until parity tests are expanded. | The search surface spans Python and C Gumbel/PUCT, persistent UCI roots, tablebase overrides, legal masks, searchmoves, timing, stop/ponder state, and walker pools. Existing tests pin several edges, but a refactor without more parity coverage could silently change node accounting or root policy behavior. | Add explicit C/Python parity tests for root legal masks, terminal roots, single-legal roots, tablebase-solved roots, and searchmoves before extracting shared search helpers. |
| 2026-05-06 | S006 | Share result-string to WDL label mapping in production finalization. | `bench/play_batch_timing.py` already used `selfplay.game._result_to_wdl`, while `selfplay/finalize.py` carried a manual copy. Routing production through the shared helper removes drift around `"*"`/draw handling and keeps result-label tests meaningful for both paths. | Keep benchmark sample assembly separate for now; its record tuple omits production-only legal masks, TB overrides, and selfplay tags. |
| 2026-05-06 | S005 | Move runtime-mutable trainer weight key ownership to `config_keys.py`. | The same trainer-weight keys were listed in the YAML allowlist and in trainable runtime sync/salvage paths. A small shared constant removes that drift risk without centralizing broader PBT/report semantics. | Defer report-dict consolidation until metric field ownership can be tested against `status.csv` and `result.json` consumers. |
| 2026-05-06 | S004 | Centralize missing replay-field defaults in `replay/shard.py`. | `disk_buffer.py` and `tune/_utils.py` both reimplemented the same required/default and optional-schema zero-fill logic. `replay/shard.py` already owns `_OPTIONAL_FIELD_SPECS`, validation, serialization, and lazy loading, so the default-array constructor belongs there too. | Keep dataset collation separate for now because it maps storage names to trainer tensor names; revisit only with a full storage-to-batch field mapping. |
| 2026-05-06 | S003 | Defer server upload extraction for now. | The live upload path and crash recovery share subtle invariants: stream/hash before validation, quarantine invalid tarballs, atomically promote extracted zarr to `_pending` before ack, update dedupe state under `upload_lock`, delete duplicate pending shards, and re-seed accumulators from `_pending`/`_in_flight` on restart. The current comments and tests keep that ordering auditable inside `create_app`; moving pieces without an upload-state object would make review harder. | Revisit only with a small `UploadState`/`UploadPaths` design and tests covering duplicate retry, failed promote, failed compaction, restart recovery, and per-trial isolation. |
| 2026-05-06 | S002 | Extract only worker-side inference compile/sync helpers into `worker_inference.py`; leave manifest, upload, lease, and selfplay state inside `WorkerSession`. | `worker.py` mixes lifecycle state with pure inference utility behavior. Moving FP8/compile/evaluator-sync helpers reduces top-level worker coupling without changing when model swaps, uploads, or evaluator updates occur. Broader splits would currently risk hiding state transitions across model SHA, lease ID, and buffered uploads. | Verify worker model-update tests plus dispatcher tests; continue S002 later only after identifying a state owner, likely for manifest polling or pending upload drainage. |
| 2026-05-06 | S001 | Accept a dedicated no-behavior-change cleanup that replaces tuple queue items with internal dataclass request/handle objects. | The previous dispatcher queue encoded eval requests, model updates, and pending GPU work as tuple shape plus casts; explicit objects make the model-update boundary and future-scatter path auditable while preserving the same drain/submit/scatter order. | Run dispatcher tests, lint, and type check before commit; defer larger worker/inference boundary changes to S002 or a heavy-thinking pass. |
| 2026-05-06 | S000 | Keep bug fixes in `REVIEW_BUG_HUNT.md`; track refactors here. | Bug-hunt tracker is complete and should stay auditable. Simplification needs a separate queue to avoid mixing cleanup with correctness findings. | Start with S001-S003 inventory. |

## Heavy Thinking Prompts

Use these prompts in GPT Pro when a subsystem needs deeper design review before
implementation.

### Inference/Worker Boundary

```text
You are reviewing a Python/PyTorch chess selfplay worker with shared GPU
inference. The codebase has DirectGPUEvaluator, ThreadedDispatcher,
SharedSlotBroker/SlotBroker, model hot-swap, torch.compile/cudagraph cache,
and worker processes that download manifests/assets and run selfplay batches.

Goal: propose a simplification plan that preserves runtime behavior but reduces
model-load/swap/queueing complexity. Focus on clear ownership boundaries,
failure modes, and minimum tests/benchmarks required before refactoring.
Avoid aesthetic refactors. Return a prioritized plan with risks and rejection
criteria.
```

### Replay/Server Upload Boundary

```text
You are reviewing a distributed replay upload path for a chess training system.
Workers upload replay shards; server validates schema, quarantines invalid
uploads, dedupes retries, stages accepted shards durably before ack, recovers
pending/in-flight shards after crash, and compacts shards for trainer ingest.

Goal: identify which parts should be extracted into smaller helpers or modules
without weakening atomicity/durability. Explain the invariants that must remain
visible in code, proposed test cases, and cases where extraction would make the
system harder to audit.
```

### Trial Config And Reporting

```text
You are reviewing Ray Tune/PBT config plumbing for a chess engine training
system. Config keys flow through YAML flattening, TrialConfig, live reload,
PBT mutations, worker process signatures, Trainer loss weights, status.csv,
result.json, and publish manifests.

Goal: design a lower-duplication key ownership scheme that avoids silent stale
worker settings or missing live loss-weight sync. Include migration steps,
tests, and what not to centralize.
```
