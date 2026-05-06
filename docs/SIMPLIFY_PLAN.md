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

## Candidate Inventory

| ID | Area | Files | Complexity Signal | Decision | Verification |
|----|------|-------|-------------------|----------|--------------|
| S001 | Inference queue typing and model swap | `chess_anti_engine/inference_threaded.py`, `chess_anti_engine/inference.py` | Queue items, model updates, compile fallback, and cudagraph probing share one dispatcher loop. Recent lint fix needed casts to describe the implicit union. | Implemented small internal typed request/handle objects for `ThreadedDispatcher`; no queue ordering or async pipeline behavior changes intended. | `python3 -m pytest tests/test_threaded_dispatcher.py tests/test_gpu_dispatcher.py -q`; `ruff check chess_anti_engine/inference_threaded.py tests/test_threaded_dispatcher.py`; `basedpyright chess_anti_engine/inference_threaded.py` |
| S002 | Worker lifecycle size | `chess_anti_engine/worker.py`, `worker_config.py`, `worker_assets.py`, `worker_inference.py` | `worker.py` is the largest Python file and owns config, manifest sync, assets, Stockfish, selfplay, uploads, metrics, and cleanup. | First boundary implemented: worker-side inference compile/sync helpers live in `worker_inference.py`; defer larger lifecycle splits until a state-owner boundary is clearer. | `python3 -m pytest tests/test_worker_model_update.py tests/test_threaded_dispatcher.py tests/test_gpu_dispatcher.py -q`; `ruff check chess_anti_engine/worker.py chess_anti_engine/worker_inference.py`; `basedpyright chess_anti_engine/worker.py chess_anti_engine/worker_inference.py` |
| S003 | Server upload state machine | `chess_anti_engine/server/app.py` | Upload handling now preserves durability, but staging, quarantine, dedupe, recovery, and compaction still live in one large module. | Deferred. Keep upload/recovery ordering visible until there is a dedicated upload-state object; extraction now would hide atomicity and dedupe invariants. | `python3 -m pytest tests/test_server_upload_* tests/test_server_trial_lease.py -q` |
| S004 | Replay schema compatibility | `chess_anti_engine/replay/shard.py`, `disk_buffer.py`, `tune/_utils.py`, `tune/replay_exchange.py` | Optional-array validation, union/intersection behavior, old NPZ support, and server zarr paths are spread across several modules. | Implemented shared `zeros_for_storage_field()` in `replay/shard.py` and reused it in disk/tune mixed-schema concatenation. | `python3 -m pytest tests/test_replay_* tests/test_collation.py tests/test_tune_utils.py -q`; `ruff check chess_anti_engine/replay/shard.py chess_anti_engine/replay/disk_buffer.py chess_anti_engine/tune/_utils.py`; `basedpyright chess_anti_engine/replay/shard.py chess_anti_engine/replay/disk_buffer.py chess_anti_engine/tune/_utils.py` |
| S005 | Trial config/report plumbing | `chess_anti_engine/tune/trial_config.py`, `trainable_config_ops.py`, `trainable_report.py`, `trainable_phases.py` | Runtime-mutable config keys and report fields are repeated across parser, sync, report, and status output. | Build a key registry only if it can replace multiple allowlists without obscuring PBT ownership rules. | `python3 -m pytest tests/test_trial_config.py tests/test_trainable_config_ops.py tests/test_trainable_rng_checkpoint.py -q` |
| S006 | Selfplay target assembly | `chess_anti_engine/selfplay/finalize.py`, `state.py`, `manager.py`, `bench/play_batch_timing.py` | Benchmark mirrors and production target assembly can drift. | Prefer shared pure helpers for target construction; keep benchmark-only timing code separate. | `python3 -m pytest tests/test_selfplay_* tests/test_batch_process_ply.py tests/test_bench_result_labeling.py -q` |
| S007 | Search implementation parity | `chess_anti_engine/mcts/gumbel.py`, `gumbel_c.py`, `puct.py`, `puct_c.py`, `uci/search.py` | Python and C paths have overlapping terminal/root/legal-mask behavior. | Add parity helpers/tests before refactoring hot loops. | `python3 -m pytest tests/test_gumbel_* tests/test_mcts_* tests/test_uci_searchmoves.py -q` |
| S008 | Script config/path helpers | `scripts/*` | Many scripts now have safer defaults but still parse roots, run dirs, and Stockfish paths independently. | Add shared script helper only if at least three scripts can delete nontrivial duplicated path logic. | `python3 -m py_compile $(find scripts -name '*.py' -type f)` |

## Current Slice

Inventory and rank the first three simplify candidates:

- [x] S001: read `inference_threaded.py` and dispatcher tests for a no-behavior-change typed queue cleanup.
- [x] S002: map `worker.py` responsibilities and identify extraction boundaries that would reduce risk.
- [x] S003: map `server/app.py` upload helpers and decide whether upload staging should be extracted.

Next slice:

- [x] S004: map replay shard schema compatibility and duplicate optional-field handling.
- [ ] S005: map trial config/report plumbing duplication.
- [ ] S006: map selfplay target/result assembly and benchmark mirrors.

## Decisions

Record accepted/rejected simplification decisions here.

| Date | ID | Decision | Reason | Follow-up |
|------|----|----------|--------|-----------|
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
