# Adversarial Review Tracker

This document is the durable tracker for project-wide adversarial review and
follow-up simplification work. The goal is to find correctness bugs,
training-quality bugs, runtime reliability bugs, efficiency/performance bugs,
scalability bugs, security bugs, missing tests, and code structure that should
be simplified before it becomes harder to reason about.

Current strategy: review by component and contract boundary, not only by commit.
Commit-level review is still useful for newly landed changes, but the long-term
pass should ask whether each subsystem is correct, minimal, observable, and easy
to change.

## Current Review Cycle

Started: 2026-05-06

Working rules:

- Review read-first, patch-second. Record attack vectors even when no bug is
  found.
- Every finding should name the violated invariant, exact file/line evidence,
  likely runtime effect, and the smallest verification command.
- Fix commits should stay grouped by component unless a cross-cutting contract
  bug demands otherwise.
- The `/simplify` pass is separate from bug fixing unless simplification is the
  lowest-risk way to remove a bug.

### Active Queue

| Order | Track | Component | Files | Status | Notes |
|-------|-------|-----------|-------|--------|-------|
| 1 | adversarial | Selfplay labels and finalization | `chess_anti_engine/selfplay/finalize.py`, `state.py`, `stockfish_turn.py`, `tablebase.py` | deep | Highest training-data corruption risk. Review POV, result labels, TB adjudication, draw handling, optional target fields. |
| 2 | adversarial | Replay and shard ingest | `chess_anti_engine/replay/*`, `worker_buffer.py`, server/worker shard paths | active | Schema drift, stale shards, missing arrays, holdout leakage, old-run compatibility. |
| 3 | adversarial | Worker lifecycle | `worker.py`, `worker_assets.py`, `worker_pool.py`, `stockfish/*` | pending | Manifest restart keys, model swaps, Stockfish restarts, partial updates, pause/resume. |
| 4 | adversarial | Tune trainable lifecycle | `chess_anti_engine/tune/*` | pending | Config reload, donor/salvage overlays, PID state restore, manifests, async eval timing. |
| 5 | adversarial | Loss and trainer contracts | `chess_anti_engine/train/losses.py`, `trainer.py`, `targets.py` | pending | Blend weights, masks, target normalization, old shard behavior, live weight sync. |
| 6 | adversarial | MCTS and search engines | `chess_anti_engine/mcts/*`, `uci/search.py`, `tablebase.py` | pending | Memory ownership, root reuse, terminal handling, solved/TB propagation, concurrent mutation. |
| 7 | adversarial | Distributed server | `chess_anti_engine/server/*`, `tune/distributed_runtime.py` | pending | Upload durability, auth, leases, path safety, backpressure, recovery after crash. |
| 8 | adversarial | UCI runtime | `chess_anti_engine/uci/*`, `stockfish/uci.py` | pending | Protocol state, time controls, ponderhit, cancellation, subprocess hangs. |
| 9 | simplify | Hot-path code quality and efficiency | selfplay, replay, MCTS, inference, worker | pending | Remove duplicated orchestration, clarify contracts, reduce allocations/sync/I/O stalls. |
| 10 | simplify | Config and ops ergonomics | `configs/*`, `scripts/*`, `AGENTS.md`, `CLAUDE.md` | pending | Stale knobs, unclear ownership, scripts that assume one run layout. |

### Current Slice Checklist

Selfplay labels and finalization:

- [x] Map result-label convention: `wdl_target` class order, POV, scalar value,
  categorical target, and draw handling.
- [ ] Verify terminal result source precedence: natural game end, timeout
  adjudication, tablebase adjudication, tablebase rescore.
- [x] Verify Stockfish WDL and cp-logistic target POV through curriculum games.
- [x] Verify Stockfish annotations cover both colors in full selfplay games.
- [x] Verify selfplay vs curriculum counters and `is_selfplay` tags survive slot
  recycling.
- [ ] Verify optional targets (`sf_wdl`, `sf_policy_target`, `search_wdl`,
  legal masks, future policy, volatility) are masked or defaulted safely.
- [ ] Identify simplification candidates only after correctness review.

Replay and shard ingest:

- [x] Map canonical shard schema: required arrays, optional flag/value pairs,
  sparse policy/mask storage, legacy NPZ compatibility, and zarr writer/loader.
- [x] Verify optional target flag/value pairs reject corrupt shards instead of
  synthesizing present zero labels.
- [ ] Verify disk buffer resume/window logic and prefetch cannot sample deleted
  or schema-incompatible shards.
- [ ] Verify server upload, pending recovery, and compaction preserve accepted
  samples across crash/restart interleavings.
- [ ] Verify worker upload/delete flow cannot drop an unaccepted shard.
- [ ] Identify replay simplification candidates only after correctness review.

Current notes:

- WDL class convention in finalization is `0=sample POV win`, `1=draw`,
  `2=sample POV loss`; categorical scalar maps win/draw/loss to `+1/0/-1`.
- Curriculum games are counted in `w/d/l`; full-selfplay games are tracked in
  selfplay counters but intentionally excluded from curriculum winrate.
- Finding F006 opened/fixed in this cycle: full-selfplay net-color turns missed
  SF annotations because the next network turn replaced the "latest" record
  before any SF label query ran.
- Finding F007 opened/fixed in this cycle: tablebase path swaps could close a
  handle still being probed by another thread; tablebase handles are now cached
  per path and failed opens still close their temporary handle.
- Finding F008 opened/fixed in this cycle: shard validation accepted optional
  `has_*` flags without the paired value array, causing present-but-zero
  auxiliary targets after load.

### `/simplify` Review Criteria

Use this as a separate pass after a component's correctness review:

- Efficiency: unnecessary allocations, board/FEN roundtrips, repeated zarr
  opens, GPU sync points, unbounded queues, avoidable serialization.
- Code quality: functions with mixed responsibilities, hidden state mutation,
  weak names, implicit contracts, broad exception handling, unclear units/POV.
- Code reuse: duplicated config plumbing, duplicate result-label logic, multiple
  retry/backoff implementations, parallel shard schemas.
- Refactoring: extract only when it reduces real complexity or makes invariants
  testable. Avoid broad style churn.
- Tests: add regression tests for behavior, not just shape or smoke coverage.

### Heavy-Thinking Prompt Bank

Use these prompts when a question deserves GPT Pro/deep research rather than a
quick local code review.

#### Architecture Simplification Prompt

```text
You are reviewing a Python chess-engine training codebase with distributed
selfplay, replay shards, MCTS, Stockfish supervision, tablebase adjudication,
and Ray Tune/PBT orchestration. I want an adversarial simplification plan, not
a rewrite. Given the files and notes below, identify the highest-leverage
refactors that reduce correctness risk, duplicated logic, and operational
fragility while preserving behavior. For each proposal include: invariant
protected, files touched, migration risk, tests required, and a smallest
incremental PR sequence. Challenge any refactor that is aesthetic only.

Context:
[paste component map, file snippets, current findings, and constraints]
```

#### Training-Target Semantics Prompt

```text
Act as a skeptical ML systems reviewer. This chess project trains on game WDL,
Stockfish WDL/cp-logistic targets, search WDL, policy targets, future policy,
categorical value targets, and tablebase relabeling. Review the target-blending
semantics for hidden POV/sign/class-order bugs, feedback loops, label leakage,
and draw/adjudication bias. Produce concrete invariants and test cases that
would catch silent training corruption. Do not assume comments are correct.

Context:
[paste relevant finalize/loss/replay/tablebase snippets and metric definitions]
```

#### Distributed Runtime Prompt

```text
Act as an adversarial distributed-systems reviewer. This repo has a training
server publishing model manifests, workers polling/downloading assets, workers
uploading replay shards, trainables ingesting shards, and restart/resume scripts.
Find crash-consistency, stale-data, auth/path-safety, backpressure, partial
publish, and lifecycle bugs. Prioritize concrete failure interleavings and
minimal fixes with tests.

Context:
[paste server app, worker polling/upload, manifest publish, ingest, and scripts]
```

#### MCTS C Extension Prompt

```text
Review this Python/C MCTS extension as a memory-safety and algorithmic
correctness auditor. Focus on ownership, reset/reuse behavior, concurrent
walkers, virtual loss, solved-node propagation, tablebase overrides, NumPy
buffer lifetimes, and Python fallback parity. Give concrete attack vectors and
tests or sanitizer/profiling commands.

Context:
[paste _mcts_tree.c regions, gumbel_c.py, puct_c.py, and tests]
```

## Status Legend

- `pending`: not reviewed
- `skim`: checked for obvious runtime/config/API/path issues
- `deep`: reviewed for invariants, call sites, edge cases, tests, and failure modes
- `finding`: one or more findings recorded
- `fixed`: finding addressed and verified
- `n/a`: intentionally excluded from bug-hunt scope

## Finding Categories

- `Correctness`: behavior is wrong for legal chess, data contracts, user-visible output, or expected API behavior.
- `Training Quality`: behavior may train the model toward the wrong target, inject biased/noisy data, degrade evaluation, or silently change learning dynamics.
- `Reliability`: crashes, hangs, corrupt state, bad cleanup, retry bugs, race conditions, and lifecycle failures.
- `Efficiency`: avoidable CPU/GPU cost, poor batching, repeated conversions, excessive allocations, expensive logging, I/O stalls, or slow hot paths.
- `Scalability`: bottlenecks or state growth that appear under more workers, larger replay, larger MCTS budgets, or longer runs.
- `Security`: unsafe upload/download paths, credential handling, auth bypass, archive traversal, or unsafe subprocess usage.
- `Compatibility`: version/device/platform issues, optional dependency breaks, and C-extension/Python fallback drift.
- `Test Gap`: missing or weak regression coverage, especially where tests assert shapes but not semantics.

## Severity Guide

- `Critical`: can corrupt training data/checkpoints, silently train the wrong objective, expose credentials, or make production runs unusable.
- `High`: likely correctness bug, common crash/hang, severe performance loss in a hot path, or missing test around core invariants.
- `Medium`: plausible bug under specific configs or scale, moderate inefficiency, weak error handling, or important test gap.
- `Low`: cleanup-level issue, rare edge case, misleading diagnostics, or small efficiency issue outside hot paths.

## Findings

| ID | Severity | Category | Component | File | Summary | Evidence | Test/Benchmark Needed | Status |
|----|----------|----------|-----------|------|---------|----------|-----------------------|--------|
| F001 | High | Reliability / Test Gap | Selfplay | `tests/test_play_batch_helpers.py`, `chess_anti_engine/selfplay/manager.py` | Tracked test suite failed during collection because helper tests imported `_PlayBatchState`, `_init_play_batch_state`, and module-level `_sf_terminal_result`, but `manager.py` only had the nested timeout helper. | `pytest tests/test_play_batch_helpers.py` failed with `ImportError: cannot import name '_PlayBatchState'`. | Added module-level helper/state API and routed timeout adjudication through shared helper. Verified `tests/test_play_batch_helpers.py` -> `16 passed`; broader selfplay slice -> `38 passed`. | fixed |
| F002 | High | Reliability | Distributed server | `chess_anti_engine/server/app.py` | Accepted shard uploads are deleted from the temp extraction path and buffered only in process memory until compaction flush. If the server process crashes before target-size/age/lifespan flush, replay samples can be lost even though the worker received `stored: true`. | `_upload_shard_impl` loads `shard_arrs`, deletes `tmp` and `tmp_zarr`, appends `arrays_to_samples(shard_arrs)` to `upload_accumulators`, and only writes a zarr in `_flush_buffered_upload_to_inbox`. Accumulators are in-memory only. | Add crash-recovery test: upload below compaction threshold, simulate process restart before flush, assert accepted samples are recoverable or that response does not claim durable storage until persisted. | open |
| F003 | Medium | Reliability | Operational scripts | `scripts/diagnose.py` | Diagnostic script hardcodes `runs/pbt2_small/tune`, discovers the latest trial at import time, and has no CLI override despite documenting only `PYTHONPATH=. python3 scripts/diagnose.py`. It can crash before argument parsing on machines without that run, or diagnose the wrong run when the active config differs. | Top-level `TRIAL_DIR = sorted(Path("runs/pbt2_small/tune").glob("train_trial_*"), ...)[-1]` executes before any `main`/CLI. Replay path later uses `TRIAL_DIR / "selfplay_shards"`, while current distributed paths often use replay roots elsewhere. | Add `--run/--trial-dir/--config/--device` args and move discovery into `main`; add a smoke test for missing run producing a clear error. | open |
| F004 | Medium | Reliability | Stockfish integration | `chess_anti_engine/stockfish/uci.py` | Real Stockfish subprocess reads have no timeout. A stalled engine or protocol deadlock can hang selfplay/arena code while holding the Stockfish lock. | `_wait_for` and `search` call `self.proc.stdout.readline()` in unbounded loops. UCI smoke tests use timeout readers around the engine process, but Stockfish integration tests mostly use mocks and do not exercise a stalled real subprocess. | Add a timeout/error path around Stockfish reads, terminate/restart the process on timeout, and add a fake-process regression test that never emits `uciok`, `readyok`, or `bestmove`. | open |
| F005 | Low | Test Hygiene | Pytest config | `pyproject.toml`, `tests/test_mcts_thread_safety.py` | Full-suite run emitted `PytestUnknownMarkWarning` for the thread-safety stress test's `slow` marker. | `pytest` passed with one warning before config change. | Registered the `slow` marker in `pyproject.toml`; `pytest tests/test_mcts_thread_safety.py` now passes without the warning. | fixed |
| F006 | High | Training Quality | Selfplay | `chess_anti_engine/selfplay/manager.py`, `tests/test_selfplay_fraction.py` | Full-selfplay games annotated only one color's network turns with SF targets. The assigned net-color turn advanced to the selfplay-opponent network turn before any SF label query ran, so the earlier `_NetRecord` was no longer the latest record and never received `sf_wdl` / SF policy targets. | In `_run_step`, SF label queries were only submitted for `sp_opp_idxs` after `run_network_turn`; `net_idxs` that were also selfplay skipped annotation. A direct repro before the fix produced two selfplay samples with `[False, True]` for `s.sf_wdl is not None`. | Tightened `test_full_selfplay_generates_both_side_samples_and_no_pid_wdl_stats` to require all full-selfplay samples have `sf_wdl` and `sf_policy_target`. Verified selfplay/finalize slice: `33 passed`. | fixed |
| F007 | Medium | Reliability | Tablebase | `chess_anti_engine/tablebase.py`, `tests/test_tablebase_cache.py` | The shared Syzygy cache used one global handle and closed the previous handle whenever another path was requested. A UCI option swap or concurrent training/search probe with a different path could invalidate an in-flight caller's returned handle after `get_tablebase()` released its lock. | `get_tablebase()` returned the cached handle without a usage lock, while the different-path branch closed `_tablebase` before installing the new handle. Call sites probe the returned handle outside the cache lock. | Replaced the single global with a path-keyed cache that leaves existing handles open; added fake-tablebase regression proving path swaps do not close active handles and failed opens still close the temporary handle. Verified `tests/test_tablebase_cache.py` -> `2 passed`. | fixed |
| F008 | High | Training Quality / Reliability | Replay shards | `chess_anti_engine/replay/shard.py`, `tests/test_replay_shard_validation.py` | Optional shard targets could be marked present without storing the paired value array. Loading such a shard synthesized a zero-filled value and `arrays_to_samples()` treated it as a real SF/search/policy/mask target, silently corrupting auxiliary supervision. | `validate_arrays()` only checked `x`, `policy_target`, and `wdl_target`; `arrays_to_samples()` populated missing optional arrays with zeros and then trusted `has_*` flags from the shard. | Added optional field validation for flag shape, value shape, finite float values, and active flag without value. Verified replay shard/collation/disk-buffer slice: `33 passed`. | fixed |

## Review Passes

| Pass | Status | Goal | Main Outputs |
|------|--------|------|--------------|
| 0. Architecture map | complete | Map entrypoints, long-running loops, generated state, and cross-component contracts. | Component map, critical data-flow notes |
| 1. Chess semantics | complete | Verify board encoding, move encoding, legal move masking, color perspective, result labeling, and symmetry transforms. | Correctness findings, edge-case tests |
| 2. Search/selfplay targets | complete | Verify MCTS/Gumbel/PUCT behavior, terminal handling, visit targets, temperature sampling, Stockfish target alignment, and game adjudication. | Correctness/training-quality findings |
| 3. Replay/training/model | complete | Verify replay shards, augmentation, batching, loss targets, optimizer/trainer lifecycle, checkpoint/load/export paths. | Training-quality and reliability findings |
| 4. Distributed runtime | complete | Verify server, worker, worker pool, leases, asset cache, uploads/downloads, backpressure, auth, and cleanup. | Reliability/security/scalability findings |
| 5. UCI/runtime engine | complete | Verify UCI protocol behavior, time management, ponderhit, model loading, walker pool, and engine search integration. | Runtime correctness findings |
| 6. Efficiency/performance | complete | Inspect and profile hot paths for repeated conversions, poor batching, object churn, sync points, I/O stalls, and unbounded memory. | Static hot-path sweep, targeted test evidence |
| 7. Config/tune/scripts | complete | Verify YAML/CLI precedence, Tune/PBT behavior, shell/script assumptions, salvage/restart, and operational workflows. | Reliability/config findings |
| 8. Tests and gaps | complete | Review tests for semantic strength, missing regression coverage, and expensive or flaky cases. | Test-gap findings and proposed tests |
| 9. Triage and fix plan | complete | Group findings into fix batches by risk and subsystem. | Prioritized implementation plan |

## Initial Architecture Map

This is a lightweight map for orientation before starting the actual review.
Keep it updated when the implementation contradicts or refines these notes.

### Primary Runtime Modes

| Mode | Entrypoint | Expected Flow | Main Review Risks |
|------|------------|---------------|-------------------|
| Single-trial training | `python -m chess_anti_engine.run --mode train` | Parse YAML/CLI, create one distributed trial path, launch local server/worker pieces, selfplay, ingest replay, train, checkpoint. | Config precedence, local process lifecycle, checkpoint/replay consistency, Stockfish path requirements |
| Tune/PBT training | `python -m chess_anti_engine.run --mode tune` | Ray Tune harness launches trainables, publishes per-trial model manifests, workers upload shards, PBT/GPBT mutates configs/checkpoints. | Exploit/explore state copying, replay mixing, metric semantics, process cleanup, scalable I/O |
| Volunteer worker | `python -m chess_anti_engine.worker` / `worker_pool` | Poll manifest, download assets, run selfplay vs Stockfish, stage/upload shards, cache models/books/binaries. | Auth, asset cache atomicity, retries, stale manifests, upload validation, local oversubscription |
| HTTP server | `python -m chess_anti_engine.server.run_server` | Serve manifests/assets and accept authenticated shard uploads into trial inboxes. | Upload security, path traversal, blocking disk I/O, lease balancing, manifest consistency |
| UCI engine | `deepfin` / `python -m chess_anti_engine.uci` | Load checkpoint/model, handle UCI commands, run search, emit info/bestmove. | Protocol state, time controls, ponderhit, legal bestmove, search responsiveness |
| Scripts/ops | `scripts/*.py`, `scripts/*.sh` | Benchmark, profile, monitor, salvage, restart, bootstrap, and diagnose runs. | Stale assumptions, unsafe file/process handling, misleading benchmark evidence |

### Training Data Flow

| Step | Producers | Consumers | Contract To Verify |
|------|-----------|-----------|--------------------|
| Board position | `selfplay.manager`, `opening`, `stockfish`, `mcts` | `encoding`, `moves`, MCTS, replay shard writer | Position is recorded before the selected move, with correct side-to-move and legal move set. |
| Feature tensor | `encoding.encode`, C extensions | model, inference dispatcher, trainer collation | Shape, dtype, orientation, metadata planes, and Python/C parity stay consistent. |
| Policy target | MCTS visits, Stockfish MultiPV shaping, legal masks | replay, loss function | Policy indices match LC0 4672 mapping and are from the intended current-player perspective. |
| Value/WDL targets | game result, Stockfish WDL, timeout adjudication, tablebase | replay, losses, metrics | Result labels, WDL distribution, future targets, and value sign use one convention. |
| Replay shard | `replay.shard`, worker buffer, server upload | disk replay, trainable ingestion, trainer dataset | Schema, sparse policy encoding, metadata, validation, and version handling are robust. |
| Training batch | replay buffer/dataset/collate | `Trainer.train_steps`, losses | Tensors have expected shape/dtype/device and optional fields have safe defaults/masks. |
| Checkpoint/manifest | trainer/tune report/distributed runtime | workers, UCI loader, salvage/recovery | Model weights, optimizer/scheduler/SWA/PID/config metadata are complete and atomically published. |

### Critical Contract Boundaries

These boundaries should receive the deepest cross-file review because bugs here
are likely to be silent.

- `moves.encode` <-> `encoding` <-> `mcts`: policy index legality, board orientation, terminal positions.
- `selfplay.manager` <-> `replay.shard`: recorded board, selected move, policy target, WDL/value target, and metadata.
- `replay.dataset` <-> `train.losses`: optional target masks, sparse/dense policy reconstruction, dtype/device handling.
- `train.trainer` <-> `tune.trainable_report`: checkpoint state, metric reporting, best/latest model promotion.
- `tune.distributed_runtime` <-> `server.app` <-> `worker`: manifest freshness, upload/download atomicity, trial leases.
- `uci.engine` <-> `uci.search` <-> `mcts.gumbel_c`: UCI command state, search cancellation, bestmove legality, timing.

## Execution Order

The review should be thorough, but this order reduces the chance of missing
systemic bugs by checking data contracts before peripheral flows.

| Step | Component | Depth | Why First | Exit Criteria |
|------|-----------|-------|-----------|---------------|
| 1 | Architecture and contract map | deep | Establish shared conventions before judging implementation. | Core data-flow contracts are marked known/unknown, and open questions are recorded. |
| 2 | Encoding and move policy | deep | Most downstream components assume this is correct. | Python/C parity, legal move edge cases, mirroring, and mask/index semantics reviewed. |
| 3 | MCTS/search target production | deep | Silent target bugs can ruin training while tests still pass. | Value backup perspective, terminal handling, visit targets, and batching reviewed. |
| 4 | Selfplay result/record generation | deep | Connects gameplay to replay targets. | Result labeling, Stockfish WDL POV, temperature, openings, timeout adjudication reviewed. |
| 5 | Replay and collation | deep | Bad storage/collation can silently corrupt training batches. | Schema validation, sparse policy, augmentation, sampling, and disk behavior reviewed. |
| 6 | Losses/trainer/checkpoints | deep | Verifies the model is trained on the intended objective and can resume. | Target semantics, masks, optimizer/scheduler/SWA/checkpoint state reviewed. |
| 7 | Distributed server/worker/tune | deep | Operational reliability and scale risks are concentrated here. | Auth, leases, uploads, manifests, PBT state copying, cleanup, and retries reviewed. |
| 8 | UCI engine | deep | Separate user-facing runtime with timing/protocol risks. | Protocol state, time controls, ponderhit, search cancellation, bestmove legality reviewed. |
| 9 | Stockfish/tablebase integration | deep | External engine/tablebase semantics affect labels and search. | UCI parsing, pooling, PID direction, Syzygy overrides, cleanup reviewed. |
| 10 | Efficiency/performance pass | deep on hot paths, skim elsewhere | Performance bugs span components and need static plus runtime evidence. | Hot-path findings have benchmark/profile commands or static evidence. |
| 11 | Scripts/docs/tests | skim-to-deep by risk | Operational assumptions and test gaps should be checked after code contracts. | Stale docs/scripts and missing tests mapped to findings. |
| 12 | Triage | deep | Prevent scattered fixes. | Findings grouped into fix batches with verification commands. |

## First Review Slice

Start with this small slice when the actual review begins.

1. Mark architecture contracts as `known`, `unknown`, or `contradicted` after reading the key call sites.
2. Deep review `chess_anti_engine/moves/encode.py` with `tests/test_move_encoding_lc0_4672.py` and mirroring tests.
3. Deep review `chess_anti_engine/encoding/encode.py`, `lc0.py`, `features.py`, and C-extension parity tests.
4. Record any policy-index, board-orientation, legality, promotion, castling, en-passant, or mirroring findings.
5. Run only targeted encoding/move tests before moving to MCTS.

Suggested first commands:

```bash
python -m pytest tests/test_encoding_basic.py tests/test_encode_optimization.py tests/test_move_encoding_lc0_4672.py tests/test_mirror_augmentation.py tests/test_cboard_terminal.py
```

## Core Data-Flow Contracts

Track these contracts during every component review. Any uncertainty should
become either a finding or an explicit open question.

| Contract | Status | Notes |
|----------|--------|-------|
| Board state -> feature tensor has stable shape, dtype, orientation, and side-to-move semantics. | pending | |
| python-chess move -> LC0 4672 policy index is total for all legal moves and invalid for illegal moves. | pending | |
| Policy logits -> legal move probabilities masks illegal moves without changing index semantics. | pending | |
| MCTS visit counts -> training policy target preserves the current-player perspective. | pending | |
| Game result / Stockfish WDL -> value target uses the same perspective expected by loss/model. | pending | |
| Mirroring/augmentation transforms board features, moves, policy vectors, and value targets consistently. | pending | |
| Replay shard schema is versioned/validated and compatible with dataset collation. | pending | |
| Trainer checkpoint contains all state needed to resume without silent optimizer/scheduler/PID drift. | pending | |
| Server manifest/model/book/binary publishing is atomic enough for concurrent workers. | pending | |
| Worker uploads are authenticated, bounded, path-safe, and recoverable after transient failure. | pending | |
| UCI engine respects protocol, time controls, ponderhit clock side, and graceful shutdown. | pending | |

## Component Map

| Component | Risk | Review Depth | Files |
|-----------|------|--------------|-------|
| Entrypoints, config, packaging | High | pending | `chess_anti_engine/run.py`, `chess_anti_engine/worker.py`, `chess_anti_engine/worker_pool.py`, `chess_anti_engine/worker_config.py`, `chess_anti_engine/version.py`, `setup.py`, `pyproject.toml`, `configs/default.yaml`, `configs/pbt2_small.yaml` |
| Encoding and move policy | Critical | deep | `chess_anti_engine/encoding/*`, `chess_anti_engine/moves/*`, `chess_anti_engine/utils/bitboards.py` |
| Model and inference | High | in progress | `chess_anti_engine/model/*`, `chess_anti_engine/inference.py`, `chess_anti_engine/inference_dispatcher.py`, `chess_anti_engine/onnx/*` |
| MCTS/search | Critical | deep | `chess_anti_engine/mcts/*`, `chess_anti_engine/tablebase.py` |
| Selfplay and arena | Critical | in progress | `chess_anti_engine/selfplay/*`, `chess_anti_engine/arena.py` |
| Replay and training data | Critical | deep | `chess_anti_engine/replay/*`, `chess_anti_engine/worker_buffer.py` |
| Training | Critical | deep | `chess_anti_engine/train/*` |
| Distributed server | High | in progress | `chess_anti_engine/server/*` |
| Worker assets/cache/pool | High | in progress | `chess_anti_engine/worker_assets.py`, `chess_anti_engine/worker.py`, `chess_anti_engine/worker_pool.py`, `chess_anti_engine/worker_buffer.py`, `chess_anti_engine/worker_config.py` |
| Stockfish integration | High | in progress | `chess_anti_engine/stockfish/*` |
| Tune/PBT runtime | High | pending | `chess_anti_engine/tune/*` |
| UCI engine | High | deep | `chess_anti_engine/uci/*` |
| Evaluation and benchmarks | Medium | pending | `chess_anti_engine/eval/*`, `chess_anti_engine/bench/*`, `scripts/bench_*.py`, `scripts/profile_*.py`, `scripts/e2e_strength_test.py` |
| Operational scripts | Medium | pending | `scripts/*.sh`, `scripts/graceful_restart.py`, `scripts/generate_bootstrap.py`, `scripts/train_bootstrap.py`, `scripts/status.py`, `scripts/diagnose.py`, `scripts/pbt_*`, `scripts/*audit*`, `scripts/match_checkpoints.py`, `scripts/reinit_value_heads.py`, `scripts/blunder_check*.py` |
| Documentation/specs | Low | pending | `README.md`, `AGENTS.md`, `CLAUDE.md`, `prompt.md`, `spec.md`, `future_ideas.md`, `TRAINING_LOG.md`, `bench_results.md`, `docs/*`, `tcec.md` |
| Tests | High | pending | `tests/*` |

## Component Checklists

### Entrypoints, Config, Packaging

Files:

- [ ] `chess_anti_engine/run.py`
- [ ] `chess_anti_engine/worker.py`
- [ ] `chess_anti_engine/worker_pool.py`
- [ ] `chess_anti_engine/worker_config.py`
- [ ] `chess_anti_engine/version.py`
- [ ] `setup.py`
- [ ] `pyproject.toml`
- [ ] `configs/default.yaml`
- [ ] `configs/pbt2_small.yaml`

Correctness/reliability:

- [ ] YAML defaults and CLI overrides have deterministic precedence.
- [ ] Required operational arguments fail early with actionable errors.
- [ ] Mode selection matches documented behavior (`train` still uses local distributed path).
- [ ] Optional dependencies fail lazily and clearly.
- [ ] Runtime output paths avoid clobbering unrelated runs.

Efficiency/scalability:

- [ ] Startup does not import heavy optional stacks unless needed.
- [ ] Config parsing and worker launch do not repeatedly reload large assets.
- [ ] Default knobs do not create accidental CPU oversubscription.

Tests:

- [ ] `tests/test_trial_config.py`
- [ ] `tests/test_worker_config_yaml.py`
- [ ] `tests/test_train_import_lazy.py`
- [ ] `tests/test_run_bootstrap.py`
- [ ] `tests/test_tune_distributed_worker_cmd.py`

### Encoding and Move Policy

Status: `deep` for Python/C legal-move parity, policy-index roundtrips,
mirroring, LC0 feature shape/dtype/orientation smoke coverage, and CBoard
terminal/history tests. No correctness finding recorded in this slice.

Evidence:

- `python3 -m compileall -q chess_anti_engine tests scripts` passed.
- Before building extensions, `pytest ... tests/test_cboard_terminal.py` failed at collection because `_lc0_ext` was unavailable in the raw worktree.
- `python3 setup.py build_ext --inplace` succeeded in the isolated worktree.
- `pytest tests/test_encoding_basic.py tests/test_encode_optimization.py tests/test_move_encoding_lc0_4672.py tests/test_mirror_augmentation.py tests/test_cboard_terminal.py` passed: `44 passed`.
- Extra adversarial check: C legal-move indices matched Python fallback on 208 positions, including promotion, castling, en-passant, and random walks.

Files:

- [ ] `chess_anti_engine/encoding/__init__.py`
- [ ] `chess_anti_engine/encoding/encode.py`
- [ ] `chess_anti_engine/encoding/features.py`
- [ ] `chess_anti_engine/encoding/lc0.py`
- [ ] `chess_anti_engine/encoding/cboard_encode.py`
- [ ] `chess_anti_engine/encoding/_features_ext.c`
- [ ] `chess_anti_engine/encoding/_features_ext.pyi`
- [ ] `chess_anti_engine/encoding/_features_impl.h`
- [ ] `chess_anti_engine/encoding/_lc0_ext.c`
- [ ] `chess_anti_engine/encoding/_lc0_ext.pyi`
- [ ] `chess_anti_engine/encoding/_cboard_impl.h`
- [ ] `chess_anti_engine/moves/__init__.py`
- [ ] `chess_anti_engine/moves/encode.py`
- [ ] `chess_anti_engine/utils/bitboards.py`

Correctness:

- [ ] Feature tensor channels, dtype, board orientation, repetition/castling/en-passant semantics are documented or inferable.
- [ ] Python and C-extension encoders agree exactly for representative and edge positions.
- [ ] Legal move encoding covers promotions, underpromotions, castling, en passant, pins, check evasions, and terminal boards.
- [ ] LC0 4672 mapping is bijective for supported moves and rejects/handles unsupported moves predictably.
- [ ] Mirroring transforms policy indices consistently.

Efficiency:

- [ ] Hot encoding paths avoid avoidable board copies, Python loops, and per-move allocations.
- [ ] C-extension fallback behavior is clear and does not silently regress throughput.
- [ ] Tests or benchmarks compare Python/C extension speed where relevant.

Tests:

- [ ] `tests/test_encoding_basic.py`
- [ ] `tests/test_encode_optimization.py`
- [ ] `tests/test_move_encoding_lc0_4672.py`
- [ ] `tests/test_cboard_terminal.py`
- [ ] `tests/test_mirror_augmentation.py`

### Model and Inference

Files:

- [ ] `chess_anti_engine/model/__init__.py`
- [ ] `chess_anti_engine/model/tiny.py`
- [ ] `chess_anti_engine/model/transformer.py`
- [ ] `chess_anti_engine/inference.py`
- [ ] `chess_anti_engine/inference_dispatcher.py`
- [ ] `chess_anti_engine/onnx/__init__.py`
- [ ] `chess_anti_engine/onnx/export.py`

Correctness:

- [ ] Model outputs match expected policy/value/WDL/auxiliary target shapes.
- [ ] Device, dtype, autocast, eval/train mode, and no-grad boundaries are correct.
- [ ] Inference and training forward paths agree on input layout and output semantics.
- [ ] ONNX export preserves numerics within documented tolerances.

Efficiency:

- [ ] Batched inference avoids repeated `.to(device)`, `.cpu()`, `.numpy()`, or scalar syncs in hot paths.
- [ ] Dispatcher queues are bounded and coalesce requests without causing starvation.
- [ ] GPU/CPU fallback paths avoid excessive model reloads or reinitialization.

Tests:

- [ ] `tests/test_transformer_forward.py`
- [ ] `tests/test_inference_broker.py`
- [ ] `tests/test_gpu_dispatcher.py`
- [ ] `tests/test_multi_gpu_dispatcher.py`
- [ ] `tests/test_batch_coalescing.py`
- [ ] `tests/test_onnx_export_smoke.py`
- [ ] `tests/test_onnx_export_int8_smoke.py`

### MCTS and Search

Status: `deep` for Python Gumbel/PUCT, C-backed Gumbel/PUCT interfaces,
terminal value perspective, root legal masks, visit target extraction, and
selfplay call sites that pass precomputed logits and per-game budgets.

Evidence:

- `pytest tests/test_gumbel_mcts_smoke.py tests/test_gumbel_root_many_edge_cases.py tests/test_gumbel_budget_usage.py tests/test_mcts_c_tree.py tests/test_mcts_thread_safety.py tests/test_mcts_virtual_loss.py tests/test_progressive_budget.py tests/test_soft_policy.py tests/test_batch_process_ply.py` passed: `45 passed`.
- Initial warning: `tests/test_mcts_thread_safety.py` used unregistered `@pytest.mark.slow`; fixed in F005.

Files:

- [ ] `chess_anti_engine/mcts/__init__.py`
- [ ] `chess_anti_engine/mcts/gumbel.py`
- [ ] `chess_anti_engine/mcts/gumbel_c.py`
- [ ] `chess_anti_engine/mcts/puct.py`
- [ ] `chess_anti_engine/mcts/puct_c.py`
- [ ] `chess_anti_engine/mcts/_mcts_tree.c`
- [ ] `chess_anti_engine/mcts/_mcts_tree.pyi`
- [ ] `chess_anti_engine/tablebase.py`

Correctness/training quality:

- [ ] Search value signs are from the intended player perspective at every backup.
- [ ] Terminal, checkmate, stalemate, repetition, fifty-move, and tablebase states are handled consistently.
- [ ] Gumbel/PUCT selection, expansion, masking, visit counting, and target extraction align with tests/spec.
- [ ] Virtual loss/thread-safety behavior cannot corrupt visits or values.
- [ ] Temperature and budget behavior do not bias targets unexpectedly.

Efficiency:

- [ ] Model evals are batched where possible and not duplicated for the same state.
- [ ] Tree nodes avoid excessive Python object churn and repeated legal move generation.
- [ ] C-tree and Python tree behavior agree; fallback paths do not become accidental defaults.
- [ ] Instrumentation/logging is cheap enough for long selfplay runs.

Tests:

- [ ] `tests/test_gumbel_mcts_smoke.py`
- [ ] `tests/test_gumbel_root_many_edge_cases.py`
- [ ] `tests/test_gumbel_budget_usage.py`
- [ ] `tests/test_mcts_c_tree.py`
- [ ] `tests/test_mcts_thread_safety.py`
- [ ] `tests/test_mcts_virtual_loss.py`
- [ ] `tests/test_progressive_budget.py`
- [ ] `tests/test_soft_policy.py`

### Selfplay and Arena

Status: `in progress`. Result labeling, timeout adjudication helper behavior,
Stockfish WDL POV flip, temperature/opening tests, continuous play, arena/match
labeling, and selfplay fraction tests have been run. One test/API drift finding
was fixed as F001.

Evidence:

- Initial `pytest ... tests/test_play_batch_helpers.py ...` failed at collection with the missing helper imports recorded in F001.
- After fix, `pytest tests/test_play_batch_helpers.py` passed: `16 passed`.
- After fix, `pytest tests/test_selfplay_result_labeling.py tests/test_selfplay_sf_wdl_pov.py tests/test_selfplay_timeout_adjudication.py tests/test_selfplay_fraction.py tests/test_threaded_selfplay.py tests/test_play_batch_continuous.py tests/test_temperature_schedule.py tests/test_opening_start_positions.py tests/test_arena_match_smoke.py tests/test_match_result_labeling.py tests/test_bench_result_labeling.py` passed: `38 passed`.

Files:

- [ ] `chess_anti_engine/selfplay/__init__.py`
- [ ] `chess_anti_engine/selfplay/budget.py`
- [ ] `chess_anti_engine/selfplay/config.py`
- [ ] `chess_anti_engine/selfplay/game.py`
- [ ] `chess_anti_engine/selfplay/manager.py`
- [ ] `chess_anti_engine/selfplay/match.py`
- [ ] `chess_anti_engine/selfplay/opening.py`
- [ ] `chess_anti_engine/selfplay/temperature.py`
- [ ] `chess_anti_engine/arena.py`

Correctness/training quality:

- [ ] Game results, timeout adjudication, Stockfish WDL labels, and final value targets share one perspective convention.
- [ ] Selfplay records match the board state before the selected move and the policy target for that move.
- [ ] Opening book/random-start positions preserve legal side-to-move and avoid accidental result leakage.
- [ ] Temperature schedule and sampling behavior match training intent.
- [ ] Arena/latest-vs-best decisions cannot promote broken checkpoints due to mislabeling.

Efficiency:

- [ ] Selfplay loop amortizes Stockfish/model calls and avoids repeated feature encoding where possible.
- [ ] Batching knobs do not cause unbounded latency or memory use.
- [ ] Manager shutdown and restart avoid orphaned processes and leaked workers.

Tests:

- [ ] `tests/test_selfplay_result_labeling.py`
- [ ] `tests/test_selfplay_sf_wdl_pov.py`
- [ ] `tests/test_selfplay_timeout_adjudication.py`
- [ ] `tests/test_selfplay_fraction.py`
- [ ] `tests/test_threaded_selfplay.py`
- [ ] `tests/test_play_batch_continuous.py`
- [ ] `tests/test_play_batch_helpers.py`
- [ ] `tests/test_batch_process_ply.py`
- [ ] `tests/test_temperature_schedule.py`
- [ ] `tests/test_opening_start_positions.py`
- [ ] `tests/test_arena_match_smoke.py`
- [ ] `tests/test_match_result_labeling.py`
- [ ] `tests/test_bench_result_labeling.py`

### Replay and Training Data

Status: `deep` for replay sample schema, sparse/dense policy and mask storage,
array collation, augmentation, shard validation, disk replay sampling, and
worker upload shard handling covered by current tests.

Evidence:

- `pytest tests/test_replay_disk_buffer.py tests/test_replay_shard_npz.py tests/test_replay_shard_validation.py tests/test_replay_surprise_sampling.py tests/test_collation.py tests/test_shard_path_iter.py tests/test_worker_small_uploads.py tests/test_is_selfplay_tagging.py tests/test_mirror_augmentation.py` passed: `56 passed`.

Files:

- [ ] `chess_anti_engine/replay/__init__.py`
- [ ] `chess_anti_engine/replay/augment.py`
- [ ] `chess_anti_engine/replay/buffer.py`
- [ ] `chess_anti_engine/replay/dataset.py`
- [ ] `chess_anti_engine/replay/disk_buffer.py`
- [ ] `chess_anti_engine/replay/shard.py`
- [ ] `chess_anti_engine/worker_buffer.py`

Correctness/training quality:

- [ ] Shard schemas are validated before ingestion and have version compatibility checks.
- [ ] Augmentation transforms all coupled fields together.
- [ ] Replay sampling preserves intended mixture, surprise weighting, freshness, and train/selfplay tagging.
- [ ] Collation preserves dtype, shape, device expectations, and target semantics.
- [ ] Disk-buffer compaction cannot delete active or partially uploaded shards.

Efficiency/scalability:

- [ ] Sampling avoids loading full shards repeatedly when indexing would suffice.
- [ ] Compaction, validation, and upload staging are bounded for long runs.
- [ ] Memory mapping/cache behavior is explicit for large replay buffers.

Tests:

- [ ] `tests/test_replay_disk_buffer.py`
- [ ] `tests/test_replay_shard_npz.py`
- [ ] `tests/test_replay_shard_validation.py`
- [ ] `tests/test_replay_surprise_sampling.py`
- [ ] `tests/test_collation.py`
- [ ] `tests/test_shard_path_iter.py`
- [ ] `tests/test_worker_small_uploads.py`
- [ ] `tests/test_is_selfplay_tagging.py`

### Training

Status: `deep` for loss-mask semantics, optional target handling, trainer
warmup/checkpoint/SWA smoke behavior, optimizer smoke tests, transformer
forward shape behavior, and feature dropout.

Evidence:

- `pytest tests/test_compute_loss.py tests/test_losses.py tests/test_hlgauss_target.py tests/test_volatility_target.py tests/test_trainer_warmup.py tests/test_warmup_lr_schedule.py tests/test_training_e2e.py tests/test_swa_export.py tests/test_cosmos_optimizer.py tests/test_cosmos_fast_optimizer.py tests/test_muon_optimizer.py tests/test_transformer_forward.py tests/test_feature_dropout.py` passed: `45 passed`.

Files:

- [ ] `chess_anti_engine/train/__init__.py`
- [ ] `chess_anti_engine/train/cosmos.py`
- [ ] `chess_anti_engine/train/cosmos_fast.py`
- [ ] `chess_anti_engine/train/losses.py`
- [ ] `chess_anti_engine/train/muon.py`
- [ ] `chess_anti_engine/train/targets.py`
- [ ] `chess_anti_engine/train/trainer.py`

Correctness/training quality:

- [ ] Loss functions consume targets with the same semantics produced by selfplay/replay.
- [ ] WDL/value/volatility/HL-Gauss targets are numerically stable and perspective-correct.
- [ ] Optimizer, scheduler, SWA, warmup, checkpoint save/load, and resume restore all required state.
- [ ] Mixed precision and gradient scaling are safe across devices.

Efficiency:

- [ ] Training step avoids unnecessary host/device transfers and repeated target construction.
- [ ] DataLoader and replay sampling do not starve the GPU.
- [ ] Optimizers avoid avoidable Python overhead in tight update paths.

Tests:

- [ ] `tests/test_compute_loss.py`
- [ ] `tests/test_losses.py`
- [ ] `tests/test_hlgauss_target.py`
- [ ] `tests/test_volatility_target.py`
- [ ] `tests/test_trainer_warmup.py`
- [ ] `tests/test_warmup_lr_schedule.py`
- [ ] `tests/test_training_e2e.py`
- [ ] `tests/test_swa_export.py`
- [ ] `tests/test_cosmos_optimizer.py`
- [ ] `tests/test_cosmos_fast_optimizer.py`
- [ ] `tests/test_muon_optimizer.py`

### Distributed Server, Workers, and Assets

Status: `in progress`. Server upload security/compaction, trial leases,
distributed backpressure, worker pool/cached assets/config, small uploads, E2E
smoke, and Tune worker command tests have been run. F002 records an open
durability risk in upload compaction.

Evidence:

- `pytest tests/test_server_upload_security.py tests/test_server_upload_compaction.py tests/test_server_trial_lease.py tests/test_distributed_selfplay_backpressure.py tests/test_worker_pool.py tests/test_worker_cached_assets.py tests/test_worker_config_yaml.py tests/test_worker_small_uploads.py tests/test_e2e_smoke.py tests/test_tune_distributed_worker_cmd.py tests/test_trial_config.py` passed: `70 passed`.

Files:

- [ ] `chess_anti_engine/server/__init__.py`
- [ ] `chess_anti_engine/server/app.py`
- [ ] `chess_anti_engine/server/auth.py`
- [ ] `chess_anti_engine/server/lease.py`
- [ ] `chess_anti_engine/server/manage_users.py`
- [ ] `chess_anti_engine/server/run_server.py`
- [ ] `chess_anti_engine/worker.py`
- [ ] `chess_anti_engine/worker_assets.py`
- [ ] `chess_anti_engine/worker_buffer.py`
- [ ] `chess_anti_engine/worker_config.py`
- [ ] `chess_anti_engine/worker_pool.py`

Correctness/reliability/security:

- [ ] Upload endpoints authenticate correctly and validate shard size, path, archive, and schema.
- [ ] Downloads and asset cache writes are atomic under concurrent workers.
- [ ] Leases are sticky enough for load balance but do not strand workers forever.
- [ ] Worker retry/backoff handles transient server/network/model errors without corrupting local state.
- [ ] Worker pool startup/shutdown handles child failures and signal propagation.
- [ ] Credentials are not logged and saved config honors opt-out flags.

Efficiency/scalability:

- [ ] Workers do not repeatedly download identical models/books/binaries.
- [ ] Server upload handling avoids blocking the event loop on heavy validation or disk I/O.
- [ ] Backpressure prevents unbounded queues, upload accumulation, and trainer starvation.

Tests:

- [ ] `tests/test_server_upload_security.py`
- [ ] `tests/test_server_upload_compaction.py`
- [ ] `tests/test_server_trial_lease.py`
- [ ] `tests/test_distributed_selfplay_backpressure.py`
- [ ] `tests/test_worker_pool.py`
- [ ] `tests/test_worker_cached_assets.py`
- [ ] `tests/test_worker_config_yaml.py`
- [ ] `tests/test_worker_small_uploads.py`
- [ ] `tests/test_e2e_smoke.py`
- [ ] `tests/test_profile_distributed.py`

### Stockfish Integration

Status: `in progress`. PID direction/state tests and worker-pool overlap have
been run. Static review notes that `StockfishUCI` uses blocking reads without
an explicit timeout; F004 records the open reliability finding.

Evidence:

- Included in UCI/Stockfish command below: `tests/test_pid_inverse_regret.py`, `tests/test_difficulty_state.py`, `tests/test_worker_pool.py`.

Files:

- [ ] `chess_anti_engine/stockfish/__init__.py`
- [ ] `chess_anti_engine/stockfish/pid.py`
- [ ] `chess_anti_engine/stockfish/pool.py`
- [ ] `chess_anti_engine/stockfish/uci.py`

Correctness/reliability:

- [ ] UCI command sequencing handles engine startup, readiness, position updates, MultiPV, limits, stop, and quit.
- [ ] PID difficulty controller updates in the intended direction and clamps correctly.
- [ ] Pool cleanup cannot leak Stockfish subprocesses under exceptions/timeouts.
- [ ] Stockfish WDL/probability parsing is robust to missing or variant info fields.

Efficiency:

- [ ] Pooling avoids per-move engine startup.
- [ ] MultiPV/node/time settings do not accidentally multiply CPU cost beyond config.
- [ ] Blocking reads have timeouts and do not stall all selfplay.

Tests:

- [ ] `tests/test_pid_inverse_regret.py`
- [ ] `tests/test_difficulty_state.py`
- [ ] `tests/test_worker_pool.py`

### Tune and PBT Runtime

Files:

- [ ] `chess_anti_engine/tune/__init__.py`
- [ ] `chess_anti_engine/tune/_utils.py`
- [ ] `chess_anti_engine/tune/distributed_runtime.py`
- [ ] `chess_anti_engine/tune/gpbt.py`
- [ ] `chess_anti_engine/tune/harness.py`
- [ ] `chess_anti_engine/tune/process_cleanup.py`
- [ ] `chess_anti_engine/tune/recovery.py`
- [ ] `chess_anti_engine/tune/replay_exchange.py`
- [ ] `chess_anti_engine/tune/trainable.py`
- [ ] `chess_anti_engine/tune/trainable_config_ops.py`
- [ ] `chess_anti_engine/tune/trainable_init.py`
- [ ] `chess_anti_engine/tune/trainable_metrics.py`
- [ ] `chess_anti_engine/tune/trainable_phases.py`
- [ ] `chess_anti_engine/tune/trainable_report.py`
- [ ] `chess_anti_engine/tune/trial_config.py`

Correctness/training quality/reliability:

- [ ] PBT exploit/explore copies only intended state and does not mix incompatible replay/checkpoints.
- [ ] Metrics reported to Ray match actual trial state and promotion criteria.
- [ ] Recovery/salvage behavior cannot resume with stale manifests or wrong optimizer/PID state unless requested.
- [ ] Pause/restart and cleanup logic handles partial iterations.

Efficiency/scalability:

- [ ] Trainer/server/worker local orchestration avoids orphaned processes and port conflicts.
- [ ] Metrics and checkpointing frequency do not dominate training time.
- [ ] Replay exchange does not repeatedly copy unchanged large files.

Tests:

- [ ] `tests/test_trial_config.py`
- [ ] `tests/test_tune_distributed_worker_cmd.py`
- [ ] `tests/test_e2e_smoke.py`

### UCI Engine

Status: `deep` for protocol parsing/formatting, UCI smoke behavior, time
manager, ponderhit clock-side regression, and walker pool tests.

Evidence:

- `pytest tests/test_uci_protocol.py tests/test_uci_smoke.py tests/test_uci_time_manager.py tests/test_uci_ponderhit_clock.py tests/test_uci_walker_pool.py tests/test_pid_inverse_regret.py tests/test_difficulty_state.py tests/test_worker_pool.py` passed: `78 passed`.

Files:

- [ ] `chess_anti_engine/uci/__init__.py`
- [ ] `chess_anti_engine/uci/__main__.py`
- [ ] `chess_anti_engine/uci/engine.py`
- [ ] `chess_anti_engine/uci/model_loader.py`
- [ ] `chess_anti_engine/uci/protocol.py`
- [ ] `chess_anti_engine/uci/score.py`
- [ ] `chess_anti_engine/uci/search.py`
- [ ] `chess_anti_engine/uci/subprocess_client.py`
- [ ] `chess_anti_engine/uci/time_manager.py`
- [ ] `chess_anti_engine/uci/walker_pool.py`

Correctness/reliability:

- [ ] UCI parser accepts common GUI command sequences and rejects malformed input safely.
- [ ] `go`, `stop`, `ponderhit`, `isready`, `ucinewgame`, and `quit` state transitions are correct.
- [ ] Time manager uses the correct side clock/increment after move and ponder transitions.
- [ ] Search returns legal best moves and reasonable scores under empty/terminal/legal-limited states.
- [ ] Subprocess client handles child engine errors and shutdown.

Efficiency:

- [ ] Walker pool/model loading is reused across searches.
- [ ] Per-search allocations and feature encodes are bounded.
- [ ] Stop/ponderhit responsiveness is not blocked by long synchronous work.

Tests:

- [ ] `tests/test_uci_protocol.py`
- [ ] `tests/test_uci_smoke.py`
- [ ] `tests/test_uci_time_manager.py`
- [ ] `tests/test_uci_ponderhit_clock.py`
- [ ] `tests/test_uci_walker_pool.py`

### Evaluation, Benchmarks, and Operational Scripts

Files:

- [ ] `chess_anti_engine/eval/__init__.py`
- [ ] `chess_anti_engine/eval/puzzles.py`
- [ ] `chess_anti_engine/bench/__init__.py`
- [ ] `chess_anti_engine/bench/play_batch_timing.py`
- [ ] `scripts/bench_*.py`
- [ ] `scripts/profile_*.py`
- [ ] `scripts/bench_batch_wait.sh`
- [ ] `scripts/cuda_sanity_check.py`
- [ ] `scripts/deepfin`
- [ ] `scripts/deepfin.bat`
- [ ] `scripts/diagnose.py`
- [ ] `scripts/e2e_strength_test.py`
- [ ] `scripts/generate_bootstrap.py`
- [ ] `scripts/graceful_restart.py`
- [ ] `scripts/hourly_pbt_audit.sh`
- [ ] `scripts/lint.sh`
- [ ] `scripts/match_checkpoints.py`
- [ ] `scripts/monitor_pbt.sh`
- [ ] `scripts/pbt_30m_poll.py`
- [ ] `scripts/pbt_hourly_audit.py`
- [ ] `scripts/poll_pbt_30m.sh`
- [ ] `scripts/reinit_value_heads.py`
- [ ] `scripts/status.py`
- [ ] `scripts/train.sh`
- [ ] `scripts/train_bootstrap.py`
- [ ] `scripts/blunder_check.py`
- [ ] `scripts/blunder_check_cp.py`

Correctness/reliability:

- [ ] Scripts run from repo root and fail clearly when required files/env vars are absent.
- [ ] Shell scripts quote paths and handle PID/log/stale-process states safely.
- [ ] Benchmarks measure what their names claim and do not accidentally benchmark setup overhead.
- [ ] Operational scripts avoid deleting or overwriting unrelated run data.

Efficiency:

- [ ] Benchmark/profiling scripts include warmup, stable iteration counts, and useful output.
- [ ] Polling scripts use reasonable intervals and avoid expensive repeated scans.

Tests:

- [ ] `tests/test_arena_match_smoke.py`
- [ ] `tests/test_profile_distributed.py`

### Documentation and Specs

Files:

- [ ] `README.md`
- [ ] `AGENTS.md`
- [ ] `CLAUDE.md`
- [ ] `prompt.md`
- [ ] `spec.md`
- [ ] `future_ideas.md`
- [ ] `TRAINING_LOG.md`
- [ ] `bench_results.md`
- [ ] `docs/experiments/value_head_arch.md`
- [ ] `tcec.md`

Review:

- [ ] Docs match CLI behavior, config names, and current architecture.
- [ ] Operational instructions do not recommend unsafe or stale workflows.
- [ ] Training notes/specs do not conflict with implemented target semantics.

### Tests

Status: `in progress`. Full suite passes on the review branch after F001 fix.

Evidence:

- Before F005, `pytest` passed with one warning: `417 passed, 1 warning in 68.66s`.
- After registering the marker, `pytest tests/test_mcts_thread_safety.py` passed: `5 passed`.
- Final full-suite verification passed without warnings: `417 passed in 77.97s`.

Files:

- [ ] `tests/__init__.py`
- [ ] `tests/test_arena_match_smoke.py`
- [ ] `tests/test_atomic_io.py`
- [ ] `tests/test_batch_coalescing.py`
- [ ] `tests/test_batch_process_ply.py`
- [ ] `tests/test_bench_result_labeling.py`
- [ ] `tests/test_cboard_terminal.py`
- [ ] `tests/test_collation.py`
- [ ] `tests/test_compute_loss.py`
- [ ] `tests/test_cosmos_fast_optimizer.py`
- [ ] `tests/test_cosmos_optimizer.py`
- [ ] `tests/test_difficulty_state.py`
- [ ] `tests/test_distributed_selfplay_backpressure.py`
- [ ] `tests/test_e2e_smoke.py`
- [ ] `tests/test_encode_optimization.py`
- [ ] `tests/test_encoding_basic.py`
- [ ] `tests/test_feature_dropout.py`
- [ ] `tests/test_gpu_dispatcher.py`
- [ ] `tests/test_gumbel_budget_usage.py`
- [ ] `tests/test_gumbel_mcts_smoke.py`
- [ ] `tests/test_gumbel_root_many_edge_cases.py`
- [ ] `tests/test_hlgauss_target.py`
- [ ] `tests/test_inference_broker.py`
- [ ] `tests/test_is_selfplay_tagging.py`
- [ ] `tests/test_losses.py`
- [ ] `tests/test_match_result_labeling.py`
- [ ] `tests/test_mcts_c_tree.py`
- [ ] `tests/test_mcts_thread_safety.py`
- [ ] `tests/test_mcts_virtual_loss.py`
- [ ] `tests/test_mirror_augmentation.py`
- [ ] `tests/test_move_encoding_lc0_4672.py`
- [ ] `tests/test_multi_gpu_dispatcher.py`
- [ ] `tests/test_muon_optimizer.py`
- [ ] `tests/test_onnx_export_int8_smoke.py`
- [ ] `tests/test_onnx_export_smoke.py`
- [ ] `tests/test_opening_start_positions.py`
- [ ] `tests/test_pid_inverse_regret.py`
- [ ] `tests/test_play_batch_continuous.py`
- [ ] `tests/test_play_batch_helpers.py`
- [ ] `tests/test_profile_distributed.py`
- [ ] `tests/test_progressive_budget.py`
- [ ] `tests/test_replay_disk_buffer.py`
- [ ] `tests/test_replay_shard_npz.py`
- [ ] `tests/test_replay_shard_validation.py`
- [ ] `tests/test_replay_surprise_sampling.py`
- [ ] `tests/test_run_bootstrap.py`
- [ ] `tests/test_selfplay_fraction.py`
- [ ] `tests/test_selfplay_result_labeling.py`
- [ ] `tests/test_selfplay_sf_wdl_pov.py`
- [ ] `tests/test_selfplay_timeout_adjudication.py`
- [ ] `tests/test_server_trial_lease.py`
- [ ] `tests/test_server_upload_compaction.py`
- [ ] `tests/test_server_upload_security.py`
- [ ] `tests/test_shard_path_iter.py`
- [ ] `tests/test_soft_policy.py`
- [ ] `tests/test_swa_export.py`
- [ ] `tests/test_temperature_schedule.py`
- [ ] `tests/test_threaded_selfplay.py`
- [ ] `tests/test_train_import_lazy.py`
- [ ] `tests/test_trainer_warmup.py`
- [ ] `tests/test_training_e2e.py`
- [ ] `tests/test_transformer_forward.py`
- [ ] `tests/test_trial_config.py`
- [ ] `tests/test_tune_distributed_worker_cmd.py`
- [ ] `tests/test_uci_ponderhit_clock.py`
- [ ] `tests/test_uci_protocol.py`
- [ ] `tests/test_uci_smoke.py`
- [ ] `tests/test_uci_time_manager.py`
- [ ] `tests/test_uci_walker_pool.py`
- [ ] `tests/test_volatility_target.py`
- [ ] `tests/test_warmup_lr_schedule.py`
- [ ] `tests/test_worker_cached_assets.py`
- [ ] `tests/test_worker_config_yaml.py`
- [ ] `tests/test_worker_pool.py`
- [ ] `tests/test_worker_small_uploads.py`

Review:

- [ ] Identify tests that only verify shapes/imports but not semantic invariants.
- [ ] Identify core behavior covered only by smoke tests.
- [ ] Identify tests that are flaky, time-dependent, network-dependent, or too slow for regular use.
- [ ] Map each high/critical finding to a regression test before fixing.

## Review Procedure

Use this procedure for each component.

1. Read the component's public API and call sites.
2. Write down the component's contract in the component notes.
3. Read implementation files and compare them to the contract.
4. Read existing tests and identify whether they catch the suspected failure modes.
5. Record findings immediately in the findings table.
6. Mark files as `skim` or `deep` only after the call sites and tests have been checked.
7. Defer fixes unless the issue is tiny, obvious, local, and can be tested immediately.

## Efficiency Review Procedure

Efficiency issues are bugs when they materially affect training throughput,
resource use, scale behavior, or iteration speed.

Static checks:

- [ ] Search for repeated `.to(`, `.cpu()`, `.numpy()`, `.item()`, tensor constructors, and board copies in loops.
- [ ] Search for full-shard loads, directory scans, JSON/YAML parsing, model loads, and checkpoint writes in hot loops.
- [ ] Inspect queues, buffers, dicts, and lists for bounded growth.
- [ ] Inspect locks, blocking I/O, subprocess reads, and network calls in async/threaded paths.
- [ ] Inspect logging and metrics for expensive per-position/per-node work.

Runtime checks:

- [ ] Measure encoding positions/sec on representative positions.
- [ ] Measure MCTS nodes/sec and model evals/sec with Python and C paths.
- [ ] Measure selfplay games/sec for small fixed configs.
- [ ] Measure training step time and DataLoader/replay wait time.
- [ ] Measure worker/server upload and manifest polling overhead.
- [ ] Compare 1 worker vs multiple workers for throughput and backpressure.

Suggested profiling commands:

```bash
python -m pytest tests/test_encode_optimization.py
python scripts/profile_mcts.py
python scripts/profile_mcts_detail.py
python scripts/profile_selfplay.py
python scripts/profile_training.py
python scripts/profile_distributed.py
python scripts/bench_multi_worker.py
python scripts/bench_pipeline.py
python scripts/bench_uci_engine.py
```

Record benchmark evidence in findings when possible. If a script requires
Stockfish, CUDA, or a running server, record the missing prerequisite instead
of treating the script as failed.

## Baseline Verification Commands

Use targeted tests during component review and broader tests after fix batches.

Fast syntax/import confidence:

```bash
python3 -m py_compile $(rg --files -g '*.py' chess_anti_engine tests scripts)
python -m pytest tests/test_train_import_lazy.py
```

Core semantic slices:

```bash
python -m pytest tests/test_encoding_basic.py tests/test_move_encoding_lc0_4672.py tests/test_mirror_augmentation.py
python -m pytest tests/test_gumbel_mcts_smoke.py tests/test_mcts_c_tree.py tests/test_mcts_virtual_loss.py
python -m pytest tests/test_selfplay_result_labeling.py tests/test_selfplay_sf_wdl_pov.py tests/test_batch_process_ply.py
python -m pytest tests/test_replay_shard_validation.py tests/test_replay_disk_buffer.py tests/test_collation.py
python -m pytest tests/test_compute_loss.py tests/test_losses.py tests/test_training_e2e.py
python -m pytest tests/test_server_upload_security.py tests/test_worker_pool.py tests/test_e2e_smoke.py
python -m pytest tests/test_uci_protocol.py tests/test_uci_time_manager.py tests/test_uci_ponderhit_clock.py
```

Full suite:

```bash
python -m pytest
```

Static analysis when dependencies are installed:

```bash
ruff check .
pyright
pylint chess_anti_engine
```

## Current Focus

Component: first full review pass complete

Goal: fix the high-signal tracked-test regression, record open reliability findings, and keep the runtime checkout untouched.

Last updated: 2026-04-24

## Triage Queue

Use this section after findings are recorded.

| Batch | Status | Findings | Fix Strategy | Verification |
|-------|--------|----------|--------------|--------------|
| A. Critical correctness | pending | | | |
| B. Training target quality | pending | | | |
| C. Runtime/distributed reliability | pending | | | |
| D. Efficiency hot paths | pending | | | |
| E. Security hardening | pending | | | |
| F. Test gaps | pending | | | |

## Open Questions

- [ ] Which runs/configs are considered operationally important enough to preserve behavior exactly?
- [ ] What minimum hardware baseline should efficiency findings use: CPU-only, single CUDA GPU, or current production host?
- [ ] Which Stockfish version/path should be treated as the review baseline?
- [ ] Are benchmark regressions findings only after measurement, or should obvious hot-path issues be recorded from static review?
