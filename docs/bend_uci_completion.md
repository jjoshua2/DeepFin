# Bend UCI engine completion contract

Status at 2026-10-01: **in progress**, not a qualified replacement for production.
Training stays in Python. The deliverable is a native UCI engine whose protocol,
game/history state, legal moves, features, legal-policy conversion, search and
session control are Bend-owned. Python may export a trained model and serve as an
external test oracle; the engine must not invoke a Python chess/search/evaluation
shim at runtime.

Native C++ LibTorch/AOTI tensor execution is an explicit dependency boundary.
This plan does not claim that transformer mathematics, CUDA kernels or training
are Bend-authored. If that boundary changes, it needs a separate design and
qualification rather than relabeling the existing backend.

## Current executable paths, traced from their callers

| Product | Actual path | Present capability and important limit |
| --- | --- | --- |
| Standalone material UCI | `standalone/main.bend` → `Protocol`/`TimeControl` → `Search` | Bend state/legal/history/search; bounded 4096-node fresh trees; no trained model |
| Standalone neural UCI | Same main → `EngineRuntime` → `NativeEvaluation` → `model_bridge.cpp` | Exact-bound CPU F32 batch 1; opt-in single-forward async; no Python runtime |
| Persistent live roots | `multi_root/live.bend` → `LiveRun` | Separate headless CPU-F32 product, 1–16 roots, generations, FIFO, one physical batch slot |
| CUDA/BF16 tensor backend | `batch_backend/main.bend` → batch effect → `cuda_execution.cpp` | Separate synchronous probe; not device-qualified and not reachable from UCI's singleton opener |

The live-root runner does not establish shared-tree multi-walker UCI parity.
The CUDA binding explicitly rejects singleton UCI use. Neither can be counted as
a completed UCI integration merely because its component tests pass.

The repository contains historical, append-only progress sections. In particular,
early standalone sections saying there is no encoder/model path were superseded
by the later complete-input and selected-leaf native composition. Runtime callers
and their actual readouts are authoritative. No historical evidence is upgraded
by this document.

## First repository milestone: actual UCI clock controls

`TimeControl.bend` parses wtime/btime/winc/binc/movestogo without changing the
existing Search or native-inference ABI. It selects the clock from the accepted
root's turn and feeds the resulting milliseconds into the real search loop's
deadline, including its async-pending cancellation path.

- Times/increments are unsigned milliseconds, 0..86,400,000; movestogo is 1..1000
- Default horizon is 30 moves; allocation is min(60,000, remaining minus a
  50 ms reserve, remaining/horizon + 3/4 increment), rounded with integer division
- A positive usable clock receives at least 1 ms; a clock at/below the reserve
  yields a legal unsearched fallback with zero selections/forwards
- The active side's clock is required when any clock field is supplied; the
  other side's clock is optional. Duplicate/missing/invalid values are rejected
- Explicit movetime wins after all supplied syntax is validated; clocks with
  infinite are rejected. Explicit nodes/depth/evals still constrain the search
- Without explicit nodes, a timed search uses the existing 65,536-simulation
  safety ceiling. Arena/horizon exhaustion can finish sooner than the clock
- `neural_work.movetime_ms` records the actual allocation. The existing
  cooperative deadline is not a hard real-time latency guarantee

The GitHub Actions **Bend UCI engine** job compiles the real main in generic and
UBSan modes, runs a standard independent UCI client with clocks, checks both turns,
increments, movetime precedence, zero-work fallback and malformed commands, then
runs the existing position/history/perft/stop/rules regressions. No Python engine
is launched inside the candidate. Its report starts unqualified and becomes
qualified only after all assertions pass. Until exact-head CI passes, these are
planned checks, not claimed results.

## Remaining acceptance gates, in dependency order

1. Complete native UCI session contract: advertised and effective setoption
   settings; clock overhead; go ponder/ponderhit with a fresh post-hit clock;
   searchmoves root restriction; explicit unsupported options; transactional
   position/ucinewgame; stop/quit exactly once and physical retirement. Supported
   maximum history/search/arena bounds must be explicit and suitable for games.
   The native bridge's current process-wide 65,536-forward diagnostic limit must
   be removed or replaced with a tournament-suitable, checked lifecycle
2. Search integration: configurable arenas and useful continuation beyond the
   initial diagnostic horizon; native PUCT parameters and selection/backup
   semantics; shared-tree walkers if required for the production comparison.
   Independent headless roots are not a substitute for shared-tree concurrency
3. Chess outcome contract: complete legal generator proof on the separate track,
   history-sensitive repetition and 50/75-move semantics, claim choices versus
   automatic draws, mate precedence and legal terminal/fallback behavior.
   No invented UCI resign/claim move: the GUI owns adjudication unless an
   explicitly documented tournament interface is added
4. Connect the CUDA/BF16 native backend to Bend-owned search/session scheduling,
   with logical cancellation distinct from physical completion, stale identity
   rejection, exact per-row histories, bounded queues and device failure handling.
   Multiple GPUs require a later explicit dispatcher/ownership contract
5. Package and deploy one trusted trained checkpoint as below. Qualify actual
   chess inputs and complete legal policy/WDL outputs against independent
   references; confirm that the isolated engine process has no Python dependency
6. Separately measure single-GPU and later multi-GPU memory, time to bestmove,
   accepted/wasted rows and useful EPS. Compare against the actual Python PUCT
   baseline using its realized settings, not historical prose defaults. Equal
   simulations, equal neural rows and equal wall time answer different questions.
   Tournament strength needs a preregistered game/match protocol and uncertainty

Passing a component gate closes only that gate. User-requested completion is not
established until the runnable native product and real trained-model target pass.

## Python-exported trained model contract

The exporter is `neural_probe.checkpoint.load_checkpoint/export_checkpoint`.
The input must be an immutable trusted `trainer.pt` (or explicit file), selected
`model` or `swa_model` weights, embedded supported architecture schema, exact
weights, compact `lc0_1858` policy, root or root-legacy-meta history,
`v1`/146 or `v2_threats`/175 features and history_rep_fix=true. Unsupported
metadata is rejected; no guessed architecture, missing layer initialization or
silent weight/model fallback. Training provenance must be supplied independently;
export does not infer that weights were trained.

Each package needs its matching v3 JSON sidecar, package SHA256, checkpoint SHA256,
selected weights key, complete encoding, fixed batch, explicit device/index/dtype
and exact release Torch build. CPU UCI currently requires F32 batch one.
CUDA requires BF16 and matching CUDA/Torch/LibTorch/toolkit/device compatibility,
with the same intended CUDA_VISIBLE_DEVICES mapping. An AOTI package contains
executable code and must come from a trusted producer.

Build dependencies: the pinned Bend fork/Bun, C/C++ toolchain, CMake, OpenSSL
development files and compatible LibTorch. Runtime dependencies: native engine,
exact package, compatible native shared libraries and (for CUDA) compatible GPU
driver/runtime. Python/checkpoint are export/qualification inputs, not required
engine runtime files. Build/startup identity checks do not prove numerical parity.

## Bounded real-model qualification proposal (not authorized or executed)

Before any PC/GPU step, obtain approval naming the machine/device, immutable
checkpoint/package paths and SHA256s, selected weights key, trusted build/toolchain
and new output directory outside active checkouts. Inspect the live process/GPU
inventory first; run only with an idle GPU or an explicitly approved paused window.
Do not restart training, change live YAML, switch its checkout or reuse live caches.

Proposed first GPU window: one GPU, one process, one batch bucket (1), two CPU
threads, one compile worker, no training/tournament/self-play, at most 15 minutes
elapsed GPU access and at most 8 GiB peak process VRAM. A separate export/build
budget must be approved if a compatible exact package does not already exist.
The 8 GiB ceiling is a kill condition, not an allocator reservation; if the model
cannot fit, stop and propose a revised budget. A watchdog records allocation and
terminates the qualification process on ceiling/time breach. No automatic retry
with larger budgets, different package, lower precision or relaxed tolerance.

Use 32 preregistered legal FEN/history fixtures spanning both turns, castling,
promotion, EP/pinned EP, repetition and feature boundaries, then 8 bounded UCI
searches at 8 real-neural rows each. First compare native BF16 against independent
eager BF16 on the exact device/input, and separately BF16 against FP32 for trained
model policy/WDL fidelity. Before execution, preregister model-specific logit
atol/rtol, legal-policy divergence/top-choice and WDL tolerances from the intended
acceptance requirement; no post-failure relaxation. Current universal values would
be guesses, so this proposal remains blocked until those exact gates are fixed.

Record original input/output traces locally, package/build/device identities,
all case results, memory peaks, cancellation/retirement counters, failures and
total elapsed time. Raw private model/input traces are not uploaded by default.
Stop at pass/fail for this exact package/device; a pass does not authorize
deployment, further buckets, multi-GPU work, performance tuning or a tournament.
