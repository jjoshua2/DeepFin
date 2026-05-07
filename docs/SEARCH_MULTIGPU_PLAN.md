# Multi-GPU Search Plan

## Current Fit

The repo already has most of the low-level infrastructure needed for the
first multi-GPU search milestone:

- `DirectGPUEvaluator` supports pinned input/output buffers and two-slot
  async evaluation.
- `PucvChunker` runs single-GPU batched virtual-loss PUCT.
- `MultiGpuPucvPool` runs one evaluator worker per GPU against a shared
  `MCTSTree`.
- `run_gumbel_root_many_c` already implements Gumbel root selection and
  sequential halving, but its pipeline is aimed at batches of positions,
  not one UCI root spread across many GPUs.

This means the first production target should be the existing shared-tree
PUCT path, not a new root-partitioned Gumbel search. The final deployment
target is an 8x RTX 5090 machine, while the current local machine has one RTX
5090. That local GPU is a good proxy for per-GPU kernel shape, compile mode,
batch buckets, gather size, and pending-accounting effects. It is not a good
proxy for host scheduling, PCIe contention, CPU feed rate, or eight independent
evaluator workers, so real scaling claims should wait for the 8x machine.

## Milestones

1. Wire the existing `MultiGpuPucvPool` into the UCI executable behind an
   explicit option. Use per-device evaluator factories so compiled/cudagraph
   state is created on the worker thread that replays it.
2. Add pool instrumentation: per-worker batches, leaves, average batch size,
   evaluation time, wall time, and worker share.
3. Benchmark the current alternatives under fixed nodes and fixed time on the
   one-GPU box: classic Gumbel, single-GPU PUCV, and walker pool. Use CPU or
   same-device simulated `MultiGpuPucvPool` only for correctness/stress, not
   for speedup claims.
4. Add an optional pending-aware selection mode in the C tree. Keep the
   current virtual-loss behavior as the baseline until fixed-node and Elo
   data justify changing the default.
5. Add an immutable evaluation cache keyed by full chess state before adding
   speculative prefetch. Cache network outputs only, not mutable tree stats.
6. Add conservative speculative prefetch only when instrumentation shows
   under-filled queues. Start with PV/top-policy continuation prefetch, then
   consider Stockfish-head continuations if those heads are exposed and
   calibrated in the inference path.
7. Build Gumbel-partitioned shared search as a separate coordinator after the
   shared-tree pool has a measured ceiling. Use local trees per surviving root
   candidate plus shared immutable eval cache before attempting a mutable DAG.

## Implementation Status

- Done: explicit `UseMultiGpuPUCV` / `--multi-gpu-pucv` wiring, per-device
  factory construction, pool stats, and benchmark launcher support.
- Done: optional `PUCVPendingMode=virtual-mean` / `--pucv-pending-mode
  virtual-mean` for batched PUCV selection. The default remains `legacy`.
- Done: exact encoded-row cache infrastructure for the plain
  `evaluate_encoded` path, exposed as `EvalCacheEntries` /
  `--eval-cache-entries`.
- Done: cache-aware miss compaction for the single-thread DirectGPU PUCV
  in-place path and shared-cache support for the multi-GPU PUCV pool.
- Next: run the one-GPU benchmark matrix when training is not using the local
  GPU, then decide whether virtual-mean pending should become the default for
  single-GPU PUCV or stay experimental. Defer real multi-GPU `--pucv-sweep`
  until a multi-GPU machine is available.

## Eval Cache Notes

The first cache key must be at least as specific as the encoded network input.
Bare `CBoard.zobrist_hash` is not safe enough because the current encoder
includes en-passant, halfmove clock, repetition, and history planes in addition
to the current board. The initial `EncodedEvalCache` therefore keys by an exact
hash of each encoded float32 row and caches immutable policy/WDL outputs only.
For single-thread PUCV, the C descent emits a fast row fingerprint and the
Python cache verifies candidate hits with an exact encoded-row digest before
reuse. Cache hits integrate without occupying evaluator slots, while misses
are packed densely into the DirectGPU input buffer before
`evaluate_inplace_async`.

For the multi-GPU PUCV pool, workers share one thread-safe cache object and
stats report per-run cache hits/misses. On the 8x machine, treat cache hit
rate and scheduler balance together: too much cross-worker cache contention can
erase the benefit of saved GPU rows.

## One-GPU Benchmark Matrix

- Modes: classic Gumbel, single-GPU `UseVL`, walker pool.
- Gather: `128`, `256`, `384`, `512`, `768`, `1024` for `UseVL`.
- Pending mode: `legacy`, `virtual-mean`.
- Modes: fixed nodes first, fixed wall-clock second.
- Metrics: NPS, average batch, final move stability, PV stability, and root
  visit distribution.
- Interpretation: use the one-GPU RTX 5090 results to choose per-worker
  defaults for the 8x5090 deployment, not to estimate eight-GPU scaling.

## Future Multi-GPU Benchmark Matrix

- Devices: `1`, `2`, `4`, `8` where available.
- Gather: `128`, `256`, `384`, `512`, `768`, `1024`.
- Virtual loss: `0`, `1`, `3`, `5`.
- Modes: fixed nodes first, fixed wall-clock second.
- Metrics: NPS, unique NN evals/sec, average batch, p95 batch, per-GPU share,
  wall p50/p95, root visit balance, and final move stability.

## Guardrails

- Do not treat multi-GPU throughput as strength. Fixed-node quality must be
  checked separately from fixed-time speed.
- Do not introduce a mutable global DAG until repetition and 50-move semantics
  are handled explicitly.
- Keep speculative prefetch low priority and cache-backed so wrong guesses do
  not distort the search.
- Keep deterministic benchmark mode in mind: stable seeds, stable request
  ordering, fixed batch buckets, and speculation disabled.
- Do not interpret same-device simulated pools as multi-GPU speed tests. They
  are useful for races, routing, vloss conservation, and option plumbing only.
- Do interpret single RTX 5090 benchmarks as meaningful for per-device
  inference settings because the deployment GPUs match the local GPU class.
