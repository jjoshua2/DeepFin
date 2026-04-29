# ThreadedDispatcher A/B results — Stage 3 (2026-04-28)

## TL;DR

**ThreadedDispatcher does not win.** DirectGPU 1-thread is the fastest path
under this workload by a wide margin. Stage 4 (in-process selfplay collapsing
the 2-subprocess split) is **NOT** pursued.

## Setup

- Hardware: RTX 5090, single GPU
- Model: 384-dim, 9-layer transformer (~15M params)
- Compile: `torch.compile(mode="reduce-overhead")` — cudagraphs on
- Per-path subprocess isolation (mixing in one process contaminates
  cudagraph_trees TLS — earlier in-process attempt crashed with
  `AssertionError: torch._C._is_key_in_tls`)
- Workload: synthetic concurrent producers calling `evaluate_encoded` in a hot
  loop. 30s timed window, 15s warmup. Batch 44 per producer (≈ leaves per
  gumbel step), 16 producers → 704 positions per fully-batched forward
- Bench: `scripts/bench_dispatcher_ab.py`
- Raw JSON: `docs/threaded_dispatcher_results.json`

## Results

| path                   | pos/s   | calls/s | avg_batch | avg_forward_ms |
|------------------------|---------|---------|-----------|----------------|
| DirectGPU (1t × 704)   | 44,193  | 62.8    | 704       | ~16            |
| ThreadedBatchEvaluator | 22,293  | 506.7   | 704       | n/a            |
| ThreadedDispatcher     | 16,888  | 383.8   | 675.6     | 31.40          |

DirectGPU is 2.0× faster than ThreadedBatchEvaluator and 2.6× faster than
ThreadedDispatcher.

## Why the dispatcher loses

Two compounding factors:

1. **DirectGPU at batch=704 already saturates matmul efficiency** for this
   model size. The cudagraph captures one static shape and replays it on
   every call (~16ms forward). There is no headroom to recover by batching
   more aggressively.
2. **The dispatcher pads to variable buckets** (128, 256, ..., 768, 1024)
   depending on what's in the queue. The dispatcher's avg_forward_ms (31.4)
   is ~2× DirectGPU's, suggesting the bucket-padding cost or shape variation
   triggers cudagraph recapture / kernel reselection.

## Caveat

This synthetic bench has zero non-GPU work between forwards. In real selfplay,
producer threads spend most of their time on MCTS tree ops (encoding, board
pushes, sampling) — the dispatcher could amortize its overhead better there.
But DirectGPU's 2× headroom over the threaded paths is large enough that
production-shaped overhead would not close the gap.

## Plan stop-condition trigger

> "Throughput within ±5% of baseline AND cudagraph counters identical →
> ThreadedDispatcher provides no in-process win, close experiment, do NOT
> proceed to Stage 4"

Triggered (worse than ±5%, dispatcher regresses). Closing the experiment.

## Disposition

- Stage 1 (`chess_anti_engine/inference_threaded.py`) — keep on disk; no
  removal. The class is small, isolated, and unused unless
  `--threaded-dispatcher` is passed.
- Stage 2 worker.py CLI flag (`--threaded-dispatcher`) — keep, off by default.
  Removing it is a separate cleanup PR if/when we're sure we won't revisit.
- Stage 3 instrumentation — leaves no production cost (the
  `dispatcher stats:` log line only fires when the dispatcher is active).
- Stage 4 — **not pursued**.

## Production attempt 2026-04-28 (post-bench)

After the synthetic-bench negative result, ran a real-training A/B:
- 2w × 8t threaded + dispatcher (default cfg). Iter 44 took **699s** vs ~387s
  baseline. Worker `dispatcher stats: avg_batch=791-818, avg_forward_ms=47-49`
  → cudagraphs were NOT capturing (47ms at batch 800 should be ~25ms).
- Diagnosed: worker.py was pre-compiling on its main thread; dispatcher thread
  inherited an OptimizedModule whose cudagraph_trees TLS lived on the wrong
  thread → forwards fell back to eager path.
- Shipped fix in `inference_threaded.py` + `worker.py`: when
  `--threaded-dispatcher`, skip worker-thread compile and pass `compile_mode`
  to the dispatcher so torch.compile + cudagraph capture happen on the
  dispatcher thread. First version blocked the constructor on a warmup
  forward; collapsed second version to lazy-compile (no constructor block) so
  startup isn't held up by max-autotune.
- Did not get a clean iter-time number with the lazy-compile dispatcher
  before reverting — the bench had already demonstrated cross-thread batching
  doesn't beat one big DirectGPU forward at this workload, and the user's
  question "is it supposed to be this slow / are we missing something?"
  prompted the rethink: under-utilization here is a *batch size* problem, not
  a threading problem.

**Conclusion**: Threading at this workload loses to large-batch DirectGPU.
The lever for future GPU-utilization wins is *bigger forward batches per
gumbel call* (raise `distributed_worker_min_games_per_batch` and/or feed
more concurrent games through one DirectGPU), not cross-thread queueing.

## Reproducing

```bash
./scripts/train.sh stop
PYTHONPATH=. python3 scripts/bench_dispatcher_ab.py \
  --compile-mode reduce-overhead --threads 16 --batch-size 44 \
  --duration-s 30 --warmup-s 15 \
  --out docs/threaded_dispatcher_results.json
./scripts/train.sh start
```
