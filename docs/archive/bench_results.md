# Selfplay Throughput Benchmarks

Date: 2026-04-10
Hardware: RTX 5090 (32GB), 16 CPU cores, WSL2
Config: 128 MCTS sims, 5000 SF nodes, 256 games/worker, bootstrap os=210

## Winner: 4 workers × 1 thread, compiled (~118s/iter in production)

## Multi-Worker (separate processes, DirectGPU compiled, 1 thread each)

| workers | sf/w | games/s | time(s) | VRAM | speedup |
|---------|------|---------|---------|------|---------|
| 1       | 8    | 1.35    | 189.4   | 2.8G | 1.0x    |
| 2       | 4    | 2.33    | 219.8   | 2.8G | 1.7x    |
| 3       | 4    | 3.35    | 229.5   | 2.8G | 2.5x    |
| 4       | 4    | 4.27    | 240.0   | 2.8G | 3.2x    |
| 6       | 3    | 5.94    | 258.7   | 2.8G | 4.4x    |
| 8       | 2    | 7.31    | 280.3   | 2.8G | 5.4x    |

Scaling nearly linear in isolation. BUT in production with trainer competing for GPU:
- 8w: ~450s/iter (VRAM 97%, GPU contention with training step)
- 4w: ~118s/iter (VRAM 75%, good headroom)

## Python Threading (ThreadPoolExecutor + ThreadedBatchEvaluator) — SLOWER

| threads | sf_wk | games/s | time(s) | VRAM |
|---------|-------|---------|---------|------|
| 1       | 8     | 1.37    | 187.2   | 2.8G |
| 4       | 8     | 1.16    | 220.1   | 2.5G |
| 8       | 8     | 1.11    | 231.0   | 2.7G |
| 16      | 16    | 0.99    | 259.2   | 3.4G |

In production (4w × 2t): ~471s/iter vs 118s/iter with 1t. 4x slower.
Root cause: ThreadedBatchEvaluator queue/event overhead per inference call.

## Pipeline Async (overlap GPU + CPU in single thread) — NO GAIN

| config              | games/s | time(s) |
|---------------------|---------|---------|
| 1w baseline         | 1.35    | 189.4   |
| 1w + pipeline async | 1.26    | 202.5   |

Slightly slower due to double-buffer bookkeeping. GPU call (CUDA graph replay)
is ~1ms — too fast to overlap meaningfully.

## C pthreads in batch_process_ply — NO GAIN

Per-ply post-processing is not the bottleneck. 1.35 → 1.35 games/s.

## Key Insights

1. Multi-process works because each process has its own CUDA context + compiled
   model. CUDA naturally interleaves work across processes.
2. Threading fails because torch.compile CUDA graphs can't run concurrently on
   different streams in the same process. ThreadedBatchEvaluator adds overhead
   without improving GPU utilization.
3. Pipeline async fails because CUDA graph replay is near-instant (~1ms). There's
   barely any GPU idle time to fill with CPU work.
4. In production, 8 workers saturates VRAM (31.7/32.6 GB) and GPU contention
   with the trainer makes iterations 4x slower than 4 workers.
5. AOT pre-compiled kernels are thread-safe (no CUDA graphs) but untested in
   production yet. Could enable LC0-style multi-threading in the future.
