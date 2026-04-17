#!/usr/bin/env python3
"""Benchmark DirectGPUEvaluator in eager mode (no compile) with multiple workers."""
from __future__ import annotations
import multiprocessing as mp
import numpy as np
import time


def worker_fn(worker_id, results_dict, batch_size=256, n_iters=30):
    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.inference import DirectGPUEvaluator

    model = build_model(ModelConfig(
        kind="transformer", embed_dim=384, num_layers=9, num_heads=12,
        ffn_mult=1.5, use_smolgen=True, use_nla=False,
    ))
    model = model.cuda().eval()
    # NO torch.compile — eager mode only
    ev = DirectGPUEvaluator(model, device="cuda", max_batch=512)

    x = np.random.randn(batch_size, 146, 8, 8).astype(np.float32)
    for _ in range(5):
        ev.evaluate_encoded(x)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        ev.evaluate_encoded(x)
    elapsed = time.perf_counter() - t0
    results_dict[worker_id] = batch_size * n_iters / elapsed


def main():
    mp.set_start_method("spawn", force=True)
    import torch

    for n_workers in [1, 2, 4, 8, 16]:
        mgr = mp.Manager()
        results = mgr.dict()
        procs = []
        for i in range(n_workers):
            p = mp.Process(target=worker_fn, args=(i, results))
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=120)

        total_nps = sum(results.values())
        avg_nps = total_nps / len(results) if results else 0
        free, total = torch.cuda.mem_get_info()
        print(
            f"{n_workers:>2} eager workers: {total_nps:>8.0f} total pos/s  "
            f"({avg_nps:.0f}/worker)  VRAM free={free/1e9:.1f}GB"
        )


if __name__ == "__main__":
    main()
