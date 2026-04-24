#!/usr/bin/env python3
"""Benchmark max-autotune DirectGPU at various worker counts."""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np


def worker_fn(worker_id, results_dict, batch_size=170, n_iters=30):
    import torch

    from chess_anti_engine.inference import DirectGPUEvaluator
    from chess_anti_engine.model import ModelConfig, build_model

    model = build_model(ModelConfig(
        kind="transformer", embed_dim=384, num_layers=9, num_heads=12,
        ffn_mult=1.5, use_smolgen=True, use_nla=False,
    ))
    model = model.cuda().eval()
    model = torch.compile(model, mode="max-autotune")
    ev = DirectGPUEvaluator(model, device="cuda", max_batch=512)

    x = np.random.randn(batch_size, 146, 8, 8).astype(np.float32)
    for _ in range(10):
        ev.evaluate_encoded(x)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        ev.evaluate_encoded(x)
    elapsed = time.perf_counter() - t0
    results_dict[worker_id] = batch_size * n_iters / elapsed


def main():
    mp.set_start_method("spawn", force=True)
    import torch

    for n_workers in [1, 2, 4, 8]:
        mgr = mp.Manager()
        results = mgr.dict()
        procs = []
        for i in range(n_workers):
            p = mp.Process(target=worker_fn, args=(i, results))
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=600)

        total_nps = sum(results.values())
        avg_nps = total_nps / len(results) if results else 0
        free, total = torch.cuda.mem_get_info()
        print(
            f"{n_workers:>2} max-autotune workers: {total_nps:>8.0f} total pos/s  "
            f"({avg_nps:.0f}/worker)  VRAM free={free/1e9:.1f}GB"
        )


if __name__ == "__main__":
    main()
