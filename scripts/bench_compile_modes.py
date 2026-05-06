#!/usr/bin/env python3
"""Benchmark different torch.compile modes for multi-worker DirectGPU."""
from __future__ import annotations

import multiprocessing as mp
import time
from typing import cast

import numpy as np


def worker_fn(worker_id, results_dict, compile_mode, batch_size=170, n_iters=30):
    import torch

    from chess_anti_engine.inference import DirectGPUEvaluator
    from chess_anti_engine.model import ModelConfig, build_model

    model = build_model(ModelConfig(
        kind="transformer", embed_dim=384, num_layers=9, num_heads=12,
        ffn_mult=1.5, use_smolgen=True, use_nla=False,
    ))
    model = model.cuda().eval()
    if compile_mode != "none":
        model = cast(torch.nn.Module, torch.compile(model, mode=compile_mode))
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

    n_workers = 4  # Test with 4 workers

    for mode in ["none", "default", "reduce-overhead", "max-autotune"]:
        mgr = mp.Manager()
        results = mgr.dict()
        procs = []
        for i in range(n_workers):
            p = mp.Process(target=worker_fn, args=(i, results, mode))
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=300)

        total_nps = sum(results.values())
        avg_nps = total_nps / len(results) if results else 0
        free, _total = torch.cuda.mem_get_info()
        print(
            f"{mode:>20} (4w): {total_nps:>8.0f} total pos/s  "
            f"({avg_nps:.0f}/worker)  VRAM free={free/1e9:.1f}GB"
        )


if __name__ == "__main__":
    main()
