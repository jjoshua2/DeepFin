#!/usr/bin/env python3
"""Train the initial bootstrap net on random game data.

Two phases: (1) threaded loading into buffer, (2) tight GPU training loop.
Threads share memory so no pickle overhead for large numpy arrays.

Usage:
    PYTHONPATH=. python3 scripts/train_bootstrap.py --config configs/pbt2_small.yaml
    PYTHONPATH=. python3 scripts/train_bootstrap.py --config configs/pbt2_small.yaml --max-positions 2000000
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.replay import ReplayBuffer
from chess_anti_engine.replay.shard import load_npz
from chess_anti_engine.train import Trainer
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


def _loader_thread(shard_paths: list[str], q: queue.Queue) -> None:
    for path in shard_paths:
        samples, _ = load_npz(path)
        q.put(samples)
    q.put(None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bootstrap net on random game data")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--bootstrap-dir", type=str, default=None, help="Override bootstrap_dir from config")
    parser.add_argument("--max-positions", type=int, default=0, help="Max positions to load (0=all)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs over the data")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--loaders", type=int, default=16, help="Number of parallel loader threads")
    parser.add_argument("--out", type=str, default="data/bootstrap/bootstrap_net_v2.pt", help="Output checkpoint path")
    args = parser.parse_args()

    cfg = load_yaml_file(args.config)
    flat = flatten_run_config_defaults(cfg)

    device = str(flat.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = args.batch_size

    model = build_model(
        ModelConfig(
            kind=str(flat.get("model", "transformer")),
            embed_dim=int(flat.get("embed_dim", 128)),
            num_layers=int(flat.get("num_layers", 4)),
            num_heads=int(flat.get("num_heads", 4)),
            ffn_mult=float(flat.get("ffn_mult", 2)),
            use_smolgen=not bool(flat.get("no_smolgen", False)),
            use_nla=bool(flat.get("use_nla", False)),
            use_gradient_checkpointing=bool(flat.get("gradient_checkpointing", False)),
        )
    )

    trainer = Trainer(
        model,
        device=device,
        lr=float(flat.get("lr", 3e-4)),
        log_dir=None,
        use_amp=not bool(flat.get("no_amp", False)),
        warmup_steps=int(flat.get("warmup_steps", 500)),
        lr_eta_min=float(flat.get("lr_eta_min", 1e-5)),
        lr_T0=int(flat.get("lr_T0", 2000)),
        lr_T_mult=int(flat.get("lr_T_mult", 2)),
        optimizer=str(flat.get("optimizer", "nadamw")),
        w_policy=float(flat.get("w_policy", 1.0)),
        w_soft=float(flat.get("w_soft", 0.5)),
        w_future=float(flat.get("w_future", 0.15)),
        w_wdl=float(flat.get("w_wdl", 1.0)),
        w_sf_move=float(flat.get("w_sf_move", 0.15)),
        w_sf_eval=float(flat.get("w_sf_eval", 0.15)),
        w_categorical=float(flat.get("w_categorical", 0.10)),
        w_sf_volatility=float(flat.get("w_sf_volatility", 0.05)),
        w_moves_left=float(flat.get("w_moves_left", 0.02)),
        w_sf_wdl=float(flat.get("w_sf_wdl", 1.0)),
    )

    bootstrap_dir = Path(args.bootstrap_dir or flat.get("bootstrap_dir", "data/bootstrap"))
    max_pos = args.max_positions
    shard_paths = [str(p) for p in sorted(bootstrap_dir.glob("*.npz"))]
    num_shards = len(shard_paths)
    print(f"Found {num_shards} shards in {bootstrap_dir}", flush=True)
    print(f"batch_size={batch_size}, loaders={args.loaders}", flush=True)

    rng = np.random.default_rng(42)

    # ── Phase 1: threaded loading into buffer ──
    t0 = time.time()
    buf_cap = max_pos if max_pos > 0 else 50_000_000  # large enough to hold all bootstrap data
    buf = ReplayBuffer(buf_cap, rng=rng)
    total_loaded = 0
    shard_count = 0

    rng.shuffle(shard_paths)
    q: queue.Queue = queue.Queue(maxsize=args.loaders * 2)
    n_loaders = args.loaders
    chunks = [shard_paths[i::n_loaders] for i in range(n_loaders)]
    threads = []
    for chunk in chunks:
        t = threading.Thread(target=_loader_thread, args=(chunk, q), daemon=True)
        t.start()
        threads.append(t)

    done_count = 0
    while done_count < n_loaders:
        item = q.get()
        if item is None:
            done_count += 1
            continue
        n = len(item)
        if max_pos > 0 and total_loaded + n > max_pos:
            item = item[:max_pos - total_loaded]
            n = len(item)
        buf.add_many(item)
        total_loaded += n
        shard_count += 1
        del item
        if shard_count % 50 == 0:
            print(f"  Loading: {shard_count}/{num_shards} shards, {total_loaded:,} pos [{time.time()-t0:.0f}s]", flush=True)
        if max_pos > 0 and total_loaded >= max_pos:
            break

    for t in threads:
        t.join(timeout=5)

    load_time = time.time() - t0
    print(f"Loaded {total_loaded:,} positions from {shard_count} shards in {load_time:.0f}s", flush=True)

    # ── Phase 2: tight GPU training loop ──
    steps_per_epoch = max(1, total_loaded // batch_size)
    total_steps = 0

    for epoch in range(args.epochs):
        metrics = trainer.train_steps(buf, batch_size=batch_size, steps=steps_per_epoch)
        total_steps += steps_per_epoch
        elapsed = time.time() - t0
        gpu_time = elapsed - load_time
        print(
            f"  Epoch {epoch+1}/{args.epochs}: {steps_per_epoch} steps, "
            f"loss={metrics.loss:.4f} wdl={metrics.wdl_loss:.4f} ml={metrics.moves_left_loss:.4f} "
            f"[{elapsed:.0f}s total, {total_steps/max(1,gpu_time):.1f} steps/s GPU]",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(out_path)
    print(f"\nSaved bootstrap checkpoint to {out_path}", flush=True)
    print(f"Total: {total_loaded:,} positions, {total_steps} steps, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
