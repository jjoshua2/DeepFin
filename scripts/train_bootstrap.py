#!/usr/bin/env python3
"""Train the initial bootstrap net on random game data.

Run ONCE, then all future trials/tunes load the saved checkpoint.

Streams shards from disk one at a time to avoid OOM (14.2M positions
would need ~500+ GB RAM if loaded all at once).

Usage:
    PYTHONPATH=. python3 scripts/train_bootstrap.py --config configs/pbt2_small.yaml
    PYTHONPATH=. python3 scripts/train_bootstrap.py --config configs/pbt2_small.yaml --max-positions 100000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.replay import ReplayBuffer
from chess_anti_engine.replay.shard import load_npz
from chess_anti_engine.train import Trainer
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bootstrap net on random game data")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--bootstrap-dir", type=str, default=None, help="Override bootstrap_dir from config")
    parser.add_argument("--max-positions", type=int, default=0, help="Max positions to load (0=all)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs over the data")
    parser.add_argument("--out", type=str, default="data/bootstrap/bootstrap_net.pt", help="Output checkpoint path")
    args = parser.parse_args()

    cfg = load_yaml_file(args.config)
    flat = flatten_run_config_defaults(cfg)

    device = str(flat.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = int(flat.get("batch_size", 256))

    model = build_model(
        ModelConfig(
            kind=str(flat.get("model", "transformer")),
            embed_dim=int(flat.get("embed_dim", 128)),
            num_layers=int(flat.get("num_layers", 4)),
            num_heads=int(flat.get("num_heads", 4)),
            ffn_mult=int(flat.get("ffn_mult", 2)),
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
    shard_paths = sorted(bootstrap_dir.glob("*.npz"))
    print(f"Found {len(shard_paths)} shards in {bootstrap_dir}", flush=True)

    # Stream training: load one shard at a time into a small buffer, train, discard.
    # Buffer holds ~2 shards worth for some mixing. Peak RAM ~3-4 GB.
    rng = np.random.default_rng(42)
    buf = ReplayBuffer(50_000, rng=rng)

    t0 = time.time()
    total_positions = 0
    total_steps = 0
    metrics = None

    for epoch in range(args.epochs):
        epoch_shards = list(shard_paths)
        rng.shuffle(epoch_shards)
        epoch_positions = 0

        for shard_idx, shard_path in enumerate(epoch_shards):
            samples, _ = load_npz(shard_path)
            if max_pos > 0 and epoch_positions + len(samples) > max_pos:
                samples = samples[:max_pos - epoch_positions]

            n = len(samples)
            buf.add_many(samples)
            epoch_positions += n
            total_positions += n
            del samples  # free shard memory immediately

            # Train on this shard's worth of data.
            steps = max(1, n // batch_size)
            if len(buf) >= batch_size:
                metrics = trainer.train_steps(buf, batch_size=batch_size, steps=steps)
                total_steps += steps

            if (shard_idx + 1) % 50 == 0 and metrics is not None:
                elapsed = time.time() - t0
                pos_per_sec = total_positions / max(1, elapsed)
                print(
                    f"  Epoch {epoch+1} shard {shard_idx+1}/{len(epoch_shards)}: "
                    f"{epoch_positions:,} pos, {total_steps} steps, "
                    f"loss={metrics.loss:.4f} wdl={metrics.wdl_loss:.4f} ml={metrics.moves_left_loss:.4f} "
                    f"[{elapsed:.0f}s, {pos_per_sec:.0f} pos/s]",
                    flush=True,
                )

            if max_pos > 0 and epoch_positions >= max_pos:
                break

        elapsed = time.time() - t0
        loss_str = f"loss={metrics.loss:.4f}" if metrics else "no training"
        print(
            f"  Epoch {epoch+1}/{args.epochs} done: {epoch_positions:,} positions, "
            f"{total_steps} steps, {loss_str} [{elapsed:.0f}s]",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(out_path)
    print(f"\nSaved bootstrap checkpoint to {out_path}", flush=True)
    print(f"Total: {total_positions:,} positions, {total_steps} steps, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
