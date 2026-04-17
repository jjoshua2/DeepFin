#!/usr/bin/env python3
"""Profile the training loop end-to-end on GPU with synthetic replay data.

Exercises: DiskReplayBuffer → collation → feature dropout → model forward →
loss computation → backward → ZClip → optimizer step → LR schedule → SWA.

No Stockfish required — fills the replay buffer with realistic synthetic samples.

Usage:
    python scripts/profile_training.py [--steps 50] [--batch-size 128] [--embed-dim 256]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.model.transformer import ChessNet, TransformerConfig
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.train import Trainer


def _make_sample(rng: np.random.Generator) -> ReplaySample:
    x = rng.standard_normal((146, 8, 8)).astype(np.float32)
    pol = rng.random(POLICY_SIZE).astype(np.float32)
    pol /= pol.sum()
    wdl = int(rng.integers(0, 3))

    s = ReplaySample(x=x, policy_target=pol, wdl_target=wdl, priority=1.0,
                     has_policy=True, is_network_turn=True)
    s.sf_wdl = rng.dirichlet([1, 1, 1]).astype(np.float32)
    s.sf_move_index = int(rng.integers(0, POLICY_SIZE))
    s.moves_left = float(rng.random())
    s.categorical_target = np.ones(32, dtype=np.float32) / 32.0
    s.policy_soft_target = pol.copy()
    s.future_policy_target = pol.copy()
    s.volatility_target = rng.random(3).astype(np.float32)
    s.sf_volatility_target = rng.random(3).astype(np.float32)
    s.has_future = True
    s.has_volatility = True
    s.has_sf_volatility = True
    s.legal_mask = (rng.random(POLICY_SIZE) > 0.5).astype(np.float32)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--optimizer", type=str, default="nadamw")
    ap.add_argument("--use-compile", action="store_true")
    ap.add_argument("--use-amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--feature-dropout-p", type=float, default=0.3)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--swa-start", type=int, default=5)
    ap.add_argument("--swa-freq", type=int, default=5)
    ap.add_argument("--prefetch", action="store_true", default=True)
    ap.add_argument("--no-prefetch", action="store_true")
    args = ap.parse_args()

    use_amp = args.use_amp and not args.no_amp
    prefetch = args.prefetch and not args.no_prefetch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: embed={args.embed_dim}, layers={args.num_layers}, heads={args.num_heads}")
    print(f"Training: steps={args.steps}, batch={args.batch_size}, accum={args.accum_steps}, "
          f"amp={use_amp}, compile={args.use_compile}, optimizer={args.optimizer}")
    print(f"Dropout: {args.feature_dropout_p}, SWA: start={args.swa_start} freq={args.swa_freq}")
    print(f"Prefetch: {prefetch}")
    print()

    # --- Build model ---
    t0 = time.perf_counter()
    cfg = TransformerConfig(
        in_planes=146,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        use_smolgen=True,
        use_nla=False,
    )
    model = ChessNet(cfg)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")
    t_model = time.perf_counter() - t0
    print(f"Model build: {t_model:.3f}s")

    # --- Fill replay buffer ---
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="profile_train_")

    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    buf = DiskReplayBuffer(
        args.num_samples,
        shard_dir=tmpdir + "/replay",
        rng=rng,
        shuffle_cap=args.num_samples,
        shard_size=500,
    )
    samples = [_make_sample(rng) for _ in range(args.num_samples)]
    buf.add_many(samples)
    del samples
    t_fill = time.perf_counter() - t0
    print(f"Buffer fill ({args.num_samples} samples): {t_fill:.3f}s")

    # --- Build trainer ---
    t0 = time.perf_counter()
    trainer = Trainer(
        model,
        device=device,
        lr=3e-4,
        log_dir=tmpdir + "/tb",
        use_amp=use_amp,
        feature_dropout_p=args.feature_dropout_p,
        optimizer=args.optimizer,
        warmup_steps=args.warmup_steps,
        warmup_lr_start=1e-5,
        accum_steps=args.accum_steps,
        swa_start=args.swa_start,
        swa_freq=args.swa_freq,
        use_compile=args.use_compile,
        prefetch_batches=prefetch,
    )
    t_trainer = time.perf_counter() - t0
    print(f"Trainer init: {t_trainer:.3f}s")

    # --- Warmup run (1 step to trigger compilation / lazy inits) ---
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    print("\nWarmup step...")
    t0 = time.perf_counter()
    trainer.train_steps(buf, batch_size=args.batch_size, steps=1)
    if device == "cuda":
        torch.cuda.synchronize()
    t_warmup = time.perf_counter() - t0
    print(f"Warmup: {t_warmup:.3f}s")

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak GPU memory after warmup: {peak_mb:.0f} MB")
        torch.cuda.reset_peak_memory_stats()

    # --- Profiled training run ---
    print(f"\nProfiled run: {args.steps} steps...")
    if device == "cuda":
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    metrics = trainer.train_steps(buf, batch_size=args.batch_size, steps=args.steps)
    if device == "cuda":
        torch.cuda.synchronize()
    t_train = time.perf_counter() - t_start

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # --- Report ---
    steps_per_s = args.steps / max(t_train, 1e-9)
    samples_per_s = metrics.train_samples_seen / max(t_train, 1e-9)
    ms_per_step = t_train * 1000 / max(args.steps, 1)
    opt_ms_per_step = metrics.opt_step_time_s * 1000 / max(args.steps, 1)

    print(f"\n{'='*60}")
    print("TRAINING PROFILING RESULTS")
    print(f"{'='*60}")
    print(f"Steps:           {args.steps}")
    print(f"Batch size:      {args.batch_size} (× {args.accum_steps} accum = {args.batch_size * args.accum_steps} eff)")
    print(f"Total time:      {t_train:.3f}s")
    print(f"Steps/s:         {steps_per_s:.1f}")
    print(f"Samples/s:       {samples_per_s:.0f}")
    print(f"ms/step:         {ms_per_step:.1f}")
    print(f"  opt step ms:   {opt_ms_per_step:.1f}")
    print(f"  other ms:      {ms_per_step - opt_ms_per_step:.1f} (data load + fwd + bwd + grad clip)")
    print(f"Train time (reported): {metrics.train_time_s:.3f}s")
    print(f"Opt step time:   {metrics.opt_step_time_s:.3f}s ({metrics.opt_step_time_s/max(t_train,1e-9)*100:.1f}% of total)")
    print()
    print(f"Loss:            {metrics.loss:.4f}")
    print(f"  policy:        {metrics.policy_loss:.4f}")
    print(f"  wdl:           {metrics.wdl_loss:.4f}")
    print(f"  sf_move:       {metrics.sf_move_loss:.4f}")
    print(f"  sf_eval:       {metrics.sf_eval_loss:.4f}")
    print(f"  soft:          {metrics.soft_policy_loss:.4f}")
    print(f"  future:        {metrics.future_policy_loss:.4f}")
    print(f"  categorical:   {metrics.categorical_loss:.4f}")
    print(f"  volatility:    {metrics.volatility_loss:.4f}")
    print(f"  sf_volatility: {metrics.sf_volatility_loss:.4f}")
    print(f"  moves_left:    {metrics.moves_left_loss:.4f}")
    print(f"  sf_move_acc:   {metrics.sf_move_acc:.4f}")
    if device == "cuda":
        print(f"\nPeak GPU memory: {peak_mb:.0f} MB")
    print(f"Final LR:        {trainer.opt.param_groups[0]['lr']:.6f}")
    print(f"Trainer step:    {trainer.step}")

    # Cleanup
    import shutil
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
