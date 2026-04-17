"""Diagnostic script: model quality analysis on current replay buffer.

Usage:
    PYTHONPATH=. python3 scripts/diagnose.py

Measures:
  - WDL head accuracy: predicted outcome vs actual game outcome
  - WDL calibration: mean predicted win prob for actual wins/draws/losses
  - Policy sharpness: entropy of model predictions vs target entropy
  - Policy top-1 accuracy: does model's best move match MCTS best move?
  - Policy top-5 accuracy: is MCTS best move in model's top-5?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.replay import DiskReplayBuffer
from chess_anti_engine.train import Trainer, trainer_kwargs_from_config
from chess_anti_engine.utils import load_yaml_file, flatten_run_config_defaults

TRIAL_DIR = sorted(
    Path("runs/pbt2_small/tune").glob("train_trial_*"),
    key=lambda p: p.stat().st_mtime,
)[-1]
print(f"Trial: {TRIAL_DIR.name}")

# Find latest checkpoint
ckpts = sorted(TRIAL_DIR.glob("checkpoint_*"))
if not ckpts:
    print("No checkpoints found")
    sys.exit(1)
ckpt_dir = ckpts[-1]
print(f"Checkpoint: {ckpt_dir.name}")

# --- Load config ---
cfg = flatten_run_config_defaults(load_yaml_file("configs/pbt2_small.yaml"))

# --- Build model ---
model_cfg = ModelConfig(
    kind=str(cfg.get("model", "transformer")),
    embed_dim=int(cfg.get("embed_dim", 384)),
    num_layers=int(cfg.get("num_layers", 9)),
    num_heads=int(cfg.get("num_heads", 8)),
    ffn_mult=float(cfg.get("ffn_mult", 2.0)),
    use_smolgen=not bool(cfg.get("no_smolgen", False)),
)
model = build_model(model_cfg)

trainer_kw = trainer_kwargs_from_config(
    cfg | {"device": "cuda"},
    log_dir=TRIAL_DIR / "tb_diag",
)
trainer = Trainer(model, **trainer_kw)
trainer.load(ckpt_dir / "trainer.pt")
trainer.model.eval()
device = trainer.device
print(f"Device: {device}")

# --- Load replay buffer ---
shard_dir = TRIAL_DIR / "selfplay_shards"
buf = DiskReplayBuffer(
    capacity=200_000,
    shard_dir=shard_dir,
    rng=np.random.default_rng(42),
)
print(f"Buffer size: {len(buf):,} positions")

N = min(2048, len(buf))
print(f"Sampling {N} positions...")
arrs = buf.sample_batch_arrays(N, wdl_balance=False)

x = torch.from_numpy(np.asarray(arrs["x"], dtype=np.float32)).to(device)
policy_target = np.asarray(arrs["policy_target"], dtype=np.float32)
wdl_target = np.asarray(arrs["wdl_target"], dtype=np.int64)
has_policy = np.asarray(arrs.get("has_policy", np.ones(N, dtype=np.uint8)), dtype=bool)

# --- Run inference ---
print("Running inference...")
with torch.no_grad():
    outputs = trainer.model(x)

policy_logits = outputs["policy_own"].cpu().float().numpy()   # (N, 4672)
wdl_logits = outputs["wdl"].cpu().float().numpy()             # (N, 3)

# ---- WDL ACCURACY ----
wdl_probs = np.exp(wdl_logits - wdl_logits.max(axis=1, keepdims=True))
wdl_probs /= wdl_probs.sum(axis=1, keepdims=True)
wdl_pred = wdl_probs.argmax(axis=1)

valid = (wdl_target >= 0) & (wdl_target <= 2)
acc = (wdl_pred[valid] == wdl_target[valid]).mean()
print("\n=== WDL Head ===")
print(f"Accuracy (top-1):      {acc*100:.1f}%")

# Calibration: mean predicted prob for each true class
for cls, name in [(0,"win"), (1,"draw"), (2,"loss")]:
    mask = (wdl_target == cls) & valid
    if mask.sum() > 0:
        mean_prob = wdl_probs[mask, cls].mean()
        count = mask.sum()
        print(f"  Avg P({name}) when true {name}: {mean_prob:.3f}  (n={count})")

# WDL distribution
unique, counts = np.unique(wdl_target[valid], return_counts=True)
total = valid.sum()
print(f"  Target distribution: W={counts[0] if 0 in unique else 0} D={counts[1] if 1 in unique else 0} L={counts[2] if 2 in unique else 0} ({total} total)")

# ---- POLICY SHARPNESS ----
eps = 1e-9

# Model prediction entropy
policy_probs = np.exp(policy_logits - policy_logits.max(axis=1, keepdims=True))
policy_probs /= policy_probs.sum(axis=1, keepdims=True)
model_entropy = -(policy_probs * np.log(policy_probs + eps)).sum(axis=1).mean()

# Target entropy (only positions with a policy)
hp = has_policy & (policy_target.sum(axis=1) > 0)
target_entropy = 0.0
if hp.sum() > 0:
    pt = policy_target[hp]
    pt = pt / pt.sum(axis=1, keepdims=True)
    target_entropy = -(pt * np.log(pt + eps)).sum(axis=1).mean()

print("\n=== Policy Head ===")
print(f"Model prediction entropy:  {model_entropy:.3f}  (lower = sharper)")
print(f"Target entropy:            {target_entropy:.3f}  (MCTS improved policy)")
print(f"Entropy ratio (m/t):       {model_entropy/max(target_entropy,eps):.2f}  (1.0 = matched, >1 = too diffuse)")

# ---- POLICY ACCURACY ----
if hp.sum() > 0:
    pt = policy_target[hp]
    pl = policy_logits[hp]

    # Top-1: model's argmax matches target's argmax
    target_best = pt.argmax(axis=1)
    model_best = pl.argmax(axis=1)
    top1_acc = (model_best == target_best).mean()

    # Top-5: target's best move is in model's top-5
    model_top5 = np.argsort(pl, axis=1)[:, -5:]
    top5_acc = np.array([target_best[i] in model_top5[i] for i in range(len(target_best))]).mean()

    print(f"Top-1 accuracy:            {top1_acc*100:.1f}%  (model best == MCTS best)")
    print(f"Top-5 accuracy:            {top5_acc*100:.1f}%  (MCTS best in model top-5)")

print()
