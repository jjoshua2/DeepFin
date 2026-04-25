"""Diagnostic script: model quality analysis on current replay buffer.

Usage:
    PYTHONPATH=. python3 scripts/diagnose.py
    PYTHONPATH=. python3 scripts/diagnose.py --run runs/pbt2_small
    PYTHONPATH=. python3 scripts/diagnose.py --trial-dir runs/pbt2_small/tune/train_trial_XXX
    PYTHONPATH=. python3 scripts/diagnose.py --config configs/pbt2_small.yaml --device cpu

Measures:
  - WDL head accuracy: predicted outcome vs actual game outcome
  - WDL calibration: mean predicted win prob for actual wins/draws/losses
  - Policy sharpness: entropy of model predictions vs target entropy
  - Policy top-1 accuracy: does model's best move match MCTS best move?
  - Policy top-5 accuracy: is MCTS best move in model's top-5?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.replay import DiskReplayBuffer
from chess_anti_engine.train import Trainer, trainer_kwargs_from_config
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


def _resolve_trial_dir(args: argparse.Namespace) -> Path:
    """Resolve the trial dir from the (mutually-exclusive) --trial-dir / --run flags.

    Without args, falls back to the historical default (runs/pbt2_small/tune)
    but errors loudly with the searched path so the user can correct it
    instead of getting an opaque IndexError at import time (F003).
    """
    if args.trial_dir:
        td = Path(args.trial_dir).expanduser().resolve()
        if not td.is_dir():
            sys.exit(f"--trial-dir does not exist: {td}")
        return td
    run_dir = Path(args.run).expanduser().resolve() if args.run else Path("runs/pbt2_small")
    tune_dir = run_dir / "tune"
    if not tune_dir.is_dir():
        sys.exit(
            f"No tune directory at {tune_dir}. Pass --run <run-dir> or --trial-dir <path>."
        )
    candidates = sorted(
        tune_dir.glob("train_trial_*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        sys.exit(f"No train_trial_* dirs under {tune_dir}.")
    return candidates[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    p.add_argument("--run", default=None,
                   help="Run dir (defaults to runs/pbt2_small). Latest train_trial_* picked.")
    p.add_argument("--trial-dir", default=None,
                   help="Specific train_trial_* directory; overrides --run.")
    p.add_argument("--config", default="configs/pbt2_small.yaml",
                   help="YAML config (default: configs/pbt2_small.yaml).")
    p.add_argument("--device", default="cuda",
                   help="Torch device (default: cuda; use cpu if no GPU).")
    p.add_argument("--n", type=int, default=2048,
                   help="Number of replay positions to sample (default: 2048).")
    args = p.parse_args()

    trial_dir = _resolve_trial_dir(args)
    print(f"Trial: {trial_dir.name}")

    ckpts = sorted(trial_dir.glob("checkpoint_*"))
    if not ckpts:
        sys.exit(f"No checkpoints under {trial_dir}")
    ckpt_dir = ckpts[-1]
    print(f"Checkpoint: {ckpt_dir.name}")

    cfg = flatten_run_config_defaults(load_yaml_file(args.config))

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
        cfg | {"device": args.device},
        log_dir=trial_dir / "tb_diag",
    )
    trainer = Trainer(model, **trainer_kw)
    trainer.load(ckpt_dir / "trainer.pt")
    trainer.model.eval()
    device = trainer.device
    print(f"Device: {device}")

    shard_dir = trial_dir / "selfplay_shards"
    buf = DiskReplayBuffer(
        capacity=200_000,
        shard_dir=shard_dir,
        rng=np.random.default_rng(42),
    )
    print(f"Buffer size: {len(buf):,} positions")

    n = min(int(args.n), len(buf))
    print(f"Sampling {n} positions...")
    arrs = buf.sample_batch_arrays(n, wdl_balance=False)

    x = torch.from_numpy(np.asarray(arrs["x"], dtype=np.float32)).to(device)
    policy_target = np.asarray(arrs["policy_target"], dtype=np.float32)
    wdl_target = np.asarray(arrs["wdl_target"], dtype=np.int64)
    has_policy = np.asarray(arrs.get("has_policy", np.ones(n, dtype=np.uint8)), dtype=bool)

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

    for cls, name in [(0, "win"), (1, "draw"), (2, "loss")]:
        mask = (wdl_target == cls) & valid
        if mask.sum() > 0:
            mean_prob = wdl_probs[mask, cls].mean()
            count = mask.sum()
            print(f"  Avg P({name}) when true {name}: {mean_prob:.3f}  (n={count})")

    unique, counts = np.unique(wdl_target[valid], return_counts=True)
    total = valid.sum()
    print(f"  Target distribution: W={counts[0] if 0 in unique else 0} D={counts[1] if 1 in unique else 0} L={counts[2] if 2 in unique else 0} ({total} total)")

    # ---- POLICY SHARPNESS ----
    eps = 1e-9
    policy_probs = np.exp(policy_logits - policy_logits.max(axis=1, keepdims=True))
    policy_probs /= policy_probs.sum(axis=1, keepdims=True)
    model_entropy = -(policy_probs * np.log(policy_probs + eps)).sum(axis=1).mean()

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
        target_best = pt.argmax(axis=1)
        model_best = pl.argmax(axis=1)
        top1_acc = (model_best == target_best).mean()
        model_top5 = np.argsort(pl, axis=1)[:, -5:]
        top5_acc = np.array([target_best[i] in model_top5[i] for i in range(len(target_best))]).mean()
        print(f"Top-1 accuracy:            {top1_acc*100:.1f}%  (model best == MCTS best)")
        print(f"Top-5 accuracy:            {top5_acc*100:.1f}%  (MCTS best in model top-5)")

    print()


if __name__ == "__main__":
    main()
