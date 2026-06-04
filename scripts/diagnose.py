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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_replay import sample_replay_arrays

from chess_anti_engine.moves import COMPACT_POLICY_SIZE, POLICY_SIZE, policy_batch_to_encoding, policy_batch_to_full
from chess_anti_engine.replay.shard import INPUT_HISTORY_ENCODING_ARRAY_KEY, iter_shard_paths
from chess_anti_engine.tune.replay_exchange import _trial_replay_shard_dir
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


def _resolve_replay_dir(args: argparse.Namespace, *, cfg: dict, trial_dir: Path) -> Path:
    if args.replay_dir:
        replay_dir = Path(args.replay_dir).expanduser().resolve()
        if not replay_dir.is_dir():
            sys.exit(f"--replay-dir does not exist: {replay_dir}")
        return replay_dir

    candidates: list[Path] = [
        _trial_replay_shard_dir(config=cfg, trial_dir=trial_dir),
        trial_dir / "selfplay_shards",
    ]
    seen: set[Path] = set()
    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        checked.append(candidate)
        if candidate.is_dir() and iter_shard_paths(candidate):
            return candidate
    checked_s = ", ".join(str(p) for p in checked)
    sys.exit(f"No replay shards found. Checked: {checked_s}. Pass --replay-dir <path>.")


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    p.add_argument("--run", default=None,
                   help="Run dir (defaults to runs/pbt2_small). Latest train_trial_* picked.")
    p.add_argument("--trial-dir", default=None,
                   help="Specific train_trial_* directory; overrides --run.")
    p.add_argument("--config", default="configs/pbt2_small.yaml",
                   help="YAML config (default: configs/pbt2_small.yaml).")
    p.add_argument("--replay-dir", default=None,
                   help="Replay shard directory; defaults to the trial replay_shards path.")
    p.add_argument("--device", default="cuda",
                   help="Torch device (default: cuda; use cpu if no GPU).")
    p.add_argument("--n", type=int, default=2048,
                   help="Number of replay positions to sample (default: 2048).")
    args = p.parse_args()

    import numpy as np
    import torch

    from chess_anti_engine.model import build_model, model_config_from_flat_config
    from chess_anti_engine.train import Trainer, trainer_kwargs_from_config
    from chess_anti_engine.train.trainer import select_input_history_arrays

    trial_dir = _resolve_trial_dir(args)
    print(f"Trial: {trial_dir.name}")

    ckpts = sorted(trial_dir.glob("checkpoint_*"))
    if not ckpts:
        sys.exit(f"No checkpoints under {trial_dir}")
    ckpt_dir = ckpts[-1]
    print(f"Checkpoint: {ckpt_dir.name}")

    cfg = flatten_run_config_defaults(load_yaml_file(args.config))

    model_cfg = model_config_from_flat_config(cfg)
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

    shard_dir = _resolve_replay_dir(args, cfg=cfg, trial_dir=trial_dir)
    arrs, total_positions, shard_count = sample_replay_arrays(
        shard_dir,
        int(args.n),
        rng=np.random.default_rng(42),
        fields=(
            "x", "x_lc0_root", "has_x_lc0_root", INPUT_HISTORY_ENCODING_ARRAY_KEY,
            "policy_target", "wdl_target", "has_policy",
        ),
    )
    print(f"Replay: {shard_dir}")
    print(f"Replay size: {total_positions:,} positions across {shard_count:,} shards")

    n = int(arrs["x"].shape[0])
    print(f"Sampling {n} positions...")
    arrs = select_input_history_arrays(
        arrs,
        input_history_encoding=model_cfg.input_history_encoding,
    )

    x = torch.from_numpy(np.asarray(arrs["x"], dtype=np.float32)).to(device)
    policy_target = np.asarray(arrs["policy_target"], dtype=np.float32)
    wdl_target = np.asarray(arrs["wdl_target"], dtype=np.int64)
    has_policy = np.asarray(arrs.get("has_policy", np.ones(n, dtype=np.uint8)), dtype=bool)

    print("Running inference...")
    with torch.no_grad():
        outputs = trainer.model(x)

    policy_logits = outputs["policy_own"].cpu().float().numpy()   # (N, 4672)
    wdl_logits = outputs["wdl"].cpu().float().numpy()             # (N, 3)
    if policy_target.ndim == 2 and int(policy_target.shape[1]) != int(policy_logits.shape[1]):
        if int(policy_logits.shape[1]) == COMPACT_POLICY_SIZE:
            policy_target = policy_batch_to_encoding(
                policy_target,
                policy_encoding=model_cfg.policy_encoding,
            )
        elif int(policy_logits.shape[1]) == POLICY_SIZE and int(policy_target.shape[1]) == COMPACT_POLICY_SIZE:
            policy_target = policy_batch_to_full(
                policy_target,
                policy_encoding="lc0_1858",
            )

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
    target_counts = {int(k): int(v) for k, v in zip(unique, counts)}
    total = valid.sum()
    print(
        f"  Target distribution: W={target_counts.get(0, 0)} "
        f"D={target_counts.get(1, 0)} L={target_counts.get(2, 0)} ({total} total)"
    )

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
