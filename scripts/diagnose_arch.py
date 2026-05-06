"""Architecture-utilization diagnostic.

Reports:
  - Embed-dim utilization: per-channel L2 norm distribution + dead-channel count
    (a "dead" channel has norm below 1% of the median — model effectively wastes it).
  - Per-layer residual contribution: ||block_out - block_in|| / ||block_in||
    across the residual stream. Tells you whether layer N is doing meaningful work
    relative to layers around it. Newly-added layers that read 0.0 are not training.
  - Per-head attention output magnitude per layer. A consistently-zero head
    contribution = dead head.

Usage:
    PYTHONPATH=. python3 scripts/diagnose_arch.py
    PYTHONPATH=. python3 scripts/diagnose_arch.py --trial-dir runs/pbt2_small/tune/train_trial_XXX --n 1024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chess_anti_engine.replay.shard import iter_shard_paths
from chess_anti_engine.tune.replay_exchange import _trial_replay_shard_dir
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


def _resolve_trial_dir(args: argparse.Namespace) -> Path:
    if args.trial_dir:
        td = Path(args.trial_dir).expanduser().resolve()
        if not td.is_dir():
            sys.exit(f"--trial-dir does not exist: {td}")
        return td
    run_dir = Path(args.run).expanduser().resolve() if args.run else Path("runs/pbt2_small")
    tune_dir = run_dir / "tune"
    if not tune_dir.is_dir():
        sys.exit(f"No tune directory at {tune_dir}.")
    candidates = sorted(tune_dir.glob("train_trial_*"), key=lambda p: p.stat().st_mtime)
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
    p.add_argument("--run", default=None)
    p.add_argument("--trial-dir", default=None)
    p.add_argument("--config", default="configs/pbt2_small.yaml")
    p.add_argument("--replay-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--dead-frac", type=float, default=0.01,
                   help="Channel is 'dead' if its norm < dead_frac × median norm (default 0.01).")
    args = p.parse_args()

    import numpy as np
    import torch

    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.replay import DiskReplayBuffer
    from chess_anti_engine.train import Trainer, trainer_kwargs_from_config

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
        cfg | {"device": args.device}, log_dir=trial_dir / "tb_diag_arch",
    )
    trainer = Trainer(model, **trainer_kw)
    trainer.load(ckpt_dir / "trainer.pt")
    trainer.model.eval()
    device = trainer.device
    print(f"Device: {device}  embed={model_cfg.embed_dim} layers={model_cfg.num_layers} heads={model_cfg.num_heads}")

    shard_dir = _resolve_replay_dir(args, cfg=cfg, trial_dir=trial_dir)
    print(f"Replay: {shard_dir}")
    buf = DiskReplayBuffer(capacity=200_000, shard_dir=shard_dir, rng=np.random.default_rng(42))
    n = min(int(args.n), len(buf))
    print(f"Sampling {n} positions...")
    arrs = buf.sample_batch_arrays(n, wdl_balance=False)
    x = torch.from_numpy(np.asarray(arrs["x"], dtype=np.float32)).to(device)

    # --- hook embed output, every block's (input, output, attn-head-output) ---
    captures: dict[str, torch.Tensor] = {}
    head_caps: list[tuple[int, torch.Tensor]] = []

    def hook_embed_out(_mod, _inp, out):
        captures["embed_out"] = out.detach().float()

    def make_block_hook(idx: int):
        def hook(_mod, inp, out):
            captures[f"block_in_{idx}"] = inp[0].detach().float()
            captures[f"block_out_{idx}"] = out.detach().float()
        return hook

    def make_attn_outproj_hook(idx: int):
        # Captures the input to out_proj: (B,T,D) where D=H*head_dim — split per head.
        def hook(_mod, inp, _out):
            head_caps.append((idx, inp[0].detach().float()))
        return hook

    handles = []
    # Embed gate output is computed inline (not a sub-module). Hook self.embed Linear,
    # then post-process in the script — close enough for "is this dim getting any signal".
    handles.append(trainer.model.embed.register_forward_hook(hook_embed_out))
    for i, blk in enumerate(trainer.model.blocks):
        handles.append(blk.register_forward_hook(make_block_hook(i)))
        handles.append(blk.out_proj.register_forward_hook(make_attn_outproj_hook(i)))

    print("Forward pass...")
    with torch.no_grad():
        _ = trainer.model(x)

    for h in handles:
        h.remove()

    def _concentration(per_ch_norms: np.ndarray) -> dict:
        """Concentration of signal across channels.

        - participation_ratio = (Σ e_i)² / Σ e_i² where e_i = ||c_i||² (per-channel
          energy). For uniform energy across D channels, PR = D; for one-hot, PR = 1.
          Interpretable as "effective number of channels carrying the signal".
        - top-k% energy: smallest k such that the top-k channels by energy
          account for ≥ X% of total energy.
        """
        e = per_ch_norms.astype(np.float64) ** 2
        total = float(e.sum())
        pr = float(total ** 2 / max((e ** 2).sum(), 1e-30))
        sorted_e = np.sort(e)[::-1]
        cum = np.cumsum(sorted_e) / max(total, 1e-30)
        return {
            "pr": pr,
            "top50": int(np.searchsorted(cum, 0.50) + 1),
            "top80": int(np.searchsorted(cum, 0.80) + 1),
            "top95": int(np.searchsorted(cum, 0.95) + 1),
        }

    # ---- EMBED DIM UTILIZATION ----
    embed_out = captures["embed_out"]               # (B,64,D)
    per_ch = embed_out.norm(dim=(0, 1)).cpu().numpy()  # (D,)
    median = float(np.median(per_ch))
    dead_thresh = args.dead_frac * median
    n_dead = int((per_ch < dead_thresh).sum())
    conc = _concentration(per_ch)
    print(f"\n=== Embed dim utilization (D={model_cfg.embed_dim}) ===")
    print("  Per-channel L2 norm of embed output (over batch+squares):")
    print(f"    median={median:.3f}  min={per_ch.min():.4f}  max={per_ch.max():.3f}")
    print(f"    p10={np.percentile(per_ch, 10):.3f}  p90={np.percentile(per_ch, 90):.3f}")
    print(f"  Dead channels (<{args.dead_frac*100:.1f}% median): {n_dead} / {model_cfg.embed_dim}")
    print(f"  Effective rank (participation ratio): {conc['pr']:.1f} / {model_cfg.embed_dim}")
    print(f"  Top-k channels carrying ≥X% of energy: "
          f"50%→{conc['top50']}  80%→{conc['top80']}  95%→{conc['top95']}")

    # ---- PER-LAYER RESIDUAL CONTRIBUTION + CONCENTRATION ----
    print("\n=== Per-layer residual + channel concentration ===")
    print("  layer | ||delta||/||in|| | dead | eff-rank | top50% | top80% | top95%")
    print("  ------+------------------+------+----------+--------+--------+-------")
    for i in range(model_cfg.num_layers):
        bi = captures[f"block_in_{i}"]
        bo = captures[f"block_out_{i}"]
        rel = (bo - bi).norm().item() / max(bi.norm().item(), 1e-9)
        per_ch_layer = bo.norm(dim=(0, 1)).cpu().numpy()
        med_layer = float(np.median(per_ch_layer))
        dead_layer = int((per_ch_layer < args.dead_frac * med_layer).sum())
        c = _concentration(per_ch_layer)
        print(f"  {i:5d} | {rel:16.3f} | {dead_layer:4d} | "
              f"{c['pr']:7.1f}  | {c['top50']:6d} | {c['top80']:6d} | {c['top95']:6d}")

    # ---- PER-HEAD ATTENTION OUTPUT MAGNITUDE ----
    print("\n=== Per-head attention magnitude ===")
    print("  layer | mean head-norm | min head-norm | max head-norm | weak heads (<10% mean)")
    print("  ------+----------------+---------------+---------------+----------------------")
    head_dim = model_cfg.embed_dim // model_cfg.num_heads
    for idx, attn_in in head_caps:
        # attn_in is (B,T,D). Reshape to per-head: (B,T,H,head_dim).
        # torch.norm with multi-dim is restricted to matrix-norm; do it manually.
        b_, t_, _ = attn_in.shape
        reshaped = attn_in.view(b_, t_, model_cfg.num_heads, head_dim)
        per_head = reshaped.pow(2).sum(dim=(0, 1, 3)).sqrt().cpu().numpy()
        mn = float(per_head.mean())
        weak = int((per_head < 0.1 * mn).sum()) if mn > 0 else 0
        print(f"  {idx:5d} | {mn:14.3f} | {per_head.min():13.3f} | {per_head.max():13.3f} | {weak:4d} / {model_cfg.num_heads}")


if __name__ == "__main__":
    main()
