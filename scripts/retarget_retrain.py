#!/usr/bin/env python3
"""Offline target-retuning driver: retrain a checkpoint over existing replay
with SF targets REBUILT from sparse MultiPV labels under different params.

For each ``--variant name:k=v,k=v`` this runs a fixed training budget from
the same starting checkpoint and replay window with
``train.rebuild_sf_targets`` forced on and the given target params
overridden, then writes one checkpoint per variant. This is what turns
sf_policy_temp / cp-logistic / label-smoothing questions into offline
retrains instead of weeks-long live A/Bs — judge the resulting checkpoints
per docs/eval_protocol.md (arena_standard matched_sims vs the base
checkpoint).

Overridable param keys: sf_policy_temp, sf_policy_label_smooth,
sf_wdl_use_cp_logistic, sf_wdl_cp_slope, sf_wdl_cp_draw_width
(anything else in the flat config also works, e.g. lr).

Usage::

    PYTHONPATH=. python3 scripts/retarget_retrain.py \\
        --config configs/pbt2_small.yaml \\
        --checkpoint <trial>/checkpoint_000123/trainer.pt \\
        --replay-dir <trial>/replay_shards \\
        --steps 800 --out-dir runs/retarget \\
        --variant base: \\
        --variant sharp:sf_policy_temp=0.006 \\
        --variant smooth:sf_policy_temp=0.05,sf_policy_label_smooth=0.02
"""
from __future__ import annotations

import argparse
from typing import Any, cast
import json
import time
from pathlib import Path

import numpy as np

from chess_anti_engine.model import build_model, load_state_dict_tolerant, model_config_from_flat_config
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


def _parse_variant(spec: str) -> tuple[str, dict]:
    name, _, body = spec.partition(":")
    if not name:
        raise SystemExit(f"--variant needs a name before ':', got {spec!r}")
    overrides: dict = {}
    for pair in filter(None, body.split(",")):
        k, _, v = pair.partition("=")
        if not _:
            raise SystemExit(f"variant override must be k=v, got {pair!r}")
        lowered = v.strip().lower()
        if lowered in ("true", "false"):
            overrides[k.strip()] = lowered == "true"
        else:
            try:
                overrides[k.strip()] = float(v)
            except ValueError:
                overrides[k.strip()] = v.strip()
    return name, overrides


def _run_variant(
    *,
    name: str,
    overrides: dict,
    base_config: dict,
    checkpoint: Path,
    replay_dir: Path,
    steps: int,
    batch_size: int,
    device: str,
    out_dir: Path,
) -> dict:
    config = dict(base_config)
    config["rebuild_sf_targets"] = True
    config.update(overrides)

    model_cfg = model_config_from_flat_config(config)
    model = build_model(model_cfg)
    import torch

    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_state_dict_tolerant(model, state, label=f"retarget-{name}")

    kwargs = trainer_kwargs_from_config(config)
    kwargs["device"] = device
    trainer = Trainer(model, model_config=model_cfg, **kwargs)

    rng = np.random.default_rng(int(config.get("seed", 0)))
    buf = DiskReplayBuffer(
        10**9, shard_dir=replay_dir, rng=rng,
        shuffle_cap=int(config.get("shuffle_buffer_size", 20_000)),
    )
    try:
        t0 = time.time()
        metrics = trainer.train_steps(cast(Any, buf), batch_size=int(batch_size), steps=int(steps))
        duration = time.time() - t0
    finally:
        buf.close()

    out_path = out_dir / f"{name}.pt"
    trainer.save(out_path)
    summary = {
        "variant": name,
        "overrides": overrides,
        "steps": int(steps),
        "duration_s": round(duration, 1),
        "checkpoint": str(out_path),
        "final_metrics": {
            k: float(v)
            for k, v in vars(metrics).items()
            if isinstance(v, (int, float))
        },
    }
    print(f"[retarget] {name}: trained {steps} steps in {duration:.0f}s -> {out_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--replay-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default: train.batch_size from the config")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/retarget"))
    ap.add_argument("--variant", action="append", required=True,
                    help="name:k=v,k=v (':' with empty body = config defaults)")
    args = ap.parse_args()

    base_config = flatten_run_config_defaults(load_yaml_file(args.config))
    batch_size = int(args.batch_size or base_config.get("batch_size", 256))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for spec in args.variant:
        name, overrides = _parse_variant(spec)
        summaries.append(_run_variant(
            name=name, overrides=overrides, base_config=base_config,
            checkpoint=args.checkpoint, replay_dir=args.replay_dir,
            steps=args.steps, batch_size=batch_size, device=args.device,
            out_dir=args.out_dir,
        ))

    report = args.out_dir / "retarget_report.json"
    report.write_text(json.dumps(summaries, indent=2))
    print(f"[retarget] report written to {report}")


if __name__ == "__main__":
    main()
