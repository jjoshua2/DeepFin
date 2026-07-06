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

Every variant retrains from a COLD optimizer: only the checkpoint's model
weights are restored; optimizer moments / scheduler / step counters are
deliberately discarded so all variants share the identical fresh-AdamW
starting point (warmup per the config). The deltas between variants stay
meaningful, but the absolute trajectories differ from what a live retune
that kept the optimizer state would produce. Each variant is seeded
identically so dropout/sampling noise doesn't pollute the A/B.

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

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.model import build_model, load_state_dict_tolerant, model_config_from_flat_config
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.uci.model_loader import model_config_from_arch
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
        if k.strip() == "rebuild_sf_targets":
            # A variant override here would silently beat the CLI flag AND
            # could give the two arms different targets, so the A/B would
            # no longer isolate the knob it claims to. One global flag only.
            raise SystemExit(
                "rebuild_sf_targets is not variant-overridable — use the "
                "global --rebuild-sf-targets / --no-rebuild-sf-targets flag "
                "so every arm trains on the same targets"
            )
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
    rebuild_sf_targets: bool = True,
    gpu_mem_fraction: float = 0.0,
    allow_yaml_arch: bool = False,
) -> dict:
    config = dict(base_config)
    config["rebuild_sf_targets"] = bool(rebuild_sf_targets)
    config.update(overrides)

    import torch

    if device.startswith("cuda") and gpu_mem_fraction:
        # Cap the SELECTED device so sidecar retrains can't OOM a live
        # trainer sharing the GPU (same convention as the yardstick scripts).
        idx = (int(device.split(":", 1)[1]) if ":" in device
               else torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(float(gpu_mem_fraction), idx)

    # Identical seed per variant so dropout masks and the buffer's random
    # draws match across arms; sampling-knob overrides still (intentionally)
    # change WHICH rows those draws select.
    torch.manual_seed(int(config.get("seed", 0)))

    # ARCHITECTURE from the checkpoint's own ``arch`` key, NOT the YAML: the
    # config only carries training/target/sampling params, so building the
    # model from it silently drops every checkpoint tensor whose shape the
    # YAML disagrees on (observed: embed.weight + per-layer ffn when the YAML
    # arch drifted from the trained net) — a partially random-init model that
    # invalidates the A/B. The stored arch reconstructs the exact model, and
    # the load is strict: any dropped tensor aborts the run.
    # map_location stays "cpu": the Trainer moves the model to `device`, and
    # GPU-loading the whole checkpoint (optimizer state included) would spike
    # exactly the VRAM that --gpu-mem-fraction is trying to protect.
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    arch_used = isinstance(ckpt, dict) and bool(ckpt.get("arch"))
    if arch_used:
        model_cfg = model_config_from_arch(ckpt["arch"])
    elif allow_yaml_arch:
        print(f"[retarget-{name}] WARNING: checkpoint has no embedded arch; "
              "building the model from the YAML config. If the YAML arch has "
              "drifted from the trained net, mismatched tensors keep their "
              "random init and the A/B is invalid.")
        model_cfg = model_config_from_flat_config(config)
    else:
        raise SystemExit(
            f"{checkpoint} has no embedded arch key, so the model would be "
            "built from the YAML config — the exact silent partial-load this "
            "script guards against. Re-save the checkpoint with a current "
            "Trainer, or pass --allow-yaml-arch if you have verified the "
            "YAML architecture matches the trained net."
        )
    model = build_model(model_cfg)

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_state_dict_tolerant(
        model, state, label=f"retarget-{name}", require_complete=arch_used,
    )

    kwargs = trainer_kwargs_from_config(config)
    kwargs["device"] = device
    trainer = Trainer(model, model_config=model_cfg, **kwargs)

    rng = np.random.default_rng(int(config.get("seed", 0)))
    # Mirror the production sampling ctor (tune/trainable_init.py) — without
    # these the offline retrain samples with NO surprise weighting / draw cap
    # and any replay_* sampling override in a --variant is a silent no-op,
    # so A/Bs of sampling knobs (e.g. replay_sf_gap_priority_weight) would
    # measure nothing. Values resolve through TrialConfig.from_dict — the
    # documented single source of truth for defaults — so this cannot drift
    # from production when the YAML omits a key. Only `capacity` differs
    # (unbounded: the offline A/B uses the whole provided replay dir, not the
    # live run's growing window) and `input_planes`, which must match the
    # checkpoint arch rather than the YAML (the flat config has no top-level
    # input_extra_features key — reading it there yields None -> 146, which
    # rejects 175-plane v2_threats shards and empties the buffer).
    tc = TrialConfig.from_dict(config)
    buf = DiskReplayBuffer(
        10**9, shard_dir=replay_dir, rng=rng,
        input_planes=input_plane_count(model_cfg.input_extra_features),
        upgrade_v1_planes=tc.replay_upgrade_v1_planes,
        shuffle_cap=tc.shuffle_buffer_size,
        shard_size=tc.shard_size,
        refresh_interval=tc.shuffle_refresh_interval,
        refresh_shards=tc.shuffle_refresh_shards,
        draw_cap_frac=tc.shuffle_draw_cap_frac,
        wl_max_ratio=tc.shuffle_wl_max_ratio,
        sf_gap_priority_weight=tc.replay_sf_gap_priority_weight,
        fast_low_surprise_priority=tc.replay_fast_low_surprise_priority,
        diff_focus_pol_scale=tc.diff_focus_pol_scale,
        diff_focus_q_weight=tc.diff_focus_q_weight,
    )
    try:
        if len(buf) == 0:
            raise SystemExit(
                f"replay buffer is empty: no shards under {replay_dir} match "
                f"input_planes={input_plane_count(model_cfg.input_extra_features)} "
                "(wrong --replay-dir, or plane-count mismatch with the "
                "checkpoint arch)"
            )
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
    ap.add_argument("--rebuild-sf-targets", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="--no-rebuild-sf-targets trains on the shards' stored "
                         "targets exactly as live training does — required when "
                         "the A/B is a sampling knob, not a target param")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.0,
                    help="cap this process's share of the selected CUDA device "
                         "(0 = uncapped); set when running beside a live trainer")
    ap.add_argument("--allow-yaml-arch", action="store_true",
                    help="permit building the model from the YAML config when "
                         "the checkpoint has no embedded arch key (UNSAFE: a "
                         "drifted YAML arch silently random-inits mismatched "
                         "tensors and invalidates the A/B)")
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
            out_dir=args.out_dir, rebuild_sf_targets=args.rebuild_sf_targets,
            gpu_mem_fraction=args.gpu_mem_fraction,
            allow_yaml_arch=args.allow_yaml_arch,
        ))

    report = args.out_dir / "retarget_report.json"
    report.write_text(json.dumps(summaries, indent=2))
    print(f"[retarget] report written to {report}")


if __name__ == "__main__":
    main()
