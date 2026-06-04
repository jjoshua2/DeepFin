#!/usr/bin/env python3
"""Shrink per-layer transformer FFNs in a trainer checkpoint.

The script copies all same-shaped parameters into a freshly built target model
and prunes FFN hidden units by a deterministic weight-norm score:

    score(unit) = ||up[row]||² + ||down[:, col]||² + bias²

Optimizer and scheduler state are dropped because parameter shapes and optimizer
slot indices are no longer safe across the architecture change.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from chess_anti_engine.model import (
    ARCH_SCHEMA_VERSION,
    ModelConfig,
    build_model,
    normalize_ffn_mult_by_layer,
)
from chess_anti_engine.uci.model_loader import (
    _find_params_json,
    _model_config_from_arch,
    _model_config_from_params,
)

_FFN_PARAM_RE = re.compile(r"^blocks\.(\d+)\.ffn\.(0|2)\.(weight|bias)$")


def _checkpoint_model_config(ckpt: dict[str, Any], ckpt_path: Path) -> ModelConfig:
    if isinstance(ckpt.get("arch"), dict):
        return _model_config_from_arch(ckpt["arch"])

    params_path = _find_params_json(ckpt_path)
    if params_path is not None:
        with params_path.open() as fh:
            return _model_config_from_params(json.load(fh))

    raise SystemExit(
        f"{ckpt_path} has no embedded arch and no params.json in its ancestor "
        "tree; refusing to shrink FFNs with a guessed architecture"
    )


def _ffn_hidden_sizes(state: dict[str, torch.Tensor], *, num_layers: int) -> tuple[int, ...]:
    sizes: list[int] = []
    for layer_idx in range(num_layers):
        key = f"blocks.{layer_idx}.ffn.0.weight"
        if key not in state:
            raise ValueError(f"missing FFN weight: {key}")
        sizes.append(int(state[key].shape[0]))
    return tuple(sizes)


def _select_units(
    state: dict[str, torch.Tensor],
    *,
    layer_idx: int,
    target_hidden: int,
) -> torch.Tensor:
    up_w = state[f"blocks.{layer_idx}.ffn.0.weight"].detach().float()
    up_b = state[f"blocks.{layer_idx}.ffn.0.bias"].detach().float()
    down_w = state[f"blocks.{layer_idx}.ffn.2.weight"].detach().float()
    source_hidden = int(up_w.shape[0])
    if int(down_w.shape[1]) != source_hidden:
        raise ValueError(f"layer {layer_idx}: inconsistent FFN hidden size")
    if target_hidden > source_hidden:
        raise ValueError(
            f"layer {layer_idx}: target hidden {target_hidden} exceeds source hidden {source_hidden}"
        )
    if target_hidden <= 0:
        raise ValueError(f"layer {layer_idx}: target hidden must be > 0")
    if target_hidden == source_hidden:
        return torch.arange(source_hidden, dtype=torch.long)

    score = up_w.square().sum(dim=1) + down_w.square().sum(dim=0) + up_b.square()
    keep = torch.topk(score, k=target_hidden, largest=True, sorted=False).indices
    return keep.sort().values


def shrink_model_state_dict(
    source: dict[str, torch.Tensor],
    target_template: dict[str, torch.Tensor],
    *,
    target_hidden_sizes: tuple[int, ...],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Return a target-shaped state dict with FFN hidden units pruned."""
    out = {key: value.detach().clone() for key, value in target_template.items()}
    copied: list[str] = []
    mismatched: list[str] = []

    for key, target_value in target_template.items():
        if _FFN_PARAM_RE.match(key):
            continue
        source_value = source.get(key)
        if source_value is None:
            mismatched.append(f"{key}: missing in source")
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            mismatched.append(f"{key}: source {tuple(source_value.shape)} target {tuple(target_value.shape)}")
            continue
        out[key] = source_value.detach().clone()
        copied.append(key)

    if mismatched:
        joined = "\n  ".join(mismatched)
        raise ValueError(f"non-FFN parameter mismatch:\n  {joined}")

    num_layers = len(target_hidden_sizes)
    _ffn_hidden_sizes(source, num_layers=num_layers)
    for layer_idx, target_hidden in enumerate(target_hidden_sizes):
        keep = _select_units(source, layer_idx=layer_idx, target_hidden=int(target_hidden))
        prefix = f"blocks.{layer_idx}.ffn"
        out[f"{prefix}.0.weight"] = source[f"{prefix}.0.weight"].detach().index_select(0, keep).clone()
        out[f"{prefix}.0.bias"] = source[f"{prefix}.0.bias"].detach().index_select(0, keep).clone()
        out[f"{prefix}.2.weight"] = source[f"{prefix}.2.weight"].detach().index_select(1, keep).clone()
        out[f"{prefix}.2.bias"] = source[f"{prefix}.2.bias"].detach().clone()
        copied.extend([
            f"{prefix}.0.weight",
            f"{prefix}.0.bias",
            f"{prefix}.2.weight",
            f"{prefix}.2.bias",
        ])

    return out, copied


def shrink_checkpoint(
    ckpt: dict[str, Any],
    *,
    ckpt_path: Path,
    target_schedule: tuple[float, ...],
) -> tuple[dict[str, Any], ModelConfig, dict[str, Any]]:
    source_cfg = _checkpoint_model_config(ckpt, ckpt_path)
    normalized_schedule = normalize_ffn_mult_by_layer(target_schedule, num_layers=source_cfg.num_layers)
    if normalized_schedule is None:
        raise ValueError("target schedule must not be empty")
    target_cfg = dataclasses.replace(
        source_cfg,
        ffn_mult=float(sum(normalized_schedule) / len(normalized_schedule)),
        ffn_mult_by_layer=normalized_schedule,
    )
    target_model = build_model(target_cfg)
    target_template = target_model.state_dict()
    target_hidden_sizes = _ffn_hidden_sizes(target_template, num_layers=target_cfg.num_layers)

    source_state = ckpt.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("checkpoint has no model state dict")

    new_ckpt = dict(ckpt)
    new_state, copied = shrink_model_state_dict(
        source_state,
        target_template,
        target_hidden_sizes=target_hidden_sizes,
    )
    new_ckpt["model"] = new_state
    if isinstance(new_ckpt.get("swa_model"), dict):
        new_ckpt["swa_model"] = shrink_model_state_dict(
            new_ckpt["swa_model"],
            target_template,
            target_hidden_sizes=target_hidden_sizes,
        )[0]
    new_ckpt["arch"] = {
        "_schema_version": ARCH_SCHEMA_VERSION,
        **dataclasses.asdict(target_cfg),
    }
    new_ckpt.pop("opt", None)
    new_ckpt.pop("scheduler", None)

    source_hidden = _ffn_hidden_sizes(source_state, num_layers=source_cfg.num_layers)
    stats = {
        "source_hidden": source_hidden,
        "target_hidden": target_hidden_sizes,
        "copied_params": len(copied),
        "dropped_optimizer": "opt" in ckpt,
        "dropped_scheduler": "scheduler" in ckpt,
    }
    return new_ckpt, target_cfg, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_checkpoint", type=Path)
    ap.add_argument("output_checkpoint", type=Path)
    ap.add_argument(
        "--target-ffn-mult-by-layer",
        required=True,
        help="Comma-separated target FFN multipliers, one per layer.",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite output checkpoint if it exists.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.input_checkpoint.exists():
        raise SystemExit(f"missing input checkpoint: {args.input_checkpoint}")
    if args.output_checkpoint.exists() and not args.force and not args.dry_run:
        raise SystemExit(f"output exists (use --force): {args.output_checkpoint}")

    target_schedule = normalize_ffn_mult_by_layer(args.target_ffn_mult_by_layer)
    if target_schedule is None:
        raise SystemExit("--target-ffn-mult-by-layer must not be empty")

    ckpt = torch.load(args.input_checkpoint, map_location="cpu", weights_only=False)
    new_ckpt, target_cfg, stats = shrink_checkpoint(
        ckpt,
        ckpt_path=args.input_checkpoint,
        target_schedule=target_schedule,
    )

    print(f"source_hidden={list(stats['source_hidden'])}")
    print(f"target_hidden={list(stats['target_hidden'])}")
    print(f"target_ffn_mult_by_layer={list(target_cfg.ffn_mult_by_layer or ())}")
    print(f"copied_params={stats['copied_params']}")
    print("dropped optimizer/scheduler state")

    if args.dry_run:
        print("dry-run; not writing")
        return

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.output_checkpoint.exists() and args.force:
        bak = args.output_checkpoint.with_suffix(args.output_checkpoint.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(args.output_checkpoint, bak)
            print(f"backed up existing output to {bak}")
    torch.save(new_ckpt, args.output_checkpoint)
    print(f"wrote {args.output_checkpoint}")


if __name__ == "__main__":
    main()
