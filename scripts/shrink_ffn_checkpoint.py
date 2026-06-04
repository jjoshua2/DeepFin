#!/usr/bin/env python3
"""Shrink per-layer transformer FFNs in a trainer checkpoint.

The script copies all same-shaped parameters into a freshly built target model
and prunes FFN hidden units by either a deterministic weight-norm score:

    score(unit) = ||up[row]||² + ||down[:, col]||² + bias²

or an activation-aware replay calibration score:

    score(unit) = E[activation²] * ||down[:, col]||²

Optimizer and scheduler state are dropped because parameter shapes and optimizer
slot indices are no longer safe across the architecture change.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
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


def _weight_unit_scores(state: dict[str, torch.Tensor], *, layer_idx: int) -> torch.Tensor:
    up_w = state[f"blocks.{layer_idx}.ffn.0.weight"].detach().float()
    up_b = state[f"blocks.{layer_idx}.ffn.0.bias"].detach().float()
    down_w = state[f"blocks.{layer_idx}.ffn.2.weight"].detach().float()
    return up_w.square().sum(dim=1) + down_w.square().sum(dim=0) + up_b.square()


def _validate_unit_score(
    score: torch.Tensor,
    *,
    layer_idx: int,
    source_hidden: int,
) -> torch.Tensor:
    score = score.detach().float().cpu().flatten()
    if int(score.numel()) != source_hidden:
        raise ValueError(
            f"layer {layer_idx}: score size {int(score.numel())} does not match "
            f"source hidden {source_hidden}"
        )
    if not torch.isfinite(score).all():
        raise ValueError(f"layer {layer_idx}: unit scores must be finite")
    return score


def _select_units(
    state: dict[str, torch.Tensor],
    *,
    layer_idx: int,
    target_hidden: int,
    unit_scores_by_layer: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    up_w = state[f"blocks.{layer_idx}.ffn.0.weight"].detach().float()
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

    score = _weight_unit_scores(state, layer_idx=layer_idx)
    if unit_scores_by_layer is not None and layer_idx in unit_scores_by_layer:
        score = unit_scores_by_layer[layer_idx]
    score = _validate_unit_score(score, layer_idx=layer_idx, source_hidden=source_hidden)
    keep = torch.topk(score, k=target_hidden, largest=True, sorted=False).indices
    return keep.sort().values


def shrink_model_state_dict(
    source: dict[str, torch.Tensor],
    target_template: dict[str, torch.Tensor],
    *,
    target_hidden_sizes: tuple[int, ...],
    unit_scores_by_layer: dict[int, torch.Tensor] | None = None,
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
        keep = _select_units(
            source,
            layer_idx=layer_idx,
            target_hidden=int(target_hidden),
            unit_scores_by_layer=unit_scores_by_layer,
        )
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


def _source_state_dict(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    source_state = ckpt.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("checkpoint has no model state dict")
    return cast(dict[str, torch.Tensor], source_state)


def _device_from_arg(raw: str) -> torch.device:
    value = str(raw).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def activation_unit_scores_from_x(
    ckpt: dict[str, Any],
    *,
    ckpt_path: Path,
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Score FFN units by replay activation energy times down-projection norm."""
    source_cfg = _checkpoint_model_config(ckpt, ckpt_path)
    source_state = _source_state_dict(ckpt)
    source_hidden = _ffn_hidden_sizes(source_state, num_layers=source_cfg.num_layers)

    model = build_model(source_cfg)
    model.load_state_dict(source_state)
    model.to(device)
    model.eval()

    sums = [torch.zeros(hidden, dtype=torch.float64) for hidden in source_hidden]
    counts = [0 for _ in source_hidden]
    handles: list[Any] = []

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("activation scoring requires a transformer model with blocks")

    def make_hook(layer_idx: int) -> Any:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"layer {layer_idx}: expected tensor activation, got {type(output).__name__}")
            act = output.detach().float()
            sums[layer_idx] += act.square().sum(dim=(0, 1)).cpu().double()
            counts[layer_idx] += int(act.shape[0]) * int(act.shape[1])

        return hook

    for layer_idx, block in enumerate(blocks):
        handles.append(block.ffn[1].register_forward_hook(make_hook(layer_idx)))

    try:
        total = int(x.shape[0])
        if total <= 0:
            raise ValueError("calibration tensor is empty")
        batch = max(1, int(batch_size))
        with torch.no_grad():
            for start in range(0, total, batch):
                xb = x[start:start + batch].to(device=device, dtype=torch.float32, non_blocking=True)
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        model(xb)
                else:
                    with contextlib.nullcontext():
                        model(xb)
    finally:
        for handle in handles:
            handle.remove()

    scores: dict[int, torch.Tensor] = {}
    for layer_idx, count in enumerate(counts):
        if count <= 0:
            raise ValueError(f"layer {layer_idx}: no calibration activations captured")
        down_w = source_state[f"blocks.{layer_idx}.ffn.2.weight"].detach().float()
        mean_act_sq = sums[layer_idx].float() / float(count)
        scores[layer_idx] = mean_act_sq * down_w.square().sum(dim=0)

    stats = {
        "activation_score_positions": int(x.shape[0]),
        "activation_score_batch_size": max(1, int(batch_size)),
        "activation_score_device": str(device),
        "activation_score_layers": len(scores),
    }
    return scores, stats


def activation_unit_scores_from_replay(
    ckpt: dict[str, Any],
    *,
    ckpt_path: Path,
    replay_dir: Path,
    positions: int,
    batch_size: int,
    device: torch.device,
    prefer_recorded_lc0_root: bool,
    synthetic_lc0_root_history: bool,
    lc0_root_legacy_meta: bool,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    from chess_anti_engine.replay.shard import iter_shard_paths
    from chess_anti_engine.replay.shard import load_shard_arrays
    from chess_anti_engine.train.trainer import select_input_history_arrays

    # Kept for CLI parity with offline_replay_epoch. The selector follows the
    # checkpoint's input_history_encoding and uses recorded lc0-root tensors when
    # present, matching the sidecar's --prefer-recorded-lc0-root behavior.
    del prefer_recorded_lc0_root, synthetic_lc0_root_history, lc0_root_legacy_meta

    source_cfg = _checkpoint_model_config(ckpt, ckpt_path)
    shard_paths = iter_shard_paths(replay_dir)
    if not shard_paths:
        raise ValueError(f"no replay shards found in {replay_dir}")
    left = max(1, int(positions))
    chunks: list[np.ndarray] = []
    for shard_path in shard_paths:
        if left <= 0:
            break
        arrs, _meta = load_shard_arrays(shard_path, lazy=False)
        arrs = select_input_history_arrays(
            arrs,
            input_history_encoding=source_cfg.input_history_encoding,
        )
        shard_x = np.asarray(arrs["x"])
        n = min(left, int(shard_x.shape[0]))
        if n <= 0:
            continue
        chunks.append(np.asarray(shard_x[:n], dtype=np.float32))
        left -= n
    if not chunks:
        raise ValueError(f"could not load calibration positions from {replay_dir}")
    x = torch.as_tensor(np.concatenate(chunks, axis=0), dtype=torch.float32)
    return activation_unit_scores_from_x(
        ckpt,
        ckpt_path=ckpt_path,
        x=x,
        batch_size=batch_size,
        device=device,
    )


def shrink_checkpoint(
    ckpt: dict[str, Any],
    *,
    ckpt_path: Path,
    target_schedule: tuple[float, ...],
    unit_scores_by_layer: dict[int, torch.Tensor] | None = None,
    score_mode: str = "weight",
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

    source_state = _source_state_dict(ckpt)

    new_ckpt = dict(ckpt)
    new_state, copied = shrink_model_state_dict(
        source_state,
        target_template,
        target_hidden_sizes=target_hidden_sizes,
        unit_scores_by_layer=unit_scores_by_layer,
    )
    new_ckpt["model"] = new_state
    if isinstance(new_ckpt.get("swa_model"), dict):
        new_ckpt["swa_model"] = shrink_model_state_dict(
            cast(dict[str, torch.Tensor], new_ckpt["swa_model"]),
            target_template,
            target_hidden_sizes=target_hidden_sizes,
            unit_scores_by_layer=unit_scores_by_layer,
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
        "score_mode": score_mode,
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
    ap.add_argument(
        "--score-mode",
        choices=("weight", "activation"),
        default="weight",
        help="FFN unit scoring method used to choose hidden units to keep.",
    )
    ap.add_argument("--calibration-replay-dir", type=Path)
    ap.add_argument("--calibration-positions", type=int, default=65536)
    ap.add_argument("--calibration-batch-size", type=int, default=512)
    ap.add_argument("--calibration-device", default="auto")
    ap.add_argument("--prefer-recorded-lc0-root", action="store_true")
    ap.add_argument("--synthetic-lc0-root-history", action="store_true")
    ap.add_argument("--lc0-root-legacy-meta", action="store_true")
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
    unit_scores_by_layer: dict[int, torch.Tensor] | None = None
    score_stats: dict[str, Any] = {}
    if args.score_mode == "activation":
        if args.calibration_replay_dir is None:
            raise SystemExit("--score-mode activation requires --calibration-replay-dir")
        if not args.calibration_replay_dir.exists():
            raise SystemExit(f"missing calibration replay dir: {args.calibration_replay_dir}")
        unit_scores_by_layer, score_stats = activation_unit_scores_from_replay(
            ckpt,
            ckpt_path=args.input_checkpoint,
            replay_dir=args.calibration_replay_dir,
            positions=int(args.calibration_positions),
            batch_size=int(args.calibration_batch_size),
            device=_device_from_arg(args.calibration_device),
            prefer_recorded_lc0_root=bool(args.prefer_recorded_lc0_root),
            synthetic_lc0_root_history=bool(args.synthetic_lc0_root_history),
            lc0_root_legacy_meta=bool(args.lc0_root_legacy_meta),
        )
    new_ckpt, target_cfg, stats = shrink_checkpoint(
        ckpt,
        ckpt_path=args.input_checkpoint,
        target_schedule=target_schedule,
        unit_scores_by_layer=unit_scores_by_layer,
        score_mode=str(args.score_mode),
    )
    stats.update(score_stats)

    print(f"source_hidden={list(stats['source_hidden'])}")
    print(f"target_hidden={list(stats['target_hidden'])}")
    print(f"target_ffn_mult_by_layer={list(target_cfg.ffn_mult_by_layer or ())}")
    print(f"score_mode={stats['score_mode']}")
    if score_stats:
        print(f"activation_score_positions={score_stats['activation_score_positions']}")
        print(f"activation_score_device={score_stats['activation_score_device']}")
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
