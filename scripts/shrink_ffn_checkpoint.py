#!/usr/bin/env python3
"""Shrink per-layer transformer FFNs in a trainer checkpoint.

The script copies all same-shaped parameters into a freshly built target model
and prunes FFN hidden units by either a deterministic weight-norm score:

    score(unit) = ||up[row]||² + ||down[:, col]||² + bias²

an activation-aware replay calibration score:

    score(unit) = E[activation²] * ||down[:, col]||²

or a first-order Taylor score on replay loss:

    score(unit) = E[|activation * dL/dactivation|]

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
from chess_anti_engine.replay.dataset import collate_arrays
from chess_anti_engine.train.losses import compute_loss
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

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


def _hidden_sizes_from_schedule(
    *,
    embed_dim: int,
    schedule: tuple[float, ...],
) -> tuple[int, ...]:
    return tuple(int(int(embed_dim) * float(mult)) for mult in schedule)


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


def _validate_unit_indices(
    keep: torch.Tensor,
    *,
    layer_idx: int,
    source_hidden: int,
    target_hidden: int,
) -> torch.Tensor:
    keep = keep.detach().cpu().long().flatten()
    if target_hidden <= 0:
        raise ValueError(f"layer {layer_idx}: target hidden must be > 0")
    if int(keep.numel()) != int(target_hidden):
        raise ValueError(
            f"layer {layer_idx}: keep index count {int(keep.numel())} does not match "
            f"target hidden {int(target_hidden)}"
        )
    if int(keep.min().item()) < 0 or int(keep.max().item()) >= int(source_hidden):
        raise ValueError(
            f"layer {layer_idx}: keep indices out of range for source hidden {int(source_hidden)}"
        )
    if int(torch.unique(keep).numel()) != int(target_hidden):
        raise ValueError(f"layer {layer_idx}: keep indices must be unique")
    return keep.sort().values


def select_unit_indices_by_layer(
    state: dict[str, torch.Tensor],
    *,
    target_hidden_sizes: tuple[int, ...],
    unit_scores_by_layer: dict[int, torch.Tensor] | None = None,
) -> dict[int, torch.Tensor]:
    """Select source FFN hidden-unit indices once for every layer."""
    _ffn_hidden_sizes(state, num_layers=len(target_hidden_sizes))
    return {
        layer_idx: _select_units(
            state,
            layer_idx=layer_idx,
            target_hidden=int(target_hidden),
            unit_scores_by_layer=unit_scores_by_layer,
        )
        for layer_idx, target_hidden in enumerate(target_hidden_sizes)
    }


def shrink_model_state_dict(
    source: dict[str, torch.Tensor],
    target_template: dict[str, torch.Tensor],
    *,
    target_hidden_sizes: tuple[int, ...],
    unit_scores_by_layer: dict[int, torch.Tensor] | None = None,
    unit_indices_by_layer: dict[int, torch.Tensor] | None = None,
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
    source_hidden_sizes = _ffn_hidden_sizes(source, num_layers=num_layers)
    for layer_idx, target_hidden in enumerate(target_hidden_sizes):
        if unit_indices_by_layer is None:
            keep = _select_units(
                source,
                layer_idx=layer_idx,
                target_hidden=int(target_hidden),
                unit_scores_by_layer=unit_scores_by_layer,
            )
        elif layer_idx in unit_indices_by_layer:
            keep = _validate_unit_indices(
                unit_indices_by_layer[layer_idx],
                layer_idx=layer_idx,
                source_hidden=int(source_hidden_sizes[layer_idx]),
                target_hidden=int(target_hidden),
            )
        else:
            raise ValueError(f"missing selected FFN unit indices for layer {layer_idx}")
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


def _clone_state_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value


def _unwrap_swa_state_dict(state: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any], bool]:
    """Return prunable model weights plus SWA metadata from a saved SWA state dict."""
    has_module_prefix = any(str(key).startswith("module.") for key in state)
    if not has_module_prefix:
        return cast(dict[str, torch.Tensor], state), {}, False

    model_state: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {}
    for key, value in state.items():
        key_str = str(key)
        if key_str.startswith("module."):
            model_key = key_str.removeprefix("module.").removeprefix("_orig_mod.")
            model_state[model_key] = cast(torch.Tensor, value)
        else:
            metadata[key_str] = _clone_state_value(value)
    return model_state, metadata, True


def _rewrap_swa_state_dict(
    model_state: dict[str, torch.Tensor],
    *,
    metadata: dict[str, Any],
    was_averaged_model: bool,
) -> dict[str, Any]:
    if not was_averaged_model:
        return model_state

    return {
        **metadata,
        **{f"module.{key}": value for key, value in model_state.items()},
    }


def _shrink_swa_state_dict(
    state: dict[str, Any],
    target_template: dict[str, torch.Tensor],
    *,
    target_hidden_sizes: tuple[int, ...],
    unit_scores_by_layer: dict[int, torch.Tensor] | None,
    unit_indices_by_layer: dict[int, torch.Tensor],
) -> dict[str, Any]:
    source_state, metadata, was_averaged_model = _unwrap_swa_state_dict(state)
    shrunk_state = shrink_model_state_dict(
        source_state,
        target_template,
        target_hidden_sizes=target_hidden_sizes,
        unit_scores_by_layer=unit_scores_by_layer,
        unit_indices_by_layer=unit_indices_by_layer,
    )[0]
    return _rewrap_swa_state_dict(
        shrunk_state,
        metadata=metadata,
        was_averaged_model=was_averaged_model,
    )


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
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:  # skylos: ignore
            del _module, _inputs
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


def _concat_calibration_chunks(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ValueError("cannot concatenate empty calibration chunks")
    common = set(chunks[0])
    for chunk in chunks[1:]:
        common &= set(chunk)
    out: dict[str, np.ndarray] = {}
    for key in sorted(common):
        values = [np.asarray(chunk[key]) for chunk in chunks]
        if all(value.ndim == 0 for value in values):
            first = values[0].item()
            if all(value.item() == first for value in values[1:]):
                out[key] = np.asarray(values[0])
            continue
        if any(value.ndim == 0 for value in values):
            continue
        out[key] = np.concatenate(values, axis=0)
    return out


def _load_calibration_arrays_from_replay(
    *,
    source_cfg: ModelConfig,
    replay_dir: Path,
    positions: int,
    prefer_recorded_lc0_root: bool,
    synthetic_lc0_root_history: bool,
    lc0_root_legacy_meta: bool,
) -> dict[str, np.ndarray]:
    from chess_anti_engine.replay.shard import iter_shard_paths
    from chess_anti_engine.replay.shard import load_shard_arrays
    from chess_anti_engine.train.trainer import select_input_history_arrays

    # Kept for CLI parity with offline_replay_epoch. The selector follows the
    # checkpoint's input_history_encoding and uses recorded lc0-root tensors when
    # present, matching the sidecar's --prefer-recorded-lc0-root behavior.
    del prefer_recorded_lc0_root, synthetic_lc0_root_history, lc0_root_legacy_meta

    shard_paths = iter_shard_paths(replay_dir)
    if not shard_paths:
        raise ValueError(f"no replay shards found in {replay_dir}")
    left = max(1, int(positions))
    chunks: list[dict[str, np.ndarray]] = []
    for shard_path in shard_paths:
        if left <= 0:
            break
        arrs, _meta = load_shard_arrays(shard_path, lazy=False)
        arrs = select_input_history_arrays(
            arrs,
            input_history_encoding=source_cfg.input_history_encoding,
        )
        n_rows = int(np.asarray(arrs["x"]).shape[0])
        n = min(left, n_rows)
        if n <= 0:
            continue
        idx = np.arange(n, dtype=np.int64)
        chunks.append({
            key: np.asarray(value)[idx] if np.asarray(value).ndim > 0 else np.asarray(value)
            for key, value in arrs.items()
        })
        left -= n
    if not chunks:
        raise ValueError(f"could not load calibration positions from {replay_dir}")
    return _concat_calibration_chunks(chunks)


def activation_unit_scores_from_arrays(
    ckpt: dict[str, Any],
    *,
    ckpt_path: Path,
    arrs: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    x = torch.as_tensor(arrs["x"], dtype=torch.float32)
    return activation_unit_scores_from_x(
        ckpt,
        ckpt_path=ckpt_path,
        x=x,
        batch_size=batch_size,
        device=device,
    )


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
    source_cfg = _checkpoint_model_config(ckpt, ckpt_path)
    arrs = _load_calibration_arrays_from_replay(
        source_cfg=source_cfg,
        replay_dir=replay_dir,
        positions=positions,
        prefer_recorded_lc0_root=prefer_recorded_lc0_root,
        synthetic_lc0_root_history=synthetic_lc0_root_history,
        lc0_root_legacy_meta=lc0_root_legacy_meta,
    )
    return activation_unit_scores_from_arrays(
        ckpt,
        ckpt_path=ckpt_path,
        arrs=arrs,
        batch_size=batch_size,
        device=device,
    )


def _loss_kwargs_from_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    cfg = flatten_run_config_defaults(load_yaml_file(config_path))
    keys = (
        "w_policy",
        "w_soft",
        "w_future",
        "w_wdl",
        "w_sf_move",
        "w_sf_eval",
        "w_categorical",
        "w_volatility",
        "w_sf_volatility",
        "w_moves_left",
        "sf_wdl_frac",
        "search_wdl_frac",
        "sf_wdl_conf_power",
        "sf_wdl_draw_scale",
        "sf_wdl_temperature",
        "sf_search_dampen_sf_low",
        "sf_search_dampen_sf_high",
        "use_adjusted_wdl_target",
        "adjusted_wdl_regret_source",
        "adjusted_wdl_regret_scale",
        "adjusted_wdl_regret_cap",
    )
    return {key: cfg[key] for key in keys if key in cfg}


def taylor_unit_scores_from_arrays(
    ckpt: dict[str, Any],
    *,
    ckpt_path: Path,
    arrs: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    loss_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Score FFN units by first-order loss sensitivity."""
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
    layer_outputs: list[torch.Tensor | None] = [None for _ in source_hidden]

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("Taylor scoring requires a transformer model with blocks")

    def make_hook(layer_idx: int) -> Any:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:  # skylos: ignore
            del _module, _inputs
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"layer {layer_idx}: expected tensor activation, got {type(output).__name__}")
            output.retain_grad()
            layer_outputs[layer_idx] = output

        return hook

    for layer_idx, block in enumerate(blocks):
        handles.append(block.ffn[1].register_forward_hook(make_hook(layer_idx)))

    try:
        total = int(np.asarray(arrs["x"]).shape[0])
        if total <= 0:
            raise ValueError("calibration arrays are empty")
        batch = max(1, int(batch_size))
        loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
        for start in range(0, total, batch):
            chunk = {
                key: np.asarray(value)[start:start + batch]
                for key, value in arrs.items()
                if np.asarray(value).ndim > 0 and int(np.asarray(value).shape[0]) == total
            }
            model.zero_grad(set_to_none=True)
            layer_outputs = [None for _ in source_hidden]
            tensor_batch = collate_arrays(chunk, device=str(device))
            out = model(tensor_batch["x"])
            loss = compute_loss(out, tensor_batch, **loss_kwargs)["total"]
            loss.backward()
            for layer_idx, activation in enumerate(layer_outputs):
                if activation is None or activation.grad is None:
                    raise ValueError(f"layer {layer_idx}: no Taylor activation gradient captured")
                grad = activation.grad.detach()
                act = activation.detach()
                sums[layer_idx] += (act * grad).abs().sum(dim=(0, 1)).cpu().double()
                counts[layer_idx] += int(act.shape[0]) * int(act.shape[1])
    finally:
        for handle in handles:
            handle.remove()

    scores: dict[int, torch.Tensor] = {}
    for layer_idx, count in enumerate(counts):
        if count <= 0:
            raise ValueError(f"layer {layer_idx}: no Taylor activations captured")
        scores[layer_idx] = (sums[layer_idx].float() / float(count)).cpu()

    stats = {
        "taylor_score_positions": int(np.asarray(arrs["x"]).shape[0]),
        "taylor_score_batch_size": max(1, int(batch_size)),
        "taylor_score_device": str(device),
        "taylor_score_layers": len(scores),
    }
    return scores, stats


def taylor_unit_scores_from_replay(
    ckpt: dict[str, Any],
    *,
    ckpt_path: Path,
    replay_dir: Path,
    positions: int,
    batch_size: int,
    device: torch.device,
    config_path: Path | None,
    prefer_recorded_lc0_root: bool,
    synthetic_lc0_root_history: bool,
    lc0_root_legacy_meta: bool,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    source_cfg = _checkpoint_model_config(ckpt, ckpt_path)
    arrs = _load_calibration_arrays_from_replay(
        source_cfg=source_cfg,
        replay_dir=replay_dir,
        positions=positions,
        prefer_recorded_lc0_root=prefer_recorded_lc0_root,
        synthetic_lc0_root_history=synthetic_lc0_root_history,
        lc0_root_legacy_meta=lc0_root_legacy_meta,
    )
    return taylor_unit_scores_from_arrays(
        ckpt,
        ckpt_path=ckpt_path,
        arrs=arrs,
        batch_size=batch_size,
        device=device,
        loss_kwargs=_loss_kwargs_from_config(config_path),
    )


def _score_retained_fracs(
    unit_scores_by_layer: dict[int, torch.Tensor] | None,
    *,
    target_hidden_sizes: tuple[int, ...],
) -> tuple[float, ...] | None:
    if unit_scores_by_layer is None:
        return None
    fracs: list[float] = []
    for layer_idx, target_hidden in enumerate(target_hidden_sizes):
        score = unit_scores_by_layer.get(layer_idx)
        if score is None:
            fracs.append(0.0)
            continue
        score = score.detach().float().flatten().clamp_min(0)
        total = float(score.sum().item())
        if total <= 0.0:
            fracs.append(0.0)
            continue
        keep_sum = float(torch.topk(score, k=int(target_hidden), largest=True, sorted=False).values.sum().item())
        fracs.append(keep_sum / total)
    return tuple(fracs)


def optimize_target_schedule_by_scores(
    *,
    source_hidden_sizes: tuple[int, ...],
    source_embed_dim: int,
    budget_schedule: tuple[float, ...],
    unit_scores_by_layer: dict[int, torch.Tensor],
    min_ffn_mult: float,
    hidden_multiple: int,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Redistribute a target FFN budget to maximize retained unit score."""
    if len(source_hidden_sizes) != len(budget_schedule):
        raise ValueError(
            f"source layers {len(source_hidden_sizes)} != budget schedule layers {len(budget_schedule)}"
        )
    multiple = max(1, int(hidden_multiple))
    embed_dim = int(source_embed_dim)
    requested_hidden = _hidden_sizes_from_schedule(embed_dim=embed_dim, schedule=budget_schedule)
    total_budget = int(sum(requested_hidden))
    min_raw = int(embed_dim * float(min_ffn_mult))
    min_hidden = tuple(
        min(int(src), ((max(1, min_raw) + multiple - 1) // multiple) * multiple)
        for src in source_hidden_sizes
    )
    min_budget = int(sum(min_hidden))
    source_budget = int(sum(source_hidden_sizes))
    if total_budget < min_budget:
        raise ValueError(
            f"target hidden budget {total_budget} is below min budget {min_budget} "
            f"from min_ffn_mult={min_ffn_mult}"
        )
    if total_budget > source_budget:
        raise ValueError(f"target hidden budget {total_budget} exceeds source hidden budget {source_budget}")
    if (total_budget - min_budget) % multiple != 0:
        raise ValueError(
            f"target budget {total_budget} minus min budget {min_budget} must be divisible by "
            f"hidden_multiple={multiple}"
        )

    target_hidden = list(min_hidden)
    block_candidates: list[tuple[float, int]] = []
    for layer_idx, source_hidden in enumerate(source_hidden_sizes):
        score = unit_scores_by_layer.get(layer_idx)
        if score is None:
            raise ValueError(f"missing unit scores for layer {layer_idx}")
        score = _validate_unit_score(score, layer_idx=layer_idx, source_hidden=int(source_hidden))
        sorted_score = torch.sort(score, descending=True).values
        for start in range(int(min_hidden[layer_idx]), int(source_hidden), multiple):
            stop = min(start + multiple, int(source_hidden))
            if stop - start != multiple:
                continue
            gain = float(sorted_score[start:stop].sum().item())
            block_candidates.append((gain, layer_idx))

    blocks_needed = (total_budget - min_budget) // multiple
    if blocks_needed > len(block_candidates):
        raise ValueError(
            f"need {blocks_needed} hidden blocks but only {len(block_candidates)} are available"
        )
    block_candidates.sort(key=lambda item: item[0], reverse=True)
    for candidate in block_candidates[:blocks_needed]:
        layer_idx = candidate[1]
        target_hidden[layer_idx] += multiple

    optimized_schedule = tuple(float(hidden) / float(embed_dim) for hidden in target_hidden)
    stats = {
        "optimized_target_hidden": tuple(target_hidden),
        "optimized_target_ffn_mult_by_layer": optimized_schedule,
        "optimized_total_hidden_budget": total_budget,
        "optimized_min_hidden": min_hidden,
        "optimized_hidden_multiple": multiple,
    }
    return optimized_schedule, stats


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
    unit_indices_by_layer = select_unit_indices_by_layer(
        source_state,
        target_hidden_sizes=target_hidden_sizes,
        unit_scores_by_layer=unit_scores_by_layer,
    )

    new_ckpt = dict(ckpt)
    new_state, copied = shrink_model_state_dict(
        source_state,
        target_template,
        target_hidden_sizes=target_hidden_sizes,
        unit_scores_by_layer=unit_scores_by_layer,
        unit_indices_by_layer=unit_indices_by_layer,
    )
    new_ckpt["model"] = new_state
    if isinstance(new_ckpt.get("swa_model"), dict):
        new_ckpt["swa_model"] = _shrink_swa_state_dict(
            new_ckpt["swa_model"],
            target_template,
            target_hidden_sizes=target_hidden_sizes,
            unit_scores_by_layer=unit_scores_by_layer,
            unit_indices_by_layer=unit_indices_by_layer,
        )
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
        "cut_hidden": tuple(int(src) - int(dst) for src, dst in zip(source_hidden, target_hidden_sizes, strict=True)),
        "copied_params": len(copied),
        "score_mode": score_mode,
        "score_retained_frac": _score_retained_fracs(
            unit_scores_by_layer,
            target_hidden_sizes=target_hidden_sizes,
        ),
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
        choices=("weight", "activation", "taylor"),
        default="weight",
        help="FFN unit scoring method used to choose hidden units to keep.",
    )
    ap.add_argument("--config", type=Path, help="Training config used for Taylor loss weights.")
    ap.add_argument("--calibration-replay-dir", type=Path)
    ap.add_argument("--calibration-positions", type=int, default=65536)
    ap.add_argument("--calibration-batch-size", type=int, default=512)
    ap.add_argument("--calibration-device", default="auto")
    ap.add_argument(
        "--optimize-target-schedule",
        action="store_true",
        help=(
            "Use the requested target schedule only as a total hidden-unit budget, "
            "then redistribute units across layers by marginal unit score."
        ),
    )
    ap.add_argument("--min-ffn-mult", type=float, default=1.0)
    ap.add_argument("--target-hidden-multiple", type=int, default=96)
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
    if args.score_mode in ("activation", "taylor"):
        if args.calibration_replay_dir is None:
            raise SystemExit(f"--score-mode {args.score_mode} requires --calibration-replay-dir")
        if not args.calibration_replay_dir.exists():
            raise SystemExit(f"missing calibration replay dir: {args.calibration_replay_dir}")
    if args.score_mode == "activation":
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
    elif args.score_mode == "taylor":
        unit_scores_by_layer, score_stats = taylor_unit_scores_from_replay(
            ckpt,
            ckpt_path=args.input_checkpoint,
            replay_dir=args.calibration_replay_dir,
            positions=int(args.calibration_positions),
            batch_size=int(args.calibration_batch_size),
            device=_device_from_arg(args.calibration_device),
            config_path=args.config,
            prefer_recorded_lc0_root=bool(args.prefer_recorded_lc0_root),
            synthetic_lc0_root_history=bool(args.synthetic_lc0_root_history),
            lc0_root_legacy_meta=bool(args.lc0_root_legacy_meta),
        )
    if args.optimize_target_schedule:
        if unit_scores_by_layer is None:
            raise SystemExit("--optimize-target-schedule requires activation or Taylor scores")
        source_cfg = _checkpoint_model_config(ckpt, args.input_checkpoint)
        source_state = _source_state_dict(ckpt)
        target_schedule, optimize_stats = optimize_target_schedule_by_scores(
            source_hidden_sizes=_ffn_hidden_sizes(source_state, num_layers=source_cfg.num_layers),
            source_embed_dim=int(source_cfg.embed_dim),
            budget_schedule=target_schedule,
            unit_scores_by_layer=unit_scores_by_layer,
            min_ffn_mult=float(args.min_ffn_mult),
            hidden_multiple=int(args.target_hidden_multiple),
        )
        score_stats.update(optimize_stats)
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
    print(f"cut_hidden={list(stats['cut_hidden'])}")
    print(f"target_ffn_mult_by_layer={list(target_cfg.ffn_mult_by_layer or ())}")
    print(f"score_mode={stats['score_mode']}")
    if score_stats:
        print(f"score_positions={score_stats.get('activation_score_positions', score_stats.get('taylor_score_positions'))}")
        print(f"score_device={score_stats.get('activation_score_device', score_stats.get('taylor_score_device'))}")
        if "optimized_target_hidden" in score_stats:
            print(f"optimized_target_hidden={list(score_stats['optimized_target_hidden'])}")
            print(
                "optimized_target_ffn_mult_by_layer="
                f"{[round(float(v), 4) for v in score_stats['optimized_target_ffn_mult_by_layer']]}"
            )
    retained = stats.get("score_retained_frac")
    if retained is not None:
        print(f"score_retained_frac={[round(float(v), 4) for v in retained]}")
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
