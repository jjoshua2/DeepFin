#!/usr/bin/env python3
"""Per-head gradient-norm contribution to the SHARED TRUNK of the production net.

Offline, read-only diagnostic: does the VALUE head deliver a gradient-starved
signal to the shared trunk relative to the POLICY head?

For each real replay batch we run the faithful training forward + ``compute_loss``
(chess_anti_engine.train.losses), then for each differentiable per-component loss
tensor C we compute the L2 norm of d C / d(trunk params) via ``torch.autograd.grad``
(no optimizer step, measurement only). We report raw and loss-weighted trunk-grad
norms, per-component and grouped, plus the headline primary-policy vs primary-value
ratio.

Grouping rule (docs/rl_loop_audit.md I3/I7): the weighted denominator is the
set of components the optimizer actually sums into ``total`` — every one of
them, each in exactly one group, so the group shares sum to 100%. The
``wdl_onehot_ce`` diagnostic carries no gradient into the update and is
reported in a separate DIAGNOSTIC block, never in the denominator. The first
measurement of this probe put it in the denominator with weight ``w_wdl`` but
in no group, which is why POLICY/VALUE/OTHER summed to 80.62% instead of 100%.

Re-run:
    PYTHONPATH=. .venv/bin/python scripts/probe_head_grad_share.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.encoding.lc0 import uses_lc0_root_legacy_meta
from chess_anti_engine.model import build_model
from chess_anti_engine.train.losses import compute_loss
from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

# Reuse the offline harness helpers so batch prep is byte-identical to training.
from scripts.offline_replay_epoch import (
    _ArraySampler,
    _as_replay_buffer,
    _convert_policy_targets,
    _model_config_from_offline_config,
    _select_configured_input_history,
)

DEFAULT_CONFIG_PATH = "configs/pbt2_small.yaml"
DEFAULT_CKPT_PATH = "scratchpad/gradshare_probe/ckpt118/trainer.pt"
DEFAULT_REPLAY_DIR = (
    "runs/pbt2_small/replay/"
    "train_trial_4c17c_00000_0_lr=0.0003_2026-07-11_13-16-47/replay_shards"
)
DEFAULT_RESULT_PATH = "scratchpad/gradshare_probe/result.json"
DEFAULT_BATCH_SIZE = 128
DEFAULT_N_BATCHES = 8
DEFAULT_GPU_MEM_FRACTION = 0.15

# Head module attribute prefixes on ChessNet (transformer.py). Everything else
# (embed, blocks, layer_smolgens/smolgen, width_adapters, global-board
# preprocess/adapter, gates, phase_output_adapter) is the SHARED TRUNK.
# ⚑ `policy_embedding` is the SHARED POLICY ADAPTER: policy-only, but shared by
# every policy head, so it is neither a private head nor the global trunk. Listing
# it here keeps it OUT of the trunk bucket -- unlisted, a policy-only module would
# inflate the very "trunk share" this probe reports, and pre/post numbers would not
# be the same measurement. `_classify_params` returns it as its own bucket so the
# number is REPORTED rather than folded into either side -- lumping it with the
# private head params would bias trunk share DOWN, and with the trunk UP, and the
# whole question this instrument is being pointed at is how much lands on it.
POLICY_SHARED_PREFIXES = frozenset({"policy_embedding"})

HEAD_PREFIXES = frozenset({
    "policy_own", "policy_soft", "policy_sf", "policy_future",
    "value_wdl", "value_sf_eval", "value_categorical",
    # Distinct top-level attribute, so the `value_categorical` entry above does
    # NOT cover it: classification splits on the first dotted component and
    # tests set membership. Left out, the coupled aux head's own 32-way Linear
    # counts as TRUNK and inflates exactly the value-head trunk share this
    # probe exists to measure.
    "value_categorical_coupled",
    "volatility", "sf_volatility", "moves_left",
})

# Loss component -> its weight key in Trainer._loss_kwargs. EXACTLY the terms
# losses.compute_loss sums into ``total``; nothing else may enter the weighted
# denominator. ``wdl_ce`` is deliberately absent: compute_loss returns it as an
# alias of ``blended_wdl_ce``, and counting both would double the value share.
COMPONENT_WEIGHT_KEY = {
    "policy_ce": "w_policy",
    "soft_policy_ce": "w_soft",
    "future_policy_ce": "w_future",
    "sf_own_ce": "w_sf_own",
    "sf_own_regret": "w_sf_own_regret",
    "blended_wdl_ce": "w_wdl",
    "sf_move_ce": "w_sf_move",
    "sf_eval_ce": "w_sf_eval",
    "categorical_ce": "w_categorical",
    "volatility": "w_volatility",
    "sf_volatility": "w_sf_volatility",
    "moves_left": "w_moves_left",
}
COMPONENT_ORDER = list(COMPONENT_WEIGHT_KEY.keys())

# Reported for context only — no gradient reaches the update through it, so it
# is excluded from every share.
DIAGNOSTIC_COMPONENTS = ("wdl_onehot_ce",)

POLICY_GROUP = (
    "policy_ce", "soft_policy_ce", "future_policy_ce", "sf_move_ce",
    "sf_own_ce", "sf_own_regret",
)
VALUE_GROUP = ("blended_wdl_ce", "sf_eval_ce", "categorical_ce")
OTHER_GROUP = ("volatility", "sf_volatility", "moves_left")
GROUPS: dict[str, tuple[str, ...]] = {
    "policy": POLICY_GROUP,
    "value": VALUE_GROUP,
    "other": OTHER_GROUP,
}


def check_grouping_partitions_components() -> None:
    """Every trained component in exactly one group — else shares can't sum to 100%."""
    seen: set[str] = set()
    for name, members in GROUPS.items():
        for comp in members:
            if comp in seen:
                raise ValueError(f"component {comp!r} appears in more than one group")
            if comp not in COMPONENT_WEIGHT_KEY:
                raise ValueError(f"group {name!r} lists unknown component {comp!r}")
            seen.add(comp)
    missing = sorted(set(COMPONENT_WEIGHT_KEY) - seen)
    if missing:
        raise ValueError(f"components in no group (shares would not sum to 100%): {missing}")
    overlap = sorted(set(DIAGNOSTIC_COMPONENTS) & seen)
    if overlap:
        raise ValueError(f"diagnostic components must stay out of the groups: {overlap}")


def _classify_params(
    model: torch.nn.Module,
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[str], list[str], list[str]]:
    """Split parameters into (trunk, trunk_names, head_names, policy_shared_names).

    Three buckets, not two: the shared policy adapter is policy-only yet shared by
    every policy head, so it is neither the global trunk nor a private head.
    """
    trunk: list[tuple[str, torch.nn.Parameter]] = []
    trunk_names: list[str] = []
    head_names: list[str] = []
    policy_shared_names: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top = name.split(".", 1)[0]
        if top in POLICY_SHARED_PREFIXES:
            policy_shared_names.append(name)
        elif top in HEAD_PREFIXES:
            head_names.append(name)
        else:
            trunk.append((name, param))
            trunk_names.append(name)
    return trunk, trunk_names, head_names, policy_shared_names


def _load_one_shard_arrays(model_cfg: Any, *, replay_dir: str) -> dict[str, np.ndarray]:
    from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays

    paths = iter_shard_paths(Path(replay_dir))
    if not paths:
        raise SystemExit(f"no shards under {replay_dir}")
    # Mid/recent shard: back off from the very newest (may be mid-write); walk
    # backward until one loads + prepares cleanly.
    candidates = paths[-6:-1][::-1] or paths[::-1]
    lc0_legacy = uses_lc0_root_legacy_meta(model_cfg.input_history_encoding)
    target_planes = input_plane_count(model_cfg.input_extra_features)
    last_err: Exception | None = None
    for path in candidates:
        try:
            arrs, _meta = load_shard_arrays(path, lazy=False)
            arrs = _select_configured_input_history(
                arrs,
                input_history_encoding=model_cfg.input_history_encoding,
                prefer_recorded_lc0_root=True,
                synthetic_lc0_root_history=False,
                lc0_root_legacy_meta=lc0_legacy,
                target_input_planes=target_planes,
                upgrade_v1_planes=False,
            )
            arrs = _convert_policy_targets(arrs, policy_encoding=model_cfg.policy_encoding)
            print(json.dumps({"event": "shard_loaded", "path": str(path),
                              "positions": int(np.asarray(arrs["x"]).shape[0])}), flush=True)
            return arrs
        except Exception as exc:
            last_err = exc
            print(json.dumps({"event": "shard_skip", "path": str(path),
                              "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    raise SystemExit(f"could not load any shard: {last_err}")


def _grad_norm(
    component: torch.Tensor, trunk_params: list[torch.nn.Parameter],
) -> float:
    # ``allow_unused=True`` yields None for params this component never
    # touches, whatever the stubs say; widen the type so the guard is honest.
    grads = cast(
        "tuple[torch.Tensor | None, ...]",
        torch.autograd.grad(component, trunk_params, retain_graph=True, allow_unused=True),
    )
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().pow(2).sum().item())
    return math.sqrt(total)


def _build_trainer(device: str, args: argparse.Namespace) -> tuple[Trainer, Any]:
    cfg = flatten_run_config_defaults(load_yaml_file(args.config))
    model_cfg = _model_config_from_offline_config(cfg)
    model = build_model(model_cfg)
    trainer_kwargs = trainer_kwargs_from_config(cfg, log_dir=Path(args.out).parent / "tb")
    trainer_kwargs.update(
        device=device,
        optimizer="aurora",
        matrix_optimizer_scope="mlp_out",
        use_compile=False,
        prefetch_batches=False,
        model_config=model_cfg,
    )
    trainer = Trainer(model, **trainer_kwargs)
    trainer.load(Path(args.checkpoint))
    return trainer, model_cfg


def _measure(
    trainer: Trainer, model_cfg: Any, args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Mean trunk-grad norm and mean loss value per component over N batches."""
    model = trainer.model
    loss_kwargs = trainer._loss_kwargs
    trunk, _names, _heads, _pol_shared = _classify_params(model)
    trunk_params = [p for _, p in trunk]
    measured = [*COMPONENT_ORDER, *DIAGNOSTIC_COMPONENTS]

    raw_sums: dict[str, float] = dict.fromkeys(measured, 0.0)
    raw_counts: dict[str, int] = dict.fromkeys(measured, 0)
    loss_val_sums: dict[str, float] = dict.fromkeys(measured, 0.0)

    arrs = _load_one_shard_arrays(model_cfg, replay_dir=args.replay_dir)
    sampler = _ArraySampler(arrs, np.random.default_rng(0))

    n_done = 0
    for batch in trainer._iter_prefetched_batches(
        _as_replay_buffer(sampler),
        batch_size=int(args.batch_size),
        mirror_prob=0.0,
        count=int(args.n_batches),
    ):
        with trainer._amp_context():
            _rel = batch.get("relations")
            out = model(batch["x"], relations=_rel) if _rel is not None else model(batch["x"])
            losses = compute_loss(out, batch, **loss_kwargs)
        for comp in measured:
            if comp not in losses:
                continue
            tensor = losses[comp]
            if not tensor.requires_grad:
                continue
            raw_sums[comp] += _grad_norm(tensor, trunk_params)
            raw_counts[comp] += 1
            loss_val_sums[comp] += float(tensor.detach().item())
        # Free the graph for this batch.
        del out, losses
        n_done += 1
        print(json.dumps({"event": "batch_done", "batch": n_done}), flush=True)

    raw_norm = {c: raw_sums[c] / raw_counts[c] for c in measured if raw_counts[c] > 0}
    loss_val = {c: loss_val_sums[c] / raw_counts[c] for c in measured if raw_counts[c] > 0}
    weights = {ck: float(loss_kwargs.get(wk, 0.0)) for ck, wk in COMPONENT_WEIGHT_KEY.items()}
    return raw_norm, loss_val, weights


def _run(device: str, args: argparse.Namespace) -> dict[str, Any]:
    check_grouping_partitions_components()
    trainer, model_cfg = _build_trainer(device, args)
    trainer.model.train()
    _trunk, trunk_names, head_names, pol_shared_names = _classify_params(trainer.model)

    print(json.dumps({
        "event": "param_classification",
        "trunk_param_tensors": len(trunk_names),
        "head_param_tensors": len(head_names),
        "policy_shared_param_tensors": len(pol_shared_names),
        "trunk_examples": trunk_names[:6],
        "head_examples": head_names[:6],
        "policy_shared": pol_shared_names,
        # ⚑ Non-empty => "trunk share" EXCLUDES the shared policy adapter, so this
        # run's numbers are NOT comparable with any run recorded before the adapter
        # existed (e.g. the 2026-08-12 soft 94.1% / own 94.7% pair).
        "trunk_share_comparable_with_pre_adapter_runs": not pol_shared_names,
    }, indent=2), flush=True)

    raw_norm, loss_val, weights = _measure(trainer, model_cfg, args)
    print(json.dumps({"event": "loss_weights", "weights": weights}, indent=2), flush=True)

    # Denominators cover the TRAINED components only.
    trained = [c for c in COMPONENT_ORDER if c in raw_norm]
    weighted_norm = {c: weights[c] * raw_norm[c] for c in trained}
    total_weighted = sum(weighted_norm.values()) or 1.0
    total_raw = sum(raw_norm[c] for c in trained) or 1.0

    table = [
        {
            "component": comp,
            "loss_value": loss_val[comp],
            "raw_trunk_grad_norm": raw_norm[comp],
            "w": weights[comp],
            "weighted_trunk_grad_norm": weighted_norm[comp],
            "weighted_pct_share": 100.0 * weighted_norm[comp] / total_weighted,
            "raw_pct_share": 100.0 * raw_norm[comp] / total_raw,
        }
        for comp in trained
    ]
    diagnostics = [
        {
            "component": comp,
            "loss_value": loss_val[comp],
            "raw_trunk_grad_norm": raw_norm[comp],
            "note": "diagnostic only - excluded from every share, no gradient in `total`",
        }
        for comp in DIAGNOSTIC_COMPONENTS
        if comp in raw_norm
    ]

    groups: dict[str, dict[str, Any]] = {}
    for name, members in GROUPS.items():
        weighted = sum(weighted_norm.get(c, 0.0) for c in members)
        raw = sum(raw_norm.get(c, 0.0) for c in members)
        groups[name] = {
            "members": list(members),
            "weighted": weighted,
            "raw": raw,
            "weighted_pct": 100.0 * weighted / total_weighted,
            "raw_pct": 100.0 * raw / total_raw,
        }
    groups_weighted_pct_sum = sum(float(g["weighted_pct"]) for g in groups.values())
    if abs(groups_weighted_pct_sum - 100.0) > 1e-6:
        raise ValueError(
            f"group weighted shares sum to {groups_weighted_pct_sum:.4f}%, not 100% - "
            "the grouping no longer partitions the trained components"
        )

    prim_pol = raw_norm.get("policy_ce", 0.0)
    prim_val = raw_norm.get("blended_wdl_ce", 0.0)
    w_prim_pol = weighted_norm.get("policy_ce", 0.0)
    w_prim_val = weighted_norm.get("blended_wdl_ce", 0.0)
    headline = {
        "primary_policy": "policy_ce",
        "primary_value": "blended_wdl_ce",
        "raw_policy_norm": prim_pol,
        "raw_value_norm": prim_val,
        "raw_ratio_policy_over_value": (prim_pol / prim_val) if prim_val > 0 else None,
        "weighted_policy_norm": w_prim_pol,
        "weighted_value_norm": w_prim_val,
        "weighted_ratio_policy_over_value": (w_prim_pol / w_prim_val) if w_prim_val > 0 else None,
    }

    return {
        "device": device,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "replay_dir": str(args.replay_dir),
        "batch_size": int(args.batch_size),
        "n_batches": int(args.n_batches),
        "trunk_param_tensors": len(trunk_names),
        "head_param_tensors": len(head_names),
        "trunk_examples": trunk_names[:8],
        "head_examples": head_names[:8],
        "loss_weights": weights,
        "table": table,
        "diagnostics": diagnostics,
        "groups": groups,
        "groups_weighted_pct_sum": groups_weighted_pct_sum,
        "headline": headline,
        "total_weighted_norm": total_weighted,
        "total_raw_norm": total_raw,
    }


def _print_report(result: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(f"PER-HEAD TRUNK GRADIENT-NORM SHARE  (device={result['device']})")
    print("=" * 100)
    print(f"trunk param tensors: {result['trunk_param_tensors']}   "
          f"head param tensors: {result['head_param_tensors']}")
    print(f"trunk examples: {result['trunk_examples'][:4]}")
    print(f"head  examples: {result['head_examples'][:4]}")
    print("-" * 100)
    hdr = (f"{'component':<18}{'loss':>8}{'raw_norm':>13}{'w':>9}"
           f"{'wtd_norm':>13}{'wtd%':>9}{'raw%':>9}")
    print(hdr)
    print("-" * 100)
    for row in result["table"]:
        print(f"{row['component']:<18}{row['loss_value']:>8.3f}"
              f"{row['raw_trunk_grad_norm']:>13.5f}{row['w']:>9.3f}"
              f"{row['weighted_trunk_grad_norm']:>13.5f}"
              f"{row['weighted_pct_share']:>8.2f}%{row['raw_pct_share']:>8.2f}%")
    print("-" * 100)
    g = result["groups"]
    print("GROUP SHARES (weighted % | raw %) - trained components only, sums to 100%:")
    for name in ("policy", "value", "other"):
        members = "+".join(g[name]["members"])
        print(f"  {name.upper():<6} ({members}): "
              f"{g[name]['weighted_pct']:6.2f}% | {g[name]['raw_pct']:6.2f}%")
    print(f"  SUM: {result['groups_weighted_pct_sum']:6.2f}%")
    if result["diagnostics"]:
        print("-" * 100)
        print("DIAGNOSTIC (no gradient in `total`, excluded from the shares above):")
        for row in result["diagnostics"]:
            print(f"  {row['component']:<18}loss {row['loss_value']:>8.3f}  "
                  f"raw_norm {row['raw_trunk_grad_norm']:.5f}")
    print("-" * 100)
    h = result["headline"]
    rr = h["raw_ratio_policy_over_value"]
    wr = h["weighted_ratio_policy_over_value"]
    print("HEADLINE  primary-policy (policy_ce) vs primary-value (blended_wdl_ce):")
    print(f"  raw trunk-grad norms:      policy {h['raw_policy_norm']:.5f}  "
          f"value {h['raw_value_norm']:.5f}  -> ratio {rr:.3f}x"
          if rr is not None else "  raw ratio: n/a")
    print(f"  weighted trunk-grad norms: policy {h['weighted_policy_norm']:.5f}  "
          f"value {h['weighted_value_norm']:.5f}  -> ratio {wr:.3f}x"
          if wr is not None else "  weighted ratio: n/a")
    print("=" * 100 + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-head trunk gradient-norm share")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--replay-dir", default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--out", default=DEFAULT_RESULT_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-batches", type=int, default=DEFAULT_N_BATCHES)
    parser.add_argument("--gpu-mem-fraction", type=float, default=DEFAULT_GPU_MEM_FRACTION)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction), 0)
        device = "cuda"
    try:
        result = _run(device, args)
    except torch.cuda.OutOfMemoryError as exc:
        print(json.dumps({"event": "cuda_oom_fallback_cpu", "error": str(exc)}), flush=True)
        torch.cuda.empty_cache()
        result = _run("cpu", args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _print_report(result)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
