#!/usr/bin/env python3
"""Train candidate optimizer scopes for one fixed replay epoch.

This bypasses selfplay/Ray entirely: it streams existing replay shards once,
optionally converts policy targets to the requested output encoding, and writes
one JSONL row per candidate. It is intended for architecture/optimizer probes
where generating fresh games would confound the comparison.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves import (
    COMPACT_POLICY_SIZE,
    FULL_TO_COMPACT_POLICY,
    POLICY_ENCODING_AZ_4672,
    POLICY_ENCODING_LC0_1858,
    POLICY_SIZE,
    normalize_policy_encoding,
    policy_batch_to_encoding,
)
from chess_anti_engine.replay.dataset import LEGAL_MASK_FIELDS
from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays
from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


POLICY_FIELDS = (
    "policy_target",
    "sf_policy_target",
    "policy_soft_target",
    "future_policy_target",
)

_LC0_HISTORY_STEPS = 8
_LC0_PIECE_PLANES = 12


class _FixedBatch:
    def __init__(self, arrs: dict[str, np.ndarray], rng: np.random.Generator):
        self._arrs = arrs
        self.rng = rng

    def sample_batch_arrays(self, _batch_size: int, *, wdl_balance: bool = True) -> dict[str, np.ndarray]:
        del wdl_balance
        return self._arrs


class _ArraySampler:
    def __init__(self, arrs: dict[str, np.ndarray], rng: np.random.Generator):
        self._arrs = arrs
        self.rng = rng
        self._n = int(np.asarray(arrs["x"]).shape[0])

    def sample_batch_arrays(self, batch_size: int, *, wdl_balance: bool = True) -> dict[str, np.ndarray]:
        del wdl_balance
        idx = self.rng.integers(0, self._n, size=int(batch_size), dtype=np.int64)
        return _slice_arrays(self._arrs, idx)


def _slice_arrays(arrs: dict[str, Any], idx: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in arrs.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            out[key] = np.array(arr, copy=True)
        else:
            out[key] = np.array(arr[idx], copy=True, order="C")
    return out


def _renorm_rows(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    sums = out.sum(axis=1, keepdims=True)
    good = sums[:, 0] > 0.0
    if np.any(good):
        out[good] = out[good] / sums[good]
    return out


def _convert_policy_targets(arrs: dict[str, Any], *, policy_encoding: str) -> dict[str, np.ndarray]:
    enc = normalize_policy_encoding(policy_encoding)
    out = {k: np.asarray(v) for k, v in arrs.items()}
    if enc == POLICY_ENCODING_AZ_4672:
        return out

    for key in POLICY_FIELDS:
        if key not in out:
            continue
        arr = np.asarray(out[key])
        if arr.ndim == 2 and int(arr.shape[1]) == POLICY_SIZE:
            out[key] = _renorm_rows(policy_batch_to_encoding(arr, policy_encoding=enc))
    for key in LEGAL_MASK_FIELDS:
        if key not in out:
            continue
        arr = np.asarray(out[key])
        if arr.ndim == 2 and int(arr.shape[1]) == POLICY_SIZE:
            out[key] = policy_batch_to_encoding(arr, policy_encoding=enc).astype(arr.dtype, copy=False)

    if "sf_move_index" in out:
        idx = np.asarray(out["sf_move_index"], dtype=np.int64)
        valid = (idx >= 0) & (idx < POLICY_SIZE)
        compact = np.full(idx.shape, -1, dtype=np.int64)
        compact[valid] = FULL_TO_COMPACT_POLICY[idx[valid]]
        has = compact >= 0
        out["sf_move_index"] = np.where(has, compact, 0).astype(np.int64)
        if "has_sf_move" in out:
            out["has_sf_move"] = (np.asarray(out["has_sf_move"], dtype=np.float32) * has.astype(np.float32))
    out["_policy_size"] = np.array(COMPACT_POLICY_SIZE, dtype=np.int32)
    return out


def _legacy_x_to_synthetic_lc0_root(x: np.ndarray) -> np.ndarray:
    """Approximate LC0 root-history layout from stored legacy replay tensors.

    Replay shards store already-encoded ``x`` planes, not boards or move
    histories.  This can still remap the legacy piece-history planes because
    adjacent plies alternate side-to-move: odd history slots need us/them swap
    plus a vertical flip to become root-POV.  Metadata is best-effort: castling,
    rule50, repetition, and bias can be moved; LC0's color flag cannot be
    recovered from side-to-move-normalized legacy planes and is left zero.
    """
    src = np.asarray(x)
    if src.ndim != 4 or src.shape[1] < 112:
        raise ValueError(f"expected x with shape (N, >=112, 8, 8), got {src.shape}")
    out = np.array(src, copy=True, order="C")
    out[:, :112, :, :] = 0

    for hist_idx in range(_LC0_HISTORY_STEPS):
        legacy_start = hist_idx * _LC0_PIECE_PLANES
        root_start = hist_idx * (_LC0_PIECE_PLANES + 1)
        planes = src[:, legacy_start:legacy_start + _LC0_PIECE_PLANES, :, :]
        if hist_idx % 2 == 0:
            out[:, root_start:root_start + _LC0_PIECE_PLANES, :, :] = planes
        else:
            out[:, root_start:root_start + 6, :, :] = planes[:, 6:12, ::-1, :]
            out[:, root_start + 6:root_start + 12, :, :] = planes[:, 0:6, ::-1, :]

        rep_plane = 103 + hist_idx
        if rep_plane < 111:
            out[:, root_start + 12, :, :] = src[:, rep_plane, :, :]

    # Legacy metadata: us-K, us-Q, them-K, them-Q, EP, side-to-move, rule50.
    # Root metadata:   us-Q, us-K, them-Q, them-K, color, rule50, zero, bias.
    out[:, 104, :, :] = src[:, 97, :, :]
    out[:, 105, :, :] = src[:, 96, :, :]
    out[:, 106, :, :] = src[:, 99, :, :]
    out[:, 107, :, :] = src[:, 98, :, :]
    out[:, 108, :, :] = 0  # True root color is not recoverable from replay x.
    out[:, 109, :, :] = src[:, 102, :, :]
    out[:, 110, :, :] = 0
    out[:, 111, :, :] = src[:, 111, :, :]
    return out


def _maybe_synthetic_history(arrs: dict[str, Any], *, synthetic_lc0_root_history: bool) -> dict[str, np.ndarray]:
    if not synthetic_lc0_root_history:
        return {k: np.asarray(v) for k, v in arrs.items()}
    out = {k: np.asarray(v) for k, v in arrs.items()}
    out["x"] = _legacy_x_to_synthetic_lc0_root(np.asarray(out["x"]))
    return out


def _select_recorded_input_history(
    arrs: dict[str, Any],
    *,
    input_history_encoding: str,
    prefer_recorded_lc0_root: bool,
) -> dict[str, np.ndarray]:
    out = {k: np.asarray(v) for k, v in arrs.items()}
    if (
        prefer_recorded_lc0_root
        and str(input_history_encoding).lower() in ("lc0_root", "lc0", "root", "root_pov", "lc0_13")
        and "x_lc0_root" in out
    ):
        has = np.asarray(out.get("has_x_lc0_root", np.zeros((out["x"].shape[0],), dtype=np.uint8)))
        if bool(np.all(has != 0)):
            out["x"] = np.asarray(out["x_lc0_root"])
    return out


def _concat(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = set.intersection(*(set(c.keys()) for c in chunks))
    out: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        first = np.asarray(chunks[0][key])
        if first.ndim == 0:
            out[key] = np.array(first, copy=True)
        else:
            out[key] = np.concatenate([np.asarray(c[key]) for c in chunks], axis=0)
    return out


def _candidate_spec(name: str) -> tuple[str, str]:
    raw = str(name).strip().lower()
    aliases = {
        "adamw": ("adamw", "default"),
        "aurora": ("aurora", "block_all"),
        "aurora_blocks": ("aurora", "block_all"),
        "aurora_block_all": ("aurora", "block_all"),
        "aurora_all_block": ("aurora", "mlp_attn_all"),
        "aurora_mlp_attn_all": ("aurora", "mlp_attn_all"),
        "aurora_mlp": ("aurora", "mlp_only"),
        "aurora_mlp_only": ("aurora", "mlp_only"),
        "aurora_mlp_out": ("aurora", "mlp_out"),
        "aurora_mlp_o": ("aurora", "mlp_out"),
        "aurora_mlp_out_v": ("aurora", "mlp_out_v"),
        "aurora_mlp_v_out": ("aurora", "mlp_out_v"),
        "aurora_mlp_attn_ov": ("aurora", "mlp_out_v"),
        "aurora_all": ("aurora", "mlp_attn_all"),
        "muon_mlp": ("muon", "mlp_only"),
        "muon_mlp_only": ("muon", "mlp_only"),
        "muon_mlp_out": ("muon", "mlp_out"),
        "muon_mlp_o": ("muon", "mlp_out"),
        "muon_mlp_out_v": ("muon", "mlp_out_v"),
        "muon_mlp_attn_ov": ("muon", "mlp_out_v"),
        "muon_all": ("muon", "mlp_attn_all"),
        "muon_all_block": ("muon", "mlp_attn_all"),
        "cosmos": ("cosmos", "default"),
        "cosmos_mlp": ("cosmos", "mlp_only"),
        "cosmos_mlp_only": ("cosmos", "mlp_only"),
        "cosmos_mlp_out": ("cosmos", "mlp_out"),
        "cosmos_mlp_out_v": ("cosmos", "mlp_out_v"),
        "cosmos_mlp_attn_ov": ("cosmos", "mlp_out_v"),
        "cosmos_all": ("cosmos", "mlp_attn_all"),
        "cosmos_fast": ("cosmos_fast", "default"),
        "cosmos_fast_mlp": ("cosmos_fast", "mlp_only"),
        "cosmos_fast_mlp_only": ("cosmos_fast", "mlp_only"),
        "cosmos_fast_mlp_out": ("cosmos_fast", "mlp_out"),
        "cosmos_fast_mlp_out_v": ("cosmos_fast", "mlp_out_v"),
        "cosmos_fast_mlp_attn_ov": ("cosmos_fast", "mlp_out_v"),
        "cosmos_fast_all": ("cosmos_fast", "mlp_attn_all"),
        "soap": ("soap", "default"),
        "soap_mlp": ("soap", "mlp_only"),
        "soap_mlp_only": ("soap", "mlp_only"),
        "soap_mlp_out": ("soap", "mlp_out"),
        "soap_mlp_out_v": ("soap", "mlp_out_v"),
        "soap_mlp_attn_ov": ("soap", "mlp_out_v"),
        "soap_all": ("soap", "mlp_attn_all"),
    }
    if raw not in aliases:
        raise ValueError(f"unknown candidate {name!r}; expected one of {sorted(aliases)}")
    return aliases[raw]


def _parse_optional_float_arg(value: str) -> float | None:
    raw = str(value).strip().lower()
    if raw in ("none", "null", "off", "false"):
        return None
    return float(raw)


def _parse_bool_choice(value: str) -> bool:
    raw = str(value).strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected one of on/off/true/false, got {value!r}")


def _model_config_from_flat(cfg: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        kind=str(cfg.get("model", "transformer")),
        embed_dim=int(cfg.get("embed_dim", 256)),
        num_layers=int(cfg.get("num_layers", 6)),
        num_heads=int(cfg.get("num_heads", 8)),
        ffn_mult=float(cfg.get("ffn_mult", 2.0)),
        use_smolgen=bool(cfg.get("use_smolgen", True)),
        use_nla=bool(cfg.get("use_nla", False)),
        use_qk_rmsnorm=bool(cfg.get("use_qk_rmsnorm", False)),
        use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        input_pos_encoding=str(cfg.get("input_pos_encoding", "none")),
        qkv_projection=str(cfg.get("qkv_projection", "fused")),
        use_deepnorm=bool(cfg.get("use_deepnorm", False)),
        policy_encoding=normalize_policy_encoding(cfg.get("policy_encoding", POLICY_ENCODING_AZ_4672)),
        input_history_encoding=str(cfg.get("input_history_encoding", "legacy")),
        input_global_embedding=str(cfg.get("input_global_embedding", "none")),
        input_global_embedding_channels=int(cfg.get("input_global_embedding_channels", 0)),
        input_square_embedding=str(cfg.get("input_square_embedding", "none")),
        smolgen_mode=str(cfg.get("smolgen_mode", "shared")),
        smolgen_bias_scale=str(cfg.get("smolgen_bias_scale", "none")),
        smolgen_bias_norm=str(cfg.get("smolgen_bias_norm", "none")),
        arc_attention_bias=str(cfg.get("arc_attention_bias", "none")),
        smolgen_relation_basis=bool(cfg.get("smolgen_relation_basis", False)),
        smolgen_relation_norm=str(cfg.get("smolgen_relation_norm", "none")),
        smolgen_relation_coeff_norm=str(cfg.get("smolgen_relation_coeff_norm", "none")),
        smolgen_relation_scale=str(cfg.get("smolgen_relation_scale", "none")),
    )


def _train_candidate(
    *,
    candidate: str,
    cfg: dict[str, Any],
    shard_paths: list[Path],
    eval_arrs: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    optimizer, scope = _candidate_spec(candidate)
    run_dir = Path(args.out_dir) / candidate
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    model_cfg = _model_config_from_flat(cfg)
    model = build_model(model_cfg)
    trainer_kwargs = trainer_kwargs_from_config(cfg, log_dir=run_dir / "tb")
    trainer_kwargs.update(
        optimizer=optimizer,
        matrix_optimizer_scope=scope,
        matrix_lr_multiplier=float(args.matrix_lr_multiplier),
        matrix_weight_decay=float(args.matrix_weight_decay),
        aux_weight_decay=float(args.aux_weight_decay),
        weight_decay_mode=str(args.weight_decay_mode),
        soda_scope=str(args.soda_scope),
        soda_start_step=int(args.soda_start_step),
        aurora_uw_floor=float(args.aurora_uw_floor),
        aurora_pp_iterations=int(args.aurora_pp_iterations),
        aurora_pp_beta=float(args.aurora_pp_beta),
        aurora_polar_steps=int(args.aurora_polar_steps),
        aurora_polar_method=str(args.aurora_polar_method),
        aurora_polar_dtype=str(args.aurora_polar_dtype),
        aurora_polar_safety=float(args.aurora_polar_safety),
        use_compile=bool(args.compile),
        use_amp=not bool(args.no_amp),
        prefetch_batches=False,
        model_config=model_cfg,
    )
    if args.zclip_max_norm is not None:
        trainer_kwargs["zclip_max_norm"] = _parse_optional_float_arg(args.zclip_max_norm)
    if args.zclip_z_thresh is not None:
        trainer_kwargs["zclip_z_thresh"] = float(args.zclip_z_thresh)
    if args.zclip_clip_factor is not None:
        trainer_kwargs["zclip_clip_factor"] = float(args.zclip_clip_factor)
    trainer = Trainer(model, **trainer_kwargs)
    if args.init_checkpoint:
        trainer.load(Path(args.init_checkpoint))
    rng = np.random.default_rng(int(args.seed))

    steps = 0
    positions = 0
    last_metrics: TrainMetrics | None = None
    t0 = time.time()
    for shard_i, shard in enumerate(shard_paths, start=1):
        arrs, _meta = load_shard_arrays(shard, lazy=False)
        arrs = _select_recorded_input_history(
            arrs,
            input_history_encoding=model_cfg.input_history_encoding,
            prefer_recorded_lc0_root=bool(args.prefer_recorded_lc0_root),
        )
        arrs = _maybe_synthetic_history(
            arrs,
            synthetic_lc0_root_history=bool(args.synthetic_lc0_root_history),
        )
        arrs = _convert_policy_targets(arrs, policy_encoding=model_cfg.policy_encoding)
        n = int(np.asarray(arrs["x"]).shape[0])
        order = rng.permutation(n)
        for start in range(0, n, int(args.batch_size)):
            idx = order[start:start + int(args.batch_size)]
            batch = _slice_arrays(arrs, idx)
            last_metrics = trainer.train_steps(_FixedBatch(batch, rng), batch_size=int(idx.shape[0]), steps=1)
            steps += 1
            positions += int(idx.shape[0])
        if shard_i % int(args.report_every_shards) == 0:
            elapsed = time.time() - t0
            row: dict[str, Any] = {
                "event": "progress",
                "candidate": candidate,
                "shards": shard_i,
                "positions": positions,
                "steps": steps,
                "positions_per_s": positions / max(elapsed, 1e-9),
            }
            if last_metrics is not None:
                metric_values = dataclasses.asdict(last_metrics)
                for key in (
                    "loss",
                    "policy_loss",
                    "soft_policy_loss",
                    "future_policy_loss",
                    "wdl_loss",
                    "blended_wdl_loss",
                    "sf_move_loss",
                    "sf_move_acc",
                    "policy_own_acc_top1",
                    "policy_future_acc_top1",
                    "aurora_uw_ratio_min",
                    "aurora_uw_ratio_p10",
                    "aurora_uw_ratio_median",
                    "aurora_uw_ratio_p90",
                    "aurora_uw_scale_max",
                    "aurora_uw_floored_frac",
                    "aurora_uw_effective_ratio_min",
                    "aurora_uw_effective_ratio_median",
                ):
                    row[f"last_{key}"] = metric_values[key]
            print(json.dumps(row), flush=True)

    eval_steps = max(1, int(args.eval_steps))
    eval_metrics = trainer.eval_steps(
        _ArraySampler(eval_arrs, np.random.default_rng(int(args.seed) + 999)),
        batch_size=int(args.batch_size),
        steps=eval_steps,
    )
    out = {
        "candidate": candidate,
        "optimizer": optimizer,
        "matrix_optimizer_scope": scope,
        "matrix_lr_multiplier": float(args.matrix_lr_multiplier),
        "matrix_weight_decay": float(args.matrix_weight_decay),
        "aux_weight_decay": float(args.aux_weight_decay),
        "weight_decay_mode": str(args.weight_decay_mode),
        "soda_scope": str(args.soda_scope),
        "soda_start_step": int(args.soda_start_step),
        "aurora_uw_floor": float(args.aurora_uw_floor),
        "aurora_pp_iterations": int(args.aurora_pp_iterations),
        "aurora_pp_beta": float(args.aurora_pp_beta),
        "aurora_polar_steps": int(args.aurora_polar_steps),
        "aurora_polar_method": str(args.aurora_polar_method),
        "aurora_polar_dtype": str(args.aurora_polar_dtype),
        "aurora_polar_safety": float(args.aurora_polar_safety),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else "",
        "positions": positions,
        "steps": steps,
        "elapsed_s": time.time() - t0,
        "model_config": dataclasses.asdict(model_cfg),
        **{f"eval_{k}": v for k, v in dataclasses.asdict(eval_metrics).items()},
    }
    trainer.save(run_dir / "trainer.pt")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/bt4_aurora_asha.yaml")
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--out-dir", default="runs/offline_replay_epoch")
    ap.add_argument(
        "--candidates",
        nargs="+",
        default=["adamw", "aurora_mlp_only", "aurora_mlp_out", "aurora_mlp_attn_all", "aurora_blocks"],
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-positions", type=int, default=2048)
    ap.add_argument("--eval-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--init-checkpoint",
        default="",
        help="Optional Trainer.save() checkpoint to load before streaming the replay epoch.",
    )
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--report-every-shards", type=int, default=50)
    ap.add_argument("--embed-dim", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--ffn-mult", type=float, default=None)
    ap.add_argument("--smolgen", type=_parse_bool_choice, default=None, help="Override model.use_smolgen.")
    ap.add_argument("--smolgen-mode", choices=["shared", "per_layer"], default=None)
    ap.add_argument("--smolgen-bias-scale", choices=["none", "layer", "layer_head"], default=None)
    ap.add_argument("--smolgen-bias-norm", choices=["none", "center", "center_rms"], default=None)
    ap.add_argument("--arc-attention-bias", choices=["none", "basic"], default=None)
    ap.add_argument("--smolgen-relation-basis", type=_parse_bool_choice, default=None)
    ap.add_argument(
        "--smolgen-relation-norm",
        choices=["none", "branch_center", "branch_center_rms", "basis_center", "basis_center_rms"],
        default=None,
    )
    ap.add_argument("--smolgen-relation-coeff-norm", choices=["none", "rms"], default=None)
    ap.add_argument("--smolgen-relation-scale", choices=["none", "layer", "layer_head"], default=None)
    ap.add_argument("--input-pos-encoding", choices=["none", "arc", "arc_adapter"], default=None)
    ap.add_argument("--qkv-projection", choices=["fused", "split"], default=None)
    ap.add_argument("--input-global-embedding", choices=["none", "bt4_board"], default=None)
    ap.add_argument("--input-global-embedding-channels", type=int, default=None)
    ap.add_argument("--input-square-embedding", choices=["none", "add", "ma_gate"], default=None)
    ap.add_argument("--deepnorm", type=_parse_bool_choice, default=None, help="Override model.use_deepnorm.")
    ap.add_argument("--policy-encoding", default=None)
    ap.add_argument("--input-history-encoding", default=None)
    ap.add_argument(
        "--synthetic-lc0-root-history",
        action="store_true",
        help="Approximate LC0 root-history planes by remapping stored legacy replay x tensors.",
    )
    ap.add_argument(
        "--prefer-recorded-lc0-root",
        action="store_true",
        help="Use shard x_lc0_root as x when training an lc0_root input model and the field is present.",
    )
    ap.add_argument("--matrix-lr-multiplier", type=float, default=20.0)
    ap.add_argument("--matrix-weight-decay", type=float, default=1e-4)
    ap.add_argument("--aux-weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--weight-decay-mode",
        choices=["weight_decay", "soda"],
        default="weight_decay",
        help="Use ordinary optimizer weight decay, or replace nonzero decay groups with SODA averaging.",
    )
    ap.add_argument(
        "--soda-scope",
        choices=["decay", "hidden_matrix_only"],
        default="decay",
        help="SODA scope: preserve old nonzero-decay behavior, or force hidden matrix groups only.",
    )
    ap.add_argument(
        "--soda-start-step",
        type=int,
        default=0,
        help="Optimizer step at which SODA averaging starts; beta still uses the global step count.",
    )
    ap.add_argument("--aurora-uw-floor", type=float, default=0.0)
    ap.add_argument("--aurora-pp-iterations", type=int, default=2)
    ap.add_argument("--aurora-pp-beta", type=float, default=0.5)
    ap.add_argument("--aurora-polar-steps", type=int, default=12)
    ap.add_argument("--aurora-polar-method", choices=["simple", "polar_express"], default="simple")
    ap.add_argument("--aurora-polar-dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    ap.add_argument("--aurora-polar-safety", type=float, default=1.01)
    ap.add_argument("--zclip-max-norm", default=None)
    ap.add_argument("--zclip-z-thresh", type=float, default=None)
    ap.add_argument("--zclip-clip-factor", type=float, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    cfg = flatten_run_config_defaults(load_yaml_file(args.config))
    cfg["policy_encoding"] = normalize_policy_encoding(args.policy_encoding or cfg.get("policy_encoding", POLICY_ENCODING_LC0_1858))
    if args.embed_dim is not None:
        cfg["embed_dim"] = int(args.embed_dim)
    if args.num_layers is not None:
        cfg["num_layers"] = int(args.num_layers)
    if args.num_heads is not None:
        cfg["num_heads"] = int(args.num_heads)
    if args.ffn_mult is not None:
        cfg["ffn_mult"] = float(args.ffn_mult)
    if args.input_history_encoding is not None:
        cfg["input_history_encoding"] = str(args.input_history_encoding)
    if args.smolgen is not None:
        cfg["use_smolgen"] = bool(args.smolgen)
    if args.smolgen_mode is not None:
        cfg["smolgen_mode"] = str(args.smolgen_mode)
    if args.smolgen_bias_scale is not None:
        cfg["smolgen_bias_scale"] = str(args.smolgen_bias_scale)
    if args.smolgen_bias_norm is not None:
        cfg["smolgen_bias_norm"] = str(args.smolgen_bias_norm)
    if args.arc_attention_bias is not None:
        cfg["arc_attention_bias"] = str(args.arc_attention_bias)
    if args.smolgen_relation_basis is not None:
        cfg["smolgen_relation_basis"] = bool(args.smolgen_relation_basis)
    if args.smolgen_relation_norm is not None:
        cfg["smolgen_relation_norm"] = str(args.smolgen_relation_norm)
    if args.smolgen_relation_coeff_norm is not None:
        cfg["smolgen_relation_coeff_norm"] = str(args.smolgen_relation_coeff_norm)
    if args.smolgen_relation_scale is not None:
        cfg["smolgen_relation_scale"] = str(args.smolgen_relation_scale)
    if args.input_pos_encoding is not None:
        cfg["input_pos_encoding"] = str(args.input_pos_encoding)
    if args.qkv_projection is not None:
        cfg["qkv_projection"] = str(args.qkv_projection)
    if args.input_global_embedding is not None:
        cfg["input_global_embedding"] = str(args.input_global_embedding)
    if args.input_global_embedding_channels is not None:
        cfg["input_global_embedding_channels"] = int(args.input_global_embedding_channels)
    if args.input_square_embedding is not None:
        cfg["input_square_embedding"] = str(args.input_square_embedding)
    if args.deepnorm is not None:
        cfg["use_deepnorm"] = bool(args.deepnorm)
    if args.lr is not None:
        cfg["lr"] = float(args.lr)
    model_cfg = _model_config_from_flat(cfg)

    shard_paths = iter_shard_paths(args.replay_dir)
    if args.max_shards > 0:
        shard_paths = shard_paths[:int(args.max_shards)]
    if not shard_paths:
        raise SystemExit(f"no replay shards found in {args.replay_dir}")

    eval_chunks: list[dict[str, np.ndarray]] = []
    eval_left = int(args.eval_positions)
    for shard in shard_paths:
        if eval_left <= 0:
            break
        arrs, _meta = load_shard_arrays(shard, lazy=False)
        arrs = _select_recorded_input_history(
            arrs,
            input_history_encoding=model_cfg.input_history_encoding,
            prefer_recorded_lc0_root=bool(args.prefer_recorded_lc0_root),
        )
        arrs = _maybe_synthetic_history(
            arrs,
            synthetic_lc0_root_history=bool(args.synthetic_lc0_root_history),
        )
        arrs = _convert_policy_targets(arrs, policy_encoding=model_cfg.policy_encoding)
        n = min(eval_left, int(np.asarray(arrs["x"]).shape[0]))
        eval_chunks.append(_slice_arrays(arrs, np.arange(n, dtype=np.int64)))
        eval_left -= n
    eval_arrs = _concat(eval_chunks)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    print(
        json.dumps({
            "event": "start",
            "shards": len(shard_paths),
            "candidates": args.candidates,
            "policy_encoding": model_cfg.policy_encoding,
            "input_history_encoding": model_cfg.input_history_encoding,
            "input_pos_encoding": model_cfg.input_pos_encoding,
            "qkv_projection": model_cfg.qkv_projection,
            "input_global_embedding": model_cfg.input_global_embedding,
            "input_global_embedding_channels": model_cfg.input_global_embedding_channels,
            "input_square_embedding": model_cfg.input_square_embedding,
            "use_smolgen": model_cfg.use_smolgen,
            "smolgen_mode": model_cfg.smolgen_mode,
            "smolgen_bias_scale": model_cfg.smolgen_bias_scale,
            "smolgen_bias_norm": model_cfg.smolgen_bias_norm,
            "arc_attention_bias": model_cfg.arc_attention_bias,
            "smolgen_relation_basis": model_cfg.smolgen_relation_basis,
            "smolgen_relation_norm": model_cfg.smolgen_relation_norm,
            "smolgen_relation_coeff_norm": model_cfg.smolgen_relation_coeff_norm,
            "smolgen_relation_scale": model_cfg.smolgen_relation_scale,
            "use_deepnorm": model_cfg.use_deepnorm,
            "synthetic_lc0_root_history": bool(args.synthetic_lc0_root_history),
            "prefer_recorded_lc0_root": bool(args.prefer_recorded_lc0_root),
            "weight_decay_mode": str(args.weight_decay_mode),
            "soda_scope": str(args.soda_scope),
            "soda_start_step": int(args.soda_start_step),
            "aurora_pp_iterations": int(args.aurora_pp_iterations),
            "aurora_pp_beta": float(args.aurora_pp_beta),
            "aurora_polar_steps": int(args.aurora_polar_steps),
            "aurora_polar_method": str(args.aurora_polar_method),
            "aurora_polar_dtype": str(args.aurora_polar_dtype),
            "aurora_polar_safety": float(args.aurora_polar_safety),
            "results": str(results_path),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else "",
        }),
        flush=True,
    )
    with results_path.open("a", encoding="utf-8") as fh:
        for candidate in args.candidates:
            row = _train_candidate(
                candidate=candidate,
                cfg=cfg,
                shard_paths=shard_paths,
                eval_arrs=eval_arrs,
                args=args,
            )
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            print(json.dumps({"event": "candidate_done", **row}), flush=True)


if __name__ == "__main__":
    main()
