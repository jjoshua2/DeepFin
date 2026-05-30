from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from zclip import ZClip

from chess_anti_engine.utils.amp import inference_autocast
from chess_anti_engine.utils.atomic import atomic_write

try:
    from torch.utils.tensorboard import SummaryWriter as _ImportedSummaryWriter

    _SummaryWriter: Any = _ImportedSummaryWriter
except Exception:  # pragma: no cover
    class _FallbackSummaryWriter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:  # skylos: ignore (stub signature parity)
            pass

        def add_scalar(self, *_args: Any, **_kwargs: Any) -> None:  # skylos: ignore (stub signature parity)
            pass

        def close(self) -> None:
            pass

    _SummaryWriter = _FallbackSummaryWriter

from chess_anti_engine.encoding.lc0 import (
    LC0_FULL,
    normalize_lc0_history_encoding,
    uses_lc0_root_history,
    uses_lc0_root_legacy_meta,
)
from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig
from chess_anti_engine.moves import (
    COMPACT_POLICY_SIZE,
    COMPACT_TO_FULL_POLICY,
    FULL_TO_COMPACT_POLICY,
    POLICY_SIZE,
)
from chess_anti_engine.replay.augment import (
    maybe_mirror_batch_arrays,
    maybe_mirror_samples,
)
from chess_anti_engine.replay.buffer import ReplayBuffer, ReplaySample
from chess_anti_engine.replay.dataset import collate, collate_arrays
from chess_anti_engine.replay.shard import INPUT_HISTORY_ENCODING_ARRAY_KEY

from .aurora import AuroraWithAuxAdam
from .compile_probe import CompileProbe, apply_compile
from .cosmos import COSMOS
from .cosmos_fast import COSMOSFast
from .losses import (
    align_policy_target,
    apply_policy_mask_to_logits,
    compute_loss,
    wdl_brier_ece_from_stats,
    wdl_calibration_stats,
)
from .muon import MuonWithAuxAdam
from .soda import SODAWeightDecayWrapper, mark_soda_weight_decay_groups

SummaryWriter = _SummaryWriter  # skylos: ignore (used via runtime fallback)

_LC0_HISTORY_STEPS = 8
_LC0_PIECE_PLANES = 12
_INPUT_HISTORY_SELECTED_KEY = "_input_history_encoding_selected"


@lru_cache(maxsize=16)
def _policy_index_lut(
    lut_name: str,
    *,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    if lut_name == "full_to_compact":
        values = FULL_TO_COMPACT_POLICY
    elif lut_name == "compact_to_full":
        values = COMPACT_TO_FULL_POLICY
    else:
        raise ValueError(f"unknown policy index LUT {lut_name!r}")
    device = torch.device(device_type) if device_index is None else torch.device(device_type, device_index)
    return torch.as_tensor(values, dtype=torch.long, device=device)


def _policy_index_lut_for(target: torch.Tensor, lut_name: str) -> torch.Tensor:
    return _policy_index_lut(
        lut_name,
        device_type=target.device.type,
        device_index=target.device.index,
    )


def _metadata_strings(
    arrs: dict[str, np.ndarray],
    key: str,
    *,
    n: int | None = None,
) -> np.ndarray | None:
    value = arrs.get(key)
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    flat = array.reshape(-1).astype(str)
    if n is not None:
        if flat.size == 1:
            return np.full((n,), str(flat[0]), dtype=object)
        if flat.size != n:
            raise ValueError(f"replay metadata {key} has {flat.size} values for {n} rows")
    return flat.astype(object, copy=False)


def _single_metadata_string(arrs: dict[str, np.ndarray], key: str) -> str | None:
    flat = _metadata_strings(arrs, key)
    if flat is None:
        return None
    first = str(flat[0])
    if any(str(item) != first for item in flat[1:]):
        raise ValueError(f"mixed replay metadata {key}: {sorted({str(item) for item in flat})}")
    return first


def _legacy_x_to_synthetic_lc0_root(x: np.ndarray) -> np.ndarray:
    """Best-effort LC0-root history remap for legacy replay tensors."""
    src = np.asarray(x)
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
    out[:, 104, :, :] = src[:, 97, :, :]
    out[:, 105, :, :] = src[:, 96, :, :]
    out[:, 106, :, :] = src[:, 99, :, :]
    out[:, 107, :, :] = src[:, 98, :, :]
    out[:, 108, :, :] = 0
    out[:, 109, :, :] = src[:, 102, :, :]
    out[:, 110, :, :] = 0
    out[:, 111, :, :] = src[:, 111, :, :]
    return out


def _apply_lc0_root_legacy_meta(root_x: np.ndarray, legacy_x: np.ndarray) -> np.ndarray:
    root = np.array(root_x, copy=True, order="C")
    legacy = np.asarray(legacy_x)
    root[:, 109, :, :] = legacy[:, 102, :, :].astype(root.dtype, copy=False)
    root[:, 110, :, :] = legacy[:, 100, :, :].astype(root.dtype, copy=False)
    return root


def select_input_history_arrays(
    arrs: dict[str, np.ndarray],
    *,
    input_history_encoding: str,
) -> dict[str, np.ndarray]:
    """Select the replay input tensor matching the model's configured history layout."""
    hist_enc = normalize_lc0_history_encoding(input_history_encoding)
    n = int(np.asarray(arrs["x"]).shape[0])
    stored_value = arrs.get(INPUT_HISTORY_ENCODING_ARRAY_KEY)
    if stored_value is not None:
        stored_array = np.asarray(stored_value)
        flat_stored = stored_array.reshape(-1)
        if flat_stored.size == 1 or (
            flat_stored.size == n
            and bool(np.all(flat_stored == flat_stored[0]))
        ):
            text = str(flat_stored[0]).strip()
            scalar_enc = normalize_lc0_history_encoding(text) if text else ""
            if not uses_lc0_root_history(hist_enc):
                if scalar_enc and uses_lc0_root_history(scalar_enc):
                    raise ValueError(
                        "replay x contains rows stored as LC0-root, "
                        f"cannot train {hist_enc!r} input"
                    )
                return arrs
            selected_enc = _single_metadata_string(arrs, _INPUT_HISTORY_SELECTED_KEY)
            if selected_enc is not None:
                if selected_enc != hist_enc:
                    raise ValueError(
                        f"replay input history already selected as {selected_enc!r}, "
                        f"cannot reselect as {hist_enc!r}"
                    )
                return arrs
            if scalar_enc == hist_enc:
                out = dict(arrs)
                out[_INPUT_HISTORY_SELECTED_KEY] = np.asarray(hist_enc)
                return out
            if scalar_enc and uses_lc0_root_history(scalar_enc):
                raise ValueError(f"replay x contains incompatible stored history encodings: {[scalar_enc]}")
    elif not uses_lc0_root_history(hist_enc):
        return arrs

    stored_raw = _metadata_strings(arrs, INPUT_HISTORY_ENCODING_ARRAY_KEY, n=n)
    stored_enc = np.full((n,), "", dtype=object)
    if stored_raw is not None:
        for i, raw in enumerate(stored_raw):
            text = str(raw).strip()
            stored_enc[i] = normalize_lc0_history_encoding(text) if text else ""
    if not uses_lc0_root_history(hist_enc):
        root_rows = np.array(
            [uses_lc0_root_history(str(enc)) if str(enc) else False for enc in stored_enc],
            dtype=bool,
        )
        if bool(np.any(root_rows)):
            raise ValueError(
                f"replay x contains {int(root_rows.sum())} rows stored as LC0-root, "
                f"cannot train {hist_enc!r} input"
            )
        return arrs
    selected_enc = _single_metadata_string(arrs, _INPUT_HISTORY_SELECTED_KEY)
    if selected_enc is not None:
        if selected_enc != hist_enc:
            raise ValueError(
                f"replay input history already selected as {selected_enc!r}, "
                f"cannot reselect as {hist_enc!r}"
            )
        return arrs
    already_root = stored_enc == hist_enc
    mismatched_root = np.array(
        [
            bool(enc) and enc != hist_enc and uses_lc0_root_history(str(enc))
            for enc in stored_enc
        ],
        dtype=bool,
    )
    if bool(np.any(mismatched_root)):
        bad = sorted({str(enc) for enc in stored_enc[mismatched_root]})
        raise ValueError(f"replay x contains incompatible stored history encodings: {bad}")
    if bool(np.all(already_root)):
        out = dict(arrs)
        out[_INPUT_HISTORY_SELECTED_KEY] = np.asarray(hist_enc)
        return out
    out = dict(arrs)
    legacy_x = np.asarray(out["x"])
    convert_rows = ~already_root
    root_x = np.array(legacy_x, copy=True, order="C")
    if bool(np.any(convert_rows)):
        root_x[convert_rows] = _legacy_x_to_synthetic_lc0_root(legacy_x[convert_rows])
    if "x_lc0_root" in out:
        recorded = np.asarray(out["x_lc0_root"])
        has = np.asarray(out.get("has_x_lc0_root", np.ones((legacy_x.shape[0],), dtype=np.uint8))) != 0
        if recorded.shape == legacy_x.shape and has.shape == (legacy_x.shape[0],):
            use_recorded = has & convert_rows
            root_x[use_recorded] = recorded[use_recorded]
    if uses_lc0_root_legacy_meta(hist_enc):
        if bool(np.any(convert_rows)):
            patched = _apply_lc0_root_legacy_meta(root_x[convert_rows], legacy_x[convert_rows])
            root_x[convert_rows] = patched
    out["x"] = root_x
    out[_INPUT_HISTORY_SELECTED_KEY] = np.asarray(hist_enc)
    out[INPUT_HISTORY_ENCODING_ARRAY_KEY] = np.full((n,), hist_enc, dtype=object)
    return out


def select_input_history_samples(
    samples: list[ReplaySample],
    *,
    input_history_encoding: str,
) -> list[ReplaySample]:
    """Select configured input planes for in-memory replay samples."""
    hist_enc = normalize_lc0_history_encoding(input_history_encoding)
    if not samples:
        return samples
    if not uses_lc0_root_history(hist_enc):
        root_samples = [
            sample
            for sample in samples
            if sample.input_history_encoding
            and uses_lc0_root_history(sample.input_history_encoding)
        ]
        if root_samples:
            raise ValueError(
                f"replay samples contain {len(root_samples)} rows stored as LC0-root, "
                f"cannot train {hist_enc!r} input"
            )
        return samples

    legacy_x = np.stack([np.asarray(s.x) for s in samples], axis=0)
    arrs: dict[str, np.ndarray] = {
        "x": legacy_x,
        INPUT_HISTORY_ENCODING_ARRAY_KEY: np.asarray(
            [
                normalize_lc0_history_encoding(sample.input_history_encoding)
                if sample.input_history_encoding
                else ""
                for sample in samples
            ],
            dtype=object,
        ),
    }
    if any(s.x_lc0_root is not None for s in samples):
        recorded = np.zeros_like(legacy_x)
        has = np.zeros((len(samples),), dtype=np.uint8)
        for i, sample in enumerate(samples):
            if sample.x_lc0_root is not None:
                recorded[i] = np.asarray(sample.x_lc0_root)
                has[i] = 1
        arrs["x_lc0_root"] = recorded
        arrs["has_x_lc0_root"] = has

    selected = select_input_history_arrays(arrs, input_history_encoding=hist_enc)["x"]
    return [
        dataclasses.replace(sample, x=selected[i], input_history_encoding=hist_enc)
        for i, sample in enumerate(samples)
    ]


class _ChainedOptimizer(torch.optim.Optimizer):
    """Expose multiple optimizers as one optimizer for trainer scheduling."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        params = [
            param
            for opt in optimizers
            for group in opt.param_groups
            for param in group["params"]
        ]
        self._initializing = True
        super().__init__(params, defaults={})
        self._initializing = False
        self.optimizers = optimizers
        self.param_groups = [
            group
            for opt in optimizers
            for group in opt.param_groups
        ]
        self.state.clear()

    def add_param_group(self, param_group: dict[str, Any]) -> None:  # type: ignore[override]
        if getattr(self, "_initializing", False):
            torch.optim.Optimizer.add_param_group(self, param_group)
            return
        del param_group
        raise NotImplementedError(
            "_ChainedOptimizer cannot route add_param_group() to a child optimizer; "
            "add new parameters to the intended child optimizer directly."
        )

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], float] | None = None) -> float:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        loss = 0.0
        for i, opt in enumerate(self.optimizers):
            loss_i = opt.step(closure if i == 0 else None)
            if loss_i is not None:
                loss = float(loss_i)
        return loss

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        for opt, opt_state in zip(self.optimizers, state_dict["optimizers"], strict=True):
            opt.load_state_dict(opt_state)
        self.param_groups = [
            group
            for opt in self.optimizers
            for group in opt.param_groups
        ]


def _split_decay_groups(
    model: torch.nn.Module,
    *,
    hidden_filter: Callable[[str, torch.nn.Parameter], bool] | None = None,
) -> tuple[list, list, list, list]:
    """Bucket parameters into (hidden_decay, hidden_no_decay, aux_decay, aux_no_decay).

    ``no_decay`` are 1-D tensors and biases (norms, biases — selective weight
    decay convention). ``hidden_filter`` separates a "trunk" subset (hidden_*)
    from the rest (aux_*); without it, hidden_* are empty and all params go
    into aux_*.
    """
    hidden_decay: list = []
    hidden_no_decay: list = []
    aux_decay: list = []
    aux_no_decay: list = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = param.ndim <= 1 or name.endswith(".bias")
        is_hidden = bool(hidden_filter(name, param)) if hidden_filter else False
        if is_hidden and not is_no_decay:
            hidden_decay.append(param)
        elif is_hidden:
            hidden_no_decay.append(param)
        elif is_no_decay:
            aux_no_decay.append(param)
        else:
            aux_decay.append(param)
    return hidden_decay, hidden_no_decay, aux_decay, aux_no_decay


def _extract_named_params(
    groups: tuple[list, ...],
    model: torch.nn.Module,
    predicate: Callable[[str, torch.nn.Parameter], bool],
) -> list:
    """Remove matching parameters from ``groups`` and return them in model order."""
    wanted_ids = {
        id(param)
        for name, param in model.named_parameters()
        if param.requires_grad and predicate(name, param)
    }
    if not wanted_ids:
        return []
    extracted: list = []
    for group in groups:
        kept = []
        for param in group:
            if id(param) in wanted_ids:
                extracted.append(param)
            else:
                kept.append(param)
        group[:] = kept
    return extracted


def _matrix_optimizer_filter(
    scope: str,
    *,
    include_embed_default: bool,
) -> Callable[[str, torch.nn.Parameter], bool]:
    """Return the 2D matrix subset owned by Muon/Aurora-style optimizers."""
    scope = str(scope or "default").lower()

    def _is_block_matrix(name: str, param: torch.nn.Parameter) -> bool:
        return param.ndim >= 2 and name.startswith("blocks.")

    def _is_ffn(name: str) -> bool:
        return ".ffn." in name

    def _is_attn_out(name: str) -> bool:
        return ".out_proj." in name

    def _is_attn_input(name: str) -> bool:
        return any(part in name for part in (".qkv_proj.", ".q_proj.", ".k_proj.", ".v_proj."))

    def _is_attn_v(name: str) -> bool:
        return ".v_proj." in name

    if scope in ("default", "", "legacy"):
        return lambda name, p: p.ndim >= 2 and (
            (include_embed_default and name == "embed.weight") or name.startswith("blocks.")
        )
    if scope in ("blocks", "block_all", "all_blocks", "all_block"):
        return _is_block_matrix
    if scope in ("mlp", "mlp_only", "ffn", "ffn_only"):
        return lambda name, p: _is_block_matrix(name, p) and _is_ffn(name)
    if scope in ("mlp_out", "mlp_o", "mlp_attn_o", "mlp_attention_o"):
        return lambda name, p: _is_block_matrix(name, p) and (_is_ffn(name) or _is_attn_out(name))
    if scope in ("mlp_out_v", "mlp_v_out", "mlp_attn_ov", "mlp_attention_ov"):
        return lambda name, p: _is_block_matrix(name, p) and (
            _is_ffn(name) or _is_attn_out(name) or _is_attn_v(name)
        )
    if scope in ("mlp_attn_all", "mlp_attention_all", "attn_mlp", "all_attention_mlp"):
        return lambda name, p: _is_block_matrix(name, p) and (
            _is_ffn(name) or _is_attn_out(name) or _is_attn_input(name)
        )
    raise ValueError(
        f"Unknown matrix_optimizer_scope {scope!r}. Supported: default, block_all, "
        "mlp_only, mlp_out, mlp_out_v, mlp_attn_all"
    )


@dataclass
class TrainMetrics:
    loss: float
    policy_loss: float
    soft_policy_loss: float
    future_policy_loss: float
    wdl_loss: float
    sf_move_loss: float
    sf_move_acc: float
    sf_eval_loss: float
    categorical_loss: float
    volatility_loss: float
    sf_volatility_loss: float
    moves_left_loss: float
    blended_wdl_loss: float = 0.0
    channel_balance: float = 0.0
    sf_search_agree_frac: float = 0.0
    sf_search_disagree_sf_low_frac: float = 0.0
    sf_search_disagree_sf_high_frac: float = 0.0
    sf_move_acc_top5: float = 0.0
    policy_own_acc_top1: float = 0.0
    policy_own_acc_top5: float = 0.0
    policy_future_acc_top1: float = 0.0
    policy_future_acc_top5: float = 0.0
    train_time_s: float = 0.0
    opt_step_time_s: float = 0.0
    train_steps_done: int = 0
    train_samples_seen: int = 0
    aurora_uw_floor: float = 0.0
    aurora_uw_count: float = 0.0
    aurora_uw_ratio_min: float = 0.0
    aurora_uw_ratio_p10: float = 0.0
    aurora_uw_ratio_median: float = 0.0
    aurora_uw_ratio_p90: float = 0.0
    aurora_uw_scale_max: float = 0.0
    aurora_uw_floored_frac: float = 0.0
    aurora_uw_effective_ratio_min: float = 0.0
    aurora_uw_effective_ratio_median: float = 0.0
  # Per-source loss split (observation-only; only meaningful once shards carry is_selfplay).
    policy_loss_selfplay: float = 0.0
    policy_loss_curriculum: float = 0.0
    wdl_loss_selfplay: float = 0.0
    wdl_loss_curriculum: float = 0.0
    frac_is_selfplay: float = 0.0
    frac_tagged: float = 0.0
  # Per-game-phase loss split (bucketed by moves_left).
    policy_loss_open: float = 0.0
    policy_loss_mid: float = 0.0
    policy_loss_end: float = 0.0
    wdl_loss_open: float = 0.0
    wdl_loss_mid: float = 0.0
    wdl_loss_end: float = 0.0
  # Value-head calibration (populated on holdout eval).
    wdl_brier: float = 0.0
    wdl_ece: float = 0.0


# Map compute_loss dict keys → TrainMetrics field names where they differ.
# Keys not listed pass through unchanged (e.g. the split losses are same-named).
_LOSS_KEY_TO_METRIC_FIELD = {
    "policy_ce": "policy_loss",
    "soft_policy_ce": "soft_policy_loss",
    "future_policy_ce": "future_policy_loss",
    "wdl_ce": "wdl_loss",
    "sf_move_ce": "sf_move_loss",
    "sf_eval_ce": "sf_eval_loss",
    "categorical_ce": "categorical_loss",
    "volatility": "volatility_loss",
    "sf_volatility": "sf_volatility_loss",
    "moves_left": "moves_left_loss",
    "blended_wdl_ce": "blended_wdl_loss",
}
_TRAIN_METRICS_FIELDS = frozenset(f.name for f in dataclasses.fields(TrainMetrics))


def _loss_sums_to_metric_kwargs(sums: dict[str, float], n: float) -> dict[str, float]:
    """Convert accumulated per-batch loss sums into TrainMetrics kwargs.

    Keys that don't map to a TrainMetrics field are dropped silently so
    compute_loss can add experimental scalars before TrainMetrics catches up.
    """
    out: dict[str, float] = {}
    for k, v in sums.items():
        field = _LOSS_KEY_TO_METRIC_FIELD.get(k, k)
        if field in _TRAIN_METRICS_FIELDS:
            out[field] = v / n
    return out


def trainer_kwargs_from_config(config: dict, *, log_dir: Path | None = None) -> dict:
    """Extract Trainer constructor kwargs from a flat config dict.

    Single source of truth for config → Trainer mapping, used by both
    run.py (single mode) and tune/trainable.py.  Callers can override
    individual keys in the returned dict before passing to Trainer().

    Accepts ``grad_clip`` as an alias for ``zclip_max_norm`` (the argparse
    name used in run.py).
    """
    def _f(key: str, default: float, typ: type = float) -> Any:
        return typ(config.get(key, default))

  # Handle grad_clip -> zclip_max_norm alias. None disables the fixed hard cap
  # while leaving adaptive z-score clipping active.
    zclip_max_norm_raw = config.get("zclip_max_norm", config.get("grad_clip", 1.0))
    zclip_max_norm = None if zclip_max_norm_raw is None else float(zclip_max_norm_raw)

  # w_sf_volatility falls back to w_volatility if not explicitly set
    w_volatility = _f("w_volatility", 0.05)
    w_sf_volatility_raw = config.get("w_sf_volatility")
    w_sf_volatility = float(w_sf_volatility_raw) if w_sf_volatility_raw is not None else w_volatility

    kw: dict[str, Any] = dict(
        device=str(config.get("device", "cpu")),
        lr=_f("lr", 3e-4),
        zclip_z_thresh=_f("zclip_z_thresh", 2.5),
        zclip_alpha=_f("zclip_alpha", 0.97),
        zclip_clip_factor=_f("zclip_clip_factor", 1.0),
        zclip_max_norm=zclip_max_norm,
        use_amp=bool(config.get("use_amp", True)),
        feature_dropout_p=_f("feature_dropout_p", 0.3),
        fdp_king_safety=config.get("fdp_king_safety"),
        fdp_pins=config.get("fdp_pins"),
        fdp_pawns=config.get("fdp_pawns"),
        fdp_mobility=config.get("fdp_mobility"),
        fdp_outposts=config.get("fdp_outposts"),
        w_volatility=w_volatility,
        resid_channel_dropout=_f("resid_channel_dropout", 0.0),
        resid_channel_balance_weight=_f("resid_channel_balance_weight", 0.0),
        accum_steps=_f("accum_steps", 1, int),
        warmup_steps=_f("warmup_steps", 1500, int),
        warmup_lr_start=config.get("warmup_lr_start"),
        lr_eta_min=_f("lr_eta_min", 1e-5),
        lr_T0=_f("lr_T0", 5000, int),
        lr_T_mult=_f("lr_T_mult", 2, int),
        use_compile=bool(config.get("use_compile", False)),
        compile_mode=str(config.get("compile_mode", "reduce-overhead")),
        optimizer=str(config.get("optimizer", "nadamw")),
        matrix_optimizer_scope=str(config.get("matrix_optimizer_scope", "default")),
        matrix_lr_multiplier=_f("matrix_lr_multiplier", 20.0),
        matrix_weight_decay=_f("matrix_weight_decay", 1e-4),
        aux_weight_decay=_f("aux_weight_decay", 1e-4),
        global_board_preprocess_lr_multiplier=_f("global_board_preprocess_lr_multiplier", 1.0),
        global_board_preprocess_weight_decay=_f("global_board_preprocess_weight_decay", 0.0),
        global_board_adapter_lr_multiplier=_f("global_board_adapter_lr_multiplier", 1.0),
        global_board_adapter_weight_decay=_f("global_board_adapter_weight_decay", 0.0),
        weight_decay_mode=str(config.get("weight_decay_mode", "weight_decay")),
        soda_scope=str(config.get("soda_scope", "decay")),
        soda_start_step=_f("soda_start_step", 0, int),
        aurora_uw_floor=_f("aurora_uw_floor", 0.0),
        aurora_pp_iterations=_f("aurora_pp_iterations", 2, int),
        aurora_pp_beta=_f("aurora_pp_beta", 0.5),
        aurora_polar_steps=_f("aurora_polar_steps", 12, int),
        aurora_polar_method=str(config.get("aurora_polar_method", "simple")),
        aurora_polar_dtype=str(config.get("aurora_polar_dtype", "auto")),
        aurora_polar_safety=_f("aurora_polar_safety", 1.01),
        cosmos_rank=_f("cosmos_rank", 64, int),
        cosmos_gamma=_f("cosmos_gamma", 0.2),
        swa_start=_f("swa_start", 0, int),
        swa_freq=_f("swa_freq", 50, int),
        w_policy=_f("w_policy", 1.0),
        w_soft=_f("w_soft", 0.5),
        w_future=_f("w_future", 0.15),
        w_wdl=_f("w_wdl", 1.0),
        w_sf_move=_f("w_sf_move", 0.15),
        w_sf_eval=_f("w_sf_eval", 0.15),
        w_categorical=_f("w_categorical", 0.10),
        w_sf_volatility=w_sf_volatility,
        w_moves_left=_f("w_moves_left", 0.02),
        sf_wdl_frac=_f("sf_wdl_frac", 0.0),
        search_wdl_frac=_f("search_wdl_frac", 0.0),
        sf_wdl_conf_power=_f("sf_wdl_conf_power", 0.0),
        sf_wdl_draw_scale=_f("sf_wdl_draw_scale", 1.0),
        sf_wdl_temperature=_f("sf_wdl_temperature", 1.0),
        sf_search_dampen_sf_low=_f("sf_search_dampen_sf_low", 0.0),
        sf_search_dampen_sf_high=_f("sf_search_dampen_sf_high", 0.0),
        use_adjusted_wdl_target=bool(config.get("use_adjusted_wdl_target", False)),
        adjusted_wdl_regret_source=str(config.get("adjusted_wdl_regret_source", "sum")),
        adjusted_wdl_regret_scale=_f("adjusted_wdl_regret_scale", 1.0),
        adjusted_wdl_regret_cap=_f("adjusted_wdl_regret_cap", 0.0),
    )
    if log_dir is not None:
        kw["log_dir"] = log_dir
    return kw


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str,
        lr: float,
        zclip_z_thresh: float = 2.5,
        zclip_alpha: float = 0.97,
        zclip_clip_factor: float = 1.0,
        zclip_max_norm: float | None = 1.0,
        log_dir: Path | None = None,
        use_amp: bool = True,
        feature_dropout_p: float = 0.3,
        fdp_king_safety: float | None = None,
        fdp_pins: float | None = None,
        fdp_pawns: float | None = None,
        fdp_mobility: float | None = None,
        fdp_outposts: float | None = None,
        w_volatility: float = 0.05,
        accum_steps: int = 1,
        warmup_steps: int = 1500,
        warmup_lr_start: float | None = None,
        lr_eta_min: float = 1e-5,
        lr_T0: int = 5000,
        lr_T_mult: int = 2,
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        optimizer: str = "nadamw",
        matrix_optimizer_scope: str = "default",
        matrix_lr_multiplier: float = 20.0,
        matrix_weight_decay: float = 1e-4,
        aux_weight_decay: float = 1e-4,
        global_board_preprocess_lr_multiplier: float = 1.0,
        global_board_preprocess_weight_decay: float = 0.0,
        global_board_adapter_lr_multiplier: float = 1.0,
        global_board_adapter_weight_decay: float = 0.0,
        weight_decay_mode: str = "weight_decay",
        soda_scope: str = "decay",
        soda_start_step: int = 0,
        aurora_uw_floor: float = 0.0,
        aurora_pp_iterations: int = 2,
        aurora_pp_beta: float = 0.5,
        aurora_polar_steps: int = 12,
        aurora_polar_method: str = "simple",
        aurora_polar_dtype: str = "auto",
        aurora_polar_safety: float = 1.01,
        cosmos_rank: int = 64,
        cosmos_gamma: float = 0.2,
        swa_start: int = 0,
        swa_freq: int = 50,
        mirror_prob: float = 0.5,
  # Loss weights (all tunable for Ray Tune ablations)
        w_policy: float = 1.0,
        w_soft: float = 0.5,
        w_future: float = 0.15,
        w_wdl: float = 1.0,
        w_sf_move: float = 0.15,
        w_sf_eval: float = 0.15,
        w_categorical: float = 0.10,
        resid_channel_dropout: float = 0.0,
        resid_channel_balance_weight: float = 0.0,
        w_sf_volatility: float | None = None,
        w_moves_left: float = 0.02,
        sf_wdl_frac: float = 0.0,
        search_wdl_frac: float = 0.0,
        sf_wdl_conf_power: float = 0.0,
        sf_wdl_draw_scale: float = 1.0,
        sf_wdl_temperature: float = 1.0,
        sf_search_dampen_sf_low: float = 0.0,
        sf_search_dampen_sf_high: float = 0.0,
        use_adjusted_wdl_target: bool = False,
        adjusted_wdl_regret_source: str = "sum",
        adjusted_wdl_regret_scale: float = 1.0,
        adjusted_wdl_regret_cap: float = 0.0,
        tb_log_interval: int = 10,
        prefetch_batches: bool = True,
        model_config: ModelConfig | None = None,
    ):
        self.device = device
  # Declared as nn.Module; torch.compile (below) wraps it in a Module
  # subclass at runtime, but its stub types return Callable — cast on
  # assignment there to keep attribute access (.train/.eval/.state_dict)
  # type-checked here.
        self.model: torch.nn.Module = model.to(device)
  # Optional — when provided, `save()` and the SWA export embed it
  # into the checkpoint so standalone loaders (UCI engine) don't need
  # a sibling params.json. Kept optional for backward compatibility
  # with direct Trainer() construction in tests.
        self._model_config = model_config
        self._input_history_encoding = normalize_lc0_history_encoding(
            model_config.input_history_encoding if model_config is not None else None
        )

        optimizer = str(optimizer).lower()
        matrix_optimizer_scope = str(matrix_optimizer_scope).lower()
        weight_decay_mode = str(weight_decay_mode).lower()
        if weight_decay_mode not in ("weight_decay", "soda"):
            raise ValueError(
                f"Unknown weight_decay_mode {weight_decay_mode!r}. "
                "Supported: weight_decay, soda"
            )
        soda_scope = str(soda_scope or "decay").lower()
        soda_scope_aliases = {
            "decay": "decay",
            "weight_decay": "decay",
            "nonzero_decay": "decay",
            "hidden_matrix_only": "hidden_matrix_only",
        }
        if soda_scope not in soda_scope_aliases:
            raise ValueError(
                f"Unknown soda_scope {soda_scope!r}. "
                "Supported: decay, hidden_matrix_only "
                "(aliases: weight_decay, nonzero_decay)"
            )
        soda_scope = soda_scope_aliases[soda_scope]
        soda_start_step = max(0, int(soda_start_step))
        use_soda_weight_decay = False

        def _mark_soda(param_groups: list[dict], *, hidden_group_indices: tuple[int, ...] = (0,)) -> bool:
            if weight_decay_mode != "soda":
                return False
            if soda_scope == "decay":
                return mark_soda_weight_decay_groups(param_groups, start_step=soda_start_step)
            for group in param_groups:
                group["weight_decay"] = 0.0
            marked = False
            for idx in hidden_group_indices:
                if 0 <= idx < len(param_groups):
                    marked = (
                        mark_soda_weight_decay_groups(
                            [param_groups[idx]],
                            start_step=soda_start_step,
                            force=True,
                        )
                        or marked
                    )
            return marked

        if optimizer in ("muon", "aurora"):
            if optimizer == "muon":
                hidden_filter = _matrix_optimizer_filter(matrix_optimizer_scope, include_embed_default=True)
            else:
                hidden_filter = _matrix_optimizer_filter(matrix_optimizer_scope, include_embed_default=False)
            hd, hnd, ad, and_ = _split_decay_groups(
                self.model,
                hidden_filter=hidden_filter,
            )
            global_board_preprocess_params = _extract_named_params(
                (hd, hnd, ad, and_),
                self.model,
                lambda name, _param: name.startswith("global_board_preprocess."),
            )
            global_board_adapter_params = _extract_named_params(
                (hd, hnd, ad, and_),
                self.model,
                lambda name, _param: name.startswith("global_board_adapter."),
            )
  # Muon/Aurora trunk gets a larger LR than the AdamW fallback for heads/norms.
  # Keep one Tune-search LR and derive trunk LR so search stays simple.
            matrix_lr = float(lr) * float(matrix_lr_multiplier)
            matrix_wd = float(matrix_weight_decay)
            aux_wd = float(aux_weight_decay)
            branch_groups = []
            if global_board_preprocess_params:
                branch_groups.append(
                    {
                        "params": global_board_preprocess_params,
                        "weight_decay": float(global_board_preprocess_weight_decay),
                        "use_muon": False,
                        "use_aurora": False,
                        "lr": float(lr) * float(global_board_preprocess_lr_multiplier),
                    }
                )
            if global_board_adapter_params:
                branch_groups.append(
                    {
                        "params": global_board_adapter_params,
                        "weight_decay": float(global_board_adapter_weight_decay),
                        "use_muon": False,
                        "use_aurora": False,
                        "lr": float(lr) * float(global_board_adapter_lr_multiplier),
                    }
                )
            if optimizer == "muon":
                param_groups = [
                    {"params": hd, "weight_decay": matrix_wd, "use_muon": True, "lr": matrix_lr},
                    {"params": hnd, "weight_decay": 0.0, "use_muon": True, "lr": matrix_lr},
                    {"params": ad, "weight_decay": aux_wd, "use_muon": False, "lr": float(lr)},
                    {"params": and_, "weight_decay": 0.0, "use_muon": False, "lr": float(lr)},
                    *branch_groups,
                ]
                use_soda_weight_decay = _mark_soda(param_groups)
                self.opt = MuonWithAuxAdam(param_groups)
            else:
                param_groups = [
                    {
                        "params": hd,
                        "weight_decay": matrix_wd,
                        "use_aurora": True,
                        "lr": matrix_lr,
                        "aurora_uw_floor": float(aurora_uw_floor),
                    },
                    {"params": hnd, "weight_decay": 0.0, "use_aurora": False, "lr": float(lr)},
                    {"params": ad, "weight_decay": aux_wd, "use_aurora": False, "lr": float(lr)},
                    {"params": and_, "weight_decay": 0.0, "use_aurora": False, "lr": float(lr)},
                    *branch_groups,
                ]
                use_soda_weight_decay = _mark_soda(param_groups)
                self.opt = AuroraWithAuxAdam(
                    param_groups,
                    aurora_pp_iterations=int(aurora_pp_iterations),
                    aurora_pp_beta=float(aurora_pp_beta),
                    aurora_polar_steps=int(aurora_polar_steps),
                    aurora_polar_method=str(aurora_polar_method),
                    aurora_polar_dtype=str(aurora_polar_dtype),
                    aurora_polar_safety=float(aurora_polar_safety),
                )
        elif optimizer == "cosmos_fast":
            hd, hnd, ad, and_ = _split_decay_groups(
                self.model,
                hidden_filter=_matrix_optimizer_filter(matrix_optimizer_scope, include_embed_default=False),
            )
            matrix_wd = float(matrix_weight_decay)
            aux_wd = float(aux_weight_decay)
            param_groups = [
                {"params": hd, "weight_decay": matrix_wd, "use_cosmos_fast": True},
                {"params": hnd, "weight_decay": 0.0, "use_cosmos_fast": True},
                {"params": ad, "weight_decay": aux_wd, "use_cosmos_fast": False},
                {"params": and_, "weight_decay": 0.0, "use_cosmos_fast": False},
            ]
            use_soda_weight_decay = _mark_soda(param_groups)
            self.opt = COSMOSFast(
                param_groups,
                lr=lr,
                rank=int(cosmos_rank),
                gamma=float(cosmos_gamma),
            )
        elif optimizer == "cosmos" and matrix_optimizer_scope not in ("default", "", "legacy"):
            hd, hnd, ad, and_ = _split_decay_groups(
                self.model,
                hidden_filter=_matrix_optimizer_filter(matrix_optimizer_scope, include_embed_default=False),
            )
            matrix_wd = float(matrix_weight_decay)
            aux_wd = float(aux_weight_decay)
            param_groups = [
                {"params": hd, "weight_decay": matrix_wd, "use_cosmos": True},
                {"params": hnd, "weight_decay": 0.0, "use_cosmos": False},
                {"params": ad, "weight_decay": aux_wd, "use_cosmos": False},
                {"params": and_, "weight_decay": 0.0, "use_cosmos": False},
            ]
            use_soda_weight_decay = _mark_soda(param_groups)
            self.opt = COSMOS(
                param_groups,
                lr=lr,
                rank=int(cosmos_rank),
                gamma=float(cosmos_gamma),
            )
        else:
  # Selective weight decay: apply only to non-bias, non-LayerNorm parameters.
            if weight_decay_mode == "soda" and soda_scope == "hidden_matrix_only":
                hd, hnd, ad, and_ = _split_decay_groups(
                    self.model,
                    hidden_filter=_matrix_optimizer_filter(
                        matrix_optimizer_scope,
                        include_embed_default=False,
                    ),
                )
                param_groups = [
                    {"params": hd, "weight_decay": 0.0},
                    {"params": hnd, "weight_decay": 0.0},
                    {"params": ad, "weight_decay": 0.0},
                    {"params": and_, "weight_decay": 0.0},
                ]
                use_soda_weight_decay = _mark_soda(param_groups)
            else:
                _, _, decay_params, no_decay_params = _split_decay_groups(self.model)
                param_groups = [
                    {"params": decay_params, "weight_decay": 1e-4},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ]
                use_soda_weight_decay = _mark_soda(param_groups)

        if optimizer == "nadamw":
  # NAdam with decoupled weight decay (spec: β1=0.9, β2=0.98, ε=1e-7).
  # PyTorch NAdam supports decoupled_weight_decay since 2.x; param-group
  # weight_decay values are applied per-group.
            self.opt = torch.optim.NAdam(
                param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-7,
                decoupled_weight_decay=True,
            )
        elif optimizer == "adamw":
            self.opt = torch.optim.AdamW(param_groups, lr=lr)
        elif optimizer == "muon":
            pass
        elif optimizer == "aurora":
            pass
        elif optimizer == "cosmos":
            if matrix_optimizer_scope in ("default", "", "legacy"):
                self.opt = COSMOS(
                    param_groups,
                    lr=lr,
                    weight_decay=0.0 if weight_decay_mode == "soda" else 1e-4,
                )
        elif optimizer == "cosmos_fast":
            pass
        elif optimizer == "soap":
  # SOAP: Shampoo-like second-order optimizer. Prefer a local
  # `soap.py`; otherwise fall back to pytorch-optimizer's SOAP.
            try:
                from soap import SOAP  # type: ignore[import] # optional local module
            except ImportError as exc:
                try:
                    from pytorch_optimizer import (
                        SOAP,  # type: ignore[import] # optional dep
                    )
                except ImportError:
                    raise ImportError(
                        "SOAP optimizer requires either a local `soap.py` module "
                        "or the `pytorch-optimizer` package. "
                        "Install with: pip install pytorch-optimizer"
                    ) from exc
            if matrix_optimizer_scope in ("default", "", "legacy"):
                try:
                    self.opt = SOAP(param_groups, lr=lr)
                except TypeError:
                    self.opt = SOAP(self.model.parameters(), lr=lr)
            else:
                hd, hnd, ad, and_ = _split_decay_groups(
                    self.model,
                    hidden_filter=_matrix_optimizer_filter(matrix_optimizer_scope, include_embed_default=False),
                )
                matrix_wd = float(matrix_weight_decay)
                aux_wd = float(aux_weight_decay)
                soap_groups = [{"params": hd, "weight_decay": matrix_wd}]
                adam_groups = [
                    {"params": hnd, "weight_decay": 0.0, "lr": float(lr)},
                    {"params": ad, "weight_decay": aux_wd, "lr": float(lr)},
                    {"params": and_, "weight_decay": 0.0, "lr": float(lr)},
                ]
                if weight_decay_mode == "soda" and soda_scope == "hidden_matrix_only":
                    for group in adam_groups:
                        group["weight_decay"] = 0.0
                    use_soda_weight_decay = mark_soda_weight_decay_groups(
                        soap_groups,
                        start_step=soda_start_step,
                        force=True,
                    )
                else:
                    soap_soda = _mark_soda(soap_groups)
                    adam_soda = _mark_soda(adam_groups, hidden_group_indices=())
                    use_soda_weight_decay = soap_soda or adam_soda
                self.opt = _ChainedOptimizer(
                    [
                        SOAP(soap_groups, lr=lr, weight_decay=0.0),
                        torch.optim.AdamW(adam_groups, lr=lr),
                    ]
                )
        else:
            raise ValueError(
                f"Unknown optimizer {optimizer!r}. Supported: nadamw, adamw, muon, aurora, cosmos, cosmos_fast, soap"
            )

        if use_soda_weight_decay:
            self.opt = SODAWeightDecayWrapper(self.opt)
        max_grad_norm = None if zclip_max_norm is None else float(zclip_max_norm)
        self.zclip = ZClip(
            mode="zscore",
            alpha=float(zclip_alpha),
            z_thresh=float(zclip_z_thresh),
            max_grad_norm=max_grad_norm,  # pyright: ignore[reportArgumentType] # zclip accepts None to disable the hard cap.
            clip_factor=float(zclip_clip_factor),
            warmup_steps=25,
        )
        self.writer = SummaryWriter(log_dir=str(log_dir or "tb"))
        self.step = 0
        self._tb_log_interval = max(1, int(tb_log_interval))
        self._prefetch_batches = bool(prefetch_batches)

        self.use_amp = bool(use_amp)
        self._amp_dtype = torch.bfloat16 if device.startswith("cuda") else None

        self.feature_dropout_p = float(feature_dropout_p)
        self._base_input_planes = int(LC0_FULL.num_planes)
  # Per-group dropout: (start_offset_from_base, num_planes, dropout_prob)
  # Groups: king_safety(10), pins(6), pawns(8), mobility(6), outposts(4).
  # Per-group overrides default to a -1 sentinel in TrialConfig (not None),
  # so treat any negative value as "fall back to global feature_dropout_p".
        _fdp = float(feature_dropout_p)

        def _resolve_fdp(v: float | None) -> float:
            return _fdp if v is None or v < 0 else float(v)

        self._feature_group_dropout = [
            (0, 10, _resolve_fdp(fdp_king_safety)),
            (10, 6, _resolve_fdp(fdp_pins)),
            (16, 8, _resolve_fdp(fdp_pawns)),
            (24, 6, _resolve_fdp(fdp_mobility)),
            (30, 4, _resolve_fdp(fdp_outposts)),
        ]
        self.w_policy = float(w_policy)
        self.w_soft = float(w_soft)
        self.w_future = float(w_future)
        self.w_wdl = float(w_wdl)
        self.w_sf_move = float(w_sf_move)
        self.w_sf_eval = float(w_sf_eval)
        self.w_categorical = float(w_categorical)
        self.w_volatility = float(w_volatility)
        self.resid_channel_dropout = max(0.0, min(0.95, float(resid_channel_dropout)))
        if hasattr(self.model, "resid_channel_dropout"):
            setattr(self.model, "resid_channel_dropout", self.resid_channel_dropout)
        self.resid_channel_balance_weight = max(0.0, float(resid_channel_balance_weight))
        if hasattr(self.model, "resid_channel_balance_weight"):
            setattr(self.model, "resid_channel_balance_weight", self.resid_channel_balance_weight)
        self.w_sf_volatility = float(w_sf_volatility) if w_sf_volatility is not None else float(w_volatility)
        self.w_moves_left = float(w_moves_left)
        self.sf_wdl_frac = float(sf_wdl_frac)
        self.search_wdl_frac = float(search_wdl_frac)
        self.sf_wdl_conf_power = float(sf_wdl_conf_power)
        self.sf_wdl_draw_scale = float(sf_wdl_draw_scale)
        self.sf_wdl_temperature = float(sf_wdl_temperature)
        self.sf_search_dampen_sf_low = float(sf_search_dampen_sf_low)
        self.sf_search_dampen_sf_high = float(sf_search_dampen_sf_high)
        self.use_adjusted_wdl_target = bool(use_adjusted_wdl_target)
        self.adjusted_wdl_regret_source = str(adjusted_wdl_regret_source)
        self.adjusted_wdl_regret_scale = float(adjusted_wdl_regret_scale)
        self.adjusted_wdl_regret_cap = float(adjusted_wdl_regret_cap)

  # Data augmentation: mirror positions left-right (files) with given probability.
        self.mirror_prob = float(mirror_prob)

  # Optional torch.compile for training throughput. Failures and recompile
  # thrash are surfaced via apply_compile + CompileProbe instead of being
  # swallowed silently.
        self.model = apply_compile(
            self.model,
            mode=(compile_mode if use_compile else "off"),
            device=device,
        )
        self._compile_probe = CompileProbe()
        self._compile_probe.snapshot_baseline()
        self._compile_probe_steps_remaining = 10  # report after first 10 steps

  # Gradient accumulation
        self.accum_steps = max(1, int(accum_steps))

  # LR schedule: linear warmup then cosine annealing with warm restarts
        self._peak_lr = float(lr)
        self._warmup_steps = int(warmup_steps)
        if warmup_lr_start is None:
            self._warmup_lr_start = max(0.0, float(lr_eta_min))
        else:
            self._warmup_lr_start = max(0.0, float(warmup_lr_start))
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=int(lr_T0), T_mult=int(lr_T_mult), eta_min=float(lr_eta_min),
        )
        self._set_initial_lrs()

  # Stochastic Weight Averaging (SWA): maintain a running average of model
  # weights for smoother, more generalizable exported networks.
        self._swa_start = int(swa_start)
        self._swa_freq = max(1, int(swa_freq))
        self._swa_model: torch.optim.swa_utils.AveragedModel | None = None
        self._init_swa()

    def _init_swa(self) -> None:
        """(Re)initialize SWA from current model weights.

        Must be called after any external weight load (bootstrap, salvage)
        because AveragedModel deep-copies at creation time.
        """
        if self._swa_start >= 0:  # 0 = start immediately, <0 = disabled
            self._swa_model = torch.optim.swa_utils.AveragedModel(self.model)
        else:
            self._swa_model = None

    def _should_log_step_scalars(self) -> bool:
        return (self.step % self._tb_log_interval) == 0

    def _zclip_step(self, *, collect_stats: bool) -> tuple[float, dict[str, float] | None]:
        if not collect_stats:
            return float(self.zclip.step(self.model)), None

        was_initialized = bool(getattr(self.zclip, "initialized", False))
        mean_raw = getattr(self.zclip, "mean", 0.0)
        var_raw = getattr(self.zclip, "var", 0.0)
        mean = float(mean_raw if mean_raw is not None else 0.0)
        var = float(var_raw if var_raw is not None else 0.0)
        total_norm = float(self.zclip.step(self.model))
        clip_val = None
        if was_initialized:
            std = var ** 0.5
            mode = str(getattr(self.zclip, "mode", "zscore"))
            z_thresh = float(getattr(self.zclip, "z_thresh", 0.0))
            if mode == "percentile":
                threshold = mean + z_thresh * std
                if total_norm > threshold:
                    clip_val = threshold
            elif mode == "zscore":
                z_score = (total_norm - mean) / (std + float(getattr(self.zclip, "eps", 1e-8)))
                if z_score > z_thresh:
                    clip_option = str(getattr(self.zclip, "clip_option", "adaptive_scaling"))
                    if clip_option == "adaptive_scaling":
                        eta = z_score / z_thresh
                        clip_val = (mean + (z_thresh * std) / eta) * float(getattr(self.zclip, "clip_factor", 1.0))
                    elif clip_option == "mean":
                        clip_val = mean
        adaptive_clip = float(clip_val) if clip_val is not None else total_norm
        max_grad_norm = getattr(self.zclip, "max_grad_norm", None)
        effective_clip = adaptive_clip
        if max_grad_norm is not None:
            effective_clip = min(effective_clip, float(max_grad_norm))

        clipped = effective_clip < total_norm
        stats = {
            "total_norm": total_norm,
            "effective_clip": float(effective_clip),
            "adaptive_clip": 1.0 if clip_val is not None and adaptive_clip < total_norm else 0.0,
            "hard_clip": 1.0 if max_grad_norm is not None and clipped and effective_clip == float(max_grad_norm) else 0.0,
            "clipped": 1.0 if clipped else 0.0,
        }
        return total_norm, stats

    @property
    def _loss_kwargs(self) -> dict[str, Any]:
        return dict(
            w_policy=self.w_policy, w_soft=self.w_soft, w_future=self.w_future,
            w_wdl=self.w_wdl, w_sf_move=self.w_sf_move, w_sf_eval=self.w_sf_eval,
            w_categorical=self.w_categorical, w_volatility=self.w_volatility,
            w_sf_volatility=self.w_sf_volatility, w_moves_left=self.w_moves_left,
            sf_wdl_frac=self.sf_wdl_frac, search_wdl_frac=self.search_wdl_frac,
            sf_wdl_conf_power=self.sf_wdl_conf_power,
            sf_wdl_draw_scale=self.sf_wdl_draw_scale,
            sf_wdl_temperature=self.sf_wdl_temperature,
            sf_search_dampen_sf_low=self.sf_search_dampen_sf_low,
            sf_search_dampen_sf_high=self.sf_search_dampen_sf_high,
            use_adjusted_wdl_target=self.use_adjusted_wdl_target,
            adjusted_wdl_regret_source=self.adjusted_wdl_regret_source,
            adjusted_wdl_regret_scale=self.adjusted_wdl_regret_scale,
            adjusted_wdl_regret_cap=self.adjusted_wdl_regret_cap,
        )

    def _amp_context(self):
        # Pinned to bf16: training has no GradScaler, so an FP16 fallback
        # would silently underflow gradients on non-BF16 CUDA cards. The
        # ``inference_autocast`` helper would auto-fallback to FP16 there.
        return inference_autocast(device=self.device, enabled=self.use_amp, dtype="bf16")

    @staticmethod
    def _extract_loss_scalars(losses: dict[str, torch.Tensor]) -> dict[str, float]:
        """Extract all non-total loss component scalars from compute_loss output.

        Single GPU sync via stack-then-tolist instead of one ``.item()`` per
        component (~13 syncs per microbatch otherwise — meaningful at high
        accum_steps).
        """
        keys = [k for k in losses if k != "total"]
        if not keys:
            return {}
        stacked = torch.stack([losses[k].detach() for k in keys])
        values = stacked.tolist()
        return dict(zip(keys, values, strict=True))

    def _log_metrics(self, metrics: TrainMetrics, tag: str) -> None:
        """Log all TrainMetrics fields to TensorBoard under the given tag."""
        for field_name, value in dataclasses.asdict(metrics).items():
            self.writer.add_scalar(f"{tag}/{field_name}", float(value), self.step)

    @staticmethod
    def _build_metrics(
        sums: dict[str, float],
        acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        n: float,
        **extras: Any,
    ) -> TrainMetrics:
        """Common tail of ``_compute_metrics`` and ``train_steps`` — averages
        loss sums by ``n`` and computes per-head accuracy ratios from
        (numerator, denominator) GPU-tensor pairs in ``acc_sums``. The
        tensors get materialized to CPU floats here in a single sync per
        head rather than per-microbatch.
        """
        def _acc(name: str) -> float:
            val = acc_sums.get(name)
            if val is None:
                return 0.0
            num, den = val
            den_f = float(den.item())
            if den_f <= 0:
                return 0.0
            return float(num.item()) / den_f

        return TrainMetrics(
            **_loss_sums_to_metric_kwargs(sums, n),  # dict[str,float] splat covers int fields (step counters) at runtime
            sf_move_acc=_acc("sf_move_acc"),
            sf_move_acc_top5=_acc("sf_move_acc_top5"),
            policy_own_acc_top1=_acc("policy_own_acc_top1"),
            policy_own_acc_top5=_acc("policy_own_acc_top5"),
            policy_future_acc_top1=_acc("policy_future_acc_top1"),
            policy_future_acc_top5=_acc("policy_future_acc_top5"),
            **extras,
        )

    def _sample_batch_host(
        self,
        buf: ReplayBuffer,
        *,
        batch_size: int,
        mirror_prob: float,
    ) -> dict[str, np.ndarray] | list:
        if hasattr(buf, "sample_batch_arrays"):
            arrs = buf.sample_batch_arrays(batch_size)
            arrs = select_input_history_arrays(
                arrs,
                input_history_encoding=self._input_history_encoding,
            )
            return maybe_mirror_batch_arrays(
                arrs,
                rng=buf.rng,
                prob=mirror_prob,
                input_history_encoding=self._input_history_encoding,
            )

        samples = buf.sample_batch(batch_size)
        samples = select_input_history_samples(
            samples,
            input_history_encoding=self._input_history_encoding,
        )
        return maybe_mirror_samples(
            samples,
            rng=buf.rng,
            prob=mirror_prob,
            input_history_encoding=self._input_history_encoding,
        )

    def _host_batch_to_tensors(self, batch: dict[str, np.ndarray] | list) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            return collate_arrays(batch, device=self.device)
        return collate(batch, device=self.device)

    def _iter_prefetched_batches(
        self,
        buf: ReplayBuffer,
        *,
        batch_size: int,
        mirror_prob: float,
        count: int,
    ):
        n = int(count)
        if n <= 0:
            return
        if not self._prefetch_batches or n == 1:
            for _ in range(n):
                host_batch = self._sample_batch_host(buf, batch_size=batch_size, mirror_prob=mirror_prob)
                yield self._host_batch_to_tensors(host_batch)
            return

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self._sample_batch_host,
                buf,
                batch_size=batch_size,
                mirror_prob=mirror_prob,
            )
            for idx in range(n):
                host_batch = future.result()
                if idx + 1 < n:
                    future = pool.submit(
                        self._sample_batch_host,
                        buf,
                        batch_size=batch_size,
                        mirror_prob=mirror_prob,
                    )
                yield self._host_batch_to_tensors(host_batch)

    def reset_optimizer_reference_weights(self) -> None:
        """Refresh optimizer reference weights after model-only loads."""
        reset_anchors = getattr(self.opt, "reset_anchors", None)
        if callable(reset_anchors):
            reset_anchors()

    def _base_lrs(self) -> list[float]:
        base_lrs = list(getattr(self._scheduler, "base_lrs", []))
        if base_lrs:
            return [float(v) for v in base_lrs]
        return [float(pg.get("lr", self._peak_lr)) for pg in self.opt.param_groups]

    def _reference_lr_from_bases(self, base_lrs: list[float] | None = None) -> float:
        vals = [float(v) for v in (base_lrs if base_lrs is not None else self._base_lrs()) if float(v) > 0.0]
        if vals:
            return min(vals)
        return max(float(self._peak_lr), 1e-12)

    def _warmup_start_lr_for(self, base_lr: float) -> float:
        peak = max(float(self._peak_lr), 1e-12)
        return float(self._warmup_lr_start) * float(base_lr) / peak

    def _set_initial_lrs(self) -> None:
        if self._warmup_steps <= 0:
            return
        for pg, base_lr in zip(self.opt.param_groups, self._base_lrs(), strict=True):
            pg["lr"] = self._warmup_start_lr_for(float(base_lr))

    def _update_lr(self) -> None:
        """Apply linear warmup, then hand off to cosine schedule."""
        if self.step < self._warmup_steps:
  # Called after optimizer.step(); set the LR for the *next* training step.
            next_frac = min(1.0, float(self.step + 1) / max(1, self._warmup_steps))
            for pg, base_lr in zip(self.opt.param_groups, self._base_lrs(), strict=True):
                start_lr = self._warmup_start_lr_for(float(base_lr))
                pg["lr"] = start_lr + (float(base_lr) - start_lr) * next_frac
        else:
            self._scheduler.step(self.step - self._warmup_steps)

    def _resolve_old_bases(self, n_groups: int) -> list[float]:
        """Per-group base LRs to scale from. Falls back to ``self._peak_lr`` if
        the scheduler hasn't recorded any, and pads/truncates to ``n_groups``."""
        old = (
            [float(v) for v in self._scheduler.base_lrs]
            if hasattr(self._scheduler, "base_lrs") else []
        )
        if not old:
            old = [float(self._peak_lr)] * n_groups
        if len(old) < n_groups:
            old.extend([old[-1]] * (n_groups - len(old)))
        return old[:n_groups]

    def _rescale_active_lr_for_group(
        self,
        pg: dict,
        *,
        old_base: float,
        new_base: float,
        scheduler_last_lr_for_group: float | None,
    ) -> None:
        """Rebase one optimizer group's active ``lr``. Cold-start cases (old_base==0)
        recreate the warmup-phase LR from ``self.step`` and ``self._warmup_steps``."""
        if old_base > 0.0:
            pg["lr"] = float(pg.get("lr", 0.0)) * (new_base / old_base)
            return
        if self.step < self._warmup_steps:
            frac = self.step / max(1, self._warmup_steps)
            warm_start = self._warmup_start_lr_for(new_base)
            pg["lr"] = warm_start + (new_base - warm_start) * frac
        else:
            pg["lr"] = float(scheduler_last_lr_for_group) if scheduler_last_lr_for_group is not None else new_base

    def set_peak_lr(self, lr: float, *, rescale_current: bool = True) -> None:
        """Rebase LR schedule to a new peak while preserving schedule phase.

        PB2 mutates `lr` in the trial config. When a trial restores from another
        trial's checkpoint, optimizer/scheduler state is cloned too, so we need
        to explicitly rebind the schedule to the mutated peak LR.
        """
        new_peak = float(lr)
        if new_peak <= 0.0:
            return

        n_groups = len(self.opt.param_groups)
        old_bases = self._resolve_old_bases(n_groups)
        ref_old = max(float(self._peak_lr), self._reference_lr_from_bases(old_bases))
        scale = new_peak / ref_old

        new_bases = [(ob * scale if ob > 0.0 else new_peak) for ob in old_bases]
        self._peak_lr = new_peak

  # Keep scheduler phase but rebase amplitude.
        if hasattr(self._scheduler, "base_lrs"):
            self._scheduler.base_lrs = list(new_bases)
        if hasattr(self._scheduler, "_last_lr"):
            last_lrs = self._scheduler._last_lr
            if last_lrs:
                self._scheduler._last_lr = [float(v) * scale for v in last_lrs]

  # Keep optimizer param-group metadata aligned.
        for pg, ob, nb in zip(self.opt.param_groups, old_bases, new_bases, strict=True):
            if "initial_lr" in pg:
                pg["initial_lr"] = float(pg["initial_lr"]) * (nb / ob) if ob > 0.0 else nb

        if not rescale_current:
            return

  # Rebase currently active optimizer LR so training continues at same phase.
        last_lrs = getattr(self._scheduler, "_last_lr", None)
        for i, (pg, ob, nb) in enumerate(zip(self.opt.param_groups, old_bases, new_bases, strict=True)):
            sched_lr = (
                float(last_lrs[i]) if isinstance(last_lrs, list) and i < len(last_lrs) else None
            )
            self._rescale_active_lr_for_group(
                pg, old_base=ob, new_base=nb, scheduler_last_lr_for_group=sched_lr,
            )

    @staticmethod
    def _policy_accuracy_stats(
        out: dict, batch: dict,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Top-1/top-5 accuracy for policy_own, policy_sf, policy_future.

        Returns (numerator, denominator) GPU 0-d tensors per head — accumulated
        on-device by callers so per-microbatch ``.item()`` syncs stay out of
        the inner loop. Per-head legal masks (``legal_mask`` at t,
        ``sf_legal_mask`` at t+1 opp-POV, ``future_legal_mask`` at t+2
        net-POV) are gated on ``has_*`` flags so old shards without them
        fall through unmasked.
        """
        stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        def _topk(
            logits: torch.Tensor, target: torch.Tensor, mask_f: torch.Tensor,
            k_values: tuple[int, ...],
        ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
            total = mask_f.sum()
            max_k = max(k_values)
            _, top_idx = torch.topk(logits, k=max_k, dim=-1)
            match = (top_idx == target.unsqueeze(-1))
            return {
                k: ((match[:, :k].any(dim=-1).to(torch.float32) * mask_f).sum(), total)
                for k in k_values
            }

        def _align_index(
            target: torch.Tensor, *, source_width: int, dst_width: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if source_width == dst_width:
                return target, torch.ones_like(target, dtype=torch.bool)
            if source_width == POLICY_SIZE and dst_width == COMPACT_POLICY_SIZE:
                lut = _policy_index_lut_for(target, "full_to_compact")
                valid = (target >= 0) & (target < POLICY_SIZE)
                safe_target = target.clamp(0, POLICY_SIZE - 1).to(torch.long)
                mapped = lut.index_select(0, safe_target)
                return mapped, valid & (mapped >= 0)
            if source_width == COMPACT_POLICY_SIZE and dst_width == POLICY_SIZE:
                lut = _policy_index_lut_for(target, "compact_to_full")
                valid = (target >= 0) & (target < COMPACT_POLICY_SIZE)
                safe_target = target.clamp(0, COMPACT_POLICY_SIZE - 1).to(torch.long)
                mapped = lut.index_select(0, safe_target)
                return mapped, valid
            raise ValueError(f"policy index width {source_width} is incompatible with logits width {dst_width}")

        def _policy_width_from_batch(*keys: str) -> int:
            for key in keys:
                value = batch.get(key)
                if value is not None and value.ndim >= 2:
                    return int(value.shape[-1])
            return POLICY_SIZE

        pol_logits = out.get("policy") if "policy" in out else out.get("policy_own")
        pol_target = batch.get("policy_t")
        has_policy = batch.get("has_policy")
        if pol_logits is not None and pol_target is not None and has_policy is not None:
            logits = apply_policy_mask_to_logits(pol_logits.detach(), batch, "legal_mask", "has_legal_mask")
            tgt = torch.argmax(align_policy_target(pol_target, int(pol_logits.shape[-1])), dim=-1)
            tk = _topk(logits, tgt, has_policy.to(torch.float32), (1, 5))
            stats["policy_own_acc_top1"] = tk[1]
            stats["policy_own_acc_top5"] = tk[5]

        sf_logits = out.get("policy_sf")
        has_sf_move = batch.get("has_sf_move")
        if sf_logits is not None and has_sf_move is not None and "sf_move_index" in batch:
            dst_width = int(sf_logits.shape[-1])
            logits = apply_policy_mask_to_logits(sf_logits.detach(), batch, "sf_legal_mask", "has_sf_legal_mask")
            source_width = _policy_width_from_batch("sf_legal_mask", "sf_policy_t", "policy_t")
            tgt, valid_idx = _align_index(batch["sf_move_index"], source_width=source_width, dst_width=dst_width)
            tk = _topk(logits, tgt, has_sf_move.to(torch.float32) * valid_idx.to(torch.float32), (1, 5))
            stats["sf_move_acc"] = tk[1]
            stats["sf_move_acc_top5"] = tk[5]

        fut_logits = out.get("policy_future")
        fut_target = batch.get("future_policy_t")
        has_future = batch.get("has_future")
        if fut_logits is not None and fut_target is not None and has_future is not None:
            logits = apply_policy_mask_to_logits(fut_logits.detach(), batch, "future_legal_mask", "has_future_legal_mask")
            tgt = torch.argmax(align_policy_target(fut_target, int(fut_logits.shape[-1])), dim=-1)
            tk = _topk(logits, tgt, has_future.to(torch.float32), (1, 5))
            stats["policy_future_acc_top1"] = tk[1]
            stats["policy_future_acc_top5"] = tk[5]

        return stats

    @torch.no_grad()
    def _compute_metrics(
        self, *, buf: ReplayBuffer, batch_size: int, steps: int, tag: str,
        model_override: torch.nn.Module | None = None,
    ) -> TrainMetrics:
        sums: dict[str, float] = {}
        acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
  # Accumulate on-device so each eval batch adds ~0 host syncs; one .item()
  # at the end produces the global Brier + ECE.
        calib_accum: dict[str, torch.Tensor] = {}

        mirror_p = self.mirror_prob if str(tag).startswith("train") else 0.0

  # ``model_override`` lets the async test-eval path drive the loop on a
  # snapshot model while ``self.model`` is being mutated by the next iter's
  # train phase. ``self.model`` is read elsewhere (notably by
  # _policy_accuracy_stats — but that takes ``out`` already, not ``self.model``).
        eval_model = model_override if model_override is not None else self.model

        for batch in self._iter_prefetched_batches(
            buf, batch_size=batch_size, mirror_prob=mirror_p, count=int(steps),
        ):
            with self._amp_context():
                out = eval_model(batch["x"])
                losses = compute_loss(out, batch, **self._loss_kwargs)

            scalars = self._extract_loss_scalars(losses)
            scalars["loss"] = float(losses["total"].item())
            for k, v in scalars.items():
                sums[k] = sums.get(k, 0.0) + v

            for name, (n_, d_) in self._policy_accuracy_stats(out, batch).items():
                prev = acc_sums.get(name)
                acc_sums[name] = (n_, d_) if prev is None else (prev[0] + n_, prev[1] + d_)

            wdl_logits = out.get("wdl")
            wdl_target = batch.get("wdl_t")
            if wdl_logits is not None and wdl_target is not None and wdl_target.numel() > 0:
                stats = wdl_calibration_stats(wdl_logits.detach(), wdl_target)
                for k, v in stats.items():
                    calib_accum[k] = calib_accum.get(k, torch.zeros_like(v)) + v

        wdl_brier, wdl_ece = wdl_brier_ece_from_stats(calib_accum) if calib_accum else (0.0, 0.0)
        metrics = self._build_metrics(
            sums, acc_sums, float(max(1, steps)),
            wdl_brier=wdl_brier, wdl_ece=wdl_ece,
        )
        self._log_metrics(metrics, tag)
        return metrics

    def _apply_feature_group_dropout(self, x: torch.Tensor) -> None:
        """Per-group classical-feature dropout applied in-place on x[:, base:, ...]."""
        base = int(self._base_input_planes)
        if x.shape[1] <= base:
            return
        for g_off, g_len, g_p in self._feature_group_dropout:
            if g_p > 0.0:
                drop = (torch.rand((x.shape[0], 1, 1, 1), device=x.device) < g_p).to(x.dtype)
                x[:, base + g_off : base + g_off + g_len, :, :] *= (1.0 - drop)

    def _run_optimizer_step(
        self, *, step_sums: dict[str, float], step_acc_sums: dict, buf: ReplayBuffer, batch_size: int,
    ) -> tuple[int, float]:
        """Run accum_steps microbatches, do zclip + opt.step + lr update.

        Mutates step_sums/step_acc_sums in place. Returns (step_n_micro, opt_step_time_s).
        """
        self.opt.zero_grad(set_to_none=True)
        step_n_micro = 0
        for batch in self._iter_prefetched_batches(
            buf, batch_size=batch_size,
            mirror_prob=self.mirror_prob, count=self.accum_steps,
        ):
            self._apply_feature_group_dropout(batch["x"])
            with self._amp_context():
                out = self.model(batch["x"])
                losses = compute_loss(out, batch, **self._loss_kwargs)
                balance_loss = getattr(self.model, "_last_channel_balance_loss", None)
                if balance_loss is not None and self.resid_channel_balance_weight > 0.0:
                    losses["channel_balance"] = balance_loss
                    losses["total"] = losses["total"] + self.resid_channel_balance_weight * balance_loss
                loss = losses["total"] / self.accum_steps
            loss.backward()

            scalars = self._extract_loss_scalars(losses)
            scalars["loss"] = float(loss.item() * self.accum_steps)
            for k, v in scalars.items():
                step_sums[k] = step_sums.get(k, 0.0) + v

            with torch.no_grad():
                for name, (n_, d_) in self._policy_accuracy_stats(out, batch).items():
                    prev = step_acc_sums.get(name)
                    step_acc_sums[name] = (n_, d_) if prev is None else (prev[0] + n_, prev[1] + d_)

            step_n_micro += 1

        should_log = self._should_log_step_scalars()
        grad_norm, zclip_stats = self._zclip_step(collect_stats=should_log)
        if should_log:
            self.writer.add_scalar("train/grad_norm", float(grad_norm), self.step)
            if zclip_stats is not None:
                self.writer.add_scalar("zclip/total_norm", zclip_stats["total_norm"], self.step)
                self.writer.add_scalar("zclip/effective_clip", zclip_stats["effective_clip"], self.step)
                self.writer.add_scalar("zclip/adaptive_clipped", zclip_stats["adaptive_clip"], self.step)
                self.writer.add_scalar("zclip/hard_clipped", zclip_stats["hard_clip"], self.step)
                self.writer.add_scalar("zclip/clipped", zclip_stats["clipped"], self.step)
        opt_step_start = time.perf_counter()
        self.opt.step()
        opt_step_time_s = time.perf_counter() - opt_step_start
        self._update_lr()
        return step_n_micro, opt_step_time_s

    def train_steps(self, buf: ReplayBuffer, *, batch_size: int, steps: int) -> TrainMetrics:
        self.model.train()
        train_wall_start = time.perf_counter()

        sums: dict[str, float] = {}
        acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        n_micro = 0
        opt_step_time_s = 0.0
        train_steps_done = 0

        _log = logging.getLogger(__name__)

        for _ in range(int(steps)):
          for _attempt in range(3):
            step_sums: dict[str, float] = {}
            step_acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            try:
                step_n_micro, this_opt_time = self._run_optimizer_step(
                    step_sums=step_sums, step_acc_sums=step_acc_sums,
                    buf=buf, batch_size=batch_size,
                )
                opt_step_time_s += this_opt_time
            except RuntimeError as exc:
                if "CUDA" not in str(exc) or _attempt >= 2:
                    raise
                _log.warning("Transient CUDA error (attempt %d/3), retrying: %s", _attempt + 1, exc)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                time.sleep(0.5 * (_attempt + 1))
                self.opt.zero_grad(set_to_none=True)
                continue

  # Success — commit metrics from this step.
            for k, v in step_sums.items():
                sums[k] = sums.get(k, 0.0) + v
            for name, (n_, d_) in step_acc_sums.items():
                prev = acc_sums.get(name)
                acc_sums[name] = (n_, d_) if prev is None else (prev[0] + n_, prev[1] + d_)
            n_micro += step_n_micro

            if (
                self._swa_model is not None
                and self.step >= self._swa_start
                and self.step % self._swa_freq == 0
            ):
                self._swa_model.update_parameters(self.model)

            if self._should_log_step_scalars():
                self.writer.add_scalar("train/loss", float(step_sums.get("loss", 0.0) / max(1, step_n_micro)), self.step)
                self.writer.add_scalar("train/lr", self.opt.param_groups[0]["lr"], self.step)
            self.step += 1
            train_steps_done += 1
            break

        train_time_s = time.perf_counter() - train_wall_start
        train_samples_seen = int(n_micro * batch_size)
        metrics = self._build_metrics(
            sums, acc_sums, float(max(1, n_micro)),
            train_time_s=float(train_time_s),
            opt_step_time_s=float(opt_step_time_s),
            train_steps_done=int(train_steps_done),
            train_samples_seen=int(train_samples_seen),
            **getattr(self.opt, "last_uw_stats", {}),
        )
        self._log_metrics(metrics, "train_avg")

  # Compile probe: report once after the first batch of train steps that
  # reach the configured threshold. This catches: (a) compile not engaged,
  # (b) graphs failing to capture, (c) per-step recompile thrash.
        if self._compile_probe_steps_remaining > 0 and train_steps_done > 0:
            self._compile_probe_steps_remaining -= train_steps_done
            if self._compile_probe_steps_remaining <= 0:
                self._compile_probe.report(step_count=10)

  # Log throughput stats that aren't in TrainMetrics
        self.writer.add_scalar("train_avg/steps_per_s", float(train_steps_done / max(train_time_s, 1e-9)), self.step)
        self.writer.add_scalar("train_avg/samples_per_s", float(train_samples_seen / max(train_time_s, 1e-9)), self.step)
        self.writer.add_scalar("train_avg/opt_steps_per_s", float(train_steps_done / max(opt_step_time_s, 1e-9)) if opt_step_time_s > 0.0 else 0.0, self.step)
        return metrics

    @torch.no_grad()
    def eval_steps(self, buf: ReplayBuffer, *, batch_size: int, steps: int) -> TrainMetrics:
        self.model.eval()
        return self._compute_metrics(buf=buf, batch_size=batch_size, steps=steps, tag="eval")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Strip torch.compile's `_orig_mod.` prefix before saving so the
        # checkpoint is wrap-agnostic. Without this, a save under
        # `use_compile=true` produces keys like `_orig_mod.embed.weight`,
        # which a later load into an unwrapped trainer will silently drop
        # (every key looks "unexpected"), leaving the trainer fresh-init.
        # That's the failure mode that destroyed the model on 2026-04-27.
        def _strip_compile_prefix(sd: dict) -> dict:
            return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

        state: dict[str, Any] = {
            "model": _strip_compile_prefix(self.model.state_dict()),
            "opt": self.opt.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "step": self.step,
            "peak_lr": float(self._peak_lr),
        }
        if self._model_config is not None:
            state["arch"] = {
                "_schema_version": ARCH_SCHEMA_VERSION,
                **dataclasses.asdict(self._model_config),
            }
        if self._swa_model is not None:
            state["swa_model"] = _strip_compile_prefix(self._swa_model.state_dict())
  # Atomic write so workers polling for new checkpoints never see a partial
  # file (matches the export_swa path; previously diverged).
        atomic_write(path, lambda tmp: torch.save(state, str(tmp)))

    def load(self, path: Path) -> None:
        from chess_anti_engine.model import load_state_dict_tolerant

        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        load_state_dict_tolerant(self.model, ckpt["model"], label="resume")
        fresh_opt_state = self.opt.state_dict()
        fresh_scheduler_state = self._scheduler.state_dict()
        optimizer_state_loaded = True
        try:
            self.opt.load_state_dict(ckpt["opt"])
        except (ValueError, KeyError, RuntimeError) as exc:
            optimizer_state_loaded = False
            logging.getLogger(__name__).warning(
                "Optimizer state incompatible with new model layout, "
                "reinitialising optimizer: %s", exc,
            )
            self.opt.load_state_dict(fresh_opt_state)
            self._scheduler.load_state_dict(fresh_scheduler_state)
            self.reset_optimizer_reference_weights()
        if "scheduler" in ckpt and optimizer_state_loaded:
            try:
                self._scheduler.load_state_dict(ckpt["scheduler"])
            except (ValueError, KeyError, RuntimeError) as exc:
                logging.getLogger(__name__).warning(
                    "Scheduler state incompatible with current optimizer layout, "
                    "reinitialising scheduler: %s", exc,
                )
                self._scheduler.load_state_dict(fresh_scheduler_state)
        if "peak_lr" in ckpt:
            self._peak_lr = float(ckpt["peak_lr"])
        else:
            self._peak_lr = self._reference_lr_from_bases()
        if "swa_model" in ckpt and self._swa_model is not None:
            try:
                self._swa_model.load_state_dict(ckpt["swa_model"])
            except (RuntimeError, KeyError) as exc:
                logging.getLogger(__name__).warning(
                    "SWA model state incompatible, reinitialising: %s", exc,
                )
        self.step = int(ckpt.get("step", 0))

    def export_swa(self, path: Path, dataloader: Any = None) -> None:
        """Export the SWA-averaged model weights.

        If a dataloader is provided, batch normalization statistics are updated
        using ``torch.optim.swa_utils.update_bn``.

        Written atomically to avoid races with workers downloading the file
        while the learner is writing it.
        """
        if self._swa_model is not None and dataloader is not None:
            torch.optim.swa_utils.update_bn(
                dataloader,
                self._swa_model,
                device=torch.device(self.device),
            )
        state_dict = (
            self.model.state_dict()
            if self._swa_model is None
            else self._swa_model.module.state_dict()
        )
        export: dict[str, Any] = {"model": state_dict}
        if self._model_config is not None:
            export["arch"] = {
                "_schema_version": ARCH_SCHEMA_VERSION,
                **dataclasses.asdict(self._model_config),
            }
        atomic_write(path, lambda tmp: torch.save(export, str(tmp)))
