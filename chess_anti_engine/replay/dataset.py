from __future__ import annotations

import numpy as np
import torch

from chess_anti_engine.moves import POLICY_SIZE as _PS

from .buffer import ReplaySample
from .shard import LEGAL_MASK_FIELDS, LEGAL_MASK_HAS_FIELDS


def _to_tensor(arr: np.ndarray, *, device: str) -> torch.Tensor:
    t = torch.from_numpy(arr)
    if str(device).startswith("cuda"):
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def _zeros_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype, device=device)


def _to_optional_tensor(
    arrs: dict[str, np.ndarray],
    key: str,
    *,
    shape: tuple[int, ...],
    dtype_np: np.dtype | type,
    dtype_torch: torch.dtype,
    device: str,
) -> torch.Tensor:
    arr = arrs.get(key)
    if arr is None:
        return _zeros_tensor(shape, dtype=dtype_torch, device=device)
    return _to_tensor(np.asarray(arr, dtype=dtype_np), device=device)


  # Optional float-vector fields. Each entry: (sample_attr, target_key, has_key, shape_per_sample).
  # Values are .astype(float32) into target[i] when present. Drives the per-sample loop in collate().
_OPTIONAL_FLOAT_FIELDS: tuple[tuple[str, str, str, tuple[int, ...]], ...] = (
    ("sf_wdl",              "sf_wdl",          "has_sf_wdl",       (3,)),
    ("sf_policy_target",    "sf_policy_t",     "has_sf_policy",    (_PS,)),
    ("categorical_target",  "categorical_t",   "has_categorical",  (32,)),
    ("policy_soft_target",  "policy_soft_t",   "has_policy_soft",  (_PS,)),
    ("future_policy_target","future_policy_t", "has_future",       (_PS,)),
    ("volatility_target",   "volatility_t",    "has_volatility",   (3,)),
    ("sf_volatility_target","sf_volatility_t", "has_sf_volatility",(3,)),
)


def _build_collate_arrays(samples: list[ReplaySample]) -> dict[str, np.ndarray]:
    """Build per-sample numpy arrays for every collate field. Returns the array
    dict; ``collate`` then converts every entry to a tensor."""
    n = len(samples)
    out: dict[str, np.ndarray] = {
        "x": np.stack([s.x for s in samples], axis=0).astype(np.float32, copy=False),
        "policy_t": np.stack([s.policy_target for s in samples], axis=0).astype(np.float32, copy=False),
        "wdl_t": np.array([s.wdl_target for s in samples], dtype=np.int64),
        "has_policy": np.array(
            [1.0 if getattr(s, "has_policy", True) else 0.0 for s in samples], dtype=np.float32,
        ),
        "sf_move_index": np.zeros((n,), dtype=np.int64),
        "has_sf_move": np.zeros((n,), dtype=np.float32),
        "moves_left": np.zeros((n,), dtype=np.float32),
        "has_moves_left": np.zeros((n,), dtype=np.float32),
        "is_network_turn": np.zeros((n,), dtype=np.bool_),
        "is_selfplay": np.zeros((n,), dtype=np.bool_),
        "has_is_selfplay": np.zeros((n,), dtype=np.float32),
    }
    for src, target, has, shape in _OPTIONAL_FLOAT_FIELDS:
        out[target] = np.zeros((n, *shape), dtype=np.float32)
        out[has] = np.zeros((n,), dtype=np.float32)
    for k in LEGAL_MASK_FIELDS:
        out[k] = np.zeros((n, _PS), dtype=np.float32)
    for k in LEGAL_MASK_HAS_FIELDS:
        out[k] = np.zeros((n,), dtype=np.float32)

    for i, s in enumerate(samples):
        for src, target, has, _ in _OPTIONAL_FLOAT_FIELDS:
            v = getattr(s, src, None)
            if v is not None:
                out[target][i] = v.astype(np.float32, copy=False)
                out[has][i] = 1.0
        if s.sf_move_index is not None:
            out["sf_move_index"][i] = int(s.sf_move_index)
            out["has_sf_move"][i] = 1.0
        if s.moves_left is not None:
            out["moves_left"][i] = float(s.moves_left)
            out["has_moves_left"][i] = 1.0
        if s.is_network_turn is not None:
            out["is_network_turn"][i] = bool(s.is_network_turn)
        if s.is_selfplay is not None:
            out["is_selfplay"][i] = bool(s.is_selfplay)
            out["has_is_selfplay"][i] = 1.0
        for mk, hk in zip(LEGAL_MASK_FIELDS, LEGAL_MASK_HAS_FIELDS, strict=True):
            v = getattr(s, mk, None)
            if v is not None:
                out[mk][i] = v.astype(np.float32, copy=False)
                out[hk][i] = 1.0
    return out


def collate(samples: list[ReplaySample], *, device: str) -> dict[str, torch.Tensor]:
    arrs = _build_collate_arrays(samples)
    return {k: _to_tensor(v, device=device) for k, v in arrs.items()}


def collate_arrays(arrs: dict[str, np.ndarray], *, device: str) -> dict[str, torch.Tensor]:
    x = np.asarray(arrs["x"], dtype=np.float32)
    n = int(x.shape[0])

    policy_t = np.asarray(arrs["policy_target"], dtype=np.float32)
    wdl_t = np.asarray(arrs["wdl_target"], dtype=np.int64)

    out = {
        "x": _to_tensor(x, device=device),
        "policy_t": _to_tensor(policy_t, device=device),
        "wdl_t": _to_tensor(wdl_t, device=device),
    }
    if "has_policy" in arrs:
        out["has_policy"] = _to_tensor(np.asarray(arrs["has_policy"], dtype=np.float32), device=device)

    optional_specs = (
        ("sf_wdl", (n, 3), np.float32, torch.float32),
        ("has_sf_wdl", (n,), np.float32, torch.float32),
        ("sf_move_index", (n,), np.int64, torch.int64),
        ("has_sf_move", (n,), np.float32, torch.float32),
        ("sf_policy_t", (n, policy_t.shape[1]), np.float32, torch.float32, "sf_policy_target"),
        ("has_sf_policy", (n,), np.float32, torch.float32),
        ("moves_left", (n,), np.float32, torch.float32),
        ("has_moves_left", (n,), np.float32, torch.float32),
        ("is_network_turn", (n,), np.bool_, torch.bool),
        ("is_selfplay", (n,), np.bool_, torch.bool),
        ("has_is_selfplay", (n,), np.float32, torch.float32),
        ("categorical_t", (n, 32), np.float32, torch.float32, "categorical_target"),
        ("has_categorical", (n,), np.float32, torch.float32),
        ("policy_soft_t", (n, policy_t.shape[1]), np.float32, torch.float32, "policy_soft_target"),
        ("has_policy_soft", (n,), np.float32, torch.float32),
        ("future_policy_t", (n, policy_t.shape[1]), np.float32, torch.float32, "future_policy_target"),
        ("has_future", (n,), np.float32, torch.float32),
        ("volatility_t", (n, 3), np.float32, torch.float32, "volatility_target"),
        ("has_volatility", (n,), np.float32, torch.float32),
        ("sf_volatility_t", (n, 3), np.float32, torch.float32, "sf_volatility_target"),
        ("has_sf_volatility", (n,), np.float32, torch.float32),
        *[(k, (n, _PS), np.float32, torch.float32) for k in LEGAL_MASK_FIELDS],
        *[(k, (n,), np.float32, torch.float32) for k in LEGAL_MASK_HAS_FIELDS],
    )
    for spec in optional_specs:
        if len(spec) == 4:
            out_key, shape, np_dtype, torch_dtype = spec
            src_key = out_key
        else:
            out_key, shape, np_dtype, torch_dtype, src_key = spec
        if src_key in arrs:
            out[out_key] = _to_optional_tensor(
                arrs, src_key, shape=shape, dtype_np=np_dtype, dtype_torch=torch_dtype, device=device
            )
    return out
