from __future__ import annotations

import numpy as np
import torch

from .buffer import ReplaySample
from .shard import (
    LEGAL_MASK_FIELDS,
    LEGAL_MASK_HAS_FIELDS,
    SF_MULTIPV_RAW_COLS,
    SF_MULTIPV_RAW_MAX,
)


def _to_tensor(
    arr: np.ndarray,
    *,
    device: str,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    t = torch.from_numpy(arr)
    if str(device).startswith("cuda"):
        pinned = t.pin_memory()
        if dtype is not None:
            return pinned.to(device=device, dtype=dtype, non_blocking=True)
        return pinned.to(device=device, non_blocking=True)
    if dtype is not None:
        return t.to(device=device, dtype=dtype)
    return t.to(device=device)


def _transfer_array(value: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    """Keep safely widenable replay values compact until device transfer."""
    arr = np.asarray(value)
    target = np.dtype(dtype)
    if arr.dtype.itemsize <= target.itemsize and np.can_cast(arr.dtype, target, casting="safe"):
        return arr
    return np.asarray(arr, dtype=target)


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
    source = _transfer_array(arr, dtype=dtype_np)
    return _to_tensor(source, device=device, dtype=dtype_torch)


  # Optional float-vector fields. Each entry: (sample_attr, target_key, has_key, shape_per_sample).
  # Values are .astype(float32) into target[i] when present. Drives the per-sample loop in collate().
_POLICY_SHAPE = (-1,)

_OPTIONAL_FLOAT_FIELDS: tuple[tuple[str, str, str, tuple[int, ...]], ...] = (
    ("sf_wdl",              "sf_wdl",          "has_sf_wdl",       (3,)),
    ("search_wdl",          "search_wdl",      "has_search_wdl",   (3,)),
    ("sf_policy_target",    "sf_policy_t",     "has_sf_policy",    _POLICY_SHAPE),
    ("categorical_target",  "categorical_t",   "has_categorical",  (32,)),
    ("policy_soft_target",  "policy_soft_t",   "has_policy_soft",  _POLICY_SHAPE),
    ("future_policy_target","future_policy_t", "has_future",       _POLICY_SHAPE),
    ("sf_p0_policy_target", "sf_p0_policy_t",  "has_sf_p0",        _POLICY_SHAPE),
    ("sf_p0_regret",        "sf_p0_regret_t",  "has_sf_p0_regret", _POLICY_SHAPE),
    ("volatility_target",   "volatility_t",    "has_volatility",   (3,)),
    ("sf_volatility_target","sf_volatility_t", "has_sf_volatility",(3,)),
)
_OPTIONAL_SCALAR_FIELDS: tuple[tuple[str, str, str, np.dtype | type], ...] = (
    ("future_sf_regret_sum", "future_sf_regret_sum", "has_future_sf_regret_sum", np.float32),
    ("future_sf_regret_d95", "future_sf_regret_d95", "has_future_sf_regret_d95", np.float32),
    ("future_sf_regret_d98", "future_sf_regret_d98", "has_future_sf_regret_d98", np.float32),
    ("future_sf_regret_max", "future_sf_regret_max", "has_future_sf_regret_max", np.float32),
    ("future_sf_regret_h4", "future_sf_regret_h4", "has_future_sf_regret_h4", np.float32),
    ("future_sf_regret_h6", "future_sf_regret_h6", "has_future_sf_regret_h6", np.float32),
    ("future_sf_regret_h12", "future_sf_regret_h12", "has_future_sf_regret_h12", np.float32),
    ("future_sf_regret_h24", "future_sf_regret_h24", "has_future_sf_regret_h24", np.float32),
    ("future_sf_regret_h50", "future_sf_regret_h50", "has_future_sf_regret_h50", np.float32),
    ("future_sf_regret_count", "future_sf_regret_count", "has_future_sf_regret_count", np.int32),
)
_OPTIONAL_EXPLICIT_HAS_ATTRS: dict[str, str] = {
    "future_policy_target": "has_future",
    "sf_p0_policy_target": "has_sf_p0",
    "sf_p0_regret": "has_sf_p0_regret",
    "volatility_target": "has_volatility",
    "sf_volatility_target": "has_sf_volatility",
}


def _sample_has_optional(s: ReplaySample, src: str) -> bool:
    explicit = _OPTIONAL_EXPLICIT_HAS_ATTRS.get(src)
    if explicit is not None:
        flag = getattr(s, explicit, None)
        if flag is not None:
            return bool(flag)
    return getattr(s, src, None) is not None


def _build_collate_arrays(samples: list[ReplaySample]) -> dict[str, np.ndarray]:
    """Build per-sample numpy arrays for every collate field. Returns the array
    dict; ``collate`` then converts every entry to a tensor."""
    n = len(samples)
    policy_size = int(np.asarray(samples[0].policy_target).shape[0])
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
    if any(getattr(s, "relations", None) is not None for s in samples):
        rel = np.zeros((n, 5, 64, 64), dtype=np.uint8)
        has_rel = np.zeros((n,), dtype=np.float32)
        for i, s_ in enumerate(samples):
            r = getattr(s_, "relations", None)
            if r is not None:
                rel[i] = np.asarray(r, dtype=np.uint8)
                has_rel[i] = 1.0
        out["relations"] = rel
        out["has_relations"] = has_rel
    for _src, target, has, shape in _OPTIONAL_FLOAT_FIELDS:  # skylos: ignore — spec tuple unpack
        shape = (policy_size,) if shape == _POLICY_SHAPE else shape
        out[target] = np.zeros((n, *shape), dtype=np.float32)
        out[has] = np.zeros((n,), dtype=np.float32)
    for _src, target, has, dtype in _OPTIONAL_SCALAR_FIELDS:  # skylos: ignore — spec tuple unpack
        out[target] = np.zeros((n,), dtype=dtype)
        out[has] = np.zeros((n,), dtype=np.float32)
    for k in LEGAL_MASK_FIELDS:
        out[k] = np.zeros((n, policy_size), dtype=np.float32)
    for k in LEGAL_MASK_HAS_FIELDS:
        out[k] = np.zeros((n,), dtype=np.float32)

    for i, s in enumerate(samples):
        for src, target, has, _ in _OPTIONAL_FLOAT_FIELDS:
            v = getattr(s, src, None)
            if v is not None and _sample_has_optional(s, src):
                out[target][i] = v.astype(np.float32, copy=False)
                out[has][i] = 1.0
        for src, target, has, dtype in _OPTIONAL_SCALAR_FIELDS:
            v = getattr(s, src, None)
            if v is not None:
                out[target][i] = int(v) if np.issubdtype(np.dtype(dtype), np.integer) else float(v)
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
    x = _transfer_array(arrs["x"], dtype=np.float32)
    n = int(x.shape[0])

    policy_t = _transfer_array(arrs["policy_target"], dtype=np.float32)
    wdl_t = _transfer_array(arrs["wdl_target"], dtype=np.int64)

    out = {
        "x": _to_tensor(x, device=device, dtype=torch.float32),
        "policy_t": _to_tensor(policy_t, device=device, dtype=torch.float32),
        "wdl_t": _to_tensor(wdl_t, device=device, dtype=torch.int64),
    }
    if "has_policy" in arrs:
        has_policy = _transfer_array(arrs["has_policy"], dtype=np.float32)
        out["has_policy"] = _to_tensor(has_policy, device=device, dtype=torch.float32)
  # Dynamic relation matrices ride along as uint8; the model casts on device.
  # Rows without relations are zero matrices == zero bias, so a mixed batch
  # (old shards + new) degrades exactly like the absent-tensor case.
    if "relations" in arrs and "has_relations" in arrs and bool(np.any(np.asarray(arrs["has_relations"]))):
        out["relations"] = _to_tensor(np.asarray(arrs["relations"], dtype=np.uint8), device=device)

    optional_specs = (
        ("sf_wdl", (n, 3), np.float32, torch.float32),
        ("has_sf_wdl", (n,), np.float32, torch.float32),
        ("search_wdl", (n, 3), np.float32, torch.float32),
        ("has_search_wdl", (n,), np.float32, torch.float32),
        ("future_sf_regret_sum", (n,), np.float32, torch.float32),
        ("has_future_sf_regret_sum", (n,), np.float32, torch.float32),
        ("future_sf_regret_d95", (n,), np.float32, torch.float32),
        ("has_future_sf_regret_d95", (n,), np.float32, torch.float32),
        ("future_sf_regret_d98", (n,), np.float32, torch.float32),
        ("has_future_sf_regret_d98", (n,), np.float32, torch.float32),
        ("future_sf_regret_max", (n,), np.float32, torch.float32),
        ("has_future_sf_regret_max", (n,), np.float32, torch.float32),
        ("future_sf_regret_h4", (n,), np.float32, torch.float32),
        ("has_future_sf_regret_h4", (n,), np.float32, torch.float32),
        ("future_sf_regret_h6", (n,), np.float32, torch.float32),
        ("has_future_sf_regret_h6", (n,), np.float32, torch.float32),
        ("future_sf_regret_h12", (n,), np.float32, torch.float32),
        ("has_future_sf_regret_h12", (n,), np.float32, torch.float32),
        ("future_sf_regret_h24", (n,), np.float32, torch.float32),
        ("has_future_sf_regret_h24", (n,), np.float32, torch.float32),
        ("future_sf_regret_h50", (n,), np.float32, torch.float32),
        ("has_future_sf_regret_h50", (n,), np.float32, torch.float32),
        ("future_sf_regret_count", (n,), np.int32, torch.int32),
        ("has_future_sf_regret_count", (n,), np.float32, torch.float32),
        ("sf_move_index", (n,), np.int64, torch.int64),
        ("has_sf_move", (n,), np.float32, torch.float32),
        # Sparse MultiPV rows: only present when the trainer requests them
        # (train.sf_policy_sparse_ce) — _sample_batch_host drops them
        # otherwise so the default H2D payload is unchanged.
        ("sf_multipv_raw", (n, SF_MULTIPV_RAW_MAX, SF_MULTIPV_RAW_COLS), np.int32, torch.int32),
        ("has_sf_multipv_raw", (n,), np.float32, torch.float32),
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
        ("sf_p0_policy_t", (n, policy_t.shape[1]), np.float32, torch.float32, "sf_p0_policy_target"),
        ("has_sf_p0", (n,), np.float32, torch.float32),
        ("sf_p0_regret_t", (n, policy_t.shape[1]), np.float32, torch.float32, "sf_p0_regret"),
        ("has_sf_p0_regret", (n,), np.float32, torch.float32),
        ("volatility_t", (n, 3), np.float32, torch.float32, "volatility_target"),
        ("has_volatility", (n,), np.float32, torch.float32),
        ("sf_volatility_t", (n, 3), np.float32, torch.float32, "sf_volatility_target"),
        ("has_sf_volatility", (n,), np.float32, torch.float32),
        *[(k, (n, policy_t.shape[1]), np.float32, torch.float32) for k in LEGAL_MASK_FIELDS],
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
