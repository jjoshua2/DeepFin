from __future__ import annotations

import numpy as np
import torch

from chess_anti_engine.moves import POLICY_SIZE as _PS

from .buffer import ReplaySample


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


def collate(samples: list[ReplaySample], *, device: str) -> dict[str, torch.Tensor]:
    x = np.stack([s.x for s in samples], axis=0).astype(np.float32, copy=False)
    policy_t = np.stack([s.policy_target for s in samples], axis=0).astype(np.float32, copy=False)
    wdl_t = np.array([s.wdl_target for s in samples], dtype=np.int64)

    has_policy = np.array([1.0 if getattr(s, "has_policy", True) else 0.0 for s in samples], dtype=np.float32)

    sf_wdl = np.zeros((len(samples), 3), dtype=np.float32)
    has_sf_wdl = np.zeros((len(samples),), dtype=np.float32)

    sf_move_index = np.zeros((len(samples),), dtype=np.int64)
    has_sf_move = np.zeros((len(samples),), dtype=np.float32)

    sf_policy_t = np.zeros_like(policy_t, dtype=np.float32)
    has_sf_policy = np.zeros((len(samples),), dtype=np.float32)

    moves_left = np.zeros((len(samples),), dtype=np.float32)
    has_moves_left = np.zeros((len(samples),), dtype=np.float32)
    is_network_turn = np.zeros((len(samples),), dtype=np.bool_)

    categorical_t = np.zeros((len(samples), 32), dtype=np.float32)
    has_categorical = np.zeros((len(samples),), dtype=np.float32)

    policy_soft_t = np.zeros_like(policy_t, dtype=np.float32)
    has_policy_soft = np.zeros((len(samples),), dtype=np.float32)
    future_policy_t = np.zeros_like(policy_t, dtype=np.float32)
    has_future = np.zeros((len(samples),), dtype=np.float32)

    volatility_t = np.zeros((len(samples), 3), dtype=np.float32)
    has_volatility = np.zeros((len(samples),), dtype=np.float32)

    sf_volatility_t = np.zeros((len(samples), 3), dtype=np.float32)
    has_sf_volatility = np.zeros((len(samples),), dtype=np.float32)

    legal_mask_t = np.zeros((len(samples), _PS), dtype=np.float32)
    has_legal_mask = np.zeros((len(samples),), dtype=np.float32)

    for i, s in enumerate(samples):
        if s.sf_wdl is not None:
            sf_wdl[i] = s.sf_wdl.astype(np.float32, copy=False)
            has_sf_wdl[i] = 1.0
        if s.sf_move_index is not None:
            sf_move_index[i] = int(s.sf_move_index)
            has_sf_move[i] = 1.0
        if getattr(s, "sf_policy_target", None) is not None:
            sf_policy_t[i] = s.sf_policy_target.astype(np.float32, copy=False)
            has_sf_policy[i] = 1.0
        if s.moves_left is not None:
            moves_left[i] = float(s.moves_left)
            has_moves_left[i] = 1.0
        if s.is_network_turn is not None:
            is_network_turn[i] = bool(s.is_network_turn)
        if s.categorical_target is not None:
            categorical_t[i] = s.categorical_target.astype(np.float32, copy=False)
            has_categorical[i] = 1.0
        if s.policy_soft_target is not None:
            policy_soft_t[i] = s.policy_soft_target.astype(np.float32, copy=False)
            has_policy_soft[i] = 1.0
        if s.future_policy_target is not None:
            future_policy_t[i] = s.future_policy_target.astype(np.float32, copy=False)
            has_future[i] = 1.0
        if s.volatility_target is not None:
            volatility_t[i] = s.volatility_target.astype(np.float32, copy=False)
            has_volatility[i] = 1.0
        if getattr(s, "sf_volatility_target", None) is not None:
            sf_volatility_t[i] = s.sf_volatility_target.astype(np.float32, copy=False)
            has_sf_volatility[i] = 1.0
        if getattr(s, "legal_mask", None) is not None:
            legal_mask_t[i] = s.legal_mask.astype(np.float32, copy=False)
            has_legal_mask[i] = 1.0

    return {
        "x": _to_tensor(x, device=device),
        "policy_t": _to_tensor(policy_t, device=device),
        "wdl_t": _to_tensor(wdl_t, device=device),
        "has_policy": _to_tensor(has_policy, device=device),
        "sf_wdl": _to_tensor(sf_wdl, device=device),
        "has_sf_wdl": _to_tensor(has_sf_wdl, device=device),
        "sf_move_index": _to_tensor(sf_move_index, device=device),
        "has_sf_move": _to_tensor(has_sf_move, device=device),
        "sf_policy_t": _to_tensor(sf_policy_t, device=device),
        "has_sf_policy": _to_tensor(has_sf_policy, device=device),
        "moves_left": _to_tensor(moves_left, device=device),
        "has_moves_left": _to_tensor(has_moves_left, device=device),
        "is_network_turn": _to_tensor(is_network_turn, device=device),
        "categorical_t": _to_tensor(categorical_t, device=device),
        "has_categorical": _to_tensor(has_categorical, device=device),
        "policy_soft_t": _to_tensor(policy_soft_t, device=device),
        "has_policy_soft": _to_tensor(has_policy_soft, device=device),
        "future_policy_t": _to_tensor(future_policy_t, device=device),
        "has_future": _to_tensor(has_future, device=device),
        "volatility_t": _to_tensor(volatility_t, device=device),
        "has_volatility": _to_tensor(has_volatility, device=device),
        "sf_volatility_t": _to_tensor(sf_volatility_t, device=device),
        "has_sf_volatility": _to_tensor(has_sf_volatility, device=device),
        "legal_mask": _to_tensor(legal_mask_t, device=device),
        "has_legal_mask": _to_tensor(has_legal_mask, device=device),
    }


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
        ("legal_mask", (n, _PS), np.float32, torch.float32),
        ("has_legal_mask", (n,), np.float32, torch.float32),
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
