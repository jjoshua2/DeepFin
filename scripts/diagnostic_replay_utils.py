"""Shared replay-scanning helpers for live diagnostic scripts."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.replay.shard import shard_index


MAX_SKIPPED_SHARD_DETAILS = 20


def latest_replay_dir(run_dir: Path, trial_dir: Path | None = None) -> Path:
    if trial_dir is not None:
        replay_dir = run_dir / "replay" / trial_dir.name / "replay_shards"
        if replay_dir.is_dir():
            return replay_dir
    replay_dirs = [p for p in (run_dir / "replay").glob("train_trial_*/replay_shards") if p.is_dir()]
    if not replay_dirs:
        raise FileNotFoundError(f"No replay shard directories under {run_dir / 'replay'}")
    return max(replay_dirs, key=lambda p: p.stat().st_mtime)


def select_shards(replay_dir: Path, max_shards: int) -> list[Path]:
    shards = sorted(replay_dir.glob("shard_*.zarr"), key=shard_index)
    return shards if max_shards <= 0 else shards[-max_shards:]


def record_skipped_shard(scan: dict[str, Any], shard: Path, exc: Exception) -> None:
    if len(scan["skipped_shards"]) < MAX_SKIPPED_SHARD_DETAILS:
        scan["skipped_shards"].append({"shard": str(shard), "reason": repr(exc)})
    else:
        scan["skipped_shards_omitted"] += 1


def bool_field(arrs: dict[str, Any], name: str, n: int) -> np.ndarray:
    if name not in arrs:
        return np.zeros((n,), dtype=bool)
    return np.asarray(arrs[name], dtype=bool)


def float_field(arrs: dict[str, Any], name: str, n: int, default: float = math.nan) -> np.ndarray:
    if name not in arrs:
        return np.full((n,), default, dtype=np.float64)
    return np.asarray(arrs[name], dtype=np.float64)


def int_field(arrs: dict[str, Any], name: str, n: int, default: int = -1) -> np.ndarray:
    if name not in arrs:
        return np.full((n,), default, dtype=np.int64)
    return np.asarray(arrs[name], dtype=np.int64)


def optional_float_array(
    arrs: dict[str, Any],
    value_name: str,
    has_name: str,
    n: int,
) -> np.ndarray:
    out = np.full((n,), math.nan, dtype=np.float64)
    if value_name not in arrs:
        return out
    values = np.asarray(arrs[value_name], dtype=np.float64)
    mask = bool_field(arrs, has_name, n) if has_name in arrs else np.isfinite(values)
    out[mask] = values[mask]
    return out


def optional_float_array_with_mask(
    arrs: dict[str, Any],
    value_name: str,
    has_name: str,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = optional_float_array(arrs, value_name, has_name, n)
    has = bool_field(arrs, has_name, n) if has_name in arrs else np.isfinite(values)
    return values, has


def optional_int_array(
    arrs: dict[str, Any],
    value_name: str,
    has_name: str,
    n: int,
) -> np.ndarray:
    out = np.full((n,), -1, dtype=np.int64)
    if value_name not in arrs:
        return out
    values = np.asarray(arrs[value_name], dtype=np.int64)
    mask = bool_field(arrs, has_name, n) if has_name in arrs else values >= 0
    out[mask] = values[mask]
    return out


def normalize_wdl(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wdl = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(wdl).all(axis=1)
    non_negative = (wdl >= 0.0).all(axis=1)
    sums = wdl.sum(axis=1)
    valid = finite & non_negative & (sums > 1e-12)
    out = np.zeros_like(wdl, dtype=np.float64)
    out[valid] = wdl[valid] / sums[valid, None]
    return out, valid


def final_q_from_wdl_target(wdl_target: np.ndarray) -> np.ndarray:
    target = np.asarray(wdl_target, dtype=np.int64)
    return np.where(target == 0, 1.0, np.where(target == 1, 0.0, -1.0)).astype(np.float64)


def rmse(pred: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> float:
    if target.size == 0:
        return math.nan
    err = (pred - target) ** 2
    if weights is None:
        return float(np.sqrt(np.mean(err)))
    return float(np.sqrt(np.average(err, weights=weights)))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return math.nan
    a0 = a - float(np.mean(a))
    b0 = b - float(np.mean(b))
    denom = float(np.sqrt(np.dot(a0, a0) * np.dot(b0, b0)))
    if denom <= 1e-12:
        return math.nan
    return float(np.dot(a0, b0) / denom)


def fit_simplex(features: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    n_features = int(features.shape[1])
    if weights is None:
        weights = np.ones((features.shape[0],), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights)

    root_w = np.sqrt(weights / max(1e-12, float(np.mean(weights))))
    xw = features * root_w[:, None]
    yw = target * root_w
    best_w = np.full((n_features,), 1.0 / max(1, n_features), dtype=np.float64)
    best_err = math.inf
    for mask in range(1, 1 << n_features):
        active = [i for i in range(n_features) if (mask >> i) & 1]
        x = xw[:, active]
        if len(active) == 1:
            # With the simplex sum-to-one constraint, a one-feature active set has weight 1.0.
            w_active = np.ones((1,), dtype=np.float64)
        else:
            gram = x.T @ x
            rhs = x.T @ yw
            ones = np.ones((len(active), 1), dtype=np.float64)
            kkt = np.block([[gram, ones], [ones.T, np.zeros((1, 1), dtype=np.float64)]])
            vec = np.concatenate([rhs, np.ones((1,), dtype=np.float64)])
            try:
                sol = np.linalg.solve(kkt, vec)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(kkt, vec, rcond=None)[0]
            w_active = sol[: len(active)]
        if np.any(w_active < -1e-9):
            continue
        w = np.zeros((n_features,), dtype=np.float64)
        w[np.asarray(active, dtype=np.int64)] = np.maximum(0.0, w_active)
        total = float(w.sum())
        if total <= 1e-12:
            continue
        w /= total
        err = float(np.average((features @ w - target) ** 2, weights=weights))
        if err < best_err:
            best_err = err
            best_w = w
    return best_w
