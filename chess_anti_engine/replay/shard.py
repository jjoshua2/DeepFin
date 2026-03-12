from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.moves import POLICY_SIZE

from .buffer import ReplaySample


SHARD_VERSION = 1


@dataclass(frozen=True)
class ShardMeta:
    version: int = SHARD_VERSION

    # Optional provenance (useful for moderation / leaderboard later)
    username: str | None = None
    run_id: str | None = None
    generated_at_unix: int | None = None

    # Optional model provenance
    model_sha256: str | None = None
    model_step: int | None = None

    # Convenience stats
    games: int | None = None
    positions: int | None = None

    # Game-level outcomes from the network's perspective (for PID difficulty control)
    wins: int | None = None
    draws: int | None = None
    losses: int | None = None
    total_game_plies: int | None = None
    adjudicated_games: int | None = None
    total_draw_games: int | None = None
    selfplay_games: int | None = None
    selfplay_adjudicated_games: int | None = None
    selfplay_draw_games: int | None = None
    curriculum_games: int | None = None
    curriculum_adjudicated_games: int | None = None
    curriculum_draw_games: int | None = None


def _u8(x: np.ndarray) -> np.ndarray:
    return x.astype(np.uint8, copy=False)


def _f16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16, copy=False)


def _copy_row(arr: np.ndarray, i: int, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if dtype is None:
        return np.array(arr[i], copy=True, order="C")
    return np.array(arr[i], dtype=dtype, copy=True, order="C")


def samples_to_arrays(samples: list[ReplaySample]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("cannot serialize empty shard")

    x = _f16(np.stack([s.x for s in samples], axis=0))
    policy_target = _f16(np.stack([s.policy_target for s in samples], axis=0))
    wdl_target = np.array([int(s.wdl_target) for s in samples], dtype=np.int8)

    # Required-ish metadata for training
    priority = np.array([float(getattr(s, "priority", 1.0)) for s in samples], dtype=np.float32)
    has_policy = _u8(np.array([1 if getattr(s, "has_policy", True) else 0 for s in samples], dtype=np.uint8))

    # Aux targets + masks (mirrors replay/dataset.py behavior)
    sf_wdl = np.zeros((len(samples), 3), dtype=np.float16)
    has_sf_wdl = np.zeros((len(samples),), dtype=np.uint8)

    sf_move_index = np.zeros((len(samples),), dtype=np.int32)
    has_sf_move = np.zeros((len(samples),), dtype=np.uint8)

    sf_policy_target = np.zeros_like(policy_target, dtype=np.float16)
    has_sf_policy = np.zeros((len(samples),), dtype=np.uint8)

    moves_left = np.zeros((len(samples),), dtype=np.float16)
    has_moves_left = np.zeros((len(samples),), dtype=np.uint8)

    is_network_turn = np.zeros((len(samples),), dtype=np.uint8)
    has_is_network_turn = np.zeros((len(samples),), dtype=np.uint8)

    categorical_target = np.zeros((len(samples), 32), dtype=np.float16)
    has_categorical = np.zeros((len(samples),), dtype=np.uint8)

    policy_soft_target = np.zeros_like(policy_target, dtype=np.float16)
    has_policy_soft = np.zeros((len(samples),), dtype=np.uint8)

    future_policy_target = np.zeros_like(policy_target, dtype=np.float16)
    has_future = np.zeros((len(samples),), dtype=np.uint8)

    volatility_target = np.zeros((len(samples), 3), dtype=np.float16)
    has_volatility = np.zeros((len(samples),), dtype=np.uint8)

    sf_volatility_target = np.zeros((len(samples), 3), dtype=np.float16)
    has_sf_volatility = np.zeros((len(samples),), dtype=np.uint8)

    legal_mask = np.zeros((len(samples), POLICY_SIZE), dtype=np.uint8)
    has_legal_mask = np.zeros((len(samples),), dtype=np.uint8)

    for i, s in enumerate(samples):
        if s.sf_wdl is not None:
            sf_wdl[i] = np.asarray(s.sf_wdl, dtype=np.float16)
            has_sf_wdl[i] = 1
        if s.sf_move_index is not None:
            sf_move_index[i] = int(s.sf_move_index)
            has_sf_move[i] = 1
        if s.sf_policy_target is not None:
            sf_policy_target[i] = np.asarray(s.sf_policy_target, dtype=np.float16)
            has_sf_policy[i] = 1
        if s.moves_left is not None:
            moves_left[i] = np.float16(float(s.moves_left))
            has_moves_left[i] = 1
        if s.is_network_turn is not None:
            is_network_turn[i] = 1 if bool(s.is_network_turn) else 0
            has_is_network_turn[i] = 1
        if s.categorical_target is not None:
            categorical_target[i] = np.asarray(s.categorical_target, dtype=np.float16)
            has_categorical[i] = 1
        if s.policy_soft_target is not None:
            policy_soft_target[i] = np.asarray(s.policy_soft_target, dtype=np.float16)
            has_policy_soft[i] = 1
        if s.future_policy_target is not None:
            future_policy_target[i] = np.asarray(s.future_policy_target, dtype=np.float16)
            has_future[i] = 1
        if s.volatility_target is not None:
            volatility_target[i] = np.asarray(s.volatility_target, dtype=np.float16)
            has_volatility[i] = 1
        if getattr(s, "sf_volatility_target", None) is not None:
            sf_volatility_target[i] = np.asarray(s.sf_volatility_target, dtype=np.float16)
            has_sf_volatility[i] = 1
        if getattr(s, "legal_mask", None) is not None:
            legal_mask[i] = np.asarray(s.legal_mask, dtype=np.uint8)
            has_legal_mask[i] = 1

    return {
        # required
        "x": x,
        "policy_target": policy_target,
        "wdl_target": wdl_target,
        # priorities / masks
        "priority": priority,
        "has_policy": has_policy,
        # aux
        "sf_wdl": sf_wdl,
        "has_sf_wdl": has_sf_wdl,
        "sf_move_index": sf_move_index,
        "has_sf_move": has_sf_move,
        "sf_policy_target": sf_policy_target,
        "has_sf_policy": has_sf_policy,
        "moves_left": moves_left,
        "has_moves_left": has_moves_left,
        "is_network_turn": is_network_turn,
        "has_is_network_turn": has_is_network_turn,
        "categorical_target": categorical_target,
        "has_categorical": has_categorical,
        "policy_soft_target": policy_soft_target,
        "has_policy_soft": has_policy_soft,
        "future_policy_target": future_policy_target,
        "has_future": has_future,
        "volatility_target": volatility_target,
        "has_volatility": has_volatility,
        "sf_volatility_target": sf_volatility_target,
        "has_sf_volatility": has_sf_volatility,
        "legal_mask": legal_mask,
        "has_legal_mask": has_legal_mask,
    }


def _require_shape(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"bad shard field {name}: expected shape {shape}, got {tuple(arr.shape)}")


def validate_arrays(arrs: dict[str, np.ndarray]) -> None:
    # required
    if "x" not in arrs or "policy_target" not in arrs or "wdl_target" not in arrs:
        raise ValueError("shard missing required fields")

    x = np.asarray(arrs["x"])
    policy = np.asarray(arrs["policy_target"])
    wdl = np.asarray(arrs["wdl_target"])

    if x.ndim != 4:
        raise ValueError(f"x must be (N,C,8,8); got {x.shape}")
    if x.shape[-2:] != (8, 8):
        raise ValueError(f"x must end with (8,8); got {x.shape}")

    if policy.ndim != 2:
        raise ValueError(f"policy_target must be (N,A); got {policy.shape}")
    if policy.shape[0] != x.shape[0]:
        raise ValueError("policy_target N mismatch")
    if int(policy.shape[1]) != int(POLICY_SIZE):
        raise ValueError(f"policy_target A mismatch: expected {POLICY_SIZE}, got {policy.shape[1]}")

    if wdl.ndim != 1 or wdl.shape[0] != x.shape[0]:
        raise ValueError("wdl_target must be (N,) matching x")

    # Numeric sanity (reject NaNs/Infs and obviously broken distributions)
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN/Inf")
    if not np.isfinite(policy).all():
        raise ValueError("policy_target contains NaN/Inf")

    if (policy < -1e-6).any():
        raise ValueError("policy_target contains negative values")

    row_sums = policy.astype(np.float64).sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("policy_target has rows with non-positive sum")

    # WDL labels must be in {0,1,2}
    wdl_i = wdl.astype(np.int64, copy=False)
    if ((wdl_i < 0) | (wdl_i > 2)).any():
        raise ValueError("wdl_target out of range")


def arrays_to_samples(arrs: dict[str, np.ndarray]) -> list[ReplaySample]:
    validate_arrays(arrs)

    x = np.asarray(arrs["x"])
    policy = np.asarray(arrs["policy_target"])
    wdl = np.asarray(arrs["wdl_target"]).astype(np.int64, copy=False)

    n = int(x.shape[0])

    priority = np.asarray(arrs.get("priority", np.ones((n,), dtype=np.float32)), dtype=np.float32)
    has_policy = np.asarray(arrs.get("has_policy", np.ones((n,), dtype=np.uint8)), dtype=np.uint8)

    sf_wdl = np.asarray(arrs.get("sf_wdl", np.zeros((n, 3), dtype=np.float16)))
    has_sf_wdl = np.asarray(arrs.get("has_sf_wdl", np.zeros((n,), dtype=np.uint8)), dtype=np.uint8)

    sf_move_index = np.asarray(arrs.get("sf_move_index", np.zeros((n,), dtype=np.int32)), dtype=np.int32)
    has_sf_move = np.asarray(arrs.get("has_sf_move", np.zeros((n,), dtype=np.uint8)), dtype=np.uint8)

    sf_policy_target = np.asarray(arrs.get("sf_policy_target", np.zeros_like(policy, dtype=np.float16)))
    has_sf_policy = np.asarray(arrs.get("has_sf_policy", np.zeros((n,), dtype=np.uint8)), dtype=np.uint8)

    moves_left = np.asarray(arrs.get("moves_left", np.zeros((n,), dtype=np.float16)))
    has_moves_left = np.asarray(arrs.get("has_moves_left", np.zeros((n,), dtype=np.uint8)), dtype=np.uint8)

    is_network_turn = np.asarray(arrs.get("is_network_turn", np.zeros((n,), dtype=np.uint8)), dtype=np.uint8)
    has_is_network_turn = np.asarray(arrs.get("has_is_network_turn", np.zeros((n,), dtype=np.uint8)), dtype=np.uint8)

    categorical = np.asarray(arrs.get("categorical_target", np.zeros((n, 32), dtype=np.float16)))
    has_categorical = np.asarray(arrs.get("has_categorical", np.zeros((n,), dtype=np.uint8)), dtype=np.uint8)

    policy_soft = np.asarray(arrs.get("policy_soft_target", np.zeros_like(policy, dtype=np.float16)))
    has_policy_soft = np.asarray(arrs.get("has_policy_soft", np.zeros((n,), dtype=np.uint8)), dtype=np.uint8)

    future_policy = np.asarray(arrs.get("future_policy_target", np.zeros_like(policy, dtype=np.float16)))
    has_future = np.asarray(arrs.get("has_future", np.zeros((n,), dtype=np.uint8)), dtype=np.uint8)

    vol = np.asarray(arrs.get("volatility_target", np.zeros((n, 3), dtype=np.float16)))
    has_vol = np.asarray(arrs.get("has_volatility", np.zeros((n,), dtype=np.uint8)), dtype=np.uint8)

    sf_vol = np.asarray(arrs.get("sf_volatility_target", np.zeros((n, 3), dtype=np.float16)))
    has_sf_vol = np.asarray(arrs.get("has_sf_volatility", np.zeros((n,), dtype=np.uint8)), dtype=np.uint8)

    legal_mask_arr = np.asarray(arrs.get("legal_mask", np.zeros((n, POLICY_SIZE), dtype=np.uint8)), dtype=np.uint8)
    has_legal_mask = np.asarray(arrs.get("has_legal_mask", np.zeros((n,), dtype=np.uint8)), dtype=np.uint8)

    out: list[ReplaySample] = []
    for i in range(n):
        s = ReplaySample(
            x=_copy_row(x, i),
            policy_target=_copy_row(policy, i),
            wdl_target=int(wdl[i]),
            priority=float(priority[i]),
            has_policy=bool(int(has_policy[i]) != 0),
        )
        if bool(int(has_sf_wdl[i]) != 0):
            s.sf_wdl = _copy_row(sf_wdl, i)
        if bool(int(has_sf_move[i]) != 0):
            s.sf_move_index = int(sf_move_index[i])
        if bool(int(has_sf_policy[i]) != 0):
            s.sf_policy_target = _copy_row(sf_policy_target, i)
        if bool(int(has_moves_left[i]) != 0):
            s.moves_left = float(moves_left[i])
        if bool(int(has_is_network_turn[i]) != 0):
            s.is_network_turn = bool(int(is_network_turn[i]) != 0)
        if bool(int(has_categorical[i]) != 0):
            s.categorical_target = _copy_row(categorical, i)
        if bool(int(has_policy_soft[i]) != 0):
            s.policy_soft_target = _copy_row(policy_soft, i)
        if bool(int(has_future[i]) != 0):
            s.future_policy_target = _copy_row(future_policy, i)
            s.has_future = True
        if bool(int(has_vol[i]) != 0):
            s.volatility_target = _copy_row(vol, i)
            s.has_volatility = True
        if bool(int(has_sf_vol[i]) != 0):
            s.sf_volatility_target = _copy_row(sf_vol, i)
            s.has_sf_volatility = True
        if bool(int(has_legal_mask[i]) != 0):
            s.legal_mask = _copy_row(legal_mask_arr, i, dtype=np.uint8)
        out.append(s)

    return out


def save_npz(path: str | Path, *, samples: list[ReplaySample], meta: ShardMeta | dict[str, Any] | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    arrs = samples_to_arrays(samples)

    if meta is None:
        meta_obj = ShardMeta(positions=len(samples))
    elif isinstance(meta, ShardMeta):
        meta_obj = meta
    else:
        meta_obj = ShardMeta(**meta)

    meta_json = json.dumps(asdict(meta_obj), sort_keys=True)
    np.savez_compressed(str(p), **arrs, meta_json=np.array(meta_json))
    return p


def load_npz(path: str | Path) -> tuple[list[ReplaySample], dict[str, Any]]:
    p = Path(path)
    with np.load(str(p), allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files if k != "meta_json"}
        meta_json = z["meta_json"].item() if "meta_json" in z.files else "{}"

    meta = json.loads(str(meta_json)) if meta_json else {}
    samples = arrays_to_samples(arrs)
    return samples, meta
