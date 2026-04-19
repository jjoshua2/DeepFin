from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.moves import POLICY_SIZE

from .buffer import ReplaySample

try:  # pragma: no cover - import availability is environment-dependent
    import zarr
    from numcodecs import Blosc
except Exception:  # pragma: no cover
    zarr = None
    Blosc = None


SHARD_VERSION = 1
LOCAL_SHARD_SUFFIX = ".zarr"
LEGACY_SHARD_SUFFIX = ".npz"
_SHARD_FIELDS = (
    "x",
    "policy_target",
    "wdl_target",
    "priority",
    "has_policy",
    "sf_wdl",
    "has_sf_wdl",
    "sf_move_index",
    "has_sf_move",
    "sf_policy_target",
    "has_sf_policy",
    "moves_left",
    "has_moves_left",
    "is_network_turn",
    "has_is_network_turn",
    "categorical_target",
    "has_categorical",
    "policy_soft_target",
    "has_policy_soft",
    "future_policy_target",
    "has_future",
    "volatility_target",
    "has_volatility",
    "sf_volatility_target",
    "has_sf_volatility",
    "legal_mask",
    "has_legal_mask",
)

_REQUIRED_STORAGE_FIELDS = (
    "x",
    "policy_target",
    "wdl_target",
    "priority",
    "has_policy",
)

_OPTIONAL_STORAGE_PAIRS = (
    ("sf_wdl", "has_sf_wdl"),
    ("sf_move_index", "has_sf_move"),
    ("sf_policy_target", "has_sf_policy"),
    ("moves_left", "has_moves_left"),
    ("is_network_turn", "has_is_network_turn"),
    ("categorical_target", "has_categorical"),
    ("policy_soft_target", "has_policy_soft"),
    ("future_policy_target", "has_future"),
    ("volatility_target", "has_volatility"),
    ("sf_volatility_target", "has_sf_volatility"),
    ("legal_mask", "has_legal_mask"),
)


# ---------------------------------------------------------------------------
# Sparse policy storage for in-memory shuffle buffers
# ---------------------------------------------------------------------------
# Policy arrays are (N, 4672) but only ~30-40 entries are non-zero per row.
# Storing as padded-sparse (values + column indices + lengths) saves ~10x
# memory per policy field in the shuffle buffer.

_POLICY_SPARSE_FIELDS = ("policy_target", "sf_policy_target", "policy_soft_target", "future_policy_target")


def _padded_positions(nnz: np.ndarray, rows: np.ndarray, N: int) -> np.ndarray:
    """Compute within-row position for each nonzero element (for padded-sparse layout)."""
    row_starts = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(nnz, out=row_starts[1:])
    return np.arange(len(rows), dtype=np.int64) - row_starts[rows]


def _sparsify_policy(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (N, P) float16 dense policy → padded sparse (vals, cols, nnz)."""
    N = dense.shape[0]
    nz_mask = dense != 0
    nnz = nz_mask.sum(axis=1).astype(np.uint16)
    K = int(nnz.max()) if N > 0 else 0
    if K == 0:
        return (np.zeros((N, 0), dtype=dense.dtype),
                np.zeros((N, 0), dtype=np.uint16),
                nnz)
    rows, col_idxs = np.nonzero(nz_mask)
    vals_flat = dense[rows, col_idxs]
    positions = _padded_positions(nnz, rows, N)
    out_vals = np.zeros((N, K), dtype=dense.dtype)
    out_cols = np.zeros((N, K), dtype=np.uint16)
    out_vals[rows, positions] = vals_flat
    out_cols[rows, positions] = col_idxs.astype(np.uint16)
    return out_vals, out_cols, nnz


def _densify_policy(vals: np.ndarray, cols: np.ndarray, nnz: np.ndarray,
                    policy_size: int) -> np.ndarray:
    """Convert padded sparse (vals, cols, nnz) → (N, P) dense."""
    N = vals.shape[0]
    K = vals.shape[1] if vals.ndim == 2 else 0
    out = np.zeros((N, policy_size), dtype=vals.dtype)
    if K == 0 or N == 0:
        return out
    valid = np.arange(K, dtype=np.uint16)[None, :] < nnz[:, None]
    rows, ks = np.nonzero(valid)
    out[rows, cols[rows, ks]] = vals[rows, ks]
    return out


def sparsify_chunk(arrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert dense policy arrays in a chunk dict to padded-sparse format."""
    out = dict(arrs)
    for key in _POLICY_SPARSE_FIELDS:
        if key not in out:
            continue
        dense = out[key]
        if dense.ndim != 2 or dense.shape[1] <= 0:
            continue
        vals, cols, nnz = _sparsify_policy(dense)
        out[key] = vals
        out[f"{key}_cols"] = cols
        out[f"{key}_nnz"] = nnz
    # legal_mask: store as indices only (values are always 1)
    if "legal_mask" in out:
        mask = out["legal_mask"]
        if mask.ndim == 2 and mask.shape[1] > 0:
            N = mask.shape[0]
            nz_mask = mask != 0
            nnz = nz_mask.sum(axis=1).astype(np.uint16)
            K = int(nnz.max()) if N > 0 else 0
            if K > 0:
                rows, col_idxs = np.nonzero(nz_mask)
                positions = _padded_positions(nnz, rows, N)
                idx_arr = np.zeros((N, K), dtype=np.uint16)
                idx_arr[rows, positions] = col_idxs.astype(np.uint16)
            else:
                idx_arr = np.zeros((N, 0), dtype=np.uint16)
            out["legal_mask"] = idx_arr
            out["legal_mask_nnz"] = nnz
    return out


def densify_chunk(arrs: dict[str, np.ndarray], policy_size: int = POLICY_SIZE) -> dict[str, np.ndarray]:
    """Convert padded-sparse policy arrays back to dense format."""
    out = dict(arrs)
    for key in _POLICY_SPARSE_FIELDS:
        cols_key = f"{key}_cols"
        nnz_key = f"{key}_nnz"
        if cols_key not in out:
            continue
        dense = _densify_policy(out[key], out[cols_key], out[nnz_key], policy_size)
        out[key] = dense
        del out[cols_key]
        del out[nnz_key]
    # legal_mask
    if "legal_mask_nnz" in out:
        idx_arr = out["legal_mask"]
        nnz = out["legal_mask_nnz"]
        N = idx_arr.shape[0]
        K = idx_arr.shape[1] if idx_arr.ndim == 2 else 0
        mask = np.zeros((N, policy_size), dtype=np.uint8)
        if K > 0 and N > 0:
            valid = np.arange(K, dtype=np.uint16)[None, :] < nnz[:, None]
            rows, ks = np.nonzero(valid)
            mask[rows, idx_arr[rows, ks]] = 1
        out["legal_mask"] = mask
        del out["legal_mask_nnz"]
    return out


@dataclass(frozen=True)
class ShardMeta:
    version: int = SHARD_VERSION
    username: str | None = None
    run_id: str | None = None
    generated_at_unix: int | None = None
    model_sha256: str | None = None
    model_step: int | None = None
    games: int | None = None
    positions: int | None = None
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
    plies_win: int | None = None
    plies_draw: int | None = None
    plies_loss: int | None = None
    checkmate_games: int | None = None
    stalemate_games: int | None = None


def _u8(x: np.ndarray) -> np.ndarray:
    return x.astype(np.uint8, copy=False)


def _f16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16, copy=False)


def _copy_row(arr: np.ndarray, i: int, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if dtype is None:
        return np.array(arr[i], copy=True, order="C")
    return np.array(arr[i], dtype=dtype, copy=True, order="C")


def _meta_dict(meta: ShardMeta | dict[str, Any] | None, *, positions: int) -> dict[str, Any]:
    if meta is None:
        return asdict(ShardMeta(positions=int(positions)))
    if isinstance(meta, ShardMeta):
        return asdict(meta)
    return asdict(ShardMeta(**meta))


def prune_storage_arrays(arrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop universally-absent optional fields before writing a shard.

    The loader and replay buffers already synthesize zero defaults for missing
    optional arrays, so persisting all-zero target tensors and has-flags wastes
    disk, I/O, and decode CPU without changing training semantics.
    """
    validate_arrays(arrs)
    out: dict[str, np.ndarray] = {name: np.asarray(arrs[name]) for name in _REQUIRED_STORAGE_FIELDS}
    for value_name, flag_name in _OPTIONAL_STORAGE_PAIRS:
        flag = np.asarray(arrs.get(flag_name, np.zeros((out["x"].shape[0],), dtype=np.uint8)), dtype=np.uint8)
        if np.any(flag):
            out[flag_name] = flag
            if value_name in arrs:
                out[value_name] = np.asarray(arrs[value_name])
    return out


def local_shard_path(shard_dir: str | Path, index: int) -> Path:
    return Path(shard_dir) / f"shard_{int(index):06d}{LOCAL_SHARD_SUFFIX}"


def local_iter_shard_path(shard_dir: str | Path, index: int) -> Path:
    return local_shard_path(shard_dir, index)


def iter_shard_paths(shard_dir: str | Path) -> list[Path]:
    d = Path(shard_dir)
    out = list(d.glob(f"shard_*{LOCAL_SHARD_SUFFIX}")) + list(d.glob(f"shard_*{LEGACY_SHARD_SUFFIX}"))
    return sorted(out)


def shard_exists(shard_dir: str | Path, index: int) -> bool:
    return find_shard_path(shard_dir, index) is not None


def find_shard_path(shard_dir: str | Path, index: int) -> Path | None:
    d = Path(shard_dir)
    stem = f"shard_{int(index):06d}"
    for suffix in (LOCAL_SHARD_SUFFIX, LEGACY_SHARD_SUFFIX):
        p = d / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def shard_index(path: str | Path) -> int:
    p = Path(path)
    name = p.name
    if name.endswith(LOCAL_SHARD_SUFFIX):
        name = name[: -len(LOCAL_SHARD_SUFFIX)]
    elif name.endswith(LEGACY_SHARD_SUFFIX):
        name = name[: -len(LEGACY_SHARD_SUFFIX)]
    try:
        return int(name.split("_")[1])
    except Exception:
        return -1


def shard_positions(path: str | Path) -> int:
    p = Path(path)
    if p.suffix == LOCAL_SHARD_SUFFIX:
        if zarr is None:
            return 0
        try:
            g = zarr.open_group(str(p), mode="r")
            return int(g["x"].shape[0])  # type: ignore[arg-type,union-attr]
        except Exception:
            return 0
    try:
        with np.load(str(p), allow_pickle=False) as z:
            return int(z["x"].shape[0]) if "x" in z.files else 0
    except Exception:
        return 0


def copy_or_link_shard(src: str | Path, dst: str | Path) -> Path:
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    rel = None
    try:
        rel = Path(os.path.relpath(src_p, start=dst_p.parent))
    except Exception:
        rel = None
    try:
        os.symlink(str(rel if rel is not None else src_p), str(dst_p), target_is_directory=src_p.is_dir())
        return dst_p
    except FileExistsError:
        return dst_p
    except OSError:
        pass
    if src_p.is_dir():
        if dst_p.exists():
            shutil.rmtree(dst_p, ignore_errors=True)
        shutil.copytree(str(src_p), str(dst_p))
    else:
        shutil.copy2(str(src_p), str(dst_p))
    return dst_p


def delete_shard_path(path: str | Path) -> None:
    p = Path(path)
    if p.is_symlink():
        p.unlink(missing_ok=True)
    elif p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        p.unlink(missing_ok=True)


def samples_to_arrays(samples: list[ReplaySample]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("cannot serialize empty shard")

    x = _f16(np.stack([s.x for s in samples], axis=0))
    policy_target = _f16(np.stack([s.policy_target for s in samples], axis=0))
    wdl_target = np.array([int(s.wdl_target) for s in samples], dtype=np.int8)

    priority = np.array([float(getattr(s, "priority", 1.0)) for s in samples], dtype=np.float32)
    has_policy = _u8(np.array([1 if getattr(s, "has_policy", True) else 0 for s in samples], dtype=np.uint8))

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
        "x": x,
        "policy_target": policy_target,
        "wdl_target": wdl_target,
        "priority": priority,
        "has_policy": has_policy,
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


def validate_arrays(arrs: dict[str, np.ndarray]) -> None:
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
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN/Inf")
    if not np.isfinite(policy).all():
        raise ValueError("policy_target contains NaN/Inf")
    if (policy < -1e-6).any():
        raise ValueError("policy_target contains negative values")
    row_sums = policy.astype(np.float64).sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("policy_target has rows with non-positive sum")
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


def save_npz(
    path: str | Path,
    *,
    samples: list[ReplaySample],
    meta: ShardMeta | dict[str, Any] | None = None,
    compress: bool = True,
) -> Path:
    arrs = samples_to_arrays(samples)
    return save_npz_arrays(path, arrs=arrs, meta=meta, compress=compress)


def save_npz_arrays(
    path: str | Path,
    *,
    arrs: dict[str, np.ndarray],
    meta: ShardMeta | dict[str, Any] | None = None,
    compress: bool = True,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stored = prune_storage_arrays(arrs)
    meta_json = json.dumps(_meta_dict(meta, positions=int(np.asarray(stored["x"]).shape[0])), sort_keys=True)
    if bool(compress):
        np.savez_compressed(str(p), **stored, meta_json=np.array(meta_json))
    else:
        np.savez(str(p), **stored, meta_json=np.array(meta_json))
    return p


def _local_chunks(arr: np.ndarray) -> tuple[int, ...]:
    n = int(arr.shape[0])
    lead = min(max(1, n), 512)
    if arr.ndim == 1:
        return (lead,)
    return (lead, *arr.shape[1:])


def save_local_shard_arrays(
    path: str | Path,
    *,
    arrs: dict[str, np.ndarray],
    meta: ShardMeta | dict[str, Any] | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stored = prune_storage_arrays(arrs)
    if zarr is None or Blosc is None:
        return save_npz_arrays(p.with_suffix(LEGACY_SHARD_SUFFIX), arrs=stored, meta=meta)
    # Write to a temp path then atomic-rename to avoid races with concurrent
    # readers/writers that can cause "Directory not empty" on rmtree.
    tmp = p.with_name(f"._{os.getpid()}_{p.name}")
    try:
        g = zarr.open_group(str(tmp), mode="w")
        g.attrs.update(_meta_dict(meta, positions=int(np.asarray(stored["x"]).shape[0])))
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
        for name, value in stored.items():
            arr = np.asarray(value)
            g.create_dataset(name, data=arr, chunks=_local_chunks(arr), compressor=compressor, overwrite=True)
        # Atomic replace: remove old, rename new.
        if p.exists():
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        tmp.rename(p)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return p


def load_npz_arrays(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    p = Path(path)
    with np.load(str(p), allow_pickle=False) as z:
        arrs = {k: np.array(z[k], copy=False) for k in z.files if k != "meta_json"}
        meta_json = z["meta_json"].item() if "meta_json" in z.files else "{}"
    meta = json.loads(str(meta_json)) if meta_json else {}
    validate_arrays(arrs)
    return arrs, meta


def load_shard_arrays(
    path: str | Path,
    *,
    lazy: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    p = Path(path)
    if p.suffix == LOCAL_SHARD_SUFFIX:
        if zarr is None:
            raise RuntimeError("zarr support is not available")
        g = zarr.open_group(str(p), mode="r")
        meta = dict(g.attrs.asdict())
        if lazy:
            arrs = {name: g[name] for name in _SHARD_FIELDS if name in g}
            return arrs, meta
        arrs = {name: np.asarray(g[name]) for name in _SHARD_FIELDS if name in g}
        validate_arrays(arrs)
        return arrs, meta
    return load_npz_arrays(p)


def load_npz(path: str | Path) -> tuple[list[ReplaySample], dict[str, Any]]:
    arrs, meta = load_npz_arrays(path)
    return arrays_to_samples(arrs), meta
