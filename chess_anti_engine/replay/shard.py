from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS

from .buffer import ReplaySample

SHARD_VERSION = 1
LOCAL_SHARD_SUFFIX = ".zarr"
LEGACY_SHARD_SUFFIX = ".npz"

# Server-managed staging dir for crash-recoverable uploads. Lives at
# ``inbox_root/_pending`` and is replayed by ``server.app.create_app`` on
# startup; learner-side ingest skips it (see ``_iter_shard_paths_nested``)
# so the same samples don't reach replay through both channels.
PENDING_DIR_NAME = "_pending"

# Server-managed flush staging. While a compacted shard is being written,
# the contributing pending zarrs are moved into
# ``inbox_root/_in_flight/<flush_token>/``. The same ``flush_token`` is
# embedded in the compacted shard's filename, so recovery can decide per
# in-flight group whether the flush committed (matching compacted shard
# exists → safe to delete the group) or crashed before commit (no match →
# move the contents back to ``_pending`` for re-seeding).
IN_FLIGHT_DIR_NAME = "_in_flight"


def is_tmp_shard_name(name: str) -> bool:
    """In-progress upload staging name: tmp directories the server is mid-write
    on (or the ``._tmp_*`` ``Path.replace`` stems numpy/zarr leave behind).
    """
    return name.startswith("tmp_") or name.startswith("._tmp_")


@dataclass(frozen=True)
class _OptFieldSpec:
    """Storage spec for one optional shard field.

    ``arr`` is the value array name, ``flag`` is the per-sample uint8 flag
    indicating presence. ``shape`` is the trailing shape (no batch dim).
    Single source of truth for the schema — ``_SHARD_FIELDS``,
    ``_OPTIONAL_STORAGE_PAIRS``, the zero-fill allocations in
    ``samples_to_arrays``/``arrays_to_samples``, and the lazy zarr loader
    all derive from this.
    """
    arr: str
    flag: str
    shape: tuple[int, ...]
    dtype: np.dtype


_POLICY_SHAPE: tuple[int, ...] = (POLICY_SIZE,)
_F16: np.dtype = np.dtype(np.float16)
_U8_DT: np.dtype = np.dtype(np.uint8)
_I32_DT: np.dtype = np.dtype(np.int32)

_OPTIONAL_FIELD_SPECS: tuple[_OptFieldSpec, ...] = (
    _OptFieldSpec("sf_wdl",               "has_sf_wdl",            (3,),          _F16),
    _OptFieldSpec("sf_move_index",        "has_sf_move",           (),            _I32_DT),
    _OptFieldSpec("sf_policy_target",     "has_sf_policy",         _POLICY_SHAPE, _F16),
    _OptFieldSpec("moves_left",           "has_moves_left",        (),            _F16),
    _OptFieldSpec("is_network_turn",      "has_is_network_turn",   (),            _U8_DT),
    _OptFieldSpec("is_selfplay",          "has_is_selfplay",       (),            _U8_DT),
    _OptFieldSpec("categorical_target",   "has_categorical",       (DEFAULT_CATEGORICAL_BINS,), _F16),
    _OptFieldSpec("policy_soft_target",   "has_policy_soft",       _POLICY_SHAPE, _F16),
    _OptFieldSpec("future_policy_target", "has_future",            _POLICY_SHAPE, _F16),
    _OptFieldSpec("volatility_target",    "has_volatility",        (3,),          _F16),
    _OptFieldSpec("sf_volatility_target", "has_sf_volatility",     (3,),          _F16),
    _OptFieldSpec("legal_mask",           "has_legal_mask",        _POLICY_SHAPE, _U8_DT),
    _OptFieldSpec("sf_legal_mask",        "has_sf_legal_mask",     _POLICY_SHAPE, _U8_DT),
    _OptFieldSpec("future_legal_mask",    "has_future_legal_mask", _POLICY_SHAPE, _U8_DT),
)

_REQUIRED_STORAGE_FIELDS: tuple[str, ...] = (
    "x",
    "policy_target",
    "wdl_target",
    "priority",
    "has_policy",
)

_OPTIONAL_STORAGE_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (s.arr, s.flag) for s in _OPTIONAL_FIELD_SPECS
)

_SHARD_FIELDS: tuple[str, ...] = (
    *_REQUIRED_STORAGE_FIELDS,
    *(name for s in _OPTIONAL_FIELD_SPECS for name in (s.arr, s.flag)),
)

# Legal-mask fields: per-head masks in different positions/POVs. Stored as
# packed indices in shards since values are always 0/1.
LEGAL_MASK_FIELDS: tuple[str, ...] = ("legal_mask", "sf_legal_mask", "future_legal_mask")
LEGAL_MASK_HAS_FIELDS: tuple[str, ...] = ("has_legal_mask", "has_sf_legal_mask", "has_future_legal_mask")


# ---------------------------------------------------------------------------
# Sparse policy storage for in-memory shuffle buffers
# ---------------------------------------------------------------------------
# Policy arrays are (N, 4672) but only ~30-40 entries are non-zero per row.
# Storing as padded-sparse (values + column indices + lengths) saves ~10x
# memory per policy field in the shuffle buffer.

POLICY_SPACE_FIELDS = ("policy_target", "sf_policy_target", "policy_soft_target", "future_policy_target")


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
    for key in POLICY_SPACE_FIELDS:
        if key not in out:
            continue
        dense = out[key]
        if dense.ndim != 2 or dense.shape[1] <= 0:
            continue
        vals, cols, nnz = _sparsify_policy(dense)
        out[key] = vals
        out[f"{key}_cols"] = cols
        out[f"{key}_nnz"] = nnz
  # legal masks: store as indices only (values are always 1)
    for key in LEGAL_MASK_FIELDS:
        if key not in out:
            continue
        mask = out[key]
        if mask.ndim != 2 or mask.shape[1] <= 0:
            continue
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
        out[key] = idx_arr
        out[f"{key}_nnz"] = nnz
    return out


def densify_chunk(arrs: dict[str, np.ndarray], policy_size: int = POLICY_SIZE) -> dict[str, np.ndarray]:
    """Convert padded-sparse policy arrays back to dense format."""
    out = dict(arrs)
    for key in POLICY_SPACE_FIELDS:
        cols_key = f"{key}_cols"
        nnz_key = f"{key}_nnz"
        if cols_key not in out:
            continue
        dense = _densify_policy(out[key], out[cols_key], out[nnz_key], policy_size)
        out[key] = dense
        del out[cols_key]
        del out[nnz_key]
  # legal masks
    for key in LEGAL_MASK_FIELDS:
        nnz_key = f"{key}_nnz"
        if nnz_key not in out:
            continue
        idx_arr = out[key]
        nnz = out[nnz_key]
        N = idx_arr.shape[0]
        K = idx_arr.shape[1] if idx_arr.ndim == 2 else 0
        mask = np.zeros((N, policy_size), dtype=np.uint8)
        if K > 0 and N > 0:
            valid = np.arange(K, dtype=np.uint16)[None, :] < nnz[:, None]
            rows, ks = np.nonzero(valid)
            mask[rows, idx_arr[rows, ks]] = 1
        out[key] = mask
        del out[nnz_key]
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
    tb_adjudicated_games: int | None = None
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

    ``priority`` is required downstream but legacy/partial shards may be missing
    it; synthesize ones to match the ``arrs.get("priority", ones)`` default in
    ``arrays_to_samples`` rather than crashing ingest.
    """
    validate_arrays(arrs)
    n = int(np.asarray(arrs["x"]).shape[0])
    out: dict[str, np.ndarray] = {}
    for name in _REQUIRED_STORAGE_FIELDS:
        if name == "priority" and name not in arrs:
            out[name] = np.ones((n,), dtype=np.float32)
        else:
            out[name] = np.asarray(arrs[name])
    for value_name, flag_name in _OPTIONAL_STORAGE_PAIRS:
        flag = np.asarray(arrs.get(flag_name, np.zeros((out["x"].shape[0],), dtype=np.uint8)), dtype=np.uint8)
        if np.any(flag):
            out[flag_name] = flag
            if value_name in arrs:
                out[value_name] = np.asarray(arrs[value_name])
    return out


def local_shard_path(shard_dir: str | Path, index: int) -> Path:
    return Path(shard_dir) / f"shard_{int(index):06d}{LOCAL_SHARD_SUFFIX}"


def iter_shard_paths(shard_dir: str | Path) -> list[Path]:
    """List local replay shards (``shard_NNNNNN.zarr``) under *shard_dir*."""
    return sorted(Path(shard_dir).glob(f"shard_*{LOCAL_SHARD_SUFFIX}"))


def find_shard_path(shard_dir: str | Path, index: int) -> Path | None:
    p = local_shard_path(shard_dir, index)
    return p if p.exists() else None


def shard_index(path: str | Path) -> int:
    stem = Path(path).stem
    try:
        return int(stem.split("_")[1])
    except Exception:
        return -1


def shard_positions(path: str | Path) -> int:
    p = Path(path)
    try:
        g = zarr.open_group(str(p), mode="r")
        return int(g["x"].shape[0])  # type: ignore[arg-type,union-attr] # zarr Group item may be Group or Array at type level
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


# ---------------------------------------------------------------------------
# Worker → server upload wire format
# ---------------------------------------------------------------------------
# Workers tar their local zarr shard directory and POST the bytes as a single
# upload. The server detects the .zarr.tar filename, extracts safely, and
# parses with load_shard_arrays. These two functions own that wire format so
# it stays in sync across producer, consumer, and tests.

UPLOAD_TAR_SUFFIX = LOCAL_SHARD_SUFFIX + ".tar"


def pack_shard_for_upload(shard_path: str | Path) -> tuple[str, io.BytesIO]:
    """Tar a local zarr shard directory for HTTP upload.

    Returns ``(upload_filename, stream)``. The filename carries
    ``UPLOAD_TAR_SUFFIX`` so the server can dispatch by name.
    """
    p = Path(shard_path)
    if p.suffix != LOCAL_SHARD_SUFFIX or not p.is_dir():
        raise ValueError(f"expected a {LOCAL_SHARD_SUFFIX} directory, got {p}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        tf.add(str(p), arcname=p.name)
    buf.seek(0)
    return p.stem + UPLOAD_TAR_SUFFIX, buf


def extract_uploaded_shard_tar(tar_path: str | Path, dest: str | Path) -> Path:
    """Safely extract a worker-uploaded zarr tarball at *tar_path* into *dest*.

    *dest* must not already exist; it is created by this function. Raises
    ``ValueError`` on any tar member that would escape the extract dir — links
    (sym or hard), absolute paths, ``..`` traversal, non-regular files, or
    resolved paths outside *dest*. On success, returns the zarr group root
    (either *dest* itself or a single nested child dir containing ``.zgroup``).

    Defense in depth: the manual member walk rejects link-based escape attacks
    before any bytes touch the filesystem; ``tarfile.extractall(filter="data")``
    strips mode/uid/gid bits and catches anything the walk missed.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    dest_resolved = dest.resolve()
    with tarfile.open(str(tar_path), mode="r:") as tf:
        for member in tf.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"rejected link member: {member.name!r}")
            if not (member.isreg() or member.isdir()):
                raise ValueError(f"rejected non-regular member: {member.name!r}")
            name = member.name
            if not name:
                raise ValueError("rejected empty member name")
            parts = Path(name).parts
            if Path(name).is_absolute() or any(p == ".." for p in parts):
                raise ValueError(f"rejected traversal path: {name!r}")
            resolved = (dest / name).resolve()
            if resolved != dest_resolved and not str(resolved).startswith(
                str(dest_resolved) + os.sep
            ):
                raise ValueError(f"tar member escapes extract dir: {name!r}")
        tf.extractall(str(dest), filter="data")
    entries = list(dest.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and (entries[0] / ".zgroup").exists():
        return entries[0]
    return dest


def samples_to_arrays(samples: list[ReplaySample]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("cannot serialize empty shard")
    n = len(samples)

    arrs: dict[str, np.ndarray] = {
        "x": _f16(np.stack([s.x for s in samples], axis=0)),
        "policy_target": _f16(np.stack([s.policy_target for s in samples], axis=0)),
        "wdl_target": np.array([int(s.wdl_target) for s in samples], dtype=np.int8),
        "priority": np.array([float(getattr(s, "priority", 1.0)) for s in samples], dtype=np.float32),
        "has_policy": _u8(np.array(
            [1 if getattr(s, "has_policy", True) else 0 for s in samples], dtype=np.uint8,
        )),
    }
    for spec in _OPTIONAL_FIELD_SPECS:
        arrs[spec.arr] = np.zeros((n, *spec.shape), dtype=spec.dtype)
        arrs[spec.flag] = np.zeros((n,), dtype=np.uint8)

    for i, s in enumerate(samples):
        if s.sf_wdl is not None:
            arrs["sf_wdl"][i] = np.asarray(s.sf_wdl, dtype=np.float16)
            arrs["has_sf_wdl"][i] = 1
        if s.sf_move_index is not None:
            arrs["sf_move_index"][i] = int(s.sf_move_index)
            arrs["has_sf_move"][i] = 1
        if s.sf_policy_target is not None:
            arrs["sf_policy_target"][i] = np.asarray(s.sf_policy_target, dtype=np.float16)
            arrs["has_sf_policy"][i] = 1
        if s.moves_left is not None:
            arrs["moves_left"][i] = np.float16(float(s.moves_left))
            arrs["has_moves_left"][i] = 1
        if s.is_network_turn is not None:
            arrs["is_network_turn"][i] = 1 if bool(s.is_network_turn) else 0
            arrs["has_is_network_turn"][i] = 1
        if s.is_selfplay is not None:
            arrs["is_selfplay"][i] = 1 if bool(s.is_selfplay) else 0
            arrs["has_is_selfplay"][i] = 1
        if s.categorical_target is not None:
            arrs["categorical_target"][i] = np.asarray(s.categorical_target, dtype=np.float16)
            arrs["has_categorical"][i] = 1
        if s.policy_soft_target is not None:
            arrs["policy_soft_target"][i] = np.asarray(s.policy_soft_target, dtype=np.float16)
            arrs["has_policy_soft"][i] = 1
        if s.future_policy_target is not None:
            arrs["future_policy_target"][i] = np.asarray(s.future_policy_target, dtype=np.float16)
            arrs["has_future"][i] = 1
        if s.volatility_target is not None:
            arrs["volatility_target"][i] = np.asarray(s.volatility_target, dtype=np.float16)
            arrs["has_volatility"][i] = 1
        if s.sf_volatility_target is not None:
            arrs["sf_volatility_target"][i] = np.asarray(s.sf_volatility_target, dtype=np.float16)
            arrs["has_sf_volatility"][i] = 1
        for mk, hk in zip(LEGAL_MASK_FIELDS, LEGAL_MASK_HAS_FIELDS, strict=True):
            v = getattr(s, mk, None)
            if v is not None:
                arrs[mk][i] = np.asarray(v, dtype=np.uint8)
                arrs[hk][i] = 1

    return arrs


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
  # fp32 accumulation of fp16 inputs is plenty for a non-positive-sum check
  # and avoids a full f64 upcast (~2x faster, half the memory on f16 shards).
    row_sums = policy.sum(axis=1, dtype=np.float32)
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

    opt: dict[str, np.ndarray] = {}
    for spec in _OPTIONAL_FIELD_SPECS:
        opt[spec.arr] = np.asarray(arrs.get(spec.arr, np.zeros((n, *spec.shape), dtype=spec.dtype)))
        opt[spec.flag] = np.asarray(arrs.get(spec.flag, np.zeros((n,), dtype=np.uint8)), dtype=np.uint8)

    out: list[ReplaySample] = []
    for i in range(n):
        s = ReplaySample(
            x=_copy_row(x, i),
            policy_target=_copy_row(policy, i),
            wdl_target=int(wdl[i]),
            priority=float(priority[i]),
            has_policy=bool(has_policy[i]),
        )
        if opt["has_sf_wdl"][i]:
            s.sf_wdl = _copy_row(opt["sf_wdl"], i)
        if opt["has_sf_move"][i]:
            s.sf_move_index = int(opt["sf_move_index"][i])
        if opt["has_sf_policy"][i]:
            s.sf_policy_target = _copy_row(opt["sf_policy_target"], i)
        if opt["has_moves_left"][i]:
            s.moves_left = float(opt["moves_left"][i])
        if opt["has_is_network_turn"][i]:
            s.is_network_turn = bool(opt["is_network_turn"][i])
        if opt["has_is_selfplay"][i]:
            s.is_selfplay = bool(opt["is_selfplay"][i])
        if opt["has_categorical"][i]:
            s.categorical_target = _copy_row(opt["categorical_target"], i)
        if opt["has_policy_soft"][i]:
            s.policy_soft_target = _copy_row(opt["policy_soft_target"], i)
        if opt["has_future"][i]:
            s.future_policy_target = _copy_row(opt["future_policy_target"], i)
            s.has_future = True
        if opt["has_volatility"][i]:
            s.volatility_target = _copy_row(opt["volatility_target"], i)
            s.has_volatility = True
        if opt["has_sf_volatility"][i]:
            s.sf_volatility_target = _copy_row(opt["sf_volatility_target"], i)
            s.has_sf_volatility = True
        for mk, hk in zip(LEGAL_MASK_FIELDS, LEGAL_MASK_HAS_FIELDS, strict=True):
            if opt[hk][i]:
                setattr(s, mk, _copy_row(opt[mk], i, dtype=np.uint8))
        out.append(s)
    return out


def save_npz(
    path: str | Path,
    *,
    samples: list[ReplaySample],
    meta: ShardMeta | dict[str, Any] | None = None,
    compress: bool = True,
) -> Path:
    """Write *samples* as a legacy ``.npz`` shard.

    Used by the bootstrap tooling (``scripts/generate_bootstrap.py`` /
    ``scripts/train_bootstrap.py``) only. The production pipeline writes
    zarr via ``save_local_shard_arrays``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stored = prune_storage_arrays(samples_to_arrays(samples))
    meta_json = json.dumps(_meta_dict(meta, positions=int(np.asarray(stored["x"]).shape[0])), sort_keys=True)
    saver = np.savez_compressed if compress else np.savez
    saver(str(p), **stored, meta_json=np.array(meta_json)) # numpy's savez stubs reject dict[str, ndarray] splat
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
  # Write to a temp path then atomic-rename to avoid races with concurrent
  # readers/writers that can cause "Directory not empty" on rmtree.
  # Prefix matches the ingest-side tmp filter (_is_tmp_shard_name) so a
  # crashed-mid-write tmp dir isn't mistaken for a real shard on resume.
    tmp = p.with_name(f"._tmp_{os.getpid()}_{p.name}")
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


def load_shard_arrays(
    path: str | Path,
    *,
    lazy: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a shard's arrays + meta, dispatching on suffix.

    Handles current ``.zarr`` shards (lazy or eager) and legacy ``.npz`` files
    (always eager). ``.npz`` support exists for the bootstrap pipeline and
    as a defensive read path for archival shards; the production writer is
    ``save_local_shard_arrays``.
    """
    p = Path(path)
    if p.suffix == LEGACY_SHARD_SUFFIX:
        with np.load(str(p), allow_pickle=False) as z:
            arrs = {k: np.array(z[k], copy=False) for k in z.files if k != "meta_json"}
            meta_json = z["meta_json"].item() if "meta_json" in z.files else "{}"
        meta = json.loads(str(meta_json)) if meta_json else {}
        validate_arrays(arrs)
        return arrs, meta
    g = zarr.open_group(str(p), mode="r")
    meta = dict(g.attrs.asdict())
    if lazy:
        arrs = {name: g[name] for name in _SHARD_FIELDS if name in g}
        return arrs, meta
    arrs = {name: np.asarray(g[name]) for name in _SHARD_FIELDS if name in g}
    validate_arrays(arrs)
    return arrs, meta


def load_npz(path: str | Path) -> tuple[list[ReplaySample], dict[str, Any]]:
    """Read a legacy ``.npz`` shard into ``ReplaySample`` objects.

    Used by ``scripts/train_bootstrap.py``; prefer ``load_shard_arrays`` for
    everything else.
    """
    arrs, meta = load_shard_arrays(path)
    return arrays_to_samples(arrs), meta
