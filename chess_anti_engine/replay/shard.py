from __future__ import annotations

import io
import json
import os
import secrets
import shutil
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc

from chess_anti_engine.moves import (
    COMPACT_POLICY_SIZE,
    POLICY_ENCODING_AZ_4672,
    POLICY_ENCODING_LC0_1858,
    POLICY_SIZE,
    normalize_policy_encoding,
    policy_size_for_encoding,
)
from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS

from .buffer import ReplaySample

SHARD_VERSION = 2  # v2: sparse MultiPV label storage (sf_multipv_raw/sf_label_meta)

# Sparse MultiPV raw-label layout. Per SF-labeled sample we keep the exact
# candidate list the targets were built from, so sf_policy_temp /
# label-smoothing / cp->WDL params can be re-tuned offline
# (train/target_builder.py) without a fresh live run.
#   sf_multipv_raw: (SF_MULTIPV_RAW_MAX, 5) int16, one row per legal MultiPV
#     line in rank order, padded with SF_MULTIPV_PAD_ROW:
#       col 0: move policy index in the SHARD's policy encoding (-1 = pad)
#       col 1: raw cp score clamped to +/-32000 (SF_CP_SENTINEL = no cp)
#       col 2: mate distance, clamped to +/-127 (0 = no mate; collides with
#         UCI "score mate 0", which is unreachable here because labels are
#         only requested for positions that still have legal moves)
#       col 3: native SF wdl W permille (-1 = absent)
#       col 4: native SF wdl D permille (-1 = absent; L = 1000 - W - D)
#   sf_label_meta: (6,) int32 for the record-level eval:
#       [nodes (-1 unknown), depth (-1 unknown), eval_cp (SF_CP_SENTINEL),
#        eval_mate (0 = none), eval_wdl_w (-1), eval_wdl_d (-1)]
SF_MULTIPV_RAW_MAX = 48   # production sf_multipv is 40; cap with headroom
SF_MULTIPV_RAW_COLS = 5
SF_LABEL_META_LEN = 6
SF_CP_SENTINEL = -32768
SF_MULTIPV_PAD_ROW = (-1, SF_CP_SENTINEL, 0, -1, -1)
LOCAL_SHARD_SUFFIX = ".zarr"
LEGACY_SHARD_SUFFIX = ".npz"
DEFAULT_MAX_SHARD_POSITIONS = 50_000
DEFAULT_MAX_SHARD_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
INPUT_HISTORY_ENCODING_ARRAY_KEY = "_input_history_encoding"
POLICY_ENCODING_ARRAY_KEY = "_policy_encoding"
# Scalar "true"/"false" marker (string, so the buffer's str()-based scalar
# merge compares it safely). Always materialized by samples_to_arrays and
# load_shard_arrays; a missing shard attr provably means the flag was off.
HISTORY_REP_FIX_ARRAY_KEY = "_history_rep_fix"


def history_rep_fix_from_arrays(arrs: dict[str, Any]) -> bool:
    """Read the history_rep_fix marker from a chunk dict (absent = off)."""
    raw = np.asarray(arrs.get(HISTORY_REP_FIX_ARRAY_KEY, np.asarray("false")))
    return bool(raw.size) and str(raw.reshape(-1)[0]).strip().lower() == "true"

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
_I64_DT: np.dtype = np.dtype(np.int64)

_OPTIONAL_FIELD_SPECS: tuple[_OptFieldSpec, ...] = (
    # x_lc0_root's stored plane count follows x (146 v1 / 175 v2_threats);
    # zeros_for_storage_field special-cases it on x_planes. The spec shape is
    # the v1 fallback for paths that don't know the runtime plane count.
    _OptFieldSpec("x_lc0_root",           "has_x_lc0_root",        (146, 8, 8),   _F16),
    # Dynamic board-relation matrices (model.use_dynamic_relations); binary.
    _OptFieldSpec("relations",            "has_relations",         (5, 64, 64),   _U8_DT),
    _OptFieldSpec("priority_policy_kl",   "has_priority_policy_kl",(),            _F16),
    _OptFieldSpec("priority_q_delta",     "has_priority_q_delta",  (),            _F16),
    _OptFieldSpec("priority_sf_search_gap","has_priority_sf_search_gap", (),       _F16),
    _OptFieldSpec("game_id",              "has_game_id",           (),            _I64_DT),
    _OptFieldSpec("ply_index",            "has_ply_index",         (),            _I32_DT),
    _OptFieldSpec("sf_wdl",               "has_sf_wdl",            (3,),          _F16),
    _OptFieldSpec("sf_move_index",        "has_sf_move",           (),            _I32_DT),
    _OptFieldSpec("sf_played_move_index", "has_sf_played_move",    (),            _I32_DT),
    _OptFieldSpec("sf_played_rank",       "has_sf_played_rank",    (),            _I32_DT),
    _OptFieldSpec("sf_played_regret",     "has_sf_played_regret",  (),            _F16),
    _OptFieldSpec("future_sf_regret_sum", "has_future_sf_regret_sum", (),         _F16),
    _OptFieldSpec("future_sf_regret_d95", "has_future_sf_regret_d95", (),         _F16),
    _OptFieldSpec("future_sf_regret_d98", "has_future_sf_regret_d98", (),         _F16),
    _OptFieldSpec("future_sf_regret_max", "has_future_sf_regret_max", (),         _F16),
    _OptFieldSpec("future_sf_regret_h4",  "has_future_sf_regret_h4",  (),         _F16),
    _OptFieldSpec("future_sf_regret_h6",  "has_future_sf_regret_h6",  (),         _F16),
    _OptFieldSpec("future_sf_regret_h12", "has_future_sf_regret_h12", (),         _F16),
    _OptFieldSpec("future_sf_regret_h24", "has_future_sf_regret_h24", (),         _F16),
    _OptFieldSpec("future_sf_regret_h50", "has_future_sf_regret_h50", (),         _F16),
    _OptFieldSpec("future_sf_regret_count", "has_future_sf_regret_count", (),     _I32_DT),
    _OptFieldSpec("sf_policy_target",     "has_sf_policy",         _POLICY_SHAPE, _F16),
    _OptFieldSpec("sf_multipv_raw",       "has_sf_multipv_raw",
                  (SF_MULTIPV_RAW_MAX, SF_MULTIPV_RAW_COLS), np.dtype(np.int16)),
    _OptFieldSpec("sf_label_meta",        "has_sf_label_meta",     (SF_LABEL_META_LEN,), _I32_DT),
    _OptFieldSpec("moves_left",           "has_moves_left",        (),            _F16),
    _OptFieldSpec("is_network_turn",      "has_is_network_turn",   (),            _U8_DT),
    _OptFieldSpec("is_selfplay",          "has_is_selfplay",       (),            _U8_DT),
    _OptFieldSpec("categorical_target",   "has_categorical",       (DEFAULT_CATEGORICAL_BINS,), _F16),
    _OptFieldSpec("policy_soft_target",   "has_policy_soft",       _POLICY_SHAPE, _F16),
    _OptFieldSpec("future_policy_target", "has_future",            _POLICY_SHAPE, _F16),
    _OptFieldSpec("volatility_target",    "has_volatility",        (3,),          _F16),
    _OptFieldSpec("sf_volatility_target", "has_sf_volatility",     (3,),          _F16),
    _OptFieldSpec("search_wdl",           "has_search_wdl",        (3,),          _F16),
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
_OPTIONAL_DISTRIBUTION_FIELDS = frozenset({
    "sf_wdl",
    "sf_policy_target",
    "categorical_target",
    "policy_soft_target",
    "future_policy_target",
    "search_wdl",
})

_SHARD_FIELDS: tuple[str, ...] = (
    *_REQUIRED_STORAGE_FIELDS,
    *(name for s in _OPTIONAL_FIELD_SPECS for name in (s.arr, s.flag)),
)


def zeros_for_storage_field(
    name: str,
    *,
    n: int,
    x_planes: int,
    policy_size: int = POLICY_SIZE,
    categorical_bins: int = DEFAULT_CATEGORICAL_BINS,
) -> np.ndarray:
    """Default array for a missing stored replay field.

    Required fields have explicit defaults because their shapes can depend on
    runtime policy/input dimensions. Optional fields are driven by
    ``_OPTIONAL_FIELD_SPECS`` so mixed-schema shard concatenation stays in sync
    with validation and serialization.
    """
    if name in ("x", "x_lc0_root"):
        return np.zeros((n, x_planes, 8, 8), dtype=np.float16)
    if name == "policy_target":
        return np.zeros((n, policy_size), dtype=np.float16)
    if name == "wdl_target":
        return np.zeros((n,), dtype=np.int8)
    if name == "priority":
        return np.ones((n,), dtype=np.float32)
    if name == "has_policy":
        return np.ones((n,), dtype=np.uint8)
    for spec in _OPTIONAL_FIELD_SPECS:
        if name == spec.flag:
            return np.zeros((n,), dtype=np.uint8)
        if name == spec.arr:
            shape = (categorical_bins,) if name == "categorical_target" else spec.shape
            if name in POLICY_SIZED_FIELDS:
                shape = (policy_size,)
            return np.zeros((n, *shape), dtype=spec.dtype)
    raise KeyError(f"unknown replay field {name!r}")

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
POLICY_SIZED_FIELDS = frozenset((*POLICY_SPACE_FIELDS, *LEGAL_MASK_FIELDS))
POLICY_INDEX_FIELDS: tuple[tuple[str, str], ...] = (
    ("sf_move_index", "has_sf_move"),
    ("sf_played_move_index", "has_sf_played_move"),
)


def _policy_encoding_for_size(policy_size: int) -> str:
    size = int(policy_size)
    if size == int(COMPACT_POLICY_SIZE):
        return POLICY_ENCODING_LC0_1858
    if size == int(POLICY_SIZE):
        return POLICY_ENCODING_AZ_4672
    raise ValueError(
        f"policy size must be {POLICY_SIZE} or {COMPACT_POLICY_SIZE}, got {size}",
    )


def _scalar_metadata_string(arrs: dict[str, Any], key: str) -> str | None:
    if key not in arrs:
        return None
    arr = np.asarray(arrs[key])
    if arr.ndim > 1:
        raise ValueError(f"{key} must be scalar or (N,)")
    values = [str(v) for v in arr.reshape(-1).tolist() if str(v)]
    if not values:
        return None
    first = values[0]
    if any(value != first for value in values[1:]):
        labels = sorted(set(values))
        raise ValueError(f"mixed replay metadata {key}: {labels}")
    return first


def _policy_metadata_from_arrays(arrs: dict[str, Any]) -> tuple[str, int]:
    policy_shape = _shape_of(arrs["policy_target"])
    if len(policy_shape) != 2:
        raise ValueError(f"policy_target must be (N,A); got {policy_shape}")
    policy_size = int(policy_shape[1])
    declared = _scalar_metadata_string(arrs, POLICY_ENCODING_ARRAY_KEY)
    policy_encoding = (
        normalize_policy_encoding(declared)
        if declared is not None
        else _policy_encoding_for_size(policy_size)
    )
    expected_size = int(policy_size_for_encoding(policy_encoding))
    if expected_size != policy_size:
        raise ValueError(
            f"{POLICY_ENCODING_ARRAY_KEY}={policy_encoding!r} expects policy width "
            f"{expected_size}, got {policy_size}",
        )
    return policy_encoding, policy_size


def _attach_identity_meta_arrays(arrs: dict[str, Any], meta: dict[str, Any]) -> None:
    """Materialize shard-attr encoding-identity fields as scalar chunk arrays.

    ``history_rep_fix`` is materialized unconditionally (a missing attr
    provably means off), so every loaded chunk carries the marker and the
    replay buffer's scalar-metadata merge can hard-fail on mixed encodings.
    """
    if meta.get("input_history_encoding") is not None:
        arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY] = np.asarray(str(meta["input_history_encoding"]))
    if meta.get("policy_encoding") is not None:
        arrs[POLICY_ENCODING_ARRAY_KEY] = np.asarray(str(meta["policy_encoding"]))
    arrs[HISTORY_REP_FIX_ARRAY_KEY] = np.asarray(
        "true" if bool(meta.get("history_rep_fix") or False) else "false"
    )


def _attach_policy_metadata(arrs: dict[str, Any], meta: dict[str, Any]) -> None:
    policy_encoding, policy_size = _policy_metadata_from_arrays(arrs)
    declared = meta.get("policy_encoding")
    if declared is not None:
        meta_encoding = normalize_policy_encoding(str(declared))
        if meta_encoding != policy_encoding:
            raise ValueError(
                f"policy_encoding mismatch: metadata has {meta_encoding!r}, "
                f"arrays have {policy_encoding!r}",
            )
        policy_encoding = meta_encoding
    if "policy_size" in meta and int(meta["policy_size"]) != policy_size:
        raise ValueError(
            f"policy_size mismatch: metadata has {int(meta['policy_size'])}, "
            f"policy_target has {policy_size}",
        )
    meta["policy_encoding"] = policy_encoding
    meta["policy_size"] = policy_size
    arrs[POLICY_ENCODING_ARRAY_KEY] = np.asarray(policy_encoding)
    arrs["_policy_size"] = np.array(policy_size, dtype=np.int32)


def _meta_with_policy(meta: ShardMeta | dict[str, Any] | None, *, arrs: dict[str, Any]) -> dict[str, Any]:
    attrs = _meta_dict(meta, positions=int(_shape_of(arrs["x"])[0]))
    policy_encoding, policy_size = _policy_metadata_from_arrays(arrs)
    if attrs.get("policy_encoding") is not None:
        policy_encoding = normalize_policy_encoding(str(attrs["policy_encoding"]))
        expected_size = int(policy_size_for_encoding(policy_encoding))
        if expected_size != policy_size:
            raise ValueError(
                f"policy_encoding {policy_encoding!r} expects policy width "
                f"{expected_size}, got {policy_size}",
            )
    if attrs.get("policy_size") is not None and int(attrs["policy_size"]) != policy_size:
        raise ValueError(
            f"policy_size {int(attrs['policy_size'])} does not match policy width {policy_size}",
        )
    attrs["policy_encoding"] = policy_encoding
    attrs["policy_size"] = policy_size
    return attrs


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
    if (
        "policy_target" in out
        and np.asarray(out["policy_target"]).ndim == 2
        and ("_policy_size" not in out or POLICY_ENCODING_ARRAY_KEY not in out)
    ):
        policy_encoding, policy_size = _policy_metadata_from_arrays(out)
        out["_policy_size"] = np.array(policy_size, dtype=np.int32)
        out[POLICY_ENCODING_ARRAY_KEY] = np.asarray(policy_encoding)
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
    input_history_encoding: str | None = None
    # Whether the gated repetition-plane fix was active during encoding.
    # Absent in shards from before the field existed, which provably means
    # off — readers should treat None as False.
    history_rep_fix: bool | None = None
    policy_encoding: str | None = None
    policy_size: int | None = None
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
    sf_d6_sum: float | None = None
    sf_d6_n: int | None = None
    diff_focus_records: int | None = None
    diff_focus_kept: int | None = None
    diff_focus_keep_prob_sum: float | None = None
    diff_focus_keep_limited: int | None = None
    diff_focus_sample_weight_sum: float | None = None
    diff_focus_sample_weight_limited: int | None = None
    diff_focus_priority_sum: float | None = None
    diff_focus_priority_sq_sum: float | None = None
    diff_focus_priority_min: float | None = None
    diff_focus_priority_max: float | None = None
    gumbel_policy_diag_n: int | None = None
    gumbel_policy_top_prob_sum: float | None = None
    gumbel_policy_action_prob_sum: float | None = None
    gumbel_policy_entropy_sum: float | None = None
    gumbel_policy_eff_moves_sum: float | None = None
    gumbel_policy_candidate_mass_sum: float | None = None
    gumbel_policy_non_candidate_top_prob_sum: float | None = None
    gumbel_policy_argmax_is_candidate_sum: int | None = None
    gumbel_policy_argmax_is_action_sum: int | None = None
    gumbel_policy_legal_count_sum: int | None = None
    gumbel_policy_candidate_count_sum: int | None = None
    outcome_stats: dict[str, int] | None = None


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
    if INPUT_HISTORY_ENCODING_ARRAY_KEY in arrs:
        out[INPUT_HISTORY_ENCODING_ARRAY_KEY] = np.asarray(arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY])
    # The rep-fix marker is part of the same encoding identity as the history
    # encoding above; preserve it too, else pruned shards reload as
    # history_rep_fix=False and silently mix fixed/unfixed repetition planes
    # across a training window (the mixed-value guard in samples_to_arrays only
    # fires when SOME rows still carry the marker).
    if HISTORY_REP_FIX_ARRAY_KEY in arrs:
        out[HISTORY_REP_FIX_ARRAY_KEY] = np.asarray(arrs[HISTORY_REP_FIX_ARRAY_KEY])
    policy_encoding, policy_size = _policy_metadata_from_arrays(out)
    out[POLICY_ENCODING_ARRAY_KEY] = np.asarray(policy_encoding)
    out["_policy_size"] = np.array(policy_size, dtype=np.int32)
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


def extract_uploaded_shard_tar(
    tar_path: str | Path,
    dest: str | Path,
    *,
    max_extract_bytes: int | None = None,
) -> Path:
    """Safely extract a worker-uploaded zarr tarball at *tar_path* into *dest*.

    *dest* must not already exist; it is created by this function. Raises
    ``ValueError`` on any tar member that would escape the extract dir — links
    (sym or hard), absolute paths, ``..`` traversal, non-regular files, or
    resolved paths outside *dest*. On success, returns the zarr group root
    (either *dest* itself or a single nested child dir containing ``.zgroup``).

    Defense in depth: the manual member walk rejects link-based escape attacks
    and oversized declared payloads before any bytes touch the filesystem;
    ``tarfile.extractall(filter="data")`` strips mode/uid/gid bits and catches
    anything the walk missed.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    dest_resolved = dest.resolve()
    with tarfile.open(str(tar_path), mode="r:") as tf:
        declared_bytes = 0
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
            if member.isreg():
                declared_bytes += max(0, int(member.size))
                if max_extract_bytes is not None and declared_bytes > int(max_extract_bytes):
                    raise ValueError(
                        f"tar declared payload too large: {declared_bytes} > {int(max_extract_bytes)} bytes"
                    )
        try:
            tf.extractall(str(dest), filter="data")
        except TypeError:
            # Python 3.10.12 and newer patched tarfile support ``filter``.
            # Some supported 3.11 patch releases may not; the explicit member
            # prewalk above already rejects links, device nodes, and traversal.
            tf.extractall(str(dest))
    entries = list(dest.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and (entries[0] / ".zgroup").exists():
        return entries[0]
    return dest


  # Each entry: (sample_attr, target_arr, has_arr, scalar_caster_or_None_for_array_asarray).
  # ``None`` ⇒ generic ``np.asarray(v, dtype=spec.dtype)`` — used for the float16
  # vector heads. Custom callables handle scalar packing (int/bool/half).
_SCALAR_FIELDS: tuple[tuple[str, str, str, "object"], ...] = (
    (
        "priority_policy_kl", "priority_policy_kl", "has_priority_policy_kl",
        lambda v: np.float16(float(v)),
    ),
    (
        "priority_q_delta", "priority_q_delta", "has_priority_q_delta",
        lambda v: np.float16(float(v)),
    ),
    (
        "priority_sf_search_gap", "priority_sf_search_gap", "has_priority_sf_search_gap",
        lambda v: np.float16(float(v)),
    ),
    ("game_id",           "game_id",            "has_game_id",           int),
    ("ply_index",         "ply_index",          "has_ply_index",         int),
    ("sf_move_index",     "sf_move_index",  "has_sf_move",          int),
    ("sf_played_move_index", "sf_played_move_index", "has_sf_played_move", int),
    ("sf_played_rank",    "sf_played_rank",     "has_sf_played_rank",    int),
    (
        "sf_played_regret", "sf_played_regret", "has_sf_played_regret",
        lambda v: np.float16(float(v)),
    ),
    (
        "future_sf_regret_sum", "future_sf_regret_sum", "has_future_sf_regret_sum",
        lambda v: np.float16(float(v)),
    ),
    (
        "future_sf_regret_d95", "future_sf_regret_d95", "has_future_sf_regret_d95",
        lambda v: np.float16(float(v)),
    ),
    (
        "future_sf_regret_d98", "future_sf_regret_d98", "has_future_sf_regret_d98",
        lambda v: np.float16(float(v)),
    ),
    (
        "future_sf_regret_max", "future_sf_regret_max", "has_future_sf_regret_max",
        lambda v: np.float16(float(v)),
    ),
    (
        "future_sf_regret_h4", "future_sf_regret_h4", "has_future_sf_regret_h4",
        lambda v: np.float16(float(v)),
    ),
    (
        "future_sf_regret_h6", "future_sf_regret_h6", "has_future_sf_regret_h6",
        lambda v: np.float16(float(v)),
    ),
    (
        "future_sf_regret_h12", "future_sf_regret_h12", "has_future_sf_regret_h12",
        lambda v: np.float16(float(v)),
    ),
    (
        "future_sf_regret_h24", "future_sf_regret_h24", "has_future_sf_regret_h24",
        lambda v: np.float16(float(v)),
    ),
    (
        "future_sf_regret_h50", "future_sf_regret_h50", "has_future_sf_regret_h50",
        lambda v: np.float16(float(v)),
    ),
    ("future_sf_regret_count", "future_sf_regret_count", "has_future_sf_regret_count", int),
    ("moves_left",        "moves_left",     "has_moves_left",       lambda v: np.float16(float(v))),
    ("is_network_turn",   "is_network_turn","has_is_network_turn",  lambda v: 1 if bool(v) else 0),
    ("is_selfplay",       "is_selfplay",    "has_is_selfplay",      lambda v: 1 if bool(v) else 0),
)
_VECTOR_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("x_lc0_root",           "x_lc0_root",           "has_x_lc0_root"),
    ("relations",            "relations",            "has_relations"),
    ("sf_wdl",               "sf_wdl",               "has_sf_wdl"),
    ("sf_policy_target",     "sf_policy_target",     "has_sf_policy"),
    ("categorical_target",   "categorical_target",   "has_categorical"),
    ("policy_soft_target",   "policy_soft_target",   "has_policy_soft"),
    ("future_policy_target", "future_policy_target", "has_future"),
    ("volatility_target",    "volatility_target",    "has_volatility"),
    ("sf_volatility_target", "sf_volatility_target", "has_sf_volatility"),
    ("search_wdl",           "search_wdl",           "has_search_wdl"),
)
# Integer-valued vector fields keep their exact dtype (the generic
# _VECTOR_FIELDS loop casts through float16, which corrupts ints > 2048).
_INT_VECTOR_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("sf_multipv_raw", "sf_multipv_raw", "has_sf_multipv_raw"),
    ("sf_label_meta",  "sf_label_meta",  "has_sf_label_meta"),
)

_VECTOR_EXPLICIT_HAS_ATTRS: dict[str, str] = {
    "future_policy_target": "has_future",
    "volatility_target": "has_volatility",
    "sf_volatility_target": "has_sf_volatility",
}


def _sample_has_vector_field(s: ReplaySample, src: str) -> bool:
    explicit = _VECTOR_EXPLICIT_HAS_ATTRS.get(src)
    if explicit is not None:
        flag = getattr(s, explicit, None)
        if flag is not None:
            return bool(flag)
    return getattr(s, src, None) is not None


def samples_to_arrays(samples: list[ReplaySample]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("cannot serialize empty shard")
    n = len(samples)
    policy_size = int(np.asarray(samples[0].policy_target).shape[0])

    arrs: dict[str, np.ndarray] = {
        "x": _f16(np.stack([s.x for s in samples], axis=0)),
        "policy_target": _f16(np.stack([s.policy_target for s in samples], axis=0)),
        "wdl_target": np.array([int(s.wdl_target) for s in samples], dtype=np.int8),
        "priority": np.array([float(getattr(s, "priority", 1.0)) for s in samples], dtype=np.float32),
        "has_policy": _u8(np.array(
            [1 if getattr(s, "has_policy", True) else 0 for s in samples], dtype=np.uint8,
        )),
    }
    history_values = [
        str(v)
        for s in samples
        if (v := getattr(s, "input_history_encoding", None)) is not None and str(v)
    ]
    if history_values:
        first_history = history_values[0]
        if len(history_values) != n or any(v != first_history for v in history_values[1:]):
            raise ValueError("mixed ReplaySample input_history_encoding values")
        arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY] = np.asarray(first_history)
    # history_rep_fix is replay identity (same encoding name, different
    # planes). Always materialized — absent on a sample provably means off —
    # so the buffer's scalar-metadata merge can hard-fail on mixed chunks.
    rep_fix_values = {bool(getattr(s, "history_rep_fix", False)) for s in samples}
    if len(rep_fix_values) > 1:
        raise ValueError("mixed ReplaySample history_rep_fix values")
    arrs[HISTORY_REP_FIX_ARRAY_KEY] = np.asarray(
        "true" if (rep_fix_values and rep_fix_values.pop()) else "false"
    )
    x_planes = int(arrs["x"].shape[1])
    for spec in _OPTIONAL_FIELD_SPECS:
        if spec.arr == "x_lc0_root":
  # Alternate-input planes follow x's width (146 v1 / 175 v2_threats).
            shape: tuple[int, ...] = (x_planes, 8, 8)
        elif spec.arr in POLICY_SIZED_FIELDS:
            shape = (policy_size,)
        else:
            shape = spec.shape
        arrs[spec.arr] = np.zeros((n, *shape), dtype=spec.dtype)
        arrs[spec.flag] = np.zeros((n,), dtype=np.uint8)

    for i, s in enumerate(samples):
        for src, target, has, cast in _SCALAR_FIELDS:
            v = getattr(s, src, None)
            if v is not None:
                arrs[target][i] = cast(v)  # pyright: ignore[reportCallIssue]
                arrs[has][i] = 1
        for src, target, has in _VECTOR_FIELDS:
            v = getattr(s, src, None)
            if v is not None and _sample_has_vector_field(s, src):
                arrs[target][i] = np.asarray(v, dtype=np.float16)
                arrs[has][i] = 1
        for src, target, has in _INT_VECTOR_FIELDS:
            v = getattr(s, src, None)
            if v is not None:
                arrs[target][i] = np.asarray(v, dtype=arrs[target].dtype)
                arrs[has][i] = 1
        for mk, hk in zip(LEGAL_MASK_FIELDS, LEGAL_MASK_HAS_FIELDS, strict=True):
            v = getattr(s, mk, None)
            if v is not None:
                arrs[mk][i] = np.asarray(v, dtype=np.uint8)
                arrs[hk][i] = 1

    return arrs


def _shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.asarray(value).shape
    return tuple(int(dim) for dim in shape)


def _dtype_of(value: Any) -> np.dtype:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = np.asarray(value).dtype
    return np.dtype(dtype)


def _declared_nbytes(value: Any) -> int:
    nbytes = int(_dtype_of(value).itemsize)
    for dim in _shape_of(value):
        nbytes *= int(dim)
    return int(nbytes)


def validate_array_declarations(
    arrs: dict[str, Any],
    *,
    max_positions: int | None = None,
    max_uncompressed_bytes: int | None = None,
) -> None:
    if "x" not in arrs or "policy_target" not in arrs or "wdl_target" not in arrs:
        raise ValueError("shard missing required fields")

    x_shape = _shape_of(arrs["x"])
    policy_shape = _shape_of(arrs["policy_target"])
    wdl_shape = _shape_of(arrs["wdl_target"])

    if len(x_shape) != 4:
        raise ValueError(f"x must be (N,C,8,8); got {x_shape}")
    if x_shape[-2:] != (8, 8):
        raise ValueError(f"x must end with (8,8); got {x_shape}")
    if len(policy_shape) != 2:
        raise ValueError(f"policy_target must be (N,A); got {policy_shape}")
    if policy_shape[0] != x_shape[0]:
        raise ValueError("policy_target N mismatch")
    policy_size = int(policy_shape[1])
    if policy_size not in (int(POLICY_SIZE), int(COMPACT_POLICY_SIZE)):
        raise ValueError(
            f"policy_target A mismatch: expected {POLICY_SIZE} or {COMPACT_POLICY_SIZE}, "
            f"got {policy_shape[1]}",
        )
    _policy_metadata_from_arrays(arrs)
    if len(wdl_shape) != 1 or wdl_shape[0] != x_shape[0]:
        raise ValueError("wdl_target must be (N,) matching x")
    if "_policy_size" in arrs:
        declared_policy_sizes = np.asarray(arrs["_policy_size"])
        if np.any(declared_policy_sizes != policy_size):
            min_declared = int(np.min(declared_policy_sizes))
            max_declared = int(np.max(declared_policy_sizes))
            raise ValueError(
                f"_policy_size mismatch: attr range [{min_declared}, {max_declared}], "
                f"policy_target has {policy_size}",
            )
    n = int(x_shape[0])
    if max_positions is not None and int(max_positions) > 0 and n > int(max_positions):
        raise ValueError(f"shard has too many positions: {n} > {int(max_positions)}")
    if max_uncompressed_bytes is not None and int(max_uncompressed_bytes) > 0:
        total_bytes = sum(
            _declared_nbytes(value)
            for key, value in arrs.items()
            if not str(key).startswith("_")
        )
        if total_bytes > int(max_uncompressed_bytes):
            raise ValueError(
                f"shard declared uncompressed arrays too large: "
                f"{total_bytes} > {int(max_uncompressed_bytes)} bytes",
            )

    for spec in _OPTIONAL_FIELD_SPECS:
        if spec.flag in arrs and _shape_of(arrs[spec.flag]) != (n,):
            raise ValueError(f"{spec.flag} must be (N,) matching x")
        if spec.arr in arrs:
            if spec.arr == "x_lc0_root":
                expected_tail: tuple[int, ...] = (int(x_shape[1]), 8, 8)
            elif spec.arr in POLICY_SIZED_FIELDS:
                expected_tail = (policy_size,)
            else:
                expected_tail = spec.shape
            expected_shape = (n, *expected_tail)
            value_shape = _shape_of(arrs[spec.arr])
            if value_shape != expected_shape:
                raise ValueError(
                    f"{spec.arr} shape mismatch: expected {expected_shape}, got {value_shape}",
                )


def validate_arrays(arrs: dict[str, np.ndarray]) -> None:
    validate_array_declarations(arrs)

    x = np.asarray(arrs["x"])
    policy = np.asarray(arrs["policy_target"])
    wdl = np.asarray(arrs["wdl_target"])

    policy_size = int(policy.shape[1])
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

    n = int(x.shape[0])
    for value_name, flag_name in POLICY_INDEX_FIELDS:
        if value_name not in arrs:
            continue
        idx = np.asarray(arrs[value_name]).astype(np.int64, copy=False)
        if idx.shape != (n,):
            raise ValueError(f"{value_name} must be (N,) matching x")
        if flag_name in arrs:
            active_rows = np.asarray(arrs[flag_name]) != 0
        else:
            active_rows = np.ones((n,), dtype=np.bool_)
        if np.any(active_rows):
            active_idx = idx[active_rows]
            if ((active_idx < 0) | (active_idx >= policy_size)).any():
                raise ValueError(
                    f"{value_name} active rows out of range for policy width {policy_size}",
                )
    for spec in _OPTIONAL_FIELD_SPECS:
        flag_present = spec.flag in arrs
        value_present = spec.arr in arrs
        active = False

        if flag_present:
            flag = np.asarray(arrs[spec.flag])
            if flag.ndim != 1 or flag.shape[0] != n:
                raise ValueError(f"{spec.flag} must be (N,) matching x")
            active = bool(np.any(flag != 0))

        if value_present:
            value = np.asarray(arrs[spec.arr])
            if spec.arr == "x_lc0_root":
                expected_tail: tuple[int, ...] = (int(x.shape[1]), 8, 8)
            elif spec.arr in POLICY_SIZED_FIELDS:
                expected_tail = (policy_size,)
            else:
                expected_tail = spec.shape
            expected_shape = (n, *expected_tail)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{spec.arr} shape mismatch: expected {expected_shape}, got {value.shape}",
                )
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"{spec.arr} contains NaN/Inf")
            if spec.arr in _OPTIONAL_DISTRIBUTION_FIELDS and flag_present:
                active_rows = np.asarray(arrs[spec.flag]) != 0
                if np.any(active_rows):
                    active_value = value[active_rows]
                    if (active_value < -1e-6).any():
                        raise ValueError(f"{spec.arr} active rows contain negative values")
                    row_sums = active_value.sum(axis=tuple(range(1, active_value.ndim)), dtype=np.float32)
                    if (row_sums <= 0).any():
                        raise ValueError(f"{spec.arr} active rows have non-positive sum")

        if active and not value_present:
            raise ValueError(f"{spec.flag} is set but {spec.arr} is missing")


def arrays_to_samples(arrs: dict[str, np.ndarray]) -> list[ReplaySample]:
    validate_arrays(arrs)

    x = np.asarray(arrs["x"])
    policy = np.asarray(arrs["policy_target"])
    wdl = np.asarray(arrs["wdl_target"]).astype(np.int64, copy=False)
    n = int(x.shape[0])

    priority = np.asarray(arrs.get("priority", np.ones((n,), dtype=np.float32)), dtype=np.float32)
    has_policy = np.asarray(arrs.get("has_policy", np.ones((n,), dtype=np.uint8)), dtype=np.uint8)
    input_history = np.asarray(arrs.get(INPUT_HISTORY_ENCODING_ARRAY_KEY, np.asarray("")))
    if input_history.ndim not in (0, 1):
        raise ValueError(f"{INPUT_HISTORY_ENCODING_ARRAY_KEY} must be scalar or (N,)")
    if input_history.ndim == 1 and input_history.shape != (n,):
        raise ValueError(f"{INPUT_HISTORY_ENCODING_ARRAY_KEY} must be scalar or (N,)")
    rep_fix_flag = history_rep_fix_from_arrays(arrs)

    opt: dict[str, np.ndarray] = {}
    for spec in _OPTIONAL_FIELD_SPECS:
        shape = (int(policy.shape[1]),) if spec.arr in POLICY_SIZED_FIELDS else spec.shape
        opt[spec.arr] = np.asarray(arrs.get(spec.arr, np.zeros((n, *shape), dtype=spec.dtype)))
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
        hist_value = input_history.item() if input_history.ndim == 0 else input_history[i]
        if str(hist_value):
            s.input_history_encoding = str(hist_value)
        s.history_rep_fix = rep_fix_flag
        if opt["has_x_lc0_root"][i]:
            s.x_lc0_root = _copy_row(opt["x_lc0_root"], i)
        if opt["has_relations"][i]:
            s.relations = _copy_row(opt["relations"], i)
        if opt["has_priority_policy_kl"][i]:
            s.priority_policy_kl = float(opt["priority_policy_kl"][i])
        if opt["has_priority_q_delta"][i]:
            s.priority_q_delta = float(opt["priority_q_delta"][i])
        if opt["has_priority_sf_search_gap"][i]:
            s.priority_sf_search_gap = float(opt["priority_sf_search_gap"][i])
        if opt["has_game_id"][i]:
            s.game_id = int(opt["game_id"][i])
        if opt["has_ply_index"][i]:
            s.ply_index = int(opt["ply_index"][i])
        if opt["has_sf_wdl"][i]:
            s.sf_wdl = _copy_row(opt["sf_wdl"], i)
        if opt["has_sf_multipv_raw"][i]:
            s.sf_multipv_raw = _copy_row(opt["sf_multipv_raw"], i)
        if opt["has_sf_label_meta"][i]:
            s.sf_label_meta = _copy_row(opt["sf_label_meta"], i)
        if opt["has_sf_move"][i]:
            s.sf_move_index = int(opt["sf_move_index"][i])
        if opt["has_sf_played_move"][i]:
            s.sf_played_move_index = int(opt["sf_played_move_index"][i])
        if opt["has_sf_played_rank"][i]:
            s.sf_played_rank = int(opt["sf_played_rank"][i])
        if opt["has_sf_played_regret"][i]:
            s.sf_played_regret = float(opt["sf_played_regret"][i])
        if opt["has_future_sf_regret_sum"][i]:
            s.future_sf_regret_sum = float(opt["future_sf_regret_sum"][i])
        if opt["has_future_sf_regret_d95"][i]:
            s.future_sf_regret_d95 = float(opt["future_sf_regret_d95"][i])
        if opt["has_future_sf_regret_d98"][i]:
            s.future_sf_regret_d98 = float(opt["future_sf_regret_d98"][i])
        if opt["has_future_sf_regret_max"][i]:
            s.future_sf_regret_max = float(opt["future_sf_regret_max"][i])
        if opt["has_future_sf_regret_h4"][i]:
            s.future_sf_regret_h4 = float(opt["future_sf_regret_h4"][i])
        if opt["has_future_sf_regret_h6"][i]:
            s.future_sf_regret_h6 = float(opt["future_sf_regret_h6"][i])
        if opt["has_future_sf_regret_h12"][i]:
            s.future_sf_regret_h12 = float(opt["future_sf_regret_h12"][i])
        if opt["has_future_sf_regret_h24"][i]:
            s.future_sf_regret_h24 = float(opt["future_sf_regret_h24"][i])
        if opt["has_future_sf_regret_h50"][i]:
            s.future_sf_regret_h50 = float(opt["future_sf_regret_h50"][i])
        if opt["has_future_sf_regret_count"][i]:
            s.future_sf_regret_count = int(opt["future_sf_regret_count"][i])
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
        if opt["has_search_wdl"][i]:
            s.search_wdl = _copy_row(opt["search_wdl"], i)
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
    meta_json = json.dumps(_meta_with_policy(meta, arrs=stored), sort_keys=True)
    saver: Callable[..., Any] = np.savez_compressed if compress else np.savez
    saver(str(p), **stored, meta_json=np.array(meta_json))
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
    tmp = p.with_name(f"._tmp_{os.getpid()}_{secrets.token_hex(8)}_{p.name}")
    try:
        g = zarr.open_group(str(tmp), mode="w")
        attrs = _meta_with_policy(meta, arrs=stored)
        if (
            attrs.get("input_history_encoding") is None
            and INPUT_HISTORY_ENCODING_ARRAY_KEY in arrs
        ):
            hist_arr = np.asarray(arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY])
            if hist_arr.size:
                attrs["input_history_encoding"] = str(hist_arr.reshape(-1)[0])
        if attrs.get("history_rep_fix") is None and HISTORY_REP_FIX_ARRAY_KEY in arrs:
            # Buffer-written window shards carry the marker as a chunk array;
            # persist it as the attr so reloads rematerialize it.
            attrs["history_rep_fix"] = history_rep_fix_from_arrays(arrs)
        g.attrs.update(attrs)
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
        for name, value in stored.items():
            if str(name).startswith("_"):
                continue
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
        _attach_identity_meta_arrays(arrs, meta)
        _attach_policy_metadata(arrs, meta)
        validate_arrays(arrs)
        return arrs, meta
    g = zarr.open_group(str(p), mode="r")
    meta = dict(g.attrs.asdict())
    if lazy:
        arrs: dict[str, Any] = {name: g[name] for name in _SHARD_FIELDS if name in g}
        _attach_identity_meta_arrays(arrs, meta)
        _attach_policy_metadata(arrs, meta)
        validate_array_declarations(arrs)
        return arrs, meta
    arrs = {name: np.asarray(g[name]) for name in _SHARD_FIELDS if name in g}
    _attach_identity_meta_arrays(arrs, meta)
    _attach_policy_metadata(arrs, meta)
    validate_arrays(arrs)
    return arrs, meta


def load_npz(path: str | Path) -> tuple[list[ReplaySample], dict[str, Any]]:
    """Read a legacy ``.npz`` shard into ``ReplaySample`` objects.

    Used by ``scripts/train_bootstrap.py``; prefer ``load_shard_arrays`` for
    everything else.
    """
    arrs, meta = load_shard_arrays(path)
    return arrays_to_samples(arrs), meta
