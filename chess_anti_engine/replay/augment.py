from __future__ import annotations

import numpy as np

from chess_anti_engine.encoding.lc0 import uses_lc0_root_history
from chess_anti_engine.moves.encode import (
    COMPACT_MIRROR_POLICY_MAP,
    COMPACT_POLICY_SIZE,
    MIRROR_POLICY_MAP,
    POLICY_SIZE,
    mirror_policy,
    mirror_policy_batch,
    mirror_policy_index,
)

from .buffer import ReplaySample
from .shard import LEGAL_MASK_FIELDS, POLICY_SPACE_FIELDS

# Fields whose rows live in POLICY_SIZE move space and must be remapped under
# the mirror permutation. Includes all policy-like targets and legal masks.
_MIRROR_POLICY_FIELDS = (*POLICY_SPACE_FIELDS, *LEGAL_MASK_FIELDS)
_LEGACY_CASTLING_SWAP_PAIRS = ((96, 97), (98, 99))
_LC0_ROOT_CASTLING_SWAP_PAIRS = ((104, 105), (106, 107))


def _castling_swap_pairs(input_history_encoding: str | None) -> tuple[tuple[int, int], ...]:
    return _LC0_ROOT_CASTLING_SWAP_PAIRS if uses_lc0_root_history(input_history_encoding) else _LEGACY_CASTLING_SWAP_PAIRS


def _swap_constant_planes(x: np.ndarray, pairs: tuple[tuple[int, int], ...]) -> None:
    for a, b in pairs:
        if x.shape[-3] <= max(a, b):
            continue
        tmp = x[..., a, :, :].copy()
        x[..., a, :, :] = x[..., b, :, :]
        x[..., b, :, :] = tmp


def _mirror_map_for_policy_width(width: int) -> np.ndarray:
    if int(width) == int(COMPACT_POLICY_SIZE):
        return COMPACT_MIRROR_POLICY_MAP
    if int(width) == int(POLICY_SIZE):
        return MIRROR_POLICY_MAP
    raise ValueError(f"policy width must be {POLICY_SIZE} or {COMPACT_POLICY_SIZE}, got {width}")


def _mirror_policy_index_for_width(index: int, width: int) -> int:
    if int(width) == int(POLICY_SIZE):
        return mirror_policy_index(index)
    mapping = _mirror_map_for_policy_width(width)
    if index < 0 or index >= int(mapping.shape[0]):
        raise ValueError(f"policy index {index} out of range for policy width {width}")
    return int(mapping[index])


def _mirror_policy_index_array(values: np.ndarray, mask: np.ndarray, *, policy_width: int) -> np.ndarray:
    out = np.array(values, copy=True, order="C")
    idx = out[mask].astype(np.int64, copy=False)
    mirrored_idx = np.array(idx, copy=True)
    valid = mirrored_idx >= 0
    if np.any(valid):
        mirror_map = _mirror_map_for_policy_width(policy_width)
        mirrored_idx[valid] = mirror_map[mirrored_idx[valid]]
    out[mask] = mirrored_idx.astype(out.dtype, copy=False)
    return out


_FILE_FLIP_SQUARES = np.arange(64) ^ 7  # a1<->h1 etc.; ranks unchanged


def mirror_relations(rel: np.ndarray) -> np.ndarray:
    """File-flip dynamic relation matrices on BOTH square axes.

    Relations are side-to-move oriented (rank flips already baked in), so a
    left-right board mirror permutes from- and to-squares with sq ^ 7.
    Accepts (..., K, 64, 64).
    """
    arr = np.asarray(rel)
    if arr.shape[-2:] != (64, 64):
        raise ValueError(f"relations must end with (64, 64); got {arr.shape}")
    return np.ascontiguousarray(
        arr[..., _FILE_FLIP_SQUARES, :][..., :, _FILE_FLIP_SQUARES]
    )


def mirror_x(x: np.ndarray, *, input_history_encoding: str | None = None) -> np.ndarray:
    """Mirror an encoded (C,8,8) position tensor left-right (file flip)."""
    arr = np.asarray(x)
    if arr.ndim != 3 or tuple(arr.shape[-2:]) != (8, 8):
        raise ValueError(f"x must be (C,8,8); got {arr.shape}")
  # Flip file axis (last axis). Force a positive-stride array.
    out = arr[:, :, ::-1].copy()
    _swap_constant_planes(out, _castling_swap_pairs(input_history_encoding))
    return out


def _mirror_x_batch(x: np.ndarray, mask: np.ndarray, *, input_history_encoding: str | None = None) -> np.ndarray:
    out = np.array(x, copy=True, order="C")
    out[mask] = out[mask, :, :, ::-1].copy()
    rows = np.flatnonzero(mask)
    for a, b in _castling_swap_pairs(input_history_encoding):
        if out.shape[1] <= max(a, b):
            continue
        tmp = out[rows, a, :, :].copy()
        out[rows, a, :, :] = out[rows, b, :, :]
        out[rows, b, :, :] = tmp
    return out


def mirror_sample(s: ReplaySample, *, input_history_encoding: str | None = None) -> ReplaySample:
    """Create the left-right mirrored version of a ReplaySample."""
    x_m = mirror_x(s.x, input_history_encoding=input_history_encoding)
    x_lc0_root = getattr(s, "x_lc0_root", None)

    pol_m = mirror_policy(s.policy_target)

    out = ReplaySample(
        x=x_m,
        policy_target=pol_m,
        wdl_target=int(s.wdl_target),
        priority=float(getattr(s, "priority", 1.0)),
        priority_policy_kl=getattr(s, "priority_policy_kl", None),
        priority_q_delta=getattr(s, "priority_q_delta", None),
        priority_sf_search_gap=getattr(s, "priority_sf_search_gap", None),
        game_id=getattr(s, "game_id", None),
        ply_index=getattr(s, "ply_index", None),
        has_policy=bool(getattr(s, "has_policy", True)),
        x_lc0_root=None if x_lc0_root is None else mirror_x(x_lc0_root, input_history_encoding="lc0_root"),
        relations=(
            None if getattr(s, "relations", None) is None
            else mirror_relations(np.asarray(s.relations))
        ),
        input_history_encoding=getattr(s, "input_history_encoding", None),
        history_rep_fix=bool(getattr(s, "history_rep_fix", False)),
    )

  # Aux targets
    out.sf_wdl = None if s.sf_wdl is None else np.asarray(s.sf_wdl, dtype=np.float32)
    policy_width = int(np.asarray(s.policy_target).shape[-1])
    if s.sf_move_index is None:
        out.sf_move_index = None
    else:
        out.sf_move_index = _mirror_policy_index_for_width(
            int(s.sf_move_index),
            policy_width,
        )
    if s.sf_played_move_index is None:
        out.sf_played_move_index = None
    else:
        out.sf_played_move_index = _mirror_policy_index_for_width(
            int(s.sf_played_move_index),
            policy_width,
        )
    out.sf_played_rank = getattr(s, "sf_played_rank", None)
    out.sf_played_regret = getattr(s, "sf_played_regret", None)
    out.sf_policy_target = None if s.sf_policy_target is None else mirror_policy(s.sf_policy_target)
    out.moves_left = None if s.moves_left is None else float(s.moves_left)
    out.is_network_turn = None if s.is_network_turn is None else bool(s.is_network_turn)
    out.is_selfplay = None if s.is_selfplay is None else bool(s.is_selfplay)

    out.categorical_target = None if s.categorical_target is None else np.asarray(s.categorical_target, dtype=np.float32)

    out.policy_soft_target = None if s.policy_soft_target is None else mirror_policy(s.policy_soft_target)
    out.future_policy_target = None if s.future_policy_target is None else mirror_policy(s.future_policy_target)
    out.has_future = getattr(s, "has_future", None)
    out.sf_p0_policy_target = None if s.sf_p0_policy_target is None else mirror_policy(s.sf_p0_policy_target)
    out.has_sf_p0 = getattr(s, "has_sf_p0", None)
  # Per-move value vector: same move-index permutation as a policy under reflection.
    out.sf_p0_regret = None if s.sf_p0_regret is None else mirror_policy(s.sf_p0_regret)
    out.has_sf_p0_regret = getattr(s, "has_sf_p0_regret", None)

    out.volatility_target = None if s.volatility_target is None else np.asarray(s.volatility_target, dtype=np.float32)
    out.has_volatility = getattr(s, "has_volatility", None)

    out.sf_volatility_target = None if getattr(s, "sf_volatility_target", None) is None else np.asarray(s.sf_volatility_target, dtype=np.float32)
    out.has_sf_volatility = getattr(s, "has_sf_volatility", None)

    out.search_wdl = None if getattr(s, "search_wdl", None) is None else np.asarray(s.search_wdl, dtype=np.float32)
    out.future_sf_regret_sum = getattr(s, "future_sf_regret_sum", None)
    out.future_sf_regret_d95 = getattr(s, "future_sf_regret_d95", None)
    out.future_sf_regret_d98 = getattr(s, "future_sf_regret_d98", None)
    out.future_sf_regret_max = getattr(s, "future_sf_regret_max", None)
    out.future_sf_regret_h4 = getattr(s, "future_sf_regret_h4", None)
    out.future_sf_regret_h6 = getattr(s, "future_sf_regret_h6", None)
    out.future_sf_regret_h12 = getattr(s, "future_sf_regret_h12", None)
    out.future_sf_regret_h24 = getattr(s, "future_sf_regret_h24", None)
    out.future_sf_regret_h50 = getattr(s, "future_sf_regret_h50", None)
    out.future_sf_regret_count = getattr(s, "future_sf_regret_count", None)

    for name in LEGAL_MASK_FIELDS:
        v = getattr(s, name, None)
        if v is None:
            setattr(out, name, None)
        else:
            mirrored = mirror_policy(np.asarray(v, dtype=np.float32))
            setattr(out, name, mirrored.astype(np.asarray(v).dtype, copy=False))

  # Sparse MultiPV labels: move-index column mirrors with policy space,
  # scores/wdl and the record-level meta are mirror-invariant.
    raw = getattr(s, "sf_multipv_raw", None)
    if raw is not None:
        out.sf_multipv_raw = _mirror_sparse_multipv(
            np.asarray(raw)[None, ...],
            np.array([True]),
            policy_width=policy_width,
        )[0]
    meta = getattr(s, "sf_label_meta", None)
    out.sf_label_meta = None if meta is None else np.array(meta, copy=True)

    return out


def maybe_mirror_samples(
    samples: list[ReplaySample],
    *,
    rng: np.random.Generator,
    prob: float,
    input_history_encoding: str | None = None,
) -> list[ReplaySample]:
    """Apply mirroring augmentation to a batch of samples with given probability."""
    p = float(prob)
    if p <= 0.0:
        return samples

    return [
        mirror_sample(s, input_history_encoding=input_history_encoding) if float(rng.random()) < p else s
        for s in samples
    ]


def maybe_mirror_batch_arrays(
    arrs: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
    prob: float,
    input_history_encoding: str | None = None,
) -> dict[str, np.ndarray]:
    """Apply mirroring augmentation to array-backed replay batches."""
    p = float(prob)
    if p <= 0.0:
        return arrs

    x = np.asarray(arrs["x"])
    n = int(x.shape[0])
    if n <= 0:
        return arrs

    mask = rng.random(n) < p
    if not np.any(mask):
        return arrs

    out = dict(arrs)
    out["x"] = _mirror_x_batch(np.asarray(arrs["x"]), mask, input_history_encoding=input_history_encoding)
    if "x_lc0_root" in arrs:
        out["x_lc0_root"] = _mirror_x_batch(
            np.asarray(arrs["x_lc0_root"]),
            mask,
            input_history_encoding="lc0_root",
        )

    for key in _MIRROR_POLICY_FIELDS:
        if key in arrs:
            src_dtype = np.asarray(arrs[key]).dtype
            out[key] = np.array(arrs[key], copy=True, order="C")
            mirrored = mirror_policy_batch(out[key][mask])
            out[key][mask] = mirrored.astype(src_dtype, copy=False)

    if "relations" in arrs:
        out["relations"] = np.array(arrs["relations"], copy=True, order="C")
        out["relations"][mask] = mirror_relations(out["relations"][mask])

    policy_width = int(np.asarray(arrs["policy_target"]).shape[1])
    for key in ("sf_move_index", "sf_played_move_index"):
        if key in arrs:
            out[key] = _mirror_policy_index_array(
                np.asarray(arrs[key]),
                mask,
                policy_width=policy_width,
            )

    if "sf_multipv_raw" in arrs:
        out["sf_multipv_raw"] = _mirror_sparse_multipv(
            np.asarray(arrs["sf_multipv_raw"]), mask, policy_width=policy_width,
        )

    return out


def _mirror_sparse_multipv(
    raw: np.ndarray, mask: np.ndarray, *, policy_width: int,
) -> np.ndarray:
    """Remap sparse MultiPV move indices (col 0) under the mirror permutation.

    Scores/wdl columns are mirror-invariant; only the move index column
    lives in policy space. Pad rows (idx -1) pass through.
    """
    out = np.array(raw, copy=True, order="C")
    rows = out[mask]
    idx = rows[..., 0].astype(np.int64)
    valid = idx >= 0
    if np.any(valid):
        mirror_map = _mirror_map_for_policy_width(policy_width)
        idx[valid] = mirror_map[idx[valid]]
        rows[..., 0] = idx.astype(rows.dtype, copy=False)
        out[mask] = rows
    return out
