from __future__ import annotations

import numpy as np

from chess_anti_engine.moves.encode import (
    MIRROR_POLICY_MAP,
    mirror_policy,
    mirror_policy_batch,
    mirror_policy_index,
)

from .buffer import ReplaySample
from .shard import LEGAL_MASK_FIELDS, POLICY_SPACE_FIELDS

# Fields whose rows live in POLICY_SIZE move space and must be remapped under
# the mirror permutation. Includes all policy-like targets and legal masks.
_MIRROR_POLICY_FIELDS = (*POLICY_SPACE_FIELDS, *LEGAL_MASK_FIELDS)


def mirror_x(x: np.ndarray) -> np.ndarray:
    """Mirror an encoded (C,8,8) position tensor left-right (file flip)."""
    arr = np.asarray(x)
    if arr.ndim != 3 or tuple(arr.shape[-2:]) != (8, 8):
        raise ValueError(f"x must be (C,8,8); got {arr.shape}")
  # Flip file axis (last axis). Force a positive-stride array.
    return arr[:, :, ::-1].copy()


def mirror_sample(s: ReplaySample) -> ReplaySample:
    """Create the left-right mirrored version of a ReplaySample."""
    x_m = mirror_x(s.x)

    pol_m = mirror_policy(s.policy_target)

    out = ReplaySample(
        x=x_m,
        policy_target=pol_m,
        wdl_target=int(s.wdl_target),
        priority=float(getattr(s, "priority", 1.0)),
        has_policy=bool(getattr(s, "has_policy", True)),
    )

  # Aux targets
    out.sf_wdl = None if s.sf_wdl is None else np.asarray(s.sf_wdl, dtype=np.float32)
    out.sf_move_index = None if s.sf_move_index is None else int(mirror_policy_index(int(s.sf_move_index)))
    out.sf_policy_target = None if s.sf_policy_target is None else mirror_policy(s.sf_policy_target)
    out.moves_left = None if s.moves_left is None else float(s.moves_left)
    out.is_network_turn = None if s.is_network_turn is None else bool(s.is_network_turn)
    out.is_selfplay = None if s.is_selfplay is None else bool(s.is_selfplay)

    out.categorical_target = None if s.categorical_target is None else np.asarray(s.categorical_target, dtype=np.float32)

    out.policy_soft_target = None if s.policy_soft_target is None else mirror_policy(s.policy_soft_target)
    out.future_policy_target = None if s.future_policy_target is None else mirror_policy(s.future_policy_target)
    out.has_future = getattr(s, "has_future", None)

    out.volatility_target = None if s.volatility_target is None else np.asarray(s.volatility_target, dtype=np.float32)
    out.has_volatility = getattr(s, "has_volatility", None)

    out.sf_volatility_target = None if getattr(s, "sf_volatility_target", None) is None else np.asarray(s.sf_volatility_target, dtype=np.float32)
    out.has_sf_volatility = getattr(s, "has_sf_volatility", None)

    for name in LEGAL_MASK_FIELDS:
        v = getattr(s, name, None)
        if v is None:
            setattr(out, name, None)
        else:
            mirrored = mirror_policy(np.asarray(v, dtype=np.float32))
            setattr(out, name, mirrored.astype(np.asarray(v).dtype, copy=False))

    return out


def maybe_mirror_samples(
    samples: list[ReplaySample],
    *,
    rng: np.random.Generator,
    prob: float,
) -> list[ReplaySample]:
    """Apply mirroring augmentation to a batch of samples with given probability."""
    p = float(prob)
    if p <= 0.0:
        return samples

    return [
        mirror_sample(s) if float(rng.random()) < p else s
        for s in samples
    ]


def maybe_mirror_batch_arrays(
    arrs: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
    prob: float,
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
    out["x"] = np.array(arrs["x"], copy=True, order="C")
    out["x"][mask] = out["x"][mask, :, :, ::-1].copy()

    for key in _MIRROR_POLICY_FIELDS:
        if key in arrs:
            src_dtype = np.asarray(arrs[key]).dtype
            out[key] = np.array(arrs[key], copy=True, order="C")
            mirrored = mirror_policy_batch(out[key][mask])
            out[key][mask] = mirrored.astype(src_dtype, copy=False)

    if "sf_move_index" in arrs:
        out["sf_move_index"] = np.array(arrs["sf_move_index"], copy=True, order="C")
        idx = out["sf_move_index"][mask].astype(np.int64, copy=False)
        out["sf_move_index"][mask] = MIRROR_POLICY_MAP[idx].astype(out["sf_move_index"].dtype, copy=False)

    return out
