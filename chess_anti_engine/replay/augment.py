from __future__ import annotations

import numpy as np

from chess_anti_engine.moves.encode import mirror_policy, mirror_policy_index

from .buffer import ReplaySample


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

    out.categorical_target = None if s.categorical_target is None else np.asarray(s.categorical_target, dtype=np.float32)

    out.policy_soft_target = None if s.policy_soft_target is None else mirror_policy(s.policy_soft_target)
    out.future_policy_target = None if s.future_policy_target is None else mirror_policy(s.future_policy_target)
    out.has_future = getattr(s, "has_future", None)

    out.volatility_target = None if s.volatility_target is None else np.asarray(s.volatility_target, dtype=np.float32)
    out.has_volatility = getattr(s, "has_volatility", None)

    out.sf_volatility_target = None if getattr(s, "sf_volatility_target", None) is None else np.asarray(s.sf_volatility_target, dtype=np.float32)
    out.has_sf_volatility = getattr(s, "has_sf_volatility", None)

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

    out: list[ReplaySample] = []
    for s in samples:
        if float(rng.random()) < p:
            out.append(mirror_sample(s))
        else:
            out.append(s)
    return out
