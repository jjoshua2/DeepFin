"""Upgrade stored v1 replay chunks (146 input planes) to v2_threats (175).

Zero-padding v1 shards into a v2_threats run (the original warm-start
fallback) starves the 29 threat planes of signal on the entire old
window. This module recomputes them for real: decode the step-0
bitboards from the stored planes, rerun the extra-feature kernel, and
append the threat block — validating the 34 recomputed v1 planes against
the stored ones so a decode bug can never silently corrupt training
inputs. See ``encoding/plane_decode.py`` for why this is exact.

Used by the offline converter (``scripts/convert_shards_v2_threats.py``)
and the on-the-fly path in ``DiskReplayBuffer`` (``upgrade_v1_planes``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.encoding.features import (
    EXTRA_FEATURES_V1,
    EXTRA_FEATURES_V2_THREATS,
    extra_feature_plane_count,
)
from chess_anti_engine.encoding.plane_decode import recompute_extra_planes

from .shard import INPUT_HISTORY_ENCODING_ARRAY_KEY

V1_INPUT_PLANES = input_plane_count(EXTRA_FEATURES_V1)          # 146
V2_INPUT_PLANES = input_plane_count(EXTRA_FEATURES_V2_THREATS)  # 175
_V1_EXTRA = extra_feature_plane_count(EXTRA_FEATURES_V1)        # 34
_LC0_PLANES = V1_INPUT_PLANES - _V1_EXTRA                       # 112
# fp16 storage quantizes the float32 extra planes; every plane value lies
# in [-1, 1] where fp16 spacing is <= 2^-10, so 2e-3 passes round-trip
# noise while any decode error (wrong square/POV/EP) lands far outside.
_VALIDATE_ATOL = 2e-3


@dataclass(frozen=True)
class UpgradeStats:
    rows: int
    upgraded_rows: int
    dropout_rows: int  # rows whose stored extra block was all-zero


def chunk_history_encoding(arrs: dict[str, np.ndarray]) -> str | None:
    raw = arrs.get(INPUT_HISTORY_ENCODING_ARRAY_KEY)
    if raw is None:
        return None
    arr = np.asarray(raw)
    return str(arr.reshape(-1)[0]) if arr.size else None


def _validated_threat_block(
    stored: np.ndarray,
    recomputed: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Append-block for one array, validated against its stored v1 planes.

    Rows whose stored extra block is all-zero (encode-time feature dropout
    in ancient shards, or unflagged all-zero optional rows) get an
    all-zero threat block — matching the dropout convention of zeroing
    the whole extra block — and are excluded from validation. Returns
    ``(block, live_row_mask)``.
    """
    stored_v1 = np.asarray(stored[:, _LC0_PLANES:V1_INPUT_PLANES], dtype=np.float32)
    live = np.any(stored_v1, axis=(1, 2, 3))
    diff = np.abs(recomputed[:, :_V1_EXTRA] - stored_v1)
    bad = live & (np.max(diff, axis=(1, 2, 3), initial=0.0) > _VALIDATE_ATOL)
    if bad.any():
        row = int(np.flatnonzero(bad)[0])
        plane = int(np.unravel_index(np.argmax(diff[row]), diff[row].shape)[0])
        raise ValueError(
            f"{label}: recomputed v1 extra planes disagree with stored planes "
            f"on {int(bad.sum())}/{stored.shape[0]} rows (first: row {row}, "
            f"extra plane {plane}, max |diff| {float(diff[row].max()):.4f}) — "
            f"refusing to append threat planes from a bad decode"
        )
    block = recomputed[:, _V1_EXTRA:].astype(stored.dtype)
    block[~live] = 0
    return block, live


def upgrade_arrays_to_v2_threats(
    arrs: dict[str, np.ndarray],
    *,
    history_encoding: str | None = None,
) -> tuple[dict[str, np.ndarray], UpgradeStats]:
    """Return a chunk dict with ``x`` (and ``x_lc0_root``) at 175 planes.

    Chunks already at 175 planes pass through unchanged; anything other
    than 146/175 is a hard error. The input dict is never mutated.
    """
    x = np.asarray(arrs["x"])
    planes = int(x.shape[1])
    if planes == V2_INPUT_PLANES:
        return arrs, UpgradeStats(rows=int(x.shape[0]), upgraded_rows=0, dropout_rows=0)
    if planes != V1_INPUT_PLANES:
        raise ValueError(
            f"cannot upgrade chunk with {planes} input planes to v2_threats; "
            f"expected {V1_INPUT_PLANES} or {V2_INPUT_PLANES}"
        )
    if history_encoding is None:
        history_encoding = chunk_history_encoding(arrs)

    recomputed = recompute_extra_planes(x, history_encoding)
    x_block, x_live = _validated_threat_block(x, recomputed, label="x")
    out = dict(arrs)
    out["x"] = np.concatenate([x, x_block], axis=1)
    dropout_rows = int(np.sum(~x_live))

    x_root = arrs.get("x_lc0_root")
    if x_root is not None:
        x_root = np.asarray(x_root)
        if int(x_root.shape[1]) != V1_INPUT_PLANES:
            raise ValueError(
                f"x_lc0_root stores {int(x_root.shape[1])} planes but x stores "
                f"{V1_INPUT_PLANES}; shard schema guarantees they match"
            )
        # Same position, same side-to-move frame ⇒ same extra block; the
        # recompute from x carries over. Unflagged rows are stored as
        # zeros and keep a zero threat block via the all-zero gate.
        root_block, _ = _validated_threat_block(x_root, recomputed, label="x_lc0_root")
        out["x_lc0_root"] = np.concatenate([x_root, root_block], axis=1)

    return out, UpgradeStats(
        rows=int(x.shape[0]),
        upgraded_rows=int(x.shape[0]),
        dropout_rows=dropout_rows,
    )
