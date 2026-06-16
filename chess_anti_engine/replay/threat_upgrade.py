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

from chess_anti_engine.encoding import input_plane_count, version_for_input_planes
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


def _zero_padded_rows(arr: np.ndarray) -> np.ndarray:
    """Rows whose v1 extra block is live but whose threat block is all zero.

    The threat block of any live position contains the nonzero attack-union
    planes (both kings always attack something), so live-v1/zero-threat rows
    can only come from the earlier zero-pad load path — never from a native
    v2_threats encode.
    """
    v1_live = np.any(arr[:, _LC0_PLANES:V1_INPUT_PLANES], axis=(1, 2, 3))
    threats_zero = ~np.any(arr[:, V1_INPUT_PLANES:V2_INPUT_PLANES], axis=(1, 2, 3))
    return v1_live & threats_zero


def _repair_zero_padded_v2(
    arrs: dict[str, np.ndarray],
    history_encoding: str | None,
) -> tuple[dict[str, np.ndarray], UpgradeStats]:
    """Recompute threat planes for 175-plane rows that were zero-padded.

    Chunks that went through the pre-upgrade zero-pad path may have been
    re-persisted at 175 planes with empty threat blocks; the width gate
    alone would treat them as native v2 forever.
    """
    x = np.asarray(arrs["x"])
    n = int(x.shape[0])
    x_pad = _zero_padded_rows(x)
    x_root = arrs.get("x_lc0_root")
    root_pad = None
    if x_root is not None:
        x_root = np.asarray(x_root)
        root_pad = _zero_padded_rows(x_root)
    need = x_pad if root_pad is None else (x_pad | root_pad)
    if not need.any():
        return arrs, UpgradeStats(rows=n, upgraded_rows=0, dropout_rows=0)

    idx = np.flatnonzero(need)
    # plane_decode only reads planes below 146, so 175-wide rows decode fine.
    recomputed = recompute_extra_planes(x[idx], history_encoding)
    out = dict(arrs)
    block, _ = _validated_threat_block(x[idx], recomputed, label="x (pad repair)")
    sel_x = x_pad[idx]
    if sel_x.any():
        new_x = np.array(x, copy=True)
        new_x[idx[sel_x], V1_INPUT_PLANES:] = block[sel_x]
        out["x"] = new_x
    if root_pad is not None and root_pad[idx].any():
        root_block, _ = _validated_threat_block(
            np.asarray(x_root)[idx], recomputed, label="x_lc0_root (pad repair)",
        )
        sel_r = root_pad[idx]
        new_root = np.array(x_root, copy=True)
        new_root[idx[sel_r], V1_INPUT_PLANES:] = root_block[sel_r]
        out["x_lc0_root"] = new_root
    return out, UpgradeStats(rows=n, upgraded_rows=int(idx.size), dropout_rows=0)


def upgrade_arrays_to_v2_threats(
    arrs: dict[str, np.ndarray],
    *,
    history_encoding: str | None = None,
) -> tuple[dict[str, np.ndarray], UpgradeStats]:
    """Return a chunk dict with ``x`` (and ``x_lc0_root``) at 175 planes.

    146-plane chunks gain recomputed threat planes; 175-plane chunks are
    scanned for previously zero-padded rows and repaired (native v2 rows
    pass through untouched). Anything other than 146/175 is a hard error.
    The input dict is never mutated.
    """
    x = np.asarray(arrs["x"])
    planes = int(x.shape[1])
    if history_encoding is None:
        history_encoding = chunk_history_encoding(arrs)
    if planes == V2_INPUT_PLANES:
        return _repair_zero_padded_v2(arrs, history_encoding)
    if planes != V1_INPUT_PLANES:
        raise ValueError(
            f"cannot upgrade chunk with {planes} input planes to v2_threats; "
            f"expected {V1_INPUT_PLANES} or {V2_INPUT_PLANES}"
        )

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


def _validated_extra_block(
    stored: np.ndarray,
    recomputed: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Full extra block (v1 + threats + checks), validated against stored v1.

    Like :func:`_validated_threat_block`, but returns the *entire* recomputed
    extra block (from index 0) rather than only the planes appended after v1.
    The first 34 recomputed planes are validated against the stored v1 block
    exactly as before — the v1 planes are version-independent, so any decode
    error lands far outside ``_VALIDATE_ATOL``. Rows whose stored v1 block is
    all-zero (encode-time dropout) get an all-zero extra block and are excluded
    from validation. Returns ``(block, live_row_mask)``.
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
            f"refusing to append extra planes from a bad decode"
        )
    block = recomputed.astype(stored.dtype)
    block[~live] = 0
    return block, live


def upgrade_arrays_to_planes(
    arrs: dict[str, np.ndarray],
    target_planes: int,
    *,
    history_encoding: str | None = None,
) -> tuple[dict[str, np.ndarray], UpgradeStats]:
    """Return a chunk dict with ``x`` (and ``x_lc0_root``) at ``target_planes``.

    Version-general twin of :func:`upgrade_arrays_to_v2_threats`: the target
    extra-features version is derived from ``target_planes`` (175→v2_threats,
    179→v3_checks, …). The full extra block is recomputed from the stored 112
    LC0 base planes — which works whether the stored shard is 146 (v1) or 175
    (v2) wide — and validated against the stored v1 planes before the LC0 base
    and recomputed extra block are concatenated to ``target_planes``. The
    all-zero-row dropout convention and ``x_lc0_root`` handling match the v2
    path. The input dict is never mutated.

    A 175-plane target delegates to :func:`upgrade_arrays_to_v2_threats` so the
    zero-padded-row repair path stays intact for backward compatibility.
    """
    target = int(target_planes)
    if target == V2_INPUT_PLANES:
        return upgrade_arrays_to_v2_threats(arrs, history_encoding=history_encoding)
    version = version_for_input_planes(target)

    x = np.asarray(arrs["x"])
    planes = int(x.shape[1])
    if planes not in (V1_INPUT_PLANES, V2_INPUT_PLANES):
        raise ValueError(
            f"cannot upgrade chunk with {planes} input planes to {version} "
            f"({target} planes); expected {V1_INPUT_PLANES} or {V2_INPUT_PLANES}"
        )
    if history_encoding is None:
        history_encoding = chunk_history_encoding(arrs)

    recomputed = recompute_extra_planes(x, history_encoding, version=version)
    x_block, x_live = _validated_extra_block(x, recomputed, label="x")
    out = dict(arrs)
    out["x"] = np.concatenate([x[:, :_LC0_PLANES], x_block], axis=1)
    dropout_rows = int(np.sum(~x_live))

    x_root = arrs.get("x_lc0_root")
    if x_root is not None:
        x_root = np.asarray(x_root)
        if int(x_root.shape[1]) != planes:
            raise ValueError(
                f"x_lc0_root stores {int(x_root.shape[1])} planes but x stores "
                f"{planes}; shard schema guarantees they match"
            )
        # Same position, same side-to-move frame ⇒ same extra block; the
        # recompute from x carries over. Unflagged rows are stored as zeros
        # and keep a zero extra block via the all-zero live gate.
        root_block, _ = _validated_extra_block(x_root, recomputed, label="x_lc0_root")
        out["x_lc0_root"] = np.concatenate([x_root[:, :_LC0_PLANES], root_block], axis=1)

    return out, UpgradeStats(
        rows=int(x.shape[0]),
        upgraded_rows=int(x.shape[0]),
        dropout_rows=dropout_rows,
    )
