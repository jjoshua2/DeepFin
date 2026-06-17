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
    EXTRA_FEATURES_V2_LEAN,
    EXTRA_FEATURES_V2_THREATS,
    _V2_LEAN_KEEP,
    extra_feature_plane_count,
)
from chess_anti_engine.encoding.plane_decode import recompute_extra_planes

from .shard import INPUT_HISTORY_ENCODING_ARRAY_KEY

V1_INPUT_PLANES = input_plane_count(EXTRA_FEATURES_V1)          # 146
V2_INPUT_PLANES = input_plane_count(EXTRA_FEATURES_V2_THREATS)  # 175
V2_LEAN_INPUT_PLANES = input_plane_count(EXTRA_FEATURES_V2_LEAN)  # 143
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


def _zero_padded_rows_to(arr: np.ndarray, target_planes: int) -> np.ndarray:
    """Version-general twin of :func:`_zero_padded_rows`.

    Rows whose v1 extra block is live but whose *entire* post-v1 extra block
    ``[146:target]`` is all zero — the signature of the pre-upgrade zero-pad
    load path persisting a v1 chunk at ``target`` width. A native encode at any
    recognized version always fills attack-union planes for a live position, so
    a live-v1/zero-extra row can only be a zero-pad artifact, never real data.
    """
    v1_live = np.any(arr[:, _LC0_PLANES:V1_INPUT_PLANES], axis=(1, 2, 3))
    extra_zero = ~np.any(arr[:, V1_INPUT_PLANES:target_planes], axis=(1, 2, 3))
    return v1_live & extra_zero


def _repair_zero_padded_extra(
    arrs: dict[str, np.ndarray],
    target_planes: int,
    version: str,
    history_encoding: str | None,
) -> tuple[dict[str, np.ndarray], UpgradeStats]:
    """Recompute the extra block for equal-width rows that were zero-padded.

    Version-general twin of :func:`_repair_zero_padded_v2` for any recognized
    multi-version ``target_planes`` (179 v3_checks, 181 v3_xray, …). A stored
    chunk already at the target width is normally native and passes through
    untouched (cheap — only the all-zero gate scan runs); only rows that the
    pre-upgrade zero-pad path left with an empty extra block get recomputed,
    validated against their stored v1 prefix exactly like the cross-version
    path. ``x`` width is unchanged.
    """
    x = np.asarray(arrs["x"])
    n = int(x.shape[0])
    x_pad = _zero_padded_rows_to(x, target_planes)
    x_root = arrs.get("x_lc0_root")
    root_pad = None
    if x_root is not None:
        x_root = np.asarray(x_root)
        root_pad = _zero_padded_rows_to(x_root, target_planes)
    need = x_pad if root_pad is None else (x_pad | root_pad)
    if not need.any():
        return arrs, UpgradeStats(rows=n, upgraded_rows=0, dropout_rows=0)

    idx = np.flatnonzero(need)
    # plane_decode reads only planes below 146, so wider rows decode fine.
    recomputed = recompute_extra_planes(x[idx], history_encoding, version=version)
    out = dict(arrs)
    block, _ = _validated_extra_block(x[idx], recomputed, label="x (pad repair)")
    sel_x = x_pad[idx]
    if sel_x.any():
        new_x = np.array(x, copy=True)
        new_x[idx[sel_x], _LC0_PLANES:] = block[sel_x]
        out["x"] = new_x
    if root_pad is not None and root_pad[idx].any():
        root_block, _ = _validated_extra_block(
            np.asarray(x_root)[idx], recomputed, label="x_lc0_root (pad repair)",
        )
        sel_r = root_pad[idx]
        new_root = np.array(x_root, copy=True)
        new_root[idx[sel_r], _LC0_PLANES:] = root_block[sel_r]
        out["x_lc0_root"] = new_root
    return out, UpgradeStats(rows=n, upgraded_rows=int(idx.size), dropout_rows=0)


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


def _validated_extra_block_from_lean(
    stored: np.ndarray,
    recomputed: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`_validated_extra_block` but for a v2_lean-width *source*.

    A stored v2_lean chunk has an intact 112-plane LC0 base (so the recompute
    is exact) but its 31-plane extra block ``[112:143]`` is the gathered lean
    subset, NOT the 34-plane v1 prefix — so the v1-prefix decode check can't
    run. The lean planes (pins + the v2 attack/threat families) sit at the same
    extra-block indices in every v2/v3 version, so we instead validate the
    decode by gathering the recomputed block through ``_V2_LEAN_KEEP`` and
    comparing it to the stored lean block. Liveness keys on the stored lean
    block (any nonzero plane) rather than the absent v1 block. Returns
    ``(block, live_row_mask)`` for the requested target version.
    """
    stored_lean = np.asarray(stored[:, _LC0_PLANES:V2_LEAN_INPUT_PLANES], dtype=np.float32)
    live = np.any(stored_lean, axis=(1, 2, 3))
    gathered = recomputed[:, _V2_LEAN_KEEP]
    diff = np.abs(gathered - stored_lean)
    bad = live & (np.max(diff, axis=(1, 2, 3), initial=0.0) > _VALIDATE_ATOL)
    if bad.any():
        row = int(np.flatnonzero(bad)[0])
        plane = int(np.unravel_index(np.argmax(diff[row]), diff[row].shape)[0])
        raise ValueError(
            f"{label}: recomputed lean planes disagree with stored v2_lean "
            f"planes on {int(bad.sum())}/{stored.shape[0]} rows (first: row "
            f"{row}, lean plane {plane}, max |diff| {float(diff[row].max()):.4f}) "
            f"— refusing to rebuild extra planes from a bad decode"
        )
    block = recomputed.astype(stored.dtype)
    block[~live] = 0
    return block, live


def _gather_v2_lean_block(v2_arrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Select v2_lean planes from a full-v2 (175-plane) chunk: the 112 LC0 base
    + the kept extra planes (``_V2_LEAN_KEEP``). ``x`` and ``x_lc0_root`` are
    gathered identically; all other keys pass through. Input dict is not mutated.
    """
    out = dict(v2_arrs)
    for key in ("x", "x_lc0_root"):
        v = out.get(key)
        if v is None:
            continue
        v = np.asarray(v)
        out[key] = np.concatenate(
            [v[:, :_LC0_PLANES], v[:, _LC0_PLANES:][:, _V2_LEAN_KEEP]], axis=1
        )
    return out


def _recompute_extra_block_to_version(
    arrs: dict[str, np.ndarray],
    x: np.ndarray,
    planes: int,
    version: str,
    history_encoding: str | None,
) -> tuple[dict[str, np.ndarray], UpgradeStats]:
    """Rebuild ``x`` (and ``x_lc0_root``) at ``version`` from the LC0 base.

    Recomputes the full extra block (v1 prefix + later planes) from the stored
    112 LC0 base planes — which works for ANY recognized stored width, since the
    recompute reads only the step-0/EP planes (all < 146) and the first 34
    recomputed planes are validated against the stored v1 block ``[112:146]`` (a
    true prefix preserved across every version). The stored extra block beyond
    v1 is never read, so a 146/175/179/181 chunk all recompute correctly. Used
    by the general cross-version path and as the full-v2 step of the v2_lean
    downgrade. The input dict is never mutated.

    A 143-plane v2_lean *source* has no v1 prefix to validate against (its extra
    block is the gathered lean subset, not the 34 v1 planes), so the decode is
    instead validated by gathering the recompute through ``_V2_LEAN_KEEP`` —
    bit-stable across every v2/v3 version — and comparing to the stored lean
    block. This lets a run leave v2_lean (143) and feed its replay back into a
    v2_threats / v3 target via recompute-from-base rather than refusing it.
    """
    src_is_lean = int(planes) == V2_LEAN_INPUT_PLANES
    validate = _validated_extra_block_from_lean if src_is_lean else _validated_extra_block
    recomputed = recompute_extra_planes(x, history_encoding, version=version)
    x_block, x_live = validate(x, recomputed, label="x")
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
        root_block, _ = validate(x_root, recomputed, label="x_lc0_root")
        out["x_lc0_root"] = np.concatenate([x_root[:, :_LC0_PLANES], root_block], axis=1)

    return out, UpgradeStats(
        rows=int(x.shape[0]),
        upgraded_rows=int(x.shape[0]),
        dropout_rows=dropout_rows,
    )


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
    LC0 base planes — which works for ANY recognized stored width, since the
    recompute reads only the step-0/EP planes (all < 146) and validates against
    the stored v1 block ``[112:146]`` (a true prefix preserved across every
    version). The stored extra block beyond v1 is never read, so a 146 (v1),
    175 (v2_threats), 179 (v3_checks) or 181 (v3_xray) shard can all be
    recomputed to the target version — crucially, this is the correct path for
    cross-version mismatches (e.g. a stored 179 chunk loaded by a 181 model),
    which must NOT fall through to zero-padding. The all-zero-row dropout
    convention and ``x_lc0_root`` handling match the v2 path. The input dict is
    never mutated.

    When the stored width already equals the target, native chunks pass through
    untouched, but rows that the pre-upgrade zero-pad path persisted at the
    target width with an all-zero extra block are repaired in place (recomputed
    only for those rows) — so an equal-width load is NOT an unconditional
    passthrough. A 175-plane target delegates to
    :func:`upgrade_arrays_to_v2_threats` so the zero-padded-row repair path
    stays intact for backward compatibility — except a v2_lean (143) *source*,
    which that path's 146/175 guard would reject; a lean source is routed
    through the general recompute-from-base helper for any target so a run can
    leave v2_lean and feed its replay back into v2_threats / a v3 width.
    """
    target = int(target_planes)
    x = np.asarray(arrs["x"])
    planes = int(x.shape[1])
    # A v2_lean (143) source has no v1 prefix and is rejected by
    # upgrade_arrays_to_v2_threats' 146/175 guard, so bypass the 175 short-circuit
    # for it; the recompute-from-base helper below validates lean sources via the
    # lean gather instead. Equal-width (lean→lean) still flows to the v2_lean
    # branch's native passthrough.
    src_is_lean = planes == V2_LEAN_INPUT_PLANES
    if target == V2_INPUT_PLANES and not src_is_lean:
        return upgrade_arrays_to_v2_threats(arrs, history_encoding=history_encoding)
    version = version_for_input_planes(target)

    # Accept any recognized stored width: the recompute reads only planes below
    # 146 (the LC0 base, identical across every version) and validates either the
    # v1 prefix or — for a lean source — the lean gather, so a 143/146/175/179/181
    # stored chunk all recompute correctly. An unrecognized width has no
    # trustworthy prefix to validate against.
    try:
        version_for_input_planes(planes)
    except ValueError as exc:
        raise ValueError(
            f"cannot upgrade chunk with {planes} input planes to {version} "
            f"({target} planes): stored width is not a recognized version"
        ) from exc
    if history_encoding is None:
        history_encoding = chunk_history_encoding(arrs)

    if version == EXTRA_FEATURES_V2_LEAN:
        # v2_lean is a strict SUBSET of v2 — its block is NOT a v1 superset, so it
        # can't be validated against the stored v1 prefix directly (the validation
        # in _validated_extra_block assumes recomputed[:34] == stored v1 block).
        # Upgrade to the full v2 block (which validates the v1 prefix + recomputes
        # the 63-plane block from the decoded base for ANY recognized stored
        # width), then GATHER the kept lean planes. Mirrors the compute side
        # exactly (features.py computes full v2 then gathers via _V2_LEAN_KEEP).
        if planes == target:
            # Native v2_lean chunk: stored extra block is already lean. The v2
            # validation path can't run (its stored [112:146] is not a v1 block),
            # so pass through untouched. v2_lean is new, so no zero-padded-row
            # repair is owed (no pre-upgrade path ever wrote v2_lean-width rows).
            return dict(arrs), UpgradeStats(
                rows=int(x.shape[0]), upgraded_rows=0, dropout_rows=0
            )
        # Rebuild the full v2 block from the LC0 base for ANY recognized stored
        # width (146/175/179/181), then gather the lean planes. Routing through
        # upgrade_arrays_to_v2_threats here would hard-fail any width other than
        # 146/175 (e.g. a v3 179/181 chunk downgraded to v2_lean), and the
        # ValueError would then hit _pad_x_planes' truncation guard and kill the
        # chunk; recompute-from-base handles every width like the v2/v3 paths.
        v2_out, stats = _recompute_extra_block_to_version(
            arrs, x, planes, EXTRA_FEATURES_V2_THREATS, history_encoding,
        )
        return _gather_v2_lean_block(v2_out), stats

    if planes == target:
        # Equal width: native chunks pass through untouched (only the all-zero
        # gate scan runs); rows the pre-upgrade zero-pad path left empty get
        # their extra block recomputed in place. Mirrors the v2 repair path.
        return _repair_zero_padded_extra(arrs, target, version, history_encoding)

    return _recompute_extra_block_to_version(arrs, x, planes, version, history_encoding)
