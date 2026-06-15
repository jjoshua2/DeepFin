"""Rebuild SF-derived training targets from sparse MultiPV labels.

Shards (schema v2+) store the raw MultiPV candidate rows the live pipeline
built its targets from (``sf_multipv_raw``/``sf_label_meta``; layout in
replay/shard.py). These pure functions replay the exact live construction
(selfplay/stockfish_turn.py) with arbitrary parameters, so target questions —
``sf_policy_temp``, ``sf_policy_label_smooth``, cp→logistic ``slope`` /
``draw_width``, logistic-vs-native WDL — become offline retrains
(scripts/retarget_retrain.py) instead of fresh live runs.

With params equal to the capture-time config, the rebuilt targets match the
stored ones to float precision (parity-tested in
tests/test_sparse_multipv_labels.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chess_anti_engine.replay.shard import SF_CP_SENTINEL
from chess_anti_engine.stockfish.wdl import cp_to_wdl
from chess_anti_engine.train.targets import (
    DEFAULT_CATEGORICAL_BINS,
    categorical_target_value,
    hlgauss_target,
)


@dataclass(frozen=True)
class SfTargetParams:
    """Knobs of the SF target construction (defaults match compute paths'
    function defaults; pass live config values for parity)."""

    sf_policy_temp: float = 0.25
    sf_policy_label_smooth: float = 0.05
    sf_wdl_use_cp_logistic: bool = False
    sf_wdl_cp_slope: float = 0.010
    sf_wdl_cp_draw_width: float = 60.0


@dataclass(frozen=True)
class CategoricalTargetParams:
    """Knobs of the categorical (HL-Gauss) value target (mirror the selfplay
    ``categorical_*`` / ``hlgauss_sigma`` config; defaults match GameConfig)."""

    blend_frac: float = 0.0
    num_bins: int = DEFAULT_CATEGORICAL_BINS
    sigma: float = 0.04


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def _row_score(
    cp: int, mate: int, wdl_w: int, wdl_d: int, params: SfTargetParams,
) -> float | None:
    """w + 0.5*d for one MultiPV row — mirrors stockfish_turn._pv_wdl_score.

    Logistic path: cp/mate → normalized (w, d, l). Native path: SF's own
    permille wdl, deliberately NOT rescaled (the live path uses raw permille
    values; reproducing them exactly is the point).
    """
    has_cp = cp != SF_CP_SENTINEL
    has_mate = mate != 0
    if params.sf_wdl_use_cp_logistic and (has_cp or has_mate):
        wdl = cp_to_wdl(
            cp if has_cp else None,
            mate if has_mate else None,
            slope=params.sf_wdl_cp_slope,
            draw_width_cp=params.sf_wdl_cp_draw_width,
        )
        return float(wdl[0]) + 0.5 * float(wdl[1])
    if wdl_w < 0 or wdl_d < 0:
        return None
    return float(wdl_w) + 0.5 * float(wdl_d)


def rebuild_sf_policy_target(
    multipv_raw: np.ndarray,
    *,
    legal_indices: np.ndarray,
    policy_size: int,
    params: SfTargetParams,
) -> np.ndarray | None:
    """(K, 5) raw rows → dense SF policy target in the rows' move space.

    Mirrors stockfish_turn._build_sf_policy_target: softmax over candidate
    scores / temp, scatter-add, optional legal-set smoothing, renormalize.
    Returns None when no scoreable rows remain (caller keeps the stored
    target — matches the live fallback which never produced sparse rows in
    that case).
    """
    rows = np.asarray(multipv_raw)
    rows = rows[rows[:, 0] >= 0]
    idxs: list[int] = []
    scores: list[float] = []
    for move_idx, cp, mate, w, d in rows.tolist():
        score = _row_score(int(cp), int(mate), int(w), int(d), params)
        if score is None:
            continue
        idxs.append(int(move_idx))
        scores.append(score)
    if not idxs:
        return None

    p_top = _softmax(
        np.array(scores, dtype=np.float64) / max(1e-6, float(params.sf_policy_temp))
    ).astype(np.float32, copy=False)
    p_sf = np.zeros((int(policy_size),), dtype=np.float32)
    for a, p in zip(idxs, p_top, strict=True):
        p_sf[a] += float(p)

    smooth = float(params.sf_policy_label_smooth)
    legal = np.asarray(legal_indices, dtype=np.int64)
    if smooth > 0.0 and legal.size > 0:
        p_sf *= 1.0 - smooth
        p_sf[legal] += smooth / float(legal.size)

    total = float(p_sf.sum())
    if total > 0:
        p_sf /= total
    return p_sf


def rebuild_sf_wdl(label_meta: np.ndarray, params: SfTargetParams) -> np.ndarray | None:
    """(6,) label meta → record-POV (W, D, L) — mirrors
    stockfish_turn._sf_result_wdl_for_record including the POV flip."""
    meta = np.asarray(label_meta).reshape(-1)
    cp, mate = int(meta[2]), int(meta[3])
    wdl_w, wdl_d = int(meta[4]), int(meta[5])
    has_cp = cp != SF_CP_SENTINEL
    has_mate = mate != 0
    if params.sf_wdl_use_cp_logistic and (has_cp or has_mate):
        wdl_stm = np.asarray(cp_to_wdl(
            cp if has_cp else None,
            mate if has_mate else None,
            slope=params.sf_wdl_cp_slope,
            draw_width_cp=params.sf_wdl_cp_draw_width,
        ), dtype=np.float32)
        return wdl_stm[::-1].copy()  # flip_wdl_pov
    if wdl_w < 0 or wdl_d < 0:
        return None
    wdl_stm = np.array(
        [float(wdl_w), float(wdl_d), float(1000 - wdl_w - wdl_d)], dtype=np.float32,
    )
    return wdl_stm[::-1].copy()


def rebuild_sf_targets_in_arrays(
    arrs: dict[str, np.ndarray], *, params: SfTargetParams,
) -> dict[str, np.ndarray]:
    """Recompute ``sf_policy_target`` / ``sf_wdl`` in a sampled batch dict.

    Only rows carrying sparse labels are rebuilt; rows without them (old
    shards) keep their stored targets. Returns ``arrs`` (mutated in place on
    fresh copies of the touched fields).

    Cost boundary: this is a per-row Python loop (~16.6 ms per 256-row batch
    at policy width 1858). Acceptable for the stated offline use — flag-gated
    and overlapped by the host-side prefetch thread — but it must be
    vectorized before ``rebuild_sf_targets`` could ever become a live-training
    default.
    """
    has_raw = np.asarray(arrs.get("has_sf_multipv_raw", ()), dtype=bool)
    if has_raw.size and has_raw.any() and "sf_multipv_raw" in arrs and "sf_policy_target" in arrs:
        pol = np.array(arrs["sf_policy_target"], copy=True)
        policy_size = int(pol.shape[1])
        legal_dense = arrs.get("sf_legal_mask")
        for i in np.flatnonzero(has_raw):
            if legal_dense is not None and i < len(legal_dense):
                legal_idx = np.flatnonzero(np.asarray(legal_dense[i]))
            else:
                legal_idx = np.zeros((0,), dtype=np.int64)
            rebuilt = rebuild_sf_policy_target(
                np.asarray(arrs["sf_multipv_raw"][i]),
                legal_indices=legal_idx,
                policy_size=policy_size,
                params=params,
            )
            if rebuilt is not None:
                pol[i] = rebuilt.astype(pol.dtype, copy=False)
        arrs["sf_policy_target"] = pol

    has_meta = np.asarray(arrs.get("has_sf_label_meta", ()), dtype=bool)
    if has_meta.size and has_meta.any() and "sf_label_meta" in arrs and "sf_wdl" in arrs:
        wdl = np.array(arrs["sf_wdl"], copy=True)
        for i in np.flatnonzero(has_meta):
            rebuilt_wdl = rebuild_sf_wdl(np.asarray(arrs["sf_label_meta"][i]), params)
            if rebuilt_wdl is not None:
                wdl[i] = rebuilt_wdl.astype(wdl.dtype, copy=False)
        arrs["sf_wdl"] = wdl
    return arrs


def rebuild_categorical_target_in_arrays(
    arrs: dict[str, np.ndarray], *, params: CategoricalTargetParams,
) -> dict[str, np.ndarray]:
    """Recompute ``categorical_target`` in a sampled batch from the stored hard
    outcome (``wdl_target``) blended with SF's eval (``sf_wdl``).

    Mirrors selfplay/finalize (``categorical_target_value`` + ``hlgauss_target``)
    so the offline sidecar can screen ``categorical_blend_frac`` on stored shards
    — the live finalize path bakes this target at capture time, so without a
    rebuild an offline replay of old shards would always measure the control.
    Rows without an SF eval (``has_sf_wdl == 0`` / missing) fall back to the
    ternary outcome. No-op when ``blend_frac <= 0`` (the stored target already
    equals the ternary HL-Gauss) or when the required fields are absent.

    Cost boundary: per-row numpy HL-Gauss, same as the live finalize path.
    Acceptable for the flag-gated offline use (overlapped by the host prefetch
    thread); vectorize before it could become a live-training default.
    """
    if float(params.blend_frac) <= 0.0:
        return arrs
    if "categorical_target" not in arrs or "wdl_target" not in arrs:
        return arrs
    wdl = np.asarray(arrs["wdl_target"]).reshape(-1)
    n = int(wdl.shape[0])
    cat = np.array(arrs["categorical_target"], copy=True)
    if n == 0 or cat.ndim != 2:
        return arrs
    num_bins = int(cat.shape[1])  # match stored width so the head's shape holds
    sf_wdl = arrs.get("sf_wdl")
    has_sf = np.asarray(
        arrs.get("has_sf_wdl", np.zeros((n,), dtype=np.float32))
    ).reshape(-1)
    for i in range(n):
        scalar_v = 1.0 if int(wdl[i]) == 0 else (0.0 if int(wdl[i]) == 1 else -1.0)
        row = (
            np.asarray(sf_wdl[i])
            if sf_wdl is not None and i < has_sf.shape[0] and bool(has_sf[i])
            else None
        )
        value = categorical_target_value(scalar_v, row, blend_frac=params.blend_frac)
        cat[i] = hlgauss_target(
            value, num_bins=num_bins, sigma=params.sigma,
        ).astype(cat.dtype, copy=False)
    arrs["categorical_target"] = cat
    return arrs
