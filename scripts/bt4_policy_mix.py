#!/usr/bin/env python3
"""Bank raw BT4 priors and arithmetically mix them into policy targets.

This is an offline tool for NNUE-bootstrap control corpora:

``label``
    Run one BT4 network evaluation per stored position, using the replay row's
    true LC0 history planes.  Write a reusable, row-aligned Zarr sidecar.  The
    sidecar stores the legal-normalized raw BT4 prior in float32 and a
    fingerprint of every source row; it never modifies the source corpus.

``mix``
    Copy the source corpus and replace only ``policy_target`` with
    either a global arithmetic mix or an audit-approved source-set treatment.
    Set treatments preserve the source mass on exact or near maxima and let a
    separately temperature-scaled BT4 prior redistribute only ``alpha`` of
    that mass.  Stockfish-derived value targets and every other replay column
    are copied unchanged.  Source and sidecar fingerprints are checked before
    a shard is edited, and the completed output is published by one directory
    rename.

``audit``
    Reconstruct the stored d9 target on the frozen deep-SF audit set, apply the
    exact same treatment, and bank paired per-position regret observations and
    a pre-training gate verdict. With ``--sf-audit-mode descriptive``, an
    immutable ``--experiment-record`` identifies why playing strength rather
    than SF agreement decides admission. Fidelity and lineage still must pass;
    materialization requires the same mode and unchanged record bytes.

The raw sidecar is intentionally separate from the mixed corpus.  Once BT4 has
been evaluated, alternative arithmetic weights can be materialized without
another GPU pass.  This is not the older R/V/G geometric target surgery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np
import zarr
from numcodecs import Blosc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_anti_engine.encoding.lc0 import x_to_lc0_planes
from chess_anti_engine.eval.audit import (
    PHASE_NAMES,
    SOURCE_NAMES,
    expected_and_top1_regret,
    load_audit_set,
    move_regrets,
)
from chess_anti_engine.eval.rvg_surgery import FINGERPRINT_BYTES, position_fingerprints
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE, POLICY_ENCODING_LC0_1858
from chess_anti_engine.moves.leela_index import compact_index_for_move
from chess_anti_engine.replay.shard import iter_shard_paths
from chess_anti_engine.stockfish.wdl import cp_to_wdl_array, mate_to_effective_cp
from scripts import sf_d9_rank_sidecar as sf_ranks
from scripts.bt4_policy_dump import (
    DEFAULT_ONNX,
    board_from_stored_x,
    file_sha256,
    legal_move_policy,
    open_session,
    remap_provenance,
    resolve_policy_output,
    shard_encoding,
)


SIDECAR_SCHEMA = 1
MIX_SCHEMA = 1
SIDECAR_SUMMARY = "bt4_policy_sidecar_summary.json"
MIX_SUMMARY = "bt4_policy_mix_summary.json"
DERIVE_SUMMARY = "derive_targets_summary.json"
POLICY_FIELD = "policy_target"
SIDECAR_POLICY_FIELD = "bt4_policy"
SIDECAR_KEY_FIELD = "source_key"
MIX_SCOPES = ("global", "top-max-ties", "near-max-ratio", "sf-cp-window")
TREATMENT_ALGORITHMS = {
    "global": "legal-normalized-global-arithmetic-v1",
    "top-max-ties": "stored-top-set-only-v1",
    "near-max-ratio": "stored-near-max-set-only-v1",
    "sf-cp-window": "stored-top-ties-union-sf-d9-cp-window-v1",
}
SOURCE_TARGET_CONTRACT = {
    "scheme": "uniform-d9",
    "depth": 9,
    "policy_temp": 0.0005,
    "floor": 0.0,
    "cp_slope": 0.006,
    "cp_draw_width": 120.0,
    "policy_encoding": POLICY_ENCODING_LC0_1858,
    "storage_dtype": "float16",
}
AUDIT_RULER_CONTRACT = {
    "audit_set_sha256": "d8e26efa0b010450abf9374693afc45027db6d146571785ab897af5061144df2",
    "d9_labels_sha256": "0f56bdc0aa453b6dbfdad5cf1744e4937b3eeae274f6b1051d683dd4e8aa4f64",
    "bt4_cache_sha256": "622cdfeda7d71c211e57719ba4d0807252934e6c46d0acfadcc098c251168294",
    "bootstrap_unit": "position (frozen audit set carries no game id)",
    "bootstrap_replicates": 10_000,
    "bootstrap_seed": 20260903,
}
SELECTED_MASS_DRIFT_MEAN_MAX = 0.0002
SELECTED_MASS_DRIFT_ROW_MAX = 0.005
# A legal distribution is normalized before float16 storage. Its total may
# round by more than the set-scoped mean bound; one float16 epsilon covers
# rounding at unit mass, including underflow across the 1858 legal slots.
GLOBAL_MASS_DRIFT_MAX = float(np.finfo(np.float16).eps)
_COMPRESSOR = Blosc(cname="zstd", clevel=2, shuffle=Blosc.BITSHUFFLE)


def validate_alpha(alpha: float) -> float:
    """Return a real treatment weight, refusing identity or extrapolation."""
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(
            f"--alpha must be finite, positive, and at most 1, got {alpha!r}",
        )
    return value


def validate_bt4_temperature(temperature: float) -> float:
    """Return a finite positive teacher temperature."""
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "--bt4-temperature must be finite and positive, "
            f"got {temperature!r}",
        )
    return value


def validate_near_max_ratio(ratio: float) -> float:
    """Return a strict near-maximum probability ratio."""
    value = float(ratio)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            "--near-max-ratio must be finite, positive, and below 1, "
            f"got {ratio!r}",
        )
    return value


def validate_sf_rank_cap(rank_cap: int) -> int:
    value = int(rank_cap)
    if value < 2 or value > 255:
        raise ValueError(f"--sf-rank-cap must be in [2,255], got {rank_cap!r}")
    return value


def validate_sf_cp_window(window_cp: float) -> float:
    value = float(window_cp)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"--sf-cp-window must be finite and positive, got {window_cp!r}",
        )
    return value


def treatment_spec(
    *,
    scope: str,
    alpha: float,
    bt4_temperature: float,
    near_max_ratio: float,
    sf_rank_cap: int = 3,
    sf_cp_window: float = 10.0,
) -> dict[str, Any]:
    """Return the complete, mechanically comparable treatment identity."""
    if scope not in MIX_SCOPES:
        raise ValueError(f"unknown mix scope {scope!r}; expected one of {MIX_SCOPES}")
    weight = validate_alpha(alpha)
    temperature = validate_bt4_temperature(bt4_temperature)
    ratio = (
        validate_near_max_ratio(near_max_ratio)
        if scope == "near-max-ratio"
        else None
    )
    rank_cap = validate_sf_rank_cap(sf_rank_cap) if scope == "sf-cp-window" else None
    cp_window = (
        validate_sf_cp_window(sf_cp_window) if scope == "sf-cp-window" else None
    )
    spec = {
        "scope": scope,
        "alpha": weight,
        "algorithm": TREATMENT_ALGORITHMS[scope],
        "bt4_temperature": temperature,
        "near_max_ratio": ratio,
    }
    if scope == "sf-cp-window":
        spec.update({"sf_rank_cap": rank_cap, "sf_cp_window": cp_window})
    return spec


def _sha_array(value: np.ndarray) -> str:
    data = np.ascontiguousarray(value)
    return hashlib.sha256(data.tobytes(order="C")).hexdigest()


def _source_keys(x: np.ndarray, encoding: str) -> np.ndarray:
    keys = position_fingerprints(x, input_history_encoding=encoding)
    if len(keys) != int(x.shape[0]) or any(
        len(key) != FINGERPRINT_BYTES for key in keys
    ):
        raise ValueError("source row fingerprints have the wrong count or width")
    if not keys:
        return np.zeros((0, FINGERPRINT_BYTES), dtype=np.uint8)
    return (
        np.frombuffer(b"".join(keys), dtype=np.uint8)
        .reshape(
            len(keys),
            FINGERPRINT_BYTES,
        )
        .copy()
    )


def _require_columns(group: Any, shard: Path) -> None:
    missing = [
        field
        for field in (
            "x",
            POLICY_FIELD,
            "legal_mask",
            "has_policy",
            "has_legal_mask",
            "wdl_target",
            "search_wdl",
            "has_search_wdl",
        )
        if field not in group
    ]
    if missing:
        raise ValueError(f"{shard} is missing required columns {missing}")
    rows = int(group["x"].shape[0])
    for flag in ("has_policy", "has_legal_mask", "has_search_wdl"):
        values = np.asarray(group[flag][:])
        if values.shape != (rows,) or not bool(np.all(values != 0)):
            raise ValueError(f"{shard}: {flag} must be active on all {rows} rows")


def _source_policy_layout(source_paths: list[Path]) -> tuple[int, set[str]]:
    """Validate the stored policy layout and return corpus rows/dtypes."""
    rows = 0
    dtypes: set[str] = set()
    for path in source_paths:
        group: Any = zarr.open_group(str(path), mode="r")
        policy = group[POLICY_FIELD]
        legal = group["legal_mask"]
        if policy.ndim != 2 or int(policy.shape[1]) != COMPACT_POLICY_SIZE:
            raise ValueError(
                f"{path}: {POLICY_FIELD} must have shape "
                f"(N, {COMPACT_POLICY_SIZE}), got {policy.shape}",
            )
        if tuple(legal.shape) != tuple(policy.shape):
            raise ValueError(
                f"{path}: legal_mask shape {legal.shape} does not match "
                f"{POLICY_FIELD} {policy.shape}",
            )
        if int(group["x"].shape[0]) != int(policy.shape[0]):
            raise ValueError(f"{path}: x and {POLICY_FIELD} row counts differ")
        rows += int(policy.shape[0])
        dtypes.add(np.dtype(policy.dtype).name)
    return rows, dtypes


def _normalized_legal(
    policy: np.ndarray,
    legal_mask: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    p = np.asarray(policy, dtype=np.float64)
    legal = np.asarray(legal_mask) != 0
    if p.shape != legal.shape or p.ndim != 2:
        raise ValueError(
            f"{name}: policy and legal mask shapes differ: {p.shape} vs {legal.shape}"
        )
    if not np.isfinite(p).all() or bool(np.any(p < 0.0)):
        raise ValueError(f"{name}: policy contains negative or non-finite values")
    p = np.where(legal, p, 0.0)
    totals = p.sum(axis=1)
    bad = np.flatnonzero(~np.isfinite(totals) | (totals <= 0.0))
    if bad.size:
        raise ValueError(f"{name}: {bad.size} rows have no positive legal mass")
    return p / totals[:, None]


def _tempered_bt4_policy(
    bt4: np.ndarray,
    legal_mask: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    """Legal-normalize and temperature-scale BT4 without reviving zero mass."""
    temp = validate_bt4_temperature(temperature)
    legal = np.asarray(legal_mask) != 0
    policy = _normalized_legal(bt4, legal, name="BT4 policy")
    if temp == 1.0:
        return policy

    positive = legal & (policy > 0.0)
    logits = np.full(policy.shape, -np.inf, dtype=np.float64)
    logits[positive] = np.log(policy[positive]) / temp
    row_max = np.max(logits, axis=1, keepdims=True)
    scaled = np.where(positive, np.exp(logits - row_max), 0.0)
    return _normalized_legal(scaled, legal, name="temperature-scaled BT4 policy")


def _source_candidate_set(
    source: np.ndarray,
    legal_mask: np.ndarray,
    *,
    scope: str,
    near_max_ratio: float,
    sf_rank_indices: np.ndarray | None = None,
    sf_rank_gaps_cp: np.ndarray | None = None,
    sf_rank_cap: int = 3,
    sf_cp_window: float = 10.0,
) -> np.ndarray:
    """Select the stored-source set whose mass BT4 may redistribute."""
    legal = np.asarray(legal_mask) != 0
    if scope == "global":
        return legal
    source_values = np.asarray(source, dtype=np.float64)
    legal_source = np.where(legal, source_values, -np.inf)
    row_max = np.max(legal_source, axis=1, keepdims=True)
    if scope == "top-max-ties":
        return legal & (legal_source == row_max)
    if scope == "near-max-ratio":
        ratio = validate_near_max_ratio(near_max_ratio)
        return legal & (legal_source >= row_max * ratio)
    if scope == "sf-cp-window":
        if sf_rank_indices is None or sf_rank_gaps_cp is None:
            raise ValueError("sf-cp-window requires d9 rank indices and cp gaps")
        indices = np.asarray(sf_rank_indices)
        gaps_cp = np.asarray(sf_rank_gaps_cp, dtype=np.float64)
        if indices.ndim != 2 or gaps_cp.shape != indices.shape:
            raise ValueError(
                "d9 rank indices/gaps must be same-shaped two-dimensional arrays",
            )
        if indices.shape[0] != source_values.shape[0]:
            raise ValueError("d9 rank sidecar row count does not match source policy")
        rank_cap = validate_sf_rank_cap(sf_rank_cap)
        if indices.shape[1] < rank_cap:
            raise ValueError(
                f"d9 rank sidecar width {indices.shape[1]} is below cap {rank_cap}",
            )
        cp_window = validate_sf_cp_window(sf_cp_window)
        selected = legal & (legal_source == row_max)
        rows = np.arange(indices.shape[0])
        for rank in range(rank_cap):
            compact = indices[:, rank].astype(np.int64, copy=False)
            include = (
                (compact >= 0)
                & (compact < source_values.shape[1])
                & np.isfinite(gaps_cp[:, rank])
                & (gaps_cp[:, rank] <= cp_window)
            )
            selected[rows[include], compact[include]] = True
        if bool(np.any(selected & ~legal)):
            raise ValueError("d9 cp-window selected an illegal move")
        return selected
    raise ValueError(f"scope {scope!r} has no source candidate set")


def arithmetic_policy_mix(
    source: np.ndarray,
    bt4: np.ndarray,
    legal_mask: np.ndarray,
    *,
    alpha: float,
    bt4_temperature: float = 1.0,
) -> np.ndarray:
    """Literal probability mixture, legal-normalized on both input teachers."""
    weight = validate_alpha(alpha)
    sf = _normalized_legal(source, legal_mask, name="source policy")
    lc0 = _tempered_bt4_policy(
        bt4,
        legal_mask,
        temperature=bt4_temperature,
    )
    mixed = (1.0 - weight) * sf + weight * lc0
    mixed = _normalized_legal(mixed, legal_mask, name="mixed policy")
    return mixed.astype(np.float32)


def mix_policy_targets(
    source: np.ndarray,
    bt4: np.ndarray,
    legal_mask: np.ndarray,
    *,
    alpha: float,
    scope: str,
    bt4_temperature: float = 1.0,
    near_max_ratio: float = 0.5,
    sf_rank_indices: np.ndarray | None = None,
    sf_rank_gaps_cp: np.ndarray | None = None,
    sf_rank_cap: int = 3,
    sf_cp_window: float = 10.0,
) -> np.ndarray:
    """Apply the selected literal probability-space treatment.

    Set-scoped treatments preserve the source target's total mass on the
    selected set and use BT4 only to redistribute ``alpha`` of that mass.
    ``top-max-ties`` selects exact stored maxima; ``near-max-ratio`` also
    selects moves within the requested fraction of the stored maximum.  A
    one-move selected set is byte-identical after storage.
    """
    if scope not in MIX_SCOPES:
        raise ValueError(f"unknown mix scope {scope!r}; expected one of {MIX_SCOPES}")
    weight = validate_alpha(alpha)
    if scope == "global":
        return arithmetic_policy_mix(
            source,
            bt4,
            legal_mask,
            alpha=weight,
            bt4_temperature=bt4_temperature,
        )

    source_stored = np.asarray(source)
    legal = np.asarray(legal_mask) != 0
    _normalized_legal(source_stored, legal, name="source policy")
    if bool(np.any(source_stored[~legal] != 0.0)):
        raise ValueError("source policy has nonzero mass on illegal moves")
    lc0 = _normalized_legal(bt4, legal, name="BT4 policy")
    selected = _source_candidate_set(
        source_stored,
        legal,
        scope=scope,
        near_max_ratio=near_max_ratio,
        sf_rank_indices=sf_rank_indices,
        sf_rank_gaps_cp=sf_rank_gaps_cp,
        sf_rank_cap=sf_rank_cap,
        sf_cp_window=sf_cp_window,
    )
    selected_count = selected.sum(axis=1)
    if bool(np.any(selected_count <= 0)):
        raise ValueError("source policy has a row without a selected legal move")

    bt4_selected_raw = np.where(selected, lc0, 0.0)
    no_bt4_selected_mass = np.flatnonzero(
        bt4_selected_raw.sum(axis=1) <= 0.0,
    )
    if no_bt4_selected_mass.size:
        set_name = {
            "top-max-ties": "source top-tie",
            "near-max-ratio": "source near-max",
            "sf-cp-window": "source d9 cp-window",
        }[scope]
        raise ValueError(
            f"BT4 policy has no mass on the {set_name} set for "
            f"{no_bt4_selected_mass.size} rows",
        )
    bt4_selected = _tempered_bt4_policy(
        bt4_selected_raw,
        selected,
        temperature=bt4_temperature,
    )
    source_values = source_stored.astype(np.float64, copy=False)
    source_selected = np.where(selected, source_values, 0.0)
    selected_mass = source_selected.sum(axis=1, keepdims=True)
    redistributed = (
        (1.0 - weight) * source_selected
        + weight * selected_mass * bt4_selected
    )
    mixed = source_values.copy()
    mixed[selected] = redistributed[selected]
    # Make one-move selected rows exact identities, not merely floating-point
    # reconstructions. Replay policies are float16 and their rounded row sum
    # need not be exactly one.
    singleton = selected_count == 1
    mixed[singleton] = source_stored[singleton]
    return mixed.astype(np.float32)


def _entropy_sum(policy: np.ndarray) -> float:
    p = np.asarray(policy, dtype=np.float64)
    positive = p > 0.0
    return float(
        -np.sum(np.where(positive, p * np.log(np.where(positive, p, 1.0)), 0.0))
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".writing")
    if tmp.exists():
        raise FileExistsError(f"stale temporary summary exists: {tmp}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def functional_remap_identity(provenance: object) -> dict[str, Any]:
    """Return remap-relevant provenance, excluding unrelated repository HEAD."""
    if not isinstance(provenance, dict):
        raise ValueError("remap provenance must be a dictionary")
    commit = provenance.get("commit")
    dirty = provenance.get("dirty")
    blobs = provenance.get("blobs")
    if not isinstance(commit, str) or not commit:
        raise ValueError("remap provenance has no commit")
    if not isinstance(dirty, bool):
        raise ValueError("remap provenance has no boolean dirty flag")
    if not isinstance(blobs, dict) or not blobs or not all(
        isinstance(path, str) and isinstance(blob, str) and path and blob
        for path, blob in blobs.items()
    ):
        raise ValueError("remap provenance has no valid source blob map")
    return {"commit": commit, "dirty": dirty, "blobs": blobs}


def _sidecar_identity(
    source_group: Any,
    source_path: Path,
    *,
    source_x: np.ndarray | None = None,
) -> tuple[str, np.ndarray, str, str]:
    _require_columns(source_group, source_path)
    encoding = shard_encoding(source_group, source_path)
    x = np.asarray(source_group["x"][:] if source_x is None else source_x)
    keys = _source_keys(x, encoding)
    policy = np.asarray(source_group[POLICY_FIELD][:])
    return encoding, keys, _sha_array(keys), _sha_array(policy)


def _validate_sidecar(
    sidecar_path: Path,
    *,
    source_path: Path,
    source_keys: np.ndarray,
    source_key_sha: str,
    source_policy_sha: str,
    onnx_sha: str | None = None,
    providers: list[str] | None = None,
    policy_output: str | None = None,
) -> dict[str, Any]:
    if not sidecar_path.is_dir():
        raise ValueError(f"missing BT4 sidecar shard {sidecar_path}")
    group: Any = zarr.open_group(str(sidecar_path), mode="r")
    attrs = dict(group.attrs)
    expected: dict[str, Any] = {
        "bt4_policy_sidecar_schema": SIDECAR_SCHEMA,
        "source_shard": source_path.name,
        "positions": int(source_keys.shape[0]),
        "source_key_sha256": source_key_sha,
        "source_policy_sha256": source_policy_sha,
        "policy_encoding": POLICY_ENCODING_LC0_1858,
        "policy_size": COMPACT_POLICY_SIZE,
        "teacher_evaluations_per_position": 1,
        "search_nodes": 0,
        "stored_dtype": "float32",
    }
    if onnx_sha is not None:
        expected["onnx_sha256"] = onnx_sha
    if providers is not None:
        expected["providers"] = providers
    if policy_output is not None:
        expected["policy_output"] = policy_output
    bad = {
        key: (attrs.get(key), value)
        for key, value in expected.items()
        if attrs.get(key) != value
    }
    if bad:
        raise ValueError(f"{sidecar_path}: provenance mismatch {bad}")
    if SIDECAR_KEY_FIELD not in group or SIDECAR_POLICY_FIELD not in group:
        raise ValueError(f"{sidecar_path}: missing sidecar arrays")
    key_array = group[SIDECAR_KEY_FIELD]
    if tuple(key_array.shape) != tuple(source_keys.shape) or np.dtype(
        key_array.dtype
    ) != np.dtype(np.uint8):
        raise ValueError(
            f"{sidecar_path}: source key layout does not match "
            f"{source_keys.shape}/uint8",
        )
    stored_keys = np.asarray(key_array[:], dtype=np.uint8)
    if not np.array_equal(stored_keys, source_keys):
        raise ValueError(f"{sidecar_path}: row fingerprints do not match {source_path}")
    raw_array = group[SIDECAR_POLICY_FIELD]
    source_group: Any = zarr.open_group(str(source_path), mode="r")
    if tuple(raw_array.shape) != tuple(source_group[POLICY_FIELD].shape):
        raise ValueError(
            f"{sidecar_path}: policy shape {raw_array.shape} does not match source"
        )
    if np.dtype(raw_array.dtype) != np.dtype(np.float32):
        raise ValueError(f"{sidecar_path}: BT4 policy storage must be float32")
    raw = np.asarray(raw_array[:], dtype=np.float32)
    if not np.isfinite(raw).all() or bool(np.any(raw < 0.0)):
        raise ValueError(
            f"{sidecar_path}: BT4 policy contains negative or non-finite values"
        )
    return attrs


def _source_game_ply_sha(source_group: Any, source_path: Path) -> str:
    for field in ("game_id", "ply_index", "has_game_id", "has_ply_index"):
        if field not in source_group:
            raise ValueError(f"{source_path}: missing row-identity field {field}")
    if not bool(np.all(np.asarray(source_group["has_game_id"][:]) != 0)) or not bool(
        np.all(np.asarray(source_group["has_ply_index"][:]) != 0)
    ):
        raise ValueError(f"{source_path}: game/ply identity is not active on every row")
    game_ids = np.asarray(source_group["game_id"][:], dtype=np.int64)
    plies = np.asarray(source_group["ply_index"][:], dtype=np.int32)
    return sf_ranks._sha_arrays(game_ids, plies)


def _validate_sf_rank_sidecar(
    rank_path: Path,
    *,
    source_group: Any,
    source_path: Path,
    source_summary_sha256: str,
    required_top_k: int,
) -> dict[str, Any]:
    if not rank_path.is_dir():
        raise ValueError(f"missing SF d9 rank sidecar shard {rank_path}")
    group: Any = zarr.open_group(str(rank_path), mode="r")
    attrs = dict(group.attrs)
    rows = int(source_group[POLICY_FIELD].shape[0])
    expected = {
        "sf_d9_rank_sidecar_schema": sf_ranks.SCHEMA,
        "source_shard": source_path.name,
        "source_rows": rows,
        "source_row_identity_sha256": _source_game_ply_sha(
            source_group,
            source_path,
        ),
        "source_derive_summary_sha256": source_summary_sha256,
        "depth": 9,
        "index_encoding": POLICY_ENCODING_LC0_1858,
        "gap_definition": "rank1_effective_cp-minus-ranked_effective_cp",
    }
    bad = {
        key: (attrs.get(key), value)
        for key, value in expected.items()
        if attrs.get(key) != value
    }
    top_k = int(attrs.get("top_k", -1))
    if top_k < required_top_k:
        bad["top_k"] = (top_k, f">={required_top_k}")
    if bad:
        raise ValueError(f"{rank_path}: provenance mismatch {bad}")
    expected_layout = {
        sf_ranks.INDEX_FIELD: ((rows, top_k), np.dtype(np.uint16)),
        sf_ranks.GAP_FIELD: ((rows, top_k), np.dtype(np.float32)),
        sf_ranks.COUNT_FIELD: ((rows,), np.dtype(np.uint8)),
    }
    for field, (shape, dtype) in expected_layout.items():
        if field not in group:
            raise ValueError(f"{rank_path}: missing {field}")
        array = group[field]
        if tuple(array.shape) != shape or np.dtype(array.dtype) != dtype:
            raise ValueError(
                f"{rank_path}: {field} is {array.shape}/{array.dtype}, "
                f"expected {shape}/{dtype}",
            )
    indices = np.asarray(group[sf_ranks.INDEX_FIELD][:], dtype=np.uint16)
    gaps = np.asarray(group[sf_ranks.GAP_FIELD][:], dtype=np.float32)
    counts = np.asarray(group[sf_ranks.COUNT_FIELD][:], dtype=np.uint8)
    if sf_ranks._sha_arrays(indices, gaps, counts) != attrs.get("payload_sha256"):
        raise ValueError(f"{rank_path}: payload digest mismatch")
    if bool(np.any(counts < 1)) or bool(np.any(counts > top_k)):
        raise ValueError(f"{rank_path}: rank counts must be between one and top-k")
    valid = np.arange(top_k)[None, :] < counts[:, None]
    if bool(np.any(indices[~valid] != sf_ranks.INVALID_INDEX)) or not bool(
        np.all(np.isinf(gaps[~valid]))
    ):
        raise ValueError(f"{rank_path}: malformed padding")
    if bool(np.any(indices[valid] >= COMPACT_POLICY_SIZE)) or not np.isfinite(
        gaps[valid]
    ).all():
        raise ValueError(f"{rank_path}: invalid rank payload")
    if bool(np.any(gaps[valid] < 0.0)) or bool(np.any(gaps[:, 0] != 0.0)):
        raise ValueError(f"{rank_path}: invalid rank gaps")
    for row_index, count in enumerate(counts.astype(np.int64)):
        if np.unique(indices[row_index, :count]).size != count:
            raise ValueError(f"{rank_path}: repeated ranked move")
        if bool(np.any(np.diff(gaps[row_index, :count]) < -1e-6)):
            raise ValueError(f"{rank_path}: rank gaps are not nondecreasing")
    legal = np.asarray(source_group["legal_mask"][:]) != 0
    row_index = np.repeat(np.arange(rows), counts.astype(np.int64))
    if not bool(np.all(legal[row_index, indices[valid].astype(np.int64)])):
        raise ValueError(f"{rank_path}: ranked move is illegal")
    stored_policy = np.asarray(source_group[POLICY_FIELD][:], dtype=np.float32)
    legal_policy = np.where(legal, stored_policy, -np.inf)
    if not bool(
        np.all(
            legal_policy[np.arange(rows), indices[:, 0].astype(np.int64)]
            == np.max(legal_policy, axis=1)
        )
    ):
        raise ValueError(f"{rank_path}: rank-1 is not a stored-policy maximum")
    return attrs


@dataclass
class LabelStats:
    rows: int = 0
    shards: int = 0
    entropy_sum: float = 0.0
    top1_sum: float = 0.0
    legal_moves_sum: int = 0

    def add_attrs(self, attrs: dict[str, Any]) -> None:
        self.rows += int(attrs["positions"])
        self.shards += 1
        self.entropy_sum += float(attrs["bt4_entropy_sum"])
        self.top1_sum += float(attrs["bt4_top1_sum"])
        self.legal_moves_sum += int(attrs["legal_moves_sum"])


def label_sidecar(args: argparse.Namespace) -> int:
    source_dir = Path(args.shards).resolve()
    out_dir = Path(args.out).resolve()
    if source_dir == out_dir or source_dir in out_dir.parents:
        raise SystemExit("--out must be separate from, not inside, --shards")
    source_paths = iter_shard_paths(source_dir)
    if not source_paths:
        raise SystemExit(f"{source_dir} has no replay shards")
    if out_dir.exists() and not args.resume:
        raise SystemExit(f"{out_dir} exists; pass --resume or use a fresh path")
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_sha = file_sha256(args.onnx)
    sess, input_name, input_dtype, providers = open_session(
        str(args.onnx),
        gpu_mem_gb=float(args.gpu_mem_gb),
        threads=int(args.threads),
    )
    policy_index = resolve_policy_output(sess, args.policy_output)
    policy_name = sess.get_outputs()[policy_index].name
    print(f"[bt4-sidecar] providers={providers} policy={policy_name}", flush=True)

    stats = LabelStats()
    started = time.time()
    for number, source_path in enumerate(source_paths, start=1):
        source: Any = zarr.open_group(str(source_path), mode="r")
        x = np.asarray(source["x"][:])
        encoding, keys, key_sha, policy_sha = _sidecar_identity(
            source,
            source_path,
            source_x=x,
        )
        rows = int(x.shape[0])
        width = int(source[POLICY_FIELD].shape[1])
        if width != COMPACT_POLICY_SIZE:
            raise ValueError(
                f"{source_path}: {POLICY_FIELD} width must be "
                f"{COMPACT_POLICY_SIZE}, got {width}",
            )
        if tuple(source["legal_mask"].shape) != tuple(source[POLICY_FIELD].shape):
            raise ValueError(
                f"{source_path}: legal_mask shape {source['legal_mask'].shape} does "
                f"not match {POLICY_FIELD} {source[POLICY_FIELD].shape}",
            )
        target_path = out_dir / source_path.name
        if target_path.exists():
            if not args.resume:
                raise SystemExit(f"{target_path} exists without --resume")
            attrs = _validate_sidecar(
                target_path,
                source_path=source_path,
                source_keys=keys,
                source_key_sha=key_sha,
                source_policy_sha=policy_sha,
                onnx_sha=onnx_sha,
                providers=providers,
                policy_output=policy_name,
            )
            stats.add_attrs(attrs)
            print(f"[bt4-sidecar] resume verified {target_path.name}", flush=True)
            continue

        writing = target_path.with_name(target_path.name + ".writing")
        if writing.exists():
            raise SystemExit(f"stale partial sidecar exists: {writing}")
        legal_mask = np.asarray(source["legal_mask"][:])
        raw_policy = np.zeros((rows, width), dtype=np.float32)
        entropy_sum = 0.0
        top1_sum = 0.0
        legal_moves_sum = 0

        batch_size = int(args.batch_size)
        for start in range(0, rows, batch_size):
            stop = min(rows, start + batch_size)
            planes = x_to_lc0_planes(
                x[start:stop],
                input_history_encoding=encoding,
            )
            boards = [
                board_from_stored_x(
                    x[row],
                    planes[row - start],
                    input_history_encoding=encoding,
                )
                for row in range(start, stop)
            ]
            for offset, board in enumerate(boards):
                row = start + offset
                expected = {
                    compact_index_for_move(board, move) for move in board.legal_moves
                }
                stored = set(np.flatnonzero(legal_mask[row] > 0).tolist())
                if expected != stored:
                    raise ValueError(
                        f"{source_path}:{row}: decoded legal moves disagree with legal_mask",
                    )
            feats = planes.astype(input_dtype, copy=False)
            output = sess.run([policy_name], {input_name: feats})[0]
            logits = np.asarray(output, dtype=np.float32)
            for offset, board in enumerate(boards):
                row = start + offset
                ucis, probs = legal_move_policy(board, logits[offset])
                indices = [
                    compact_index_for_move(board, chess.Move.from_uci(uci))
                    for uci in ucis
                ]
                if len(set(indices)) != len(indices) or any(
                    index < 0 for index in indices
                ):
                    raise ValueError(
                        f"{source_path}:{row}: compact policy mapping is not one-to-one"
                    )
                raw_policy[row, np.asarray(indices, dtype=np.int64)] = probs.astype(
                    np.float32
                )
                entropy_sum += _entropy_sum(probs[None, :])
                top1_sum += float(probs.max())
                legal_moves_sum += len(indices)

        normalized = _normalized_legal(raw_policy, legal_mask, name=str(target_path))
        if float(np.max(np.abs(normalized - raw_policy))) > 2e-6:
            raise ValueError(f"{target_path}: generated BT4 policy is not normalized")
        group: Any = zarr.open_group(str(writing), mode="w")
        row_chunk = min(512, max(1, rows))
        group.create_dataset(
            SIDECAR_KEY_FIELD,
            data=keys,
            chunks=(row_chunk, FINGERPRINT_BYTES),
            compressor=_COMPRESSOR,
        )
        group.create_dataset(
            SIDECAR_POLICY_FIELD,
            data=raw_policy,
            chunks=(row_chunk, width),
            compressor=_COMPRESSOR,
        )
        attrs = {
            "bt4_policy_sidecar_schema": SIDECAR_SCHEMA,
            "source_shard": source_path.name,
            "source_dir": str(source_dir),
            "positions": rows,
            "source_key_sha256": key_sha,
            "source_policy_sha256": policy_sha,
            "input_history_encoding": encoding,
            "policy_encoding": POLICY_ENCODING_LC0_1858,
            "policy_size": width,
            "onnx_path": str(Path(args.onnx).resolve()),
            "onnx_sha256": onnx_sha,
            "policy_output": policy_name,
            "providers": providers,
            "teacher_evaluations_per_position": 1,
            "search_nodes": 0,
            "stored_dtype": "float32",
            "bt4_entropy_sum": entropy_sum,
            "bt4_top1_sum": top1_sum,
            "legal_moves_sum": legal_moves_sum,
        }
        group.attrs.update(attrs)
        os.replace(writing, target_path)
        stats.add_attrs(attrs)
        elapsed = max(time.time() - started, 1e-9)
        print(
            f"[bt4-sidecar] {number}/{len(source_paths)} shards, {stats.rows} rows, "
            f"{stats.rows / elapsed:.1f} pos/s",
            flush=True,
        )

    source_names = {path.name for path in source_paths}
    extras = sorted(
        path.name
        for path in out_dir.glob("shard_*.zarr")
        if path.name not in source_names
    )
    if extras:
        raise SystemExit(
            f"{out_dir} has sidecar shards absent from source: {extras[:10]}"
        )
    summary = {
        "schema": SIDECAR_SCHEMA,
        "kind": "bt4_raw_legal_policy_sidecar",
        "source_dir": str(source_dir),
        "source_shards": len(source_paths),
        "rows": stats.rows,
        "sidecar_shards": stats.shards,
        "policy_encoding": POLICY_ENCODING_LC0_1858,
        "onnx": {"path": str(Path(args.onnx).resolve()), "sha256": onnx_sha},
        "policy_output": policy_name,
        "providers": providers,
        "teacher_evaluations_per_position": 1,
        "search_nodes": 0,
        "stored_dtype": "float32",
        "mean_entropy_nats": stats.entropy_sum / stats.rows,
        "mean_top1_probability": stats.top1_sum / stats.rows,
        "mean_legal_moves": stats.legal_moves_sum / stats.rows,
        "remap": remap_provenance(),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if stats.shards != len(source_paths):
        raise SystemExit(
            f"sidecar completeness failure: {stats.shards}/{len(source_paths)} shards",
        )
    _atomic_json(out_dir / SIDECAR_SUMMARY, summary)
    print(f"[bt4-sidecar] complete: {stats.rows} rows -> {out_dir}", flush=True)
    return 0


@dataclass
class MixStats:
    rows: int = 0
    shards: int = 0
    changed_rows: int = 0
    source_top_tied_rows: int = 0
    source_candidate_multi_rows: int = 0
    candidate_set_wider_rows: int = 0
    changed_unique_max_rows: int = 0
    source_bt4_top1_agree: int = 0
    mixed_source_top1_agree: int = 0
    mixed_bt4_top1_agree: int = 0
    source_top1_ge_0_99_rows: int = 0
    mixed_top1_ge_0_99_rows: int = 0
    source_entropy_sum: float = 0.0
    bt4_entropy_sum: float = 0.0
    mixed_entropy_sum: float = 0.0
    l1_from_source_sum: float = 0.0
    selected_mass_abs_drift_sum: float = 0.0
    selected_mass_abs_drift_max: float = 0.0


def _effective_cp(cp: int | None, mate: int | None) -> float:
    if mate is not None:
        return float(mate_to_effective_cp(int(mate)))
    if cp is not None:
        return float(cp)
    raise ValueError("Stockfish line has neither cp nor mate")


def _stored_d9_policy(
    legal_ucis: list[str],
    lines: list[list[Any]],
) -> np.ndarray:
    scores: dict[str, float] = {}
    for line in lines:
        if len(line) < 4:
            raise ValueError(f"malformed d9 line {line!r}")
        _rank, move, cp, mate = line[:4]
        scores.setdefault(str(move), _effective_cp(cp, mate))
    if set(scores) != set(legal_ucis):
        missing = sorted(set(legal_ucis) - set(scores))
        extra = sorted(set(scores) - set(legal_ucis))
        raise ValueError(
            f"d9 full-width move-set mismatch: missing={missing[:5]} extra={extra[:5]}",
        )
    eff_cp = np.asarray([scores[move] for move in legal_ucis], dtype=np.float64)
    wdl = cp_to_wdl_array(eff_cp, slope=0.006, draw_width_cp=120.0)
    q = wdl[:, 0].astype(np.float64) - wdl[:, 2].astype(np.float64)
    scaled = q / 0.0005
    probs = np.exp(scaled - float(np.max(scaled)))
    probs /= float(probs.sum())
    # Reproduce derive_corpus_targets.shard_stored: float64 -> float32 ->
    # float16. The corpus is mixed from what the trainer actually reads.
    return probs.astype(np.float32).astype(np.float16).astype(np.float32)


def _audit_d9_rank_arrays(
    legal_ucis: Sequence[str],
    lines: Sequence[Sequence[Any]],
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Represent audit-format d9 ranks in the row-local policy ordering."""
    by_move = {move: index for index, move in enumerate(legal_ucis)}
    ranked = sorted(lines, key=lambda line: int(line[0]))
    indices = np.full((1, top_k), -1, dtype=np.int64)
    gaps_cp = np.full((1, top_k), np.inf, dtype=np.float64)
    if not ranked:
        raise ValueError("audit d9 block is empty")
    best = _effective_cp(ranked[0][2], ranked[0][3])
    for offset, line in enumerate(ranked[:top_k]):
        if len(line) < 4:
            raise ValueError(f"malformed d9 line {line!r}")
        rank, move, cp, mate = line[:4]
        if int(rank) != offset + 1 or str(move) not in by_move:
            raise ValueError("audit d9 ranks or moves are malformed")
        score = _effective_cp(cp, mate)
        gap = best - score
        if gap < -1e-6:
            raise ValueError("audit d9 ranks disagree with effective-cp scores")
        indices[0, offset] = by_move[str(move)]
        gaps_cp[0, offset] = max(0.0, gap)
    return indices, gaps_cp


def _depth_lines(row: dict[str, Any], depth: int) -> list[list[Any]]:
    matches = [entry for entry in row.get("depths", []) if entry.get("depth") == depth]
    if len(matches) != 1:
        raise ValueError(
            f"{row.get('key')}: expected one complete depth-{depth} block, "
            f"found {len(matches)}",
        )
    return list(matches[0].get("lines", []))


def _load_keyed_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["key"])
            if key in rows:
                raise ValueError(f"{path}:{line_number}: duplicate key {key}")
            rows[key] = row
    return rows


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size < 2 or n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    batch = 256
    for start in range(0, n_boot, batch):
        stop = min(n_boot, start + batch)
        draws = rng.integers(0, vals.size, size=(stop - start, vals.size))
        means[start:stop] = vals[draws].mean(axis=1)
    quantiles = np.quantile(means, [0.025, 0.975])
    return float(quantiles[0]), float(quantiles[1])


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
    }


def _validate_source_target_contract(
    summary: dict[str, Any],
    *,
    storage_dtypes: set[str],
) -> None:
    scheme = summary.get("scheme")
    cp_map = summary.get("cp_map")
    policy = summary.get("policy")
    realized = summary.get("realized")
    if not all(isinstance(value, dict) for value in (scheme, cp_map, policy, realized)):
        raise ValueError("source derive summary is missing target-provenance sections")
    assert isinstance(scheme, dict)
    assert isinstance(cp_map, dict)
    assert isinstance(policy, dict)
    assert isinstance(realized, dict)
    depths = realized.get("realized_base_depth_histogram")
    observed = {
        "scheme": scheme.get("canonical"),
        "depth": 9 if isinstance(depths, dict) and set(depths) == {"9"} else depths,
        "policy_temp": summary.get("temp_requested"),
        "floor": summary.get("floor_requested"),
        "cp_slope": cp_map.get("cp_slope"),
        "cp_draw_width": cp_map.get("cp_draw_width"),
        "policy_encoding": policy.get("encoding"),
        "storage_dtype": (
            next(iter(storage_dtypes))
            if len(storage_dtypes) == 1
            else sorted(storage_dtypes)
        ),
    }
    bad: dict[str, tuple[Any, Any]] = {
        key: (observed.get(key), expected)
        for key, expected in SOURCE_TARGET_CONTRACT.items()
        if observed.get(key) != expected
    }
    if bad:
        raise ValueError(f"source target does not match the audited d9 contract: {bad}")


def _audit_admission(
    mode: str, experiment_record: Path | None,
) -> dict[str, Any]:
    """Identify the declared role of the SF ruler, independently of its result."""
    if mode not in {"gate", "descriptive"}:
        raise ValueError(f"unknown SF audit mode {mode!r}")
    if mode == "gate":
        if experiment_record is not None:
            raise ValueError("--experiment-record requires --sf-audit-mode descriptive")
        return {"mode": mode, "experiment_record": None}
    if experiment_record is None:
        raise ValueError("descriptive SF audit requires --experiment-record")
    path = Path(experiment_record).resolve()
    if not path.is_file() or not path.read_bytes().strip():
        raise ValueError(f"experiment record must be a nonempty file: {path}")
    return {
        "mode": mode,
        "experiment_record": {"path": str(path), "sha256": file_sha256(path)},
    }


def _mass_drift_bounds(scope: str) -> tuple[float, float]:
    if scope == "global":
        return GLOBAL_MASS_DRIFT_MAX, GLOBAL_MASS_DRIFT_MAX
    return SELECTED_MASS_DRIFT_MEAN_MAX, SELECTED_MASS_DRIFT_ROW_MAX


def _load_audit_receipt(
    path: Path,
    *,
    alpha: float,
    scope: str,
    bt4_temperature: float,
    near_max_ratio: float,
    sf_rank_cap: int = 3,
    sf_cp_window: float = 10.0,
    sf_audit_mode: str = "gate",
    experiment_record: Path | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing audit receipt {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": 1,
        "kind": "bt4_policy_mix_frozen_deep_sf_audit",
    }
    bad: dict[str, tuple[Any, Any]] = {
        key: (receipt.get(key), value)
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    admission = _audit_admission(sf_audit_mode, experiment_record)
    recorded_admission = receipt.get(
        "admission", {"mode": "gate", "experiment_record": None},
    )
    if recorded_admission != admission:
        bad["admission"] = (recorded_admission, admission)
    expected_treatment = treatment_spec(
        scope=scope,
        alpha=alpha,
        bt4_temperature=bt4_temperature,
        near_max_ratio=near_max_ratio,
        sf_rank_cap=sf_rank_cap,
        sf_cp_window=sf_cp_window,
    )
    treatment = receipt.get("treatment")
    if not isinstance(treatment, dict):
        bad["treatment"] = (treatment, expected_treatment)
    else:
        for key, expected_value in expected_treatment.items():
            if treatment.get(key) != expected_value:
                bad[f"treatment.{key}"] = (treatment.get(key), expected_value)
    if receipt.get("source_target_contract") != SOURCE_TARGET_CONTRACT:
        bad["source_target_contract"] = (
            receipt.get("source_target_contract"),
            SOURCE_TARGET_CONTRACT,
        )
    ruler = receipt.get("ruler")
    if not isinstance(ruler, dict):
        bad["ruler"] = (ruler, AUDIT_RULER_CONTRACT)
    else:
        for key, expected_value in AUDIT_RULER_CONTRACT.items():
            if ruler.get(key) != expected_value:
                bad[f"ruler.{key}"] = (ruler.get(key), expected_value)
    gate = receipt.get("gate")
    if isinstance(gate, dict) and not audit_training_permitted(
        verdict=str(gate.get("verdict")),
        treatment_invariants_passed=gate.get("treatment_invariants_passed") is True,
        sf_audit_mode=sf_audit_mode,
    ):
        bad["gate.verdict"] = (gate.get("verdict"), f"admissible under {sf_audit_mode}")
    if not isinstance(gate, dict) or gate.get("training_permitted") is not True:
        bad["gate.training_permitted"] = (
            gate.get("training_permitted") if isinstance(gate, dict) else None,
            True,
        )
    if (
        not isinstance(gate, dict)
        or gate.get("treatment_invariants_passed") is not True
    ):
        bad["gate.treatment_invariants_passed"] = (
            gate.get("treatment_invariants_passed")
            if isinstance(gate, dict)
            else None,
            True,
        )
    invariants = receipt.get("treatment_invariants")
    if not isinstance(invariants, dict):
        bad["treatment_invariants"] = (invariants, "dict")
    else:
        required_invariants: dict[str, Any] = {
            "selected_mass_drift_within_bounds": True,
        }
        if scope == "global":
            required_invariants["mass_reference"] = "normalized_total_legal_mass"
        if scope in {"near-max-ratio", "sf-cp-window"}:
            required_invariants.update(
                {
                    "near_max_extended": True,
                    "candidate_set_wider_rows": lambda value: int(value) > 0,
                    "changed_unique_max_rows": lambda value: int(value) > 0,
                }
            )
        if scope == "top-max-ties":
            required_invariants.update(
                {
                    "top_tie_unique_max_identity": True,
                    "changed_unique_max_rows": 0,
                }
            )
        if (
            scope == "top-max-ties"
            and bt4_temperature != 1.0
            and alpha == 1.0
        ):
            required_invariants.update(
                {
                    "temperature_one_top1_preserved": True,
                    "temperature_one_top1_mismatch_rows": 0,
                }
            )
        if (
            scope == "sf-cp-window"
            and bt4_temperature != 1.0
            and alpha == 1.0
        ):
            required_invariants.update(
                {
                    "temperature_rank_preserved_before_storage": True,
                    "temperature_prestorage_top1_mismatch_rows": 0,
                }
            )
        for key, requirement in required_invariants.items():
            observed = invariants.get(key)
            if callable(requirement):
                try:
                    matches = bool(requirement(observed))
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = observed == requirement
            if not matches:
                bad[f"treatment_invariants.{key}"] = (observed, requirement)
    if bad:
        raise ValueError(f"audit receipt does not admit this treatment: {bad}")
    return receipt


def audit_training_permitted(
    *,
    verdict: str,
    treatment_invariants_passed: bool,
    sf_audit_mode: str = "gate",
) -> bool:
    """Fidelity is mandatory; an explicitly descriptive ruler does not gate."""
    if sf_audit_mode not in {"gate", "descriptive"}:
        raise ValueError(f"unknown SF audit mode {sf_audit_mode!r}")
    return (
        verdict in {"graduate_win", "graduate_tie", "kill"}
        and (sf_audit_mode == "descriptive" or verdict != "kill")
        and treatment_invariants_passed
    )


def audit_mix(args: argparse.Namespace) -> int:
    """Score a source/BT4 policy treatment on the frozen deep-SF ruler."""
    admission = _audit_admission(
        str(getattr(args, "sf_audit_mode", "gate")),
        getattr(args, "experiment_record", None),
    )
    alpha = validate_alpha(float(args.alpha))
    scope = str(args.scope)
    if scope not in MIX_SCOPES:
        raise SystemExit(f"--scope must be one of {MIX_SCOPES}")
    bt4_temperature = validate_bt4_temperature(float(args.bt4_temperature))
    near_max_ratio = float(args.near_max_ratio)
    sf_rank_cap = int(getattr(args, "sf_rank_cap", 3))
    sf_cp_window = float(getattr(args, "sf_cp_window", 10.0))
    treatment = treatment_spec(
        scope=scope,
        alpha=alpha,
        bt4_temperature=bt4_temperature,
        near_max_ratio=near_max_ratio,
        sf_rank_cap=sf_rank_cap,
        sf_cp_window=sf_cp_window,
    )
    audit_path = Path(args.audit_set).resolve()
    d9_path = Path(args.d9_labels).resolve()
    bt4_path = Path(args.bt4_cache).resolve()
    out_path = Path(args.json).resolve()
    if out_path.exists():
        raise SystemExit(f"audit receipt already exists: {out_path}")

    positions = load_audit_set(audit_path)
    d9_rows = _load_keyed_jsonl(d9_path)
    bt4_rows = _load_keyed_jsonl(bt4_path)
    expected_keys = {position.key for position in positions}
    for name, rows in (("d9", d9_rows), ("BT4", bt4_rows)):
        if set(rows) != expected_keys:
            raise SystemExit(
                f"{name} key-set mismatch: missing "
                f"{len(expected_keys - set(rows))}, extra "
                f"{len(set(rows) - expected_keys)}",
            )

    per_position: list[dict[str, Any]] = []
    source_metrics: list[tuple[float, float]] = []
    candidate_metrics: list[tuple[float, float]] = []
    for position in positions:
        board = chess.Board(position.fen)
        legal_ucis = [move.uci() for move in board.legal_moves]
        d9_row = d9_rows[position.key]
        if bool(d9_row.get("timed_out")):
            raise ValueError(f"{position.key}: d9 label timed out")
        source = _stored_d9_policy(legal_ucis, _depth_lines(d9_row, 9))
        if scope == "sf-cp-window":
            sf_rank_indices, sf_rank_gaps_cp = _audit_d9_rank_arrays(
                legal_ucis,
                _depth_lines(d9_row, 9),
                top_k=sf_rank_cap,
            )
        else:
            sf_rank_indices = None
            sf_rank_gaps_cp = None

        topk = bt4_rows[position.key].get("topk", [])
        bt4_by_move = {str(move): float(prob) for move, prob in topk}
        if set(bt4_by_move) != set(legal_ucis):
            raise ValueError(f"{position.key}: BT4 cache is not full legal width")
        bt4 = np.asarray([bt4_by_move[move] for move in legal_ucis], dtype=np.float64)
        if not np.isfinite(bt4).all() or bool(np.any(bt4 < 0.0)) or bt4.sum() <= 0.0:
            raise ValueError(f"{position.key}: invalid BT4 cache probabilities")
        bt4 /= float(bt4.sum())

        legal = np.ones((1, len(legal_ucis)), dtype=np.uint8)
        candidate = mix_policy_targets(
            source[None, :],
            bt4[None, :],
            legal,
            alpha=alpha,
            scope=scope,
            bt4_temperature=bt4_temperature,
            near_max_ratio=near_max_ratio,
            sf_rank_indices=sf_rank_indices,
            sf_rank_gaps_cp=sf_rank_gaps_cp,
            sf_rank_cap=sf_rank_cap,
            sf_cp_window=sf_cp_window,
        )
        # The materializer casts back to the source policy_target dtype. The
        # 20M source is float16, so audit what the trainer will actually read.
        candidate_unstored = candidate
        candidate_stored = candidate_unstored.astype(np.float16).astype(np.float32)
        reference_unstored = mix_policy_targets(
            source[None, :],
            bt4[None, :],
            legal,
            alpha=alpha,
            scope=scope,
            bt4_temperature=1.0,
            near_max_ratio=near_max_ratio,
            sf_rank_indices=sf_rank_indices,
            sf_rank_gaps_cp=sf_rank_gaps_cp,
            sf_rank_cap=sf_rank_cap,
            sf_cp_window=sf_cp_window,
        )
        reference_stored = reference_unstored.astype(np.float16).astype(np.float32)
        candidate = _normalized_legal(
            candidate_stored,
            legal,
            name="stored candidate",
        )[0]
        reference = _normalized_legal(
            reference_stored,
            legal,
            name="stored temperature-1 reference",
        )[0]
        regrets = move_regrets(position, legal_ucis)
        source_pair = expected_and_top1_regret(source, regrets)
        candidate_pair = expected_and_top1_regret(candidate, regrets)
        source_metrics.append(source_pair)
        candidate_metrics.append(candidate_pair)
        source_top = int(np.argmax(source))
        candidate_top = int(np.argmax(candidate))
        top_count = int(np.sum(source == np.max(source)))
        candidate_set = _source_candidate_set(
            source[None, :],
            legal,
            scope=scope,
            near_max_ratio=near_max_ratio,
            sf_rank_indices=sf_rank_indices,
            sf_rank_gaps_cp=sf_rank_gaps_cp,
            sf_rank_cap=sf_rank_cap,
            sf_cp_window=sf_cp_window,
        )[0]
        candidate_count = int(candidate_set.sum())
        selected_source_mass = (
            1.0 if scope == "global"
            else float(np.sum(source[candidate_set], dtype=np.float64))
        )
        selected_candidate_mass = float(
            np.sum(candidate_stored[0, candidate_set], dtype=np.float64)
        )
        source_entropy = _entropy_sum(source[None, :])
        candidate_entropy = _entropy_sum(candidate[None, :])
        per_position.append(
            {
                "key": position.key,
                "phase": position.phase,
                "source": position.source,
                "source_expected_regret_cp": source_pair[0],
                "source_top1_regret_cp": source_pair[1],
                "candidate_expected_regret_cp": candidate_pair[0],
                "candidate_top1_regret_cp": candidate_pair[1],
                "source_top1": legal_ucis[source_top],
                "candidate_top1": legal_ucis[candidate_top],
                "source_top1_probability": float(source[source_top]),
                "candidate_top1_probability": float(candidate[candidate_top]),
                "source_entropy_nats": source_entropy,
                "candidate_entropy_nats": candidate_entropy,
                "source_top_tie_count": top_count,
                "source_candidate_count": candidate_count,
                "candidate_set_wider_than_top_tie": candidate_count > top_count,
                "policy_changed": not np.array_equal(candidate_stored[0], source),
                "selected_mass_abs_drift": abs(
                    selected_candidate_mass - selected_source_mass
                ),
                "temperature_one_top1": legal_ucis[int(np.argmax(reference))],
                "matches_temperature_one_top1": (
                    int(np.argmax(reference)) == candidate_top
                ),
                "matches_temperature_one_top1_before_storage": (
                    int(np.argmax(reference_unstored[0]))
                    == int(np.argmax(candidate_unstored[0]))
                ),
                "top1_changed": source_top != candidate_top,
            }
        )

    source_array = np.asarray(source_metrics, dtype=np.float64)
    candidate_array = np.asarray(candidate_metrics, dtype=np.float64)
    delta = candidate_array - source_array
    expected_ci = _bootstrap_mean_ci(
        delta[:, 0],
        n_boot=int(args.boot),
        seed=int(args.seed),
    )
    top1_ci = _bootstrap_mean_ci(
        delta[:, 1],
        n_boot=int(args.boot),
        seed=int(args.seed) + 1,
    )
    if expected_ci[0] > 0.0:
        verdict = "kill"
    elif expected_ci[1] < 0.0:
        verdict = "graduate_win"
    else:
        verdict = "graduate_tie"

    def group_summary(indices: np.ndarray) -> dict[str, Any]:
        group_delta = delta[indices]
        return {
            "rows": int(indices.sum()),
            "source_expected_regret_cp": _metric_summary(source_array[indices, 0]),
            "candidate_expected_regret_cp": _metric_summary(
                candidate_array[indices, 0]
            ),
            "expected_regret_delta_cp": _metric_summary(group_delta[:, 0]),
            "source_top1_regret_cp": _metric_summary(source_array[indices, 1]),
            "candidate_top1_regret_cp": _metric_summary(candidate_array[indices, 1]),
            "top1_regret_delta_cp": _metric_summary(group_delta[:, 1]),
        }

    phase = np.asarray([position.phase for position in positions], dtype=np.int8)
    source_kind = np.asarray([position.source for position in positions], dtype=np.int8)
    all_rows = np.ones(len(positions), dtype=bool)
    changed_unique_max_rows = int(
        sum(
            row["policy_changed"] and row["source_top_tie_count"] == 1
            for row in per_position
        )
    )
    candidate_set_wider_rows = int(
        sum(row["candidate_set_wider_than_top_tie"] for row in per_position)
    )
    temperature_one_top1_mismatch_rows = int(
        sum(not row["matches_temperature_one_top1"] for row in per_position)
    )
    temperature_prestorage_top1_mismatch_rows = int(
        sum(
            not row["matches_temperature_one_top1_before_storage"]
            for row in per_position
        )
    )
    selected_mass_drifts = np.asarray(
        [row["selected_mass_abs_drift"] for row in per_position],
        dtype=np.float64,
    )
    mass_mean_bound, mass_row_bound = _mass_drift_bounds(scope)
    treatment_invariants = {
        "mass_reference": (
            "normalized_total_legal_mass" if scope == "global" else "stored_selected_mass"
        ),
        "candidate_set_wider_rows": candidate_set_wider_rows,
        "changed_unique_max_rows": changed_unique_max_rows,
        "temperature_one_top1_mismatch_rows": temperature_one_top1_mismatch_rows,
        "temperature_prestorage_top1_mismatch_rows": (
            temperature_prestorage_top1_mismatch_rows
        ),
        "near_max_extended": (
            scope not in {"near-max-ratio", "sf-cp-window"}
            or (candidate_set_wider_rows > 0 and changed_unique_max_rows > 0)
        ),
        "top_tie_unique_max_identity": (
            scope != "top-max-ties" or changed_unique_max_rows == 0
        ),
        "temperature_one_top1_preserved": (
            bt4_temperature == 1.0
            or alpha != 1.0
            or scope != "top-max-ties"
            or temperature_one_top1_mismatch_rows == 0
        ),
        "temperature_rank_preserved_before_storage": (
            bt4_temperature == 1.0
            or alpha != 1.0
            or scope != "sf-cp-window"
            or temperature_prestorage_top1_mismatch_rows == 0
        ),
        "selected_mass_drift_within_bounds": (
            float(selected_mass_drifts.mean()) <= mass_mean_bound
            and float(selected_mass_drifts.max(initial=0.0))
            <= mass_row_bound
        ),
    }
    treatment_invariants_passed = all(
        bool(treatment_invariants[key])
        for key in (
            "near_max_extended",
            "top_tie_unique_max_identity",
            "temperature_one_top1_preserved",
            "temperature_rank_preserved_before_storage",
            "selected_mass_drift_within_bounds",
        )
    )
    training_permitted = audit_training_permitted(
        verdict=verdict,
        treatment_invariants_passed=treatment_invariants_passed,
        sf_audit_mode=admission["mode"],
    )
    report: dict[str, Any] = {
        "schema": 1,
        "kind": "bt4_policy_mix_frozen_deep_sf_audit",
        "admission": admission,
        "treatment": treatment,
        "source_target_contract": SOURCE_TARGET_CONTRACT,
        "ruler": {
            "audit_set": str(audit_path),
            "audit_set_sha256": file_sha256(audit_path),
            "d9_labels": str(d9_path),
            "d9_labels_sha256": file_sha256(d9_path),
            "bt4_cache": str(bt4_path),
            "bt4_cache_sha256": file_sha256(bt4_path),
            "d9_depth": 9,
            "d9_policy_temp": 0.0005,
            "cp_slope": 0.006,
            "cp_draw_width": 120.0,
            "source_storage_dtype": "float16",
            "bt4_cache_input_history": "fen_only",
            "bootstrap_unit": "position (frozen audit set carries no game id)",
            "bootstrap_replicates": int(args.boot),
            "bootstrap_seed": int(args.seed),
        },
        "overall": group_summary(all_rows),
        "expected_regret_delta_95ci_cp": list(expected_ci),
        "top1_regret_delta_95ci_cp": list(top1_ci),
        "source_top_tied_fraction": float(
            np.mean([row["source_top_tie_count"] > 1 for row in per_position])
        ),
        "source_candidate_multi_fraction": float(
            np.mean([row["source_candidate_count"] > 1 for row in per_position])
        ),
        "candidate_set_wider_than_top_tie_fraction": float(
            np.mean(
                [row["candidate_set_wider_than_top_tie"] for row in per_position]
            )
        ),
        "treatment_invariants": treatment_invariants,
        "top1_changed_fraction": float(
            np.mean([row["top1_changed"] for row in per_position])
        ),
        "shape": {
            "source_mean_entropy_nats": float(
                np.mean([row["source_entropy_nats"] for row in per_position])
            ),
            "candidate_mean_entropy_nats": float(
                np.mean([row["candidate_entropy_nats"] for row in per_position])
            ),
            "source_top1_ge_0_99_fraction": float(
                np.mean(
                    [row["source_top1_probability"] >= 0.99 for row in per_position]
                )
            ),
            "candidate_top1_ge_0_99_fraction": float(
                np.mean(
                    [row["candidate_top1_probability"] >= 0.99 for row in per_position]
                )
            ),
            "selected_mass_abs_drift_mean": float(selected_mass_drifts.mean()),
            "selected_mass_abs_drift_max": float(
                selected_mass_drifts.max(initial=0.0)
            ),
            "selected_mass_abs_drift_bounds": {
                "mean_max": mass_mean_bound,
                "row_max": mass_row_bound,
            },
        },
        "by_phase": {
            name: group_summary(phase == index)
            for index, name in enumerate(PHASE_NAMES)
        },
        "by_source": {
            name: group_summary(source_kind == index)
            for index, name in enumerate(SOURCE_NAMES)
        },
        "gate": {
            "metric": "candidate minus source expected deep-SF regret in cp",
            "kill_if": (
                "paired 95% interval is wholly above zero"
                if admission["mode"] == "gate"
                else "SF comparison is descriptive; fidelity and lineage remain mandatory"
            ),
            "verdict": verdict,
            "treatment_invariants_passed": treatment_invariants_passed,
            "training_permitted": training_permitted,
            "value_gate": "not applicable; mix leaves all value columns byte-identical",
        },
        "per_position": per_position,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_path, report)
    overall = report["overall"]
    print(
        f"[bt4-audit] SF comparison={verdict}, mode={admission['mode']}, "
        f"training_permitted={training_permitted}: n={len(positions)}, "
        f"expected delta={overall['expected_regret_delta_cp']['mean']:.4f} cp "
        f"[{expected_ci[0]:.4f}, {expected_ci[1]:.4f}], "
        f"top1 delta={overall['top1_regret_delta_cp']['mean']:.4f} cp "
        f"[{top1_ci[0]:.4f}, {top1_ci[1]:.4f}] -> {out_path}",
        flush=True,
    )
    return 0 if training_permitted else 2


def mix_corpus(args: argparse.Namespace) -> int:
    sf_audit_mode = str(getattr(args, "sf_audit_mode", "gate"))
    experiment_record = getattr(args, "experiment_record", None)
    admission = _audit_admission(sf_audit_mode, experiment_record)
    alpha = validate_alpha(float(args.alpha))
    scope = str(args.scope)
    mass_mean_bound, mass_row_bound = _mass_drift_bounds(scope)
    if scope not in MIX_SCOPES:
        raise SystemExit(f"--scope must be one of {MIX_SCOPES}")
    bt4_temperature = validate_bt4_temperature(float(args.bt4_temperature))
    near_max_ratio = float(args.near_max_ratio)
    sf_rank_cap = int(getattr(args, "sf_rank_cap", 3))
    sf_cp_window = float(getattr(args, "sf_cp_window", 10.0))
    treatment_specification = treatment_spec(
        scope=scope,
        alpha=alpha,
        bt4_temperature=bt4_temperature,
        near_max_ratio=near_max_ratio,
        sf_rank_cap=sf_rank_cap,
        sf_cp_window=sf_cp_window,
    )
    source_dir = Path(args.shards).resolve()
    sidecar_dir = Path(args.sidecar).resolve()
    rank_sidecar_dir = (
        None
        if getattr(args, "sf_rank_sidecar", None) is None
        else Path(args.sf_rank_sidecar).resolve()
    )
    out_dir = Path(args.out).resolve()
    expected_rows = int(args.expected_rows)
    expected_shards = int(args.expected_shards)
    expected_source_summary_sha256 = str(args.expected_source_summary_sha256)
    if expected_rows <= 0 or expected_shards <= 0:
        raise SystemExit("--expected-rows and --expected-shards must be positive")
    if len(expected_source_summary_sha256) != 64:
        raise SystemExit("--expected-source-summary-sha256 must be a SHA-256 hex digest")
    if out_dir.exists():
        raise SystemExit(
            f"{out_dir} exists; mixed corpora are immutable, use a fresh path"
        )
    if source_dir == out_dir or source_dir in out_dir.parents:
        raise SystemExit("--out must be separate from, not inside, --shards")
    if sidecar_dir == source_dir:
        raise SystemExit("--sidecar must be separate from --shards")
    if sidecar_dir == out_dir or sidecar_dir in out_dir.parents:
        raise SystemExit("--out must be separate from, not inside, --sidecar")
    if scope == "sf-cp-window" and rank_sidecar_dir is None:
        raise SystemExit("sf-cp-window requires --sf-rank-sidecar")
    if scope != "sf-cp-window" and rank_sidecar_dir is not None:
        raise SystemExit("--sf-rank-sidecar is valid only for sf-cp-window")
    if rank_sidecar_dir is not None and (
        rank_sidecar_dir in out_dir.parents or rank_sidecar_dir == out_dir
    ):
        raise SystemExit("--out must be separate from, not inside, --sf-rank-sidecar")
    writing = out_dir.with_name(out_dir.name + ".writing")
    if writing.exists():
        raise SystemExit(f"stale partial mixed corpus exists: {writing}")
    source_paths = iter_shard_paths(source_dir)
    if not source_paths:
        raise SystemExit(f"{source_dir} has no replay shards")
    derive_summary_source = source_dir / DERIVE_SUMMARY
    if not derive_summary_source.is_file():
        raise SystemExit(
            f"{source_dir} has no {DERIVE_SUMMARY}; refusing a corpus whose "
            "base target provenance cannot be carried forward",
        )
    derive_summary_original = json.loads(
        derive_summary_source.read_text(encoding="utf-8")
    )
    source_summary_sha256 = file_sha256(derive_summary_source)
    if source_summary_sha256 != expected_source_summary_sha256:
        raise SystemExit(
            "source derive-summary SHA-256 mismatch: "
            f"{source_summary_sha256} != {expected_source_summary_sha256}",
        )
    source_rows, source_policy_dtypes = _source_policy_layout(source_paths)
    if source_rows != expected_rows or len(source_paths) != expected_shards:
        raise SystemExit(
            "source corpus size mismatch: "
            f"rows={source_rows}/{expected_rows}, "
            f"shards={len(source_paths)}/{expected_shards}",
        )
    _validate_source_target_contract(
        derive_summary_original,
        storage_dtypes=source_policy_dtypes,
    )
    audit_receipt_path = Path(args.audit_receipt).resolve()
    _load_audit_receipt(
        audit_receipt_path,
        alpha=alpha,
        scope=scope,
        bt4_temperature=bt4_temperature,
        near_max_ratio=near_max_ratio,
        sf_rank_cap=sf_rank_cap,
        sf_cp_window=sf_cp_window,
        sf_audit_mode=sf_audit_mode,
        experiment_record=experiment_record,
    )
    side_summary_path = sidecar_dir / SIDECAR_SUMMARY
    if not side_summary_path.is_file():
        raise SystemExit(f"missing completed sidecar summary {side_summary_path}")
    side_summary = json.loads(side_summary_path.read_text(encoding="utf-8"))
    expected_side = {
        "schema": SIDECAR_SCHEMA,
        "kind": "bt4_raw_legal_policy_sidecar",
        "source_dir": str(source_dir),
        "source_shards": len(source_paths),
        "sidecar_shards": len(source_paths),
        "rows": source_rows,
        "policy_encoding": POLICY_ENCODING_LC0_1858,
        "teacher_evaluations_per_position": 1,
        "search_nodes": 0,
        "stored_dtype": "float32",
    }
    side_bad = {
        key: (side_summary.get(key), value)
        for key, value in expected_side.items()
        if side_summary.get(key) != value
    }
    if side_bad:
        raise SystemExit(
            f"sidecar summary does not describe this source corpus: {side_bad}"
        )
    side_remap = functional_remap_identity(side_summary.get("remap"))
    current_remap = functional_remap_identity(remap_provenance())
    if side_remap != current_remap:
        raise SystemExit(
            "sidecar remap implementation does not match the current code: "
            f"{side_remap} != {current_remap}"
        )
    side_onnx = side_summary.get("onnx")
    if not isinstance(side_onnx, dict) or not isinstance(side_onnx.get("sha256"), str):
        raise SystemExit("sidecar summary has no ONNX SHA-256")
    side_providers = side_summary.get("providers")
    if not isinstance(side_providers, list) or not all(
        isinstance(provider, str) for provider in side_providers
    ):
        raise SystemExit("sidecar summary has no realized provider list")
    side_policy_output = side_summary.get("policy_output")
    if not isinstance(side_policy_output, str):
        raise SystemExit("sidecar summary has no resolved policy output")

    rank_summary_path: Path | None = None
    rank_summary_sha: str | None = None
    rank_outputs: dict[str, Mapping[str, Any]] | None = None
    if rank_sidecar_dir is not None:
        rank_summary_path = rank_sidecar_dir / sf_ranks.SUMMARY_NAME
        if not rank_summary_path.is_file():
            raise SystemExit(f"missing completed SF rank summary {rank_summary_path}")
        rank_summary = json.loads(rank_summary_path.read_text(encoding="utf-8"))
        expected_rank = {
            "schema": sf_ranks.SCHEMA,
            "kind": "sf_d9_rank_gap_sidecar",
            "source_dir": str(source_dir),
            "source_derive_summary_sha256": source_summary_sha256,
            "rows": source_rows,
            "shards": len(source_paths),
            "depth": 9,
            "index_encoding": POLICY_ENCODING_LC0_1858,
            "gap_definition": "rank1_effective_cp-minus-ranked_effective_cp",
        }
        rank_bad = {
            key: (rank_summary.get(key), value)
            for key, value in expected_rank.items()
            if rank_summary.get(key) != value
        }
        if int(rank_summary.get("top_k", -1)) < sf_rank_cap:
            rank_bad["top_k"] = (rank_summary.get("top_k"), f">={sf_rank_cap}")
        if rank_bad:
            raise SystemExit(
                f"SF rank summary does not describe this source corpus: {rank_bad}",
            )
        output_items = rank_summary.get("outputs")
        if not isinstance(output_items, list) or not all(
            isinstance(item, Mapping) for item in output_items
        ):
            raise SystemExit("SF rank summary has no per-shard output receipts")
        output_names = [str(item.get("path")) for item in output_items]
        expected_names = [path.name for path in source_paths]
        if output_names != expected_names:
            raise SystemExit(
                "SF rank summary output order does not match the source shards",
            )
        rank_outputs = {str(item["path"]): item for item in output_items}
        rank_summary_sha = file_sha256(rank_summary_path)

    shutil.copytree(source_dir, writing)
    stats = MixStats()
    try:
        for source_path in source_paths:
            source: Any = zarr.open_group(str(source_path), mode="r")
            encoding, keys, key_sha, policy_sha = _sidecar_identity(source, source_path)
            side_path = sidecar_dir / source_path.name
            _validate_sidecar(
                side_path,
                source_path=source_path,
                source_keys=keys,
                source_key_sha=key_sha,
                source_policy_sha=policy_sha,
                onnx_sha=side_onnx["sha256"],
                providers=side_providers,
                policy_output=side_policy_output,
            )
            side: Any = zarr.open_group(str(side_path), mode="r")
            rank: Any | None = None
            rank_attrs: dict[str, Any] | None = None
            if rank_sidecar_dir is not None:
                rank_path = rank_sidecar_dir / source_path.name
                rank_attrs = _validate_sf_rank_sidecar(
                    rank_path,
                    source_group=source,
                    source_path=source_path,
                    source_summary_sha256=source_summary_sha256,
                    required_top_k=sf_rank_cap,
                )
                if rank_outputs is None:  # pragma: no cover - guarded above
                    raise AssertionError("rank summary receipts were not loaded")
                summary_output = rank_outputs[source_path.name]
                expected_output = {
                    "path": source_path.name,
                    "rows": int(source[POLICY_FIELD].shape[0]),
                    "source_row_identity_sha256": rank_attrs[
                        "source_row_identity_sha256"
                    ],
                    "payload_sha256": rank_attrs["payload_sha256"],
                }
                output_bad = {
                    key: (summary_output.get(key), value)
                    for key, value in expected_output.items()
                    if summary_output.get(key) != value
                }
                if output_bad:
                    raise ValueError(
                        f"{rank_path}: summary output receipt mismatch {output_bad}",
                    )
                rank = zarr.open_group(str(rank_path), mode="r")
            destination_path = writing / source_path.name
            destination: Any = zarr.open_group(str(destination_path), mode="a")
            rows = int(source["x"].shape[0])
            chunk_rows = int(source[POLICY_FIELD].chunks[0])
            for start in range(0, rows, chunk_rows):
                stop = min(rows, start + chunk_rows)
                sf_stored = np.asarray(source[POLICY_FIELD][start:stop])
                bt4_stored = np.asarray(side[SIDECAR_POLICY_FIELD][start:stop])
                legal = np.asarray(source["legal_mask"][start:stop])
                sf_rank_indices = (
                    None
                    if rank is None
                    else np.asarray(rank[sf_ranks.INDEX_FIELD][start:stop])
                )
                sf_rank_gaps_cp = (
                    None
                    if rank is None
                    else np.asarray(rank[sf_ranks.GAP_FIELD][start:stop])
                )
                sf = _normalized_legal(sf_stored, legal, name=f"{source_path}:source")
                bt4 = _normalized_legal(bt4_stored, legal, name=f"{source_path}:BT4")
                mixed = mix_policy_targets(
                    sf_stored,
                    bt4,
                    legal,
                    alpha=alpha,
                    scope=scope,
                    bt4_temperature=bt4_temperature,
                    near_max_ratio=near_max_ratio,
                    sf_rank_indices=sf_rank_indices,
                    sf_rank_gaps_cp=sf_rank_gaps_cp,
                    sf_rank_cap=sf_rank_cap,
                    sf_cp_window=sf_cp_window,
                )
                stored = mixed.astype(destination[POLICY_FIELD].dtype, copy=False)
                destination[POLICY_FIELD][start:stop] = stored
                reread = np.asarray(destination[POLICY_FIELD][start:stop])
                if not np.array_equal(reread, stored):
                    raise ValueError(f"{destination_path}: policy write/read mismatch")
                if scope == "global" and bool(np.any(reread[legal == 0] != 0.0)):
                    raise ValueError(f"{destination_path}: global policy has illegal mass")

                sf_top = np.argmax(sf, axis=1)
                bt4_top = np.argmax(bt4, axis=1)
                mixed_read = _normalized_legal(
                    reread,
                    legal,
                    name=f"{destination_path}:stored mixed policy",
                )
                mixed_top = np.argmax(mixed_read, axis=1)
                stats.rows += stop - start
                changed = np.any(stored != sf_stored, axis=1)
                legal_bool = legal != 0
                source_legal = np.where(legal_bool, sf_stored, -np.inf)
                source_top = source_legal == np.max(
                    source_legal,
                    axis=1,
                    keepdims=True,
                )
                source_tied = source_top.sum(axis=1) > 1
                candidate_set = _source_candidate_set(
                    sf_stored,
                    legal_bool,
                    scope=scope,
                    near_max_ratio=near_max_ratio,
                    sf_rank_indices=sf_rank_indices,
                    sf_rank_gaps_cp=sf_rank_gaps_cp,
                    sf_rank_cap=sf_rank_cap,
                    sf_cp_window=sf_cp_window,
                )
                candidate_count = candidate_set.sum(axis=1)
                source_top_count = source_top.sum(axis=1)
                stats.changed_rows += int(changed.sum())
                stats.source_top_tied_rows += int(source_tied.sum())
                stats.source_candidate_multi_rows += int(
                    np.sum(candidate_count > 1)
                )
                stats.candidate_set_wider_rows += int(
                    np.sum(candidate_count > source_top_count)
                )
                source_selected_mass = (
                    np.ones(stop - start, dtype=np.float64) if scope == "global"
                    else np.sum(
                        np.where(candidate_set, sf_stored, 0.0), axis=1, dtype=np.float64,
                    )
                )
                mixed_selected_mass = np.sum(
                    np.where(candidate_set, reread, 0.0),
                    axis=1,
                    dtype=np.float64,
                )
                selected_mass_abs_drift = np.abs(
                    mixed_selected_mass - source_selected_mass
                )
                stats.selected_mass_abs_drift_sum += float(
                    selected_mass_abs_drift.sum()
                )
                stats.selected_mass_abs_drift_max = max(
                    stats.selected_mass_abs_drift_max,
                    float(selected_mass_abs_drift.max(initial=0.0)),
                )
                stats.changed_unique_max_rows += int(np.sum(changed & ~source_tied))
                stats.source_bt4_top1_agree += int(np.sum(sf_top == bt4_top))
                stats.mixed_source_top1_agree += int(np.sum(mixed_top == sf_top))
                stats.mixed_bt4_top1_agree += int(np.sum(mixed_top == bt4_top))
                stats.source_top1_ge_0_99_rows += int(
                    np.sum(np.max(sf, axis=1) >= 0.99)
                )
                stats.mixed_top1_ge_0_99_rows += int(
                    np.sum(np.max(mixed_read, axis=1) >= 0.99)
                )
                stats.source_entropy_sum += _entropy_sum(sf)
                stats.bt4_entropy_sum += _entropy_sum(bt4)
                stats.mixed_entropy_sum += _entropy_sum(mixed_read)
                stats.l1_from_source_sum += float(np.abs(mixed_read - sf).sum())

            destination.attrs.update(
                {
                    "policy_target_mix_schema": MIX_SCHEMA,
                    "policy_target_mix_kind": scope,
                    "policy_target_mix_algorithm": TREATMENT_ALGORITHMS[scope],
                    "policy_target_mix_alpha": alpha,
                    "policy_target_mix_bt4_temperature": bt4_temperature,
                    "policy_target_mix_near_max_ratio": (
                        near_max_ratio if scope == "near-max-ratio" else None
                    ),
                    "policy_target_mix_sf_rank_cap": (
                        sf_rank_cap if scope == "sf-cp-window" else None
                    ),
                    "policy_target_mix_sf_cp_window": (
                        sf_cp_window if scope == "sf-cp-window" else None
                    ),
                    "policy_target_mix_sf_rank_sidecar": (
                        str(rank_sidecar_dir) if rank_sidecar_dir is not None else None
                    ),
                    "policy_target_mix_sf_rank_payload_sha256": (
                        rank_attrs.get("payload_sha256")
                        if rank_attrs is not None
                        else None
                    ),
                    "policy_target_mix_source": "stored_stockfish_nnue_bootstrap",
                    "policy_target_mix_external": "bt4_raw_one_eval",
                    "policy_target_mix_sidecar": str(sidecar_dir),
                    "policy_target_mix_sidecar_schema": SIDECAR_SCHEMA,
                    "policy_target_mix_source_key_sha256": key_sha,
                    "policy_target_mix_source_policy_sha256": policy_sha,
                    "policy_target_mix_onnx_sha256": dict(side.attrs)["onnx_sha256"],
                    "policy_target_mix_value_columns_unchanged": True,
                    "policy_target_mix_input_history_encoding": encoding,
                }
            )
            stats.shards += 1

        denom = max(stats.rows, 1)
        treatment = {
            "schema": MIX_SCHEMA,
            "kind": treatment_specification["scope"],
            "algorithm": treatment_specification["algorithm"],
            "formula": (
                "mixed=(1-alpha)*stored_stockfish+alpha*bt4_raw_one_eval"
                if scope == "global"
                else (
                    "redistribute alpha of stored source top-tie mass by "
                    "temperature-scaled BT4 prior"
                    if scope == "top-max-ties"
                    else (
                        "redistribute alpha of stored source near-max mass by "
                        "temperature-scaled BT4 prior"
                        if scope == "near-max-ratio"
                        else "redistribute alpha of stored top-tie union d9 "
                        "cp-window mass by temperature-scaled BT4 prior"
                    )
                )
            ),
            "alpha": alpha,
            "bt4_temperature": bt4_temperature,
            "near_max_ratio": treatment_specification["near_max_ratio"],
            "sf_rank_cap": treatment_specification.get("sf_rank_cap"),
            "sf_cp_window": treatment_specification.get("sf_cp_window"),
            "source_dir": str(source_dir),
            "source_derive_summary_sha256": source_summary_sha256,
            "expected_rows": expected_rows,
            "expected_shards": expected_shards,
            "sidecar_dir": str(sidecar_dir),
            "sidecar_summary": {
                "path": str(side_summary_path),
                "sha256": file_sha256(side_summary_path),
            },
            "sf_rank_sidecar_dir": (
                str(rank_sidecar_dir) if rank_sidecar_dir is not None else None
            ),
            "sf_rank_sidecar_summary": (
                None
                if rank_summary_path is None
                else {
                    "path": str(rank_summary_path),
                    "sha256": rank_summary_sha,
                }
            ),
            "audit_receipt": {
                "path": str(audit_receipt_path),
                "sha256": file_sha256(audit_receipt_path),
            },
            "admission": admission,
            "rows": stats.rows,
            "shards": stats.shards,
            "changed_rows": stats.changed_rows,
            "changed_fraction": stats.changed_rows / denom,
            "source_top_tied_rows": stats.source_top_tied_rows,
            "source_top_tied_fraction": stats.source_top_tied_rows / denom,
            "source_candidate_multi_rows": stats.source_candidate_multi_rows,
            "source_candidate_multi_fraction": (
                stats.source_candidate_multi_rows / denom
            ),
            "candidate_set_wider_rows": stats.candidate_set_wider_rows,
            "candidate_set_wider_fraction": stats.candidate_set_wider_rows / denom,
            "changed_unique_max_rows": stats.changed_unique_max_rows,
            "source_bt4_top1_agreement": stats.source_bt4_top1_agree / denom,
            "mixed_source_top1_agreement": stats.mixed_source_top1_agree / denom,
            "mixed_bt4_top1_agreement": stats.mixed_bt4_top1_agree / denom,
            "mean_entropy_nats": {
                "source": stats.source_entropy_sum / denom,
                "bt4": stats.bt4_entropy_sum / denom,
                "mixed": stats.mixed_entropy_sum / denom,
            },
            "top1_ge_0_99_fraction": {
                "source": stats.source_top1_ge_0_99_rows / denom,
                "mixed": stats.mixed_top1_ge_0_99_rows / denom,
            },
            "mean_l1_from_source": stats.l1_from_source_sum / denom,
            "selected_mass_abs_drift": {
                "reference": (
                    "normalized_total_legal_mass" if scope == "global" else "stored_selected_mass"
                ),
                "mean": stats.selected_mass_abs_drift_sum / denom,
                "max": stats.selected_mass_abs_drift_max,
                "mean_bound": mass_mean_bound,
                "row_bound": mass_row_bound,
            },
            "mutated_arrays": [POLICY_FIELD],
            "value_columns_unchanged": ["wdl_target", "search_wdl"],
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if stats.shards != len(source_paths):
            raise ValueError(f"mixed only {stats.shards}/{len(source_paths)} shards")
        if stats.changed_rows <= 0 or stats.l1_from_source_sum <= 0.0:
            raise ValueError(
                "the requested BT4 treatment changed no target rows; refusing "
                "an inert corpus",
            )
        if scope == "top-max-ties" and stats.changed_unique_max_rows != 0:
            raise ValueError(
                "top-max-ties changed a unique-maximum source row; refusing "
                "a treatment outside its declared scope",
            )
        if scope in {"near-max-ratio", "sf-cp-window"} and (
            stats.changed_unique_max_rows <= 0 or stats.candidate_set_wider_rows <= 0
        ):
            raise ValueError(
                f"{scope} did not extend beyond exact source ties; refusing "
                "an inert breadth treatment",
            )
        if (
            stats.selected_mass_abs_drift_sum / denom
            > mass_mean_bound
            or stats.selected_mass_abs_drift_max > mass_row_bound
        ):
            raise ValueError(
                "float16 "
                + ("total legal" if scope == "global" else "selected-set")
                + " mass drift exceeds the preregistered bounds",
            )

        derive_summary_path = writing / DERIVE_SUMMARY
        if _audit_admission(sf_audit_mode, experiment_record) != admission:
            raise ValueError("experiment record changed during materialization")
        derive_summary = json.loads(derive_summary_path.read_text(encoding="utf-8"))
        if "policy_target_postprocess" in derive_summary:
            raise ValueError("source derive summary already names a policy postprocess")
        derive_summary["policy_target_postprocess"] = treatment
        _atomic_json(derive_summary_path, derive_summary)
        _atomic_json(writing / MIX_SUMMARY, treatment)
        os.replace(writing, out_dir)
    except BaseException:
        print(
            f"[bt4-mix] FAILED; preserving partial output at {writing}", file=sys.stderr
        )
        raise

    print(
        f"[bt4-mix] complete: {stats.rows} rows, scope={scope}, "
        f"alpha={alpha:.6f}, bt4_temperature={bt4_temperature:.6f}, "
        f"near_max_ratio={treatment_specification['near_max_ratio']}, "
        f"sf_rank_cap={treatment_specification.get('sf_rank_cap')}, "
        f"sf_cp_window={treatment_specification.get('sf_cp_window')}, "
        f"top1 preserved={stats.mixed_source_top1_agree / stats.rows:.4%} -> {out_dir}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    label = sub.add_parser("label", help="run raw one-evaluation BT4 policy inference")
    label.add_argument("--shards", type=Path, required=True)
    label.add_argument("--out", type=Path, required=True)
    label.add_argument("--onnx", type=Path, default=Path(DEFAULT_ONNX))
    label.add_argument("--batch-size", type=int, default=1024)
    label.add_argument("--threads", type=int, default=16)
    label.add_argument("--gpu-mem-gb", type=float, default=0.0)
    label.add_argument("--policy-output", default=None)
    label.add_argument("--resume", action="store_true")

    mix = sub.add_parser("mix", help="materialize an arithmetic source/BT4 target mix")
    mix.add_argument("--shards", type=Path, required=True)
    mix.add_argument("--sidecar", type=Path, required=True)
    mix.add_argument("--out", type=Path, required=True)
    mix.add_argument("--alpha", type=float, required=True)
    mix.add_argument("--scope", choices=MIX_SCOPES, default="top-max-ties")
    mix.add_argument("--bt4-temperature", type=float, default=1.0)
    mix.add_argument("--near-max-ratio", type=float, default=0.5)
    mix.add_argument("--sf-rank-sidecar", type=Path, default=None)
    mix.add_argument("--sf-rank-cap", type=int, default=3)
    mix.add_argument("--sf-cp-window", type=float, default=10.0)
    mix.add_argument("--expected-rows", type=int, required=True)
    mix.add_argument("--expected-shards", type=int, required=True)
    mix.add_argument("--expected-source-summary-sha256", required=True)
    mix.add_argument("--audit-receipt", type=Path, required=True)

    audit = sub.add_parser(
        "audit",
        help="gate a treatment on the frozen deep-SF audit and bank the receipt",
    )
    audit.add_argument("--audit-set", type=Path, required=True)
    audit.add_argument("--d9-labels", type=Path, required=True)
    audit.add_argument("--bt4-cache", type=Path, required=True)
    audit.add_argument("--json", type=Path, required=True)
    audit.add_argument("--alpha", type=float, required=True)
    audit.add_argument("--scope", choices=MIX_SCOPES, default="top-max-ties")
    audit.add_argument("--bt4-temperature", type=float, default=1.0)
    audit.add_argument("--near-max-ratio", type=float, default=0.5)
    audit.add_argument("--sf-rank-cap", type=int, default=3)
    audit.add_argument("--sf-cp-window", type=float, default=10.0)
    audit.add_argument("--boot", type=int, default=10_000)
    audit.add_argument("--seed", type=int, default=20260903)
    for command in (audit, mix):
        command.add_argument(
            "--sf-audit-mode", choices=("gate", "descriptive"), default="gate",
            help="whether deep-SF agreement gates admission (default) or is descriptive",
        )
        command.add_argument(
            "--experiment-record", type=Path, default=None,
            help="immutable preregistration snapshot required in descriptive mode; path and SHA-256 must match the audit receipt",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "label":
        if int(args.batch_size) <= 0 or int(args.threads) < 0:
            raise SystemExit("--batch-size must be positive and --threads non-negative")
        return label_sidecar(args)
    if args.command == "mix":
        return mix_corpus(args)
    if args.command == "audit":
        if int(args.boot) <= 0:
            raise SystemExit("--boot must be positive")
        return audit_mix(args)
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
