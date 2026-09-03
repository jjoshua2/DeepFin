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
    either a global arithmetic mix or the audit-approved top-tie treatment.
    The latter preserves the source mass on its equal maxima and lets BT4
    redistribute only ``alpha`` of that mass.  Stockfish-derived value targets
    and every other replay column are copied unchanged.  Source and sidecar
    fingerprints are checked before a shard is edited, and the completed
    output is published by one directory rename.

``audit``
    Reconstruct the stored d9 target on the frozen deep-SF audit set, apply the
    exact same treatment, and bank paired per-position regret observations and
    a pre-training gate verdict.

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
MIX_SCOPES = ("global", "top-max-ties")
TREATMENT_ALGORITHMS = {
    "global": "legal-normalized-global-arithmetic-v1",
    "top-max-ties": "stored-top-set-only-v1",
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
_COMPRESSOR = Blosc(cname="zstd", clevel=2, shuffle=Blosc.BITSHUFFLE)


def validate_alpha(alpha: float) -> float:
    """Return a real treatment weight, refusing identity or extrapolation."""
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(
            f"--alpha must be finite, positive, and at most 1, got {alpha!r}",
        )
    return value


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


def arithmetic_policy_mix(
    source: np.ndarray,
    bt4: np.ndarray,
    legal_mask: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Literal probability mixture, legal-normalized on both input teachers."""
    weight = validate_alpha(alpha)
    sf = _normalized_legal(source, legal_mask, name="source policy")
    lc0 = _normalized_legal(bt4, legal_mask, name="BT4 policy")
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
) -> np.ndarray:
    """Apply the selected literal probability-space treatment.

    ``top-max-ties`` changes only rows whose stored source target has more
    than one legal maximum.  It preserves the source target's total mass on
    that tied set and uses BT4 only to redistribute ``alpha`` of that mass.
    A unique source maximum is therefore byte-identical after storage.
    """
    if scope not in MIX_SCOPES:
        raise ValueError(f"unknown mix scope {scope!r}; expected one of {MIX_SCOPES}")
    weight = validate_alpha(alpha)
    if scope == "global":
        return arithmetic_policy_mix(source, bt4, legal_mask, alpha=weight)

    source_stored = np.asarray(source)
    legal = np.asarray(legal_mask) != 0
    _normalized_legal(source_stored, legal, name="source policy")
    if bool(np.any(source_stored[~legal] != 0.0)):
        raise ValueError("source policy has nonzero mass on illegal moves")
    lc0 = _normalized_legal(bt4, legal, name="BT4 policy")
    legal_source = np.where(legal, source_stored, -np.inf)
    top = legal & (legal_source == np.max(legal_source, axis=1, keepdims=True))
    top_count = top.sum(axis=1)
    if bool(np.any(top_count <= 0)):
        raise ValueError("source policy has a row without a legal maximum")

    bt4_top = np.where(top, lc0, 0.0)
    source_values = source_stored.astype(np.float64, copy=False)
    source_top = np.where(top, source_values, 0.0)
    top_mass = source_top.sum(axis=1, keepdims=True)
    bt4_top_mass = bt4_top.sum(axis=1, keepdims=True)
    no_bt4_tie_mass = np.flatnonzero(bt4_top_mass[:, 0] <= 0.0)
    if no_bt4_tie_mass.size:
        raise ValueError(
            "BT4 policy has no mass on the source top-tie set for "
            f"{no_bt4_tie_mass.size} rows",
        )
    bt4_top /= bt4_top_mass
    redistributed = (1.0 - weight) * source_top + weight * top_mass * bt4_top
    mixed = source_values.copy()
    mixed[top] = redistributed[top]
    # Make unique-maximum rows an exact identity arm, not merely a
    # floating-point reconstruction of one. Replay policies are stored in
    # float16 and their rounded row sum need not be exactly one.
    unique = top_count == 1
    mixed[unique] = source_stored[unique]
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


def _load_audit_receipt(
    path: Path,
    *,
    alpha: float,
    scope: str,
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
    treatment = receipt.get("treatment")
    if not isinstance(treatment, dict):
        bad["treatment"] = (
            treatment,
            {
                "scope": scope,
                "alpha": alpha,
                "algorithm": TREATMENT_ALGORITHMS[scope],
            },
        )
    else:
        if treatment.get("scope") != scope:
            bad["treatment.scope"] = (treatment.get("scope"), scope)
        if treatment.get("alpha") != alpha:
            bad["treatment.alpha"] = (treatment.get("alpha"), alpha)
        if treatment.get("algorithm") != TREATMENT_ALGORITHMS[scope]:
            bad["treatment.algorithm"] = (
                treatment.get("algorithm"),
                TREATMENT_ALGORITHMS[scope],
            )
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
    if not isinstance(gate, dict) or gate.get("training_permitted") is not True:
        bad["gate.training_permitted"] = (
            gate.get("training_permitted") if isinstance(gate, dict) else None,
            True,
        )
    if bad:
        raise ValueError(f"audit receipt does not admit this treatment: {bad}")
    return receipt


def audit_mix(args: argparse.Namespace) -> int:
    """Score a source/BT4 policy treatment on the frozen deep-SF ruler."""
    alpha = validate_alpha(float(args.alpha))
    scope = str(args.scope)
    if scope not in MIX_SCOPES:
        raise SystemExit(f"--scope must be one of {MIX_SCOPES}")
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
        )
        # The materializer casts back to the source policy_target dtype. The
        # 20M source is float16, so audit what the trainer will actually read.
        candidate = candidate.astype(np.float16).astype(np.float32)
        candidate = _normalized_legal(candidate, legal, name="stored candidate")[0]
        regrets = move_regrets(position, legal_ucis)
        source_pair = expected_and_top1_regret(source, regrets)
        candidate_pair = expected_and_top1_regret(candidate, regrets)
        source_metrics.append(source_pair)
        candidate_metrics.append(candidate_pair)
        source_top = int(np.argmax(source))
        candidate_top = int(np.argmax(candidate))
        top_count = int(np.sum(source == np.max(source)))
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
    report: dict[str, Any] = {
        "schema": 1,
        "kind": "bt4_policy_mix_frozen_deep_sf_audit",
        "treatment": {
            "scope": scope,
            "alpha": alpha,
            "algorithm": TREATMENT_ALGORITHMS[scope],
        },
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
            "kill_if": "paired 95% interval is wholly above zero",
            "verdict": verdict,
            "training_permitted": verdict != "kill",
            "value_gate": "not applicable; mix leaves all value columns byte-identical",
        },
        "per_position": per_position,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_path, report)
    overall = report["overall"]
    print(
        f"[bt4-audit] {verdict}: n={len(positions)}, "
        f"expected delta={overall['expected_regret_delta_cp']['mean']:.4f} cp "
        f"[{expected_ci[0]:.4f}, {expected_ci[1]:.4f}], "
        f"top1 delta={overall['top1_regret_delta_cp']['mean']:.4f} cp "
        f"[{top1_ci[0]:.4f}, {top1_ci[1]:.4f}] -> {out_path}",
        flush=True,
    )
    return 0 if verdict != "kill" else 2


def mix_corpus(args: argparse.Namespace) -> int:
    alpha = validate_alpha(float(args.alpha))
    scope = str(args.scope)
    if scope not in MIX_SCOPES:
        raise SystemExit(f"--scope must be one of {MIX_SCOPES}")
    source_dir = Path(args.shards).resolve()
    sidecar_dir = Path(args.sidecar).resolve()
    out_dir = Path(args.out).resolve()
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
    source_rows, source_policy_dtypes = _source_policy_layout(source_paths)
    _validate_source_target_contract(
        derive_summary_original,
        storage_dtypes=source_policy_dtypes,
    )
    audit_receipt_path = Path(args.audit_receipt).resolve()
    _load_audit_receipt(audit_receipt_path, alpha=alpha, scope=scope)
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
        "remap": remap_provenance(),
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
            destination_path = writing / source_path.name
            destination: Any = zarr.open_group(str(destination_path), mode="a")
            rows = int(source["x"].shape[0])
            chunk_rows = int(source[POLICY_FIELD].chunks[0])
            for start in range(0, rows, chunk_rows):
                stop = min(rows, start + chunk_rows)
                sf_stored = np.asarray(source[POLICY_FIELD][start:stop])
                bt4_stored = np.asarray(side[SIDECAR_POLICY_FIELD][start:stop])
                legal = np.asarray(source["legal_mask"][start:stop])
                sf = _normalized_legal(sf_stored, legal, name=f"{source_path}:source")
                bt4 = _normalized_legal(bt4_stored, legal, name=f"{source_path}:BT4")
                mixed = mix_policy_targets(
                    sf_stored,
                    bt4,
                    legal,
                    alpha=alpha,
                    scope=scope,
                )
                stored = mixed.astype(destination[POLICY_FIELD].dtype, copy=False)
                destination[POLICY_FIELD][start:stop] = stored
                reread = np.asarray(destination[POLICY_FIELD][start:stop])
                if not np.array_equal(reread, stored):
                    raise ValueError(f"{destination_path}: policy write/read mismatch")

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
                stats.changed_rows += int(changed.sum())
                stats.source_top_tied_rows += int(source_tied.sum())
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
            "kind": scope,
            "algorithm": TREATMENT_ALGORITHMS[scope],
            "formula": (
                "mixed=(1-alpha)*stored_stockfish+alpha*bt4_raw_one_eval"
                if scope == "global"
                else "redistribute alpha of stored source top-tie mass by BT4 prior"
            ),
            "alpha": alpha,
            "source_dir": str(source_dir),
            "sidecar_dir": str(sidecar_dir),
            "sidecar_summary": {
                "path": str(side_summary_path),
                "sha256": file_sha256(side_summary_path),
            },
            "audit_receipt": {
                "path": str(audit_receipt_path),
                "sha256": file_sha256(audit_receipt_path),
            },
            "rows": stats.rows,
            "shards": stats.shards,
            "changed_rows": stats.changed_rows,
            "changed_fraction": stats.changed_rows / denom,
            "source_top_tied_rows": stats.source_top_tied_rows,
            "source_top_tied_fraction": stats.source_top_tied_rows / denom,
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

        derive_summary_path = writing / DERIVE_SUMMARY
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
        f"[bt4-mix] complete: {stats.rows} rows, alpha={alpha:.6f}, "
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
    audit.add_argument("--boot", type=int, default=10_000)
    audit.add_argument("--seed", type=int, default=20260903)
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
