"""Admit exact stored categorical labels/masks for a frozen descriptive row set.

No target reconstruction or model work. Execute reads only in a hard-limited
process. Returned targets are float32 casts of native float16 values, NOT already
renormalized: production soft_cross_entropy performs its own normalization.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from chess_anti_engine.replay.shard import load_shard_arrays
from scripts.stored_wdl_direct import read_direct_shard, _bounded_bytes
from scripts.stored_wdl_rows import _charge_rows

FIELDS = ("categorical_target", "has_categorical", "is_network_turn", "has_is_network_turn")
MAX_TARGET_DECODE = 1024**2
SUM_TOLERANCE = 1e-3
SEMANTICS = (
    "stored 32-bin categorical target; float16 cast to float32 unchanged; "
    "native soft_cross_entropy normalizes by row sum clamp_min(1e-8); "
    "effective categorical mask=is_network_turn*has_categorical; no reconstruction"
)


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def admit_values(raw: np.ndarray, has_cat: np.ndarray, net: np.ndarray,
                 has_net: np.ndarray) -> dict:
    """Validate but retain every ordered row, including explicitly masked rows."""
    raw, has_cat, net, has_net = (np.array(v, copy=True) for v in (raw, has_cat, net, has_net))
    n = raw.shape[0] if raw.ndim else 0
    if raw.dtype != np.dtype("float16") or raw.shape != (n, 32) or not 1 <= n <= 256:
        raise ValueError("bounded native float16 N32 categorical targets required")
    for mask in (has_cat, net, has_net):
        if mask.dtype != np.dtype("uint8") or mask.shape != (n,) or not np.isin(mask, (0, 1)).all():
            raise ValueError("explicit binary uint8 row masks required")
    if not np.all(has_net == 1):
        raise ValueError("explicit network-turn presence required; no legacy default")
    targets = raw.astype(np.float32)
    sums = targets.sum(axis=1, dtype=np.float32)
    if (not np.isfinite(targets).all() or (targets < 0).any() or (targets > 1).any() or
            (np.abs(sums[(has_cat == 1) | (sums != 0)] - 1) > SUM_TOLERANCE).any()):
        raise ValueError("invalid present categorical distribution or excessive quantization")
    # Keep absent rows exactly, including all-zero rows; no filtering or repair.
    denominator = np.maximum(sums, np.float32(1e-8))
    normalized = targets / denominator[:, None]
    return {"raw_targets": raw, "targets": targets,
            "raw_categorical_mask": has_cat, "raw_network_mask": net,
            "raw_network_presence": has_net,
            "categorical_mask": has_cat.astype(np.float32), "network_mask": net.astype(np.float32),
            "effective_mask": (has_cat * net).astype(np.float32),
            "normalization": {"performed": False, "applied_by": "native soft_cross_entropy",
                              "row_sum_clamp_min": 1e-8, "admission_sum_tolerance": SUM_TOLERANCE,
                              "max_present_abs_sum_error": float(np.max(
                                  np.abs(sums[has_cat == 1] - 1), initial=0)),
                              "max_potential_abs_correction": float(np.max(np.abs(normalized-targets)))},
            "source_value_sha256": {name: _hash(a) for name, a in zip(
                FIELDS, (raw, has_cat, net, has_net), strict=True)},
            "mask_counts": {"rows": n, "network": int(net.sum()),
                            "categorical_present": int(has_cat.sum()),
                            "effective_categorical": int((has_cat * net).sum())}}


def _recheck(path: Path, accepted: dict) -> dict:
    if str(path.resolve()) != accepted["source_shard"]:
        raise ValueError("frozen source shard path mismatch")
    current = read_direct_shard(
        path, expected_attrs_sha256=accepted["attrs_sha256"],
        expected_seal_sha256=accepted["producer_seal_sha256"],
        expected_rows=len(accepted["row_ids"]),
        expected_history_rep_fix=accepted["expected_history_rep_fix"],
    )
    for key in ("rows_sha256", "row_ids", "group_ids", "consumed_values_sha256", "array_metadata"):
        if current[key] != accepted[key]:
            raise ValueError("original ordered inputs/identities/flags/metadata changed")
    return current


def read_categorical_targets(path: Path, accepted: dict) -> dict:
    """Rebind fixed inputs before and after bounded native categorical/mask reads.

    Accepted must be the frozen direct-reader JSON receipt, not arbitrary newly
    generated identities. This function verifies consistency, not its authority.
    """
    accepted = json.loads(json.dumps(accepted))
    before = _recheck(path, accepted)
    n = len(accepted["row_ids"])
    metadata = {}
    for name in FIELDS:
        raw = _bounded_bytes(path / name / ".zarray", 65536)
        m = json.loads(raw)
        shape, dtype = ([n, 32], "<f2") if name == "categorical_target" else ([n], "|u1")
        chunks = m.get("chunks", [])
        if (m.get("zarr_format") != 2 or m.get("shape") != shape or m.get("dtype") != dtype or
                len(chunks) != len(shape) or any(type(c) is not int or c < 1 for c in chunks) or
                math.prod(chunks) * np.dtype(dtype).itemsize > MAX_TARGET_DECODE):
            raise ValueError("bounded exact categorical/mask declarations required")
        metadata[name] = {"sha256": hashlib.sha256(raw).hexdigest(), **m}
    arrays, _ = load_shard_arrays(path, lazy=True)
    for name, m in metadata.items():
        a = arrays[name]
        if (list(a.shape) != m["shape"] or list(a.chunks) != m["chunks"] or
                np.dtype(a.dtype).str != m["dtype"]):
            raise ValueError("categorical metadata changed before decode")
    cost = sum(_charge_rows(arrays[name], 0, n) for name in FIELDS)
    if cost > MAX_TARGET_DECODE:
        raise ValueError("categorical decoded chunk budget exceeded")
    values = [np.asarray(arrays[name][:]).copy() for name in FIELDS]
    admitted = admit_values(*values)
    for name, m in metadata.items():
        if hashlib.sha256(_bounded_bytes(path / name / ".zarray", 65536)).hexdigest() != m["sha256"]:
            raise ValueError("categorical metadata changed during read")
    after = _recheck(path, accepted)
    binding = {"rows_sha256": accepted["rows_sha256"], "row_ids": accepted["row_ids"],
               "group_ids": accepted["group_ids"], "source_shard": accepted["source_shard"],
               "attrs_sha256": accepted["attrs_sha256"],
               "producer_seal_sha256": accepted["producer_seal_sha256"],
               "source_value_sha256": admitted["source_value_sha256"], "target_metadata": metadata,
               "semantics": SEMANTICS}
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**admitted, "target_description": SEMANTICS, "categorical_binding": binding,
            "categorical_sha256": digest, "target_metadata": metadata,
            "decoded_chunk_bytes_charged": cost + before["decoded_chunk_bytes_charged"] + after["decoded_chunk_bytes_charged"],
            "limits": {"target_decoded_bytes": MAX_TARGET_DECODE, "rows": 256},
            "interpretation": "Native stored target/mask admission only; no reconstruction, loss, gradient or strength measurement"}
