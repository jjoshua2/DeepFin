"""Same-row producer search-target agreement, not truth or final training loss.

Extraction requires an external hard-limited process. No model forwards.
Float16 source labels are retained and explicit bounded normalization reported.
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
from scripts.stored_wdl_compare import _simplex, targets_digest

SUM_TOLERANCE = 1e-3
LOG_FLOOR = 1e-12
DESCRIPTION = (
    "producer search_wdl; source-declared STM W,D,L; float16 explicitly renormalized"
)


def normalize_targets(raw: np.ndarray) -> tuple[np.ndarray, dict]:
    raw = np.array(raw, copy=True)
    if (
        raw.dtype != np.dtype("float16")
        or raw.ndim != 2
        or raw.shape[1] != 3
        or not 1 <= len(raw) <= 256
    ):
        raise ValueError("bounded source float16 WDL required")
    values = raw.astype(np.float64)
    sums = values.sum(axis=1)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or (np.abs(sums - 1) > SUM_TOLERANCE).any()
    ):
        raise ValueError("invalid or excessive float16 simplex rounding")
    normalized = values / sums[:, None]
    return normalized, {
        "sum_tolerance": SUM_TOLERANCE,
        "max_abs_sum_error": float(np.abs(sums - 1).max()),
        "max_abs_normalization_correction": float(np.abs(normalized - values).max()),
        "raw_float16_sha256": hashlib.sha256(raw.astype("<f2").tobytes()).hexdigest(),
    }


def read_search_targets(path: Path, accepted: dict) -> dict:
    """Verify every original row/flag again; never filter or select by predictions."""
    accepted = json.loads(json.dumps(accepted))
    n = len(accepted["row_ids"])
    current = read_direct_shard(
        path,
        expected_attrs_sha256=accepted["attrs_sha256"],
        expected_seal_sha256=accepted["producer_seal_sha256"],
        expected_rows=n,
        expected_history_rep_fix=True,
    )
    for key in ("rows_sha256", "row_ids", "group_ids", "consumed_values_sha256"):
        if current[key] != accepted[key]:
            raise ValueError("original ordered input/identity/flags changed")
    metadata = {}
    for name, shape, dtype in (
        ("search_wdl", [n, 3], "<f2"),
        ("has_search_wdl", [n], "|u1"),
    ):
        raw = _bounded_bytes(path / name / ".zarray", 65536)
        m = json.loads(raw)
        chunks = m.get("chunks", [])
        if (
            m.get("zarr_format") != 2
            or m.get("shape") != shape
            or m.get("dtype") != dtype
            or len(chunks) != len(shape)
            or any(type(c) is not int or c < 1 for c in chunks)
            or math.prod(chunks) * np.dtype(dtype).itemsize > 1024**2
        ):
            raise ValueError("bounded exact target metadata required")
        metadata[name] = {"sha256": hashlib.sha256(raw).hexdigest(), **m}
    arrays, _ = load_shard_arrays(path, lazy=True)
    for name, m in metadata.items():
        a = arrays[name]
        if (
            list(a.shape) != m["shape"]
            or list(a.chunks) != m["chunks"]
            or np.dtype(a.dtype).str != m["dtype"]
        ):
            raise ValueError("target metadata changed before decode")
        if _charge_rows(a, 0, n) > 1024**2:
            raise ValueError("target decode budget exceeded")
    presence = np.asarray(arrays["has_search_wdl"][:]).copy()
    raw_targets = np.asarray(arrays["search_wdl"][:]).copy()
    if not np.all(presence == 1):
        raise ValueError(
            "ALL original rows must have search targets; filtering prohibited"
        )
    target, correction = normalize_targets(raw_targets)
    for name, m in metadata.items():
        if (
            hashlib.sha256(_bounded_bytes(path / name / ".zarray", 65536)).hexdigest()
            != m["sha256"]
        ):
            raise ValueError("target metadata changed during read")
    if (
        hashlib.sha256(
            _bounded_bytes(path / "directory_producer_seal.json", 4_000_000)
        ).hexdigest()
        != accepted["producer_seal_sha256"]
    ):
        raise ValueError("seal changed during target read")
    if (
        hashlib.sha256(_bounded_bytes(path / ".zattrs", 65536)).hexdigest()
        != accepted["attrs_sha256"]
    ):
        raise ValueError("attributes changed during target read")
    return {
        "raw_targets": raw_targets,
        "targets": target,
        "normalization": correction,
        "target_metadata": metadata,
        "presence_sha256": hashlib.sha256(presence.tobytes()).hexdigest(),
        "target_description": DESCRIPTION,
        "producer_nonfinite_draw_fallback_incidence": "unknown; presence does not distinguish fallback from genuine draw",
        "targets_sha256": targets_digest(
            target,
            accepted["row_ids"],
            accepted["group_ids"],
            rows_sha256=accepted["rows_sha256"],
            target_description=DESCRIPTION,
        ),
    }


def score_agreement(donor, candidate, targets, groups) -> dict:
    """Arithmetic-only kernel; actual saved-output scoring must use bound_agreement."""
    groups = tuple(groups)
    if (
        not 1 <= len(groups) <= 256
        or any(not isinstance(g, str) or not g for g in groups)
        or len(set(groups)) < 2
    ):
        raise ValueError("bounded multiple supplied groups required")
    a, b, t = (
        _simplex(v, len(groups), "WDL").copy() for v in (donor, candidate, targets)
    )
    result: dict = {
        "target_description": DESCRIPTION,
        "log_probability_floor": LOG_FLOOR,
        "interpretation": "Producer search-target agreement only; not final training mixture, correctness, heldout or strength",
    }
    for name, loss in (
        ("brier", lambda p: ((p - t) ** 2).sum(1)),
        ("logloss", lambda p: -(t * np.log(np.maximum(p, LOG_FLOOR))).sum(1)),
    ):
        da, db = loss(a), loss(b)
        delta = db - da
        per = [
            {
                "group": g,
                "rows": groups.count(g),
                "candidate_minus_donor": float(delta[np.asarray(groups) == g].mean()),
            }
            for g in sorted(set(groups))
        ]
        result[name] = {
            "donor_row_mean": float(da.mean()),
            "candidate_row_mean": float(db.mean()),
            "candidate_minus_donor_row_mean": float(delta.mean()),
            "candidate_minus_donor_equal_group_mean": float(
                np.mean([v["candidate_minus_donor"] for v in per])
            ),
            "per_supplied_group": per,
            "lower_is_better_agreement": True,
        }
    return result


def bound_agreement(
    *,
    accepted_bytes: bytes,
    plan_bytes: bytes,
    donor_bytes: bytes,
    candidate_bytes: bytes,
    expected_hashes: dict[str, str],
    extracted: dict,
) -> dict:
    """Validate externally pinned immutable receipt bytes before real scoring.

    Expected hashes are independently supplied identity pins, not authenticity.
    Target identity includes original ordered inputs, rows, groups and semantics.
    """
    blobs = {
        "accepted": accepted_bytes,
        "plan": plan_bytes,
        "donor": donor_bytes,
        "candidate": candidate_bytes,
    }
    for key, blob in blobs.items():
        if len(blob) > 4 * 1024**2 or hashlib.sha256(
            blob
        ).hexdigest() != expected_hashes.get(key):
            raise ValueError("saved receipt identity mismatch")
    accepted, plan, donor, candidate = (json.loads(blobs[k]) for k in blobs)
    rows, groups = accepted["row_ids"], accepted["group_ids"]
    digest = accepted["rows_sha256"]
    if (
        plan["rows_sha256"] != digest
        or accepted["source_revision"] != plan["source_revision"]
    ):
        raise ValueError("input/source plan binding mismatch")
    if (
        len(rows) != 254
        or len(set(rows)) != 254
        or len(groups) != 254
        or len(set(groups)) != 3
    ):
        raise ValueError("exact predeclared 254 rows and three groups required")
    if extracted["target_description"] != DESCRIPTION:
        raise ValueError("target semantics mismatch")
    normalized, correction = normalize_targets(extracted["raw_targets"])
    if correction != extracted["normalization"] or not np.array_equal(
        normalized, extracted["targets"]
    ):
        raise ValueError("raw target normalization binding mismatch")
    target_digest = targets_digest(
        extracted["targets"],
        rows,
        groups,
        rows_sha256=digest,
        target_description=DESCRIPTION,
    )
    if target_digest != extracted["targets_sha256"]:
        raise ValueError("ordered target identity mismatch")
    probabilities = []
    bindings = {}
    for name, model, receipt in (
        ("donor", "donor", donor),
        ("candidate", "cap40", candidate),
    ):
        if (
            receipt.get("schema") != "isolated-stored-wdl-forward/v1"
            or receipt.get("sha256") != plan["models"][model]["sha256"]
            or receipt.get("rows_sha256") != digest
            or receipt.get("rows") != 254
            or receipt.get("source_revision") != plan["source_revision"]
            or receipt.get("architecture") != plan["architecture"]
            or receipt.get("loader_sha256")
            != plan["source_hashes"]["scripts/stored_wdl_checkpoint.py"]
            or receipt.get("batch_size") != 16
            or receipt.get("wdl_order_declared") != ["W", "D", "L"]
            or receipt.get("pov_declared") != "stored row side to move"
            or receipt.get("inference_performed") is not True
        ):
            raise ValueError(
                "saved forward model/input/source/semantics binding mismatch"
            )
        pred = _simplex(
            np.asarray(receipt["probabilities"]), 254, "saved probabilities"
        ).copy()
        probabilities.append(pred)
        bindings[name] = {
            "receipt_sha256": expected_hashes[name],
            "checkpoint_sha256": receipt["sha256"],
            "probabilities_sha256": hashlib.sha256(
                pred.astype("<f8").tobytes()
            ).hexdigest(),
        }
    result = score_agreement(probabilities[0], probabilities[1], extracted["targets"], groups)
    result["bindings"] = {
        "saved_receipt_sha256": dict(expected_hashes),
        "models": bindings,
        "rows_sha256": digest,
        "targets_sha256": target_digest,
        "source_revision": plan["source_revision"],
        "architecture": plan["architecture"],
        "target_normalization": extracted["normalization"],
    }
    result["producer_draw_fallback_incidence"] = "unknown; all draw targets retained"
    result["semantic_POV_control"] = (
        "source-declared only; real semantic control unperformed"
    )
    return result
