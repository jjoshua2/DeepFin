"""Coverage audit only: stored SF labels describe action-conditioned children.

Never compare these values against root predictions or reconstruct child inputs.
Call reads only within an externally hard-limited reviewed process.
"""

import hashlib
import json
import math
from pathlib import Path
import numpy as np
from chess_anti_engine.replay.shard import load_shard_arrays
from scripts.stored_wdl_categorical import _recheck
from scripts.stored_wdl_direct import _bounded_bytes
from scripts.stored_wdl_rows import _charge_rows

FIELDS = ("sf_wdl", "has_sf_wdl", "sf_label_meta", "has_sf_label_meta")


def summarize(values: dict) -> dict:
    raw, mask, meta, mmask = (values[k] for k in FIELDS)
    n = len(raw)
    if (
        not 1 <= n <= 256
        or raw.shape != (n, 3)
        or raw.dtype != np.float16
        or meta.shape != (n, 6)
        or meta.dtype != np.int32
    ):
        raise ValueError("bounded native label/meta shapes required")
    for m in (mask, mmask):
        if m.dtype != np.uint8 or m.shape != (n,) or not np.isin(m, (0, 1)).all():
            raise ValueError("explicit binary SF masks required")
    v = raw.astype(np.float64)
    present = mask == 1
    mp = mmask == 1
    finite = np.isfinite(v).all(1)
    nonnegative = (v >= 0).all(1)
    bounded = (v <= 1).all(1)
    sums = v.sum(1)
    near = np.abs(sums - 1) <= 0.001

    def column(i):
        a = meta[mp, i]
        known = a[a >= 0]
        return {
            "present_metadata_rows": int(mp.sum()),
            "unknown_negative_rows": int((a < 0).sum()),
            "known_min": int(known.min()) if len(known) else None,
            "known_max": int(known.max()) if len(known) else None,
        }

    return {
        "rows": n,
        "sf_present": int(present.sum()),
        "sf_missing": int((~present).sum()),
        "present_finite": int((present & finite).sum()),
        "present_nonfinite": int((present & ~finite).sum()),
        "present_negative": int((present & (v < 0).any(1)).sum()),
        "present_above_one": int((present & (v > 1).any(1)).sum()),
        "present_near_simplex": int(
            (present & finite & nonnegative & bounded & near).sum()
        ),
        "present_max_abs_sum_error": float(np.abs(sums[present & finite] - 1).max())
        if (present & finite).any()
        else None,
        "label_and_metadata_present": int((present & mp).sum()),
        "recorded_nodes": column(0),
        "recorded_depth": column(1),
        "range_count_semantics": "Explicit comparisons: NaN is neither negative nor above one; signed infinities count in their respective range and nonfinite categories",
        "recorded_effort_semantics": "Negative metadata is unknown; zero is recorded zero, not proof of positive search effort",
        "simplex_tolerance": 0.001,
        "normalization_performed": False,
        "interpretation": "Recorded action-conditioned SF coverage only; no root truth, successor identity, model score or strength claim",
    }


def read_sf_coverage(path: Path, accepted: dict) -> dict:
    accepted = json.loads(json.dumps(accepted))
    _recheck(path, accepted)
    n = len(accepted["row_ids"])
    declarations = {}
    for name in FIELDS:
        raw = _bounded_bytes(path / name / ".zarray", 65536)
        m = json.loads(raw)
        shape, dtype = (
            ([n, 3], "<f2")
            if name == "sf_wdl"
            else (([n, 6], "<i4") if name == "sf_label_meta" else ([n], "|u1"))
        )
        chunks = m.get("chunks", [])
        if (
            m.get("zarr_format") != 2
            or m.get("shape") != shape
            or m.get("dtype") != dtype
            or len(chunks) != len(shape)
            or any(type(c) is not int or c < 1 for c in chunks)
            or math.prod(chunks) * np.dtype(dtype).itemsize > 1024**2
        ):
            raise ValueError("bounded exact SF declarations required")
        declarations[name] = {"sha256": hashlib.sha256(raw).hexdigest(), **m}
    arrays, _ = load_shard_arrays(path, lazy=True)
    for name, m in declarations.items():
        a = arrays[name]
        if (
            list(a.shape) != m["shape"]
            or list(a.chunks) != m["chunks"]
            or np.dtype(a.dtype).str != m["dtype"]
        ):
            raise ValueError("opened SF metadata mismatch")
    if sum(_charge_rows(arrays[name], 0, n) for name in FIELDS) > 1024**2:
        raise ValueError("SF decoded byte cap exceeded")
    values = {name: np.asarray(arrays[name][:]).copy() for name in FIELDS}
    summary = summarize(values)
    for name, m in declarations.items():
        if (
            hashlib.sha256(_bounded_bytes(path / name / ".zarray", 65536)).hexdigest()
            != m["sha256"]
        ):
            raise ValueError("SF metadata changed during read")
    _recheck(path, accepted)
    absent = {
        name: not (path / name / ".zarray").exists()
        for name in ("sf_played_move_index", "has_sf_played_move")
    }
    return {
        "raw_arrays": values,
        "summary": summary,
        "row_ids": accepted["row_ids"],
        "group_ids": accepted["group_ids"],
        "rows_sha256": accepted["rows_sha256"],
        "attrs_sha256": accepted["attrs_sha256"],
        "producer_seal_sha256": accepted["producer_seal_sha256"],
        "source_shard": str(path.resolve()),
        "metadata": declarations,
        "raw_value_sha256": {
            k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
            for k, v in values.items()
        },
        "physical_action_fields_absent": absent,
        "child_identity_qualified": False,
        "child_comparison_blocker": "Exact played action and full successor/history binding not established; SF recommended move is not played action",
    }
