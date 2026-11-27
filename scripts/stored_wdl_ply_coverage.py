"""Exact supplied game/ply successor-key coverage only.

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

FIELDS = ("ply_index", "has_ply_index")


def summarize(plies: np.ndarray, mask: np.ndarray, groups: tuple) -> dict:
    groups = tuple(groups)
    n = len(groups)
    if not 1 <= n <= 256 or any(not isinstance(g, str) or not g for g in groups):
        raise ValueError("bounded supplied groups required")
    if (
        plies.dtype != np.int32
        or plies.shape != (n,)
        or mask.dtype != np.uint8
        or mask.shape != (n,)
        or not np.isin(mask, (0, 1)).all()
    ):
        raise ValueError("native ply and binary presence required")
    keys: dict[tuple[str, int], list[int]] = {}
    valid = (mask == 1) & (plies >= 0)
    for i in range(n):
        if valid[i]:
            keys.setdefault((groups[i], int(plies[i])), []).append(i)
    pairs = []
    missing = ambiguous = duplicate_parent = 0
    for i in range(n):
        if not valid[i]:
            continue
        key = (groups[i], int(plies[i]))
        if len(keys[key]) != 1:
            duplicate_parent += 1
            continue
        successor = keys.get((groups[i], int(plies[i]) + 1), [])
        if not successor:
            missing += 1
        elif len(successor) != 1:
            ambiguous += 1
        else:
            pairs.append([i, successor[0]])
    return {
        "rows": n,
        "present": int((mask == 1).sum()),
        "missing_presence": int((mask == 0).sum()),
        "present_negative_ply": int(((mask == 1) & (plies < 0)).sum()),
        "valid_rows": int(valid.sum()),
        "unique_valid_keys": len(keys),
        "duplicate_key_count": sum(len(v) > 1 for v in keys.values()),
        "duplicate_parent_rows": duplicate_parent,
        "missing_successor_rows": missing,
        "ambiguous_successor_rows": ambiguous,
        "unique_successor_pairs": pairs,
        "unique_successor_count": len(pairs),
        "all_rows_have_unique_successor_key": len(pairs) == n,
        "interpretation": "Supplied game/ply key coverage only; no legal-transition, history, played-action or child-position qualification",
    }


def read_ply_coverage(path: Path, accepted: dict) -> dict:
    accepted = json.loads(json.dumps(accepted))
    _recheck(path, accepted)
    n = len(accepted["row_ids"])
    declarations = {}
    for name in FIELDS:
        raw = _bounded_bytes(path / name / ".zarray", 65536)
        m = json.loads(raw)
        shape, dtype = [n], "<i4" if name == "ply_index" else "|u1"
        chunks = m.get("chunks", [])
        if (
            m.get("zarr_format") != 2
            or m.get("shape") != shape
            or m.get("dtype") != dtype
            or len(chunks) != len(shape)
            or any(type(c) is not int or c < 1 for c in chunks)
            or math.prod(chunks) * np.dtype(dtype).itemsize > 1024**2
        ):
            raise ValueError("bounded exact ply declarations required")
        declarations[name] = {"sha256": hashlib.sha256(raw).hexdigest(), **m}
    arrays, _ = load_shard_arrays(path, lazy=True)
    for name, m in declarations.items():
        a = arrays[name]
        if (
            list(a.shape) != m["shape"]
            or list(a.chunks) != m["chunks"]
            or np.dtype(a.dtype).str != m["dtype"]
        ):
            raise ValueError("opened ply metadata mismatch")
    if sum(_charge_rows(arrays[name], 0, n) for name in FIELDS) > 1024**2:
        raise ValueError("ply decoded byte cap exceeded")
    values = {name: np.asarray(arrays[name][:]).copy() for name in FIELDS}
    summary = summarize(
        values["ply_index"], values["has_ply_index"], tuple(accepted["group_ids"])
    )
    for name, m in declarations.items():
        if (
            hashlib.sha256(_bounded_bytes(path / name / ".zarray", 65536)).hexdigest()
            != m["sha256"]
        ):
            raise ValueError("ply metadata changed during read")
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
