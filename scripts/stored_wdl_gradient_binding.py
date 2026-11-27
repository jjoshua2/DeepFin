"""Identity gates for a bounded offline shared-value-body probe, no execution."""

import hashlib
import json
import numpy as np
from scripts.stored_wdl_compare import rows_digest


def pinned_json(blob: object, expected_sha256: str) -> dict:
    if (
        not isinstance(blob, bytes)
        or len(blob) > 4 * 1024**2
        or hashlib.sha256(blob).hexdigest() != expected_sha256
    ):
        raise ValueError("pinned receipt identity mismatch")
    result = json.loads(blob)
    if not isinstance(result, dict):
        raise ValueError("object receipt required")
    return result


def bind_arrays(
    accepted: dict,
    search_receipt: dict,
    x: np.ndarray,
    raw_search: np.ndarray,
    categorical: dict,
) -> dict:
    """Snapshot bounded admitted arrays then revalidate all ordered bindings.

    Caller must obtain accepted/search receipts through independently pinned
    bytes, and record new categorical admission digest before model execution.
    """
    arrays: dict = {
        "x": (x, (len(accepted["row_ids"]), 175, 8, 8), np.dtype("float32")),
        "search": (raw_search, (len(accepted["row_ids"]), 3), np.dtype("float16")),
        "cat": (
            categorical["raw_targets"],
            (len(accepted["row_ids"]), 32),
            np.dtype("float16"),
        ),
    }
    n = len(accepted["row_ids"])
    if not 1 <= n <= 256:
        raise ValueError("bounded rows required")
    for key, (a, shape, dtype) in arrays.items():
        if not isinstance(a, np.ndarray) or a.shape != shape or a.dtype != dtype:
            raise ValueError("bounded exact input array required")
        arrays[key] = (a.copy(), shape, dtype)
    x, raw_search, raw_cat = (arrays[k][0] for k in ("x", "search", "cat"))
    if (
        rows_digest(x, accepted["row_ids"], accepted["group_ids"])
        != accepted["rows_sha256"]
    ):
        raise ValueError("ordered input identity mismatch")
    binding = search_receipt["bindings"]
    if (
        binding["rows_sha256"] != accepted["rows_sha256"]
        or hashlib.sha256(raw_search.astype("<f2").tobytes()).hexdigest()
        != binding["target_normalization"]["raw_float16_sha256"]
    ):
        raise ValueError("search target identity mismatch")
    cb = json.loads(json.dumps(categorical["categorical_binding"]))
    sha = hashlib.sha256(
        json.dumps(cb, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if sha != categorical["categorical_sha256"]:
        raise ValueError("categorical receipt identity mismatch")
    for key in (
        "rows_sha256",
        "row_ids",
        "group_ids",
        "source_shard",
        "attrs_sha256",
        "producer_seal_sha256",
    ):
        if cb[key] != accepted[key]:
            raise ValueError("categorical ordered binding mismatch")
    masks = []
    for key, field in [
        ("raw_categorical_mask", "has_categorical"),
        ("raw_network_mask", "is_network_turn"),
        ("raw_network_presence", "has_is_network_turn"),
    ]:
        a = categorical[key]
        if not isinstance(a, np.ndarray) or a.dtype != np.uint8 or a.shape != (n,):
            raise ValueError("bounded raw mask required")
        a = a.copy()
        if (
            not np.isin(a, (0, 1)).all()
            or hashlib.sha256(a.tobytes()).hexdigest()
            != cb["source_value_sha256"][field]
        ):
            raise ValueError("mask identity mismatch")
        masks.append(a)
    if not (masks[2] == 1).all():
        raise ValueError("network presence required")
    if (
        hashlib.sha256(raw_cat.tobytes()).hexdigest()
        != cb["source_value_sha256"]["categorical_target"]
    ):
        raise ValueError("categorical target identity mismatch")
    # Ignore caller-owned derived arrays: derive exact raw casts afresh.
    return {
        "x": x,
        "search_targets": raw_search.astype(np.float32),
        "categorical_targets": raw_cat.astype(np.float32),
        "network_mask": masks[1].astype(np.float32),
        "categorical_mask": masks[0].astype(np.float32),
        "rows_sha256": accepted["rows_sha256"],
        "categorical_sha256": sha,
        "search_raw_sha256": binding["target_normalization"]["raw_float16_sha256"],
    }
