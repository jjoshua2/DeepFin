"""Saved-only partial child SF agreement; arithmetic is not a strength test."""

import hashlib
import io
import json
import numpy as np
from scripts.stored_wdl_search_agreement import normalize_targets, score_agreement
from scripts.stored_wdl_compare import _simplex


def child_score(donor, candidate, raw, groups, pairs):
    """Arithmetic kernel; use bound_child_score for actual artifact scoring."""
    n = len(groups)
    a, b = (_simplex(v, n, "probabilities").copy() for v in (donor, candidate))
    frozen = tuple(tuple(p) for p in pairs)
    if (
        not frozen
        or len(set(frozen)) != len(frozen)
        or len({p[0] for p in frozen}) != len(frozen)
        or any(
            len(p) != 2
            or any(type(i) is not int or not 0 <= i < n for i in p)
            or p[0] == p[1]
            or groups[p[0]] != groups[p[1]]
            for p in frozen
        )
    ):
        raise ValueError("unique same-group ordered pair map required")
    parent, child = zip(*frozen, strict=True)
    targets, correction = normalize_targets(np.asarray(raw)[list(parent)])
    # Stored SF label is already parent POV. Flip CHILD predictions once only.
    result = score_agreement(
        a[list(child)][:, [2, 1, 0]],
        b[list(child)][:, [2, 1, 0]],
        targets,
        tuple(groups[i] for i in parent),
    )
    result.update(
        target_description="Stored action-conditioned child SF labels in parent POV",
        interpretation="Partial cohort action-conditioned child SF agreement only; not root truth, heldout, independent high-budget truth or strength",
        perspective="Successor-row W/L flipped exactly once into parent POV; targets unchanged except declared f16 renormalization",
        normalization=correction,
        pairs=[list(p) for p in frozen],
        prior_exclusions=n - len(frozen),
    )
    return result


def bound_child_score(blobs: dict[str, bytes], expected: dict[str, str]) -> dict:
    keys = {
        "accepted",
        "plan",
        "donor",
        "candidate",
        "sf",
        "sf_raw",
        "keys",
        "transition",
        "transition_review",
    }
    if set(blobs) != keys or set(expected) != keys:
        raise ValueError("complete external receipt pins required")
    for k, blob in blobs.items():
        if len(blob) > 4 * 1024**2 or hashlib.sha256(blob).hexdigest() != expected[k]:
            raise ValueError("artifact identity mismatch")
    j = {k: json.loads(v) for k, v in blobs.items() if k != "sf_raw"}
    accepted, plan, sf, key, transition = (
        j[k] for k in ("accepted", "plan", "sf", "keys", "transition")
    )
    rows, groups, digest = (
        accepted[k] for k in ("row_ids", "group_ids", "rows_sha256")
    )
    if (
        len(rows) != 254
        or len(set(rows)) != 254
        or len(groups) != 254
        or len(set(groups)) != 3
    ):
        raise ValueError("fixed panel mismatch")
    for rec in (sf, key):
        if any(rec[k] != accepted[k] for k in ("row_ids", "group_ids", "rows_sha256")):
            raise ValueError("ordered row/group binding mismatch")
    if (
        plan["rows_sha256"] != digest
        or plan["source_revision"] != accepted["source_revision"]
    ):
        raise ValueError("source/input plan mismatch")
    if (
        sf["artifact_sha256"] != expected["sf_raw"]
        or sf["input_receipt_sha256"] != expected["accepted"]
    ):
        raise ValueError("SF receipt binding mismatch")
    tp = transition["artifact_pins"]
    if (
        tp["accepted"]["sha256"] != expected["accepted"]
        or tp["pairs"]["sha256"] != expected["keys"]
    ):
        raise ValueError("transition input binding mismatch")
    pairs = key["summary"]["unique_successor_pairs"]
    classifications = transition["classifications"]
    if (
        len(pairs) != 227
        or [c["pair"] for c in classifications] != pairs
        or any(c["status"] != "qualified_narrow" for c in classifications)
        or transition["prior_missing_successors"] != 27
    ):
        raise ValueError("all fixed227 transitions must qualify without filtering")
    review = j["transition_review"]
    if (
        review.get("result_sha256") != expected["transition"]
        or review.get("fixed_pairs") != 227
        or review.get("qualified_narrow") != 227
        or review.get("exact_pair_order_moves_and_history_overlap_match") is not True
    ):
        raise ValueError("independent transition review mismatch")
    probabilities = []
    for name, model in (("donor", "donor"), ("candidate", "cap40")):
        rec = j[name]
        if (
            rec.get("schema") != "isolated-stored-wdl-forward/v1"
            or rec.get("sha256") != plan["models"][model]["sha256"]
            or rec.get("rows_sha256") != digest
            or rec.get("rows") != 254
            or rec.get("source_revision") != plan["source_revision"]
            or rec.get("architecture") != plan["architecture"]
            or rec.get("loader_sha256")
            != plan["source_hashes"]["scripts/stored_wdl_checkpoint.py"]
            or rec.get("batch_size") != 16
            or rec.get("wdl_order_declared") != ["W", "D", "L"]
            or rec.get("pov_declared") != "stored row side to move"
            or rec.get("inference_performed") is not True
        ):
            raise ValueError("forward identity/semantics mismatch")
        probabilities.append(np.asarray(rec["probabilities"]))
    with np.load(io.BytesIO(blobs["sf_raw"]), allow_pickle=False) as z:
        if set(z.files) != {
            "sf_wdl",
            "has_sf_wdl",
            "sf_label_meta",
            "has_sf_label_meta",
        }:
            raise ValueError("exact SF archive fields required")
        raw = z["sf_wdl"]
        if raw.shape != (254, 3) or raw.dtype != np.float16:
            raise ValueError("native SF target shape required")
        for name in z.files:
            arr = z[name]
            if (
                hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
                != sf["raw_value_sha256"][name]
            ):
                raise ValueError("raw SF value binding mismatch")
        if not (z["has_sf_wdl"] == 1).all():
            raise ValueError("no missing-label filtering allowed")
    result = child_score(probabilities[0], probabilities[1], raw, groups, pairs)
    result["bindings"] = dict(expected)
    result["rows_sha256"] = digest
    result["sf_recorded_budget"] = {
        k: sf["summary"][k] for k in ("recorded_nodes", "recorded_depth")
    }
    result["selection_limit"] = (
        "227 successor-available rows from original254, not random/heldout;27 missing retained as exclusions"
    )
    return result
