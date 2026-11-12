import numpy as np
import pytest
from scripts.stored_wdl_child_agreement import child_score, bound_child_score


def fixture():
    a = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]])
    raw = np.array([[0.1, 0.2, 0.7], [0, 1, 0], [0.7, 0.2, 0.1], [0, 1, 0]], np.float16)
    return a, raw, ["a", "a", "b", "b"], [[0, 1], [2, 3]]


def test_flip_child_once_not_parent():
    a, t, g, p = fixture()
    r = child_score(a, a, t, g, p)
    assert r["brier"]["donor_row_mean"] < 1e-7
    assert r["brier"]["candidate_minus_donor_row_mean"] == 0
    assert r["prior_exclusions"] == 2


def test_unflipped_prediction_is_wrong():
    a, t, g, p = fixture()
    b = a.copy()
    b[[1, 3]] = b[[1, 3]][:, [2, 1, 0]]
    assert child_score(a, b, t, g, p)["brier"]["candidate_minus_donor_row_mean"] > 0.7


@pytest.mark.parametrize(
    "pairs", [[[0, 3], [2, 1]], [[0, 1], [0, 1]], [[0, 0], [2, 3]]]
)
def test_bad_pairs(pairs):
    a, t, g, _ = fixture()
    with pytest.raises(ValueError, match="pair map"):
        child_score(a, a, t, g, pairs)


def test_missing_pins_rejected():
    with pytest.raises(ValueError, match="complete"):
        bound_child_score({}, {})


def bound_fixture():
    import io
    import json
    import hashlib

    groups = ["a"] * 85 + ["b"] * 85 + ["c"] * 84
    pairs = [[i, i + 1] for i in range(254 - 1) if groups[i] == groups[i + 1]][:227]
    common = {
        "row_ids": [str(i) for i in range(254)],
        "group_ids": groups,
        "rows_sha256": "rows",
    }
    accepted = {**common, "source_revision": "rev"}
    plan = {
        "rows_sha256": "rows",
        "source_revision": "rev",
        "models": {"donor": {"sha256": "d"}, "cap40": {"sha256": "c"}},
        "architecture": {},
        "source_hashes": {"scripts/stored_wdl_checkpoint.py": "loader"},
    }

    def prediction(sha):
        return {
            "schema": "isolated-stored-wdl-forward/v1",
            "sha256": sha,
            "rows_sha256": "rows",
            "rows": 254,
            "source_revision": "rev",
            "architecture": {},
            "loader_sha256": "loader",
            "batch_size": 16,
            "wdl_order_declared": ["W", "D", "L"],
            "pov_declared": "stored row side to move",
            "inference_performed": True,
            "probabilities": [[0.25, 0.5, 0.25]] * 254,
        }

    raw = {
        "sf_wdl": np.tile(np.array([0.25, 0.5, 0.25], np.float16), (254, 1)),
        "has_sf_wdl": np.ones(254, np.uint8),
        "sf_label_meta": np.zeros((254, 6), np.int32),
        "has_sf_label_meta": np.ones(254, np.uint8),
    }
    buf = io.BytesIO()
    np.savez(buf, **raw)
    blobs = {"sf_raw": buf.getvalue()}

    def put(k, v):
        blobs[k] = json.dumps(v, sort_keys=True).encode()
        return hashlib.sha256(blobs[k]).hexdigest()

    ah = put("accepted", accepted)
    put("plan", plan)
    put("donor", prediction("d"))
    put("candidate", prediction("c"))
    kh = put("keys", {**common, "summary": {"unique_successor_pairs": pairs}})
    put(
        "transition",
        {
            "artifact_pins": {"accepted": {"sha256": ah}, "pairs": {"sha256": kh}},
            "classifications": [
                {"pair": p, "status": "qualified_narrow"} for p in pairs
            ],
            "prior_missing_successors": 27,
        },
    )
    put(
        "sf",
        {
            **common,
            "artifact_sha256": hashlib.sha256(blobs["sf_raw"]).hexdigest(),
            "input_receipt_sha256": ah,
            "raw_value_sha256": {
                k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in raw.items()
            },
            "summary": {"recorded_nodes": {}, "recorded_depth": {}},
        },
    )
    put(
        "transition_review",
        {
            "result_sha256": hashlib.sha256(blobs["transition"]).hexdigest(),
            "fixed_pairs": 227,
            "qualified_narrow": 227,
            "exact_pair_order_moves_and_history_overlap_match": True,
        },
    )
    return blobs, {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()}


def test_bound_synthetic():
    b, h = bound_fixture()
    r = bound_child_score(b, h)
    assert len(r["pairs"]) == 227
    assert r["brier"]["donor_row_mean"] == 0


@pytest.mark.parametrize(
    ("key", "field", "value"),
    [
        ("donor", "sha256", "wrong"),
        ("candidate", "pov_declared", "root"),
        ("sf", "rows_sha256", "other"),
        ("transition", "prior_missing_successors", 26),
    ],
)
def test_bound_semantic_mismatch_even_with_new_pin(key, field, value):
    import json
    import hashlib

    b, h = bound_fixture()
    j = json.loads(b[key])
    j[field] = value
    b[key] = json.dumps(j).encode()
    h[key] = hashlib.sha256(b[key]).hexdigest()
    with pytest.raises(ValueError, match=r"mismatch|filtering"):
        bound_child_score(b, h)
