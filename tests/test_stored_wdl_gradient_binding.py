import hashlib
import json
import numpy as np
import pytest
from scripts.stored_wdl_compare import rows_digest
from scripts.stored_wdl_gradient_binding import bind_arrays, pinned_json


def fixture():
    x = np.ones((2, 175, 8, 8), np.float32)
    s = np.array([[0.2, 0.3, 0.5]] * 2, np.float16)
    raw = np.ones((2, 32), np.float16) / 32
    m = np.ones(2, np.uint8)
    a: dict = {
        "row_ids": ["a", "b"],
        "group_ids": ["g", "h"],
        "source_shard": "source",
        "attrs_sha256": "a",
        "producer_seal_sha256": "s",
    }
    a["rows_sha256"] = rows_digest(x, a["row_ids"], a["group_ids"])
    sr = {
        "bindings": {
            "rows_sha256": a["rows_sha256"],
            "target_normalization": {
                "raw_float16_sha256": hashlib.sha256(s.tobytes()).hexdigest()
            },
        }
    }
    cb = {
        **a,
        "source_value_sha256": {
            "categorical_target": hashlib.sha256(raw.tobytes()).hexdigest(),
            **{
                k: hashlib.sha256(m.tobytes()).hexdigest()
                for k in ("has_categorical", "is_network_turn", "has_is_network_turn")
            },
        },
    }
    c: dict = {
        "raw_targets": raw,
        "raw_categorical_mask": m.copy(),
        "raw_network_mask": m.copy(),
        "raw_network_presence": m.copy(),
        "categorical_binding": cb,
        "categorical_sha256": hashlib.sha256(
            json.dumps(cb, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return a, sr, x, s, c


def test_bound_snapshots_ignore_mutable_derived_values():
    args = fixture()
    args[-1]["targets"] = np.zeros((2, 32))
    out = bind_arrays(*args)
    assert np.all(out["categorical_targets"] == 1 / 32)
    args[2][:] = 0
    assert np.all(out["x"] == 1)


@pytest.mark.parametrize(
    "fault", ["input", "search", "target", "mask", "rows", "receipt"]
)
def test_mismatch_rejection(fault):
    a, sr, x, s, c = fixture()
    if fault == "input":
        x[0, 0, 0, 0] = 2
    elif fault == "search":
        s[0] = [1, 0, 0]
    elif fault == "target":
        c["raw_targets"][0] = 0
    elif fault == "mask":
        c["raw_network_mask"][0] = 0
    elif fault == "rows":
        a["row_ids"].reverse()
    else:
        c["categorical_sha256"] = "wrong"
    with pytest.raises(ValueError, match=r"identity|binding"):
        bind_arrays(a, sr, x, s, c)


def test_receipt_pin_rejects_alteration():
    blob = b'{"a":1}'
    sha = hashlib.sha256(blob).hexdigest()
    assert pinned_json(blob, sha) == {"a": 1}
    with pytest.raises(ValueError, match="identity"):
        pinned_json(b'{"a":2}', sha)
