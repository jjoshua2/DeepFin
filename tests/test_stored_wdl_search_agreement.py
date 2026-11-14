"""Synthetic native-target extraction and paired agreement, no actual data."""

from pathlib import Path
import hashlib
import numpy as np
import pytest
import zarr
from chess_anti_engine.replay.directory_seal import write_directory_producer_seal
from tests.test_stored_wdl_direct import direct_fixture
from scripts.stored_wdl_direct import read_direct_shard
from scripts import stored_wdl_search_agreement as m


def fixture(tmp_path: Path):
    path, kw = direct_fixture(tmp_path)
    g = zarr.open_group(str(path), mode="a")
    raw = np.array([[0.1, 0.2, 0.7], [0, 1, 0], [0.4, 0.4, 0.2]], np.float16)
    g.create_dataset("search_wdl", data=raw, chunks=raw.shape)
    g.create_dataset("has_search_wdl", data=np.ones(3, np.uint8), chunks=(3,))
    write_directory_producer_seal(g)
    kw["expected_seal_sha256"] = hashlib.sha256(
        (path / "directory_producer_seal.json").read_bytes()
    ).hexdigest()
    return path, read_direct_shard(path, **kw)


def test_extract_exact_native_targets(tmp_path):
    path, accepted = fixture(tmp_path)
    accepted.pop("x")
    out = m.read_search_targets(path, accepted)
    assert out["raw_targets"].dtype == np.float16
    assert out["targets"].dtype == np.float64
    assert np.allclose(out["targets"].sum(1), 1)
    assert out["normalization"]["max_abs_normalization_correction"] > 0
    assert out["targets"][1].tolist() == [
        0,
        1,
        0,
    ]  # genuine draw vs producer fallback unknowable


@pytest.mark.parametrize(
    "raw",
    [
        [[1.001, 0, 0]],
        [[-0.001, 1, 0]],
        [[0.1, 0.1, 0.1]],
        [[np.nan, 0, 1]],
        [[np.inf, 0, 0]],
    ],
)
def test_invalid_targets_rejected(raw):
    with pytest.raises(ValueError, match="invalid or excessive"):
        m.normalize_targets(np.array(raw, np.float16))


def test_presence_all_rows_required(tmp_path, monkeypatch):
    path, accepted = fixture(tmp_path)
    accepted.pop("x")
    original = m.load_shard_arrays

    def changed(*a, **kw):
        arrays, meta = original(*a, **kw)

        class Mask:
            shape = (3,)
            chunks = (3,)
            dtype = np.dtype("uint8")

            def __getitem__(self, key):
                return np.array([1, 0, 1], np.uint8)

        arrays["has_search_wdl"] = Mask()
        return arrays, meta

    monkeypatch.setattr(m, "load_shard_arrays", changed)
    with pytest.raises(ValueError, match="ALL original"):
        m.read_search_targets(path, accepted)


def test_original_row_binding(tmp_path):
    path, accepted = fixture(tmp_path)
    accepted.pop("x")
    accepted["row_ids"] = list(reversed(accepted["row_ids"]))
    with pytest.raises(ValueError, match="original ordered"):
        m.read_search_targets(path, accepted)


def test_paired_agreement_direction_and_group_weights():
    t = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], float)
    a = np.ones((3, 3)) / 3
    b = t.copy()
    out = m.score_agreement(a, b, t, ["a", "a", "b"])
    assert out["brier"]["candidate_row_mean"] == 0
    assert out["logloss"]["candidate_row_mean"] == 0
    assert all(
        v["candidate_minus_donor"] < 0 for v in out["brier"]["per_supplied_group"]
    )
    same = m.score_agreement(a, a, t, ["a", "a", "b"])
    assert same["brier"]["candidate_minus_donor_row_mean"] == 0
    reverse = m.score_agreement(b, a, t, ["a", "a", "b"])
    assert reverse["brier"]["candidate_minus_donor_row_mean"] > 0


def test_log_floor_and_wl_swap():
    a = np.array([[0, 0, 1], [0, 0, 1]], float)
    t = np.array([[1, 0, 0], [1, 0, 0]], float)
    out = m.score_agreement(a, t, t, ["a", "b"])
    assert np.isclose(out["logloss"]["donor_row_mean"], -np.log(1e-12))
    assert out["brier"]["donor_row_mean"] == 2


def bound_fixture() -> dict:
    import json

    rows = [f"row/{i}" for i in range(254)]
    groups = [str(i % 3) for i in range(254)]
    raw = np.tile(np.array([[0.2, 0.3, 0.5]], np.float16), (254, 1))
    targets, correction = m.normalize_targets(raw)
    accepted = {
        "row_ids": rows,
        "group_ids": groups,
        "rows_sha256": "a" * 64,
        "source_revision": "rev",
    }
    plan = {
        "rows_sha256": "a" * 64,
        "source_revision": "rev",
        "architecture": {"kind": "transformer"},
        "models": {"donor": {"sha256": "d" * 64}, "cap40": {"sha256": "c" * 64}},
        "source_hashes": {"scripts/stored_wdl_checkpoint.py": "l" * 64},
    }
    forward = {
        "schema": "isolated-stored-wdl-forward/v1",
        "sha256": "d" * 64,
        "rows_sha256": "a" * 64,
        "rows": 254,
        "source_revision": "rev",
        "architecture": plan["architecture"],
        "loader_sha256": "l" * 64,
        "batch_size": 16,
        "wdl_order_declared": ["W", "D", "L"],
        "pov_declared": "stored row side to move",
        "inference_performed": True,
        "probabilities": targets.tolist(),
    }
    blobs = {
        "accepted": json.dumps(accepted).encode(),
        "plan": json.dumps(plan).encode(),
        "donor": json.dumps(forward).encode(),
        "candidate": json.dumps({**forward, "sha256": "c" * 64}).encode(),
    }
    extracted = {
        "raw_targets": raw,
        "targets": targets,
        "normalization": correction,
        "target_description": m.DESCRIPTION,
        "targets_sha256": m.targets_digest(
            targets,
            rows,
            groups,
            rows_sha256="a" * 64,
            target_description=m.DESCRIPTION,
        ),
    }
    return {
        **{k + "_bytes": v for k, v in blobs.items()},
        "expected_hashes": {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()},
        "extracted": extracted,
    }


def test_bound_saved_receipts():
    out = m.bound_agreement(**bound_fixture())
    assert out["brier"]["candidate_minus_donor_row_mean"] == 0
    assert out["bindings"]["rows_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "fault", ["rows", "model", "swap", "source", "probabilities", "groups", "targets"]
)
def test_bound_receipt_mismatches(fault):
    import json

    kw = bound_fixture()
    if fault == "targets":
        kw["extracted"]["targets"] = kw["extracted"]["targets"][:, ::-1].copy()
    elif fault == "swap":
        kw["donor_bytes"], kw["candidate_bytes"] = (
            kw["candidate_bytes"],
            kw["donor_bytes"],
        )
    elif fault == "groups":
        v = json.loads(kw["accepted_bytes"])
        v["group_ids"] = list(reversed(v["group_ids"]))
        kw["accepted_bytes"] = json.dumps(v).encode()
        kw["expected_hashes"]["accepted"] = hashlib.sha256(
            kw["accepted_bytes"]
        ).hexdigest()
    else:
        v = json.loads(kw["donor_bytes"])
        key = {
            "rows": "rows_sha256",
            "model": "sha256",
            "source": "source_revision",
            "probabilities": "probabilities",
        }[fault]
        v[key] = [[1, 0, 0]] * 254 if fault == "probabilities" else "wrong"
        kw["donor_bytes"] = json.dumps(v).encode()
        if fault != "probabilities":
            kw["expected_hashes"]["donor"] = hashlib.sha256(
                kw["donor_bytes"]
            ).hexdigest()
    with pytest.raises(ValueError, match=r"binding|identity|mismatch"):
        m.bound_agreement(**kw)
