"""Synthetic categorical source/mask admission; no actual data or model execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from chess_anti_engine.replay.directory_seal import write_directory_producer_seal
from chess_anti_engine.train.losses import soft_cross_entropy
from scripts import stored_wdl_categorical as categorical
from scripts.stored_wdl_direct import read_direct_shard
from tests.test_stored_wdl_direct import direct_fixture


def fixture(tmp_path: Path) -> tuple[Path, dict]:
    path, kw = direct_fixture(tmp_path)
    group = zarr.open_group(str(path), mode="a")
    raw = np.zeros((3, 32), dtype=np.float16)
    raw[0, :3] = [.1, .2, .7]
    raw[2, 7] = 1
    for name, array in (
        ("categorical_target", raw), ("has_categorical", np.array([1, 0, 1], np.uint8)),
        ("is_network_turn", np.array([1, 1, 0], np.uint8)),
        ("has_is_network_turn", np.ones(3, np.uint8)),
    ):
        group.create_dataset(name, data=array, chunks=array.shape)
    write_directory_producer_seal(group)
    kw["expected_attrs_sha256"] = hashlib.sha256((path / ".zattrs").read_bytes()).hexdigest()
    kw["expected_seal_sha256"] = hashlib.sha256((path / "directory_producer_seal.json").read_bytes()).hexdigest()
    accepted = read_direct_shard(path, **kw)
    accepted.pop("x")
    return path, accepted


def test_exact_targets_masks_and_native_normalization(tmp_path: Path) -> None:
    path, accepted = fixture(tmp_path)
    result = categorical.read_categorical_targets(path, accepted)
    assert result["raw_targets"].dtype == np.float16
    np.testing.assert_array_equal(result["targets"], result["raw_targets"].astype(np.float32))
    np.testing.assert_array_equal(result["network_mask"], [1, 1, 0])
    np.testing.assert_array_equal(result["categorical_mask"], [1, 0, 1])
    np.testing.assert_array_equal(result["effective_mask"], [1, 0, 0])
    assert result["mask_counts"]["rows"] == 3
    assert result["normalization"]["performed"] is False
    assert result["normalization"]["max_present_abs_sum_error"] > 0
    assert len(result["categorical_sha256"]) == 64
    # Arithmetic control uses the actual production loss, not a new normalization.
    targets = torch.from_numpy(result["targets"])
    logits = torch.linspace(-1, 1, 96).reshape(3, 32)
    normalized = targets / targets.sum(1, keepdim=True).clamp_min(1e-8)
    torch.testing.assert_close(soft_cross_entropy(logits, targets),
                               -(normalized * logits.log_softmax(-1)).sum(-1))
    assert soft_cross_entropy(logits, targets)[1].item() == 0


@pytest.mark.parametrize("fault", ["dtype", "shape", "nan", "negative", "over_one", "sum",
                                    "mask", "presence"])
def test_malformed_values_fail_closed(fault: str) -> None:
    raw = np.full((3, 32), 1/32, np.float16)
    cat, net, present = (np.ones(3, np.uint8) for _ in range(3))
    if fault == "dtype":
        raw = raw.astype(np.float32)
    elif fault == "shape":
        raw = raw[:, :31]
    elif fault == "nan":
        raw[0, 0] = np.nan
    elif fault == "negative":
        raw[0, 0] = -.01
    elif fault == "over_one":
        raw[0, 0] = 1.1
    elif fault == "sum":
        raw *= .5
    elif fault == "mask":
        cat[0] = 2
    else:
        present[0] = 0
    with pytest.raises(ValueError, match=r"bounded|binary|invalid|presence"):
        categorical.admit_values(raw, cat, net, present)


def test_masks_and_raw_values_are_private_snapshots() -> None:
    raw = np.full((3, 32), 1/32, np.float16)
    cat, net, present = (np.ones(3, np.uint8) for _ in range(3))
    result = categorical.admit_values(raw, cat, net, present)
    raw[:] = 0
    cat[:] = 0
    net[:] = 0
    assert result["targets"].sum() == 3
    assert result["categorical_mask"].sum() == 3
    assert result["network_mask"].sum() == 3


def test_wrong_row_order_or_source_is_rejected(tmp_path: Path) -> None:
    path, accepted = fixture(tmp_path)
    accepted["row_ids"] = list(reversed(accepted["row_ids"]))
    with pytest.raises(ValueError, match="original ordered"):
        categorical.read_categorical_targets(path, accepted)
    accepted["source_shard"] = "different"
    with pytest.raises(ValueError, match="source shard"):
        categorical.read_categorical_targets(path, accepted)


def test_target_chunk_limit_precedes_target_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, accepted = fixture(tmp_path)
    # Keep original-input recheck independent of this deliberately malformed target declaration.
    original = categorical._recheck
    pinned = original(path, accepted)
    monkeypatch.setattr(categorical, "_recheck", lambda *_a: pinned)
    p = path / "categorical_target" / ".zarray"
    m = json.loads(p.read_text())
    m["chunks"] = [1000000, 32]
    p.write_text(json.dumps(m))
    monkeypatch.setattr(categorical, "load_shard_arrays", lambda *_a, **_k: pytest.fail("target loader reached"))
    with pytest.raises(ValueError, match="bounded exact"):
        categorical.read_categorical_targets(path, accepted)


def test_original_inputs_rechecked_after_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, accepted = fixture(tmp_path)
    original = categorical.load_shard_arrays
    def mutate(*args, **kwargs):
        arrays, meta = original(*args, **kwargs)
        group = zarr.open_group(str(path), mode="a")
        x = np.asarray(group["x"][:])
        x[0, 0, 0, 0] += .5
        group["x"][:] = x
        return arrays, meta
    monkeypatch.setattr(categorical, "load_shard_arrays", mutate)
    with pytest.raises(ValueError, match="original ordered"):
        categorical.read_categorical_targets(path, accepted)


def test_joint_digest_changes_with_masks(tmp_path: Path) -> None:
    path, accepted = fixture(tmp_path)
    first = categorical.read_categorical_targets(path, accepted)
    group = zarr.open_group(str(path), mode="a")
    group["is_network_turn"][0] = 0
    second = categorical.read_categorical_targets(path, accepted)
    assert first["categorical_sha256"] != second["categorical_sha256"]
    assert first["source_value_sha256"]["categorical_target"] == second["source_value_sha256"]["categorical_target"]
    assert second["mask_counts"]["effective_categorical"] == 0

def test_masked_zero_or_near_simplex_preserved_but_half_mass_rejected() -> None:
    raw = np.zeros((3, 32), np.float16)
    raw[1, :3] = [.1, .2, .7]
    raw[2, 0] = 1
    cat = np.array([0, 0, 1], np.uint8)
    result = categorical.admit_values(raw, cat, np.ones(3, np.uint8), np.ones(3, np.uint8))
    np.testing.assert_array_equal(result["targets"], raw.astype(np.float32))
    raw[0, 0] = .5
    with pytest.raises(ValueError, match="invalid present"):
        categorical.admit_values(raw, cat, np.ones(3, np.uint8), np.ones(3, np.uint8))


def test_scalar_target_rejected() -> None:
    with pytest.raises(ValueError, match="bounded native"):
        categorical.admit_values(np.array(1, np.float16), np.ones(1, np.uint8),
                                 np.ones(1, np.uint8), np.ones(1, np.uint8))
