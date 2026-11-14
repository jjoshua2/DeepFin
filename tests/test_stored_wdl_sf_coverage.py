import hashlib
import numpy as np
import pytest
import zarr
from chess_anti_engine.replay.directory_seal import write_directory_producer_seal
from tests.test_stored_wdl_direct import direct_fixture
from scripts.stored_wdl_direct import read_direct_shard
from scripts.stored_wdl_sf_coverage import summarize, read_sf_coverage


def values():
    return {
        "sf_wdl": np.array([[0.2, 0.3, 0.5], [0, 0, 0], [0, 1, 0]], np.float16),
        "has_sf_wdl": np.array([1, 0, 1], np.uint8),
        "sf_label_meta": np.array(
            [
                [5000, 12, 0, 0, 200, 300],
                [-1, -1, 0, 0, -1, -1],
                [6000, 14, 0, 0, 0, 1000],
            ],
            np.int32,
        ),
        "has_sf_label_meta": np.array([1, 0, 1], np.uint8),
    }


def test_coverage_no_repair_or_score():
    v = values()
    s = summarize(v)
    assert s["sf_present"] == 2
    assert s["sf_missing"] == 1
    assert s["present_near_simplex"] == 2
    assert s["recorded_nodes"]["known_min"] == 5000
    assert s["normalization_performed"] is False


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1.1])
def test_invalid_labels_counted_not_repaired(bad):
    v = values()
    v["sf_wdl"][0, 0] = bad
    s = summarize(v)
    assert s["present_near_simplex"] == 1
    assert (
        np.isnan(v["sf_wdl"][0, 0])
        if np.isnan(bad)
        else v["sf_wdl"][0, 0] == np.float16(bad)
    )


def test_read_bound_synthetic_shard(tmp_path):
    path, kw = direct_fixture(tmp_path)
    g = zarr.open_group(str(path), mode="a")
    for name, a in values().items():
        g.create_dataset(name, data=a, chunks=a.shape)
    write_directory_producer_seal(g)
    kw["expected_seal_sha256"] = hashlib.sha256(
        (path / "directory_producer_seal.json").read_bytes()
    ).hexdigest()
    accepted = read_direct_shard(path, **kw)
    accepted.pop("x")
    result = read_sf_coverage(path, accepted)
    assert result["summary"]["sf_present"] == 2
    assert result["child_identity_qualified"] is False
    assert all(result["physical_action_fields_absent"].values())
    accepted["row_ids"].reverse()
    with pytest.raises(ValueError, match="original ordered"):
        read_sf_coverage(path, accepted)


def test_bad_mask_rejected():
    v = values()
    v["has_sf_wdl"][0] = 2
    with pytest.raises(ValueError, match="binary"):
        summarize(v)


@pytest.mark.parametrize(
    ("bad", "negative", "above"), [(np.nan, 0, 0), (-np.inf, 1, 0), (np.inf, 0, 1)]
)
def test_nonfinite_range_categories(bad, negative, above):
    v = values()
    v["sf_wdl"][0, 0] = bad
    s = summarize(v)
    assert s["present_nonfinite"] == 1
    assert s["present_negative"] == negative
    assert s["present_above_one"] == above
    assert s["present_near_simplex"] == 1
