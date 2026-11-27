"""Synthetic sealed producer shards and target-free arithmetic only."""
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from chess_anti_engine.replay.directory_seal import write_directory_producer_seal
from chess_anti_engine.replay.shard import POLICY_SIZE
from scripts import stored_wdl_direct as direct


def direct_fixture(tmp_path: Path):
    path = tmp_path / "direct.zarr"
    group = zarr.open_group(str(path), mode="w")
    fields = {"x": np.ones((3, 175, 8, 8), np.float16),
              "policy_target": np.zeros((3, POLICY_SIZE), np.float32),
              "wdl_target": np.ones(3, np.int8), "has_policy": np.ones(3, np.uint8),
              "game_id": np.array([7, 7, 9], np.int64),
              "has_game_id": np.ones(3, np.uint8), "is_selfplay": np.ones(3, np.uint8),
              "has_is_selfplay": np.ones(3, np.uint8)}
    for name, data in fields.items():
        group.create_dataset(name, data=data, chunks=data.shape, write_empty_chunks=True)
    group.attrs.update(positions=3, selfplay_games=2, curriculum_games=0,
                       input_history_encoding="lc0_root_legacy_meta", history_rep_fix=True)
    write_directory_producer_seal(group)
    kw: dict[str, Any] = {
        "expected_attrs_sha256": hashlib.sha256((path / ".zattrs").read_bytes()).hexdigest(),
        "expected_seal_sha256": hashlib.sha256((path / "directory_producer_seal.json").read_bytes()).hexdigest(),
        "expected_rows": 3, "expected_history_rep_fix": True}
    return path, kw


def test_direct_all_rows_no_targets(tmp_path: Path):
    path, kw = direct_fixture(tmp_path)
    result = direct.read_direct_shard(path, **kw)
    assert result["x"].shape == (3, 175, 8, 8)
    assert result["x"].dtype == np.float32
    assert "targets" not in result
    assert result["source_rows_scanned"] == 3
    assert len(set(result["group_ids"])) == 2
    assert result["holdout_claim"] is False


@pytest.mark.parametrize("fault", ["attrs", "seal", "count", "repfix"])
def test_predeclared_identity_failures_before_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault):
    path, kw = direct_fixture(tmp_path)
    if fault in {"attrs", "seal"}:
        kw[f"expected_{fault}_sha256"] = "0" * 64
    elif fault == "count":
        kw["expected_rows"] = 4
    else:
        kw["expected_history_rep_fix"] = False
    def forbidden(*_args, **_kwargs):
        pytest.fail("array loader reached")
    monkeypatch.setattr(direct, "load_shard_arrays", forbidden)
    with pytest.raises(ValueError, match=r"identity|metadata|history"):
        direct.read_direct_shard(path, **kw)


def test_prediction_drift_weighting_and_swap():
    a = np.array([[.5,.2,.3]] * 3)
    b = np.array([[.6,.2,.2],[.6,.2,.2],[.4,.2,.4]])
    kwargs: dict[str, Any] = {"group_ids": ["g1","g1","g2"], "input_rows_sha256": "a" * 64}
    result = direct.prediction_drift(a, b, **kwargs)
    assert result["row_weighted"]["mean_expected_score_change"] == pytest.approx(.1 / 3)
    assert result["equal_supplied_group_weighted"]["mean_expected_score_change"] == pytest.approx(0)
    same = direct.prediction_drift(a, a, **kwargs)
    assert same["max_abs_WDL_change"] == 0
    swapped = direct.prediction_drift(a[:, ::-1], b[:, ::-1], **kwargs)
    assert swapped["row_weighted"]["mean_expected_score_change"] == pytest.approx(-.1 / 3)
    assert result["controls_performed"] == []


@pytest.mark.parametrize("bad", [np.array([[2.,0.,-1.]] * 3), np.zeros((3,3))])
def test_bad_predictions_refused(bad):
    with pytest.raises(ValueError, match=r"finite|sum"):
        direct.prediction_drift(bad, bad, ["g1","g1","g2"], input_rows_sha256="a" * 64)


@pytest.mark.parametrize("fault", ["missing_id", "not_selfplay", "one_group", "nan"])
def test_observed_rows_fail_closed(tmp_path: Path, fault):
    path, kw = direct_fixture(tmp_path)
    group = zarr.open_group(str(path), mode="a")
    if fault == "missing_id":
        group["has_game_id"][0] = 0
    elif fault == "not_selfplay":
        group["is_selfplay"][0] = 0
    elif fault == "one_group":
        group["game_id"][:] = 7
    else:
        group["x"][0,0,0,0] = float("nan")
    write_directory_producer_seal(group)
    kw["expected_seal_sha256"] = hashlib.sha256((path / "directory_producer_seal.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=r"finite|groups"):
        direct.read_direct_shard(path, **kw)
