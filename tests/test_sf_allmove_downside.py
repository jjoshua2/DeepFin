"""One fixed all-move downside intervention and its bounded real-writer prefix."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from scripts import sf_policy_rewrite as tool
from tests.test_sf_tactical_policy import core, fixture


def test_near_good_moves_do_not_disable_bad_move_weighting():
    obs, stored = core([50, 45, -400], [0.25, 0.25, 0.5])
    actual, _ = tool.tactical_target(obs, stored, downside=True)
    np.testing.assert_allclose(actual[:3], [1 / 3, 1 / 3, 1 / 3], atol=0.0003)
    assert actual[0] / actual[1] == stored[0] / stored[1]


def test_strict_raw_boundary_and_nonuniform_inside_set_odds():
    obs, stored = core([0, -300, -300.00000001, -800], [0.5, 0.25, 0.125, 0.125])
    actual, _ = tool.tactical_target(obs, stored, downside=True)
    expected = np.array([0.5, 0.25, 0.0625, 0.0625])
    expected /= expected.sum()
    np.testing.assert_array_equal(
        actual[:4], expected.astype(np.float32).astype(np.float16)
    )


@pytest.mark.parametrize(
    ("scores", "masses"),
    [
        ([0, -500], [1.0, 0.0]),
        ([0, -500, -600], [0.0, 0.25, 0.75]),
        ([99900, 0], [0.25, 0.75]),
        ([-99900, -50000], [0.25, 0.75]),
        ([0], [1.0]),
    ],
)
def test_exact_noop_including_zero_flagged_mass_and_any_mate(scores, masses):
    obs, stored = core(scores, masses)
    actual, _ = tool.tactical_target(obs, stored, downside=True)
    assert actual.tobytes() == stored.tobytes()


def test_prefix_writer_matches_full_rows_and_never_joins_tail(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    args.tactical_recipe = "allmove-downside300"
    full = tool.rewrite(args)
    pilot = copy.copy(args)
    pilot.out = str(tmp_path / "pilot")
    pilot.pilot_shards = 1
    pilot.pilot_max_raw_rows = 5
    original = tool.derive.iter_corpus_rows
    visited = []

    def bounded(path):
        for raw in original(path):
            visited.append(raw["game_id"])
            assert len(visited) <= 5, "pilot decoded tail beyond selected emitted shard"
            yield raw

    monkeypatch.setattr(tool.derive, "iter_corpus_rows", bounded)
    result = tool.rewrite(pilot)
    assert (
        result["status"],
        result["rows"],
        result["raw_limit"],
        result["rows_dropped_no_result"],
    ) == ("PILOT_COMPLETE_NOT_TRAINING", 4, 5, 1)
    assert len(visited) == 5
    assert len(result["outputs"]) == 1
    assert not (Path(pilot.out) / tool.derive.SUMMARY_NAME).exists()
    assert result["algorithm"] == tool.DOWNSIDE_ALGORITHM != tool.TACTICAL_ALGORITHM
    shard = result["outputs"][0]["path"]
    full_group = zarr.open_group(str(Path(args.out) / shard), mode="r")
    prefix_group = zarr.open_group(str(Path(pilot.out) / shard), mode="r")
    for col in tool.ARRAYS:
        np.testing.assert_array_equal(full_group[col][:], prefix_group[col][:])
    parent = Path(args.tactical_bt4_source) / shard
    for col in tool.ARRAYS - {"policy_target"}:
        for f in (parent / col).iterdir():
            assert (
                f.read_bytes() == (Path(pilot.out) / shard / col / f.name).read_bytes()
            )
    assert full["rows"] == 6


def test_prefix_physical_cap_counts_dropped_rows(tmp_path):
    args, _ = fixture(tmp_path)
    args.tactical_recipe = "allmove-downside300"
    args.pilot_shards = 1
    args.pilot_max_raw_rows = 4
    with pytest.raises(ValueError, match="pilot raw-row cap exhausted"):
        tool.rewrite(args)
    failure = json.loads(Path(args.out + ".writing/failed.json").read_text())
    assert failure["raw_rows_read"] == 4
    assert failure["rows_written"] == 0


def test_downside_retains_real_alignment_refusal(tmp_path):
    args, _ = fixture(tmp_path)
    args.tactical_recipe = "allmove-downside300"
    shard = next(Path(args.source).glob("*.zarr"))
    g = zarr.open_group(str(shard), mode="a")
    g["game_id"][0] = 9999
    with pytest.raises(ValueError, match="raw/source shuffled identity differs"):
        tool.rewrite(args)
