"""Observation selection and source identity survive the real writer paths."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zarr

from scripts import corpus_row_provenance as refs
from scripts import derive_corpus_targets as derive
from scripts import gen_sf_rooted_corpus as corpus
from tests.test_derive_corpus_targets import (
    CONFIG_SHA, depth_options, full_width_phase, history_row, narrowed_phase,
    read_rows, run_derive, write_corpus,
)
from tests.test_derive_parallel import game, shard_content, write_split_corpus


def test_phase0_policy_changes_target_without_changing_default_value(tmp_path: Path) -> None:
    row = history_row()
    initial = derive.RowBank(row).full_width_block(9)
    assert initial is not None
    moves = list(initial["order"])
    row["phases"] = [
        full_width_phase(row["fen"], {9: {move: 0.0 if i else 100.0 for i, move in enumerate(moves)}}),
        narrowed_phase({9: {moves[1]: 400.0}}),
    ]
    source = write_corpus(tmp_path, [row])
    summaries = {}
    samples = {}
    for name, flags in {
        "legacy": (), "policy": ("--policy-observation", "phase0"),
        "both": ("--policy-observation", "phase0", "--value-observation", "phase0"),
        "value_only": ("--value-observation", "phase0"),
        "explicit_default": ("--policy-observation", "latest-phase", "--value-observation", "latest-phase"),
    }.items():
        summaries[name] = run_derive(source, tmp_path / name, "uniform-d9", *flags)
        samples[name] = read_rows(tmp_path / name)[0][0]
    assert not np.array_equal(samples["legacy"].policy_target, samples["policy"].policy_target)
    np.testing.assert_array_equal(samples["legacy"].search_wdl, samples["policy"].search_wdl)
    np.testing.assert_array_equal(samples["policy"].policy_target, samples["both"].policy_target)
    assert not np.array_equal(samples["policy"].search_wdl, samples["both"].search_wdl)
    np.testing.assert_array_equal(samples["legacy"].policy_target, samples["value_only"].policy_target)
    np.testing.assert_array_equal(samples["both"].search_wdl, samples["value_only"].search_wdl)
    assert shard_content(tmp_path / "legacy") == shard_content(tmp_path / "explicit_default")
    assert summaries["policy"]["scheme"]["value_source"] == derive.VALUE_SOURCE_DEEPEST
    assert summaries["both"]["scheme"]["value_source"] == derive.VALUE_SOURCE_PHASE0
    assert "depth None" not in summaries["policy"]["value_channels"]["search_wdl"]


def test_phase0_requires_complete_support_and_rejects_unsupported_schemes() -> None:
    row = history_row()
    block = derive.RowBank(row).full_width_block(9)
    assert block is not None
    row["phases"] = [full_width_phase(row["fen"], {9: block["values"]}, complete={9: False})]
    phase0 = replace(derive.parse_scheme("uniform-d9"), policy_observation="phase0")
    with pytest.raises(derive.EnvelopeMiss, match="complete block"):
        derive.apply_scheme(derive.RowBank(row), phase0)
    for spec in ("top4-d11-rest-d9", "nodes-100"):
        with pytest.raises(ValueError, match="uniform"):
            replace(derive.parse_scheme(spec), policy_observation="phase0")


def test_row_references_distinguish_source_aliases_and_float16_quantization(tmp_path: Path) -> None:
    row = history_row()
    derived = derive.TargetDeriver(depth_options(derive.parse_scheme("uniform-d9"))).derive_row(row)
    assert derived is not None
    x = derived.sample.x
    first = refs.reference(row, tmp_path / "run06" / "w00.jsonl.zst", 3, CONFIG_SHA, x)
    second = refs.reference(row, tmp_path / "run07" / "w00.jsonl.zst", 3, CONFIG_SHA, x)
    assert first["source_namespace"] != second["source_namespace"]
    assert first["input_key"] == row["input_key"]
    assert first["input_key"] != first["stored_input_key"]
    path = tmp_path / refs.FILENAME
    pin = refs.write(path, [first, second], np.stack([x, x]),
                     np.array([row["game_id"]] * 2), np.array([row["ply"]] * 2))
    assert refs.read(path, rows=2) == [first, second]
    assert pin["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert refs.RECORD_DTYPE.itemsize == 56
    changed = x.copy()
    changed[0, 0, 0] += 0.5
    with pytest.raises(ValueError, match="does not match"):
        refs.write(tmp_path / "bad.npz", [first], np.stack([changed]),
                   np.array([row["game_id"]]), np.array([row["ply"]]))
    assert not (tmp_path / "bad.npz").exists()


@pytest.mark.parametrize("workers", [1, 2])
def test_filtered_shuffled_rows_keep_physical_source_reference(tmp_path: Path, workers: int) -> None:
    rows = [history_row(game_id=i, result=None if i == 2 else 1.0) for i in range(8)]
    # Cross both raw shard and spill/output boundaries. Resultless physical row 2 drops.
    source = write_split_corpus(tmp_path, rows, [4, 4])
    baseline = tmp_path / "baseline"
    run_derive(source, baseline, "uniform-d9", "--rows-per-shard", "3", "--seed", "9")
    output = tmp_path / "with_refs"
    flags = ("--workers", str(workers), "--spill-chunk-rows", "2") if workers > 1 else ()
    summary = run_derive(source, output, "uniform-d9", "--rows-per-shard", "3",
                         "--seed", "9", "--row-provenance", *flags)
    expected_order = []
    rng = np.random.default_rng(9)
    survivors = [i for i in range(8) if i != 2]
    for start in range(0, len(survivors), 3):
        chunk = survivors[start:start + 3]
        expected_order.extend(chunk[int(i)] for i in rng.permutation(len(chunk)))
    seen = []
    for shard in sorted(output.glob("shard_*.zarr")):
        group = zarr.open_group(str(shard), mode="r")
        baseline_group = zarr.open_group(str(baseline / shard.name), mode="r")
        assert set(group.array_keys()) == set(baseline_group.array_keys())
        for name in group.array_keys():
            np.testing.assert_array_equal(group[name][:], baseline_group[name][:])
        stored_x = np.asarray(group["x"][:])
        stored_games = np.asarray(group["game_id"][:])
        rows_here = refs.read(shard / refs.FILENAME, rows=len(stored_x))
        for i, ref in enumerate(rows_here):
            game_id = int(stored_games[i])
            assert ref["game_id"] == game_id
            assert ref["source_shard"] == f"w00-{game_id // 4:05d}.jsonl.zst"
            assert ref["source_row"] == game_id % 4
            assert ref["input_key"] == rows[game_id]["input_key"]
            assert ref["stored_input_key"] == corpus.input_tensor_key(stored_x[i])
            seen.append(game_id)
    assert seen == expected_order
    assert summary["row_provenance"]["path_in_shard"] == refs.FILENAME


def test_grouped_refs_survive_worker_handoff_and_both_drop_paths(tmp_path: Path) -> None:
    rows = [row for gid in range(3) for row in game(gid, 4)]
    rows[1]["phases"][0]["per_depth"] = [
        block for block in rows[1]["phases"][0]["per_depth"] if block["depth"] != 9
    ]
    rows[6]["result"] = None
    source = write_split_corpus(tmp_path, rows, [3, 4, 5])
    flags = ("--rows-per-shard", "3", "--value-scheme", "qzphase",
             "--max-envelope-misses", "1", "--row-provenance")
    outputs = [tmp_path / "sequential", tmp_path / "parallel"]
    for output, extra in zip(outputs, [(), ("--workers", "2", "--spill-chunk-rows", "2")]):
        run_derive(source, output, "uniform-d9", *flags, *extra)
    assert shard_content(outputs[0]) == shard_content(outputs[1])
    physical = []
    for shard in sorted(outputs[0].glob("shard_*.zarr")):
        counterpart = outputs[1] / shard.name / refs.FILENAME
        assert (shard / refs.FILENAME).read_bytes() == counterpart.read_bytes()
        group = zarr.open_group(str(shard), mode="r")
        for ref in refs.read(shard / refs.FILENAME, rows=len(np.asarray(group["x"][:]))):
            offset = {"w00-00000.jsonl.zst": 0, "w00-00001.jsonl.zst": 3,
                      "w00-00002.jsonl.zst": 7}[ref["source_shard"]]
            index = offset + ref["source_row"]
            assert ref["input_key"] == rows[index]["input_key"]
            assert (ref["game_id"], ref["ply"]) == (rows[index]["game_id"], rows[index]["ply"])
            physical.append(index)
    assert sorted(physical) == [i for i in range(12) if i not in (1, 6)]
