"""Stored B100 attenuation, categorical mates and real legacy source joins."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from chess_anti_engine.replay.shard import load_shard_arrays
from scripts import bt4_policy_mix as policy
from scripts import sf_policy_rewrite as tool
from tests.test_sf_policy_rewrite import CONFIG_SHA, args, raw_row, run, write_corpus


def core(scores: list[float], masses: list[float] | None = None):
    obs = tool.Observation(
        0,
        0,
        np.arange(len(scores), dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        "0" * 32,
    )
    stored = np.zeros(1858, dtype=np.float16)
    stored[: len(scores)] = (
        masses if masses is not None else np.full(len(scores), 1 / len(scores))
    )
    return obs, stored


def test_gap_floor_and_acceptable_move_ratios() -> None:
    obs, base = core([500, 400, 300, -2000], [0.4, 0.3, 0.2, 0.1])
    result, diagnostic = tool.tactical_target(obs, base)
    legal = result[:4].astype(np.float64)
    # First two moves retain their original odds; 200cp deficit gets exp(-1),
    # the far worse move gets relative floor .1 rather than absolute mass .1.
    assert legal[0] / legal[1] == pytest.approx(
        float(base[0]) / float(base[1]), rel=0.002
    )
    assert (legal[2] / legal[0]) / (float(base[2]) / float(base[0])) == pytest.approx(
        np.exp(-1), rel=0.002
    )
    assert (legal[3] / legal[0]) / (float(base[3]) / float(base[0])) == pytest.approx(
        0.1, rel=0.002
    )
    assert legal[3] < 0.1
    assert np.all(result[4:] == 0)
    assert diagnostic["category"] == "no_mate"
    assert diagnostic["support_losses"] == 0


@pytest.mark.parametrize(
    ("scores", "category", "expected_weights"),
    [
        ([99900, 50000, 100, -99900], "winning_mate_available", [1, 1, 0.1, 0.1]),
        ([100, 0, -100, -99900], "losing_mate_alternatives", [1, 1, np.exp(-1), 0.1]),
        ([-50000, -80000, -99900, -100000], "all_forced_losses", [1, 1, 1, 1]),
        ([32000, 31901, 31900, 31980], "no_mate", [1, 1, 1, 1]),
    ],
)
def test_mate_categories_and_identity(scores, category, expected_weights) -> None:
    obs, base = core(scores, [0.4, 0.3, 0.2, 0.1])
    result, diagnostic = tool.tactical_target(obs, base)
    assert diagnostic["category"] == category
    odds = result[:4].astype(np.float64) / base[:4].astype(np.float64)
    np.testing.assert_allclose(odds / odds[0], expected_weights, rtol=0.002)
    if all(weight == 1 for weight in expected_weights):
        assert result.tobytes() == base.tobytes()


@pytest.mark.parametrize(
    "score", [32001, -32001, 49999, 99900.1, 100001, float("nan"), float("inf")]
)
def test_ambiguous_score_domain_refused(score: float) -> None:
    obs, base = core([score, 0])
    with pytest.raises(ValueError, match=r"scores|domain"):
        tool.tactical_target(obs, base)


@pytest.mark.parametrize(
    "defect",
    ["dtype", "illegal", "mass", "negative", "nan", "duplicate", "float_index"],
)
def test_bad_base_or_mapping_refused(defect: str) -> None:
    obs, base = core([0, -200])
    if defect == "dtype":
        base = base.astype(np.float32)
    elif defect == "illegal":
        base[1], base[8] = 0, 0.5
    elif defect == "mass":
        base *= 0.5
    elif defect == "negative":
        base[0] = -0.5
    elif defect == "nan":
        base[0] = np.nan
    elif defect == "duplicate":
        obs.indices[:] = 0
    else:
        obs.indices = obs.indices.astype(np.float64)
    with pytest.raises(ValueError, match=r"B100|compact indices"):
        tool.tactical_target(obs, base)


def test_float16_support_loss_is_reported() -> None:
    obs, base = core([0, -2000], [1, float(np.nextafter(np.float16(0), np.float16(1)))])
    result, diagnostic = tool.tactical_target(obs, base)
    assert result[0] == 1
    assert result[1] == 0
    assert diagnostic["support_losses"] == 1
    assert diagnostic["ideal_to_stored_relative_error_max"] == 1
    assert 0 < diagnostic["ideal_to_stored_TV"] < 1e-7


def fixture(tmp_path: Path):
    rows = [raw_row(game_id=i) for i in range(7)]
    rows[2]["result"] = None
    for i, row in enumerate(rows):
        phase = row["phases"][0]
        phase["anomalies"]["bound_lines"] = (
            i  # Dropped UCI bounds are not retained scores.
        )
        lines = phase["per_depth"][0]["lines"]
        for j, line in enumerate(lines):
            line[2] = 500.123456789 - (j * (30 + i))
        if i in (1, 6):
            lines[-1][2] = -99900
        elif i == 3:
            lines[0][2], lines[1][2] = 99900, 50000
            lines[-1][2] = -99900
        elif i == 4:
            for j, line in enumerate(lines):
                line[2] = -50000 - 100 * j
    raw = write_corpus(
        tmp_path, rows, row_schema=3, staircase=[{"depth": 9, "width": "all"}]
    )
    sf = tmp_path / "sf"
    run(raw, sf, "--limit", "7", temp=0.0005, rows_per_shard=4)
    b100 = tmp_path / "B100"
    shutil.copytree(sf, b100)
    summary = json.loads((sf / tool.derive.SUMMARY_NAME).read_text())
    mix = {
        "kind": "global",
        "algorithm": "legal-normalized-global-arithmetic-v1",
        "alpha": 1.0,
        "bt4_temperature": 0.5,
        "rows": 6,
        "expected_shards": 2,
        "source_dir": str(sf),
        "source_derive_summary_sha256": tool.file_sha256(sf / tool.derive.SUMMARY_NAME),
        "mutated_arrays": ["policy_target"],
    }
    for shard in b100.glob("*.zarr"):
        group: Any = zarr.open_group(str(shard), mode="a")
        legal = group["legal_mask"][:]
        external = np.broadcast_to(np.arange(1, 1859), legal.shape).astype(np.float32)
        group["policy_target"][:] = policy.mix_policy_targets(
            group["policy_target"][:],
            external,
            legal,
            alpha=1.0,
            scope="global",
            bt4_temperature=0.5,
        )
        group.attrs.update(
            policy_target_mix_kind="global",
            policy_target_mix_alpha=1.0,
            policy_target_mix_bt4_temperature=0.5,
        )
    summary["policy_target_postprocess"] = mix
    (b100 / tool.derive.SUMMARY_NAME).write_text(json.dumps(summary))
    (b100 / "bt4_policy_mix_summary.json").write_text(json.dumps(mix))
    invocation = args(
        raw,
        sf,
        tmp_path / "out",
        "--tactical-bt4-source",
        str(b100),
        "--expected-bt4-summary-sha256",
        tool.file_sha256(b100 / tool.derive.SUMMARY_NAME),
        "--expected-bt4-mix-sha256",
        tool.file_sha256(b100 / "bt4_policy_mix_summary.json"),
    )
    return invocation, rows


def test_real_shuffled_producer_loader_and_compressed_nonpolicy_parity(
    tmp_path, monkeypatch
) -> None:
    invocation, rows = fixture(tmp_path)

    def no_reencode(*_args, **_kwargs):
        pytest.fail("inherited history must not be re-encoded")

    monkeypatch.setattr(tool.derive.TargetDeriver, "_encode", no_reencode)
    parent_states = {
        str(p): tool.rank._storage_identity(p)
        for root in (invocation.source, invocation.tactical_bt4_source)
        for p in Path(root).glob("*.zarr")
    }
    result = tool.rewrite(invocation)
    assert result["kind"] == "bt4_sf_tactical_policy_attenuation"
    assert (result["rows"], result["shards"], result["raw_limit"]) == (6, 2, 7)
    assert (result["rows_dropped_no_result"], result["changed_rows"]) == (1, 5)
    assert result["categories"] == {
        "no_mate": 2,
        "losing_mate_alternatives": 2,
        "winning_mate_available": 1,
        "all_forced_losses": 1,
    }
    assert (
        result["source_derive_summary_sha256"] == invocation.expected_bt4_summary_sha256
    )
    assert (
        result["sf_derive_summary_sha256"] == invocation.expected_source_summary_sha256
    )
    assert result["recipe"]["gap_cp"] == result["recipe"]["decay_cp"] == 100
    assert result["recipe"]["relative_floor"] == 0.1
    assert result["winning_mate_zero_base_mass_rows"] == 0
    out = Path(invocation.out)
    assert not (out / tool.SUMMARY).exists()
    assert not (out / "bt4_policy_mix_summary.json").exists()
    assert (
        json.loads((out / tool.TACTICAL_SUMMARY).read_text())["algorithm"]
        == tool.TACTICAL_ALGORITHM
    )
    summary = json.loads((out / tool.derive.SUMMARY_NAME).read_text())
    assert summary["policy_target_postprocess"] == json.loads(
        json.dumps({k: v for k, v in result.items() if k != "outputs"})
    )
    for spec in result["outputs"]:
        source = Path(invocation.tactical_bt4_source) / spec["path"]
        original, _ = load_shard_arrays(source)
        actual, metadata = load_shard_arrays(out / spec["path"])
        assert metadata["policy_target_rewrite"]["algorithm"] == tool.TACTICAL_ALGORITHM
        assert not any(key.startswith("policy_target_mix_") for key in metadata)
        for key in tool.ARRAYS - {"policy_target"}:
            np.testing.assert_array_equal(actual[key], original[key])
            for path in (source / key).iterdir():
                assert (
                    path.read_bytes()
                    == (out / spec["path"] / key / path.name).read_bytes()
                )
                assert (
                    path.stat().st_ino
                    != (out / spec["path"] / key / path.name).stat().st_ino
                )
        for i, game in enumerate(actual["game_id"]):
            obs = tool.observation(rows[int(game)], CONFIG_SHA)
            expected, _ = tool.tactical_target(obs, original["policy_target"][i])
            np.testing.assert_array_equal(actual["policy_target"][i], expected)
        assert spec["source_policy_sha256"] == tool.rank._sha_arrays(
            original["policy_target"]
        )
    for path, before in parent_states.items():
        assert tool.rank._storage_identity(Path(path)) == before
    with pytest.raises(ValueError, match="already exists"):
        tool.rewrite(invocation)


@pytest.mark.parametrize(
    "defect",
    [
        "pin",
        "recipe",
        "lineage",
        "attrs",
        "nonpolicy",
        "chunk",
        "illegal_policy",
        "raw_bounds",
        "raw_nodes",
        "raw_score",
    ],
)
def test_parent_and_raw_refusals(tmp_path, monkeypatch, defect) -> None:
    invocation, _ = fixture(tmp_path)
    parent = Path(invocation.tactical_bt4_source)
    group: Any = zarr.open_group(str(parent / "shard_000000.zarr"), mode="a")
    if defect == "pin":
        invocation.expected_bt4_mix_sha256 = "0" * 64
    elif defect in {"recipe", "lineage"}:
        path = parent / tool.derive.SUMMARY_NAME
        summary = json.loads(path.read_text())
        if defect == "recipe":
            summary["policy_target_postprocess"]["alpha"] = 0.5
            mix = parent / "bt4_policy_mix_summary.json"
            mix.write_text(json.dumps(summary["policy_target_postprocess"]))
            invocation.expected_bt4_mix_sha256 = tool.file_sha256(mix)
        else:
            summary["seed"] += 1
        path.write_text(json.dumps(summary))
        invocation.expected_bt4_summary_sha256 = tool.file_sha256(path)
    elif defect == "attrs":
        group.attrs["policy_target_mix_alpha"] = 0.5
    elif defect == "nonpolicy":
        group["search_wdl"][0] = [0.25, 0.5, 0.25]
    elif defect == "chunk":
        del group["x"].chunk_store[group["x"]._chunk_key((0, 0, 0, 0))]
    elif defect == "illegal_policy":
        row = group["policy_target"][0]
        row[:] = 0
        row[np.flatnonzero(group["legal_mask"][0] == 0)[0]] = 1
        group["policy_target"][0] = row
    else:
        original = tool.derive.iter_corpus_rows

        def corrupt(path):
            for row in original(path):
                phase = row["phases"][0]
                if defect == "raw_bounds":
                    del phase["anomalies"]["bound_lines"]
                elif defect == "raw_nodes":
                    phase["per_depth"][0]["lines"][0][3] = True
                elif row["game_id"] == 0:
                    phase["per_depth"][0]["lines"][-1][2] = -32001
                yield row

        monkeypatch.setattr(tool.derive, "iter_corpus_rows", corrupt)
    with pytest.raises(
        ValueError, match=r"B100|bound-line|nodes|domain|reconstruction|chunk"
    ):
        tool.rewrite(invocation)
    assert not Path(invocation.out).exists()


@pytest.mark.parametrize("defect", ["parent", "output", "inventory", "stop"])
def test_late_mutation_preserves_failure(tmp_path, monkeypatch, defect) -> None:
    invocation, _ = fixture(tmp_path)
    atomic = tool.rank._atomic_json
    copier = tool.copy_shard

    def mutate_storage(root):
        group: Any = zarr.open_group(str(root / "shard_000000.zarr"), mode="a")
        group["policy_target"][0] = group["policy_target"][1]

    def copy_then_mutate(src, dst):
        result = copier(src, dst)
        if defect == "parent":
            mutate_storage(Path(invocation.tactical_bt4_source))
        elif defect == "inventory":
            (Path(invocation.tactical_bt4_source) / "shard_000999.zarr").mkdir(
                exist_ok=True
            )
        return result

    def mutate(path, value):
        atomic(path, value)
        if path.name == tool.derive.SUMMARY_NAME:
            if defect == "stop":
                (tmp_path / "STOP").touch()
            elif defect == "output":
                mutate_storage(Path(invocation.out + ".writing"))

    monkeypatch.setattr(tool, "copy_shard", copy_then_mutate)
    monkeypatch.setattr(tool.rank, "_atomic_json", mutate)
    with pytest.raises(ValueError, match=r"changed|inventory|STOP"):
        tool.rewrite(invocation)
    assert not Path(invocation.out).exists()
    assert (Path(invocation.out + ".writing") / "failed.json").exists()


@pytest.mark.parametrize(
    "defect", ["missing_pin", "orphan_pin", "temperature", "space"]
)
def test_tactical_cli_does_not_ignore_flags(tmp_path, defect):
    invocation, _ = fixture(tmp_path)
    if defect == "missing_pin":
        invocation.expected_bt4_summary_sha256 = None
    elif defect == "orphan_pin":
        invocation.tactical_bt4_source = None
    elif defect == "temperature":
        invocation.temperature = 0.01
    else:
        invocation.score_space = "effective-cp"
    with pytest.raises(ValueError, match=r"both pins|cannot override"):
        tool.rewrite(invocation)
    assert not Path(invocation.out).exists()
    assert not Path(invocation.out + ".writing").exists()


def test_winning_mates_with_zero_b100_mass_are_reported_not_invented():
    obs, base = core([99900, 50000, 100, -99900], [0, 0, 0.6, 0.4])
    result, diagnostic = tool.tactical_target(obs, base)
    assert diagnostic["category"] == "winning_mate_available"
    assert diagnostic["winning_mate_zero_base_mass"] == 1
    assert np.all(result[:2] == 0)
    np.testing.assert_array_equal(result, base)
