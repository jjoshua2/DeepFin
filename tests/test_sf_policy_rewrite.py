"""Exercise the producer against actual derived/shuffled history fixtures."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import derive_corpus_targets as derive
from scripts import sf_policy_rewrite as rewrite
from tests.test_derive_corpus_targets import (
    CONFIG_SHA,
    history_row,
    write_corpus,
    write_manifest,
)
from tests.test_derive_parallel import run, write_split_corpus


def raw_row(**kwargs: Any) -> dict[str, Any]:
    row = history_row(**kwargs)
    lines = row["phases"][0]["per_depth"][0]["lines"]
    lines.sort(key=lambda line: -line[2])
    for index, line in enumerate(lines, 1):
        line[0] = index
    return row


def fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    rows = [raw_row(game_id=i) for i in range(7)]
    rows[2]["result"] = None
    # Distinct scores on each row make a wrong within-shard join observable.
    for i, row in enumerate(rows):
        for line in row["phases"][0]["per_depth"][0]["lines"]:
            line[2] = float(line[2]) * (i + 1) + 0.0000000123
    raw = write_corpus(
        tmp_path, rows, row_schema=3, staircase=[{"depth": 9, "width": "all"}]
    )
    source = tmp_path / "derived"
    run(raw, source, "--limit", "7", temp=0.0005, rows_per_shard=4)
    return raw, source, rows


def args(raw: Path, source: Path, out: Path, *extra: str) -> Any:
    return rewrite.build_parser().parse_args(
        [
            "--raw",
            str(raw),
            "--source",
            str(source),
            "--out",
            str(out),
            "--expected-source-summary-sha256",
            rewrite.file_sha256(source / derive.SUMMARY_NAME),
            "--minimum-free-gib",
            "0",
            *extra,
        ]
    )


def test_actual_shuffled_rewrite_preserves_nonpolicy_and_default_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, source, rows = fixture(tmp_path)

    def no_encoding(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("rewrite must not re-encode stored history")

    monkeypatch.setattr(derive.TargetDeriver, "_encode", no_encoding)
    identity = tmp_path / "identity"
    assert (
        rewrite.main(
            [
                "--raw",
                str(raw),
                "--source",
                str(source),
                "--out",
                str(identity),
                "--expected-source-summary-sha256",
                rewrite.file_sha256(source / derive.SUMMARY_NAME),
                "--minimum-free-gib",
                "0",
            ]
        )
        == 0
    )
    result = rewrite.rewrite(
        args(
            raw,
            source,
            tmp_path / "cp",
            "--score-space",
            "effective-cp",
            "--temperature",
            "10",
        )
    )
    assert (result["rows"], result["shards"], result["rows_dropped_no_result"]) == (
        6,
        2,
        1,
    )
    assert result["changed_rows"] == 6
    assert "Inherited" in result["history_lineage"]
    for shard in sorted(source.glob("shard_*.zarr")):
        original: Any = zarr.open_group(str(shard), mode="r")
        control: Any = zarr.open_group(str(identity / shard.name), mode="r")
        cp: Any = zarr.open_group(str(tmp_path / "cp" / shard.name), mode="r")
        np.testing.assert_array_equal(
            original["policy_target"][:], control["policy_target"][:]
        )
        for column in rewrite.ARRAYS - {"policy_target"}:
            for path in (shard / column).iterdir():
                assert (
                    path.read_bytes()
                    == (tmp_path / "cp" / shard.name / column / path.name).read_bytes()
                )
        attrs = dict(cp.attrs)
        attrs.pop("policy_target_rewrite")
        assert attrs == dict(original.attrs)
        for i, game in enumerate(cp["game_id"][:]):
            obs = rewrite.observation(rows[int(game)], CONFIG_SHA)
            scores = np.array(
                [
                    line[2]
                    for line in rows[int(game)]["phases"][0]["per_depth"][0]["lines"]
                ]
            )
            weights = np.exp((scores - scores.max()) / 10.0)
            expected = (weights / weights.sum()).astype(np.float32).astype(np.float16)
            np.testing.assert_array_equal(cp["policy_target"][i][obs.indices], expected)
        assert np.array_equal(cp["game_id"][:], original["game_id"][:])


@pytest.mark.parametrize(
    "defect",
    ["duplicate", "illegal", "nonfinite", "truncated", "later_phase", "config"],
)
def test_full_raw_admission_refuses(defect: str) -> None:
    row = raw_row()
    phase = row["phases"][0]
    lines = phase["per_depth"][0]["lines"]
    if defect == "duplicate":
        lines[-1][1] = lines[0][1]
    elif defect == "illegal":
        lines[-1][1] = "a1a8"
    elif defect == "nonfinite":
        lines[0][2] = float("nan")
    elif defect == "truncated":
        lines.pop()
    elif defect == "later_phase":
        phase["index"] = 1
    else:
        row["run"]["config_sha256"] = "0" * 64
    with pytest.raises((ValueError, derive.CorpusIntegrityError)):
        rewrite.observation(row, CONFIG_SHA)


def test_original_float64_scores_and_mate_distance_are_preserved() -> None:
    row = raw_row()
    lines = row["phases"][0]["per_depth"][0]["lines"]
    for i, line in enumerate(lines):
        line[2] = 10000.123456789 - i * 10.123456789
    obs = rewrite.observation(row, CONFIG_SHA)
    assert obs.scores.dtype == np.float64
    np.testing.assert_array_equal(obs.scores, [line[2] for line in lines])
    probabilities = rewrite.target(obs, "effective-cp", 10)[obs.indices]
    assert probabilities[0] > probabilities[1] > probabilities[2]
    # q is saturated at these scores, whereas effective-cp retains mate distances.
    q = rewrite.target(obs, "q", 0.0005)[obs.indices]
    assert len(np.unique(q)) == 1
    assert not np.array_equal(q, probabilities)


@pytest.mark.parametrize("defect", ["summary", "old_policy", "row_join", "attrs"])
def test_source_mismatch_refuses_without_success(tmp_path: Path, defect: str) -> None:
    raw, source, _ = fixture(tmp_path)
    if defect == "summary":
        path = source / derive.SUMMARY_NAME
        summary = json.loads(path.read_text())
        summary["temp_requested"] = 0.1
        path.write_text(json.dumps(summary))
    else:
        g: Any = zarr.open_group(str(source / "shard_000000.zarr"), mode="a")
        if defect == "old_policy":
            g["policy_target"][0] = 0
        elif defect == "row_join":
            g["game_id"][0] = 999
        else:
            g.attrs["derive_value_scheme"] = "different"
    with pytest.raises(ValueError, match="differs"):
        rewrite.rewrite(args(raw, source, tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_late_source_mutation_retains_partial_and_refuses_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, source, _ = fixture(tmp_path)
    original = rewrite.copy_shard

    def changing_copy(src: Path, dst: Path) -> dict[str, str]:
        proof = original(src, dst)
        if src.name == "shard_000001.zarr":
            path = source / "shard_000000.zarr" / "x" / ".zarray"
            path.write_bytes(path.read_bytes() + b"\n")
        return proof

    monkeypatch.setattr(rewrite, "copy_shard", changing_copy)
    with pytest.raises(ValueError, match="source changed before publication"):
        rewrite.rewrite(args(raw, source, tmp_path / "out"))
    assert (tmp_path / "out.writing" / "failed.json").is_file()
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="partial already exists"):
        rewrite.rewrite(args(raw, source, tmp_path / "out"))


def test_source_contract_preserves_historical_limits(tmp_path: Path) -> None:
    raw, source, _ = fixture(tmp_path)
    summary = json.loads((source / derive.SUMMARY_NAME).read_text())
    original = copy.deepcopy(summary)
    rewrite.source_contract(summary, derive.read_corpus_record(raw))
    assert summary == original


def test_raw_prefix_crosses_closed_shards_and_stops_inside_last(tmp_path: Path) -> None:
    rows = [raw_row(game_id=i) for i in range(9)]
    rows[2]["result"] = None
    raw = write_split_corpus(tmp_path, rows, [3, 3, 3])
    staircase = [{"depth": 9, "width": "all"}]
    manifest = write_manifest(raw, row_schema=3, staircase=staircase)
    progress = [
        json.loads(line)
        for line in (raw / "w00.progress.jsonl").read_text().splitlines()
    ]
    (raw / "summary.json").write_text(
        json.dumps(
            {
                **manifest,
                "run_id": "test_corpus",
                "run_finished": True,
                "shards": progress,
            }
        )
    )
    source = tmp_path / "derived"
    run(raw, source, "--limit", "7", temp=0.0005, rows_per_shard=4)
    result = rewrite.rewrite(args(raw, source, tmp_path / "out"))
    assert [item["rows_consumed"] for item in result["raw_files"].values()] == [3, 3, 1]
    assert result["raw_limit"] == 7
    assert result["rows"] == 6
    assert result["changed_rows"] == 0


def test_stop_during_final_summary_prevents_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, source, _ = fixture(tmp_path)
    atomic = rewrite.rank._atomic_json

    def stop_after_summary(path: Path, value: dict[str, Any]) -> None:
        atomic(path, value)
        if path.name == derive.SUMMARY_NAME:
            (tmp_path / "STOP").write_text("stop before publication")

    monkeypatch.setattr(rewrite.rank, "_atomic_json", stop_after_summary)
    with pytest.raises(ValueError, match="STOP requested"):
        rewrite.rewrite(args(raw, source, tmp_path / "out"))
    assert not (tmp_path / "out").exists()
    assert (tmp_path / "out.writing" / "failed.json").is_file()


def test_completed_summary_only_legacy_source(tmp_path: Path) -> None:
    raw, source, _ = fixture(tmp_path)
    (raw / "manifest.json").unlink()
    result = rewrite.rewrite(args(raw, source, tmp_path / "out"))
    assert result["rows"] == 6
    assert result["changed_rows"] == 0
    assert result["raw_manifest_present"] is False
    assert str(raw / "summary.json") in result["metadata_sha256"]
    assert str(raw / "manifest.json") not in result["metadata_sha256"]


def test_manifest_appearing_after_legacy_admission_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, source, _ = fixture(tmp_path)
    path = raw / "manifest.json"
    original_manifest = path.read_bytes()
    path.unlink()
    copy_shard = rewrite.copy_shard

    def add_manifest(src: Path, dst: Path) -> dict[str, str]:
        copied = copy_shard(src, dst)
        path.write_bytes(original_manifest)
        return copied

    monkeypatch.setattr(rewrite, "copy_shard", add_manifest)
    with pytest.raises(ValueError, match="raw manifest presence changed"):
        rewrite.rewrite(args(raw, source, tmp_path / "out"))
    assert not (tmp_path / "out").exists()
    assert (tmp_path / "out.writing" / "failed.json").exists()
