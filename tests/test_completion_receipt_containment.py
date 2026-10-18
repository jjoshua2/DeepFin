"""Pure path/JSON fixtures for completion receipts; no collector processes."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import ceres_collection_batches as tool


def _raw_chunk(tmp_path: Path, output: Path, completion: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    return {"id": "c0", "start_shard": 0, "max_shards": 1,
            "expected_rows": 32, "expected_padding_rows": 0,
            "output_directory": str(output), "working_directory": str(tmp_path),
            "argv": [sys.executable], "timeout_seconds": tool.MIN_CHUNK_SECONDS,
            "completion_mode": tool.COMPLETION_FIXED, "expected_completion": str(completion)}


def _validate(raw: dict, tmp_path: Path) -> dict:
    return tool.validate_chunk(raw, 0, tmp_path / "state", set(), [], [])


def _consumption_chunk(output: Path, completion: Path | None = None) -> dict:
    return {"id": "c0", "start_shard": 0, "max_shards": 1,
            "expected_rows": 32, "expected_padding_rows": 0,
            "output_directory": output, "expected_completion": completion,
            "completion_mode": tool.COMPLETION_FIXED if completion is not None else tool.COMPLETION_CERES}


def _payload() -> dict:
    counts = {"real_rows": 32, "padding_rows": 0, "input_rows": 32, "calls": 1}
    return {"complete": True, "rows": 32, "shards": 1, "new_shards": 1,
            "selection": [{"path": "shard_000000.zarr", "rows": 32}],
            "collection_counts": counts, "new_collection_counts": counts}


def _write_receipt(path: Path, *, ended: float | None = None) -> str:
    body = _payload()
    if ended is not None:
        body["ended_unix"] = ended
    blob = json.dumps(body).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def _forbid_receipt_reads(monkeypatch) -> None:
    def forbidden(_path):
        pytest.fail("escaped receipt reached JSON or hash read")

    monkeypatch.setattr(tool, "load_json", forbidden)
    monkeypatch.setattr(tool, "sha256_file", forbidden)


@pytest.mark.parametrize("escape", ["dotdot", "sibling-prefix", "normalized-root"])
def test_plan_rejects_normalized_receipt_escape(tmp_path: Path, escape: str) -> None:
    output = tmp_path / "outputs" / "chunk"
    completion = {
        "dotdot": output / ".." / "outside.json",
        "sibling-prefix": output / ".." / "chunk-other" / "completed.json",
        "normalized-root": output / "nested" / "..",
    }[escape]
    raw = _raw_chunk(tmp_path, output, completion)
    with pytest.raises(ValueError, match="not inside the output directory"):
        _validate(raw, tmp_path)


@pytest.mark.parametrize("layout", ["direct", "nested", "normalized"])
def test_plan_accepts_contained_receipts_without_changing_path(tmp_path: Path, layout: str) -> None:
    output = tmp_path / "outputs" / "chunk"
    completion = {"direct": output / "completed.json",
                  "nested": output / "nested" / "completed.json",
                  "normalized": output / "nested" / ".." / "completed.json"}[layout]
    validated = _validate(_raw_chunk(tmp_path, output, completion), tmp_path)
    assert validated["expected_completion"] == completion
    assert validated["output_directory"] == output
    assert not output.exists()


@pytest.mark.parametrize("escape", ["dotdot", "sibling-prefix"])
def test_consumption_rejects_traversal_before_receipt_reads(tmp_path: Path, monkeypatch, escape: str) -> None:
    output = tmp_path / "chunk"
    output.mkdir()
    completion = (output / ".." / "outside.json" if escape == "dotdot" else
                  output / ".." / "chunk-other" / "completed.json")
    _forbid_receipt_reads(monkeypatch)
    with pytest.raises(ValueError, match="not inside the output directory"):
        tool.validate_completion(_consumption_chunk(output, completion), 10, 20)


@pytest.mark.parametrize("after_plan", [False, True])
def test_consumption_rechecks_escaping_symlink_parent(tmp_path: Path, monkeypatch, after_plan: bool) -> None:
    output, outside = tmp_path / "outputs" / "chunk", tmp_path / "outside"
    completion = output / "link" / "completed.json"
    if after_plan:
        chunk = _validate(_raw_chunk(tmp_path, output, completion), tmp_path)
    else:
        chunk = _consumption_chunk(output, completion)
    output.mkdir(parents=True)
    outside.mkdir()
    (output / "link").symlink_to(outside, target_is_directory=True)
    _forbid_receipt_reads(monkeypatch)
    with pytest.raises(ValueError, match="not inside the output directory"):
        tool.validate_completion(chunk, 10, 20)


@pytest.mark.parametrize("layout", ["direct", "nested", "normalized", "contained-link", "output-parent-alias"])
def test_consumption_preserves_contained_receipts_and_digest(tmp_path: Path, layout: str) -> None:
    parent = tmp_path / "outputs"
    parent.mkdir()
    if layout == "output-parent-alias":
        alias = tmp_path / "logical-outputs"
        alias.symlink_to(parent, target_is_directory=True)
        parent = alias
    output = parent / "chunk"
    completion = output / "completed.json"
    if layout == "nested":
        completion = output / "nested" / "completed.json"
    elif layout == "normalized":
        completion = output / "nested" / ".." / "completed.json"
    elif layout == "contained-link":
        completion = output / "link" / "completed.json"
    chunk = _validate(_raw_chunk(tmp_path, output, completion), tmp_path)
    output.mkdir()
    if layout == "contained-link":
        target = output / "real"
        target.mkdir()
        (output / "link").symlink_to(target, target_is_directory=True)
    elif layout == "normalized":
        (output / "nested").mkdir()
    expected_sha = _write_receipt(completion)
    assert tool.validate_completion(chunk, 10, 20) == (completion, expected_sha)


@pytest.mark.parametrize("target_location", ["inside", "outside"])
def test_final_receipt_symlink_still_refused(tmp_path: Path, monkeypatch, target_location: str) -> None:
    output = tmp_path / "chunk"
    output.mkdir()
    target = output / "target.json" if target_location == "inside" else tmp_path / "target.json"
    completion = output / "completed.json"
    completion.symlink_to(target)
    _forbid_receipt_reads(monkeypatch)
    with pytest.raises(ValueError, match="receipt is a symlink"):
        tool.validate_completion(_consumption_chunk(output, completion), 10, 20)


@pytest.mark.parametrize("mode", ["valid", "child-only", "ambiguous", "stale"])
def test_ceres_completion_mode_preserved(tmp_path: Path, mode: str) -> None:
    output = tmp_path / "chunk"
    run = output / "invocations" / "run-one"
    expected_sha = ""
    if mode == "child-only":
        _write_receipt(run / "child_completed.json", ended=15)
    else:
        expected_sha = _write_receipt(run / "completed.json", ended=5 if mode == "stale" else 15)
    if mode == "ambiguous":
        _write_receipt(output / "invocations" / "run-two" / "completed.json", ended=15)
    chunk = _consumption_chunk(output)
    if mode == "valid":
        assert tool.validate_completion(chunk, 10, 20) == (run / "completed.json", expected_sha)
    else:
        message = {"child-only": "parent completed.json missing", "ambiguous": "ambiguous", "stale": "ended_unix"}[mode]
        with pytest.raises(ValueError, match=message):
            tool.validate_completion(chunk, 10, 20)
