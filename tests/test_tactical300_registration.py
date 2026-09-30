"""Synthetic registered-teacher, cohort-uniqueness and inventory regressions."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import tactical300_registration as registration
from scripts import tactical300_calibration as calibration


def test_registered_teacher_is_accepted_without_opening_model() -> None:
    registration.validate_teacher({
        "onnx": {"sha256": registration.BT4_MODEL_SHA256, "path": "unopened.onnx"},
        "policy_output": registration.BT4_POLICY_OUTPUT,
    })


@pytest.mark.parametrize("teacher", [
    None, {}, {"onnx": None},
    {"onnx": {"sha256": "0" * 64}, "policy_output": "/output/policy"},
    {"onnx": {"sha256": registration.BT4_MODEL_SHA256}, "policy_output": "wrong"},
])
def test_unregistered_teacher_is_refused(teacher: object) -> None:
    with pytest.raises(ValueError, match="B100"):
        registration.validate_teacher(teacher)


def test_identity_set_spans_derived_shards() -> None:
    seen: set[tuple[str, str, int]] = set()
    first_shard = [("source_a", "raw_0", 0), ("source_a", "raw_0", 1)]
    second_shard = [("source_b", "raw_0", 0), ("source_a", "raw_0", 0)]
    for row in first_shard:
        registration.claim_source_identity(seen, row)
    registration.claim_source_identity(seen, second_shard[0])
    with pytest.raises(ValueError, match="duplicate source-qualified"):
        registration.claim_source_identity(seen, second_shard[1])
    assert len(seen) == 3


@pytest.mark.parametrize("mutation", ["add", "remove", "rename"])
def test_inventory_is_revalidated_before_publication(tmp_path: Path, mutation: str) -> None:
    original = tmp_path / "shard_000000.zarr"
    original.mkdir()
    names = [original.name]
    assert registration.derived_inventory(tmp_path, names) == [original]
    if mutation == "add":
        (tmp_path / "shard_000001.zarr").mkdir()
    elif mutation == "remove":
        original.rmdir()
    else:
        original.rename(tmp_path / "shard_000001.zarr")
    with pytest.raises(ValueError, match="inventory differs"):
        registration.derived_inventory(tmp_path, names)


@pytest.mark.parametrize(("top", "mates", "expected"), [
    ({"a", "b"}, {"a"}, False),
    ({"a", "b"}, {"a", "b"}, True),
    ({"a"}, {"a", "b"}, True),
    (set(), {"a"}, False),
    ({"a"}, set(), False),
])
def test_tied_top_set_requires_all_moves_to_be_final_mates(
    top: set[str], mates: set[str], expected: bool,
) -> None:
    assert registration.top_set_is_final_mate(top, mates) is expected


def test_changed_producer_source_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = {"producer.py": "before"}
    monkeypatch.setattr(
        calibration, "_producer_sha256", lambda: {"producer.py": "after"}
    )
    with pytest.raises(ValueError, match="producer source changed"):
        calibration._require_unchanged_producers(snapshot)
