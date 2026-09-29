"""Regression checks for mixed-pack metadata and physical shard identities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.replay.mixed_source_pack_identity import (
    PackSourceIdentity,
    PreservedParentPack,
    SourceGameIdentity,
)


@pytest.mark.parametrize("field", ["source_namespace", "opening_stratum"])
@pytest.mark.parametrize("value", [1, True, b"source", None, ""])
def test_source_fields_require_nonempty_strings(field: str, value: Any) -> None:
    values: dict[str, Any] = {
        "run_manifest_sha256": "a" * 64,
        "source_namespace": "bt4",
        "opening_stratum": "root_a",
    }
    values[field] = value
    with pytest.raises(ValueError, match="nonempty strings"):
        PackSourceIdentity(**values)
    with pytest.raises(ValueError, match="nonempty strings"):
        SourceGameIdentity(**values, game_id=7)


@pytest.mark.parametrize("value", [1, True, b"a" * 64, None])
def test_manifest_hash_requires_string(value: Any) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        PackSourceIdentity(value, "bt4", "root_a")


@pytest.mark.parametrize("cross_source", [False, True])
def test_zip_hard_links_are_rejected_before_staging(
    tmp_path: Path, cross_source: bool,
) -> None:
    first = PackSourceIdentity("a" * 64, "bt4", "root_a")
    second = PackSourceIdentity("b" * 64, "ceres", "root_a")
    other = second if cross_source else first
    pack = PreservedParentPack([first, second], shard_format="zip_stored")
    root = tmp_path / "pack"
    original = pack.source_parent(root, first) / "shard_000000.zarr.zip"
    linked = pack.source_parent(root, other) / "shard_000001.zarr.zip"
    original.parent.mkdir(parents=True)
    linked.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"staging fixture, not a qualified ZIP")
    os.link(original, linked)
    stage = tmp_path / "stage"
    with pytest.raises(ValueError, match="physical shards must be distinct"):
        pack.stage_shards(root, stage, [(first, original), (other, linked)])
    assert not stage.exists()
    assert original.read_bytes() == linked.read_bytes()


def test_distinct_zip_files_with_equal_bytes_can_be_staged(tmp_path: Path) -> None:
    source = PackSourceIdentity("a" * 64, "bt4", "root_a")
    pack = PreservedParentPack([source], shard_format="zip_stored")
    root = tmp_path / "pack"
    parent = pack.source_parent(root, source)
    parent.mkdir(parents=True)
    first = parent / "shard_000000.zarr.zip"
    second = parent / "shard_000001.zarr.zip"
    first.write_bytes(b"same unqualified fixture bytes")
    second.write_bytes(first.read_bytes())
    stage = tmp_path / "stage"
    pack.stage_shards(root, stage, [(source, first), (source, second)])
    assert len(list(stage.iterdir())) == 2
    assert (stage / first.name).resolve() == first
    assert (stage / second.name).resolve() == second
