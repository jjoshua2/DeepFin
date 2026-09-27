"""Small synthetic identity fixtures; never open registered E payload."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pytest
import zarr
from numcodecs import Blosc

from scripts.selected_e_game_index import (
    FixtureClosure, FixtureShard, index_fixture, select_complete_games,
)


def fixture(root: Path, cases: list[tuple[int, str, list[int]]],
            *, flags: list[list[bool]] | None = None) -> FixtureClosure:
    specs = []
    for ordinal, (cohort, source, games) in enumerate(cases):
        name = f"shard_{ordinal:06d}.zarr"
        group = zarr.open_group(str(root / name), mode="w")
        ids = np.asarray(games, dtype="<i8")
        has = np.asarray(flags[ordinal] if flags else [True] * len(ids), dtype="|b1")
        for column, data in (("game_id", ids), ("has_game_id", has)):
            group.create_dataset(column, data=data, chunks=(max(1, min(128, len(data))),),
                                 compressor=Blosc(cname="zstd", clevel=2, shuffle=2))
        specs.append(FixtureShard(
            cohort_index=cohort, cohort_manifest_sha256=f"{cohort + 1:064x}",
            shard_ordinal=ordinal, shard_name=name, shard_rows=len(ids),
            resolved_base_parent=source,
            game_id_sha256=sha256(ids.tobytes()).hexdigest(),
            has_game_id_sha256=sha256(has.tobytes()).hexdigest(),
        ))
    closure = FixtureClosure("11" * 32, "22" * 32, tuple(specs), "00" * 32)
    return replace(closure, first_sealed_identity_sha256=closure.identity_sha256())


def test_cross_shard_unsorted_games_and_source_scoping(tmp_path: Path) -> None:
    closure = fixture(tmp_path, [(0, "/source/a", [7, 9, 7]),
                                 (1, "/source/a", [9, 7]),
                                 (2, "/source/b", [7, 10])])
    index = index_fixture(tmp_path, closure)
    assert index[("/source/a", 7)] == [(0, 0), (0, 2), (1, 1)]
    assert index[("/source/b", 7)] == [(2, 0)]
    sample = select_complete_games(closure, index, seed_sha256="33" * 32,
                                   target_rows=6, required_cohorts=frozenset({0, 1, 2}))
    assert sample["status"] == "NO-LAUNCH"
    assert len(sample["rows"]) == 6
    assert sample == select_complete_games(closure, index, seed_sha256="33" * 32,
                                           target_rows=6,
                                           required_cohorts=frozenset({0, 1, 2}))
    selected = {(r["resolved_base_parent"], r["game_id"]) for r in sample["rows"]}
    for key in selected:
        assert len([r for r in sample["rows"] if (r["resolved_base_parent"], r["game_id"]) == key]) == len(index[key])


def test_exact_12288_and_35_cohort_coverage(tmp_path: Path) -> None:
    # One complete 384-row game in each cohort; exact subset chooses 32 games.
    # The remaining three cohorts have one-row anchors to force all 35 in.
    cases = [(c, f"/source/{c}", [c] * (384 if c < 32 else 1)) for c in range(35)]
    # 12,288 cannot include three extra anchors, so add 3 replaceable 383-row
    # games and rely on exact subset selection to find the required total.
    cases = [(c, f"/source/{c}", [c] * (383 if c < 3 else 384 if c < 32 else 1))
             for c in range(35)]
    closure = fixture(tmp_path, cases)
    index = index_fixture(tmp_path, closure)
    sample = select_complete_games(closure, index, seed_sha256="44" * 32)
    assert len(sample["rows"]) == 12_288
    assert sample["covered_cohorts"] == list(range(35))
    assert sample["selected_games"] == 35


def test_missing_identity_and_incomplete_index_refused(tmp_path: Path) -> None:
    closure = fixture(tmp_path, [(0, "/source/a", [1, 1])], flags=[[True, False]])
    with pytest.raises(ValueError, match="without game identity"):
        index_fixture(tmp_path, closure)
    closure = fixture(tmp_path / "second", [(0, "/source/a", [1, 1])])
    index = index_fixture(tmp_path / "second", closure)
    index[("/source/a", 1)].pop()
    with pytest.raises(ValueError, match="omits"):
        select_complete_games(closure, index, seed_sha256="33" * 32,
                              target_rows=1, required_cohorts=frozenset({0}))


def test_seal_digest_and_member_closure_refused(tmp_path: Path) -> None:
    closure = fixture(tmp_path, [(0, "/source/a", [1, 2, 3])])
    with pytest.raises(ValueError, match="first-sealed"):
        index_fixture(tmp_path, replace(closure, first_sealed_identity_sha256="00" * 32))
    (tmp_path / "unlisted.zarr").mkdir()
    with pytest.raises(ValueError, match="unlisted members"):
        index_fixture(tmp_path, closure)


def test_linked_root_and_metadata_refused(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.mkdir()
    closure = fixture(original, [(0, "/source/a", [1, 2])])
    alias = tmp_path / "alias"
    alias.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="fixture root contains a link"):
        index_fixture(alias, closure)
    metadata = original / closure.shards[0].shard_name / ".zgroup"
    raw = metadata.read_bytes()
    metadata.unlink()
    metadata.symlink_to(original / closure.shards[0].shard_name / "game_id" / ".zarray")
    with pytest.raises(ValueError, match="linked"):
        index_fixture(original, closure)
    metadata.unlink()
    metadata.write_bytes(raw)


def test_symlink_fifo_size_and_hash_refused(tmp_path: Path) -> None:
    closure = fixture(tmp_path, [(0, "/source/a", [1, 2, 3])])
    column = tmp_path / closure.shards[0].shard_name / "game_id"
    chunk = column / "0"
    raw = chunk.read_bytes()
    chunk.unlink()
    chunk.symlink_to(column / ".zarray")
    with pytest.raises(ValueError, match="linked"):
        index_fixture(tmp_path, closure)
    chunk.unlink()
    chunk.write_bytes(raw)
    bad = replace(closure, shards=(
        replace(closure.shards[0], game_id_sha256="00" * 32),
    ))
    bad = replace(bad, first_sealed_identity_sha256=bad.identity_sha256())
    with pytest.raises(ValueError, match="SHA-256 differs"):
        index_fixture(tmp_path, bad)
    chunk.unlink()
    os.mkfifo(chunk)
    with pytest.raises(ValueError, match="special"):
        index_fixture(tmp_path, closure)
    chunk.unlink()
    chunk.write_bytes(b"x" * (192 * 1024 + 1))
    with pytest.raises(ValueError, match="oversized"):
        index_fixture(tmp_path, closure)


def test_no_exact_whole_game_total_refused(tmp_path: Path) -> None:
    closure = fixture(tmp_path, [(0, "/source/a", [1, 1, 2, 2])])
    with pytest.raises(ValueError, match="no exact row total"):
        select_complete_games(closure, index_fixture(tmp_path, closure),
                              seed_sha256="33" * 32, target_rows=3,
                              required_cohorts=frozenset())


def test_chunk_count_cap_refused_before_decode(tmp_path: Path) -> None:
    closure = fixture(tmp_path, [(0, "/source/a", list(range(600)))])
    metadata = tmp_path / closure.shards[0].shard_name / "game_id" / ".zarray"
    value = json.loads(metadata.read_text())
    value["chunks"] = [1]
    metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="chunk-count cap"):
        index_fixture(tmp_path, closure)


def test_nonbinary_bool_byte_refused(tmp_path: Path) -> None:
    closure = fixture(tmp_path, [(0, "/source/a", [1, 1, 1])])
    chunk = tmp_path / closure.shards[0].shard_name / "has_game_id" / "0"
    values = bytes([1, 2, 1])
    chunk.write_bytes(Blosc(cname="zstd", clevel=2, shuffle=2).encode(values))
    changed = replace(closure, shards=(replace(
        closure.shards[0], has_game_id_sha256=sha256(values).hexdigest()),))
    changed = replace(changed, first_sealed_identity_sha256=changed.identity_sha256())
    with pytest.raises(ValueError, match="nonbinary"):
        index_fixture(tmp_path, changed)
