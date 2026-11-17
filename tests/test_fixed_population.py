"""Actual synthetic shard/buffer/gather coverage; no model or historical data."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.replay.fixed_population import load_fixed_population, manifest_identity
from chess_anti_engine.replay.stored_row_once import StoredRowOnceReplay
from scripts import core_finite_learner as core
from tests.test_core_selfplay_union_manifest import ARCH, compiled, frozen, legacy, plan


def population_fixture(tmp_path: Path, branches: int = 6) -> tuple[dict[str, Any], dict[str, str]]:
    refs = [{"kind": "legacy", "manifest": legacy(tmp_path, generation=g, collection=i)}
            for i, g in enumerate((1, 8, 8, 37, 2, 99)[:branches], 1)]
    return compiled(tmp_path, plan(tmp_path, refs, expected_sources=branches, expected_rows=4 * branches))


def test_one_fixed_assembly_preserves_all_branches_rng_and_natural_duplicate_rows(tmp_path: Path) -> None:
    m, mr = population_fixture(tmp_path)
    rng = np.random.default_rng(19)
    before = copy.deepcopy(rng.bit_generator.state)
    pop = load_fixed_population(ARCH, m, mr, rng=rng, max_rows=24, max_decoded_bytes=64 << 20)
    assert len(pop) == 24
    assert pop.rng is rng
    assert rng.bit_generator.state == before
    assert tuple(g for g, _, _ in pop.membership) == (1, 8, 8, 37, 2, 99)
    assert pop.identity == manifest_identity(m, mr)
    pop.check_identity(pop.identity)
    origin = pop.origin_map
    origin["origins"].clear()
    assert len(pop.origin_map["origins"]) == 6
    # All24 boards deliberately identical; none is semantically deduplicated.
    arrays = pop._buffer._gather_rows(np.arange(24))
    assert arrays["x"].shape[0] == 24
    assert np.all(arrays["x"] == 1)


def test_real_quota_sampler_covers_every_admitted_source_row_once(tmp_path: Path) -> None:
    m, mr = population_fixture(tmp_path)
    pop = load_fixed_population(ARCH, m, mr, rng=np.random.default_rng(3), max_rows=24, max_decoded_bytes=64 << 20)
    seen: list[Any] = []
    replay = StoredRowOnceReplay(pop, seed=17, batch_size=4, expected_rows=24, on_draw=seen.extend)
    for _ in range(6):
        arrays = replay.sample_batch_arrays(4)
        assert arrays["x"].shape[0] == 4
    assert len(seen) == 24
    wanted = {(g, sha, row) for g, sha, count in pop.membership for row in range(count)}
    assert {(r.generation, r.source_sha256, r.row_index) for r in seen} == wanted
    with pytest.raises(ValueError, match="stored-row pass exhausted"):
        replay.sample_batch_arrays(4)


def test_global_declared_cap_and_row_cap_refuse_before_eager_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.replay import shard

    m, mr = population_fixture(tmp_path)
    calls = []
    original = shard.load_shard_arrays

    def observed(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(shard, "load_shard_arrays", observed)
    with pytest.raises(ValueError, match="before eager decode"):
        load_fixed_population(ARCH, m, mr, rng=np.random.default_rng(1), max_rows=24, max_decoded_bytes=1)
    assert calls == []
    with pytest.raises(ValueError, match="row cap"):
        load_fixed_population(ARCH, m, mr, rng=np.random.default_rng(1), max_rows=23, max_decoded_bytes=64 << 20)
    assert calls == []


@pytest.mark.parametrize("fault", ["history", "target", "selfplay", "duplicate_ply", "game_id"])
def test_rehashed_wrong_arrays_fail_real_loader_predicates(tmp_path: Path, fault: str) -> None:
    """Alter valid synthetic arrays and reseal descriptors; no guard is mocked."""
    from chess_anti_engine.replay.fixed_population import digest, origin_map
    import zarr

    m, _ = population_fixture(tmp_path, branches=1)
    s = m["sources"][0]
    old_path = Path(s["shard"]["path"])
    new_path = tmp_path / "changed.zarr"
    shutil.copytree(old_path, new_path)
    for p in [new_path, *new_path.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    group: Any = zarr.open_group(str(new_path), mode="a")
    if fault == "history":
        # A changed selected identity must not silently accept the valid shard
        # produced for the original history mode. No invalid shard writer is
        # needed to reach the caller's actual history predicate.
        m["architecture"]["input_history_encoding"] = "wrong_history"
        for origin in m["origins"]:
            origin["identity"]["architecture"]["input_history_encoding"] = "wrong_history"
    elif fault == "target":
        group["wdl_target"][:] = np.full(4, 3, dtype=np.int8)
    elif fault == "selfplay":
        group["is_selfplay"][:] = np.zeros(4, dtype=np.uint8)
    elif fault == "duplicate_ply":
        group["ply_index"][:] = np.zeros(4, dtype=np.int32)
    else:
        group["game_id"][:] = np.full(4, 8, dtype=np.int64)
    core.seal(new_path)
    files = [{"path": p.relative_to(new_path).as_posix(), "bytes": p.stat().st_size,
              "sha256": core.file_sha(p)} for p in sorted(new_path.rglob("*")) if p.is_file()]
    s["shard"] = {"path": str(new_path), "files": files}
    s["source_sha256"] = digest(files)
    # The origin IDs bind the architecture; when changing that selected field
    # rehash the graph rather than relying on an earlier digest refusal.
    old_origin = s["origin_id"]
    for origin in m["origins"]:
        origin["origin_id"] = digest({k: v for k, v in origin.items() if k != "origin_id"})
        if old_origin == s["origin_id"]:
            s["origin_id"] = origin["origin_id"]
    m["origin_map_sha256"] = digest(origin_map(m))
    mr = frozen(tmp_path / "changed-manifest.json", m)
    with pytest.raises((ValueError, TypeError)):
        load_fixed_population(m["architecture"], m, mr, rng=np.random.default_rng(3), max_rows=4, max_decoded_bytes=64 << 20)


def test_changed_manifest_or_origin_map_cannot_resume_same_identity(tmp_path: Path) -> None:
    m, mr = population_fixture(tmp_path, branches=1)
    pop = load_fixed_population(ARCH, m, mr, rng=np.random.default_rng(3), max_rows=4, max_decoded_bytes=64 << 20)
    identity = pop.identity
    identity["origin_map_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint identity"):
        pop.check_identity(identity)
    changed = copy.deepcopy(m)
    changed["sources"][0]["generation"] += 1
    mr2 = frozen(tmp_path / "changed-map.json", changed)
    with pytest.raises(ValueError, match="actual generation"):
        manifest_identity(changed, mr2)


def test_loader_refuses_payload_inventory_mutation_without_model_access(tmp_path: Path) -> None:
    m, mr = population_fixture(tmp_path, branches=1)
    root = Path(m["sources"][0]["shard"]["path"])
    path = root / ".zattrs"
    path.chmod(0o644)
    value = json.loads(path.read_bytes())
    value["tampered"] = True
    path.write_text(json.dumps(value))
    path.chmod(0o444)
    with pytest.raises(ValueError, match="inventory mismatch"):
        load_fixed_population(ARCH, m, mr, rng=np.random.default_rng(3), max_rows=4, max_decoded_bytes=64 << 20)
