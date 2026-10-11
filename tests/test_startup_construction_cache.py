from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.replay import game_epoch as ge
from chess_anti_engine.replay import startup_cache as cache
from tests.test_exact_host_overlap import _iterate, _trainer
from tests.test_game_aware_epoch_replay import _write
from tests.test_target_overlay_v2_main_port import _qualified
from scripts.lc0_control_train import stage_shards


def _counter(arrays: Any) -> dict[str, float]:
    return {"policy": float(np.asarray(arrays["has_policy"]).sum())}


def _fraction_counter(arrays: Any) -> dict[str, float]:
    return {"policy": 1e16 if np.any(np.asarray(arrays["game_id"]) == 0) else 1.0}


def _open(data: Path, **extra: Any) -> ge.GameAwareEpochBuffer:
    kwargs: dict[str, Any] = {
        "shard_dir": data,
        "batch_size": 4,
        "seed": 131,
        "input_planes": 146,
        "input_history_encoding": "legacy",
        "history_rep_fix": False,
        "mirror_augmentation": True,
        "plan_workers": 1,
        "load_workers": 1,
        "max_working_set_bytes": 96 * 1024**2,
        "objective_mask_counter": _counter,
        "startup_cache_recipe": {"target": "fixture-policy-v1"},
    }
    kwargs.update(extra)
    return ge.GameAwareEpochBuffer(**kwargs)


def _data(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "data",
        [
            [(game, game * 10 + row) for row in range(3) for game in range(8)],
            [(game, game * 10 + row) for row in range(3, 5) for game in range(8)],
        ],
    )


def _reference(buffer: ge.GameAwareEpochBuffer, path: Path) -> dict[str, str]:
    assert buffer.startup_cache_sha256 is not None
    return {"path": str(path), "sha256": buffer.startup_cache_sha256}


def _equal_arrays(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert left[name].dtype == right[name].dtype
        assert np.array_equal(left[name], right[name])


@pytest.mark.parametrize(("seed", "overlap"), [(131, False), (177, True)])
def test_real_constructor_tensor_rng_and_receipt_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int, overlap: bool
) -> None:
    data, artifact = _data(tmp_path), tmp_path / "construction"
    cold = _open(
        data, seed=seed, host_batch_overlap=overlap, startup_cache_write=artifact
    )
    reference = _reference(cold, artifact)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("cached constructor repeated census/objective/planning")

    for name in ("_scan_shards", "_attach_objective_mask_weights", "_plan_epoch"):
        monkeypatch.setattr(ge, name, forbidden)
    restored = _open(
        data, seed=seed, host_batch_overlap=overlap, startup_cache_read=reference
    )
    assert cache.canonical(cache.encode(cold.plan)) == cache.canonical(
        cache.encode(restored.plan)
    )
    for _ in range(0, cold.num_batches, 2):
        left = list(_iterate(_trainer(), cold, 2))
        right = list(_iterate(_trainer(), restored, 2))
        for a, b in zip(left, right, strict=True):
            assert a.keys() == b.keys()
            assert all(torch.equal(a[key], b[key]) for key in a)
        for name in ("rng", "_choice_rng", "_row_rng"):
            assert (
                getattr(cold, name).bit_generator.state
                == getattr(restored, name).bit_generator.state
            )
        assert cold._resident_bytes == restored._resident_bytes
        assert cold._next_shard == restored._next_shard
    assert cold.receipt() == restored.receipt()
    assert cold.receipt()["complete"]
    cold.close()
    restored.close()


def test_prefix_rebuild_and_fresh_pass_remain_distinct(tmp_path: Path) -> None:
    data, artifact = _data(tmp_path), tmp_path / "construction"
    original = _open(data, startup_cache_write=artifact)
    reference = _reference(original, artifact)
    resumed = _open(data, startup_cache_read=reference)
    fresh = _open(data, startup_cache_read=reference)
    assert fresh._batch_index == 0
    for _ in range(3):
        _equal_arrays(original.sample_batch_arrays(4), resumed.sample_batch_arrays(4))
    assert resumed._batch_index == 3
    assert fresh._batch_index == 0
    _equal_arrays(original.sample_batch_arrays(4), resumed.sample_batch_arrays(4))
    for buffer in (original, resumed, fresh):
        buffer.close()


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 177},
        {"batch_size": 8},
        {"load_workers": 2},
        {"plan_workers": 2},
        {"mirror_augmentation": False},
        {"host_batch_overlap": True},
        {"max_working_set_bytes": 97 * 1024**2},
        {"input_planes": 175},
        {"startup_cache_recipe": {"target": "other"}},
    ],
)
def test_changed_bindings_reject_before_delivery(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    data, artifact = _data(tmp_path), tmp_path / "construction"
    cold = _open(data, startup_cache_write=artifact)
    with pytest.raises(ValueError, match="binding"):
        _open(data, startup_cache_read=_reference(cold, artifact), **change)
    cold.close()


def test_same_size_changed_bytes_rejected_even_with_preserved_timestamp(
    tmp_path: Path,
) -> None:
    data, artifact = _data(tmp_path), tmp_path / "construction"
    cold = _open(data, startup_cache_write=artifact)
    source = next(
        path for path in data.rglob("*") if path.is_file() and path.name == "0"
    )
    stamp = source.stat()
    raw = bytearray(source.read_bytes())
    raw[-1] ^= 1
    source.write_bytes(raw)
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="binding"):
        _open(data, startup_cache_read=_reference(cold, artifact))
    cold.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-array",
        "dtype",
        "order",
        "rng",
        "namespace",
        "rows",
        "resident",
        "extra-field",
    ],
)
def test_rehashed_invalid_state_rejected(tmp_path: Path, mutation: str) -> None:
    data, artifact = _data(tmp_path), tmp_path / "construction"
    cold = _open(data, startup_cache_write=artifact)
    state = json.loads((artifact / "state.json").read_bytes())
    fields = state["items"]["plan"]["fields"]
    if mutation == "missing-array":
        del fields["load_counts"]
    elif mutation == "dtype":
        fields["batch_rows"]["dtype"] = "<u4"
    elif mutation == "order":
        items = state["items"]["shuffled_records"]["items"]
        items[0] = items[1]
    elif mutation == "rng":
        state["items"]["rng"]["items"]["rng"]["items"]["state"]["items"]["state"] += 1
    elif mutation == "namespace":
        state["items"]["census"]["items"][0]["fields"]["game_keys"]["hex"] = "00" * 64
    elif mutation == "rows":
        fields["rows"] += 1
    elif mutation == "resident":
        fields["resident_bytes_after_batch"]["hex"] = (
            "ff" * 8 + fields["resident_bytes_after_batch"]["hex"][16:]
        )
    else:
        fields["unknown"] = 1
    raw = cache.canonical(state)
    (artifact / "state.json").write_bytes(raw)
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    manifest["state_sha256"] = hashlib.sha256(raw).hexdigest()
    raw = cache.canonical(manifest)
    (artifact / "manifest.json").write_bytes(raw)
    reference = {"path": str(artifact), "sha256": hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ValueError, match="construction"):
        _open(data, startup_cache_read=reference)
    cold.close()


def test_partial_publication_and_atomic_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, artifact = _data(tmp_path), tmp_path / "construction"
    calls = []
    actual = cache.os.fsync

    def fail_first(descriptor: int) -> None:
        calls.append(descriptor)
        if len(calls) == 1:
            raise OSError("fixture fsync failure")
        actual(descriptor)

    monkeypatch.setattr(cache.os, "fsync", fail_first)
    with pytest.raises(OSError, match="fixture"):
        _open(data, startup_cache_write=artifact)
    assert not artifact.exists()
    pending = list(tmp_path.glob(".construction.pending-*"))
    assert len(pending) == 1
    with pytest.raises(ValueError, match="publication"):
        cache.load(pending[0], "0" * 64, {}, {})


def test_extra_file_and_overwrite_rejected(tmp_path: Path) -> None:
    data, artifact = _data(tmp_path), tmp_path / "construction"
    cold = _open(data, startup_cache_write=artifact)
    with pytest.raises(ValueError, match="already exists"):
        _open(data, startup_cache_write=artifact)
    (artifact / "extra").write_text("extra")
    with pytest.raises(ValueError, match="publication"):
        _open(data, startup_cache_read=_reference(cold, artifact))
    cold.close()


def test_competing_empty_publication_fails_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, artifact = _data(tmp_path), tmp_path / "construction"
    original = cache._publish_directory

    def competing(temporary: Path, destination: Path) -> None:
        destination.mkdir()
        original(temporary, destination)

    monkeypatch.setattr(cache, "_publish_directory", competing)
    with pytest.raises(FileExistsError, match="File exists"):
        _open(data, startup_cache_write=artifact)
    assert list(artifact.iterdir()) == []
    assert len(list(tmp_path.glob(".construction.pending-*"))) == 1


def test_actual_frontier_order(tmp_path: Path) -> None:
    data = _write(
        tmp_path / "data",
        [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 10), (6, 60)],
            [(2, 20)],
            [(3, 30)],
            [(4, 40)],
            [(5, 50)],
        ],
    )
    artifact = tmp_path / "construction"
    cold = _open(
        data,
        batch_size=3,
        seed=3,
        startup_cache_write=artifact,
        objective_mask_counter=_fraction_counter,
    )
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    state = cache.load(
        artifact,
        cold.startup_cache_sha256 or "",
        manifest["bindings"],
        {"_ShardGames": ge._ShardGames, "GameEpochPlan": ge.GameEpochPlan},
    )
    initial = [
        state["census"][int(index)].path
        for index in ge._seeded_rng(3, 0).permutation(6)
    ]
    assert initial != [record.path for record in cold._records]
    restored = _open(
        data,
        batch_size=3,
        seed=3,
        startup_cache_read=_reference(cold, artifact),
        objective_mask_counter=_fraction_counter,
    )
    for _ in range(cold.num_batches):
        _equal_arrays(cold.sample_batch_arrays(3), restored.sample_batch_arrays(3))
    cold.close()
    restored.close()


def test_qualified_overlay_constructor_and_namespace(tmp_path: Path) -> None:
    roots, _, qualification = _qualified(tmp_path)
    data = tmp_path / "stage"
    stage_shards(roots, data)
    artifact = tmp_path / "construction"
    kwargs = {
        "batch_size": 4,
        "input_planes": 175,
        "input_history_encoding": "lc0_root_legacy_meta",
        "history_rep_fix": True,
        "overlay_storage_qualification": qualification,
    }
    cold = _open(data, startup_cache_write=artifact, **kwargs)
    restored = _open(data, startup_cache_read=_reference(cold, artifact), **kwargs)
    assert cold.plan.source_count == 2
    _equal_arrays(cold.sample_batch_arrays(4), restored.sample_batch_arrays(4))
    assert cold.receipt() == restored.receipt()
    cold.close()
    restored.close()


def test_ragged_constructor_reuse(tmp_path: Path) -> None:
    data = _write(tmp_path / "data", [[(row % 100, row) for row in range(199)]])
    artifact = tmp_path / "construction"
    cold = _open(data, batch_size=100, startup_cache_write=artifact)
    restored = _open(
        data, batch_size=100, startup_cache_read=_reference(cold, artifact)
    )
    assert cold.plan.min_batch_rows == 99
    assert cold.plan.ragged_batches == 1
    for _ in range(cold.num_batches):
        _equal_arrays(cold.sample_batch_arrays(100), restored.sample_batch_arrays(100))
    assert cold.receipt() == restored.receipt()
    cold.close()
    restored.close()


def test_repeated_raw_game_ids_have_distinct_source_namespaces(tmp_path: Path) -> None:
    sources = [
        _write(
            tmp_path / f"source-{index}",
            [[(game, row) for row in range(2) for game in range(2)]],
        )
        for index in range(2)
    ]
    data, artifact = tmp_path / "stage", tmp_path / "construction"
    stage_shards(sources, data)
    cold = _open(data, startup_cache_write=artifact)
    restored = _open(data, startup_cache_read=_reference(cold, artifact))
    assert cold.plan.source_count == 2
    assert cold.plan.game_count == 4
    for _ in range(cold.num_batches):
        _equal_arrays(cold.sample_batch_arrays(4), restored.sample_batch_arrays(4))
    assert cold.receipt() == restored.receipt()
    cold.close()
    restored.close()


def test_cached_overlap_abort_preserves_incomplete_receipt(tmp_path: Path) -> None:
    data, artifact = _data(tmp_path), tmp_path / "construction"
    cold = _open(data, host_batch_overlap=True, startup_cache_write=artifact)
    restored = _open(
        data, host_batch_overlap=True, startup_cache_read=_reference(cold, artifact)
    )
    for buffer in (cold, restored):
        iterator = _iterate(_trainer(), buffer, 3)
        next(iterator)
        iterator.close()
        assert buffer.receipt()["overlap_aborted"]
        assert not buffer.receipt()["complete"]
        buffer.close()
