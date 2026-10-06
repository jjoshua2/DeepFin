"""Directory producer density seal: missing chunks fail before fill decoding.

Synthetic shards only. A 520-row shard is the smallest fixture that makes
``save_local_shard_arrays`` emit a second chunk under the production 512-row
chunker. Unsealed historical shards are not claimed valid: the legacy test
below shows that stripping the manifest and every array attribute still
zero-fills a deleted tail and can still complete an epoch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pytest
import zarr
from zarr.storage import DirectoryStore

from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.replay import packed_zarr as packed
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.directory_seal import SEAL_ATTR, SEAL_FILENAME
from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer
from chess_anti_engine.replay.shard import (
    ShardMeta,
    load_shard_arrays,
    samples_to_arrays,
    save_local_shard_arrays,
)

PLANES = 8
TAIL = 512


def _sample(row: int, *, tail_fill: bool) -> ReplaySample:
    policy = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float32)
    policy[row % COMPACT_POLICY_SIZE] = 1.0
    legal = np.zeros_like(policy, dtype=np.uint8)
    legal[row % COMPACT_POLICY_SIZE] = 1
    wdl = 0 if tail_fill and row >= TAIL else row % 3
    item = ReplaySample(
        x=np.full((PLANES, 8, 8), (row % 17) + 1, dtype=np.float32),
        policy_target=policy,
        legal_mask=legal,
        wdl_target=wdl,
        priority=1.0,
        has_policy=True,
        game_id=row,
        ply_index=row,
    )
    item.input_history_encoding = "legacy"
    item.history_rep_fix = False
    return item


def _write(
    directory: Path,
    *,
    n: int = 520,
    tail_fill: bool = False,
    density: str = "dense",
    name: str = "shard_000000.zarr",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    samples = [_sample(row, tail_fill=tail_fill) for row in range(n)]
    path = directory / name
    save_local_shard_arrays(
        path,
        arrs=samples_to_arrays(samples),
        meta=ShardMeta(
            positions=n,
            policy_encoding="lc0_1858",
            policy_size=COMPACT_POLICY_SIZE,
            input_history_encoding="legacy",
            history_rep_fix=False,
        ),
        chunk_density=density,
    )
    return path


def _expected_wdl(n: int, *, tail_fill: bool) -> np.ndarray:
    return np.array(
        [0 if tail_fill and row >= TAIL else row % 3 for row in range(n)],
        dtype=np.int8,
    )


def _epoch(directory: Path, *, rows: int) -> GameAwareEpochBuffer:
    return GameAwareEpochBuffer(
        shard_dir=directory,
        batch_size=rows,
        seed=121,
        input_planes=PLANES,
        input_history_encoding="legacy",
        history_rep_fix=False,
        mirror_augmentation=False,
        plan_workers=1,
        load_workers=1,
    )


def _forbid_fill_decode(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"getitem": 0, "decode": 0}

    def getitem(_self: object, *_args: object, **_kwargs: object) -> None:
        calls["getitem"] += 1
        raise AssertionError("Array.__getitem__")

    def decode(_self: object, *_args: object, **_kwargs: object) -> None:
        calls["decode"] += 1
        raise AssertionError("_decode_chunk")

    monkeypatch.setattr(zarr.Array, "__getitem__", getitem)
    monkeypatch.setattr(zarr.Array, "_decode_chunk", decode)
    return calls


def _seal(path: Path) -> dict[str, Any]:
    doc = json.loads((path / SEAL_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def test_invalid_chunk_density_publishes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "shard_000000.zarr"
    samples = [_sample(row, tail_fill=False) for row in range(2)]
    with pytest.raises(ValueError, match="chunk_density"):
        save_local_shard_arrays(
            path,
            arrs=samples_to_arrays(samples),
            meta=ShardMeta(positions=2, policy_encoding="lc0_1858", policy_size=COMPACT_POLICY_SIZE),
            chunk_density="sparse",
        )
    assert not path.exists()
    assert list(tmp_path.glob("._tmp_*")) == []


def test_sealed_dense_roundtrip_binds_checksums_outside_group_meta(tmp_path: Path) -> None:
    path = _write(tmp_path / "replay", n=520, tail_fill=False)
    doc = _seal(path)
    assert doc["density"] == "dense"
    assert doc["kind"] == "directory-producer-density-seal"
    wdl = doc["arrays"]["wdl_target"]
    assert isinstance(wdl, dict)
    assert wdl["elided"] == []
    assert wdl["order"] == "C"
    stored = wdl["stored"]
    assert isinstance(stored, list)
    assert [item["key"] for item in stored] == ["wdl_target/0", "wdl_target/1"]
    loaded, meta = load_shard_arrays(path)
    np.testing.assert_array_equal(loaded["wdl_target"], _expected_wdl(520, tail_fill=False))
    assert SEAL_ATTR not in meta
    assert SEAL_FILENAME not in meta
    group = zarr.open_group(str(path), mode="r")
    digest = hashlib.sha256(bytes(group.store[SEAL_FILENAME])).hexdigest()
    for name in group.array_keys():
        assert group[name].attrs[SEAL_ATTR] == digest


def test_sealed_nonzero_missing_tail_fails_before_fill_and_epoch_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "replay"
    path = _write(directory, n=520, tail_fill=False)
    tail = path / "wdl_target" / "1"
    assert tail.is_file()
    assert np.any(_expected_wdl(520, tail_fill=False)[TAIL:] != 0)
    tail.unlink()
    calls = _forbid_fill_decode(monkeypatch)
    with pytest.raises(ValueError, match="stored chunk 'wdl_target/1'"):
        load_shard_arrays(path)
    with pytest.raises(ValueError, match="directory producer seal"):
        _epoch(directory, rows=520)
    assert calls == {"getitem": 0, "decode": 0}


def test_dense_fill_chunk_stays_stored_and_deletion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(tmp_path / "replay", n=520, tail_fill=True)
    assert (path / "wdl_target" / "1").is_file()
    doc = _seal(path)
    wdl = doc["arrays"]["wdl_target"]
    assert isinstance(wdl, dict)
    assert wdl["elided"] == []
    loaded, _meta = load_shard_arrays(path)
    np.testing.assert_array_equal(loaded["wdl_target"], _expected_wdl(520, tail_fill=True))
    (path / "wdl_target" / "1").unlink()
    calls = _forbid_fill_decode(monkeypatch)
    with pytest.raises(ValueError, match="stored chunk 'wdl_target/1'"):
        load_shard_arrays(path)
    assert calls == {"getitem": 0, "decode": 0}


def test_sparse_fill_omitted_tail_decodes_as_fill(tmp_path: Path) -> None:
    path = _write(tmp_path / "replay", n=520, tail_fill=True, density="sparse-fill")
    assert not (path / "wdl_target" / "1").exists()
    assert (path / "wdl_target" / "0").is_file()
    doc = _seal(path)
    assert doc["density"] == "sparse-fill"
    wdl = doc["arrays"]["wdl_target"]
    assert isinstance(wdl, dict)
    assert wdl["elided"] == ["wdl_target/1"]
    loaded, _meta = load_shard_arrays(path)
    np.testing.assert_array_equal(loaded["wdl_target"], _expected_wdl(520, tail_fill=True))
    assert np.any(np.asarray(loaded["x"][TAIL:]) != 0)


def test_sparse_fill_refuses_to_publish_an_omitted_nonzero_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = DirectoryStore.__setitem__

    def skip(self: DirectoryStore, key: str, value: object) -> None:
        if key == "wdl_target/1":
            return
        real(self, key, value)

    monkeypatch.setattr(DirectoryStore, "__setitem__", skip)
    directory = tmp_path / "replay"
    path = directory / "shard_000000.zarr"
    with pytest.raises(ValueError, match="refusing to publish"):
        _write(directory, n=520, tail_fill=False, density="sparse-fill")
    assert not path.exists()
    assert list(directory.glob("._tmp_*")) == []


def test_swapped_chunk_bytes_fail_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(tmp_path / "replay", n=520, tail_fill=False)
    head = path / "wdl_target" / "0"
    tail = path / "wdl_target" / "1"
    head_bytes = head.read_bytes()
    tail_bytes = tail.read_bytes()
    assert head_bytes != tail_bytes
    head.write_bytes(tail_bytes)
    tail.write_bytes(head_bytes)
    calls = _forbid_fill_decode(monkeypatch)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_shard_arrays(path)
    assert calls == {"getitem": 0, "decode": 0}


def test_fortran_order_edit_fails_before_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``.zarray`` order reshapes each chunk. Checksums do not see that field."""
    path = _write(tmp_path / "replay", n=8, tail_fill=False)
    meta = path / "x" / ".zarray"
    text = meta.read_text(encoding="utf-8")
    needle = '"order": "C"'
    assert needle in text
    meta.write_text(text.replace(needle, '"order": "F"', 1), encoding="utf-8")
    calls = _forbid_fill_decode(monkeypatch)
    with pytest.raises(ValueError, match="order for 'x'"):
        load_shard_arrays(path)
    assert calls == {"getitem": 0, "decode": 0}


def test_extra_array_fails_closed_before_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(tmp_path / "replay", n=8, tail_fill=False)
    group = zarr.open_group(str(path), mode="a")
    group.create_dataset(
        "has_moves_left",
        data=np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
        overwrite=True,
    )
    calls = _forbid_fill_decode(monkeypatch)
    with pytest.raises(ValueError, match="array set does not match"):
        load_shard_arrays(path)
    assert calls == {"getitem": 0, "decode": 0}


def test_renamed_chunk_key_is_an_extra_stored_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(tmp_path / "replay", n=520, tail_fill=False)
    tail = path / "wdl_target" / "1"
    (path / "wdl_target" / "9").write_bytes(tail.read_bytes())
    tail.unlink()
    calls = _forbid_fill_decode(monkeypatch)
    with pytest.raises(ValueError, match="extra stored key 'wdl_target/9'"):
        load_shard_arrays(path)
    assert calls == {"getitem": 0, "decode": 0}


def test_removing_the_manifest_while_attributes_remain_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(tmp_path / "replay", n=520, tail_fill=False)
    (path / SEAL_FILENAME).unlink()
    calls = _forbid_fill_decode(monkeypatch)
    with pytest.raises(ValueError, match="seal file is missing"):
        load_shard_arrays(path)
    assert calls == {"getitem": 0, "decode": 0}


def test_unknown_seal_field_is_rejected_when_the_attribute_hash_matches(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "replay", n=8, tail_fill=False)
    doc = _seal(path)
    doc["note"] = "ignore-me"
    payload = json.dumps(
        doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    (path / SEAL_FILENAME).write_bytes(payload)
    group = zarr.open_group(str(path), mode="a")
    raw = group.store[SEAL_FILENAME]
    digest = hashlib.sha256(bytes(raw)).hexdigest()
    for name in list(group.array_keys()):
        group[name].attrs[SEAL_ATTR] = digest
    with pytest.raises(ValueError, match="schema keys"):
        load_shard_arrays(path)


def test_unsealed_missing_tail_still_zero_fills_and_can_complete_an_epoch(
    tmp_path: Path,
) -> None:
    """Stripping the seal is the legacy condition, not a validation.

    Historical directory shards have no manifest and no array attribute. This
    test builds that on-disk condition from a sealed write, then deletes a
    nonzero tail. The load zero-fills and the epoch receipt can still complete.
    """
    directory = tmp_path / "replay"
    path = _write(directory, n=520, tail_fill=False)
    honest = _expected_wdl(520, tail_fill=False)
    assert np.any(honest[TAIL:] != 0)
    group = zarr.open_group(str(path), mode="a")
    del group.store[SEAL_FILENAME]
    for name in list(group.array_keys()):
        del group[name].attrs[SEAL_ATTR]
    assert not (path / SEAL_FILENAME).exists()
    assert all(SEAL_ATTR not in group[name].attrs for name in group.array_keys())
    (path / "wdl_target" / "1").unlink()
    loaded, _meta = load_shard_arrays(path)
    got = np.asarray(loaded["wdl_target"])
    np.testing.assert_array_equal(got[:TAIL], honest[:TAIL])
    assert np.all(got[TAIL:] == 0)
    assert np.bincount(got, minlength=3).tolist() != np.bincount(honest, minlength=3).tolist()
    buf = _epoch(directory, rows=520)
    batches = [
        buf.sample_batch_arrays(buf.plan.batch_size)
        for _ in range(buf.num_batches)
    ]
    assert len(batches) == 1
    epoch_wdl = np.asarray(batches[0]["wdl_target"])
    assert np.bincount(epoch_wdl, minlength=3).tolist() == np.bincount(got, minlength=3).tolist()
    assert buf.receipt()["complete"] is True
    assert buf.receipt()["rows_realized"] == 520


def test_zero_row_sealed_shard_loads_and_is_not_scheduled(tmp_path: Path) -> None:
    directory = tmp_path / "replay"
    path = _write(directory, n=4, tail_fill=False)
    arrs, _meta = load_shard_arrays(path)
    rows = int(arrs["x"].shape[0])
    empty = {
        name: (
            np.asarray(value)[:0]
            if np.asarray(value).ndim >= 1 and np.asarray(value).shape[0] == rows
            else np.asarray(value)
        )
        for name, value in arrs.items()
    }
    empty_path = directory / "shard_000001.zarr"
    save_local_shard_arrays(empty_path, arrs=empty)
    assert (empty_path / SEAL_FILENAME).is_file()
    loaded, _meta = load_shard_arrays(empty_path)
    assert int(loaded["x"].shape[0]) == 0
    buf = _epoch(directory, rows=4)
    assert buf.plan.shard_count == 1
    assert buf.plan.rows == 4
    batches = [
        buf.sample_batch_arrays(buf.plan.batch_size)
        for _ in range(buf.num_batches)
    ]
    assert len(batches) == 1
    assert sorted(np.asarray(batches[0]["game_id"]).tolist()) == [0, 1, 2, 3]
    assert buf.receipt()["complete"] is True


def test_packed_root_seal_is_opaque_and_a_missing_member_fails_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = tmp_path / "admitted.zarr.zip"
    with zipfile.ZipFile(admitted, "w") as archive:
        archive.writestr(".zgroup", '{"zarr_format":2}')
        archive.writestr(".zattrs", "{}")
        archive.writestr(SEAL_FILENAME, b"{}")
    packed.content_sha256(admitted)
    nested = tmp_path / "nested.zarr.zip"
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr(".zgroup", '{"zarr_format":2}')
        archive.writestr(".zattrs", "{}")
        archive.writestr(f"wdl_target/{SEAL_FILENAME}", b"{}")
    with pytest.raises(ValueError, match="nonordinary"):
        packed.content_sha256(nested)

    directory = tmp_path / "replay"
    path = _write(directory, n=520, tail_fill=False)
    intact = tmp_path / "shard_000000.zarr.zip"
    damaged = tmp_path / "damaged.zarr.zip"
    with zipfile.ZipFile(intact, "w", compression=zipfile.ZIP_STORED) as archive:
        for file in sorted(path.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(path).as_posix())
    with zipfile.ZipFile(damaged, "w", compression=zipfile.ZIP_STORED) as archive:
        for file in sorted(path.rglob("*")):
            if file.is_file() and file.relative_to(path).as_posix() != "wdl_target/1":
                archive.write(file, file.relative_to(path).as_posix())
    packed_loaded, _meta = load_shard_arrays(intact)
    np.testing.assert_array_equal(
        packed_loaded["wdl_target"], _expected_wdl(520, tail_fill=False),
    )
    calls = _forbid_fill_decode(monkeypatch)
    with pytest.raises(ValueError, match="stored chunk 'wdl_target/1'"):
        load_shard_arrays(damaged)
    assert calls == {"getitem": 0, "decode": 0}


def test_overlay_branch_does_not_treat_a_copied_attribute_as_this_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlay policy attrs are copied from the base and are not this seal.

    The overlay branch never opens the directory group for this check. A
    deleted nonzero tail therefore still fill-decodes there. Ordinary loads of
    the same directory do not.
    """
    from chess_anti_engine.replay import target_overlay

    directory = tmp_path / "replay"
    path = _write(directory, n=520, tail_fill=False)
    (path / "wdl_target" / "1").unlink()

    def proxies(
        shard: Path, fields: tuple[str, ...], seal: object = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del seal
        group = zarr.open_group(str(shard), mode="r")
        return (
            {name: group[name] for name in fields if name in group},
            dict(group.attrs.asdict()),
        )

    monkeypatch.setattr(target_overlay, "has_overlay", lambda _path: True)
    monkeypatch.setattr(target_overlay, "overlay_content_sha256", lambda _path, seal=None: "overlay")
    monkeypatch.setattr(target_overlay, "seal_for_overlay", lambda _path: object())
    monkeypatch.setattr(target_overlay, "overlay_proxies", proxies)
    loaded, _meta = load_shard_arrays(path, allow_target_overlay=True)
    assert np.all(np.asarray(loaded["wdl_target"])[TAIL:] == 0)
