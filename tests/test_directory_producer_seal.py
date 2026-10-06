"""Dense directory producer seal: missing chunks and .zarray edits fail closed.

Synthetic shards only. A 520-row shard is the smallest fixture that makes
``save_local_shard_arrays`` emit a second chunk under the production 512-row
chunker. The seal binds raw ``.zarray`` bytes and the stored chunk-key
inventory. It does not hash chunk payloads. Unsealed stores keep legacy fill,
including sparse omission, and are not claimed valid.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pytest
import zarr
from zarr.storage import DirectoryStore, MemoryStore

from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.replay import directory_seal, packed_zarr as packed
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.directory_seal import (
    MAX_SEAL_BYTES,
    SEAL_ATTR,
    SEAL_FILENAME,
    verify_directory_producer_seal,
    write_directory_producer_seal,
)
from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import (
    ShardMeta,
    load_shard_arrays,
    samples_to_arrays,
    save_local_shard_arrays,
    shard_positions,
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


def _snapshot(path: Path) -> dict[str, bytes]:
    return {
        file.relative_to(path).as_posix(): file.read_bytes()
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _replace_zarray(path: Path, name: str, mutate) -> None:
    meta = path / name / ".zarray"
    doc = json.loads(meta.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    mutate(doc)
    meta.write_text(json.dumps(doc), encoding="utf-8")


def test_sealed_dense_roundtrip_binds_declaration_outside_group_meta(tmp_path: Path) -> None:
    path = _write(tmp_path / "replay", n=520, tail_fill=False)
    doc = _seal(path)
    assert doc["density"] == "dense"
    assert doc["kind"] == "directory-producer-density-seal"
    wdl = doc["arrays"]["wdl_target"]
    assert isinstance(wdl, dict)
    assert set(wdl) == {"zarray_sha256", "stored"}
    assert wdl["stored"] == ["wdl_target/0", "wdl_target/1"]
    raw_zarray = (path / "wdl_target" / ".zarray").read_bytes()
    assert wdl["zarray_sha256"] == hashlib.sha256(raw_zarray).hexdigest()
    assert '"order":' in raw_zarray.decode("utf-8") or '"order": ' in raw_zarray.decode("utf-8")
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
    loaded, _meta = load_shard_arrays(path)
    np.testing.assert_array_equal(loaded["wdl_target"], _expected_wdl(520, tail_fill=True))
    (path / "wdl_target" / "1").unlink()
    calls = _forbid_fill_decode(monkeypatch)
    with pytest.raises(ValueError, match="stored chunk 'wdl_target/1'"):
        load_shard_arrays(path)
    assert calls == {"getitem": 0, "decode": 0}


def test_missing_chunk_at_publish_preserves_the_existing_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "replay"
    path = _write(directory, n=520, tail_fill=False)
    before = _snapshot(path)
    real = DirectoryStore.__setitem__

    def skip(self: DirectoryStore, key: str, value: object) -> None:
        if key == "wdl_target/1":
            return
        real(self, key, value)

    monkeypatch.setattr(DirectoryStore, "__setitem__", skip)
    with pytest.raises(ValueError, match="refusing to publish"):
        _write(directory, n=520, tail_fill=False)
    assert _snapshot(path) == before
    assert list(directory.glob("._tmp_*")) == []


def test_oversize_manifest_is_not_stored_and_preserves_the_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "replay"
    path = _write(directory, n=8, tail_fill=False)
    before = _snapshot(path)
    monkeypatch.setattr(
        directory_seal,
        "_canonical_bytes",
        lambda _doc: b"y" * (MAX_SEAL_BYTES + 1),
    )
    with pytest.raises(ValueError, match=f"cap is {MAX_SEAL_BYTES}; refusing to publish"):
        _write(directory, n=8, tail_fill=False)
    assert _snapshot(path) == before
    assert list(directory.glob("._tmp_*")) == []
    assert MAX_SEAL_BYTES == 4_000_000


def test_oversize_manifest_publishes_nothing_when_the_path_is_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "replay"
    directory.mkdir()
    monkeypatch.setattr(
        directory_seal,
        "_canonical_bytes",
        lambda _doc: b"y" * (MAX_SEAL_BYTES + 1),
    )
    samples = [_sample(row, tail_fill=False) for row in range(2)]
    path = directory / "shard_000000.zarr"
    with pytest.raises(ValueError, match="refusing to publish"):
        save_local_shard_arrays(
            path,
            arrs=samples_to_arrays(samples),
            meta=ShardMeta(positions=2, policy_encoding="lc0_1858", policy_size=COMPACT_POLICY_SIZE),
        )
    assert not path.exists()
    assert list(directory.glob("._tmp_*")) == []


def test_writer_refuses_the_real_cap_before_assigning_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "group.zarr"
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset("x", data=np.arange(4, dtype=np.float32), overwrite=True)
    monkeypatch.setattr(
        directory_seal,
        "_canonical_bytes",
        lambda _doc: b"y" * (MAX_SEAL_BYTES + 1),
    )
    with pytest.raises(ValueError, match=f"cap is {MAX_SEAL_BYTES}; refusing to publish"):
        write_directory_producer_seal(group)
    assert SEAL_FILENAME not in group.store
    assert not (path / SEAL_FILENAME).exists()


def _uint8_two_chunk_group() -> tuple[MemoryStore, Any]:
    store = MemoryStore()
    group = zarr.group(store=store)
    group.create_dataset(
        "v",
        data=np.zeros(2, dtype=np.uint8),
        chunks=(1,),
        overwrite=True,
    )
    return store, group


def _store_key_count(path: Path) -> int:
    group = zarr.open_group(str(path), mode="r")
    return len(directory_seal._store_keys(group.store))


def test_writer_and_reader_share_the_final_store_key_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sol shape: one uint8 array, two chunks, four keys before the seal.

    The manifest and the array ``.zattrs`` make six keys. A cap of four or
    five must refuse before those keys are stored. A cap of six is the
    inclusive boundary for both the writer and the reader.
    """
    store, group = _uint8_two_chunk_group()
    initial = sorted(store.keys())
    assert initial == [".zgroup", "v/.zarray", "v/0", "v/1"]

    for cap in (4, 5):
        monkeypatch.setattr(directory_seal, "MAX_STORE_KEYS", cap)
        with pytest.raises(ValueError, match=f"more than {cap} keys"):
            write_directory_producer_seal(group)
        assert sorted(store.keys()) == initial
        assert SEAL_FILENAME not in store

    monkeypatch.setattr(directory_seal, "MAX_STORE_KEYS", 6)
    write_directory_producer_seal(group)
    final = sorted(store.keys())
    assert final == [
        ".zgroup",
        SEAL_FILENAME,
        "v/.zarray",
        "v/.zattrs",
        "v/0",
        "v/1",
    ]
    verify_directory_producer_seal(group)

    monkeypatch.setattr(directory_seal, "MAX_STORE_KEYS", 5)
    with pytest.raises(ValueError, match="more than 5 keys"):
        verify_directory_producer_seal(group)
    monkeypatch.setattr(directory_seal, "MAX_STORE_KEYS", 6)
    verify_directory_producer_seal(group)


def test_existing_array_zattrs_counts_once_toward_the_final_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, group = _uint8_two_chunk_group()
    group["v"].attrs["keep"] = 1
    initial = sorted(store.keys())
    assert initial == [".zgroup", "v/.zarray", "v/.zattrs", "v/0", "v/1"]

    monkeypatch.setattr(directory_seal, "MAX_STORE_KEYS", 5)
    with pytest.raises(ValueError, match="more than 5 keys"):
        write_directory_producer_seal(group)
    assert sorted(store.keys()) == initial

    monkeypatch.setattr(directory_seal, "MAX_STORE_KEYS", 6)
    write_directory_producer_seal(group)
    assert sorted(store.keys()) == sorted([*initial, SEAL_FILENAME])
    verify_directory_producer_seal(group)


def test_final_key_cap_preserves_destination_and_matches_the_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "replay"
    path = _write(directory, n=8, tail_fill=False)
    before = _snapshot(path)
    final = _store_key_count(path)
    assert final > 4

    monkeypatch.setattr(directory_seal, "MAX_STORE_KEYS", final - 1)
    with pytest.raises(ValueError, match=f"more than {final - 1} keys"):
        _write(directory, n=8, tail_fill=False)
    assert _snapshot(path) == before
    assert list(directory.glob("._tmp_*")) == []
    with pytest.raises(ValueError, match=f"more than {final - 1} keys"):
        load_shard_arrays(path)

    monkeypatch.setattr(directory_seal, "MAX_STORE_KEYS", final)
    loaded, _meta = load_shard_arrays(path)
    assert int(loaded["x"].shape[0]) == 8
    _write(directory, n=8, tail_fill=False)
    assert _store_key_count(path) == final
    reloaded, _meta = load_shard_arrays(path)
    assert int(reloaded["x"].shape[0]) == 8
    assert list(directory.glob("._tmp_*")) == []


def test_actual_key_count_blocks_publish_when_the_projection_undercounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "replay"
    path = _write(directory, n=8, tail_fill=False)
    before = _snapshot(path)
    final = _store_key_count(path)
    monkeypatch.setattr(directory_seal, "MAX_STORE_KEYS", final - 1)
    monkeypatch.setattr(
        directory_seal,
        "_projected_store_key_count",
        lambda _listed, _group, _names: final - 1,
    )
    with pytest.raises(ValueError, match=f"more than {final - 1} keys"):
        _write(directory, n=8, tail_fill=False)
    assert _snapshot(path) == before
    assert list(directory.glob("._tmp_*")) == []


def test_reader_rejects_a_manifest_over_the_same_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(tmp_path / "replay", n=8, tail_fill=False)
    (path / SEAL_FILENAME).write_bytes(b"x" * (MAX_SEAL_BYTES + 1))
    calls = _forbid_fill_decode(monkeypatch)
    with pytest.raises(ValueError, match=f"cap is {MAX_SEAL_BYTES}; refusing fill decode"):
        load_shard_arrays(path)
    assert calls == {"getitem": 0, "decode": 0}


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("order", lambda doc: doc.__setitem__("order", "F")),
        ("dtype", lambda doc: doc.__setitem__("dtype", "<i4")),
        ("fill_value", lambda doc: doc.__setitem__("fill_value", 1.0)),
        ("shape", lambda doc: doc.__setitem__("shape", [int(doc["shape"][0]) + 1, *doc["shape"][1:]])),
        ("chunks", lambda doc: doc.__setitem__("chunks", [max(1, int(doc["chunks"][0]) - 1), *doc["chunks"][1:]])),
        ("compressor", lambda doc: doc["compressor"].__setitem__("clevel", int(doc["compressor"]["clevel"]) + 1)),
        ("filters", lambda doc: doc.__setitem__("filters", [])),
        (
            "dimension_separator",
            lambda doc: doc.__setitem__(
                "dimension_separator",
                "/" if doc.get("dimension_separator", ".") == "." else ".",
            ),
        ),
    ],
)
def test_zarray_field_edit_fails_before_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, mutate,
) -> None:
    path = _write(tmp_path / "replay", n=8, tail_fill=False)
    before = (path / "x" / ".zarray").read_bytes()
    _replace_zarray(path, "x", mutate)
    assert (path / "x" / ".zarray").read_bytes() != before, label
    calls = _forbid_fill_decode(monkeypatch)
    with pytest.raises(ValueError, match="declaration for 'x'"):
        load_shard_arrays(path, validate=False)
    assert calls == {"getitem": 0, "decode": 0}


def _window(directory: Path, capacity: int) -> DiskReplayBuffer:
    return DiskReplayBuffer(
        capacity,
        shard_dir=directory,
        rng=np.random.default_rng(0),
        read_only=False,
        shuffle_cap=16,
        shard_size=1000,
        refresh_shards=1,
        deterministic_refresh=True,
        input_planes=PLANES,
    )


def _rewrite_only_x_seal(path: Path, rows: int) -> None:
    """Point the manifest and x's attribute at a new x shape. Leave the rest."""
    _replace_zarray(path, "x", lambda doc: doc["shape"].__setitem__(0, rows))
    doc = _seal(path)
    arrays = doc["arrays"]
    assert isinstance(arrays, dict)
    spec = arrays["x"]
    assert isinstance(spec, dict)
    raw_x = (path / "x" / ".zarray").read_bytes()
    spec["zarray_sha256"] = hashlib.sha256(raw_x).hexdigest()
    payload = directory_seal._canonical_bytes(doc)
    (path / SEAL_FILENAME).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    group = zarr.open_group(str(path), mode="a")
    group["x"].attrs[SEAL_ATTR] = digest
    assert group["game_id"].attrs[SEAL_ATTR] != digest


def test_forged_sealed_shape_does_not_delete_the_honest_shard(tmp_path: Path) -> None:
    directory = tmp_path / "replay"
    honest = _write(directory, n=8, name="shard_000000.zarr")
    forged = _write(directory, n=8, name="shard_000001.zarr")
    _replace_zarray(forged, "x", lambda doc: doc["shape"].__setitem__(0, 100))
    assert shard_positions(honest) == 8
    assert shard_positions(forged) == 0
    buf = DiskReplayBuffer(
        20,
        shard_dir=directory,
        rng=np.random.default_rng(0),
        read_only=False,
        shuffle_cap=16,
        shard_size=1000,
        refresh_shards=1,
        deterministic_refresh=True,
        input_planes=PLANES,
    )
    try:
        assert honest.exists()
        assert forged.exists()
        assert buf._tracked_shard_positions() == 8
    finally:
        buf.close()
    stripped = _write(tmp_path / "stripped", n=8, name="shard_000000.zarr")
    shutil.rmtree(stripped / "priority")
    assert shard_positions(stripped) == 8


def test_partial_attribute_rewrite_does_not_inflate_window_deletion(
    tmp_path: Path,
) -> None:
    """A new digest on x alone is not a row count the window may trust.

    The manifest hash for x is rewritten to a 13-row shape and only x's seal
    attribute is updated. The other arrays still carry the old digest, which
    is the same mismatch a full load rejects. Capacity 20 must keep the honest
    8-row shard: 13 plus 8 would have deleted it.
    """
    directory = tmp_path / "replay"
    honest = _write(directory, n=8, name="shard_000000.zarr")
    forged = _write(directory, n=8, name="shard_000001.zarr")
    _rewrite_only_x_seal(forged, 13)
    assert shard_positions(honest) == 8
    assert shard_positions(forged) == 0
    with pytest.raises(ValueError, match="does not match"):
        load_shard_arrays(forged)
    buf = _window(directory, 20)
    try:
        assert honest.exists()
        assert forged.exists()
        assert buf._tracked_shard_positions() == 8
    finally:
        buf.close()


def test_window_count_is_the_shape_in_the_hashed_zarray_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file swapped after the hash read is not the count that was hashed.

    The scan records 8 from the bytes that matched the seal, then the file
    becomes shape 100. Capacity 20 therefore keeps both shards. A later read
    of that forged file counts as 0.
    """
    directory = tmp_path / "replay"
    honest = _write(directory, n=8, name="shard_000000.zarr")
    attacked = _write(directory, n=8, name="shard_000001.zarr")
    original = (attacked / "x" / ".zarray").read_bytes()
    forged_doc = json.loads(original)
    assert isinstance(forged_doc, dict)
    shape = forged_doc["shape"]
    assert isinstance(shape, list)
    shape[0] = 100
    forged = json.dumps(forged_doc).encode("utf-8")
    real_raw = directory_seal._raw_bytes
    attacked_path = attacked.resolve()

    def swap_after_hash(store: object, key: str, *, publish: bool) -> bytes:
        data = real_raw(store, key, publish=publish)
        store_path = getattr(store, "path", None)
        if (
            key == "x/.zarray"
            and isinstance(store_path, str)
            and Path(store_path).resolve() == attacked_path
        ):
            (attacked / "x" / ".zarray").write_bytes(forged)
        return data

    monkeypatch.setattr(directory_seal, "_raw_bytes", swap_after_hash)
    assert shard_positions(attacked) == 8
    assert json.loads((attacked / "x" / ".zarray").read_text(encoding="utf-8"))["shape"][0] == 100
    (attacked / "x" / ".zarray").write_bytes(original)
    buf = _window(directory, 20)
    try:
        assert honest.exists()
        assert attacked.exists()
        assert buf._tracked_shard_positions() == 16
    finally:
        buf.close()
    assert json.loads((attacked / "x" / ".zarray").read_text(encoding="utf-8"))["shape"][0] == 100
    monkeypatch.setattr(directory_seal, "_raw_bytes", real_raw)
    assert shard_positions(attacked) == 0
    assert shard_positions(honest) == 8
    assert honest.exists()
    assert attacked.exists()


def test_dangling_symlink_is_not_a_stored_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A listed name that decode will not open is a missing chunk."""
    path = _write(tmp_path / "replay", n=520, tail_fill=False)
    doc = _seal(path)
    raw_wdl = (path / "wdl_target" / ".zarray").read_bytes()
    assert directory_seal._grid_keys("wdl_target", raw_wdl) == doc["arrays"]["wdl_target"]["stored"]
    tail = path / "wdl_target" / "1"
    assert tail.is_file()
    tail.unlink()
    tail.symlink_to("missing-chunk")
    calls = _forbid_fill_decode(monkeypatch)
    with pytest.raises(ValueError, match="stored chunk 'wdl_target/1'"):
        load_shard_arrays(path)
    assert calls == {"getitem": 0, "decode": 0}


def test_dropped_sibling_declarations_do_not_inflate_the_window(tmp_path: Path) -> None:
    """Deleting sibling .zarray files does not make a rewritten x shape countable."""
    directory = tmp_path / "replay"
    honest = _write(directory, n=8, name="shard_000000.zarr")
    forged = _write(directory, n=8, name="shard_000001.zarr")
    raw_x = (forged / "x" / ".zarray").read_bytes()
    assert directory_seal._grid_keys("x", raw_x) == _seal(forged)["arrays"]["x"]["stored"]
    for child in forged.iterdir():
        if child.is_dir() and child.name != "x":
            zarray = child / ".zarray"
            if zarray.exists():
                zarray.unlink()
    _replace_zarray(forged, "x", lambda doc: doc["shape"].__setitem__(0, 13))
    seal = _seal(forged)
    arrays = seal["arrays"]
    assert isinstance(arrays, dict)
    spec = arrays["x"]
    assert isinstance(spec, dict)
    spec["zarray_sha256"] = hashlib.sha256((forged / "x" / ".zarray").read_bytes()).hexdigest()
    payload = directory_seal._canonical_bytes(seal)
    (forged / SEAL_FILENAME).write_bytes(payload)
    group = zarr.open_group(str(forged), mode="a")
    assert list(group.array_keys()) == ["x"]
    group["x"].attrs[SEAL_ATTR] = hashlib.sha256(payload).hexdigest()
    assert shard_positions(honest) == 8
    assert shard_positions(forged) == 0
    with pytest.raises(ValueError, match="array set does not match"):
        load_shard_arrays(forged)
    buf = _window(directory, 20)
    try:
        assert honest.exists()
        assert forged.exists()
        assert buf._tracked_shard_positions() == 8
    finally:
        buf.close()


def test_stale_chunk_grid_does_not_inflate_the_window(tmp_path: Path) -> None:
    """Updating every seal attribute still leaves a stale inventory at 0 rows."""
    directory = tmp_path / "replay"
    honest = _write(directory, n=8, name="shard_000000.zarr")
    forged = _write(directory, n=8, name="shard_000001.zarr")
    _replace_zarray(forged, "x", lambda doc: doc["shape"].__setitem__(0, 100))
    seal = _seal(forged)
    arrays = seal["arrays"]
    assert isinstance(arrays, dict)
    spec = arrays["x"]
    assert isinstance(spec, dict)
    assert spec["stored"] == ["x/0.0.0.0"]
    spec["zarray_sha256"] = hashlib.sha256((forged / "x" / ".zarray").read_bytes()).hexdigest()
    payload = directory_seal._canonical_bytes(seal)
    (forged / SEAL_FILENAME).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    group = zarr.open_group(str(forged), mode="a")
    for name in list(group.array_keys()):
        group[name].attrs[SEAL_ATTR] = digest
    assert shard_positions(honest) == 8
    assert shard_positions(forged) == 0
    with pytest.raises(ValueError, match="chunk grid"):
        load_shard_arrays(forged)
    buf = _window(directory, 20)
    try:
        assert honest.exists()
        assert forged.exists()
        assert buf._tracked_shard_positions() == 8
    finally:
        buf.close()


def _retarget_x_hash(path: Path) -> bytes:
    seal = _seal(path)
    arrays = seal["arrays"]
    assert isinstance(arrays, dict)
    spec = arrays["x"]
    assert isinstance(spec, dict)
    raw_x = (path / "x" / ".zarray").read_bytes()
    spec["zarray_sha256"] = hashlib.sha256(raw_x).hexdigest()
    payload = directory_seal._canonical_bytes(seal)
    (path / SEAL_FILENAME).write_bytes(payload)
    return payload


def test_same_grid_shape_edit_does_not_inflate_the_window(tmp_path: Path) -> None:
    """A longer shape that keeps the chunk keys is not a larger window."""
    directory = tmp_path / "replay"
    honest = _write(directory, n=8, name="shard_000000.zarr")
    forged = _write(directory, n=520, name="shard_000001.zarr")
    _replace_zarray(forged, "x", lambda doc: doc["shape"].__setitem__(0, 1024))
    raw_x = (forged / "x" / ".zarray").read_bytes()
    assert directory_seal._grid_keys("x", raw_x) == _seal(forged)["arrays"]["x"]["stored"]
    payload = _retarget_x_hash(forged)
    digest = hashlib.sha256(payload).hexdigest()
    group = zarr.open_group(str(forged), mode="a")
    for name in list(group.array_keys()):
        group[name].attrs[SEAL_ATTR] = digest
    assert shard_positions(honest) == 8
    assert shard_positions(forged) == 0
    with pytest.raises(ValueError, match="policy_target N mismatch"):
        load_shard_arrays(forged)
    buf = _window(directory, 1024)
    try:
        assert honest.exists()
        assert forged.exists()
        assert buf._tracked_shard_positions() == 8
    finally:
        buf.close()


def test_same_key_rewrite_without_sibling_declarations_does_not_inflate(
    tmp_path: Path,
) -> None:
    """Removing sibling .zarray files still leaves their seal attributes behind."""
    directory = tmp_path / "replay"
    honest = _write(directory, n=8, name="shard_000000.zarr")
    forged = _write(directory, n=8, name="shard_000001.zarr")
    for child in forged.iterdir():
        if child.is_dir() and child.name != "x":
            declaration = child / ".zarray"
            if declaration.exists():
                declaration.unlink()

    def _one_chunk(doc: dict[str, Any]) -> None:
        doc["shape"][0] = 100
        doc["chunks"][0] = 100

    _replace_zarray(forged, "x", _one_chunk)
    raw_x = (forged / "x" / ".zarray").read_bytes()
    assert directory_seal._grid_keys("x", raw_x) == ["x/0.0.0.0"]
    payload = _retarget_x_hash(forged)
    group = zarr.open_group(str(forged), mode="a")
    assert list(group.array_keys()) == ["x"]
    group["x"].attrs[SEAL_ATTR] = hashlib.sha256(payload).hexdigest()
    assert shard_positions(honest) == 8
    assert shard_positions(forged) == 0
    with pytest.raises(ValueError, match="array set does not match"):
        load_shard_arrays(forged)
    buf = _window(directory, 100)
    try:
        assert honest.exists()
        assert forged.exists()
        assert buf._tracked_shard_positions() == 8
    finally:
        buf.close()


def test_restamped_leftover_attrs_do_not_inflate_the_window(tmp_path: Path) -> None:
    """A leftover .zattrs restamped with the new digest is not a removed array."""
    directory = tmp_path / "replay"
    honest = _write(directory, n=8, name="shard_000000.zarr")
    forged = _write(directory, n=8, name="shard_000001.zarr")
    for child in forged.iterdir():
        if child.is_dir() and child.name != "x":
            declaration = child / ".zarray"
            if declaration.exists():
                declaration.unlink()

    def _one_chunk(doc: dict[str, Any]) -> None:
        doc["shape"][0] = 13
        doc["chunks"][0] = 13

    _replace_zarray(forged, "x", _one_chunk)
    raw_x = (forged / "x" / ".zarray").read_bytes()
    assert directory_seal._grid_keys("x", raw_x) == ["x/0.0.0.0"]
    payload = _retarget_x_hash(forged)
    digest = hashlib.sha256(payload).hexdigest()
    for child in forged.iterdir():
        attrs = child / ".zattrs"
        if not child.is_dir() or not attrs.is_file():
            continue
        doc = json.loads(attrs.read_text(encoding="utf-8"))
        assert isinstance(doc, dict)
        doc[SEAL_ATTR] = digest
        attrs.write_text(json.dumps(doc), encoding="utf-8")
    assert shard_positions(honest) == 8
    assert shard_positions(forged) == 0
    with pytest.raises(ValueError, match="array set does not match"):
        load_shard_arrays(forged)
    buf = _window(directory, 20)
    try:
        assert honest.exists()
        assert forged.exists()
        assert buf._tracked_shard_positions() == 8
    finally:
        buf.close()


def test_collapsed_float_chunk_grid_does_not_inflate_the_window(
    tmp_path: Path,
) -> None:
    """A shape past the float mantissa must not keep a one-key inventory."""
    directory = tmp_path / "replay"
    honest = _write(directory, n=8, name="shard_000000.zarr")
    forged = _write(directory, n=8, name="shard_000001.zarr")
    stored_before = _seal(forged)["arrays"]["x"]["stored"]
    assert stored_before == ["x/0.0.0.0"]
    huge = 2**53

    def _collapse(doc: dict[str, Any]) -> None:
        doc["shape"][0] = huge + 1
        doc["chunks"][0] = huge

    for child in forged.iterdir():
        if child.is_dir() and (child / ".zarray").is_file():
            _replace_zarray(forged, child.name, _collapse)
    seal = _seal(forged)
    arrays = seal["arrays"]
    assert isinstance(arrays, dict)
    for name, spec in arrays.items():
        assert isinstance(spec, dict)
        raw = (forged / name / ".zarray").read_bytes()
        spec["zarray_sha256"] = hashlib.sha256(raw).hexdigest()
    payload = directory_seal._canonical_bytes(seal)
    (forged / SEAL_FILENAME).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    raw_x = (forged / "x" / ".zarray").read_bytes()
    edited = json.loads(raw_x)
    assert math.ceil(edited["shape"][0] / edited["chunks"][0]) == 1
    assert (edited["shape"][0] + edited["chunks"][0] - 1) // edited["chunks"][0] == 2
    assert _seal(forged)["arrays"]["x"]["stored"] == stored_before
    for child in forged.iterdir():
        attrs = child / ".zattrs"
        if not child.is_dir() or not attrs.is_file():
            continue
        doc = json.loads(attrs.read_text(encoding="utf-8"))
        assert isinstance(doc, dict)
        doc[SEAL_ATTR] = digest
        attrs.write_text(json.dumps(doc), encoding="utf-8")
    group = zarr.open_group(str(forged), mode="r")
    cdata = group["x"].cdata_shape
    assert isinstance(cdata, tuple)
    assert cdata[0] == 1
    with pytest.raises(ValueError, match="not the float grid"):
        directory_seal._grid_keys("x", raw_x)
    assert shard_positions(honest) == 8
    assert shard_positions(forged) == 0
    with pytest.raises(ValueError, match="not the float grid"):
        verify_directory_producer_seal(zarr.open_group(str(forged), mode="r"))
    buf = _window(directory, 20)
    try:
        assert honest.exists()
        assert forged.exists()
        assert buf._tracked_shard_positions() == 8
    finally:
        buf.close()


def test_restored_zarray_must_match_the_metadata_cached_at_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open caches order F. The sealed bytes are what get installed, even if the file is forged again after the hash."""
    samples = []
    for row in range(2):
        grid = np.zeros((PLANES, 8, 8), dtype=np.float32)
        grid[0, 0, 0] = 1.0
        grid[1, 7, 7] = 2.0
        policy = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float32)
        policy[0] = 1.0
        legal = np.zeros_like(policy, dtype=np.uint8)
        legal[0] = 1
        item = ReplaySample(
            x=grid,
            policy_target=policy,
            legal_mask=legal,
            wdl_target=row % 3,
            priority=1.0,
            has_policy=True,
            game_id=row,
            ply_index=row,
        )
        item.input_history_encoding = "legacy"
        item.history_rep_fix = False
        samples.append(item)
    path = tmp_path / "shard.zarr"
    save_local_shard_arrays(
        path,
        arrs=samples_to_arrays(samples),
        meta=ShardMeta(
            positions=2,
            policy_encoding="lc0_1858",
            policy_size=COMPACT_POLICY_SIZE,
            input_history_encoding="legacy",
            history_rep_fix=False,
        ),
    )
    original = (path / "x" / ".zarray").read_bytes()
    forged_doc = json.loads(original)
    forged_doc["order"] = "F"
    forged = json.dumps(forged_doc).encode()
    (path / "x" / ".zarray").write_bytes(forged)
    from chess_anti_engine.replay import shard as shard_mod

    real_guard = shard_mod._reject_unsafe_shard_codecs
    restored = {"done": False}

    def restore_after_open(proxies: dict[str, Any]) -> None:
        if not restored["done"]:
            (path / "x" / ".zarray").write_bytes(original)
            restored["done"] = True
        real_guard(proxies)

    real_raw = directory_seal._raw_bytes

    def forge_after_hash(store: object, key: str, *, publish: bool) -> bytes:
        data = real_raw(store, key, publish=publish)
        if key == "x/.zarray":
            (path / "x" / ".zarray").write_bytes(forged)
        return data

    monkeypatch.setattr(shard_mod, "_reject_unsafe_shard_codecs", restore_after_open)
    monkeypatch.setattr(directory_seal, "_raw_bytes", forge_after_hash)
    loaded, _meta = load_shard_arrays(path, validate=True)
    assert float(loaded["x"][0, 1, 7, 7]) == 2.0
    assert json.loads((path / "x" / ".zarray").read_text(encoding="utf-8"))["order"] == "F"


def test_verify_reads_the_manifest_and_zarray_not_chunk_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(tmp_path / "replay", n=520, tail_fill=False)
    seen: list[str] = []
    real = directory_seal._raw_bytes

    def wrapped(store: object, key: str, *, publish: bool) -> bytes:
        seen.append(key)
        return real(store, key, publish=publish)

    monkeypatch.setattr(directory_seal, "_raw_bytes", wrapped)
    loaded, _meta = load_shard_arrays(path, lazy=True, validate=False)
    assert int(loaded["x"].shape[0]) == 520
    assert SEAL_FILENAME in seen
    assert any(key.endswith(".zarray") for key in seen)
    assert all(key == SEAL_FILENAME or key.endswith(".zarray") for key in seen)


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


def test_unsealed_sparse_omission_still_fill_decodes(tmp_path: Path) -> None:
    """A store that never had this seal keeps Zarr fill-elision.

    ``write_empty_chunks=False`` is the existing unsealed behavior. The omitted
    fill chunk is not required, and the nonzero ``x`` tail is still the stored
    chunk. This contract does not validate that store.
    """
    from numcodecs import Blosc

    path = tmp_path / "plain.zarr"
    samples = [_sample(row, tail_fill=True) for row in range(520)]
    group = zarr.open_group(str(path), mode="w")
    compressor = Blosc(cname="zstd", clevel=2, shuffle=Blosc.BITSHUFFLE)
    for name, value in samples_to_arrays(samples).items():
        if str(name).startswith("_"):
            continue
        arr = np.asarray(value)
        lead = min(max(1, int(arr.shape[0])), 512)
        chunks = (lead,) if arr.ndim == 1 else (lead, *arr.shape[1:])
        group.create_dataset(
            name,
            data=arr,
            chunks=chunks,
            compressor=compressor,
            overwrite=True,
            write_empty_chunks=False,
        )
    assert not (path / "wdl_target" / "1").exists()
    assert (path / "x" / "1.0.0.0").is_file()
    assert not (path / SEAL_FILENAME).exists()
    loaded, _meta = load_shard_arrays(path, validate=False)
    np.testing.assert_array_equal(
        np.asarray(loaded["wdl_target"]), _expected_wdl(520, tail_fill=True),
    )
    assert np.any(np.asarray(loaded["x"][TAIL:]) != 0)


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
    """Overlay loads skip this seal. A deleted tail still fill-decodes there.

    This test records that limitation. It does not claim the overlay tree is
    protected. Ordinary loads of the same directory still refuse the hole.
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
