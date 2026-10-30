"""Owned numeric ZIP fixtures; no Ceres, Torch, native or engine imports.

The explicit AST seam executes the exact named reader definitions, not copied
reader logic. It omits unrelated imports and game/encoding machinery. Hostile
metadata tests install recording-and-raising codec sentinels, never a codec or
a serialized payload. Run only in an authorized disposable test workspace.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import io
import json
import os
import stat
import zipfile
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numcodecs
import numpy as np
import pytest
import zarr

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "chess_anti_engine/source/ceres_saved_game.py"
HELPER = ROOT / "chess_anti_engine/replay/codec_safety.py"
STRUCTURED_OBJECT_WITNESS = {
    "dtype": [["payload", "|O"]], "filters": [{"id": "pickle"}], "fill_value": "AA==",
}


def _extract(path: Path, names: set[str], namespace: dict[str, Any]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    chosen: list[ast.stmt] = [ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0)]
    found: set[str] = set()
    for node in tree.body:
        defined: set[str] = set()
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            defined = {node.name}
        elif isinstance(node, ast.Assign):
            defined = {target.id for target in node.targets if isinstance(target, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined = {node.target.id}
        if defined & names:
            assert defined <= names, f"unexpected shared assignment in {path}"
            chosen.append(node)
            found |= defined
    assert found == names, f"inert seam must be audited for source drift: {names - found}"
    module = ast.fix_missing_locations(ast.Module(body=chosen, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def _reader() -> SimpleNamespace:
    namespace: dict[str, Any] = {
        "Any": Any, "Iterator": Iterator, "MutableMapping": MutableMapping,
        "MappingProxyType": MappingProxyType, "Path": Path, "cast": cast,
        "gzip": gzip, "hashlib": hashlib, "io": io, "json": json, "os": os,
        "stat": stat, "zipfile": zipfile, "np": np, "zarr": zarr,
        "SavedCeresGame": lambda arrays, rows, game, model, source: SimpleNamespace(
            arrays=arrays, rows=rows, game=game, model_sha256=model,
            source_archive_sha256=source),
        "_check_game": lambda *_args, **_kwargs: None,
    }
    _extract(HELPER, {"_SAFE_SHARD_DTYPE_KINDS", "_SAFE_SHARD_COMPRESSOR_IDS",
                      "_reject_unsafe_shard_codecs"}, namespace)
    _extract(SOURCE, {
        "ARRAY_NAMES", "SCHEMA", "SOURCE_SCHEMA", "MAX_ROWS", "MAX_ARCHIVE_BYTES",
        "MAX_METADATA_BYTES", "MAX_ZIP_MEMBERS", "_need", "_sha", "_canonical",
        "_file_sha", "_zip_json", "_ApprovedSavedGameStore",
        "_approved_saved_game_metadata", "_arrays_from_zip", "load_saved_game",
        "_read_written",
    }, namespace)
    return SimpleNamespace(**namespace)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "x": np.arange(4 * 175 * 8 * 8, dtype=np.int32).reshape(4, 175, 8, 8)
               .astype(np.float16) / np.float16(1000),
        "value": np.arange(12, dtype=np.float16).reshape(4, 3),
        "value2": np.arange(12, dtype=np.float16).reshape(4, 3) + np.float16(1),
        "legal_offsets": np.array([0, 2, 5, 7, 10], dtype=np.uint32),
        "legal_compact": np.arange(10, dtype=np.uint16),
        "legal_leela": np.arange(10, dtype=np.uint16) + np.uint16(10),
        "legal_logits": np.arange(10, dtype=np.float16),
    }


def _archive(
    tmp_path: Path, *, member: str = "x", changes: dict[str, Any] | None = None,
    missing: bool = False, subgroup: bool = False, compressor: Any = None,
) -> tuple[Path, dict[str, np.ndarray]]:
    arrays = _arrays()
    store = zarr.storage.MemoryStore()
    group = zarr.group(store=store)
    for name, array in arrays.items():
        group.create_dataset(name, data=array, chunks=(1, *array.shape[1:]),
                             compressor=compressor, filters=None)
    key = f"{member}/.zarray"
    if changes is not None:
        metadata = json.loads(store[key])
        metadata.update(changes)
        store[key] = json.dumps(metadata).encode()
    if missing:
        del store[key]
    if subgroup:
        store[f"{member}/.zgroup"] = b'{"zarr_format":2}'
    path = tmp_path / "owned-numeric.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in store:
            archive.writestr(name, store[name])
    store.close()
    return path, arrays


def _no_decode(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[Any]]:
    codec_calls: list[Any] = []
    slice_calls: list[Any] = []

    def codec(config: Any) -> Any:
        codec_calls.append(config)
        raise AssertionError("codec sentinel must never construct or return a codec")

    def sliced(_self: Any, selection: Any) -> Any:
        slice_calls.append(selection)
        raise AssertionError("unsafe metadata reached array slicing")

    monkeypatch.setattr(numcodecs, "get_codec", codec)
    monkeypatch.setattr(zarr.core, "get_codec", codec)
    monkeypatch.setattr(zarr.Array, "__getitem__", sliced)
    return codec_calls, slice_calls


@pytest.mark.parametrize(("member", "changes", "message"), [
    ("x", STRUCTURED_OBJECT_WITNESS, "raw dtype must be primitive numeric"),
    ("x", {"dtype": "|O", "filters": [{"id": "pickle"}]}, "raw dtype differs"),
    ("legal_logits", {"filters": [{"id": "pickle"}]}, "raw filters"),
    ("legal_compact", {"compressor": {"id": "pickle"}}, "raw compressor"),
    ("value2", {"compressor": {"id": "unregistered-future-codec"}}, "raw compressor"),
    ("value", {"compressor": []}, "raw compressor"),
    ("legal_offsets", {"dtype": "<i4"}, "raw dtype differs"),
    ("legal_leela", {"dtype": None}, "raw dtype must be primitive numeric"),
    ("x", {"dtype": ">f2"}, "raw dtype differs"),
    ("x", {"filters": {}}, "raw filters"),
])
def test_raw_metadata_refuses_before_any_codec_or_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    member: str, changes: dict[str, Any], message: str,
) -> None:
    reader = _reader()
    path, _ = _archive(tmp_path, member=member, changes=changes)
    codecs, slices = _no_decode(monkeypatch)
    with pytest.raises(ValueError, match=message):
        reader._arrays_from_zip(path, 1, 3)
    assert codecs == []
    assert slices == []


def test_structured_witness_reaches_codec_sentinel_without_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discriminate the witness without ever returning a codec or decoding fill."""
    path, _ = _archive(tmp_path, changes=STRUCTURED_OBJECT_WITNESS)
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("x/.zarray")
    codecs, slices = _no_decode(monkeypatch)
    with pytest.raises(zarr.errors.MetadataError, match="error decoding metadata") as caught:
        zarr.meta.Metadata2.decode_array_metadata(raw)
    assert codecs == [{"id": "pickle"}]
    assert isinstance(caught.value.__cause__, AssertionError)
    assert "never construct or return a codec" in str(caught.value.__cause__)
    assert slices == []


@pytest.mark.parametrize(("missing", "subgroup", "message"), [
    (True, False, "unreadable raw metadata"),
    (True, True, "unreadable raw metadata"),
    (False, True, "not a v2 array"),
])
def test_absent_or_subgroup_declaration_refuses_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    missing: bool, subgroup: bool, message: str,
) -> None:
    reader = _reader()
    path, _ = _archive(tmp_path, member="legal_logits", missing=missing, subgroup=subgroup)
    codecs, slices = _no_decode(monkeypatch)
    with pytest.raises(ValueError, match=message):
        reader._arrays_from_zip(path, 1, 3)
    assert codecs == []
    assert slices == []


@pytest.mark.parametrize("compressed", [False, True])
def test_numeric_partial_read_uses_guarded_proxies_and_rebases_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compressed: bool,
) -> None:
    reader = _reader()
    compressor = numcodecs.Blosc(cname="zstd", clevel=2, shuffle=numcodecs.Blosc.BITSHUFFLE)
    path, arrays = _archive(tmp_path, compressor=compressor if compressed else None)
    guarded: list[dict[str, Any]] = []
    decoded: list[Any] = []
    guard = reader._reject_unsafe_shard_codecs
    sliced = zarr.Array.__getitem__

    def check(proxies: dict[str, Any]) -> None:
        guard(proxies)
        guarded.append(proxies)

    def track(array: Any, selection: Any) -> Any:
        assert len(guarded) == 1
        assert array is guarded[0][array.name.lstrip("/")]
        decoded.append((array.name, selection))
        return sliced(array, selection)

    monkeypatch.setitem(reader._arrays_from_zip.__globals__, "_reject_unsafe_shard_codecs", check)
    monkeypatch.setattr(zarr.Array, "__getitem__", track)
    actual = reader._arrays_from_zip(path, 1, 3)
    assert len(decoded) == 7
    for name in ("x", "value", "value2"):
        np.testing.assert_array_equal(actual[name], arrays[name][1:3])
    np.testing.assert_array_equal(actual["legal_offsets"], [0, 3, 5])
    for name in ("legal_compact", "legal_leela", "legal_logits"):
        np.testing.assert_array_equal(actual[name], arrays[name][2:7])


@pytest.mark.parametrize("member", ["x", "value", "value2", "legal_offsets",
                                    "legal_compact", "legal_leela", "legal_logits"])
@pytest.mark.parametrize("mutation", ["change", "delete", "insert"])
def test_approved_metadata_reused_or_refused_if_source_mutates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member: str, mutation: str,
) -> None:
    reader = _reader()
    path, arrays = _archive(tmp_path)
    zip_store = zarr.ZipStore
    reads: dict[str, int] = {}
    opened: list[Any] = []
    changed: list[Any] = []
    codec_calls: list[Any] = []
    slice_calls: list[Any] = []
    sliced = zarr.Array.__getitem__

    class ChangingStore(MutableMapping[str, bytes]):
        def __init__(self, filename: str, mode: str) -> None:
            self.store = zip_store(filename, mode=mode)
            self.replacements: dict[str, bytes] = {}
            self.deleted: set[str] = set()
            opened.append(self.store)

        def __getitem__(self, key: str) -> bytes:
            reads[key] = reads.get(key, 0) + 1
            if key in self.deleted:
                raise KeyError(key)
            if key in self.replacements:
                return self.replacements[key]
            return self.store[key]

        def __iter__(self) -> Iterator[str]:
            return iter((set(self.store) - self.deleted) | self.replacements.keys())

        def __len__(self) -> int:
            return sum(1 for _ in self)

        def __contains__(self, key: object) -> bool:
            return key not in self.deleted and (key in self.replacements or key in self.store)

        def __setitem__(self, _key: str, _value: bytes) -> None:
            raise AssertionError("unexpected write")

        def __delitem__(self, _key: str) -> None:
            raise AssertionError("unexpected delete")

        def close(self) -> None:
            self.store.close()

    preflight = reader._approved_saved_game_metadata

    def change_after_validation(store: ChangingStore) -> dict[str, bytes]:
        approved = preflight(store)
        key = f"{member}/.zarray"
        doc = json.loads(approved[key])
        doc.update(dtype=[["payload", "|O"]], filters=[{"id": "pickle"}], fill_value="AA==")
        if mutation == "change":
            store.replacements[key] = json.dumps(doc).encode()
            assert store.replacements[key] != approved[key]
        elif mutation == "delete":
            store.deleted.add(key)
        else:
            store.replacements["unexpected/.zarray"] = json.dumps(doc).encode()
        changed.append(store)
        return approved

    def codec(config: Any) -> Any:
        codec_calls.append(config)
        raise AssertionError("mutation must not construct or return a codec")

    def track(array: Any, selection: Any) -> Any:
        slice_calls.append(selection)
        return sliced(array, selection)

    monkeypatch.setattr(zarr, "ZipStore", ChangingStore)
    monkeypatch.setattr(numcodecs, "get_codec", codec)
    monkeypatch.setattr(zarr.core, "get_codec", codec)
    monkeypatch.setattr(zarr.Array, "__getitem__", track)
    monkeypatch.setitem(reader._arrays_from_zip.__globals__, "_approved_saved_game_metadata",
                        change_after_validation)
    if mutation == "change":
        actual = reader._arrays_from_zip(path, 1, 3)
        np.testing.assert_array_equal(actual["x"], arrays["x"][1:3])
        assert len(slice_calls) == 7
    else:
        with pytest.raises(ValueError, match="archive array names differ"):
            reader._arrays_from_zip(path, 1, 3)
        assert slice_calls == []
    assert all(reads[f"{name}/.zarray"] == 1 for name in reader.ARRAY_NAMES)
    assert len(changed) == 1
    assert codec_calls == []
    assert opened[0].zf.fp is None


@pytest.mark.parametrize("compressor_id", [None, "blosc", "zstd", "lz4", "gzip",
                                           "zlib", "bz2", "lzma"])
def test_preflight_accepts_only_explicit_byte_compressor_family(
    tmp_path: Path, compressor_id: str | None,
) -> None:
    reader = _reader()
    compressor = None if compressor_id is None else {"id": compressor_id}
    path, _ = _archive(tmp_path, changes={"compressor": compressor, "filters": []})
    with zarr.ZipStore(str(path), mode="r") as store:
        approved = reader._approved_saved_game_metadata(store)
    assert json.loads(approved["x/.zarray"])["compressor"] == compressor


@pytest.mark.parametrize("failure", ["preflight", "bounds", "slice"])
def test_zip_store_closes_on_each_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    reader = _reader()
    path, _ = _archive(tmp_path, changes={"filters": [{"id": "pickle"}]}
                       if failure == "preflight" else None)
    opened: list[Any] = []
    zip_store = zarr.ZipStore

    def tracked(filename: str, mode: str) -> Any:
        store = zip_store(filename, mode=mode)
        opened.append(store)
        return store

    monkeypatch.setattr(zarr, "ZipStore", tracked)
    if failure == "slice":
        def refuse_slice(_self: Any, _selection: Any) -> Any:
            raise ValueError("owned slice failure")
        monkeypatch.setattr(zarr.Array, "__getitem__", refuse_slice)
    message = {"preflight": "raw filters", "bounds": "stored row shape/count",
               "slice": "owned slice failure"}[failure]
    with pytest.raises(ValueError, match=message):
        reader._arrays_from_zip(path, 1, 9 if failure == "bounds" else 3)
    assert len(opened) == 1
    assert opened[0].zf.fp is None


def test_snapshot_is_read_only_and_does_not_alias_approval_dict(tmp_path: Path) -> None:
    reader = _reader()
    path, _ = _archive(tmp_path)
    with zarr.ZipStore(str(path), mode="r") as store:
        approved = reader._approved_saved_game_metadata(store)
        raw = approved["x/.zarray"]
        view = reader._ApprovedSavedGameStore(store, approved)
        approved["x/.zarray"] = b"unapproved replacement"
        assert view["x/.zarray"] == raw
        assert view.get("x/.zarray") == raw
        with pytest.raises(TypeError, match="read-only"):
            view["x/.zarray"] = b"replacement"
        with pytest.raises(TypeError, match="read-only"):
            del view["x/.zarray"]


def test_source_reader_preserves_hash_check_before_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = _reader()
    calls: list[Any] = []
    namespace = reader.load_saved_game.__globals__
    monkeypatch.setitem(namespace, "_file_sha", lambda _path: "different")

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        raise AssertionError("hash refusal must precede metadata and arrays")

    monkeypatch.setitem(namespace, "_zip_json", forbidden)
    monkeypatch.setitem(namespace, "_arrays_from_zip", forbidden)
    with pytest.raises(ValueError, match="saved source archive SHA differs"):
        reader.load_saved_game(Path("unused-owned-path"), "expected", 7)
    assert calls == []


def test_written_reader_preserves_receipt_refusal_before_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _reader()
    namespace = reader._read_written.__globals__
    monkeypatch.setitem(namespace, "_file_sha", lambda _path: "expected")
    docs = {
        "manifest.json": {"schema": reader.SCHEMA, "training_ready": False,
                          "source_qualified": False, "row_count": 1,
                          "source_archive_sha256": "wrong"},
        "rows.json.gz": [{}], "game.json.gz": {"game_id": 7},
    }
    monkeypatch.setitem(namespace, "_zip_json", lambda _path, name, **_kw: docs[name])

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("receipt refusal must precede arrays")

    monkeypatch.setitem(namespace, "_arrays_from_zip", forbidden)
    receipt = {"archive_sha256": "expected", "rows": 1,
               "source_archive_sha256": "expected-source"}
    with pytest.raises(ValueError, match="published manifest/metadata receipt mismatch"):
        reader._read_written(Path("unused-owned-path"), receipt)
