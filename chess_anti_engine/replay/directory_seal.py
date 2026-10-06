"""Opt-in directory producer density seal for shards written by save_local_shard_arrays.

``save_local_shard_arrays`` writes ``directory_producer_seal.json`` into the
Zarr store and records that file's sha256 on every array attribute
``directory_producer_seal_sha256``. The seal is a manifest of the chunk keys
the producer actually stored, plus the sha256 of each key's raw store bytes
(the compressed chunk file, not a decoded array). It also binds each
array's ``.zarray`` ``order``: decode reshapes a chunk with that field, so a
C-to-F edit changes the returned values without touching the compressed
bytes. ``load_shard_arrays`` checks the seal after the codec allowlist and
before any chunk is decoded.

Two producer densities, one per shard:

- ``dense`` (the writer default). ``write_empty_chunks=True``. Every declared
  chunk key is stored. A missing key fails even when the missing values would
  have equalled the fill value, because a deleted nonzero chunk and a deleted
  fill chunk look the same once Zarr has substituted the fill.
- ``sparse-fill``. ``write_empty_chunks=False``. Omitted keys are listed as
  ``elided`` only after the producer checks that each omitted in-memory slice
  equals the array fill value. A nonzero omitted slice refuses publication and
  leaves no shard behind. After elision the store cannot tell a fill omission
  from a deleted nonzero chunk, so the reader trusts ``elided`` only as the
  complement of the checksummed ``stored`` set: stored and elided are disjoint,
  their union is the declared grid, elided keys are absent, and stored keys
  match the recorded raw-byte checksums.

Readers: a shard with neither the seal file nor any array attribute keeps the
legacy fill behavior. That absence is not a validation of historical data.
Removing the seal file while any array attribute remains fails closed. Removing
the seal file and every array attribute together drops the shard back onto the
legacy path; a deleted chunk then zero-fills again. This seal is not a
signature: a rewrite that updates the chunk bytes, the manifest, and every
array attribute together is a new producer statement.

The manifest lives in the store, so a directory content hash and an upload tar
both carry it, and ``ShardMeta`` does not see it (it is not a group attribute).
Overlay loads do not run this check. ``begin_policy_shard`` copies policy
``.zattrs``, so an overlay tree may contain the attribute without containing
the manifest; that copy is not a seal of the overlay. Packed ZIP admission
treats the JSON as one opaque root member. The packed dense chunk-grid rule is
a separate contract and is not implemented here.

Verification hashes stored chunk bytes again at load. An earlier tree hash is
not reused: the file can change between the two reads. The second pass is
bounded by the same chunk cap the writer uses (100_000 declared chunks per
array, 200_000 store keys, 4 MiB of manifest).
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import zarr

SEAL_FILENAME = "directory_producer_seal.json"
SEAL_ATTR = "directory_producer_seal_sha256"
SCHEMA = 1
KIND = "directory-producer-density-seal"
DENSITY_DENSE = "dense"
DENSITY_SPARSE_FILL = "sparse-fill"
DENSITIES = frozenset({DENSITY_DENSE, DENSITY_SPARSE_FILL})
MAX_DECLARED_CHUNKS = 100_000
MAX_STORE_KEYS = 200_000
MAX_SEAL_BYTES = 4_000_000

_DOC_KEYS = frozenset({"schema", "kind", "density", "arrays"})
_ARRAY_KEYS = frozenset({
    "shape", "chunks", "dtype", "fill_value", "order", "dimension_separator",
    "stored", "elided",
})
_STORED_KEYS = frozenset({"key", "sha256"})
_HEX64 = frozenset("0123456789abcdef")


def write_directory_producer_seal(
    group: Any,
    sources: Mapping[str, np.ndarray],
    density: str,
) -> None:
    """Bind a density manifest to the arrays just written into ``group``.

    Raises before the caller publishes the directory when a dense chunk is
    missing or a sparse-fill omission is not the fill value.
    """
    if density not in DENSITIES:
        raise ValueError(
            f"chunk_density must be 'dense' or 'sparse-fill', got {density!r}"
        )
    names = _array_names(group)
    if set(names) != set(sources):
        raise ValueError(
            "directory producer seal sources do not match the written arrays: "
            f"written {sorted(names)}, sources {sorted(sources)}"
        )
    specs = {
        name: _spec_for_array(group[name], np.asarray(sources[name]), density)
        for name in names
    }
    payload = _canonical_bytes({
        "schema": SCHEMA,
        "kind": KIND,
        "density": density,
        "arrays": specs,
    })
    store = group.store
    store[SEAL_FILENAME] = payload
    stored = _raw_bytes(store, SEAL_FILENAME)
    if stored != payload:
        raise RuntimeError(
            "directory producer seal bytes were transformed by the store; "
            "refusing to publish"
        )
    digest = hashlib.sha256(payload).hexdigest()
    for name in names:
        group[name].attrs[SEAL_ATTR] = digest
    if _raw_bytes(store, SEAL_FILENAME) != payload:
        raise RuntimeError(
            "binding directory producer seal attributes rewrote the manifest; "
            "refusing to publish"
        )


def verify_directory_producer_seal(group: Any) -> None:
    """No-op only when the seal file and every array attribute are absent.

    Any other shape fails closed before the caller decodes chunk bytes.
    """
    store = group.store
    names = _array_names(group)
    bound = [name for name in names if _attr_bound(group[name])]
    present = SEAL_FILENAME in store
    if not present and not bound:
        return
    if not present:
        raise ValueError(
            "directory producer seal file is missing while array attributes "
            f"still bind {bound}; refusing fill decode"
        )
    raw = _raw_bytes(store, SEAL_FILENAME)
    if len(raw) > MAX_SEAL_BYTES:
        raise ValueError(
            f"directory producer seal is {len(raw)} bytes; cap is "
            f"{MAX_SEAL_BYTES}; refusing fill decode"
        )
    try:
        text = raw.decode("utf-8")
        doc = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "directory producer seal is not utf-8 JSON; refusing fill decode"
        ) from exc
    if not isinstance(doc, dict) or not _exact_keys(doc, _DOC_KEYS):
        raise ValueError(
            "directory producer seal schema keys are "
            f"{sorted(doc) if isinstance(doc, dict) else type(doc).__name__}; "
            "refusing fill decode"
        )
    if type(doc["schema"]) is not int or doc["schema"] != SCHEMA or doc["kind"] != KIND:
        raise ValueError(
            "directory producer seal schema/kind is not "
            f"{SCHEMA}/{KIND}; refusing fill decode"
        )
    if doc["density"] not in DENSITIES:
        raise ValueError(
            f"directory producer seal density {doc['density']!r} is not "
            "dense or sparse-fill; refusing fill decode"
        )
    arrays = doc["arrays"]
    if not isinstance(arrays, dict) or not _exact_keys(arrays, frozenset(names)):
        raise ValueError(
            "directory producer seal array set does not match the group; "
            "refusing fill decode"
        )
    if _canonical_bytes(doc) != raw:
        raise ValueError(
            "directory producer seal is not canonical JSON; refusing fill decode"
        )
    digest = hashlib.sha256(raw).hexdigest()
    keys_by_store: dict[int, list[str]] = {}
    for name in names:
        arr = group[name]
        if SEAL_ATTR not in arr.attrs or arr.attrs[SEAL_ATTR] != digest:
            raise ValueError(
                f"directory producer seal attribute on {name!r} does not match "
                "the manifest bytes; refusing fill decode"
            )
        spec = arrays[name]
        if not isinstance(spec, dict) or not _exact_keys(spec, _ARRAY_KEYS):
            raise ValueError(
                f"directory producer seal entry for {name!r} has unexpected "
                "fields; refusing fill decode"
            )
        chunk_store = arr.chunk_store
        ident = id(chunk_store)
        if ident not in keys_by_store:
            keys_by_store[ident] = _store_keys(chunk_store)
        _verify_array(name, arr, spec, doc["density"], keys_by_store[ident])


def _array_names(group: Any) -> list[str]:
    names = list(group.array_keys())
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or name in (SEAL_FILENAME, ".zgroup", ".zattrs", ".zmetadata")
            or "/" in name
            or "\\" in name
            or name.startswith(".")
        ):
            raise ValueError(
                f"directory producer seal refuses array name {name!r}"
            )
    return names


def _attr_bound(arr: Any) -> bool:
    try:
        return SEAL_ATTR in arr.attrs
    except Exception as exc:
        raise ValueError(
            f"directory producer seal cannot read attributes of {getattr(arr, 'path', arr)!r}; "
            "refusing fill decode"
        ) from exc


def _spec_for_array(arr: Any, source: np.ndarray, density: str) -> dict[str, Any]:
    _require_v2(arr)
    shape = _shape_of(arr)
    chunks = _chunks_of(arr)
    if tuple(int(dim) for dim in source.shape) != shape:
        raise ValueError(
            f"{arr.path} source shape {tuple(source.shape)} does not match "
            f"the written shape {shape}; refusing to publish"
        )
    if np.dtype(source.dtype).str != np.dtype(arr.dtype).str:
        raise ValueError(
            f"{arr.path} source dtype {np.dtype(source.dtype).str} does not "
            f"match the written dtype {np.dtype(arr.dtype).str}; refusing to publish"
        )
    separator = _separator_of(arr)
    declared = _declared_keys(arr)
    present = {
        key for key in _store_keys(arr.chunk_store)
        if key.startswith(_prefix(arr)) and not _is_array_meta(arr, key)
    }
    stored: list[dict[str, str]] = []
    elided: list[str] = []
    for key in declared:
        if key in present:
            stored.append({
                "key": key,
                "sha256": hashlib.sha256(_raw_bytes(arr.chunk_store, key)).hexdigest(),
            })
        elif density == DENSITY_DENSE:
            raise ValueError(
                f"dense directory producer omitted stored chunk {key}; "
                "refusing to publish"
            )
        elif _chunk_matches_fill(source, shape, chunks, _coord_for_key(arr, key), arr.fill_value):
            elided.append(key)
        else:
            raise ValueError(
                f"sparse-fill elided chunk {key} differs from fill_value; "
                "refusing to publish"
            )
    extra = sorted(present.difference(declared))
    if extra:
        raise ValueError(
            f"directory producer saw unexpected chunk keys {extra[:4]}; "
            "refusing to publish"
        )
    stored.sort(key=lambda item: item["key"])
    elided.sort()
    return {
        "shape": list(shape),
        "chunks": list(chunks),
        "dtype": np.dtype(arr.dtype).str,
        "fill_value": _encode_fill(arr.fill_value, np.dtype(arr.dtype)),
        "order": _order_of(arr),
        "dimension_separator": separator,
        "stored": stored,
        "elided": elided,
    }


def _verify_array(
    name: str,
    arr: Any,
    spec: Mapping[str, Any],
    density: str,
    store_keys: list[str],
) -> None:
    _require_v2(arr)
    shape = _shape_of(arr)
    chunks = _chunks_of(arr)
    if _int_list(spec["shape"], label=f"{name} shape") != list(shape):
        raise ValueError(
            f"directory producer seal shape for {name!r} does not match the "
            "array; refusing fill decode"
        )
    if _int_list(spec["chunks"], label=f"{name} chunks") != list(chunks):
        raise ValueError(
            f"directory producer seal chunks for {name!r} do not match the "
            "array; refusing fill decode"
        )
    if spec["dtype"] != np.dtype(arr.dtype).str:
        raise ValueError(
            f"directory producer seal dtype for {name!r} does not match the "
            "array; refusing fill decode"
        )
    if spec["order"] != _order_of(arr):
        raise ValueError(
            f"directory producer seal order for {name!r} does not match "
            "the array; refusing fill decode"
        )
    if spec["dimension_separator"] != _separator_of(arr):
        raise ValueError(
            f"directory producer seal separator for {name!r} does not match "
            "the array; refusing fill decode"
        )
    if spec["fill_value"] != _encode_fill(arr.fill_value, np.dtype(arr.dtype)):
        raise ValueError(
            f"directory producer seal fill_value for {name!r} does not match "
            "the array; refusing fill decode"
        )
    if density == DENSITY_DENSE and spec["elided"]:
        raise ValueError(
            f"dense directory producer seal lists elided chunks for {name!r}; "
            "refusing fill decode"
        )
    declared = _declared_keys(arr)
    stored_map = _stored_map(name, spec["stored"])
    elided = _elided_list(name, spec["elided"])
    if set(stored_map) & set(elided):
        raise ValueError(
            f"directory producer seal lists {name!r} chunks as both stored "
            "and elided; refusing fill decode"
        )
    if set(stored_map).union(elided) != set(declared):
        raise ValueError(
            f"directory producer seal chunk grid for {name!r} does not match "
            "the declared coordinates; refusing fill decode"
        )
    prefix = _prefix(arr)
    present = {
        key for key in store_keys
        if key.startswith(prefix) and not _is_array_meta(arr, key)
    }
    extra = sorted(present.difference(stored_map))
    if extra:
        raise ValueError(
            f"directory producer seal extra stored key {extra[0]!r} under "
            f"{name!r}; refusing fill decode"
        )
    missing = sorted(set(stored_map).difference(present))
    if missing:
        raise ValueError(
            f"directory producer seal stored chunk {missing[0]!r} for "
            f"{name!r} is missing; refusing fill decode"
        )
    materialized = sorted(set(elided).intersection(present))
    if materialized:
        raise ValueError(
            f"directory producer seal elided key {materialized[0]!r} for "
            f"{name!r} is present in the store; refusing fill decode"
        )
    for key, expected in stored_map.items():
        actual = hashlib.sha256(_raw_bytes(arr.chunk_store, key)).hexdigest()
        if actual != expected:
            raise ValueError(
                f"directory producer seal checksum mismatch for {key!r}; "
                "refusing fill decode"
            )


def _require_v2(arr: Any) -> None:
    if not isinstance(arr, zarr.Array) or int(getattr(arr, "_version", 0)) != 2:
        raise ValueError(
            f"directory producer seal requires a zarr v2 array, got "
            f"{type(arr).__name__}; refusing fill decode"
        )


def _shape_of(arr: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in arr.shape)


def _chunks_of(arr: Any) -> tuple[int, ...]:
    chunks = tuple(int(dim) for dim in arr.chunks)
    if len(chunks) != len(arr.shape) or any(dim <= 0 for dim in chunks):
        raise ValueError(
            f"{arr.path} has chunks {chunks} for shape {tuple(arr.shape)}; "
            "refusing the directory producer seal"
        )
    return chunks


def _order_of(arr: Any) -> str:
    order = arr.order
    if order not in ("C", "F"):
        raise ValueError(
            f"{arr.path} order {order!r} is not 'C' or 'F'; "
            "refusing the directory producer seal"
        )
    return str(order)


def _separator_of(arr: Any) -> str:
    separator = arr._dimension_separator
    if separator not in (".", "/"):
        raise ValueError(
            f"{arr.path} dimension_separator {separator!r} is not '.' or '/'"
        )
    return str(separator)


def _prefix(arr: Any) -> str:
    path = str(arr.path).strip("/")
    return f"{path}/" if path else ""


def _is_array_meta(arr: Any, key: str) -> bool:
    prefix = _prefix(arr)
    return key in (f"{prefix}.zarray", f"{prefix}.zattrs")


def _declared_keys(arr: Any) -> list[str]:
    cdata = tuple(int(dim) for dim in arr.cdata_shape)
    if any(dim < 0 for dim in cdata):
        raise ValueError(f"{arr.path} declared a negative chunk grid {cdata}")
    if cdata and any(dim == 0 for dim in cdata):
        coords: list[tuple[int, ...]] = []
    else:
        product = 1
        for dim in cdata:
            if dim > MAX_DECLARED_CHUNKS:
                raise ValueError(
                    f"{arr.path} declares a chunk axis of {dim}; "
                    f"seal cap is {MAX_DECLARED_CHUNKS}"
                )
            product *= dim
            if product > MAX_DECLARED_CHUNKS:
                raise ValueError(
                    f"{arr.path} declares {product} chunks; "
                    f"seal cap is {MAX_DECLARED_CHUNKS}"
                )
        coords = [()] if not cdata else list(itertools.product(*(range(dim) for dim in cdata)))
    keys = [str(arr._chunk_key(coord)) for coord in coords]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{arr.path} declared duplicate chunk keys")
    return keys


def _coord_for_key(arr: Any, key: str) -> tuple[int, ...]:
    """Inverse of the v2 chunk key this array just produced.

    Used only while sealing a source array, to slice the in-memory chunk that
    the store omitted. The reader never needs it: elided keys are not decoded.
    """
    prefix = _prefix(arr)
    if not key.startswith(prefix):
        raise ValueError(f"chunk key {key!r} is outside {arr.path}")
    body = key[len(prefix):]
    if body == "":
        return ()
    separator = _separator_of(arr)
    parts = body.split(separator)
    if not parts or any(not part.isdecimal() for part in parts):
        raise ValueError(f"chunk key {key!r} is not a declared coordinate")
    return tuple(int(part) for part in parts)


def _chunk_matches_fill(
    source: np.ndarray,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    coord: tuple[int, ...],
    fill: Any,
) -> bool:
    if len(coord) != len(shape):
        raise ValueError(
            f"chunk coordinate {coord} does not match rank {len(shape)}"
        )
    slices: list[slice] = []
    for size, chunk, index in zip(shape, chunks, coord, strict=True):
        start = index * chunk
        if start >= size:
            return False
        slices.append(slice(start, min(start + chunk, size)))
    piece = np.asarray(source[tuple(slices)]) if slices else np.asarray(source)
    if fill is None:
        return False
    if piece.dtype.kind == "f" and np.isnan(np.asarray(fill, dtype=piece.dtype)).item():
        return bool(np.isnan(piece).all())
    try:
        broadcast = np.broadcast_to(np.array(fill, dtype=piece.dtype), piece.shape)
    except (TypeError, ValueError):
        return False
    return bool(np.array_equal(piece, broadcast))


def _encode_fill(value: Any, dtype: np.dtype) -> Any:
    if value is None:
        return None
    kind = dtype.kind
    if kind == "f":
        number = np.asarray(value, dtype=dtype)
        if np.isnan(number).item():
            return "NaN"
        if np.isposinf(number).item():
            return "Infinity"
        if np.isneginf(number).item():
            return "-Infinity"
        return float(number)
    if kind in "ui":
        return int(np.asarray(value, dtype=dtype))
    if kind == "b":
        return bool(np.asarray(value, dtype=dtype))
    raise ValueError(
        f"directory producer seal cannot encode fill_value for dtype {dtype!s}"
    )


def _store_keys(store: Any) -> list[str]:
    found: list[str] = []
    for key in store:
        found.append(key if isinstance(key, str) else str(key))
        if len(found) > MAX_STORE_KEYS:
            raise ValueError(
                "directory producer seal store lists more than "
                f"{MAX_STORE_KEYS} keys; refusing the directory producer seal"
            )
    return found


def _raw_bytes(store: Any, key: str) -> bytes:
    try:
        raw = store[key]
    except KeyError as exc:
        raise ValueError(
            f"directory producer seal stored chunk {key!r} is missing; "
            "refusing fill decode"
        ) from exc
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, memoryview):
        return raw.tobytes()
    if isinstance(raw, np.ndarray):
        return raw.tobytes()
    raise ValueError(
        f"directory producer seal cannot hash {key!r} from "
        f"{type(raw).__name__}"
    )


def _canonical_bytes(doc: Mapping[str, Any]) -> bytes:
    return json.dumps(
        doc,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _exact_keys(value: Mapping[Any, Any], expected: frozenset[str]) -> bool:
    found: set[str] = set()
    for key in value:
        if not isinstance(key, str):
            return False
        found.add(key)
    return sorted(found) == sorted(expected)


def _int_list(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(
            f"directory producer seal {label} is not a list of ints; "
            "refusing fill decode"
        )
    return [int(item) for item in value]


def _stored_map(name: str, value: Any) -> dict[str, str]:
    if not isinstance(value, list):
        raise ValueError(
            f"directory producer seal stored list for {name!r} is not a list; "
            "refusing fill decode"
        )
    if len(value) > MAX_DECLARED_CHUNKS:
        raise ValueError(
            f"directory producer seal stored list for {name!r} exceeds "
            f"{MAX_DECLARED_CHUNKS}; refusing fill decode"
        )
    out: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict) or not _exact_keys(item, _STORED_KEYS):
            raise ValueError(
                f"directory producer seal stored entry for {name!r} is "
                "malformed; refusing fill decode"
            )
        key = item["key"]
        digest = item["sha256"]
        if not isinstance(key, str) or not key or key in out:
            raise ValueError(
                f"directory producer seal stored key {key!r} for {name!r} "
                "is duplicated or empty; refusing fill decode"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in _HEX64 for char in digest)
        ):
            raise ValueError(
                f"directory producer seal checksum for {key!r} is not "
                "lowercase sha256; refusing fill decode"
            )
        out[key] = digest
    return out


def _elided_list(name: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(
            f"directory producer seal elided list for {name!r} is not a list; "
            "refusing fill decode"
        )
    if len(value) > MAX_DECLARED_CHUNKS:
        raise ValueError(
            f"directory producer seal elided list for {name!r} exceeds "
            f"{MAX_DECLARED_CHUNKS}; refusing fill decode"
        )
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(
            f"directory producer seal elided keys for {name!r} are malformed; "
            "refusing fill decode"
        )
    if len(value) != len(set(value)):
        raise ValueError(
            f"directory producer seal elided keys for {name!r} repeat; "
            "refusing fill decode"
        )
    return list(value)
