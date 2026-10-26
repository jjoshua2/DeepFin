"""Dense directory producer seal for shards written by ``save_local_shard_arrays``.

The writer stores ``directory_producer_seal.json`` in the Zarr store and records
that file's sha256 on every array attribute ``directory_producer_seal_sha256``.
The manifest is a dense inventory:

- ``density`` is ``dense``. The production writer uses Zarr's default
  ``write_empty_chunks=True`` and does not grow a sparse-fill mode.
- Each array entry is the sha256 of that array's raw ``.zarray`` bytes. Decode
  uses the whole file (shape, chunks, dtype, fill value, order, filters,
  compressor, dimension separator, zarr format), so any edit of those bytes
  fails before fill decoding. Zarr caches that metadata at open and decode
  keeps using the cache. The check therefore installs these approved bytes
  onto the array object that will be decoded, so a file restored between the
  open and this read cannot leave a different order or dtype in that cache.
- Each array entry lists the stored chunk keys and nothing about their payload
  bytes. The list must equal the grid declared by the bound ``.zarray``. A
  missing, extra, or renamed key fails before fill decoding.

A store with neither the seal file nor any array attribute keeps the legacy
fill path. That includes a legitimate unsealed store written with
``write_empty_chunks=False``. Absence of the seal is not a validation of
historical data. Removing the file while any array attribute remains fails
closed. Removing the file and every array attribute together is the legacy
path again.

The writer refuses a manifest larger than ``MAX_SEAL_BYTES`` before it stores
the file. It also refuses when the final key count, including this manifest
and every array ``.zattrs``, would exceed ``MAX_STORE_KEYS``. A projected
overrun is refused before the manifest is stored. The reader counts those
final keys, so a count taken before the manifest and attributes exist would
publish a store the reader rejects. ``save_local_shard_arrays`` runs both
checks before its existing delete-then-rename, so a refusal leaves the
previous destination in place and does not publish the temporary directory.
This module does not change that replace. The seal is not a MAC: a rewrite that
updates the ``.zarray`` bytes, the chunk keys, the manifest, and every array
attribute together is a new producer statement. Replacing a chunk's bytes
without renaming the key is outside this contract.

Window accounting reads the row count from the ``.zarray`` bytes that matched
the seal, and only when the chunk grid in those bytes equals the sealed
inventory. Present arrays must agree on that row count and carry the
manifest digest. Removing ``priority`` entirely does not zero the count.
Any other declared array that is gone refuses it, so a forged ``x`` cannot
inflate deletion by deleting the siblings that would disagree. The manifest
must declare ``x``, ``policy_target``, ``wdl_target``, and ``has_policy``
before the window will count rows. The load check requires the arrays that
are present to share one row count before decode. A leftover
``.zattrs`` refuses the count, including one restamped with the new digest.
The chunk count is the integer grid; a float division that
would report fewer chunks is refused. A mismatch counts as no rows, so a
forged shape cannot inflate deletion. An unsealed shard still returns the
shape stored on disk. A directory entry that decode will not open as a
file, including a dangling symlink, is not a stored chunk.

Overlay loads do not run this check. ``begin_policy_shard`` copies policy
``.zattrs``, so an overlay tree can carry the attribute without this manifest.
That copy is not a seal of the overlay, and a missing overlay chunk can still
fill-decode. Packed ZIP admission allowlists the root name as one opaque
member. Packed admission independently enforces the chunk grid declared by
each array; the loader then applies this producer seal before decoding.

Verification re-reads the manifest and each ``.zarray`` on every load,
including ``validate=False``. It does not read chunk payloads. Caps: 100_000
declared chunks per array, four axes (the shard schema maximum), 4_000_000
JSON-escaped key bytes across the manifest, 200_000 final store keys,
8_000_000 listed store-key bytes, and 4_000_000 manifest bytes. Grid key cost
is checked before coordinates are enumerated; coordinates are consumed lazily.
The store-key cap is that final count on both the writer and the reader.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping
from typing import Any

import zarr

SEAL_FILENAME = "directory_producer_seal.json"
SEAL_ATTR = "directory_producer_seal_sha256"
SCHEMA = 1
KIND = "directory-producer-density-seal"
DENSITY_DENSE = "dense"
MAX_DECLARED_CHUNKS = 100_000
MAX_STORE_KEYS = 200_000
MAX_SEAL_BYTES = 4_000_000
# Shard declarations have at most four axes (N,C,8,8). Bound representation
# as well as cardinality: a small grid can otherwise contain enormous keys.
MAX_ARRAY_RANK = 4
MAX_GRID_KEY_BYTES = MAX_SEAL_BYTES
MAX_STORE_KEY_BYTES = 2 * MAX_SEAL_BYTES
# Quarantine strips ``priority`` and still needs the honest row count.
# Every other declared array must remain, or the window count is refused.
_WINDOW_OPTIONAL_ARRAYS = frozenset({"priority"})
# These names carry one row per position. A manifest that drops one of them
# can no longer prove the siblings agree, so the window count is refused.
_WINDOW_REQUIRED_ARRAYS = ("x", "policy_target", "wdl_target", "has_policy")

_DOC_KEYS = frozenset({"schema", "kind", "density", "arrays"})
_ARRAY_KEYS = frozenset({"zarray_sha256", "stored"})
_HEX64 = frozenset("0123456789abcdef")


def write_directory_producer_seal(group: Any) -> None:
    """Bind a dense inventory to the arrays just written into ``group``.

    Raises before the caller publishes the directory when a declared chunk is
    missing, the manifest exceeds ``MAX_SEAL_BYTES``, or the final store key
    count would exceed ``MAX_STORE_KEYS``. An oversized manifest is not stored.
    A projected key-count overrun is refused before the manifest is stored.
    The actual key count is checked again after the manifest and attributes
    are bound, still before the caller publishes.
    """
    names = _array_names(group)
    store = group.store
    listed = _store_keys(store)
    inventory = _partition_store_keys(group, listed)
    specs: dict[str, Any] = {}
    key_bytes = 0
    for name in names:
        spec = _spec_for_array(group[name], inventory)
        key_bytes += _key_bytes(spec["stored"])
        _refuse_key_bytes(key_bytes, MAX_GRID_KEY_BYTES)
        specs[name] = spec
    payload = _canonical_bytes({
        "schema": SCHEMA,
        "kind": KIND,
        "density": DENSITY_DENSE,
        "arrays": specs,
    })
    if len(payload) > MAX_SEAL_BYTES:
        raise ValueError(
            f"directory producer seal is {len(payload)} bytes; cap is "
            f"{MAX_SEAL_BYTES}; refusing to publish"
        )
    _refuse_store_key_count(_projected_store_key_count(listed, group, names))
    store[SEAL_FILENAME] = payload
    stored = _raw_bytes(store, SEAL_FILENAME, publish=True)
    if stored != payload:
        raise RuntimeError(
            "directory producer seal bytes were transformed by the store; "
            "refusing to publish"
        )
    digest = hashlib.sha256(payload).hexdigest()
    for name in names:
        group[name].attrs[SEAL_ATTR] = digest
    if _raw_bytes(store, SEAL_FILENAME, publish=True) != payload:
        raise RuntimeError(
            "binding directory producer seal attributes rewrote the manifest; "
            "refusing to publish"
        )
    for name in names:
        if _sha256(group[name], publish=True) != specs[name]["zarray_sha256"]:
            raise RuntimeError(
                "binding directory producer seal attributes rewrote .zarray; "
                "refusing to publish"
            )
    # Reader ``_store_keys`` counts the manifest and per-array attributes.
    # Recount after they exist, while the caller still holds the temp directory.
    _store_keys(store)


def verify_directory_producer_seal(
    group: Any,
    opened: Mapping[str, Any] | None = None,
) -> None:
    """No-op only when the seal file and every array attribute are absent.

    Any other shape fails closed before the caller decodes chunk bytes.
    ``opened`` is the array objects decode will use. When a name is present
    there, the approved ``.zarray`` bytes are installed onto that object.
    """
    loaded = _manifest(group)
    if loaded is None:
        return
    doc, digest = loaded
    names = _array_names(group)
    arrays = doc["arrays"]
    if not isinstance(arrays, dict) or not _exact_keys(arrays, frozenset(names)):
        raise ValueError(
            "directory producer seal array set does not match the group; "
            "refusing fill decode"
        )
    listed = _store_keys(group.store)
    inventory = _partition_store_keys(group, listed)
    rows: int | None = None
    for name in names:
        arr = opened[name] if opened is not None and name in opened else group[name]
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
        count = _verify_array(name, arr, spec, inventory)
        if rows is None:
            rows = count
        elif count != rows:
            raise ValueError(
                f"directory producer seal row count for {name!r} is {count}, "
                f"not {rows}; refusing fill decode"
            )


def sealed_row_count(group: Any, name: str) -> int | None:
    """Rows in a sealed ``.zarray``, or None when the shard is unsealed.

    Every array still in the group must carry the manifest digest and the
    same row count, taken from ``.zarray`` bytes whose integer chunk grid
    matches the sealed inventory. ``priority`` may be absent. Any other
    declared array that is absent refuses the count. A ``.zattrs`` left
    behind by a removed declaration refuses the count. A declaration,
    digest, grid, actual stored-key inventory, or length mismatch raises.
    """
    loaded = _manifest(group)
    if loaded is None:
        return None
    doc, digest = loaded
    arrays = doc["arrays"]
    if not isinstance(arrays, dict) or name not in arrays:
        raise ValueError(
            f"directory producer seal does not declare {name!r}; "
            "refusing fill decode"
        )
    _require_declared_rows(arrays)
    present = _array_names(group)
    for found in present:
        arr = group[found]
        if SEAL_ATTR not in arr.attrs or arr.attrs[SEAL_ATTR] != digest:
            raise ValueError(
                f"directory producer seal attribute on {found!r} does not "
                "match the manifest bytes; refusing fill decode"
            )
    if name not in present:
        raise ValueError(
            f"directory producer seal array {name!r} is missing; "
            "refusing fill decode"
        )
    for missing in sorted(set(arrays).difference(present)):
        _require_no_conflicting_attr(group.store, missing)
        if missing not in _WINDOW_OPTIONAL_ARRAYS:
            raise ValueError(
                f"directory producer seal array {missing!r} is missing; "
                "refusing fill decode"
            )
    listed = _store_keys(group.store)
    inventory = _partition_store_keys(group, listed)
    rows: int | None = None
    for found in present:
        spec = arrays.get(found)
        if not isinstance(spec, dict):
            raise ValueError(
                f"directory producer seal entry for {found!r} has unexpected "
                "fields; refusing fill decode"
            )
        arr = group[found]
        raw = _declaration_bytes(found, arr, spec, digest)
        stored = _stored_keys(found, spec["stored"])
        if _grid_keys(found, raw, expected_count=len(stored)) != stored:
            raise ValueError(
                f"directory producer seal chunk grid for {found!r} does not "
                "match the declared coordinates; refusing fill decode"
            )
        _verify_stored_inventory(found, arr, stored, inventory)
        count = _rows_in_zarray(found, raw)
        if rows is None:
            rows = count
        elif count != rows:
            raise ValueError(
                f"directory producer seal row count for {found!r} is {count}, "
                f"not {rows}; refusing fill decode"
            )
    if rows is None:
        raise ValueError(
            "directory producer seal has no array row count; refusing fill decode"
        )
    return rows


def _require_declared_rows(arrays: Mapping[str, Any]) -> None:
    """The row-bearing arrays have to be in the manifest, not only on disk."""
    absent = [name for name in _WINDOW_REQUIRED_ARRAYS if name not in arrays]
    if absent:
        raise ValueError(
            "directory producer seal does not declare "
            f"{', '.join(absent)}; refusing fill decode"
        )


def _require_no_conflicting_attr(store: Any, name: Any) -> None:
    """A removed array may leave nothing. A leftover attribute refuses the count."""
    if (
        not isinstance(name, str)
        or not name
        or "/" in name
        or "\\" in name
        or name.startswith(".")
    ):
        raise ValueError(
            f"directory producer seal refuses array name {name!r}"
        )
    key = f"{name}/.zattrs"
    try:
        present = key in store
    except Exception as exc:
        raise ValueError(
            f"directory producer seal cannot stat {key!r}; refusing fill decode"
        ) from exc
    if not present:
        return
    raise ValueError(
        f"directory producer seal attribute on {name!r} remains after its "
        "declaration was removed; refusing fill decode"
    )


def _manifest(group: Any) -> tuple[dict[str, Any], str] | None:
    """The canonical manifest and its digest, or None when the shard is unsealed."""
    store = group.store
    names = _array_names(group)
    bound = [name for name in names if _attr_bound(group[name])]
    present = SEAL_FILENAME in store
    if not present and not bound:
        return None
    if not present:
        raise ValueError(
            "directory producer seal file is missing while array attributes "
            f"still bind {bound}; refusing fill decode"
        )
    raw = _raw_bytes(store, SEAL_FILENAME, publish=False)
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
    if doc["density"] != DENSITY_DENSE:
        raise ValueError(
            f"directory producer seal density {doc['density']!r} is not "
            "dense; refusing fill decode"
        )
    arrays = doc["arrays"]
    if not isinstance(arrays, dict):
        raise ValueError("directory producer seal arrays is not an object")
    key_bytes = 0
    key_count = 0
    for name, spec in arrays.items():
        if not isinstance(spec, dict) or not _exact_keys(spec, _ARRAY_KEYS):
            raise ValueError(f"directory producer seal entry for {name!r} has unexpected fields")
        _hex_digest(name, spec["zarray_sha256"])
        stored = _stored_keys(name, spec["stored"])
        key_count += len(stored)
        _refuse_store_key_count(key_count)
        key_bytes += _key_bytes(stored)
        _refuse_key_bytes(key_bytes, MAX_GRID_KEY_BYTES)
    if _canonical_bytes(doc) != raw:
        raise ValueError(
            "directory producer seal is not canonical JSON; refusing fill decode"
        )
    return doc, hashlib.sha256(raw).hexdigest()


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
            "directory producer seal cannot read attributes of "
            f"{getattr(arr, 'path', arr)!r}; refusing fill decode"
        ) from exc


def _spec_for_array(arr: Any, inventory: Mapping[str, list[str]]) -> dict[str, Any]:
    _require_v2(arr)
    declared = _declared_keys(arr)
    present = _present_chunks(arr, inventory)
    missing = sorted(set(declared).difference(present))
    if missing:
        raise ValueError(
            f"dense directory producer omitted stored chunk {missing[0]}; "
            "refusing to publish"
        )
    extra = sorted(present.difference(declared))
    if extra:
        raise ValueError(
            f"directory producer saw unexpected chunk keys {extra[:4]}; "
            "refusing to publish"
        )
    return {
        "zarray_sha256": _sha256(arr, publish=True),
        "stored": declared,
    }


def _declaration_bytes(
    name: str,
    arr: Any,
    spec: Mapping[str, Any],
    digest: str,
) -> bytes:
    """Raw ``.zarray`` bytes that match the seal. The caller parses these bytes."""
    _require_v2(arr)
    if SEAL_ATTR not in arr.attrs or arr.attrs[SEAL_ATTR] != digest:
        raise ValueError(
            f"directory producer seal attribute on {name!r} does not match "
            "the manifest bytes; refusing fill decode"
        )
    if not isinstance(spec, dict) or not _exact_keys(spec, _ARRAY_KEYS):
        raise ValueError(
            f"directory producer seal entry for {name!r} has unexpected "
            "fields; refusing fill decode"
        )
    raw = _raw_bytes(arr.store, _zarray_key(arr), publish=False)
    if hashlib.sha256(raw).hexdigest() != _hex_digest(name, spec["zarray_sha256"]):
        raise ValueError(
            f"directory producer seal declaration for {name!r} does not match "
            "the .zarray bytes; refusing fill decode"
        )
    return raw


def _rows_in_zarray(name: str, raw: bytes) -> int:
    try:
        doc = json.loads(raw)
        shape = doc["shape"]
        rows = shape[0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"directory producer seal declaration for {name!r} has no row "
            "count; refusing fill decode"
        ) from exc
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise ValueError(
            f"directory producer seal declaration for {name!r} has row count "
            f"{rows!r}; refusing fill decode"
        )
    return rows


def _verify_array(
    name: str,
    arr: Any,
    spec: Mapping[str, Any],
    inventory: Mapping[str, list[str]],
) -> int:
    _require_v2(arr)
    raw = _raw_bytes(arr.store, _zarray_key(arr), publish=False)
    if hashlib.sha256(raw).hexdigest() != _hex_digest(name, spec["zarray_sha256"]):
        raise ValueError(
            f"directory producer seal declaration for {name!r} does not match "
            "the .zarray bytes; refusing fill decode"
        )
    stored = _stored_keys(name, spec["stored"])
    # Check cheap schema/cardinality/cost against the bound bytes before reload
    # and enumeration. Keep the approved metadata installation before decode.
    _grid_declaration(name, raw, expected_count=len(stored))
    _install_approved_metadata(name, arr, raw)
    declared = _declared_keys(arr)
    if stored != declared:
        raise ValueError(
            f"directory producer seal chunk grid for {name!r} does not match "
            "the declared coordinates; refusing fill decode"
        )
    _verify_stored_inventory(name, arr, stored, inventory)
    return _rows_in_zarray(name, raw)


def _verify_stored_inventory(
    name: str, arr: Any, stored: list[str], inventory: Mapping[str, list[str]],
) -> None:
    """Same bounded decoder-openable inventory check for load and row count."""
    present = _present_chunks(arr, inventory)
    extra = sorted(present.difference(stored))
    if extra:
        raise ValueError(
            f"directory producer seal extra stored key {extra[0]!r} under "
            f"{name!r}; refusing fill decode"
        )
    missing = sorted(set(stored).difference(present))
    if missing:
        raise ValueError(
            f"directory producer seal stored chunk {missing[0]!r} for "
            f"{name!r} is missing; refusing fill decode"
        )


class _ApprovedBytes:
    """Store view that serves one already-hashed ``.zarray`` payload."""

    def __init__(self, store: Any, key: str, raw: bytes) -> None:
        self._store = store
        self._key = key
        self._raw = raw

    def __getitem__(self, item: str) -> Any:
        if item == self._key:
            return self._raw
        return self._store[item]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


def _install_approved_metadata(name: str, arr: Any, raw: bytes) -> None:
    """Reload ``arr`` from the bytes just hashed, not from a later file read.

    ``group[name]`` constructs a new array, so this has to run on the object
    the caller will decode. ``cache_metadata`` then keeps that metadata.
    """
    store = arr._store
    key = _zarray_key(arr)
    arr._store = _ApprovedBytes(store, key, raw)
    try:
        arr._load_metadata()
    except Exception as exc:
        raise ValueError(
            f"directory producer seal declaration for {name!r} could not be "
            "decoded; refusing fill decode"
        ) from exc
    finally:
        arr._store = store
    try:
        decoded = store._metadata_class.decode_array_metadata(raw)
    except Exception as exc:
        raise ValueError(
            f"directory producer seal declaration for {name!r} could not be "
            "decoded; refusing fill decode"
        ) from exc
    cached = getattr(arr, "_meta", None)
    if not isinstance(cached, Mapping) or not _meta_matches(cached, decoded):
        raise ValueError(
            f"directory producer seal declaration for {name!r} does not match "
            "the cached .zarray metadata; refusing fill decode"
        )


def _meta_matches(cached: Any, decoded: Any) -> bool:
    if isinstance(cached, Mapping) and isinstance(decoded, Mapping):
        if set(cached) != set(decoded):
            return False
        return all(_meta_matches(cached[key], decoded[key]) for key in cached)
    if isinstance(cached, (list, tuple)) and isinstance(decoded, (list, tuple)):
        return len(cached) == len(decoded) and all(
            _meta_matches(left, right) for left, right in zip(cached, decoded, strict=True)
        )
    if _both_nan(cached, decoded):
        return True
    return bool(cached == decoded)


def _both_nan(left: Any, right: Any) -> bool:
    if isinstance(left, (str, bytes, bool, int, Mapping, list, tuple)):
        return False
    if isinstance(right, (str, bytes, bool, int, Mapping, list, tuple)):
        return False
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return False
    return math.isnan(left_float) and math.isnan(right_float)


def _require_v2(arr: Any) -> None:
    if not isinstance(arr, zarr.Array) or int(getattr(arr, "_version", 0)) != 2:
        raise ValueError(
            f"directory producer seal requires a zarr v2 array, got "
            f"{type(arr).__name__}; refusing fill decode"
        )


def _sha256(arr: Any, *, publish: bool) -> str:
    raw = _raw_bytes(arr.store, _zarray_key(arr), publish=publish)
    return hashlib.sha256(raw).hexdigest()


def _zarray_key(arr: Any) -> str:
    return f"{_prefix(arr)}.zarray"


def _prefix(arr: Any) -> str:
    path = str(arr.path).strip("/")
    return f"{path}/" if path else ""


def _is_array_meta(arr: Any, key: str) -> bool:
    prefix = _prefix(arr)
    return key in (f"{prefix}.zarray", f"{prefix}.zattrs")


def _partition_store_keys(group: Any, listed: list[str]) -> dict[str, list[str]]:
    """Index a bounded listing once, retaining slash-separated descendants.

    Array names are immediate children of this group. Use its path so seals
    on nested groups retain the same full-prefix inventory semantics.
    """
    path = str(group.path).strip("/")
    prefix = f"{path}/" if path else ""
    inventory: dict[str, list[str]] = {}
    for key in listed:
        if not key.startswith(prefix):
            continue
        name, separator, _ = key[len(prefix):].partition("/")
        if separator:
            inventory.setdefault(f"{prefix}{name}/", []).append(key)
    return inventory


def _present_chunks(arr: Any, inventory: Mapping[str, list[str]]) -> set[str]:
    """Chunk keys decode can open. A dangling name is not one of them."""
    store = arr.store
    prefix = _prefix(arr)
    present: set[str] = set()
    for key in inventory.get(prefix, ()):
        if _is_array_meta(arr, key):
            continue
        if _store_has_file(store, key):
            present.add(key)
    return present


def _store_has_file(store: Any, key: str) -> bool:
    """Whether ``key in store`` is true, the check chunk decode uses."""
    try:
        return bool(key in store)
    except Exception as exc:
        raise ValueError(
            f"directory producer seal cannot stat {key!r}; "
            "refusing the directory producer seal"
        ) from exc


def _grid_declaration(
    name: str, raw: bytes, *, expected_count: int | None = None,
) -> tuple[list[int], str]:
    """Bound the raw declaration before reloading metadata or enumerating keys."""
    try:
        doc = json.loads(raw)
        shape = doc["shape"]
        chunks = doc["chunks"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"directory producer seal declaration for {name!r} has no chunk "
            "grid; refusing fill decode"
        ) from exc
    if (
        not isinstance(shape, list)
        or not isinstance(chunks, list)
        or not shape
        or len(shape) != len(chunks)
    ):
        raise ValueError(
            f"directory producer seal declaration for {name!r} has no chunk "
            "grid; refusing fill decode"
        )
    separator = doc.get("dimension_separator", ".")
    if separator not in (".", "/"):
        raise ValueError(
            f"directory producer seal declaration for {name!r} has chunk "
            f"separator {separator!r}; refusing fill decode"
        )
    dims = _bounded_grid(name, shape, chunks)
    if expected_count is not None and math.prod(dims) != expected_count:
        raise ValueError(
            f"directory producer seal chunk grid for {name!r} does not match "
            "the declared coordinates; refusing fill decode"
        )
    return dims, separator


def _grid_keys(name: str, raw: bytes, *, expected_count: int | None = None) -> list[str]:
    """Chunk keys declared by these ``.zarray`` bytes, in inventory order."""
    dims, separator = _grid_declaration(name, raw, expected_count=expected_count)
    if 0 in dims:
        return []
    keys = [
        f"{name}/{separator.join(str(coord) for coord in point)}"
        for point in itertools.product(*(range(dim) for dim in dims))
    ]
    keys.sort()
    return keys


def _refuse_key_bytes(size: int, cap: int) -> None:
    if size > cap:
        raise ValueError(f"directory producer seal key representation exceeds {cap} bytes")


def _key_bytes(keys: list[str]) -> int:
    # JSON escaping is at least as large as the string's in-memory characters.
    return sum(len(json.dumps(key, ensure_ascii=True)) for key in keys)


def _bounded_grid(name: str, shape: Any, chunks: Any) -> list[int]:
    """Refuse unsupported rank, cardinality and aggregate key cost before product.

    Count the decimal characters in each axis without creating coordinates or
    range pools. Empty axes still undergo schema checks, but have no key cost.
    """
    if (
        not isinstance(shape, (list, tuple))
        or not isinstance(chunks, (list, tuple))
        or len(shape) != len(chunks)
        or len(shape) > MAX_ARRAY_RANK
    ):
        raise ValueError(f"directory producer seal {name!r} rank/grid exceeds supported rank {MAX_ARRAY_RANK}")
    dims: list[int] = []
    for length, chunk in zip(shape, chunks, strict=True):
        if type(length) is not int or type(chunk) is not int or length < 0 or chunk <= 0:
            raise ValueError(f"directory producer seal {name!r} has invalid shape/chunks")
        dims.append(_chunk_count(length, chunk))
    product = math.prod(dims)
    if product > MAX_DECLARED_CHUNKS:
        raise ValueError(f"{name} declares {product} chunks; seal cap is {MAX_DECLARED_CHUNKS}")
    if not product:
        return dims
    # Prefix, quotes, separators, then the sum of axis digits across the grid.
    cost = product * (len(json.dumps(name, ensure_ascii=True)) + 1 + max(0, len(dims) - 1))
    for dim in dims:
        digits = dim
        power = 10
        while power < dim:
            digits += dim - power
            power *= 10
        cost += digits * (product // dim)
    _refuse_key_bytes(cost, MAX_GRID_KEY_BYTES)
    return dims


def _chunk_count(length: int, chunk: int) -> int:
    """Chunks that cover ``length``. Refuse a float division that collapses."""
    exact = (length + chunk - 1) // chunk if length else 0
    try:
        floated = math.ceil(length / chunk)
    except OverflowError as exc:
        raise ValueError(
            f"directory producer seal chunk count for length {length} "
            f"chunk {chunk} is not an exact integer grid"
        ) from exc
    if type(floated) is not int or exact != floated:
        raise ValueError(
            f"directory producer seal chunk count for length {length} "
            f"chunk {chunk} is {exact}, not the float grid {floated}"
        )
    return exact


def _declared_keys(arr: Any) -> list[str]:
    dims = _bounded_grid(str(arr.path), arr.shape, arr.chunks)
    if 0 in dims:
        return []
    keys = [str(arr._chunk_key(coord)) for coord in itertools.product(*(range(dim) for dim in dims))]
    _refuse_key_bytes(_key_bytes(keys), MAX_GRID_KEY_BYTES)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{arr.path} declared duplicate chunk keys")
    keys.sort()
    return keys


def _projected_store_key_count(listed: list[str], group: Any, names: list[str]) -> int:
    """Keys after the manifest and any missing per-array ``.zattrs`` are added."""
    present = set(listed)
    count = len(listed)
    if SEAL_FILENAME not in present:
        count += 1
    for name in names:
        if f"{_prefix(group[name])}.zattrs" not in present:
            count += 1
    return count


def _refuse_store_key_count(count: int) -> None:
    if count > MAX_STORE_KEYS:
        raise ValueError(
            "directory producer seal store lists more than "
            f"{MAX_STORE_KEYS} keys; refusing the directory producer seal"
        )


def _store_keys(store: Any) -> list[str]:
    found: list[str] = []
    key_bytes = 0
    for key in store:
        key = key if isinstance(key, str) else str(key)
        key_bytes += _key_bytes([key])
        _refuse_key_bytes(key_bytes, MAX_STORE_KEY_BYTES)
        _refuse_store_key_count(len(found) + 1)
        found.append(key)
    return found


def _raw_bytes(store: Any, key: str, *, publish: bool) -> bytes:
    try:
        raw = store[key]
    except KeyError as exc:
        action = "refusing to publish" if publish else "refusing fill decode"
        raise ValueError(
            f"directory producer seal entry {key!r} is missing; {action}"
        ) from exc
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, memoryview):
        return raw.tobytes()
    raise ValueError(
        f"directory producer seal cannot read {key!r} from {type(raw).__name__}"
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


def _hex_digest(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX64 for char in value)
    ):
        raise ValueError(
            f"directory producer seal declaration for {name!r} is not "
            "lowercase sha256; refusing fill decode"
        )
    return value


def _stored_keys(name: str, value: Any) -> list[str]:
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
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(
            f"directory producer seal stored keys for {name!r} are malformed; "
            "refusing fill decode"
        )
    _refuse_key_bytes(_key_bytes(value), MAX_GRID_KEY_BYTES)
    if len(value) != len(set(value)):
        raise ValueError(
            f"directory producer seal stored keys for {name!r} repeat; "
            "refusing fill decode"
        )
    return value
