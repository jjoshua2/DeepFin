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
the file. ``save_local_shard_arrays`` runs that check before its existing
delete-then-rename, so a refusal leaves the previous destination in place and
does not publish the temporary directory. This module does not change that
replace. The seal is not a MAC: a rewrite that updates the ``.zarray`` bytes,
the chunk keys, the manifest, and every array attribute together is a new
producer statement. Replacing a chunk's bytes without renaming the key is
outside this contract.

Overlay loads do not run this check. ``begin_policy_shard`` copies policy
``.zattrs``, so an overlay tree can carry the attribute without this manifest.
That copy is not a seal of the overlay, and a missing overlay chunk can still
fill-decode. Packed ZIP admission allowlists the root name as one opaque
member. It does not apply a dense chunk-grid rule.

Verification re-reads the manifest and each ``.zarray`` on every load,
including ``validate=False``. It does not read chunk payloads. Caps: 100_000
declared chunks per array, 200_000 store keys, 4_000_000 manifest bytes.
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

_DOC_KEYS = frozenset({"schema", "kind", "density", "arrays"})
_ARRAY_KEYS = frozenset({"zarray_sha256", "stored"})
_HEX64 = frozenset("0123456789abcdef")


def write_directory_producer_seal(group: Any) -> None:
    """Bind a dense inventory to the arrays just written into ``group``.

    Raises before the caller publishes the directory when a declared chunk is
    missing or the manifest exceeds ``MAX_SEAL_BYTES``. The oversized bytes
    are not stored.
    """
    names = _array_names(group)
    store = group.store
    listed = _store_keys(store)
    specs = {
        name: _spec_for_array(group[name], listed)
        for name in names
    }
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


def verify_directory_producer_seal(
    group: Any,
    opened: Mapping[str, Any] | None = None,
) -> None:
    """No-op only when the seal file and every array attribute are absent.

    Any other shape fails closed before the caller decodes chunk bytes.
    ``opened`` is the array objects decode will use. When a name is present
    there, the approved ``.zarray`` bytes are installed onto that object.
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
    listed = _store_keys(store)
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
        _verify_array(name, arr, spec, listed)


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


def _spec_for_array(arr: Any, listed: list[str]) -> dict[str, Any]:
    _require_v2(arr)
    declared = _declared_keys(arr)
    present = _present_chunks(arr, listed)
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


def _verify_array(
    name: str,
    arr: Any,
    spec: Mapping[str, Any],
    listed: list[str],
) -> None:
    _require_v2(arr)
    raw = _raw_bytes(arr.store, _zarray_key(arr), publish=False)
    if hashlib.sha256(raw).hexdigest() != _hex_digest(name, spec["zarray_sha256"]):
        raise ValueError(
            f"directory producer seal declaration for {name!r} does not match "
            "the .zarray bytes; refusing fill decode"
        )
    _install_approved_metadata(name, arr, raw)
    declared = _declared_keys(arr)
    stored = _stored_keys(name, spec["stored"])
    if stored != declared:
        raise ValueError(
            f"directory producer seal chunk grid for {name!r} does not match "
            "the declared coordinates; refusing fill decode"
        )
    present = _present_chunks(arr, listed)
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


def _present_chunks(arr: Any, listed: list[str]) -> set[str]:
    prefix = _prefix(arr)
    return {
        key for key in listed
        if key.startswith(prefix) and not _is_array_meta(arr, key)
    }


def _declared_keys(arr: Any) -> list[str]:
    chunks = tuple(int(dim) for dim in arr.chunks)
    if len(chunks) != len(arr.shape) or any(dim <= 0 for dim in chunks):
        raise ValueError(
            f"{arr.path} has chunks {chunks} for shape {tuple(arr.shape)}; "
            "refusing the directory producer seal"
        )
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
    keys.sort()
    return keys


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
    if len(value) != len(set(value)):
        raise ValueError(
            f"directory producer seal stored keys for {name!r} repeat; "
            "refusing fill decode"
        )
    return list(value)
