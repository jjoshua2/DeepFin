"""Shared no-decode safety gate for Zarr shard and target-overlay arrays."""
from __future__ import annotations

from typing import Any

import numpy as np
import zarr

# --- Untrusted-deserialization guard for uploaded shards (issue #411) --------
#
# An uploaded ``.zarr`` array is decoded through the codec its own ``.zarray``
# declares. A numcodecs *object codec* (Pickle/JSON/MsgPack/VLen*) attached as a
# filter or compressor runs an attacker-chosen deserializer -- ``pickle`` calls
# ``__reduce__`` -- the moment a chunk is materialized on the training host.
# ``allow_pickle=False`` closes the twin ``.npz`` sink in ``load_shard_arrays``;
# this closes the ``.zarr`` sibling. It is an ALLOWLIST, not a denylist: every
# permitted dtype/compressor is a fixed numeric/byte-transform that provably
# cannot deserialize an object, so the next hostile codec is rejected by
# default rather than needing to be enumerated.
#
# Calibrated against what production actually writes (``save_local_shard_arrays``
# is the ONLY zarr writer; it emits ``Blosc(cname="zstd")`` with no filters) and
# empirically confirmed against 14 real shards under ``runs/`` on 2026-08-13:
# every array used dtype in {u1,i1,i2,i4,i8,f2,f4}, compressor ``blosc``, and
# NO filters. The dtype allowlist is by *kind* (bool/int/uint/float) so a future
# numeric width needs no change here; the compressor allowlist admits the family
# of pure byte-transform (de)compressors so a compressor swap does not take the
# fleet to zero, while still rejecting every object codec.
_SAFE_SHARD_DTYPE_KINDS: frozenset[str] = frozenset({"b", "i", "u", "f"})
_SAFE_SHARD_COMPRESSOR_IDS: frozenset[str] = frozenset(
    {"blosc", "zstd", "lz4", "gzip", "zlib", "bz2", "lzma"}
)


def _reject_unsafe_shard_codecs(proxies: dict[str, Any]) -> None:
    """Reject uploaded zarr arrays whose dtype/codec could run a deserializer.

    ⚑ FAILS CLOSED. Every member must be POSITIVELY identified as an
    allowlisted numeric ``zarr.Array``; anything this function cannot classify
    -- a sub-group in a field slot, a member with no readable ``dtype``, an
    unreadable ``.zarray`` -- is REJECTED, not defaulted. The obvious spelling
    ``np.dtype(getattr(arr, "dtype", None))`` is a trap: it maps a MISSING
    dtype to ``float64``, whose kind ``f`` this allowlist permits, so the
    guard would accept exactly the members it failed to understand. A security
    gate whose default is ACCEPT is the "gate that cannot fail" shape. The
    downstream ``validate_arrays`` is NOT the backstop here -- it lives in a
    different function that a later reordering could move or skip.

    ⚑ Takes the SAME proxy objects the loader goes on to materialize, not the
    group to re-walk. Two independent walks over ``_SHARD_FIELDS`` is precisely
    how a guard and its loader drift apart later; passing the built dict makes
    "the guard inspected what was decoded" true by construction, and halves the
    lazy-path cost.

    Reads ``.zarray`` metadata only (``dtype``/``filters``/``compressor``) --
    no chunk is decoded -- so it is safe on lazy, untrusted proxies BEFORE
    materialization.
    """
    for name, arr in proxies.items():
        if not isinstance(arr, zarr.Array):
            raise ValueError(
                f"shard member {name!r} is a {type(arr).__name__}, not a zarr array; "
                f"refusing to decode (untrusted-deserialization guard)",
            )
        raw_dtype = getattr(arr, "dtype", None)
        if raw_dtype is None:
            raise ValueError(
                f"shard array {name!r} declares no dtype; "
                f"refusing to decode (untrusted-deserialization guard)",
            )
        try:
            dtype = np.dtype(raw_dtype)
        except TypeError as exc:
            raise ValueError(
                f"shard array {name!r} declares an unreadable dtype {raw_dtype!r}; "
                f"refusing to decode (untrusted-deserialization guard)",
            ) from exc
        if dtype.kind not in _SAFE_SHARD_DTYPE_KINDS:
            raise ValueError(
                f"shard array {name!r} declares non-numeric dtype {dtype!s}; "
                f"refusing to decode (untrusted-deserialization guard)",
            )
        filters = getattr(arr, "filters", None)
        if filters:
            filter_ids = [getattr(f, "codec_id", type(f).__name__) for f in filters]
            raise ValueError(
                f"shard array {name!r} declares filters {filter_ids}; only "
                f"filter-free numeric arrays are accepted (untrusted-deserialization guard)",
            )
        compressor = getattr(arr, "compressor", None)
        if compressor is not None:
            codec_id = getattr(compressor, "codec_id", type(compressor).__name__)
            if codec_id not in _SAFE_SHARD_COMPRESSOR_IDS:
                raise ValueError(
                    f"shard array {name!r} uses disallowed compressor {codec_id!r}; "
                    f"only byte-transform compressors are accepted "
                    f"(untrusted-deserialization guard)",
                )
