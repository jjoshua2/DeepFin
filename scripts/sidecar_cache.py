"""Shared identity layouts and disk-cache reservations for corpus sidecars."""

import numpy as np

# The compact provenance format and adapter identity cache share this layout.
RAW_IDENTITY_DTYPE = np.dtype([
    ("source", "<u4"), ("row", "<u4"), ("game_id", "<i8"),
    ("ply", "<i4"), ("worker_id", "<i4"),
    ("input_key", "u1", (16,)), ("stored_input_key", "u1", (16,)),
])


def rank_identity_dtype(top_k: int) -> np.dtype:
    return np.dtype([
        ("game_id", "<i8"), ("ply", "<i4"), ("worker_id", "<i4"),
        ("input_key", "u1", (16,)), ("stored_input_key", "u1", (16,)),
        ("indices", "<u2", (top_k,)), ("gaps", "<f4", (top_k,)),
        ("count", "u1"), ("valid", "u1"),
    ])


def rank_cache_bytes(rows: int, raw_shards: int, top_k: int) -> int:
    """Exact rank admission reservation, including seen bits and shard headers."""
    return rows * (rank_identity_dtype(top_k).itemsize + 1) + raw_shards * 512


def raw_identity_cache_bytes(rows: int, raw_shards: int) -> int:
    """Conservative adapter reservation for every selected row/shard.

    Each first shard access reserves its array plus 4096 bytes above accumulated
    actual NPY sizes. Reserving that header allowance for every shard covers any
    first-access order, including a shard whose rows are later all filtered out.
    """
    return rows * RAW_IDENTITY_DTYPE.itemsize + raw_shards * 4096
