from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache

# opening_source_code values that mark a seed-origin game (blind-spot FEN seeds).
_SEED_SOURCE_CODES = frozenset({2, 3})  # 2 = fenlist, 3 = fenlist_sf_refute


def position_key(fen: str) -> str:
    """Position identity: first 4 FEN fields (placement, stm, castling, ep)."""
    parts = str(fen).split()
    return " ".join(parts[:4])


def opening_source_code(source: str) -> int:
    s = str(source or "")
    if s.startswith("fenlist_sf_refute"):
        return 3
    if s.startswith("fenlist"):
        return 2
    if s.startswith(("book", "pgn")):
        return 1
    if s == "start" or s.startswith("start"):
        return 0
    if s.startswith("salvage"):
        return 4
    return 255


def content_seed_id(fen: str) -> int:
    """Stable, path-independent id for a seed position: a hash of its
    ``position_key``, masked to non-negative int32 (the shard field dtype).

    Distributed workers load the seed list from an ephemeral sha-named cache
    copy, so a manifest keyed by list PATH cannot be found at finalize time.
    Hashing the start position instead makes ``seed_id`` reproducible anywhere:
    the same seed FEN always maps to the same id, and id->FEN/severity/family
    mapping is recovered offline by hashing the seed list with this function.
    Collision odds across a few hundred seeds are ~1e-5 (31-bit space).
    """
    digest = hashlib.blake2b(position_key(fen).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def _manifest_path_for(list_path: str | None) -> str | None:
    if not list_path:
        return None
    return str(list_path) + ".manifest.json"


@lru_cache(maxsize=8)
def _load_by_key(manifest_path: str, mtime: float) -> dict[str, tuple[int, int]]:
    # mtime in the key busts the cache when the file changes.
    del mtime
    with open(manifest_path, encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[str, tuple[int, int]] = {}
    for k, v in data.get("by_key", {}).items():
        out[k] = (int(v[0]), int(v[1]))
    return out


def resolve_seed_ids(
    fen: str, list_path: str | None, *, source_code: int = 255,
) -> tuple[int, int]:
    """Return ``(seed_id, seed_family_id)`` for a game's start FEN.

    Only seed-origin rows (``source_code`` in {2,3}) get an id; everything else
    returns ``(-1, -1)``. When a path-keyed manifest is present (single-process
    / local runs) its curated ids win; otherwise — the distributed case — fall
    back to the content hash so tagging still works. ``family_id`` mirrors
    ``seed_id`` as a v1 placeholder until near-transposition grouping ships.
    """
    if int(source_code) not in _SEED_SOURCE_CODES:
        return (-1, -1)
    mp = _manifest_path_for(list_path)
    if mp and os.path.exists(mp):
        try:
            hit = _load_by_key(mp, os.path.getmtime(mp)).get(position_key(fen))
        except Exception:
            hit = None
        if hit is not None:
            return hit
    cid = content_seed_id(fen)
    return (cid, cid)
