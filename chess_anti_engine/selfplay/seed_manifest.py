from __future__ import annotations

import json
import os
from functools import lru_cache


def position_key(fen: str) -> str:
    """Position identity: first 4 FEN fields (placement, stm, castling, ep)."""
    parts = str(fen).split()
    return " ".join(parts[:4])


_SOURCE_CODES = {  # prefix-matched; longest/most-specific first in resolver
    "fenlist_sf_refute": 3,
    "fenlist": 2,
    "book": 1,
    "pgn": 1,
    "start": 0,
    "salvage": 4,
}


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


def resolve_seed_ids(fen: str, list_path: str | None) -> tuple[int, int]:
    """Return (seed_id, seed_family_id) for a game's start FEN, or (-1,-1)."""
    mp = _manifest_path_for(list_path)
    if not mp or not os.path.exists(mp):
        return (-1, -1)
    try:
        by_key = _load_by_key(mp, os.path.getmtime(mp))
    except Exception:
        return (-1, -1)
    return by_key.get(position_key(fen), (-1, -1))
