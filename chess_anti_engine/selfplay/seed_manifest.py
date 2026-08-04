from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache

log = logging.getLogger(__name__)

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
    if s.startswith("start"):
        return 0
    if s.startswith("salvage"):
        return 4
    if s.startswith("random"):
        return 5
    return 255


def content_seed_id(fen: str) -> int:
    """Stable, path-independent id for a seed position: a hash of its
    ``position_key``, masked to non-negative int32 (the shard field dtype).

    Distributed workers load the seed list from an ephemeral sha-named cache
    copy, so a manifest keyed by list PATH cannot be found at finalize time.
    Hashing the start position instead makes ``seed_id`` reproducible anywhere:
    the same seed FEN always maps to the same id, and id->FEN/severity/family
    mapping is recovered offline by hashing the seed list with this function.
    Collision odds across a few hundred seeds are ~2e-5 (31-bit space).
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
    dropped = 0
    first_bad: tuple[str, object] | None = None
    for k, v in data.get("by_key", {}).items():
        sid, fid = int(v[0]), int(v[1])
        # Range-check curated ids against the shard's int32 dtype here, at
        # load time — an out-of-range id from a hand-built manifest would
        # otherwise crash shard SERIALIZATION on the worker, far from the
        # cause. Bad entries are dropped (the row falls back to content hash).
        if 0 <= sid <= 0x7FFFFFFF and 0 <= fid <= 0x7FFFFFFF:
            out[k] = (sid, fid)
        else:
            dropped += 1
            if first_bad is None:
                first_bad = (k, v)
    if dropped:
        # ⚑ Logged, not silent. The drop is correct (it keeps a bad id out of
        # the shard writer), but a hand-built manifest with one bad row used to
        # yield a silent MIX of curated and hashed aliases within a single run,
        # with nothing anywhere naming the offending entry.
        log.warning(
            "seed manifest %s: dropped %d entry/entries whose ids are outside "
            "non-negative int32 (shard dtype); first was key=%r value=%r. Those "
            "seeds carry no curated alias",
            manifest_path, dropped,
            first_bad[0] if first_bad else None,
            first_bad[1] if first_bad else None,
        )
    return out


def curated_seed_ids(fen: str, list_path: str | None) -> tuple[int, int] | None:
    """The manifest's curated ``(seed_id, seed_family_id)``, or None.

    An ALIAS, never the row's identity — see :func:`resolve_seed_ids`. Curated
    ids are the seed list's enumeration index, which ``build_seed_manifest.py``
    documents as "only stable WITHIN one list version — the retire step
    rewrites the list, shifting indices". They are useful for reading a local
    run against the list a human is editing, and unusable for anything
    longitudinal.

    Returns None when there is no manifest, when the manifest does not list
    this position, or when the manifest cannot be read.
    """
    mp = _manifest_path_for(list_path)
    if not mp or not os.path.exists(mp):
        return None
    try:
        by_key = _load_by_key(mp, os.path.getmtime(mp))
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        # ⚑ Narrowed from a bare `except Exception`, which used to swallow ANY
        # failure here and degrade to the other id scheme with no warning — a
        # corrupt or truncated manifest silently changed which identity every
        # seed row got. These are the failures a malformed file actually
        # produces: unreadable (OSError), not JSON or not an int (ValueError),
        # wrong types (TypeError), wrong shape (KeyError/IndexError). Anything
        # else is a bug in this module and propagates.
        log.warning(
            "seed manifest %s is unreadable (%s); seed rows keep their content "
            "-hash identity and simply carry no curated alias", mp, exc,
        )
        return None
    return by_key.get(position_key(fen))


def resolve_seed_ids(fen: str, *, source_code: int = 255) -> tuple[int, int]:
    """Return ``(seed_id, seed_family_id)`` for a game's start FEN.

    Only seed-origin rows (``source_code`` in {2,3}) get an id; everything else
    returns ``(-1, -1)``.

    ⚑ ALWAYS the content hash, in every run mode. It used to be the curated
    manifest id when ``<list_path>.manifest.json`` happened to exist and the
    content hash otherwise, which meant **the same seed FEN got two different
    ids depending on a file's presence**: a local/single-process run tagged it
    with the curated enumeration index, a distributed run with the hash. Both
    are valid non-negative int32s, so the split was invisible at the row level
    and silently divided one seed into two in any ``seed_id``-grouped analysis
    — the same family as the standing "shard indices COLLIDE across lineages,
    never join by basename" rule.

    Making the hash primary is the direction that does NOT break longitudinal
    joins, and the evidence is in the repo rather than in this docstring:

    * production selfplay is distributed, and a worker's seed list is an
      ephemeral sha-named cache copy with no manifest beside it, so **every
      production shard already carries content-hash ids**. The curated branch
      only ever fired in local runs. Flipping the primary changes no shard that
      exists;
    * ``scripts/build_seed_manifest.py`` already says so itself — "for any
      cross-version analysis use ``content_id`` ... which is stable by
      construction", and it emits a ``by_content_id`` map described as the
      thing "enrichment analysis joins shard seed_id -> ";
    * the curated id is the list's enumeration index, which the retire step
      shifts whenever the list is rewritten. Keeping it as primary would have
      preserved the *less* stable of the two identities.

    The curated pair is still available via :func:`curated_seed_ids` as an
    alias for local tooling. ``family_id`` mirrors ``seed_id`` as a v1
    placeholder until near-transposition grouping ships; when it does, it
    should key off the content id so it stays mode-independent.

    ⚑ ``list_path`` is GONE from this signature, deliberately. Keeping it and
    ignoring it would be a parameter the caller believes matters and that does
    nothing -- this codebase's signature defect wearing a different hat. With
    it removed, path-dependence cannot be reintroduced here without changing
    the signature, which makes the guarantee structural rather than a promise
    in a docstring.
    """
    if int(source_code) not in _SEED_SOURCE_CODES:
        return (-1, -1)
    cid = content_seed_id(fen)
    return (cid, cid)
