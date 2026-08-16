"""Wire-format vocabulary for provenance-stamped audit caches.

⚑ THIS MODULE MUST STAY A LEAF. It exists so a reader can recognise a stamp
header without paying for the machinery that PRODUCES one.

`chess_anti_engine/eval/__init__.py` eagerly imports `.puzzles`, which imports
torch, so anything under `chess_anti_engine.eval` costs ~4.0 s and ~750 MB
(measured: 74 -> 1209 modules). `scripts/paired_compare.py` is deliberately
stdlib+numpy and runs on the training box every monitoring cycle via
`scripts/monitor_fen.sh`, so importing the sentinel from `eval.audit_cache`
would have put a multi-second, multi-hundred-megabyte transient on the machine
that is training. Importing this module instead costs ~0.01 s and ~4 MB.

Keep it that way: **stdlib only, no project imports**. `eval/audit_cache.py`
re-exports every name here, so consumers that already pay for torch are
unaffected and there is exactly ONE definition of each literal — duplicating
them would invite precisely the drift the stamp exists to prevent.
`tests/test_audit_cache_stamp.py` asserts `paired_compare` imports without
torch, so a regression here fails a test rather than slowing production.
"""
from __future__ import annotations

#: Bumped only when the STAMP's own layout changes, not when a version does.
AUDIT_CACHE_FORMAT = 1

#: Sentinel key that distinguishes the header record from a data row. A data
#: row is keyed by `key`; nothing else in the schema uses this name. Readers
#: that merely need to SKIP the header — `scripts/paired_compare.py` — need
#: this and nothing else.
STAMP_FORMAT_KEY = "audit_cache_format"

#: Row count, written by `write_audit_cache` and ENFORCED by `read_audit_cache`.
#: The stamp otherwise binds only to line 1, so without this a stamp lifted from
#: a good cache would certify a truncated file, or two caches concatenated.
ROW_COUNT_KEY = "rows"

#: Human-readable scoring-set path, recorded for the report banner.
AUDIT_SET_KEY = "audit_set"

#: Content digest of the scoring set — the value actually COMPARED. A path
#: string is not a provenance value: `data/audit_set_v1.jsonl` and
#: `/abs/path/data/audit_set_v1.jsonl` name the same file and compare unequal,
#: while two different files can share a basename. Every other field in this
#: stamp is a derived digest; this one is too.
AUDIT_SET_DIGEST_KEY = "audit_set_digest"

#: Stamp keys that may legitimately DIFFER between two dumps being compared.
#: Everything else in a stamp is ruler identity and must match, so this is
#: expressed as an EXCLUDE set rather than an include list: a version field
#: added to the stamp later is then guarded automatically, whereas an include
#: list would silently fail to cover it. Same reasoning as
#: `scripts/paired_compare.py` skipping on the SENTINEL rather than
#: special-casing one key.
#:
#:   - `STAMP_FORMAT_KEY` is equal by construction on any pair a reader accepts.
#:   - `ROW_COUNT_KEY` differs whenever two dumps cover different position
#:     counts, which a paired join handles by intersecting.
#:   - `AUDIT_SET_KEY` is a human-readable PATH, and this module's own comment on
#:     `AUDIT_SET_DIGEST_KEY` says why a path is not a provenance value. The
#:     digest is the field that must agree, and it is deliberately not excluded.
STAMP_NON_IDENTITY_KEYS: frozenset[str] = frozenset({
    STAMP_FORMAT_KEY,
    ROW_COUNT_KEY,
    AUDIT_SET_KEY,
})

__all__ = [
    "AUDIT_CACHE_FORMAT",
    "AUDIT_SET_DIGEST_KEY",
    "AUDIT_SET_KEY",
    "ROW_COUNT_KEY",
    "STAMP_FORMAT_KEY",
    "STAMP_NON_IDENTITY_KEYS",
]
