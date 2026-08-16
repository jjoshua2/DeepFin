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

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

#: Bumped only when the STAMP's own layout changes, not when a version does.
AUDIT_CACHE_FORMAT = 1

#: Sentinel key that distinguishes the header record from a data row. A data
#: row is keyed by `key`; nothing else in the schema uses this name. Readers
#: that merely need to SKIP the header — `scripts/paired_compare.py` — need
#: this and nothing else.
#:
#: ⚑ The sentinel is ENFORCED, not merely conventional: `write_audit_cache`
#: refuses to write a data row carrying it, so "line begins with this key" is a
#: sound header test for an ad-hoc `grep`/`jq` reader too. `json.dumps(...,
#: sort_keys=True)` on the header puts it first alphabetically, which is why
#: `grep -v '^{"audit_cache_format":'` is exact rather than a heuristic.
STAMP_FORMAT_KEY = "audit_cache_format"

#: Digest of the policy index map the cache's move columns were written under.
POLICY_MAP_VERSION_KEY = "policy_map_version"

#: Digest of the audit ruler (scoring behaviour) that produced the numbers.
AUDIT_RULER_VERSION_KEY = "audit_ruler_version"

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

#: Human-readable path of the matched-rows NPZ an `--input-encoding stored` run
#: read its 175-plane inputs from, recorded for the report banner.
MATCHED_ROWS_KEY = "matched_rows"

#: Content digest of that NPZ — the value actually COMPARED, for the same reason
#: `AUDIT_SET_DIGEST_KEY` is: in `stored` mode the scores depend on the snapshot's
#: CONTENTS, and two runs over different snapshots at the same path (or the same
#: snapshot at two paths) must not be confused for one another.
MATCHED_ROWS_DIGEST_KEY = "matched_rows_digest"

#: Label of the net a foreign-net cache SCORED — `scripts/foreign_net_audit.py`
#: records it in the stamp. It names the comparison's SUBJECT, not its ruler.
NET_KEY = "net"

#: The keys `audit_cache_stamp` writes for EVERY stamped cache, whatever the
#: writer. `chess_anti_engine.eval.audit_cache.audit_cache_stamp` builds its
#: header from exactly these plus the caller's `extra`, and
#: `tests/test_audit_cache_stamp.py::test_the_core_stamp_keys_match_the_writer`
#: fails if the two drift apart.
#:
#: The distinction is load-bearing for `require_same_stamp`: a key ONE side
#: declares is version skew between two writer builds when it is an `extra`, and
#: a malformed or hand-made stamp when it is a CORE key — every stamped dump ever
#: written carries all of these, so their absence cannot be explained by skew.
#: The first warns, the second refuses.
#:
#: ⚑ `STAMP_FORMAT_KEY` is a member for completeness and is UNREACHABLE through
#: that path: `is_stamp_record` recognises a header BY that key, so a stamp
#: lacking it is read as a data row and never becomes a one-sided stamp key. The
#: refusal is live for the other two. Stated so nobody later reads the set as
#: three enforced keys when the behaviour pins two.
CORE_STAMP_KEYS: frozenset[str] = frozenset({
    STAMP_FORMAT_KEY,
    POLICY_MAP_VERSION_KEY,
    AUDIT_RULER_VERSION_KEY,
})

#: Stamp keys that may legitimately DIFFER between two dumps being compared.
#: Everything else in a stamp is ruler identity and must match, so this is
#: expressed as an EXCLUDE set rather than an include list: an include list
#: would have to be edited in lockstep with the writer and would fail silently
#: when it was not. Same reasoning as `scripts/paired_compare.py` skipping on
#: the SENTINEL rather than special-casing one key.
#:
#: ⚑ NOT "guarded the day it appears" — that claim was here and it was wrong.
#: A new stamp field is guarded from the day BOTH writers emit it. On the day
#: only one does, the key is one-sided, and for an `extra` key `require_same_stamp`
#: WARNS and continues (exit 0) rather than refusing; for a `CORE_STAMP_KEYS` key
#: it refuses. See `require_same_stamp` for the full argument — including why the
#: earlier rationale here, which named `scripts/audit_compare_buckets.py`, was
#: false: that script never calls `require_same_stamp`.
#:
#: ⚑⚑ `STAMP_FORMAT_KEY` IS NOT EXCLUDED, AND THE RATIONALE THAT EXCLUDED IT
#: WAS FALSE FOR THE READER THAT MATTERS. It used to say the key is "equal by
#: construction on any pair a reader accepts" — true of `read_audit_cache_stamp`,
#: which RAISES on a format mismatch, and false of `scripts/paired_compare.py`,
#: whose `load_dump` tested only key PRESENCE (`if STAMP_FORMAT_KEY in r`) and
#: never read the value. MEASURED: dump A at format 1 vs dump B at format 99,
#: everything else identical → exit 0, verdict printed. A justification that
#: names one reader's invariant and is applied to another reader is the same
#: defect as the stamp being recognised and then ignored, one level up. It is
#: latent while the constant is 1 and live the day it is bumped, which is
#: precisely when a stale reader is most dangerous. `load_dump` now also
#: range-checks the declared format, so the two guards are independent.
#:
#:   - `ROW_COUNT_KEY` differs whenever two dumps cover different position
#:     counts, which a paired join handles by intersecting. ⚑ It is excluded
#:     only from the CROSS-DUMP comparison; each dump's count is still binding
#:     on its own body, and every reader must enforce that (see
#:     `read_audit_cache`, and `load_dump` since the #442 review).
#:   - `AUDIT_SET_KEY` is a human-readable PATH, and this module's own comment on
#:     `AUDIT_SET_DIGEST_KEY` says why a path is not a provenance value. The
#:     digest is the field that must agree, and it is deliberately not excluded.
#:   - `NET_KEY` names the net a foreign-net cache SCORED, and the whole purpose
#:     of a paired comparison is that the two sides ran different nets. Treating
#:     it as ruler identity refuses `paired_compare.py A.jsonl B.jsonl --join-key
#:     key --field exp_regret` over two `foreign_net_audit.py` caches — the
#:     comparison the tool exists for — on the one field guaranteed to differ.
#:     `value_regret.py` already keeps its net label on the ROWS for this reason.
#:     ⚑ The subject of a measurement is not its ruler; excluding it costs no
#:     coverage, because no writer uses `net` to describe HOW it scored.
#:   - `MATCHED_ROWS_KEY` is a PATH, excluded for exactly the reason
#:     `AUDIT_SET_KEY` is. `MATCHED_ROWS_DIGEST_KEY` is NOT excluded: in
#:     `--input-encoding stored` mode the scores are a function of that
#:     snapshot's contents.
STAMP_NON_IDENTITY_KEYS: frozenset[str] = frozenset({
    ROW_COUNT_KEY,
    AUDIT_SET_KEY,
    MATCHED_ROWS_KEY,
    NET_KEY,
})


def is_stamp_record(rec: object) -> bool:
    """True when `rec` is a provenance HEADER rather than a data row.

    The one place this test is spelled out. Every reader of a stamped cache
    needs it, and each one that re-implements it is a chance to get it wrong in
    a different direction — `scripts/paired_compare.py` folded the header into
    its per-row ruler set, and `tests/test_onnx_net_source.py` counted it as a
    scored position (`assert len(rows) == 2` reading 3).
    """
    return isinstance(rec, dict) and STAMP_FORMAT_KEY in rec


def iter_data_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield a stamped-or-unstamped JSONL cache's DATA rows, header skipped.

    ⚑ THE SKIPPING READER, NOT THE VALIDATING ONE. It does not check the format
    version, the row count, or the stamp's identity fields, and it accepts an
    unstamped legacy dump unchanged. Use it where a consumer only ever wanted
    the rows — an ad-hoc count, a field histogram, a test. Where the numbers
    feed a verdict, use `chess_anti_engine.eval.audit_cache.read_audit_cache`
    (validating) or `scripts/paired_compare.py` (validating and paired).

    It lives in the leaf module so a consumer can skip a header without paying
    ~4 s / 750 MB for torch — see this module's docstring.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if is_stamp_record(rec):
                continue
            yield rec


__all__ = [
    "AUDIT_CACHE_FORMAT",
    "AUDIT_RULER_VERSION_KEY",
    "AUDIT_SET_DIGEST_KEY",
    "AUDIT_SET_KEY",
    "CORE_STAMP_KEYS",
    "MATCHED_ROWS_DIGEST_KEY",
    "MATCHED_ROWS_KEY",
    "NET_KEY",
    "POLICY_MAP_VERSION_KEY",
    "ROW_COUNT_KEY",
    "STAMP_FORMAT_KEY",
    "STAMP_NON_IDENTITY_KEYS",
    "is_stamp_record",
    "iter_data_rows",
]
