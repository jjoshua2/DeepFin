"""Provenance-stamped foreign-net audit caches, and a reader that REFUSES.

A cache row (`best_move`, `topk`, `exp_regret`, `top1_regret`) is a joint
product of two things that both change under our feet:

* the **policy map** — the UCI -> slot lookup used to gather a foreign net's
  logits at the legal moves. `moves/leela_index.py` fixed a castling mis-map on
  2026-08-13 (`e1g1` resolved to slot 102, an ordinary king slide, instead of
  103): 296 moves across 256 of the 4000 audit positions;
* the **audit ruler** — `eval/audit.py`'s mate->cp folding and per-move regret.
  `a84aaf846` (#360, 2026-08-05) unified the mate mapping and added
  `AUDIT_REGRET_CAP_CP`. Mean `gap_cp` over the same keys reads 70.36 under the
  old ruler and 2006.85 under the current one.

Neither is visible in the file. `data/lc0/bt4_audit_cache.jsonl` (2026-06-26)
carried both defects for seven weeks while two scripts DEFAULTED to it, and a
published BT4 figure read 23.79 where the correct value is 20.08. The docstring
that warned about it was read by nobody, because a docstring is not a guard.

So: every cache carries a stamp as its FIRST JSONL record, and every read
validates that stamp before the rows are parsed.

DISPOSITION of the file this module was written for: `data/lc0/bt4_audit_cache.jsonl`
was DELETED on 2026-08-13, not repaired. It is clean in `wdl` and wrong in
`best_move` / `topk` / `exp_regret` / `top1_regret`, but its only reader
(`scripts/audit_compare_buckets.py`) prints the WDL calibration table and the
contaminated regret / agreement rows in ONE report, so no consumer wants the
clean column on its own. Keeping a half-good file would mean teaching the reader
which COLUMNS to trust — a per-field trust judgement, which is the class of
judgement that failed here in the first place. Regenerate instead: it is one
command, and the net is still banked under `data/lc0/`.

⚑ A CORRECTED 4000-row cache is banked at `scratchpad/gates/fix_bt4_4000.jsonl`
(md5 `aa82d8b96ae105464f494bd690f15a11`). **It is NOT stamped and MUST NOT be
hand-stamped**, because only half its provenance can be checked:

* its **RULER provenance VERIFIES** — `gap_cp` is a pure function of the frozen
  audit set and today's ruler, and recomputing it agrees on **4000/4000** rows;
* its **POLICY-MAP provenance is UNVERIFIED and unverifiable from the file** —
  `best_move`, `topk`, `exp_regret` and `top1_regret` are functions of the NET'S
  LOGITS, so confirming which map gathered them needs a GPU forward pass. (61 of
  its rows have a castle as `best_move`, which is consistent with the fixed map
  but is not proof: under the buggy map a castle could still hold the max logit.)

Stamping it by hand would assert exactly the half that cannot be checked, inside
the mechanism built to stop that. Regenerate with the command in `_REGENERATE`
when a GPU pause window is available, and let the writer stamp it.

**Absence of the stamp is a FAILURE, not a pass.** An "if present, check it"
guard would silently accept exactly the file that caused this, and would be the
project's signature defect (a value accepted and then ignored) rebuilt inside
the fix for it. There is deliberately no override flag on the READ path.

Both versions are DERIVED, never hand-maintained, on two independent legs:

* a **structural** digest — the AST of the functions that define the map / the
  ruler, with docstrings stripped, so a comment or docstring edit does NOT
  invalidate banked caches but any change to the code does;
* a **behavioural** digest — the functions run over a fixed probe covering the
  cases that historically broke (castling both colours, promotions, en passant,
  mate folding, the regret cap, the criticality edges), so a change in a
  TRANSITIVE dependency that the AST leg cannot see still bumps the version.

⚑ NOT AIRTIGHT, and the gaps are stated rather than papered over. The
structural leg covers only the functions named in `_RULER_SOURCES` /
`_MAP_SOURCES`; a behaviour change inside a helper they call is invisible to it
and is caught only if the behavioural probe happens to exercise it. The
behavioural leg covers only the probe positions; a map bug on a board shape the
probe does not contain is invisible to it. The two legs are chosen to fail in
different directions, and both fail CLOSED (a version that changes when it did
not need to costs a re-run; a version that fails to change costs a wrong
number). Adding a case to a probe is a deliberate cache invalidation — that is
the intended cost of making the guard stricter.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.eval import audit
from chess_anti_engine.moves.encode import move_to_index
from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_UCI_TO_IDX
from chess_anti_engine.moves.leela_index import leela_index_for_move
from chess_anti_engine.stockfish.wdl import mate_to_effective_cp
from chess_anti_engine.utils import sha256_file
from chess_anti_engine.utils.audit_cache_format import (
    AUDIT_CACHE_FORMAT,
    AUDIT_RULER_VERSION_KEY,
    AUDIT_SET_DIGEST_KEY,
    AUDIT_SET_KEY,
    CORE_STAMP_KEYS,
    MATCHED_ROWS_DIGEST_KEY,
    MATCHED_ROWS_KEY,
    NET_KEY,
    POLICY_MAP_VERSION_KEY,
    ROW_COUNT_KEY,
    STAMP_FORMAT_KEY,
    STAMP_NON_IDENTITY_KEYS,
    is_stamp_record,
    iter_data_rows,
)

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
    "AuditCacheError",
    "audit_cache_stamp",
    "audit_ruler_version",
    "audit_set_provenance",
    "ensure_cache_writable",
    "is_stamp_record",
    "iter_data_rows",
    "policy_map_version",
    "read_audit_cache",
    "read_audit_cache_by_key",
    "read_audit_cache_stamp",
    "require_same_audit_set",
    "stamp_summary",
    "write_audit_cache",
]

# The stamp's wire-format vocabulary lives in a LEAF module and is re-exported
# here, not re-declared. `scripts/paired_compare.py` needs only the sentinel to
# skip a header, and importing it from this module would drag in
# `chess_anti_engine.eval.__init__` -> `.puzzles` -> torch: measured 0.01 s /
# 14 MB against 3.97 s / 749 MB, on a script `scripts/monitor_fen.sh` runs
# against the live training box every monitoring cycle. Duplicating the
# literals instead would invite exactly the drift the stamp exists to prevent.

_REGENERATE = (
    "regenerate it with:\n"
    "    PYTHONPATH=. python3 scripts/foreign_net_audit.py \\\n"
    "        --onnx <net.onnx> --cache-out <this path> --out <report.md>\n"
    "  (or --checkpoint <ckpt> for one of our own nets); pass --force-cache-out\n"
    "  only when you intend to replace an existing banked file."
)


class AuditCacheError(RuntimeError):
    """A cache is unstamped, stale, malformed, or would be clobbered."""


# ---------------------------------------------------------------------------
# Structural leg: the AST of the defining code, docstrings stripped.
# ---------------------------------------------------------------------------


def _strip_docstrings(tree: ast.Module) -> ast.Module:
    """Drop every docstring node in place, so prose edits do not bump a version."""
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            continue
        body: list[ast.stmt] = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
    return tree


def _structure_digest(fn: Any) -> str:
    """AST dump of `fn`'s source, docstring-free and location-free."""
    src = textwrap.dedent(inspect.getsource(fn))
    return ast.dump(_strip_docstrings(ast.parse(src)), include_attributes=False)


def _fmt(x: float) -> str:
    """Stable text for a float probe output (12 significant digits)."""
    return format(float(x), ".12g")


# ---------------------------------------------------------------------------
# Policy map version
# ---------------------------------------------------------------------------

#: Probe boards. Every historically-broken shape is here on purpose: castling
#: both colours and both sides (the 2026-08-13 defect), promotions and
#: under-promotions with and without a capture, en passant, and the startpos.
#: Adding a FEN deliberately invalidates every banked cache.
_MAP_PROBE_FENS: tuple[str, ...] = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "1n6/P6k/8/8/8/8/6K1/8 w - - 0 1",
    "8/6k1/8/8/8/8/p6K/1N6 b - - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
)

_MAP_SOURCES = (leela_index_for_move, move_to_index)


def _map_behaviour_lines() -> list[str]:
    """Both mappers' output on every legal move of every probe board."""
    lines: list[str] = []
    for fen in _MAP_PROBE_FENS:
        board = chess.Board(fen)
        for uci in sorted(m.uci() for m in board.legal_moves):
            move = chess.Move.from_uci(uci)
            lines.append(
                f"{fen}|{uci}|{leela_index_for_move(board, move)}"
                f"|{move_to_index(move, board)}"
            )
    return lines


@lru_cache(maxsize=1)
def policy_map_version() -> str:
    """Digest of the UCI -> policy-slot mapping a cache's moves were gathered under.

    Three legs: the whole canonical 1858 table (exhaustive over the table), the
    AST of both mappers, and both mappers' behaviour on `_MAP_PROBE_FENS`.
    """
    h = hashlib.sha256()
    for uci, idx in sorted(LC0_1858_UCI_TO_IDX.items()):
        h.update(f"{uci}={idx}\n".encode())
    for fn in _MAP_SOURCES:
        h.update(_structure_digest(fn).encode())
    for line in _map_behaviour_lines():
        h.update(f"{line}\n".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Audit ruler version
# ---------------------------------------------------------------------------

#: One synthetic audit record, covering the axes `a84aaf846` moved: a mate line
#: of each sign (mate->cp folding), a duplicate move (first listing wins), an
#: unscoreable line (skipped, not fatal), and a spread wide enough that
#: `AUDIT_REGRET_CAP_CP` binds on at least one move.
_RULER_PROBE_RECORD = json.dumps(
    {
        "key": "stamp-probe",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "phase": 2,
        "source": 0,
        "multipv": [
            {"move": "e2e4", "cp": 35},
            {"move": "d2d4", "cp": 12},
            {"move": "g1f3", "mate": 5},
            {"move": "b1c3", "mate": -3},
            {"move": "e2e4", "cp": -900},
            {"move": "a2a3"},
            {"move": "h2h3", "cp": -1500},
        ],
        "wdl": [500, 300, 200],
        "nodes": 1_000_000,
        "depth": 40,
    },
    sort_keys=True,
)

_RULER_SOURCES = (
    audit.parse_audit_record,
    audit.move_regrets,
    audit.expected_and_top1_regret,
    audit.criticality_gap,
    audit.criticality_bucket,
    mate_to_effective_cp,
)


def _ruler_behaviour_lines() -> list[str]:
    """The ruler's own output on `_RULER_PROBE_RECORD` and the bucket edges."""
    pos = audit.parse_audit_record(_RULER_PROBE_RECORD)
    board = chess.Board(pos.fen)
    ucis = sorted(m.uci() for m in board.legal_moves)
    regrets = audit.move_regrets(pos, ucis)
    # A deterministic, deliberately non-uniform distribution: a uniform one
    # would make E[regret] insensitive to the ORDER the regrets come back in.
    probs = np.arange(1, len(ucis) + 1, dtype=np.float64)
    exp_r, top1_r = audit.expected_and_top1_regret(probs, regrets)
    gap = audit.criticality_gap(pos.move_cp)

    lines = [f"best_cp={_fmt(pos.best_cp)}", f"n_move_cp={len(pos.move_cp)}"]
    lines += [f"cp:{u}={_fmt(v)}" for u, v in sorted(pos.move_cp.items())]
    lines += [f"regret:{u}={_fmt(r)}" for u, r in zip(ucis, regrets, strict=True)]
    lines.append(f"exp={_fmt(exp_r)} top1={_fmt(top1_r)} gap={_fmt(gap)}")
    lines += [
        f"bucket({_fmt(g)})={audit.criticality_bucket(g)}"
        for g in (0.0, 19.999, 20.0, 49.999, 50.0, 99.999, 100.0, 1e9)
    ]
    lines += [f"mate({m})={_fmt(mate_to_effective_cp(m))}" for m in (-5, -1, 0, 1, 5)]
    return lines


@lru_cache(maxsize=1)
def audit_ruler_version() -> str:
    """Digest of the scoring rules a cache's regret columns were produced under.

    Three legs: the AST of the ruler functions, the live values of the
    constants they read, and the ruler's behaviour on `_RULER_PROBE_RECORD`.
    The constants are read at call time (not frozen at import), so a patched
    `AUDIT_REGRET_CAP_CP` or a moved bucket edge shows up here.
    """
    h = hashlib.sha256()
    for fn in _RULER_SOURCES:
        h.update(_structure_digest(fn).encode())
    h.update(f"cap={_fmt(audit.AUDIT_REGRET_CAP_CP)}\n".encode())
    h.update(
        ("edges=" + ",".join(_fmt(e) for e in audit.CRITICALITY_GAP_EDGES) + "\n").encode()
    )
    h.update(("names=" + ",".join(audit.CRITICALITY_BUCKET_NAMES) + "\n").encode())
    for line in _ruler_behaviour_lines():
        h.update(f"{line}\n".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Stamp, write, read
# ---------------------------------------------------------------------------


def audit_cache_stamp(**extra: Any) -> dict[str, Any]:
    """The header record written as line 1 of every audit cache."""
    return {
        STAMP_FORMAT_KEY: AUDIT_CACHE_FORMAT,
        POLICY_MAP_VERSION_KEY: policy_map_version(),
        AUDIT_RULER_VERSION_KEY: audit_ruler_version(),
        **extra,
    }


def ensure_cache_writable(path: Path, *, force: bool = False) -> None:
    """Refuse to clobber an existing cache. Call this BEFORE the expensive work.

    `write_audit_cache` repeats the check, but by then a multi-minute forward
    pass has already been paid for and the operator is staring at a traceback
    instead of a report. Every foreign-net cache is a banked measurement that
    something else joins against, so the default is refuse, not overwrite.
    """
    if force or not path.exists():
        return
    raise AuditCacheError(
        f"refusing to overwrite the existing audit cache {path}.\n"
        "  A banked cache is a measurement other analyses join against; "
        "overwriting it in place\n"
        "  destroys the comparison silently. Write somewhere else with "
        "--cache-out, or pass\n"
        "  --force-cache-out if replacing this file is what you mean to do."
    )


def write_audit_cache(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    force: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a stamped cache: header record first, then one row per line.

    `rows` is recorded by the WRITER, after `**extra`, so a caller cannot
    supply a count that disagrees with what was actually written. The reader
    enforces it — see `read_audit_cache`.

    ⚑ A DATA ROW MAY NOT CARRY `STAMP_FORMAT_KEY`. Every reader — this module's
    `read_audit_cache`, `scripts/paired_compare.load_dump`,
    `utils.audit_cache_format.iter_data_rows`, and any `grep`/`jq` one-liner an
    operator writes — separates header from body on that one key. That is only
    sound if the writer cannot emit a row that answers to it, so the writer
    refuses instead of documenting a convention. Enforced here rather than in
    the reader because a reader can only guess which of the two records is the
    real header, and by then the file is already banked.
    """
    ensure_cache_writable(path, force=force)
    materialised = list(rows)
    for i, row in enumerate(materialised):
        if STAMP_FORMAT_KEY in row:
            raise AuditCacheError(
                f"{path}: data row {i} carries the header sentinel "
                f"'{STAMP_FORMAT_KEY}'. Every reader tells the provenance "
                "header from the body by that key alone, so a row holding it "
                "would be read as a second header (or the body as one row "
                "short). Rename the field."
            )
    stamp = audit_cache_stamp(**dict(extra or {}))
    stamp[ROW_COUNT_KEY] = len(materialised)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(stamp, sort_keys=True) + "\n")
        for row in materialised:
            fh.write(json.dumps(row) + "\n")
    return stamp


def _reject(path: Path, what: str) -> AuditCacheError:
    return AuditCacheError(f"{path}: {what}\n  {_REGENERATE}")


def read_audit_cache_stamp(path: Path) -> dict[str, Any]:
    """Validate a cache's provenance, reading ONLY its header line.

    Raises `AuditCacheError` on a missing file, an unstamped file, or a stamp
    that does not match the CURRENT policy map / audit ruler. Call this before
    loading anything else, so a stale cache costs no work at all.

    ⚑ This validates PROVENANCE ONLY. It reads one line, so by construction it
    cannot check that the header describes the body — the row-count binding
    lives in `read_audit_cache`. A caller that uses this as its *only* check
    accepts a valid stamp over a truncated or concatenated file; use it for the
    cheap pre-flight and `read_audit_cache` to actually consume rows.
    """
    if not path.exists():
        raise _reject(path, "audit cache not found")
    with path.open(encoding="utf-8") as fh:
        first = fh.readline()
    if not first.strip():
        raise _reject(path, "audit cache is empty")
    try:
        header: object = json.loads(first)
    except json.JSONDecodeError as exc:
        raise _reject(path, f"first line is not JSON ({exc})") from exc
    if not isinstance(header, dict) or STAMP_FORMAT_KEY not in header:
        raise _reject(
            path,
            "UNSTAMPED audit cache — its first line carries no "
            f"'{STAMP_FORMAT_KEY}' provenance header, so it was written before "
            "provenance stamps existed and there is NO evidence of which policy "
            "map or audit ruler produced it. Absence of the stamp is a failure, "
            "not a pass: the file this guard exists for "
            "(data/lc0/bt4_audit_cache.jsonl, 2026-06-26) looked exactly like "
            "this and was wrong on both axes. It cannot be certified after the "
            "fact",
        )
    stamp: dict[str, Any] = header
    found_format = stamp.get(STAMP_FORMAT_KEY)
    if found_format != AUDIT_CACHE_FORMAT:
        raise _reject(
            path,
            f"stamp format {found_format!r}, this code writes "
            f"{AUDIT_CACHE_FORMAT!r}",
        )
    for field, current in (
        ("policy_map_version", policy_map_version()),
        ("audit_ruler_version", audit_ruler_version()),
    ):
        found = stamp.get(field)
        if found == current:
            continue
        detail = (
            f"absent from the stamp (expected {current!r})"
            if found is None
            else f"{found!r}, current is {current!r}"
        )
        why = (
            "the UCI -> policy-slot map changed"
            if field == "policy_map_version"
            else "eval/audit.py's mate->cp folding, regret cap or "
            "criticality edges changed"
        )
        raise _reject(
            path,
            f"STALE audit cache — {field} is {detail}. Since it was written, "
            f"{why}, so its regret and move columns are not comparable to "
            "anything produced today",
        )
    return stamp


def read_audit_cache(path: Path) -> list[dict[str, Any]]:
    """Stamp-checked cache read. The guard fires before any row is parsed.

    Also enforces the stamp's row count. The provenance header binds to line 1
    only, so without this a stamp lifted from a good cache would certify a
    truncated file, and two stamped caches concatenated would read as one.
    """
    stamp = read_audit_cache_stamp(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if lineno == 1 or not line.strip():
                continue
            try:
                row: object = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _reject(path, f"line {lineno} is not JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise _reject(path, f"line {lineno} is not a JSON object")
            if is_stamp_record(row):
                raise _reject(
                    path,
                    f"line {lineno} is a second provenance header — this looks "
                    "like two caches concatenated, which the line-1 stamp cannot "
                    "describe",
                )
            rows.append(row)
    declared = stamp.get(ROW_COUNT_KEY)
    if not isinstance(declared, int):
        raise _reject(
            path,
            f"stamp carries no integer '{ROW_COUNT_KEY}' count "
            f"(found {declared!r}), so the header cannot vouch for the body",
        )
    if declared != len(rows):
        raise _reject(
            path,
            f"stamp declares {declared} rows but the file holds {len(rows)} — "
            "TRUNCATED, appended to, or carrying a stamp lifted from another "
            "cache",
        )
    return rows


def read_audit_cache_by_key(path: Path) -> dict[str, dict[str, Any]]:
    """`read_audit_cache` keyed by each row's position `key`.

    ⚑ DUPLICATE KEYS ARE REFUSED, not last-won. A dict build silently collapses
    two rows claiming the same position into one, and the losers are invisible:
    absent from the join and absent from any count, so the caller reads a clean
    result over a smaller and biased sample. That is documented audit invariant
    L14, and `scripts/paired_compare.load_dump` was hardened against exactly it
    — "there is no principled winner between two rows claiming the same
    position", so stop rather than guess.

    The row-count binding does NOT cover this: it counts LINES, so two rows with
    one key satisfy it. Latent rather than live today (`per_position_277.jsonl`
    measures 4000 rows / 4000 unique keys), which is the moment to close it.
    """
    out: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for lineno, row in enumerate(read_audit_cache(path), start=2):  # line 1 = stamp
        if "key" not in row:
            raise _reject(path, f"line {lineno} has no 'key' field")
        key = str(row["key"])
        if key in out:
            duplicates.append(key)
            continue
        out[key] = row
    if duplicates:
        unique = sorted(set(duplicates))
        raise _reject(
            path,
            f"{len(duplicates)} duplicate rows across {len(unique)} repeated "
            f"'key' values, e.g. {unique[:3]} — two runs concatenated, a re-run "
            "appended, or the wrong file. A join cannot pick between two rows "
            "claiming the same position, and the row-count binding cannot see "
            "this because it counts lines, not distinct keys",
        )
    return out


def audit_set_provenance(path: Path) -> dict[str, Any]:
    """Stamp fields identifying the scoring set: readable path + CONTENT digest.

    Both are recorded and they do different jobs. The path is for the human
    reading the report banner; the DIGEST is what `require_same_audit_set`
    compares, because a path string is not a provenance value —
    `data/audit_set_v1.jsonl` and `/abs/path/data/audit_set_v1.jsonl` name one
    file and compare unequal, while two genuinely different files can share a
    basename. Every other field in this stamp is derived from content rather
    than from a name; this makes the last one consistent with them.
    """
    return {AUDIT_SET_KEY: str(path), AUDIT_SET_DIGEST_KEY: sha256_file(path)[:16]}


def require_same_audit_set(
    a: Mapping[str, Any], b: Mapping[str, Any], *, label_a: str, label_b: str,
) -> None:
    """Refuse to join two caches scored over DIFFERENT audit sets.

    Recording a provenance value and then never reading it is precisely the
    defect class this module exists to kill — a value accepted and silently
    ignored — so this is COMPARED, not merely stored. Measured before it
    existed: a 4000-row report printed happily with one side stamped
    `audit_set_v1` and the other `audit_set_v9_DIFFERENT`, and the banner did
    not even show the field.

    Compares the CONTENT DIGEST, so the same file reached by a relative and an
    absolute path is accepted and two different files are refused however they
    are spelled. Absent on either side is a REFUSAL, not a pass, for the same
    reason the version fields are: a cache whose scoring set is unrecorded is a
    cache whose scoring set is unknown.
    """
    da, db = a.get(AUDIT_SET_DIGEST_KEY), b.get(AUDIT_SET_DIGEST_KEY)
    if da is not None and da == db:
        return
    if da is None or db is None:
        detail = (
            f"{label_a} records {da!r} and {label_b} records {db!r} for "
            f"'{AUDIT_SET_DIGEST_KEY}'"
        )
    else:
        detail = (
            f"{label_a} scored {a.get(AUDIT_SET_KEY)!r} ({da}), "
            f"{label_b} scored {b.get(AUDIT_SET_KEY)!r} ({db})"
        )
    raise AuditCacheError(
        f"these two caches were not scored over the same audit set — {detail}.\n"
        "  Joining them would pair positions from different label sets and "
        "report the difference\n"
        "  as if it were a difference between the NETS. Re-run both sides "
        "against one audit set."
    )


def stamp_summary(stamp: Mapping[str, Any], fields: Sequence[str] = ()) -> str:
    """One-line provenance banner for a report header."""
    keys = list(fields) or ["policy_map_version", "audit_ruler_version"]
    return " ".join(f"{k}={stamp.get(k)!r}" for k in keys)
