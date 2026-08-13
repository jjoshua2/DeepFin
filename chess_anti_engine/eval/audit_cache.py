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

__all__ = [
    "AUDIT_CACHE_FORMAT",
    "STAMP_FORMAT_KEY",
    "AuditCacheError",
    "audit_cache_stamp",
    "audit_ruler_version",
    "ensure_cache_writable",
    "policy_map_version",
    "read_audit_cache",
    "read_audit_cache_by_key",
    "read_audit_cache_stamp",
    "stamp_summary",
    "write_audit_cache",
]

#: Bumped only when the STAMP's own layout changes, not when a version does.
AUDIT_CACHE_FORMAT = 1

#: Sentinel key that distinguishes the header record from a data row. A data
#: row is keyed by `key`; nothing else in the schema uses this name.
STAMP_FORMAT_KEY = "audit_cache_format"

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
        "policy_map_version": policy_map_version(),
        "audit_ruler_version": audit_ruler_version(),
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
    """Write a stamped cache: header record first, then one row per line."""
    ensure_cache_writable(path, force=force)
    stamp = audit_cache_stamp(**dict(extra or {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(stamp, sort_keys=True) + "\n")
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return stamp


def _reject(path: Path, what: str) -> AuditCacheError:
    return AuditCacheError(f"{path}: {what}\n  {_REGENERATE}")


def read_audit_cache_stamp(path: Path) -> dict[str, Any]:
    """Validate a cache's provenance, reading ONLY its header line.

    Raises `AuditCacheError` on a missing file, an unstamped file, or a stamp
    that does not match the CURRENT policy map / audit ruler. Call this before
    loading anything else, so a stale cache costs no work at all.
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
    """Stamp-checked cache read. The guard fires before any row is parsed."""
    read_audit_cache_stamp(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if lineno == 1 or not line.strip():
                continue
            row: object = json.loads(line)
            if not isinstance(row, dict):
                raise _reject(path, f"line {lineno} is not a JSON object")
            rows.append(row)
    return rows


def read_audit_cache_by_key(path: Path) -> dict[str, dict[str, Any]]:
    """`read_audit_cache` keyed by each row's position `key`."""
    out: dict[str, dict[str, Any]] = {}
    for row in read_audit_cache(path):
        out[str(row["key"])] = row
    return out


def stamp_summary(stamp: Mapping[str, Any], fields: Sequence[str] = ()) -> str:
    """One-line provenance banner for a report header."""
    keys = list(fields) or ["policy_map_version", "audit_ruler_version"]
    return " ".join(f"{k}={stamp.get(k)!r}" for k in keys)
