#!/usr/bin/env python3
"""Repair a schema-1 corpus (bare-FEN rows) to schema 2 (rows with their history).

WHY.  A schema-1 row is a bare FEN.  The deriver encodes it with history slot 0
filled and slots 1..7 plus every repetition plane ZERO, an input distribution
live play never produces (ledger 2026-09-01: the champion flips 46.5% of its
top-1 moves and loses +20.2 cp of regret between that input and the eight real
frames play fills).  The labels are almost all fine: a cold-TT fixed-depth
search differs between ``position fen <fen>`` and ``position fen <root>
moves ...`` on 0/1200 rows WITHOUT a repeated position in the reversible
segment and on 12.5% of rows WITH one (same ledger entry).  So the repair is

* R1 -- rebuild every row's move window from the corpus itself and bank it as
  schema 2, exactly as the generator would have (its own ``history_for`` on a
  replayed board, so the root definition is the generator's and not a copy);
* R2 -- PARKED (ledger 2026-09-01 "OPERATOR DECISIONS").  The original
  Stockfish labels are copied BYTE-FOR-BYTE; every row is TAGGED instead
  (``rep_in_window`` / ``rep_in_segment`` / ``cur_position_repeat_count`` /
  ``label_regime``) so a consumer knows which labels a history-aware search
  could have moved.  Measured: re-searching a repeat row cold changes its top-1
  on 53.85% of rows, of which only ~13% is the history -- the rest is the
  generator's carried transposition table, which no cheap warm-up reproduces
  (``tt_warmup_check``).  The re-label path stays behind ``--relabel
  {off,banked_window,segment}``, default ``off``, for a future R2 that
  replays the original search prefix.

TWO MODES, AND THE DEFAULT IS THE NARROW ONE.  Arm B needs exactly one thing
from this tool: a correct FULL repair of the known production corpus.  So the
default is a fail-closed PRODUCTION MODE -- the corpus's entire recorded
inventory, every listed shard held to its claimed row count in both
directions, the production opening book required and pinned by sha256
(``PRODUCTION_BOOK_SHA256``), labels byte-for-byte, row schema and the
repetition-plane regime pinned and stamped, and a ``summary.json`` that says
``run_finished: true`` because nothing was refused or skipped.  Everything
optional -- ``--shards``/``--workers`` slicing, a ``--book`` override, a
``--book-sha256`` other than the pin, ``--relabel``, ``--report-json`` -- is
REFUSED BY NAME unless ``--audit-mode`` is given, and ``--audit-mode`` stamps
``production: false`` into the manifest and the summary (a slice also writes
``run_finished: false`` with a ``partial_repair`` block) so a downstream
consumer can refuse it.  Shards on disk that the inventory does not list (a
paused run's in-flight shards) are recorded -- names, decodable rows and
games -- and never repaired in either mode: the corpus does not claim them.

WHERE THE HISTORY COMES FROM.  Rows chain: ``prev.fen + prev.played_move`` is
``fen`` within one ``(worker_id, game_id)``.  Three things break the chain and
each is handled without guessing:

* THE BOOK.  Every game's opening was sampled with ``book_rng(seed, worker,
  game)`` and no wall clock, so ``sample_starting_board`` re-derives the start
  board WITH its book move stack.  It is trusted only when it is VERIFIED --
  the resampled start FEN equals the game's ply-0 row exactly (505/505 games
  on the audit sample) -- or, when the ply-0 row itself was dedup-served, when
  a UNIQUE legal path of at most ``MAX_BRIDGE_PLIES`` joins the resampled
  start to the first banked row.  Otherwise every row that needs a position
  before ply 0 is quarantined ``no_source_book_mismatch``.
* DEDUP-DROPPED PLIES.  A position already searched was served from the cache
  and never banked, so its ``played_move`` is missing.  The gap is BRIDGED by
  enumerating every legal move sequence of the gap's length from the position
  after the last banked move to the next banked FEN.  Exactly one path is an
  exact reconstruction (every intermediate position is determined); more than
  one taints every row whose window spans the gap (``ambiguous_multi_path`` --
  the intermediate frames differ, and no plausible path is ever picked); none,
  or a gap longer than ``MAX_BRIDGE_PLIES``, is ``unbridged_gap``.
* A CHAIN MISMATCH -- consecutive plies whose FENs do not chain -- is a corpus
  fault and is quarantined as ``chain_mismatch`` for every row that needs it.

THE CRITERION IS EXACT EQUIVALENCE WITH LIVE PLAY, never "has 7 frames".  A
row is written only when the board it is banked from is the board live play
had -- the game's true stack, book moves included -- so a short TRUE history
(a book line shorter than 7 plies) is kept and encodes with the live fill
semantics.  Every written row re-verifies that replaying ``history_uci`` from
``history_root_fen`` reproduces ``fen``.

PROVENANCE.  The input shards are never modified.  The output is a sibling
directory of schema-2 shards under the input names, a ``summary.json`` the
deriver reads (``derive_corpus_targets.py``), ``repair_manifest.json`` with the
per-class counts and the input manifest's sha256, per-worker
``repair_rows-wNN.jsonl.zst`` files mapping EVERY input row
``(worker_id, game_id, ply)`` to ``repaired`` / ``relabeled`` /
``quarantined:<reason>``, and ``relabel_audit.jsonl`` with the old and new
top-1 of every re-labeled row.

PARALLELISM.  ``--procs`` runs whole WORKERS in parallel: each worker's shards
are processed in order with a rolling per-game buffer (a game's rows only ever
look backward, so a row is repaired as soon as it is read), and each worker
process owns one engine for its re-labels.  The opening book is warmed ONCE in
the parent (~70 s for the production book) and inherited across the fork.

Usage::

    PYTHONPATH=. python3 scripts/repair_corpus_history.py \\
        --in data/nnue_bootstrap/run03 --out data/nnue_bootstrap/run03_s2 --procs 14
    PYTHONPATH=. python3 scripts/repair_corpus_history.py --audit-mode \\
        --in data/nnue_bootstrap/run03 --audit-only --workers 3 --shards 100-102
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import logging
import multiprocessing
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import chess

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.selfplay.opening import (
    OpeningConfig,
    sample_starting_board,
    warm_opening_book_cache,
)
from chess_anti_engine.stockfish.uci import StockfishTimeoutError, StockfishUCI
from scripts import audit_targets
from scripts import derive_corpus_targets as derive
from scripts import gen_sf_rooted_corpus as corpus

_LOG = logging.getLogger("repair_corpus_history")

#: The schema this tool reads and the one it writes.
SOURCE_SCHEMA = 1
TARGET_SCHEMA = corpus.ROW_SCHEMA

#: Longest dedup gap that is bridged by path enumeration.  Beyond it the
#: enumeration is both expensive and, on the audit sample, never unique.
MAX_BRIDGE_PLIES = 3
#: The enumeration stops as soon as a SECOND path is found: the verdict is
#: "unique or not", and a third path changes nothing.
BRIDGE_PATH_CAP = 2

#: Row classes in the provenance map.
CLASS_REPAIRED = "repaired"
CLASS_RELABELED = "relabeled"
QUARANTINE_PREFIX = "quarantined:"
REASON_AMBIGUOUS = "ambiguous_multi_path"
REASON_UNBRIDGED = "unbridged_gap"
REASON_CHAIN_MISMATCH = "chain_mismatch"
REASON_BOOK_MISMATCH = "no_source_book_mismatch"
#: An anchorless rebuild whose root could carry a pseudo-legal-only ep square
#: (the ply before it is unresolved, and the side that moved into it has a
#: pawn on its fourth rank with both squares behind it empty).  The live
#: board would have ``ep_square`` set where ``chess.Board(fen)`` has none; a
#: later return to the same placement then keys and encodes differently.
REASON_ROOT_EP = "root_ep_unverifiable"
REASON_RELABEL_TIMEOUT = "relabel_timeout"

#: ``repair.history`` values.
HISTORY_CHAINED = "chained"
HISTORY_CHAINED_BOOK = "chained+book"
HISTORY_CHAINED_BRIDGED = "chained+bridged"
HISTORY_CHAINED_BOOK_BRIDGED = "chained+book+bridged"
LABEL_ORIGINAL = "original"
LABEL_RELABELED = "relabeled"

#: Per-game book verdicts (counted in the manifest).
BOOK_EXACT = "exact"
BOOK_BRIDGED = "bridged_from_start"
BOOK_MISMATCH = "mismatch"
BOOK_UNRESOLVED = "start_gap_unresolved"

REPAIR_MANIFEST_NAME = "repair_manifest.json"
RELABEL_AUDIT_NAME = "relabel_audit.jsonl"
#: The per-row class maps live in a subdirectory: the deriver's inventory
#: check reads every ``*.jsonl.zst`` beside the shards as a shard.
PROVENANCE_DIR = "provenance"
ROW_MAP_TEMPLATE = "repair_rows-w{worker_id:02d}.jsonl.zst"
RELABEL_AUDIT_TEMPLATE = "relabel_audit-w{worker_id:02d}.jsonl"

DEFAULT_SF_SEARCH_TIMEOUT_S = 30.0
DEFAULT_BENCH_ENCODE_ROWS = 2000

#: ``--relabel``.  ``off`` (the default, and the operator's decision) keeps
#: every original label byte-for-byte.  ``window`` re-searches rows with a
#: repeated transposition key anywhere in the banked ``[root, row]`` window;
#: ``segment`` only rows with a repeat in their own reversible run (the last
#: ``halfmove_clock`` plies, all the engine's repetition scan can see).
#: Measured on run03 w03-w05 shards 100-102: window 2.50% of rows, segment
#: 1.64%.  ⚑ Neither reproduces the generator's carried-TT label (see the
#: module docstring); both are kept for a future R2 that does.
RELABEL_OFF = "off"
#: Named after the TAG each mode selects by (``rep_in_banked_window`` /
#: ``rep_in_segment``), so "window" cannot mean two things.
RELABEL_WINDOW = "banked_window"
RELABEL_SEGMENT = "segment"
RELABEL_MODES = (RELABEL_OFF, RELABEL_WINDOW, RELABEL_SEGMENT)

#: ``label_regime`` on every repaired row whose label was copied: the search
#: ran on the generator's carried transposition table and was sent the bare
#: FEN, i.e. it never saw the history the row now carries.
LABEL_REGIME_CARRIED_BLIND = "carried_tt_history_blind"
#: The generator re-ran this row's search on a FRESH engine after a wedge
#: (``cold_tt_retry``), or the row's run block says the table was cleared
#: mid-position: the label is history-blind AND cold.
LABEL_REGIME_COLD_BLIND = "cold_tt_history_blind"
LABEL_REGIME_COLD_HISTORY = "cold_tt_history_aware"

#: ⚑ THE ENCODER REGIME ``input_key`` IS HASHED UNDER, stamped in the manifest,
#: the summary and every row's ``run`` block under the GENERATOR's key name --
#: the one the deriver refuses by.  The C play-path encoder (``corpus.row_key``
#: -> ``encode_cboard``) reads a PROCESS-GLOBAL flag, ``history_rep_fix``: off,
#: its repetition planes only see partners inside the kept hash window; on, a
#: per-slot flag recorded at push time with full look-back.  Production and
#: the deriver run FIXED, so a key hashed in a fresh process that never
#: applied the flag disagrees with the deriver on every row whose repeat
#: partner sits more than the window back (measured 2026-09-01 on run03 w03
#: 100-102: 77/24,590 rows), and the deriver refuses the corpus.  ``run``
#: applies it through the generator's own ``apply_history_rep_fix`` before any
#: board exists, EVERY worker applies it again on entry (the pool forks, so
#: the parent's C global is inherited -- applied anyway, so a spawn context
#: would not silently un-fix it), and ``corpus.row_key`` itself refuses to
#: hash under any other regime.  The stamp is read off the REALIZED flag.
KEY_HISTORY_REP_FIX = corpus.KEY_HISTORY_REP_FIX
HISTORY_REP_FIX = corpus.HISTORY_REP_FIX

#: ⚑ THE PRODUCTION OPENING BOOK, PINNED BY HASH.  Every legacy corpus (run02
#: and its snapshot, run03, run04, run05) names this file in
#: ``config_requested.book``; production mode refuses any book that does not
#: hash to it, so a replaced book can never resample "plausible" openings.
#: ⚑ WHAT THIS PIN IS AND IS NOT: no independent historical record of the
#: book's hash exists -- the run manifests store its PATH only, and neither
#: git history nor the ledger ever recorded a digest -- so this is today's
#: measurement (2026-09-02), not retroactive cryptographic proof.  The
#: evidence that it is the book the runs sampled from is recorded beside it
#: in ``PRODUCTION_BOOK_PROVENANCE`` and stamped into every manifest.
PRODUCTION_BOOK_NAME = (
    "8moves_v3_plus_policybeam_final145cp_plus_uho2024_060_110_plus_2move_thinbeam_dedup.pgn.zip"
)
PRODUCTION_BOOK_SHA256 = "70d0dfa50a6b1191f1db702a093a4911b7433c612a30599517c2c2d92f0cea7c"
PRODUCTION_BOOK_SIZE = 9_663_653
PRODUCTION_BOOK_PROVENANCE: dict[str, Any] = {
    "historical_hash": None,
    "hash_measured": "2026-09-02",
    "evidence": [
        "file mtime 2026-05-28 17:31 local, unchanged since; every legacy run "
        "started later (run02 2026-08-28, run03 08-30, run04 08-31, run05 09-01)",
        "functional: the resampled book reproduces the banked ply-0 FEN of every "
        "audited game (505/505 on the calibration audit, 158/158 on the run03 "
        "w03 100-102 + 207 slice)",
        "the run manifests name this file by path; none records a digest",
    ],
}


#: Where a manifest's RELATIVE book path is resolved from (the generator was
#: launched from the repo root and stored the path as typed).
REPO_ROOT = Path(__file__).resolve().parents[1]


class RepairError(RuntimeError):
    """The corpus is not what the repair was told it is; nothing is written."""


# -- the spec -----------------------------------------------------------------


@dataclass(frozen=True)
class RepairSpec:
    """Everything a worker process needs.  Frozen and picklable across a fork."""

    in_dir: Path
    out_dir: Path | None
    seed: int
    opening: OpeningConfig
    staircase: str
    cp_slope: float
    cp_draw_width: float
    sf_binary: str
    sf_hash_mb: int
    sf_read_timeout_s: float
    sf_search_timeout_s: float
    syzygy_path: str
    nice: int
    audit_only: bool
    shard_indices: tuple[int, ...] | None
    bench_encode_rows: int
    #: The shards this run may read, BY NAME, from the corpus's own inventory
    #: (``derive.read_corpus_record``) -- never a glob, so a paused run's
    #: in-flight or abandoned last shard is not repaired.
    listed_shards: tuple[tuple[str, int], ...] = ()
    #: The deriver's inventory reading this repair was cut from.
    progress_files: tuple[str, ...] = ()
    progress_torn_tail_files: tuple[str, ...] = ()
    #: Shard files on disk that the inventory does not list (a paused run's
    #: in-flight shards).  Recorded with their decodable rows and games;
    #: NEVER repaired -- the corpus does not claim them.
    unlisted_shards: tuple[str, ...] = ()
    #: ``--audit-mode``: the optional surface is allowed and the records say
    #: ``production: false``.  Off (the default) is production mode.
    audit_mode: bool = False
    #: ``--shards``/``--workers`` restricted the input (audit mode only): the
    #: output is NOT the whole corpus and ``summary.json`` says
    #: ``run_finished: false`` with a ``partial_repair`` block.
    partial: bool = False
    #: One of ``RELABEL_MODES``; ``off`` copies every label byte-for-byte.
    relabel: str = RELABEL_OFF


def opening_config_from_manifest(
    requested: Mapping[str, Any], *, book_override: str | None = None,
) -> OpeningConfig:
    """The run's opening sampler config, spelled the way ``build_opening_config`` spells it."""
    book = book_override if book_override is not None else requested.get("book")
    if not book:
        raise RepairError(
            "the input manifest's config_requested names no opening book; a "
            "bookless run has no book stack to re-derive and this tool has no "
            "other source for pre-ply-0 positions",
        )
    book_path = Path(str(book))
    if not book_path.is_absolute():
        # run05's manifest stores the book relative to the repo root the
        # generator was launched from; resolve against THIS checkout's root.
        book_path = REPO_ROOT / book_path
    if not book_path.exists():
        raise RepairError(
            f"opening book {book!r} does not exist (resolved to {book_path}; a "
            f"relative manifest path is taken against the repo root {REPO_ROOT}). "
            "Pass --book to point at the SAME book file the run sampled from",
        )
    return OpeningConfig(
        opening_book_path=str(book_path),
        opening_book_max_plies=int(requested["book_plies"]),
        opening_book_max_games=int(requested["book_max_games"]),
        opening_book_prob=1.0,
    )


# -- shard inventory ------------------------------------------------------------


def corpus_inventory(
    in_dir: Path,
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...], derive.ProgressInventory]:
    """``(listed (shard name, rows claimed), unlisted shard names on disk, the
    deriver's inventory)``.

    ⚑ THE DERIVER'S OWN READER, ``derive.read_progress_inventory``: the shards
    the ``w*.progress.jsonl`` files claim, with the rows each record claims
    (the number every listed shard is held to in ``repair_worker``), the
    progress files whose final line was torn, and whatever is on disk that no
    record lists (a paused run's in-flight shard, a killed run's abandoned
    one) -- reported and never repaired.  Every legacy corpus this tool
    exists for is manifest + progress files (none ended with a
    ``summary.json``), and the deriver reads them the same way.
    """
    try:
        inventory = derive.read_progress_inventory(in_dir)
    except derive.CorpusIntegrityError as exc:
        raise RepairError(f"{in_dir}: {exc}") from exc
    listed = tuple(
        (path.name, int(rows)) for path, rows in zip(inventory.shards, inventory.shard_rows)
    )
    return listed, tuple(inventory.unlisted_on_disk), inventory


def shard_worker_id(name: str) -> int | None:
    """``w03-00100.jsonl.zst`` -> 3, or ``None`` for a name that is not a shard."""
    if not (name.startswith("w") and name.endswith((".jsonl.zst", ".jsonl.gz"))):
        return None
    head = name.split("-", 1)[0][1:]
    return int(head) if head.isdigit() else None


def worker_shards(
    in_dir: Path, worker_id: int, indices: Sequence[int] | None,
    listed: Sequence[tuple[str, int]],
) -> list[tuple[Path, int]]:
    """This worker's LISTED shards ``(path, rows claimed)`` in index order.

    ⚑ ``listed`` is the corpus's own inventory (``read_corpus_record``: the
    summary's shard list, or the progress files' on a run that has not
    ended), never a glob of the directory.  A paused or killed run leaves its
    in-flight shard on disk unlisted -- its zstd/JSON tail possibly torn --
    and a resume deletes and replays it; repairing it would bank rows the
    corpus itself does not claim.
    """
    found: list[tuple[int, Path, int]] = []
    for name, rows in listed:
        if shard_worker_id(name) != int(worker_id):
            continue
        index = corpus.shard_index_of(name)
        if index is None or (indices is not None and index not in indices):
            continue
        found.append((index, in_dir / name, int(rows)))
    return [(path, rows) for _, path, rows in sorted(found)]


def refuse_report_path_inside(
    report_json: str | None, in_dir: Path, out_dir: Path | None,
) -> Path | None:
    """``--report-json`` may not name anything inside the input or the output
    corpus (it would replace a shard or a record after they were written) nor
    an existing file; checked BEFORE any output exists."""
    if not report_json:
        return None
    report_path = Path(report_json).resolve()
    for label, root in (("input", in_dir), ("output", out_dir)):
        if root is not None and root.resolve() in (report_path, *report_path.parents):
            raise RepairError(
                f"--report-json {report_json} lies inside the {label} corpus "
                f"{root}; the report is written last and would replace a record there",
            )
    if report_path.exists():
        raise RepairError(f"--report-json {report_json} already exists; it is never overwritten")
    return report_path


def verify_book(book_path_s: str, book_sha256_arg: str | None) -> dict[str, Any]:
    """The opening book's hash, held to the expected one: the production pin
    unless ``--book-sha256`` names another (audit mode only).  Returns the
    manifest's ``book`` block -- what was measured, what was expected, where
    the expectation came from, and the pin's provenance."""
    book_path = Path(book_path_s)
    sha256 = sha256_of(book_path)
    size = book_path.stat().st_size
    expected = (book_sha256_arg or PRODUCTION_BOOK_SHA256).lower()
    expected_from = "--book-sha256" if book_sha256_arg else "production_pin"
    if sha256 != expected:
        raise RepairError(
            f"opening book {book_path} hashes to {sha256} ({size} bytes), not the "
            f"expected {expected} ({expected_from}; the production pin is "
            f"{PRODUCTION_BOOK_NAME} {PRODUCTION_BOOK_SHA256}, {PRODUCTION_BOOK_SIZE} "
            "bytes); the resampled openings would come from a different book "
            "than the run's",
        )
    return {
        "path": book_path_s, "sha256": sha256, "size": size,
        "expected_sha256": expected, "expected_from": expected_from,
        "matches_production_pin": sha256 == PRODUCTION_BOOK_SHA256,
        "production_pin": {
            "name": PRODUCTION_BOOK_NAME, "sha256": PRODUCTION_BOOK_SHA256,
            "size": PRODUCTION_BOOK_SIZE, **PRODUCTION_BOOK_PROVENANCE,
        },
    }


def worker_ids_in(names: Sequence[str]) -> list[int]:
    ids = {shard_worker_id(name) for name in names}
    return sorted(i for i in ids if i is not None)


def unlisted_entry(shard: Path) -> dict[str, Any]:
    """What an unlisted (in-flight) shard holds, decoded up to its damage --
    recorded so the manifest says what was left behind in rows and games
    rather than in names alone.  Rows are written whole-game at game end
    (``ShardWriter.end_game``), so what a kill tears is the zstd/JSON TAIL;
    nothing in the shard says whether its last decodable game is whole, and
    the corpus does not claim any of it, so none of it is repaired."""
    decodable, damage = iter_decodable_rows(shard)
    return {
        "path": shard.name, "decodable_rows": len(decodable),
        "decodable_games": len({int(r["game_id"]) for r in decodable}),
        "damage": damage,
    }


#: ⚑ THE OPTIONAL SURFACE, refused by name in production mode.  One row per
#: flag: ``(flag, is_set)``.  ``refuse_in_production`` walks the table, so a
#: flag missing from it is a flag production mode accepts -- keep it complete.
AUDIT_MODE_ONLY: tuple[tuple[str, Callable[[argparse.Namespace], bool]], ...] = (
    ("--shards", lambda a: bool(a.shards)),
    ("--workers", lambda a: bool(a.workers)),
    ("--book", lambda a: a.book is not None),
    ("--relabel", lambda a: a.relabel != RELABEL_OFF),
    ("--report-json", lambda a: bool(a.report_json)),
    ("--book-sha256 (other than the production pin)",
     lambda a: bool(a.book_sha256) and a.book_sha256.lower() != PRODUCTION_BOOK_SHA256),
)


def refuse_in_production(args: argparse.Namespace) -> None:
    """Production mode (no ``--audit-mode``) takes the whole recorded corpus
    with the production book and nothing else; each optional flag is refused
    by its own name."""
    if args.audit_mode:
        return
    for flag, is_set in AUDIT_MODE_ONLY:
        if is_set(args):
            raise RepairError(
                f"{flag} is an audit-mode operation and this is a production repair "
                "(the whole recorded inventory, the production book, labels "
                "byte-for-byte); pass --audit-mode to run it, which stamps "
                "production: false into the manifest and the summary",
            )


def iter_decodable_rows(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Every row of a shard up to the first damage, and what the damage was.

    For an UNLISTED shard only: a listed shard is claimed whole and any damage
    in it is a refusal, never a truncation.
    """
    rows: list[dict[str, Any]] = []
    stream = corpus.iter_shard_rows(path)
    while True:
        try:
            rows.append(next(stream))
        except StopIteration:
            return rows, None
        except Exception as exc:  # zstd frame errors and torn JSON lines alike
            return rows, f"{type(exc).__name__}: {str(exc)[:200]}"


def parse_shard_range(spec: str | None) -> tuple[int, ...] | None:
    """``"100-102,110"`` -> ``(100, 101, 102, 110)``; ``None`` means every shard."""
    if spec is None:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if sep:
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"--shards {spec!r} names no shard")
    return tuple(sorted(out))


# -- bridging -------------------------------------------------------------------


def bridge_paths(
    before: chess.Board, target_fen: str, plies: int, *, cap: int = BRIDGE_PATH_CAP,
) -> list[list[str]]:
    """Distinct legal move sequences of exactly ``plies`` from ``before`` to ``target_fen``.

    Stops after ``cap`` paths: the caller only asks whether the path is UNIQUE.
    The one prune is exact -- a capture removes one piece per ply, so a
    position with more surplus pieces than plies left cannot reach the target.
    """
    target_pieces = chess.popcount(chess.Board(target_fen).occupied)
    found: list[list[str]] = []

    def walk(board: chess.Board, depth: int, acc: list[str]) -> None:
        if len(found) >= cap:
            return
        remaining = plies - depth
        if remaining == 0:
            if board.fen() == target_fen:
                found.append(list(acc))
            return
        if chess.popcount(board.occupied) - target_pieces > remaining:
            return
        for move in list(board.legal_moves):
            board.push(move)
            acc.append(move.uci())
            walk(board, depth + 1, acc)
            acc.pop()
            board.pop()
            if len(found) >= cap:
                return

    walk(before.copy(stack=False), 0, [])
    return found


# -- per-game reconstruction ----------------------------------------------------


@dataclass
class GameState:
    """One game's chain, filled as its rows stream past.

    ``fen_at`` / ``move_at`` hold every ply whose position and played move are
    KNOWN -- banked, or bridged through a unique path.  ``gap_reason`` names
    every ply that is not, and ``bridged`` marks the ones a bridge supplied.
    ``book_moves`` is the verified book stack, or ``None`` when no position
    before ply 0 may be used.
    """

    worker_id: int
    game_id: int
    book_status: str
    book_moves: list[str] | None
    fen_at: dict[int, str] = field(default_factory=dict)
    move_at: dict[int, str] = field(default_factory=dict)
    gap_reason: dict[int, str] = field(default_factory=dict)
    bridged: set[int] = field(default_factory=set)
    last_ply: int = -1

    def fill_gap(self, start: int, stop: int, before: chess.Board, target_fen: str) -> None:
        """Resolve plies ``[start, stop)`` between ``before`` and the banked ``target_fen``."""
        gap = stop - start
        if gap > MAX_BRIDGE_PLIES:
            self.mark_gap(start, stop, REASON_UNBRIDGED)
            return
        paths = bridge_paths(before, target_fen, gap)
        if len(paths) != 1:
            self.mark_gap(
                start, stop, REASON_AMBIGUOUS if len(paths) > 1 else REASON_UNBRIDGED,
            )
            return
        board = before.copy(stack=False)
        for offset, uci in enumerate(paths[0]):
            ply = start + offset
            self.fen_at[ply] = board.fen()
            self.move_at[ply] = uci
            self.bridged.add(ply)
            board.push(chess.Move.from_uci(uci))

    def mark_gap(self, start: int, stop: int, reason: str) -> None:
        for ply in range(start, stop):
            self.gap_reason[ply] = reason

    def unresolved_in(self, start: int, stop: int) -> str | None:
        """The quarantine reason for a window over ``[start, stop)``, or ``None``."""
        reasons = {self.gap_reason[p] for p in range(start, stop) if p in self.gap_reason}
        if not reasons:
            return None
        if REASON_AMBIGUOUS in reasons:
            return REASON_AMBIGUOUS
        if REASON_CHAIN_MISMATCH in reasons:
            return REASON_CHAIN_MISMATCH
        return REASON_UNBRIDGED

    def any_bridged_in(self, start: int, stop: int) -> bool:
        return any(p in self.bridged for p in range(start, stop))


def fen_halfmove_clock(fen: str) -> int:
    return int(fen.split(" ")[4])


@dataclass(frozen=True)
class RowRepair:
    """What R1 decided for one row."""

    #: ``CLASS_REPAIRED`` or a ``quarantined:<reason>`` class.
    row_class: str
    #: The board live play had, with its full stack (repaired rows only).
    board: chess.Board | None
    history: corpus.RowHistory | None
    history_kind: str | None
    #: The repetition tags (see :class:`RepeatTags`); empty on a quarantined row.
    tags: RepeatTags
    #: The board was rebuilt from the root position itself because the ply
    #: before it is unresolved (see ``GameReconstructor._classify``).
    anchorless: bool = False

    @property
    def repaired(self) -> bool:
        return self.board is not None

    def flagged(self, mode: str) -> bool:
        if mode == RELABEL_WINDOW:
            return self.tags.banked_window
        if mode == RELABEL_SEGMENT:
            return self.tags.segment
        return False


@dataclass(frozen=True)
class RepeatTags:
    """What a history-aware reader could see that the banked label did not.

    All computed on the REPLAY of the banked window, so they describe the
    row as written.  ``frames`` (row tag ``rep_in_frames8``) is a repeat AMONG
    the 8 encoder frames (the position and the 7 before it; a repetition
    plane can also be set by a partner further back inside the window, so
    this is narrower than "any plane set"); ``segment`` is over the
    row's own reversible run (the last ``halfmove_clock`` plies, which is all
    the engine's repetition scan reads); ``banked_window`` (manifest count
    ``rep_in_banked_window``, the ``--relabel banked_window`` scope) is over
    the whole ``[root, row]`` window; ``cur_count`` is how many times the row's own
    position occurs in the segment, itself included (2 = a repetition claim
    is one move away, 3 = threefold).
    """

    banked_window: bool = False
    frames: bool = False
    segment: bool = False
    cur_count: int = 0

    def as_row_fields(self) -> dict[str, Any]:
        return {
            "rep_in_frames8": self.frames,
            "rep_in_segment": self.segment,
            "cur_position_repeat_count": self.cur_count,
        }


def quarantined(reason: str) -> RowRepair:
    return RowRepair(
        row_class=QUARANTINE_PREFIX + reason, board=None, history=None,
        history_kind=None, tags=RepeatTags(),
    )


def repeats_in(history: corpus.RowHistory, *, halfmove_clock: int) -> RepeatTags:
    """The row's repetition tags, from the replayed window.

    ⚑ The ROOT position is in every scan: the commonest shuffle is a clock-0
    position (the window's root) recurring four plies later, and a scan that
    started one position late would tag none of them.
    """
    board = chess.Board(history.root_fen)
    keys = [board._transposition_key()]
    for uci in history.uci:
        board.push(chess.Move.from_uci(uci))
        keys.append(board._transposition_key())
    frames = keys[-(corpus.HISTORY_WINDOW_PLIES + 1):]
    segment = keys[max(0, len(keys) - 1 - int(halfmove_clock)):]
    return RepeatTags(
        banked_window=len(set(keys)) != len(keys),
        frames=len(set(frames)) != len(frames),
        segment=len(set(segment)) != len(segment),
        cur_count=segment.count(keys[-1]),
    )


def could_have_double_stepped(board: chess.Board) -> bool:
    """Could the side that just moved have made a double pawn step into ``board``?

    A pawn of the side NOT to move sits on its fourth rank with the two
    squares behind it empty.  When this is false the position can carry no
    ep square at all, so the banked FEN (ep printed only when a capture is
    legal) and the live ``fen(en_passant="fen")`` are the same string.
    """
    mover = not board.turn
    rank = 3 if mover == chess.WHITE else 4
    step = -1 if mover == chess.WHITE else 1
    for square in chess.SquareSet(board.pieces(chess.PAWN, mover) & chess.BB_RANKS[rank]):
        file = chess.square_file(square)
        behind = chess.square(file, rank + step)
        origin = chess.square(file, rank + 2 * step)
        if board.piece_at(behind) is None and board.piece_at(origin) is None:
            return True
    return False


class GameReconstructor:
    """Streams one worker's rows, game by game, and repairs each row as it arrives."""

    def __init__(self, spec: RepairSpec, worker_id: int) -> None:
        self.spec = spec
        self.worker_id = int(worker_id)
        self.game: GameState | None = None
        self.seen_games: set[int] = set()
        self.book_games: Counter[str] = Counter()

    def _start_game(self, row: Mapping[str, Any]) -> GameState:
        game_id = int(row["game_id"])
        if game_id in self.seen_games:
            raise RepairError(
                f"worker {self.worker_id} game {game_id} reappears after another "
                "game's rows; games are banked contiguously and a split one "
                "cannot be chained",
            )
        self.seen_games.add(game_id)
        start = sample_starting_board(
            rng=corpus.book_rng(seed=self.spec.seed, worker_id=self.worker_id, game_id=game_id),
            cfg=self.spec.opening,
        )
        book_moves = [m.uci() for m in start.board.move_stack]
        start_fen = start.board.fen()
        ply = int(row["ply"])
        fen = str(row["fen"])
        if ply == 0:
            if start_fen == fen:
                game = GameState(self.worker_id, game_id, BOOK_EXACT, book_moves)
            else:
                game = GameState(self.worker_id, game_id, BOOK_MISMATCH, None)
        else:
            # The ply-0 position was dedup-served.  The only verification left
            # is the bridge itself: a unique path from the resampled start.
            game = GameState(self.worker_id, game_id, BOOK_BRIDGED, book_moves)
            game.fill_gap(0, ply, start.board, fen)
            if 0 in game.gap_reason:
                game.book_status = BOOK_UNRESOLVED
                game.book_moves = None
            game.last_ply = ply - 1
        self.book_games[game.book_status] += 1
        return game

    def _extend(self, game: GameState, row: Mapping[str, Any]) -> None:
        ply = int(row["ply"])
        fen = str(row["fen"])
        if ply <= game.last_ply:
            raise RepairError(
                f"worker {self.worker_id} game {game.game_id}: ply {ply} after "
                f"ply {game.last_ply}; rows of a game are banked in ply order",
            )
        prev = game.last_ply
        if prev >= 0 and prev in game.move_at:
            before = chess.Board(game.fen_at[prev])
            try:
                before.push(chess.Move.from_uci(game.move_at[prev]))
            except (ValueError, AssertionError):
                before = None
            if before is None:
                game.mark_gap(prev, ply, REASON_CHAIN_MISMATCH)
                del game.move_at[prev]
            elif ply == prev + 1:
                if before.fen() != fen:
                    game.mark_gap(prev, ply, REASON_CHAIN_MISMATCH)
                    del game.move_at[prev]
            else:
                game.fill_gap(prev + 1, ply, before, fen)
        elif prev >= 0:
            # The previous ply is itself unresolved; nothing to bridge from.
            game.mark_gap(prev + 1, ply, game.gap_reason.get(prev, REASON_UNBRIDGED))
        game.fen_at[ply] = fen
        game.move_at[ply] = str(row["played_move"])
        game.last_ply = ply

    def repair(self, row: Mapping[str, Any]) -> RowRepair:
        if int(row.get("schema", SOURCE_SCHEMA)) != SOURCE_SCHEMA:
            raise RepairError(
                f"worker {self.worker_id} game {row.get('game_id')} ply "
                f"{row.get('ply')} is schema {row.get('schema')}, not "
                f"{SOURCE_SCHEMA}; this tool repairs bare-FEN rows only",
            )
        if int(row["worker_id"]) != self.worker_id:
            raise RepairError(
                f"a row of worker {row['worker_id']} is in worker "
                f"{self.worker_id}'s shard",
            )
        game_id = int(row["game_id"])
        if self.game is None or self.game.game_id != game_id:
            self.game = self._start_game(row)
        game = self.game
        self._extend(game, row)
        return self._classify(game, int(row["ply"]))

    def _classify(self, game: GameState, ply: int) -> RowRepair:
        """The generator's root definition, evaluated on the chain.

        ``P`` is the position ``HISTORY_WINDOW_PLIES`` back; the root is ``P``
        walked back ``halfmove_clock(P)`` plies.  Three rebuilds, tried in
        order, and each yields the board live play had:

        * ANCHORED -- root at ply ``r >= 1`` and ply ``r - 1`` known: the board
          is rebuilt from ``r - 1``, so ``history_for`` sees a non-empty stack
          at the root (its reason is ``irreversible``, as live) and the root's
          raw ep square survives the push.
        * BOOK -- the root is at or before ply 0: rebuilt from the standard
          start through the verified book stack, exactly as the worker's board
          was.
        * ANCHORLESS -- the root is known but the ply before it is not (an
          unresolved gap ends exactly at the root): rebuilt from the root
          position itself.  ``history_for`` then reads the root as the game
          start, so the reason is set to ``irreversible`` explicitly (the
          chain proves it is not the game start, and its clock is 0).  What
          can differ from live is a pseudo-legal-only ep square on the root:
          ``chess.Board(fen)`` has none where the live board has one, so a
          later return to the same placement is a repetition on one board
          and not the other -- a different plane and a different
          ``input_key`` -- and Stockfish's ``Position::set``/``do_move`` key
          the square on the pseudo-legal criterion too.  Whenever the root
          COULD have been reached by a double pawn step the row is
          QUARANTINED (``root_ep_unverifiable``); only a root that provably
          carries no ep square is rebuilt this way.
        """
        window = corpus.HISTORY_WINDOW_PLIES
        p_ply = ply - window
        root: int | None = None
        if p_ply >= 0:
            if p_ply not in game.fen_at:
                return quarantined(game.unresolved_in(p_ply, ply) or REASON_UNBRIDGED)
            root = p_ply - fen_halfmove_clock(game.fen_at[p_ply])
        anchorless = False
        if root is not None and root >= 1 and (root - 1) not in game.gap_reason:
            anchor = root - 1
            reason = game.unresolved_in(anchor, ply)
            if reason is not None:
                return quarantined(reason)
            board = chess.Board(game.fen_at[anchor])
            for p in range(anchor, ply):
                board.push(chess.Move.from_uci(game.move_at[p]))
            bridged = game.any_bridged_in(anchor, ply)
            kind = HISTORY_CHAINED_BRIDGED if bridged else HISTORY_CHAINED
        elif (root is None or root <= 0) and game.book_moves is not None:
            reason = game.unresolved_in(0, ply)
            if reason is not None:
                return quarantined(reason)
            board = chess.Board()
            for uci in game.book_moves:
                board.push(chess.Move.from_uci(uci))
            for p in range(ply):
                board.push(chess.Move.from_uci(game.move_at[p]))
            bridged = game.any_bridged_in(0, ply)
            kind = HISTORY_CHAINED_BOOK_BRIDGED if bridged else HISTORY_CHAINED_BOOK
        elif root is not None and root >= 0 and root in game.fen_at:
            reason = game.unresolved_in(root, ply)
            if reason is not None:
                return quarantined(reason)
            board = chess.Board(game.fen_at[root])
            for p in range(root, ply):
                board.push(chess.Move.from_uci(game.move_at[p]))
            bridged = game.any_bridged_in(root, ply)
            kind = HISTORY_CHAINED_BRIDGED if bridged else HISTORY_CHAINED
            anchorless = True
        elif root is not None and root >= 0:
            return quarantined(game.unresolved_in(root, ply) or REASON_UNBRIDGED)
        else:
            reason = game.unresolved_in(0, ply)
            return quarantined(reason if reason is not None else REASON_BOOK_MISMATCH)
        if board.fen() != game.fen_at[ply]:
            raise RepairError(
                f"worker {self.worker_id} game {game.game_id} ply {ply}: the "
                f"rebuilt board is {board.fen()!r}, not the row's "
                f"{game.fen_at[ply]!r}",
            )
        history = corpus.history_for(board)
        if root is not None and history.plies != ply - root:
            raise RepairError(
                f"worker {self.worker_id} game {game.game_id} ply {ply}: the "
                f"generator's window has {history.plies} plies where the chain "
                f"placed the root {ply - root} plies back",
            )
        if anchorless:
            if could_have_double_stepped(chess.Board(history.root_fen)):
                return quarantined(REASON_ROOT_EP)
            history = dataclasses.replace(history, reason=corpus.HISTORY_ROOT_IRREVERSIBLE)
        tags = repeats_in(history, halfmove_clock=fen_halfmove_clock(game.fen_at[ply]))
        return RowRepair(
            row_class=CLASS_REPAIRED, board=board, history=history, history_kind=kind,
            tags=tags, anchorless=anchorless,
        )


# -- output -----------------------------------------------------------------------


class ZstdLines:
    """A ``.jsonl.zst`` writer opened ``"xb"`` -- a rerun never overwrites."""

    def __init__(self, path: Path) -> None:
        module = corpus.zstandard_module()
        if module is None:
            raise RepairError("the zstandard module is not importable")
        self.path = path
        self._binary = open(path, "xb")  # noqa: SIM115 - closed in close()
        self._raw = module.ZstdCompressor().stream_writer(self._binary)
        self._text = io.TextIOWrapper(self._raw, encoding="utf-8")
        self.rows = 0

    def write(self, record: Mapping[str, Any]) -> None:
        self._text.write(json.dumps(record, sort_keys=True) + "\n")
        self.rows += 1

    def close(self) -> None:
        # The same order ``ShardWriter.close`` uses: the text layer flushes
        # into the compressor, the compressor writes its end-of-frame.
        self._text.close()
        self._raw.close()
        self._binary.close()


def label_regime(row: Mapping[str, Any], *, relabeled: bool) -> str:
    """Where this row's label came from, for the consumer that reads it.

    Cold is decided from ``cold_tt_retry`` ALONE: the generator sets it on
    the one row whose search was re-run on a fresh engine after a wedge, so
    it is the per-row fact.  ⚑ ``run[tt_carried_across_phases]`` is NOT a
    per-row fact and is deliberately not read here: the generator stamps it
    from ``searcher.tt_cleared_mid_position == 0``, a per-WORKER counter
    that never resets, so after a worker's first wedge every later row
    carries ``False`` although its search ran on the replacement engine's
    carried table like any other.  It means "no wedge had happened in this
    worker before this row", and it is carried through verbatim in ``run``
    for the reader that wants that.
    """
    if relabeled:
        return LABEL_REGIME_COLD_HISTORY
    if bool(row.get("cold_tt_retry")):
        return LABEL_REGIME_COLD_BLIND
    return LABEL_REGIME_CARRIED_BLIND


def output_shard_name(name: str) -> str:
    """The output is always zstd, whatever codec the input shard used."""
    for suffix in (".jsonl.zst", ".jsonl.gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".jsonl.zst"
    raise RepairError(f"{name!r} is not a corpus shard name")


def top1_of_phases(phases: Sequence[Mapping[str, Any]]) -> tuple[str | None, float | None]:
    """The rank-1 move and its cp in phase 0's deepest complete block."""
    if not phases:
        return None, None
    blocks = list(phases[0].get("per_depth", []))
    complete = [b for b in blocks if b.get("complete")] or blocks
    if not complete:
        return None, None
    lines = complete[-1].get("lines", [])
    if not lines:
        return None, None
    return str(lines[0][1]), float(lines[0][2])


def repaired_row(
    row: Mapping[str, Any], repair: RowRepair, *, phases: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """The schema-2 row: the input row, the window, the provenance block."""
    if repair.history is None or repair.history_kind is None:
        raise RepairError(f"game {row.get('game_id')} ply {row.get('ply')}: a quarantined row cannot be written")
    if repair.board is None:
        raise RepairError(f"game {row.get('game_id')} ply {row.get('ply')}: no board to key")
    # ⚑ ``dict(row)``: every original value is the SAME object the input row
    # held -- ``phases`` included -- so the labels are copied, never rebuilt.
    # Only the schema-2 additions and the tags are new keys.
    out = dict(row)
    out["schema"] = TARGET_SCHEMA
    out.update(repair.history.as_row_fields())
    # The two identities the #497 fix round banks on every schema-2 row, from
    # the generator's own helpers on the rebuilt live board; the deriver
    # refuses a schema-2 row whose reconstructed tensor does not hash to
    # ``input_key``.
    out["input_key"] = corpus.row_key(repair.board)
    out["search_key"] = corpus.search_key(repair.board)
    out.update(repair.tags.as_row_fields())
    out["label_regime"] = label_regime(row, relabeled=phases is not None)
    # The encoder regime the key above was hashed under, READ off the flag
    # in force (never the constant), beside the run stamps the deriver reads.
    run_block = row.get("run")
    if not isinstance(run_block, Mapping):
        raise RepairError(f"game {row.get('game_id')} ply {row.get('ply')}: no run block")
    out["run"] = {**run_block, KEY_HISTORY_REP_FIX: bool(rep_fix.current())}
    out["repair"] = {
        "source_schema": SOURCE_SCHEMA,
        "history": repair.history_kind,
        "label": LABEL_RELABELED if phases is not None else LABEL_ORIGINAL,
    }
    if phases is not None:
        out["phases"] = list(phases)
    # ⚑ Asserted on the ROW, not on the object it was built from: this is the
    # check the deriver will repeat, on the bytes about to be written.
    replayed = chess.Board(str(out["history_root_fen"]))
    for uci in out["history_uci"]:
        replayed.push(chess.Move.from_uci(str(uci)))
    if replayed.fen() != str(out["fen"]):
        raise RepairError(
            f"game {out.get('game_id')} ply {out.get('ply')}: replaying "
            f"{out['history_uci']} from {out['history_root_fen']!r} gives "
            f"{replayed.fen()!r}, not {out['fen']!r}",
        )
    return out


# -- the worker --------------------------------------------------------------------


SearcherFactory = Callable[[corpus.SearchStats], corpus.StaircaseSearcher]


def default_searcher_factory(spec: RepairSpec) -> SearcherFactory:
    staircase = corpus.parse_staircase(spec.staircase)

    def spawn(stats: corpus.SearchStats) -> corpus.StaircaseSearcher:
        return corpus.StaircaseSearcher(
            engine=StockfishUCI(
                spec.sf_binary,
                multipv=1,
                hash_mb=int(spec.sf_hash_mb),
                syzygy_path=spec.syzygy_path,
                nice=int(spec.nice),
                threads=1,
                read_timeout_s=float(spec.sf_read_timeout_s),
            ),
            staircase=staircase,
            cp_slope=spec.cp_slope,
            cp_draw_width=spec.cp_draw_width,
            stats=stats,
            search_timeout_s=float(spec.sf_search_timeout_s),
        )

    return spawn


@dataclass
class WorkerTally:
    """Every counter one worker reports, all events."""

    rows_in: int = 0
    rows_out: int = 0
    classes: Counter[str] = field(default_factory=Counter)
    history_kinds: Counter[str] = field(default_factory=Counter)
    root_reasons: Counter[str] = field(default_factory=Counter)
    history_plies: Counter[str] = field(default_factory=Counter)
    book_games: Counter[str] = field(default_factory=Counter)
    games: int = 0
    rep_in_frames8: int = 0
    rep_in_segment: int = 0
    rep_in_banked_window: int = 0
    cur_position_repeated: int = 0
    anchorless: int = 0
    relabeled: int = 0
    relabel_top1_changed: int = 0
    relabel_timeouts: int = 0
    relabel_s: float = 0.0
    relabel_max_s: float = 0.0
    reconstruct_s: float = 0.0
    encode_rows: int = 0
    encode_s: float = 0.0
    shards: list[dict[str, Any]] = field(default_factory=list)
    label_regimes: Counter[str] = field(default_factory=Counter)

    def summary(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "games": self.games,
            "classes": dict(self.classes),
            "history_kinds": dict(self.history_kinds),
            "history_root_reasons": dict(self.root_reasons),
            "history_plies_histogram": dict(self.history_plies),
            "book_games": dict(self.book_games),
            "tags": {
                "rep_in_frames8": self.rep_in_frames8,
                "rep_in_segment": self.rep_in_segment,
                "rep_in_banked_window": self.rep_in_banked_window,
                "cur_position_repeat_count_ge2": self.cur_position_repeated,
            },
            "anchorless": self.anchorless,
            "relabel": {
                "rows": self.relabeled,
                "top1_changed": self.relabel_top1_changed,
                "timeouts": self.relabel_timeouts,
                "wall_s": self.relabel_s,
                "max_s_per_row": self.relabel_max_s,
                "rows_per_s": (self.relabeled / self.relabel_s) if self.relabel_s else None,
            },
            "bench": {
                "reconstruct_s": self.reconstruct_s,
                "reconstruct_rows_per_s": (
                    (self.rows_in / self.reconstruct_s) if self.reconstruct_s else None
                ),
                "encode_rows": self.encode_rows,
                "encode_s": self.encode_s,
                "encode_rows_per_s": (
                    (self.encode_rows / self.encode_s) if self.encode_s else None
                ),
            },
            "shards": list(self.shards),
            "label_regimes": dict(self.label_regimes),
        }


def _encoder() -> Callable[[chess.Board], Any]:
    """The deriver's own encoder, for the audit-mode benchmark only."""
    from scripts import derive_corpus_targets as derive

    deriver = derive.TargetDeriver(
        derive.DeriveOptions(
            scheme=derive.parse_scheme("uniform-d9"), temp=1.0, cp_slope=1.0,
            cp_draw_width=1.0, limit=0, seed=0, rows_per_shard=8, max_envelope_misses=0,
        ),
    )
    return deriver._encode


def repair_worker(
    spec: RepairSpec, worker_id: int, *, searcher_factory: SearcherFactory | None = None,
) -> dict[str, Any]:
    """One worker's shards, in order.  Returns the tally ``main`` merges."""
    tally = WorkerTally()
    # ⚑ In the process that will hash, whatever start method the pool used:
    # applied here, then REQUIRED (the generator's own precondition, the one
    # `corpus.row_key` re-checks on every call).
    corpus.apply_history_rep_fix()
    corpus.require_history_rep_fix()
    shards = worker_shards(spec.in_dir, worker_id, spec.shard_indices, spec.listed_shards)
    if not shards:
        raise RepairError(f"worker {worker_id} has no listed shards in {spec.in_dir}")
    reconstructor = GameReconstructor(spec, worker_id)
    lease: corpus.EngineLease | None = None
    row_map: ZstdLines | None = None
    audit: TextIO | None = None
    out_dir = spec.out_dir
    encode = _encoder() if spec.audit_only and spec.bench_encode_rows > 0 else None
    if out_dir is not None:
        row_map = ZstdLines(
            out_dir / PROVENANCE_DIR / ROW_MAP_TEMPLATE.format(worker_id=worker_id),
        )
        audit = open(  # noqa: SIM115 - closed in the finally
            out_dir / PROVENANCE_DIR / RELABEL_AUDIT_TEMPLATE.format(worker_id=worker_id),
            "x", encoding="utf-8",
        )
    try:
        for shard, claimed in shards:
            started = time.perf_counter()
            rows_in = 0
            games: list[int] = []
            source = corpus.iter_shard_rows(shard)
            writer = ZstdLines(out_dir / output_shard_name(shard.name)) if out_dir is not None else None
            for row in source:
                rows_in += 1
                if not games or games[-1] != int(row["game_id"]):
                    games.append(int(row["game_id"]))
                t0 = time.perf_counter()
                repair = reconstructor.repair(row)
                tally.reconstruct_s += time.perf_counter() - t0
                row_class = repair.row_class
                phases: list[dict[str, Any]] | None = None
                if repair.board is not None and repair.history is not None:
                    if encode is not None and tally.encode_rows < spec.bench_encode_rows:
                        t0 = time.perf_counter()
                        encode(repair.board)
                        tally.encode_s += time.perf_counter() - t0
                        tally.encode_rows += 1
                    tally.rep_in_frames8 += int(repair.tags.frames)
                    tally.rep_in_segment += int(repair.tags.segment)
                    tally.rep_in_banked_window += int(repair.tags.banked_window)
                    tally.cur_position_repeated += int(repair.tags.cur_count >= 2)
                    tally.anchorless += int(repair.anchorless)
                    if repair.flagged(spec.relabel):
                        row_class = CLASS_RELABELED
                        if not spec.audit_only and audit is not None:
                            if lease is None:
                                lease = corpus.EngineLease(
                                    searcher_factory or default_searcher_factory(spec),
                                )
                            phases, row_class = _relabel(lease, row, repair, tally, audit)
                    if not row_class.startswith(QUARANTINE_PREFIX):
                        tally.root_reasons[repair.history.reason] += 1
                        tally.history_plies[str(repair.history.plies)] += 1
                        tally.history_kinds[str(repair.history_kind)] += 1
                tally.classes[row_class] += 1
                if not row_class.startswith(QUARANTINE_PREFIX):
                    tally.label_regimes[label_regime(row, relabeled=phases is not None)] += 1
                if row_map is not None:
                    row_map.write({
                        "worker_id": int(row["worker_id"]), "game_id": int(row["game_id"]),
                        "ply": int(row["ply"]), "class": row_class,
                    })
                if writer is not None and not row_class.startswith(QUARANTINE_PREFIX):
                    writer.write(repaired_row(row, repair, phases=phases))
            if rows_in != claimed:
                # ⚑ The inventory's claim is compared, not merely carried: a
                # shard truncated at a line boundary decodes cleanly and
                # would otherwise repair a subset the corpus never recorded.
                raise RepairError(
                    f"{shard.name}: the inventory claims {claimed} rows but the "
                    f"shard decodes to {rows_in}; a listed shard is claimed whole",
                )
            tally.rows_in += rows_in
            entry = {
                "path": output_shard_name(shard.name), "source": str(shard.name),
                "rows_in": rows_in, "rows_claimed": claimed,
                "rows": writer.rows if writer is not None else None,
                "games": games, "codec": "zstd",
            }
            if writer is not None:
                writer.close()
                tally.rows_out += writer.rows
            tally.shards.append(entry)
            _LOG.info(
                "worker %02d %s: %d rows in %.1fs (%s)", worker_id, shard.name, rows_in,
                time.perf_counter() - started,
                ", ".join(f"{k}={v}" for k, v in sorted(tally.classes.items())),
            )
    finally:
        if lease is not None:
            lease.close()
        if row_map is not None:
            row_map.close()
        if audit is not None:
            audit.close()
    tally.games = len(reconstructor.seen_games)
    tally.book_games = reconstructor.book_games
    return {"worker_id": int(worker_id), **tally.summary()}


def _relabel(
    lease: corpus.EngineLease, row: Mapping[str, Any], repair: RowRepair,
    tally: WorkerTally, audit: TextIO,
) -> tuple[list[dict[str, Any]] | None, str]:
    """R2 for one flagged row: a cold search under the row's own window."""
    if repair.board is None or repair.history is None:
        raise RepairError("a quarantined row reached the re-label step")
    started = time.perf_counter()
    lease.new_game()
    try:
        search = lease.search_position(repair.board)
    except StockfishTimeoutError:
        tally.relabel_timeouts += 1
        return None, QUARANTINE_PREFIX + REASON_RELABEL_TIMEOUT
    elapsed = time.perf_counter() - started
    # The searcher builds its own window from the same board; it must be the
    # one being banked.  ⚑ The reason is compared only on an anchored board:
    # an anchorless rebuild starts its stack AT the root, so ``history_for``
    # there reads ``game_start`` where the row (correctly) says
    # ``irreversible``; the position line the engine saw is the same.
    searched = (search.history.fen, search.history.root_fen, search.history.uci)
    banked = (repair.history.fen, repair.history.root_fen, repair.history.uci)
    if searched != banked or (
        not repair.anchorless and search.history.reason != repair.history.reason
    ):
        raise RepairError(
            f"game {row['game_id']} ply {row['ply']}: the searcher's window "
            f"differs from the banked one ({search.history} vs {repair.history})",
        )
    phases = [p.as_row() for p in search.phases]
    old_move, old_cp = top1_of_phases(row.get("phases", []))
    new_move, new_cp = top1_of_phases(phases)
    tally.relabeled += 1
    tally.relabel_s += elapsed
    tally.relabel_max_s = max(tally.relabel_max_s, elapsed)
    tally.relabel_top1_changed += int(old_move != new_move)
    audit.write(json.dumps({
        "worker_id": int(row["worker_id"]), "game_id": int(row["game_id"]),
        "ply": int(row["ply"]), "fen": str(row["fen"]),
        "history_plies": repair.history.plies,
        "history_root_reason": repair.history.reason,
        "old_top1": old_move, "old_cp": old_cp, "new_top1": new_move, "new_cp": new_cp,
        "top1_changed": old_move != new_move, "search_s": elapsed,
        "cold_tt_retry": bool(lease.cold_tt_retry_last),
    }, sort_keys=True) + "\n")
    return phases, CLASS_RELABELED


# -- the run ------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_counters(results: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    total: Counter[str] = Counter()
    for r in results:
        total.update({str(k): int(v) for k, v in r[key].items()})
    return dict(sorted(total.items()))


def _engine_id_name(spec: RepairSpec) -> str | None:
    try:
        return audit_targets.engine_identity(spec.sf_binary)
    except Exception:  # identity is a stamp, not a gate; the sha256 check is the gate
        return None


def build_records(
    *, spec: RepairSpec, manifest: Mapping[str, Any], manifest_sha: str,
    results: Sequence[Mapping[str, Any]], started_utc: str, wall_s: float,
    engine_id_name: str | None, book: Mapping[str, Any],
    unlisted_entries: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(repair_manifest, summary)`` -- provenance, and the deriver-facing record.

    ``unlisted_entries`` are the in-flight shards on disk, decoded by the
    caller and repaired by nobody; ``book`` is the block ``run`` verified."""
    classes = merge_counters(results, "classes")
    rows_in = sum(int(r["rows_in"]) for r in results)
    rows_out = sum(int(r["rows_out"]) for r in results)
    relabel_rows = sum(int(r["relabel"]["rows"]) for r in results)
    relabel_s = sum(float(r["relabel"]["wall_s"]) for r in results)
    reconstruct_s = sum(float(r["bench"]["reconstruct_s"]) for r in results)
    encode_rows = sum(int(r["bench"]["encode_rows"]) for r in results)
    encode_s = sum(float(r["bench"]["encode_s"]) for r in results)
    shards = [
        {**entry, "worker_id": int(r["worker_id"])}
        for r in sorted(results, key=lambda r: int(r["worker_id"]))
        for entry in r["shards"]
    ]
    repair_manifest = {
        "schema": 1,
        "tool": "scripts/repair_corpus_history.py",
        "started_utc": started_utc,
        "wall_s": wall_s,
        "audit_only": spec.audit_only,
        # ⚑ THE MODE, for the consumer that must refuse an audit-mode output:
        # production is the whole recorded inventory, the pinned book, no
        # slicing, no relabel.
        "production": not spec.audit_mode,
        "input": {
            "dir": str(spec.in_dir),
            "manifest_sha256": manifest_sha,
            "config_sha256": manifest.get("config_sha256"),
            "run_id": (manifest.get("config_requested") or {}).get("run_id"),
            "row_schema": SOURCE_SCHEMA,
            "shards": None if spec.shard_indices is None else list(spec.shard_indices),
            "workers": [int(r["worker_id"]) for r in results],
            "listed_shards": len(spec.listed_shards),
            "rows_claimed": sum(rows for _, rows in spec.listed_shards),
            "progress_files": list(spec.progress_files),
            "progress_torn_tail_files": list(spec.progress_torn_tail_files),
            # On disk but not in the corpus's own inventory (a paused run's
            # in-flight shards): recorded with what they hold, never repaired.
            "unlisted_shards_skipped": [str(e["path"]) for e in unlisted_entries],
            "unlisted_shards": list(unlisted_entries),
        },
        "output": {"row_schema": TARGET_SCHEMA, "dir": None if spec.out_dir is None else str(spec.out_dir)},
        # ⚑ No engine ran under --relabel off, so no engine is recorded.
        "engine": None if spec.relabel == RELABEL_OFF else {
            "path": spec.sf_binary, "sha256": (manifest.get("engine") or {}).get("sha256"),
            "id_name": engine_id_name, "threads": 1, "hash_mb": spec.sf_hash_mb,
            "ucinewgame_per_row": True, "staircase": spec.staircase,
        },
        KEY_HISTORY_REP_FIX: bool(rep_fix.current()),
        "label_regimes": merge_counters(results, "label_regimes"),
        "book": {
            **book,
            "plies": spec.opening.opening_book_max_plies,
            "max_games": spec.opening.opening_book_max_games,
            "seed": spec.seed,
        },
        "bridge": {"max_plies": MAX_BRIDGE_PLIES, "path_cap": BRIDGE_PATH_CAP},
        "rows_in": rows_in,
        "rows_out": rows_out,
        "games": sum(int(r["games"]) for r in results),
        "classes": classes,
        "class_fractions": {k: v / rows_in for k, v in classes.items()} if rows_in else {},
        "quarantined": sum(v for k, v in classes.items() if k.startswith(QUARANTINE_PREFIX)),
        "relabeled": classes.get(CLASS_RELABELED, 0),
        "history_kinds": merge_counters(results, "history_kinds"),
        "history_root_reasons": merge_counters(results, "history_root_reasons"),
        "history_plies_histogram": merge_counters(results, "history_plies_histogram"),
        "book_games": merge_counters(results, "book_games"),
        "tags": {
            key: sum(int(r["tags"][key]) for r in results)
            for key in (
                "rep_in_frames8", "rep_in_segment", "rep_in_banked_window",
                "cur_position_repeat_count_ge2",
            )
        },
        "anchorless": sum(int(r["anchorless"]) for r in results),
        "relabel": {
            "mode": spec.relabel,
            "rows": relabel_rows,
            "top1_changed": sum(int(r["relabel"]["top1_changed"]) for r in results),
            "timeouts": sum(int(r["relabel"]["timeouts"]) for r in results),
            "wall_s": relabel_s,
            "max_s_per_row": max((float(r["relabel"]["max_s_per_row"]) for r in results), default=0.0),
            "rows_per_s_per_core": (relabel_rows / relabel_s) if relabel_s else None,
        },
        "bench": {
            "reconstruct_s": reconstruct_s,
            "reconstruct_rows_per_s_per_core": (rows_in / reconstruct_s) if reconstruct_s else None,
            "encode_rows": encode_rows,
            "encode_rows_per_s_per_core": (encode_rows / encode_s) if encode_s else None,
        },
        "workers": {str(r["worker_id"]): {k: v for k, v in r.items() if k != "shards"} for r in results},
        "row_map_files": [
            f"{PROVENANCE_DIR}/{ROW_MAP_TEMPLATE.format(worker_id=int(r['worker_id']))}"
            for r in results
        ],
        "relabel_audit": RELABEL_AUDIT_NAME,
        "shards": shards,
    }
    summary = {
        "schema": corpus.SUMMARY_SCHEMA,
        "row_schema": TARGET_SCHEMA,
        # ⚑ THE COMPLETENESS CLAIM the deriver turns into `corpus_complete`:
        # a production repair covers the whole recorded inventory with
        # nothing refused or skipped, so it is `true`; an audit-mode
        # --shards/--workers slice is NOT the corpus, says `false` here and
        # names the slice.
        "run_finished": not spec.partial,
        "production": not spec.audit_mode,
        **({"partial_repair": {
            "workers": [int(r["worker_id"]) for r in results],
            "shards": None if spec.shard_indices is None else list(spec.shard_indices),
            "listed_input_shards": len(spec.listed_shards),
            "repaired_shards": len(shards),
        }} if spec.partial else {}),
        "run_id": (manifest.get("config_requested") or {}).get("run_id"),
        "started_utc": started_utc,
        "wall_s": wall_s,
        "rows": rows_out,
        "games": sum(int(r["games"]) for r in results),
        "shards": [
            {"path": s["path"], "rows": s["rows"], "games": s["games"], "codec": s["codec"]}
            for s in shards
        ],
        "config_requested": manifest.get("config_requested"),
        "config_sha256": manifest.get("config_sha256"),
        "staircase_parsed": manifest.get("staircase_parsed"),
        "config_realized_by_worker": {},
        "engine": {
            **(manifest.get("engine") or {}),
            **({} if spec.relabel == RELABEL_OFF else {"relabel_id_name": engine_id_name}),
        },
        KEY_HISTORY_REP_FIX: bool(rep_fix.current()),
        "banked_rows_min_piece_count": manifest.get("banked_rows_min_piece_count", corpus.MIN_BANKED_PIECES),
        "adjudication_max_piece_count": manifest.get("adjudication_max_piece_count", corpus.ADJUDICATION_MAX_PIECES),
        "history_plies_histogram": repair_manifest["history_plies_histogram"],
        "history_root_reasons": repair_manifest["history_root_reasons"],
        "repair": {"manifest": REPAIR_MANIFEST_NAME, "source": str(spec.in_dir)},
    }
    return repair_manifest, summary


def format_report(repair_manifest: Mapping[str, Any]) -> str:
    rows_in = int(repair_manifest["rows_in"])
    lines = [
        f"rows in {rows_in}  out {repair_manifest['rows_out']}  games {repair_manifest['games']}",
        "classes:",
    ]
    for name, count in sorted(repair_manifest["classes"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name:40s} {count:9d} {count / rows_in if rows_in else 0:8.3%}")
    lines.append("book games: " + ", ".join(
        f"{k}={v}" for k, v in sorted(repair_manifest["book_games"].items())
    ))
    lines.append("history kinds: " + ", ".join(
        f"{k}={v}" for k, v in sorted(repair_manifest["history_kinds"].items())
    ))
    lines.append("label regimes: " + ", ".join(
        f"{k}={v}" for k, v in sorted(repair_manifest["label_regimes"].items())
    ))
    lines.append("root reasons: " + ", ".join(
        f"{k}={v}" for k, v in sorted(repair_manifest["history_root_reasons"].items())
    ))
    tags = repair_manifest["tags"]
    lines.append(
        "tags: " + ", ".join(
            f"{k}={v} ({v / rows_in if rows_in else 0:.3%})" for k, v in tags.items()
        )
        + f"; relabel={repair_manifest['relabel']['mode']}; anchorless rebuilds "
        f"{repair_manifest['anchorless']}",
    )
    bench = repair_manifest["bench"]
    relabel = repair_manifest["relabel"]
    rps = bench["reconstruct_rows_per_s_per_core"]
    lines.append(
        f"reconstruct {rps:,.0f} rows/s/core" if rps else "reconstruct: no timing",
    )
    if bench["encode_rows_per_s_per_core"]:
        lines.append(
            f"replay+encode {bench['encode_rows_per_s_per_core']:,.0f} rows/s/core "
            f"({bench['encode_rows']} rows)",
        )
    if relabel["rows"]:
        lines.append(
            f"relabel {relabel['rows']} rows, {relabel['rows_per_s_per_core']:.3f} "
            f"rows/s/core, top-1 changed {relabel['top1_changed']} "
            f"({relabel['top1_changed'] / relabel['rows']:.2%}), max "
            f"{relabel['max_s_per_row']:.2f}s, timeouts {relabel['timeouts']}",
        )
    return "\n".join(lines)


def build_spec(
    args: argparse.Namespace, manifest: Mapping[str, Any], *,
    listed_shards: Sequence[tuple[str, int]], unlisted_shards: Sequence[str] = (),
    progress_files: Sequence[str] = (), progress_torn_tail_files: Sequence[str] = (),
    verify_engine: bool = True,
) -> RepairSpec:
    """The spec, with the engine and tablebases verified unless a test injects a searcher."""
    requested = manifest.get("config_requested") or {}
    engine_record = manifest.get("engine") or {}
    sf_binary = str(args.engine or engine_record.get("path") or requested.get("stockfish"))
    if not args.audit_only and verify_engine and str(args.relabel) != RELABEL_OFF:
        if not Path(sf_binary).exists():
            raise RepairError(f"engine {sf_binary} does not exist; pass --engine")
        want = str(engine_record.get("sha256") or "")
        if not want:
            raise RepairError(
                "the input manifest records no engine sha256, so a re-label "
                "cannot be proven to run the corpus's own build; refused",
            )
        have = sha256_of(Path(sf_binary))
        if have != want:
            raise RepairError(
                f"engine {sf_binary} hashes to {have}, but the corpus was "
                f"labeled by {want}; a re-label on a different build is not the "
                "same label and is refused",
            )
    syzygy = str(args.syzygy_path or requested.get("syzygy_path") or "")
    relabel = str(args.relabel)
    if relabel not in RELABEL_MODES:
        raise RepairError(f"--relabel {relabel!r} is not one of {RELABEL_MODES}")
    if not args.audit_only and verify_engine and relabel != RELABEL_OFF:
        corpus.refuse_unopenable_syzygy(syzygy)
    return RepairSpec(
        in_dir=Path(args.in_dir),
        out_dir=None if args.audit_only else Path(args.out_dir),
        seed=int(requested["seed"]),
        opening=opening_config_from_manifest(requested, book_override=args.book),
        staircase=str(requested["staircase"]),
        cp_slope=float(requested["cp_slope"]),
        cp_draw_width=float(requested["cp_draw_width"]),
        sf_binary=sf_binary,
        sf_hash_mb=int(args.sf_hash_mb if args.sf_hash_mb is not None else requested.get("sf_hash_mb", corpus.DEFAULT_SF_HASH_MB)),
        sf_read_timeout_s=float(requested.get("sf_read_timeout_s", corpus.DEFAULT_SF_READ_TIMEOUT_S)),
        sf_search_timeout_s=float(args.sf_search_timeout_s),
        syzygy_path=syzygy,
        nice=int(requested.get("nice", corpus.DEFAULT_NICE)),
        audit_only=bool(args.audit_only),
        shard_indices=parse_shard_range(args.shards),
        bench_encode_rows=int(args.bench_encode_rows),
        listed_shards=tuple(listed_shards),
        progress_files=tuple(progress_files),
        progress_torn_tail_files=tuple(progress_torn_tail_files),
        unlisted_shards=tuple(unlisted_shards),
        audit_mode=bool(args.audit_mode),
        partial=bool(args.shards) or bool(args.workers),
        relabel=relabel,
    )


def run(
    args: argparse.Namespace, *, searcher_factory: SearcherFactory | None = None,
) -> dict[str, Any]:
    """The whole repair.  ``searcher_factory`` is the test seam: a scripted
    engine in place of Stockfish, which also skips the binary's sha256 and
    tablebase checks (there is no binary) and runs in-process."""
    # ⚑ FIRST, before the book warm and before any CBoard exists: the
    # generator's own regime switch and its precondition (KEY_HISTORY_REP_FIX).
    corpus.apply_history_rep_fix()
    corpus.require_history_rep_fix()
    refuse_in_production(args)
    in_dir = Path(args.in_dir)
    manifest_path = in_dir / corpus.MANIFEST_NAME
    if not manifest_path.exists():
        raise RepairError(f"{manifest_path} does not exist; the repair reads the run's config from it")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("row_schema", -1)) != SOURCE_SCHEMA:
        raise RepairError(
            f"{manifest_path} says row_schema {manifest.get('row_schema')}; this "
            f"tool repairs schema {SOURCE_SCHEMA} only",
        )
    # ⚑ The stamp is recomputed, not trusted: an edited config_requested
    # would put the wrong seed or book under every resampled opening.
    recomputed = corpus.stamp_sha256(dict(manifest.get("config_requested") or {}))
    if recomputed != str(manifest.get("config_sha256", "")):
        raise RepairError(
            f"{manifest_path} is inconsistent with itself: config_sha256 is "
            f"{manifest.get('config_sha256')!r} but config_requested hashes to "
            f"{recomputed!r}; the record of what produced the rows has been altered",
        )
    manifest_sha = sha256_of(manifest_path)
    # ⚑ The corpus's OWN inventory, through the deriver's reader (manifest +
    # progress files).  Whatever is on disk but unlisted is skipped by name
    # and reported.
    listed, unlisted, inventory = corpus_inventory(in_dir)
    _LOG.info(
        "inventory (%d progress files, %d torn tails): %d listed shards claiming "
        "%d rows, %d unlisted", len(inventory.progress_files),
        len(inventory.torn_tail_files), len(listed), inventory.rows_claimed, len(unlisted),
    )
    spec = build_spec(
        args, manifest, listed_shards=listed, unlisted_shards=unlisted,
        progress_files=tuple(inventory.progress_files),
        progress_torn_tail_files=tuple(inventory.torn_tail_files),
        verify_engine=searcher_factory is None,
    )
    workers = [int(w) for w in args.workers] if args.workers else worker_ids_in([n for n, _ in listed])
    if not workers:
        raise RepairError(f"{in_dir} lists no worker shards")
    if unlisted:
        _LOG.warning(
            "%d shard(s) on disk are not in the corpus inventory and will NOT be "
            "repaired: %s", len(unlisted), ", ".join(unlisted),
        )
    report_path = refuse_report_path_inside(args.report_json, in_dir, spec.out_dir)
    book = verify_book(str(spec.opening.opening_book_path), args.book_sha256)
    if spec.out_dir is not None:
        spec.out_dir.mkdir(parents=True, exist_ok=True)
        corpus.refuse_populated_dir(spec.out_dir)
        (spec.out_dir / PROVENANCE_DIR).mkdir()
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    _LOG.info("warming the opening book %s", spec.opening.opening_book_path)
    warm_opening_book_cache(spec.opening)
    _LOG.info("book warm in %.1fs; %d workers, %d procs", time.perf_counter() - started, len(workers), args.procs)
    engine_id_name = (
        None
        if spec.audit_only or searcher_factory is not None or spec.relabel == RELABEL_OFF
        else _engine_id_name(spec)
    )
    results: list[dict[str, Any]] = []
    if int(args.procs) <= 1 or searcher_factory is not None:
        results.extend(
            repair_worker(spec, worker_id, searcher_factory=searcher_factory)
            for worker_id in workers
        )
    else:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=int(args.procs), mp_context=context) as pool:
            futures = {pool.submit(repair_worker, spec, w): w for w in workers}
            results.extend(future.result() for future in as_completed(futures))
    results.sort(key=lambda r: int(r["worker_id"]))
    repair_manifest, summary = build_records(
        spec=spec, manifest=manifest, manifest_sha=manifest_sha, results=results,
        started_utc=started_utc, wall_s=time.perf_counter() - started,
        engine_id_name=engine_id_name, book=book,
        unlisted_entries=[unlisted_entry(in_dir / name) for name in unlisted],
    )
    if spec.out_dir is not None:
        with open(spec.out_dir / RELABEL_AUDIT_NAME, "x", encoding="utf-8") as merged:
            for worker_id in workers:
                part = spec.out_dir / PROVENANCE_DIR / RELABEL_AUDIT_TEMPLATE.format(worker_id=worker_id)
                merged.write(part.read_text(encoding="utf-8"))
                part.unlink()
        (spec.out_dir / REPAIR_MANIFEST_NAME).write_text(
            json.dumps(repair_manifest, indent=1, sort_keys=True), encoding="utf-8",
        )
        (spec.out_dir / corpus.SUMMARY_NAME).write_text(
            json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8",
        )
    if report_path is not None:
        # Exclusive: a report never replaces a file that was already there.
        with open(report_path, "x", encoding="utf-8") as fh:
            fh.write(json.dumps(repair_manifest, indent=1, sort_keys=True))
    print(format_report(repair_manifest), flush=True)
    return repair_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_dir", required=True, help="schema-1 corpus directory (never modified)")
    parser.add_argument("--out", dest="out_dir", help="output directory for the schema-2 shards (must be empty)")
    parser.add_argument("--audit-only", action="store_true", help="classify and benchmark; write nothing, run no engine")
    parser.add_argument("--procs", type=int, default=1, help="worker processes (each repairs whole workers and owns one engine)")
    parser.add_argument("--workers", nargs="*", help="worker ids to repair (default: every worker with shards)")
    parser.add_argument("--shards", help="shard index range per worker, e.g. 100-102 (default: all)")
    parser.add_argument("--engine", help="Stockfish binary (default: the manifest's; must hash to the manifest's sha256)")
    parser.add_argument("--book", help="opening book path override (same book the run used)")
    parser.add_argument("--syzygy-path", help="tablebase path override (default: the manifest's)")
    parser.add_argument("--sf-hash-mb", type=int, help="engine hash (default: the manifest's)")
    parser.add_argument("--sf-search-timeout-s", type=float, default=DEFAULT_SF_SEARCH_TIMEOUT_S, help="per-search wedge tripwire for the cold re-labels")
    parser.add_argument("--relabel", choices=RELABEL_MODES, default=RELABEL_OFF, help="R2 is PARKED: 'off' (default) copies every label byte-for-byte and only tags rows; 'banked_window'/'segment' re-search the rows the tag of that name flags, cold (NOT the generator's carried-TT label)")
    parser.add_argument("--audit-mode", action="store_true", help="allow the optional surface (--shards/--workers slices, --book, --relabel, --report-json, a --book-sha256 other than the production pin) and stamp production: false into the manifest and summary; without it every one of those is refused by name")
    parser.add_argument("--book-sha256", help="the hash the opening book must have (default: the production pin; another value is audit-mode only)")
    parser.add_argument("--bench-encode-rows", type=int, default=DEFAULT_BENCH_ENCODE_ROWS, help="--audit-only: rows to time through the deriver's encoder per worker")
    parser.add_argument("--report-json", help="audit mode: also write the repair manifest here (never inside a corpus, never over an existing file)")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if not args.audit_only and not args.out_dir:
        print("--out is required unless --audit-only", file=sys.stderr)
        return 2
    try:
        run(args)
    except (RepairError, ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
