#!/usr/bin/env python3
"""NNUE-bootstrap corpus generator: CPU-only selfplay banking FULL-WIDTH Stockfish.

No GPU, no net, no MCTS. Each worker plays games in which Stockfish itself is
both the move chooser (through a Gumbel sample over its own root values) and the
labeller, and every position it searches is banked with the COMPLETE per-depth
MultiPV stream the search emitted -- not a summary of it.

WHY THE RAW BLOCKS AND NOT A SUMMARY.  ``docs/experiment_ledger.md``'s
FREEZE-THE-OBSERVATIONS rule: an expensive run against a deterministic search
must bank the lowest-level observations it collected, because every later
correction (a different cp->wdl mapping, a depth-truncated re-fit, a re-ranked
target, a width ablation) is then a re-read of this corpus instead of a rerun --
and a rerun re-rolls the intervention.  The unit of resampling is banked too:
every row carries ``game_id`` and ``ply``, so a clustered bootstrap is possible
against a corpus that has already been written.

THE STAIRCASE
-------------
``--staircase "all:9,16:11,4:13"`` (the default) is a NARROWING SCOUT:

* phase 1 -- MultiPV = the position's legal-move count, ``go depth 9``.  Every
  legal move gets a value, which is what makes the move-selection distribution
  well defined.
* phase 2 -- the top 16 moves by phase 1's deepest full-width scores, restricted
  with UCI ``searchmoves``, MultiPV 16, ``go depth 11``.
* phase 3 -- the top 4 of those, MultiPV 4, ``go depth 13``.

``--staircase-policy g10 --staircase "all:9,8:10,4:12"`` names the frozen,
independently validated adaptive labeler.  It always runs the first two rungs,
then runs ``4:12`` exactly when d10's rank-1 minus rank-2 effective-cp margin
is at most 10 cp.  The decision and margin are banked on every row; the
manifest and summary carry the frozen rule.  The policy refuses every other
staircase instead of turning one validated threshold into an unmeasured knob.

⚑ THE TRANSPOSITION TABLE IS CARRIED ACROSS THE PHASES ON PURPOSE, and that is
disclosed rather than assumed.  Scout-then-narrow is the DEPLOYABLE scheme: the
deep phases are cheap precisely because the shallow one warmed the table, and a
generator that cleared the hash between phases would measure a scheme nobody
would ship.  Every row and the summary carry ``tt_carried_across_phases: true``
so a consumer cannot mistake these for independent searches.  The table IS
cleared (``ucinewgame``) at the start of every GAME, so one game's tree cannot
leak into the next -- ``tt_cleared_per_game`` says so.

THE MOVE WINDOW (ROW SCHEMA 2)
------------------------------
Every banked row carries the moves that led to it, and the SAME window is what
the engine is set up with.  Both halves matter and they are one change:

* THE INPUT.  A row that is only a FEN encodes with history slot 0 filled and
  slots 1..7 -- planes 13..103 -- and every repetition plane ZERO.  That is not
  a small difference: measured 2026-09-01 (``docs/experiment_ledger.md``, the
  history-materiality probe) the champion flips **46.5% of its top-1 moves** and
  loses **+20.2 cp** of regret when its history planes are filled the way play
  fills them.  A corpus built on the zero-history distribution trains on inputs
  production never sees.
* THE LABEL.  ``position fen <fen>`` hands Stockfish a position with no past,
  so its own repetition and fifty-move detection cannot fire and a drawn-by-
  repetition line scores as if it were fresh.  ``position fen <root> moves ...``
  gives it the reversible segment, which is exactly what those rules need.

``history_for`` picks the window; :func:`position_command` is the ONLY place a
``position`` line is spelled.  See both for the root definition and why it is
7 plies plus the halfmove clock rather than just 7.

⚑⚑ DEDUP IS TWO KEYS, NOT ONE.  Under schema 2 a position reached by two
routes can differ in (1) the 8-frame input tensor, (2) the repetition state the
engine's ``is_draw``/``has_game_cycle`` read, and (3) therefore the right move
-- so "same FEN" is neither "same row" nor "same label".  ``play_game`` keys
the two concepts separately (see :func:`search_key` and :func:`row_key`):

* ``search_key`` -- LABEL equivalence: ``dedup_key`` (FEN minus the fullmove
  number) plus the repetition signature of the reversible segment.  Stockfish
  consults game history only through repetition counts, so two routes with
  equal signatures get the same values (0/1200 label changes measured without a
  repeat, 2026-09-01 calibration).
* ``row_key`` -- MODEL-INPUT identity: a hash of the encoded 175-plane tensor,
  banked on the row as ``input_key``.  A route whose tensor is new is SEARCHED
  and BANKED even when its ``search_key`` is cached (the compact cache cannot
  supply the ``phases`` a row needs); a route whose tensor was seen banks
  nothing, and is served from the cache only when its ``search_key`` hits.

Measured on run03/w03 (1,717,460 rows): position-key collisions are 0.015% of
positions, all in the first 1-2 plies after a shared book exit, so the split is
a CORRECTNESS guarantee for schema 2 and not a throughput trade.

THE STREAM RULES, MEASURED AGAINST THE REAL BINARY (2026-08-27)
---------------------------------------------------------------
* The FIRST non-bound emission per ``(depth, multipv-rank)`` wins, NEVER the
  last.  On an abort Stockfish re-emits updated lines still labelled with the
  old depth, so "the last line seen for this rank" silently mixes a later
  search's number into an earlier depth's block.  Re-emissions are COUNTED
  (``re_emissions``) rather than discarded silently.
* ``upperbound`` / ``lowerbound`` lines are dropped: an aspiration bound is a
  claim about a window, not the move's score.
* A clean ``go depth`` stream emits exactly ``width x 1`` non-bound scored lines
  per iteration.  Verified against the production binary at MultiPV 20 (9
  depths x 20 lines, zero bounds), MultiPV 16 with ``searchmoves`` (11 x 16) and
  MultiPV 4 (13 x 4).  The generator ASSERTS that count per depth and counts a
  mismatch as ``emission_count_violations`` instead of crashing -- a corpus that
  dies on the millionth position because one search ended early on a proven mate
  is worse than one that records the anomaly.
* ⚑⚑ AND THE ASSERTION FIRES ON REAL, HEALTHY SEARCHES.  **Measured 2026-08-27,
  cold engine, start position, MultiPV 20:** ``go depth 4`` and ``go depth 6``
  emit their FINAL iteration TWICE (40 lines at depth 4; 40 at depth 6), while
  depths 2, 3, 5 and 9 do not.  The re-emitted lines carry the SAME rank, move
  and score -- only ``nps``/``time`` and the PV tail differ -- so it is
  Stockfish's end-of-search flush, not an abort.  That is why the emission
  counter alone is not a verdict: ``duplicate_iteration_flushes`` classifies
  exactly this benign signature (a depth at ``2 x width`` with zero
  disagreement), and ``re_emissions_disagreeing`` is the number that actually
  matters -- it is nonzero only when the banked block WOULD HAVE DIFFERED under
  a last-emission-wins rule.  A corpus with ``re_emissions_disagreeing == 0``
  and ``emission_count_violations == duplicate_iteration_flushes`` saw nothing
  but the flush.
* ``info string`` lines carry no depth and no pv and are skipped.
* A game-over position emits no PV lines at all, so it is never searched: the
  loop tests termination BEFORE it searches.

WHAT A WORKER HOLDS, AND WHAT IT DOES WHEN IT DIES
--------------------------------------------------
* THE PER-WORKER DEDUP CACHE IS COMPACT AND BOUNDED.  It stores the two things
  selection actually reads -- the uci list and one ``float32`` value vector --
  rather than the banked ``PvLine`` objects, and it holds at most
  ``--dedup-cache-max`` positions with FIFO eviction.  ⚑ AN EVICTED POSITION
  THAT RECURS IS RE-SEARCHED AND RE-BANKED, exactly as a position first reached
  by a second worker already is (the cache has always been per-worker).  The
  eviction count is in the summary next to the dup counters so a corpus states
  how much of its duplication the bound let through.
* A WORKER THAT DIES DOES NOT DESTROY THE RUN'S BOOKKEEPING.  Its slot records
  the exception and the game/ply it died on, the surviving workers finish, and
  ``summary.json`` is written with a top-level ``failed_workers``.  One residual:
  a HARD-killed worker process (OOM-kill/segfault) breaks the shared pool, so an
  in-flight peer's future fails too -- both slots are marked failed and any
  already-closed shards of theirs exist on disk unindexed by ``summary["shards"]``.
  Python-level failures never do this; ``run_worker`` catches them.  The process
  exit code is still nonzero -- a partial corpus is a fact to record, not a
  success to report.

KILL AND RESUME (``--resume``)
-----------------------------
A multi-day burn has to survive ``kill -9`` of the whole process tree, and the
crash-safety comes ENTIRELY from an append-only protocol -- no signal handler,
no clean-shutdown path, because a handler cannot run for ``SIGKILL`` and a
protocol that only works on a polite exit is a protocol that does not work.

* SHARDS ROTATE ON GAME BOUNDARIES, NEVER MID-GAME (``ShardWriter.end_game``).
  A closed shard therefore holds WHOLE games, and the progress line that
  records it names them (``"games": [...]``).
* One progress line is appended per closed shard, ``fsync``-free but
  single-line and append-only, so a ``kill -9`` can only ever tear the LAST
  line.  The reader drops a torn tail and refuses anything worse.
* ⚑ AND THE RESUME REPAIRS THAT TAIL ON DISK BEFORE IT APPENDS.  Tolerating it
  on read is not enough: the next append opens ``"a"`` and lands on the end of
  the fragment, so the record and the fragment become one line that is neither.
  A second kill would then hit the "damaged some other way" refusal and that
  worker would be unresumable by hand-editing only.  ``repair_worker_progress``
  either restores the single newline a kill stole from a WHOLE final record or
  truncates a partial one away, both idempotent and safe to be killed inside.
* A game that banked NO rows (every position dedup-served, or an immediate
  adjudication) is still a completed game.  Its id rides in the current
  shard's pending list, and a trailing run of row-less games is flushed as a
  path-less COMPLETION RECORD ``{"path": null, "rows": 0, "games": [...]}``.
  ⚑ A null-path line is a completion record, NOT a shard: it indexes no file.
* On ``--resume`` each worker reads its own progress file, replays only the
  games no line claims, DELETES every ``w<id>-*`` file no line lists (that is
  the shard the kill caught mid-write; its games are simply replayed), and
  starts its shard index one above the highest listed.  Games are
  order-independent -- every RNG stream is seeded from
  ``(seed, worker_id, game_id, tag, ply)`` -- so a replayed set produces the
  same rows it would have in one uninterrupted run.
* THE DEDUP CACHE IS RE-WARMED from this worker's own listed shards before the
  first game, because a cold cache would re-search (and re-bank) positions the
  killed session had already banked.  ⚑ It re-warms from BANKED ROWS **and
  from the cache-only events committed beside them**: a search that banked no
  row (a seen tensor under a new ``search_key``; a sub-``MIN_BANKED_PIECES``
  position) is recorded as a ``cache_event`` in the same progress record as
  the game's rows, and the resume replays rows and events strictly in the
  worker's ``seq`` order, so the re-warmed cache holds exactly what the
  killed session's cache held, in the same FIFO order.  ``dedup_rewarmed`` and
  ``dedup_cache_events_rewarmed`` in the summary are the two counts.  The
  residual is the in-flight shard the kill caught mid-write: its record was
  never committed, its rows and events are gone together, and its games are
  replayed whole -- a re-search, never a different corpus.  A progress record
  that names games but carries no ``cache_events`` predates this contract and
  is REFUSED on resume rather than adopted as "zero events".
* A resume must not change any generation-affecting setting: the requested
  config is re-stamped and its sha256 compared against the manifest's, and a
  mismatch is refused before a single game is played.  ``--workers`` is in the
  stamp, so a resume cannot re-deal the game ids either.
* ⚑ A ``summary.json`` IS NOT PROOF THAT THE RUN FINISHED, and the resume gate
  does not treat it as one.  A crash the parent process SURVIVES -- an OOM kill
  that breaks the worker pool is the measured one -- still reaches the end of
  ``run`` and still writes a summary, one whose ``failed_workers`` block names
  every worker it lost.  So the summary states its own verdict in
  ``run_finished`` (``build_summary``: true exactly when ``failed_workers`` is
  empty, the same condition ``main``'s exit code already uses), the gate reads
  THAT rather than the file's existence, and a crashed session's summary is
  moved aside to ``summary.unfinished_NN.json`` so the resumed session's own
  ``open("x")`` still has its name free at the end.  ⚑ AND SO IS ANYTHING AT
  THE ``--json`` PATH, which ``main`` writes with a SECOND ``open("x")`` after
  ``run`` returns: freeing only the in-directory name would move the same
  days-late traceback one function further out, onto a resume that had already
  banked a correct corpus.  That one is moved unconditionally -- an output
  location the operator named is not corpus identity, and it is not in
  ``config_stamp`` for the drift gate to speak to either.  ⚑ A summary that makes no
  ``run_finished`` claim -- unreadable, or written before this key existed --
  is REFUSED, not guessed at: the cost of failing closed is one manual rename,
  and the cost of failing open is games appended to a finished corpus.

WHAT IS SHARED RATHER THAN RESTATED
-----------------------------------
The mate band and the cp->wdl mapping are ``scripts/audit_label_candidates``'s,
imported: ``effective_cp_from_score`` (mate-first, ``score mate 0`` handled) and
``q_from_effective_cp``, which reaches ``gen.cp_to_wdl_array`` as a module
attribute at call time so the gate's arms and this generator are ONE function
object.  ``tests/test_gen_sf_rooted_corpus.py`` proves that by replacing the one
object and watching this file's selection move.  Openings are
``chess_anti_engine.selfplay.opening``'s -- the production sampler, through the
same ``OpeningConfig`` the live selfplay path builds.  Adjudication is
``chess_anti_engine.tablebase``'s ``tb_adjudicate_result``.  ``searchmoves``
validation is ``StockfishUCI``'s own ``_validated_searchmoves``, which matters
more here than anywhere: Stockfish SILENTLY IGNORES a root move that is not
legal, so an unvalidated narrowing list widens the search back to full width and
the phase reports a narrowing that never happened.

Usage::

    PYTHONPATH=. nice -n 15 python3 scripts/gen_sf_rooted_corpus.py \\
        --out-dir data/nnue_bootstrap/run01 --games 3 --workers 1
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib
import io
import itertools
import json
import logging
import math
import multiprocessing
import os
import statistics
import sys
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import chess.polyglot
import numpy as np

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.selfplay.opening import OpeningConfig, sample_starting_board
from chess_anti_engine.stockfish.uci import (
    StockfishTimeoutError,
    StockfishUCI,
    _parse_info_fields,
    _validated_searchmoves,
)
from chess_anti_engine.tablebase import get_tablebase, probe_wdl, tb_adjudicate_result
from chess_anti_engine.utils.engine_discovery import (
    REPO_ROOT,
    announce_engine,
    default_stockfish,
    main_checkout,
)
from scripts import audit_label_candidates as gate
from scripts import audit_targets
from scripts import gen_random_selfplay_shards as gen

_LOG = logging.getLogger("gen_sf_rooted_corpus")

#: Row schema.  1 is the first shape that banks a NARROWING STAIRCASE (several
#: phases per position, each with every depth block it emitted).  A consumer
#: that reads a single-search corpus's keys off one of these rows gets a
#: KeyError rather than a plausible wrong number.
#:
#: 2 adds the MOVE HISTORY (``history_root_fen`` / ``history_uci`` /
#: ``history_plies`` / ``history_root_reason``).  A schema-1 row is a bare FEN,
#: so a consumer encoding it fills history slot 0 and leaves slots 1..7 and
#: every repetition plane ZERO -- an input distribution live play never
#: produces (ledger 2026-09-01: the champion flips 46.5% of its top-1 moves and
#: loses +20.2 cp of regret when its history planes are filled the way play
#: fills them).  Schema 2 banks the window that makes a row's encoded input and
#: its Stockfish label both see what play sees; see :func:`history_for`.
#:
#: 3 adds the two dedup keys (``search_key``, ``input_key`` -- see the module
#: docstring) and the per-shard ``tallies`` on progress records.  ⚑ SCHEMA 2 --
#: a window WITHOUT those keys -- was never produced outside smoke runs, and it
#: is REFUSED rather than read: the resume re-warm needs both keys and the
#: deriver verifies every row against ``input_key``, so a schema-2 corpus can
#: only be regenerated (or repaired by a tool that adds the keys).  In the
#: ledger's and the spec's prose "schema 2" names THIS shape; in code it is 3.
#: ⚑ VERIFIED 2026-09-01: schema 2 never produced an accepted corpus (every
#: persistent manifest under ``data/nnue_bootstrap`` reads ``row_schema: 1``;
#: the only schema-2 rows ever written were test fixtures and scratch output),
#: so it is refused with no compatibility carried.  Schema 1 stays readable by
#: the deriver; 3 is the only schema this generator writes.
ROW_SCHEMA = 3
ROW_SCHEMA_HISTORY_WITHOUT_KEYS = 2

#: How many PREVIOUS positions the encoder keeps (``CBOARD_HISTORY_MAX`` in
#: ``chess_anti_engine/encoding/_cboard_impl.h``).  The 8 encoded frames are
#: this many plus the position itself.  Read here rather than imported because
#: it is a C compile-time constant; ``tests/test_corpus_history_banking.py``
#: pins the two together.
HISTORY_WINDOW_PLIES = 7

#: ``history_root_reason``: the window reaches back to the position right after
#: the last irreversible move (no earlier position can repeat with any frame in
#: the window, and no earlier move can matter to the engine's own repetition or
#: fifty-move detection)...
HISTORY_ROOT_IRREVERSIBLE = "irreversible"

#: ...or the game ran out of stack first, so the window is the whole game.  ⚑
#: "The game" is the board the worker plays on, whose stack CARRIES the book
#: moves the opening sampler pushed -- so for a book opening this root is the
#: standard starting position, not the book exit.  For the blind-spot FEN-list
#: branch it is that bare FEN.
HISTORY_ROOT_GAME_START = "game_start"

#: The production input encoding, spelled ONCE for the whole corpus chain.
#: :func:`row_key` hashes the tensor live play would encode for a position
#: (``encode_cboard(CBoard.from_board(board), ...)``, the UCI search-root call)
#: and ``scripts/derive_corpus_targets.py`` re-encodes the reconstructed row
#: with the SAME two constants -- imported from here, because the deriver
#: already imports this module and a second spelling is how the two sides drift
#: apart without any test noticing.  ``tests/test_corpus_history_banking.py``
#: pins the deriver's python encoder to this C call, plane for plane.
INPUT_HISTORY_ENCODING = "lc0_root_legacy_meta"
INPUT_EXTRA_FEATURES = "v2_threats"

#: ⚑⚑ THE REPETITION-PLANE REGIME, and it is part of the input identity.  The C
#: encoder :func:`row_key` goes through records per-slot repetition flags at
#: board construction under a PROCESS-GLOBAL flag
#: (``chess_anti_engine/encoding/rep_fix.py``); its C default is UNFIXED, which
#: under-reports repetitions older than the hash stack the encoder rebuilds
#: after an irreversible move.  Production plays FIXED
#: (``configs/pbt2_small.yaml`` ``history_rep_fix: true`` -- pinned by
#: ``tests/test_gen_sf_rooted_corpus.py``) and the deriver's python encoder is
#: always fixed, so an unfixed generator would (a) bank an ``input_key`` the
#: deriver can never reproduce on any row with a repeat inside the window but
#: outside the 8 frames (measured 2026-09-01 on run03: 77/24,590 rows) and
#: (b) fold two rows that differ only in a repetition plane into one.
#: ``apply_history_rep_fix`` sets it at ``run``/``run_worker`` start, before
#: any CBoard exists; :func:`row_key` REQUIRES it; the manifest, the summary
#: and every row's ``run`` block stamp it; the deriver refuses a schema-3
#: corpus whose stamp is missing or differs from its own regime.
HISTORY_REP_FIX = True
KEY_HISTORY_REP_FIX = "history_rep_fix"

#: Repetition counts above this are indistinguishable to the engine (a 3-fold
#: is a draw whether it is the third or the fifth occurrence), so the signature
#: caps them here rather than growing a key for a state that cannot change a
#: label.
_SEARCH_KEY_REPEAT_CAP = 3

#: Summary schema, versioned separately: the summary gains aggregate keys far
#: more often than a row changes shape, and bumping the row schema for a new
#: histogram would invalidate every banked row for nothing.
SUMMARY_SCHEMA = 1

#: Launch manifest, written BEFORE the first game (see write_launch_manifest).
MANIFEST_SCHEMA = 1
MANIFEST_NAME = "manifest.json"

#: The run record, written once at run END.  ⚑ Its PRESENCE is not a completion
#: claim -- a crash the parent survives writes one too -- so read the verdict it
#: states in ``run_finished`` (see :func:`summary_run_finished`), never the
#: existence of this file.
SUMMARY_NAME = "summary.json"

#: Where a crashed session's records are moved so a resume can write its own
#: (:func:`unfinished_archive_path`).  ONE spelling for both of them -- the
#: in-directory ``summary.json`` and any ``--json`` copy beside it -- because
#: two conventions for the same act is two things to learn and one to get
#: wrong.  The extension is preserved rather than appended to, so an archived
#: record is still a ``.json`` file to every tool that opens one, and the
#: infix keeps the name outside ``shard_glob`` and outside the ``.jsonl.*``
#: suffixes a consumer's inventory globs.
UNFINISHED_ARCHIVE_TEMPLATE = "{stem}.unfinished_{index:02d}{suffix}"

DEFAULT_STAIRCASE = "all:9,16:11,4:13"
STAIRCASE_POLICY_FIXED = "fixed"
STAIRCASE_POLICY_G10 = "g10"
STAIRCASE_POLICIES: tuple[str, ...] = (
    STAIRCASE_POLICY_FIXED,
    STAIRCASE_POLICY_G10,
)
G10_STAIRCASE = "all:9,8:10,4:12"
G10_DECISION_DEPTH = 10
G10_MARGIN_CP = 10.0
DEFAULT_TEMP_PLIES = 20
DEFAULT_TEMP_HIGH = 1.0
DEFAULT_TEMP_LOW = 0.3
DEFAULT_MAX_PLIES = 400
DEFAULT_SHARD_ROWS = 100_000
DEFAULT_SF_HASH_MB = 64
DEFAULT_NICE = 15
DEFAULT_SEED = 20260827
DEFAULT_RUN_ID = "gen_sf_rooted_corpus"

#: Positions one worker's dedup cache holds before it starts evicting.  A bound
#: rather than "however many the run sees": the cache is the one structure here
#: that grows with the corpus instead of with the machine, and a 100M-position
#: burn would otherwise ask for tens of GiB per worker.
#:
#: MEASURED 2026-08-27 (``entry_bytes`` on a 33-legal-move middlegame): **816
#: bytes per entry**, so this default is **~1.5 GiB per worker**.  The same
#: position cost 9,929 bytes as the ``tuple[PvLine, ...]`` the cache used to
#: hold -- 18.5 GiB at this bound, per worker, which is what made the bound and
#: the compaction one change rather than two.  ⚑ The number is not assumed: each
#: run publishes its OWN ``dedup_cache_bytes_per_entry_est``, measured off the
#: entries it actually held, because move counts differ by position mix.
DEFAULT_DEDUP_CACHE_MAX = 2_000_000

#: Deadline on ONE WHOLE ``go`` exchange (search), not one readline -- the
#: clock starts before the read loop, same semantics as the driver's own
#: ``search``.  ``StockfishUCI``'s own default is
#: 60 s, sized for the node-limited searches production runs; this generator's
#: deepest rung is a ``go depth 13`` on a cold table, which is a different order
#: of wall time, and a deadline that expires POISONS the engine (see
#: ``StockfishDesyncError``) rather than retrying.  300 s is the burn-safe
#: setting; it is a flag because the right value follows the staircase.
DEFAULT_SF_READ_TIMEOUT_S = 300.0

#: Whole-search deadline for ONE staircase ``go`` -- the wedge tripwire.  The
#: hot-TT search explosion (ledger AMENDMENT 4, 2026-08-27) turns a sub-second
#: command into minutes-to-hours at 100% CPU, and detection costs the whole
#: wait: EngineLease's respawn + cold retry (~2 s measured) is the escape
#: either way, so the only question is how long we sit before pulling it.  8 s
#: is ~4x the measured legitimate cold deep rung (~2 s) and ~80x the hot
#: median, so a trip is almost certainly an explosion; a false trip costs one
#: respawned cold search and stamps the row ``cold_tt_retry``, never a worker.
#: Deliberately NOT 1 s: that sits BELOW the legitimate cold-search time, so
#: the post-respawn retry itself would trip it and a double timeout abandons a
#: healthy game.  ``--sf-read-timeout`` stays the OUTER deadline for
#: handshakes (NNUE load, SyzygyPath init after a WSL2 drop_caches evicts the
#: tables), which can legitimately stall far past 8 s.
DEFAULT_SF_SEARCH_TIMEOUT_S = 8.0

#: ``--staircase`` phase-1 width spelling for "one PV per legal move".
WIDTH_ALL = "all"

#: A row is banked only at or above this piece count.  Below it the position is
#: in (or one move from) tablebase range, where the search is answering a
#: question the tablebase already answers exactly -- and where this generator
#: ENDS the game by adjudication rather than labelling it.
MIN_BANKED_PIECES = 7

#: At or below this piece count the game is adjudicated from local Syzygy and
#: stopped.  6, not 7: the production pair's 6-man DTZ is what makes the verdict
#: exact, and ``MIN_BANKED_PIECES`` is one above so a banked row is never a
#: position whose value the tablebase would have overruled.
ADJUDICATION_MAX_PIECES = 6

#: Phase-of-game buckets for the dedup disclosure.  PRECEDENCE IS ENDGAME
#: FIRST: a <=9-man position at ply <= 20 is an endgame that arrived early, not
#: an opening, and bucketing it by ply would put tablebase-adjacent positions in
#: the "opening" column where a reader would read them as book exits.
OPENING_MAX_PLY = 20
ENDGAME_MAX_PIECES = 9

PHASE_OPENING = "opening"
PHASE_MIDDLEGAME = "middlegame"
PHASE_ENDGAME = "endgame"
GAME_PHASES: tuple[str, ...] = (PHASE_OPENING, PHASE_MIDDLEGAME, PHASE_ENDGAME)

#: The production Syzygy pair, BY DIRECTORY NAME rather than by absolute path.
#: ``configs/pbt2_small.yaml``'s ``syzygy_path`` is these two directories under
#: the checkout root; spelling the names here and resolving the root at runtime
#: keeps a username out of a public repo AND keeps the default correct in a
#: worktree, where ``data/`` is untracked runtime output living in the MAIN
#: checkout.  ⚑ The names LIE about their contents (``syzygy_3-4-5`` holds 3-6
#: man WDL; the 6-man DTZ is the separate ``syzygy_6``) -- see CLAUDE.md.
#: ``tests/test_gen_sf_rooted_corpus.py`` pins these to the production config.
SYZYGY_DIR_NAMES: tuple[str, str] = ("syzygy_3-4-5", "syzygy_6")

#: RNG stream tags.  The opening draw and the move selection must not share a
#: stream: they are seeded from the same (seed, worker, game) material, and
#: without a tag the book draw would consume exactly the entropy ply 0's Gumbel
#: noise is derived from.
_STREAM_BOOK = 0
_STREAM_SELECT = 1

#: Written into every row so a shard is self-describing after a join.
KEY_TT_CARRIED = "tt_carried_across_phases"


# -- the staircase ------------------------------------------------------------


@dataclass(frozen=True)
class StaircasePhase:
    """One rung: a MultiPV width and the depth searched at that width.

    ``width is None`` is the ``all`` spelling -- one PV per legal move, resolved
    PER POSITION and recorded as realized, never as the ask.
    """

    width: int | None
    depth: int

    @property
    def width_label(self) -> str:
        return WIDTH_ALL if self.width is None else str(self.width)

    @property
    def sort_key(self) -> float:
        """``all`` is wider than any number, which is what orders the rungs."""
        return math.inf if self.width is None else float(self.width)

    def width_for(self, available: int) -> int:
        """The MultiPV this phase asks for given ``available`` root moves."""
        wanted = available if self.width is None else int(self.width)
        return max(1, min(int(wanted), int(available)))


def parse_staircase(spec: str) -> tuple[StaircasePhase, ...]:
    """``"all:9,16:11,4:13"`` -> three rungs, or ``ValueError``.

    Two shape rules, both refused rather than repaired:

    * WIDTHS STRICTLY DESCEND.  A rung no narrower than the one before it is not
      a narrowing, and a staircase that widens again would spend the deep budget
      on moves the shallow scout already refuted.  ``all`` sorts above every
      number, so ``16:11,all:13`` is refused by the same comparison rather than
      by a special case.
    * DEPTHS STRICTLY ASCEND.  A rung no deeper than the one before it re-runs a
      search the transposition table already holds and publishes it as a
      separate observation, which is a duplicate row wearing a new depth label.
    """
    phases: list[StaircasePhase] = []
    for raw in (part.strip() for part in spec.split(",")):
        if not raw:
            continue
        width_token, sep, depth_token = raw.partition(":")
        if not sep:
            raise ValueError(
                f"--staircase rung {raw!r} is not '<width>:<depth>'; widths are "
                f"a positive integer or {WIDTH_ALL!r}",
            )
        width_token = width_token.strip()
        if width_token == WIDTH_ALL:
            width: int | None = None
        else:
            try:
                width = int(width_token)
            except ValueError as exc:
                raise ValueError(
                    f"--staircase rung {raw!r}: width {width_token!r} is neither "
                    f"an integer nor {WIDTH_ALL!r}",
                ) from exc
            if width <= 0:
                raise ValueError(
                    f"--staircase rung {raw!r}: the MultiPV width must be positive",
                )
        try:
            depth = int(depth_token.strip())
        except ValueError as exc:
            raise ValueError(
                f"--staircase rung {raw!r}: depth {depth_token.strip()!r} is not "
                "an integer",
            ) from exc
        if depth <= 0:
            # Stockfish silently replaces `go depth 0` with a real iteration, so
            # a non-positive ask is a limit accepted and then quietly changed.
            raise ValueError(
                f"--staircase rung {raw!r}: the depth must be positive",
            )
        phases.append(StaircasePhase(width=width, depth=depth))
    if not phases:
        raise ValueError("--staircase selected no phases")
    for previous, nxt in itertools.pairwise(phases):
        if nxt.sort_key >= previous.sort_key:
            raise ValueError(
                f"--staircase widths must strictly descend: "
                f"{nxt.width_label} does not narrow {previous.width_label}",
            )
        if nxt.depth <= previous.depth:
            raise ValueError(
                f"--staircase depths must strictly ascend: depth {nxt.depth} "
                f"does not deepen depth {previous.depth}",
            )
    return tuple(phases)


def format_staircase(phases: Sequence[StaircasePhase]) -> str:
    """The canonical spelling of a parsed staircase, for the realized stamp."""
    return ",".join(f"{p.width_label}:{p.depth}" for p in phases)


def validate_staircase_policy(
    policy: str,
    phases: Sequence[StaircasePhase],
) -> str:
    """Return a valid named policy or refuse an unvalidated combination.

    G10 is a measured intervention, not a generic threshold knob: its 10 cp
    threshold was selected on one bank and independently validated on another
    for exactly ``all:9,8:10,4:12``.  Accepting another staircase under the
    same name would stamp untested work as the validated labeler.
    """
    name = str(policy)
    if name not in STAIRCASE_POLICIES:
        raise ValueError(
            f"--staircase-policy {name!r} is not one of "
            f"{', '.join(STAIRCASE_POLICIES)}",
        )
    staircase = format_staircase(phases)
    if name == STAIRCASE_POLICY_G10 and staircase != G10_STAIRCASE:
        raise ValueError(
            f"--staircase-policy {STAIRCASE_POLICY_G10!r} is validated only "
            f"with --staircase {G10_STAIRCASE!r}, got {staircase!r}",
        )
    return name


def staircase_gate_stamp(policy: str) -> dict[str, Any]:
    """The run-level meaning of the policy, including its frozen threshold."""
    name = str(policy)
    if name == STAIRCASE_POLICY_G10:
        return {
            "policy": STAIRCASE_POLICY_G10,
            "adaptive": True,
            "decision_after_phase": 1,
            "decision_depth": G10_DECISION_DEPTH,
            "metric": "effective_cp_rank1_minus_rank2",
            "extend_when": "margin_cp<=threshold_cp",
            "threshold_cp": G10_MARGIN_CP,
            "no_margin_action": "stop",
            "extended_phase": 2,
            "extended_width": 4,
            "extended_depth": 12,
        }
    if name == STAIRCASE_POLICY_FIXED:
        return {"policy": STAIRCASE_POLICY_FIXED, "adaptive": False}
    raise ValueError(f"unknown staircase policy {name!r}")


# -- the info stream ----------------------------------------------------------


@dataclass(frozen=True)
class PvLine:
    """One MultiPV rank at one depth, as first emitted.

    ``nodes`` is the CUMULATIVE search node count Stockfish reported ON THAT
    LINE, banked per line rather than only per depth: the within-iteration
    progression is what lets a later analysis price a width without rerunning
    the search.
    """

    rank: int
    move: str
    effective_cp: float
    nodes: int | None


@dataclass(frozen=True)
class DepthBlock:
    """Every rank one iteration emitted, taken whole.

    ``emissions`` counts the non-bound scored lines SEEN at this depth,
    including re-emissions that lost to the first-wins rule, so
    ``emissions != len(lines)`` is exactly the abort signature.
    """

    depth: int
    lines: tuple[PvLine, ...]
    emissions: int
    complete: bool
    nodes_at_depth: int | None


@dataclass(frozen=True)
class StreamParse:
    """One ``go`` line's whole stream, bucketed by depth.

    ``blocks`` ascends by depth.  The counters are ANOMALY counters, not errors:
    each is aggregated into the run summary so a corpus states how clean the
    searches that produced it were.

    ⚑ ``re_emissions_disagreeing`` IS THE ONE THAT MATTERS.  ``re_emissions``
    counts every line the first-wins rule dropped, and the measured Stockfish
    end-of-search flush makes that number large and harmless.  A DISAGREEING
    re-emission is the case where the banked block would have been different
    under a last-emission-wins rule -- the actual bug this parser exists to not
    have.
    """

    blocks: tuple[DepthBlock, ...]
    re_emissions: int
    re_emissions_disagreeing: int
    bound_lines: int
    unscored_lines: int
    emission_count_violations: int
    duplicate_iteration_flushes: int


def parse_depth_blocks(
    lines: Sequence[str], *, expected_lines: int,
) -> StreamParse:
    """Bucket a ``go depth`` stream into complete per-depth MultiPV blocks.

    ⚑⚑ FIRST EMISSION WINS PER ``(depth, rank)``.  Stockfish re-emits updated
    lines under an OLD depth label when a search is cut short, so keeping the
    last line seen -- which is what ``StockfishUCI``'s accumulator does,
    correctly, for its own single-PV purpose -- silently splices a later
    search's score into an earlier iteration's block.  Every number derived from
    that block is then a blend of two searches and nothing raises.

    ⚑ ``upperbound``/``lowerbound`` lines never participate, not even as the
    first emission: a bound is a claim about an aspiration window, and letting
    one win the rank would freeze a window edge in as the move's score.
    """
    first: dict[int, dict[int, PvLine]] = {}
    emissions: Counter[int] = Counter()
    disagreeing: Counter[int] = Counter()
    re_emissions = 0
    bound_lines = 0
    unscored_lines = 0
    for line in lines:
        parts = line.split()
        if not parts or parts[0] != "info":
            continue
        if "upperbound" in parts or "lowerbound" in parts:
            bound_lines += 1
            continue
        mpv, nodes, depth, cp, mate, _wdl, pv_move = _parse_info_fields(parts)
        if depth is None or pv_move is None:
            # `info string ...`, `info depth N currmove ...` and the periodic
            # nps lines: no rank to bank and no score to bank it with.
            continue
        eff = gate.effective_cp_from_score(cp, mate)
        if eff is None:
            unscored_lines += 1
            continue
        d = int(depth)
        rank = int(mpv if mpv is not None else 1)
        emissions[d] += 1
        ranks = first.setdefault(d, {})
        if rank in ranks:
            re_emissions += 1
            banked = ranks[rank]
            if banked.move != str(pv_move) or banked.effective_cp != float(eff):
                disagreeing[d] += 1
            continue
        ranks[rank] = PvLine(
            rank=rank,
            move=str(pv_move),
            effective_cp=float(eff),
            nodes=None if nodes is None else int(nodes),
        )
    blocks: list[DepthBlock] = []
    violations = 0
    flushes = 0
    for d in sorted(first):
        rank_ids = sorted(first[d])
        ranked = tuple(first[d][r] for r in rank_ids)
        seen = int(emissions[d])
        if seen != int(expected_lines):
            violations += 1
            # The MEASURED benign signature: the whole iteration emitted a
            # second time, every repeat agreeing with what was banked.
            if seen == 2 * int(expected_lines) and disagreeing[d] == 0:
                flushes += 1
        node_counts = [pv.nodes for pv in ranked if pv.nodes is not None]
        blocks.append(DepthBlock(
            depth=d,
            lines=ranked,
            emissions=seen,
            # Ranks 1..W exactly, not cardinality: a stream that emitted ranks
            # {1, 2, 4} has the right COUNT for W=3 while missing a line the
            # width promised -- and the emissions counter cannot see it either,
            # because three lines arrived.
            complete=rank_ids == list(range(1, int(expected_lines) + 1)),
            # Cumulative and monotone within an iteration, so this is the
            # largest count among the lines that WON their rank -- the
            # iteration's count as of its first emission.  ⚑ Not necessarily
            # its last: the measured end-of-search flush re-emits the same ranks
            # with a larger `nodes`, and first-emission-wins discards that
            # update, so read this as a floor on the iteration's final count.
            nodes_at_depth=max(node_counts) if node_counts else None,
        ))
    return StreamParse(
        blocks=tuple(blocks),
        re_emissions=re_emissions,
        re_emissions_disagreeing=int(sum(disagreeing.values())),
        bound_lines=bound_lines,
        unscored_lines=unscored_lines,
        emission_count_violations=violations,
        duplicate_iteration_flushes=flushes,
    )


def deepest_block_with_width(
    blocks: Sequence[DepthBlock], *, want: int,
) -> tuple[DepthBlock, bool]:
    """``(block, full_width)`` -- the deepest iteration carrying ``want`` ranks.

    Falls back to the deepest block with the MOST ranks and reports
    ``full_width=False``.  A fallback is still ONE iteration, never a mix: the
    ranking is narrower than asked for, which is a disclosed fact rather than a
    silently blended one.
    """
    if not blocks:
        raise RuntimeError(
            "the search produced no scored MultiPV line; there is no ranking to "
            "read and imputing one would invent the position's values",
        )
    full = [b for b in blocks if len(b.lines) >= int(want)]
    if full:
        return max(full, key=lambda b: b.depth), True
    best = max(blocks, key=lambda b: (len(b.lines), b.depth))
    return best, False


# -- driving the engine -------------------------------------------------------


@dataclass(frozen=True)
class PhaseResult:
    """One rung as it actually ran."""

    index: int
    width_requested: str
    width_realized: int
    #: Lines in the block the NEXT rung actually consumed -- observed off the
    #: stream, where ``width_realized`` is the MultiPV the request resolved to.
    #: A stream that under-delivers shows up as the two disagreeing.
    width_streamed: int
    depth_requested: int
    searchmoves: tuple[str, ...] | None
    parse: StreamParse

    def as_row(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "width_requested": self.width_requested,
            "width_realized": self.width_realized,
            "width_streamed": self.width_streamed,
            "depth_requested": self.depth_requested,
            "searchmoves": (
                None if self.searchmoves is None else list(self.searchmoves)
            ),
            "per_depth": [
                {
                    "depth": b.depth,
                    "complete": b.complete,
                    "emissions": b.emissions,
                    "nodes_at_depth": b.nodes_at_depth,
                    # (rank, move, effective_cp, cumulative nodes on that line)
                    "lines": [
                        [pv.rank, pv.move, pv.effective_cp, pv.nodes]
                        for pv in b.lines
                    ],
                }
                for b in self.parse.blocks
            ],
            "nodes_at_depth": {
                str(b.depth): b.nodes_at_depth for b in self.parse.blocks
            },
            "anomalies": {
                "re_emissions": self.parse.re_emissions,
                "re_emissions_disagreeing": self.parse.re_emissions_disagreeing,
                "bound_lines": self.parse.bound_lines,
                "unscored_lines": self.parse.unscored_lines,
                "emission_count_violations": self.parse.emission_count_violations,
                "duplicate_iteration_flushes": self.parse.duplicate_iteration_flushes,
            },
        }

    @property
    def nodes_total(self) -> int:
        counts = [
            b.nodes_at_depth for b in self.parse.blocks if b.nodes_at_depth is not None
        ]
        return max(counts) if counts else 0


@dataclass(frozen=True)
class StaircaseGateDecision:
    """The live G10 decision banked beside the phases it controlled."""

    margin_cp: float | None
    extended: bool
    reason: str
    decision_depth_observed: int

    def as_row(self) -> dict[str, Any]:
        return {
            **staircase_gate_stamp(STAIRCASE_POLICY_G10),
            "margin_cp": self.margin_cp,
            "extended": self.extended,
            "reason": self.reason,
            "decision_depth_observed": self.decision_depth_observed,
        }


@dataclass(frozen=True)
class PositionSearch:
    """Everything one position's staircase produced."""

    phases: tuple[PhaseResult, ...]
    #: Present exactly when an adaptive policy evaluated the final rung.
    staircase_gate: StaircaseGateDecision | None
    #: The value vector move selection runs on: the deepest FULL-WIDTH phase-1
    #: block, in rank order, from the ROOT MOVER's seat (a rooted MultiPV list
    #: already reports each root move from the mover's POV, so nothing is
    #: negated here -- the gate's `RootedStockfishArm` documents the same trap).
    values: tuple[PvLine, ...]
    value_depth: int
    value_full_width: bool
    #: The move window every rung of this staircase was searched on -- the SAME
    #: object the bank site writes into the row.  ⚑ REQUIRED, with no default:
    #: there is exactly one constructor and a defaulted ``None`` would let a
    #: future one bank a row whose window nobody built.
    history: RowHistory


@dataclass(frozen=True, eq=False, slots=True)
class SelectionValues:
    """The two arrays move selection reads, and NOTHING else.

    A ``PositionSearch`` is what gets BANKED; this is what gets CACHED, and the
    difference is the whole reason the type exists.  Selection needs a move to
    play and a value per move; the banked ``PvLine`` also carries a rank and a
    per-line node count, which are an order of magnitude more bytes and are
    already in the shard.  MEASURED 2026-08-27 on a 33-legal-move middlegame:
    **816 bytes** here against **9,929** for the ``tuple[PvLine, ...]`` this
    replaced in the cache -- 12.2x, and the difference between ~1.5 GiB and
    ~18.5 GiB per worker at ``DEFAULT_DEDUP_CACHE_MAX``.

    ⚑ ``eq=False`` on purpose: a generated ``__eq__`` over a numpy field returns
    an ARRAY, and ``a == b`` then raises "truth value is ambiguous" at whatever
    call site first compares two of these.  Identity is the honest default for a
    cache entry.

    ⚑ ``float32``, not ``float64``.  Effective cp is integral centipawns and the
    mate band tops out around 1e5, both exact in float32 (integers are exact to
    2**24), so the narrower dtype is free here.  ``q_of`` widens back to float64
    before the shared mapping sees it, and selection therefore reads the same
    number on a cache HIT as it did on the first-seen search -- which is why
    ``play_game`` builds this object once and selects through it BOTH times
    rather than selecting off the ``PvLine`` list the first time.
    """

    moves: tuple[str, ...]
    effective_cp: np.ndarray

    @classmethod
    def from_lines(cls, lines: Sequence[PvLine]) -> SelectionValues:
        """Compact one ranked block.  ⚑ The uci strings are INTERNED.

        A uci move is a from-square, a to-square and an optional promotion, so
        the spellings Stockfish can ever emit are a few thousand objects at
        most -- while a corpus worker caches millions of positions whose move
        lists are drawn from exactly that set.  Interning makes every cached
        list share one object per spelling, which is where most of the 12x came
        from: without it each entry pays ~1.8 KB for 35 strings it holds
        millions of copies of.
        """
        return cls(
            moves=tuple(sys.intern(str(pv.move)) for pv in lines),
            effective_cp=np.asarray(
                [pv.effective_cp for pv in lines], dtype=np.float32,
            ),
        )


#: Node-count observations one phase's median estimator keeps.  4096 ints is
#: 32 KB per phase and does not grow; the exact median needs every observation,
#: which at three searches per position is ~84 MB per million positions per
#: worker held for the lifetime of the run to produce four numbers.
NODES_RESERVOIR_MAX = 4096

#: Seed for the reservoir's replacement draws, mixed with the phase index.
#: Fixed rather than entropy: a summary statistic that moves between two
#: byte-identical runs is a number nobody can diff, and the reservoir is the
#: only place in this file where the SUMMARY (not the corpus) is sampled.
_RESERVOIR_SEED = 20260827


@dataclass
class NodeSamples:
    """Running node-count aggregates for one staircase phase.

    Everything here is O(1) in the number of searches except the reservoir,
    which is bounded by ``NODES_RESERVOIR_MAX``.

    ⚑ THE MEDIAN IS AN ESTIMATE AND ITS KEY SAYS SO.  ``median_est_reservoir``
    is the median of a uniform sample of the phase's node counts, not of the
    phase's node counts -- a distinction that is invisible in a summary whose
    key is called ``median``, and this file's whole posture is that a number
    states what it is.
    """

    n: int = 0
    total: int = 0
    minimum: int = 0
    maximum: int = 0
    reservoir: list[int] = field(default_factory=list)
    log2_buckets: Counter[int] = field(default_factory=Counter)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(_RESERVOIR_SEED),
    )

    def add(self, nodes: int) -> None:
        value = int(nodes)
        self.n += 1
        self.total += value
        self.minimum = value if self.n == 1 else min(self.minimum, value)
        self.maximum = value if self.n == 1 else max(self.maximum, value)
        # -1 is the "no node count reported" bucket, as it was when this was a
        # separate counter.
        self.log2_buckets[int(math.log2(value)) if value > 0 else -1] += 1
        if len(self.reservoir) < NODES_RESERVOIR_MAX:
            self.reservoir.append(value)
            return
        # Algorithm R: observation `n` replaces a uniformly chosen slot with
        # probability capacity/n, which keeps the kept sample uniform over
        # everything seen without the kept sample ever growing.
        slot = int(self.rng.integers(0, self.n))
        if slot < NODES_RESERVOIR_MAX:
            self.reservoir[slot] = value

    def summary(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": (self.total / self.n) if self.n else math.nan,
            "median_est_reservoir": (
                statistics.median(self.reservoir) if self.reservoir else math.nan
            ),
            "median_est_reservoir_n": len(self.reservoir),
            "median_est_reservoir_capacity": NODES_RESERVOIR_MAX,
            "min": self.minimum if self.n else 0,
            "max": self.maximum if self.n else 0,
            "total": int(self.total),
            # log2 buckets rather than raw values: a per-search list over a
            # million-position corpus is a second corpus, and the shape is what
            # a cost model needs. -1 is the "no node count reported" bucket.
            "log2_buckets": {
                str(k): int(v) for k, v in sorted(self.log2_buckets.items())
            },
        }


@dataclass
class SearchStats:
    """Anomaly and cost counters, accumulated over a worker's whole run."""

    searches: int = 0
    positions: int = 0
    search_s: float = 0.0
    re_emissions: int = 0
    re_emissions_disagreeing: int = 0
    bound_lines: int = 0
    unscored_lines: int = 0
    emission_count_violations: int = 0
    duplicate_iteration_flushes: int = 0
    incomplete_final_blocks: int = 0
    selection_not_full_width: int = 0
    staircase_gate_evaluations: int = 0
    staircase_gate_extended: int = 0
    staircase_gate_stopped: int = 0
    staircase_gate_forced_stops: int = 0
    #: TT-hygiene counters live HERE, not on the searcher, because a wedged
    #: engine is replaced mid-run (see EngineLease) and a fresh searcher must
    #: not zero the observations the realized stamp is built from.
    new_game_calls: int = 0
    tt_cleared_mid_position: int = 0
    engine_respawns: int = 0
    nodes_by_phase: dict[int, NodeSamples] = field(default_factory=dict)

    def add_phase(self, phase: PhaseResult) -> None:
        self.searches += 1
        self.re_emissions += phase.parse.re_emissions
        self.re_emissions_disagreeing += phase.parse.re_emissions_disagreeing
        self.bound_lines += phase.parse.bound_lines
        self.unscored_lines += phase.parse.unscored_lines
        self.emission_count_violations += phase.parse.emission_count_violations
        self.duplicate_iteration_flushes += phase.parse.duplicate_iteration_flushes
        samples = self.nodes_by_phase.get(phase.index)
        if samples is None:
            # Per-phase stream, so the phases' reservoirs do not replace the
            # same slots at the same counts as each other.
            samples = NodeSamples(
                rng=np.random.default_rng([_RESERVOIR_SEED, int(phase.index)]),
            )
            self.nodes_by_phase[phase.index] = samples
        samples.add(phase.nodes_total)

    def add_gate(self, decision: StaircaseGateDecision) -> None:
        self.staircase_gate_evaluations += 1
        if decision.extended:
            self.staircase_gate_extended += 1
        else:
            self.staircase_gate_stopped += 1
        if decision.margin_cp is None:
            self.staircase_gate_forced_stops += 1

    def summary(self) -> dict[str, Any]:
        return {
            "positions_searched": self.positions,
            "searches": self.searches,
            "search_s": self.search_s,
            "s_per_position": (
                self.search_s / self.positions if self.positions else math.nan
            ),
            "anomalies": {
                "re_emissions": self.re_emissions,
                "re_emissions_disagreeing": self.re_emissions_disagreeing,
                "bound_lines": self.bound_lines,
                "unscored_lines": self.unscored_lines,
                "emission_count_violations": self.emission_count_violations,
                "duplicate_iteration_flushes": self.duplicate_iteration_flushes,
                "incomplete_final_blocks": self.incomplete_final_blocks,
                "selection_not_full_width": self.selection_not_full_width,
            },
            "engine_respawns": self.engine_respawns,
            "staircase_gate": {
                "evaluations": self.staircase_gate_evaluations,
                "extended": self.staircase_gate_extended,
                "stopped": self.staircase_gate_stopped,
                "forced_stops": self.staircase_gate_forced_stops,
            },
            "nodes_by_phase": {
                str(idx): samples.summary()
                for idx, samples in sorted(self.nodes_by_phase.items())
            },
        }


# -- the banked move window ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class RowHistory:
    """The move window a row is banked with AND the one its label is searched on.

    ONE object serves both, on purpose.  ``search_position`` builds it, hands it
    to :func:`position_command` for the engine, and hands the same instance back
    on the ``PositionSearch`` for the bank site to write out -- so the window in
    the row is the window the engine saw, structurally, rather than by two call
    sites agreeing.

    ⚑ ``fen`` is the position's own FEN under python-chess's DEFAULT ep policy,
    which is what the row's ``fen`` field has always been and what
    ``_validated_searchmoves`` and ``dedup_key`` read.  ``root_fen`` is written
    with ``en_passant="fen"`` instead: the default policy PRINTS AN EP SQUARE
    ONLY WHEN AN EP CAPTURE IS LEGAL, so a root that sits right after a double
    pawn push with no legal capture would come back from ``chess.Board(...)``
    with ``ep_square = None`` -- and the encoder's legacy-meta ep plane (110)
    reads ``ep_square`` RAW, without the legality test.  That is a plane that
    differs from live play, so the root keeps the pseudo-legal square.
    """

    #: The position the row is about.
    fen: str
    #: The window's first position, ``en_passant="fen"``.
    root_fen: str
    #: Root -> row, oldest first.
    uci: tuple[str, ...]
    #: ``HISTORY_ROOT_IRREVERSIBLE`` or ``HISTORY_ROOT_GAME_START``.
    reason: str

    @property
    def plies(self) -> int:
        return len(self.uci)

    def as_row_fields(self) -> dict[str, Any]:
        """The schema-2 keys, spelled once."""
        return {
            "history_root_fen": self.root_fen,
            "history_uci": list(self.uci),
            "history_plies": self.plies,
            "history_root_reason": self.reason,
        }


def history_for(board: chess.Board) -> RowHistory:
    """The window ``board`` is banked and searched with.

    ⚑ THE ROOT DEFINITION, and it is a repetition-state guarantee rather than a
    convenience.  Let ``P`` be the position ``HISTORY_WINDOW_PLIES`` plies back
    (or the game's first position if fewer exist).  The root is ``P`` walked
    back ``halfmove_clock(P)`` further plies -- the first position at or before
    ``P`` whose halfmove clock is 0, i.e. the one right after the last
    irreversible move -- or the game's own start if the stack empties first.

    Two things need that much and no more:

    * **The 8 encoded frames.**  Every frame is ``P`` or later.  No position
      after an irreversible move can equal any position before it (a pawn never
      un-moves and a captured piece never comes back), so a frame's repetition
      partners all lie at or after the start of that frame's own reversible
      run -- and the earliest such run start over all 8 frames is exactly the
      root.  Anything earlier cannot change a plane; anything later can lose one.
    * **The engine's own detection for the row's position.**  Its repetition and
      fifty-move state needs the moves since the last irreversible move, which
      is a SUFFIX of the same window.

    Bound: ``7 + 100`` plies by the fifty-move rule, so ~100 UCI tokens per row;
    inside a zstd shard consecutive rows share almost all of that prefix.

    The replay is CHECKED here, not asserted downstream: a row that cannot
    reproduce its own FEN from its own window is never written.
    """
    fen = board.fen()
    walk = board.copy(stack=True)
    moves: list[str] = []
    for _ in range(HISTORY_WINDOW_PLIES):
        if not walk.move_stack:
            break
        moves.append(walk.pop().uci())
    # ⚑ Read AFTER the first walk: it is P's clock, not the row position's.  An
    # irreversible move inside the window makes the row's own clock small while
    # P's is large, and it is P's run that has to be covered.
    for _ in range(int(walk.halfmove_clock)):
        if not walk.move_stack:
            break
        moves.append(walk.pop().uci())
    moves.reverse()
    reason = (
        HISTORY_ROOT_GAME_START if not walk.move_stack
        else HISTORY_ROOT_IRREVERSIBLE
    )
    history = RowHistory(
        fen=fen,
        root_fen=walk.fen(en_passant="fen"),
        uci=tuple(moves),
        reason=reason,
    )
    replayed = chess.Board(history.root_fen)
    for uci in history.uci:
        replayed.push(chess.Move.from_uci(uci))
    if replayed.fen() != fen:
        raise RuntimeError(
            f"the banked window does not reproduce its own position: replaying "
            f"{list(history.uci)} from {history.root_fen!r} gives "
            f"{replayed.fen()!r}, not {fen!r}",
        )
    return history


def position_command(history: RowHistory) -> str:
    """The ONE ``position`` line every search in this file sends.

    ⚑ ``position fen <root> moves ...`` rather than ``position fen <row>``:
    with the moves the engine can see its own repetitions and count its own
    fifty-move clock, so the LABEL is computed under the same history the row's
    encoded input carries.  A bare-FEN send is history-blind, and a
    ``position``-sender that spelled its own line would be blind again the next
    time one is added -- hence one helper and no second speller.
    """
    if not history.uci:
        return f"position fen {history.root_fen}"
    return f"position fen {history.root_fen} moves {' '.join(history.uci)}"


class StaircaseSearcher:
    """One engine, driven through the staircase, position after position.

    ⚑ THE INFO LINES ARE READ RAW rather than through ``StockfishUCI.search``.
    That method folds the stream into one result per RANK as it goes, and the
    depth each rank came from -- the whole point of this corpus -- is not in
    that result.  The lock and the protocol section are taken exactly as
    ``search`` takes them, so a failure here poisons the engine rather than
    desyncing it; the deadline is the searcher's own ``search_timeout_s``
    (the explosion tripwire), not the engine-wide handshake deadline.  ``searchmoves`` goes through the driver's own
    ``_validated_searchmoves`` for the reason the module docstring gives.
    """

    def __init__(
        self,
        *,
        engine: StockfishUCI,
        staircase: Sequence[StaircasePhase],
        cp_slope: float,
        cp_draw_width: float,
        stats: SearchStats | None = None,
        search_timeout_s: float = DEFAULT_SF_SEARCH_TIMEOUT_S,
        staircase_policy: str = STAIRCASE_POLICY_FIXED,
    ) -> None:
        self.engine = engine
        self.staircase = tuple(staircase)
        self.staircase_policy = validate_staircase_policy(
            staircase_policy,
            self.staircase,
        )
        self.cp_slope = float(cp_slope)
        self.cp_draw_width = float(cp_draw_width)
        # The per-``go`` explosion tripwire; the engine's own read_timeout_s
        # stays on handshakes.  See DEFAULT_SF_SEARCH_TIMEOUT_S.
        self.search_timeout_s = float(search_timeout_s)
        # Shared by EngineLease across respawns: the counters below are
        # observations about the WORKER's whole run, not about one engine.
        self.stats = stats if stats is not None else SearchStats()
        # What the engine's MultiPV option currently is.  Resending an unchanged
        # value would cost a round trip per phase for nothing.
        self._engine_multipv = int(engine.multipv)
        #: True only on a search that EngineLease re-ran on a fresh engine
        #: after a wedge -- that search saw a COLD table, and the row it
        #: produced says so.  A bare searcher never retries, so it stays False.
        self.cold_tt_retry_last = False

    @property
    def new_game_calls(self) -> int:
        """``ucinewgame``s actually delivered, counted AFTER the engine call
        returns -- the worker compares this against games started, which is
        what lets ``tt_cleared_per_game`` in the realized stamp FAIL instead
        of echoing a constant."""
        return self.stats.new_game_calls

    @property
    def tt_cleared_mid_position(self) -> int:
        """Clears observed INSIDE a position's staircase, which would void the
        carried-TT premise the module docstring states.  Structurally zero
        today; the counter exists so the stamp is an observation."""
        return self.stats.tt_cleared_mid_position

    # -- protocol ---------------------------------------------------------

    def stream(
        self,
        history: RowHistory,
        *,
        depth: int,
        multipv: int,
        searchmoves: Sequence[str] | None = None,
    ) -> list[str]:
        """Drive one ``go depth`` and return every line before ``bestmove``.

        ⚑ Takes the WINDOW, not a FEN.  The engine is set up with
        :func:`position_command`, so every rung of every staircase -- including
        the cold-TT retry, which re-enters here through ``search_position`` --
        sees the same move history the row banks.  ``searchmoves`` is still
        validated against the TERMINAL position, which is what the moves name.
        """
        tokens = (
            _validated_searchmoves(history.fen, searchmoves) if searchmoves else []
        )
        with self.engine._lock, self.engine._protocol_section():
            if int(multipv) != self._engine_multipv:
                self.engine._send(f"setoption name MultiPV value {int(multipv)}")
                self.engine._send("isready")
                self.engine._wait_for("readyok")
                self._engine_multipv = int(multipv)
            self.engine._send(position_command(history))
            go_cmd = f"go depth {int(depth)}"
            if tokens:
                # ⚑ LAST PARAMETER ON THE LINE, always: per the UCI spec
                # `searchmoves` consumes every remaining token, so anything
                # appended after it is swallowed as a move.
                go_cmd = f"{go_cmd} searchmoves {' '.join(tokens)}"
            self.engine._send(go_cmd)
            # The explosion tripwire, NOT the engine-wide read deadline: a
            # staircase ``go`` is sub-second hot and ~2 s cold, while the
            # hot-TT explosion runs unbounded.  Expiry poisons the engine and
            # surfaces as StockfishTimeoutError, which EngineLease answers
            # with a respawn and one cold retry.
            deadline = time.monotonic() + self.search_timeout_s
            lines: list[str] = []
            while True:
                line = self.engine._readline_with_deadline(deadline).strip()
                if line.startswith("bestmove"):
                    return lines
                if line:
                    lines.append(line)

    def new_game(self) -> None:
        """``ucinewgame`` -- the per-GAME table clear.  See the module docstring."""
        self.engine.new_game()
        self.stats.new_game_calls += 1

    # -- the staircase ----------------------------------------------------

    def search_position(self, board: chess.Board) -> PositionSearch:
        """Run every rung on ``board`` and pick the selection value vector."""
        history = history_for(board)
        fen = history.fen
        legal = [move.uci() for move in board.legal_moves]
        if not legal:
            raise RuntimeError(
                f"search_position was handed a terminal position {fen!r}; a "
                "game-over board emits no PV lines and is never searched",
            )
        started = time.perf_counter()
        self.cold_tt_retry_last = False
        clears_at_entry = self.new_game_calls
        results: list[PhaseResult] = []
        candidates: list[str] = legal
        gate_decision: StaircaseGateDecision | None = None
        for index, phase in enumerate(self.staircase):
            if (
                self.staircase_policy == STAIRCASE_POLICY_G10
                and index == len(self.staircase) - 1
            ):
                previous = results[-1]
                previous_block, full = deepest_block_with_width(
                    previous.parse.blocks,
                    want=previous.width_realized,
                )
                if (
                    previous_block.depth != G10_DECISION_DEPTH
                    or not full
                    or not previous_block.complete
                ):
                    gate_decision = StaircaseGateDecision(
                        margin_cp=None,
                        extended=False,
                        reason="decision_block_incomplete",
                        decision_depth_observed=previous_block.depth,
                    )
                elif len(previous_block.lines) < 2:
                    gate_decision = StaircaseGateDecision(
                        margin_cp=None,
                        extended=False,
                        reason="fewer_than_two_moves",
                        decision_depth_observed=previous_block.depth,
                    )
                else:
                    margin_cp = float(
                        previous_block.lines[0].effective_cp
                        - previous_block.lines[1].effective_cp,
                    )
                    extended = margin_cp <= G10_MARGIN_CP
                    gate_decision = StaircaseGateDecision(
                        margin_cp=margin_cp,
                        extended=extended,
                        reason=(
                            "margin_at_or_below_threshold"
                            if extended else "margin_above_threshold"
                        ),
                        decision_depth_observed=previous_block.depth,
                    )
                if not gate_decision.extended:
                    break
            width = phase.width_for(len(candidates))
            # Phase 0 searches the full move list, so it names no `searchmoves`
            # and its `go` line is byte-identical to an unrestricted rooted
            # search.  Later phases always name theirs.
            restrict = None if index == 0 else tuple(candidates[:width])
            lines = self.stream(
                history, depth=phase.depth, multipv=width, searchmoves=restrict,
            )
            parse = parse_depth_blocks(lines, expected_lines=width)
            block, full = deepest_block_with_width(parse.blocks, want=width)
            if not full:
                self.stats.incomplete_final_blocks += 1
            result = PhaseResult(
                index=index,
                width_requested=phase.width_label,
                width_realized=width,
                width_streamed=len(block.lines),
                depth_requested=phase.depth,
                searchmoves=restrict,
                parse=parse,
            )
            results.append(result)
            self.stats.add_phase(result)
            candidates = [pv.move for pv in block.lines]
        self.stats.search_s += time.perf_counter() - started
        self.stats.positions += 1
        if gate_decision is not None:
            self.stats.add_gate(gate_decision)
        if self.new_game_calls != clears_at_entry:
            self.stats.tt_cleared_mid_position += 1

        # ⚑ SELECTION READS PHASE 1, not the deepest phase.  It is the deepest
        # depth at which EVERY legal move has a value; a narrowed phase has
        # values for a subset, and sampling over a subset would silently make
        # the staircase's own pruning part of the selection policy.
        value_block, full_width = deepest_block_with_width(
            results[0].parse.blocks, want=len(legal),
        )
        if not full_width:
            self.stats.selection_not_full_width += 1
        return PositionSearch(
            phases=tuple(results),
            staircase_gate=gate_decision,
            values=value_block.lines,
            value_depth=value_block.depth,
            value_full_width=full_width,
            history=history,
        )

    def q_of(self, values: SelectionValues) -> np.ndarray:
        """Root-seat q in [-1, 1] for a value vector, through the SHARED map."""
        return gate.q_from_effective_cp(
            np.asarray(values.effective_cp, dtype=np.float64),
            slope=self.cp_slope,
            draw_width_cp=self.cp_draw_width,
        )

    def realized(self) -> dict[str, Any]:
        """Settings READ OFF THE LIVE ENGINE, never echoed from the args.

        Every field here is an attribute of the object that talked to Stockfish
        (or of the process this code is running in), so a knob that was accepted
        and then dropped on the way to the engine shows up as a disagreement
        with the requested stamp rather than as a matching number.
        """
        return {
            "sf_hash_mb": self.engine.hash_mb,
            "sf_threads": self.engine.threads,
            "sf_syzygy_path": self.engine.syzygy_path,
            "sf_nice": self.engine.nice,
            "sf_binary": self.engine.path,
            "sf_read_timeout_s": self.engine.read_timeout_s,
            "sf_search_timeout_s": self.search_timeout_s,
            "sf_multipv_current": self._engine_multipv,
            "staircase": format_staircase(self.staircase),
            "staircase_policy": self.staircase_policy,
            "staircase_gate": staircase_gate_stamp(self.staircase_policy),
            "cp_slope": self.cp_slope,
            "cp_draw_width": self.cp_draw_width,
            # Observed, not asserted: the first hardcoded `True` here survived
            # to review, and a stamp that cannot fail is the repo's signature
            # defect.  `tt_cleared_per_game` needs the games count and is
            # stamped by the worker, from `new_game_calls`.
            KEY_TT_CARRIED: self.tt_cleared_mid_position == 0,
            "ucinewgame_calls": self.new_game_calls,
        }


class EngineLease:
    """A searcher that survives its engine.

    MEASURED 2026-08-27 (worker 8 game 488 ply 152, then two more workers
    within the hour): Stockfish dev-20260420 can wedge at 100% CPU on a
    narrowed ``go depth`` with a hot transposition table -- no info lines,
    ``stop`` ignored -- so no UCI-level escape exists, and ``movetime``/node
    caps sit behind the same never-reached check.  The dev-20260810 build
    wedges too, on a different position.  A wedge is ENGINE state: the same
    command on a cold table finishes in seconds.  So the recovery is a new
    engine -- close the old process group, spawn a fresh one, re-run the
    position ONCE.  The retried search saw a COLD table and its row says so
    (``cold_tt_retry``); ``engine_respawns`` in the shared stats counts every
    replacement.  A retry that wedges AGAIN propagates, and ``play_game``
    abandons the GAME rather than the worker.

    The stats object is created HERE and threaded into every searcher this
    lease spawns, so a replacement cannot zero the observations the realized
    stamp is built from.
    """

    def __init__(
        self, factory: Callable[[SearchStats], StaircaseSearcher],
    ) -> None:
        self._factory = factory
        self.stats = SearchStats()
        self.searcher = factory(self.stats)

    # -- the searcher surface ``play_game`` uses --------------------------

    def new_game(self) -> None:
        self.searcher.new_game()

    def q_of(self, values: SelectionValues) -> np.ndarray:
        return self.searcher.q_of(values)

    @property
    def tt_cleared_mid_position(self) -> int:
        return self.searcher.tt_cleared_mid_position

    @property
    def cold_tt_retry_last(self) -> bool:
        return self.searcher.cold_tt_retry_last

    def realized(self) -> dict[str, Any]:
        return self.searcher.realized()

    def close(self) -> None:
        # Suppressed on purpose, here only: close() runs on the way OUT of a
        # recorded failure, and an engine that is already gone must not turn
        # that into an unrecorded one.
        with contextlib.suppress(Exception):
            self.searcher.engine.close()

    def respawn(self) -> None:
        self.close()
        self.stats.engine_respawns += 1
        self.searcher = self._factory(self.stats)

    def search_position(self, board: chess.Board) -> PositionSearch:
        try:
            return self.searcher.search_position(board)
        except StockfishTimeoutError:
            self.respawn()
            try:
                search = self.searcher.search_position(board)
            except StockfishTimeoutError:
                # The fresh engine wedged on the same position. It abandoned a
                # protocol exchange mid-search, so the driver would refuse it
                # with StockfishDesyncError on its NEXT use -- replace it
                # again before re-raising, so the caller abandons the game
                # onto a lease that is already clean for the next one.
                self.respawn()
                raise
            self.searcher.cold_tt_retry_last = True
            return search


# -- move selection -----------------------------------------------------------


def selection_rng(
    *, seed: int, worker_id: int, game_id: int, ply: int,
) -> np.random.Generator:
    """The per-ply Gumbel stream.  NO WALL CLOCK anywhere in the material.

    A corpus whose move choices cannot be replayed is a corpus whose selection
    bias cannot be audited, so the seed material is exactly
    ``(seed, worker, game, stream tag, ply)`` and a rerun with the same
    ``--seed`` reproduces every draw.
    """
    return np.random.default_rng(
        [int(seed), int(worker_id), int(game_id), _STREAM_SELECT, int(ply)],
    )


def book_rng(*, seed: int, worker_id: int, game_id: int) -> np.random.Generator:
    """The per-game opening stream, tagged apart from the selection stream."""
    return np.random.default_rng(
        [int(seed), int(worker_id), int(game_id), _STREAM_BOOK],
    )


def temperature_for(
    ply: int, *, temp_plies: int, temp_high: float, temp_low: float,
) -> tuple[float, str]:
    """``(tau, schedule phase)`` for this ply.

    The boundary is ``ply < temp_plies`` -- ``--temp-plies 20`` means plies 0..19
    are hot, which is the count a reader of the flag expects.  ``0`` therefore
    means "never hot", a usable setting rather than an off-by-one.
    """
    if int(ply) < int(temp_plies):
        return float(temp_high), "high"
    return float(temp_low), "low"


def gumbel_choice(q: np.ndarray, *, temp: float, rng: np.random.Generator) -> int:
    """``argmax_k(q_k / tau + g_k)`` with ``g ~ Gumbel(0, 1)``.

    ⚑ ``tau`` DIVIDES THE VALUE rather than scaling the noise, so a smaller
    ``tau`` sharpens toward the argmax and a larger one flattens toward uniform.
    A non-positive ``tau`` is refused: the limit is a deterministic argmax, and
    silently taking it would turn a typo into a corpus with no exploration at
    all and no way to see that from the rows.
    """
    tau = float(temp)
    if not tau > 0.0 or not math.isfinite(tau):
        raise ValueError(
            f"the selection temperature must be finite and positive, got {temp!r}",
        )
    values = np.asarray(q, dtype=np.float64)
    noise = np.asarray(rng.gumbel(size=int(values.shape[0])), dtype=np.float64)
    return int(np.argmax(values / tau + noise))


# -- dedup --------------------------------------------------------------------


def dedup_key(board: chess.Board) -> str:
    """The FEN MINUS the fullmove counter; the halfmove clock is KEPT.

    The fullmove number cannot change a search: it is a move ordinal.  The
    halfmove clock CAN and does -- it decides how close the fifty-move rule is,
    which changes the score of every drawish line and the tablebase verdict
    outright -- so folding it out of the key would serve one position's values
    for another position's search.
    """
    return " ".join(board.fen().split(" ")[:5])


def search_key(board: chess.Board) -> str:
    """LABEL equivalence: ``dedup_key`` plus the reversible segment's repeats.

    ⚑ THE MINIMUM SAFE SIGNATURE, and why it is enough: Stockfish consults game
    history only through ``is_draw``/``has_game_cycle``, and both read
    REPETITION COUNTS over the positions since the last irreversible move --
    nothing else about the route.  So two routes whose reversible segments hold
    the same multiset of repeated positions get the same search, and the
    2026-09-01 legacy-corpus calibration measured exactly that: 0/1200 label
    changes between the history-blind and history-aware sends where no repeat
    sits in the segment, 12.50% top-1 changes where one does.

    The segment is walked back ``halfmove_clock`` plies from the position
    itself (the game start bounds it), every position's polyglot Zobrist hash
    is counted, and the hashes seen at least twice are serialised with their
    counts (capped at ``_SEARCH_KEY_REPEAT_CAP``) in a fixed order.  A route
    with no repeat serialises to ``<dedup_key>|`` -- the old key with an empty
    signature -- so the cache still folds every transposition that IS
    label-equivalent.
    """
    counts: Counter[int] = Counter()
    walk = board.copy(stack=True)
    counts[chess.polyglot.zobrist_hash(walk)] += 1
    for _ in range(int(board.halfmove_clock)):
        if not walk.move_stack:
            break
        walk.pop()
        counts[chess.polyglot.zobrist_hash(walk)] += 1
    repeats = sorted(
        (h, min(c, _SEARCH_KEY_REPEAT_CAP)) for h, c in counts.items() if c >= 2
    )
    return f"{dedup_key(board)}|{','.join(f'{h:016x}:{c}' for h, c in repeats)}"


def input_tensor_key(planes: np.ndarray) -> str:
    """The identity of one encoded input: blake2b-128 over its float32 bytes.

    Spelled once and called from BOTH ends of the chain: the generator hashes
    the tensor live play would encode (:func:`row_key`) and the deriver hashes
    the tensor it reconstructed from the banked window, and a row whose two
    hashes differ is refused there.  ``float32`` and C order are forced so the
    bytes are the same whichever encoder produced the array.
    """
    return hashlib.blake2b(
        np.ascontiguousarray(planes, dtype=np.float32).tobytes(), digest_size=16,
    ).hexdigest()


def apply_history_rep_fix() -> None:
    """Put the process in production's repetition-plane regime.

    Called at the top of ``run`` (the parent: before the engine handshake and
    before any opening book is warmed) and of ``run_worker`` (each worker is a
    SPAWNED process, so a C global set in the parent does not reach it).  No
    CBoard exists in either process before this call -- every one the
    generator builds is inside :func:`row_key`, after it -- which is what
    ``boards_discarded=True`` certifies.
    """
    rep_fix.apply(HISTORY_REP_FIX, boards_discarded=True)


def require_history_rep_fix() -> None:
    """The precondition :func:`row_key` checks, spelled once and LOUD.

    A wrong or unset regime here would hash planes the deriver cannot
    reproduce; raising is the only honest response, because the C global is
    write-only and the planes carry no mark of which regime made them.
    """
    if rep_fix.current() is not HISTORY_REP_FIX:
        raise RuntimeError(
            f"history_rep_fix is {rep_fix.current()!r} in this process but the "
            f"corpus is defined under {HISTORY_REP_FIX}; call "
            "apply_history_rep_fix() before building any board -- the C "
            "encoder would otherwise hash repetition planes the deriver "
            "never produces",
        )


def row_key(board: chess.Board) -> str:
    """MODEL-INPUT identity: the hash of the tensor live play encodes for ``board``.

    ⚑ THE C PLAY PATH, verbatim -- ``encode_cboard(CBoard.from_board(board))``
    is the call the UCI search makes on its root -- so "same row" means "the
    net would see the same 175 planes", history frames and repetition planes
    included, rather than "same FEN".  ~0.35 ms per position beside a search
    of order 100 ms.  ⚑ Under ``HISTORY_REP_FIX``, checked on every call.
    """
    require_history_rep_fix()
    return input_tensor_key(
        encode_cboard(
            CBoard.from_board(board),
            input_history_encoding=INPUT_HISTORY_ENCODING,
            input_extra_features=INPUT_EXTRA_FEATURES,
        ),
    )


def game_phase(*, ply: int, piece_count: int) -> str:
    """Which dedup bucket a position belongs to.  Endgame wins -- see the constants."""
    if int(piece_count) <= ENDGAME_MAX_PIECES:
        return PHASE_ENDGAME
    if int(ply) <= OPENING_MAX_PLY:
        return PHASE_OPENING
    return PHASE_MIDDLEGAME


#: What one ``OrderedDict`` slot costs beyond the key and value objects.
#: MEASURED 2026-08-27, CPython 3.10, ``sys.getsizeof`` on containers built with
#: 100k and 200k entries: 105.40 and 105.40 bytes per entry (a plain ``dict`` is
#: ~52; the linked list ``OrderedDict`` keeps the insertion order in is the
#: rest).  Pinned as a constant because the point of the bound is a memory
#: number a reader can act on, and one that omits the container understates the
#: cache by a fifth.
_ORDERED_DICT_SLOT_BYTES = 106


def entry_bytes(key: str, values: SelectionValues) -> int:
    """The MARGINAL bytes one more cached entry costs.

    ⚑ THE UCI SPELLINGS ARE DELIBERATELY NOT IN THIS NUMBER.
    ``SelectionValues.from_lines`` interns them, so the few thousand distinct
    spellings are ONE set of objects shared by every entry in the process --
    a few hundred KB once, for any cache size.  Charging each entry for 35
    strings it shares with millions of others would treble the reported cost and
    make the flag's memory claim wrong in the direction that leads a reader to
    under-size the cache.  The 8-byte pointers to them ARE counted, inside
    ``getsizeof(values.moves)``.
    """
    return (
        _ORDERED_DICT_SLOT_BYTES
        + sys.getsizeof(key)
        + sys.getsizeof(values)
        + sys.getsizeof(values.moves)
        # numpy's `__sizeof__` includes the data buffer for an array that owns
        # it, which this one does.
        + sys.getsizeof(values.effective_cp)
    )


class DedupCache:
    """One worker's first-seen value cache: COMPACT, and BOUNDED with FIFO.

    ⚑⚑ AN EVICTED POSITION THAT RECURS IS RE-SEARCHED AND RE-BANKED.  That is
    the documented consequence of the bound, not a defect of it, and it is the
    same thing that already happens to a position two workers both reach -- the
    cache has always been per-worker (``merge_dedup``'s ``cache_scope``).  So
    the corpus can hold two rows with one ``dedup_key``, they are two genuine
    independent searches of one position, and ``dedup_cache_evictions`` in the
    summary is how a consumer knows to expect them.

    FIFO rather than LRU on purpose.  LRU would keep the opening tree resident
    forever and evict the endgame positions a game is currently walking through,
    which is the opposite of where the cheap hits are; FIFO is also one
    ``popitem`` with no per-hit bookkeeping, and a cache read on the hot path
    should not write.

    ⚑ THE VALUES AND THE SEEN-INPUT SET CAN DRIFT APART under eviction (a
    label evicted while its tensor is still remembered, or the reverse), and
    that drift can only ever cost a RE-SEARCH or a RE-BANK -- an evicted label
    is searched again, an evicted tensor is banked again -- never a dropped
    row and never a label served for the wrong state, because serving requires
    BOTH keys to hit and banking requires the tensor to be absent (review
    round 2, Fable F3 / Grok D4: traced, no code change).
    """

    def __init__(self, *, max_entries: int) -> None:
        if int(max_entries) <= 0:
            raise ValueError(
                f"--dedup-cache-max must be positive, got {max_entries!r}; a "
                "non-positive bound evicts every entry the instant it is "
                "stored, which is the cache turned off wearing a size flag",
            )
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[str, SelectionValues] = OrderedDict()
        self.evictions = 0
        self._bytes = 0
        #: The ``row_key``s whose tensor has already been BANKED (or searched
        #: for a sub-``MIN_BANKED_PIECES`` position), bounded by the same FIFO
        #: rule as the values so a resume re-warms both from the same shards
        #: in the same order.  A seen input banks no second row.
        self._inputs: OrderedDict[str, None] = OrderedDict()
        self.input_evictions = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> SelectionValues | None:
        """Serve a position, WITHOUT touching the eviction order (FIFO)."""
        return self._entries.get(key)

    def input_seen(self, key: str) -> bool:
        """Whether a row with this input tensor has already been banked."""
        return key in self._inputs

    def remember_input(self, key: str) -> None:
        """Mark an input tensor as banked.  ⚑ AFTER the search succeeds, never
        before: a ply whose search wedged banked nothing, and marking it seen
        would deny the next route to that tensor its row."""
        if key in self._inputs:
            return
        self._inputs[key] = None
        while len(self._inputs) > self.max_entries:
            self._inputs.popitem(last=False)
            self.input_evictions += 1

    def put(self, key: str, values: SelectionValues) -> None:
        if key in self._entries:
            return
        self._entries[key] = values
        self._bytes += entry_bytes(key, values)
        while len(self._entries) > self.max_entries:
            evicted_key, evicted = self._entries.popitem(last=False)
            self._bytes -= entry_bytes(evicted_key, evicted)
            self.evictions += 1

    def summary(self) -> dict[str, Any]:
        """The realized cost of the bound, not the requested one."""
        return {
            "dedup_cache_max_entries": self.max_entries,
            "dedup_cache_entries": len(self._entries),
            "dedup_cache_evictions": self.evictions,
            "dedup_cache_bytes_est": int(self._bytes),
            "dedup_cache_bytes_per_entry_est": (
                self._bytes / len(self._entries) if self._entries else math.nan
            ),
            "dedup_cache_eviction_policy": "fifo",
            # Spelled out in the summary as well as in `--help`, because the
            # summary is what a consumer of the corpus reads.
            "dedup_cache_eviction_semantics": (
                "an evicted position that recurs is re-searched and RE-BANKED; "
                "two rows may share a dedup_key"
            ),
            "dedup_input_set_entries": len(self._inputs),
            "dedup_input_set_evictions": self.input_evictions,
        }


# -- results ------------------------------------------------------------------


def result_from_pov(result_pgn: str | None, *, white_to_move: bool) -> float | None:
    """A PGN result string as seen from THIS ROW's side to move.

    ``+1`` = the row's mover won, ``-1`` = the row's mover lost, ``0`` = draw,
    ``None`` = the game has no result (a ply cap outside tablebase range).

    ⚑ THE SIGN IS THE ROW'S, NOT WHITE'S.  Every row of one game shares a single
    game result and alternating movers, so a corpus that stored the white-POV
    number and let the consumer flip it is one join away from a value target
    that is exactly backwards on half the rows -- a well-formed, plausible,
    silently inverted label.  ``None`` is never replaced by ``0.0``: an
    unfinished game is not a draw.
    """
    if result_pgn is None:
        return None
    if result_pgn == "1/2-1/2":
        return 0.0
    if result_pgn == "1-0":
        return 1.0 if white_to_move else -1.0
    if result_pgn == "0-1":
        return -1.0 if white_to_move else 1.0
    return None


# -- shard writing ------------------------------------------------------------


def zstandard_module() -> Any | None:
    """The optional ``zstandard`` module, or ``None``.

    A function rather than a module-level ``try: import``: the codec choice is a
    behaviour a test must be able to force both ways, and a module-level import
    that already succeeded cannot be un-imported.
    """
    try:
        return importlib.import_module("zstandard")
    except ImportError:  # pragma: no cover - the dev env ships zstandard
        return None


def refuse_populated_dir(out_dir: Path) -> None:
    """A rerun into a populated directory is refused, never merged into.

    The same rule the leaf banks apply with ``open("x")`` and for the same
    reason: two runs' rows in one directory is a corpus whose configuration
    stamp describes only half of it, and nothing downstream can tell which half.

    ⚑ ``--resume`` is the ONE sanctioned way into a populated directory, and it
    buys its way in by proving the configuration is the SAME one (see
    ``refuse_resume_config_drift``) rather than by relaxing this rule.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"--out-dir {out_dir} already holds files; a corpus run refuses to "
            "merge into an existing directory (its rows would carry two "
            "configuration stamps). Choose a new directory, or pass --resume to "
            "continue the run that wrote it.",
        )


def progress_name(worker_id: int) -> str:
    """The per-worker incremental inventory's file name."""
    return f"w{int(worker_id):02d}.progress.jsonl"


def shard_glob(worker_id: int) -> str:
    """Every shard file name one worker can ever write.

    ⚑ The trailing ``-`` is load-bearing: ``w00.progress.jsonl`` must NOT match,
    or the resume sweep would delete the very file it just read.  It also
    matches EVERY codec suffix rather than only the one this session would
    write, so a partial banked as ``.jsonl.zst`` by a session that had
    ``zstandard`` cannot survive a resume that fell back to gzip -- an
    unreferenced truncated shard beside a healthy corpus is a file the next
    consumer's glob reads as data.
    """
    return f"w{int(worker_id):02d}-*"


def shard_index_of(name: str) -> int | None:
    """The rotation index in a shard file name, or ``None`` if it has none."""
    stem = name.split(".")[0]
    _, sep, digits = stem.partition("-")
    if not sep or not digits.isdigit():
        return None
    return int(digits)


class ShardWriter:
    """Per-worker JSONL shards that rotate ON GAME BOUNDARIES, opened ``"x"``.

    zstd when ``zstandard`` imports, gzip otherwise; the codec that was actually
    used is READ BACK off the writer for the summary rather than assumed from
    the import.

    ⚑⚑ ROTATION HAPPENS IN ``end_game``, NEVER IN ``write``.  A shard that
    rotated on a row count would hold a PREFIX of one game, and the resume
    protocol -- which replays every game its progress file does not claim as
    complete -- would then have no honest answer for that game: keeping the
    prefix duplicates its head, dropping the shard loses whole games banked
    beside it.  Whole games per shard is what makes "replay what is not listed"
    exactly right, so the row bound is a FLOOR the writer crosses at the next
    game boundary rather than a hard cap.
    """

    def __init__(
        self, *, out_dir: Path, worker_id: int, shard_rows: int,
        first_index: int = 0, tally_keys: Sequence[str] = (),
    ) -> None:
        if int(shard_rows) <= 0:
            raise ValueError(
                f"--shard-rows must be positive, got {shard_rows!r}; a "
                "non-positive rotation would open one file per row",
            )
        if int(first_index) < 0:
            raise ValueError(
                f"first_index must be >= 0, got {first_index!r}",
            )
        self.out_dir = out_dir
        self.worker_id = int(worker_id)
        self.shard_rows = int(shard_rows)
        #: Where the rotation counter STARTED.  Kept as an attribute so the
        #: realized stamp can read a resume's shard numbering off the writer
        #: that will do the numbering, not off the flag that asked for it.
        self.first_index = int(first_index)
        module = zstandard_module()
        self._zstd = module
        self.codec = "zstd" if module is not None else "gzip"
        self.suffix = ".jsonl.zst" if module is not None else ".jsonl.gz"
        self.shards: list[dict[str, Any]] = []
        #: Shards abandoned UNLISTED because the worker died between banking a
        #: game's rows and ending it.  Named so the slot can report them.
        self.abandoned: list[dict[str, Any]] = []
        self._index = int(first_index)
        self._rows = 0
        self._uncommitted = 0
        self._pending_games: list[int] = []
        self._text: Any = None
        self._raw: Any = None
        self._binary: Any = None
        #: Row fields tallied PER SHARD and committed WITH the shard record
        #: (``tallies``), so a histogram summed over ``shards`` can never count
        #: a row that was abandoned with its file.  ⚑ Read by SUBSCRIPT: a row
        #: missing a tallied key is a writer fault, not a zero.
        self.tally_keys = tuple(str(key) for key in tally_keys)
        self._tallies: dict[str, Counter[str]] = self._fresh_tallies()
        #: Cache-only events of the games pending in the CURRENT record,
        #: committed with it (``cache_events``) and dropped with an abandoned
        #: shard, exactly like the rows -- a resume replays those games.
        self._pending_cache_events: list[dict[str, Any]] = []

    def _fresh_tallies(self) -> dict[str, Counter[str]]:
        return {key: Counter() for key in self.tally_keys}

    def path_for(self, index: int) -> Path:
        return self.out_dir / f"w{self.worker_id:02d}-{index:05d}{self.suffix}"

    def _open(self) -> Any:
        path = self.path_for(self._index)
        if self._zstd is not None:
            # "xb" is what refuses the rerun; the compressor never sees a file
            # it did not create.
            binary = open(path, "xb")  # noqa: SIM115 - closed in close()
            raw = self._zstd.ZstdCompressor().stream_writer(binary)
            self._binary = binary
            self._raw = raw
            self._text = io.TextIOWrapper(raw, encoding="utf-8")
        else:
            self._binary = None
            self._raw = None
            self._text = gzip.open(path, "xt", encoding="utf-8")  # noqa: SIM115
        self._rows = 0
        return self._text

    def write(self, row: dict[str, Any]) -> None:
        """Bank one row.  ⚑ NEVER rotates -- see the class docstring."""
        handle = self._text if self._text is not None else self._open()
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._rows += 1
        self._uncommitted += 1
        for key in self.tally_keys:
            self._tallies[key][str(row[key])] += 1

    def note_cache_events(self, events: Sequence[Mapping[str, Any]]) -> None:
        """Bank a game's cache-only events beside its rows.  ⚑ BEFORE
        ``end_game``, like the rows: they are committed with the record."""
        self._pending_cache_events.extend(dict(e) for e in events)

    def end_game(self, game_id: int) -> None:
        """Record ``game_id`` as banked in full, and rotate if the shard is due.

        ⚑ CALLED FOR EVERY COMPLETED GAME, INCLUDING ONE THAT BANKED NO ROWS.
        A game whose every position was dedup-served (or that was adjudicated
        before it could bank anything) produced no bytes, and a resume that
        inferred completion from banked rows would replay it forever.  Its id
        joins the CURRENT shard's pending list, and ``close`` flushes a
        path-less completion record if the worker ends on a run of them.
        """
        self._pending_games.append(int(game_id))
        self._uncommitted = 0
        if self._rows >= self.shard_rows:
            self.close()

    def _append_progress(self, record: dict[str, Any]) -> None:
        # One line per CLOSED shard, appended as the shard closes, so a run
        # that dies on day 13 still names every complete shard it wrote --
        # summary.json is written once at run END and a crash takes it with
        # it. Each worker owns its own progress file, so "a" cannot interleave
        # across processes. Deliberately NOT suppressed: a metadata write that
        # fails (disk full) should kill the worker loudly, not leave a corpus
        # whose inventory quietly stopped growing.
        #
        # ⚑ ONE line, ONE `write`, append mode: that is the whole crash-safety
        # story, and it is why `kill -9` can only ever tear the LAST line of
        # this file (which `read_worker_progress` drops).
        with open(self.out_dir / progress_name(self.worker_id), "a",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def close(self) -> None:
        games = sorted(self._pending_games)
        self._pending_games = []
        cache_events = self._pending_cache_events
        self._pending_cache_events = []
        if self._text is None:
            if games:
                # ⚑ A COMPLETION RECORD, NOT A SHARD: `path` is null because
                # these games banked no rows, so there is no file to index.
                # It never enters `self.shards` -- the summary's inventory
                # must name files that exist.
                self._append_progress({
                    "path": None, "rows": 0, "codec": self.codec, "games": games,
                    "cache_events": cache_events,
                })
            return
        self._text.close()
        if self._raw is not None:
            self._raw.close()
        if self._binary is not None:
            self._binary.close()
        record = {
            "path": str(self.path_for(self._index)),
            "rows": self._rows,
            "codec": self.codec,
            "games": games,
            "tallies": {
                key: dict(counter) for key, counter in self._tallies.items()
            },
            "cache_events": cache_events,
        }
        self._tallies = self._fresh_tallies()
        if self._uncommitted:
            # ⚑⚑ A GAME'S ROWS ARE IN THIS FILE AND THE GAME NEVER ENDED --
            # the worker died between `write` and `end_game` (a disk error, an
            # OOM in the row loop). Listing it would put a HALF game in the
            # inventory under a `games` list that does not name it, and the
            # next resume would replay that game and bank its head TWICE. So
            # the file is left UNLISTED: a resume deletes it and replays every
            # game it held, which costs work and cannot duplicate a row. This
            # is what makes "a listed shard holds only whole games" structural
            # rather than a property of the happy path.
            _LOG.error(
                "worker %d abandoning %s unlisted: %d row(s) of a game that "
                "never ended; a resume will delete it and replay its games",
                self.worker_id, record["path"], self._uncommitted,
            )
            # ⚑ `games` is DROPPED with the file rather than carried into the
            # next record: their rows are in the file a resume is about to
            # delete, so recording them complete would lose them silently --
            # the one failure worse than replaying them.
            self.abandoned.append({**record, "uncommitted_rows": self._uncommitted})
        else:
            self.shards.append(record)
            self._append_progress(record)
        self._text = None
        self._raw = None
        self._binary = None
        self._rows = 0
        self._uncommitted = 0
        self._index += 1


# -- resume -------------------------------------------------------------------


def _decode_shard_line(path: Path, number: int, line: str) -> dict[str, Any]:
    """One shard line, or a refusal that NAMES THE FILE.

    ⚑ MEASURED on a copy of a live production shard: a shard truncated
    mid-write decompresses cleanly right up to a partial final line, and the
    bare ``json.JSONDecodeError`` that follows says
    ``Expecting property name ... char 4347`` and nothing else -- no path, no
    hint that the corpus is what is damaged.  A resume is read on day 13 by
    someone who did not write this file.
    """
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} line {number} is not a complete JSON row ({exc}). A shard "
            "a progress line LISTS is claimed complete, so a truncated one "
            "means the corpus is damaged: its games would be marked done "
            "against rows that are only half there.",
        ) from exc


def iter_shard_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Every banked row of one shard, in the order it was written.

    The reader half of ``ShardWriter``, and the only one -- a second decoder
    somewhere else is how a codec choice comes to disagree with itself.
    """
    if path.suffix == ".zst":
        module = zstandard_module()
        if module is None:
            raise RuntimeError(
                f"{path} is a zstd shard and this process cannot import "
                "zstandard; a resume that skipped it would replay games the "
                "corpus already holds",
            )
        with open(path, "rb") as raw, module.ZstdDecompressor().stream_reader(
            raw,
        ) as stream:
            for number, line in enumerate(
                io.TextIOWrapper(stream, encoding="utf-8"), start=1,
            ):
                if line.strip():
                    yield _decode_shard_line(path, number, line)
        return
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if line.strip():
                yield _decode_shard_line(path, number, line)


def require_current_row_schema(row: Mapping[str, Any]) -> None:
    """A resume re-warms ONLY from rows this build wrote the shape of.

    Checked before any key of the row is read: a schema-1 row has no
    ``search_key``/``input_key``, and a ``KeyError`` on one of those would name
    the symptom rather than the cause.
    """
    if int(row["schema"]) != ROW_SCHEMA:
        raise ValueError(
            f"row schema {row['schema']!r} is not {ROW_SCHEMA}; a resume that "
            "re-warmed its dedup cache from a foreign row shape would serve "
            "values it cannot read",
        )


def selection_values_from_row(row: Mapping[str, Any]) -> SelectionValues:
    """Rebuild the CACHED value vector from the row that position banked.

    ⚑ It has to be the same object the live visit cached, not merely a similar
    one: ``q_of`` runs on ``effective_cp`` and the Gumbel draw is seeded from
    ``(seed, worker, game, ply)``, so a value vector that differs in one float
    moves the played move and the resumed corpus stops being the corpus an
    uninterrupted run would have written.  The row banks exactly the block
    selection read -- phase 0, at ``selection.value_depth`` -- and the floats
    round-trip through JSON exactly, so this is a reconstruction rather than an
    approximation.  ``value_width`` is checked rather than trusted: a row shape
    that drifted must fail here, not serve a quietly wrong value vector.
    """
    require_current_row_schema(row)
    selection = row["selection"]
    depth = int(selection["value_depth"])
    blocks = [
        block for block in row["phases"][0]["per_depth"]
        if int(block["depth"]) == depth
    ]
    if len(blocks) != 1:
        raise ValueError(
            f"row for game {row.get('game_id')!r} ply {row.get('ply')!r} names "
            f"selection depth {depth} and its phase-0 stream holds "
            f"{len(blocks)} block(s) at that depth; exactly one is required to "
            "rebuild the cached value vector",
        )
    lines = blocks[0]["lines"]
    if len(lines) != int(selection["value_width"]):
        raise ValueError(
            f"row for game {row.get('game_id')!r} ply {row.get('ply')!r} banks "
            f"{len(lines)} lines at depth {depth} but claims value_width "
            f"{selection['value_width']!r}",
        )
    return SelectionValues(
        moves=tuple(sys.intern(str(line[1])) for line in lines),
        effective_cp=np.asarray(
            [float(line[2]) for line in lines], dtype=np.float32,
        ),
    )


#: Keys every progress line must carry.  ``games`` is deliberately NOT here:
#: lines written before game-boundary rotation existed do not have it, and the
#: whole point of the legacy path is that a run already burning does not have to
#: be thrown away to gain a resume.
_PROGRESS_KEYS = ("path", "rows", "codec")

#: What ``repair_worker_progress`` found, and did about it.
PROGRESS_ABSENT = "absent"
PROGRESS_INTACT = "intact"
PROGRESS_NEWLINE_RESTORED = "newline_restored"
PROGRESS_TRUNCATED = "truncated"


def _is_progress_record(line: str) -> bool:
    """Whether ``line`` is a whole progress record, not a prefix of one."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return False
    return isinstance(record, dict) and all(k in record for k in _PROGRESS_KEYS)


def repair_worker_progress(path: Path) -> str:
    """Heal what a kill left, BEFORE anything appends to this file again.

    ⚑⚑ TOLERATING A TORN TAIL IS NOT ENOUGH -- THE BYTES HAVE TO GO.
    ``_append_progress`` opens ``"a"``, so a resumed session's first record
    lands ON THE END of whatever fragment the kill left, producing a line that
    is neither the fragment nor the record.  Two ways that ends badly, and
    repeated ``kill -9`` is this feature's entire contract:

    * A PARTIAL final line.  The reader drops it, the resume proceeds, and the
      next record is glued to it.  That glued line is now mid-file, so the
      SECOND resume hits the "not the torn tail" refusal and the worker is
      unresumable without hand-editing.  Worse than the refusal: the record
      swallowed inside it is a closed shard whose games are then unknown.
    * A COMPLETE final line whose NEWLINE alone was lost.  The reader accepts
      it -- correctly, it is a whole record -- and the next append destroys it
      by concatenation.  A record that was accepted is then gone.

    So the tail is repaired in place, and both repairs are single operations
    that are safe to be killed in the middle of, because a kill leaves either
    the old state (repaired on the next resume) or the new one:

    * the final line is a whole record  -> append the ONE byte the kill stole.
    * anything else                     -> truncate to just past the last
      newline (to zero if there is none), dropping only the fragment.

    Bytes, not text: a torn write can in principle cut a multi-byte character
    in half, and ``read_text`` would then raise for the WHOLE file instead of
    for the fragment.
    """
    if not path.exists():
        return PROGRESS_ABSENT
    raw = path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return PROGRESS_INTACT
    cut = raw.rfind(b"\n") + 1  # 0 when the file holds no newline at all
    try:
        tail = raw[cut:].decode("utf-8")
    except UnicodeDecodeError:
        tail = ""
    if _is_progress_record(tail):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n")
        return PROGRESS_NEWLINE_RESTORED
    os.truncate(path, cut)
    return PROGRESS_TRUNCATED


def read_worker_progress(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """One worker's progress lines, plus whether a TORN TAIL was dropped.

    ⚑ THE TORN TAIL IS THE ONLY TOLERATED DAMAGE, and the tolerance is as
    narrow as the failure it models.  ``kill -9`` during the single append that
    writes a line can leave a final line cut short OF ITS NEWLINE; it cannot
    leave a partial line with intact lines after it, and it cannot leave a
    partial line that ends in a newline.  So exactly that signature is dropped
    and reported, and anything else is refused -- a reader that shrugged at
    damage in the middle would silently forget every shard listed below it and
    replay games the corpus already holds.
    """
    if not path.exists():
        return [], False
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    # A file that ends in a newline has no line in flight: every line it holds
    # was written whole.
    all_terminated = text.endswith("\n") or not text
    records: list[dict[str, Any]] = []
    torn = False
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if number != len(lines) or all_terminated:
                raise ValueError(
                    f"{path} line {number} of {len(lines)} is not JSON, and it "
                    "is not the torn tail a kill leaves (that is the LAST line "
                    "and it is cut short of its newline"
                    f"{'; this file ends in one' if all_terminated else ''}). "
                    "Resuming from a file damaged some other way would lose "
                    "the shards it lists.",
                ) from exc
            torn = True
            continue
        if not isinstance(record, dict):
            raise ValueError(f"{path} line {number} is not a JSON object")
        missing = [key for key in _PROGRESS_KEYS if key not in record]
        if missing:
            raise ValueError(
                f"{path} line {number} is missing {missing}; it is not a "
                "progress record this generator wrote",
            )
        records.append(record)
    return records, torn


@dataclass(frozen=True)
class ResumeState:
    """What one worker's own bookkeeping says it has already done.

    Everything here is READ OFF THE DISK the killed session left behind -- the
    progress file it appended to and the shards it closed -- never off the
    flags of the session doing the resuming.
    """

    #: Game ids no later session may replay.
    completed_games: frozenset[int]
    #: The listed shards, path-normalised into THIS invocation's out-dir.
    shards: tuple[dict[str, Any], ...]
    #: Where the shard counter continues.  ``highest listed + 1``, never a
    #: renumbering: a resume must not rewrite a byte of what is already banked.
    next_shard_index: int
    #: Killed-mid-write shard files, deleted; their games are replayed.
    deleted_partials: tuple[str, ...]
    #: A partial final progress line was dropped.  ⚑ Sourced from the REPAIR,
    #: which is the thing that observed the damage -- by the time the reader
    #: runs, the file is terminated and its own tolerance cannot fire.
    torn_tail: bool
    #: What the tail repair found: absent / intact / newline_restored /
    #: truncated.  ``newline_restored`` is the case that USED to be invisible
    #: and destructive -- an accepted record with its newline stolen.
    progress_repair: str
    #: Positions put back into the dedup cache from the listed shards.
    dedup_rewarmed: int
    #: Progress lines that carried no ``games`` list and whose game ids were
    #: therefore DERIVED by reading the shard (the pre-``end_game`` format).
    legacy_lines: int
    #: Cache-only events (a search that banked NO row: a label-only miss on a
    #: seen tensor, or a sub-``MIN_BANKED_PIECES`` position) replayed from the
    #: progress records, so the re-warmed cache holds every entry the killed
    #: session's cache held, in the same FIFO order.
    cache_events_rewarmed: int = 0
    #: Where this worker's ``seq`` counter continues: one above the highest
    #: ``seq`` re-warmed, so a resumed session's rows and events sort after
    #: everything the killed session committed.
    next_seq: int = 0

    @classmethod
    def fresh(cls) -> ResumeState:
        """The state of a worker with nothing to resume."""
        return cls(
            completed_games=frozenset(),
            shards=(),
            next_shard_index=0,
            deleted_partials=(),
            torn_tail=False,
            progress_repair=PROGRESS_ABSENT,
            dedup_rewarmed=0,
            legacy_lines=0,
        )


class WorkerSeq:
    """⚑ ONE MONOTONIC COUNTER PER WORKER, stamped on every row and every cache
    event at the moment the cache changes.  The resume replays rows and events
    strictly by it.  ``(game_id, ply)`` is NOT the live order: a worker plays
    its game ids in the order it was dealt them, and under a resume that
    order is the spec's, not ascending -- so a cache bounded by FIFO would
    come back with a different resident set (measured: ``game_ids=(2, 0)``
    with ``dedup_cache_max=1``)."""

    def __init__(self, start: int = 0) -> None:
        self.value = int(start)

    def next(self) -> int:
        current = self.value
        self.value += 1
        return current


def cache_event(
    *, seq: int, game_id: int, ply: int, search_key: str, input_key: str,
    remember_input: bool, values: SelectionValues,
) -> dict[str, Any]:
    """⚑⚑ A SEARCH THAT BANKED NO ROW, recorded so a resume can rebuild it.

    Two plies search and then bank nothing: a seen tensor whose label was not
    cached (the row is suppressed, the values are cached under the new
    ``search_key``), and a sub-``MIN_BANKED_PIECES`` position (values cached,
    tensor remembered, no row).  Both leave state in the worker's cache that
    ``resume_worker_state`` could not rebuild from banked rows -- so a killed
    and resumed worker searched where the uninterrupted one served, and under
    the carried TT that moves finite-depth values, the seeded move and every
    row after it (Codex P2, operator ruling #497 round 3).  The event carries
    exactly what the live visit put in the cache, and the record it is
    committed with (``ShardWriter.close``) is the shard's own, so the FIFO
    order is reproduced by merging rows and events on ``(game_id, ply)``.
    """
    return {
        "seq": int(seq),
        "game_id": int(game_id),
        "ply": int(ply),
        "search_key": str(search_key),
        "input_key": str(input_key),
        "remember_input": bool(remember_input),
        "values": {
            "moves": list(values.moves),
            # float32 -> python float -> JSON -> float32 is exact.
            "effective_cp": [float(cp) for cp in values.effective_cp],
        },
    }


def selection_values_from_event(event: Mapping[str, Any]) -> SelectionValues:
    """The cached object a cache event recorded, rebuilt exactly."""
    values = event["values"]
    return SelectionValues(
        moves=tuple(sys.intern(str(m)) for m in values["moves"]),
        effective_cp=np.asarray(values["effective_cp"], dtype=np.float32),
    )


def _apply_cache_event(cache: DedupCache, event: Mapping[str, Any]) -> None:
    """Replay one event in the order the live session performed it."""
    key = str(event["search_key"])
    if cache.get(key) is None:
        cache.put(key, selection_values_from_event(event))
    if bool(event.get("remember_input")):
        cache.remember_input(str(event["input_key"]))


def _seq_of(entry: Mapping[str, Any], *, what: str, worker_id: int) -> int:
    """The ``seq`` a row or event was committed under -- REQUIRED, by name."""
    if "seq" not in entry:
        raise ValueError(
            f"{progress_name(worker_id)}: a {what} carries no seq (game "
            f"{entry.get('game_id')!r} ply {entry.get('ply')!r}); the resume "
            "replays the cache strictly in seq order and cannot place it",
        )
    return int(entry["seq"])


def resume_worker_state(
    *, out_dir: Path, worker_id: int, cache: DedupCache,
) -> ResumeState:
    """Adopt one worker's killed session: its games, its shards, its cache.

    Three things happen here and they are one operation because they read the
    same bytes:

    1. THE PROGRESS FILE decides what is complete.  A line with a ``games``
       list says so outright.  A line without one predates game-boundary
       rotation, so its games are DERIVED by reading the shard: every
       ``game_id`` in it counts as complete, **including the last game, which
       that format may have cut in half at the row bound**.  Accepting the
       truncation is the deliberate call -- a shard is immutable, rewriting one
       to heal a game is a worse trade than losing the tail of at most one game
       per worker, and REPLAYING it would duplicate every row already banked.
    2. EVERY UNLISTED ``w<id>-*`` FILE IS DELETED.  That is the shard the kill
       caught mid-write: no progress line names it, nothing downstream can tell
       how far it got, and its games are simply replayed.  ⚑ Deleting is what
       makes ``open("x")`` still meaningful on the resumed run -- a leftover
       partial at the next index would otherwise refuse the first shard the
       resumed worker tries to write.
    3. THE DEDUP CACHE IS RE-WARMED from the listed shards, in the order they
       were banked, so FIFO eviction sees the same order the killed session
       gave it.

    A listed shard that is missing from disk is REFUSED rather than skipped: it
    means rows the progress file claims are gone, and continuing would mark
    their games complete while the corpus no longer holds them.

    ⚑ THE TAIL IS REPAIRED FIRST, before this run can append a byte to that
    file -- see ``repair_worker_progress``.  Dropping a torn tail on READ and
    leaving it on DISK is what turns a second kill into a permanently
    unresumable worker.
    """
    progress_path = out_dir / progress_name(worker_id)
    repair = repair_worker_progress(progress_path)
    # The repair guarantees a newline-terminated (or empty) file, so the
    # reader's own torn-tail tolerance cannot fire here; it stays live for a
    # caller that reads the file without owning it. `torn_tail` below comes
    # from the repair, which is the half that actually saw the damage.
    records, _ = read_worker_progress(progress_path)
    completed: set[int] = set()
    shards: list[dict[str, Any]] = []
    listed_names: set[str] = set()
    highest = -1
    legacy = 0
    rewarmed = 0
    events_rewarmed = 0
    next_seq = 0
    for record in records:
        raw_path = record["path"]
        # ⚑ FAIL CLOSED (Grok/Fable round 5): a games-bearing record with no
        # `cache_events` key predates the contract, and adopting it as "zero
        # events" is the accepted-then-ignored shape -- its label-only entries
        # would silently be gone. Schema 3 never produced an accepted corpus,
        # so there is nothing to carry: refused by name.
        if "games" in record and "cache_events" not in record:
            raise ValueError(
                f"{progress_name(worker_id)} holds a record for games "
                f"{record['games']} with no cache_events; it was written before "
                "cache-only searches were committed with the record, and a "
                "resume cannot rebuild the cache it left. Use a new --out-dir.",
            )
        events = sorted(
            (dict(e) for e in record.get("cache_events", [])),
            key=lambda e: _seq_of(e, what="cache event", worker_id=worker_id),
        )
        for event in events:
            next_seq = max(next_seq, _seq_of(event, what="cache event", worker_id=worker_id) + 1)
        if raw_path is None:
            # A completion record: games, no file.  It must carry them -- a
            # null path with no games names nothing at all.
            if "games" not in record:
                raise ValueError(
                    f"{progress_name(worker_id)} holds a null-path record with "
                    "no games list; it indexes neither a shard nor a game",
                )
            completed.update(int(game) for game in record["games"])
            for event in events:
                _apply_cache_event(cache, event)
                events_rewarmed += 1
            continue
        # By NAME, not by the stored string: the corpus directory may have been
        # moved or spelled differently between sessions, and the file the
        # progress line means is the one beside the progress line.
        name = Path(str(raw_path)).name
        path = out_dir / name
        if not path.exists():
            raise ValueError(
                f"{progress_name(worker_id)} lists {name} and it is not in "
                f"{out_dir}; a resume cannot mark its games complete against "
                "rows that are gone",
            )
        listed_names.add(name)
        index = shard_index_of(name)
        if index is not None:
            highest = max(highest, index)
        adopted: dict[str, Any] = {
            "path": str(path),
            "rows": int(record["rows"]),
            "codec": str(record["codec"]),
        }
        # ⚑ THE PRIOR SHARD'S TALLIES TRAVEL WITH IT, so the corpus-level
        # window histograms can be summed over prior + this session and still
        # equal the corpus-level `rows` (Codex P2, review round 2). A record
        # without them (a pre-tallies progress line) is counted as UNKNOWN
        # rows rather than as zero windows.
        if "tallies" in record:
            adopted["tallies"] = dict(record["tallies"])
        derived: set[int] = set()
        pending_events = list(events)
        # ONE pass over the shard: it re-warms the cache, and it is also where
        # a legacy line's game ids come from.
        for row in iter_shard_rows(path):
            derived.add(int(row["game_id"]))
            require_current_row_schema(row)
            # ⚑ STRICTLY BY seq: every event committed before this row is
            # replayed first, so the FIFO sequence is the one the killed
            # session's cache saw whatever order the games were dealt in.
            row_seq = _seq_of(row, what="row", worker_id=worker_id)
            next_seq = max(next_seq, row_seq + 1)
            while pending_events and _seq_of(
                pending_events[0], what="cache event", worker_id=worker_id,
            ) < row_seq:
                _apply_cache_event(cache, pending_events.pop(0))
                events_rewarmed += 1
            # ⚑ BOTH KEYS, in the order the live session filled them: the
            # values under the row's `search_key`, and the row's `input_key`
            # into the seen-input set. A re-warm that restored only the values
            # would have the resumed session search AND RE-BANK every position
            # the killed one already banked -- a duplicate row per replayed
            # transposition, and `test_a_resumed_worker_re_warms_...` reads
            # `rows == 0` off exactly this.
            key = str(row["search_key"])
            if cache.get(key) is None:
                cache.put(key, selection_values_from_row(row))
                rewarmed += 1
            cache.remember_input(str(row["input_key"]))
        for event in pending_events:
            _apply_cache_event(cache, event)
            events_rewarmed += 1
        if "games" in record:
            adopted["games"] = [int(game) for game in record["games"]]
        else:
            legacy += 1
            # ⚑ Flagged, because a derived list is a weaker claim than a
            # recorded one: the last game in a mid-game-rotated shard may hold
            # only part of its plies.
            adopted["games"] = sorted(derived)
            adopted["games_derived"] = True
        completed.update(int(game) for game in adopted["games"])
        shards.append(adopted)
    deleted: list[str] = []
    for stale in sorted(out_dir.glob(shard_glob(worker_id))):
        # Only files this writer could have written: a directory that happens
        # to match the glob is not a shard, and deleting it is not this
        # function's business.
        if stale.name in listed_names or not stale.is_file():
            continue
        stale.unlink()
        deleted.append(stale.name)
    return ResumeState(
        completed_games=frozenset(completed),
        shards=tuple(shards),
        next_shard_index=highest + 1,
        deleted_partials=tuple(deleted),
        torn_tail=repair == PROGRESS_TRUNCATED,
        progress_repair=repair,
        dedup_rewarmed=rewarmed,
        legacy_lines=legacy,
        cache_events_rewarmed=events_rewarmed,
        next_seq=next_seq,
    )


# -- the game loop ------------------------------------------------------------


@dataclass(frozen=True)
class WorkerSpec:
    """What one worker process needs.  Frozen and picklable across a spawn."""

    worker_id: int
    game_ids: tuple[int, ...]
    out_dir: Path
    sf_binary: str
    sf_hash_mb: int
    sf_read_timeout_s: float
    sf_search_timeout_s: float
    syzygy_path: str
    staircase: str
    staircase_policy: str
    seed: int
    dedup_cache_max: int
    temp_plies: int
    temp_high: float
    temp_low: float
    max_plies: int
    shard_rows: int
    nice: int
    cp_slope: float
    cp_draw_width: float
    book: str | None
    book_plies: int
    book_max_games: int
    run_id: str
    config_sha256: str
    #: Continue the killed session already banked in ``out_dir`` instead of
    #: starting one.  A plain ``bool`` so the spec stays picklable across a
    #: spawn; every fact the resume needs is on the DISK the worker owns
    #: (``w<id>.progress.jsonl`` and the shards it lists), not in this object.
    resume: bool = False


@dataclass
class DedupStats:
    """Searched and cache-served counts by phase, plus the two-key breakdown.

    Every visited ply is exactly one of ``first_seen`` (it ran a search) or
    ``hits`` (it was served from the cache with no search).  The scalar
    counters below say WHY, in terms of the two keys ``play_game`` reads:

    * ``row_key_hits`` -- the input tensor had been banked before; no row.
    * ``search_key_hits`` -- the ``search_key`` was in the cache, whether or
      not the ply was served from it.
    * ``search_key_hit_on_new_input`` -- the label was cached but the tensor
      was new: searched and banked anyway, because the compact cache cannot
      supply a row's ``phases``.  The count of rows the old position-keyed
      dedup would have DROPPED.
    * ``search_key_miss_on_seen_input`` -- the tensor was banked before but an
      older repeat outside the frames changed the engine-relevant state:
      searched, not banked.  The count of labels the old dedup would have
      SERVED WRONG.
    * ``searches`` -- every ply that ran a search (``== positions_first_seen``,
      spelled under the name the ledger uses).
    * ``rows_banked`` -- rows handed to the game's row list.  ⚑ NOT the
      committed row count: a row can still be abandoned with its shard, which
      is why the worker's ``rows`` is read off the inventory instead.
    """

    first_seen: Counter[str] = field(default_factory=Counter)
    hits: Counter[str] = field(default_factory=Counter)
    row_key_hits: int = 0
    search_key_hits: int = 0
    search_key_hit_on_new_input: int = 0
    search_key_miss_on_seen_input: int = 0
    searches: int = 0
    rows_banked: int = 0

    def summary(self) -> dict[str, Any]:
        seen = int(sum(self.first_seen.values()))
        hits = int(sum(self.hits.values()))
        total = seen + hits
        return {
            "positions_visited": total,
            "positions_first_seen": seen,
            "dup_hits": hits,
            "dup_rate": (hits / total) if total else math.nan,
            "first_seen_by_phase": {p: int(self.first_seen[p]) for p in GAME_PHASES},
            "dup_hits_by_phase": {p: int(self.hits[p]) for p in GAME_PHASES},
            "row_key_hits": int(self.row_key_hits),
            "search_key_hits": int(self.search_key_hits),
            "search_key_hit_on_new_input": int(self.search_key_hit_on_new_input),
            "search_key_miss_on_seen_input": int(self.search_key_miss_on_seen_input),
            "searches": int(self.searches),
            "rows_banked": int(self.rows_banked),
        }


@dataclass
class WorkerProgress:
    """Where a worker currently IS.  Read only when it dies.

    A crashed worker's slot has to say more than "it crashed": the game and ply
    are what makes the failure reproducible (the selection stream is seeded from
    exactly ``(seed, worker, game, tag, ply)``), and they are the only thing
    that distinguishes a bad position from a bad engine.  ⚑ Required rather than
    optional on ``play_game``: a progress recorder a caller can forget to pass
    is a value accepted and then silently ignored, which is the defect class
    this repo keeps re-finding.
    """

    game_id: int | None = None
    ply: int | None = None


@dataclass
class GameOutcome:
    rows: list[dict[str, Any]]
    plies: int
    termination: str
    result_pgn: str | None
    adjudication: dict[str, Any] | None
    opening_source: str
    #: Plies at <= 6 pieces where the tablebase could NOT answer (a missing
    #: table, or castling rights, which Syzygy's index does not encode). The
    #: game keeps playing; the count is disclosed rather than absorbed, because
    #: it is the one path on which a small-material game reaches the ply cap.
    adjudication_unavailable: int
    #: Searches that banked no row but left cache state (``cache_event``).
    cache_events: list[dict[str, Any]] = field(default_factory=list)


def _adjudicate(board: chess.Board, syzygy_path: str) -> dict[str, Any] | None:
    """Syzygy verdict for ``board``, or ``None`` when it is not probable.

    Both halves come from ``chess_anti_engine.tablebase`` rather than from a
    local mapping: ``tb_adjudicate_result`` owns the cursed-win convention and
    ``probe_wdl`` owns the 0/1/2 label, and restating either here is how the two
    conventions come to disagree.
    """
    result = tb_adjudicate_result(board, syzygy_path)
    if result is None:
        return None
    wdl = probe_wdl(board, syzygy_path)
    return {
        "kind": "syzygy",
        "result": result,
        # ⚑ From the ADJUDICATED position's own side to move, which is NOT the
        # banked row's mover.  A row's own POV number is its `result` field.
        "wdl": None if wdl is None else int(wdl),
        "pov": "terminal_position_side_to_move",
        "fen": board.fen(),
        "piece_count": int(chess.popcount(board.occupied)),
    }


def play_game(
    *,
    spec: WorkerSpec,
    searcher: StaircaseSearcher | EngineLease,
    opening_cfg: OpeningConfig,
    game_id: int,
    cache: DedupCache,
    dedup: DedupStats,
    progress: WorkerProgress,
    seq: WorkerSeq,
) -> GameOutcome:
    """One game: sample an opening, then search-select-push until it ends."""
    searcher.new_game()
    progress.game_id = int(game_id)
    progress.ply = None
    start = sample_starting_board(
        rng=book_rng(seed=spec.seed, worker_id=spec.worker_id, game_id=game_id),
        cfg=opening_cfg,
    )
    board = start.board
    rows: list[dict[str, Any]] = []
    cache_events: list[dict[str, Any]] = []
    adjudication: dict[str, Any] | None = None
    result_pgn: str | None = None
    termination = "unfinished"
    unavailable = 0
    ply = 0
    while True:
        progress.ply = ply
        if board.is_game_over(claim_draw=True):
            termination = "natural"
            result_pgn = board.result(claim_draw=True)
            break
        piece_count = int(chess.popcount(board.occupied))
        if piece_count <= ADJUDICATION_MAX_PIECES:
            adjudication = _adjudicate(board, spec.syzygy_path)
            if adjudication is not None:
                termination = "syzygy"
                result_pgn = str(adjudication["result"])
                break
            unavailable += 1
        if ply >= int(spec.max_plies):
            termination = "max_plies"
            # ⚑ NEVER a fabricated result.  A capped game outside tablebase
            # range has no outcome, and `None` is what every row of it carries.
            adjudication = (
                _adjudicate(board, spec.syzygy_path)
                if piece_count <= ADJUDICATION_MAX_PIECES else None
            )
            result_pgn = None if adjudication is None else str(adjudication["result"])
            break

        key = dedup_key(board)
        label_key = search_key(board)
        input_key = row_key(board)
        phase = game_phase(ply=ply, piece_count=piece_count)
        # ⚑⚑ TWO KEYS, TWO QUESTIONS (module docstring).  `cached` answers "is
        # this LABEL known"; `new_input` answers "has this TENSOR been banked".
        # A new tensor is searched and banked whatever the cache says, because
        # the compact cache holds two arrays and a row needs the phases; a
        # seen tensor banks nothing, and is served only when the label matches
        # too -- an older repeat outside the frames changes what the engine
        # sees without changing a single plane.
        cached = cache.get(label_key)
        new_input = not cache.input_seen(input_key)
        if cached is not None:
            dedup.search_key_hits += 1
        if not new_input:
            dedup.row_key_hits += 1
        searched = False
        # Assigned at the search; a served ply never reads it (rows and
        # events are only built on the searched branch).
        seq_no = -1
        if cached is not None and not new_input:
            dedup.hits[phase] += 1
            values = cached
            search: PositionSearch | None = None
        else:
            try:
                search = searcher.search_position(board)
            except StockfishTimeoutError:
                # An EngineLease has already spent its one fresh-engine retry
                # by the time this raises (a bare searcher has none): the ply
                # is unlabelable, so the GAME ends here -- banked rows keep
                # their labels, the game gets no result (same shape as a ply
                # cap), and the WORKER plays on instead of dying with it.
                termination = "engine_wedge"
                break
            searched = True
            # The cache changes at THIS search: rows and events both carry
            # the number, and the resume replays them by it.
            seq_no = seq.next()
            dedup.first_seen[phase] += 1
            dedup.searches += 1
            if cached is not None:
                dedup.search_key_hit_on_new_input += 1
            elif not new_input:
                dedup.search_key_miss_on_seen_input += 1
            # ⚑ Selection reads the COMPACT object on the first visit too, so a
            # cache-served ply and a first-seen ply cannot disagree about q.
            values = SelectionValues.from_lines(search.values)
            cache.put(label_key, values)
            if new_input:
                # ⚑ After the search, so a wedged ply never marks its tensor
                # banked.  Remembered whether or not a row follows: a sub-
                # `MIN_BANKED_PIECES` position can never bank one, and the
                # next route to it should be served, not re-searched.
                cache.remember_input(input_key)
            else:
                # A seen tensor whose label was NOT cached: the search was
                # needed for the move, the row is not (its tensor is already
                # in the corpus under another label).
                search = None

        temp, temp_phase = temperature_for(
            ply,
            temp_plies=spec.temp_plies,
            temp_high=spec.temp_high,
            temp_low=spec.temp_low,
        )
        chosen = gumbel_choice(
            searcher.q_of(values),
            temp=temp,
            rng=selection_rng(
                seed=spec.seed, worker_id=spec.worker_id, game_id=game_id, ply=ply,
            ),
        )
        played = values.moves[chosen]

        if search is not None and piece_count >= MIN_BANKED_PIECES:
            # ⚑ THE ROW'S WINDOW IS THE SEARCH'S WINDOW, read off the object the
            # engine was set up from rather than rebuilt here.  Two builders
            # would be two chances for the banked input and the banked label to
            # describe different histories, which is the failure this whole
            # schema exists to remove.
            history = search.history
            if history.fen != board.fen():
                raise RuntimeError(
                    "the search's window is for a different position than the "
                    f"one being banked ({history.fen!r} against "
                    f"{board.fen()!r})",
                )
            rows.append({
                "schema": ROW_SCHEMA,
                # Present ONLY on a row whose search re-ran on a fresh engine
                # after a wedge: that search saw a COLD table, not the game's
                # carried one, and the row has to say so.
                **({"cold_tt_retry": True} if searcher.cold_tt_retry_last else {}),
                "run": {
                    "run_id": spec.run_id,
                    "config_sha256": spec.config_sha256,
                    # Observed at write time, same counter as the worker stamp.
                    KEY_TT_CARRIED: searcher.tt_cleared_mid_position == 0,
                    # The REALIZED regime, read off the flag `row_key` just
                    # required, not off the constant.
                    KEY_HISTORY_REP_FIX: bool(rep_fix.current()),
                },
                "fen": board.fen(),
                # `dedup_key` is kept for consumers that join on it; the two
                # keys the generator actually decides by sit beside it.
                "dedup_key": key,
                "search_key": label_key,
                "input_key": input_key,
                "worker_id": spec.worker_id,
                "game_id": game_id,
                "ply": ply,
                "seq": seq_no,
                "stm": "w" if board.turn == chess.WHITE else "b",
                **history.as_row_fields(),
                "piece_count": piece_count,
                "game_phase": phase,
                "played_move": played,
                "selection": {
                    "temp": temp,
                    "schedule_phase": temp_phase,
                    "temp_plies": int(spec.temp_plies),
                    "value_depth": search.value_depth,
                    "value_width": len(values.moves),
                    "value_full_width": search.value_full_width,
                    "legal_moves": board.legal_moves.count(),
                    "seed_material": [
                        int(spec.seed), int(spec.worker_id), int(game_id),
                        _STREAM_SELECT, int(ply),
                    ],
                },
                **(
                    {"staircase_gate": search.staircase_gate.as_row()}
                    if search.staircase_gate is not None else {}
                ),
                "phases": [p.as_row() for p in search.phases],
                # Backfilled below, at game end.
                "result": None,
                "result_pgn": None,
                "adjudication": None,
            })
            dedup.rows_banked += 1
        elif searched:
            # ⚑ A SEARCH WITH NO ROW: the cache changed and nothing banked
            # says so. Recorded so a resume can rebuild the entry -- see
            # `cache_event`.
            cache_events.append(cache_event(
                seq=seq_no, game_id=game_id, ply=ply, search_key=label_key,
                input_key=input_key, remember_input=new_input, values=values,
            ))

        board.push(chess.Move.from_uci(played))
        ply += 1

    for row in rows:
        row["result_pgn"] = result_pgn
        row["result"] = result_from_pov(result_pgn, white_to_move=row["stm"] == "w")
        row["adjudication"] = adjudication
    return GameOutcome(
        rows=rows,
        plies=ply,
        termination=termination,
        result_pgn=result_pgn,
        adjudication=adjudication,
        opening_source=start.source,
        adjudication_unavailable=unavailable,
        cache_events=cache_events,
    )


def build_opening_config(spec: WorkerSpec) -> OpeningConfig:
    """The production opening sampler's config, from this run's flags.

    ``chess_anti_engine.selfplay.opening`` is what live selfplay draws its
    openings from, so this generator uses it rather than a private PGN reader:
    a corpus whose opening distribution differs from production's is a corpus
    whose coverage claim does not transfer.  The blind-spot FEN-list branch is
    structurally unreachable here (no list path, zero draw probability), which
    is deliberate -- seeded refutations are curriculum data and this corpus is
    meant to have none.
    """
    return OpeningConfig(
        opening_book_path=spec.book,
        opening_book_max_plies=int(spec.book_plies),
        opening_book_max_games=int(spec.book_max_games),
        opening_book_prob=1.0 if spec.book else 0.0,
    )


def apply_nice(target: int) -> int:
    """Renice this process and return the priority it ACTUALLY has."""
    want = min(19, max(0, int(target)))
    try:
        current = os.getpriority(os.PRIO_PROCESS, 0)
        if want > current:
            os.setpriority(os.PRIO_PROCESS, 0, want)
    except OSError:  # pragma: no cover - permitted downward only
        pass
    return int(os.getpriority(os.PRIO_PROCESS, 0))


def worker_failure(
    exc: BaseException, *, progress: WorkerProgress, games_completed: int,
) -> dict[str, Any]:
    """What a dead worker's slot says.

    ``str(exc)`` is empty for several real exception types, so the type name is
    recorded separately rather than folded in -- a ``failed`` entry reading
    ``""`` names nothing at all.
    """
    return {
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "last_game_id": progress.game_id,
        "last_ply": progress.ply,
        "games_completed": int(games_completed),
    }


def failed_worker_slot(spec: WorkerSpec, failure: dict[str, Any]) -> dict[str, Any]:
    """A summary slot for a worker whose PROCESS died, not just its game loop.

    ``run_worker`` records its own failures and still returns its real counters;
    this is the other half -- an OOM kill or a segfault takes the process before
    any of that runs, and the parent then has nothing but the spec.  Every key
    the merge functions read is present and zeroed (through the same
    ``DedupStats``/``SearchStats`` summaries a live worker uses, so the shapes
    cannot drift apart), and the realized stamp says UNAVAILABLE rather than
    echoing the flags -- there is no engine to have realized anything.
    """
    return {
        "worker_id": spec.worker_id,
        "failed": failure,
        "games": 0,
        "rows": 0,
        "wall_s": math.nan,
        "plies_total": 0,
        "plies_mean": math.nan,
        "plies_max": 0,
        "terminations": {},
        "adjudications": {},
        "opening_sources": {},
        "history_plies_histogram": {},
        "history_root_reasons": {},
        "history_plies_histogram_prior": {},
        "history_root_reasons_prior": {},
        "history_tallies_unknown_rows_prior": 0,
        "adjudication_unavailable_plies": 0,
        "dedup": {
            **DedupStats().summary(),
            **DedupCache(max_entries=int(spec.dedup_cache_max)).summary(),
        },
        "search": SearchStats().summary(),
        "shards": [],
        # ⚑ EMPTY, not read back off the progress file, and that is a known
        # under-count rather than an oversight: a process that died before
        # `run_worker` ran has no adopted state to report, and inventing one
        # here from the disk would make a FAILED slot claim shards this run
        # never verified. The residual is the one the module docstring already
        # names for a hard-killed worker, and `failed_workers` is what tells a
        # reader the numbers beside it are incomplete.
        "shards_prior": [],
        "shards_abandoned": [],
        "games_completed_prior": 0,
        "dedup_rewarmed": 0,
        "dedup_cache_events_rewarmed": 0,
        "dedup_rewarmed_resident": 0,
        "resumed": bool(spec.resume),
        "resume_partials_deleted": [],
        "resume_progress_torn_tail": False,
        "resume_progress_repair": PROGRESS_ABSENT,
        "resume_legacy_progress_lines": 0,
        "codec": "none",
        "realized": {"unavailable_worker_process_died": True},
    }


def run_worker(spec: WorkerSpec) -> dict[str, Any]:
    """One worker: its own engine, its own shards, its own counters.

    ⚑ IT DOES NOT RAISE ON A FATAL ERROR OF ITS OWN.  A worker that dies has
    still produced shards, counters and a realized stamp, and letting the
    exception out would take the whole run's ``summary.json`` with it -- eight
    hours of other workers' searches lost to one engine's pty.  The failure is
    RECORDED in this slot (``failed``), the run keeps going, and ``main`` exits
    nonzero on the merged list.  What it must never do is return a slot that
    looks healthy.
    """
    nice_realized = apply_nice(spec.nice)
    # ⚑ FIRST, before the resume re-warm, the opening book and every board.
    apply_history_rep_fix()
    staircase = parse_staircase(spec.staircase)
    staircase_policy = validate_staircase_policy(spec.staircase_policy, staircase)

    def spawn_searcher(stats: SearchStats) -> StaircaseSearcher:
        return StaircaseSearcher(
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
            staircase_policy=staircase_policy,
        )

    cache = DedupCache(max_entries=int(spec.dedup_cache_max))
    # ⚑ BEFORE THE ENGINE EXISTS, and outside the try: like `parse_staircase`
    # above, this is a pre-flight over the spec, and a resume that cannot be
    # trusted must not reach the point where it appends games to a corpus. It
    # sits above the lease because the `finally` that closes an engine is
    # further down still -- a refusal up here cannot leak a Stockfish process.
    resume = (
        resume_worker_state(
            out_dir=spec.out_dir, worker_id=spec.worker_id, cache=cache,
        )
        if spec.resume else ResumeState.fresh()
    )
    rewarmed_resident = len(cache)
    foreign = sorted(resume.completed_games - set(spec.game_ids))
    if foreign:
        raise ValueError(
            f"worker {spec.worker_id}'s progress file claims games {foreign} "
            "that this run did not deal it; the game ids were dealt "
            "differently, so resuming would leave the difference unplayed",
        )
    writer = ShardWriter(
        out_dir=spec.out_dir, worker_id=spec.worker_id, shard_rows=spec.shard_rows,
        first_index=resume.next_shard_index,
        tally_keys=_SHARD_TALLY_KEYS,
    )
    game_ids = tuple(
        game_id for game_id in spec.game_ids
        if game_id not in resume.completed_games
    )
    searcher = EngineLease(spawn_searcher)
    opening_cfg = build_opening_config(spec)
    dedup = DedupStats()
    seq = WorkerSeq(resume.next_seq)
    progress = WorkerProgress()
    terminations: Counter[str] = Counter()
    adjudications: Counter[str] = Counter()
    opening_sources: Counter[str] = Counter()
    plies: list[int] = []
    games = 0
    games_started = 0
    unavailable = 0
    failure: dict[str, Any] | None = None
    started = time.perf_counter()
    try:
        for game_id in game_ids:
            games_started += 1
            outcome = play_game(
                spec=spec, searcher=searcher, opening_cfg=opening_cfg,
                game_id=game_id, cache=cache, dedup=dedup, progress=progress,
                seq=seq,
            )
            for row in outcome.rows:
                writer.write(row)
            writer.note_cache_events(outcome.cache_events)
            # ⚑ AFTER the last row and BEFORE the next game: this is the
            # commit point of the whole resume protocol. A kill between the
            # last `write` and here leaves the shard unlisted, so the game is
            # replayed whole; a kill after here has the game recorded.
            writer.end_game(game_id)
            games += 1
            unavailable += outcome.adjudication_unavailable
            plies.append(outcome.plies)
            terminations[outcome.termination] += 1
            adjudications[
                "none" if outcome.adjudication is None
                else f"syzygy_wdl_{outcome.adjudication['wdl']}"
            ] += 1
            opening_sources[outcome.opening_source] += 1
    except Exception as exc:
        failure = worker_failure(exc, progress=progress, games_completed=games)
        _LOG.exception(
            "worker %d died on game %s ply %s; recording the slot and letting "
            "the run finish", spec.worker_id, progress.game_id, progress.ply,
        )
    finally:
        writer.close()
        # ⚑ The engine is the thing that most plausibly just died, and the
        # lease's close() suppresses for exactly that reason: a close() that
        # raises on the way out of a RECORDED failure would turn it back into
        # an unrecorded one.
        searcher.close()
    wall_s = time.perf_counter() - started
    # ⚑ READ OFF THE INVENTORY, not off a counter kept beside it. A row this
    # worker wrote into a shard that was then ABANDONED unlisted (see
    # `ShardWriter.close`) is not in the corpus, and a parallel counter would
    # keep claiming it -- `summary["rows"]` must equal the rows a consumer
    # reaches by iterating `summary["shards"]`, on the failure path too.
    rows_written = sum(int(shard["rows"]) for shard in writer.shards)
    # ⚑ THE SAME RULE FOR THE WINDOW HISTOGRAMS. Each shard record carries the
    # tallies of the rows it holds (`ShardWriter.tally_keys`), committed with
    # the shard and dropped with it, so `sum(histogram) == rows` is an
    # identity over the inventory rather than a happy-path coincidence -- a
    # counter kept beside the writer kept counting rows whose shard was then
    # abandoned unlisted (Codex review of PR #497).
    history_plies = merge_shard_tallies(writer.shards, "history_plies")
    history_root_reasons = merge_shard_tallies(writer.shards, "history_root_reason")
    prior = prior_shard_tallies(resume.shards)
    return {
        "worker_id": spec.worker_id,
        "failed": failure,
        "games": games,
        "rows": rows_written,
        "wall_s": wall_s,
        "plies_total": int(sum(plies)),
        "plies_mean": statistics.fmean(plies) if plies else math.nan,
        "plies_max": max(plies) if plies else 0,
        "terminations": dict(terminations),
        "adjudications": dict(adjudications),
        "opening_sources": dict(opening_sources),
        "history_plies_histogram": dict(history_plies),
        "history_root_reasons": dict(history_root_reasons),
        # The adopted shards' tallies, kept apart from this session's so the
        # merge can publish both a corpus-level and a session-level reading.
        "history_plies_histogram_prior": prior.history_plies,
        "history_root_reasons_prior": prior.history_root_reasons,
        "history_tallies_unknown_rows_prior": prior.unknown_rows,
        "adjudication_unavailable_plies": unavailable,
        "dedup": {**dedup.summary(), **cache.summary()},
        "search": searcher.stats.summary(),
        "shards": writer.shards,
        # ⚑ THE KILLED SESSION'S SHARDS, so `summary.json`'s inventory names
        # the WHOLE corpus rather than the tail this process happened to add.
        # A summary that listed only this session's files would be a complete-
        # looking document indexing half a corpus.
        "shards_prior": list(resume.shards),
        # Empty unless the worker died between banking a game's rows and
        # ending it; the file is on disk, unlisted, and the next resume
        # deletes it. Reported so a failed slot says so out loud.
        "shards_abandoned": list(writer.abandoned),
        "games_completed_prior": len(resume.completed_games),
        "dedup_rewarmed": resume.dedup_rewarmed,
        "dedup_cache_events_rewarmed": resume.cache_events_rewarmed,
        # ⚑ The same claim READ OFF THE CACHE the worker then played through,
        # measured the instant the re-warm finished. A re-warm that counted its
        # puts and handed them to a cache the game loop never sees would report
        # a healthy `dedup_rewarmed` beside a zero here -- the accepted-then-
        # ignored shape, made visible instead of assumed away.
        "dedup_rewarmed_resident": rewarmed_resident,
        "resumed": bool(spec.resume),
        "resume_partials_deleted": list(resume.deleted_partials),
        "resume_progress_torn_tail": resume.torn_tail,
        "resume_progress_repair": resume.progress_repair,
        "resume_legacy_progress_lines": resume.legacy_lines,
        "codec": writer.codec,
        "realized": {
            **searcher.realized(),
            # `play_game` clears the table as its first act, so every STARTED
            # game must have delivered exactly one `ucinewgame` -- compared
            # against games started, not completed, or a worker that died
            # mid-game would read as a TT-hygiene failure it did not commit.
            "tt_cleared_per_game": searcher.stats.new_game_calls == games_started,
            "nice": nice_realized,
            # Read off the cache object, not off the spec: a bound that never
            # reached the cache is exactly the failure the stamp exists for.
            "dedup_cache_max": cache.max_entries,
            "shard_rows": writer.shard_rows,
            # Read off the WRITER that will do the numbering. A resume that
            # accepted the flag and then restarted its shard counter at 0 would
            # collide with the killed session's first shard on `open("x")`, and
            # this is the number that says which it did.
            "shard_index_start": writer.first_index,
            "codec": writer.codec,
            "opening_book_path": opening_cfg.opening_book_path,
            "opening_book_prob": float(opening_cfg.opening_book_prob),
            "opening_book_max_plies": int(opening_cfg.opening_book_max_plies),
            "opening_book_max_games": int(opening_cfg.opening_book_max_games),
            "temp_plies": int(spec.temp_plies),
            "temp_high": float(spec.temp_high),
            "temp_low": float(spec.temp_low),
            "max_plies": int(spec.max_plies),
            "seed": int(spec.seed),
        },
    }


# -- run assembly -------------------------------------------------------------


def default_syzygy_path() -> str:
    """The production Syzygy pair, resolved against a checkout that has it.

    This checkout first, then the MAIN checkout -- the same two-root lookup
    ``engine_discovery`` does for the binary, and for the same reason: ``data/``
    is untracked runtime output, so a worktree resolves it to nothing.  Falls
    back to this checkout's spelling when neither exists, so ``--help`` stays
    stable and machine-independent and the failure is "no such directory"
    rather than a ``None`` deep inside a probe.
    """
    roots = [REPO_ROOT]
    main = main_checkout()
    if main is not None and main != REPO_ROOT:
        roots.append(main)
    for root in roots:
        pair = [root / "data" / name for name in SYZYGY_DIR_NAMES]
        if all(p.is_dir() for p in pair):
            return os.pathsep.join(str(p) for p in pair)
    return os.pathsep.join(
        str(REPO_ROOT / "data" / name) for name in SYZYGY_DIR_NAMES
    )


def refuse_unopenable_syzygy(syzygy_path: str) -> tuple[str, ...]:
    """Refuse unless EVERY named tablebase directory opens.  Returns them.

    ⚑⚑ ``get_tablebase`` ON THE WHOLE PAIR IS NOT THIS CHECK.  It adds the
    directories it can and returns a handle when AT LEAST ONE of them worked, so
    ``<real 3-4-5 dir>:/typo/syzygy_6`` opens, passes, and then silently answers
    ``None`` to every 6-man probe for the length of the burn -- the run's
    ``<=6``-man games reach the ply cap with no result instead of being
    adjudicated, and nothing anywhere says why.  That is the same shape as
    Stockfish silently ignoring an illegal ``searchmoves``: a value accepted and
    then quietly dropped.  Production's own path is a PAIR (the 6-man DTZ lives
    in the second directory, see CLAUDE.md), so the half-open case is the
    realistic one, not a corner.
    """
    components = tuple(
        part.strip() for part in str(syzygy_path).split(os.pathsep) if part.strip()
    )
    if not components:
        raise ValueError(
            f"--syzygy-path {syzygy_path!r} names no directory. Adjudication is "
            "what gives a <=6-man game its result, so a run without it would "
            "bank rows whose value target is null.",
        )
    dead = [name for name in components if get_tablebase(name) is None]
    if dead:
        raise ValueError(
            f"--syzygy-path {syzygy_path!r} names {len(components)} directories "
            f"and {', '.join(repr(d) for d in dead)} opened no tablebase. Every "
            "component must open: a half-open pair probes what the live half "
            "holds and answers None for everything else, which is a corpus "
            "whose <=6-man games silently stop being adjudicated.",
        )
    return components


def split_games(total: int, workers: int) -> list[list[int]]:
    """Deal ``total`` game ids round-robin, so every worker's ids are distinct."""
    if int(total) <= 0:
        raise ValueError(
            f"--games must be positive, got {total!r}; state how large the "
            "corpus run is rather than letting it default to unbounded",
        )
    if int(workers) < 1:
        # Refused rather than clamped: a clamp runs one worker while the
        # requested stamp says 0, which is an accepted-then-ignored knob.
        raise ValueError(f"--workers must be >= 1, got {workers!r}")
    n = int(workers)
    buckets: list[list[int]] = [[] for _ in range(n)]
    for game_id in range(int(total)):
        buckets[game_id % n].append(game_id)
    return [b for b in buckets if b]


def config_stamp(args: argparse.Namespace, *, sf_binary: str) -> dict[str, Any]:
    """The REQUESTED configuration, exactly as the CLI stated it."""
    return {
        "out_dir": str(args.out_dir),
        "games": int(args.games),
        "workers": int(args.workers),
        "staircase": str(args.staircase),
        "staircase_policy": str(args.staircase_policy),
        "seed": int(args.seed),
        "temp_plies": int(args.temp_plies),
        "temp_high": float(args.temp_high),
        "temp_low": float(args.temp_low),
        "max_plies": int(args.max_plies),
        "shard_rows": int(args.shard_rows),
        "sf_hash_mb": int(args.sf_hash_mb),
        "sf_read_timeout_s": float(args.sf_read_timeout),
        "sf_search_timeout_s": float(args.sf_search_timeout),
        "dedup_cache_max": int(args.dedup_cache_max),
        "syzygy_path": str(args.syzygy_path),
        "nice": int(args.nice),
        "cp_slope": float(args.cp_slope),
        "cp_draw_width": float(args.cp_draw_width),
        "book": None if args.book is None else str(args.book),
        "book_plies": int(args.book_plies),
        "book_max_games": int(args.book_max_games),
        "run_id": str(args.run_id),
        "stockfish": sf_binary,
    }


def stamp_sha256(stamp: dict[str, Any]) -> str:
    """A content hash of the requested config, carried in every row."""
    return hashlib.sha256(
        json.dumps(stamp, sort_keys=True).encode("utf-8"),
    ).hexdigest()


def merge_counters(results: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for result in results:
        for name, count in result[key].items():
            merged[str(name)] += int(count)
    return dict(merged)


#: The row fields ``ShardWriter`` tallies per committed shard for the worker
#: summary.  ``history_plies`` feeds ``history_plies_histogram`` and
#: ``history_root_reason`` feeds ``history_root_reasons``.
_SHARD_TALLY_KEYS: tuple[str, ...] = ("history_plies", "history_root_reason")


def merge_shard_tallies(
    shards: Sequence[Mapping[str, Any]], key: str,
) -> dict[str, int]:
    """One tallied field summed over COMMITTED shard records."""
    merged: Counter[str] = Counter()
    for shard in shards:
        for name, count in shard["tallies"][key].items():
            merged[str(name)] += int(count)
    return dict(merged)


@dataclass(frozen=True)
class PriorTallies:
    """The adopted shards' window tallies, and the rows that carried none."""

    history_plies: dict[str, int]
    history_root_reasons: dict[str, int]
    #: Rows in adopted shards whose progress record predates ``tallies``.
    #: Reported, not folded into a bucket: ``sum(histogram) + unknown == rows``.
    unknown_rows: int


def _carries_tallies(shard: Mapping[str, Any]) -> bool:
    """Whether a shard record tallies EVERY key the summary reads.

    A record with no ``tallies`` predates them; one whose ``tallies`` lacks a
    key was written by a writer configured without it.  Both are UNKNOWN rows
    rather than a zero bucket -- a partial tally is not a smaller histogram.
    """
    tallies = shard.get("tallies")
    return isinstance(tallies, Mapping) and all(k in tallies for k in _SHARD_TALLY_KEYS)


def prior_shard_tallies(shards: Sequence[Mapping[str, Any]]) -> PriorTallies:
    with_tallies = [shard for shard in shards if _carries_tallies(shard)]
    unknown = sum(int(shard["rows"]) for shard in shards if not _carries_tallies(shard))
    return PriorTallies(
        history_plies=merge_shard_tallies(with_tallies, "history_plies"),
        history_root_reasons=merge_shard_tallies(with_tallies, "history_root_reason"),
        unknown_rows=unknown,
    )


def merge_counters_across(
    results: Sequence[dict[str, Any]], *keys: str,
) -> dict[str, int]:
    """Several per-worker counters summed into one (prior + this session)."""
    merged: Counter[str] = Counter()
    for key in keys:
        for name, count in merge_counters(results, key).items():
            merged[name] += count
    return dict(merged)


#: The scalar two-key counters ``DedupStats.summary`` emits, summed by
#: ``merge_dedup``.  Listed once so the merge cannot silently skip one.
_DEDUP_SCALAR_COUNTERS: tuple[str, ...] = (
    "row_key_hits", "search_key_hits", "search_key_hit_on_new_input",
    "search_key_miss_on_seen_input", "searches", "rows_banked",
    "dedup_input_set_entries", "dedup_input_set_evictions",
)


def merge_dedup(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    seen = sum(int(r["dedup"]["positions_first_seen"]) for r in results)
    hits = sum(int(r["dedup"]["dup_hits"]) for r in results)
    total = seen + hits
    cached = sum(int(r["dedup"]["dedup_cache_entries"]) for r in results)
    cache_bytes = sum(int(r["dedup"]["dedup_cache_bytes_est"]) for r in results)
    return {
        "positions_visited": total,
        "positions_first_seen": seen,
        "dup_hits": hits,
        "dup_rate": (hits / total) if total else math.nan,
        **{
            name: sum(int(r["dedup"][name]) for r in results)
            for name in _DEDUP_SCALAR_COUNTERS
        },
        "first_seen_by_phase": {
            p: sum(int(r["dedup"]["first_seen_by_phase"][p]) for r in results)
            for p in GAME_PHASES
        },
        "dup_hits_by_phase": {
            p: sum(int(r["dedup"]["dup_hits_by_phase"][p]) for r in results)
            for p in GAME_PHASES
        },
        # ⚑ The cache is PER WORKER, so a position two workers both reach is
        # searched twice and banked twice.  Stated rather than left for a reader
        # to discover from a duplicate `dedup_key` in the corpus.
        "cache_scope": "per_worker",
        # ⚑ ... and the SAME thing happens within one worker once the bound
        # starts evicting, which is why this count sits next to `dup_hits`
        # rather than in a diagnostics corner: it is the number of duplicate
        # `dedup_key`s the corpus may contain for a reason other than scope.
        "dedup_cache_evictions": sum(
            int(r["dedup"]["dedup_cache_evictions"]) for r in results
        ),
        "dedup_cache_entries": cached,
        "dedup_cache_max_entries_per_worker": max(
            int(r["dedup"]["dedup_cache_max_entries"]) for r in results
        ),
        "dedup_cache_bytes_est": cache_bytes,
        "dedup_cache_bytes_per_entry_est": (
            cache_bytes / cached if cached else math.nan
        ),
        "dedup_cache_eviction_policy": "fifo",
        "dedup_cache_eviction_semantics": (
            "an evicted position that recurs is re-searched and RE-BANKED; "
            "two rows may share a dedup_key"
        ),
    }


def merge_search(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    positions = sum(int(r["search"]["positions_searched"]) for r in results)
    search_s = sum(float(r["search"]["search_s"]) for r in results)
    anomalies: Counter[str] = Counter()
    for result in results:
        for name, count in result["search"]["anomalies"].items():
            anomalies[str(name)] += int(count)
    counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    mins: dict[str, list[int]] = {}
    maxs: dict[str, list[int]] = {}
    buckets: dict[str, Counter[str]] = {}
    medians: dict[str, dict[str, float]] = {}
    for result in results:
        for phase, cell in result["search"]["nodes_by_phase"].items():
            name = str(phase)
            counts[name] += int(cell["n"])
            totals[name] += int(cell["total"])
            mins.setdefault(name, []).append(int(cell["min"]))
            maxs.setdefault(name, []).append(int(cell["max"]))
            medians.setdefault(name, {})[str(result["worker_id"])] = float(
                cell["median_est_reservoir"],
            )
            into = buckets.setdefault(name, Counter())
            for bucket, count in cell["log2_buckets"].items():
                into[str(bucket)] += int(count)
    nodes = {
        name: {
            "n": int(counts[name]),
            "total": int(totals[name]),
            "mean": (totals[name] / counts[name]) if counts[name] else math.nan,
            "min": min(mins[name]),
            "max": max(maxs[name]),
            # ⚑ PER WORKER, and no pooled median.  n, total, min and max merge
            # exactly; a median does not.  Concatenating the workers' equal-size
            # reservoirs would weight a worker that searched 10k positions the
            # same as one that searched 10, so the "merged median" would be a
            # number with no population -- and it would look exactly as
            # authoritative as the four beside it.  Publishing them separately
            # is the honest shape, and it is what makes the estimate reach
            # `summary.json` at all: the per-worker search block does not.
            "median_est_reservoir_by_worker": dict(sorted(
                medians[name].items(), key=lambda kv: int(kv[0]),
            )),
            "log2_buckets": dict(sorted(
                buckets[name].items(), key=lambda kv: int(kv[0]),
            )),
        }
        for name in sorted(counts)
    }
    return {
        "positions_searched": positions,
        "searches": sum(int(r["search"]["searches"]) for r in results),
        "search_s": search_s,
        "s_per_position": (search_s / positions) if positions else math.nan,
        "anomalies": dict(anomalies),
        "staircase_gate": {
            name: sum(
                int(r["search"]["staircase_gate"][name]) for r in results
            )
            for name in ("evaluations", "extended", "stopped", "forced_stops")
        },
        "nodes_by_phase": nodes,
    }


def build_summary(
    *,
    results: Sequence[dict[str, Any]],
    requested: dict[str, Any],
    config_sha: str,
    engine_record: dict[str, Any],
    engine_id_name: str | None,
    staircase: Sequence[StaircasePhase],
    started_utc: str,
    wall_s: float,
) -> dict[str, Any]:
    rows_session = sum(int(r["rows"]) for r in results)
    games_session = sum(int(r["games"]) for r in results)
    rows_prior = sum(
        int(shard["rows"]) for r in results for shard in r["shards_prior"]
    )
    games_prior = sum(int(r["games_completed_prior"]) for r in results)
    # ⚑ CORPUS TOTALS, not session totals, and deliberately under the names a
    # consumer already reads. On a fresh run the prior halves are 0 and every
    # number below is what it always was; on a RESUMED run a reader who ignores
    # every key added for the resume still gets the size of the corpus in front
    # of them rather than the size of the last shift. The session-scoped
    # numbers keep the `_this_session` suffix, which is the safe direction to
    # be wrong in: an unread key understates nothing.
    rows = rows_session + rows_prior
    games = games_session + games_prior
    search = merge_search(results)
    failed_workers = [
        {"worker_id": r["worker_id"], **r["failed"]}
        for r in results if r["failed"] is not None
    ]
    return {
        "schema": SUMMARY_SCHEMA,
        "row_schema": ROW_SCHEMA,
        # ⚑ THE SUMMARY'S OWN VERDICT, and the only thing `--resume` is allowed
        # to read as one. This file is written on the failure path too -- a
        # crash the parent process survives (an OOM kill that breaks the worker
        # pool) reaches the end of `run` and banks a summary whose counters are
        # all zero -- so its EXISTENCE says nothing. Derived here, beside the
        # list it is derived from, so the two can never disagree; the condition
        # is `main`'s exit code, spelled once.
        "run_finished": not failed_workers,
        "run_id": requested["run_id"],
        "started_utc": started_utc,
        "wall_s": wall_s,
        "rows": rows,
        "games": games,
        # ⚑ `plies_total`, `dedup`, `search`, `terminations`, `adjudications`
        # and `opening_sources` are THIS SESSION's and cannot be otherwise: a
        # killed run's summary.json was never written, so its counters died
        # with it. The progress file preserves rows, shards and games, which is
        # exactly why those three -- and only those three -- carry prior state.
        "resumed": any(bool(r["resumed"]) for r in results),
        "rows_this_session": rows_session,
        "games_this_session": games_session,
        "rows_prior": rows_prior,
        "games_completed_prior": games_prior,
        "games_completed_prior_by_worker": {
            str(r["worker_id"]): int(r["games_completed_prior"]) for r in results
        },
        "dedup_rewarmed": sum(int(r["dedup_rewarmed"]) for r in results),
        # ⚑ ITS OWN COLUMN (Grok round 5): computed per worker and then
        # dropped here was the accepted-then-ignored shape.
        "dedup_cache_events_rewarmed": sum(
            int(r["dedup_cache_events_rewarmed"]) for r in results
        ),
        "dedup_cache_events_rewarmed_by_worker": {
            str(r["worker_id"]): int(r["dedup_cache_events_rewarmed"]) for r in results
        },
        "dedup_rewarmed_resident_by_worker": {
            str(r["worker_id"]): int(r["dedup_rewarmed_resident"]) for r in results
        },
        "resume_partials_deleted": {
            str(r["worker_id"]): list(r["resume_partials_deleted"])
            for r in results if r["resume_partials_deleted"]
        },
        "resume_progress_torn_tail_workers": [
            int(r["worker_id"]) for r in results if r["resume_progress_torn_tail"]
        ],
        # ⚑ Only the workers whose progress file the kill actually damaged.
        # `newline_restored` is here because it is otherwise invisible: the
        # record was whole, only its terminator was lost, and the repair is
        # the difference between keeping that record and concatenating the
        # next one onto it.
        "resume_progress_repaired": {
            str(r["worker_id"]): str(r["resume_progress_repair"])
            for r in results
            if r["resume_progress_repair"] in (
                PROGRESS_NEWLINE_RESTORED, PROGRESS_TRUNCATED,
            )
        },
        "resume_legacy_progress_lines": sum(
            int(r["resume_legacy_progress_lines"]) for r in results
        ),
        # ⚑ Rows this run WROTE and then dropped from the inventory, because
        # the shard holding them also held a game that never ended. Empty on
        # every healthy run; never silent, because the alternative reading of
        # a shrunken row count is that the generator lost rows for no reason.
        "shards_abandoned": {
            str(r["worker_id"]): list(r["shards_abandoned"])
            for r in results if r["shards_abandoned"]
        },
        "plies_total": sum(int(r["plies_total"]) for r in results),
        "rows_per_game": (rows / games) if games else math.nan,
        # Wall clock is this session's, so the rate it divides has to be too.
        "s_per_row": (wall_s / rows_session) if rows_session else math.nan,
        # ⚑ TOP LEVEL, and empty on a healthy run.  A partial corpus that reads
        # like a complete one is the whole hazard here: every other number in
        # this file is a sum over the workers that reported, and a reader has no
        # way to know one of them stopped early unless the summary says so.
        "failed_workers": failed_workers,
        "dedup": merge_dedup(results),
        "search": search,
        "terminations": merge_counters(results, "terminations"),
        "adjudications": merge_counters(results, "adjudications"),
        "adjudication_unavailable_plies": sum(
            int(r["adjudication_unavailable_plies"]) for r in results
        ),
        "opening_sources": merge_counters(results, "opening_sources"),
        # ⚑ The window every banked row of the CORPUS carries -- prior shards'
        # tallies plus this session's, so `sum(histogram) + unknown == rows`
        # holds at this level exactly as it does per worker (Codex P2, round
        # 2: a zero-work resume used to report rows > 0 beside an empty
        # histogram). The session-only reading keeps the `_this_session`
        # suffix like every other session-scoped counter here. A corpus whose
        # rows are all `history_plies: 0` is one the schema bump did not
        # reach, and this is where that shows.
        "history_plies_histogram": merge_counters_across(
            results, "history_plies_histogram", "history_plies_histogram_prior",
        ),
        "history_root_reasons": merge_counters_across(
            results, "history_root_reasons", "history_root_reasons_prior",
        ),
        "history_plies_histogram_this_session": merge_counters(
            results, "history_plies_histogram",
        ),
        "history_root_reasons_this_session": merge_counters(
            results, "history_root_reasons",
        ),
        "history_tallies_unknown_rows": sum(
            int(r["history_tallies_unknown_rows_prior"]) for r in results
        ),
        # Prior first, then this session's, per worker: the order they were
        # banked in, which is also the order their rows must be re-warmed in.
        "shards": [
            shard for r in results
            for shard in [*r["shards_prior"], *r["shards"]]
        ],
        "config_requested": requested,
        "config_sha256": config_sha,
        "staircase_parsed": [
            {"width": p.width_label, "depth": p.depth} for p in staircase
        ],
        "staircase_gate": staircase_gate_stamp(
            str(requested.get("staircase_policy", STAIRCASE_POLICY_FIXED)),
        ),
        # ⚑ REALIZED, one entry per worker, every field read off the live engine
        # (or off this process) rather than echoed from the flags -- see
        # `StaircaseSearcher.realized`.
        "config_realized_by_worker": {
            str(r["worker_id"]): r["realized"] for r in results
        },
        "engine": {**engine_record, "id_name": engine_id_name},
        "banked_rows_min_piece_count": MIN_BANKED_PIECES,
        KEY_HISTORY_REP_FIX: HISTORY_REP_FIX,
        "adjudication_max_piece_count": ADJUDICATION_MAX_PIECES,
        "python": sys.version.split()[0],
    }


def _histogram_range(histogram: dict[str, Any]) -> str:
    """``min..max (n rows)`` for a ``{bucket: count}`` map, or ``"none"``."""
    if not histogram:
        return "none"
    buckets = sorted(int(k) for k in histogram)
    total = sum(int(v) for v in histogram.values())
    return f"{buckets[0]}..{buckets[-1]} ({total} rows)"


def format_summary(summary: dict[str, Any]) -> str:
    dedup = summary["dedup"]
    search = summary["search"]
    gate_rule = summary.get(
        "staircase_gate",
        staircase_gate_stamp(STAIRCASE_POLICY_FIXED),
    )
    gate_counts = search.get("staircase_gate", {})
    gate_line = f"staircase policy={gate_rule['policy']}"
    if gate_rule.get("adaptive"):
        gate_line += (
            f" extended={int(gate_counts.get('extended', 0))}/"
            f"{int(gate_counts.get('evaluations', 0))} "
            f"stopped={int(gate_counts.get('stopped', 0))} "
            f"forced_stops={int(gate_counts.get('forced_stops', 0))}"
        )
    lines = [
        f"games={summary['games']} rows={summary['rows']} "
        f"plies={summary['plies_total']}",
        f"positions searched={search['positions_searched']} "
        f"s/pos={search['s_per_position']:.3f} "
        f"dup_rate={dedup['dup_rate']:.4f} "
        f"(open={dedup['dup_hits_by_phase']['opening']} "
        f"mid={dedup['dup_hits_by_phase']['middlegame']} "
        f"end={dedup['dup_hits_by_phase']['endgame']})",
        f"dedup cache entries={dedup['dedup_cache_entries']} "
        f"evictions={dedup['dedup_cache_evictions']} "
        f"bytes/entry={dedup['dedup_cache_bytes_per_entry_est']:.0f}",
        gate_line,
        f"anomalies={search['anomalies']}",
        f"terminations={summary['terminations']}",
        # ⚑ `.get` for the same reason `run_finished` uses one below: this
        # formats summaries loaded off disk, and a schema-1 corpus's summary
        # has no window to report.
        f"history roots={summary.get('history_root_reasons', {})} "
        f"plies={_histogram_range(summary.get('history_plies_histogram', {}))}",
    ]
    if summary["resumed"]:
        # Said out loud, because every line above it is a CORPUS total on a
        # resumed run and the operator's mental model is "what did this shift
        # do".  Both numbers, named.
        lines.insert(0, (
            f"RESUMED: {summary['games_completed_prior']} games and "
            f"{summary['rows_prior']} rows adopted from the killed session, "
            f"{summary['dedup_rewarmed']} dedup entries and "
            f"{summary['dedup_cache_events_rewarmed']} cache-only events "
            "re-warmed; this "
            f"session played {summary['games_this_session']} games / "
            f"{summary['rows_this_session']} rows"
        ))
    lines.extend(
        f"FAILED worker {failed['worker_id']}: {failed['exception_type']}: "
        f"{failed['exception']} (game {failed['last_game_id']} "
        f"ply {failed['last_ply']}, {failed['games_completed']} games done)"
        for failed in summary["failed_workers"]
    )
    # ⚑ `.get`, and only on an explicit `False`: this formats summaries loaded
    # off disk as well as freshly built ones, and a summary written before
    # `run_finished` existed makes no claim either way. Silence is "did not
    # say", never "finished".
    if summary.get("run_finished") is False:
        lines.append(
            "RUN DID NOT FINISH (run_finished: false): every number above is "
            "what the workers that reported managed. --resume continues it.",
        )
    return "\n".join(lines)


def write_launch_manifest(
    out_dir: Path,
    *,
    requested: dict[str, Any],
    config_sha: str,
    staircase: Sequence[StaircasePhase],
    engine_record: dict[str, Any],
    engine_id_name: str | None,
) -> None:
    """Bank the launch facts BEFORE the first game, ``"x"`` like everything here.

    ``summary.json`` is written once, at run END -- so on a multi-day run a
    crash near the end would leave millions of banked rows that
    ``derive_corpus_targets`` refuses: no cp map, no staircase, no inventory.
    The manifest banks the launch-time facts immediately and the per-worker
    ``w<id>.progress.jsonl`` files (see ``ShardWriter.close``) bank the shard
    inventory incrementally.  ``complete: false`` is stamped so no reader can
    mistake this for the run record: ``summary.json`` remains the only document
    that can say the run finished, and a partial read has to say it is one.
    ⚑ *Can* say, not *does*: the summary states its verdict in ``run_finished``
    and a crashed session's says ``false``.  This key is the MANIFEST's, it is
    about which document you are holding, and it is never updated.
    """
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "row_schema": ROW_SCHEMA,
        "complete": False,
        "config_requested": requested,
        "config_sha256": config_sha,
        "staircase_parsed": [
            {"width": p.width_label, "depth": p.depth} for p in staircase
        ],
        "staircase_gate": staircase_gate_stamp(
            str(requested.get("staircase_policy", STAIRCASE_POLICY_FIXED)),
        ),
        "engine": {**engine_record, "id_name": engine_id_name},
        "banked_rows_min_piece_count": MIN_BANKED_PIECES,
        KEY_HISTORY_REP_FIX: HISTORY_REP_FIX,
        "adjudication_max_piece_count": ADJUDICATION_MAX_PIECES,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    # ⚑ "x": a manifest is written ONCE, by the session that opened the corpus.
    # A resume never reaches this function -- it VALIDATES against the manifest
    # instead (see `load_resume_manifest`), because a second manifest, or an
    # overwritten one, is the corpus losing the record of what produced its
    # first half.
    with open(out_dir / MANIFEST_NAME, "x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=_json_default)
        fh.write("\n")


def summary_run_finished(summary: object) -> bool | None:
    """The completion verdict a summary STATES, or ``None`` if it states none.

    ⚑ ``None`` is not "no", it is "this document never said" -- what a summary
    written before ``run_finished`` existed looks like, and what an unreadable
    or non-object one looks like too.  Every caller has to decide that case
    deliberately; :func:`load_resume_manifest` refuses on it, which leaves a
    genuinely completed legacy corpus behaving exactly as it always did.

    ⚑ ``isinstance(..., bool)`` rather than truthiness, on purpose.  The string
    ``"false"`` is true in Python, and a completion claim decided by truthiness
    is precisely the accepted-then-silently-ignored failure this repo keeps
    finding: it would read a hand-edited summary as finished and refuse a
    resume that should have run.
    """
    if not isinstance(summary, dict):
        return None
    claim = summary.get("run_finished")
    return claim if isinstance(claim, bool) else None


def read_summary_run_finished(path: Path) -> bool | None:
    """:func:`summary_run_finished` for a summary on disk.

    An unreadable or unparseable file is ``None`` -- ambiguous, exactly like a
    summary that predates the key -- rather than an exception, because the
    caller's response to both is the same refusal and a ``JSONDecodeError``
    out of a resume gate says less than that refusal does.
    """
    try:
        return summary_run_finished(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def unfinished_archive_path(path: Path) -> Path:
    """The first free ``<stem>.unfinished_NN<suffix>`` beside ``path``.

    Counted rather than timestamped so the name is a pure function of what is
    already on disk: a corpus that survived three crashes keeps all three
    records, in the order they happened, and no clock skew or same-minute
    collision can make one overwrite another.  WHEN each crash was is inside
    the file it names, in ``started_utc`` and ``wall_s``.
    """
    index = 0
    while True:
        candidate = path.with_name(UNFINISHED_ARCHIVE_TEMPLATE.format(
            stem=path.stem, index=index, suffix=path.suffix,
        ))
        if not candidate.exists():
            return candidate
        index += 1


def archive_unfinished_summary(out_dir: Path) -> Path | None:
    """Move a crashed session's ``summary.json`` aside, returning where it went.

    ⚑ THIS IS WHAT MAKES THE RELAXED GATE SAFE RATHER THAN MERELY PERMISSIVE.
    ``run`` banks its summary with ``open("x")``, so a resume that started with
    a stale one still in place would search for however many days and then die
    on the very last line -- the days-late failure the old blanket refusal was
    really protecting against.  Moved, never overwritten: the crashed record
    holds the ``failed_workers`` block naming what killed the run, and it is
    the only copy of it.

    ``None`` when there is no summary to move, which is every ordinary
    ``kill -9`` resume.  A summary that does not state ``run_finished: false``
    is REFUSED rather than moved: :func:`load_resume_manifest` has already
    turned those away, and a second reader willing to move one would be a gate
    you get past by calling the wrong function first.
    """
    summary_path = out_dir / SUMMARY_NAME
    if not summary_path.exists():
        return None
    if read_summary_run_finished(summary_path) is not False:
        raise ValueError(
            f"refusing to move {summary_path} aside: it does not state "
            "run_finished: false, so it is not a crashed session's record.",
        )
    archive = unfinished_archive_path(summary_path)
    summary_path.rename(archive)
    _LOG.info(
        "resume: the crashed session's %s is kept as %s; this session will "
        "write its own", SUMMARY_NAME, archive.name,
    )
    return archive


def archive_json_copy_for_resume(json_path: Path | None) -> Path | None:
    """Free the ``--json`` path the resumed session will write its copy to.

    ⚑ THE SAME DAYS-LATE EXPLOSION AS THE SUMMARY, ONE FUNCTION FURTHER OUT.
    ``main`` writes the ``--json`` copy with its own ``open("x")`` AFTER ``run``
    returns, so a crashed session that had ``--json`` left TWO files behind and
    :func:`archive_unfinished_summary` frees only one.  Repeating the original
    command would then archive the summary, search for however many days, bank
    a correct corpus and its ``summary.json`` -- and traceback on the last line
    of ``main``, leaving automation an error code beside a complete corpus.

    ⚑ NO VERDICT IS READ HERE, and that is the difference from
    :func:`archive_unfinished_summary` rather than an omission.  ``--json`` is
    an OUTPUT LOCATION the operator chose, not corpus identity: it is not in
    ``config_stamp`` (so ``refuse_resume_config_drift`` cannot speak to it),
    nothing ties it to this ``--out-dir``, and a previous run of something else
    may simply own that name.  The corpus's OWN verdict was settled before this
    is ever called -- ``load_resume_manifest`` refused every summary that said
    it finished -- so whatever sits at the aux path is stale or foreign either
    way, and refusing on it would be a NEW false refusal of exactly the kind
    this change exists to remove.  So: moved aside, never deleted, never
    overwritten, and said out loud in the log.

    ``None`` when no ``--json`` was given or nothing is at that path, which is
    every ordinary resume.
    """
    if json_path is None or not json_path.exists():
        return None
    archive = unfinished_archive_path(json_path)
    json_path.rename(archive)
    _LOG.info(
        "resume: the file already at the --json path is kept as %s; this "
        "session will write its own copy there", archive.name,
    )
    return archive


def read_launch_manifest(out_dir: Path) -> dict[str, Any]:
    """The manifest, self-checked -- and NOTHING about whether to resume onto it.

    ⚑ THE READER'S HALF, split out (review round 2): the deriver opens a
    live/killed corpus through this, and it must be able to open EVERY schema
    the deriver can dispatch on -- the 54M-row legacy corpora are schema 1 --
    while the generator's ``--resume`` must refuse the same manifest.  The
    row-schema gate is a statement about APPENDING to a corpus, so it lives in
    :func:`load_resume_manifest`; what belongs to every reader is that the
    record hashes its own ``config_requested``.
    """
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise ValueError(f"{manifest_path} does not exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored = str(manifest.get("config_sha256", ""))
    recomputed = stamp_sha256(dict(manifest.get("config_requested", {})))
    if recomputed != stored:
        raise ValueError(
            f"{manifest_path} is inconsistent with itself: config_sha256 is "
            f"{stored!r} but its own config_requested hashes to {recomputed!r}. "
            "The record of what produced the banked rows has been altered, and "
            "anything validated against it would prove nothing.",
        )
    return manifest


def load_resume_manifest(out_dir: Path) -> dict[str, Any]:
    """The manifest a ``--resume`` continues, or a refusal.

    Four refusals, all before a single game is played:

    * NO MANIFEST -- there is no run here to continue.  Told to drop the flag
      rather than quietly turned into a fresh run, because a fresh run into a
      populated directory is the thing `refuse_populated_dir` exists to stop.
    * A ``summary.json`` that says ``run_finished: true`` -- that run FINISHED.
      Resuming would append games to a completed corpus and then fail at the
      very end on the summary's own ``open("x")``, after however many days of
      searching.
    * A ``summary.json`` that states NO verdict -- unreadable, or written
      before ``run_finished`` existed.  ⚑ Refused rather than assumed either
      way.  Failing closed costs one manual rename on a legacy directory;
      failing open would append games to a corpus that really had finished,
      and no consumer could then tell which rows were which.
    * A manifest whose ``config_sha256`` does not hash its own
      ``config_requested`` -- the record of what produced the banked half has
      been edited or truncated, and every later comparison against it would be
      comparing against fiction.

    ⚑ What is NOT a refusal: a ``summary.json`` that says
    ``run_finished: false``.  A crash the parent process survives -- the
    measured one is an OOM kill breaking the worker pool -- writes a summary
    recording its ``failed_workers`` and zero session totals, and that corpus
    is exactly the one a resume is for.  Before 2026-08-29 the gate keyed on
    the file's EXISTENCE, so such a run could not be resumed at all until an
    operator renamed the crash record by hand.
    """
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise ValueError(
            f"--resume was given but {manifest_path} does not exist, so there "
            "is no run in this directory to continue. Drop --resume to start "
            "one (into an EMPTY directory).",
        )
    summary_path = out_dir / SUMMARY_NAME
    if summary_path.exists():
        finished = read_summary_run_finished(summary_path)
        if finished is None:
            raise ValueError(
                f"--resume was given but {summary_path} states no "
                "run_finished verdict -- it is unreadable, or it was written "
                "before this generator stamped one. Refused rather than "
                "guessed at, because guessing wrong appends games to a "
                "finished corpus. If that run in fact crashed (its "
                "failed_workers block says so), move the file aside and "
                "re-run --resume; if it completed, use a new --out-dir.",
            )
        if finished:
            raise ValueError(
                f"--resume was given but {summary_path} says run_finished: "
                "true: that run completed and wrote its summary. Nothing to "
                "resume; use a new --out-dir.",
            )
    manifest = read_launch_manifest(out_dir)
    # ⚑ THE ROW SCHEMA IS A REFUSAL HERE, AT THE MANIFEST, not a KeyError
    # somewhere inside a worker. The per-row check in the cache re-warm fires
    # only for a worker that HAS shards; a worker whose progress file holds
    # nothing but zero-row completion records re-warms nothing, passes, and
    # appends schema-2 rows beside the other workers' schema-1 shards -- one
    # corpus, two row shapes, and a manifest that describes only the first.
    # `run` calls this before it archives any record and before any worker
    # is dispatched, so a refused resume leaves the directory untouched.
    banked_schema = manifest.get("row_schema")
    if banked_schema == ROW_SCHEMA and manifest.get(KEY_HISTORY_REP_FIX) != HISTORY_REP_FIX:
        raise ValueError(
            f"--resume was given but {manifest_path} stamps "
            f"{KEY_HISTORY_REP_FIX}={manifest.get(KEY_HISTORY_REP_FIX)!r} and "
            f"this build encodes under {HISTORY_REP_FIX}; the banked input_keys "
            "were hashed in another repetition-plane regime and cannot be "
            "continued. Use a new --out-dir.",
        )
    if banked_schema != ROW_SCHEMA:
        raise ValueError(
            f"--resume was given but {manifest_path} was opened under row "
            f"schema {banked_schema!r} and this build writes schema "
            f"{ROW_SCHEMA}. A schema-1 row has no window to re-bank; a "
            f"schema-{ROW_SCHEMA_HISTORY_WITHOUT_KEYS} row is a window without "
            "search_key/input_key (never produced outside smoke runs) and has "
            "nothing to re-warm the two-key cache from. Regenerate the corpus "
            "(or repair it with a tool that adds the keys); use a new "
            "--out-dir.",
        )
    return manifest


def refuse_resume_config_drift(
    manifest: Mapping[str, Any], *, requested: Mapping[str, Any],
) -> None:
    """Refuse a resume that would change a generation-affecting setting.

    ⚑⚑ FIELD BY FIELD AGAINST THE MANIFEST'S OWN DICT, NOT SHA AGAINST SHA.
    A stamp comparison looks stricter and is in fact *weaker where it matters*:
    the moment this file gains or renames a stamp key, every recomputed sha
    stops matching every manifest ever written, and a run that has been burning
    for days becomes unresumable for a reason that has nothing to do with its
    configuration.  So the manifest's dict is the authority -- each key IT
    carries is compared against what this invocation asks for, and a key it
    does not carry (because the session that wrote it predates the key) is
    simply not a claim anyone made.

    ``out_dir`` is compared RESOLVED: the same directory reached by a different
    spelling is the same directory, and the corpus is the files, not the path.
    """
    stamped = dict(manifest.get("config_requested", {}))
    drifted: list[str] = []
    # Before this knob existed every staircase was fixed.  The generic rule
    # below correctly treats a newly added key as "no old claim", but this key
    # has a knowable historical meaning: allowing an old fixed
    # all:9,8:10,4:12 run to resume under G10 would mix fixed and adaptive rows
    # while retaining the old config hash and top-level fixed stamp.
    if "staircase_policy" not in stamped:
        current_policy = str(requested.get(
            "staircase_policy",
            STAIRCASE_POLICY_FIXED,
        ))
        if current_policy != STAIRCASE_POLICY_FIXED:
            drifted.append(
                f"staircase_policy: manifest {STAIRCASE_POLICY_FIXED!r} "
                f"(the pre-policy default) -> {current_policy!r}",
            )
    for key, banked in sorted(stamped.items()):
        if key not in requested:
            drifted.append(f"{key}: manifest {banked!r}, this run does not stamp it")
            continue
        current = requested[key]
        if key == "out_dir":
            if Path(str(banked)).resolve() != Path(str(current)).resolve():
                drifted.append(f"{key}: {banked!r} -> {current!r}")
            continue
        if banked != current:
            drifted.append(f"{key}: {banked!r} -> {current!r}")
    if drifted:
        raise ValueError(
            "--resume must continue the SAME run, and these settings differ "
            "from the manifest this corpus was opened with:\n  "
            + "\n  ".join(drifted)
            + "\nA corpus whose rows were produced under two configurations "
            "has a stamp that describes only half of it. Re-run with the "
            "original settings, or start a new --out-dir.",
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate, fan out, merge and write ``summary.json``."""
    apply_history_rep_fix()
    staircase = parse_staircase(str(args.staircase))
    staircase_policy = validate_staircase_policy(
        str(args.staircase_policy),
        staircase,
    )
    if float(args.temp_high) <= 0.0 or float(args.temp_low) <= 0.0:
        raise ValueError(
            f"--temp-high/--temp-low must be positive, got "
            f"{args.temp_high!r}/{args.temp_low!r}: a non-positive temperature "
            "is a deterministic argmax wearing a sampling flag",
        )
    if int(args.temp_plies) < 0:
        raise ValueError(f"--temp-plies must be >= 0, got {args.temp_plies!r}")
    if int(args.max_plies) <= 0:
        raise ValueError(f"--max-plies must be positive, got {args.max_plies!r}")
    if int(args.dedup_cache_max) <= 0:
        raise ValueError(
            f"--dedup-cache-max must be positive, got {args.dedup_cache_max!r}",
        )
    if not float(args.sf_read_timeout) > 0.0 or not math.isfinite(
        float(args.sf_read_timeout),
    ):
        raise ValueError(
            f"--sf-read-timeout must be finite and positive, got "
            f"{args.sf_read_timeout!r}: a non-positive deadline expires on the "
            "first read and poisons the engine before it has said anything",
        )
    if not float(args.sf_search_timeout) > 0.0 or not math.isfinite(
        float(args.sf_search_timeout),
    ):
        raise ValueError(
            f"--sf-search-timeout must be finite and positive, got "
            f"{args.sf_search_timeout!r}: a non-positive deadline expires on "
            "the first read of every search and poisons the engine",
        )
    if float(args.sf_search_timeout) > float(args.sf_read_timeout):
        raise ValueError(
            f"--sf-search-timeout {args.sf_search_timeout!r} exceeds "
            f"--sf-read-timeout {args.sf_read_timeout!r}: the search tripwire "
            "must sit INSIDE the outer read deadline, or the stamp would name "
            "a bound the engine never enforces",
        )
    buckets = split_games(int(args.games), int(args.workers))
    out_dir = Path(args.out_dir)
    resume = bool(args.resume)
    # Both refusals happen HERE, before the engine handshake and before a
    # single byte is written: a run that cannot legally touch this directory
    # should find out in the first second, not after the tablebase probe.
    manifest = load_resume_manifest(out_dir) if resume else None
    if not resume:
        refuse_populated_dir(out_dir)

    syzygy_path = str(args.syzygy_path)
    refuse_unopenable_syzygy(syzygy_path)
    sf_binary = str(args.stockfish)
    engine_record = announce_engine("gen_sf_rooted_corpus", sf_binary)
    try:
        engine_id_name: str | None = audit_targets.engine_identity(sf_binary)
    except (OSError, RuntimeError):  # pragma: no cover - engine handshake
        engine_id_name = None

    requested = config_stamp(args, sf_binary=sf_binary)
    config_sha = stamp_sha256(requested)
    if manifest is not None:
        refuse_resume_config_drift(manifest, requested=requested)
        # ⚑ AFTER the last refusal and before the first game: a resume that is
        # going to be turned away must leave BOTH files byte-identical to how
        # it found them, and a resume that is going ahead must free every name
        # this invocation writes with `open("x")` at the end. `None` from both
        # on an ordinary kill -9 resume, which left no summary at all.
        #
        # ⚑ Summary FIRST: it is the record whose verdict was just checked, and
        # archiving it is the step that re-checks it. The aux copy is handled
        # second and unconditionally -- see `archive_json_copy_for_resume` for
        # why an output location the operator named is not corpus identity.
        # Both live here rather than in `main` because only `run` knows when
        # the last refusal has passed, and moving a file before that would be
        # a refused resume that changed the directory anyway.
        archive_unfinished_summary(out_dir)
        archive_json_copy_for_resume(None if args.json is None else Path(args.json))
        # ⚑ THE CORPUS'S STAMP, NOT THIS SESSION'S. Every row banked before the
        # kill carries the manifest's sha, so the rows this session adds carry
        # it too -- a corpus with two stamps on rows made under one
        # configuration is a join nobody can trust, and the equality was just
        # proved field by field.
        requested = dict(manifest["config_requested"])
        config_sha = str(manifest["config_sha256"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        write_launch_manifest(
            out_dir, requested=requested, config_sha=config_sha,
            staircase=staircase, engine_record=engine_record,
            engine_id_name=engine_id_name,
        )
    specs = [
        WorkerSpec(
            worker_id=index,
            game_ids=tuple(ids),
            out_dir=out_dir,
            sf_binary=sf_binary,
            sf_hash_mb=int(args.sf_hash_mb),
            sf_read_timeout_s=float(args.sf_read_timeout),
            sf_search_timeout_s=float(args.sf_search_timeout),
            syzygy_path=syzygy_path,
            staircase=format_staircase(staircase),
            staircase_policy=staircase_policy,
            seed=int(args.seed),
            dedup_cache_max=int(args.dedup_cache_max),
            temp_plies=int(args.temp_plies),
            temp_high=float(args.temp_high),
            temp_low=float(args.temp_low),
            max_plies=int(args.max_plies),
            shard_rows=int(args.shard_rows),
            nice=int(args.nice),
            cp_slope=float(args.cp_slope),
            cp_draw_width=float(args.cp_draw_width),
            book=None if args.book is None else str(args.book),
            book_plies=int(args.book_plies),
            book_max_games=int(args.book_max_games),
            run_id=str(args.run_id),
            config_sha256=config_sha,
            resume=resume,
        )
        for index, ids in enumerate(buckets)
    ]

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    results: list[dict[str, Any]]
    if len(specs) == 1:
        # In process for a single worker: a pool adds a spawn, a second torch
        # import and a pickling hop to buy nothing, and the smoke run is the
        # case that most needs a legible traceback.
        results = [run_worker(specs[0])]
    else:
        ctx = multiprocessing.get_context("spawn")
        results = []
        with ProcessPoolExecutor(max_workers=len(specs), mp_context=ctx) as pool:
            futures = {pool.submit(run_worker, spec): spec for spec in specs}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    # ⚑ `run_worker` records its OWN failures and returns; this
                    # is the process dying under it -- an OOM kill, a segfault,
                    # a pickling error on the way home -- where there is no slot
                    # to have recorded anything. Synthesised rather than
                    # re-raised, so one dead process does not take the other
                    # workers' summary with it.
                    _LOG.exception("worker %d process died", spec.worker_id)
                    results.append(failed_worker_slot(
                        spec,
                        worker_failure(
                            exc, progress=WorkerProgress(), games_completed=0,
                        ),
                    ))
    results.sort(key=lambda r: int(r["worker_id"]))
    wall_s = time.perf_counter() - started

    summary = build_summary(
        results=results, requested=requested, config_sha=config_sha,
        engine_record=engine_record, engine_id_name=engine_id_name,
        staircase=staircase, started_utc=started_utc, wall_s=wall_s,
    )
    with open(out_dir / SUMMARY_NAME, "x", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=_json_default)
        fh.write("\n")
    return summary


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialise {type(value).__name__} into the summary")


# -- CLI ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--out-dir", type=Path, required=True,
                   help="corpus directory; refused if it already holds files "
                        "unless --resume")
    p.add_argument(
        "--resume", action="store_true",
        help="continue the run already banked in --out-dir after a kill -9 "
             "instead of starting a new one. Every worker replays exactly the "
             "games its own w<id>.progress.jsonl does not claim, deletes the "
             "shard the kill caught mid-write, re-warms its dedup cache from "
             "the shards it kept, and continues its shard numbering. REFUSED "
             "unless the directory holds a manifest.json (nothing to resume) "
             "and every generation-affecting setting matches the one it "
             "records -- a corpus may only ever have one configuration. A "
             "summary.json is only a refusal when it says run_finished: true "
             "or states no verdict at all; a crashed session's (run_finished: "
             "false) is kept as summary.unfinished_NN.json and continued, as "
             "is anything at the --json path.",
    )
    p.add_argument(
        "--games", type=int, default=0,
        help="TOTAL games across all workers. Refused at 0: an unbounded "
             "corpus run is a shell loop's job, and a default size would make "
             "the run's cost invisible in the command that produced it.",
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--staircase", default=DEFAULT_STAIRCASE,
        help=f"narrowing rungs as '<width>:<depth>,...' (default "
             f"{DEFAULT_STAIRCASE!r}). Width {WIDTH_ALL!r} means one PV per "
             "legal move. Widths must strictly descend and depths strictly "
             "ascend.",
    )
    p.add_argument(
        "--staircase-policy",
        choices=STAIRCASE_POLICIES,
        default=STAIRCASE_POLICY_FIXED,
        help=(
            f"how the parsed staircase is executed (default "
            f"{STAIRCASE_POLICY_FIXED!r}). {STAIRCASE_POLICY_G10!r} is the "
            f"independently validated {G10_STAIRCASE!r} labeler: after d10, "
            f"run the final 4:d12 rung only when the top-two effective-cp "
            f"margin is <= {G10_MARGIN_CP:g}. It is refused with every other "
            "--staircase rather than generalized silently."
        ),
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--temp-plies", type=int, default=DEFAULT_TEMP_PLIES,
        help="plies 0..N-1 sample at --temp-high; the rest at --temp-low",
    )
    p.add_argument("--temp-high", type=float, default=DEFAULT_TEMP_HIGH)
    p.add_argument("--temp-low", type=float, default=DEFAULT_TEMP_LOW)
    p.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    p.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    p.add_argument(
        "--dedup-cache-max", type=int, default=DEFAULT_DEDUP_CACHE_MAX,
        help=f"positions ONE WORKER's dedup cache holds before it evicts, FIFO "
             f"(default {DEFAULT_DEDUP_CACHE_MAX}, measured at ~816 bytes per "
             "entry = ~1.5 GiB per worker). ⚑ An evicted position that recurs "
             "is RE-SEARCHED and RE-BANKED, so the corpus can hold two rows "
             "with one dedup_key -- the same thing that already happens when "
             "two workers reach one position, and the count is published as "
             "the summary's dedup_cache_evictions. Each run publishes its own "
             "realized bytes per entry; read it before raising this.",
    )
    p.add_argument("--stockfish", type=Path, default=default_stockfish())
    p.add_argument("--sf-hash-mb", type=int, default=DEFAULT_SF_HASH_MB)
    p.add_argument(
        "--sf-read-timeout", type=float, default=DEFAULT_SF_READ_TIMEOUT_S,
        help=f"OUTER deadline in seconds on engine handshakes (boot, ucinewgame, "
             f"SyzygyPath init; default {DEFAULT_SF_READ_TIMEOUT_S}). Searches "
             "are bounded by the tighter --sf-search-timeout. Expiry POISONS "
             "the engine rather than retrying, so size it above a cold "
             "tablebase init's worst case.",
    )
    p.add_argument(
        "--sf-search-timeout", type=float, default=DEFAULT_SF_SEARCH_TIMEOUT_S,
        help=f"deadline in seconds on one WHOLE staircase search (the full go "
             f"exchange; default {DEFAULT_SF_SEARCH_TIMEOUT_S}). The hot-TT "
             "explosion tripwire: expiry poisons the engine and EngineLease "
             "respawns it and retries the position once, cold. Must not "
             "exceed --sf-read-timeout.",
    )
    p.add_argument(
        "--syzygy-path", default=default_syzygy_path(),
        help="OS-separated tablebase directories, handed to BOTH the engine "
             "and the adjudicator. Default is the production pair.",
    )
    p.add_argument("--nice", type=int, default=DEFAULT_NICE)
    p.add_argument("--cp-slope", type=float, default=gen.NNUE_CP_SLOPE)
    p.add_argument("--cp-draw-width", type=float, default=gen.NNUE_CP_DRAW_WIDTH)
    p.add_argument(
        "--book", type=Path, default=None,
        help="PGN/PGN.zip/Polyglot opening book, sampled through the "
             "PRODUCTION opening sampler. Default: the start position.",
    )
    p.add_argument("--book-plies", type=int, default=gen.DEFAULT_OPENING_PLIES)
    p.add_argument(
        "--book-max-games", type=int, default=gen.DEFAULT_OPENING_MAX_GAMES,
    )
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument(
        "--json", type=Path, default=None,
        help="also write the summary here (it always lands in --out-dir). "
             "Written with an exclusive create, so an existing file is never "
             "clobbered; on --resume anything already at this path is moved "
             "to <stem>.unfinished_NN<suffix> first, because a name this "
             "invocation cannot write is a multi-day run that fails after the "
             "corpus is already banked.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(format_summary(summary))
    if args.json is not None:
        with open(Path(args.json), "x", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True, default=_json_default)
            fh.write("\n")
    # ⚑ The summary is WRITTEN either way -- a partial corpus with a stamp beats
    # no corpus -- but the exit code is the only thing a shell loop reads, and a
    # run that lost a worker did not do what it was asked to.
    return 1 if summary["failed_workers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
