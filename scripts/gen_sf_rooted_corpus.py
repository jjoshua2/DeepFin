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

⚑ THE TRANSPOSITION TABLE IS CARRIED ACROSS THE PHASES ON PURPOSE, and that is
disclosed rather than assumed.  Scout-then-narrow is the DEPLOYABLE scheme: the
deep phases are cheap precisely because the shallow one warmed the table, and a
generator that cleared the hash between phases would measure a scheme nobody
would ship.  Every row and the summary carry ``tt_carried_across_phases: true``
so a consumer cannot mistake these for independent searches.  The table IS
cleared (``ucinewgame``) at the start of every GAME, so one game's tree cannot
leak into the next -- ``tt_cleared_per_game`` says so.

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
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import numpy as np

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
ROW_SCHEMA = 1

#: Summary schema, versioned separately: the summary gains aggregate keys far
#: more often than a row changes shape, and bumping the row schema for a new
#: histogram would invalidate every banked row for nothing.
SUMMARY_SCHEMA = 1

#: Launch manifest, written BEFORE the first game (see write_launch_manifest).
MANIFEST_SCHEMA = 1
MANIFEST_NAME = "manifest.json"

DEFAULT_STAIRCASE = "all:9,16:11,4:13"
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
class PositionSearch:
    """Everything one position's staircase produced."""

    phases: tuple[PhaseResult, ...]
    #: The value vector move selection runs on: the deepest FULL-WIDTH phase-1
    #: block, in rank order, from the ROOT MOVER's seat (a rooted MultiPV list
    #: already reports each root move from the mover's POV, so nothing is
    #: negated here -- the gate's `RootedStockfishArm` documents the same trap).
    values: tuple[PvLine, ...]
    value_depth: int
    value_full_width: bool


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
            "nodes_by_phase": {
                str(idx): samples.summary()
                for idx, samples in sorted(self.nodes_by_phase.items())
            },
        }


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
    ) -> None:
        self.engine = engine
        self.staircase = tuple(staircase)
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
        fen: str,
        *,
        depth: int,
        multipv: int,
        searchmoves: Sequence[str] | None = None,
    ) -> list[str]:
        """Drive one ``go depth`` and return every line before ``bestmove``."""
        tokens = _validated_searchmoves(fen, searchmoves) if searchmoves else []
        with self.engine._lock, self.engine._protocol_section():
            if int(multipv) != self._engine_multipv:
                self.engine._send(f"setoption name MultiPV value {int(multipv)}")
                self.engine._send("isready")
                self.engine._wait_for("readyok")
                self._engine_multipv = int(multipv)
            self.engine._send(f"position fen {fen}")
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
        fen = board.fen()
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
        for index, phase in enumerate(self.staircase):
            width = phase.width_for(len(candidates))
            # Phase 0 searches the full move list, so it names no `searchmoves`
            # and its `go` line is byte-identical to an unrestricted rooted
            # search.  Later phases always name theirs.
            restrict = None if index == 0 else tuple(candidates[:width])
            lines = self.stream(
                fen, depth=phase.depth, multipv=width, searchmoves=restrict,
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
            values=value_block.lines,
            value_depth=value_block.depth,
            value_full_width=full_width,
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

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> SelectionValues | None:
        """Serve a position, WITHOUT touching the eviction order (FIFO)."""
        return self._entries.get(key)

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
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"--out-dir {out_dir} already holds files; a corpus run refuses to "
            "merge into an existing directory (its rows would carry two "
            "configuration stamps). Choose a new directory.",
        )


class ShardWriter:
    """Rotating per-worker JSONL shards, every file opened ``"x"``.

    zstd when ``zstandard`` imports, gzip otherwise; the codec that was actually
    used is READ BACK off the writer for the summary rather than assumed from
    the import.
    """

    def __init__(
        self, *, out_dir: Path, worker_id: int, shard_rows: int,
    ) -> None:
        if int(shard_rows) <= 0:
            raise ValueError(
                f"--shard-rows must be positive, got {shard_rows!r}; a "
                "non-positive rotation would open one file per row",
            )
        self.out_dir = out_dir
        self.worker_id = int(worker_id)
        self.shard_rows = int(shard_rows)
        module = zstandard_module()
        self._zstd = module
        self.codec = "zstd" if module is not None else "gzip"
        self.suffix = ".jsonl.zst" if module is not None else ".jsonl.gz"
        self.shards: list[dict[str, Any]] = []
        self._index = 0
        self._rows = 0
        self._text: Any = None
        self._raw: Any = None
        self._binary: Any = None

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
        handle = self._text if self._text is not None else self._open()
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._rows += 1
        if self._rows >= self.shard_rows:
            self.close()

    def close(self) -> None:
        if self._text is None:
            return
        self._text.close()
        if self._raw is not None:
            self._raw.close()
        if self._binary is not None:
            self._binary.close()
        self.shards.append({
            "path": str(self.path_for(self._index)),
            "rows": self._rows,
            "codec": self.codec,
        })
        # One line per CLOSED shard, appended as the shard closes, so a run
        # that dies on day 13 still names every complete shard it wrote --
        # summary.json is written once at run END and a crash takes it with
        # it. Each worker owns its own progress file, so "a" cannot interleave
        # across processes. Deliberately NOT suppressed: a metadata write that
        # fails (disk full) should kill the worker loudly, not leave a corpus
        # whose inventory quietly stopped growing.
        progress = self.out_dir / f"w{self.worker_id:02d}.progress.jsonl"
        with open(progress, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.shards[-1], sort_keys=True) + "\n")
        self._text = None
        self._raw = None
        self._binary = None
        self._index += 1


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


@dataclass
class DedupStats:
    """First-seen and cache-served counts, split by phase of game."""

    first_seen: Counter[str] = field(default_factory=Counter)
    hits: Counter[str] = field(default_factory=Counter)

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
        phase = game_phase(ply=ply, piece_count=piece_count)
        cached = cache.get(key)
        if cached is not None:
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
            dedup.first_seen[phase] += 1
            # ⚑ Selection reads the COMPACT object on the first visit too, so a
            # cache-served ply and a first-seen ply cannot disagree about q.
            values = SelectionValues.from_lines(search.values)
            cache.put(key, values)

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
                },
                "fen": board.fen(),
                "dedup_key": key,
                "worker_id": spec.worker_id,
                "game_id": game_id,
                "ply": ply,
                "stm": "w" if board.turn == chess.WHITE else "b",
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
                "phases": [p.as_row() for p in search.phases],
                # Backfilled below, at game end.
                "result": None,
                "result_pgn": None,
                "adjudication": None,
            })

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
        "adjudication_unavailable_plies": 0,
        "dedup": {
            **DedupStats().summary(),
            **DedupCache(max_entries=int(spec.dedup_cache_max)).summary(),
        },
        "search": SearchStats().summary(),
        "shards": [],
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
    staircase = parse_staircase(spec.staircase)

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
        )

    searcher = EngineLease(spawn_searcher)
    opening_cfg = build_opening_config(spec)
    writer = ShardWriter(
        out_dir=spec.out_dir, worker_id=spec.worker_id, shard_rows=spec.shard_rows,
    )
    cache = DedupCache(max_entries=int(spec.dedup_cache_max))
    dedup = DedupStats()
    progress = WorkerProgress()
    terminations: Counter[str] = Counter()
    adjudications: Counter[str] = Counter()
    opening_sources: Counter[str] = Counter()
    plies: list[int] = []
    rows_written = 0
    games = 0
    games_started = 0
    unavailable = 0
    failure: dict[str, Any] | None = None
    started = time.perf_counter()
    try:
        for game_id in spec.game_ids:
            games_started += 1
            outcome = play_game(
                spec=spec, searcher=searcher, opening_cfg=opening_cfg,
                game_id=game_id, cache=cache, dedup=dedup, progress=progress,
            )
            for row in outcome.rows:
                writer.write(row)
            rows_written += len(outcome.rows)
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
        "adjudication_unavailable_plies": unavailable,
        "dedup": {**dedup.summary(), **cache.summary()},
        "search": searcher.stats.summary(),
        "shards": writer.shards,
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
    rows = sum(int(r["rows"]) for r in results)
    games = sum(int(r["games"]) for r in results)
    search = merge_search(results)
    return {
        "schema": SUMMARY_SCHEMA,
        "row_schema": ROW_SCHEMA,
        "run_id": requested["run_id"],
        "started_utc": started_utc,
        "wall_s": wall_s,
        "rows": rows,
        "games": games,
        "plies_total": sum(int(r["plies_total"]) for r in results),
        "rows_per_game": (rows / games) if games else math.nan,
        "s_per_row": (wall_s / rows) if rows else math.nan,
        # ⚑ TOP LEVEL, and empty on a healthy run.  A partial corpus that reads
        # like a complete one is the whole hazard here: every other number in
        # this file is a sum over the workers that reported, and a reader has no
        # way to know one of them stopped early unless the summary says so.
        "failed_workers": [
            {"worker_id": r["worker_id"], **r["failed"]}
            for r in results if r["failed"] is not None
        ],
        "dedup": merge_dedup(results),
        "search": search,
        "terminations": merge_counters(results, "terminations"),
        "adjudications": merge_counters(results, "adjudications"),
        "adjudication_unavailable_plies": sum(
            int(r["adjudication_unavailable_plies"]) for r in results
        ),
        "opening_sources": merge_counters(results, "opening_sources"),
        "shards": [shard for r in results for shard in r["shards"]],
        "config_requested": requested,
        "config_sha256": config_sha,
        "staircase_parsed": [
            {"width": p.width_label, "depth": p.depth} for p in staircase
        ],
        # ⚑ REALIZED, one entry per worker, every field read off the live engine
        # (or off this process) rather than echoed from the flags -- see
        # `StaircaseSearcher.realized`.
        "config_realized_by_worker": {
            str(r["worker_id"]): r["realized"] for r in results
        },
        "engine": {**engine_record, "id_name": engine_id_name},
        "banked_rows_min_piece_count": MIN_BANKED_PIECES,
        "adjudication_max_piece_count": ADJUDICATION_MAX_PIECES,
        "python": sys.version.split()[0],
    }


def format_summary(summary: dict[str, Any]) -> str:
    dedup = summary["dedup"]
    search = summary["search"]
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
        f"anomalies={search['anomalies']}",
        f"terminations={summary['terminations']}",
    ]
    lines.extend(
        f"FAILED worker {failed['worker_id']}: {failed['exception_type']}: "
        f"{failed['exception']} (game {failed['last_game_id']} "
        f"ply {failed['last_ply']}, {failed['games_completed']} games done)"
        for failed in summary["failed_workers"]
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
    mistake this for the completion record: ``summary.json`` remains the only
    document that says the run FINISHED, and a partial read has to say it is
    one.
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
        "engine": {**engine_record, "id_name": engine_id_name},
        "banked_rows_min_piece_count": MIN_BANKED_PIECES,
        "adjudication_max_piece_count": ADJUDICATION_MAX_PIECES,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / MANIFEST_NAME, "x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=_json_default)
        fh.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate, fan out, merge and write ``summary.json``."""
    staircase = parse_staircase(str(args.staircase))
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
    out_dir.mkdir(parents=True, exist_ok=True)
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
    with open(out_dir / "summary.json", "x", encoding="utf-8") as fh:
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
                   help="corpus directory; refused if it already holds files")
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
    p.add_argument("--json", type=Path, default=None,
                   help="also write the summary here (it always lands in --out-dir)")
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
