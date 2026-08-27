#!/usr/bin/env python3
"""Turn an NNUE-bootstrap corpus bank into training shards the control rig reads.

``scripts/gen_sf_rooted_corpus.py`` banks the LOWEST-LEVEL observations a
Stockfish staircase emitted -- every phase, every depth, every MultiPV rank,
with the cumulative node count on each line.  It does not bank a training
target, on purpose: the ledger's FREEZE-THE-OBSERVATIONS rule exists so that
choosing a target is a re-read of a corpus rather than a rerun of the search.

This is the re-read.  It maps each banked row's ``(move -> per-depth value)``
bank to ONE value per legal move under a named SCHEME, turns those values into a
policy and a value target, and writes ``shard_NNNNNN.zarr`` replay shards that
``scripts/lc0_control_train.py --shards`` consumes unchanged.

Usage::

    PYTHONPATH=. python3 scripts/derive_corpus_targets.py \\
        --corpus data/nnue_bootstrap/run01 \\
        --out data/nnue_rows/run01-uniform-d9 \\
        --scheme uniform-d9 --temp 1.0

THE SCHEMES
-----------
``uniform-d<D>``
    Every move's value at depth ``D``.  Each move is read from the DEEPEST PHASE
    that carries that move at that depth -- a narrowed phase's depth-``D`` block
    is the same iteration seen with a warmer transposition table, so it is the
    better-informed observation of the same cell.  ⚑ That means one row's values
    can come from several phases, and ``values_by_phase`` in the summary says
    how often; a ``uniform-d9`` corpus is therefore NOT "what a bare
    ``go depth 9`` would have printed".

``top<K>-d<D2>-rest-d<D1>``
    Rank every move by its ``D1`` value, read the top ``K`` at ``D2`` and leave
    the rest at ``D1``.  This is the OFFLINE HALVING SIMULATION: it prices a
    narrowing policy the generator did not run, against blocks it already
    banked.  ⚑ A move in the top ``K`` by ``D1`` that the generator's own
    narrowing did NOT carry to ``D2`` has no banked value there; that row is an
    envelope miss (see below) rather than a row quietly read at ``D1``.

``nodes-<N>``
    The deepest COMPLETE phase-0 depth whose banked cumulative node count is
    ``<= N``, per row.  ⚑ PHASE 0 ONLY, for both the depth choice and the
    values, and that is the one place this file deliberately differs from the
    two schemes above: the claim being reconstructed is "what a full-width
    ``go nodes N`` would have produced", and phase 1+ node counts are neither
    full width nor measured from a cold table.  Splicing a depth-13 narrowed
    reading into a budget arm would make its own headline false.
    A row whose SHALLOWEST complete banked depth already exceeds ``N`` is read
    at that shallowest depth and COUNTED as ``nodes_floor_hits`` -- the budget
    was not honoured for that row and the corpus says so, rather than the row
    silently pretending it was.

THE TARGETS
-----------
POLICY -- ``softmax(q / temp)`` over the scheme's values, placed on the compact
``lc0_1858`` slots ``compact_index_for_move`` assigns and zero everywhere else.
``q`` comes from ``audit_label_candidates.q_from_effective_cp``, which reaches
``gen_random_selfplay_shards.cp_to_wdl_array`` as a module attribute AT CALL
TIME -- the same one function object the generator's own move selection and the
label gate's arms use.  ``tests/test_derive_corpus_targets.py`` proves it by
replacing that single object and watching this file's targets move.  Because the
generator selects with ``argmax(q/tau + Gumbel)``, ``--temp tau`` reproduces
exactly the distribution its own play was sampled from.

VALUE -- the construction is ``data/lc0_rows``'s, mirrored:

* ``wdl_target`` = the row's EXACT game result, already stored from that row's
  own side-to-move seat by ``result_from_pov``.  0=W / 1=D / 2=L.
* ``search_wdl`` = ``cp_to_wdl_array`` of the scheme's BEST-MOVE value, i.e. the
  searched root value of the position, side-to-move POV.
* ``sf_wdl`` is ABSENT.

⚑⚑ AND THE SEARCHED VALUE GOES IN ``search_wdl`` EVEN THOUGH IT IS A STOCKFISH
EVAL.  That reads backwards until you check what the consumer does with it:
``lc0_control_train.py``'s launch guard 1 calls
``assert_pid_cannot_reassert_sf_wdl``, which refuses ANY config with
``sf_wdl_frac > 0`` -- unconditionally, with no reference to whether the shards
carry an SF label.  A value written to ``sf_wdl`` therefore could not reach a
loss on this rig under any config it will start: accepted at write time, ignored
at train time, this repo's signature defect manufactured by its own tooling.
``search_wdl`` is the channel the rig can actually weight
(``search_wdl_frac``), and it is the channel ``lc0_data_to_rows`` already uses
for "the search's own root value".  ⚑ The honesty cost is real and is paid in
the manifest: a reader joining these shards with production ones must know that
here ``search_wdl`` is Stockfish's root value and not our MCTS's, so
``value_channels`` in ``<out>/derive_targets_summary.json`` says exactly that.

⚑ NO BLEND IS BAKED INTO A ROW.  The row carries the two components; the mixing
weights are the trainer's (``game_frac`` / ``search_wdl_frac``), exactly as on
the lc0 corpus.  ``required_training_overrides`` in the summary names the
combination that passes the rig's own guards, and a test asserts that it does by
running ``run_config_problems`` against the shards this tool actually wrote.

⚑ A ROW WITH NO RESULT IS SKIPPED AND COUNTED (``rows_dropped_no_result``).  The
generator's ply cap outside tablebase range banks ``result: null`` and its own
docstring refuses to call that a draw; ``wdl_target`` is a REQUIRED shard field
with no has-flag, so there is no way to emit such a row without inventing an
outcome for it.  The count is in the summary because dropping them shifts the
corpus's position mix and a consumer has to be able to see by how much.

⚑⚑ ZERO HISTORY, AND IT IS MEASURED RATHER THAN ASSERTED
--------------------------------------------------------
A corpus row is a FEN.  It carries ``game_id`` and ``ply``, but rows are banked
only above ``MIN_BANKED_PIECES`` and only on a dedup MISS, so the plies of one
game are not contiguous and the move stack cannot be rebuilt from the corpus
alone.  ``encode_position`` on a ``chess.Board(fen)`` therefore fills history
slot 0 and leaves slots 1..7 -- planes 13..103 -- ZERO, and every repetition
plane with them.  That is the same blindness the frozen rulers score under, and
it is a real difference from both production selfplay rows and the lc0 corpus,
whose 8 frames are all real.  The summary stamps
``history_slots_nonzero_max``, measured off the planes this run actually wrote,
so the claim is a reading and not a comment.

WHAT IS SHARED RATHER THAN RESTATED
-----------------------------------
The corpus schema, the populated-directory refusal and the codec probe are
``gen_sf_rooted_corpus``'s, imported.  The cp->value mapping is the label gate's
(above).  The move -> compact-1858 map is ``moves.leela_index``'s, the same one
``lc0_data_to_rows`` uses.  The shard writer is ``replay.shard``'s, so these
files are byte-compatible with the rig's existing corpus by construction rather
than by imitation.  ⚑ No torch: the mapping needed here is move -> compact 1858,
which ``moves.leela_index`` provides off module-level tables, and the device
tensors in ``moves/torch_maps.py`` convert compact <-> AZ-4672, which this path
never does.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import numpy as np
import zarr

from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import compact_index_for_move
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    local_shard_path,
    samples_to_arrays,
    save_local_shard_arrays,
)
from scripts import audit_label_candidates as gate
from scripts import gen_random_selfplay_shards as gen
from scripts import gen_sf_rooted_corpus as corpus

#: Derived-shard schema.  Bumped when the MEANING of an emitted column changes,
#: which is a different event from the corpus row schema changing -- a consumer
#: needs both numbers to know what it is holding, so both are stamped.
DERIVE_SCHEMA = 1

#: Pinned, not a flag.  The rig merges these shards into ONE ``DiskReplayBuffer``
#: alongside whatever else is named in ``--shards``, and the buffer hard-fails on
#: mixed encoding identity -- so a per-run knob here would turn a typo into an
#: unmergeable corpus, and a wrong-but-mergeable value into 146 planes fed to a
#: 175-plane net.  Changing these is a code edit, deliberately.
INPUT_HISTORY_ENCODING = "lc0_root_legacy_meta"
INPUT_EXTRA_FEATURES = "v2_threats"

#: The rows are encoded by TODAY's encoder, which applies the repetition-plane
#: fix unconditionally.  Vacuous on a zero-history row (nothing can repeat) and
#: stamped anyway, because it is replay IDENTITY: the same encoding name with a
#: different flag is a different plane set and the buffer refuses to mix them.
HISTORY_REP_FIX = True

#: 8192 rows x 175 planes x 64 squares of float16 is ~184 MB on disk, the same
#: rotation ``lc0_data_to_rows`` uses for the same reason.
DEFAULT_ROWS_PER_SHARD = 8192

DEFAULT_SEED = 20260827

#: ``ShardMeta.run_id`` for every shard this tool writes.  The scheme is NOT
#: folded into it: the scheme, its parameters and this file's schema go into
#: their own zarr attrs (see ``_stamp_shard_attrs``) where a reader can parse
#: them, rather than into a string it would have to take apart.
SHARD_RUN_ID = "sf_rooted_corpus_targets"

SUMMARY_NAME = "derive_targets_summary.json"

#: Which phases a scheme is allowed to read a move's value from.
VALUE_SOURCE_DEEPEST = "deepest_phase_covering"
VALUE_SOURCE_PHASE0 = "phase0_only"

#: ``lc0_root_legacy_meta`` history slots and the piece planes in each.  Used
#: only to MEASURE how many slots the encoder actually filled.
_HISTORY_SLOTS = 8
_PLANES_PER_SLOT = 13
_PIECE_PLANES_PER_SLOT = 12

_SCHEME_UNIFORM = re.compile(r"^uniform-d(\d+)$")
_SCHEME_TOPK = re.compile(r"^top(\d+)-d(\d+)-rest-d(\d+)$")
_SCHEME_NODES = re.compile(r"^nodes-(\d+)$")

_SCHEME_FORMS = (
    "uniform-d<D>",
    "top<K>-d<D2>-rest-d<D1>",
    "nodes-<N>",
)


class CorpusIntegrityError(RuntimeError):
    """The corpus is not what its own summary or row schema says it is."""


class EnvelopeMiss(RuntimeError):
    """ONE row's bank cannot answer the scheme's question.

    Distinct from :class:`CorpusIntegrityError`: the corpus is well formed, this
    row simply does not carry the block the scheme asks for (an aborted search,
    or a narrowing the generator did not run).  Whether that ends the run is the
    operator's call via ``--max-envelope-misses``; it is never silent.
    """


# -- the scheme ---------------------------------------------------------------


@dataclass(frozen=True)
class Scheme:
    """A parsed target scheme.  ``canonical`` is re-derived, never echoed."""

    kind: str
    #: uniform: the depth.  topk: D1, the depth the REST are read at.
    depth: int | None = None
    #: topk only: D2, the depth the top K are read at.
    deep_depth: int | None = None
    top_k: int | None = None
    nodes: int | None = None

    @property
    def canonical(self) -> str:
        """The scheme's spelling, rebuilt from the PARSED fields.

        Stamped instead of ``args.scheme`` so a stamp that agrees with the flag
        is evidence the parse agreed with it too, rather than a copy of the
        string that would match whatever the parser had decided.
        """
        if self.kind == "uniform":
            return f"uniform-d{self.depth}"
        if self.kind == "topk":
            return f"top{self.top_k}-d{self.deep_depth}-rest-d{self.depth}"
        return f"nodes-{self.nodes}"

    @property
    def value_source(self) -> str:
        return VALUE_SOURCE_PHASE0 if self.kind == "nodes" else VALUE_SOURCE_DEEPEST

    def params(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "depth": self.depth,
            "deep_depth": self.deep_depth,
            "top_k": self.top_k,
            "nodes": self.nodes,
            "value_source": self.value_source,
        }


def parse_scheme(spec: str) -> Scheme:
    """``"top4-d13-rest-d9"`` -> a :class:`Scheme`, or ``ValueError``.

    The shape rules are the staircase's own, for the staircase's own reasons:

    * ``D2`` must strictly DEEPEN ``D1``.  ``top4-d9-rest-d9`` reads every move
      at one depth while claiming a two-tier read, and ``top4-d5-rest-d9`` would
      spend the "deep" tier on a shallower block than the rest -- a narrowing
      that makes the top moves the LEAST informed, which no consumer wants and
      no name here would disclose.
    * every depth, ``K`` and ``N`` is positive.  ``depth 0`` is the trap
      ``parse_staircase`` documents (Stockfish silently replaces ``go depth 0``
      with a real iteration), and here it would simply select no block.
    """
    text = spec.strip()
    match = _SCHEME_UNIFORM.match(text)
    if match:
        depth = int(match.group(1))
        _require_positive(text, depth=depth)
        return Scheme(kind="uniform", depth=depth)
    match = _SCHEME_TOPK.match(text)
    if match:
        top_k, deep, base = (int(match.group(i)) for i in (1, 2, 3))
        _require_positive(text, top_k=top_k, deep_depth=deep, depth=base)
        if deep <= base:
            raise ValueError(
                f"--scheme {text!r}: the top-K depth {deep} must strictly deepen "
                f"the rest depth {base}; a scheme whose 'deep' tier is not deeper "
                "is a uniform read wearing a two-tier name",
            )
        return Scheme(kind="topk", depth=base, deep_depth=deep, top_k=top_k)
    match = _SCHEME_NODES.match(text)
    if match:
        nodes = int(match.group(1))
        _require_positive(text, nodes=nodes)
        return Scheme(kind="nodes", nodes=nodes)
    raise ValueError(
        f"--scheme {spec!r} is not one of {', '.join(_SCHEME_FORMS)}",
    )


def _require_positive(spec: str, **values: int) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"--scheme {spec!r}: {name} must be positive, got {value}")


def scheme_vs_staircase_problems(
    scheme: Scheme, staircase: Sequence[dict[str, Any]],
) -> list[str]:
    """Why this corpus's STAIRCASE cannot answer this scheme, before reading rows.

    A run-level refusal on top of the per-row one: ``--scheme uniform-d20``
    against a corpus whose deepest rung is 13 is knowable from ``summary.json``,
    and finding it out one row at a time would spend a full read to say it.
    """
    if not staircase:
        return ["the corpus summary carries no staircase_parsed; nothing to check"]
    full_width_depth = int(staircase[0]["depth"])
    deepest = max(int(rung["depth"]) for rung in staircase)
    problems: list[str] = []
    # The BASE depth must be reachable at FULL WIDTH -- every move needs a value
    # there -- so it is bounded by phase 0's rung, not by the deepest one.
    if scheme.depth is not None and int(scheme.depth) > full_width_depth:
        problems.append(
            f"the scheme's base depth {scheme.depth} exceeds the corpus "
            f"envelope: the staircase's full-width phase reaches depth "
            f"{full_width_depth}",
        )
    # The DEEP depth is only ever asked of a narrowed subset, so the deepest
    # rung is its bound.
    if scheme.deep_depth is not None and int(scheme.deep_depth) > deepest:
        problems.append(
            f"the scheme's deep depth {scheme.deep_depth} exceeds the corpus "
            f"envelope: the staircase's deepest rung reaches depth {deepest}",
        )
    return problems


# -- one row's bank -----------------------------------------------------------


@dataclass(frozen=True)
class MoveValues:
    """One value per move, and the provenance of every one of them.

    ``depth_by_move``/``phase_by_move`` are not diagnostics: a scheme that reads
    two tiers is only doing so if the emitted rows say which tier each move came
    from, and those two arrays are what the summary's histograms are built from.
    """

    moves: tuple[str, ...]
    effective_cp: np.ndarray
    depth_by_move: tuple[int, ...]
    phase_by_move: tuple[int, ...]
    base_depth: int
    floor_hit: bool

    @property
    def best_index(self) -> int:
        """The scheme's best move.  Ties go to the earliest banked rank."""
        return int(np.argmax(self.effective_cp))


class RowBank:
    """Every ``(phase, depth)`` block of ONE corpus row, indexed for reading.

    Built per row and thrown away; the corpus is streamed, so nothing here is
    allowed to grow with the run.
    """

    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row
        # phase index -> depth -> {"complete": bool, "nodes": int|None,
        #                          "values": {move: cp}, "order": (move, ...)}
        self._blocks: list[dict[int, dict[str, Any]]] = []
        for phase in row["phases"]:
            by_depth: dict[int, dict[str, Any]] = {}
            for block in phase["per_depth"]:
                lines = block["lines"]
                by_depth[int(block["depth"])] = {
                    "complete": bool(block["complete"]),
                    "nodes": (
                        None if block["nodes_at_depth"] is None
                        else int(block["nodes_at_depth"])
                    ),
                    "values": {str(line[1]): float(line[2]) for line in lines},
                    "order": tuple(str(line[1]) for line in lines),
                }
            self._blocks.append(by_depth)

    @property
    def phase_count(self) -> int:
        return len(self._blocks)

    def full_width_block(self, depth: int) -> dict[str, Any] | None:
        """Phase 0's COMPLETE block at ``depth``, or None.

        Phase 0 is the only rung searched at ``MultiPV = legal move count`` with
        no ``searchmoves``, so it is the only one whose block defines "every
        move".  ``complete`` is required: an aborted iteration lists a subset of
        the ranks, and reading a subset as the move set would silently drop
        moves out of the policy's support with a legal mask that still names
        them.
        """
        if not self._blocks:
            return None
        block = self._blocks[0].get(int(depth))
        if block is None or not block["complete"]:
            return None
        return block

    def full_width_depths(self) -> list[int]:
        """The depths phase 0 banked a COMPLETE block at, ascending."""
        if not self._blocks:
            return []
        return sorted(d for d, b in self._blocks[0].items() if b["complete"])

    def value_at(self, move: str, depth: int) -> tuple[float, int] | None:
        """``(effective_cp, phase_index)`` from the DEEPEST phase carrying it."""
        for index in range(len(self._blocks) - 1, -1, -1):
            block = self._blocks[index].get(int(depth))
            if block is None:
                continue
            value = block["values"].get(move)
            if value is not None:
                return float(value), index
        return None

    def node_ladder(self) -> list[tuple[int, int]]:
        """Phase 0's ``(depth, cumulative nodes)`` rungs, ascending by depth.

        Only COMPLETE blocks that reported a node count take part: a block with
        no ``nodes_at_depth`` cannot be priced, and an incomplete one is not the
        iteration a ``go nodes`` budget would have bought.
        """
        if not self._blocks:
            return []
        return [
            (depth, int(block["nodes"]))
            for depth, block in sorted(self._blocks[0].items())
            if block["complete"] and block["nodes"] is not None
        ]

    def nodes_depth(self, budget: int) -> tuple[int, bool]:
        """``(depth, floor_hit)`` for a ``go nodes budget`` reconstruction."""
        ladder = self.node_ladder()
        if not ladder:
            raise EnvelopeMiss(
                "no complete phase-0 block carries a node count, so no node "
                "budget can be priced against this row",
            )
        affordable = [depth for depth, nodes in ladder if nodes <= int(budget)]
        if affordable:
            return max(affordable), False
        # ⚑ The FLOOR, and it is an event rather than a default: the budget did
        # not buy even the shallowest banked iteration, so the row is read at
        # that iteration and the run counts how often it had to.
        return ladder[0][0], True


def _required(value: int | None, name: str) -> int:
    """A scheme field the parser guarantees for this kind.

    ``ValueError`` rather than ``assert``: an assert is removed under ``-O`` and
    the failure would then be an index into ``None``.
    """
    if value is None:  # pragma: no cover - parse_scheme fills these per kind
        raise ValueError(f"scheme is missing {name}")
    return int(value)


def apply_scheme(bank: RowBank, scheme: Scheme) -> MoveValues:
    """Collapse one row's bank to one value per move under ``scheme``."""
    if scheme.kind == "nodes":
        depth, floor_hit = bank.nodes_depth(_required(scheme.nodes, "nodes"))
        block = bank.full_width_block(depth)
        if block is None:  # pragma: no cover - node_ladder only lists complete blocks
            raise EnvelopeMiss(f"phase 0 has no complete block at depth {depth}")
        moves = block["order"]
        return MoveValues(
            moves=moves,
            effective_cp=np.array(
                [block["values"][move] for move in moves], dtype=np.float64,
            ),
            depth_by_move=(depth,) * len(moves),
            phase_by_move=(0,) * len(moves),
            base_depth=depth,
            floor_hit=floor_hit,
        )

    base = _required(scheme.depth, "depth")
    block = bank.full_width_block(base)
    if block is None:
        raise EnvelopeMiss(
            f"phase 0 has no complete block at depth {base}; the row's "
            f"full-width envelope is {bank.full_width_depths()}",
        )
    moves = block["order"]
    values: list[float] = []
    phases: list[int] = []
    for move in moves:
        read = bank.value_at(move, base)
        if read is None:  # pragma: no cover - phase 0 carries every move at base
            raise EnvelopeMiss(f"no banked value for {move} at depth {base}")
        values.append(read[0])
        phases.append(read[1])
    depths = [base] * len(moves)

    if scheme.kind == "topk":
        top_k = _required(scheme.top_k, "top_k")
        deep = _required(scheme.deep_depth, "deep_depth")
        # ⚑ The rank is by the D1 value with the uci as the tiebreak, so the
        # top-K set is a function of the BANK and not of iteration order. Using
        # Stockfish's own D1 rank as the tiebreak would tie this scheme to the
        # generator's narrowing, which is the very thing it exists to vary.
        order = sorted(range(len(moves)), key=lambda i: (-values[i], moves[i]))
        for index in order[:top_k]:
            read = bank.value_at(moves[index], deep)
            if read is None:
                raise EnvelopeMiss(
                    f"{moves[index]} is in the top {top_k} by depth "
                    f"{base} but no phase banked it at depth {deep}; the "
                    "generator's own narrowing did not carry it that far",
                )
            values[index], phases[index] = read[0], read[1]
            depths[index] = deep

    return MoveValues(
        moves=moves,
        effective_cp=np.asarray(values, dtype=np.float64),
        depth_by_move=tuple(depths),
        phase_by_move=tuple(phases),
        base_depth=base,
        floor_hit=False,
    )


# -- targets ------------------------------------------------------------------


def validate_temp(temp: float) -> float:
    """⚑ A non-positive or non-finite temperature is REFUSED, not taken to its limit.

    Exactly as ``gen_sf_rooted_corpus.gumbel_choice`` refuses it: the limit is a
    one-hot target, and silently emitting one would turn a decimal typo into a
    corpus whose policy carries no distributional information at all and no
    column that says so.  Called from ``main`` as well as from the softmax, so a
    bad ``--temp`` is refused before the first shard is read rather than on the
    first row that happens to reach the target builder.
    """
    tau = float(temp)
    if not tau > 0.0 or not math.isfinite(tau):
        raise ValueError(
            f"--temp must be finite and positive, got {temp!r}: the limit is a "
            "one-hot target, which is a different experiment",
        )
    return tau


def softmax_at_temp(q: np.ndarray, *, temp: float) -> np.ndarray:
    """``softmax(q / temp)`` in float64, max-shifted."""
    tau = validate_temp(temp)
    scaled = np.asarray(q, dtype=np.float64) / tau
    shifted = np.exp(scaled - float(np.max(scaled)))
    return shifted / float(shifted.sum())


def recover_temp(q: np.ndarray, probs: np.ndarray) -> float | None:
    """The temperature READ BACK off an emitted policy row, or None.

    ⚑ This is the realized stamp for ``--temp``, and it is realized because it
    is computed from the OUTPUT: for any two moves,
    ``log p_i - log p_j == (q_i - q_j) / tau``, so the emitted distribution and
    the values it came from determine tau with nothing echoed from the flag.
    A knob that was parsed, stored and then not applied cannot survive it.
    ⚑ It reads the DERIVED float64 distribution, before the shard's float16
    cast, so it certifies the computation rather than the storage; what the
    cast costs is reported separately as ``policy_support_lost_to_float16``.
    Returns None when every value is equal (the distribution is uniform at every
    temperature, so the row carries no information about tau).
    """
    values = np.asarray(q, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    usable = p > 0.0
    if int(usable.sum()) < 2:
        return None
    values, p = values[usable], p[usable]
    hi, lo = int(np.argmax(values)), int(np.argmin(values))
    gap = float(values[hi] - values[lo])
    if gap <= 0.0:
        return None
    log_gap = float(np.log(p[hi]) - np.log(p[lo]))
    if log_gap <= 0.0:
        return None
    return gap / log_gap


#: ``result_from_pov``'s ``+1 / 0 / -1`` (the ROW's own side-to-move seat) as the
#: shard's ``wdl_target``.  ⚑ Both halves of this mapping are POV claims and
#: neither is checkable from the number alone, which is why the corpus stores the
#: result already rotated and this table is the only other place a sign appears.
_RESULT_TO_WDL: dict[float, int] = {1.0: 0, 0.0: 1, -1.0: 2}


def wdl_target_from_result(result: float) -> int:
    """0=W / 1=D / 2=L from the row's own-POV game result."""
    key = float(result)
    if key not in _RESULT_TO_WDL:
        raise CorpusIntegrityError(
            f"row result {result!r} is not one of +1.0/0.0/-1.0; "
            "result_from_pov emits nothing else and a fourth value would mean "
            "the corpus was written by something other than the generator",
        )
    return _RESULT_TO_WDL[key]


def history_slots_filled(planes: np.ndarray) -> int:
    """How many of the 8 history slots carry a piece.  MEASURED, per row.

    The zero-history claim in this module's docstring is a claim about the
    ENCODER's behaviour on a stackless board, and the cheapest way to keep it
    honest is to read it off the planes that were written.
    """
    filled = 0
    for slot in range(_HISTORY_SLOTS):
        start = slot * _PLANES_PER_SLOT
        if bool(np.any(planes[start : start + _PIECE_PLANES_PER_SLOT])):
            filled += 1
    return filled


# -- reading the corpus -------------------------------------------------------


def corpus_shard_paths(corpus_dir: Path) -> list[Path]:
    """The corpus's JSONL shards, found on DISK and sorted by name."""
    return [
        path
        for path in sorted(corpus_dir.iterdir())
        if path.name.endswith((".jsonl.zst", ".jsonl.gz"))
    ]


def check_shard_inventory(on_disk: Sequence[Path], summary: dict[str, Any]) -> None:
    """The shards present must be exactly the ones ``summary.json`` names.

    ⚑ BY BASENAME, because ``summary["shards"]`` stores the paths of the machine
    that produced the corpus and a corpus is routinely read from somewhere else.
    Comparing sets rather than counts is what catches the case this exists for:
    a partially copied corpus, whose missing shard would otherwise be a smaller
    training set that nothing named.
    """
    named = {Path(str(entry["path"])).name for entry in summary.get("shards", [])}
    if not named:
        # Refused, not skipped (review finding 2): the generator always writes
        # a real inventory, so an empty or missing `shards` list means a
        # damaged or foreign summary — and a lenient pass here is a gate that
        # silently does not fire, this codebase's signature defect.
        raise CorpusIntegrityError(
            "summary.json names no shards at all; the generator always records "
            "its inventory, so there is nothing to check the disk against and "
            "proceeding would train on whatever happens to be in the directory",
        )
    found = {path.name for path in on_disk}
    if named != found:
        missing = sorted(named - found)
        extra = sorted(found - named)
        raise CorpusIntegrityError(
            "the shards on disk are not the ones summary.json names: "
            f"missing {missing or 'none'}, unexpected {extra or 'none'}. A "
            "partially copied corpus would train on a subset nothing recorded.",
        )


def iter_corpus_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Stream one corpus shard's rows, zstd or gzip."""
    if path.name.endswith(".jsonl.zst"):
        module = corpus.zstandard_module()
        if module is None:
            raise CorpusIntegrityError(
                f"{path.name} is zstd-compressed but the zstandard module is not "
                "importable in this environment",
            )
        with open(path, "rb") as binary:
            reader = module.ZstdDecompressor().stream_reader(binary)
            for line in _text_lines(reader):
                yield json.loads(line)
        return
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _text_lines(reader: Any) -> Iterator[str]:
    with io.TextIOWrapper(reader, encoding="utf-8") as text:
        for line in text:
            if line.strip():
                yield line


# -- the run ------------------------------------------------------------------


@dataclass
class DeriveStats:
    """Every counter the summary reports.  All of them are events, not rates."""

    rows_read: int = 0
    rows_written: int = 0
    rows_dropped_no_result: int = 0
    rows_dropped_envelope: int = 0
    envelope_miss_examples: list[str] = field(default_factory=list)
    nodes_floor_hits: int = 0
    support_checks: int = 0
    depth_histogram: dict[int, int] = field(default_factory=dict)
    values_by_phase: dict[int, int] = field(default_factory=dict)
    deep_tier_moves: int = 0
    base_tier_moves: int = 0
    #: How many staircase rungs the ROWS carried, which is not necessarily how
    #: many ``staircase_parsed`` declares -- and ``values_by_phase`` is
    #: uninterpretable without it (a corpus of one-phase rows reads
    #: ``{"0": n}`` whether or not the scheme could ever have looked deeper).
    phases_per_row: dict[int, int] = field(default_factory=dict)
    history_slots_nonzero_max: int = 0
    repetition_planes_nonzero_rows: int = 0
    temp_recovered_n: int = 0
    temp_recovered_min: float = math.inf
    temp_recovered_max: float = -math.inf
    temp_recovered_sum: float = 0.0
    x_planes: int = 0
    policy_width: int = 0
    #: -1 until the first row is measured; a 0 sentinel would be a legal support.
    policy_support_min: int = -1
    policy_support_max: int = 0
    #: Legal moves whose probability survived the derivation but not the shard's
    #: float16 cast.  ⚑ Measured on the far side of the cast on purpose: a
    #: support counted in float64 is a number about a row nothing will ever read.
    policy_support_lost_to_float16: int = 0

    def note_temp(self, value: float) -> None:
        self.temp_recovered_n += 1
        self.temp_recovered_sum += value
        self.temp_recovered_min = min(self.temp_recovered_min, value)
        self.temp_recovered_max = max(self.temp_recovered_max, value)

    def summary(self) -> dict[str, Any]:
        mean = (
            self.temp_recovered_sum / self.temp_recovered_n
            if self.temp_recovered_n else math.nan
        )
        return {
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "rows_dropped_no_result": self.rows_dropped_no_result,
            "rows_dropped_envelope": self.rows_dropped_envelope,
            "envelope_miss_examples": list(self.envelope_miss_examples),
            "nodes_floor_hits": self.nodes_floor_hits,
            "realized_base_depth_histogram": {
                str(k): v for k, v in sorted(self.depth_histogram.items())
            },
            # ⚑ WHICH PHASE EACH VALUE CAME FROM. A `uniform-dD` corpus whose
            # values are all phase 0 and one whose top moves came from a
            # narrowed rung are different corpora with the same scheme name.
            "values_by_phase": {
                str(k): v for k, v in sorted(self.values_by_phase.items())
            },
            "deep_tier_moves": self.deep_tier_moves,
            "base_tier_moves": self.base_tier_moves,
            "phases_per_row": {
                str(k): v for k, v in sorted(self.phases_per_row.items())
            },
            "history_slots_nonzero_max": self.history_slots_nonzero_max,
            "repetition_planes_nonzero_rows": self.repetition_planes_nonzero_rows,
            "temp_recovered_from_emitted_policy": {
                "n": self.temp_recovered_n,
                "min": (
                    self.temp_recovered_min if self.temp_recovered_n else math.nan
                ),
                "max": (
                    self.temp_recovered_max if self.temp_recovered_n else math.nan
                ),
                "mean": mean,
            },
            "x_planes": self.x_planes,
            "policy_width": self.policy_width,
            "policy_support_min": self.policy_support_min,
            "policy_support_max": self.policy_support_max,
            "policy_support_lost_to_float16": self.policy_support_lost_to_float16,
            "support_checks": self.support_checks,
        }


@dataclass(frozen=True)
class DeriveOptions:
    scheme: Scheme
    temp: float
    cp_slope: float
    cp_draw_width: float
    limit: int
    seed: int
    rows_per_shard: int
    max_envelope_misses: int


class TargetDeriver:
    """Turns corpus rows into :class:`ReplaySample` rows under one scheme."""

    def __init__(self, options: DeriveOptions) -> None:
        self.options = options
        self.stats = DeriveStats()

    # -- the shared mapping ----------------------------------------------

    def q_of(self, effective_cp: np.ndarray) -> np.ndarray:
        """Effective cp -> q in [-1, 1], through the SHARED map.

        ``gate.q_from_effective_cp`` resolves ``gen.cp_to_wdl_array`` as a
        module attribute at call time, which is what makes this file, the
        generator's move selection and the label gate's arms ONE function
        object rather than three copies that agree today.
        """
        return gate.q_from_effective_cp(
            np.asarray(effective_cp, dtype=np.float64),
            slope=self.options.cp_slope,
            draw_width_cp=self.options.cp_draw_width,
        )

    def wdl_of(self, effective_cp: float) -> np.ndarray:
        """One effective cp -> (W, D, L), through the SAME object as ``q_of``."""
        wdl = gen.cp_to_wdl_array(
            np.asarray([float(effective_cp)], dtype=np.float64),
            slope=self.options.cp_slope,
            draw_width_cp=self.options.cp_draw_width,
        )
        return np.asarray(wdl, dtype=np.float32).reshape(-1)[:3]

    # -- one row ----------------------------------------------------------

    def sample_from_row(self, row: dict[str, Any]) -> ReplaySample | None:
        """One corpus row -> one replay row, or None when the row is dropped."""
        board = self._board_for(row)
        if row.get("result") is None:
            self.stats.rows_dropped_no_result += 1
            return None
        bank = RowBank(row)
        self.stats.phases_per_row[bank.phase_count] = (
            self.stats.phases_per_row.get(bank.phase_count, 0) + 1
        )
        values = apply_scheme(bank, self.options.scheme)
        if values.floor_hit:
            self.stats.nodes_floor_hits += 1
        self._check_support(board, values, row)

        q = self.q_of(values.effective_cp)
        probs = softmax_at_temp(q, temp=self.options.temp)
        recovered = recover_temp(q, probs)
        if recovered is not None:
            self.stats.note_temp(recovered)

        policy = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float64)
        legal_mask = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.uint8)
        for move_uci, prob in zip(values.moves, probs):
            index = int(compact_index_for_move(board, chess.Move.from_uci(move_uci)))
            # ⚑ Range and COLLISION, both checked. A slot outside [0, 1858) is
            # the wrong index space (the AZ-4672 ids the search uses are the
            # near miss); a slot already claimed by another legal move is a
            # many-to-one map, which would fold two moves' mass onto one entry
            # and leave a legal mask that still names both. Neither shows up as
            # a shape error and neither changes the row's sum.
            if not 0 <= index < COMPACT_POLICY_SIZE:
                raise CorpusIntegrityError(
                    f"{_row_label(row)}: {move_uci} maps to policy slot {index}, "
                    f"outside the compact lc0_1858 space [0, {COMPACT_POLICY_SIZE})",
                )
            if legal_mask[index]:
                raise CorpusIntegrityError(
                    f"{_row_label(row)}: {move_uci} collides with an earlier "
                    f"legal move on policy slot {index}",
                )
            policy[index] = float(prob)
            legal_mask[index] = 1

        planes = self._encode(board)
        self._note_shapes(planes, policy, values)

        return ReplaySample(
            x=planes,
            policy_target=policy.astype(np.float32),
            wdl_target=wdl_target_from_result(float(row["result"])),
            legal_mask=legal_mask,
            # ⚑ The SEARCHED value goes here and not in `sf_wdl`; see the module
            # docstring for the guard that makes `sf_wdl` unreachable on this rig.
            search_wdl=self.wdl_of(float(values.effective_cp[values.best_index])),
            has_policy=True,
            is_selfplay=True,
            is_network_turn=True,
            game_id=int(row["game_id"]),
            ply_index=int(row["ply"]),
            input_history_encoding=INPUT_HISTORY_ENCODING,
            history_rep_fix=HISTORY_REP_FIX,
        )

    def _board_for(self, row: dict[str, Any]) -> chess.Board:
        """The row's board, with the row's OWN metadata re-derived from it.

        ``stm`` and ``piece_count`` are banked alongside the FEN, so they are a
        free external check that the row is internally consistent -- and a
        disagreement means the fields a consumer would filter on describe a
        different position from the one it would encode.
        """
        board = chess.Board(str(row["fen"]))
        stm = "w" if board.turn == chess.WHITE else "b"
        if stm != str(row["stm"]):
            raise CorpusIntegrityError(
                f"row {row.get('game_id')}/{row.get('ply')}: stm {row['stm']!r} "
                f"disagrees with the FEN's {stm!r}",
            )
        pieces = int(chess.popcount(board.occupied))
        if pieces != int(row["piece_count"]):
            raise CorpusIntegrityError(
                f"row {row.get('game_id')}/{row.get('ply')}: piece_count "
                f"{row['piece_count']} disagrees with the FEN's {pieces}",
            )
        return board

    def _check_support(
        self, board: chess.Board, values: MoveValues, row: dict[str, Any],
    ) -> None:
        """The banked move set must be EXACTLY python-chess's legal moves.

        The external referee, and the same one ``lc0_data_to_rows`` uses:
        python-chess computes legality independently of anything in this repo,
        so a FEN that drifted from the search, a promotion spelled the other
        way, or a phase-0 block that is complete-but-narrow all show up here
        instead of as a policy whose support is quietly wrong.
        """
        legal = {move.uci() for move in board.legal_moves}
        banked = set(values.moves)
        if legal != banked:
            raise CorpusIntegrityError(
                f"row {row.get('game_id')}/{row.get('ply')}: the banked move set "
                f"is not the legal move set (only banked: {sorted(banked - legal)}; "
                f"only legal: {sorted(legal - banked)})",
            )
        if len(banked) != len(values.moves):
            raise CorpusIntegrityError(
                f"row {row.get('game_id')}/{row.get('ply')}: the banked block "
                "lists a move twice, which would fold two ranks onto one slot",
            )
        self.stats.support_checks += 1

    def _encode(self, board: chess.Board) -> np.ndarray:
        return np.asarray(
            encode_position(
                board,
                add_features=True,
                input_history_encoding=INPUT_HISTORY_ENCODING,
                input_extra_features=INPUT_EXTRA_FEATURES,
            ),
            dtype=np.float32,
        )

    def _note_shapes(
        self, planes: np.ndarray, policy: np.ndarray, values: MoveValues,
    ) -> None:
        stats = self.stats
        stats.x_planes = int(planes.shape[0])
        stats.policy_width = int(policy.shape[0])
        # ⚑ AFTER the float16 cast the shard stores. A cold temperature over a
        # wide move list pushes the tail below float16's smallest subnormal, and
        # a support counted in float64 would report moves the trainer will read
        # as zero while the legal mask still names them.
        support = int((policy.astype(np.float16) > 0).sum())
        stats.policy_support_lost_to_float16 += int((policy > 0.0).sum()) - support
        stats.policy_support_min = (
            support if stats.policy_support_min < 0
            else min(stats.policy_support_min, support)
        )
        stats.policy_support_max = max(stats.policy_support_max, support)
        stats.history_slots_nonzero_max = max(
            stats.history_slots_nonzero_max, history_slots_filled(planes),
        )
        rep_planes = planes[
            [slot * _PLANES_PER_SLOT + _PIECE_PLANES_PER_SLOT
             for slot in range(_HISTORY_SLOTS)]
        ]
        if bool(np.any(rep_planes)):
            stats.repetition_planes_nonzero_rows += 1
        stats.depth_histogram[values.base_depth] = (
            stats.depth_histogram.get(values.base_depth, 0) + 1
        )
        for phase, depth in zip(values.phase_by_move, values.depth_by_move):
            stats.values_by_phase[phase] = stats.values_by_phase.get(phase, 0) + 1
            if depth == values.base_depth:
                stats.base_tier_moves += 1
            else:
                stats.deep_tier_moves += 1


# -- driving ------------------------------------------------------------------


def _row_label(row: dict[str, Any]) -> str:
    return f"game {row.get('game_id')} ply {row.get('ply')}"


def read_corpus_summary(corpus_dir: Path) -> dict[str, Any]:
    path = corpus_dir / "summary.json"
    if not path.exists():
        raise CorpusIntegrityError(
            f"{path} does not exist; a corpus without its summary carries no "
            "config_sha256, no staircase and no shard inventory, and every "
            "stamp this tool passes through would have to be invented",
        )
    summary: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    row_schema = int(summary.get("row_schema", -1))
    if row_schema != corpus.ROW_SCHEMA:
        raise CorpusIntegrityError(
            f"corpus row schema {row_schema} != this build's "
            f"{corpus.ROW_SCHEMA}; the block keys this tool reads are not "
            "promised to mean the same thing across a schema bump",
        )
    return summary


def cp_map_params(summary: dict[str, Any]) -> tuple[float, float]:
    """The corpus's OWN cp->value parameters, not this tool's.

    ⚑ Deliberately not a flag.  The generator SELECTED its moves with these two
    numbers, so deriving targets under different ones would produce a policy
    whose ranking disagrees with the play that generated the positions, silently
    and only in the tail.  A corpus that does not record them is refused rather
    than defaulted.
    """
    requested = summary.get("config_requested") or {}
    try:
        slope = float(requested["cp_slope"])
        draw_width = float(requested["cp_draw_width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CorpusIntegrityError(
            "summary.json's config_requested carries no usable cp_slope / "
            "cp_draw_width; the cp->value mapping cannot be reconstructed and "
            "defaulting it would derive targets under a mapping the corpus was "
            "not generated with",
        ) from exc
    # ⚑ Cross-checked against every worker's REALIZED stamp (review finding 1):
    # `config_requested` is what the CLI said, the realized stamp is what the
    # searcher actually converted with. A generator defect that dropped the
    # knob on the way to the searcher would select moves under one map while
    # the request stamps another — and deriving under the requested map would
    # then disagree with the play that generated the positions. A dead
    # worker's placeholder carries no cp keys and is skipped.
    for worker_id, stamp in (summary.get("config_realized_by_worker") or {}).items():
        if not isinstance(stamp, dict) or "cp_slope" not in stamp:
            continue
        realized = (
            float(stamp["cp_slope"]),
            float(stamp.get("cp_draw_width", draw_width)),
        )
        if realized != (slope, draw_width):
            raise CorpusIntegrityError(
                f"worker {worker_id}'s realized cp map {realized} disagrees "
                f"with config_requested ({slope}, {draw_width}); the corpus "
                "was selected under a mapping the request does not describe, "
                "and targets derived under either one would be wrong about "
                "the other",
            )
    return slope, draw_width


def derive(
    *,
    corpus_dir: Path,
    out_dir: Path,
    options: DeriveOptions,
    corpus_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read the corpus, write the shards, return the summary that describes them.

    ``corpus_summary`` is threaded in by ``main`` (which already read it to
    resolve the cp mapping) so the file is validated ONCE per run; omitting it
    reads and validates it here, which is what a direct caller wants.

    ⚑ ``derive_targets_summary.json`` is written LAST, so a directory holding
    shards and no summary is a run that DIED, not a corpus.  Nothing downstream
    has to detect that: the next attempt at the same ``--out`` hits
    ``refuse_populated_dir`` and stops, which is the fail-closed direction --
    the operator deletes a half-written directory deliberately rather than
    discovering later that half of it was described by nothing.
    """
    started = datetime.now(timezone.utc).isoformat()
    # ⚑ Here rather than in ``main`` so a caller cannot get past it: two runs'
    # shards in one directory is a corpus whose manifest describes half of it,
    # which is the same rule -- and the same reused function -- the generator
    # applies to its own out-dir.
    corpus.refuse_populated_dir(out_dir)
    summary = (
        corpus_summary if corpus_summary is not None
        else read_corpus_summary(corpus_dir)
    )
    problems = scheme_vs_staircase_problems(
        options.scheme, summary.get("staircase_parsed", []),
    )
    if problems:
        raise CorpusIntegrityError(
            f"--scheme {options.scheme.canonical} cannot be answered by this "
            "corpus: " + "; ".join(problems),
        )
    shards = corpus_shard_paths(corpus_dir)
    if not shards:
        raise CorpusIntegrityError(f"{corpus_dir} holds no .jsonl.zst/.jsonl.gz shards")
    check_shard_inventory(shards, summary)

    corpus_sha = str(summary.get("config_sha256", ""))
    deriver = TargetDeriver(options)
    rng = np.random.default_rng(options.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    pending: list[ReplaySample] = []
    shard_index = 0
    tt_carried: set[bool] = set()

    for path in shards:
        for row in iter_corpus_rows(path):
            if options.limit and deriver.stats.rows_read >= options.limit:
                break
            deriver.stats.rows_read += 1
            tt_carried.add(_check_row_identity(row, corpus_sha))
            try:
                sample = deriver.sample_from_row(row)
            except EnvelopeMiss as exc:
                deriver.stats.rows_dropped_envelope += 1
                if len(deriver.stats.envelope_miss_examples) < 8:
                    deriver.stats.envelope_miss_examples.append(
                        f"{_row_label(row)}: {exc}",
                    )
                if deriver.stats.rows_dropped_envelope > options.max_envelope_misses:
                    raise CorpusIntegrityError(
                        f"{_row_label(row)} cannot answer "
                        f"--scheme {options.scheme.canonical}: {exc}. That is "
                        f"{deriver.stats.rows_dropped_envelope} envelope miss(es) "
                        f"against --max-envelope-misses "
                        f"{options.max_envelope_misses}. Dropping rows changes "
                        "which positions the corpus contains, so the tolerance "
                        "is stated rather than assumed.",
                    ) from exc
                continue
            if sample is None:
                continue
            pending.append(sample)
            if len(pending) >= options.rows_per_shard:
                written.append(
                    _flush(out_dir, shard_index, pending, options, rng, corpus_sha),
                )
                deriver.stats.rows_written += len(pending)
                pending = []
                shard_index += 1
        if options.limit and deriver.stats.rows_read >= options.limit:
            break

    if pending:
        written.append(_flush(out_dir, shard_index, pending, options, rng, corpus_sha))
        deriver.stats.rows_written += len(pending)
    if not written:
        raise CorpusIntegrityError(
            "no rows survived; nothing was written. Read the drop counters "
            "before rerunning: an empty corpus and a corpus every row of which "
            "was dropped are different problems.",
        )

    out = build_summary(
        options=options,
        stats=deriver.stats,
        corpus_dir=corpus_dir,
        corpus_summary=summary,
        shards=written,
        started_utc=started,
        tt_carried=tt_carried,
    )
    (out_dir / SUMMARY_NAME).write_text(
        json.dumps(out, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return out


def _check_row_identity(row: dict[str, Any], corpus_sha: str) -> bool:
    """Every row belongs to the run ``summary.json`` describes.  Returns its TT flag.

    ⚑ The join key is the row's OWN ``config_sha256``, not the directory it was
    found in.  Two runs' rows in one directory would carry one summary's stamps
    on both halves, which is exactly what ``refuse_populated_dir`` exists to
    prevent upstream -- and this is the check that notices when it was defeated
    by hand.
    """
    schema = int(row.get("schema", -1))
    if schema != corpus.ROW_SCHEMA:
        raise CorpusIntegrityError(
            f"{_row_label(row)}: row schema {schema} != {corpus.ROW_SCHEMA}",
        )
    run = row.get("run")
    if not isinstance(run, dict) or corpus.KEY_TT_CARRIED not in run:
        raise CorpusIntegrityError(
            f"{_row_label(row)}: no run block carrying "
            f"{corpus.KEY_TT_CARRIED}; the derived shards could not disclose "
            "whether these searches shared a transposition table",
        )
    row_sha = str(run["config_sha256"])
    if corpus_sha and row_sha != corpus_sha:
        raise CorpusIntegrityError(
            f"{_row_label(row)}: config_sha256 {row_sha} does not match the "
            f"corpus summary's {corpus_sha}; these rows are from another run",
        )
    return bool(run[corpus.KEY_TT_CARRIED])


def _flush(
    out_dir: Path,
    index: int,
    samples: list[ReplaySample],
    options: DeriveOptions,
    rng: np.random.Generator,
    corpus_sha: str,
) -> dict[str, Any]:
    """Write one shard.  ``--seed`` permutes the rows inside it and nothing else.

    ⚑ Stated because the flag invites the other reading: the seed does NOT
    choose which rows are kept (``--limit`` takes a PREFIX of the corpus, which
    is a prefix of GAMES) and it does not touch a single target value.  It
    breaks up the game-contiguous order the corpus is written in, which is the
    order a mid-budget checkpoint would otherwise split by game.
    """
    order = rng.permutation(len(samples))
    ordered = [samples[int(i)] for i in order]
    path = local_shard_path(out_dir, index)
    save_local_shard_arrays(
        path,
        arrs=samples_to_arrays(ordered),
        meta=ShardMeta(
            run_id=SHARD_RUN_ID,
            input_history_encoding=INPUT_HISTORY_ENCODING,
            history_rep_fix=HISTORY_REP_FIX,
            policy_encoding="lc0_1858",
            policy_size=COMPACT_POLICY_SIZE,
            positions=len(ordered),
        ),
    )
    _stamp_shard_attrs(path, options, corpus_sha)
    return {"path": path.name, "rows": len(ordered)}


def _stamp_shard_attrs(path: Path, options: DeriveOptions, corpus_sha: str) -> None:
    """Put the scheme, its parameters and the code schema ON the shard.

    ⚑ A SECOND WRITE, and it has to be: ``save_local_shard_arrays`` funnels its
    meta through ``ShardMeta(**meta)``, which raises on any key that is not one
    of its declared fields -- so there is no way to carry a scheme name through
    it, and a scheme recorded only in the run's ``summary.json`` would be lost
    the moment one shard was copied somewhere else.  ``load_shard_arrays``
    returns the whole attrs dict, so these keys reach any reader that wants
    them and are ignored by every reader that does not.
    """
    group = zarr.open_group(str(path), mode="a")
    group.attrs.update({
        "derive_schema": DERIVE_SCHEMA,
        "derive_scheme": options.scheme.canonical,
        "derive_scheme_params": options.scheme.params(),
        "derive_temp": float(options.temp),
        "derive_cp_slope": float(options.cp_slope),
        "derive_cp_draw_width": float(options.cp_draw_width),
        "derive_corpus_config_sha256": corpus_sha,
        "derive_corpus_row_schema": corpus.ROW_SCHEMA,
    })


def build_summary(
    *,
    options: DeriveOptions,
    stats: DeriveStats,
    corpus_dir: Path,
    corpus_summary: dict[str, Any],
    shards: Sequence[dict[str, Any]],
    started_utc: str,
    tt_carried: set[bool],
) -> dict[str, Any]:
    """The output manifest.  Every knob appears as a REALIZED reading."""
    return {
        "schema": DERIVE_SCHEMA,
        "started_utc": started_utc,
        "tool": "scripts/derive_corpus_targets.py",
        "corpus": {
            "dir": corpus_dir.name,
            "config_sha256": corpus_summary.get("config_sha256"),
            "run_id": corpus_summary.get("run_id"),
            "row_schema": corpus.ROW_SCHEMA,
            "staircase_parsed": corpus_summary.get("staircase_parsed"),
            # Passed THROUGH, from the rows themselves rather than the summary:
            # a consumer must not mistake these for independent searches.
            corpus.KEY_TT_CARRIED: sorted(tt_carried),
            "banked_rows_min_piece_count": corpus_summary.get(
                "banked_rows_min_piece_count",
            ),
        },
        # ⚑ Rebuilt from the PARSED scheme object, not echoed from the flag.
        "scheme": {"canonical": options.scheme.canonical, **options.scheme.params()},
        "temp_requested": options.temp,
        "cp_map": {
            "q_function": (
                f"{gate.q_from_effective_cp.__module__}."
                f"{gate.q_from_effective_cp.__qualname__}"
            ),
            # ⚑ Resolved AT RUN TIME off the module attribute the mapping
            # actually goes through, so a monkeypatched or swapped object shows
            # up in the corpus that was written under it.
            "wdl_function": (
                f"{gen.cp_to_wdl_array.__module__}."
                f"{gen.cp_to_wdl_array.__qualname__}"
            ),
            "cp_slope": options.cp_slope,
            "cp_draw_width": options.cp_draw_width,
            "source": "the corpus's own config_requested",
        },
        "input": {
            "input_history_encoding": INPUT_HISTORY_ENCODING,
            "input_extra_features": INPUT_EXTRA_FEATURES,
            "history_rep_fix": HISTORY_REP_FIX,
            "history_frames_total": _HISTORY_SLOTS,
            "zero_history": stats.history_slots_nonzero_max <= 1,
            "why_zero_history": (
                "a corpus row is a FEN; banked plies are non-contiguous "
                "(dedup misses above MIN_BANKED_PIECES only) so the move stack "
                "cannot be rebuilt, and encode_position fills slot 0 only"
            ),
        },
        "policy": {
            "encoding": "lc0_1858",
            "width": COMPACT_POLICY_SIZE,
            "construction": "softmax(q / temp) over the scheme's values",
        },
        # ⚑ Both flags are COMPATIBILITY stamps, and both are lies of the same
        # shape `lc0_data_to_rows` already tells: no network played these moves
        # (Stockfish did) and the "selfplay" was engine-vs-itself. They are set
        # so the rows survive the trainer's network-turn filter, and they are
        # named here so a reader does not take them for provenance.
        "row_flags": {
            "is_network_turn": (
                "true — no network played these plies; set so the rows are not "
                "dropped by the train-on-network-turns filter"
            ),
            "is_selfplay": "true — Stockfish against itself, no curriculum arm",
            "priority": "1.0 — no surprise weighting exists for a corpus row",
        },
        "value_channels": {
            "wdl_target": (
                "the corpus row's exact game result, already stored from that "
                "row's own side-to-move seat (result_from_pov). 0=W/1=D/2=L"
            ),
            "search_wdl": (
                "cp_to_wdl_array of the SCHEME's best-move value: Stockfish's "
                "searched root value, side-to-move POV. ⚑ NOT our MCTS's "
                "value, which is what this column means on production shards"
            ),
            "sf_wdl": (
                "ABSENT — deliberately. lc0_control_train.py's launch guard 1 "
                "(assert_pid_cannot_reassert_sf_wdl) refuses any config with "
                "sf_wdl_frac > 0 regardless of what the shards carry, so a "
                "value written here could never reach a loss on this rig"
            ),
            "categorical_target": "ABSENT — outcome-derived, and wdl_target carries the outcome",
            "moves_left": "ABSENT — a corpus row does not know its game's length",
        },
        "value_blend": {
            "baked_into_rows": False,
            "note": (
                "the row carries the two components; the mixing weights are the "
                "trainer's, exactly as on data/lc0_rows"
            ),
        },
        "required_training_overrides": {
            "sf_wdl_frac": 0.0,
            "sf_wdl_frac_floor": 0.0,
            "search_wdl_frac": (
                "the whole non-outcome share. These shards carry no sf_wdl, so "
                "losses.py would redirect any SF share onto the raw game "
                "outcome; the searched value is in search_wdl"
            ),
        },
        "limit_requested": options.limit,
        "seed": options.seed,
        "seed_effect": "permutes rows WITHIN each shard; changes no target value",
        "rows_per_shard": options.rows_per_shard,
        "max_envelope_misses": options.max_envelope_misses,
        "realized": stats.summary(),
        "shards": list(shards),
        "python": sys.version.split()[0],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def format_summary(out: dict[str, Any]) -> str:
    realized = out["realized"]
    recovered = realized["temp_recovered_from_emitted_policy"]
    lines = [
        f"scheme={out['scheme']['canonical']} temp={out['temp_requested']} "
        f"value_source={out['scheme']['value_source']}",
        f"rows read={realized['rows_read']} written={realized['rows_written']} "
        f"dropped(no result)={realized['rows_dropped_no_result']} "
        f"dropped(envelope)={realized['rows_dropped_envelope']}",
        f"nodes_floor_hits={realized['nodes_floor_hits']} "
        f"base depths={realized['realized_base_depth_histogram']} "
        f"values by phase={realized['values_by_phase']}",
        f"temp recovered from the emitted policy: n={recovered['n']} "
        f"min={recovered['min']:.6f} max={recovered['max']:.6f}",
        f"x planes={realized['x_planes']} policy width={realized['policy_width']} "
        f"support {realized['policy_support_min']}..{realized['policy_support_max']} "
        f"history slots filled<={realized['history_slots_nonzero_max']}",
        f"shards={len(out['shards'])}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scheme", required=True, help=" | ".join(_SCHEME_FORMS))
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="stop after this many CORPUS ROWS READ (0 = the whole corpus). "
             "Rows dropped by a scheme or a missing result still count as read.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rows-per-shard", type=int, default=DEFAULT_ROWS_PER_SHARD)
    parser.add_argument(
        "--max-envelope-misses", type=int, default=0,
        help="how many rows may be dropped for lacking the block the scheme "
             "asks for before the run refuses. 0 (default) refuses on the first.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scheme = parse_scheme(str(args.scheme))
    temp = validate_temp(float(args.temp))
    if int(args.rows_per_shard) <= 0:
        raise ValueError(
            f"--rows-per-shard must be positive, got {args.rows_per_shard!r}",
        )
    if int(args.max_envelope_misses) < 0:
        raise ValueError(
            f"--max-envelope-misses must be >= 0, got {args.max_envelope_misses!r}",
        )
    corpus_dir = Path(args.corpus)
    corpus_summary = read_corpus_summary(corpus_dir)
    slope, draw_width = cp_map_params(corpus_summary)
    out = derive(
        corpus_dir=corpus_dir,
        out_dir=Path(args.out),
        corpus_summary=corpus_summary,
        options=DeriveOptions(
            scheme=scheme,
            temp=temp,
            cp_slope=slope,
            cp_draw_width=draw_width,
            limit=max(0, int(args.limit)),
            seed=int(args.seed),
            rows_per_shard=int(args.rows_per_shard),
            max_envelope_misses=int(args.max_envelope_misses),
        ),
    )
    print(format_summary(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
