#!/usr/bin/env python3
"""Frozen-driver / shadow-arm LABEL readout for the native NNUE value arms.

``scripts/nnue_gumbel_readout.py`` measures THROUGHPUT and refuses to report
label quality, and its own JSON says why: each arm drives its own Gumbel search,
so an arm that changes a leaf value changes which later leaves exist, and the
per-arm banks are trace artifacts of three different populations.  That is a
selection effect -- evaluators cannot be compared on positions each evaluator
chose for itself -- and its ``quality_scope`` block names the fix:

    "A frozen-driver/shadow-arm experiment is required for paired deep-SF
     attribution."

This file is that experiment.

ONE arm is the DRIVER.  It alone steers the production C Gumbel search through
``gen_random_selfplay_shards.play_game`` -- the same seam
``nnue_gumbel_readout.py`` uses, so the game loop, the C ``MCTSTree``, the root
budget, the pending-leaf ``CBoard`` binding and the terminal handling are the
generator's, not a second search written here.  Every other arm is an OBSERVER:
it runs in its OWN long-lived context, it is handed exactly the positions the
driver's search produced, and its answer is DISCARDED.  Nothing an observer
computes can reach the tree, because nothing an observer computes is ever
returned to it.

⚑ THE DRIVER ARM IS ALSO AN OBSERVER, in a second context of its own.  That is
not redundancy: the driver's search context and its observer context see the
same positions under the same configuration, so their values must agree
element-for-element, and ``driver_observer_disagreements`` is the live wiring
proof that the observers really are looking at the driver's positions.  It is
also the instrument that watches the DAG substrate: a memo cannot change an
answer (#472), so a disagreement means it did.

Three populations are observed, and each is a superset of the last:

* ROOT -- the position the search is about to run on, once per ply.
* ROOT CHILDREN -- the position after each legal root move, once per ply.  These
  are what the per-arm shard LABEL is built from; see below.
* SEARCH LEAVES -- every pending-leaf batch the driver's tree produced.  This is
  the paired evaluator sample ``quality_scope`` said did not exist: identical
  positions, N arms, one row each.  Banked only with
  ``--bank-leaf-observations``; the evaluation happens either way, because the
  observers' provider counters over an identical leaf population are themselves
  the paired behaviour comparison.

WHAT EACH CELL'S ``policy_target`` MEANS -- read this before quoting a number:

* ``search_gumbel__<driver>`` is the production Gumbel improved policy the
  DRIVER's search returned.  It is the same label rule
  ``gen_random_selfplay_shards`` writes, so this cell -- and only this cell --
  is comparable with the banked G2 native cells.  It is identical in every
  respect except that it is one cell, not three: there is no per-arm version of
  it, which is the whole point.
* ``oneply__<arm>`` is that ARM's own 1-ply ranking of the root's legal moves,
  as a policy vector: ``softmax(sigma * -q(child))`` over the root's children,
  where ``q`` is the arm's value through the same cp-logistic the generator
  uses.  No search, no tree, no selection -- a pure function of the arm's
  evaluation of a position set that is identical across every arm by
  construction (the root's children are a property of the root, and the roots
  are the driver's).

  ⚑ ``--oneply-sigma`` IS INERT FOR THE PRIMARY METRIC AND LOAD-BEARING FOR THE
  REST.  ``score_shard_labels.py``'s primary is ``top1_regret_cp``, the deep-SF
  regret of the target's argmax, and a softmax is monotone -- so no value of
  sigma can change it.  ``expected_regret_cp`` and the blunder rates are
  functions of sigma, so they are comparable ACROSS THE ``oneply__`` CELLS OF
  ONE RUN (one sigma) and are NOT comparable with the ``search_gumbel__`` cell
  or with the banked G2 series, which were produced by a different label rule.

SHALLOW-STOCKFISH OBSERVER ARMS -- ``sf-<nodes>``:

``--sf-observer 512`` adds an arm named ``sf-512`` that answers with a real
Stockfish search at ``go nodes 512`` instead of an NNUE evaluation.  It is an
OBSERVER ONLY (``--driver sf-*`` is refused): it never returns a value into the
tree, it labels the same rows as every other arm, and it emits
``oneply__sf-512`` beside the native cells.  A ladder of them
(``--sf-observer 512 --sf-observer 2048 --sf-observer 8192``) makes LABEL
QUALITY a measured function of LABEL SEARCH EFFORT on positions that are
identical by construction, which is the one comparison the per-arm banks could
never support.

Four things about these arms that a reader must not have to infer:

* **They see ROOTS and ROOT CHILDREN, and NOT search leaves.**  The 1-ply label
  is built from the root's children alone, and the leaf population is
  ``sims``-many positions per ply against the children's ~30 -- so fanning an
  8192-node engine out over it would multiply the run's cost by the simulation
  count to produce a column nothing reads.  ``leaf_positions`` is therefore 0
  for every ``sf-`` arm and the report says which populations the arm observed
  rather than leaving a zero to be read as a failure.
* **The engine answers from the EVALUATED position's seat**, exactly as the
  native arms do: UCI ``score cp`` is from the side to move.  ``probe_root``
  applies the root-mover negation for every arm alike -- measured, not assumed
  (4k3/8/8/8/8/8/8/3QK3 reads +566 with White to move and -537 with Black).
* **``score cp`` goes through the generator's own cp-logistic**, the same
  ``cp_to_wdl_array`` object ``NnueArmValueSource.q_from_values`` calls, at the
  same ``--nnue-cp-slope`` / ``--nnue-cp-draw-width``.  ``--nnue-cp-per-unit``
  is NOT applied: it converts internal NNUE units to centipawns and Stockfish
  already reports centipawns.  Mates fold through ``mate_to_effective_cp``, THE
  single mate-to-score home -- with ``score mate 0`` (a side to move that is
  already checkmated) mapped to a LOSS, because that function reads the sign off
  its argument and ``0`` is non-negative.
* **The transposition table is cleared once per GAME** (``ucinewgame``), never
  per position: per position it would cost a protocol round trip on every one of
  the ~30 children, and the banked hazard this repo already carries is a DIRTY
  TT across a whole label run, not warmth inside one game.  The policy is
  stamped into every ``sf-`` cell's ``cell_meta.json`` rather than left to the
  reader.  ⚑ An engine that DIES and is restarted mid-run gets a cold table in
  the middle of a game -- a different instrument for the rows after it -- so a
  restart is counted and makes the run INADMISSIBLE rather than being retried
  quietly.

Everything here is UNIFORM PRIOR.  ``gen_random_selfplay_shards`` uses no net at
all: the policy logits handed to the search are zeros, so the root candidate set
is chosen by the Gumbel draw alone and ``--all-root-moves`` (on by default) is
what makes every legal root move a candidate.  No claim in this file, its
output, or its sidecars depends on a learned prior, because there is not one.

⚑⚑ ``--dag-node-cap`` DOES NOT BOUND MEMORY, AND THE OOM IS WHY THIS FILE HAS A
SECOND KNOB.  ``_arm_providers.h`` sets ``q.nodes_used = 0`` once per
``cae_arm_qsearch_eval_mode`` call, so the cap is a PER-CALL quiescence node
budget; the canonical position store it is named after grows across calls and
the cap never looks at it.  MEASURED on this machine's build: 3000 evaluations
at ``dag_node_cap`` 0 interned 1,019,128 nodes for 5.79 GB, and the SAME 3000 at
``dag_node_cap`` 4096 interned 994,061 for the same 5.79 GB with 16
``dag_budget_trips`` -- the cap bound the search and not the store.  The store
costs ~5.68 kB per node (an NNUE accumulator payload per interned position), and
``arm_dag_reset`` returns ``node_count`` to 0 while RETAINING the allocation, so
peak memory is set by the largest ``node_count`` ever reached between resets.
``--dag-max-nodes`` is therefore a store watchdog: after every observed batch,
a DAG-backed arm whose ``node_count`` exceeds it is reset.  Same 6000-evaluation
stream, watchdog at 100,000: 13 resets, peak 1.45 GB against 5.79 GB unwatched.
Peak RSS is reported per worker, because a bound nobody reads is not a bound.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import resource
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import load_shard_arrays
from chess_anti_engine.stockfish.uci import StockfishResult, StockfishUCI
from chess_anti_engine.stockfish.wdl import cp_to_wdl_array, mate_to_effective_cp
from chess_anti_engine.utils import engine_discovery
from chess_anti_engine.utils.numpy_helpers import softmax_1d
from scripts import gen_random_selfplay_shards as gen
from scripts import nnue_gumbel_readout as readout

_LOG = logging.getLogger("nnue_shadow_label_readout")

#: Report schema. 1 is the first shape that reports paired label quality at all;
#: `nnue_gumbel_readout`'s schema counter is independent of this one and the two
#: files' reports are not interchangeable.
#: 2 adds the ``sf-<nodes>`` shallow-Stockfish observer family: ``observers``
#: entries now carry a ``family`` discriminator, an ``sf-`` entry has no
#: ``provider_stats`` at all (there is no C counter surface behind it), and
#: every entry carries a ``cost`` block. A schema-1 reader that subscripts
#: ``provider_stats`` on an ``sf-`` arm must fail rather than silently read a
#: native arm's shape into a Stockfish one.
REPORT_SCHEMA = 2

SEARCH_CELL_PREFIX = "search_gumbel__"
ONEPLY_CELL_PREFIX = "oneply__"

#: Arm-name prefix of the shallow-Stockfish observer family. An arm called
#: ``sf-2048`` is a real engine at ``go nodes 2048``; the node budget is IN THE
#: NAME because the cell directory, the shard rows and the pairwise agreement
#: table are all keyed by arm, and a ladder whose rungs were called ``sf-a`` /
#: ``sf-b`` would need the report beside it to be readable at all.
SF_ARM_PREFIX = "sf-"

#: One engine, one thread. Above 1 a node-limited Stockfish search stops being
#: reproducible at a fixed ``go nodes N``, which would make a label a function
#: of thread scheduling; the flag exists so a run that trades that away has to
#: say so, not so the default is a choice.
DEFAULT_SF_THREADS = 1

#: Transposition-table size, PINNED rather than inherited from whatever the
#: engine's own default happens to be on this build. It is part of the label
#: instrument: the same node budget with a different table is a different
#: search, and an artifact that does not name it cannot be compared with the
#: next one. 16 MB is Stockfish's own default, so pinning it changes nothing
#: today and freezes it against a future build that changes its mind.
DEFAULT_SF_HASH_MB = 16

#: Schema of the ``sf-`` observer's banked JSONL rows. Independent of the
#: generator's ``LEAF_BANK_SCHEMA``: the columns are an engine's, not an arm
#: context's, and a reader must not be able to mistake one file for the other.
SF_BANK_SCHEMA = 1

#: ⚑ ONE MESSAGE, TWO GATES. The parser refuses an ``sf-`` ``--driver`` and so
#: does ``run``; a caller that reaches either must be told the same thing, and a
#: shared literal is what keeps the two from drifting into "invalid choice".
_SF_DRIVER_REFUSAL = (
    "{arm} is an OBSERVER-ONLY arm and cannot be the --driver. The driver steers "
    "the production C Gumbel search through gen_random_selfplay_shards.play_game, "
    "which consumes a native NNUE value source; a Stockfish arm has no such seam, "
    "and a run whose positions were chosen by an engine would not be measuring "
    "that engine's labels on the native arms' positions -- which is the entire "
    "point of the frozen driver. Pass it as --sf-observer <nodes> instead."
)

#: Engine failures that justify a RESTART rather than a crash. Both
#: `StockfishTimeoutError` and `StockfishDesyncError` derive from `RuntimeError`,
#: and a dead child surfaces as `RuntimeError("Stockfish process exited")` or as
#: an `OSError` from the write side. ⚑ `ValueError` is deliberately absent: a
#: newline in a FEN is our bug, not the engine's, and restarting around it would
#: turn a defect into a statistic.
_SF_ENGINE_FAILURES: tuple[type[BaseException], ...] = (RuntimeError, OSError)

#: Store watchdog default, in canonical DAG nodes. At the measured ~5.68 kB per
#: node (see the module docstring) and the store's power-of-two payload table
#: this holds a DAG-backed arm's context near 1.5 GB, so the default matrix --
#: one DAG observer plus one FastQ observer per worker -- fits a couple of
#: workers on a box that is also training. It is a RESOURCE bound: it changes
#: how often the memo is dropped and therefore the arm's wall time, never its
#: answers (a memo that changed an answer would show up as a
#: driver-vs-observer disagreement, which is an inadmissible reason).
DEFAULT_DAG_MAX_NODES = 100_000

#: Measured on this repo's production build, 2026-08-26: `memory_bytes` /
#: `node_count` over a 3000-evaluation random-playout stream. Reported beside
#: every DAG peak so a reader can price a cap in bytes without re-deriving it.
MEASURED_DAG_BYTES_PER_NODE = 5681

#: Rows per shard. Shards close on a GAME boundary at or past this, exactly as
#: the generator does, so every cell's shards hold whole games.
DEFAULT_SHARD_SIZE = gen.DEFAULT_SHARD_SIZE

_ROLE_ROOT = "root"
_ROLE_LEAF = "leaf"
_ROLE_CHILD = "root-child"


def oneply_sigma_default() -> float:
    """The production Gumbel target's own sharpness scale, not a round number.

    ``gumbel._sigma_scale`` is ``c_scale * (c_visit + max_visit)`` and the
    generator pins ``gumbel_target_max_visit_cap``, so the target the search
    stores is built at a sigma of at most
    ``SELFPLAY_GUMBEL_C_SCALE * (c_visit + cap)``.  Starting the 1-ply label
    there puts the two cells' sharpness in the same family; it does not make
    their secondary metrics comparable, and nothing here claims it does.
    """
    return float(gen.SELFPLAY_GUMBEL_C_SCALE) * (
        float(GumbelConfig.c_visit) + float(gen.DEFAULT_TARGET_MAX_VISIT_CAP)
    )


# ── observers ────────────────────────────────────────────────────────────────


class DagBackedSource(Protocol):
    """What the store watchdog needs of an arm source.

    A protocol rather than ``ReadoutArmSource`` because the watchdog guards the
    DRIVER's source and every OBSERVER's, and because a bound that cannot be
    exercised without a 5 GB allocation is a bound nobody tests.
    """

    def dag_stats(self) -> dict[str, int] | None: ...

    def reset_game(self) -> None: ...


@dataclass
class DagStoreWatch:
    """Bound a DAG-backed arm's canonical store by RESETTING it.

    ⚑⚑ THIS EXISTS BECAUSE ``--dag-node-cap`` IS NOT A MEMORY BOUND, MEASURED
    RATHER THAN ASSUMED. ``_arm_providers.h`` sets ``q.nodes_used = 0`` once per
    ``cae_arm_qsearch_eval_mode`` CALL, so the cap bounds one evaluation's
    quiescence and never looks at the store the flag is named after: 3000
    evaluations at cap 0 interned 1,019,128 nodes for 5.79 GB, and the same 3000
    at cap 4096 interned 994,061 for the same 5.79 GB with 16 trips.

    ⚑ IT MUST BE ARM-STATE-ONLY, AND IT IS THE DRIVER'S TOO. A reset drops a
    memo, and #472's bit-identity result says a memo cannot change an answer --
    so a watched DAG driver plays the same games as an unwatched one, only
    slower. The claim is not left as an argument: the driver's watch fires
    identically in both passes of ``--prove-shadow-inertness``, and the
    driver-vs-observer value agreement compares two contexts whose watchdogs
    fire at different moments.

    The peaks are read BEFORE the reset, so ``nodes_peak`` is the largest the
    store actually reached rather than the value it was left at. It can exceed
    ``max_nodes`` by up to one batch's growth; that overshoot is the reason the
    peak is reported instead of the cap.
    """

    max_nodes: int
    resets: int = 0
    nodes_peak: int = 0
    memory_peak: int = 0

    def observe(self, source: DagBackedSource) -> None:
        stats = source.dag_stats()
        if stats is None:
            return
        nodes = int(stats["node_count"])
        self.nodes_peak = max(self.nodes_peak, nodes)
        self.memory_peak = max(self.memory_peak, int(stats["memory_bytes"]))
        if self.max_nodes > 0 and nodes > self.max_nodes:
            source.reset_game()
            self.resets += 1

    def merge(self, other: DagStoreWatch) -> None:
        self.resets += other.resets
        self.nodes_peak = max(self.nodes_peak, other.nodes_peak)
        self.memory_peak = max(self.memory_peak, other.memory_peak)


@dataclass
class ObserverStats:
    """What one observer arm saw, counted by POPULATION rather than pooled.

    ``NnueArmValueSource.stats`` counts roots and everything-else, which is the
    right split for the generator and the wrong one here: the root children are
    neither search leaves nor roots, and pooling them into ``leaves`` would put
    a per-ply population inside a per-leaf rate.
    """

    root_positions: int = 0
    child_positions: int = 0
    leaf_positions: int = 0
    root_batches: int = 0
    child_batches: int = 0
    leaf_batches: int = 0
    eval_s: float = 0.0

    def merge(self, other: ObserverStats) -> None:
        self.root_positions += other.root_positions
        self.child_positions += other.child_positions
        self.leaf_positions += other.leaf_positions
        self.root_batches += other.root_batches
        self.child_batches += other.child_batches
        self.leaf_batches += other.leaf_batches
        self.eval_s += other.eval_s

    def add_batch(self, role: str, n: int) -> None:
        if role == _ROLE_ROOT:
            self.root_batches += 1
            self.root_positions += int(n)
        elif role == _ROLE_CHILD:
            self.child_batches += 1
            self.child_positions += int(n)
        else:
            self.leaf_batches += 1
            self.leaf_positions += int(n)


class ObserverArm:
    """One arm under test, in a context the search cannot reach.

    ⚑ THE RETURN VALUE GOES NOWHERE.  ``evaluate`` hands its q back to this
    file's own recorder and to nothing else; the driver's q is what
    ``q_for_boards`` returns to the tree.  That is the whole shadow property,
    and it is structural rather than asserted -- but it is asserted anyway, by
    the digest gate, because "structural" is what everyone says about the wiring
    that turns out to be wrong.
    """

    #: Native arms are cheap enough per position to be fanned out over the
    #: search-leaf population as well as the root's children, and that fan-out
    #: is what makes ``driver_observer_disagreements`` -- the wiring proof --
    #: possible at all. The ``sf-`` family sets this False; see
    #: ``StockfishObserverArm``.
    observes_leaves: bool = True

    def __init__(self, *, source: readout.ReadoutArmSource, dag_max_nodes: int) -> None:
        self.source = source
        self.arm = source.arm
        self.stats = ObserverStats()
        self.dag_watch = DagStoreWatch(max_nodes=int(dag_max_nodes))

    def begin_game(self) -> None:
        """Per-GAME hook. A no-op here: the native arms' canonical-store reset
        cadence is ``--dag-reset``'s to choose, and firing a reset from here too
        would take that choice away from the flag that owns it."""

    def evaluate(
        self, boards: list[CBoard], *, role: str, cluster: tuple[int, int] | None,
    ) -> np.ndarray:
        # ⚑ The role goes through VERBATIM. `q_for_boards` treats every
        # non-"root" role as a leaf for its own counters and writes the string
        # it was given into the bank, so passing `_ROLE_CHILD` is what lets a
        # banked row say which of the three populations it came from. This
        # object's own `stats` is the split that is not pooled.
        started = time.perf_counter()
        q = self.source.q_for_boards(boards, role=role, cluster=cluster)
        self.stats.eval_s += time.perf_counter() - started
        self.stats.add_batch(role, len(boards))
        self.dag_watch.observe(self.source)
        return np.asarray(q, dtype=np.float64)

    def reset_dag(self) -> None:
        self.source.reset_game()

    def provider_stats(self) -> dict[str, int]:
        return self.source.provider_stats()

    def close(self) -> None:
        self.source.close()


# ── the shallow-Stockfish observer family ────────────────────────────────────


def is_sf_arm(arm: str) -> bool:
    return str(arm).startswith(SF_ARM_PREFIX)


def sf_arm_name(nodes: int) -> str:
    return f"{SF_ARM_PREFIX}{int(nodes)}"


def sf_effective_cp(*, cp: int | None, mate: int | None) -> float:
    """One UCI score -> effective centipawns, on the pipeline's own scale.

    ⚑⚑ ``score mate 0`` IS A LOSS AND ``mate_to_effective_cp`` CALLS IT A WIN.
    Stockfish emits ``info depth 0 score mate 0`` + ``bestmove (none)`` for a
    side to move that is ALREADY checkmated (measured on this build's engine),
    and ``mate_to_effective_cp`` reads the sign off its argument -- ``0`` is
    non-negative, so it returns ``+100000``.  Handed through unchanged that
    turns every mate delivered by a root move into a value of ``+1`` for the
    side that just got mated, and after ``probe_root``'s negation the root
    mover's own mating move becomes the WORST-ranked move in the label.  It
    fails no shape check, sums to one, and is dense over the right move set.
    The generator's native path carries the same trap in the same shape --
    ``NnueArmValueSource.q_from_values`` documents it -- and the native arm
    answers ``-1.0`` on a mated position, which is the number this branch
    reproduces.

    Mate precedence over cp is the UCI convention and matches ``cp_to_wdl``:
    an engine emits at most one of the two per info line.
    """
    if mate is not None:
        if int(mate) == 0:
            return -abs(mate_to_effective_cp(0))
        return mate_to_effective_cp(int(mate))
    if cp is not None:
        return float(cp)
    raise RuntimeError(
        "Stockfish returned neither a cp nor a mate score for a position; a "
        "search with no score at all cannot be turned into a value, and "
        "substituting a draw here would put a fabricated label in a cell whose "
        "whole purpose is label quality",
    )


def sf_q_from_effective_cp(
    eff_cp: np.ndarray, *, cp_slope: float, cp_draw_width: float,
) -> np.ndarray:
    """Effective cp -> q in [-1, 1], through THE generator's cp-logistic.

    ⚑ ``cp_to_wdl_array`` IS THE SAME FUNCTION OBJECT
    ``NnueArmValueSource.q_from_values`` calls -- both import it from
    ``chess_anti_engine.stockfish.wdl`` -- and ``W - L`` is that method's own
    final line.  A second implementation with the same formula would be a
    second thing to keep in step; ``test_the_sf_arm_reuses_the_generators_cp_logistic``
    asserts the identity rather than the arithmetic.

    ⚑ ``--nnue-cp-per-unit`` IS DELIBERATELY NOT APPLIED.  It is the slope of
    internal NNUE units to centipawns, and Stockfish already reports
    centipawns; multiplying by 0.28 here would shrink every Stockfish score by
    ~3.6x, land the whole ladder inside the draw zone, and look exactly like
    "shallow Stockfish is a flat evaluator".
    """
    wdl = cp_to_wdl_array(
        np.asarray(eff_cp, dtype=np.float64),
        slope=float(cp_slope), draw_width_cp=float(cp_draw_width),
    )
    return (
        wdl[..., 0].astype(np.float64) - wdl[..., 2].astype(np.float64)
    )


@dataclass(frozen=True)
class SfArmConfig:
    """One rung of the node ladder, fully self-describing.

    The binary digest is resolved ONCE in the parent and carried down, for the
    same reason ``NnueArmValueSource`` takes ``pack_file_sha256``: hashing a
    ~60 MB engine inside every worker's measured window prices the instrument
    into the measurement.  The workers re-hash anyway, once, at setup, and the
    aggregation refuses a run whose workers did not all map these bytes.
    """

    arm: str
    nodes: int
    threads: int
    hash_mb: int
    binary: Path
    binary_sha256: str
    binary_source: str

    def consumed(self) -> dict[str, int]:
        """Every number this arm's engine was configured with, as the plan.

        The same shape ``ResolvedArmConfig.consumed`` returns for a native arm,
        so ``arm_config`` / ``arm_config_realized`` mean the same thing in both
        families' report blocks.
        """
        return {
            "nodes": int(self.nodes),
            "threads": int(self.threads),
            "hash_mb": int(self.hash_mb),
        }


@dataclass
class SfArmStats:
    """What the engine behind one ``sf-`` arm actually did.

    ⚑ ``restarts`` IS AN ADMISSIBILITY COUNTER, NOT A HEALTH GAUGE.  A restarted
    engine resumes with a COLD transposition table in the middle of a game, so
    every row after it was labelled by a measurably different instrument from
    the rows before it.  The alternative -- catch, log, continue -- is this
    repository's signature defect exactly: a value accepted and then silently
    ignored.
    """

    searches: int = 0
    cp_scores: int = 0
    mate_scores: int = 0
    mate_zero_scores: int = 0
    #: Positions whose FEN does NOT reconstruct the full search state, i.e. the
    #: board carried repetition history the UCI ``position fen`` line cannot
    #: transmit. The native arms see the `CBoard` and its hash stack; the engine
    #: sees a FEN. Counted rather than argued about — the 50-move clock DOES
    #: travel (it is a FEN field), only repetition does not.
    positions_without_repetition_history: int = 0
    engine_new_games: int = 0
    restarts: int = 0
    first_restart_error: str = ""
    #: ⚑⚑ THE TAKE-EFFECT PROOF FOR ``--sf-observer <nodes>``, read off the
    #: CONSUMER. ``go nodes N`` is a request; these are the ``info ... nodes``
    #: counts Stockfish itself reported back, so a budget that never reached the
    #: engine shows up here as the wrong number rather than as nothing at all.
    #: A terminal position reports no node count (depth 0, ``bestmove (none)``),
    #: which is why ``engine_nodes_reported`` is a separate denominator from
    #: ``searches``. ``engine_nodes_min`` starts at -1 = "never set".
    engine_nodes_reported: int = 0
    engine_nodes_sum: int = 0
    engine_nodes_min: int = -1
    engine_nodes_max: int = 0

    def observe_nodes(self, nodes: int | None) -> None:
        if nodes is None:
            return
        value = int(nodes)
        self.engine_nodes_reported += 1
        self.engine_nodes_sum += value
        self.engine_nodes_max = max(self.engine_nodes_max, value)
        self.engine_nodes_min = (
            value if self.engine_nodes_min < 0 else min(self.engine_nodes_min, value)
        )

    def merge(self, other: SfArmStats) -> None:
        self.searches += other.searches
        self.cp_scores += other.cp_scores
        self.mate_scores += other.mate_scores
        self.mate_zero_scores += other.mate_zero_scores
        self.positions_without_repetition_history += (
            other.positions_without_repetition_history
        )
        self.engine_new_games += other.engine_new_games
        self.restarts += other.restarts
        if not self.first_restart_error:
            self.first_restart_error = other.first_restart_error
        self.engine_nodes_reported += other.engine_nodes_reported
        self.engine_nodes_sum += other.engine_nodes_sum
        self.engine_nodes_max = max(self.engine_nodes_max, other.engine_nodes_max)
        if other.engine_nodes_min >= 0:
            self.engine_nodes_min = (
                other.engine_nodes_min if self.engine_nodes_min < 0
                else min(self.engine_nodes_min, other.engine_nodes_min)
            )


class StockfishObserverArm:
    """A real Stockfish search, as a shadow arm on the driver's positions.

    ⚑ IT CANNOT REACH THE TREE, AND THE PROOF COVERS IT ANYWAY.  This object
    holds a subprocess and a FEN; it is handed COPIES of the driver's boards and
    its answer goes to this file's recorder and nowhere else.  That is
    structural -- and ``--prove-shadow-inertness`` replays the same games with
    the whole observer set detached and requires bit-identical digests, because
    "structural" is what everyone says about the wiring that turns out to be
    wrong.

    ⚑ ROOTS AND ROOT CHILDREN ONLY (``observes_leaves = False``).  The 1-ply
    label is a function of the root's children; the leaf population is
    ``sims``-many positions per ply, so fanning an engine over it would multiply
    the run's cost by the simulation count for a column nothing reads.  The
    report publishes which populations the arm observed, so the zero is a stated
    scope rather than a number to be read as a failure.
    """

    observes_leaves: bool = False

    def __init__(
        self,
        *,
        config: SfArmConfig,
        cp_slope: float,
        cp_draw_width: float,
        nice: int = 0,
        bank: Path | None = None,
        bank_identity: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.arm = str(config.arm)
        self.cp_slope = float(cp_slope)
        self.cp_draw_width = float(cp_draw_width)
        self.nice = int(nice)
        self.stats = ObserverStats()
        self.sf_stats = SfArmStats()
        self.bank_identity = dict(bank_identity or {})
        self.bank_rows = 0
        if bank is not None:
            bank.parent.mkdir(parents=True, exist_ok=True)
        self.leaf_bank_path = bank
        # "x" for the same reason the generator's bank uses it: a rerun that
        # appended would produce one file whose rows came from two runs.
        self._bank = None if bank is None else bank.open("x")
        self._engine = self._open_engine()

    # ── engine lifecycle ────────────────────────────────────────────────
    def _open_engine(self) -> StockfishUCI:
        return StockfishUCI(
            str(self.config.binary),
            nodes=int(self.config.nodes),
            hash_mb=int(self.config.hash_mb),
            nice=int(self.nice),
            threads=int(self.config.threads),
        )

    def realized(self) -> dict[str, int]:
        """What the LIVE engine object was configured with, not what was planned.

        ⚑ THIS IS WEAKER THAN THE NATIVE ARMS' REQUESTED-VS-REALIZED CHECK AND
        THE REPORT SAYS SO.  UCI has no readback: an engine cannot be asked what
        ``Threads`` or ``Hash`` it ended up with, so the strongest available
        statement about those two is the value written to its stdin.  The one
        knob that IS observed at the consumer is the node budget --
        ``SfArmStats.engine_nodes_*`` are the counts Stockfish reported back
        about its own searches.
        """
        return {
            "nodes": int(self._engine.nodes),
            "threads": int(self._engine.threads),
            "hash_mb": int(self._engine.hash_mb or 0),
        }

    def begin_game(self) -> None:
        """Clear the transposition table between GAMES, never between positions.

        ⚑ THE CADENCE IS THE POINT AND IT IS STAMPED INTO ``cell_meta.json``.
        Per position it costs a ``ucinewgame`` + ``isready`` round trip on every
        one of the root's ~30 children and buys independence nothing in this
        harness compares across; per RUN it reproduces the banked hazard this
        repository already carries (production SF labels run on a dirty TT).
        Per game is the cadence the driver's own games have.
        """
        self._engine.new_game()
        self.sf_stats.engine_new_games += 1

    def close(self) -> None:
        try:
            self._engine.close()
        finally:
            if self._bank is not None:
                self._bank.close()
                self._bank = None

    def _restart(self, exc: BaseException) -> None:
        self.sf_stats.restarts += 1
        if not self.sf_stats.first_restart_error:
            self.sf_stats.first_restart_error = f"{type(exc).__name__}: {exc}"
        _LOG.error(
            "INADMISSIBLE: %s engine failed mid-run (%s: %s) and is being "
            "restarted; the rows after this point were labelled with a COLD "
            "transposition table, which is a different instrument from the rows "
            "before it",
            self.arm, type(exc).__name__, exc,
        )
        try:
            self._engine.close()
        except _SF_ENGINE_FAILURES:  # pragma: no cover - close() swallows most
            pass
        self._engine = self._open_engine()

    def _search(self, fen: str) -> StockfishResult:
        """One node-limited search, with EXACTLY ONE restart of tolerance.

        ⚑ A SECOND CONSECUTIVE FAILURE RAISES.  One dead engine is a fact to
        record and press on from with the run marked inadmissible; two in a row
        on the same position is a broken binary or a broken board, and a harness
        that kept retrying would turn that into a hang with a plausible report
        at the end of it.
        """
        try:
            return self._engine.search(fen, nodes=int(self.config.nodes))
        except _SF_ENGINE_FAILURES as exc:
            self._restart(exc)
        return self._engine.search(fen, nodes=int(self.config.nodes))

    # ── the observer surface ────────────────────────────────────────────
    def evaluate(
        self, boards: list[CBoard], *, role: str, cluster: tuple[int, int] | None,
    ) -> np.ndarray:
        """Every board in the batch, one engine search each, from ITS OWN seat.

        The return is the same units and the same convention every other arm
        returns: q in [-1, 1] from the EVALUATED position's side to move.
        ``probe_root`` owns the root-mover negation and applies it to all arms
        alike.
        """
        started = time.perf_counter()
        eff_cp = np.empty((len(boards),), dtype=np.float64)
        results: list[StockfishResult] = []
        for i, board in enumerate(boards):
            if int(board.hash_stack_len) != 0:
                self.sf_stats.positions_without_repetition_history += 1
            result = self._search(board.fen())
            self.sf_stats.searches += 1
            self.sf_stats.observe_nodes(result.nodes)
            if result.mate is not None:
                self.sf_stats.mate_scores += 1
                if int(result.mate) == 0:
                    self.sf_stats.mate_zero_scores += 1
            elif result.cp is not None:
                self.sf_stats.cp_scores += 1
            eff_cp[i] = sf_effective_cp(cp=result.cp, mate=result.mate)
            results.append(result)
        q = sf_q_from_effective_cp(
            eff_cp, cp_slope=self.cp_slope, cp_draw_width=self.cp_draw_width,
        )
        if self._bank is not None:
            self._bank_batch(boards, results, eff_cp, q, role=role, cluster=cluster)
        # ⚑ The wall clock is stopped AFTER the bank write, deliberately: the
        # cost axis this arm exists to supply is what the operator pays for the
        # column, and the banking is part of producing it.
        self.stats.eval_s += time.perf_counter() - started
        self.stats.add_batch(role, len(boards))
        return q

    def _bank_batch(
        self,
        boards: list[CBoard],
        results: list[StockfishResult],
        eff_cp: np.ndarray,
        q: np.ndarray,
        *,
        role: str,
        cluster: tuple[int, int] | None,
    ) -> None:
        """One JSONL row per evaluated position: the RAW UCI score, and its key.

        The raw ``cp``/``mate`` pair is the only thing that cannot be recovered
        later: the effective cp, the logistic and q are all pure functions of it
        and the three constants banked beside it. Without the raw score a slope
        correction or a different value map is a rerun of the engine, and a
        rerun against a node-limited search is not a reanalysis -- it re-rolls
        the intervention.
        """
        assert self._bank is not None
        game, ply = (-1, -1) if cluster is None else cluster
        for board, result, eff, value in zip(
            boards, results, eff_cp.tolist(), q.tolist(), strict=True,
        ):
            self._bank.write(
                json.dumps(
                    {
                        "schema": SF_BANK_SCHEMA,
                        "arm_family": "stockfish",
                        "arm": self.arm,
                        "role": role,
                        "fen": board.fen(),
                        "halfmove_clock": int(board.halfmove_clock),
                        "hash_stack_len": int(board.hash_stack_len),
                        "fen_reconstructs_full_search_state": bool(
                            board.hash_stack_len == 0,
                        ),
                        "cp": None if result.cp is None else int(result.cp),
                        "mate": None if result.mate is None else int(result.mate),
                        "effective_cp": float(eff),
                        "q_from_evaluated_seat": float(value),
                        "engine_nodes_reported": (
                            None if result.nodes is None else int(result.nodes)
                        ),
                        "engine_depth": (
                            None if result.depth is None else int(result.depth)
                        ),
                        "bestmove": str(result.bestmove_uci),
                        "game": int(game),
                        "ply": int(ply),
                        "cp_slope": self.cp_slope,
                        "cp_draw_width": self.cp_draw_width,
                        **self.bank_identity,
                        **self.config.consumed(),
                        "sf_binary_sha256": self.config.binary_sha256,
                    },
                    sort_keys=True,
                )
                + "\n",
            )
            self.bank_rows += 1


class ShadowObserver(Protocol):
    """What the worker loop and the fan-out need of an observer, either family.

    Two implementations -- the native ``ObserverArm`` and
    ``StockfishObserverArm`` -- and they differ in exactly one thing the caller
    can see, ``observes_leaves``. The protocol exists so that difference is read
    off the object rather than re-decided at each call site, which is how the
    leaf fan-out would eventually acquire a Stockfish engine by accident.
    """

    arm: str
    stats: ObserverStats

    @property
    def observes_leaves(self) -> bool: ...

    def evaluate(
        self, boards: list[CBoard], *, role: str, cluster: tuple[int, int] | None,
    ) -> np.ndarray: ...

    def begin_game(self) -> None: ...

    def close(self) -> None: ...


class ArmObserver(Protocol):
    """What ``probe_root`` needs of an arm, so the probe can be tested alone.

    ``ObserverArm`` satisfies this structurally. The protocol exists because the
    root probe is the one piece of this file whose CONVENTION can be wrong
    without anything raising -- a target built on the un-negated child value is
    a perfectly well-formed probability vector over the right move set -- and a
    convention is only pinned by a test that can supply its own values.
    """

    arm: str

    def evaluate(
        self, boards: list[CBoard], *, role: str, cluster: tuple[int, int] | None,
    ) -> np.ndarray: ...


@dataclass
class PlyProbe:
    """One ply's frozen root, and every arm's 1-ply answer about it."""

    game: int
    ply_ordinal: int
    legal_full_indices: tuple[int, ...]
    #: arm -> q of each legal move FROM THE ROOT MOVER'S SEAT, aligned with
    #: ``legal_full_indices``.
    q_mover: dict[str, np.ndarray]


class ProbeRecorder:
    """The per-ply root probes of one game, in ply order."""

    def __init__(self) -> None:
        self.game: int = -1
        self.probes: list[PlyProbe] = []

    def begin_game(self, game: int) -> None:
        self.game = int(game)
        self.probes = []

    def add(self, probe: PlyProbe) -> None:
        if probe.game != self.game:
            raise RuntimeError(
                f"probe recorded for game {probe.game} while game {self.game} "
                "was being played; the evaluator's cluster key and the game "
                "loop disagree",
            )
        if probe.ply_ordinal != len(self.probes):
            raise RuntimeError(
                f"probe for game {probe.game} arrived at ply ordinal "
                f"{probe.ply_ordinal} with {len(self.probes)} already recorded; "
                "the per-ply hook did not fire exactly once per ply",
            )
        self.probes.append(probe)


#: The pseudo-arm the pairwise table uses for the driver's Gumbel target, so the
#: search cell appears in the same resolution table as the 1-ply cells.
SEARCH_LABEL = "search_gumbel_target"


@dataclass
class ArgmaxAgreement:
    """How often two cells' targets pick the SAME move, per row.

    ⚑⚑ THIS IS THE INSTRUMENT'S RESOLUTION, AND IT HAS TO BE READ FIRST.
    ``score_shard_labels.py``'s primary metric is ``top1_regret_cp`` -- the
    deep-SF regret of the target's ARGMAX -- so two cells whose argmax agrees on
    every row have IDENTICAL top1 numbers no matter how differently they value
    the moves underneath.  A paired delta of 0.00 cp between such cells is not a
    finding about the arms; it is the ruler reporting that it was handed the
    same column twice.  Measured on the 48-row smoke: qsearch and qsearch-DAG
    agreed on 48/48 (the #472 oracle) and so did FastQ, which is exactly the
    regime in which the deep-SF spend would have bought nothing.  Read
    ``rows_disagreeing`` before committing Stockfish time.
    """

    rows: int = 0
    rows_disagreeing: int = 0

    def add(self, same: bool) -> None:
        self.rows += 1
        self.rows_disagreeing += int(not same)

    def merge(self, other: ArgmaxAgreement) -> None:
        self.rows += other.rows
        self.rows_disagreeing += other.rows_disagreeing

    def summary(self) -> dict[str, float]:
        return {
            "rows": float(self.rows),
            "rows_disagreeing": float(self.rows_disagreeing),
            "disagreement_rate": (
                self.rows_disagreeing / self.rows if self.rows else float("nan")
            ),
        }


def pair_key(left: str, right: str) -> str:
    return f"{left}|{right}"


@dataclass
class AgreementStats:
    """Driver search context vs driver-arm observer context, position by position."""

    compared: int = 0
    disagreements: int = 0
    first_disagreement: str = ""

    def merge(self, other: AgreementStats) -> None:
        self.compared += other.compared
        self.disagreements += other.disagreements
        if not self.first_disagreement:
            self.first_disagreement = other.first_disagreement


class ShadowFanoutSource(readout.ReadoutArmSource):
    """The DRIVER's search source, with the observer fan-out hooked onto it.

    ⚑ THE HOOK IS ON THE SOURCE, NOT THE EVALUATOR, and that is forced rather
    than chosen: ``play_game`` evaluates the ROOT through
    ``native_root_logits(cb, source=evaluator.nnue_source)`` and the leaves
    through ``evaluator.evaluate_encoded``, so the source is the one object both
    populations pass through.  Hooking the evaluator would see the leaves and
    miss every root, and the root is where the frozen position set comes from.

    ⚑ THE DRIVER IS EVALUATED FIRST, ALWAYS.  ``super().q_for_boards`` runs
    before any observer touches the board list, so even an observer that
    mutated a board could not change the value the search receives for it.  The
    tree-level version of that question -- whether an observer changed anything
    the search does LATER -- is not arguable and is not argued: it is settled by
    ``--prove-shadow-inertness``, which replays the same games with the
    observers detached and requires bit-identical digests.
    """

    def __init__(
        self,
        *,
        observers: tuple[ShadowObserver, ...],
        recorder: ProbeRecorder,
        agreement: AgreementStats,
        dag_watch: DagStoreWatch,
        **kwargs: Any,
    ) -> None:
        self._observers = tuple(observers)
        # ⚑ DERIVED FROM THE OBSERVERS, NOT PASSED IN BESIDE THEM. Which arms
        # can afford the search-leaf population is a property of the arm (an
        # `sf-` engine cannot), and a second argument saying so would be a
        # second place for the answer to live.
        self._leaf_observers = tuple(o for o in observers if o.observes_leaves)
        self._recorder = recorder
        self._agreement = agreement
        self.dag_watch = dag_watch
        super().__init__(**kwargs)

    def q_for_boards(
        self,
        boards: list[CBoard],
        *,
        role: str = _ROLE_LEAF,
        cluster: tuple[int, int] | None = None,
    ) -> np.ndarray:
        q = super().q_for_boards(boards, role=role, cluster=cluster)
        # ⚑ BEFORE THE OBSERVER BRANCH, SO IT FIRES IN BOTH PROOF PASSES. A
        # DAG-backed DRIVER needs the store watchdog exactly as an observer does
        # -- FastQ and qsearch-DAG both intern -- and a watchdog that ran only
        # when observers were attached would be a difference between the two
        # passes rather than something the proof covers.
        self.dag_watch.observe(self)
        if not self._observers:
            return q
        if role == _ROLE_ROOT:
            self._observe_root(boards, cluster)
        else:
            self._observe_leaves(boards, q, cluster)
        return q

    def _observe_root(
        self, boards: list[CBoard], cluster: tuple[int, int] | None,
    ) -> None:
        if len(boards) != 1 or cluster is None:
            raise RuntimeError(
                f"root evaluation arrived with {len(boards)} boards and cluster "
                f"{cluster!r}; play_game evaluates exactly one root per ply and "
                "binds its (game, ply) key first",
            )
        self._recorder.add(
            probe_root(boards[0], observers=self._observers, cluster=cluster),
        )

    def _observe_leaves(
        self, boards: list[CBoard], driver_q: np.ndarray, cluster: tuple[int, int] | None,
    ) -> None:
        for observer in self._leaf_observers:
            observed = observer.evaluate(boards, role=_ROLE_LEAF, cluster=cluster)
            if observer.arm != self.arm:
                continue
            self._agreement.compared += int(observed.size)
            if np.array_equal(observed, np.asarray(driver_q, dtype=np.float64)):
                continue
            bad = int(np.argmax(observed != np.asarray(driver_q, dtype=np.float64)))
            self._agreement.disagreements += int(
                np.count_nonzero(observed != np.asarray(driver_q, dtype=np.float64)),
            )
            if not self._agreement.first_disagreement:
                self._agreement.first_disagreement = (
                    f"{boards[bad].fen()}: driver {float(driver_q[bad])!r} vs "
                    f"observer {float(observed[bad])!r}"
                )


def probe_root(
    root: CBoard,
    *,
    observers: Sequence[ArmObserver],
    cluster: tuple[int, int],
) -> PlyProbe:
    """Every arm's 1-ply answer about ONE frozen root, on identical children.

    The children are a property of the ROOT, not of any arm, so this is where
    the position set stops depending on who is evaluating it.
    """
    legal = tuple(int(i) for i in root.legal_move_indices())
    if not legal:
        raise RuntimeError(
            "root has no legal moves; play_game does not search a terminal "
            "position, so this batch is not a root",
        )
    children: list[CBoard] = []
    for action in legal:
        child = root.copy()
        child.push_index(action)
        children.append(child)
    q_mover: dict[str, np.ndarray] = {}
    for observer in observers:
        observer.evaluate([root], role=_ROLE_ROOT, cluster=cluster)
        # ⚑ NEGATED. The arm answers from the CHILD's side to move, which is the
        # root's opponent; a target built on the un-negated value ranks the
        # root's moves exactly backwards and is still a well-formed probability
        # vector over the right move set, so nothing downstream can see it.
        # Pinned by test_the_oneply_target_is_read_from_the_root_movers_seat.
        q_mover[observer.arm] = -np.asarray(
            observer.evaluate(children, role=_ROLE_CHILD, cluster=cluster),
            dtype=np.float64,
        )
    return PlyProbe(
        game=int(cluster[0]),
        ply_ordinal=int(cluster[1]),
        legal_full_indices=legal,
        q_mover=q_mover,
    )


def oneply_policy_vector(
    legal_full_indices: tuple[int, ...], q_mover: np.ndarray, *, sigma: float,
) -> np.ndarray:
    """One arm's 1-ply ranking of the root's legal moves, as a (4672,) target."""
    values = np.asarray(q_mover, dtype=np.float64)
    if values.shape != (len(legal_full_indices),):
        raise ValueError(
            f"{values.shape} arm values for {len(legal_full_indices)} legal "
            "moves: the probe and the root disagree about the move set",
        )
    if not bool(np.isfinite(values).all()):
        raise ValueError(
            "a non-finite arm value reached the 1-ply target; a softmax would "
            "turn it into a one-hot or a NaN row and neither would raise",
        )
    probs = np.zeros((gen.POLICY_SIZE,), dtype=np.float32)
    probs[np.asarray(legal_full_indices, dtype=np.int64)] = softmax_1d(
        float(sigma) * values,
    ).astype(np.float32)
    return probs


def oneply_outcome(
    outcome: gen.GameOutcome, probes: list[PlyProbe], *, arm: str, sigma: float,
) -> gen.GameOutcome:
    """The driver's game, relabelled with ONE arm's 1-ply targets.

    Every field except ``policy_probs`` is the driver's, so the emitted rows
    share the driver's positions, legal masks, game outcome, ``game_id`` and ply
    ordering -- which is what makes the cells row-aligned and the comparison
    paired.
    """
    if len(probes) != len(outcome.records):
        raise RuntimeError(
            f"{len(probes)} root probes for {len(outcome.records)} stored plies: "
            "the per-ply hook and the game loop did not see the same game",
        )
    records: list[gen.PlyRecord] = []
    for record, probe in zip(outcome.records, probes, strict=True):
        stored = tuple(
            int(i) for i in np.flatnonzero(np.asarray(record.legal_mask)).tolist()
        )
        if stored != probe.legal_full_indices:
            raise RuntimeError(
                f"ply {record.ply_index}: the search stored {len(stored)} legal "
                f"moves and the root probe saw {len(probe.legal_full_indices)}; "
                "the probe is not looking at the position the search searched",
            )
        records.append(
            replace(
                record,
                policy_probs=oneply_policy_vector(
                    probe.legal_full_indices, probe.q_mover[arm], sigma=sigma,
                ),
            ),
        )
    return replace(outcome, records=records)


# ── plan / cells ─────────────────────────────────────────────────────────────


def search_cell_name(driver: str) -> str:
    return f"{SEARCH_CELL_PREFIX}{driver}"


def oneply_cell_name(arm: str) -> str:
    return f"{ONEPLY_CELL_PREFIX}{arm}"


def cell_names(driver: str, arms: tuple[str, ...]) -> tuple[str, ...]:
    return (search_cell_name(driver), *(oneply_cell_name(a) for a in arms))


@dataclass(frozen=True)
class RunConfig:
    """Everything a worker needs, in one picklable box."""

    driver: readout.ResolvedArmConfig
    observers: tuple[readout.ResolvedArmConfig, ...]
    pack: Path
    out_dir: Path
    games: int
    workers: int
    seed: int
    sims: int
    topk: int
    max_plies: int
    all_root_moves: bool
    cp_per_internal_unit: float
    cp_slope: float
    cp_draw_width: float
    oneply_sigma: float
    dag_max_nodes: int
    dag_reset_every: int
    shard_size: int
    bank_path: Path | None
    run_id: str
    nice: int
    emit_shards: bool = True
    attach_observers: bool = True
    #: The shallow-Stockfish ladder. A SEPARATE tuple rather than a wider
    #: ``observers``, because ``ResolvedArmConfig`` is the NNUE extension's plan
    #: -- it fills its defaults out of ``_nnue_ext`` and its ``consumed()`` looks
    #: the arm up in ``ARM_SPECS``. An ``sf-`` rung has neither, and forcing it
    #: through that type is exactly how it would acquire a plausible NNUE knob
    #: dict that nothing reads. Defaulted to empty so every existing caller and
    #: every native-only run means what it did before.
    sf_observers: tuple[SfArmConfig, ...] = ()

    @property
    def nnue_arms(self) -> tuple[str, ...]:
        return tuple(c.arm for c in self.observers)

    @property
    def sf_arms(self) -> tuple[str, ...]:
        return tuple(c.arm for c in self.sf_observers)

    @property
    def arms(self) -> tuple[str, ...]:
        """Every LABELLING arm, native first, in cell order.

        ⚑ The order is part of the artifact: ``cells``, the pairwise argmax
        table's keys and the shard directories are all derived from it.
        """
        return (*self.nnue_arms, *self.sf_arms)

    @property
    def cells(self) -> tuple[str, ...]:
        return cell_names(self.driver.arm, self.arms)


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    game_indices: tuple[int, ...]
    cfg: RunConfig


@dataclass
class WorkerResult:
    worker_id: int
    games: int
    plies: int
    setup_s: float
    elapsed_s: float
    peak_rss_bytes: int
    game_records: list[readout.GameRecord]
    terminations: dict[str, int]
    root_budget: gen.RootBudgetStats
    driver_provider_stats: dict[str, int]
    observer_provider_stats: dict[str, dict[str, int]]
    observer_stats: dict[str, ObserverStats]
    driver_dag_watch: DagStoreWatch
    observer_dag_watch: dict[str, DagStoreWatch]
    agreement: AgreementStats
    argmax_pairs: dict[str, ArgmaxAgreement]
    shards: dict[str, list[dict[str, Any]]]
    rows: int
    bank_rows: int
    kernel: str
    pack_file_sha256: str
    pack_source_sha256: str
    nice_realized: int
    nnue_ext_sha256: str = ""
    nnue_ext_loaded_build_id: str = ""
    mcts_ext_sha256: str = ""
    mcts_ext_loaded_build_id: str = ""
    lc0_ext_sha256: str = ""
    lc0_ext_loaded_build_id: str = ""
    arm_config_realized: dict[str, dict[str, int]] = field(default_factory=dict)
    #: ``sf-`` arms are kept in their own maps rather than folded into
    #: ``observer_*``: the aggregation reads a native arm's ``provider_stats``
    #: by SUBSCRIPT off its own C surface, and an ``sf-`` arm has no such
    #: surface at all. Merging the two would force a ``.get(..., 0)`` there,
    #: which is the exact shape the file already refuses one comment above it.
    sf_observer_stats: dict[str, ObserverStats] = field(default_factory=dict)
    sf_arm_stats: dict[str, SfArmStats] = field(default_factory=dict)
    #: This worker's OWN hash of the engine binary. The parent hashed it before
    #: the run; a disagreement means the workers did not all run one engine.
    sf_binary_sha256: str = ""


def _base_gen_config(cfg: RunConfig) -> gen.GenConfig:
    """The generator config the DRIVER's game loop runs under.

    ``value_source`` is the driver's arm because the driver is what the search
    consumes; the observers are not a generator concept at all.
    """
    return gen.GenConfig(
        out_dir=cfg.out_dir,
        games=cfg.games,
        workers=cfg.workers,
        sims=cfg.sims,
        topk=cfg.topk,
        c_scale=gen.SELFPLAY_GUMBEL_C_SCALE,
        policy_temp=gen.DEFAULT_POLICY_TEMP,
        temperature=gen.DEFAULT_TEMPERATURE,
        gumbel_scale=gen.DEFAULT_GUMBEL_SCALE,
        target_max_visit_cap=gen.DEFAULT_TARGET_MAX_VISIT_CAP,
        target_untempered_prior=gen.DEFAULT_TARGET_UNTEMPERED_PRIOR,
        vloss_weight=gen.DEFAULT_VLOSS_WEIGHT,
        target_batch=gen.DEFAULT_TARGET_BATCH,
        value_source=cfg.driver.arm,
        all_root_moves=cfg.all_root_moves,
        nnue_pack=cfg.pack,
        nnue_cp_per_unit=cfg.cp_per_internal_unit,
        nnue_cp_slope=cfg.cp_slope,
        nnue_cp_draw_width=cfg.cp_draw_width,
        max_plies=cfg.max_plies,
        shard_size=cfg.shard_size,
        seed=cfg.seed,
        nice=cfg.nice,
        run_id=cfg.run_id,
    )


def cell_dir(out_dir: Path, cell: str) -> Path:
    return Path(out_dir) / "cells" / cell


def _worker_bank_path(
    base: Path | None, *, role: str, arm: str, worker_id: int,
) -> Path | None:
    """One bank file per (role, arm, worker); ``"x"`` mode refuses a rerun."""
    if base is None:
        return None
    suffix = base.suffix or ".jsonl"
    stem = base.name[: -len(suffix)] if base.name.endswith(suffix) else base.name
    return base.with_name(f"{stem}.{role}.{arm}.w{worker_id:02d}{suffix}")


def _build_source(
    *,
    config: readout.ResolvedArmConfig,
    cfg: RunConfig,
    bank: Path | None,
    identity: dict[str, Any],
    pack_sha: str,
    observers: tuple[ShadowObserver, ...] | None = None,
    recorder: ProbeRecorder | None = None,
    agreement: AgreementStats | None = None,
    dag_watch: DagStoreWatch | None = None,
) -> readout.ReadoutArmSource:
    kwargs: dict[str, Any] = {
        "config": config,
        "pack": cfg.pack,
        "pack_file_sha256": pack_sha,
        "cp_per_internal_unit": cfg.cp_per_internal_unit,
        "cp_slope": cfg.cp_slope,
        "cp_draw_width": cfg.cp_draw_width,
        "leaf_bank": bank,
        "bank_identity": identity,
    }
    if observers is None:
        return readout.ReadoutArmSource(**kwargs)
    if recorder is None or agreement is None or dag_watch is None:
        raise ValueError(
            "a fan-out source needs a recorder, an agreement tally and a DAG "
            "store watch",
        )
    return ShadowFanoutSource(
        observers=observers, recorder=recorder, agreement=agreement,
        dag_watch=dag_watch, **kwargs,
    )


def _peak_rss_bytes() -> int:
    """This process's high-water RSS. Linux reports ``ru_maxrss`` in KiB."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _run_worker(spec: WorkerSpec) -> WorkerResult:
    cfg = spec.cfg
    nice_realized = readout._apply_nice(cfg.nice)
    rep_fix.apply(True)
    setup_started = time.perf_counter()
    base = _base_gen_config(cfg)
    gcfg = gen.build_gumbel_config(base)
    opening_cfg = gen.build_opening_config(base)
    pack_stamp = readout._file_stamp(cfg.pack)
    pack_sha = pack_stamp[0]
    identity = {
        "run_id": cfg.run_id,
        "seed": int(cfg.seed),
        "worker_id": int(spec.worker_id),
        "driver_arm": cfg.driver.arm,
        "population_kind": "frozen_driver_shadow",
    }
    recorder = ProbeRecorder()
    agreement = AgreementStats()
    driver_dag_watch = DagStoreWatch(max_nodes=int(cfg.dag_max_nodes))
    # ⚑ ONE CONTEXT PER OBSERVER, OPENED IN SEQUENCE. `set_arm_config` /
    # `fastq_set_config` are PROCESS-wide globals that `arm_open` snapshots, so
    # a context keeps the configuration that was live when it opened; building
    # them one at a time is what lets three arms with three configurations
    # coexist, and `NnueArmValueSource`'s requested-vs-realized check is what
    # proves each one got its own.
    observers: list[ObserverArm] = [
        ObserverArm(
            source=_build_source(
                config=observer_config,
                cfg=cfg,
                bank=_worker_bank_path(
                    cfg.bank_path, role="observer",
                    arm=observer_config.arm, worker_id=spec.worker_id,
                ),
                identity={**identity, "observer_arm": observer_config.arm},
                pack_sha=pack_sha,
            ),
            dag_max_nodes=cfg.dag_max_nodes,
        )
        for observer_config in (cfg.observers if cfg.attach_observers else ())
    ]
    # ⚑ ONE ENGINE PER (WORKER, ARM), OPENED HERE AND REUSED FOR EVERY POSITION.
    # A process spawn plus a UCI handshake costs far more than a 512-node search,
    # so an engine per position would price the harness's own plumbing into the
    # cost axis the ladder exists to measure.
    sf_observers: list[StockfishObserverArm] = [
        StockfishObserverArm(
            config=sf_config,
            cp_slope=cfg.cp_slope,
            cp_draw_width=cfg.cp_draw_width,
            nice=nice_realized,
            bank=_worker_bank_path(
                cfg.bank_path, role="observer",
                arm=sf_config.arm, worker_id=spec.worker_id,
            ),
            bank_identity={**identity, "observer_arm": sf_config.arm},
        )
        for sf_config in (cfg.sf_observers if cfg.attach_observers else ())
    ]
    sf_binary_sha256 = (
        readout._sha256_file(cfg.sf_observers[0].binary) if cfg.sf_observers else ""
    )
    all_observers: tuple[ShadowObserver, ...] = (*observers, *sf_observers)
    driver = _build_source(
        config=cfg.driver,
        cfg=cfg,
        bank=_worker_bank_path(
            cfg.bank_path, role="driver", arm=cfg.driver.arm, worker_id=spec.worker_id,
        ),
        identity={**identity, "observer_arm": ""},
        pack_sha=pack_sha,
        observers=all_observers,
        recorder=recorder,
        agreement=agreement,
        dag_watch=driver_dag_watch,
    )
    readout._assert_file_unchanged("NNUE pack while opening", cfg.pack, pack_stamp)
    evaluator = readout.ReadoutEvaluator(
        source=driver,
        expected_planes=gen.input_plane_count(base.input_extra_features),
        input_history_encoding=base.input_history_encoding,
        input_extra_features=base.input_extra_features,
    )
    from chess_anti_engine.encoding import _lc0_ext as lc0_ext
    from chess_anti_engine.mcts import _mcts_tree as mcts_ext

    _, lc0_sha, lc0_build_id, _ = readout._module_identity(lc0_ext)
    _, nnue_sha, nnue_build_id, _ = readout._module_identity(readout._load_ext())
    _, mcts_sha, mcts_build_id, _ = readout._module_identity(mcts_ext)
    setup_s = time.perf_counter() - setup_started

    budget = gen.RootBudgetStats()
    terminations = dict.fromkeys(gen.TERMINATIONS, 0)
    records: list[readout.GameRecord] = []
    pending: dict[str, list[ReplaySample]] = {cell: [] for cell in cfg.cells}
    shards: dict[str, list[dict[str, Any]]] = {cell: [] for cell in cfg.cells}
    tally = gen.ShardTally()
    shard_idx = int(spec.worker_id)
    stride = max(1, int(cfg.workers))
    search_cell = search_cell_name(cfg.driver.arm)
    labels: tuple[str, ...] = (SEARCH_LABEL, *cfg.arms)
    argmax_pairs: dict[str, ArgmaxAgreement] = {
        pair_key(left, right): ArgmaxAgreement()
        for i, left in enumerate(labels) for right in labels[i + 1:]
    }

    def flush() -> None:
        nonlocal tally, shard_idx
        if not pending[search_cell]:
            return
        sizes = {cell: len(rows) for cell, rows in pending.items()}
        if len(set(sizes.values())) != 1:
            raise RuntimeError(
                f"cells hold different row counts at a flush boundary: {sizes}. "
                "One shard index would then name different rows in different "
                "cells and the row-ordinal sampler would unpair them.",
            )
        for cell in cfg.cells:
            shards[cell].append(
                gen.write_shard(
                    out_dir=cell_dir(cfg.out_dir, cell),
                    index=shard_idx,
                    rows=pending[cell],
                    cfg=base,
                    tally=tally,
                ),
            )
            pending[cell] = []
        shard_idx += stride
        tally = gen.ShardTally()

    started = time.perf_counter()
    try:
        for position, game_index in enumerate(spec.game_indices):
            if cfg.dag_reset_every > 0 and position % cfg.dag_reset_every == 0:
                driver.reset_game()
                for observer in observers:
                    observer.reset_dag()
            # ⚑ UNCONDITIONAL, AND SEPARATE FROM `--dag-reset` ON PURPOSE. The
            # DAG cadence is a memo-retention knob whose answers cannot change
            # (#472); a Stockfish transposition table CAN change an answer, so
            # its clearing cadence is a property of the instrument and is not
            # available for an operator flag to turn off by accident.
            for observer in all_observers:
                observer.begin_game()
            recorder.begin_game(int(game_index))
            rng = np.random.default_rng(int(cfg.seed) + int(game_index))
            outcome = gen.play_game(
                cfg=base,
                gcfg=gcfg,
                evaluator=evaluator,
                rng=rng,
                opening_cfg=opening_cfg,
                budget=budget,
                game_index=int(game_index),
            )
            terminations[outcome.termination] = (
                terminations.get(outcome.termination, 0) + 1
            )
            records.append(
                readout.GameRecord(
                    game=int(game_index),
                    plies=int(outcome.plies),
                    result=str(outcome.result),
                    termination=str(outcome.termination),
                    digest=readout.game_digest(
                        game_index=int(game_index),
                        start_fen=str(outcome.start_fen),
                        move_trace=str(outcome.move_trace),
                        result=str(outcome.result),
                        termination=str(outcome.termination),
                    ),
                    search_digest=readout.search_output_digest(outcome.records),
                ),
            )
            if not cfg.emit_shards:
                continue
            # ⚑ THE ARGMAX IS TAKEN OFF THE EMITTED ROW, NOT OFF THE 4672-SPACE
            # VECTOR IT CAME FROM. `rows_from_game` re-encodes the target into
            # compact-1858, which is the column `score_shard_labels.py` reads,
            # and ties break by index -- so an argmax computed one space earlier
            # can name a different move than the one the ruler will score.
            search_rows = gen.rows_from_game(outcome, cfg=base)
            pending[search_cell].extend(search_rows)
            chosen: dict[str, list[int]] = {
                SEARCH_LABEL: [int(np.argmax(r.policy_target)) for r in search_rows],
            }
            for arm in cfg.arms:
                arm_rows = gen.rows_from_game(
                    oneply_outcome(
                        outcome, recorder.probes, arm=arm, sigma=cfg.oneply_sigma,
                    ),
                    cfg=base,
                )
                pending[oneply_cell_name(arm)].extend(arm_rows)
                chosen[arm] = [int(np.argmax(r.policy_target)) for r in arm_rows]
            for i, left in enumerate(labels):
                for right in labels[i + 1:]:
                    tally_pair = argmax_pairs[pair_key(left, right)]
                    for a, b in zip(chosen[left], chosen[right], strict=True):
                        tally_pair.add(a == b)
            tally.add(outcome)
            if len(pending[search_cell]) >= int(cfg.shard_size):
                flush()
        if cfg.emit_shards:
            flush()
    finally:
        evaluator.close()
        # ⚑ EVERY observer, including the engines: a `sf-` arm holds a
        # subprocess, and one that outlives the worker is a ~2.6 GB orphan the
        # audit-R2 note in `stockfish/uci.py` exists about.
        for closing in all_observers:
            closing.close()
    elapsed = time.perf_counter() - started
    readout._assert_file_unchanged("NNUE pack", cfg.pack, pack_stamp)
    return WorkerResult(
        worker_id=spec.worker_id,
        games=len(spec.game_indices),
        plies=sum(int(r.plies) for r in records),
        setup_s=setup_s,
        elapsed_s=elapsed,
        peak_rss_bytes=_peak_rss_bytes(),
        game_records=records,
        terminations=terminations,
        root_budget=budget,
        driver_provider_stats=driver.provider_stats(),
        observer_provider_stats={o.arm: o.provider_stats() for o in observers},
        observer_stats={o.arm: o.stats for o in observers},
        driver_dag_watch=driver_dag_watch,
        observer_dag_watch={o.arm: o.dag_watch for o in observers},
        agreement=agreement,
        argmax_pairs=argmax_pairs,
        shards=shards,
        rows=sum(int(s["rows"]) for s in shards[search_cell]),
        bank_rows=(
            int(driver.bank_rows)
            + sum(int(o.source.bank_rows) for o in observers)
            + sum(int(o.bank_rows) for o in sf_observers)
        ),
        kernel=driver.kernel,
        pack_file_sha256=driver.pack_file_sha256,
        pack_source_sha256=driver.pack_source_sha256,
        nice_realized=nice_realized,
        nnue_ext_sha256=nnue_sha,
        nnue_ext_loaded_build_id=nnue_build_id,
        mcts_ext_sha256=mcts_sha,
        mcts_ext_loaded_build_id=mcts_build_id,
        lc0_ext_sha256=lc0_sha,
        lc0_ext_loaded_build_id=lc0_build_id,
        arm_config_realized={
            cfg.driver.arm: dict(driver.realized),
            **{o.arm: dict(o.source.realized) for o in observers},
            **{o.arm: o.realized() for o in sf_observers},
        },
        sf_observer_stats={o.arm: o.stats for o in sf_observers},
        sf_arm_stats={o.arm: o.sf_stats for o in sf_observers},
        sf_binary_sha256=sf_binary_sha256,
    )


def _build_worker_specs(cfg: RunConfig) -> list[WorkerSpec]:
    buckets: list[list[int]] = [[] for _ in range(cfg.workers)]
    for game in range(cfg.games):
        buckets[game % cfg.workers].append(game)
    return [
        WorkerSpec(worker_id=i, game_indices=tuple(games), cfg=cfg)
        for i, games in enumerate(buckets) if games
    ]


def _run_workers(cfg: RunConfig) -> list[WorkerResult]:
    specs = _build_worker_specs(cfg)
    if not specs:
        raise ValueError("no games to play")
    if len(specs) == 1:
        return [_run_worker(specs[0])]
    with ProcessPoolExecutor(max_workers=len(specs)) as pool:
        results = list(pool.map(_run_worker, specs))
    # `map` already yields in submission order; sorting states the invariant the
    # aggregation depends on rather than assuming it.
    results.sort(key=lambda r: r.worker_id)
    return results


# ── the shadow-inertness proof ───────────────────────────────────────────────


def driver_digests(results: list[WorkerResult]) -> dict[str, str]:
    records = [rec for r in results for rec in r.game_records]
    return {
        "games_digest": readout.games_digest(records),
        "searches_digest": readout.searches_digest(records),
    }


def prove_shadow_inertness(cfg: RunConfig, *, games: int) -> dict[str, Any]:
    """Replay the same games with and without observers; digests must be EQUAL.

    ⚑ A DIFFERING DIGEST IS A HARD ERROR, not a warning.  The observers exist
    to be inert; an observer that moved the driver's trajectory or its improved
    policy has made every cell in the run a different experiment from every
    other, and the resulting shards would be a paired comparison of positions
    that were not the same positions.  There is no partial credit available
    here, so there is no non-fatal reading to report.

    Both passes run in one worker over the same game indices.  Every input to a
    game is a pure function of the seed -- ``np.random.default_rng(seed +
    game_index)`` per game, ``sample_action_with_temperature`` draws nothing at
    ``DEFAULT_TEMPERATURE`` 0.0, and ``sample_starting_board`` short-circuits
    with no book -- so the only thing that differs between the passes is
    whether the observers are attached.
    """
    if games <= 0:
        raise ValueError("--prove-games must be >= 1 to prove anything")
    probe = replace(
        cfg,
        games=int(games),
        workers=1,
        emit_shards=False,
        bank_path=None,
        attach_observers=True,
    )
    with_observers = driver_digests(_run_workers(probe))
    # ⚑ BOTH FAMILIES ARE DETACHED. `attach_observers=False` already builds
    # neither, and emptying both tuples as well is what makes the second pass a
    # config a reader can see is observer-free rather than one that depends on
    # a flag being honoured in two places.
    without = driver_digests(
        _run_workers(
            replace(probe, attach_observers=False, observers=(), sf_observers=()),
        ),
    )
    if with_observers != without:
        raise RuntimeError(
            "SHADOW INERTNESS PROOF FAILED: the driver's trajectory and/or its "
            "improved-policy search output changed when the observer arms were "
            f"attached. with_observers={with_observers} without={without}. The "
            "observers are not shadows and no cell of this run is paired with "
            "any other.",
        )
    return {
        "games": int(games),
        "with_observers": with_observers,
        "without_observers": without,
        "digests_agree": True,
    }


# ── cell alignment ───────────────────────────────────────────────────────────

#: Columns every cell must reproduce exactly. ``policy_target`` is deliberately
#: absent: it is the one column the cells are supposed to differ in.
_ALIGNED_COLUMNS: tuple[str, ...] = (
    "x", "legal_mask", "wdl_target", "has_policy", "game_id", "ply_index",
)


def assert_cells_are_row_aligned(out_dir: Path, cells: tuple[str, ...]) -> dict[str, Any]:
    """Every cell must hold the SAME rows in the SAME order.

    ⚑ THIS IS WHAT MAKES THE COMPARISON PAIRED, AND IT IS NOT FREE.
    ``scratchpad/az_purity/score_shard_labels.py`` samples a cell by ROW
    ORDINAL over its shards sorted by name -- so two cells are scored on the
    same positions only if their shard names, their per-shard row counts and
    their row order all match.  Nothing downstream checks it, and a cell that
    silently lost a game would still produce a plausible number.
    """
    if len(cells) < 2:
        raise ValueError("row alignment needs at least two cells to compare")
    reference = cells[0]
    ref_paths = sorted(p.name for p in cell_dir(out_dir, reference).glob("shard_*"))
    if not ref_paths:
        raise RuntimeError(f"cell {reference} wrote no shards under {out_dir}")
    checked = 0
    for cell in cells[1:]:
        names = sorted(p.name for p in cell_dir(out_dir, cell).glob("shard_*"))
        if names != ref_paths:
            raise RuntimeError(
                f"cell {cell} has shards {names} and cell {reference} has "
                f"{ref_paths}: the cells are not row-aligned and a row-ordinal "
                "sampler would score them on different positions",
            )
    for name in ref_paths:
        ref_arrays, _ = load_shard_arrays(cell_dir(out_dir, reference) / name)
        for cell in cells[1:]:
            arrays, _ = load_shard_arrays(cell_dir(out_dir, cell) / name)
            for column in _ALIGNED_COLUMNS:
                if not np.array_equal(
                    np.asarray(ref_arrays[column]), np.asarray(arrays[column]),
                ):
                    raise RuntimeError(
                        f"cell {cell} shard {name} column {column!r} differs from "
                        f"cell {reference}: the cells are not the same positions",
                    )
            checked += 1
    return {
        "cells": list(cells),
        "shards_per_cell": len(ref_paths),
        "cross_cell_shard_comparisons": checked,
        "aligned_columns": list(_ALIGNED_COLUMNS),
    }


# ── aggregation ──────────────────────────────────────────────────────────────


def _merge_provider(
    dicts: list[dict[str, int]], arm: str,
) -> tuple[dict[str, int], dict[str, list[int]]]:
    return readout.merge_provider_stats(dicts, readout.key_classes_for(arm))


def _observer_cost(stats: ObserverStats) -> dict[str, float | str]:
    """The COST AXIS of the quality-vs-cost curve, MEASURED rather than assumed.

    ⚑⚑ THE POINT OF A NODE LADDER IS A CURVE WITH TWO AXES, AND THE SECOND ONE
    IS ONLY EVER GUESSED AT.  "512 nodes is sub-millisecond, 8192 may dominate"
    is a plausible sentence that no artifact in this repository has ever
    contained a number for.  ``eval_s`` is wall time inside the observer's own
    ``evaluate``, summed over the arm's whole population, so the ratio between
    two rungs of one run is a like-for-like read on identical positions.

    ⚑ IT IS CPU-SECONDS, NOT WALL SECONDS, WHENEVER ``--workers > 1``: the
    per-worker figures are SUMMED, exactly as the populations are.  Divide by
    ``workers_active`` for a wall-clock reading, and do not compare it against
    ``search_wall_s`` without doing so.
    """
    positions = (
        int(stats.root_positions) + int(stats.child_positions)
        + int(stats.leaf_positions)
    )
    return {
        "eval_s": float(stats.eval_s),
        "positions": float(positions),
        "s_per_position": (
            float(stats.eval_s) / positions if positions else float("nan")
        ),
        "positions_per_s": (
            positions / float(stats.eval_s) if stats.eval_s > 0.0 else float("nan")
        ),
        "seconds_are": "cpu (summed over workers), not wall",
    }


def _sf_cp_mapping(cfg: RunConfig) -> dict[str, Any]:
    """The exact score-to-value map an ``sf-`` cell's labels went through."""
    return {
        "function": "chess_anti_engine.stockfish.wdl.cp_to_wdl_array",
        "shared_with": "NnueArmValueSource.q_from_values (the same function object)",
        "cp_slope": float(cfg.cp_slope),
        "cp_draw_width": float(cfg.cp_draw_width),
        "q": "W - L",
        # ⚑ Named so a reader cannot wonder whether it was applied. It converts
        # internal NNUE units to centipawns; Stockfish already reports
        # centipawns, so applying it would shrink every score by ~3.6x and put
        # the whole ladder inside the draw zone.
        "cp_per_internal_unit_applied": False,
        "cp_per_internal_unit": float(cfg.cp_per_internal_unit),
        "mate": "chess_anti_engine.stockfish.wdl.mate_to_effective_cp",
        "mate_zero": (
            "score mate 0 = the side to move is ALREADY checkmated, mapped to a "
            "LOSS; mate_to_effective_cp reads the sign off its argument and 0 is "
            "non-negative, so the unguarded call returns a WIN"
        ),
    }


def _aggregate(
    cfg: RunConfig,
    results: list[WorkerResult],
    *,
    wall_s: float,
    proof: dict[str, Any],
    alignment: dict[str, Any] | None,
) -> dict[str, Any]:
    records = [rec for r in results for rec in r.game_records]
    agreement = AgreementStats()
    for r in results:
        agreement.merge(r.agreement)
    reasons: list[str] = []
    if agreement.disagreements:
        reasons.append(
            f"the driver's search context and its own {cfg.driver.arm} observer "
            f"context disagreed on {agreement.disagreements} of "
            f"{agreement.compared} leaf values (first: "
            f"{agreement.first_disagreement}); the observers are not evaluating "
            "the positions the driver searched, or the arm is not a pure "
            "function of the position",
        )
    if agreement.compared == 0:
        reasons.append(
            "no driver-vs-observer value comparison was made at all: the driver "
            "arm is not among the observer arms, so nothing proves the observers "
            "saw the driver's leaves",
        )

    driver_provider, driver_conflicts = _merge_provider(
        [r.driver_provider_stats for r in results], cfg.driver.arm,
    )
    if driver_conflicts:
        reasons.append(f"driver workers disagreed about arm configuration: {driver_conflicts}")
    driver_watch = DagStoreWatch(max_nodes=cfg.dag_max_nodes)
    for r in results:
        driver_watch.merge(r.driver_dag_watch)

    observers: dict[str, Any] = {}
    for arm in cfg.nnue_arms:
        provider, conflicts = _merge_provider(
            [r.observer_provider_stats[arm] for r in results], arm,
        )
        stats = ObserverStats()
        watch = DagStoreWatch(max_nodes=cfg.dag_max_nodes)
        for r in results:
            stats.merge(r.observer_stats[arm])
            watch.merge(r.observer_dag_watch[arm])
        if conflicts:
            reasons.append(
                f"observer {arm} workers disagreed about arm configuration: {conflicts}",
            )
        # ⚑ TWO SURFACES, TWO KEY NAMES, AND `.get(..., 0)` WOULD HIDE BOTH.
        # The qsearch family publishes `dag_budget_trips` and FastQ publishes
        # `budget_trips`; a default of 0 on a missing key reports the HEALTHY
        # value about a counter that vanished. Subscript the one this arm's
        # surface owns.
        #
        # ⚑ AND THE SAME NUMBER MEANS DIFFERENT THINGS ON THE TWO SURFACES. A
        # non-zero `budget_trips` is FastQ-4+ DOING ITS JOB -- the node cap is
        # what the arm is -- while a non-zero `dag_budget_trips` means the
        # qsearch-DAG arm stood pat where the un-capped oracle searched, i.e.
        # the arm under test is not the arm the flag names. Both are published
        # with the key they came from rather than collapsed into one verdict.
        trip_key = (
            "budget_trips" if readout.ARM_SPECS[arm].stats_surface == "fastq"
            else "dag_budget_trips"
        )
        observers[arm] = {
            # ⚑ A DISCRIMINATOR, NOT DECORATION. The two families' blocks do not
            # have the same keys -- an `sf-` arm has no `provider_stats` at all
            # -- so a consumer must be able to branch on something other than
            # the arm's name spelling.
            "family": "nnue",
            "arm_config": next(
                c.consumed() for c in cfg.observers if c.arm == arm
            ),
            "arm_config_realized": readout._agree(
                [r.arm_config_realized[arm] for r in results], f"{arm} realized",
            ),
            "populations": asdict(stats),
            "observed_populations": [_ROLE_ROOT, _ROLE_CHILD, _ROLE_LEAF],
            "cost": _observer_cost(stats),
            "provider_stats": provider,
            "provider_stats_conflicts": conflicts,
            "node_budget_trip_counter": trip_key,
            "node_budget_trips": int(provider[trip_key]),
            # ⚑ `memory_peak` is the C layer's own figure, not `nodes_peak`
            # times a bytes-per-node constant. The store allocates a
            # power-of-two payload table, so the product understates the
            # resident bytes -- measured, 156,291 nodes held 1.44 GB against a
            # 0.89 GB product -- and publishing the estimate beside the
            # measurement would put a guess next to the thing it guessed at.
            "dag_store_watch": asdict(watch),
        }

    for sf_config in cfg.sf_observers:
        arm = sf_config.arm
        stats = ObserverStats()
        sf_stats = SfArmStats()
        for r in results:
            stats.merge(r.sf_observer_stats[arm])
            sf_stats.merge(r.sf_arm_stats[arm])
        if sf_stats.restarts:
            # ⚑⚑ AN INADMISSIBLE REASON, NOT A WARNING. A restarted engine
            # resumes with a COLD transposition table part-way through a game,
            # so the rows after it were produced by a measurably different
            # instrument from the rows before it -- and nothing in the shards
            # marks where the boundary was. The harness could not have skipped
            # the affected rows either: the cells must stay row-aligned, so
            # dropping a row from one cell would unpair every cell.
            reasons.append(
                f"observer {arm} restarted its Stockfish engine "
                f"{sf_stats.restarts} time(s) mid-run (first: "
                f"{sf_stats.first_restart_error}); the rows after a restart "
                "were labelled with a cold transposition table and are not the "
                "same instrument as the rows before it",
            )
        observers[arm] = {
            "family": "stockfish",
            "arm_config": sf_config.consumed(),
            "arm_config_realized": readout._agree(
                [r.arm_config_realized[arm] for r in results], f"{arm} realized",
            ),
            # ⚑ THE REALIZED BLOCK IS WEAKER HERE THAN FOR A NATIVE ARM, AND
            # SAYING SO IS THE POINT. UCI has no option readback, so `threads`
            # and `hash_mb` are the values written to the engine's stdin, not
            # values read out of it. The node budget IS observed at the
            # consumer, under `engine_stats.engine_nodes_*`.
            "arm_config_realized_is_a_readback": False,
            "node_budget_observed_at_the_engine": {
                "requested": int(sf_config.nodes),
                "reported_min": int(sf_stats.engine_nodes_min),
                "reported_max": int(sf_stats.engine_nodes_max),
                "reported_mean": (
                    sf_stats.engine_nodes_sum / sf_stats.engine_nodes_reported
                    if sf_stats.engine_nodes_reported else float("nan")
                ),
                "searches_reporting_nodes": int(sf_stats.engine_nodes_reported),
                "searches": int(sf_stats.searches),
            },
            "populations": asdict(stats),
            # ⚑ `leaf_positions` is 0 BY DESIGN, and this is what says so. The
            # 1-ply label is a function of the root's children; the leaf
            # population is `sims`-many positions per ply, so fanning an engine
            # over it would multiply the run's cost by the simulation count for
            # a column nothing reads.
            "observed_populations": [_ROLE_ROOT, _ROLE_CHILD],
            "cost": _observer_cost(stats),
            "engine": {
                "binary": str(sf_config.binary),
                "binary_sha256": sf_config.binary_sha256,
                "binary_source": sf_config.binary_source,
                "nodes": int(sf_config.nodes),
                "threads": int(sf_config.threads),
                "hash_mb": int(sf_config.hash_mb),
                "tt_cleared": "per_game",
                "one_engine_per": "(worker, arm)",
            },
            "engine_stats": asdict(sf_stats),
            "cp_mapping": _sf_cp_mapping(cfg),
        }

    if cfg.sf_observers:
        expected_sha = cfg.sf_observers[0].binary_sha256
        observed = sorted({r.sf_binary_sha256 for r in results})
        if observed != [expected_sha]:
            reasons.append(
                f"workers hashed a Stockfish binary {observed} that is not the "
                f"{expected_sha!r} the parent resolved before the run: the "
                "sf- cells were not labelled by one engine",
            )

    argmax_pairs: dict[str, ArgmaxAgreement] = {}
    for r in results:
        for key, value in r.argmax_pairs.items():
            argmax_pairs.setdefault(key, ArgmaxAgreement()).merge(value)
    # ⚑ NOT an inadmissible reason: a run whose cells pick the same move on
    # every row is a valid measurement of the arms and a REFUTATION of the
    # premise that deep SF could tell them apart. It is surfaced here because
    # the deep-SF pass costs hours and its answer for such a pair is already
    # known -- exactly 0.00 cp, from arithmetic rather than from Stockfish.
    zero_resolution = sorted(
        key for key, value in argmax_pairs.items()
        if value.rows > 0 and value.rows_disagreeing == 0
    )

    kernels = sorted({r.kernel for r in results})
    pack_file_shas = sorted({r.pack_file_sha256 for r in results})
    build_ids = {
        "_lc0_ext": sorted({r.lc0_ext_loaded_build_id for r in results}),
        "_nnue_ext": sorted({r.nnue_ext_loaded_build_id for r in results}),
        "_mcts_tree": sorted({r.mcts_ext_loaded_build_id for r in results}),
    }
    bad_build_ids = {k: v for k, v in build_ids.items() if len(v) != 1 or not v[0]}
    if bad_build_ids:
        reasons.append(
            f"workers did not execute one proven loaded native image: {bad_build_ids}",
        )
    if len(kernels) != 1:
        reasons.append(f"workers ran different NNUE kernels {kernels}")
    if len(pack_file_shas) != 1:
        reasons.append(f"workers mapped different pack bytes {pack_file_shas}")
    nice_realized = sorted({int(r.nice_realized) for r in results})
    if len(nice_realized) > 1:
        reasons.append(f"workers ran at different niceness {nice_realized}")

    terminations: dict[str, int] = {}
    for r in results:
        for key, value in r.terminations.items():
            terminations[key] = terminations.get(key, 0) + int(value)
    # ⚑ The workers ship the ACCUMULATOR, not its summary. Merging summaries
    # would mean re-deriving `legal_sum` from a published mean, which turns an
    # exact count into a rounded reconstruction of itself.
    budget = gen.RootBudgetStats()
    for r in results:
        budget.merge(r.root_budget)

    return {
        "driver": {
            "arm": cfg.driver.arm,
            "arm_config": cfg.driver.consumed(),
            "arm_config_realized": readout._agree(
                [r.arm_config_realized[cfg.driver.arm] for r in results],
                "driver realized",
            ),
            "provider_stats": driver_provider,
            "provider_stats_conflicts": driver_conflicts,
            "dag_store_watch": asdict(driver_watch),
        },
        "observers": observers,
        "shadow_inertness_proof": proof,
        "driver_observer_agreement": asdict(agreement),
        "cell_alignment": alignment,
        "label_argmax_agreement": {
            key: value.summary() for key, value in sorted(argmax_pairs.items())
        },
        "zero_resolution_pairs": zero_resolution,
        "games": sum(r.games for r in results),
        "plies": sum(r.plies for r in results),
        "rows_per_cell": sum(r.rows for r in results),
        # Every cell's shards with the content digest `shard_digest` reads back
        # through the real loader, so a later reanalysis can prove it scored the
        # bytes this run wrote.
        "shards": {
            cell: sorted(
                (s for r in results for s in r.shards.get(cell, [])),
                key=lambda s: int(s["index"]),
            )
            for cell in cfg.cells
        },
        "workers_active": len(results),
        "wall_s": wall_s,
        "search_wall_s": max((r.elapsed_s for r in results), default=0.0),
        "setup_wall_s": max((r.setup_s for r in results), default=0.0),
        "peak_rss_bytes_per_worker": {
            str(r.worker_id): int(r.peak_rss_bytes) for r in results
        },
        "peak_rss_bytes_max": max((int(r.peak_rss_bytes) for r in results), default=0),
        "terminations": terminations,
        "root_budget": budget.summary(),
        "games_digest": readout.games_digest(records),
        "searches_digest": readout.searches_digest(records),
        "games_detail": [asdict(rec) for rec in sorted(records, key=lambda r: r.game)],
        "bank_rows": sum(r.bank_rows for r in results),
        "kernel": kernels[0] if len(kernels) == 1 else kernels,
        "nice_realized": nice_realized,
        "inadmissible_reasons": reasons,
        "admissible": not reasons,
    }


def _cell_meta(cfg: RunConfig, cell: str, *, arm: str | None) -> dict[str, Any]:
    """What ``policy_target`` MEANS in this directory, beside the shards.

    A shard directory that does not say which label rule produced its policy
    column is a corpus whose only description lives in whatever report the
    reader happens to also have.
    """
    if arm is None:
        return {
            "cell": cell,
            "label_rule": "production Gumbel improved policy from the driver's search",
            "driver_arm": cfg.driver.arm,
            # ⚑ COMPARABLE AS A SAMPLE, NOT ROW FOR ROW, and the difference is
            # the seeding. `gen_random_selfplay_shards.run_worker` draws every
            # game of a worker from ONE stream; this harness seeds each game as
            # `seed + game_index`, which is what makes the same games replayable
            # with and without the observers -- and therefore what makes the
            # inertness proof possible at all. The LABEL RULE and the search
            # settings are the generator's; the game draws are not the same
            # games.
            "comparable_with_banked_gen0_cells": "distributionally, not row-for-row",
            "differs_from_the_generator_in": "per-game rng seeding (seed + game_index)",
            "run_id": cfg.run_id,
            "seed": int(cfg.seed),
            "sims_floor": int(cfg.sims),
            "all_root_moves": bool(cfg.all_root_moves),
            "prior": "uniform (no network; gen_random_selfplay_shards)",
        }
    meta: dict[str, Any] = {
        "cell": cell,
        "label_rule": (
            "softmax(oneply_sigma * -q(child)) over the root's legal moves, where "
            "q is this arm's value through the generator's cp-logistic"
        ),
        "arm": arm,
        "arm_family": "stockfish" if is_sf_arm(arm) else "nnue",
        "driver_arm": cfg.driver.arm,
        "oneply_sigma": float(cfg.oneply_sigma),
        "sigma_is_inert_for": "top1_regret_cp (a softmax cannot move an argmax)",
        "sigma_is_load_bearing_for": [
            "expected_regret_cp", "blunder_100", "blunder_300",
        ],
        "comparable_with_banked_gen0_cells": False,
        "run_id": cfg.run_id,
        "seed": int(cfg.seed),
        "prior": "uniform (no network; gen_random_selfplay_shards)",
    }
    sf_config = next((c for c in cfg.sf_observers if c.arm == arm), None)
    if sf_config is None:
        return meta
    # ⚑ A BARE CELL DIRECTORY IS WHAT TRAVELS, so everything that makes these
    # labels a DIFFERENT instrument from the native cells' has to be in it: the
    # engine's identity, its node budget, and -- the one this repository has a
    # banked burn about -- what happened to the transposition table.
    meta.update({
        "label_source": "stockfish search (observer only; never steered the tree)",
        "sf_binary": str(sf_config.binary),
        "sf_binary_sha256": sf_config.binary_sha256,
        "sf_binary_source": sf_config.binary_source,
        "sf_nodes": int(sf_config.nodes),
        "sf_threads": int(sf_config.threads),
        "sf_hash_mb": int(sf_config.hash_mb),
        "sf_tt_policy": {
            "cleared": "per_game",
            "command": "ucinewgame + isready at the start of every game",
            "not_cleared_between_positions": True,
            "why": (
                "per position costs a protocol round trip on each of the root's "
                "~30 children; per RUN reproduces the banked dirty-TT hazard "
                "(production SF labels run on a dirty TT). Per game is the "
                "cadence the driver's own games have."
            ),
            "a_restart_resets_it_mid_game": (
                "and is counted as an INADMISSIBLE reason, not absorbed"
            ),
        },
        "sf_position_state": (
            "UCI 'position fen' only: the 50-move clock travels (a FEN field), "
            "repetition history does NOT. See engine_stats."
            "positions_without_repetition_history in the run report."
        ),
        "sf_seat": (
            "UCI score is from the EVALUATED position's side to move, the same "
            "seat the native arms answer from; probe_root applies the root-mover "
            "negation for every arm alike"
        ),
        "cp_mapping": _sf_cp_mapping(cfg),
    })
    return meta


def run(cfg: RunConfig, *, prove_games: int) -> dict[str, Any]:
    if cfg.games <= 0 or cfg.workers <= 0 or cfg.max_plies <= 0:
        raise ValueError("games, workers and max_plies must be positive")
    if not cfg.pack.is_file():
        raise FileNotFoundError(cfg.pack)
    # ⚑ THE REFUSAL LIVES HERE AS WELL AS IN THE PARSER. The CLI's `--driver`
    # rejects an `sf-` name with its own message, but `run` is the entry point a
    # test or a sibling script calls, and a gate that only exists in argparse is
    # a gate the programmatic path does not have.
    if is_sf_arm(cfg.driver.arm):
        raise ValueError(_SF_DRIVER_REFUSAL.format(arm=cfg.driver.arm))
    if cfg.sf_observers:
        binaries = sorted({
            (str(c.binary), c.binary_sha256) for c in cfg.sf_observers
        })
        if len(binaries) != 1:
            raise ValueError(
                f"the sf- ladder names more than one engine binary: {binaries}. "
                "A ladder whose rungs are different engines measures the engine, "
                "not the node budget.",
            )
    if cfg.driver.arm not in cfg.arms:
        raise ValueError(
            f"the driver arm {cfg.driver.arm!r} must also be an observer arm: its "
            "observer context is the control that proves the observers see the "
            "driver's own positions",
        )
    gen.build_gumbel_config(_base_gen_config(cfg))
    for cell in cfg.cells:
        directory = cell_dir(cfg.out_dir, cell)
        if directory.exists() and any(directory.glob("shard_*")):
            raise FileExistsError(
                f"{directory} already holds shards; a rerun that appended would "
                "produce one cell whose rows came from two runs",
            )
        directory.mkdir(parents=True, exist_ok=True)

    pack_sha = readout._sha256_file(cfg.pack)
    git_start = readout._git_provenance()
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
  # Annotated: the skipped literal alone infers `dict[str, int | None | bool]`,
  # and the union with the proof's `dict[str, Any]` then rejects
  # `int(proof.get("games", 0))` at the cell_meta stamp below.
    proof: dict[str, Any] = (
        prove_shadow_inertness(cfg, games=prove_games) if prove_games > 0
        else {"games": 0, "digests_agree": None, "skipped": True}
    )
    results = _run_workers(cfg)
    wall_s = time.perf_counter() - started
    alignment = (
        assert_cells_are_row_aligned(cfg.out_dir, cfg.cells) if cfg.emit_shards
        else None
    )
    report = _aggregate(cfg, results, wall_s=wall_s, proof=proof, alignment=alignment)
    git_end = readout._git_provenance()
    if not (
        readout._git_provenance_available(git_start)
        and readout._git_provenance_available(git_end)
    ):
        report["inadmissible_reasons"].append(
            "tracked source provenance is unavailable at one or both endpoints",
        )
    elif git_start != git_end:
        report["inadmissible_reasons"].append(
            "tracked source provenance changed while the run was in flight",
        )
    worker_pack_shas = sorted({r.pack_file_sha256 for r in results})
    if worker_pack_shas != [pack_sha]:
        report["inadmissible_reasons"].append(
            f"workers mapped a pack whose file hash {worker_pack_shas} is not the "
            f"{pack_sha} the parent hashed before the run: the cells did not read "
            "one set of weights",
        )
  # ⚑⚑ NO PROOF, NO QUALITY CLAIM. `--prove-games 0` used to skip the
  # inertness proof and still publish `paired_evaluator_quality: true` and
  # `admissible: true` — the one field that authorises spending hours of deep
  # Stockfish asserted the one property this file exists to establish, without
  # it ever being checked. That is this repo's signature defect (a gate that
  # cannot fail), caught by the independent review of af15fc8e8. A skipped
  # proof is a legitimate thing to RUN (a quick look at throughput), but the
  # artifact it leaves must say what it is: unproven, inadmissible as a paired
  # ruler input.
    if bool(proof.get("skipped")):
        report["inadmissible_reasons"].append(
            "the shadow-inertness proof was skipped (--prove-games 0): "
            "paired_evaluator_quality is UNPROVEN for these cells — rerun with "
            "--prove-games >= 1 before spending any Stockfish on them",
        )
    report["admissible"] = not report["inadmissible_reasons"]
    report["schema"] = REPORT_SCHEMA
    proof_proven = bool(proof.get("digests_agree")) and not bool(proof.get("skipped"))
    report["quality_scope"] = {
        "population": "frozen_driver_shadow",
        "paired_evaluator_quality": proof_proven,
        "deep_sf_paired_input_admissible": bool(report["admissible"]) and proof_proven,
        "primary_ruler": "scratchpad/az_purity/score_shard_labels.py (top1_regret_cp)",
        "note": (
            "one driver chose every position; every arm labelled the same rows in "
            "the same order. The search_gumbel__ cell carries the production "
            "Gumbel target and is the only cell comparable with the banked gen-0 "
            "series; the oneply__ cells carry per-arm 1-ply targets whose "
            "secondary metrics depend on --oneply-sigma."
        ),
    }
    report["provenance"] = {
        "run_id": cfg.run_id,
        "started_utc": started_utc,
        "pack_path": str(cfg.pack),
        "pack_file_sha256": pack_sha,
        "pack_source_sha256": sorted({r.pack_source_sha256 for r in results}),
        "out_dir": str(cfg.out_dir),
        "cells": list(cfg.cells),
        "driver_arm": cfg.driver.arm,
        "observer_arms": list(cfg.arms),
        "nnue_observer_arms": list(cfg.nnue_arms),
        "sf_observer_arms": list(cfg.sf_arms),
        "sf_engine": (
            {
                "binary": str(cfg.sf_observers[0].binary),
                "binary_sha256": cfg.sf_observers[0].binary_sha256,
                "binary_source": cfg.sf_observers[0].binary_source,
                "threads": int(cfg.sf_observers[0].threads),
                "hash_mb": int(cfg.sf_observers[0].hash_mb),
                "node_ladder": [int(c.nodes) for c in cfg.sf_observers],
                "tt_cleared": "per_game",
            }
            if cfg.sf_observers else None
        ),
        "games": int(cfg.games),
        "workers_requested": int(cfg.workers),
        "seed": int(cfg.seed),
        "sims_floor": int(cfg.sims),
        "topk": int(cfg.topk),
        "max_plies": int(cfg.max_plies),
        "all_root_moves": bool(cfg.all_root_moves),
        "oneply_sigma": float(cfg.oneply_sigma),
        "dag_max_nodes": int(cfg.dag_max_nodes),
        "dag_reset": readout._dag_reset_label(cfg.dag_reset_every),
        "dag_bytes_per_node_measured": MEASURED_DAG_BYTES_PER_NODE,
        "shard_size": int(cfg.shard_size),
        "banking": cfg.bank_path is not None,
        "cp_per_internal_unit": float(cfg.cp_per_internal_unit),
        "cp_slope": float(cfg.cp_slope),
        "cp_draw_width": float(cfg.cp_draw_width),
        "nice_requested": int(cfg.nice),
        "prove_games": int(prove_games),
        **git_start,
        "python": sys.version.split()[0],
    }
    if cfg.emit_shards:
        for cell in cfg.cells:
            arm = None if cell.startswith(SEARCH_CELL_PREFIX) else cell[len(ONEPLY_CELL_PREFIX):]
            readout._atomic_write_text(
                cell_dir(cfg.out_dir, cell) / "cell_meta.json",
                json.dumps(
                    {
                        **_cell_meta(cfg, cell, arm=arm),
                        "report_schema": REPORT_SCHEMA,
                      # ⚑ The proof verdict TRAVELS WITH THE SHARDS. cell_meta
                      # is the only artifact a reader holding a bare cell
                      # directory has, and before this stamp it could not tell
                      # a proven-inert corpus from a --prove-games 0 one — the
                      # distinction that decides whether the rows are a valid
                      # paired-quality input at all (review of af15fc8e8, F1).
                        "shadow_inertness_proof": {
                            "games": int(proof.get("games", 0)),
                            "digests_agree": proof.get("digests_agree"),
                            "skipped": bool(proof.get("skipped", False)),
                        },
                        "paired_evaluator_quality": (
                            report["quality_scope"]["paired_evaluator_quality"]
                        ),
                    },
                    indent=2, sort_keys=True,
                ) + "\n",
            )
    for reason in report["inadmissible_reasons"]:
        _LOG.error("INADMISSIBLE: %s", reason)
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────


def _driver_arm_argument(text: str) -> str:
    """``--driver``'s ``type``, so an ``sf-`` name gets the REAL reason.

    argparse checks ``type`` before ``choices``, so this fires first and the
    operator is told that the arm is observer-only rather than that it is an
    "invalid choice" alongside three names that look nothing like it.
    """
    if is_sf_arm(text):
        raise argparse.ArgumentTypeError(_SF_DRIVER_REFUSAL.format(arm=text))
    return text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument(
        "--driver", type=_driver_arm_argument, choices=readout.READOUT_ARMS,
        default=readout.ARM_QSEARCH,
        help="the ONE arm that steers the search and chooses every position. "
             "An sf- arm is refused here: it is observer-only.",
    )
    p.add_argument(
        "--arm", choices=readout.READOUT_ARMS, action="append", default=None,
        help="an arm to LABEL with; repeatable. Defaults to all three. The "
             "driver arm is always included, as the control.",
    )
    p.add_argument(
        "--sf-observer", type=int, action="append", default=None, metavar="NODES",
        help="add a shallow-Stockfish OBSERVER arm at `go nodes NODES`, named "
             "sf-<NODES>; repeatable, once per rung of the ladder. It labels the "
             "same rows as every native arm and never steers the search, so a "
             "ladder makes label quality a function of label search effort on "
             "positions that are identical by construction. It sees ROOTS and "
             "ROOT CHILDREN only -- never the search leaves, which are "
             "sims-many per ply.",
    )
    p.add_argument(
        "--sf-binary", type=Path, default=None,
        help="the engine every sf- arm runs. Defaults to this repo's shared "
             "discovery (CAE_STOCKFISH, then the checkout's published engine, "
             "then the main checkout's, then PATH/distro).",
    )
    p.add_argument(
        "--sf-threads", type=int, default=DEFAULT_SF_THREADS,
        help=f"Threads per sf- engine (default {DEFAULT_SF_THREADS}). Above 1 a "
             "node-limited search stops being reproducible at a fixed budget, "
             "which makes the label a function of thread scheduling.",
    )
    p.add_argument(
        "--sf-hash-mb", type=int, default=DEFAULT_SF_HASH_MB,
        help=f"Hash per sf- engine (default {DEFAULT_SF_HASH_MB} MB, which is "
             "Stockfish's own default). Pinned rather than inherited so the "
             "table size is part of the artifact: the same node budget with a "
             "different table is a different search.",
    )
    p.add_argument("--nnue-pack", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--sims", type=int, default=gen.DEFAULT_SIMS)
    p.add_argument("--topk", type=int, default=gen.MAX_LEGAL_MOVES)
    p.add_argument("--max-plies", type=int, default=gen.DEFAULT_MAX_PLIES)
    p.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    p.add_argument(
        "--all-root-moves", action=argparse.BooleanOptionalAction, default=True,
        help="every legal root move is a candidate, so every root child is "
             "expanded; the 1-ply cells label the same move set either way, but "
             "the search cell's target is only complete with this on",
    )
    p.add_argument(
        "--oneply-sigma", type=float, default=None,
        help="sharpness of the 1-ply cells' softmax over q. INERT for "
             "top1_regret_cp; load-bearing for expected regret and blunder "
             f"rates. Default {oneply_sigma_default():.4g}, the production "
             "Gumbel target's own sigma at the pinned max-visit cap.",
    )
    p.add_argument(
        "--dag-max-nodes", type=int, default=DEFAULT_DAG_MAX_NODES,
        help="canonical-store watchdog: reset a DAG-backed arm once its "
             "node_count exceeds this. 0 disables it, which is how the DAG arm "
             "OOMs -- --dag-node-cap is a per-call quiescence budget and does "
             "NOT bound the store.",
    )
    p.add_argument(
        "--dag-reset", default="game",
        help="DAG persistence cadence between games: 'game', 'never' or "
             "'every-N-games'. Independent of --dag-max-nodes, which fires "
             "inside a game.",
    )
    p.add_argument(
        "--prove-games", type=int, default=4,
        help="games to replay with and without the observers before the run. A "
             "digest mismatch is a HARD ERROR. 0 skips the proof, which makes "
             "every paired claim in the output unproven.",
    )
    p.add_argument("--run-id", default="nnue_shadow_label_readout")
    p.add_argument("--nice", type=int, default=gen.DEFAULT_NICE)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument(
        "--bank-leaf-observations", type=Path, default=None,
        help="bank one JSONL row per evaluated position for the driver and for "
             "EVERY observer. The role and arm are in each file name.",
    )

    p.add_argument("--nnue-resolver-max-depth", type=int, default=None)
    p.add_argument("--nnue-qsearch-max-ply", type=int, default=None)
    p.add_argument("--nnue-qsearch-check-plies", type=int, default=None)
    p.add_argument("--dag-node-cap", type=int, default=None)
    p.add_argument(
        "--allow-binding-dag-node-cap", action="store_true",
        help="permit --dag-node-cap > 0 when nnue-qsearch-dag is the DRIVER, "
             "accepting that the capped arm is a different search",
    )
    p.add_argument("--fastq-max-qply", type=int, default=None)
    p.add_argument("--fastq-node-cap", type=int, default=None)
    p.add_argument("--fastq-delta-margin", type=int, default=None)
    p.add_argument("--fastq-recapture-exempt", type=int, choices=(0, 1), default=None)

    p.add_argument("--nnue-cp-per-unit", type=float, default=gen.NNUE_CP_PER_INTERNAL_UNIT)
    p.add_argument("--nnue-cp-slope", type=float, default=gen.NNUE_CP_SLOPE)
    p.add_argument("--nnue-cp-draw-width", type=float, default=gen.NNUE_CP_DRAW_WIDTH)
    return p


def resolve_sf_observers(args: argparse.Namespace) -> tuple[SfArmConfig, ...]:
    """The ``sf-`` ladder, with the engine resolved and hashed ONCE.

    ⚑ A DUPLICATE RUNG IS REFUSED, NOT DEDUPLICATED.  ``--sf-observer 512
    --sf-observer 512`` names one cell twice; silently collapsing it would leave
    the operator with a ladder that has fewer rungs than they asked for and a
    report that never says so.

    ⚑ A MISSING ENGINE IS REFUSED, NOT DEGRADED.  ``engine_discovery`` has a
    ``default_stockfish`` that returns a path whether or not anything is there,
    which is right for ``--help`` stability and wrong here: an sf- run with no
    engine must not reach the first ``StockfishUCI`` and die inside a pty
    handshake.
    """
    # ⚑ ATTRIBUTE ACCESS, NEVER `getattr(args, ..., None)`. A default here would
    # turn "this caller's Namespace predates the flag" into "no Stockfish arms
    # were requested", which is the same shape as reading `lr` with `.get`: the
    # single most consequential value in the call reported as absent. A
    # Namespace that has not got these keys must raise.
    requested = [int(n) for n in (args.sf_observer or ())]
    if not requested:
        return ()
    seen: set[int] = set()
    duplicates = sorted({n for n in requested if n in seen or seen.add(n)})
    if duplicates:
        raise ValueError(
            f"--sf-observer was given the same node budget twice: {duplicates}. "
            "Each rung is one cell, so a repeat would name one directory twice.",
        )
    bad = sorted(n for n in requested if n <= 0)
    if bad:
        raise ValueError(f"--sf-observer needs a positive node budget, got {bad}")
    threads = int(args.sf_threads)
    hash_mb = int(args.sf_hash_mb)
    if threads < 1:
        raise ValueError(f"--sf-threads must be >= 1, got {threads}")
    if hash_mb < 1:
        raise ValueError(f"--sf-hash-mb must be >= 1, got {hash_mb}")
    supplied = args.sf_binary
    if supplied is not None:
        binary_path: str | None = str(supplied)
        source = "explicit"
    else:
        binary_path, source = engine_discovery.resolve_stockfish()
    if not binary_path or not Path(binary_path).is_file():
        raise ValueError(
            "--sf-observer was requested but no Stockfish binary resolved "
            f"(tried {binary_path!r}, source {source!r}). Pass --sf-binary or "
            f"set ${engine_discovery.ENV_VAR}.",
        )
    identity = engine_discovery.engine_identity(binary_path)
    digest = identity["sha256"]
    if not digest:
        raise ValueError(
            f"could not hash the Stockfish binary at {binary_path}: an sf- cell "
            "whose engine has no content digest cannot be compared with the next "
            "run's, and the path alone is not an identity",
        )
    return tuple(
        SfArmConfig(
            arm=sf_arm_name(nodes),
            nodes=int(nodes),
            threads=threads,
            hash_mb=hash_mb,
            binary=Path(binary_path),
            binary_sha256=str(digest),
            binary_source=str(identity["source"]),
        )
        for nodes in requested
    )


def config_from_args(args: argparse.Namespace) -> RunConfig:
    driver = str(args.driver)
    if is_sf_arm(driver):
        raise ValueError(_SF_DRIVER_REFUSAL.format(arm=driver))
    arms = list(dict.fromkeys(args.arm or list(readout.READOUT_ARMS)))
    if driver not in arms:
        arms.insert(0, driver)
    readout._validate_matrix_knobs(args, arms)
    # ⚑ The DAG cap guard is the driver's alone. On an OBSERVER a binding cap
    # makes the arm under test the capped arm -- a different arm, published as
    # `per_call_node_cap_bound` -- which is a legitimate thing to measure. On
    # the DRIVER it changes which leaves exist for everybody.
    if (
        driver == readout.ARM_QSEARCH_DAG
        and args.dag_node_cap is not None
        and int(args.dag_node_cap) > 0
        and not bool(args.allow_binding_dag_node_cap)
    ):
        raise ValueError(
            "--dag-node-cap > 0 makes the nnue-qsearch-dag DRIVER stand pat where "
            "the un-capped arm searched, so the frozen position set would be the "
            "capped arm's. Pass --allow-binding-dag-node-cap to accept that.",
        )
    sigma = (
        oneply_sigma_default() if args.oneply_sigma is None else float(args.oneply_sigma)
    )
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            f"--oneply-sigma must be finite and > 0, got {sigma!r}: a "
            "non-positive sigma flattens or inverts every 1-ply target without "
            "failing any check",
        )
    if int(args.shard_size) <= 0:
        raise ValueError("--shard-size must be positive")
    if int(args.max_plies) <= 0:
        raise ValueError("--max-plies must be positive")
    if int(args.dag_max_nodes) < 0:
        raise ValueError("--dag-max-nodes must be >= 0 (0 disables the watchdog)")
    # ⚑ THE SHARED RESOLVER'S DAG-CAP REFUSAL IS ABOUT THE SIBLING TOOL'S
    # DECOMPOSITION, NOT ABOUT THIS ONE. `nnue_gumbel_readout` needs
    # nnue-qsearch-dag to stay bit-identical to nnue-qsearch, so any binding cap
    # voids it. Here the DAG arm is an arm UNDER TEST: a capped arm is simply a
    # different arm, it labels the same rows as every other, and its trips are
    # published as `node_budget_trips`. So observers are resolved with the
    # allowance on and the DRIVER is guarded above, where a cap really does
    # change which positions exist for everybody.
    observer_args = argparse.Namespace(
        **{**vars(args), "allow_binding_dag_node_cap": True},
    )
    return RunConfig(
        driver=readout.resolve_arm_config(
            observer_args, arm=driver, strict_foreign_knobs=False,
        ),
        observers=tuple(
            readout.resolve_arm_config(
                observer_args, arm=a, strict_foreign_knobs=False,
            )
            for a in arms
        ),
        pack=Path(args.nnue_pack),
        out_dir=Path(args.out_dir),
        games=int(args.games),
        workers=int(args.workers),
        seed=int(args.seed),
        sims=int(args.sims),
        topk=int(args.topk),
        max_plies=int(args.max_plies),
        all_root_moves=bool(args.all_root_moves),
        cp_per_internal_unit=float(args.nnue_cp_per_unit),
        cp_slope=float(args.nnue_cp_slope),
        cp_draw_width=float(args.nnue_cp_draw_width),
        oneply_sigma=sigma,
        dag_max_nodes=int(args.dag_max_nodes),
        dag_reset_every=readout.parse_dag_reset(str(args.dag_reset)),
        shard_size=int(args.shard_size),
        bank_path=(
            None if args.bank_leaf_observations is None
            else Path(args.bank_leaf_observations)
        ),
        run_id=str(args.run_id),
        nice=int(args.nice),
        sf_observers=resolve_sf_observers(args),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    report = run(cfg, prove_games=int(args.prove_games))
    text = json.dumps(report, indent=2, sort_keys=True, default=_json_default)
    print(text)
    if args.json is not None:
        readout._atomic_write_text(Path(args.json), text + "\n")
    # The artifact is written first and the failure reported after, for the same
    # reason `nnue_gumbel_readout.main` does it: a gate that raises before the
    # JSON exists destroys the evidence for the finding it just made.
    return 0 if report["admissible"] else 2


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"cannot serialise {type(value).__name__} into the report")


if __name__ == "__main__":
    raise SystemExit(main())
