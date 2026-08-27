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
from chess_anti_engine.utils.numpy_helpers import softmax_1d
from scripts import gen_random_selfplay_shards as gen
from scripts import nnue_gumbel_readout as readout

_LOG = logging.getLogger("nnue_shadow_label_readout")

#: Report schema. 1 is the first shape that reports paired label quality at all;
#: `nnue_gumbel_readout`'s schema counter is independent of this one and the two
#: files' reports are not interchangeable.
REPORT_SCHEMA = 1

SEARCH_CELL_PREFIX = "search_gumbel__"
ONEPLY_CELL_PREFIX = "oneply__"

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

    def __init__(self, *, source: readout.ReadoutArmSource, dag_max_nodes: int) -> None:
        self.source = source
        self.arm = source.arm
        self.stats = ObserverStats()
        self.dag_watch = DagStoreWatch(max_nodes=int(dag_max_nodes))

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
        observers: tuple[ObserverArm, ...],
        recorder: ProbeRecorder,
        agreement: AgreementStats,
        dag_watch: DagStoreWatch,
        **kwargs: Any,
    ) -> None:
        self._observers = tuple(observers)
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
        for observer in self._observers:
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

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(c.arm for c in self.observers)

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
    observers: tuple[ObserverArm, ...] | None = None,
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
    driver = _build_source(
        config=cfg.driver,
        cfg=cfg,
        bank=_worker_bank_path(
            cfg.bank_path, role="driver", arm=cfg.driver.arm, worker_id=spec.worker_id,
        ),
        identity={**identity, "observer_arm": ""},
        pack_sha=pack_sha,
        observers=tuple(observers),
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
        for observer in observers:
            observer.close()
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
        bank_rows=int(driver.bank_rows) + sum(int(o.source.bank_rows) for o in observers),
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
        },
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
    without = driver_digests(
        _run_workers(replace(probe, attach_observers=False, observers=())),
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
    for arm in cfg.arms:
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
            "arm_config": next(
                c.consumed() for c in cfg.observers if c.arm == arm
            ),
            "arm_config_realized": readout._agree(
                [r.arm_config_realized[arm] for r in results], f"{arm} realized",
            ),
            "populations": asdict(stats),
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
    return {
        "cell": cell,
        "label_rule": (
            "softmax(oneply_sigma * -q(child)) over the root's legal moves, where "
            "q is this arm's value through the generator's cp-logistic"
        ),
        "arm": arm,
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


def run(cfg: RunConfig, *, prove_games: int) -> dict[str, Any]:
    if cfg.games <= 0 or cfg.workers <= 0 or cfg.max_plies <= 0:
        raise ValueError("games, workers and max_plies must be positive")
    if not cfg.pack.is_file():
        raise FileNotFoundError(cfg.pack)
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
    proof = (
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
    report["admissible"] = not report["inadmissible_reasons"]
    report["schema"] = REPORT_SCHEMA
    report["quality_scope"] = {
        "population": "frozen_driver_shadow",
        "paired_evaluator_quality": True,
        "deep_sf_paired_input_admissible": bool(report["admissible"]),
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
                    {**_cell_meta(cfg, cell, arm=arm), "report_schema": REPORT_SCHEMA},
                    indent=2, sort_keys=True,
                ) + "\n",
            )
    for reason in report["inadmissible_reasons"]:
        _LOG.error("INADMISSIBLE: %s", reason)
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument(
        "--driver", choices=readout.READOUT_ARMS, default=readout.ARM_QSEARCH,
        help="the ONE arm that steers the search and chooses every position",
    )
    p.add_argument(
        "--arm", choices=readout.READOUT_ARMS, action="append", default=None,
        help="an arm to LABEL with; repeatable. Defaults to all three. The "
             "driver arm is always included, as the control.",
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


def config_from_args(args: argparse.Namespace) -> RunConfig:
    driver = str(args.driver)
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
