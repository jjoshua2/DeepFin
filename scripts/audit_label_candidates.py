#!/usr/bin/env python3
"""Audit-first kill/keep gate for NNUE bootstrap-label CANDIDATES.

``docs/eval_protocol.md``'s standing rule is that every training-target
candidate is scored against the FROZEN deep-SF audit set BEFORE any training
compute, and one that loses the direct audit is killed without training. The
NNUE-bootstrap lane had no such gate: ``scripts/nnue_shadow_label_readout.py``
produces paired per-arm label CELLS (shards), and scoring those needs a fresh
deep-SF pass over positions the arms themselves generated. This file is the
cheap half that comes first -- the arms scored on positions that 1M-node
MultiPV Stockfish labelled months ago, at zero new Stockfish cost.

WHAT IT MEASURES, IN ONE SENTENCE: for each frozen audit position, every
candidate labeller ranks the root's legal moves by its OWN value, and the
resulting move is scored against the banked deep-SF labels with
``scripts/audit_targets.py``'s own functions.

⚑ THE 1-PLY LABEL RULE IS THE SHADOW HARNESS'S, IMPORTED RATHER THAN RESTATED.
``nnue_shadow_label_readout.probe_root`` evaluates every legal child and
NEGATES to the root mover's seat; ``oneply_policy_vector`` turns those values
into the same ``softmax(sigma * -q(child))`` target its ``oneply__`` cells
carry. A second copy of that convention is how a sign error ships: an
un-negated target ranks the root's moves exactly backwards and is still a
well-formed probability vector over the right move set, so nothing downstream
raises. The only thing this file supplies is the EVALUATOR.

⚑ AND THE METRICS ARE ``audit_targets``'s, FOR THE SAME REASON.
``eval.audit.move_regrets`` (the censoring rule for unlisted moves, and the
1000cp cap), ``audit_targets.sf_reference_sets`` (score-tied top1/top10 sets)
and ``eval.audit.expected_and_top1_regret`` are imported, not reproduced, so a
row this tool emits is definitionally the same shape as a row
``audit_targets.py --dump-per-position`` emits and the comparison against the
banked production-target rows is a TABLE JOIN. No such comparison is computed
here: those numbers are banked elsewhere and re-deriving them under a different
ruler is exactly the failure the shared instrument prevents.

THE CENSORING RULE, STATED BECAUSE IT BOUNDS EVERY REGRET NUMBER BELOW.
``move_regrets`` gives a legal move the deep MultiPV did not list the regret of
the WORST LISTED line -- a FLOOR, biased optimistic -- and clamps every regret
to ``AUDIT_REGRET_CAP_CP`` (1000cp) so a mate line does not turn a mean over
positions into a mate detector. The frozen set is MultiPV=10, so an arm that
picks an 11th-ranked move is credited with the 10th-ranked move's regret and
its true cost is unknown. ``top1_move_unlisted_rate`` is reported per arm for
exactly that reason: it is the fraction of the headline regret that is a lower
bound rather than a measurement, and a candidate whose rate is high has a
regret number that cannot be compared with a candidate whose rate is low.

⚑⚑ THE DIRTY-TT SIDECAR IS NEVER OPENED. ``audit_targets.py`` caches SHALLOW
Stockfish labels beside the audit set; that cache was produced on a shared,
warm transposition table and is banned as a label source by standing rule. This
tool reads the frozen deep set and nothing else, refuses an ``--audit-set``
pointing at the cache, and ``tests/test_audit_label_candidates.py`` proves by
EXECUTION -- every ``open`` during a full run is recorded -- that no path
naming it is touched.

THE ARMS
--------
``nnue-static`` / ``nnue-qsearch`` / ``nnue-qsearch-dag`` / ``nnue-fastq``
    The native NNUE providers, opened through the same two classes the
    generator and the throughput readout use, with the same
    requested-vs-realized knob check. FastQ's four knobs are settable
    (``--fastq-max-qply`` and friends) so a FastQ quality ladder runs through
    this gate rather than beside it.

``sf-<nodes>`` / ``sf-d<depth>`` (e.g. ``sf-512``, ``sf-d9``) -- PER-CHILD
    The SAME 1-ply construction with Stockfish ``go nodes <N>`` -- or
    ``go depth <D>`` for the ``-d`` spelling -- on each CHILD as the
    evaluator. The score is read off the final info line, mates go
    through ``stockfish.wdl.mate_to_effective_cp`` (THE single mate home, the
    one ``parse_audit_record`` used to build the frozen set's own labels), and
    the effective cp then goes through the SAME ``cp_to_wdl_array`` the native
    arms' ``q_from_values`` uses. Evaluator swapped, label rule identical --
    that is the point of the arm, and
    ``test_the_sf_arm_and_the_nnue_arm_share_one_cp_mapping_object`` proves the
    two share one function object by execution rather than by inspection.

``sfroot-<nodes>[-mpv<W>]`` / ``sfroot-d<depth>[-mpv<W>]`` -- ROOTED
    (e.g. ``sfroot-2048-mpv20``, ``sfroot-d9-mpv20``.)
    ONE ``go nodes <N>`` -- or ``go depth <D>`` -- on the POSITION at MultiPV
    ``W``, and the arm's ranking is the MultiPV list. It shares one tree
    across the PVs, so it is
    cheaper and stronger per node than the per-child arm -- and it is a
    DIFFERENT construction, which is why both are in the table: the delta
    between them is the thing being measured, not an implementation detail.

    ``W`` defaults to 20 (the banked label-width finding: MultiPV 20 reaches
    95.3% of the bad-tail mass against MultiPV 6's 60.9%, and the width cost is
    sublinear -- ~7x at MultiPV 40). ``-mpvall`` sets MultiPV to the position's
    own legal-move count, per position. The realized width is recorded per arm
    beside the wall time, so width-versus-cost is measured rather than assumed.

    ⚑⚑ THE SEAT IS THE OTHER WAY ROUND, AND THAT IS THE WHOLE HAZARD OF THIS
    ARM. A rooted MultiPV search reports each root move's score from the ROOT
    MOVER's POV; the per-child arms report from the CHILD's and ``probe_root``
    negates. So the rooted arm's values go into ``oneply_policy_vector``
    UN-negated. Pinned by
    ``test_the_rooted_arm_reads_scores_from_the_root_movers_seat``.

    ⚑ AND THE DEPTH IS READ AS A COMPLETE SET, NEVER AS "THE LAST LINES SEEN".
    A node-limited search stops mid-iteration, so the tail of the stream is
    typically depth D's ranks 1..k next to depth D-1's ranks 1..W. Keeping the
    last line per rank -- which is what ``StockfishUCI``'s own accumulator does,
    correctly, for its own single-PV purpose -- would build a ranking out of two
    different searches. ``rooted_ranking_from_info_lines`` takes the DEEPEST
    depth whose MultiPV set is complete, and says so in the stats.

    ⚑ AND THAT RULE IS NOT RELAXED FOR THE ``-d`` ARMS. A depth-limited search
    emits the SAME per-depth MultiPV blocks -- it simply stops after the one it
    was asked for -- so the final complete set at the requested depth is what
    the same rule selects, and the incomplete fallback still covers the search
    that ends early (a proven mate, a tablebase hit, an engine that ran out of
    root moves at the requested width). Reading "the last lines seen" would be
    the identical bug here, and reading "the requested depth" would be worse:
    it would report the ASK as the realized depth of a search that never got
    there.

⚑ TT HYGIENE IS DISCLOSED, NOT ASSUMED. One persistent engine per Stockfish
arm, ``Threads=1``, and a ``ucinewgame`` + ``Clear Hash`` at run start, so the
run begins cold. WITHIN the run the transposition table is shared across
searches -- that is what "one persistent engine" means, and this repo has a
banked finding about labels produced on a warm shared TT. The verdict is
stamped in each arm's ``tt_hygiene``; ``--sf-fresh-per-position`` buys a cold
TT per position at a large cost in wall time.

Usage::

    PYTHONPATH=. nice -n 15 python3 scripts/audit_label_candidates.py \\
        --audit-set data/audit_set_v1.jsonl \\
        --nnue-pack data/nnue/nn-f68ec79f0fe3.pack \\
        --arms nnue-static,nnue-fastq,nnue-qsearch,sf-512,sfroot-2048-mpv20 \\
        --limit 50 --json /tmp/gate.json --dump-per-position /tmp/gate_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import chess
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.eval.audit import (
    AUDIT_REGRET_CAP_CP,
    AuditPosition,
    expected_and_top1_regret,
    legal_full_indices,
    load_audit_set,
    move_regrets,
)
from chess_anti_engine.stockfish.uci import (
    StockfishResult,
    StockfishUCI,
    _parse_info_fields,
)
from chess_anti_engine.stockfish.wdl import mate_to_effective_cp
from chess_anti_engine.utils.engine_discovery import find_stockfish
from scripts import audit_targets
from scripts import gen_random_selfplay_shards as gen
from scripts import nnue_gumbel_readout as readout
from scripts import nnue_shadow_label_readout as shadow

_LOG = logging.getLogger("audit_label_candidates")

#: Report schema. 1 is the first shape that scores label CANDIDATES (rather
#: than a trained net's policy head) against the frozen deep-SF set; a consumer
#: that reads an ``audit_targets`` report's keys off this file gets a KeyError
#: rather than a plausible wrong number.
#:
#: ⚑ THE DEPTH ARMS DID NOT BUMP IT, deliberately. They ADD keys
#: (``limit_kind``, ``depth_requested``, and the per-child arm's realized
#: ``depth_*``) and change no existing one: every ``sf-<nodes>`` /
#: ``sfroot-<nodes>`` column a schema-1 consumer already reads is byte-for-byte
#: what it was. ``nodes`` is null only on an arm spelling that did not exist at
#: schema 1, so no reader can be holding a number for it.
REPORT_SCHEMA = 1

#: The native arms this gate can open. ``nnue-static`` comes from the generator
#: (it is one of the two arms ``NnueArmValueSource`` was written for) and the
#: other three from the readout's widened source; both lists are imported so a
#: new arm in either place appears here without an edit.
NATIVE_ARMS: tuple[str, ...] = (
    gen.VALUE_SOURCE_NNUE_STATIC, *readout.READOUT_ARMS,
)

#: The arms that intern a canonical DAG store, and therefore the ONLY consumers
#: of ``--dag-max-nodes`` and ``--dag-reset-every``. Derived from the readout's
#: own ``ArmSpec.uses_dag`` for the same reason ``NATIVE_ARMS`` is derived: a new
#: DAG-backed arm there must not need an edit here to be recognised.
#:
#: ⚑ IT IS NOT ``{nnue-qsearch-dag}``. ``nnue-fastq`` is DAG-backed too, and
#: ``open_arms`` hands it a live ``dag_source`` -- so a hand-written set that
#: named only the obvious arm would refuse a knob the selected arm does read,
#: which is the same defect as accepting one it does not.
DAG_BACKED_ARMS: frozenset[str] = frozenset(
    name for name, spec in readout.ARM_SPECS.items() if spec.uses_dag
)

#: ``--dag-reset-every``'s default: drop a DAG-backed arm's memo every position.
#: It lives here rather than in ``add_argument`` because the flag's argparse
#: default is ``None`` -- "not supplied", which is what lets ``validate_knobs``
#: tell an explicit value from an absent one -- so the number itself needs a home
#: both the resolver and a test can read.
DEFAULT_DAG_RESET_EVERY = 1

#: ``sf-<nodes>``. The node budget is IN THE ARM NAME rather than in a flag,
#: because a ladder is several arms in one run and a single ``--sf-nodes``
#: would silently give them all the same budget.
SF_ARM_PREFIX = "sf-"

#: ``sfroot-<nodes>[-mpv<W|all>]``: one rooted MultiPV search per position
#: instead of one search per child. Same reason for the same spelling.
SFROOT_ARM_PREFIX = "sfroot-"

#: The ``d`` that turns a node budget into a DEPTH budget: ``sf-d9``,
#: ``sfroot-d9-mpv20``. It is a separate character rather than a separate flag
#: for the reason the budget itself is in the name -- a depth ladder is several
#: arms in one run -- and it is a PREFIX of the number rather than a suffix so
#: that ``sf-d9`` and ``sf-9`` cannot be confused by a reader OR by the regex:
#: the two parse to different limits, publish different arm names, and open
#: different engines.
SF_DEPTH_MARKER = "d"

#: Default rooted MultiPV width. The banked label-width finding is that MultiPV
#: 20 reaches 95.3% of the bad-tail mass against MultiPV 6's 60.9%, and the cost
#: is sublinear (~7x at MultiPV 40) -- so the default buys the tail rather than
#: the cheapest number. ``-mpvall`` widens it to the position's legal-move count.
DEFAULT_ROOTED_MULTIPV = 20

#: ``width is None`` means "as many PVs as this position has legal moves",
#: resolved PER POSITION and recorded as realized.
MULTIPV_ALL = "all"

_SF_ARM_RE = re.compile(r"^sf-(d?)(\d+)$")
_SFROOT_ARM_RE = re.compile(r"^sfroot-(d?)(\d+)(?:-mpv(\d+|all))?$")

#: The one position "cluster" every probe belongs to. The shadow harness keys a
#: probe by (game, ply); audit positions are independent draws, so the game is
#: constant and the ordinal is the row index.
_CLUSTER_GAME = 0


@dataclass(frozen=True)
class SfArmSpec:
    """A parsed Stockfish arm name, canonicalised.

    ``name`` is what the report keys on. ``sfroot-2048`` and
    ``sfroot-2048-mpv20`` are the SAME arm, so both parse to the canonical
    spelling -- otherwise a run naming both would open two engines, pay twice,
    and publish two identical columns as if they were a comparison.

    ⚑ EXACTLY ONE OF ``nodes`` / ``depth`` IS SET, enforced in ``__post_init__``
    rather than left to the parser. The two are not interchangeable budgets:
    ``go nodes N`` stops mid-iteration at a depth nobody chose, ``go depth D``
    runs the iteration out at a node count nobody chose, and a spec carrying
    both would let one silently become the other on any path that reads the
    field it happens to prefer. ``sf-d9`` and ``sf-9`` are therefore two
    different arms with two different names, never aliases.
    """

    rooted: bool
    nodes: int | None
    depth: int | None
    width: int | None
    name: str

    def __post_init__(self) -> None:
        if (self.nodes is None) == (self.depth is None):
            raise ValueError(
                f"{self.name!r}: a Stockfish arm carries exactly one search "
                f"limit, got nodes={self.nodes!r} depth={self.depth!r}",
            )

    @property
    def limit_kind(self) -> str:
        """``"nodes"`` or ``"depth"`` -- what the ``go`` line limits on."""
        return "nodes" if self.nodes is not None else "depth"

    @property
    def go_limit(self) -> str:
        """The ``go`` line's limit clause: ``nodes <N>`` or ``depth <D>``."""
        return (
            f"nodes {self.nodes}" if self.nodes is not None
            else f"depth {self.depth}"
        )


def _parse_sf_limit(
    arm: str, marker: str, digits: str,
) -> tuple[int | None, int | None]:
    """``("d", "9")`` -> ``(None, 9)``; ``("", "512")`` -> ``(512, None)``.

    The ``(nodes, depth)`` pair ``SfArmSpec`` then refuses if it is not exactly
    one. A non-positive budget is refused HERE and in the arm's own words,
    because ``sf-d0`` is a typo a reader of the report could never recover
    from: a zero depth is a limit Stockfish quietly replaces with a real
    iteration.
    """
    value = int(digits)
    if marker == SF_DEPTH_MARKER:
        if value <= 0:
            raise ValueError(f"{arm!r}: the depth budget must be positive")
        return None, value
    if value <= 0:
        raise ValueError(f"{arm!r}: the node budget must be positive")
    return value, None


def parse_sf_arm(arm: str) -> SfArmSpec | None:
    """``"sf-512"`` / ``"sfroot-d9-mpv20"`` -> a spec; anything else -> None."""
    per_child = _SF_ARM_RE.match(arm)
    if per_child is not None:
        nodes, depth = _parse_sf_limit(arm, per_child.group(1), per_child.group(2))
        return SfArmSpec(
            rooted=False, nodes=nodes, depth=depth, width=None, name=arm,
        )
    rooted = _SFROOT_ARM_RE.match(arm)
    if rooted is None:
        return None
    nodes, depth = _parse_sf_limit(arm, rooted.group(1), rooted.group(2))
    raw_width = rooted.group(3)
    if raw_width is None:
        width: int | None = DEFAULT_ROOTED_MULTIPV
    elif raw_width == MULTIPV_ALL:
        width = None
    else:
        width = int(raw_width)
        if width <= 0:
            raise ValueError(f"{arm!r}: the MultiPV width must be positive")
    label = MULTIPV_ALL if width is None else str(width)
    # ⚑ THE LIMIT TOKEN IS REBUILT FROM THE PARSED LIMIT, not copied out of the
    # input, so the canonical name of a depth arm keeps its `d`. Dropping it
    # here would canonicalise `sfroot-d9` onto `sfroot-9-mpv20` -- two different
    # searches publishing one column, which is the collision the whole
    # canonicalisation exists to prevent rather than to cause.
    limit_token = f"{SF_DEPTH_MARKER}{depth}" if nodes is None else str(nodes)
    return SfArmSpec(
        rooted=True, nodes=nodes, depth=depth, width=width,
        name=f"{SFROOT_ARM_PREFIX}{limit_token}-mpv{label}",
    )


def q_from_effective_cp(
    eff_cp: np.ndarray, *, slope: float, draw_width_cp: float,
) -> np.ndarray:
    """Effective cp -> q in [-1, 1], through the NATIVE ARMS' OWN mapping.

    ⚑ ``gen.cp_to_wdl_array`` is reached as a module ATTRIBUTE, at call time, on
    purpose. ``NnueArmValueSource.q_from_values`` resolves the same name out of
    the same module namespace, so the two paths are one function object and a
    test can prove it by replacing that one object and watching both answers
    move. A local ``from ... import cp_to_wdl_array`` here would bind a second
    reference and the proof would silently stop covering the Stockfish arms.

    W - L, dropping the draw channel, exactly as ``q_from_values`` does.
    """
    wdl = gen.cp_to_wdl_array(
        np.asarray(eff_cp, dtype=np.float64),
        slope=float(slope), draw_width_cp=float(draw_width_cp),
    )
    return wdl[..., 0].astype(np.float64) - wdl[..., 2].astype(np.float64)


def effective_cp_from_score(cp: int | None, mate: int | None) -> float | None:
    """One UCI score pair -> effective cp, or ``None`` when neither was given.

    ⚑ MATE FIRST, and ``is not None`` rather than truthiness. ``score mate 0``
    is a real score, and ``if mate:`` routes it to the cp branch where a missing
    cp then reads as "unscored". Mates go through ``mate_to_effective_cp``, the
    single mate home -- the same function ``eval.audit.parse_audit_record`` used
    to build the frozen set's own ``move_cp``, so the two bands agree.
    """
    if mate is not None:
        return float(mate_to_effective_cp(int(mate)))
    if cp is not None:
        return float(cp)
    return None


#: A checkmated position's effective cp, from the MATED side's seat. Measured
#: against ``arm_handle_eval``: a checkmated child reads -100000 internal units
#: and ``q_from_values`` maps it to q = -1. ``mate_to_effective_cp(0)`` is the
#: positive end of the same band, so the sign is applied here rather than folded
#: into the argument -- the identical trap ``q_from_values`` documents, where a
#: mate AT depth 0 has zero plies and both colours would come back as a WIN.
MATED_EFFECTIVE_CP = -abs(mate_to_effective_cp(0))


def terminal_effective_cp(board: CBoard) -> float | None:
    """A game-over board's effective cp from ITS OWN side-to-move seat, or None.

    The values are the native arm's, measured rather than chosen -- see the
    module docstring. ``is_game_over`` covers stalemate, insufficient material
    and the fifty-move rule, and the arm returns 0 for all three.
    """
    if board.is_checkmate():
        return MATED_EFFECTIVE_CP
    if board.is_game_over():
        return 0.0
    return None


# ── the label one arm produces for one position ──────────────────────────────


@dataclass(frozen=True)
class Label:
    """One arm's answer about one position, in ``audit``'s own move order.

    ``probs`` is a distribution over ``legal_ucis``; ``chosen`` is the arm's
    move. They agree for every construction whose move IS the argmax, and the
    scorer counts the rows where they do not rather than picking one silently.
    """

    probs: np.ndarray
    chosen: str
    #: The lowest-level per-move observation this arm made, keyed by uci, and
    #: what it is. Banked in the per-position dump so a later estimator is a
    #: re-read rather than a rerun against a deterministic search.
    values: dict[str, float]
    values_kind: str


@dataclass
class ArmCost:
    """Wall time and volume, split by population.

    ROOT and CHILD are counted apart because they are different populations
    with different sizes: the root is one board per position and the children
    are the whole legal move list. Pooling them would put a per-position figure
    inside a per-evaluation rate.

    ⚑ THE PROBE'S ROOT EVALUATION IS DISCARDED AND STILL COUNTED.
    ``probe_root`` evaluates the root before the children (it is where the
    harness's arms get their root counter), the 1-ply label does not use the
    answer, and a cost axis that hid it would understate what running that
    labeller costs by one evaluation per position.
    """

    root_positions: int = 0
    child_positions: int = 0
    batches: int = 0
    eval_s: float = 0.0

    def add(self, role: str, n: int, seconds: float) -> None:
        if role == shadow._ROLE_ROOT:
            self.root_positions += int(n)
        else:
            self.child_positions += int(n)
        self.batches += 1
        self.eval_s += float(seconds)

    def summary(self, positions: int) -> dict[str, float]:
        evaluations = self.root_positions + self.child_positions
        return {
            "eval_s_total": self.eval_s,
            "eval_s_per_position": self.eval_s / positions if positions else math.nan,
            "eval_s_per_board": (
                self.eval_s / evaluations if evaluations else math.nan
            ),
            "root_positions": float(self.root_positions),
            "child_positions": float(self.child_positions),
            "batches": float(self.batches),
        }


# ── arms ─────────────────────────────────────────────────────────────────────


class ReportableArm(Protocol):
    """What the run loop and the report need of ANY arm."""

    arm: str

    def begin_position(self, *, reset_memo: bool) -> None: ...

    def cost(self) -> ArmCost: ...

    def stamp(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class ChildArm(ReportableArm, Protocol):
    """An arm ``probe_root`` can drive: a batched value over the root's children.

    Structurally a ``shadow.ArmObserver``. Every such arm is handed ONE
    ``probe_root`` call per position, so the position set is identical across
    them by construction rather than by argument.
    """

    def evaluate(
        self, boards: list[CBoard], *, role: str, cluster: tuple[int, int] | None,
    ) -> np.ndarray: ...


class SearchEngine(Protocol):
    """What a Stockfish arm needs of an engine.

    Narrower than ``StockfishUCI`` on purpose: the arm needs a search and a
    close, and a test that has to build a whole UCI subprocess to check a sign
    convention is a test nobody runs.
    """

    hash_mb: int | None

    def search(
        self, fen: str, *, nodes: int | None = ..., depth: int | None = ...,
    ) -> StockfishResult: ...

    def new_game(self) -> None: ...

    def close(self) -> None: ...


class NnueCandidateArm:
    """A native NNUE arm as a ``ChildArm``.

    ⚑ IT ADDS NO EVALUATION LOGIC. ``q_for_boards`` is the source's, which is
    the generator's, which is where the cp mapping and the mate band live. This
    wrapper times the call, counts the population and drives the DAG store
    watchdog -- and the watchdog is ``shadow.DagStoreWatch``, imported, because
    ``--dag-node-cap`` is a PER-CALL quiescence budget and does not bound the
    canonical store that OOMs (that measurement is the shadow harness's).
    """

    def __init__(
        self,
        *,
        source: gen.NnueArmValueSource,
        dag_source: readout.ReadoutArmSource | None,
        dag_max_nodes: int,
    ) -> None:
        self.source = source
        self.arm = source.arm
        self._dag_source = dag_source
        self._cost = ArmCost()
        self.dag_watch = shadow.DagStoreWatch(max_nodes=int(dag_max_nodes))

    def evaluate(
        self, boards: list[CBoard], *, role: str, cluster: tuple[int, int] | None,
    ) -> np.ndarray:
        started = time.perf_counter()
        q = self.source.q_for_boards(boards, role=role, cluster=cluster)
        self._cost.add(role, len(boards), time.perf_counter() - started)
        self.dag_watch.observe(self)
        return np.asarray(q, dtype=np.float64)

    # ── the DagBackedSource surface the watchdog drives ──────────────────────
    def dag_stats(self) -> dict[str, int] | None:
        return None if self._dag_source is None else self._dag_source.dag_stats()

    def reset_game(self) -> None:
        if self._dag_source is not None:
            self._dag_source.reset_game()

    def begin_position(self, *, reset_memo: bool) -> None:
        """Drop the canonical memo between INDEPENDENT positions.

        Audit rows are deduped, unrelated draws, so a memo carried across them
        cannot help a real labelling run and would make a DAG-backed arm's
        measured cost depend on the audit set's ordering. #472 says a memo
        cannot change an ANSWER, so this is a cost decision only.
        """
        if reset_memo:
            self.reset_game()

    def cost(self) -> ArmCost:
        return self._cost

    def stamp(self) -> dict[str, Any]:
        return {
            "kind": "nnue",
            "construction": "oneply_child",
            "arm_config_requested": dict(self.source.requested),
            "arm_config_realized": dict(self.source.realized),
            "kernel": self.source.kernel,
            "pack_file_sha256": self.source.pack_file_sha256,
            "pack_source_sha256": self.source.pack_source_sha256,
            "provider_stats": {
                k: int(v) for k, v in self.source.provider_stats().items()
            },
            # ⚑ `None`, not a zeroed dict, for an arm with no canonical store.
            # A watchdog reading 0 resets and a 0-byte peak is exactly what a
            # BROKEN watchdog on a DAG arm looks like, and publishing that shape
            # for an arm with no store to watch is a value that reads as a
            # measurement and is not one.
            "dag_store_watch": (
                None if self._dag_source is None else {
                    "max_nodes": self.dag_watch.max_nodes,
                    "resets": self.dag_watch.resets,
                    "nodes_peak": self.dag_watch.nodes_peak,
                    "memory_peak": self.dag_watch.memory_peak,
                }
            ),
        }

    def close(self) -> None:
        self.source.close()


class StockfishCandidateArm:
    """``go nodes <N>`` -- or ``go depth <D>`` -- on every child, as a ``ChildArm``.

    ⚑ THE ONLY THING THAT DIFFERS FROM A NATIVE ARM IS THE EVALUATOR. The seat
    (the child's own side to move), the mate mapping and the cp -> q logistic
    are the native arms', reached through the same objects; ``probe_root``
    supplies the negation to the root mover's seat for both.

    ⚑ AND THE LIMIT IS NAMED ON EVERY SEARCH, never left to the engine's own
    ``nodes`` default. ``StockfishUCI.search`` falls back to ``self.nodes`` when
    a call names no limit, so an arm that forgot to pass one would run a
    2000-node search and report it under whatever budget its NAME advertises.
    """

    def __init__(
        self,
        *,
        spec: SfArmSpec,
        engine: SearchEngine,
        cp_slope: float,
        cp_draw_width: float,
        fresh_per_position: bool,
    ) -> None:
        self.arm = spec.name
        self.nodes = None if spec.nodes is None else int(spec.nodes)
        self.depth = None if spec.depth is None else int(spec.depth)
        self.limit_kind = spec.limit_kind
        self.engine = engine
        self.cp_slope = float(cp_slope)
        self.cp_draw_width = float(cp_draw_width)
        self.fresh_per_position = bool(fresh_per_position)
        self._cost = ArmCost()
        self._searches = 0
        self._terminals = 0
        #: The depth the ENGINE reported, one entry per search it ran. The
        #: requested budget is `self.nodes` / `self.depth` and is not a
        #: measurement of anything: a node arm's realized depth is whatever the
        #: budget bought, and a depth arm's can still fall SHORT of the ask when
        #: the search ends early on a proven mate or a tablebase hit.
        self._depths: list[int] = []

    def effective_cp(self, board: CBoard) -> float:
        """One child's effective cp from ITS OWN seat -- terminal, mate or cp."""
        terminal = terminal_effective_cp(board)
        if terminal is not None:
            self._terminals += 1
            return terminal
        result = (
            self.engine.search(board.fen(), nodes=self.nodes) if self.nodes is not None
            else self.engine.search(board.fen(), depth=self.depth)
        )
        self._searches += 1
        if result.depth is not None:
            self._depths.append(int(result.depth))
        eff = effective_cp_from_score(result.cp, result.mate)
        if eff is None:
            # Refusing beats imputing: a silent 0.0 here is a draw claim about a
            # position nobody evaluated.
            raise RuntimeError(
                f"{self.arm}: Stockfish returned no cp and no mate for "
                f"{board.fen()!r} (bestmove {result.bestmove_uci!r}); the arm "
                "cannot label a position it did not score",
            )
        return eff

    def evaluate(
        self, boards: list[CBoard], *, role: str, cluster: tuple[int, int] | None,
    ) -> np.ndarray:
        del cluster  # this arm banks through the per-position dump, not a sidecar
        started = time.perf_counter()
        eff = np.asarray(
            [self.effective_cp(board) for board in boards], dtype=np.float64,
        )
        q = q_from_effective_cp(
            eff, slope=self.cp_slope, draw_width_cp=self.cp_draw_width,
        )
        self._cost.add(role, len(boards), time.perf_counter() - started)
        return q

    def begin_position(self, *, reset_memo: bool) -> None:
        del reset_memo  # the DAG memo is a native-arm concept; the TT is not it
        if self.fresh_per_position:
            self.engine.new_game()

    def cost(self) -> ArmCost:
        return self._cost

    def stamp(self) -> dict[str, Any]:
        return {
            "kind": "stockfish",
            "construction": "oneply_child",
            **_limit_stamp(self),
            "multipv": 1,
            "threads": 1,
            "hash_mb": self.engine.hash_mb,
            "searches": self._searches,
            "terminal_children_resolved_without_search": self._terminals,
            **_depth_stamp(self._depths),
            "tt_hygiene": _tt_hygiene(self.fresh_per_position),
        }

    def close(self) -> None:
        self.engine.close()


def _tt_hygiene(fresh_per_position: bool) -> str:
    return (
        "cold at run start (ucinewgame + Clear Hash); a fresh ucinewgame per "
        "POSITION" if fresh_per_position else
        "cold at run start (ucinewgame + Clear Hash); SHARED across searches "
        "thereafter"
    )


def _limit_stamp(arm: StockfishCandidateArm | RootedStockfishArm) -> dict[str, Any]:
    """The arm's REQUESTED search limit, with the unused half left NULL.

    ⚑ A DEPTH ARM PUBLISHES ``nodes: null``, NOT A NUMBER. There is no node
    budget to report -- ``go depth 9`` spends whatever the iteration costs --
    and printing the engine constructor's default there would put a value on the
    face of the report that reads as the budget the arm ran under and was never
    a limit on anything. The same holds mirrored for ``depth_requested`` on a
    node arm, whose depth is an OUTCOME (see ``_depth_stamp``) rather than an
    ask. ``limit_kind`` is what a reader keys on to tell the two apart without
    inferring it from which field is null.
    """
    return {
        "limit_kind": arm.limit_kind,
        "nodes": arm.nodes,
        "depth_requested": arm.depth,
    }


def _depth_stamp(depths: Sequence[int]) -> dict[str, Any]:
    """REALIZED search depth, read off the engine's own replies.

    ⚑ NOT ``depth_requested``, and the gap between the two is the point. A node
    arm has no requested depth at all and these are the only depth numbers it
    can report; a depth arm's realized depth normally equals its ask, and where
    it does NOT -- a search that ended early on a proven mate or a tablebase hit
    -- the difference is exactly what a reader needs to see. An arm that echoed
    its own request here would report a depth the engine never reached and
    nothing would raise.
    """
    return {
        "depth_mean": statistics.fmean(depths) if depths else math.nan,
        "depth_min": min(depths) if depths else 0,
        "depth_max": max(depths) if depths else 0,
    }


# ── the rooted MultiPV arm ───────────────────────────────────────────────────


@dataclass(frozen=True)
class RootedRanking:
    """One rooted search's move ranking, taken from ONE depth.

    ``moves`` is ``(multipv_rank, uci, effective_cp)`` in rank order.
    ``complete`` says whether that depth carried the full width the search was
    asked for; an incomplete one is still a single-depth ranking (never a mix),
    it is simply narrower than requested.
    """

    depth: int
    complete: bool
    moves: tuple[tuple[int, str, float], ...]


def rooted_ranking_from_info_lines(
    lines: Sequence[str], *, expected_lines: int,
) -> RootedRanking:
    """The DEEPEST COMPLETE MultiPV set in a rooted search's info stream.

    ⚑⚑ THE BUG THIS FUNCTION EXISTS TO NOT HAVE. A node-limited search is cut
    off mid-iteration, so the tail of the stream is depth D's ranks 1..k
    followed by nothing -- while depth D-1 has all W. Keeping "the last line
    seen per rank" (which is exactly what ``StockfishUCI``'s accumulator does,
    correctly, for its single-PV purpose) then builds one ranking out of two
    searches: rank 1 from depth D, rank 5 from depth D-1. Every number derived
    from it is a blend of two evaluations and nothing raises.

    So the lines are bucketed BY DEPTH and one bucket is chosen whole:

    * the deepest depth whose bucket holds ``expected_lines`` ranks, else
    * the deepest depth whose bucket holds rank 1 (``complete=False``).

    ``upperbound`` / ``lowerbound`` lines are dropped: an aspiration-window
    bound is a claim about a window, not the move's score, and mixing one into
    a ranking silently reorders it.
    """
    by_depth: dict[int, dict[int, tuple[str, float]]] = {}
    for line in lines:
        parts = line.split()
        if not parts or parts[0] != "info":
            continue
        if "upperbound" in parts or "lowerbound" in parts:
            continue
        mpv, _nodes, depth, cp, mate, _wdl, pv_move = _parse_info_fields(parts)
        if depth is None or pv_move is None:
            continue
        eff = effective_cp_from_score(cp, mate)
        if eff is None:
            continue
        by_depth.setdefault(int(depth), {})[int(mpv or 1)] = (str(pv_move), eff)
    if not by_depth:
        raise RuntimeError(
            "the rooted search produced no scored MultiPV line; there is no "
            "ranking to read and imputing one would invent the arm's answer",
        )
    complete = [d for d, ranks in by_depth.items() if len(ranks) >= expected_lines]
    if complete:
        depth = max(complete)
        is_complete = True
    else:
        with_best = [d for d, ranks in by_depth.items() if 1 in ranks]
        if not with_best:
            raise RuntimeError(
                "no depth in the rooted search carried MultiPV rank 1: the arm "
                "has no chosen move",
            )
        depth = max(with_best)
        is_complete = False
    ranks = by_depth[depth]
    return RootedRanking(
        depth=int(depth),
        complete=is_complete,
        moves=tuple(
            (rank, ranks[rank][0], ranks[rank][1]) for rank in sorted(ranks)
        ),
    )


class RootedStockfishArm:
    """ONE ``go nodes <N>`` -- or ``go depth <D>`` -- per POSITION at MultiPV ``W``.

    ⚑ THIS IS NOT A ``ChildArm``, and the difference is the seat. The MultiPV
    list scores each ROOT MOVE from the root mover's own POV, so there is
    nothing to negate; ``probe_root``'s negation would inverted it exactly.
    ``label`` therefore builds the target itself -- through the SAME
    ``oneply_policy_vector``, so the softmax and the legal-index placement are
    still the harness's and only the seat differs.

    ⚑ AND ``chosen`` IS PV1, NOT THE ARGMAX. They coincide unless two listed
    moves carry exactly the same cp, in which case ``np.argmax`` breaks the tie
    by the audit set's legal-move order and the engine breaks it by search
    order. Both readings are defensible; the arm's own answer is PV1, so that
    is what is scored, and the rows where the two differ are COUNTED
    (``chosen_disagrees_with_probs_argmax``) rather than absorbed.
    """

    def __init__(
        self,
        *,
        spec: SfArmSpec,
        engine: StockfishUCI,
        cp_slope: float,
        cp_draw_width: float,
        fresh_per_position: bool,
    ) -> None:
        self.arm = spec.name
        self.nodes = None if spec.nodes is None else int(spec.nodes)
        self.depth = None if spec.depth is None else int(spec.depth)
        self.limit_kind = spec.limit_kind
        self.go_limit = spec.go_limit
        self.width = spec.width
        self.engine = engine
        self.cp_slope = float(cp_slope)
        self.cp_draw_width = float(cp_draw_width)
        self.fresh_per_position = bool(fresh_per_position)
        self._cost = ArmCost()
        self._searches = 0
        self._incomplete_depths = 0
        self._realized_widths: list[int] = []
        self._depths: list[int] = []
        # What the engine's MultiPV option currently is. `StockfishUCI` sets it
        # at construction for a fixed width; the "all" arm moves it per
        # position, and resending an unchanged value would cost a round trip
        # per position for nothing.
        self._engine_multipv = int(engine.multipv)

    def requested_width(self, legal_moves: int) -> int:
        """MultiPV for THIS position. ``all`` clamps to the legal move count."""
        wanted = legal_moves if self.width is None else int(self.width)
        return max(1, min(int(wanted), int(legal_moves)))

    def search_lines(self, fen: str, *, multipv: int) -> list[str]:
        """Drive one rooted search and return every line the engine emitted.

        ⚑ THE INFO LINES ARE READ HERE RATHER THAN THROUGH ``search``, because
        ``StockfishUCI`` folds them into one result per rank as it goes and the
        depth each rank came from is not in that result. Widening the
        production selfplay class for a ruler script is the wrong trade (the
        same call ``audit_targets.engine_identity`` makes about the handshake),
        so the protocol section, the lock and the deadline are taken exactly as
        ``search`` takes them.
        """
        with self.engine._lock, self.engine._protocol_section():
            if multipv != self._engine_multipv:
                self.engine._send(f"setoption name MultiPV value {multipv}")
                self.engine._send("isready")
                self.engine._wait_for("readyok")
                self._engine_multipv = multipv
            self.engine._send(f"position fen {fen}")
            # ⚑ ONE limit clause, taken from the SPEC rather than assembled
            # here, so this hand-built `go` line and the per-child arm's
            # `search` keyword cannot come to differ about which budget the
            # `-d` arms run under.
            self.engine._send(f"go {self.go_limit}")
            deadline = time.monotonic() + self.engine.read_timeout_s
            lines: list[str] = []
            while True:
                line = self.engine._readline_with_deadline(deadline).strip()
                if line.startswith("bestmove"):
                    return lines
                if line:
                    lines.append(line)

    def label(
        self,
        *,
        board: chess.Board,
        legal_ucis: list[str],
        legal_idxs: np.ndarray,
        sigma: float,
    ) -> Label:
        started = time.perf_counter()
        width = self.requested_width(len(legal_ucis))
        lines = self.search_lines(board.fen(), multipv=width)
        self._cost.add(shadow._ROLE_ROOT, 1, time.perf_counter() - started)
        self._searches += 1
        ranking = rooted_ranking_from_info_lines(lines, expected_lines=width)
        self._realized_widths.append(len(ranking.moves))
        self._depths.append(ranking.depth)
        if not ranking.complete:
            self._incomplete_depths += 1

        where = {uci: i for i, uci in enumerate(legal_ucis)}
        listed = [(rank, uci, cp) for rank, uci, cp in ranking.moves if uci in where]
        if not listed:
            raise RuntimeError(
                f"{self.arm}: none of the rooted search's MultiPV moves "
                f"{[m for _, m, _ in ranking.moves]} is an encodable legal move "
                f"of {board.fen()!r}",
            )
        # ⚑ NOT NEGATED. See the class docstring: these scores are already the
        # root mover's. `oneply_policy_vector` places the softmax at the given
        # 4672 action ids and leaves every other entry at zero, so the legal
        # moves the search did not list get exactly 0.0 rather than an imputed
        # value -- they cannot be the argmax and they contribute nothing to the
        # expected regret.
        q_root = q_from_effective_cp(
            np.asarray([cp for _, _, cp in listed], dtype=np.float64),
            slope=self.cp_slope, draw_width_cp=self.cp_draw_width,
        )
        full = shadow.oneply_policy_vector(
            tuple(int(legal_idxs[where[uci]]) for _, uci, _ in listed),
            q_root, sigma=sigma,
        )
        return Label(
            probs=np.asarray(full, dtype=np.float64)[legal_idxs],
            chosen=listed[0][1],
            values={uci: float(cp) for _, uci, cp in listed},
            values_kind="effective_cp_root_seat",
        )

    def begin_position(self, *, reset_memo: bool) -> None:
        del reset_memo  # the DAG memo is a native-arm concept; the TT is not it
        if self.fresh_per_position:
            self.engine.new_game()

    def cost(self) -> ArmCost:
        return self._cost

    def stamp(self) -> dict[str, Any]:
        return {
            "kind": "stockfish",
            "construction": "rooted_multipv",
            **_limit_stamp(self),
            "multipv_requested": MULTIPV_ALL if self.width is None else self.width,
            "multipv_realized_mean": (
                statistics.fmean(self._realized_widths) if self._realized_widths
                else math.nan
            ),
            "multipv_realized_min": (
                min(self._realized_widths) if self._realized_widths else 0
            ),
            "multipv_realized_max": (
                max(self._realized_widths) if self._realized_widths else 0
            ),
            **_depth_stamp(self._depths),
            # Positions where NO depth carried the full requested width, so the
            # ranking is the deepest single depth that had a best move. Not an
            # error and not a mix; a narrower ranking than asked for.
            "positions_without_a_complete_multipv_depth": self._incomplete_depths,
            "threads": 1,
            "hash_mb": self.engine.hash_mb,
            "searches": self._searches,
            "tt_hygiene": _tt_hygiene(self.fresh_per_position),
        }

    def close(self) -> None:
        self.engine.close()


def depth_requested_of(arm: ReportableArm) -> int | None:
    """The arm's REQUESTED search depth, or ``None`` for an arm with no depth limit.

    Carried into every per-position dump row, because a dump outlives the report
    it was written beside and gets joined to other dumps: without it a
    ``sf-d9`` row and a ``sf-512`` row are two identical shapes whose only
    difference lives in an arm NAME a joiner is free to rename. ``None`` is the
    honest answer for a node arm and for every native arm -- they have no depth
    ASK, only the realized depth in ``_depth_stamp``.

    ⚑ ``isinstance``, not ``getattr(arm, "depth", None)``. A name looked up as a
    string returns ``None`` for a typo exactly as it does for an arm with no
    depth, so the silent-failure mode of that spelling is the one this repo is
    built to refuse: every row would publish ``null`` and every test asserting
    ``null`` for the native arms would still pass.
    """
    if isinstance(arm, StockfishCandidateArm | RootedStockfishArm):
        return arm.depth
    return None


def clear_transposition_table(engine: StockfishUCI) -> None:
    """``ucinewgame`` + ``Clear Hash``, so an arm starts genuinely cold.

    ``new_game`` is the public ``ucinewgame`` handshake and is what Stockfish
    documents as its clear point. ``Clear Hash`` is sent as well because it is
    the option every UCI GUI uses for this and costs one round trip: an arm
    whose first hundred positions inherited a previous run's table would report
    a node budget it did not spend.
    """
    engine.new_game()
    # `StockfishUCI` exposes no generic setoption, deliberately: it sits on the
    # production selfplay path and widening it for a ruler script is the wrong
    # trade. The lock and the protocol section are taken exactly as `new_game`
    # takes them, so a failure here poisons the engine rather than desyncing it.
    with engine._lock, engine._protocol_section():
        engine._send("setoption name Clear Hash")
        engine._send("isready")
        engine._wait_for("readyok")


# ── configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GateConfig:
    """Everything one gate run needs."""

    audit_set: Path
    pack: Path
    arms: tuple[str, ...]
    native_configs: dict[str, readout.ResolvedArmConfig]
    sf_specs: dict[str, SfArmSpec]
    static_resolver_max_depth: int | None
    limit: int
    oneply_sigma: float
    cp_per_internal_unit: float
    cp_slope: float
    cp_draw_width: float
    dag_max_nodes: int
    dag_reset_every: int
    sf_binary: Path | None
    sf_hash_mb: int
    sf_fresh_per_position: bool
    nice: int
    run_id: str
    dump_per_position: Path | None = None
    dump_move_values: bool = False
    bank_observations: Path | None = None

    @property
    def native_arms(self) -> tuple[str, ...]:
        return tuple(a for a in self.arms if a in NATIVE_ARMS)

    @property
    def dag_arms(self) -> tuple[str, ...]:
        """The selected arms with a canonical DAG store -- possibly none.

        The store knobs are stamped into the report only when this is non-empty:
        a ``dag_max_nodes`` printed beside a run with no DAG store reads as a
        setting that shaped the numbers, and it did not.
        """
        return tuple(a for a in self.arms if a in DAG_BACKED_ARMS)


def _static_arm_source(cfg: GateConfig, **kwargs: Any) -> gen.NnueArmValueSource:
    """``nnue-static`` through the GENERATOR's own knob gating.

    ``resolve_arm_knob_defaults`` is what decides that the static arm reads
    ``resolver_max_depth`` and NOTHING else -- an earlier revision of it
    published ``qsearch_max_ply`` in a static run's realized line, which is this
    repo's signature defect with a receipt on top. Calling it rather than
    restating the rule is what keeps this file on the right side of that fix.
    """
    depth, qmax, qchk = gen.resolve_arm_knob_defaults(
        argparse.Namespace(
            value_source=gen.VALUE_SOURCE_NNUE_STATIC,
            nnue_resolver_max_depth=cfg.static_resolver_max_depth,
            nnue_qsearch_max_ply=None,
            nnue_qsearch_check_plies=None,
        ),
    )
    if depth is None or qmax is not None or qchk is not None:
        raise RuntimeError(
            "the generator resolved a quiescence knob for nnue-static; the "
            "static arm consumes resolver_max_depth alone",
        )
    return gen.NnueArmValueSource(
        arm=gen.VALUE_SOURCE_NNUE_STATIC,
        pack=cfg.pack,
        cp_per_internal_unit=cfg.cp_per_internal_unit,
        cp_slope=cfg.cp_slope,
        cp_draw_width=cfg.cp_draw_width,
        resolver_max_depth=int(depth),
        **kwargs,
    )


def _bank_path(base: Path | None, arm: str) -> Path | None:
    if base is None:
        return None
    suffix = base.suffix or ".jsonl"
    stem = base.name[: -len(suffix)] if base.name.endswith(suffix) else base.name
    path = base.with_name(f"{stem}.{arm}{suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def open_arms(
    cfg: GateConfig, *, pack_sha: str | None,
) -> tuple[list[ChildArm], list[RootedStockfishArm]]:
    """One long-lived context per arm, opened IN SEQUENCE.

    ``set_arm_config`` / ``fastq_set_config`` are PROCESS-wide globals that
    ``arm_open`` snapshots, so a context keeps the configuration that was live
    when it opened. Building them one at a time is what lets several arms with
    several configurations coexist, and ``NnueArmValueSource``'s
    requested-vs-realized check is what proves each one got its own.
    """
    child: list[ChildArm] = []
    rooted: list[RootedStockfishArm] = []
    identity = {"run_id": cfg.run_id, "population_kind": "frozen_audit_labels"}
    try:
        for name in cfg.arms:
            spec = cfg.sf_specs.get(name)
            if spec is not None:
                if cfg.sf_binary is None:
                    raise ValueError(
                        f"{name} needs a Stockfish binary; none was given and "
                        "none was discovered",
                    )
                # ⚑ ``nodes`` IS OMITTED FOR A DEPTH ARM RATHER THAN INVENTED.
                # The constructor argument is only the fallback for a
                # ``search()`` that names no limit, and both arms below name
                # theirs on every call -- so a depth arm has no node budget to
                # give, and handing one over would put a number on an object a
                # later reader could mistake for the arm's configuration.
                #
                # Two calls rather than a ``**kwargs`` dict: an unpacked
                # ``dict[str, int]`` type-checks as if it could fill ANY
                # constructor parameter, and the one it landed on was
                # ``syzygy_path``.
                multipv = (
                    1 if not spec.rooted
                    else (spec.width or DEFAULT_ROOTED_MULTIPV)
                )
                engine = (
                    StockfishUCI(
                        str(cfg.sf_binary), multipv=multipv,
                        hash_mb=cfg.sf_hash_mb, nice=max(0, cfg.nice),
                    )
                    if spec.nodes is None else
                    StockfishUCI(
                        str(cfg.sf_binary), nodes=int(spec.nodes),
                        multipv=multipv, hash_mb=cfg.sf_hash_mb,
                        nice=max(0, cfg.nice),
                    )
                )
                if spec.rooted:
                    rooted.append(RootedStockfishArm(
                        spec=spec, engine=engine, cp_slope=cfg.cp_slope,
                        cp_draw_width=cfg.cp_draw_width,
                        fresh_per_position=cfg.sf_fresh_per_position,
                    ))
                else:
                    child.append(StockfishCandidateArm(
                        spec=spec, engine=engine, cp_slope=cfg.cp_slope,
                        cp_draw_width=cfg.cp_draw_width,
                        fresh_per_position=cfg.sf_fresh_per_position,
                    ))
                # ⚑ THE COLD START HAPPENS AFTER THE ARM IS REGISTERED, and the
                # order is the whole point. `clear_transposition_table` talks
                # UCI -- it can time out, or find a child that died during the
                # handshake -- and the cleanup below can only close arms that
                # are already in these lists. Clearing first left a live
                # Stockfish subprocess owned by nobody for the rest of the run.
                # No arm searches during construction, so deferring the clear
                # costs nothing: the table is still cold at the first search.
                clear_transposition_table(engine)
                continue
            bank = _bank_path(cfg.bank_observations, name)
            source: gen.NnueArmValueSource
            dag_source: readout.ReadoutArmSource | None
            if name == gen.VALUE_SOURCE_NNUE_STATIC:
                source = _static_arm_source(
                    cfg, leaf_bank=bank, pack_file_sha256=pack_sha,
                    bank_identity={**identity, "arm": name},
                )
                dag_source = None
            else:
                readout_source = readout.ReadoutArmSource(
                    config=cfg.native_configs[name],
                    pack=cfg.pack,
                    cp_per_internal_unit=cfg.cp_per_internal_unit,
                    cp_slope=cfg.cp_slope,
                    cp_draw_width=cfg.cp_draw_width,
                    leaf_bank=bank,
                    pack_file_sha256=pack_sha,
                    bank_identity={**identity, "arm": name},
                )
                source = readout_source
                dag_source = (
                    readout_source if readout_source.spec.uses_dag else None
                )
            child.append(NnueCandidateArm(
                source=source, dag_source=dag_source,
                dag_max_nodes=cfg.dag_max_nodes,
            ))
    except BaseException:
        for arm in (*child, *rooted):
            arm.close()
        raise
    return child, rooted


# ── scoring ──────────────────────────────────────────────────────────────────


@dataclass
class ArmScore:
    """One arm's paired per-position booleans and regrets, unaggregated.

    ⚑ THE ROWS ARE KEPT, NOT JUST THEIR MEAN. A later correction -- a
    stratification by phase, a clustered interval, a different regret cap --
    then costs a re-read of this list instead of a rerun, and a rerun against a
    deterministic evaluator is not merely expensive: it re-rolls the
    intervention and confounds any drift with the method change.
    """

    top1_regret_cp: list[float] = field(default_factory=list)
    expected_regret_cp: list[float] = field(default_factory=list)
    top1_agree: list[bool] = field(default_factory=list)
    in_top10: list[bool] = field(default_factory=list)
    top1_move_listed: list[bool] = field(default_factory=list)
    chosen_disagrees_with_argmax: int = 0

    def add(
        self,
        *,
        top1: float,
        expected: float,
        agree: bool,
        out_of_top10: bool | None,
        listed: bool,
        argmax_agrees: bool,
    ) -> None:
        self.top1_regret_cp.append(float(top1))
        self.expected_regret_cp.append(float(expected))
        self.top1_agree.append(bool(agree))
        if out_of_top10 is not None:
            self.in_top10.append(not out_of_top10)
        self.top1_move_listed.append(bool(listed))
        self.chosen_disagrees_with_argmax += int(not argmax_agrees)

    def summary(self) -> dict[str, float]:
        n = len(self.top1_regret_cp)
        return {
            "positions": float(n),
            "top1_agree_rate": sum(self.top1_agree) / n if n else math.nan,
            "top10_agree_rate": (
                sum(self.in_top10) / len(self.in_top10) if self.in_top10
                else math.nan
            ),
            "top10_scorable_positions": float(len(self.in_top10)),
            "top1_regret_cp_mean": (
                statistics.fmean(self.top1_regret_cp) if n else math.nan
            ),
            "top1_regret_cp_median": (
                statistics.median(self.top1_regret_cp) if n else math.nan
            ),
            # ⚑ SIGMA-DEPENDENT, unlike everything above it. A softmax cannot
            # move an argmax, so every top1 statistic is inert in
            # `--oneply-sigma`; this one is not, and it is comparable only
            # across the arms of ONE run.
            "expected_regret_cp_mean_at_sigma": (
                statistics.fmean(self.expected_regret_cp) if n else math.nan
            ),
            # The share of the headline regret that is a FLOOR rather than a
            # measurement -- see the censoring rule in the module docstring.
            "top1_move_unlisted_rate": (
                1.0 - sum(self.top1_move_listed) / n if n else math.nan
            ),
            "top1_move_unlisted_positions": float(
                len(self.top1_move_listed) - sum(self.top1_move_listed),
            ),
            # Nonzero only for an arm whose chosen move is not its own argmax
            # (the rooted arm on an exact cp tie). Published rather than
            # resolved silently in one direction.
            "chosen_disagrees_with_probs_argmax": float(
                self.chosen_disagrees_with_argmax,
            ),
        }


def child_label(
    probe: shadow.PlyProbe,
    *,
    arm: str,
    legal_ucis: list[str],
    legal_idxs: np.ndarray,
    sigma: float,
) -> Label:
    """One 1-ply arm's target, reindexed onto ``audit``'s legal-move order.

    ⚑ THE REINDEX IS NOT COSMETIC. ``CBoard.legal_move_indices`` and
    ``eval.audit.legal_full_indices`` agree on the SET of 4672 action ids and
    NOT on their order, and ``audit_targets`` takes its argmax over the audit
    order -- so two moves an arm values identically must break the tie the way
    the banked candidate rows broke it, or this tool's booleans stop being
    joinable with theirs. Going through the (4672,) vector is what makes the
    reindex a lookup rather than a re-derivation of the move list.
    """
    full = shadow.oneply_policy_vector(
        probe.legal_full_indices, probe.q_mover[arm], sigma=sigma,
    )
    probs = np.asarray(full, dtype=np.float64)[legal_idxs]
    order = probe_order_for(probe.legal_full_indices, legal_idxs)
    q_mover = np.asarray(probe.q_mover[arm], dtype=np.float64)[order]
    return Label(
        probs=probs,
        chosen=legal_ucis[int(np.argmax(probs))],
        values={uci: float(v) for uci, v in zip(legal_ucis, q_mover, strict=True)},
        values_kind="q_root_seat",
    )


def probe_order_for(
    probe_indices: tuple[int, ...], legal_idxs: np.ndarray,
) -> np.ndarray:
    """Position in the probe's move order for each of ``audit``'s legal moves."""
    where = {int(a): i for i, a in enumerate(probe_indices)}
    return np.asarray([where[int(a)] for a in legal_idxs], dtype=np.int64)


def _refuse_move_set_drift(
    pos: AuditPosition, probe: shadow.PlyProbe, legal_idxs: np.ndarray,
) -> None:
    probed = set(probe.legal_full_indices)
    audited = {int(a) for a in legal_idxs}
    if probed == audited:
        return
    raise RuntimeError(
        f"{pos.key}: the probe saw {len(probed)} legal moves and the audit "
        f"scorer saw {len(audited)} (symmetric difference "
        f"{sorted(probed ^ audited)}) for {pos.fen!r}. The arms would be scored "
        "on a different move set from the deep-SF labels.",
    )


#: What every number in the report MEANS, carried with it. A dump outlives the
#: session that made it and gets joined to other dumps; a censoring rule that
#: lives only in a docstring is a rule the joiner does not have.
METRIC_DEFINITIONS: dict[str, Any] = {
    "label_rule_oneply_child": (
        "argmax over the root's legal moves of softmax(oneply_sigma * -q(child)), "
        "where q is the arm's value through the generator's cp-logistic; built by "
        "nnue_shadow_label_readout.probe_root + oneply_policy_vector"
    ),
    "label_rule_rooted_multipv": (
        "one `go nodes N` (or `go depth D` for a -d arm) at MultiPV W on the "
        "position itself; the chosen move is PV1 of the DEEPEST COMPLETE MultiPV "
        "depth and the ranking is that depth's list, scored from the ROOT MOVER's "
        "seat (no negation) through the same cp-logistic and the same "
        "oneply_policy_vector softmax"
    ),
    "search_limit_kind": (
        "a Stockfish arm limits on NODES (`sf-512`, `sfroot-2048-mpv20`) or on "
        "DEPTH (`sf-d9`, `sfroot-d9-mpv20`), never both. limit_kind names which; "
        "`nodes` and `depth_requested` are the ASK and exactly one of them is "
        "non-null; depth_mean/min/max are the depth the ENGINE reported and are a "
        "measurement on both kinds -- for a node arm the only depth number there "
        "is, and for a depth arm the place a search that ended early (proven mate, "
        "tablebase hit) shows up as a realized depth below the ask"
    ),
    "top1_agree_rate": (
        "share of positions whose chosen move is in the deep-SF co-best set, "
        "built BY SCORE with ties included (audit_targets.sf_reference_sets)"
    ),
    "top10_agree_rate": (
        "share of positions whose chosen move is in the score-tied deep-SF top-10 "
        "set; positions whose MultiPV list is shorter than 10 cannot support the "
        "claim and are excluded from the denominator, never counted as successes"
    ),
    "top1_regret_cp": (
        "eval.audit.move_regrets of the chosen move: deep-SF best_cp minus the "
        f"move's listed cp, clamped to {AUDIT_REGRET_CAP_CP:.0f}cp"
    ),
    "censoring_rule_for_unlisted_moves": (
        "a legal move the deep MultiPV did not list is given the regret of the "
        "WORST LISTED line -- a FLOOR, biased optimistic, not a measurement. The "
        "frozen set is MultiPV=10, so an arm choosing an 11th-ranked move is "
        "credited with the 10th's regret. top1_move_unlisted_rate is the share of "
        "rows on which the headline regret is that floor."
    ),
    "regret_cap_cp": AUDIT_REGRET_CAP_CP,
    "sigma_inert_for": ["top1_agree_rate", "top10_agree_rate", "top1_regret_cp"],
    "sigma_load_bearing_for": ["expected_regret_cp_mean_at_sigma"],
    "comparison_scope": (
        "definitionally identical to audit_targets.py --dump-per-position candidate "
        "rows (same move_regrets, same sf_reference_sets, same argmax order), so a "
        "comparison with the banked production-target rows is a TABLE JOIN on key. "
        "No such comparison is computed here."
    ),
    "banned_label_source": (
        "the shallow-SF sidecar beside the audit set was produced on a dirty shared "
        "transposition table and is never read by this tool"
    ),
}


def run(cfg: GateConfig) -> dict[str, Any]:
    """Score every arm on the frozen set and return the stamped report."""
    if not cfg.arms:
        raise ValueError("no arms selected")
    audit_stamp = readout._file_stamp(cfg.audit_set)
    audit_sha = audit_stamp[0]
    pack_stamp = readout._file_stamp(cfg.pack) if cfg.native_arms else None
    git_start = readout._git_provenance()
    started_utc = datetime.now(timezone.utc).isoformat()
    nice_realized = readout._apply_nice(cfg.nice)

    positions = load_audit_set(cfg.audit_set)
    total_positions = len(positions)
    if cfg.limit > 0:
        positions = positions[: cfg.limit]

    child_arms, rooted_arms = open_arms(
        cfg, pack_sha=None if pack_stamp is None else pack_stamp[0],
    )
    all_arms: list[ReportableArm] = [*child_arms, *rooted_arms]
    scores: dict[str, ArmScore] = {a.arm: ArmScore() for a in all_arms}
    depth_requested = {a.arm: depth_requested_of(a) for a in all_arms}
    dump: list[dict[str, Any]] = []
    skipped_no_legal = 0
    started = time.perf_counter()
    try:
        for ordinal, pos in enumerate(positions):
            board = chess.Board(pos.fen)
            legal_ucis, legal_idxs = legal_full_indices(board)
            if not legal_ucis:
                skipped_no_legal += 1
                continue
            reset_memo = (
                cfg.dag_reset_every > 0 and ordinal % cfg.dag_reset_every == 0
            )
            for arm in all_arms:
                arm.begin_position(reset_memo=reset_memo)
            labels: dict[str, Label] = {}
            if child_arms:
                probe = shadow.probe_root(
                    CBoard.from_board(board),
                    observers=child_arms,
                    cluster=(_CLUSTER_GAME, ordinal),
                )
                _refuse_move_set_drift(pos, probe, legal_idxs)
                for arm in child_arms:
                    labels[arm.arm] = child_label(
                        probe, arm=arm.arm, legal_ucis=legal_ucis,
                        legal_idxs=legal_idxs, sigma=cfg.oneply_sigma,
                    )
            for rooted in rooted_arms:
                labels[rooted.arm] = rooted.label(
                    board=board, legal_ucis=legal_ucis, legal_idxs=legal_idxs,
                    sigma=cfg.oneply_sigma,
                )
            regrets = move_regrets(pos, legal_ucis)
            sf_top1, sf_top10 = audit_targets.sf_reference_sets(pos.move_cp)
            row: dict[str, Any] = {
                "key": pos.key,
                "fen": pos.fen,
                "phase": pos.phase,
                "source": pos.source,
                "n_legal": len(legal_ucis),
                "n_listed": len(pos.move_cp),
                "best_cp": float(pos.best_cp),
                "sf_top1": sorted(sf_top1),
                "sf_top10": sorted(sf_top10),
                "arm": {},
            }
            for arm in all_arms:
                label = labels[arm.arm]
                expected, _ = expected_and_top1_regret(label.probs, regrets)
                chosen_index = legal_ucis.index(label.chosen)
                top1 = float(regrets[chosen_index])
                argmax_agrees = int(np.argmax(label.probs)) == chosen_index
                out_of_top10 = (
                    None if not sf_top10 else bool(label.chosen not in sf_top10)
                )
                listed = label.chosen in pos.move_cp
                scores[arm.arm].add(
                    top1=top1, expected=expected,
                    agree=label.chosen in sf_top1, out_of_top10=out_of_top10,
                    listed=listed, argmax_agrees=argmax_agrees,
                )
                cell: dict[str, Any] = {
                    "move": label.chosen,
                    "top1_regret_cp": top1,
                    "expected_regret_cp_at_sigma": float(expected),
                    "top1_agree": bool(label.chosen in sf_top1),
                    "out_of_top10": out_of_top10,
                    # ⚑ False means this row's regret is the WORST-LISTED FLOOR,
                    # not a measurement. Carried per row so a downstream
                    # re-analysis can drop or re-weight the censored rows
                    # without rerunning a single evaluation.
                    "top1_move_listed_by_deep_sf": bool(listed),
                    "chosen_is_probs_argmax": bool(argmax_agrees),
                    # ⚑ THE ARM'S ASK, so a joined dump can tell a depth arm
                    # from a node arm without the arm's name. `null` for every
                    # arm that has no depth limit -- see `depth_requested_of`.
                    "depth_requested": depth_requested[arm.arm],
                }
                if cfg.dump_move_values:
                    cell["values_kind"] = label.values_kind
                    cell["values"] = label.values
                row["arm"][arm.arm] = cell
            if cfg.dump_per_position is not None:
                dump.append(row)
    finally:
        for arm in all_arms:
            arm.close()
    wall_s = time.perf_counter() - started

    scored = len(positions) - skipped_no_legal
    reasons: list[str] = []
    if scored <= 0:
        reasons.append("no audit position was scored")
    readout._assert_file_unchanged("audit set", cfg.audit_set, audit_stamp)
    if pack_stamp is not None:
        readout._assert_file_unchanged("NNUE pack", cfg.pack, pack_stamp)
    git_end = readout._git_provenance()
    if not (
        readout._git_provenance_available(git_start)
        and readout._git_provenance_available(git_end)
    ):
        reasons.append(
            "tracked source provenance is unavailable at one or both endpoints",
        )
    elif git_start != git_end:
        reasons.append("tracked source provenance changed while the run was in flight")

    by_name = {arm.arm: arm for arm in all_arms}
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        # In SELECTION order, not construction order: the table is read beside
        # the command line that produced it.
        "arms": {
            name: {
                **by_name[name].stamp(),
                **scores[name].summary(),
                "cost": by_name[name].cost().summary(scored),
            }
            for name in cfg.arms
        },
        "positions_scored": scored,
        "positions_skipped_no_encodable_legal_move": skipped_no_legal,
        "audit_positions_in_file": total_positions,
        "wall_s": wall_s,
        "metric_definitions": METRIC_DEFINITIONS,
        "provenance": {
            "run_id": cfg.run_id,
            "started_utc": started_utc,
            "audit_set": str(cfg.audit_set),
            "audit_set_sha256": audit_sha,
            # ⚑ REQUESTED AND REALIZED, the same split as `nice_*` below and as
            # each arm's `multipv_requested` / `multipv_realized_*`. `limit` is
            # what the CLI ASKED for (0 = all); `limit_realized` is how many
            # rows the slice actually yielded, which is smaller whenever the
            # audit set is shorter than the ask. Publishing the ask alone is a
            # value that reads as a measurement of what was scored and is not
            # one -- and it is inert to whether the slice is applied at all.
            "limit": int(cfg.limit),
            "limit_realized": len(positions),
            "pack_path": str(cfg.pack) if cfg.native_arms else None,
            "pack_file_sha256": None if pack_stamp is None else pack_stamp[0],
            "oneply_sigma": float(cfg.oneply_sigma),
            # ⚑ NULL RATHER THAN A NUMBER WHEN NO SELECTED ARM CONSUMED IT.
            # `cp_per_internal_unit` converts the NATIVE arms' internal units;
            # the Stockfish arms are handed cp already and never touch it, so
            # stamping it on an SF-only run publishes a knob that reads as
            # having shaped the numbers when nothing read it. `cp_slope` and
            # `cp_draw_width` are NOT gated: `q_from_effective_cp` is on every
            # arm's path, Stockfish arms included.
            "cp_per_internal_unit": (
                float(cfg.cp_per_internal_unit) if cfg.native_arms else None
            ),
            "cp_slope": float(cfg.cp_slope),
            "cp_draw_width": float(cfg.cp_draw_width),
            "dag_max_nodes": (
                int(cfg.dag_max_nodes) if cfg.dag_arms else None
            ),
            "dag_reset_every_positions": (
                int(cfg.dag_reset_every) if cfg.dag_arms else None
            ),
            "sf_binary": None if cfg.sf_binary is None else str(cfg.sf_binary),
            "sf_binary_sha256": (
                None if cfg.sf_binary is None
                else readout._sha256_file(cfg.sf_binary)
            ),
            "sf_engine_id": (
                None if cfg.sf_binary is None
                else audit_targets.engine_identity(str(cfg.sf_binary))
            ),
            "nice_requested": int(cfg.nice),
            "nice_realized": int(nice_realized),
            **git_start,
            "python": sys.version.split()[0],
        },
        "inadmissible_reasons": reasons,
        "admissible": not reasons,
    }
    if cfg.dump_per_position is not None:
        readout._atomic_write_text(
            cfg.dump_per_position,
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in dump),
        )
    for reason in reasons:
        _LOG.error("INADMISSIBLE: %s", reason)
    return report


def format_table(report: dict[str, Any]) -> str:
    """The arms side by side, in selection order."""
    header = (
        f"{'arm':<22}{'n':>6}{'top1%':>8}{'top10%':>8}{'regret_mean':>13}"
        f"{'regret_med':>12}{'unlisted%':>11}{'s/pos':>10}"
    )
    lines = [header, "-" * len(header)]
    for name, cell in report["arms"].items():
        lines.append(
            f"{name:<22}{int(cell['positions']):>6}"
            f"{100.0 * cell['top1_agree_rate']:>8.1f}"
            f"{100.0 * cell['top10_agree_rate']:>8.1f}"
            f"{cell['top1_regret_cp_mean']:>13.1f}"
            f"{cell['top1_regret_cp_median']:>12.1f}"
            f"{100.0 * cell['top1_move_unlisted_rate']:>11.1f}"
            f"{cell['cost']['eval_s_per_position']:>10.4f}",
        )
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument(
        "--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"),
        help="the FROZEN deep-SF audit set. Pointing this at the shallow-SF "
             "sidecar is refused: that cache ran on a dirty shared TT.",
    )
    p.add_argument("--nnue-pack", type=Path, default=None)
    p.add_argument(
        "--arms", default=",".join(NATIVE_ARMS),
        help="comma-separated candidate arms. Native: "
             f"{', '.join(NATIVE_ARMS)}. Stockfish per-child: sf-<nodes> (e.g. "
             "sf-512) or sf-d<depth> (e.g. sf-d9). Stockfish rooted MultiPV: "
             "sfroot-<nodes>[-mpv<W|all>] or sfroot-d<depth>[-mpv<W|all>] "
             f"(e.g. sfroot-2048-mpv20, sfroot-d9); W defaults to "
             f"{DEFAULT_ROOTED_MULTIPV} and 'all' means one PV per legal move. "
             "A 'd' before the number makes the arm DEPTH-limited (go depth D) "
             "instead of node-limited (go nodes N); sf-d9 and sf-9 are two "
             "different arms. Rooted names are canonicalised to their -mpv form, "
             f"so sfroot-2048 and sfroot-2048-mpv{DEFAULT_ROOTED_MULTIPV} are one "
             "arm.",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="score only the first N audit positions (0 = all). The set's own "
             "order, so two runs at the same N score the same rows.",
    )
    p.add_argument(
        "--oneply-sigma", type=float, default=None,
        help="sharpness of the label softmax. INERT for every top1 statistic "
             f"this tool reports. Default {shadow.oneply_sigma_default():.4g}, "
             "the shadow harness's own default.",
    )
    p.add_argument("--stockfish", type=Path, default=None,
                   help="Stockfish binary for the sf-/sfroot- arms (auto-discovered)")
    p.add_argument("--sf-hash-mb", type=int, default=16)
    p.add_argument(
        "--sf-fresh-per-position", action="store_true",
        help="ucinewgame before each position, so the TT cannot carry one "
             "position's search into the next. Expensive; the default is one "
             "cold start per RUN and the verdict is stamped in the report.",
    )
    # ⚑ `default=None` MEANS "NOT SUPPLIED", not "no value". Both knobs are
    # refused on a run with no DAG-backed arm, and a refusal that cannot tell an
    # explicit value from argparse's own default cannot fire. The numbers they
    # fall back to are `shadow.DEFAULT_DAG_MAX_NODES` and
    # `DEFAULT_DAG_RESET_EVERY`, resolved in `config_from_args`.
    p.add_argument(
        "--dag-max-nodes", type=int, default=None,
        help="canonical-store watchdog for the DAG-backed arms. 0 disables it, "
             "which is how the DAG arm OOMs. Default "
             f"{shadow.DEFAULT_DAG_MAX_NODES}.",
    )
    p.add_argument(
        "--dag-reset-every", type=int, default=None,
        help="drop a DAG-backed arm's memo every N positions (0 = never). "
             "Audit rows are independent draws, so the default "
             f"({DEFAULT_DAG_RESET_EVERY}) keeps one row's cost from depending "
             "on the row before it.",
    )
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--dump-per-position", type=Path, default=None)
    p.add_argument(
        "--dump-move-values", action="store_true",
        help="add every arm's per-legal-move raw observation to the dump, so a "
             "later estimator is a re-read rather than a rerun",
    )
    p.add_argument(
        "--bank-observations", type=Path, default=None,
        help="bank the NATIVE arms' raw per-position scores through the "
             "generator's own leaf bank; one file per arm",
    )
    p.add_argument("--run-id", default="audit_label_candidates")
    p.add_argument("--nice", type=int, default=15)

    p.add_argument("--nnue-resolver-max-depth", type=int, default=None)
    p.add_argument("--nnue-qsearch-max-ply", type=int, default=None)
    p.add_argument("--nnue-qsearch-check-plies", type=int, default=None)
    p.add_argument("--dag-node-cap", type=int, default=None)
    p.add_argument("--fastq-max-qply", type=int, default=None)
    p.add_argument("--fastq-node-cap", type=int, default=None)
    p.add_argument("--fastq-delta-margin", type=int, default=None)
    p.add_argument("--fastq-recapture-exempt", type=int, choices=(0, 1), default=None)

    # ⚑ Also `default=None` for "not supplied", and for the same reason: this
    # one is read by the NATIVE arms alone and is refused without one. Its
    # fallback is `gen.NNUE_CP_PER_INTERNAL_UNIT`. `--nnue-cp-slope` and
    # `--nnue-cp-draw-width` keep their eager defaults -- every arm consumes
    # them, so there is nothing to refuse and nothing to tell apart.
    p.add_argument("--nnue-cp-per-unit", type=float, default=None)
    p.add_argument("--nnue-cp-slope", type=float, default=gen.NNUE_CP_SLOPE)
    p.add_argument("--nnue-cp-draw-width", type=float, default=gen.NNUE_CP_DRAW_WIDTH)
    return p


def parse_arms(spec: str) -> tuple[tuple[str, ...], dict[str, SfArmSpec]]:
    """``--arms`` -> canonical names in order, plus the Stockfish specs."""
    names: list[str] = []
    sf_specs: dict[str, SfArmSpec] = {}
    for raw in (a.strip() for a in spec.split(",")):
        if not raw:
            continue
        if raw in NATIVE_ARMS:
            canonical = raw
        else:
            parsed = parse_sf_arm(raw)
            if parsed is None:
                raise ValueError(
                    f"unknown arm {raw!r}: expected one of {NATIVE_ARMS}, "
                    f"{SF_ARM_PREFIX}<nodes>, "
                    f"{SF_ARM_PREFIX}{SF_DEPTH_MARKER}<depth>, "
                    f"{SFROOT_ARM_PREFIX}<nodes>[-mpv<W|all>], or "
                    f"{SFROOT_ARM_PREFIX}{SF_DEPTH_MARKER}<depth>[-mpv<W|all>]",
                )
            canonical = parsed.name
            sf_specs[canonical] = parsed
        if canonical not in names:
            names.append(canonical)
    if not names:
        raise ValueError("--arms selected nothing")
    return tuple(names), sf_specs


def validate_knobs(args: argparse.Namespace, arms: Sequence[str]) -> None:
    """Refuse a knob NO selected arm consumes.

    ⚑ NOT ``readout._validate_matrix_knobs``, and the difference is exactly one
    arm. That helper's qsearch family is ``{nnue-qsearch, nnue-qsearch-dag}``,
    so it refuses ``--nnue-resolver-max-depth`` on a run whose only native arm
    is ``nnue-static`` -- and ``cae_arm_static_eval`` is precisely the consumer
    that DOES read it. Restating the rule is the smaller error than silently
    dropping a knob the selected arm consumes.

    ⚑ THE RULE APPLIES IN BOTH DIRECTIONS, and the second half was the gap. The
    per-arm knobs below were refused while the WHOLE-RUN ones -- the pack, the
    cp-per-unit conversion, the DAG store bounds, the native leaf bank -- were
    accepted on a Stockfish-only run, where nothing reads any of them.
    ``--nnue-cp-per-unit`` was the worst of the four because ``run`` STAMPED it
    into ``provenance``: a value no arm consumed, published beside the numbers
    as if it had shaped them.
    """
    selected = set(arms)
    native = selected & set(NATIVE_ARMS)
    dag_backed = selected & DAG_BACKED_ARMS
    resolver_family = {gen.VALUE_SOURCE_NNUE_STATIC, readout.ARM_QSEARCH,
                       readout.ARM_QSEARCH_DAG}
    qsearch_family = {readout.ARM_QSEARCH, readout.ARM_QSEARCH_DAG}
    if args.nnue_resolver_max_depth is not None and not (selected & resolver_family):
        raise ValueError(
            "--nnue-resolver-max-depth is read by nnue-static and the qsearch "
            "family; no such arm is selected",
        )
    if any(
        v is not None
        for v in (args.nnue_qsearch_max_ply, args.nnue_qsearch_check_plies)
    ) and not (selected & qsearch_family):
        raise ValueError(
            "--nnue-qsearch-* knobs were supplied but no qsearch-family arm is "
            "selected",
        )
    if args.dag_node_cap is not None and readout.ARM_QSEARCH_DAG not in selected:
        raise ValueError(
            "--dag-node-cap was supplied but nnue-qsearch-dag is not selected",
        )
    if any(
        v is not None for v in (
            args.fastq_max_qply, args.fastq_node_cap,
            args.fastq_delta_margin, args.fastq_recapture_exempt,
        )
    ) and readout.ARM_FASTQ not in selected:
        raise ValueError(
            "--fastq-* knobs were supplied but nnue-fastq is not selected",
        )
    if args.nnue_pack is not None and not native:
        raise ValueError(
            f"--nnue-pack is the NATIVE arms' weights and only {NATIVE_ARMS} "
            "evaluate with it; no native arm is selected, so the pack would be "
            "read by nothing and not even stamped",
        )
    if args.nnue_cp_per_unit is not None and not native:
        raise ValueError(
            "--nnue-cp-per-unit converts the NATIVE arms' internal units to "
            "cp; the Stockfish arms are handed cp already and never read it. "
            "No native arm is selected, and the report used to stamp this "
            "value anyway.",
        )
    if args.bank_observations is not None and not native:
        raise ValueError(
            "--bank-observations banks the NATIVE arms' raw scores through the "
            "generator's own leaf bank; no native arm is selected, so no bank "
            "file would be written. The Stockfish arms bank through "
            "--dump-per-position --dump-move-values instead.",
        )
    if any(
        v is not None for v in (args.dag_max_nodes, args.dag_reset_every)
    ) and not dag_backed:
        raise ValueError(
            "--dag-max-nodes / --dag-reset-every bound and reset a DAG-backed "
            f"arm's canonical store; none of {sorted(DAG_BACKED_ARMS)} is "
            "selected, so there is no store to watch or drop",
        )


def config_from_args(args: argparse.Namespace) -> GateConfig:
    arms, sf_specs = parse_arms(str(args.arms))
    validate_knobs(args, arms)
    audit_set = Path(args.audit_set)
    if audit_set.name.endswith(audit_targets.SHALLOW_SF_CACHE_SUFFIX):
        raise ValueError(
            f"{audit_set} is the shallow-SF cache, not the frozen deep set. It "
            "was produced on a dirty shared transposition table and is banned "
            "as a label source; point --audit-set at the deep JSONL.",
        )
    native = tuple(a for a in arms if a in NATIVE_ARMS)
    if native and args.nnue_pack is None:
        raise ValueError(
            f"the native arms {native} need --nnue-pack: the weights are what "
            "they evaluate with",
        )
    sf_binary: Path | None = None
    if sf_specs:
        found = str(args.stockfish) if args.stockfish else find_stockfish()
        if not found:
            raise ValueError(
                f"the arms {tuple(sf_specs)} need a Stockfish binary and none "
                "was discovered; pass --stockfish",
            )
        sf_binary = Path(found)
    sigma = (
        shadow.oneply_sigma_default() if args.oneply_sigma is None
        else float(args.oneply_sigma)
    )
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            f"--oneply-sigma must be finite and > 0, got {sigma!r}: a "
            "non-positive sigma flattens or inverts every label without "
            "failing any check",
        )
    # ⚑ `None` here is "not supplied", which `validate_knobs` has already used
    # to decide whether the knob had a consumer at all. The fallbacks are the
    # values the eager argparse defaults used to carry, so an unset run is
    # unchanged and only an EXPLICIT knob on a consumer-less run is refused.
    cp_per_internal_unit = (
        gen.NNUE_CP_PER_INTERNAL_UNIT if args.nnue_cp_per_unit is None
        else float(args.nnue_cp_per_unit)
    )
    dag_max_nodes = (
        shadow.DEFAULT_DAG_MAX_NODES if args.dag_max_nodes is None
        else int(args.dag_max_nodes)
    )
    dag_reset_every = (
        DEFAULT_DAG_RESET_EVERY if args.dag_reset_every is None
        else int(args.dag_reset_every)
    )
    if dag_max_nodes < 0 or dag_reset_every < 0:
        raise ValueError("--dag-max-nodes and --dag-reset-every must be >= 0")
    # ⚑ The readout resolver's DAG-cap refusal is about the SIBLING tool's
    # decomposition: it needs nnue-qsearch-dag to stay bit-identical to
    # nnue-qsearch. Here the DAG arm is an arm UNDER TEST, so a capped arm is
    # simply a different candidate whose trips are published in its provider
    # stats. Nothing in this file compares the two as an oracle.
    resolver_args = argparse.Namespace(
        **{**vars(args), "allow_binding_dag_node_cap": True},
    )
    return GateConfig(
        audit_set=audit_set,
        pack=Path(args.nnue_pack) if args.nnue_pack is not None else Path(),
        arms=arms,
        native_configs={
            a: readout.resolve_arm_config(
                resolver_args, arm=a, strict_foreign_knobs=False,
            )
            for a in native if a != gen.VALUE_SOURCE_NNUE_STATIC
        },
        sf_specs=sf_specs,
        static_resolver_max_depth=(
            None if args.nnue_resolver_max_depth is None
            else int(args.nnue_resolver_max_depth)
        ),
        limit=int(args.limit),
        oneply_sigma=sigma,
        cp_per_internal_unit=cp_per_internal_unit,
        cp_slope=float(args.nnue_cp_slope),
        cp_draw_width=float(args.nnue_cp_draw_width),
        dag_max_nodes=dag_max_nodes,
        dag_reset_every=dag_reset_every,
        sf_binary=sf_binary,
        sf_hash_mb=int(args.sf_hash_mb),
        sf_fresh_per_position=bool(args.sf_fresh_per_position),
        nice=int(args.nice),
        run_id=str(args.run_id),
        dump_per_position=(
            None if args.dump_per_position is None else Path(args.dump_per_position)
        ),
        dump_move_values=bool(args.dump_move_values),
        bank_observations=(
            None if args.bank_observations is None else Path(args.bank_observations)
        ),
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialise {type(value).__name__} into the report")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    report = run(config_from_args(args))
    print(format_table(report))
    text = json.dumps(report, indent=2, sort_keys=True, default=_json_default)
    if args.json is not None:
        readout._atomic_write_text(Path(args.json), text + "\n")
    else:
        print(text)
    # The artifact is written before the verdict is returned, for the same
    # reason `nnue_gumbel_readout.main` does it: a gate that exits before the
    # JSON exists destroys the evidence for the finding it just made.
    return 0 if report["admissible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
