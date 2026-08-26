#!/usr/bin/env python3
"""Production-Gumbel readout for the native qsearch/DAG/FastQ value arms.

This is deliberately a *driver* around ``gen_random_selfplay_shards.play_game``
rather than another search implementation.  The game loop, C ``MCTSTree``
Gumbel search, per-root simulation budget, root-value plumbing, pending-leaf
``CBoard`` binding, target construction, and terminal handling are therefore the
same code used by the generation-zero shard generator.

The one thing this file owns is the experiment boundary that the generator does
not yet have:

* ``nnue-qsearch`` is the non-DAG control;
* ``nnue-qsearch-dag`` runs the identical qsearch on the canonical position DAG;
* ``nnue-fastq`` runs FastQ-4+ on that same DAG.

⚑⚑ THE FIRST TWO CELLS ARE A FREE ORACLE, AND THIS TOOL SPENDS IT.  #472 proved
the DAG retrofit bit-identical to the non-DAG qsearch, and every other input to
a game is a pure function of the seed: ``gumbel_c.py`` draws exactly
``legal_idx.size`` uniforms per ply, ``sample_action_with_temperature`` draws
none at ``temperature <= 0`` (production, and this tool's, ``DEFAULT_TEMPERATURE``
is ``0.0``), and ``sample_starting_board`` short-circuits with no book.  So
``nnue-qsearch`` and ``nnue-qsearch-dag`` at one seed MUST play byte-identical
games.  Each cell therefore publishes a per-game digest over
``(game_index, start_fen, move_trace, result, termination)`` and a ``games_digest``
over all of them.  **Differing digests VOID the decomposition**: the DAG cell is
then no longer the same experiment as the control and no wall-clock difference
between them can be attributed to the substrate.

DAG-backed arms persist across *plies within one game* and are reset before the
next game by default.  ``--dag-reset`` exposes the cadence, because
``docs/fastq_design.md`` §4.4 says the persistence policy is to be CHOSEN FROM
MEASUREMENT and this harness is that measurement -- a harness that can only
produce the one preselected point cannot inform the choice it exists to inform.

Nothing here installs a DAG-backed provider in ``MCTSTree``.  Those providers
correctly declare ``requires_gil`` and the tree refuses them.  The existing gen-0
path instead asks the tree for its pending leaf ``CBoard`` objects and evaluates
that batch externally through ``arm_handle_eval``; this driver uses exactly that
path, so the concurrency guard is neither weakened nor bypassed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from chess_anti_engine.encoding import rep_fix
from scripts import gen_random_selfplay_shards as gen


_LOG = logging.getLogger("nnue_gumbel_readout")

ARM_QSEARCH = "nnue-qsearch"
ARM_QSEARCH_DAG = "nnue-qsearch-dag"
ARM_FASTQ = "nnue-fastq"
READOUT_ARMS: tuple[str, ...] = (ARM_QSEARCH, ARM_QSEARCH_DAG, ARM_FASTQ)
#: The two cells #472's bit-identity proof makes comparable, in report order.
ORACLE_ARMS: tuple[str, str] = (ARM_QSEARCH, ARM_QSEARCH_DAG)

#: Report schema. 2 replaced the single-cell top level with ``cells`` +
#: ``repeats`` + ``provenance``; a consumer that reads schema 1's flat keys off
#: a schema 2 file gets a KeyError rather than a plausible wrong number.
REPORT_SCHEMA = 2

StatsSurface = Literal["arm", "fastq"]

DAG_RESET_EVERY_GAME = 1
DAG_RESET_NEVER = 0
_DAG_RESET_EVERY_N = re.compile(r"^every-(\d+)-games$")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    uses_dag: bool
    stats_surface: StatsSurface
    consumes_qsearch_knobs: bool
    consumes_fastq_knobs: bool


ARM_SPECS: dict[str, ArmSpec] = {
    ARM_QSEARCH: ArmSpec(ARM_QSEARCH, False, "arm", True, False),
    ARM_QSEARCH_DAG: ArmSpec(ARM_QSEARCH_DAG, True, "arm", True, False),
    ARM_FASTQ: ArmSpec(ARM_FASTQ, True, "fastq", False, True),
}


# ── provider counter classification ──────────────────────────────────────────
# ⚑⚑ EVERY KEY MUST BE CLASSIFIED BEFORE IT CAN BE MERGED, and that rule is the
# generator's, reused rather than re-transcribed.  `NnueArmStats.merge` RAISES on
# an unclassified `arm_stats` key ("a new C counter must be classified before it
# can be merged"); an earlier revision of this file dropped that guard and
# silently SUMMED whatever it was handed.  The three keys that cost was
# `dag_node_count` / `dag_edge_count` / `dag_memory_bytes`: they are the CURRENT
# SIZE of a worker's store, reset by `arm_dag_reset`, so their cross-worker
# "total" means "the sum of N workers' last-reset-window endpoints" -- a resource
# figure, and emphatically not a count of positions the run saw.  Filing them as
# counters merges them identically and loses exactly that caveat, which is this
# repo's signature defect wearing a plausible name.
@dataclass(frozen=True)
class KeyClasses:
    """How each key of one provider's stats dict merges across workers."""

    counters: frozenset[str]
    peaks: frozenset[str]
    config: frozenset[str]
    store_sizes: frozenset[str]

    def classify(self, key: str) -> str:
        if key in self.config:
            return "config"
        if key in self.peaks:
            return "peak"
        if key in self.store_sizes:
            return "store_size"
        if key in self.counters:
            return "counter"
        raise ValueError(
            f"provider stats key {key!r} is not classified as a counter, a peak, "
            "a store size or a config value; add it to the right set in "
            "scripts/nnue_gumbel_readout.py (or, for the arm surface, to "
            "_CTX_*_KEYS in scripts/gen_random_selfplay_shards.py) before "
            "merging it",
        )


#: The qsearch/DAG surface: the generator's own classification, not a copy of
#: it.  A new C counter therefore has to be classified in exactly one place.
ARM_KEY_CLASSES = KeyClasses(
    counters=gen._CTX_COUNTER_KEYS,
    peaks=gen._CTX_PEAK_KEYS,
    config=gen._CTX_CONFIG_KEYS,
    store_sizes=gen._CTX_STORE_SIZE_KEYS,
)

#: FastQ's surface has no equivalent in the generator (it never opens the arm),
#: so its classification lives here.  `max_ply_seen` is the only peak; the four
#: knobs are the context's own snapshot and must AGREE across workers; FastQ
#: publishes no store size (the DAG's own sizes come from `arm_dag_stats`).
FASTQ_KEY_CLASSES = KeyClasses(
    counters=frozenset({
        "calls", "nodes", "evasion_nodes", "nodes_created",
        "nodes_created_in_check", "nnue_evals", "hits_within_call",
        "hits_cross_call", "quiet_certificates", "quiet_certificate_hits",
        "quiet_returns", "see_prunes", "delta_prunes", "recapture_exemptions",
        "stand_pat_cutoffs", "move_cutoffs", "budget_trips", "path_ceilings",
        "cycle_draws", "terminal_mate", "terminal_draw",
    }),
    peaks=frozenset({"max_ply_seen"}),
    config=frozenset({
        "max_qply", "node_cap", "delta_margin", "see_recapture_exempt",
    }),
    store_sizes=frozenset(),
)


def key_classes_for(arm: str) -> KeyClasses:
    return (
        FASTQ_KEY_CLASSES if ARM_SPECS[arm].stats_surface == "fastq"
        else ARM_KEY_CLASSES
    )


@dataclass(frozen=True)
class ResolvedArmConfig:
    arm: str
    resolver_max_depth: int | None = None
    qsearch_max_ply: int | None = None
    qsearch_check_plies: int | None = None
    dag_node_cap: int | None = None
    fastq_max_qply: int | None = None
    fastq_node_cap: int | None = None
    fastq_delta_margin: int | None = None
    fastq_recapture_exempt: int | None = None

    def consumed(self) -> dict[str, int]:
        spec = ARM_SPECS[self.arm]
        if spec.consumes_qsearch_knobs:
            out = {
                "resolver_max_depth": _required(self.resolver_max_depth, "resolver_max_depth"),
                "qsearch_max_ply": _required(self.qsearch_max_ply, "qsearch_max_ply"),
                "qsearch_check_plies": _required(
                    self.qsearch_check_plies, "qsearch_check_plies",
                ),
            }
            if self.arm == ARM_QSEARCH_DAG:
                out["dag_node_cap"] = _required(self.dag_node_cap, "dag_node_cap")
            return out
        return {
            "max_qply": _required(self.fastq_max_qply, "fastq_max_qply"),
            "node_cap": _required(self.fastq_node_cap, "fastq_node_cap"),
            "delta_margin": _required(self.fastq_delta_margin, "fastq_delta_margin"),
            "see_recapture_exempt": _required(
                self.fastq_recapture_exempt, "fastq_recapture_exempt",
            ),
        }


def _required(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"internal error: consumed arm knob {name} was not resolved")
    return int(value)


def _load_ext() -> Any:
    from chess_anti_engine.nnue import _nnue_ext

    return _nnue_ext


def resolve_arm_config(
    args: argparse.Namespace, ext: Any | None = None, *, arm: str | None = None,
) -> ResolvedArmConfig:
    """Resolve only knobs the selected provider consumes; refuse every other one.

    The refusal is intentional.  A flag that is accepted, printed, and then not
    read by the selected C provider is this repository's signature defect.  The
    CLI therefore uses ``None`` as "not supplied", fills defaults from the
    extension only for the selected arm, and errors if a caller supplies a knob
    belonging to another arm.
    """
    # ⚑ A SEPARATE NAME WITH AN EXPLICIT `Any`, not a reassignment of `ext`.
    # Assigning an `Any` value back into a parameter declared `Any | None`
    # re-narrows it to the DECLARED type, so every `ext.X` below stayed an
    # `Optional` member access to basedpyright -- 25 findings, and CI red.  The
    # fix is to give the resolved handle its own name rather than to silence the
    # rule: the rule was reading the code correctly.
    resolved_ext: Any = _load_ext() if ext is None else ext
    selected = str(args.arm) if arm is None else str(arm)
    if selected not in ARM_SPECS:
        raise ValueError(f"arm must be one of {READOUT_ARMS}, got {selected!r}")
    spec = ARM_SPECS[selected]

    q_values = (
        args.nnue_resolver_max_depth,
        args.nnue_qsearch_max_ply,
        args.nnue_qsearch_check_plies,
        args.dag_node_cap,
    )
    f_values = (
        args.fastq_max_qply,
        args.fastq_node_cap,
        args.fastq_delta_margin,
        args.fastq_recapture_exempt,
    )
    if spec.consumes_qsearch_knobs and any(v is not None for v in f_values):
        raise ValueError(
            f"{selected} does not consume --fastq-* knobs; remove them rather "
            "than recording settings the selected provider will ignore",
        )
    if spec.consumes_fastq_knobs and any(v is not None for v in q_values):
        raise ValueError(
            f"{selected} does not consume qsearch/resolver/DAG-qsearch knobs; "
            "remove them rather than recording settings the selected provider "
            "will ignore",
        )

    if spec.consumes_qsearch_knobs:
        dag_cap = args.dag_node_cap
        if selected == ARM_QSEARCH and dag_cap is not None:
            raise ValueError(
                "nnue-qsearch has no DAG and cannot consume --dag-node-cap",
            )
        # ⚑⚑ A BINDING CAP VOIDS THE CONTROL, AND SILENTLY. `set_arm_config`'s
        # own docstring says it: above 0 a node that trips the cap "stands pat
        # and increments dag_budget_trips, so an arm with a binding cap no
        # longer matches the oracle". The DAG cell exists to be bit-identical to
        # the non-DAG one; a cap turns it into a different search whose wall
        # time is not attributable to the substrate, and the per-game digests
        # would diverge with no explanation on the face of the report.
        if (
            selected == ARM_QSEARCH_DAG
            and dag_cap is not None
            and int(dag_cap) > 0
            and not bool(getattr(args, "allow_binding_dag_node_cap", False))
        ):
            raise ValueError(
                "--dag-node-cap > 0 makes nnue-qsearch-dag stop matching the "
                "nnue-qsearch oracle (set_arm_config: a node that trips the cap "
                "stands pat), so the control cell would no longer be a control. "
                "Pass --allow-binding-dag-node-cap if you are deliberately "
                "measuring the capped arm and accept that the decomposition is "
                "void.",
            )
        return ResolvedArmConfig(
            arm=selected,
            resolver_max_depth=(
                int(resolved_ext.RESOLVER_MAX_DEPTH)
                if args.nnue_resolver_max_depth is None
                else int(args.nnue_resolver_max_depth)
            ),
            qsearch_max_ply=(
                int(resolved_ext.QSEARCH_MAX_PLY)
                if args.nnue_qsearch_max_ply is None
                else int(args.nnue_qsearch_max_ply)
            ),
            qsearch_check_plies=(
                int(resolved_ext.QSEARCH_CHECK_PLIES)
                if args.nnue_qsearch_check_plies is None
                else int(args.nnue_qsearch_check_plies)
            ),
            dag_node_cap=(
                int(resolved_ext.QSEARCH_DAG_NODE_CAP)
                if selected == ARM_QSEARCH_DAG and dag_cap is None
                else (None if dag_cap is None else int(dag_cap))
            ),
        )

    return ResolvedArmConfig(
        arm=selected,
        fastq_max_qply=(
            int(resolved_ext.FASTQ_MAX_QPLY)
            if args.fastq_max_qply is None else int(args.fastq_max_qply)
        ),
        fastq_node_cap=(
            int(resolved_ext.FASTQ_NODE_CAP)
            if args.fastq_node_cap is None else int(args.fastq_node_cap)
        ),
        fastq_delta_margin=(
            int(resolved_ext.FASTQ_DELTA_MARGIN)
            if args.fastq_delta_margin is None else int(args.fastq_delta_margin)
        ),
        fastq_recapture_exempt=(
            int(resolved_ext.FASTQ_RECAPTURE_EXEMPT)
            if args.fastq_recapture_exempt is None
            else int(args.fastq_recapture_exempt)
        ),
    )


def readout_arm_config_plan(config: ResolvedArmConfig) -> gen.ArmConfigPlan:
    """The setter/stats pair the selected provider owns, as one picklable plan.

    ⚑ ``consumed`` is the CALLER's dict, and the plan hands it to
    ``NnueArmValueSource`` as ``requested`` -- not the setter's return value.
    The setter's return is the producer's copy of the request after any clamp
    the setter applied, and a clamp that also landed in the context would make
    the requested-vs-realized check compare a clamped value with itself.
    """
    spec = ARM_SPECS[config.arm]
    consumed = config.consumed()
    if spec.consumes_qsearch_knobs:
        # `set_arm_config` RESETS dag_node_cap when the argument is omitted, so
        # the non-DAG arm passes an explicit 0 rather than relying on call
        # history to leave it there.
        dag_cap = consumed["dag_node_cap"] if config.arm == ARM_QSEARCH_DAG else 0
        return gen.ArmConfigPlan(
            setter="set_arm_config",
            setter_args=(
                consumed["resolver_max_depth"],
                consumed["qsearch_max_ply"],
                consumed["qsearch_check_plies"],
                dag_cap,
            ),
            consumed=consumed,
            stats="arm_stats",
            consumes_qsearch=True,
        )
    return gen.ArmConfigPlan(
        setter="fastq_set_config",
        setter_args=(
            consumed["max_qply"],
            consumed["node_cap"],
            consumed["delta_margin"],
            consumed["see_recapture_exempt"],
        ),
        consumed=consumed,
        stats="fastq_stats",
        consumes_qsearch=False,
    )


def _atomic_write_text(path: Path, text: str) -> None:

    """Replace a result artifact all-or-nothing on the same filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_name: str | None = None

    try:

        with tempfile.NamedTemporaryFile(

            mode="w", encoding="utf-8", dir=path.parent,

            prefix=f".{path.name}.", suffix=".tmp", delete=False,

        ) as f:

            tmp_name = f.name

            f.write(text)

            f.flush()

            os.fsync(f.fileno())

        os.replace(tmp_name, path)

        tmp_name = None

    finally:

        if tmp_name is not None:

            try:

                Path(tmp_name).unlink()

            except FileNotFoundError:

                pass





def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


#: This process's niceness before the harness touched it, as a one-element list
#: so the update needs no ``global``. See ``_apply_nice``.
_NICE_BASELINE: list[int] = []


def _apply_nice(delta: int) -> int:
    """Renice this worker to ``baseline + delta``, and REPORT what it got.

    ⚑ Two failures, not one.  Swallowing the ``OSError`` (the shape this file
    shipped with) hides a renice that could not be applied; reporting the
    REQUESTED value hides it just as thoroughly when the call succeeded but was
    clamped.  ``os.nice(0)`` returns the current niceness without changing it,
    so the realized figure is read back from the process rather than restated.

    ⚑⚑ AND IT IS ABSOLUTE, NOT AN INCREMENT, WHICH THE REPORTING IS WHAT
    EXPOSED.  ``os.nice`` is relative, and with ``--workers 1`` the worker body
    runs in the PARENT process, so a matrix of three cells reniced the same
    process three times: measured, the first cell ran at 10 and the third at 19
    (the kernel's ceiling).  The cells were then not one experiment, and the
    old code -- which published the requested 10 and nothing else -- could not
    have said so.
    """
    current = int(os.nice(0))
    if not _NICE_BASELINE:
        _NICE_BASELINE.append(current)
    target = _NICE_BASELINE[0] + int(delta)
    step = target - current
    if step > 0:
        try:
            os.nice(step)
        except OSError as exc:  # pragma: no cover - only without privileges
            _LOG.warning("could not renice by %d: %s", step, exc)
    elif step < 0:
        # Lowering niceness needs privileges; say so rather than reporting the
        # request as if it had been honoured.
        _LOG.warning(
            "cannot lower niceness from %d to %d without privileges", current, target,
        )
    return int(os.nice(0))


class ReadoutArmSource(gen.NnueArmValueSource):
    """``NnueArmValueSource`` generalized to the two DAG-backed readout providers.

    ⚑ THIS SUBCLASS ADDS NO STATE.  It widens the parent's arm whitelist and
    hands it a different ``ArmConfigPlan``; everything else -- the batch/Q
    conversion, the leaf bank, the requested-vs-realized check, the two pack
    hashes -- is the parent's, called through ``super().__init__()``.  The
    previous shape reproduced the parent's constructor to get past the
    whitelist, and had already drifted from it: it never set
    ``consumes_qsearch``.  ``test_the_readout_arm_source_adds_no_state_of_its_own``
    is what keeps that from happening again.

    The stats surface is chosen by the plan, so the FastQ handle is NEVER passed
    to ``arm_stats`` (which correctly raises) and a qsearch handle is never
    passed to ``fastq_stats``.
    """

    _ALLOWED_ARMS: ClassVar[tuple[str, ...]] = READOUT_ARMS

    def __init__(
        self,
        *,
        config: ResolvedArmConfig,
        pack: Path,
        cp_per_internal_unit: float,
        cp_slope: float,
        cp_draw_width: float,
        leaf_bank: Path | None = None,
        ext: Any | None = None,
        pack_file_sha256: str | None = None,
        bank_identity: dict[str, Any] | None = None,
    ) -> None:
        if leaf_bank is not None:
            leaf_bank.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(
            arm=config.arm,
            pack=Path(pack),
            cp_per_internal_unit=cp_per_internal_unit,
            cp_slope=cp_slope,
            cp_draw_width=cp_draw_width,
            leaf_bank=leaf_bank,
            plan=readout_arm_config_plan(config),
            ext=ext,
            pack_file_sha256=pack_file_sha256,
            bank_identity=bank_identity,
        )

    @property
    def spec(self) -> ArmSpec:
        """Derived from ``arm``, never stored: a second copy can disagree."""
        return ARM_SPECS[self.arm]

    def dag_stats(self) -> dict[str, int] | None:
        if not self.spec.uses_dag:
            return None
        return dict(self._ext.arm_dag_stats(self._handle))

    def reset_game(self) -> None:
        """Reset only canonical graph state; provider counters remain cumulative."""
        if self.spec.uses_dag:
            self._ext.arm_dag_reset(self._handle)


class ReadoutEvaluator(gen.UniformPriorEvaluator):
    """The generator evaluator with its native-source whitelist widened explicitly.

    ⚑ Both whitelists are widened, not one.  The parent cross-checks
    ``(value_source in _NATIVE_VALUE_SOURCES) != (nnue_source is not None)``;
    widening only ``_ALLOWED_VALUE_SOURCES`` would let a readout arm through
    that check with its source quietly optional.
    """

    _ALLOWED_VALUE_SOURCES: ClassVar[tuple[str, ...]] = READOUT_ARMS
    _NATIVE_VALUE_SOURCES: ClassVar[tuple[str, ...]] = READOUT_ARMS

    def __init__(
        self,
        *,
        source: ReadoutArmSource,
        expected_planes: int,
        input_history_encoding: str,
        input_extra_features: str,
    ) -> None:
        super().__init__(
            value_source=source.arm,
            expected_planes=expected_planes,
            nnue_source=source,
            input_history_encoding=input_history_encoding,
            input_extra_features=input_extra_features,
        )


@dataclass(frozen=True)
class RunConfig:
    """One CELL: one arm, one repeat, everything the workers need to reproduce it."""

    arm_config: ResolvedArmConfig
    pack: Path
    pack_file_sha256: str
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
    bank_path: Path | None
    run_id: str
    nice: int
    dag_reset_every: int = DAG_RESET_EVERY_GAME
    repeat: int = 0


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    game_indices: tuple[int, ...]
    cfg: RunConfig


@dataclass(frozen=True)
class GameRecord:
    """One game's identity, so two cells can be shown to have played the same run.

    ⚑ ``_run_worker`` used to keep ``outcome.plies`` and a termination bucket and
    throw ``result`` / ``move_trace`` / ``start_fen`` away.  Those three are the
    entire content of the qsearch-vs-DAG oracle: without them the two cells are
    two wall-clock numbers with no evidence they searched the same positions.
    """

    game: int
    plies: int
    result: str
    termination: str
    digest: str


def game_digest(
    *, game_index: int, start_fen: str, move_trace: str, result: str, termination: str,
) -> str:
    """The per-game identity the qsearch/DAG oracle compares.

    Everything a differing search could change is in it: where the game started,
    every move played, how it ended, and which terminal rule ended it.
    """
    payload = f"{game_index}:{start_fen}:{move_trace}:{result}:{termination}"
    return hashlib.sha256(payload.encode()).hexdigest()


def games_digest(records: list[GameRecord]) -> str:
    """One digest over a cell's whole game set, ordered by game index."""
    h = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.game):
        h.update(f"{record.game}:{record.digest}\n".encode())
    return h.hexdigest()


@dataclass
class DagGameStats:
    games: int = 0
    nodes_sum: int = 0
    edges_sum: int = 0
    hits_sum: int = 0
    probes_sum: int = 0
    inserts_sum: int = 0
    state_inits_sum: int = 0
    state_makes_sum: int = 0
    memory_peak: int = 0
    nodes_peak: int = 0
    edges_peak: int = 0
    #: Snapshots where ``state_inits + state_makes != node_count``. See
    #: ``arm_dag_stats``'s own docstring: the arm's interning "has to keep
    #: satisfying" it, so a non-zero count is a broken substrate, not a finding.
    state_identity_violations: int = 0

    def add(self, stats: dict[str, int]) -> None:
        """Fold one ``arm_dag_stats()`` snapshot in, by its REAL key names.

        ⚑ SUBSCRIPT, NOT ``.get``.  This read used to be
        ``stats.get("nodes", stats.get("node_count", 0))``; the C layer publishes
        ``node_count`` and has never published ``nodes``, so the live branch was
        the fallback -- and the test fake returned the dead names, so every test
        exercised the branch production cannot reach.  Deleting the fallback in
        that shape left the suite green and made every real run report
        ``nodes_per_game: 0.0``.
        """
        self.games += 1
        nodes = int(stats["node_count"])
        edges = int(stats["edge_count"])
        memory = int(stats["memory_bytes"])
        state_inits = int(stats["state_inits"])
        state_makes = int(stats["state_makes"])
        self.nodes_sum += nodes
        self.edges_sum += edges
        self.hits_sum += int(stats["hits"])
        self.probes_sum += int(stats["probes"])
        self.inserts_sum += int(stats["inserts"])
        self.state_inits_sum += state_inits
        self.state_makes_sum += state_makes
        if state_inits + state_makes != nodes:
            self.state_identity_violations += 1
        self.memory_peak = max(self.memory_peak, memory)
        self.nodes_peak = max(self.nodes_peak, nodes)
        self.edges_peak = max(self.edges_peak, edges)

    def merge(self, other: DagGameStats) -> None:
        self.games += other.games
        self.nodes_sum += other.nodes_sum
        self.edges_sum += other.edges_sum
        self.hits_sum += other.hits_sum
        self.probes_sum += other.probes_sum
        self.inserts_sum += other.inserts_sum
        self.state_inits_sum += other.state_inits_sum
        self.state_makes_sum += other.state_makes_sum
        self.state_identity_violations += other.state_identity_violations
        self.memory_peak = max(self.memory_peak, other.memory_peak)
        self.nodes_peak = max(self.nodes_peak, other.nodes_peak)
        self.edges_peak = max(self.edges_peak, other.edges_peak)

    def summary(self) -> dict[str, float]:
        if self.games == 0:
            return {}
        return {
            "games": float(self.games),
            "nodes_per_game": self.nodes_sum / self.games,
            "edges_per_game": self.edges_sum / self.games,
            "canonical_hit_rate": self.hits_sum / max(1, self.probes_sum),
            "hits": float(self.hits_sum),
            "probes": float(self.probes_sum),
            "inserts": float(self.inserts_sum),
            "state_inits": float(self.state_inits_sum),
            "state_makes": float(self.state_makes_sum),
            "nodes_peak_per_game": float(self.nodes_peak),
            "edges_peak_per_game": float(self.edges_peak),
            "memory_peak_per_worker_bytes": float(self.memory_peak),
        }


@dataclass
class WorkerResult:
    worker_id: int
    games: int
    plies: int
    setup_s: float
    elapsed_s: float
    terminations: dict[str, int]
    policy_shape: dict[str, float]
    root_budget: dict[str, float]
    provider_stats: dict[str, int]
    dag: DagGameStats
    game_records: list[GameRecord]
    eval_batches: int
    eval_rows: int
    arm_batches: int
    arm_leaves: int
    arm_roots: int
    mate_band_leaves: int
    mate_band_roots: int
    bank_rows: int
    bank_file: str | None
    kernel: str
    pack_file_sha256: str
    pack_source_sha256: str
    nice_realized: int
    arm_config_requested: dict[str, int] = field(default_factory=dict)
    arm_config_realized: dict[str, int] = field(default_factory=dict)


def _base_gen_config(cfg: RunConfig) -> gen.GenConfig:
    # `out_dir` is unused by play_game; every other value below is consumed by
    # the imported production game/search path.  Keep search defaults sourced
    # from the generator module rather than copying their numeric spellings.
    return gen.GenConfig(
        out_dir=Path("."),
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
        value_source=cfg.arm_config.arm,
        all_root_moves=cfg.all_root_moves,
        nnue_pack=cfg.pack,
        nnue_cp_per_unit=cfg.cp_per_internal_unit,
        nnue_cp_slope=cfg.cp_slope,
        nnue_cp_draw_width=cfg.cp_draw_width,
        max_plies=cfg.max_plies,
        seed=cfg.seed,
        nice=cfg.nice,
        run_id=cfg.run_id,
    )


def _worker_bank_path(
    base: Path | None, *, arm: str, repeat: int, worker_id: int,
) -> Path | None:
    """One bank file per (arm, repeat, worker).

    ⚑ THE ARM IS IN THE NAME.  Without it the three cells of one matrix, aimed
    at one ``--bank-leaf-observations`` path as the doc told a reader to do,
    merge into one file whose rows come from three different searches.  The
    repeat index is there for the same reason, and because the file is opened
    ``"x"``: a rerun must fail loudly rather than append into last run's rows.
    """
    if base is None:
        return None
    suffix = base.suffix or ".jsonl"
    stem = base.name[: -len(suffix)] if base.name.endswith(suffix) else base.name
    return base.with_name(f"{stem}.{arm}.r{repeat:02d}.w{worker_id:02d}{suffix}")


def _run_worker(spec: WorkerSpec) -> WorkerResult:
    cfg = spec.cfg
    nice_realized = _apply_nice(cfg.nice)
    rep_fix.apply(True)
    # ⚑ SETUP IS TIMED SEPARATELY FROM SEARCH. `ext.load()` plus the arm_open()
    # mmap is per-worker fixed cost with nothing to do with the arm's speed, and
    # folding it into the throughput number makes a cheap arm look slower the
    # more workers it is split across. The pack FILE hash is not here at all any
    # more -- the parent does it once, before its own clock starts.
    setup_started = time.perf_counter()
    base = _base_gen_config(cfg)
    gcfg = gen.build_gumbel_config(base)
    opening_cfg = gen.build_opening_config(base)
    source = ReadoutArmSource(
        config=cfg.arm_config,
        pack=cfg.pack,
        pack_file_sha256=cfg.pack_file_sha256,
        cp_per_internal_unit=cfg.cp_per_internal_unit,
        cp_slope=cfg.cp_slope,
        cp_draw_width=cfg.cp_draw_width,
        leaf_bank=_worker_bank_path(
            cfg.bank_path, arm=cfg.arm_config.arm, repeat=cfg.repeat,
            worker_id=spec.worker_id,
        ),
        bank_identity={
            "run_id": cfg.run_id,
            "seed": int(cfg.seed),
            "repeat": int(cfg.repeat),
            "worker_id": int(spec.worker_id),
        },
    )
    evaluator = ReadoutEvaluator(
        source=source,
        expected_planes=gen.input_plane_count(base.input_extra_features),
        input_history_encoding=base.input_history_encoding,
        input_extra_features=base.input_extra_features,
    )
    setup_s = time.perf_counter() - setup_started
    policy = gen.PolicyShapeStats()
    budget = gen.RootBudgetStats()
    dag_games = DagGameStats()
    terminations = dict.fromkeys(gen.TERMINATIONS, 0)
    records: list[GameRecord] = []
    plies = 0
    started = time.perf_counter()
    try:
        for position, game_index in enumerate(spec.game_indices):
            # The reset cadence is a MEASURED choice, not a constant: see
            # `--dag-reset`. `every-1-games` is the preregistered default and
            # the shape this harness shipped with -- reset before every game,
            # including the first, so the lifecycle is one rule rather than a
            # special first-game branch. Reset retains allocations, which is
            # what lets memory_peak measure the worker's resident capacity while
            # semantic nodes stay scoped to the cadence window.
            if (
                cfg.dag_reset_every > 0
                and position % cfg.dag_reset_every == 0
            ):
                source.reset_game()
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
            plies += int(outcome.plies)
            terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1
            records.append(GameRecord(
                game=int(game_index),
                plies=int(outcome.plies),
                result=str(outcome.result),
                termination=str(outcome.termination),
                digest=game_digest(
                    game_index=int(game_index),
                    start_fen=str(outcome.start_fen),
                    move_trace=str(outcome.move_trace),
                    result=str(outcome.result),
                    termination=str(outcome.termination),
                ),
            ))
            for row in outcome.records:
                policy.add(gen.policy_tv_to_uniform(row.policy_probs, row.legal_mask))
            dag = source.dag_stats()
            if dag is not None:
                dag_games.add(dag)
    finally:
        evaluator.close()
    elapsed = time.perf_counter() - started
    return WorkerResult(
        worker_id=spec.worker_id,
        games=len(spec.game_indices),
        plies=plies,
        setup_s=setup_s,
        elapsed_s=elapsed,
        terminations=terminations,
        policy_shape=policy.summary(),
        root_budget=budget.summary(),
        provider_stats=source.provider_stats(),
        dag=dag_games,
        game_records=records,
        eval_batches=evaluator.eval_calls,
        eval_rows=evaluator.eval_rows,
        arm_batches=source.stats.batches,
        arm_leaves=source.stats.leaves,
        arm_roots=source.stats.roots,
        mate_band_leaves=source.stats.mate_band_leaves,
        mate_band_roots=source.stats.mate_band_roots,
        bank_rows=source.bank_rows,
        bank_file=(
            None if source.leaf_bank_path is None else source.leaf_bank_path.name
        ),
        kernel=source.kernel,
        pack_file_sha256=source.pack_file_sha256,
        pack_source_sha256=source.pack_source_sha256,
        nice_realized=nice_realized,
        arm_config_requested=dict(source.requested),
        arm_config_realized=dict(source.realized),
    )


def merge_provider_stats(
    dicts: list[dict[str, int]], classes: KeyClasses,
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Merge N workers' provider counters by each key's declared class.

    Returns ``(merged, conflicts)``.  ⚑ A config key whose workers DISAGREE is
    RECORDED, not raised on: raising here throws away a finished multi-hour run
    at the aggregation step, and the parent (``NnueArmStats.merge``) already
    settled the question -- it publishes ``context_conflicts`` and keeps the
    lower value, because "a merged run whose workers disagreed is not one cell"
    is a finding to report rather than a reason to have no report.

    An UNCLASSIFIED key still raises.  That is not the same call: an unknown key
    means a new C counter reached this merge without anyone deciding how it
    merges, so there is no correct number to publish for it at all.
    """
    out: dict[str, int] = {}
    conflicts: dict[str, list[int]] = {}
    for d in dicts:
        for key, raw in d.items():
            value = int(raw)
            kind = classes.classify(key)
            if key not in out:
                out[key] = value
            elif kind == "config":
                if out[key] != value:
                    conflicts[key] = sorted(set(conflicts.get(key, [])) | {out[key], value})
                    # Keep the LOWER value, exactly as NnueArmStats.merge does:
                    # for a ceiling the run is only as deep as its shallowest
                    # worker, and the conflict is reported rather than smoothed.
                    out[key] = min(out[key], value)
            elif kind == "peak":
                out[key] = max(out[key], value)
            else:
                out[key] = out[key] + value
    return out, conflicts


def _fastq_rates(provider: dict[str, int]) -> dict[str, float | None]:
    """FastQ's per-call rates, or ``null`` when nothing ran.

    ⚑ SUBSCRIPT, AND ``None`` AT ZERO CALLS.  These were seven
    ``provider.get(key, 0) / max(1, provider.get("calls", 0))`` expressions, and
    both halves lied.  A missing ``budget_trips`` read 0.0 -- which the doc's
    read-order names as the HEALTHY value, so a counter that vanished from the C
    layer would report as a clean run.  And ``max(1, calls)`` made "nothing
    tripped over 4 million calls" and "nothing ran at all" the same number.
    """
    calls = int(provider["calls"])
    if calls == 0:
        return {
            "calls": 0.0,
            "budget_trip_rate": None,
            "created_nodes_per_call": None,
            "nnue_evals_per_call": None,
            "within_call_hits_per_call": None,
            "cross_call_hits_per_call": None,
            "see_prunes_per_call": None,
            "delta_prunes_per_call": None,
        }
    return {
        "calls": float(calls),
        "budget_trip_rate": int(provider["budget_trips"]) / calls,
        "created_nodes_per_call": int(provider["nodes_created"]) / calls,
        "nnue_evals_per_call": int(provider["nnue_evals"]) / calls,
        "within_call_hits_per_call": int(provider["hits_within_call"]) / calls,
        "cross_call_hits_per_call": int(provider["hits_cross_call"]) / calls,
        "see_prunes_per_call": int(provider["see_prunes"]) / calls,
        "delta_prunes_per_call": int(provider["delta_prunes"]) / calls,
    }


def _aggregate(results: list[WorkerResult], cfg: RunConfig, wall_s: float) -> dict[str, Any]:
    arm = cfg.arm_config.arm
    spec = ARM_SPECS[arm]
    provider, conflicts = merge_provider_stats(
        [r.provider_stats for r in results], key_classes_for(arm),
    )
    dag = DagGameStats()
    for r in results:
        dag.merge(r.dag)
    records = [rec for r in results for rec in r.game_records]
    plies = sum(r.plies for r in results)
    games = sum(r.games for r in results)
    arm_leaves = sum(r.arm_leaves for r in results)
    arm_roots = sum(r.arm_roots for r in results)
    arm_batches = sum(r.arm_batches for r in results)
    total_calls = arm_leaves + arm_roots
    terminations: dict[str, int] = {}
    for r in results:
        for key, value in r.terminations.items():
            terminations[key] = terminations.get(key, 0) + int(value)
    # ⚑ TWO WINDOWS, AND THEY ARE NOT THE SAME NUMBER. `wall_s` is the parent's
    # end-to-end clock: it includes pool startup and every worker's `ext.load()`
    # + `arm_open()` mmap. `search_wall_s` is the widest worker's SEARCH window
    # only, which is what a per-arm throughput comparison is about. The report
    # publishes both, each named for the window it measures, because a single
    # `plies_per_s` that silently mixed them is how the headline cell came to be
    # the one paying for the banking.
    search_wall_s = max((r.elapsed_s for r in results), default=0.0)
    reasons: list[str] = []
    if conflicts:
        reasons.append(f"workers disagreed about arm configuration: {conflicts}")
    if dag.state_identity_violations:
        reasons.append(
            f"{dag.state_identity_violations} arm_dag_stats snapshots violate "
            "state_inits + state_makes == node_count",
        )

    identities: dict[str, Any] = {
        "dag_state_identity_ok": dag.state_identity_violations == 0,
        "dag_state_identity_violations": dag.state_identity_violations,
    }
    if spec.stats_surface == "fastq":
        # docs/fastq_design.md §7: the evaluate-once invariant, as an ASSERTABLE
        # counter identity. Every term is already in this report, and nothing
        # was checking it.
        evaluate_once_ok = (
            int(provider["nnue_evals"]) + int(provider["nodes_created_in_check"])
            == int(provider["nodes_created"])
        )
        identities["evaluate_once_identity_ok"] = evaluate_once_ok
        identities["evaluate_once_identity"] = {
            "nnue_evals": int(provider["nnue_evals"]),
            "nodes_created_in_check": int(provider["nodes_created_in_check"]),
            "nodes_created": int(provider["nodes_created"]),
        }
        if not evaluate_once_ok:
            reasons.append(
                "evaluate-once identity violated: nnue_evals "
                f"{provider['nnue_evals']} + nodes_created_in_check "
                f"{provider['nodes_created_in_check']} != nodes_created "
                f"{provider['nodes_created']}",
            )
    if arm == ARM_QSEARCH_DAG and int(provider["dag_budget_trips"]) > 0:
        reasons.append(
            f"dag_budget_trips {provider['dag_budget_trips']} > 0: the DAG cell's "
            "node cap BOUND, so this arm stood pat where the oracle searched and "
            "is no longer the nnue-qsearch control",
        )

    readout: dict[str, Any] = {
        "arm": arm,
        "repeat": cfg.repeat,
        "games": games,
        # Active is the throughput denominator; requested is provenance.
        # When games < workers, _build_worker_specs drops empty buckets.
        "workers": len(results),
        "workers_requested": cfg.workers,
        "plies": plies,
        "wall_s": wall_s,
        "search_wall_s": search_wall_s,
        "setup_wall_s": max((r.setup_s for r in results), default=0.0),
        # End-to-end, including pool startup and per-worker context setup.
        "plies_per_s": plies / max(wall_s, 1e-12),
        "games_per_h": games * 3600.0 / max(wall_s, 1e-12),
        # Search only, over the widest worker's search window.
        "search_plies_per_s": plies / max(search_wall_s, 1e-12),
        "search_games_per_h": games * 3600.0 / max(search_wall_s, 1e-12),
        # ⚑ RENAMED. This is a sum of `time.perf_counter` deltas, i.e. WALL time
        # summed over workers -- it was called `worker_cpu_s`, which is a
        # different quantity (`time.process_time`) that this has never measured.
        "worker_wall_s": sum(r.elapsed_s for r in results),
        "worker_setup_wall_s": sum(r.setup_s for r in results),
        "arm_config": cfg.arm_config.consumed(),
        "arm_config_realized": _agree(
            [r.arm_config_realized for r in results], "arm_config_realized",
        ),
        "search": {
            "sims_floor": cfg.sims,
            "topk": cfg.topk,
            "all_root_moves": cfg.all_root_moves,
            "c_scale": gen.SELFPLAY_GUMBEL_C_SCALE,
            "policy_temp": gen.DEFAULT_POLICY_TEMP,
            "temperature": gen.DEFAULT_TEMPERATURE,
            "gumbel_scale": gen.DEFAULT_GUMBEL_SCALE,
            "vloss_weight": gen.DEFAULT_VLOSS_WEIGHT,
        },
        "arm_io": {
            "leaves": arm_leaves,
            "roots": arm_roots,
            "calls": total_calls,
            "batches": arm_batches,
            "leaves_per_batch": arm_leaves / max(1, arm_batches),
            "mate_band_leaves": sum(r.mate_band_leaves for r in results),
            "mate_band_roots": sum(r.mate_band_roots for r in results),
            "nnue_evals_per_top_level_call": (
                float(provider["nnue_evals"]) / max(1, total_calls)
            ),
        },
        "provider_stats": provider,
        "provider_stats_classification": _classification(provider, key_classes_for(arm)),
        "provider_stats_conflicts": conflicts,
        "identities": identities,
        "dag_per_game": dag.summary(),
        "dag_reset": _dag_reset_label(cfg.dag_reset_every),
        "terminations": terminations,
        "games_digest": games_digest(records),
        "games_detail": [asdict(rec) for rec in sorted(records, key=lambda r: r.game)],
        "bank_rows": sum(r.bank_rows for r in results),
        "bank_files": [r.bank_file for r in results if r.bank_file is not None],
        "nice_realized": sorted({r.nice_realized for r in results}),
        "inadmissible_reasons": reasons,
        "admissible": not reasons,
        # `game_records` is dropped here: it is republished whole as
        # `games_detail` above, and a per-worker copy doubles the largest array
        # in the file for no extra fact.
        "workers_detail": [
            {k: v for k, v in asdict(r).items() if k != "game_records"}
            for r in results
        ],
    }
    if arm == ARM_FASTQ:
        readout["fastq"] = _fastq_rates(provider)
    return readout


def _classification(provider: dict[str, int], classes: KeyClasses) -> dict[str, list[str]]:
    """Which merge rule produced each published number.

    ⚑ The store-size keys are the reason this block exists.  Their cross-worker
    merge is a SUM, and the sum is a resource endpoint total -- how many nodes
    and bytes the N stores held at their last snapshot -- not a count of
    positions the run visited.  A reader who assumes "counter" for everything in
    ``provider_stats`` reads ``dag_node_count`` as the second thing.
    """
    out: dict[str, list[str]] = {
        "counters": [], "peaks": [], "config": [], "store_endpoint_sizes": [],
    }
    bucket = {
        "counter": "counters", "peak": "peaks", "config": "config",
        "store_size": "store_endpoint_sizes",
    }
    for key in sorted(provider):
        out[bucket[classes.classify(key)]].append(key)
    return out


def _agree(dicts: list[dict[str, int]], what: str) -> dict[str, Any]:
    """One dict when every worker reported the same one; otherwise a conflict map."""
    if not dicts:
        return {}
    merged: dict[str, Any] = dict(dicts[0])
    for d in dicts[1:]:
        for key, value in d.items():
            if merged.get(key) != value:
                _LOG.error("%s disagrees on %s: %r vs %r", what, key, merged.get(key), value)
                merged[key] = sorted({merged.get(key), value}, key=repr)
    return merged


def _dag_reset_label(every: int) -> str:
    if every <= 0:
        return "never"
    if every == 1:
        return "game"
    return f"every-{every}-games"


def parse_dag_reset(text: str) -> int:
    """``game`` -> 1, ``never`` -> 0, ``every-N-games`` -> N."""
    if text == "game":
        return DAG_RESET_EVERY_GAME
    if text == "never":
        return DAG_RESET_NEVER
    match = _DAG_RESET_EVERY_N.match(text)
    if match is None:
        raise ValueError(
            "--dag-reset must be 'game', 'never' or 'every-N-games', got "
            f"{text!r}",
        )
    every = int(match.group(1))
    if every <= 0:
        raise ValueError("--dag-reset every-N-games needs N >= 1")
    return every


def _build_worker_specs(cfg: RunConfig) -> list[WorkerSpec]:
    buckets: list[list[int]] = [[] for _ in range(cfg.workers)]
    for game in range(cfg.games):
        buckets[game % cfg.workers].append(game)
    return [
        WorkerSpec(worker_id=i, game_indices=tuple(games), cfg=cfg)
        for i, games in enumerate(buckets) if games
    ]


def run_cell(cfg: RunConfig) -> dict[str, Any]:
    """One arm, one repeat."""
    if cfg.games <= 0 or cfg.workers <= 0:
        raise ValueError("games and workers must be positive")
    if not cfg.pack.is_file():
        raise FileNotFoundError(cfg.pack)
    # ⚑ VALIDATE THE SEARCH CONFIG IN THE PARENT, BEFORE ANY SPAWN. It was only
    # ever built inside `_run_worker`, so a bad `--topk` surfaced as N identical
    # pickled exceptions after pool startup -- once per worker, with the real
    # message buried in a future's traceback.
    gen.build_gumbel_config(_base_gen_config(cfg))
    specs = _build_worker_specs(cfg)
    started = time.perf_counter()
    if len(specs) == 1:
        results = [_run_worker(specs[0])]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=len(specs)) as pool:
            futures = {pool.submit(_run_worker, spec): spec for spec in specs}
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda r: r.worker_id)
    return _aggregate(results, cfg, time.perf_counter() - started)


@dataclass(frozen=True)
class ReadoutPlan:
    """The whole matrix: every cell, every repeat, and what they share."""

    arm_configs: tuple[ResolvedArmConfig, ...]
    pack: Path
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
    bank_path: Path | None
    run_id: str
    nice: int
    dag_reset_every: int
    repeats: int

    def cell(self, arm_config: ResolvedArmConfig, *, repeat: int, pack_sha: str) -> RunConfig:
        return RunConfig(
            arm_config=arm_config,
            pack=self.pack,
            pack_file_sha256=pack_sha,
            games=self.games,
            workers=self.workers,
            seed=self.seed,
            sims=self.sims,
            topk=self.topk,
            max_plies=self.max_plies,
            all_root_moves=self.all_root_moves,
            cp_per_internal_unit=self.cp_per_internal_unit,
            cp_slope=self.cp_slope,
            cp_draw_width=self.cp_draw_width,
            bank_path=self.bank_path,
            run_id=self.run_id,
            nice=self.nice,
            dag_reset_every=self.dag_reset_every,
            repeat=repeat,
        )


def _oracle(cells: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Compare the qsearch and qsearch-DAG per-game digests, repeat by repeat.

    ⚑ Present only when BOTH cells ran in this report.  Two digests from two
    invocations are comparable too -- they are a pure function of the seed and
    the settings, both of which ``provenance`` pins -- but this tool can only
    speak for the runs it made, so it says ``available: false`` rather than
    implying a comparison it did not perform.
    """
    left, right = ORACLE_ARMS
    if left not in cells or right not in cells:
        return {"arms": list(ORACLE_ARMS), "available": False, "digests_agree": None}
    per_repeat: list[dict[str, Any]] = []
    for a, b in zip(cells[left], cells[right], strict=False):
        per_repeat.append({
            "repeat": a["repeat"],
            left: a["games_digest"],
            right: b["games_digest"],
            "agree": a["games_digest"] == b["games_digest"],
        })
    return {
        "arms": list(ORACLE_ARMS),
        "available": True,
        "digests_agree": bool(per_repeat) and all(r["agree"] for r in per_repeat),
        "per_repeat": per_repeat,
    }


def run(plan: ReadoutPlan) -> dict[str, Any]:
    """Run every cell of the matrix, interleaving repeats.

    ⚑ THE ORDER IS (repeat, then cell), NOT (cell, then repeat). A matrix that
    runs all of arm A before any of arm B measures the machine's first hour
    against its third; interleaving spreads any drift -- thermal, another job
    arriving, page cache warming -- across the cells instead of loading it onto
    whichever ran last.
    """
    if plan.repeats <= 0:
        raise ValueError("--repeats must be >= 1")
    if not plan.arm_configs:
        raise ValueError("at least one --arm is required")
    if not plan.pack.is_file():
        raise FileNotFoundError(plan.pack)
    # ⚑ ONCE, IN THE PARENT, BEFORE ANY CLOCK STARTS. Hashing a 100+ MB pack is
    # pure I/O with nothing to do with the arm; it used to happen inside every
    # worker, inside the window `plies_per_s` divides by.
    pack_sha = _sha256_file(plan.pack)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    cells: dict[str, list[dict[str, Any]]] = {}
    order: list[dict[str, Any]] = []
    for repeat in range(plan.repeats):
        for arm_config in plan.arm_configs:
            cfg = plan.cell(arm_config, repeat=repeat, pack_sha=pack_sha)
            _LOG.info("cell: arm=%s repeat=%d", arm_config.arm, repeat)
            readout = run_cell(cfg)
            cells.setdefault(arm_config.arm, []).append(readout)
            order.append({"arm": arm_config.arm, "repeat": repeat})
    wall_s = time.perf_counter() - started

    every = [c for runs in cells.values() for c in runs]
    kernels = sorted({
        str(w["kernel"]) for c in every for w in c["workers_detail"]
    })
    source_shas = sorted({
        str(w["pack_source_sha256"]) for c in every for w in c["workers_detail"]
    })
    file_shas = sorted({
        str(w["pack_file_sha256"]) for c in every for w in c["workers_detail"]
    })
    reasons = [
        f"cell {c['arm']} repeat {c['repeat']}: {reason}"
        for c in every for reason in c["inadmissible_reasons"]
    ]
    if file_shas != [pack_sha]:
        reasons.append(
            f"workers mapped a pack whose file hash {file_shas} is not the "
            f"{pack_sha} the parent hashed before the run: the cells did not "
            "read one set of weights",
        )
    if len(kernels) > 1:
        reasons.append(
            f"workers ran different NNUE kernels {kernels}: avx2 and scalar "
            "differ by a multi-fold wall factor, so these cells are not one "
            "experiment",
        )
    nice_realized = sorted({int(n) for c in every for n in c["nice_realized"]})
    if len(nice_realized) > 1:
        reasons.append(
            f"cells ran at different niceness {nice_realized}: scheduling "
            "priority is a throughput input, so these cells are not one "
            "experiment",
        )
    oracle = _oracle(cells)
    if oracle["available"] and not oracle["digests_agree"]:
        reasons.append(
            "the nnue-qsearch and nnue-qsearch-dag per-game digests DIFFER: the "
            "two cells did not play the same games, so no wall-clock difference "
            "between them is attributable to the DAG substrate. The "
            "decomposition is VOID.",
        )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "provenance": {
            "run_id": plan.run_id,
            "started_utc": started_utc,
            "wall_s": wall_s,
            "pack_path": str(plan.pack),
            "pack_file_sha256": pack_sha,
            "pack_source_sha256": source_shas[0] if len(source_shas) == 1 else source_shas,
            "kernel": kernels[0] if len(kernels) == 1 else kernels,
            "seed": plan.seed,
            "games_per_cell": plan.games,
            "workers": plan.workers,
            "sims_floor": plan.sims,
            "topk": plan.topk,
            "max_plies": plan.max_plies,
            "all_root_moves": plan.all_root_moves,
            "cp_per_internal_unit": plan.cp_per_internal_unit,
            "cp_slope": plan.cp_slope,
            "cp_draw_width": plan.cp_draw_width,
            "nice_requested": plan.nice,
            "nice_realized": nice_realized,
            # ⚑ BANKING IS PART OF THE TIMED WINDOW, so whether it was on is
            # part of every throughput number in this file. It is one flag for
            # the WHOLE matrix by construction now: the previous doc turned it
            # on for the FastQ cell only, i.e. on exactly the cell whose
            # speedup was the headline.
            "banking": plan.bank_path is not None,
            "bank_path": None if plan.bank_path is None else str(plan.bank_path),
            "dag_reset": _dag_reset_label(plan.dag_reset_every),
            "repeats": plan.repeats,
            "arms": [c.arm for c in plan.arm_configs],
            "python": sys.version.split()[0],
        },
        "order": order,
        "oracle": oracle,
        "cells": cells,
        "inadmissible_reasons": reasons,
        "admissible": not reasons,
    }
    for reason in reasons:
        _LOG.error("INADMISSIBLE: %s", reason)
    return report


def assert_admissible(report: dict[str, Any]) -> None:
    """Raise unless every gate in ``report`` passed.

    The library-side hard failure.  ``main`` does not call it before writing the
    JSON: a run that violates a counter identity after six hours should leave
    its evidence on disk and exit non-zero, not vanish into a traceback.  A
    consumer that wants the exception has this.
    """
    if not report["admissible"]:
        raise RuntimeError(
            "readout is inadmissible: " + "; ".join(report["inadmissible_reasons"]),
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arm", choices=READOUT_ARMS, action="append", required=True,
        help="repeat to run several cells in one interleaved matrix",
    )
    p.add_argument("--nnue-pack", type=Path, required=True)
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--sims", type=int, default=gen.DEFAULT_SIMS)
    p.add_argument("--topk", type=int, default=gen.MAX_LEGAL_MOVES)
    p.add_argument("--max-plies", type=int, default=gen.DEFAULT_MAX_PLIES)
    p.add_argument(
        "--repeats", type=int, default=1,
        help="run the whole cell set this many times, interleaved, so the "
             "throughput comparison can be replicated instead of being one "
             "fixed-order run",
    )
    p.add_argument(
        "--dag-reset", default="game",
        help="DAG persistence cadence: 'game' (default), 'never', or "
             "'every-N-games'. fastq_design.md 4.4 leaves this to be chosen "
             "from measurement; this is the knob that measures it.",
    )
    p.add_argument(
        "--all-root-moves", action=argparse.BooleanOptionalAction, default=True,
        help="match the native-arm readout cells: every legal root move is a candidate",
    )
    p.add_argument("--run-id", default="nnue_gumbel_readout")
    p.add_argument("--nice", type=int, default=gen.DEFAULT_NICE)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument(
        "--bank-leaf-observations", type=Path, default=None,
        help="bank one JSONL row per evaluated position, for EVERY cell. The "
             "arm and repeat are in each file name; banking is inside the timed "
             "window, so it is all cells or none.",
    )

    # qsearch / qsearch-DAG only. None means resolve from the compiled extension.
    p.add_argument("--nnue-resolver-max-depth", type=int, default=None)
    p.add_argument("--nnue-qsearch-max-ply", type=int, default=None)
    p.add_argument("--nnue-qsearch-check-plies", type=int, default=None)
    p.add_argument("--dag-node-cap", type=int, default=None)
    p.add_argument(
        "--allow-binding-dag-node-cap", action="store_true",
        help="permit --dag-node-cap > 0 on nnue-qsearch-dag, accepting that a "
             "binding cap stops that cell being the nnue-qsearch control",
    )

    # FastQ only. None means resolve from the compiled extension.
    p.add_argument("--fastq-max-qply", type=int, default=None)
    p.add_argument("--fastq-node-cap", type=int, default=None)
    p.add_argument("--fastq-delta-margin", type=int, default=None)
    p.add_argument("--fastq-recapture-exempt", type=int, choices=(0, 1), default=None)

    p.add_argument("--nnue-cp-per-unit", type=float, default=gen.NNUE_CP_PER_INTERNAL_UNIT)
    p.add_argument("--nnue-cp-slope", type=float, default=gen.NNUE_CP_SLOPE)
    p.add_argument("--nnue-cp-draw-width", type=float, default=gen.NNUE_CP_DRAW_WIDTH)
    return p


def plan_from_args(args: argparse.Namespace) -> ReadoutPlan:
    arms = list(dict.fromkeys(args.arm))
    return ReadoutPlan(
        arm_configs=tuple(resolve_arm_config(args, arm=a) for a in arms),
        pack=Path(args.nnue_pack),
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
        bank_path=(
            None if args.bank_leaf_observations is None
            else Path(args.bank_leaf_observations)
        ),
        run_id=str(args.run_id),
        nice=int(args.nice),
        dag_reset_every=parse_dag_reset(str(args.dag_reset)),
        repeats=int(args.repeats),
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args()
    report = run(plan_from_args(args))
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        path = Path(args.json)
        _atomic_write_text(path, text + "\n")
    # ⚑ THE ARTIFACT IS WRITTEN FIRST, THEN THE FAILURE IS REPORTED. A gate that
    # raises before the JSON exists destroys the evidence for the finding it
    # just made; exit code 2 with `admissible: false` on disk is a hard failure
    # a caller cannot miss and a reader can still investigate.
    return 0 if report["admissible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
