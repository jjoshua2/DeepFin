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

DAG-backed arms persist across *plies within one game* and are reset before the
next game.  That is the persistence policy preregistered in
``docs/fastq_design.md``: it preserves the cross-ply reuse we want to measure
without letting unrelated games accumulate hundreds of MB in one worker store.

Nothing here installs a DAG-backed provider in ``MCTSTree``.  Those providers
correctly declare ``requires_gil`` and the tree refuses them.  The existing gen-0
path instead asks the tree for its pending leaf ``CBoard`` objects and evaluates
that batch externally through ``arm_handle_eval``; this driver uses exactly that
path, so the concurrency guard is neither weakened nor bypassed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from chess_anti_engine.encoding import rep_fix
from scripts import gen_random_selfplay_shards as gen


ARM_QSEARCH = "nnue-qsearch"
ARM_QSEARCH_DAG = "nnue-qsearch-dag"
ARM_FASTQ = "nnue-fastq"
READOUT_ARMS: tuple[str, ...] = (ARM_QSEARCH, ARM_QSEARCH_DAG, ARM_FASTQ)

StatsSurface = Literal["arm", "fastq"]


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


def resolve_arm_config(args: argparse.Namespace, ext: Any | None = None) -> ResolvedArmConfig:
    """Resolve only knobs the selected provider consumes; refuse every other one.

    The refusal is intentional.  A flag that is accepted, printed, and then not
    read by the selected C provider is this repository's signature defect.  The
    CLI therefore uses ``None`` as "not supplied", fills defaults from the
    extension only for the selected arm, and errors if a caller supplies a knob
    belonging to another arm.
    """
    ext = _load_ext() if ext is None else ext
    arm = str(args.arm)
    if arm not in ARM_SPECS:
        raise ValueError(f"arm must be one of {READOUT_ARMS}, got {arm!r}")
    spec = ARM_SPECS[arm]

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
            f"{arm} does not consume --fastq-* knobs; remove them rather than "
            "recording settings the selected provider will ignore",
        )
    if spec.consumes_fastq_knobs and any(v is not None for v in q_values):
        raise ValueError(
            f"{arm} does not consume qsearch/resolver/DAG-qsearch knobs; remove "
            "them rather than recording settings the selected provider will ignore",
        )

    if spec.consumes_qsearch_knobs:
        dag_cap = args.dag_node_cap
        if arm == ARM_QSEARCH and dag_cap is not None:
            raise ValueError(
                "nnue-qsearch has no DAG and cannot consume --dag-node-cap",
            )
        return ResolvedArmConfig(
            arm=arm,
            resolver_max_depth=(
                int(ext.RESOLVER_MAX_DEPTH)
                if args.nnue_resolver_max_depth is None
                else int(args.nnue_resolver_max_depth)
            ),
            qsearch_max_ply=(
                int(ext.QSEARCH_MAX_PLY)
                if args.nnue_qsearch_max_ply is None
                else int(args.nnue_qsearch_max_ply)
            ),
            qsearch_check_plies=(
                int(ext.QSEARCH_CHECK_PLIES)
                if args.nnue_qsearch_check_plies is None
                else int(args.nnue_qsearch_check_plies)
            ),
            dag_node_cap=(
                int(ext.QSEARCH_DAG_NODE_CAP)
                if arm == ARM_QSEARCH_DAG and dag_cap is None
                else (None if dag_cap is None else int(dag_cap))
            ),
        )

    return ResolvedArmConfig(
        arm=arm,
        fastq_max_qply=(
            int(ext.FASTQ_MAX_QPLY)
            if args.fastq_max_qply is None else int(args.fastq_max_qply)
        ),
        fastq_node_cap=(
            int(ext.FASTQ_NODE_CAP)
            if args.fastq_node_cap is None else int(args.fastq_node_cap)
        ),
        fastq_delta_margin=(
            int(ext.FASTQ_DELTA_MARGIN)
            if args.fastq_delta_margin is None else int(args.fastq_delta_margin)
        ),
        fastq_recapture_exempt=(
            int(ext.FASTQ_RECAPTURE_EXEMPT)
            if args.fastq_recapture_exempt is None
            else int(args.fastq_recapture_exempt)
        ),
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class ReadoutArmSource(gen.NnueArmValueSource):
    """NnueArmValueSource generalized to the two DAG-backed readout providers.

    The inherited batch/Q conversion and banking code is intentionally reused;
    only construction and the stats surface differ.  In particular the FastQ
    handle is NEVER passed to ``arm_stats`` (which correctly raises), and a
    qsearch handle is NEVER passed to ``fastq_stats``.
    """

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
    ) -> None:
        ext = _load_ext() if ext is None else ext
        self._ext = ext
        self.arm = config.arm
        self.spec = ARM_SPECS[self.arm]
        self.pack = Path(pack)
        self.cp_per_internal_unit = float(cp_per_internal_unit)
        self.cp_slope = float(cp_slope)
        self.cp_draw_width = float(cp_draw_width)
        for name, value in (
            ("cp_per_internal_unit", self.cp_per_internal_unit),
            ("cp_slope", self.cp_slope),
            ("cp_draw_width", self.cp_draw_width),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.cp_per_internal_unit <= 0 or self.cp_slope <= 0 or self.cp_draw_width < 0:
            raise ValueError(
                "cp_per_internal_unit/cp_slope must be > 0 and cp_draw_width >= 0",
            )

        requested = config.consumed()
        if self.spec.consumes_qsearch_knobs:
            stored = dict(
                ext.set_arm_config(
                    requested["resolver_max_depth"],
                    requested["qsearch_max_ply"],
                    requested["qsearch_check_plies"],
                    requested.get("dag_node_cap", 0),
                ),
            )
        else:
            stored = dict(
                ext.fastq_set_config(
                    requested["max_qply"],
                    requested["node_cap"],
                    requested["delta_margin"],
                    requested["see_recapture_exempt"],
                ),
            )

        # Configuration MUST precede open(): both provider families snapshot at init.
        self._handle = ext.arm_open(self.arm, str(self.pack))
        snapshot = self.provider_stats()
        self.consumed_keys = tuple(requested)
        self.requested = {k: int(stored[k]) for k in self.consumed_keys}
        self.realized = {k: int(snapshot[k]) for k in self.consumed_keys}
        if self.realized != self.requested:
            raise RuntimeError(
                f"{self.arm} context did not realize the requested knobs: "
                f"requested={self.requested} realized={self.realized}",
            )

        self.stats = gen.NnueArmStats(context=dict(snapshot))
        self.pack_source_sha256 = str(ext.source_sha256(ext.load(str(self.pack))))
        self.pack_file_sha256 = _sha256_file(self.pack)
        self.kernel = "avx2" if ext.simd_active() else "scalar"
        if leaf_bank is not None:
            leaf_bank.parent.mkdir(parents=True, exist_ok=True)
        self._bank = None if leaf_bank is None else leaf_bank.open("a", encoding="utf-8")
        self.leaf_bank_path = leaf_bank
        self.bank_rows = 0
        self.mate_band_floor = float(
            ext.RESOLVER_MATE_BASE - ext.RESOLVER_MAX_PLIES * ext.RESOLVER_MATE_PLY_STEP,
        )
        self.mate_base = float(ext.RESOLVER_MATE_BASE)
        self.mate_ply_step = float(ext.RESOLVER_MATE_PLY_STEP)

    def provider_stats(self) -> dict[str, int]:
        if self.spec.stats_surface == "fastq":
            return dict(self._ext.fastq_stats(self._handle))
        return dict(self._ext.arm_stats(self._handle))

    def dag_stats(self) -> dict[str, int] | None:
        if not self.spec.uses_dag:
            return None
        return dict(self._ext.arm_dag_stats(self._handle))

    def reset_game(self) -> None:
        """Reset only canonical graph state; provider counters remain cumulative."""
        if self.spec.uses_dag:
            self._ext.arm_dag_reset(self._handle)

    def refresh_context_stats(self) -> None:
        self.stats.context = self.provider_stats()


class ReadoutEvaluator(gen.UniformPriorEvaluator):
    """The generator evaluator with its native-source whitelist widened explicitly."""

    def __init__(
        self,
        *,
        source: ReadoutArmSource,
        expected_planes: int,
        input_history_encoding: str,
        input_extra_features: str,
    ) -> None:
        # Do not call the parent constructor: its whitelist intentionally names
        # only the two production generator arms.  Everything after that guard is
        # ordinary state initialization, reproduced here so a new provider cannot
        # sneak into the original generator by changing a global tuple.
        if source.arm not in READOUT_ARMS:
            raise ValueError(f"readout source must be one of {READOUT_ARMS}")
        self.value_source = source.arm
        self.expected_planes = int(expected_planes)
        self.random_salt = 0
        self.nnue_source = source
        self.input_history_encoding = str(input_history_encoding)
        self.input_extra_features = str(input_extra_features)
        self.eval_calls = 0
        self.eval_rows = 0
        self._tree = None
        self._cluster = None


@dataclass(frozen=True)
class RunConfig:
    arm_config: ResolvedArmConfig
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


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    game_indices: tuple[int, ...]
    cfg: RunConfig


@dataclass
class DagGameStats:
    games: int = 0
    nodes_sum: int = 0
    edges_sum: int = 0
    hits_sum: int = 0
    probes_sum: int = 0
    inserts_sum: int = 0
    memory_peak: int = 0
    nodes_peak: int = 0
    edges_peak: int = 0

    def add(self, stats: dict[str, int]) -> None:
        self.games += 1
        nodes = int(stats.get("nodes", stats.get("node_count", 0)))
        edges = int(stats.get("edges", stats.get("edge_count", 0)))
        memory = int(stats.get("memory_bytes", 0))
        self.nodes_sum += nodes
        self.edges_sum += edges
        self.hits_sum += int(stats.get("hits", 0))
        self.probes_sum += int(stats.get("probes", 0))
        self.inserts_sum += int(stats.get("inserts", 0))
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
            "nodes_peak_per_game": float(self.nodes_peak),
            "edges_peak_per_game": float(self.edges_peak),
            "memory_peak_per_worker_bytes": float(self.memory_peak),
        }


@dataclass
class WorkerResult:
    worker_id: int
    games: int
    plies: int
    elapsed_s: float
    terminations: dict[str, int]
    policy_shape: dict[str, float]
    root_budget: dict[str, float]
    provider_stats: dict[str, int]
    dag: DagGameStats
    eval_batches: int
    eval_rows: int
    arm_batches: int
    arm_leaves: int
    arm_roots: int
    mate_band_leaves: int
    mate_band_roots: int
    bank_rows: int
    bank_file: str | None


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


def _worker_bank_path(base: Path | None, worker_id: int) -> Path | None:
    if base is None:
        return None
    suffix = base.suffix or ".jsonl"
    stem = base.name[: -len(suffix)] if base.name.endswith(suffix) else base.name
    return base.with_name(f"{stem}.w{worker_id:02d}{suffix}")


def _run_worker(spec: WorkerSpec) -> WorkerResult:
    cfg = spec.cfg
    if cfg.nice:
        try:
            os.nice(int(cfg.nice))
        except OSError:
            pass
    rep_fix.apply(True)
    base = _base_gen_config(cfg)
    gcfg = gen.build_gumbel_config(base)
    opening_cfg = gen.build_opening_config(base)
    source = ReadoutArmSource(
        config=cfg.arm_config,
        pack=cfg.pack,
        cp_per_internal_unit=cfg.cp_per_internal_unit,
        cp_slope=cfg.cp_slope,
        cp_draw_width=cfg.cp_draw_width,
        leaf_bank=_worker_bank_path(cfg.bank_path, spec.worker_id),
    )
    evaluator = ReadoutEvaluator(
        source=source,
        expected_planes=gen.input_plane_count(base.input_extra_features),
        input_history_encoding=base.input_history_encoding,
        input_extra_features=base.input_extra_features,
    )
    policy = gen.PolicyShapeStats()
    budget = gen.RootBudgetStats()
    dag_games = DagGameStats()
    terminations = dict.fromkeys(gen.TERMINATIONS, 0)
    plies = 0
    started = time.perf_counter()
    try:
        for game_index in spec.game_indices:
            # Reset BEFORE every game, including the first, so the lifecycle is
            # one rule rather than a special first-game branch.  Reset retains
            # allocations, which is exactly what lets memory_peak measure the
            # worker's actual resident capacity while semantic nodes stay game-local.
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
            for row in outcome.records:
                policy.add(gen.policy_tv_to_uniform(row.policy_probs, row.legal_mask))
            dag = source.dag_stats()
            if dag is not None:
                dag_games.add(dag)
    finally:
        evaluator.close()
    elapsed = time.perf_counter() - started
    provider = source.provider_stats()
    return WorkerResult(
        worker_id=spec.worker_id,
        games=len(spec.game_indices),
        plies=plies,
        elapsed_s=elapsed,
        terminations=terminations,
        policy_shape=policy.summary(),
        root_budget=budget.summary(),
        provider_stats=provider,
        dag=dag_games,
        eval_batches=evaluator.eval_calls,
        eval_rows=evaluator.eval_rows,
        arm_batches=source.stats.batches,
        arm_leaves=source.stats.leaves,
        arm_roots=source.stats.roots,
        mate_band_leaves=source.stats.mate_band_leaves,
        mate_band_roots=source.stats.mate_band_roots,
        bank_rows=source.bank_rows,
        bank_file=(
            None if source.leaf_bank_path is None else str(source.leaf_bank_path)
        ),
    )


def _sum_numeric_dicts(dicts: list[dict[str, int]], config_keys: set[str], peak_keys: set[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dicts:
        for key, raw in d.items():
            value = int(raw)
            if key in config_keys:
                if key in out and out[key] != value:
                    raise RuntimeError(
                        f"workers realized different {key}: {out[key]} vs {value}",
                    )
                out[key] = value
            elif key in peak_keys:
                out[key] = max(out.get(key, value), value)
            else:
                out[key] = out.get(key, 0) + value
    return out


def _aggregate(results: list[WorkerResult], cfg: RunConfig, wall_s: float) -> dict[str, Any]:
    spec = ARM_SPECS[cfg.arm_config.arm]
    if spec.stats_surface == "fastq":
        config_keys = {"max_qply", "node_cap", "delta_margin", "see_recapture_exempt"}
        peak_keys = {"max_ply_seen"}
    else:
        config_keys = {
            "resolver_max_depth", "qsearch_max_ply", "qsearch_check_plies",
            "dag_node_cap", "dag_enabled",
        }
        peak_keys = {"max_depth_seen", "qmax_ply_seen"}
    provider = _sum_numeric_dicts(
        [r.provider_stats for r in results], config_keys, peak_keys,
    )
    dag = DagGameStats()
    for r in results:
        dag.merge(r.dag)
    plies = sum(r.plies for r in results)
    games = sum(r.games for r in results)
    arm_leaves = sum(r.arm_leaves for r in results)
    arm_roots = sum(r.arm_roots for r in results)
    total_calls = arm_leaves + arm_roots
    terminations: dict[str, int] = {}
    for r in results:
        for key, value in r.terminations.items():
            terminations[key] = terminations.get(key, 0) + int(value)

    readout: dict[str, Any] = {
        "schema": 1,
        "run_id": cfg.run_id,
        "arm": cfg.arm_config.arm,
        "arm_config": cfg.arm_config.consumed(),
        "games": games,
        "workers": cfg.workers,
        "plies": plies,
        "wall_s": wall_s,
        "plies_per_s": plies / max(wall_s, 1e-12),
        "games_per_h": games * 3600.0 / max(wall_s, 1e-12),
        "worker_cpu_s": sum(r.elapsed_s for r in results),
        "search": {
            "sims_floor": cfg.sims,
            "topk": cfg.topk,
            "all_root_moves": cfg.all_root_moves,
        },
        "arm_io": {
            "leaves": arm_leaves,
            "roots": arm_roots,
            "calls": total_calls,
            "batches": sum(r.arm_batches for r in results),
            "leaves_per_batch": arm_leaves / max(1, sum(r.arm_batches for r in results)),
            "mate_band_leaves": sum(r.mate_band_leaves for r in results),
            "mate_band_roots": sum(r.mate_band_roots for r in results),
            "nnue_evals_per_top_level_call": (
                float(provider.get("nnue_evals", 0)) / max(1, total_calls)
            ),
        },
        "provider_stats": provider,
        "dag_per_game": dag.summary(),
        "terminations": terminations,
        "bank_rows": sum(r.bank_rows for r in results),
        "bank_files": [r.bank_file for r in results if r.bank_file is not None],
        "workers_detail": [asdict(r) for r in results],
    }
    if cfg.arm_config.arm == ARM_FASTQ:
        calls = max(1, int(provider.get("calls", 0)))
        readout["fastq"] = {
            "budget_trip_rate": int(provider.get("budget_trips", 0)) / calls,
            "created_nodes_per_call": int(provider.get("nodes_created", 0)) / calls,
            "nnue_evals_per_call": int(provider.get("nnue_evals", 0)) / calls,
            "within_call_hits_per_call": int(provider.get("hits_within_call", 0)) / calls,
            "cross_call_hits_per_call": int(provider.get("hits_cross_call", 0)) / calls,
            "see_prunes_per_call": int(provider.get("see_prunes", 0)) / calls,
            "delta_prunes_per_call": int(provider.get("delta_prunes", 0)) / calls,
        }
    return readout


def _build_worker_specs(cfg: RunConfig) -> list[WorkerSpec]:
    buckets: list[list[int]] = [[] for _ in range(cfg.workers)]
    for game in range(cfg.games):
        buckets[game % cfg.workers].append(game)
    return [
        WorkerSpec(worker_id=i, game_indices=tuple(games), cfg=cfg)
        for i, games in enumerate(buckets) if games
    ]


def run(cfg: RunConfig) -> dict[str, Any]:
    if cfg.games <= 0 or cfg.workers <= 0:
        raise ValueError("games and workers must be positive")
    if not cfg.pack.is_file():
        raise FileNotFoundError(cfg.pack)
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=READOUT_ARMS, required=True)
    p.add_argument("--nnue-pack", type=Path, required=True)
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--sims", type=int, default=gen.DEFAULT_SIMS)
    p.add_argument("--topk", type=int, default=gen.MAX_LEGAL_MOVES)
    p.add_argument("--max-plies", type=int, default=gen.DEFAULT_MAX_PLIES)
    p.add_argument(
        "--all-root-moves", action=argparse.BooleanOptionalAction, default=True,
        help="match the native-arm readout cells: every legal root move is a candidate",
    )
    p.add_argument("--run-id", default="nnue_gumbel_readout")
    p.add_argument("--nice", type=int, default=gen.DEFAULT_NICE)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--bank-leaf-observations", type=Path, default=None)

    # qsearch / qsearch-DAG only. None means resolve from the compiled extension.
    p.add_argument("--nnue-resolver-max-depth", type=int, default=None)
    p.add_argument("--nnue-qsearch-max-ply", type=int, default=None)
    p.add_argument("--nnue-qsearch-check-plies", type=int, default=None)
    p.add_argument("--dag-node-cap", type=int, default=None)

    # FastQ only. None means resolve from the compiled extension.
    p.add_argument("--fastq-max-qply", type=int, default=None)
    p.add_argument("--fastq-node-cap", type=int, default=None)
    p.add_argument("--fastq-delta-margin", type=int, default=None)
    p.add_argument("--fastq-recapture-exempt", type=int, choices=(0, 1), default=None)

    p.add_argument("--nnue-cp-per-unit", type=float, default=gen.NNUE_CP_PER_INTERNAL_UNIT)
    p.add_argument("--nnue-cp-slope", type=float, default=gen.NNUE_CP_SLOPE)
    p.add_argument("--nnue-cp-draw-width", type=float, default=gen.NNUE_CP_DRAW_WIDTH)
    return p


def config_from_args(args: argparse.Namespace) -> RunConfig:
    arm_config = resolve_arm_config(args)
    return RunConfig(
        arm_config=arm_config,
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
    )


def main() -> int:
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    result = run(cfg)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
