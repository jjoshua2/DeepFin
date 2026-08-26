#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READOUT = ROOT / "scripts/nnue_gumbel_readout.py"
GEN = ROOT / "scripts/gen_random_selfplay_shards.py"
TEST = ROOT / "tests/test_nnue_gumbel_readout.py"
DOC = ROOT / "docs/nnue_gumbel_readout.md"


def read(path: Path) -> str:
    return path.read_text()


def write(path: Path, text: str) -> None:
    path.write_text(text)


def replace_once(path: Path, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, got {count}: {old[:100]!r}")
    write(path, text.replace(old, new, 1))


def replace_between(path: Path, start: str, end: str, new: str) -> None:
    text = read(path)
    i = text.find(start)
    if i < 0:
        raise RuntimeError(f"{path}: start marker missing: {start!r}")
    j = text.find(end, i)
    if j < 0:
        raise RuntimeError(f"{path}: end marker missing: {end!r}")
    if text.find(start, i + 1) >= 0 and text.find(start, i + 1) < j:
        raise RuntimeError(f"{path}: ambiguous nested start marker: {start!r}")
    write(path, text[:i] + new.rstrip() + "\n\n" + text[j:])


# ---------------------------------------------------------------------------
# scripts/nnue_gumbel_readout.py
# ---------------------------------------------------------------------------
replace_once(READOUT, "import re\nimport sys\n", "import re\nimport subprocess\nimport sys\n")
replace_once(
    READOUT,
    "REPORT_SCHEMA = 3\n\nStatsSurface = Literal[\"arm\", \"fastq\"]\n",
    """#: 4 adds a search-output digest to the qsearch/DAG oracle and native-binary
#: provenance. A schema-3 reader must not mistake game-trajectory equality for
#: the stronger search-output equality this version requires.
REPORT_SCHEMA = 4

QUALITY_SCOPE: dict[str, object] = {
    "population": "end_to_end_arm_selected",
    "paired_evaluator_quality": False,
    "deep_sf_paired_input_admissible": False,
    "reason": (
        "each arm drives its own Gumbel search, so FastQ may change which later "
        "leaves exist; per-arm banks are trace artifacts, not a paired evaluator "
        "quality sample. A frozen-driver/shadow-arm experiment is required for "
        "paired deep-SF attribution."
    ),
}

StatsSurface = Literal["arm", "fastq"]
""",
)

replace_once(
    READOUT,
    """def resolve_arm_config(
    args: argparse.Namespace, ext: Any | None = None, *, arm: str | None = None,
) -> ResolvedArmConfig:
""",
    """def resolve_arm_config(
    args: argparse.Namespace, ext: Any | None = None, *, arm: str | None = None,
    strict_foreign_knobs: bool = True,
) -> ResolvedArmConfig:
""",
)
replace_once(
    READOUT,
    """    if spec.consumes_qsearch_knobs and any(v is not None for v in f_values):
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
""",
    """    if strict_foreign_knobs:
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
""",
)
replace_once(
    READOUT,
    """def readout_arm_config_plan(config: ResolvedArmConfig) -> gen.ArmConfigPlan:
""",
    """def _validate_matrix_knobs(args: argparse.Namespace, arms: list[str]) -> None:
    """Refuse a supplied knob only when NO selected cell consumes it.

    Per-cell resolution still copies only the fields its provider reads. The
    distinction matters for a mixed matrix: a FastQ knob is foreign to qsearch
    but live in the FastQ cell, so rejecting it while resolving qsearch makes
    non-default matrix experiments impossible.
    """
    selected = set(arms)
    qsearch_family = {ARM_QSEARCH, ARM_QSEARCH_DAG}
    q_common = (
        args.nnue_resolver_max_depth,
        args.nnue_qsearch_max_ply,
        args.nnue_qsearch_check_plies,
    )
    fastq = (
        args.fastq_max_qply,
        args.fastq_node_cap,
        args.fastq_delta_margin,
        args.fastq_recapture_exempt,
    )
    if any(v is not None for v in q_common) and not (selected & qsearch_family):
        raise ValueError("qsearch/resolver knobs were supplied but no qsearch-family arm is selected")
    if args.dag_node_cap is not None and ARM_QSEARCH_DAG not in selected:
        raise ValueError("--dag-node-cap was supplied but nnue-qsearch-dag is not selected")
    if any(v is not None for v in fastq) and ARM_FASTQ not in selected:
        raise ValueError("--fastq-* knobs were supplied but nnue-fastq is not selected")


def readout_arm_config_plan(config: ResolvedArmConfig) -> gen.ArmConfigPlan:
""",
)

replace_once(
    READOUT,
    """def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


#: This process's niceness before the harness touched it, as a one-element list
""",
    """def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _module_identity(module: Any) -> tuple[str, str]:
    raw = getattr(module, "__file__", None)
    if not raw:
        raise RuntimeError(f"native module {module!r} has no __file__; cannot prove binary identity")
    path = Path(str(raw)).resolve()
    return str(path), _sha256_file(path)


def _git_provenance() -> dict[str, object]:
    """Best-effort source revision. Native binary hashes remain authoritative."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip())
        return {"git_head": head, "git_tracked_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_head": None, "git_tracked_dirty": None}


#: This process's niceness before the harness touched it, as a one-element list
""",
)

# Stronger oracle: search output digest lives beside trajectory digest.
replace_once(
    READOUT,
    """    termination: str
    digest: str


def game_digest(
""",
    """    termination: str
    digest: str
    search_digest: str = ""


def search_output_digest(rows: list[Any]) -> str:
    """Digest the exact improved-policy/search output for every stored ply."""
    h = hashlib.sha256()
    for row in rows:
        h.update(int(row.ply_index).to_bytes(8, "little", signed=True))
        policy = np.asarray(row.policy_probs, dtype="<f4")
        legal = np.asarray(row.legal_mask, dtype=np.uint8)
        h.update(policy.tobytes(order="C"))
        h.update(legal.tobytes(order="C"))
    return h.hexdigest()


def game_digest(
""",
)
replace_once(
    READOUT,
    """def games_digest(records: list[GameRecord]) -> str:
    """One digest over a cell's whole game set, ordered by game index."""
    h = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.game):
        h.update(f"{record.game}:{record.digest}\\n".encode())
    return h.hexdigest()


@dataclass
class DagGameStats:
""",
    """def games_digest(records: list[GameRecord]) -> str:
    """One digest over a cell's whole game set, ordered by game index."""
    h = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.game):
        h.update(f"{record.game}:{record.digest}\\n".encode())
    return h.hexdigest()


def searches_digest(records: list[GameRecord]) -> str:
    """One digest over each game's improved-policy/search outputs."""
    h = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.game):
        h.update(f"{record.game}:{record.search_digest}\\n".encode())
    return h.hexdigest()


@dataclass
class DagGameStats:
""",
)

replace_between(
    READOUT,
    "    def add(self, stats: dict[str, int]) -> None:\n",
    "    def merge(self, other: DagGameStats) -> None:\n",
    '''    def add(
        self, stats: dict[str, int], *, previous: dict[str, int] | None = None,
    ) -> None:
        """Fold one snapshot as DELTA work plus absolute resource peaks.

        `arm_dag_stats()` counters are cumulative since the last reset. Under
        `--dag-reset never/every-N`, summing each absolute snapshot counts early
        games repeatedly. `previous=None` means a reset boundary; otherwise all
        cumulative fields must be monotone and only their deltas are charged to
        this game. The construction identity is checked on the absolute snapshot
        before any differencing.
        """
        self.games += 1
        prev = {} if previous is None else previous

        def delta(key: str) -> int:
            current = int(stats[key])
            before = int(prev.get(key, 0))
            if current < before:
                raise ValueError(
                    f"DAG cumulative counter {key} went backwards without a "
                    f"declared reset: {before} -> {current}"
                )
            return current - before

        nodes = int(stats["node_count"])
        edges = int(stats["edge_count"])
        memory = int(stats["memory_bytes"])
        state_inits = int(stats["state_inits"])
        state_makes = int(stats["state_makes"])

        self.nodes_sum += delta("node_count")
        self.edges_sum += delta("edge_count")
        self.hits_sum += delta("hits")
        self.probes_sum += delta("probes")
        self.inserts_sum += delta("inserts")
        self.state_inits_sum += delta("state_inits")
        self.state_makes_sum += delta("state_makes")
        if state_inits + state_makes != nodes:
            self.state_identity_violations += 1
        self.memory_peak = max(self.memory_peak, memory)
        self.nodes_peak = max(self.nodes_peak, nodes)
        self.edges_peak = max(self.edges_peak, edges)''',
)

replace_once(
    READOUT,
    """    nice_realized: int
    arm_config_requested: dict[str, int] = field(default_factory=dict)
""",
    """    nice_realized: int
    nnue_ext_path: str = ""
    nnue_ext_sha256: str = ""
    mcts_ext_path: str = ""
    mcts_ext_sha256: str = ""
    arm_config_requested: dict[str, int] = field(default_factory=dict)
""",
)

# Worker-local pack hash and binary provenance. Hash is outside the search timer.
replace_once(
    READOUT,
    """    opening_cfg = gen.build_opening_config(base)
    source = ReadoutArmSource(
""",
    """    opening_cfg = gen.build_opening_config(base)
    # Hash in THIS worker, immediately around the open. The parent hash is an
    # expectation, not evidence of what a worker actually mapped.
    worker_pack_sha_before = _sha256_file(cfg.pack)
    source = ReadoutArmSource(
""",
)
replace_once(
    READOUT,
    """        pack=cfg.pack,
        pack_file_sha256=cfg.pack_file_sha256,
""",
    """        pack=cfg.pack,
        pack_file_sha256=worker_pack_sha_before,
""",
)
replace_once(
    READOUT,
    """    evaluator = ReadoutEvaluator(
        source=source,
        expected_planes=gen.input_plane_count(base.input_extra_features),
        input_history_encoding=base.input_history_encoding,
        input_extra_features=base.input_extra_features,
    )
    setup_s = time.perf_counter() - setup_started
""",
    """    worker_pack_sha_after = _sha256_file(cfg.pack)
    if worker_pack_sha_after != worker_pack_sha_before:
        raise RuntimeError(
            "NNUE pack changed while this worker was opening it: "
            f"{worker_pack_sha_before} -> {worker_pack_sha_after}"
        )
    evaluator = ReadoutEvaluator(
        source=source,
        expected_planes=gen.input_plane_count(base.input_extra_features),
        input_history_encoding=base.input_history_encoding,
        input_extra_features=base.input_extra_features,
    )
    from chess_anti_engine.mcts import _mcts_tree as mcts_ext
    nnue_ext_path, nnue_ext_sha = _module_identity(source._ext)
    mcts_ext_path, mcts_ext_sha = _module_identity(mcts_ext)
    setup_s = time.perf_counter() - setup_started
""",
)
replace_once(
    READOUT,
    """    records: list[GameRecord] = []
    plies = 0
    started = time.perf_counter()
""",
    """    records: list[GameRecord] = []
    dag_previous: dict[str, int] | None = None
    plies = 0
    started = time.perf_counter()
""",
)
replace_once(
    READOUT,
    """            ):
                source.reset_game()
            rng = np.random.default_rng(int(cfg.seed) + int(game_index))
""",
    """            ):
                source.reset_game()
                dag_previous = None
            rng = np.random.default_rng(int(cfg.seed) + int(game_index))
""",
)
replace_once(
    READOUT,
    """                digest=game_digest(
                    game_index=int(game_index),
                    start_fen=str(outcome.start_fen),
                    move_trace=str(outcome.move_trace),
                    result=str(outcome.result),
                    termination=str(outcome.termination),
                ),
            ))
""",
    """                digest=game_digest(
                    game_index=int(game_index),
                    start_fen=str(outcome.start_fen),
                    move_trace=str(outcome.move_trace),
                    result=str(outcome.result),
                    termination=str(outcome.termination),
                ),
                search_digest=search_output_digest(outcome.records),
            ))
""",
)
replace_once(
    READOUT,
    """            dag = source.dag_stats()
            if dag is not None:
                dag_games.add(dag)
""",
    """            dag = source.dag_stats()
            if dag is not None:
                dag_games.add(dag, previous=dag_previous)
                dag_previous = dag
""",
)
replace_once(
    READOUT,
    """        pack_source_sha256=source.pack_source_sha256,
        nice_realized=nice_realized,
""",
    """        pack_source_sha256=source.pack_source_sha256,
        nice_realized=nice_realized,
        nnue_ext_path=nnue_ext_path,
        nnue_ext_sha256=nnue_ext_sha,
        mcts_ext_path=mcts_ext_path,
        mcts_ext_sha256=mcts_ext_sha,
""",
)
replace_once(
    READOUT,
    """            "worker_id": int(spec.worker_id),
        },
""",
    """            "worker_id": int(spec.worker_id),
            "population_kind": "end_to_end_arm_selected",
        },
""",
)

replace_once(
    READOUT,
    """        "games_digest": games_digest(records),
        "games_detail": [asdict(rec) for rec in sorted(records, key=lambda r: r.game)],
""",
    """        "games_digest": games_digest(records),
        "searches_digest": searches_digest(records),
        "games_detail": [asdict(rec) for rec in sorted(records, key=lambda r: r.game)],
""",
)

replace_between(
    READOUT,
    "def _oracle(cells: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:\n",
    "def run(plan: ReadoutPlan) -> dict[str, Any]:\n",
    '''def _oracle(cells: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Require both trajectory AND improved-policy/search-output parity."""
    left, right = ORACLE_ARMS
    if left not in cells or right not in cells:
        return {
            "arms": list(ORACLE_ARMS), "available": False,
            "digests_agree": None, "game_digests_agree": None,
            "search_digests_agree": None,
        }
    if len(cells[left]) != len(cells[right]):
        return {
            "arms": list(ORACLE_ARMS), "available": True,
            "digests_agree": False, "game_digests_agree": False,
            "search_digests_agree": False,
            "reason": "qsearch and qsearch-DAG repeat counts differ",
        }
    per_repeat: list[dict[str, Any]] = []
    for a, b in zip(cells[left], cells[right], strict=True):
        game_agree = a["games_digest"] == b["games_digest"]
        search_agree = a["searches_digest"] == b["searches_digest"]
        per_repeat.append({
            "repeat": a["repeat"],
            "game_digest": {left: a["games_digest"], right: b["games_digest"]},
            "search_digest": {left: a["searches_digest"], right: b["searches_digest"]},
            "game_agree": game_agree,
            "search_agree": search_agree,
            "agree": game_agree and search_agree,
        })
    return {
        "arms": list(ORACLE_ARMS),
        "available": True,
        "game_digests_agree": bool(per_repeat) and all(r["game_agree"] for r in per_repeat),
        "search_digests_agree": bool(per_repeat) and all(r["search_agree"] for r in per_repeat),
        "digests_agree": bool(per_repeat) and all(r["agree"] for r in per_repeat),
        "per_repeat": per_repeat,
    }''',
)

# Add actual binary provenance and source revision to run().
replace_once(
    READOUT,
    """    file_shas = sorted({
        str(w["pack_file_sha256"]) for c in every for w in c["workers_detail"]
    })
    reasons = [
""",
    """    file_shas = sorted({
        str(w["pack_file_sha256"]) for c in every for w in c["workers_detail"]
    })
    nnue_binary_shas = sorted({
        str(w["nnue_ext_sha256"]) for c in every for w in c["workers_detail"]
    })
    mcts_binary_shas = sorted({
        str(w["mcts_ext_sha256"]) for c in every for w in c["workers_detail"]
    })
    nnue_binary_paths = sorted({
        str(w["nnue_ext_path"]) for c in every for w in c["workers_detail"]
    })
    mcts_binary_paths = sorted({
        str(w["mcts_ext_path"]) for c in every for w in c["workers_detail"]
    })
    git_meta = _git_provenance()
    reasons = [
""",
)
replace_once(
    READOUT,
    """    if len(kernels) > 1:
        reasons.append(
            f"workers ran different NNUE kernels {kernels}: avx2 and scalar "
            "differ by a multi-fold wall factor, so these cells are not one "
            "experiment",
        )
""",
    """    if len(kernels) > 1:
        reasons.append(
            f"workers ran different NNUE kernels {kernels}: avx2 and scalar "
            "differ by a multi-fold wall factor, so these cells are not one "
            "experiment",
        )
    if len(nnue_binary_shas) != 1 or len(mcts_binary_shas) != 1:
        reasons.append(
            "workers loaded different native binaries: "
            f"_nnue_ext={nnue_binary_shas}, _mcts_tree={mcts_binary_shas}"
        )
""",
)
replace_once(
    READOUT,
    """        reasons.append(
            "the nnue-qsearch and nnue-qsearch-dag per-game digests DIFFER: the "
            "two cells did not play the same games, so no wall-clock difference "
            "between them is attributable to the DAG substrate. The "
            "decomposition is VOID.",
        )
""",
    """        reasons.append(
            "the nnue-qsearch and nnue-qsearch-dag oracle DIFFERED in game "
            "trajectory and/or improved-policy search output; the two cells did "
            "not execute the same search, so no wall-clock difference is "
            "attributable purely to the DAG substrate. The decomposition is VOID.",
        )
""",
)
replace_once(
    READOUT,
    """            "kernel": kernels[0] if len(kernels) == 1 else kernels,
            "seed": plan.seed,
""",
    """            "kernel": kernels[0] if len(kernels) == 1 else kernels,
            "nnue_ext_path": nnue_binary_paths[0] if len(nnue_binary_paths) == 1 else nnue_binary_paths,
            "nnue_ext_sha256": nnue_binary_shas[0] if len(nnue_binary_shas) == 1 else nnue_binary_shas,
            "mcts_ext_path": mcts_binary_paths[0] if len(mcts_binary_paths) == 1 else mcts_binary_paths,
            "mcts_ext_sha256": mcts_binary_shas[0] if len(mcts_binary_shas) == 1 else mcts_binary_shas,
            **git_meta,
            "seed": plan.seed,
""",
)
replace_once(
    READOUT,
    """        "oracle": oracle,
        "cells": cells,
""",
    """        "oracle": oracle,
        "quality_scope": dict(QUALITY_SCOPE),
        "cells": cells,
""",
)

replace_once(
    READOUT,
    """def plan_from_args(args: argparse.Namespace) -> ReadoutPlan:
    arms = list(dict.fromkeys(args.arm))
    return ReadoutPlan(
        arm_configs=tuple(resolve_arm_config(args, arm=a) for a in arms),
""",
    """def plan_from_args(args: argparse.Namespace) -> ReadoutPlan:
    arms = list(dict.fromkeys(args.arm))
    _validate_matrix_knobs(args, arms)
    return ReadoutPlan(
        arm_configs=tuple(
            resolve_arm_config(args, arm=a, strict_foreign_knobs=False) for a in arms
        ),
""",
)

# ---------------------------------------------------------------------------
# scripts/gen_random_selfplay_shards.py — make history loss explicit in banks.
# ---------------------------------------------------------------------------
replace_once(
    GEN,
    """#: the artifact. A reanalysis that guessed them would silently run the mate rows
#: through the centipawn slope, which is the banked N1 defect in a new scale.
LEAF_BANK_SCHEMA = 2
""",
    """#: the artifact. A reanalysis that guessed them would silently run the mate rows
#: through the centipawn slope, which is the banked N1 defect in a new scale.
#:
#: 3 (2026-08-26) records whether FEN is sufficient to reconstruct the native
#: arm's repetition/search state. FEN carries the halfmove clock but not the
#: CBoard repetition hash stack; rows with a non-empty stack are explicitly NOT
#: admissible to a FEN-only history-sensitive scorer.
LEAF_BANK_SCHEMA = 3
""",
)
replace_once(
    GEN,
    """                        "fen": board.fen(),
                        "value": int(value),
""",
    """                        "fen": board.fen(),
                        "halfmove_clock": int(board.halfmove_clock),
                        "hash_stack_len": int(board.hash_stack_len),
                        "hist_len": int(board.hist_len),
                        "fen_reconstructs_full_search_state": bool(board.hash_stack_len == 0),
                        "value": int(value),
""",
)

# ---------------------------------------------------------------------------
# Tests — append mutation-resistant coverage for this fix round.
# ---------------------------------------------------------------------------
marker = "# --- #474 current-head measurement-integrity regression tests ---"
if marker in read(TEST):
    raise RuntimeError("integrity tests already appended")
with TEST.open("a") as f:
    f.write(r'''


# --- #474 current-head measurement-integrity regression tests ---

def test_multi_arm_matrix_scopes_explicit_knobs_to_the_consuming_cells(tmp_path: Path) -> None:
    args = readout.build_parser().parse_args([
        "--arm", readout.ARM_QSEARCH,
        "--arm", readout.ARM_QSEARCH_DAG,
        "--arm", readout.ARM_FASTQ,
        "--nnue-pack", str(tmp_path / "pack"),
        "--nnue-qsearch-max-ply", "3",
        "--dag-node-cap", "0",
        "--fastq-max-qply", "6",
    ])
    plan = readout.plan_from_args(args)
    by_arm = {cfg.arm: cfg for cfg in plan.arm_configs}
    assert by_arm[readout.ARM_QSEARCH].qsearch_max_ply == 3
    assert by_arm[readout.ARM_QSEARCH].fastq_max_qply is None
    assert by_arm[readout.ARM_QSEARCH_DAG].dag_node_cap == 0
    assert by_arm[readout.ARM_FASTQ].fastq_max_qply == 6
    assert by_arm[readout.ARM_FASTQ].qsearch_max_ply is None


def test_matrix_still_refuses_a_knob_no_selected_arm_consumes(tmp_path: Path) -> None:
    args = readout.build_parser().parse_args([
        "--arm", readout.ARM_QSEARCH,
        "--nnue-pack", str(tmp_path / "pack"),
        "--fastq-max-qply", "6",
    ])
    with pytest.raises(ValueError, match="no.*nnue-fastq|nnue-fastq is not selected"):
        readout.plan_from_args(args)


def test_persistent_dag_snapshots_are_differenced_not_resummed() -> None:
    first = dict(_DAG_SNAPSHOT)
    second = dict(first)
    second.update({
        "node_count": 115,
        "edge_count": 177,
        "probes": 220,
        "hits": 54,
        "inserts": 115,
        "state_makes": 75,
        "memory_bytes": first["memory_bytes"] + 1024,
    })
    assert second["state_inits"] + second["state_makes"] == second["node_count"]
    stats = readout.DagGameStats()
    stats.add(first)
    stats.add(second, previous=first)
    summary = stats.summary()
    assert summary["nodes_per_game"] == pytest.approx((110 + 5) / 2)
    assert summary["edges_per_game"] == pytest.approx((170 + 7) / 2)
    assert summary["hits"] == 54
    assert summary["probes"] == 220
    assert summary["state_makes"] == 75


def test_dag_delta_refuses_a_counter_that_goes_backwards_without_reset() -> None:
    first = dict(_DAG_SNAPSHOT)
    second = dict(first)
    second["hits"] -= 1
    stats = readout.DagGameStats()
    stats.add(first)
    with pytest.raises(ValueError, match="went backwards"):
        stats.add(second, previous=first)


def test_search_output_digest_catches_a_target_change_that_game_digest_cannot() -> None:
    row_a = argparse.Namespace(
        ply_index=0,
        policy_probs=np.array([0.5, 0.5], dtype=np.float32),
        legal_mask=np.array([True, True]),
    )
    row_b = argparse.Namespace(
        ply_index=0,
        policy_probs=np.array([0.6, 0.4], dtype=np.float32),
        legal_mask=np.array([True, True]),
    )
    same_game = readout.game_digest(
        game_index=0, start_fen="start", move_trace="e2e4", result="*", termination="max",
    )
    left = readout.GameRecord(0, 1, "*", "max", same_game, readout.search_output_digest([row_a]))
    right = readout.GameRecord(0, 1, "*", "max", same_game, readout.search_output_digest([row_b]))
    cells = {
        readout.ARM_QSEARCH: [{"repeat": 0, "games_digest": readout.games_digest([left]),
                               "searches_digest": readout.searches_digest([left])}],
        readout.ARM_QSEARCH_DAG: [{"repeat": 0, "games_digest": readout.games_digest([right]),
                                   "searches_digest": readout.searches_digest([right])}],
    }
    oracle = readout._oracle(cells)
    assert oracle["game_digests_agree"] is True
    assert oracle["search_digests_agree"] is False
    assert oracle["digests_agree"] is False


def test_quality_scope_explicitly_forbids_paired_attribution_from_end_to_end_cells() -> None:
    assert readout.QUALITY_SCOPE["population"] == "end_to_end_arm_selected"
    assert readout.QUALITY_SCOPE["paired_evaluator_quality"] is False
    assert readout.QUALITY_SCOPE["deep_sf_paired_input_admissible"] is False


def test_native_module_identity_hashes_the_actual_loaded_file(tmp_path: Path) -> None:
    binary = tmp_path / "fake.so"
    binary.write_bytes(b"one build")
    module = argparse.Namespace(__file__=str(binary))
    path, digest = readout._module_identity(module)
    assert path == str(binary.resolve())
    assert digest == readout._sha256_file(binary)
    binary.write_bytes(b"different build")
    assert readout._sha256_file(binary) != digest


def test_leaf_bank_marks_when_fen_does_not_reconstruct_repetition_history() -> None:
    import io

    sink = io.StringIO()
    source = object.__new__(gen.NnueArmValueSource)
    source._bank = sink
    source.arm = readout.ARM_QSEARCH
    source.pack_file_sha256 = "f" * 64
    source.cp_per_internal_unit = 0.28
    source.cp_slope = 0.006
    source.cp_draw_width = 120.0
    source.mate_base = 100000.0
    source.mate_ply_step = 1.0
    source.mate_max_plies = 128.0
    source.bank_identity = {}
    source.realized = {}
    source.bank_rows = 0
    board = argparse.Namespace(
        fen=lambda: "8/8/8/8/8/8/8/K6k w - - 7 9",
        halfmove_clock=7,
        hash_stack_len=3,
        hist_len=7,
    )
    gen.NnueArmValueSource._bank_batch(
        source, [board], np.array([10.0]), np.array([False]),
        role="leaf", cluster=(2, 3),
    )
    row = json.loads(sink.getvalue())
    assert row["schema"] >= 3
    assert row["halfmove_clock"] == 7
    assert row["hash_stack_len"] == 3
    assert row["fen_reconstructs_full_search_state"] is False
''')

# ---------------------------------------------------------------------------
# Docs — state the stronger oracle and the intentionally non-paired quality scope.
# ---------------------------------------------------------------------------
replace_once(
    DOC,
    """* `games_detail[i].digest` — `sha256("<game>:<start_fen>:<move_trace>:<result>:<termination>")`
* `games_digest` — one digest over all of them, ordered by game index
""",
    """* `games_detail[i].digest` — the game trajectory/termination digest;
* `games_detail[i].search_digest` — the exact improved-policy/legal-mask output
  sequence for that game;
* `games_digest` and `searches_digest` — ordered whole-cell digests of those two
  independent views.
""",
)
replace_once(
    DOC,
    """> **If `oracle.digests_agree` is `false`, the decomposition is VOID.** The two
> cells did not play the same games, so no wall-clock difference between them is
> attributable to the DAG substrate — it is attributable to a different search.
""",
    """> **If `oracle.digests_agree` is `false`, the decomposition is VOID.** It
> now requires BOTH game-trajectory equality and improved-policy/search-output
> equality. Equal played moves alone are insufficient: sequential halving can
> absorb changed leaf values while still choosing the same move.
""",
)
replace_once(
    DOC,
    """The oracle is live, not decorative, and its sensitivity has been measured rather
than assumed. Perturbing evaluated leaf values in the DAG arm only (+500 internal
units on one leaf of the first *N* leaf batches, `--games 2 --sims 8
--max-plies 12 --seed 7`):
""",
    """The old trajectory-only oracle was measured and shown to absorb several
perturbed leaves before a move changed. That experiment is retained below as the
reason the current schema also digests the improved-policy/search output: the
stronger oracle catches a changed target even when the played move is unchanged.
Perturbing evaluated leaf values in the DAG arm only (+500 internal units on one
leaf of the first *N* leaf batches, `--games 2 --sims 8 --max-plies 12 --seed 7`):
""",
)
replace_once(
    DOC,
    """The report therefore names graph sizes `*_per_game`. `memory_peak_per_worker_bytes`
is different: because reset retains capacity it is the worker's resident DAG
allocation high-water mark, not the sum of positions seen over the run.
""",
    """The per-game work fields are DELTAS of the cumulative `arm_dag_stats()`
snapshot. This matters for `never`/`every-N-games`: summing absolute snapshots
would count game 1 again in game 2, game 3, and so on. Resource peaks remain
absolute snapshots. `memory_peak_per_worker_bytes` is the worker's resident DAG
allocation high-water mark, not the sum of positions seen over the run.
""",
)
replace_once(
    DOC,
    """The command accepts both qsearch-family and FastQ-family flags so the three cells
can be driven by one tool, but it refuses a knob the selected provider does not
consume.
""",
    """The command accepts both qsearch-family and FastQ-family flags so the three
cells can be driven by one tool. In a multi-arm matrix, a supplied knob is legal
when at least one selected cell consumes it, and each `ResolvedArmConfig` copies
only its own fields. In a single-arm invocation, foreign knobs are still refused.
""",
)
replace_once(
    DOC,
    """To bank leaf observations for the deep-SF quality readout, add the flag to the
**same** command so every cell pays it:
""",
    """To bank raw end-to-end trace observations, add the flag to the **same**
command so every cell pays it. **These files are NOT a paired deep-SF evaluator
quality sample**: each arm drives its own Gumbel search, so changing FastQ can
change which later leaves exist. A paired quality experiment needs a frozen
driver population with shadow arms that do not feed values back into MCTSTree.
This PR deliberately does not claim that attribution:
""",
)
replace_once(
    DOC,
    """Bank files are opened `"x"`: a rerun that would append into the previous run's
rows fails instead. Rows carry FEN, the raw internal value, an `is_mate` flag,
""",
    """Bank files are opened `"x"`: a rerun that would append into the previous run's
rows fails instead. Rows carry FEN plus `halfmove_clock`, `hash_stack_len`,
`hist_len`, and `fen_reconstructs_full_search_state`. FEN does NOT carry the
CBoard repetition hash stack: a row with that flag false must be excluded by any
FEN-only history-sensitive scorer rather than silently reconstructed as a fresh
position. Rows also carry the raw internal value, an `is_mate` flag,
""",
)
replace_once(
    DOC,
    """`banking` · `dag_reset` · `repeats` · `arms` · `python`.
""",
    """`banking` · `dag_reset` · `repeats` · `arms` · `python` · the Git HEAD/dirty
flag · and the actual loaded `_nnue_ext` / `_mcts_tree` paths and SHA-256 hashes.
""",
)

print("#474 measurement-integrity fixes applied")
