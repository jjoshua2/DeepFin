from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.nnue import _nnue_ext
from chess_anti_engine.replay.shard import load_shard_arrays
from scripts import gen_random_selfplay_shards as gen
from scripts import nnue_gumbel_readout as readout
from scripts import nnue_shadow_label_readout as shadow
from tests.test_nnue_gumbel_readout import FakeExt, _args
from tests.test_nnue_native_eval import write_synthetic_pack

_ARMS: tuple[str, ...] = readout.READOUT_ARMS


@pytest.fixture(scope="module")
def live_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The real net when the operator has one, otherwise the synthetic layout.

    ⚑ THE TWO ARE NOT EQUALLY POWERFUL AND THE TEST SAYS SO. The synthetic pack
    is all zeros, so every arm evaluates every position at 0 and the three arms
    are INDISTINGUISHABLE -- which means a defect that made an observer's value
    reach the search would not move a single digest under it. Set
    ``CAE_NNUE_TEST_PACK`` to a real pack to run the end-to-end body where the
    arms actually differ.
    """
    env = os.environ.get("CAE_NNUE_TEST_PACK")
    if env and Path(env).is_file():
        return Path(env)
    path = tmp_path_factory.mktemp("shadow") / "tiny.pack"
    write_synthetic_pack(path)
    return path


@pytest.fixture(autouse=True)
def _extension_defaults():
    """Leave the compiled globals as they were found.

    `set_arm_config` / `fastq_set_config` are PROCESS-wide, and this file opens
    several contexts per test; a test that inherited another's configuration
    would assert the wrong knobs by accident.
    """
    def restore() -> None:
        _nnue_ext.set_arm_config(
            _nnue_ext.RESOLVER_MAX_DEPTH,
            _nnue_ext.QSEARCH_MAX_PLY,
            _nnue_ext.QSEARCH_CHECK_PLIES,
            _nnue_ext.QSEARCH_DAG_NODE_CAP,
        )
        _nnue_ext.fastq_set_config(
            _nnue_ext.FASTQ_MAX_QPLY,
            _nnue_ext.FASTQ_NODE_CAP,
            _nnue_ext.FASTQ_DELTA_MARGIN,
            _nnue_ext.FASTQ_RECAPTURE_EXEMPT,
        )

    restore()
    yield
    restore()


def _cli_args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "driver": readout.ARM_QSEARCH,
        "arm": None,
        "oneply_sigma": None,
        "dag_max_nodes": shadow.DEFAULT_DAG_MAX_NODES,
        "dag_reset": "game",
        "shard_size": 2000,
        "max_plies": 40,
        "nnue_pack": Path("pack"),
        "out_dir": Path("out"),
        "games": 4,
        "workers": 1,
        "seed": 1,
        "sims": 16,
        "topk": gen.MAX_LEGAL_MOVES,
        "all_root_moves": True,
        "run_id": "test",
        "nice": 0,
        "bank_leaf_observations": None,
        "nnue_resolver_max_depth": None,
        "nnue_qsearch_max_ply": None,
        "nnue_qsearch_check_plies": None,
        "dag_node_cap": None,
        "allow_binding_dag_node_cap": False,
        "fastq_max_qply": None,
        "fastq_node_cap": None,
        "fastq_delta_margin": None,
        "fastq_recapture_exempt": None,
        "nnue_cp_per_unit": gen.NNUE_CP_PER_INTERNAL_UNIT,
        "nnue_cp_slope": gen.NNUE_CP_SLOPE,
        "nnue_cp_draw_width": gen.NNUE_CP_DRAW_WIDTH,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


# ── the 1-ply label ──────────────────────────────────────────────────────────


def test_the_oneply_target_is_dense_over_the_legal_set_and_nowhere_else() -> None:
    legal = (5, 900, 4000)
    probs = shadow.oneply_policy_vector(legal, np.array([0.1, -0.3, 0.9]), sigma=5.5)
    assert probs.shape == (gen.POLICY_SIZE,)
    assert pytest.approx(1.0, abs=1e-6) == float(probs.sum())
    assert sorted(np.flatnonzero(probs).tolist()) == list(legal)
    # Monotone in the arm's own value, which is the whole content of the label.
    assert int(np.argmax(probs)) == 4000


def test_the_oneply_sigma_cannot_change_the_target_argmax() -> None:
    """⚑ The claim the whole reporting rests on: sigma is inert for top1.

    ``score_shard_labels.py``'s primary metric is the deep-SF regret of the
    target's argmax, so if sigma could move an argmax the ``oneply__`` cells'
    headline number would be a function of a flag rather than of the arm.
    """
    legal = (1, 2, 3, 4)
    values = np.array([-0.42, 0.17, 0.16, -0.9])
    argmaxes = {
        int(np.argmax(shadow.oneply_policy_vector(legal, values, sigma=s)))
        for s in (0.01, 0.5, 5.5, 50.0, 500.0)
    }
    assert argmaxes == {2}


def test_a_non_finite_arm_value_is_refused_rather_than_softmaxed() -> None:
    with pytest.raises(ValueError, match="non-finite arm value"):
        shadow.oneply_policy_vector((1, 2), np.array([0.5, np.nan]), sigma=5.5)


def test_the_oneply_target_is_read_from_the_root_movers_seat() -> None:
    """⚑ The arm answers from the CHILD's seat; the label is the root mover's.

    A target built on the un-negated child value ranks the root's moves exactly
    backwards and still sums to one, is still dense over the legal set, and
    still passes every shape check in this file.
    """
    class _Observer:
        arm = "probe"

        def __init__(self) -> None:
            self.roles: list[str] = []

        def evaluate(
            self, boards: list[CBoard], *, role: str, cluster: tuple[int, int] | None,
        ) -> np.ndarray:
            del cluster
            self.roles.append(role)
            # A value that is a strictly increasing function of the child's
            # index, so "which move wins" is unambiguous in both conventions.
            return np.arange(len(boards), dtype=np.float64) / 100.0

    observer = _Observer()
    root = CBoard.from_board(chess.Board())
    probe = shadow.probe_root(root, observers=(observer,), cluster=(0, 0))

    assert observer.roles == ["root", "root-child"]
    assert len(probe.legal_full_indices) == 20
    # The arm liked the LAST child most; from the root mover's seat that is the
    # WORST move, so the label's argmax must be the FIRST legal move.
    assert int(np.argmax(probe.q_mover["probe"])) == 0
    assert float(probe.q_mover["probe"][0]) > float(probe.q_mover["probe"][-1])


def test_a_probe_whose_legal_set_differs_from_the_search_is_refused() -> None:
    record = gen.PlyRecord(
        x=np.zeros((1, 8, 8), dtype=np.float32),
        policy_probs=np.zeros((gen.POLICY_SIZE,), dtype=np.float32),
        legal_mask=np.zeros((gen.POLICY_SIZE,), dtype=bool),
        pov_white=True,
        ply_index=0,
        search_value=0.0,
    )
    record.legal_mask[[3, 9]] = True
    outcome = gen.GameOutcome(
        records=[record], result="1/2-1/2", plies=1, termination="max_plies",
        start_fen=chess.STARTING_FEN, opening_source="random", move_trace="3",
        end_ply_index=1,
    )
    probe = shadow.PlyProbe(
        game=0, ply_ordinal=0, legal_full_indices=(3, 10),
        q_mover={"a": np.array([0.1, 0.2])},
    )
    with pytest.raises(RuntimeError, match="not looking at the position"):
        shadow.oneply_outcome(outcome, [probe], arm="a", sigma=5.5)


def test_a_per_ply_hook_that_fires_out_of_order_is_refused() -> None:
    recorder = shadow.ProbeRecorder()
    recorder.begin_game(7)
    probe = shadow.PlyProbe(
        game=7, ply_ordinal=0, legal_full_indices=(1,), q_mover={"a": np.array([0.0])},
    )
    recorder.add(probe)
    with pytest.raises(RuntimeError, match="exactly once per ply"):
        recorder.add(probe)
    with pytest.raises(RuntimeError, match="the evaluator's cluster key"):
        recorder.add(
            shadow.PlyProbe(
                game=8, ply_ordinal=1, legal_full_indices=(1,),
                q_mover={"a": np.array([0.0])},
            ),
        )


# ── the canonical-store watchdog (the OOM bound) ─────────────────────────────


class _GrowingStore:
    """A DAG-backed source whose store grows until something resets it."""

    def __init__(self, *, per_batch: int = 40) -> None:
        self.nodes = 0
        self.per_batch = int(per_batch)
        self.resets = 0
        self.high_water = 0

    def evaluate_a_batch(self) -> None:
        self.nodes += self.per_batch
        self.high_water = max(self.high_water, self.nodes)

    def dag_stats(self) -> dict[str, int] | None:
        return {"node_count": self.nodes, "memory_bytes": self.nodes * 5681}

    def reset_game(self) -> None:
        self.resets += 1
        self.nodes = 0


def test_the_store_watchdog_bounds_a_growing_dag_and_reports_the_true_peak() -> None:
    """⚑ ``--dag-node-cap`` is a PER-CALL quiescence budget, not a store bound.

    Measured on this build: 3000 evaluations at cap 0 interned 1,019,128 nodes
    for 5.79 GB, and the same 3000 at cap 4096 interned 994,061 for the same
    5.79 GB with 16 trips. The store is bounded here or it is not bounded.
    """
    source = _GrowingStore(per_batch=40)
    watch = shadow.DagStoreWatch(max_nodes=100)
    for _ in range(50):
        source.evaluate_a_batch()
        watch.observe(source)
    assert watch.resets > 0
    assert source.high_water <= 120  # the cap plus at most one batch of overshoot
    # The peak is read BEFORE the reset, so it is what the store reached rather
    # than what it was left at.
    assert watch.nodes_peak == source.high_water
    assert watch.memory_peak == source.high_water * 5681


def test_a_zero_max_nodes_disables_the_watchdog_and_says_so_in_the_peak() -> None:
    source = _GrowingStore(per_batch=40)
    watch = shadow.DagStoreWatch(max_nodes=0)
    for _ in range(50):
        source.evaluate_a_batch()
        watch.observe(source)
    assert watch.resets == 0
    assert watch.nodes_peak == 2000


def test_a_non_dag_arm_is_never_reset_by_the_watchdog() -> None:
    class _NoStore:
        def __init__(self) -> None:
            self.resets = 0

        def dag_stats(self) -> dict[str, int] | None:
            return None

        def reset_game(self) -> None:
            self.resets += 1

    source = _NoStore()
    watch = shadow.DagStoreWatch(max_nodes=1)
    watch.observe(source)
    assert (source.resets, watch.resets, watch.nodes_peak) == (0, 0, 0)


class _EvalFakeExt(FakeExt):
    """``FakeExt`` plus the one entry point ``q_for_boards`` calls."""

    def arm_handle_eval(self, _handle: Any, boards: list[Any]) -> list[int]:
        return [7] * len(boards)


def test_the_driver_is_watched_even_with_no_observers_attached(tmp_path: Path) -> None:
    """⚑ The DRIVER's store needs the bound too, and it must not depend on the
    observers -- FastQ and qsearch-DAG both intern, so a DAG driver would OOM,
    and a watchdog that only ran with observers attached would be a difference
    between the two passes of the inertness proof rather than something it
    covers."""
    ext = _EvalFakeExt()
    pack = tmp_path / "tiny.pack"
    pack.write_bytes(b"fake extension owns this test")
    config = readout.resolve_arm_config(
        _args(readout.ARM_QSEARCH_DAG), ext, strict_foreign_knobs=False,
    )
    watch = shadow.DagStoreWatch(max_nodes=10)
    source = shadow.ShadowFanoutSource(
        observers=(),
        recorder=shadow.ProbeRecorder(),
        agreement=shadow.AgreementStats(),
        dag_watch=watch,
        config=config,
        pack=pack,
        cp_per_internal_unit=0.28,
        cp_slope=0.006,
        cp_draw_width=120.0,
        ext=ext,
    )
    board = CBoard.from_board(chess.Board())
    source.q_for_boards([board, board.copy()], role="leaf", cluster=(0, 0))
    # The fake's snapshot holds 110 nodes against a cap of 10, so one batch is
    # enough: the watchdog either fired or the driver is unbounded.
    assert watch.resets == 1
    assert watch.nodes_peak == 110
    assert ext.resets == 1


# ── the shadow-inertness proof ───────────────────────────────────────────────


def _digest_result(games: str, searches: str) -> shadow.WorkerResult:
    return shadow.WorkerResult(
        worker_id=0, games=1, plies=1, setup_s=0.0, elapsed_s=0.0,
        peak_rss_bytes=0,
        game_records=[
            readout.GameRecord(
                game=0, plies=1, result="1-0", termination="checkmate",
                digest=games, search_digest=searches,
            ),
        ],
        terminations={}, root_budget=gen.RootBudgetStats(),
        driver_provider_stats={}, observer_provider_stats={}, observer_stats={},
        driver_dag_watch=shadow.DagStoreWatch(max_nodes=0), observer_dag_watch={},
        agreement=shadow.AgreementStats(), argmax_pairs={}, shards={}, rows=0,
        bank_rows=0, kernel="scalar", pack_file_sha256="a" * 64,
        pack_source_sha256="b" * 64, nice_realized=0,
    )


def _plan(**overrides: Any) -> shadow.RunConfig:
    driver = readout.ResolvedArmConfig(
        arm=readout.ARM_QSEARCH, resolver_max_depth=4, qsearch_max_ply=2,
        qsearch_check_plies=1,
    )
    values: dict[str, Any] = {
        "driver": driver,
        "observers": (driver,),
        "pack": Path("pack"),
        "out_dir": Path("out"),
        "games": 2,
        "workers": 1,
        "seed": 1,
        "sims": 8,
        "topk": gen.MAX_LEGAL_MOVES,
        "max_plies": 4,
        "all_root_moves": True,
        "cp_per_internal_unit": 0.28,
        "cp_slope": 0.006,
        "cp_draw_width": 120.0,
        "oneply_sigma": 5.5,
        "dag_max_nodes": 1000,
        "dag_reset_every": 1,
        "shard_size": 2000,
        "bank_path": None,
        "run_id": "test",
        "nice": 0,
    }
    values.update(overrides)
    return shadow.RunConfig(**values)


def test_a_shadow_that_moves_the_driver_fails_the_proof_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ The gate this whole harness turns on: a differing digest RAISES.

    A shadow that perturbed the search would make every cell a different
    experiment, so there is no partial reading to report and no warning that
    would be honest.
    """
    calls: list[bool] = []

    def fake_run(cfg: shadow.RunConfig) -> list[shadow.WorkerResult]:
        attached = bool(cfg.attach_observers)
        calls.append(attached)
        return [
            _digest_result("g", "s") if attached else _digest_result("g", "OTHER")
        ]

    monkeypatch.setattr(shadow, "_run_workers", fake_run)
    with pytest.raises(RuntimeError, match="SHADOW INERTNESS PROOF FAILED"):
        shadow.prove_shadow_inertness(_plan(), games=2)
    assert calls == [True, False]


def test_the_proof_passes_only_when_both_digests_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        shadow, "_run_workers", lambda cfg: [_digest_result("g", "s")],
    )
    proof = shadow.prove_shadow_inertness(_plan(), games=3)
    assert proof["digests_agree"] is True
    assert proof["games"] == 3
    assert proof["with_observers"] == proof["without_observers"]


def test_the_proof_pass_detaches_the_observers_and_writes_no_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[shadow.RunConfig] = []

    def fake_run(cfg: shadow.RunConfig) -> list[shadow.WorkerResult]:
        seen.append(cfg)
        return [_digest_result("g", "s")]

    monkeypatch.setattr(shadow, "_run_workers", fake_run)
    shadow.prove_shadow_inertness(_plan(games=64, workers=8), games=2)
    assert [c.attach_observers for c in seen] == [True, False]
    assert [c.observers for c in seen][1] == ()
    assert all(c.games == 2 and c.workers == 1 for c in seen)
    assert not any(c.emit_shards for c in seen)


def test_zero_prove_games_cannot_stand_in_for_a_proof() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        shadow.prove_shadow_inertness(_plan(), games=0)


# ── cell alignment ───────────────────────────────────────────────────────────


def _row(game_id: int, ply: int, *, policy_index: int) -> Any:
    from chess_anti_engine.replay.sample import ReplaySample

    # The mask is FIXED across rows: cells differ only in `policy_target`, which
    # is exactly the shape the alignment check is supposed to accept.
    policy = np.zeros((gen.COMPACT_POLICY_SIZE,), dtype=np.float32)
    policy[policy_index] = 1.0
    mask = np.zeros((gen.COMPACT_POLICY_SIZE,), dtype=np.uint8)
    mask[[3, 4, 9, 11]] = 1
    return ReplaySample(
        x=np.zeros((175, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=1,
        legal_mask=mask,
        has_policy=True,
        is_selfplay=True,
        is_network_turn=True,
        moves_left=0.5,
        game_id=game_id,
        ply_index=ply,
    )


def _write_cell(out_dir: Path, cell: str, rows: list[Any]) -> None:
    directory = shadow.cell_dir(out_dir, cell)
    directory.mkdir(parents=True, exist_ok=True)
    gen.write_shard(
        out_dir=directory, index=0, rows=rows,
        cfg=gen.GenConfig(out_dir=directory, run_id="test"),
        tally=gen.ShardTally(),
    )


def test_row_aligned_cells_pass_and_report_what_they_compared(tmp_path: Path) -> None:
    rows_a = [_row(1, 0, policy_index=3), _row(1, 1, policy_index=4)]
    rows_b = [_row(1, 0, policy_index=9), _row(1, 1, policy_index=11)]
    _write_cell(tmp_path, "left", rows_a)
    _write_cell(tmp_path, "right", rows_b)
    summary = shadow.assert_cells_are_row_aligned(tmp_path, ("left", "right"))
    assert summary["shards_per_cell"] == 1
    assert summary["cross_cell_shard_comparisons"] == 1
    assert "policy_target" not in summary["aligned_columns"]


def test_cells_whose_rows_are_not_the_same_positions_are_refused(
    tmp_path: Path,
) -> None:
    """⚑ ``score_shard_labels.py`` samples by ROW ORDINAL, so this is the check
    that makes the comparison paired; nothing downstream repeats it."""
    _write_cell(tmp_path, "left", [_row(1, 0, policy_index=3)])
    _write_cell(tmp_path, "right", [_row(2, 0, policy_index=3)])
    with pytest.raises(RuntimeError, match="column 'game_id' differs"):
        shadow.assert_cells_are_row_aligned(tmp_path, ("left", "right"))


def test_cells_with_different_shard_sets_are_refused(tmp_path: Path) -> None:
    _write_cell(tmp_path, "left", [_row(1, 0, policy_index=3)])
    shadow.cell_dir(tmp_path, "right").mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="not row-aligned"):
        shadow.assert_cells_are_row_aligned(tmp_path, ("left", "right"))


# ── the resolution table ─────────────────────────────────────────────────────


def test_cells_that_never_disagree_are_flagged_before_any_stockfish_is_spent(
    tmp_path: Path,
) -> None:
    """⚑ A pair whose argmax always agrees has a top1 delta of exactly 0.00 cp
    by arithmetic. Spending hours of deep SF to be told that is the failure."""
    same = shadow.ArgmaxAgreement()
    differs = shadow.ArgmaxAgreement()
    for i in range(50):
        same.add(True)
        differs.add(i % 10 != 0)
    result = _digest_result("g", "s")
    result.argmax_pairs = {
        shadow.pair_key("a", "b"): same,
        shadow.pair_key("a", "c"): differs,
    }
    result.observer_provider_stats = {readout.ARM_QSEARCH: _qsearch_provider()}
    result.observer_stats = {readout.ARM_QSEARCH: shadow.ObserverStats()}
    result.observer_dag_watch = {readout.ARM_QSEARCH: shadow.DagStoreWatch(max_nodes=0)}
    result.driver_provider_stats = _qsearch_provider()
    result.arm_config_realized = {readout.ARM_QSEARCH: {}}
    result.agreement = shadow.AgreementStats(compared=10, disagreements=0)
    report = shadow._aggregate(
        _plan(out_dir=tmp_path), [result], wall_s=1.0,
        proof={"digests_agree": True}, alignment=None,
    )
    assert report["zero_resolution_pairs"] == ["a|b"]
    assert report["label_argmax_agreement"]["a|c"]["rows_disagreeing"] == 5.0
    assert report["label_argmax_agreement"]["a|c"]["disagreement_rate"] == 0.1


def _qsearch_provider() -> dict[str, int]:
    keys = (
        gen._CTX_COUNTER_KEYS | gen._CTX_PEAK_KEYS | gen._CTX_STORE_SIZE_KEYS
        | gen._CTX_CONFIG_KEYS
    )
    return dict.fromkeys(sorted(keys), 0)


def test_a_driver_that_disagrees_with_its_own_observer_is_inadmissible(
    tmp_path: Path,
) -> None:
    """⚑ The wiring proof. Same arm, same config, same positions -- so the two
    contexts must agree, and a disagreement means the observers are not looking
    at the driver's leaves (or the arm is not a function of the position)."""
    result = _digest_result("g", "s")
    result.driver_provider_stats = _qsearch_provider()
    result.observer_provider_stats = {readout.ARM_QSEARCH: _qsearch_provider()}
    result.observer_stats = {readout.ARM_QSEARCH: shadow.ObserverStats()}
    result.observer_dag_watch = {readout.ARM_QSEARCH: shadow.DagStoreWatch(max_nodes=0)}
    result.arm_config_realized = {readout.ARM_QSEARCH: {}}
    result.agreement = shadow.AgreementStats(
        compared=100, disagreements=3, first_disagreement="8/8/8 w - - 0 1: 1 vs 2",
    )
    report = shadow._aggregate(
        _plan(out_dir=tmp_path), [result], wall_s=1.0,
        proof={"digests_agree": True}, alignment=None,
    )
    assert report["admissible"] is False
    assert any("disagreed on 3 of 100" in r for r in report["inadmissible_reasons"])


def test_a_run_that_never_compared_the_driver_against_itself_is_inadmissible(
    tmp_path: Path,
) -> None:
    result = _digest_result("g", "s")
    result.driver_provider_stats = _qsearch_provider()
    result.observer_provider_stats = {readout.ARM_QSEARCH: _qsearch_provider()}
    result.observer_stats = {readout.ARM_QSEARCH: shadow.ObserverStats()}
    result.observer_dag_watch = {readout.ARM_QSEARCH: shadow.DagStoreWatch(max_nodes=0)}
    result.arm_config_realized = {readout.ARM_QSEARCH: {}}
    report = shadow._aggregate(
        _plan(out_dir=tmp_path), [result], wall_s=1.0,
        proof={"digests_agree": True}, alignment=None,
    )
    assert report["admissible"] is False
    assert any("nothing proves the observers" in r for r in report["inadmissible_reasons"])


def test_the_node_budget_counter_is_read_from_each_arms_own_surface(
    tmp_path: Path,
) -> None:
    """⚑ Two surfaces, two key names, and the same number means different
    things: FastQ's ``budget_trips`` is the arm doing its job, the DAG arm's
    ``dag_budget_trips`` means it stood pat where the oracle searched."""
    fastq_cfg = readout.ResolvedArmConfig(
        arm=readout.ARM_FASTQ, fastq_max_qply=4, fastq_node_cap=32,
        fastq_delta_margin=200, fastq_recapture_exempt=1,
    )
    driver = readout.ResolvedArmConfig(
        arm=readout.ARM_QSEARCH, resolver_max_depth=4, qsearch_max_ply=2,
        qsearch_check_plies=1,
    )
    fastq_provider = dict.fromkeys(
        sorted(shadow.readout.FASTQ_KEY_CLASSES.counters
               | shadow.readout.FASTQ_KEY_CLASSES.peaks
               | shadow.readout.FASTQ_KEY_CLASSES.config), 0,
    )
    fastq_provider["budget_trips"] = 17
    result = _digest_result("g", "s")
    result.driver_provider_stats = _qsearch_provider()
    result.observer_provider_stats = {
        readout.ARM_QSEARCH: _qsearch_provider(),
        readout.ARM_FASTQ: fastq_provider,
    }
    result.observer_stats = {
        readout.ARM_QSEARCH: shadow.ObserverStats(),
        readout.ARM_FASTQ: shadow.ObserverStats(),
    }
    result.observer_dag_watch = {
        readout.ARM_QSEARCH: shadow.DagStoreWatch(max_nodes=0),
        readout.ARM_FASTQ: shadow.DagStoreWatch(max_nodes=0),
    }
    result.arm_config_realized = {readout.ARM_QSEARCH: {}, readout.ARM_FASTQ: {}}
    result.agreement = shadow.AgreementStats(compared=5, disagreements=0)
    report = shadow._aggregate(
        _plan(out_dir=tmp_path, driver=driver, observers=(driver, fastq_cfg)),
        [result], wall_s=1.0, proof={"digests_agree": True}, alignment=None,
    )
    fastq = report["observers"][readout.ARM_FASTQ]
    assert fastq["node_budget_trip_counter"] == "budget_trips"
    assert fastq["node_budget_trips"] == 17
    qsearch = report["observers"][readout.ARM_QSEARCH]
    assert qsearch["node_budget_trip_counter"] == "dag_budget_trips"


# ── CLI wiring ───────────────────────────────────────────────────────────────


def test_the_driver_arm_is_always_an_observer_arm() -> None:
    cfg = shadow.config_from_args(
        _cli_args(driver=readout.ARM_FASTQ, arm=[readout.ARM_QSEARCH]),
    )
    assert cfg.driver.arm == readout.ARM_FASTQ
    assert readout.ARM_FASTQ in cfg.arms
    assert cfg.cells[0] == f"{shadow.SEARCH_CELL_PREFIX}{readout.ARM_FASTQ}"


def test_the_default_matrix_is_every_arm() -> None:
    cfg = shadow.config_from_args(_cli_args())
    assert set(cfg.arms) == set(_ARMS)
    assert len(cfg.cells) == len(_ARMS) + 1


def test_a_binding_dag_node_cap_is_refused_on_the_driver_and_allowed_on_an_observer() -> None:
    """⚑ The two roles are not the same question.

    On the DRIVER a binding cap changes which positions exist for every cell.
    On an OBSERVER it makes the arm under test the capped arm -- a different
    arm, labelling the same rows, whose trips are published.
    """
    with pytest.raises(ValueError, match="frozen position set"):
        shadow.config_from_args(
            _cli_args(driver=readout.ARM_QSEARCH_DAG, dag_node_cap=64),
        )
    allowed = shadow.config_from_args(
        _cli_args(
            driver=readout.ARM_QSEARCH_DAG, dag_node_cap=64,
            allow_binding_dag_node_cap=True,
        ),
    )
    assert allowed.driver.consumed()["dag_node_cap"] == 64
    cfg = shadow.config_from_args(
        _cli_args(driver=readout.ARM_QSEARCH, dag_node_cap=64),
    )
    dag = next(c for c in cfg.observers if c.arm == readout.ARM_QSEARCH_DAG)
    assert dag.consumed()["dag_node_cap"] == 64


def test_a_non_positive_oneply_sigma_is_refused() -> None:
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="oneply-sigma"):
            shadow.config_from_args(_cli_args(oneply_sigma=bad))


def test_the_default_sigma_is_the_production_targets_own_scale() -> None:
    assert shadow.oneply_sigma_default() == pytest.approx(
        gen.SELFPLAY_GUMBEL_C_SCALE * (50.0 + gen.DEFAULT_TARGET_MAX_VISIT_CAP),
    )


def test_a_bank_path_names_the_role_the_arm_and_the_worker() -> None:
    """⚑ Several contexts write banks in one worker; a name that dropped the arm
    would merge three arms' rows into one file with nothing able to split them."""
    path = shadow._worker_bank_path(
        Path("/x/bank.jsonl"), role="observer", arm=readout.ARM_FASTQ, worker_id=3,
    )
    assert path is not None
    assert path.name == "bank.observer.nnue-fastq.w03.jsonl"
    driver = shadow._worker_bank_path(
        Path("/x/bank.jsonl"), role="driver", arm=readout.ARM_FASTQ, worker_id=3,
    )
    assert driver is not None
    assert driver != path
    assert shadow._worker_bank_path(None, role="driver", arm="a", worker_id=0) is None


def test_a_rerun_into_a_populated_cell_directory_is_refused(tmp_path: Path) -> None:
    cfg = shadow.config_from_args(
        _cli_args(out_dir=tmp_path, nnue_pack=tmp_path / "p.pack"),
    )
    (tmp_path / "p.pack").write_bytes(b"x")
    directory = shadow.cell_dir(tmp_path, cfg.cells[0])
    directory.mkdir(parents=True)
    (directory / "shard_000000.zarr").mkdir()
    with pytest.raises(FileExistsError, match="rows came from two runs"):
        shadow.run(cfg, prove_games=0)


def test_the_run_refuses_a_driver_that_is_not_an_observer(tmp_path: Path) -> None:
    driver = readout.ResolvedArmConfig(
        arm=readout.ARM_QSEARCH, resolver_max_depth=4, qsearch_max_ply=2,
        qsearch_check_plies=1,
    )
    other = readout.ResolvedArmConfig(
        arm=readout.ARM_FASTQ, fastq_max_qply=4, fastq_node_cap=32,
        fastq_delta_margin=200, fastq_recapture_exempt=1,
    )
    pack = tmp_path / "p.pack"
    pack.write_bytes(b"x")
    cfg = _plan(driver=driver, observers=(other,), pack=pack, out_dir=tmp_path)
    with pytest.raises(ValueError, match="must also be an observer arm"):
        shadow.run(cfg, prove_games=0)


def test_the_resolved_arm_configs_come_from_the_extension(tmp_path: Path) -> None:
    """Every knob a selected arm consumes is a concrete number in the plan."""
    ext = FakeExt()
    cfg = readout.resolve_arm_config(
        _args(readout.ARM_QSEARCH_DAG), ext, strict_foreign_knobs=False,
    )
    assert cfg.consumed() == {
        "resolver_max_depth": FakeExt.RESOLVER_MAX_DEPTH,
        "qsearch_max_ply": FakeExt.QSEARCH_MAX_PLY,
        "qsearch_check_plies": FakeExt.QSEARCH_CHECK_PLIES,
        "dag_node_cap": FakeExt.QSEARCH_DAG_NODE_CAP,
    }
    del tmp_path


# ── end to end, on the real compiled arms ────────────────────────────────────


@pytest.mark.slow
def test_the_whole_harness_runs_and_every_gate_reports(
    live_pack: Path, tmp_path: Path,
) -> None:
    """One tiny run through the production C Gumbel path, all three arms.

    The pack is the synthetic all-zeros layout, so this proves WIRING -- the
    observers see the driver's positions, the driver is unmoved by them, the
    cells are row-aligned and every artifact is written -- not label quality.
    """
    cfg = shadow.config_from_args(
        _cli_args(
            nnue_pack=live_pack, out_dir=tmp_path, games=2, workers=1,
            sims=8, max_plies=6, seed=99,
            bank_leaf_observations=tmp_path / "bank.jsonl",
        ),
    )
    report = shadow.run(cfg, prove_games=1)

    assert report["admissible"] is True, report["inadmissible_reasons"]
    assert report["shadow_inertness_proof"]["digests_agree"] is True
    assert (
        report["shadow_inertness_proof"]["with_observers"]
        == report["shadow_inertness_proof"]["without_observers"]
    )
    agreement = report["driver_observer_agreement"]
    assert agreement["compared"] > 0
    assert agreement["disagreements"] == 0

    # Every arm saw the SAME positions, which is the entire point.
    populations = {
        arm: report["observers"][arm]["populations"] for arm in cfg.arms
    }
    for key in ("root_positions", "child_positions", "leaf_positions"):
        assert len({p[key] for p in populations.values()}) == 1, (key, populations)
        assert next(iter(populations.values()))[key] > 0

    # Every cell holds the same rows, and only the policy column differs.
    assert report["cell_alignment"]["shards_per_cell"] >= 1
    targets = {}
    for cell in cfg.cells:
        directory = shadow.cell_dir(tmp_path, cell)
        meta = json.loads((directory / "cell_meta.json").read_text())
        assert meta["cell"] == cell
        arrays, _ = load_shard_arrays(directory / "shard_000000.zarr")
        targets[cell] = np.asarray(arrays["policy_target"])
        assert targets[cell].shape[0] == report["rows_per_cell"]
        mask = np.asarray(arrays["legal_mask"]).astype(bool)
        assert np.all(targets[cell][~mask] == 0.0)
        assert np.allclose(targets[cell].sum(axis=1), 1.0, atol=2e-2)
    # ⚑ The synthetic pack is all zeros, so every arm values every position at 0
    # and every 1-ply ranking is the uniform one; the cells CANNOT be asserted
    # to differ here, and the run says so itself rather than being read as if
    # they had. The per-arm split is pinned by the unit tests above. With a real
    # pack (CAE_NNUE_TEST_PACK) the arms separate and the search cell must too.
    search_cell = shadow.search_cell_name(cfg.driver.arm)
    oneply_cell = shadow.oneply_cell_name(cfg.driver.arm)
    if len({t.tobytes() for t in targets.values()}) == 1:
        assert sorted(report["zero_resolution_pairs"]) == sorted(
            report["label_argmax_agreement"],
        )
    else:
        assert not np.array_equal(targets[search_cell], targets[oneply_cell])

    # The bank carries one file per (role, arm, worker), each labelled.
    banks = sorted(p.name for p in tmp_path.glob("bank.*.jsonl"))
    assert len(banks) == len(cfg.arms) + 1
    rows = [
        json.loads(line)
        for line in (tmp_path / f"bank.driver.{cfg.driver.arm}.w00.jsonl").read_text().splitlines()
    ]
    assert {r["role"] for r in rows} == {"root", "leaf"}
    assert {r["population_kind"] for r in rows} == {"frozen_driver_shadow"}
    observer_rows = [
        json.loads(line)
        for line in (
            tmp_path / f"bank.observer.{readout.ARM_FASTQ}.w00.jsonl"
        ).read_text().splitlines()
    ]
    assert {r["role"] for r in observer_rows} == {"root", "leaf", "root-child"}

    # The resolution table exists and names every pair, search target included.
    labels = {shadow.SEARCH_LABEL, *cfg.arms}
    assert len(report["label_argmax_agreement"]) == len(labels) * (len(labels) - 1) // 2
    for value in report["label_argmax_agreement"].values():
        assert value["rows"] == float(report["rows_per_cell"])

    assert report["peak_rss_bytes_max"] > 0
