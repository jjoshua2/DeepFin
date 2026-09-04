from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.analyze_varying_budget_controller import (
    HORIZONS,
    PRICES,
    PRIMARY_PRICE,
    Snapshot,
    Trajectory,
    analyze,
    build_examples,
    cluster_bootstrap_delta,
    collection_status,
    feature_vector,
    grouped_folds,
    load_trajectories,
    policy_delta,
    rollout_policy,
    value_to_go,
)


def snap(chunk: int, regret: float, *, continue_: bool = True) -> Snapshot:
    return Snapshot(
        chunk=chunk,
        nodes=2048 * chunk,
        regret_score=regret,
        visit_gap=0.1,
        visit_entropy=0.5,
        q_gap=None if chunk == 1 else 0.02,
        root_q=0.1,
        bestmove_flip=False,
        stable_chunks=max(0, chunk - 1),
        q_drift=None if chunk == 1 else 0.01,
        visit_churn=None if chunk == 1 else 0.02,
        piece_count=20,
        legal_move_count=30,
        phase=1,
        source=0,
        complexity_continue=continue_,
    )


def trajectory(key: str, regrets: list[float], *, group: str | None = None, kind: str = "source_game", continues: list[bool] | None = None) -> Trajectory:
    flags = continues or [True] * len(regrets)
    return Trajectory(
        key=key,
        group_id=group or key,
        group_kind=kind,
        snapshots=tuple(snap(index + 1, regret, continue_=flags[index]) for index, regret in enumerate(regrets)),
    )


def test_value_to_go_changes_when_only_total_horizon_changes() -> None:
    snapshots = trajectory("p", [0.010, 0.012, 0.004, 0.004]).snapshots
    assert value_to_go(snapshots, 1, 2, 0.0001) < 0.0
    assert value_to_go(snapshots, 1, 3, 0.0001) > 0.0


def test_feature_ablation_isolates_budget_from_state_and_age() -> None:
    row = trajectory("p", [0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003])
    examples = build_examples([row], [4, 6], PRIMARY_PRICE)
    short = next(example for example in examples if example.horizon == 4 and example.chunk == 1)
    long = next(example for example in examples if example.horizon == 6 and example.chunk == 1)
    assert feature_vector(short, "M_state") == feature_vector(long, "M_state")
    assert feature_vector(short, "M_age") == feature_vector(long, "M_age")
    assert feature_vector(short, "M_budget") != feature_vector(long, "M_budget")


def test_targets_remain_signed_instead_of_clamping_regressions() -> None:
    row = trajectory("p", [0.01, 0.02, 0.03, 0.04])
    targets = [example.target for example in build_examples([row], [4], 0.001)]
    assert all(math.isfinite(value) for value in targets)
    assert any(value < 0.0 for value in targets)


def test_online_policy_has_no_bank_wide_quota_effect() -> None:
    a = trajectory("a", [0.02] * 8)
    b = trajectory("b", [0.02] * 8)
    a_predictions = {("a", 4, 1): 1.0, ("a", 4, 2): -1.0, ("a", 4, 3): 99.0}
    alone = rollout_policy([a], [4], 0.0, a_predictions)
    together = rollout_policy(
        [a, b],
        [4],
        0.0,
        {**a_predictions, ("b", 4, 1): 100.0, ("b", 4, 2): 100.0, ("b", 4, 3): 100.0},
    )
    assert alone[0].stop_chunk == 2
    assert next(row for row in together if row.key == "a").stop_chunk == 2


def test_stopping_is_absorbing() -> None:
    row = trajectory("p", [0.02, 0.015, 0.010, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = rollout_policy(
        [row],
        [4],
        0.0,
        {("p", 4, 1): 1.0, ("p", 4, 2): -1.0, ("p", 4, 3): 100.0},
    )
    assert result[0].stop_chunk == 2


def test_complexity_baseline_is_also_absorbing() -> None:
    row = trajectory("p", [0.02] * 8, continues=[True, False, True, True, True, True, True, True])
    assert rollout_policy([row], [4], 0.0, complexity=True)[0].stop_chunk == 2


def test_grouped_folds_never_split_a_source_game() -> None:
    groups = ["g1", "g1", "g2", "g3", "g3", "g3"]
    folds = grouped_folds(groups, 3)
    for group in set(groups):
        assert len({int(folds[index]) for index, value in enumerate(groups) if value == group}) == 1


def test_bootstrap_resamples_group_units() -> None:
    rows = [trajectory("a", [0.02, 0.01] + [0.01] * 6, group="g1"), trajectory("b", [0.02] * 8, group="g1"), trajectory("c", [0.02, 0.01] + [0.01] * 6, group="g2")]
    budget = rollout_policy(rows, [2], 0.0, {("a", 2, 1): 1.0, ("b", 2, 1): -1.0, ("c", 2, 1): 1.0})
    age = rollout_policy(rows, [2], 0.0, {("a", 2, 1): -1.0, ("b", 2, 1): -1.0, ("c", 2, 1): -1.0})
    result = cluster_bootstrap_delta(budget, age, 200, 1)
    assert result["groups"] == 2
    mean = result["mean"]
    assert mean is not None
    assert math.isclose(float(mean), 0.02 / 3.0)


def test_policy_delta_reports_each_horizon() -> None:
    row = trajectory("p", [0.03, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    budget = rollout_policy([row], [2, 4], 0.0, {("p", 2, 1): 1.0, ("p", 4, 1): 1.0, ("p", 4, 2): 1.0, ("p", 4, 3): 1.0})
    age = rollout_policy([row], [2, 4], 0.0, {("p", 2, 1): -1.0, ("p", 4, 1): -1.0, ("p", 4, 2): -1.0, ("p", 4, 3): -1.0})
    delta = policy_delta(budget, age)
    assert set(delta["by_horizon"]) == {"2", "4"}
    assert delta["by_horizon"]["4"] > delta["by_horizon"]["2"]


def forced_row(chunk: int) -> dict[str, object]:
    stable = chunk - 1
    return {
        "schema": "deepfin.varying_budget_trajectory.v1",
        "key": "forced",
        "group_id": "position:forced",
        "group_kind": "position",
        "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
        "phase": 0,
        "source": 0,
        "chunk": chunk,
        "nodes": 2048 * chunk,
        "reference_best_cp": 0.0,
        "chosen_reference_cp": 0.0,
        "regret_score": 0.0,
        "visit_gap": 1.0,
        "visit_entropy": 0.0,
        "q_gap": None,
        "root_q": 0.0,
        "bestmove_flip": False,
        "stable_chunks": stable,
        "q_drift": None if chunk == 1 else 0.0,
        "visit_churn": None if chunk == 1 else 0.0,
        "piece_count": 2,
        "legal_move_count": 1,
        "complexity_continue": stable < 2,
        "emitted_action": 10,
        "chosen_uci": "a1a2",
        "root_actions": [10],
        "root_visits": [0],
        "root_child_q": [0.0],
    }


def test_loader_requires_exact_fixed_node_chunks_and_recomputes_fields(tmp_path: Path) -> None:
    path = tmp_path / "bank.jsonl"
    rows = [forced_row(chunk) for chunk in range(1, 9)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert len(load_trajectories(path)) == 1
    nodes = rows[3]["nodes"]
    assert isinstance(nodes, int)
    rows[3]["nodes"] = nodes + 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="fixed-node"):
        load_trajectories(path)


def test_collection_status_requires_preregistered_search_shape() -> None:
    rows = [trajectory("p", [0.02] * 8, group="g")]
    manifest: dict[str, Any] = {
        "complete": True,
        "completed_positions": 1,
        "config": {
            "schema": "deepfin.varying_budget_trajectory.v1",
            "chunk_sims": 2048,
            "max_chunks": 8,
            "walkers": 2,
            "compile_mode": "max-autotune",
            "production_shape": True,
            "device": "cuda:0",
            "git_sha": "0" * 40,
        },
        "model": {"realized_search_path": "walker"},
    }
    assert collection_status(rows, manifest)["passed"] is True
    config = manifest["config"]
    assert isinstance(config, dict)
    config["walkers"] = 1
    assert collection_status(rows, manifest)["passed"] is False


def test_final_mode_enforces_bootstrap_floor_and_manifest() -> None:
    rows = [trajectory(f"p{index}", [0.02] * 8, group=f"g{index}") for index in range(4)]
    with pytest.raises(ValueError, match="1000"):
        analyze(rows, bootstrap_samples=999, mode="final")
    with pytest.raises(ValueError, match="collector manifest"):
        analyze(rows, bootstrap_samples=1000, mode="final")


def test_small_end_to_end_run_cannot_advance() -> None:
    rows = []
    for index in range(12):
        regrets = [0.02 - 0.001 * max(0, chunk - 2) if index % 2 == 0 else 0.02 + 0.0001 * chunk for chunk in range(1, 9)]
        rows.append(trajectory(f"p{index}", regrets, group=f"g{index}"))
    result = analyze(rows, horizons=HORIZONS, prices=PRICES, primary_price=PRIMARY_PRICE, n_folds=3, bootstrap_samples=20, mode="pilot")
    assert result["experiment"]["quota_ranking"] is False
    assert result["experiment"]["reentry"] is False
    assert result["preregistered_verdict"]["verdict"] == "INSUFFICIENT_PILOT_SAMPLE"


def collector_module():
    from scripts import collect_varying_budget_trajectories as collector

    return collector


def test_collector_imports_current_search_apis() -> None:
    collector = collector_module()
    from chess_anti_engine.uci.__main__ import _make_evaluator_factory
    from chess_anti_engine.uci.search import SearchWorker

    assert collector.CHUNK_SIMS == 2048
    assert collector.MAX_CHUNKS == 8
    assert collector.WALKERS == 2
    assert "compile_mode" in inspect.signature(_make_evaluator_factory).parameters
    assert "on_chunk" in inspect.signature(SearchWorker.run).parameters
    assert hasattr(SearchWorker, "realized_search_path")


def test_resume_repair_drops_partial_and_duplicate_groups(tmp_path: Path) -> None:
    collector = collector_module()
    path = tmp_path / "bank.jsonl"
    rows = [{"schema": collector.SCHEMA, "key": "complete", "chunk": chunk} for chunk in range(1, 9)]
    rows += [{"schema": collector.SCHEMA, "key": "partial", "chunk": chunk} for chunk in range(1, 4)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"schema":')
    assert collector.repair_resume_bank(path) == {"complete"}
    assert {json.loads(line)["key"] for line in path.read_text().splitlines()} == {"complete"}


def test_group_index_rejects_legacy_and_ambiguous_ids(tmp_path: Path) -> None:
    collector = collector_module()
    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(legacy, found=np.asarray([True]), key=np.asarray(["p"]), game_id=np.asarray([0]), snapshot=np.asarray(["snapshot"]))
    assert collector.GroupIndex(legacy).lookup("p") is None

    current = tmp_path / "current.npz"
    np.savez_compressed(
        current,
        found=np.asarray([True, True, True]),
        key=np.asarray(["ok", "missing", "ambiguous"]),
        game_id=np.asarray([7, 0, 9]),
        has_game_id=np.asarray([True, False, True]),
        source_cluster_ambiguous=np.asarray([False, False, True]),
        src_shard=np.asarray(["s.zarr", "s.zarr", "t.zarr"]),
        snapshot=np.asarray(["snapshot"]),
    )
    index = collector.GroupIndex(current)
    assert index.lookup("ok") == "\0".join(("snapshot", "s.zarr", "7"))
    assert index.lookup("missing") is None
    assert index.lookup("ambiguous") is None