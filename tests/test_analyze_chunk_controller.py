from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.moves import move_to_index
from scripts.analyze_chunk_controller import (
    Transition,
    _complexity_continue,
    _require_bootstrap_resolution,
    _require_safe_output_path,
    _update_stability,
    analyze,
    cluster_bootstrap_delta,
    evaluate_horizon,
    grouped_folds,
    held_horizon_predictions,
    load_transitions,
)
from scripts.backtest_chunk_trajectory import _require_search_take_effect


def _state(value: float = 0.0, *, gap: float = 0.1, stable: float = 0.0) -> dict[str, float]:
    return {
        "visit_gap": gap,
        "visit_entropy": 0.5,
        "q_gap": 0.0,
        "q_gap_missing": 0.0,
        "bestmove_flip": 0.0,
        "stable_chunks": stable,
        "q_drift": value,
        "visit_churn": 0.1,
        "root_q": value,
        "phase": 1.0,
        "piece_count": 20.0,
        "legal_move_count": 30.0,
        "nodes": 50.0,
    }


def _transition(
    key: str,
    game: int,
    horizon: int,
    gain: float,
    *,
    current: bool,
    value: float = 0.0,
) -> Transition:
    return Transition(
        key=key,
        group_id=f"selfplay:{game}",
        horizon=horizon,
        cost=50,
        gain=gain,
        regret_before=max(gain, 0.0) + 1.0,
        regret_after=max(gain, 0.0) + 1.0 - gain,
        complexity_continue=current,
        state=_state(value),
    )


def test_evaluation_preserves_signed_corrections_and_regressions() -> None:
    rows = [
        _transition("a", 1, 100, 1.0, current=True),
        _transition("b", 2, 100, -2.0, current=True),
        _transition("c", 3, 100, 0.5, current=False),
        _transition("d", 4, 100, 0.0, current=False),
    ]
    result = evaluate_horizon(
        rows,
        np.asarray([4.0, 3.0, 2.0, 1.0]),
        np.asarray([4.0, 1.0, 3.0, 2.0]),
    )

    assert result["corrections"] == 2
    assert result["regressions"] == 1
    assert result["policies"]["complexity_predicate"]["signed_gain"] == pytest.approx(-1.0)
    assert result["policies"]["random"]["signed_gain"] == pytest.approx(-0.25)
    assert result["policies"]["oracle"]["signed_gain"] == pytest.approx(1.5)
    assert {policy["spend"] for policy in result["policies"].values()} == {100}


def test_complexity_predicate_selects_boolean_continues_at_natural_spend() -> None:
    continuing = _transition("hard", 1, 100, 2.0, current=True)
    stopping = _transition("easy", 2, 100, -3.0, current=False)
    continuing_state = dict(continuing.state)
    continuing_state["stable_chunks"] = 20.0
    stopping_state = dict(stopping.state)
    stopping_state["stable_chunks"] = 2.0
    rows = [replace(continuing, state=continuing_state), replace(stopping, state=stopping_state)]

    result = evaluate_horizon(rows, np.zeros(2), np.zeros(2))

    assert result["policies"]["complexity_predicate"]["signed_gain"] == pytest.approx(2.0)


def test_held_horizon_cv_excludes_horizon_and_source_game() -> None:
    rows = [
        _transition(f"{game}-{horizon}", game, horizon, game / 10, current=game % 2 == 0)
        for game in range(8)
        for horizon in (100, 150, 200)
    ]
    predictions, diagnostics = held_horizon_predictions(rows, "M0", n_folds=4)

    assert np.isfinite(predictions).all()
    assert diagnostics
    for fold in diagnostics:
        assert fold["horizon"] not in fold["train_horizons"]
        assert set(fold["test_groups"]).isdisjoint(fold["train_groups"])


def test_grouped_folds_never_split_a_game() -> None:
    groups = ["s:10", "s:10", "s:11", "s:12", "s:12", "s:12", "s:13", "s:14"]
    folds = grouped_folds(groups, 3)
    memberships: dict[str, int] = {}
    for fold_number, indices in enumerate(folds):
        for index in indices:
            group = groups[int(index)]
            memberships.setdefault(group, fold_number)
            assert memberships[group] == fold_number


def test_source_scoped_groups_do_not_collide() -> None:
    shard_a = "\0".join(("/same", "a.zarr", "7"))
    shard_b = "\0".join(("/same", "b.zarr", "7"))
    groups = [shard_a, shard_a, shard_b, shard_b]
    folds = grouped_folds(groups, 2)
    membership = {
        groups[int(index)]: fold_number
        for fold_number, fold in enumerate(folds)
        for index in fold
    }
    assert membership[shard_a] != membership[shard_b]


def test_trajectory_producer_uses_production_evaluator_stack_and_readback() -> None:
    from scripts import backtest_chunk_trajectory as producer

    source = inspect.getsource(producer.main)
    assert "LocalModelEvaluator" not in source
    assert "DirectGPUEvaluator" in source
    assert "ThreadSafeGPUDispatcher" in source
    assert "BatchCoalescingDispatcher" in source
    assert "SyzygyProbe" in source
    assert "mcts_extension" in source
    assert "_make_evaluator_factory" in source
    assert "compile_mode=compile_mode" in source
    assert "realized_search_values" in source
    assert "_require_search_take_effect" in source
    assert "if int(args.walkers) == 1:" in source
    assert "worker.set_minibatch_size" in source
    assert "allow_terminal_shortcuts=True" in source
    assert '"root_child_q"' in source
    assert '"pv_actions"' in source
    assert '"checkpoint_params"' in source


def test_search_take_effect_rejects_an_inert_active_parameter() -> None:
    realized: dict[str, float | int | bool | str] = {
        "concurrency_mode": "walker_puct",
        "concurrency_workers": 2,
        "c_puct": 1.75,
    }
    _require_search_take_effect(
        expected_mode="walker_puct",
        expected_workers=2,
        active_parameters={"c_puct": 1.75},
        realized=realized,
    )
    with pytest.raises(RuntimeError, match=r"c_puct requested=99\.0 realized=1\.75"):
        _require_search_take_effect(
            expected_mode="walker_puct",
            expected_workers=2,
            active_parameters={"c_puct": 99.0},
            realized=realized,
        )


def test_stability_streak_resets_while_gumbel_survivor_trails_visits() -> None:
    last, stable = _update_stability(
        -1, 0, emitted_action=11, visit_gap=0.2, action_count=3,
    )
    assert (last, stable) == (11, 0)
    last, stable = _update_stability(
        last, stable, emitted_action=11, visit_gap=-0.1, action_count=3,
    )
    assert (last, stable) == (11, 0)
    last, stable = _update_stability(
        last, stable, emitted_action=11, visit_gap=0.3, action_count=3,
    )
    assert (last, stable) == (11, 1)


def test_stable_single_legal_move_is_decided_without_root_visits() -> None:
    assert not _complexity_continue(
        stable_chunks=2, visit_gap=0.0, action_count=1,
    )
    assert _complexity_continue(
        stable_chunks=1, visit_gap=0.0, action_count=1,
    )


def test_budget_interactions_improve_held_horizon_prediction() -> None:
    rows: list[Transition] = []
    for game in range(20):
        value = game / 19.0
        for horizon in (100, 150, 200):
            remaining_fraction = 50.0 / horizon
            gain = (value - 0.5) * (remaining_fraction - 0.3)
            row = _transition(
                f"{game}-{horizon}", game, horizon, gain,
                current=game % 2 == 0, value=value,
            )
            state = dict(row.state)
            state["nodes"] = float(horizon - 50)
            rows.append(replace(row, state=state))

    m0, _ = held_horizon_predictions(rows, "M0", n_folds=5)
    m1, _ = held_horizon_predictions(rows, "M1", n_folds=5)
    truth = np.asarray([row.gain for row in rows])

    assert np.mean((m1 - truth) ** 2) < np.mean((m0 - truth) ** 2) * 0.5


def test_cluster_bootstrap_is_deterministic() -> None:
    rows = [
        _transition(f"{game}-{horizon}", game, horizon, (game - 2) / horizon,
                    current=game % 2 == 0)
        for game in range(6)
        for horizon in (100, 150)
    ]
    first = cluster_bootstrap_delta(
        rows,
        allocation_fraction=0.5, samples=20, seed=7, n_folds=3,
    )
    second = cluster_bootstrap_delta(
        rows,
        allocation_fraction=0.5, samples=20, seed=7, n_folds=3,
    )
    assert first == second


def _write_bank(path: Path, *, correct_gap: bool) -> Path:
    board = chess.Board()
    actions = [
        int(move_to_index(chess.Move.from_uci(uci), board))
        for uci in ("a2a3", "b2b3")
    ]
    entropy = float(-(0.4 * np.log(0.4) + 0.6 * np.log(0.6)))
    rows = []
    for chunk, regret, regret_cp in ((1, 0.2, 20.0), (2, 0.1, 10.0)):
        rows.append({
            "schema": "deepfin.chunk_trajectory.v2",
            "key": "k", "source_dir": "/snapshot", "shard": "s0.zarr",
            "fen": board.fen(),
            "game_id": 3, "group_id": "/snapshot\0s0.zarr\0" + "3",
            "chunk": chunk, "nodes": chunk * 50,
            "elapsed_ms": float(chunk), "regret_cp": regret_cp,
            "regret_score": regret, "regret_vs_final_cp": regret_cp - 10.0,
            "visit_gap": -0.2 if correct_gap else 0.2,
            "root_actions": actions, "root_visits": [20, 30],
            "root_visit_shares": [0.4, 0.6],
            "root_child_q": [0.1, 0.2],
            "emitted_action": actions[0], "uci": "a2a3", "bestmove_flip": False,
            "pv_actions": [actions[0]], "pv_uci": ["a2a3"],
            "stable_chunks": 0, "visit_entropy": entropy, "q_gap": -0.1,
            "complexity_predicate_continue": True,
            "q_drift": None if chunk == 1 else 0.0,
            "visit_churn": None if chunk == 1 else 0.0, "root_q": 0.0,
            "changes_to_final": False,
            "phase": 1, "piece_count": 32, "legal_move_count": 20,
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = Path(str(path) + ".meta.json")
    meta.write_text(json.dumps({
        "schema": "deepfin.chunk_trajectory.v2",
        "complete": True,
        "decision_grade": True,
        "analysis_scope": "fixed_node_horizons_only",
        "clock_conditioning_available": False,
        "root_position_history": "fen_only_from_audit_fen",
        "game_group_kind": "source_dir:shard:game_id",
        "complexity_predicate": {
            "kind": "clock_free_visit_gap_and_stability",
            "minimum_stable_chunks": 2,
            "minimum_visit_gap": 0.25,
            "single_legal_move_is_decided": True,
        },
        "producer_git_sha": "a" * 40,
        "producer_git_dirty": False,
        "producer_script": {
            "path": "/producer.py", "size": 1, "mtime_ns": 1, "sha256": "a" * 64,
        },
        "checkpoint": {
            "path": "/trainer.pt", "size": 1, "mtime_ns": 1, "sha256": "b" * 64,
        },
        "checkpoint_params": None,
        "audit_set": {
            "path": "/audit.jsonl", "size": 1, "mtime_ns": 1, "sha256": "c" * 64,
        },
        "matched_rows": {
            "path": "/matched.npz", "size": 1, "mtime_ns": 1, "sha256": "d" * 64,
        },
        "mcts_extension": {
            "path": "/_mcts_tree.so", "size": 1, "mtime_ns": 1,
            "sha256": "e" * 64, "abi_version": 9, "required_abi_version": 9,
            "gss_halving_rev": 3,
        },
        "syzygy": {
            "path": "/tb/a:/tb/b",
            "rtbw_count": 875,
            "rtbz_count": 510,
            "directories": [
                {"path": "/tb/a", "rtbw_count": 510, "rtbz_count": 145,
                 "total_bytes": 1, "inventory_sha256": "f" * 64},
                {"path": "/tb/b", "rtbw_count": 365, "rtbz_count": 365,
                 "total_bytes": 1, "inventory_sha256": "1" * 64},
            ],
        },
        "row_count": 2,
        "chunk_count": 2,
        "position_count": 1,
        "requested_position_count": 1,
        "excluded_position_count": 0,
        "excluded_positions": [],
        "requested_search": {
            "device": "cuda", "active_path": "walker_puct",
            "walkers": 2, "chunk_sims": 50,
            "active_parameters": {
                "c_puct": 1.75, "cpuct_factor": 3.89, "cpuct_base": 38739.0,
                "fpu_reduction": 0.33, "vloss_weight": 3,
                "walker_gather": 1, "policy_temp": 1.0,
            },
        },
        "requested_evaluator": {
            "stack": (
                "BatchCoalescingDispatcher("
                "ThreadSafeGPUDispatcher(DirectGPUEvaluator))"
            ),
            "direct_max_batch": 256, "outer_max_batch": 256, "n_slots": 2,
            "input_bf16": False, "legal_bf16": False,
            "compiled": True, "model_wrapper_type": "OptimizedModule",
        },
        "realized_evaluator": {
            "stack": (
                "BatchCoalescingDispatcher("
                "ThreadSafeGPUDispatcher(DirectGPUEvaluator))"
            ),
            "direct_max_batch": 256, "outer_max_batch": 256, "n_slots": 2,
            "input_bf16": False, "legal_bf16": False,
            "compiled": True, "model_wrapper_type": "OptimizedModule",
        },
        "compile": {
            "enabled": True, "mode": "max-autotune", "cache_dir": "/cache",
            "torchinductor_cache_dir": "/cache/torch", "triton_cache_dir": "/cache/triton",
        },
        "realized_search": {
            "concurrency_mode": "walker_puct", "concurrency_workers": 2,
            "chunk_sims": 50, "c_puct": 1.75, "cpuct_factor": 3.89,
            "cpuct_base": 38739.0, "fpu_reduction": 0.33,
            "vloss_weight": 3, "walker_gather": 1, "policy_temp": 1.0,
        },
        "realized_tablebase": {
            "installed": True, "cursed_as_draw": True,
            "n_wdl": 510, "n_dtz": 510, "max_pieces": 6,
            "root_probe_active": True, "leaf_probe_active": False,
        },
        "output": {"sha256": digest, "size": path.stat().st_size},
    }))
    return meta


def test_loader_refuses_leader_gap_for_nonleading_emitted_action(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=False)

    with pytest.raises(ValueError, match="emitted action's gap"):
        load_transitions(bank)


def test_loader_requires_provenance_but_allows_explicit_legacy_smoke(tmp_path: Path) -> None:
    bank = tmp_path / "legacy.jsonl"
    bank.write_text("".join([
        json.dumps({
            "key": "k", "chunk": 1, "nodes": 50, "regret_cp": 20,
            "visit_gap": 0.1, "uci": "a2a3",
        }) + "\n",
        json.dumps({
            "key": "k", "chunk": 2, "nodes": 100, "regret_cp": 10,
            "visit_gap": 0.2, "uci": "a2a3",
        }) + "\n",
    ]))

    with pytest.raises(ValueError, match="decision-grade analysis requires"):
        load_transitions(bank)
    transitions, info = load_transitions(bank, methodology_smoke=True)
    assert transitions[0].gain == 10.0
    assert info["decision_grade"] is False
    assert info["metric"] == "regret_cp"


def test_loader_accepts_verified_emitted_action_gap(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)
    transitions, info = load_transitions(bank)
    assert transitions[0].state["visit_gap"] == pytest.approx(-0.2)
    assert info["decision_grade"] is True


def test_loader_rejects_dirty_or_incomplete_producer_provenance(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["producer_git_dirty"] = True
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="producer checkout was dirty"):
        load_transitions(bank)


def test_loader_rejects_requested_parameter_that_did_not_take_effect(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["realized_search"]["c_puct"] = 2.5
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="realized active parameters"):
        load_transitions(bank)


def test_loader_rejects_unrealized_walker_gather(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["requested_search"]["active_parameters"]["walker_gather"] = 999
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="realized active parameters"):
        load_transitions(bank)


def test_loader_requires_realized_syzygy_and_native_binary(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["realized_tablebase"]["installed"] = False
    manifest["mcts_extension"]["sha256"] = "not-a-hash"
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=r"MCTS.*Syzygy|Syzygy.*MCTS"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value"),
    [("root_probe_active", False), ("leaf_probe_active", True)],
)
def test_loader_records_actual_walker_tablebase_semantics(
    tmp_path: Path, field: str, value: bool,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["realized_tablebase"][field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="production Syzygy probe"):
        load_transitions(bank)


def test_loader_rejects_partial_syzygy_inventory(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["syzygy"]["directories"][0]["rtbw_count"] = 500
    manifest["syzygy"]["directories"][1]["rtbw_count"] = 375
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="production Syzygy provenance"):
        load_transitions(bank)


def test_loader_requires_native_halving_semantic_revision(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["mcts_extension"].pop("gss_halving_rev")
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="native MCTS extension"):
        load_transitions(bank)


def test_loader_requires_resolved_checkpoint_params_provenance(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest.pop("checkpoint_params")
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="checkpoint architecture provenance"):
        load_transitions(bank)


def test_loader_validates_the_final_rows_full_cluster_identity(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows[-1]["source_dir"] = "/other"
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="group_id is not"):
        load_transitions(bank)


def test_loader_validates_final_row_stability_semantics(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows[-1]["stable_chunks"] = 1
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="stable_chunks disagrees"):
        load_transitions(bank)


def test_loader_requires_manifest_position_count_unique_trajectories(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows.extend(json.loads(json.dumps(row)) for row in rows[:])
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest.update({"row_count": 4, "position_count": 2, "requested_position_count": 2})
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unique trajectory count"):
        load_transitions(bank)


def test_loader_requires_every_trajectory_to_have_all_manifest_chunks(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    original = [json.loads(line) for line in bank.read_text().splitlines()]
    rows = [original[0]]
    for chunk, template in enumerate((original[0], original[1], original[1]), start=1):
        row = json.loads(json.dumps(template))
        row.update({
            "key": "other", "game_id": 4,
            "group_id": "\0".join(("/snapshot", "s0.zarr", "4")),
            "chunk": chunk, "nodes": chunk * 50, "elapsed_ms": float(chunk),
        })
        rows.append(row)
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest.update({"row_count": 4, "position_count": 2, "requested_position_count": 2})
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="complete consecutive manifest horizon"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("root_child_q", [0.1, float("nan")], "root child Q"),
        ("pv_actions", [0], "PV provenance"),
        ("pv_uci", ["b2b3"], "PV provenance"),
    ],
)
def test_loader_requires_reusable_root_q_and_pv_observations(
    tmp_path: Path, field: str, value: object, match: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows[-1][field] = value
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=match):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("row_index", "field", "value", "match"),
    [
        (0, "visit_entropy", 0.1, "visit_entropy disagrees"),
        (0, "q_gap", 0.0, "q_gap disagrees"),
        (1, "bestmove_flip", True, "bestmove_flip disagrees"),
        (1, "q_drift", 0.1, "q_drift disagrees"),
        (1, "visit_churn", 0.1, "visit_churn disagrees"),
        (0, "changes_to_final", True, "changes_to_final disagrees"),
        (0, "regret_vs_final_cp", 99.0, "regret_vs_final_cp disagrees"),
        (0, "root_visits", [10, 40], "shares disagree"),
    ],
)
def test_loader_recomputes_derived_trajectory_fields(
    tmp_path: Path, row_index: int, field: str, value: object, match: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows[row_index][field] = value
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=match):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("regret_score", "bad"),
        ("regret_score", float("nan")),
        ("regret_score", float("inf")),
        ("regret_score", float("-inf")),
        ("root_q", "bad"),
        ("visit_entropy", float("nan")),
        ("q_gap", float("nan")),
        ("q_drift", float("inf")),
        ("visit_churn", float("-inf")),
    ],
)
def test_loader_rejects_nonfinite_or_nonnumeric_decision_fields(
    tmp_path: Path, field: str, value: object,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    target = rows[-1] if field == "regret_score" else rows[0]
    target[field] = value
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=rf"{field} must be a finite number"):
        load_transitions(bank)


def test_output_path_cannot_replace_the_bank_or_manifest(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    _require_safe_output_path(bank, meta, tmp_path / "report.json")
    with pytest.raises(ValueError, match="must not overwrite"):
        _require_safe_output_path(bank, meta, bank)
    with pytest.raises(ValueError, match="must not overwrite"):
        _require_safe_output_path(bank, meta, meta)


def test_decision_grade_requires_enough_bootstrap_samples() -> None:
    _require_bootstrap_resolution(1, methodology_smoke=True)
    _require_bootstrap_resolution(1000, methodology_smoke=False)
    with pytest.raises(ValueError, match="at least 1000"):
        _require_bootstrap_resolution(999, methodology_smoke=False)


def test_analyze_cannot_advance_with_an_undersampled_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import analyze_chunk_controller as controller

    rows = [
        _transition(f"a-{horizon}", 1, horizon, -1.0, current=False)
        for horizon in (100, 150)
    ] + [
        _transition(f"b-{horizon}", 2, horizon, 1.0, current=True)
        for horizon in (100, 150)
    ]
    m0 = np.zeros(len(rows), dtype=np.float64)
    m1 = np.asarray([row.gain for row in rows], dtype=np.float64)

    def fake_predictions(
        transitions: list[Transition], model: str, *, n_folds: int = 5,
    ) -> tuple[np.ndarray, list[dict[str, object]]]:
        del transitions, n_folds
        return (m0 if model == "M0" else m1), []

    monkeypatch.setattr(controller, "held_horizon_predictions", fake_predictions)
    monkeypatch.setattr(
        controller,
        "cluster_bootstrap_delta",
        lambda *_args, **_kwargs: {
            "requested_samples": 1, "valid_samples": 1, "valid_fraction": 1.0,
            "mean": 1.0, "lower_95": 1.0, "upper_95": 1.0,
        },
    )

    result = analyze(
        rows,
        n_folds=2,
        bootstrap_samples=1,
        seed=0,
        allocation_fraction=0.5,
        min_capture_gain=0.0,
        min_oracle_headroom=0.0,
        min_bootstrap_valid_fraction=0.5,
    )

    assert result["bootstrap_resolution_passed"] is False
    assert result["verdict"] == "KILL_BUDGET_CONTEXT"


def test_walker_manifest_omits_gumbel_only_minibatch_setting(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())

    assert "minibatch_size" not in manifest["requested_search"]["active_parameters"]
    assert "walker_gather" in manifest["requested_search"]["active_parameters"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stack", "ThreadSafeGPUDispatcher(DirectGPUEvaluator)"),
        ("direct_max_batch", 999),
        ("input_bf16", True),
    ],
)
def test_requested_only_evaluator_changes_fail_closed(
    tmp_path: Path, field: str, value: object,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["requested_evaluator"][field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="realized evaluator"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value"),
    [("enabled", False), ("mode", "reduce-overhead"), ("cache_dir", "")],
)
def test_loader_requires_production_compile_manifest(
    tmp_path: Path, field: str, value: object,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["compile"][field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="realized evaluator"):
        load_transitions(bank)


def test_cluster_bootstrap_refits_models_inside_resamples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import analyze_chunk_controller as controller

    rows = [
        _transition(
            f"{game}-{horizon}", game, horizon, (game - 5) / horizon,
            current=game % 2 == 0,
        )
        for game in range(12)
        for horizon in (100, 150)
    ]
    original = controller._fit_ridge
    calls = 0
    alpha_calls = 0

    def counted_fit(x: np.ndarray, y: np.ndarray, alpha: float):
        nonlocal calls
        calls += 1
        return original(x, y, alpha)

    original_inner_alpha = controller._inner_alpha

    def counted_inner_alpha(
        transitions: list[Transition], model: str, n_folds: int,
    ) -> float:
        nonlocal alpha_calls
        alpha_calls += 1
        return original_inner_alpha(transitions, model, n_folds)

    monkeypatch.setattr(controller, "_fit_ridge", counted_fit)
    monkeypatch.setattr(controller, "_inner_alpha", counted_inner_alpha)
    controller.cluster_bootstrap_delta(
        rows,
        allocation_fraction=0.5,
        samples=5,
        seed=11,
        n_folds=3,
    )

    assert calls >= 4, "each usable replicate must refit both models"
    assert alpha_calls >= 4, "each usable replicate must reselect alpha in-bag"


def test_bootstrap_fails_closed_when_oracle_headroom_is_undefined() -> None:
    rows = [
        _transition(f"{game}-{horizon}", game, horizon, 0.0, current=True)
        for game in range(6)
        for horizon in (100, 150)
    ]
    result = cluster_bootstrap_delta(
        rows,
        allocation_fraction=0.5,
        samples=25,
        seed=3,
        n_folds=3,
        min_oracle_headroom=1e-4,
    )

    assert result["requested_samples"] == 25
    assert result["valid_samples"] == 0
    assert result["valid_fraction"] == 0.0
    assert result["lower_95"] is None
