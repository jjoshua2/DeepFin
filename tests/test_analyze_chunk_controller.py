from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.eval.audit import legal_full_indices, phase_bucket, position_key
from chess_anti_engine.moves import move_to_index
from scripts import analyze_chunk_controller as controller_module
from scripts.analyze_chunk_controller import (
    Transition,
    _complexity_continue,
    _evaluation_fold_ids,
    _fold_stage_counts,
    _rollout_selected_indices,
    _source_group,
    _stage_counts,
    _require_bootstrap_resolution,
    _require_safe_output_path,
    _score,
    _update_stability,
    analyze,
    cluster_bootstrap_delta,
    evaluate_horizon,
    evaluate_reachable_rollout,
    grouped_folds,
    held_horizon_predictions,
    load_transitions,
)
from scripts.backtest_chunk_trajectory import (
    _acquire_output_lock,
    _acquire_output_locks,
    _publish_output,
    _require_new_output_pair,
    _require_safe_output_paths,
    _require_search_take_effect,
    _validate_registry_search_values,
)


_TEST_GIT_FILES: dict[str, bytes] = {}


@pytest.fixture(autouse=True)
def _authenticate_test_preregistration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        controller_module,
        "_git_file_at_commit",
        lambda _commit, relative_path: _TEST_GIT_FILES.get(relative_path),
    )
    monkeypatch.setattr(
        controller_module,
        "_preregistration_is_only_source_change",
        lambda _source, _producer, _relative_path: True,
    )


def _state(value: float = 0.0, *, gap: float = 0.1, stable: float = 0.0) -> dict[str, float]:
    return {
        "visit_gap": gap,
        "visit_entropy": 0.5,
        "q_gap": 0.0,
        "q_gap_missing": 0.0,
        "bestmove_flip": 0.0,
        "stable_chunks": stable,
        "q_drift": value,
        "q_drift_missing": 0.0,
        "visit_churn": 0.1,
        "visit_churn_missing": 0.0,
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
        hard_horizon=256,
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


def test_reachable_rollout_never_allows_a_stopped_position_to_reenter() -> None:
    rows = [
        _transition(key, game, horizon, 100.0 if key == "d" and horizon > 100 else 0.0,
                    current=True)
        for game, key in enumerate(("a", "b", "c", "d"))
        for horizon in (100, 150, 200)
    ]
    scores = np.asarray([
        (-100.0 if row.key == "d" and row.horizon == 100 else 100.0)
        if row.key == "d" else float(-ord(row.key[0]))
        for row in rows
    ])
    counts = _stage_counts(4, 3, 0.5)
    assert counts == [3, 2, 1]
    selected = _rollout_selected_indices(rows, scores, counts)
    assert not any(rows[int(index)].key == "d" for index in selected)

    result = evaluate_reachable_rollout(
        rows, scores, scores, allocation_fraction=0.5,
    )
    assert result["selection_semantics"] == "nested_prefix_no_reentry"
    assert result["policies"]["M1"]["signed_gain"] == 0.0
    assert result["relaxed_oracle_signed_gain"] == 200.0


def test_reachable_rollout_uses_exact_nested_oracle_not_relaxed_bound() -> None:
    rows = [
        _transition("early", 1, 100, 100.0, current=True),
        _transition("early", 1, 150, 0.0, current=True),
        _transition("late", 2, 100, 0.0, current=True),
        _transition("late", 2, 150, 100.0, current=True),
    ]

    result = evaluate_reachable_rollout(
        rows, np.zeros(4), np.zeros(4), allocation_fraction=0.5,
    )

    assert result["stage_continue_counts"] == [1, 1]
    assert result["oracle_semantics"] == "exact_nested_stop_depth_assignment"
    assert result["reachable_oracle_signed_gain"] == 100.0
    assert result["relaxed_oracle_signed_gain"] == 200.0
    assert result["relaxation_gap"] == 100.0
    assert result["policies"]["oracle"]["signed_gain"] == 100.0


def test_reachable_gate_requires_oracle_headroom_at_every_rung() -> None:
    from scripts.analyze_chunk_controller import _minimum_reachable_rung_gain_delta

    rows = [
        _transition("early", 1, 100, 100.0, current=True),
        _transition("early", 1, 150, -100.0, current=True),
        _transition("late", 2, 100, 0.0, current=True),
        _transition("late", 2, 150, 100.0, current=True),
    ]
    scores = np.zeros(len(rows), dtype=np.float64)

    result = evaluate_reachable_rollout(
        rows, scores, scores, allocation_fraction=0.5,
    )

    assert result["oracle_over_random_headroom_mean"] == 25.0
    assert result["reachable_stage_diagnostics"][0][
        "oracle_over_random_headroom_mean"
    ] == -25.0
    assert _minimum_reachable_rung_gain_delta(
        rows, scores, scores, 0.5, min_oracle_headroom=1.0,
    ) is None


def test_fold_local_rollout_is_invariant_to_cross_fit_score_offsets() -> None:
    rows = [
        _transition(key, game, horizon, gain, current=True)
        for key, game, gain in (
            ("a", 1, 0.0),
            ("b", 2, 0.0),
            ("c", 3, 10.0),
            ("d", 4, -10.0),
        )
        for horizon in (100, 150)
    ]
    fold_ids = np.asarray([
        0 if row.key in {"a", "c"} else 1 for row in rows
    ])
    scores = np.asarray([
        {"a": 10.0, "b": 0.0, "c": 9.0, "d": -1.0}[row.key]
        for row in rows
    ])
    shifted = scores + np.where(fold_ids == 1, 1000.0, 0.0)

    pooled = evaluate_reachable_rollout(
        rows, scores, scores, allocation_fraction=0.5,
    )
    shifted_pooled = evaluate_reachable_rollout(
        rows, shifted, shifted, allocation_fraction=0.5,
    )
    assert pooled["policies"]["M1"]["signed_gain"] != (
        shifted_pooled["policies"]["M1"]["signed_gain"]
    )

    local = evaluate_reachable_rollout(
        rows, scores, scores, allocation_fraction=0.5, fold_ids=fold_ids,
    )
    shifted_local = evaluate_reachable_rollout(
        rows, shifted, shifted, allocation_fraction=0.5, fold_ids=fold_ids,
    )
    assert local["selection_semantics"] == "fold_local_nested_prefix_no_reentry"
    assert local["policies"] == shifted_local["policies"]
    assert local["reachable_stage_diagnostics"] == (
        shifted_local["reachable_stage_diagnostics"]
    )


def test_fold_local_quotas_are_nested_and_preserve_exact_global_spend() -> None:
    global_counts, fold_counts = _fold_stage_counts([5, 3, 2], 4, 0.35)

    assert global_counts == _stage_counts(10, 4, 0.35)
    assert [sum(counts[stage] for counts in fold_counts) for stage in range(4)] == (
        global_counts
    )
    assert all(counts == sorted(counts, reverse=True) for counts in fold_counts)


def test_held_horizon_cv_excludes_horizon_and_source_game() -> None:
    rows = [
        _transition(str(game), game, horizon, game / 10, current=game % 2 == 0)
        for game in range(8)
        for horizon in (100, 150, 200)
    ]
    predictions, diagnostics = held_horizon_predictions(rows, "M0", n_folds=4)

    assert np.isfinite(predictions).all()
    assert diagnostics
    for fold in diagnostics:
        assert fold["horizon"] not in fold["train_horizons"]
        assert set(fold["test_groups"]).isdisjoint(fold["train_groups"])
    fold_ids = _evaluation_fold_ids(rows, 4)
    assert all(
        len({int(fold_ids[index]) for index, row in enumerate(rows) if row.key == key}) == 1
        for key in {row.key for row in rows}
    )


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
    source_a = "\0".join(("/first", "7"))
    source_b = "\0".join(("/second", "7"))
    groups = [source_a, source_a, source_b, source_b]
    folds = grouped_folds(groups, 2)
    membership = {
        groups[int(index)]: fold_number
        for fold_number, fold in enumerate(folds)
        for index in fold
    }
    assert membership[source_a] != membership[source_b]


def test_source_game_group_unifies_one_game_split_across_shards() -> None:
    first = {"source_dir": "/snapshot", "shard": "a.zarr", "game_id": 7}
    second = {"source_dir": "/snapshot", "shard": "b.zarr", "game_id": 7}

    assert _source_group(first) == _source_group(second) == "/snapshot\0" + "7"


def test_trajectory_producer_uses_production_evaluator_stack_and_readback() -> None:
    from scripts import backtest_chunk_trajectory as producer

    source = inspect.getsource(producer._main)
    module_source = inspect.getsource(producer)
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
    assert "args.walkers != _PRODUCTION_WALKERS" in source
    assert "worker.set_minibatch_size" in source
    assert "allow_terminal_shortcuts=True" in source
    assert '"root_child_q"' in source
    assert '"pv_actions"' in source
    assert '"checkpoint_params"' in source
    assert source.index("initial_input_artifacts =") < source.index("load_audit_set(")
    assert source.index("initial_input_artifacts =") < source.index("MatchedAuditRows(")
    assert source.index("output_locks = _acquire_output_locks") < source.index(
        "model = load_model_from_checkpoint("
    )
    assert "load_model_from_checkpoint(\n        checkpoint_path," in source
    assert "tmp_path.unlink()" not in source
    assert '"raw_observations_preserved": True' in source
    assert module_source.index(
        'if __name__ == "__main__" and "--recover-publication" in sys.argv[1:]:'
    ) < module_source.index(
        "from scripts.native_import_guard import PREIMPORT_NATIVE_ARTIFACTS"
    )
    assert module_source.index(
        "from scripts.native_import_guard import PREIMPORT_NATIVE_ARTIFACTS"
    ) < module_source.index("from chess_anti_engine.eval.audit import")


def test_trajectory_preimport_guard_covers_every_loaded_project_extension() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    program = "\n".join((
        "import json, sys",
        "from pathlib import Path",
        "import scripts.backtest_chunk_trajectory",
        "from scripts.native_import_guard import EARLY_NATIVE_MODULES",
        "loaded = sorted(name for name, module in sys.modules.items() "
        "if name.startswith('chess_anti_engine.') "
        "and isinstance(getattr(module, '__file__', None), str) "
        "and Path(module.__file__).suffix in ('.so', '.pyd'))",
        "print(json.dumps({'guarded': sorted(EARLY_NATIVE_MODULES), 'loaded': loaded}))",
    ))

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    inventory = json.loads(completed.stdout)

    assert inventory["loaded"] == inventory["guarded"]
    assert "chess_anti_engine.encoding._features_ext" in inventory["loaded"]


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


def test_expected_score_is_stable_for_forced_mate_scale_cp() -> None:
    assert _score(-100_000.0) >= 0.0
    assert _score(100_000.0) <= 1.0
    assert _score(-100_000.0) < _score(0.0) < _score(100_000.0)


def test_budget_interactions_improve_held_horizon_prediction() -> None:
    rows: list[Transition] = []
    for game in range(20):
        value = game / 19.0
        for horizon in (100, 150, 200):
            remaining_fraction = 50.0 / horizon
            gain = (value - 0.5) * (remaining_fraction - 0.3)
            row = _transition(
                str(game), game, horizon, gain,
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
        _transition(str(game), game, horizon, (game - 2) / horizon,
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
    assert first["selection_semantics"] == "fold_local_nested_prefix_no_reentry"
    point_folds = _evaluation_fold_ids(rows, 3)
    point = evaluate_reachable_rollout(
        rows, np.zeros(len(rows)), np.zeros(len(rows)),
        allocation_fraction=0.5, fold_ids=point_folds,
    )
    assert first["selection_semantics"] == point["selection_semantics"]
    assert first["oracle_semantics"] == point["oracle_semantics"]


def test_bootstrap_refits_exclude_the_target_fold_and_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _transition(str(game), game, horizon, game / horizon, current=True)
        for game in range(9)
        for horizon in (100, 150, 200)
    ]
    fold_ids = _evaluation_fold_ids(rows, 3)
    group_fold = {
        row.group_id: int(fold_ids[index]) for index, row in enumerate(rows)
    }
    seen = 0

    def checked_alpha(
        train_rows: list[Transition], model: str, n_folds: int,
    ) -> float:
        nonlocal seen
        del model, n_folds
        seen += 1
        assert len({row.horizon for row in train_rows}) == 2
        assert len({group_fold[row.group_id] for row in train_rows}) == 2
        return 1.0

    monkeypatch.setattr(controller_module, "_inner_alpha", checked_alpha)
    predictions = controller_module._refit_fold_predictions(
        rows, fold_ids, model="M0", n_folds=3,
    )

    assert np.isfinite(predictions).all()
    assert seen == 9


def _write_bank(path: Path, *, correct_gap: bool) -> Path:
    board = chess.Board()
    _ucis, legal_actions = legal_full_indices(board)
    actions = [int(action) for action in legal_actions]
    emitted = int(move_to_index(chess.Move.from_uci("a2a3"), board))
    alternative = int(move_to_index(chess.Move.from_uci("b2b3"), board))
    reference_best = int(move_to_index(chess.Move.from_uci("e2e4"), board))
    emitted_index = actions.index(emitted)
    alternative_index = actions.index(alternative)
    visits = [0] * len(actions)
    visits[emitted_index] = 20
    visits[alternative_index] = 30
    shares = [float(visit) / 50.0 for visit in visits]
    child_q = [0.2] * len(actions)
    child_q[emitted_index] = 0.1
    child_q_observed = [visit > 0 for visit in visits]
    action_regret = [20.0] * len(actions)
    action_regret[emitted_index] = 10.0
    action_regret[actions.index(reference_best)] = 0.0
    action_reference = [80.0] * len(actions)
    action_reference[emitted_index] = 90.0
    action_reference[actions.index(reference_best)] = 100.0
    action_listed = [False] * len(actions)
    action_listed[emitted_index] = True
    action_listed[alternative_index] = True
    action_listed[actions.index(reference_best)] = True
    entropy = float(-(0.4 * np.log(0.4) + 0.6 * np.log(0.6)))
    best_cp = 100.0
    regret_cp = 10.0
    regret_score = (
        1.0 / (1.0 + 10.0 ** (-best_cp / 300.0))
        - 1.0 / (1.0 + 10.0 ** (-(best_cp - regret_cp) / 300.0))
    )
    rows = [
        {
            "schema": "deepfin.chunk_trajectory.v3",
            "key": position_key(board), "source_dir": "/snapshot", "shard": "s0.zarr",
            "fen": board.fen(),
            "game_id": 3, "group_id": "/snapshot\0" + "3",
            "chunk": chunk, "nodes": chunk * 50,
            "elapsed_ms": float(chunk), "regret_cp": regret_cp,
            "regret_score": regret_score, "regret_vs_final_cp": 0.0,
            "deep_reference_best_cp": best_cp,
            "deep_reference_move_cp": {
                "e2e4": 100.0, "a2a3": 90.0, "b2b3": 80.0,
            },
            "visit_gap": -0.2 if correct_gap else 0.2,
            "root_actions": actions,
            "root_visits": [visit * chunk for visit in visits],
            "root_visit_shares": shares,
            "root_child_q": child_q,
            "root_child_q_observed": child_q_observed,
            "root_action_regret_cp": action_regret,
            "root_action_reference_cp": action_reference,
            "root_action_reference_listed": action_listed,
            "emitted_reference_listed": True,
            "emitted_action": emitted, "uci": "a2a3", "bestmove_flip": False,
            "pv_actions": [emitted], "pv_uci": ["a2a3"],
            "stable_chunks": 0, "visit_entropy": entropy, "q_gap": -0.1,
            "complexity_predicate_continue": True,
            "q_drift": None if chunk == 1 else 0.0,
            "visit_churn": None if chunk == 1 else 0.0, "root_q": 0.0,
            "changes_to_final": False,
            "phase": phase_bucket(32), "source": 0,
            "piece_count": 32, "legal_move_count": 20,
            "tb_probes": 0, "tb_hits": 0,
        }
        for chunk in (1, 2, 3, 4)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = Path(str(path) + ".meta.json")
    producer_source_paths = {
        "producer_script": "scripts/backtest_chunk_trajectory.py",
        "scripts.chunk_trajectory_publication": (
            "scripts/chunk_trajectory_publication.py"
        ),
        "scripts.analyze_chunk_controller": "scripts/analyze_chunk_controller.py",
        "scripts.repo_output_guard": "scripts/repo_output_guard.py",
        "chess_anti_engine.eval.audit": "chess_anti_engine/eval/audit.py",
        "chess_anti_engine.mcts.search_options": (
            "chess_anti_engine/mcts/search_options.py"
        ),
        "chess_anti_engine.uci.search": "chess_anti_engine/uci/search.py",
        "chess_anti_engine.uci.model_loader": (
            "chess_anti_engine/uci/model_loader.py"
        ),
    }
    producer_sources: dict[str, dict[str, Any]] = {}
    for name, relative_path in producer_source_paths.items():
        source = f"synthetic source for {name}\n".encode()
        _TEST_GIT_FILES[relative_path] = source
        producer_sources[name] = {
            "path": "/producer-checkout/" + relative_path,
            "repo_relative_path": relative_path,
            "matches_producer_git_revision": True,
            "size": len(source),
            "mtime_ns": 1,
            "sha256": hashlib.sha256(source).hexdigest(),
        }
    native_builds: dict[str, dict[str, Any]] = {}
    for module in controller_module._NATIVE_MODULES:
        dependency_bytes = {
            relative_path: f"synthetic native input {relative_path}\n".encode()
            for relative_path in controller_module.extension_spec(module).dependencies
        }
        _TEST_GIT_FILES.update(dependency_bytes)
        native_builds[module] = {
            **controller_module.native_build_attestation(
                module, "a" * 40, dependency_bytes,
            ),
            "current_inputs_match_revision": True,
            "matches_producer_revision": True,
        }
    manifest: dict[str, Any] = {
        "schema": "deepfin.chunk_trajectory.v3",
        "complete": True,
        "decision_grade": True,
        "analysis_scope": "fixed_node_horizons_only",
        "clock_conditioning_available": False,
        "reference_censoring": {
            "kind": "finite_multipv_unlisted_emitted_move",
            "scope": "completed_trajectory_decision_labels",
            "decision_labels_require_listed_emitted_moves": True,
            "passed": True,
            "affected_trajectory_count": 0,
            "unlisted_emitted_row_count": 0,
            "censored_transition_count": 0,
            "affected_trajectories": [],
        },
        "elapsed_measurement": {
            "kind": "callback_instrumented_wall_time",
            "usable_for_controller_or_cost_analysis": False,
        },
        "root_position_history": "fen_only_from_audit_fen",
        "root_tree_state": "fresh_per_position_no_cross_move_reuse",
        "game_group_kind": "source_dir:game_id",
        "complexity_predicate": {
            "kind": "clock_free_visit_gap_and_stability",
            "minimum_stable_chunks": 2,
            "minimum_visit_gap": 0.25,
            "single_legal_move_is_decided": True,
        },
        "producer_git_sha": "a" * 40,
        "producer_git_dirty": False,
        "producer_script": producer_sources["producer_script"],
        "publication_helper": producer_sources[
            "scripts.chunk_trajectory_publication"
        ],
        "producer_sources": producer_sources,
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
        "features_extension": {
            "path": "/_features_ext.so", "size": 1, "mtime_ns": 1,
            "sha256": "4" * 64,
            "build_attestation": native_builds[
                "chess_anti_engine.encoding._features_ext"
            ],
            "freshness_check": {
                "modules": list(controller_module._NATIVE_MODULES),
                "minimum_gcc_major": 15,
                "production_recipe_required": True,
                "passed": True, "issues": [],
            },
        },
        "mcts_extension": {
            "path": "/_mcts_tree.so", "size": 1, "mtime_ns": 1,
            "sha256": "e" * 64, "abi_version": 9, "required_abi_version": 9,
            "gss_halving_rev": 3,
            "build_attestation": native_builds[
                "chess_anti_engine.mcts._mcts_tree"
            ],
            "freshness_check": {
                "modules": list(controller_module._NATIVE_MODULES),
                "minimum_gcc_major": 15,
                "production_recipe_required": True,
                "passed": True, "issues": [],
            },
        },
        "lc0_extension": {
            "path": "/_lc0_ext.so", "size": 1, "mtime_ns": 1,
            "sha256": "2" * 64, "cboard_encode_full": True,
            "build_attestation": native_builds[
                "chess_anti_engine.encoding._lc0_ext"
            ],
            "freshness_check": {
                "modules": list(controller_module._NATIVE_MODULES),
                "minimum_gcc_major": 15,
                "production_recipe_required": True,
                "passed": True, "issues": [],
            },
        },
        "artifact_stability": {
            "passed": True, "changed": [], "final_git_sha": "a" * 40,
            "final_git_dirty": False,
        },
        "syzygy": {
            "path": "/tb/syzygy_3-4-5:/tb/syzygy_6",
            "rtbw_count": 875,
            "rtbz_count": 510,
            "directories": [
                {"path": "/tb/syzygy_3-4-5", "rtbw_count": 510, "rtbz_count": 145,
                 "total_bytes": 1, "inventory_sha256": "f" * 64},
                {"path": "/tb/syzygy_6", "rtbw_count": 365, "rtbz_count": 365,
                 "total_bytes": 1, "inventory_sha256": "1" * 64},
            ],
        },
        "row_count": 4,
        "chunk_count": 4,
        "position_count": 1,
        "requested_position_count": 1,
        "requested_max_positions": 2,
        "excluded_position_count": 0,
        "excluded_positions": [],
        "incomplete_exclusion_count": 0,
        "source_game_group_count": 9,
        "minimum_decision_grade_source_games": 9,
        "source_group_resolution_passed": True,
        "requested_search": {
            "device": "cuda", "active_path": "walker_puct",
            "walkers": 2, "chunk_sims": 50, "max_chunks": 4,
            "active_parameters": {
                "c_puct": 1.75, "cpuct_factor": 3.89, "cpuct_base": 38739.0,
                "fpu_reduction": 0.33, "vloss_weight": 3,
                "walker_gather": 1,
            },
        },
        "requested_model_search_contract": {
            "model_input_history_encoding": "legacy",
            "model_input_extra_features": "v1",
            "model_policy_encoding": "lc0_1858",
            "model_compute_relations": False,
            "search_input_history_encoding": "legacy",
            "search_input_extra_features": "v1",
            "search_policy_encoding": "lc0_1858",
            "search_compute_relations": False,
            "evaluator_input_planes": 146,
            "walker_input_planes": 146,
            "walker_compute_relations": False,
        },
        "realized_model_search_contract": {
            "model_input_history_encoding": "legacy",
            "model_input_extra_features": "v1",
            "model_policy_encoding": "lc0_1858",
            "model_compute_relations": False,
            "search_input_history_encoding": "legacy",
            "search_input_extra_features": "v1",
            "search_policy_encoding": "lc0_1858",
            "search_compute_relations": False,
            "evaluator_input_planes": 146,
            "walker_input_planes": 146,
            "walker_compute_relations": False,
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
        "runtime": {
            "python_version": "3.10", "python_implementation": "CPython",
            "python_executable": "/usr/bin/python3", "numpy_version": "2.x",
            "python_chess_version": "1.x", "platform": "Linux-test",
            "machine": "x86_64",
            "torch_version": "2.x", "torch_cuda_version": "13.0",
            "cudnn_version": 90000, "nvidia_driver_version": "600.1",
            "requested_device": "cuda", "evaluator_device": "cuda",
            "model_parameter_devices": ["cuda:0"],
            "resolved_requested_device": "cuda:0",
            "resolved_evaluator_device": "cuda:0",
            "resolved_model_parameter_devices": ["cuda:0"],
            "cuda_device_name": "test GPU", "cuda_device_capability": [12, 0],
        },
        "realized_search": {
            "concurrency_mode": "walker_puct", "concurrency_workers": 2,
            "chunk_sims": 50, "c_puct": 1.75, "cpuct_factor": 3.89,
            "cpuct_base": 38739.0, "fpu_reduction": 0.33,
            "vloss_weight": 3, "walker_gather": 1,
        },
        "search_warmup": {
            "completed": True, "requested_nodes": 256, "realized_nodes": 256,
            "excluded_from_timing": True, "tree_reset_after": True,
            "tablebase_counters_reset_after": True,
        },
        "realized_tablebase": {
            "installed": True, "cursed_as_draw": True,
            "n_wdl": 510, "n_dtz": 510, "max_pieces": 6,
            "root_probe_active": True, "leaf_probe_active": False,
            "positive_control": {
                "fen": "7k/8/8/8/8/8/8/KQ6 w - - 0 1",
                "probes": 1, "hits": 1, "apply_return": 1, "passed": True,
            },
            "root_shortcut_positive_control": {
                "fen": "7k/8/8/8/8/8/8/KQ6 w - - 0 1",
                "bestmove_uci": "b1b7", "nodes": 1, "tbhits": 1,
                "root_declined": None, "tree_created": False, "passed": True,
            },
        },
        "output": {"sha256": digest, "size": path.stat().st_size},
    }
    preregistration = {
        "schema": "deepfin.chunk_controller_preregistration.v1",
        "producer": {
            "source_git_sha": "9" * 40,
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "checkpoint_params_sha256": None,
            "audit_set_sha256": manifest["audit_set"]["sha256"],
            "matched_rows_sha256": manifest["matched_rows"]["sha256"],
            "max_positions": manifest["requested_max_positions"],
            "requested_search": manifest["requested_search"],
            "requested_model_search_contract": manifest[
                "requested_model_search_contract"
            ],
            "requested_evaluator": manifest["requested_evaluator"],
            "compile": {
                "enabled": manifest["compile"]["enabled"],
                "mode": manifest["compile"]["mode"],
            },
            "syzygy": manifest["syzygy"],
        },
        "analysis": {
            "folds": 5,
            "bootstrap_samples": 2000,
            "seed": 0,
            "allocation_fraction": 0.2,
            "min_capture_gain": 0.05,
            "min_oracle_headroom": 1e-4,
            "min_bootstrap_valid_fraction": 0.95,
        },
    }
    document = json.dumps(preregistration, sort_keys=True, separators=(",", ":")) + "\n"
    relative_path = "tests/fixtures/chunk_controller_preregistration.json"
    manifest["preregistration"] = {
        "path": "/repo/" + relative_path,
        "repo_relative_path": relative_path,
        "size": len(document.encode()),
        "mtime_ns": 1,
        "sha256": hashlib.sha256(document.encode()).hexdigest(),
    }
    manifest["preregistration_document"] = document
    _TEST_GIT_FILES[relative_path] = document.encode()
    meta.write_text(json.dumps(manifest))
    return meta


def _rewrite_bank(bank: Path, meta: Path, rows: list[dict[str, object]]) -> None:
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))


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
    assert info["preregistered_design"] is True
    assert info["analysis_scope"] == "fresh_tree_fixed_node_horizons_only"
    assert info["cross_move_tree_reuse_tested"] is False


def test_loader_rejects_decision_grade_claim_with_insufficient_source_games(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["source_game_group_count"] = 8
    manifest["source_group_resolution_passed"] = False
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="insufficient source-game resolution"):
        load_transitions(bank)

    _transitions, info = load_transitions(bank, methodology_smoke=True)
    assert info["decision_grade"] is False


def test_loader_requires_publication_helper_provenance(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    del manifest["publication_helper"]
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="publication_helper artifact provenance"):
        load_transitions(bank)


def test_loader_rejects_publication_helper_not_bound_to_producer_revision(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    foreign_source = b"foreign publication helper\n"
    manifest["publication_helper"].update({
        "path": "/foreign-worktree/scripts/chunk_trajectory_publication.py",
        "size": len(foreign_source),
        "sha256": hashlib.sha256(foreign_source).hexdigest(),
    })
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="not bound to the producer Git revision"):
        load_transitions(bank)


def test_loader_rejects_foreign_loaded_producer_python_module(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    foreign_source = b"foreign search implementation\n"
    manifest["producer_sources"]["chess_anti_engine.uci.search"].update({
        "path": "/foreign/chess_anti_engine/uci/search.py",
        "size": len(foreign_source),
        "sha256": hashlib.sha256(foreign_source).hexdigest(),
    })
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="producer Python sources"):
        load_transitions(bank)


def test_loader_rejects_a_collection_outside_the_preregistered_design(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["requested_max_positions"] = 3
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="does not match the preregistered design"):
        load_transitions(bank)


def test_loader_authenticates_preregistration_against_the_producer_commit(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    payload = json.loads(manifest["preregistration_document"])
    payload["analysis"]["seed"] = 1
    document = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    manifest["preregistration_document"] = document
    manifest["preregistration"]["size"] = len(document.encode())
    manifest["preregistration"]["sha256"] = hashlib.sha256(document.encode()).hexdigest()
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="not tracked verbatim"):
        load_transitions(bank)


def test_loader_requires_fresh_tree_scope_provenance(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["root_tree_state"] = "production_reused_tree"
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="root_tree_state"):
        load_transitions(bank)


def test_loader_exposes_missing_drift_and_churn_to_M0(tmp_path: Path) -> None:
    from scripts.analyze_chunk_controller import _design

    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)

    transitions, _ = load_transitions(bank)

    assert transitions[0].state["q_drift_missing"] == 1.0
    assert transitions[0].state["visit_churn_missing"] == 1.0
    assert transitions[1].state["q_drift_missing"] == 0.0
    assert transitions[1].state["visit_churn_missing"] == 0.0
    assert _design(transitions, "M0")[0].tolist() != _design(transitions, "M0")[1].tolist()


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


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_chunks", 3), ("chunk_sims", 31), ("walkers", 1)],
)
def test_loader_rejects_invalid_requested_search_shape(
    tmp_path: Path, field: str, value: int,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["requested_search"][field] = value
    if field in {"chunk_sims", "walkers"}:
        realized_field = "chunk_sims" if field == "chunk_sims" else "concurrency_workers"
        manifest["realized_search"][realized_field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="requested search"):
        load_transitions(bank)


def test_loader_rejects_classic_gumbel_as_decision_grade(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["requested_search"].update({
        "active_path": "gumbel",
        "walkers": 1,
        "active_parameters": {
            "c_scale": 0.1,
            "c_visit": 50.0,
            "c_visit_root": 50.0,
            "c_scale_root": 1.0,
            "q_visit_exp_root": 1.0,
            "topk": 16,
            "policy_temp": 1.0,
            "halving_div": 2,
            "root_noise_scale": 1.0,
            "vloss_weight": 3,
            "minibatch_size": 32,
        },
    })
    manifest["realized_search"] = {
        **manifest["requested_search"]["active_parameters"],
        "concurrency_mode": "gumbel",
        "concurrency_workers": 1,
        "chunk_sims": manifest["requested_search"]["chunk_sims"],
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="requested search"):
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


def test_loader_requires_feature_encoder_binary_provenance(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["features_extension"]["sha256"] = "not-a-hash"
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="feature encoding extension"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("section", "module", "message"),
    [
        (
            "features_extension",
            "chess_anti_engine.encoding._features_ext",
            "feature encoding extension",
        ),
        (
            "lc0_extension",
            "chess_anti_engine.encoding._lc0_ext",
            "CBoard encoding extension",
        ),
        (
            "mcts_extension",
            "chess_anti_engine.mcts._mcts_tree",
            "MCTS extension",
        ),
    ],
)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_git_sha", "b" * 40),
        ("input_sha256", "c" * 64),
        ("matches_producer_revision", False),
    ],
)
def test_loader_rejects_mismatched_embedded_native_build_attestation(
    tmp_path: Path,
    section: str,
    module: str,
    message: str,
    field: str,
    value: object,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    assert manifest[section]["build_attestation"]["module"] == module
    manifest[section]["build_attestation"][field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ("features_extension", "feature encoding extension"),
        ("lc0_extension", "CBoard encoding extension"),
        ("mcts_extension", "MCTS extension"),
    ],
)
def test_loader_rejects_native_build_dependency_not_at_producer_revision(
    tmp_path: Path, section: str, message: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    dependency = next(iter(manifest[section]["build_attestation"]["dependencies"]))
    _TEST_GIT_FILES[dependency] += b" changed after foreign build"

    with pytest.raises(ValueError, match=message):
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


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("search_warmup", "completed", False, "search warmup"),
        ("search_warmup", "realized_nodes", 0, "search warmup"),
        (
            "elapsed_measurement", "usable_for_controller_or_cost_analysis", True,
            "elapsed-time instrumentation",
        ),
        ("mcts_extension", "freshness_check", {}, "native MCTS extension"),
    ],
)
def test_loader_requires_warmup_timing_and_native_freshness_provenance(
    tmp_path: Path, section: str, field: str, value: object, match: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest[section][field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=match):
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shard", "other.zarr"),
        ("game_id", 99),
        (
            "deep_reference_move_cp",
            {"e2e4": 100.0, "a2a3": 90.0, "b2b3": 80.0, "c2c3": 80.0},
        ),
    ],
)
def test_loader_requires_trajectory_identity_and_labels_to_stay_fixed(
    tmp_path: Path, field: str, value: object,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows[-1][field] = value
    if field == "game_id":
        rows[-1]["group_id"] = "\0".join((
            str(rows[-1]["source_dir"]),
            str(rows[-1]["game_id"]),
        ))
    if field == "deep_reference_move_cp":
        board = chess.Board(str(rows[-1]["fen"]))
        newly_listed = int(move_to_index(chess.Move.from_uci("c2c3"), board))
        listed_index = rows[-1]["root_actions"].index(newly_listed)
        rows[-1]["root_action_reference_listed"][listed_index] = True
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match="trajectory-invariant fields change"):
        load_transitions(bank)


def _mark_emitted_reference_unlisted(row: dict[str, object]) -> None:
    actions = row["root_actions"]
    listed = row["root_action_reference_listed"]
    references = row["root_action_reference_cp"]
    assert isinstance(actions, list)
    assert isinstance(listed, list)
    assert isinstance(references, list)
    emitted_index = actions.index(row["emitted_action"])
    listed[emitted_index] = False
    references[emitted_index] = min(float(value) for value in references)
    row["emitted_reference_listed"] = False
    move_cp = row["deep_reference_move_cp"]
    assert isinstance(move_cp, dict)
    move_cp.pop(str(row["uci"]))


@pytest.mark.parametrize(
    ("listed", "expected_unlisted_rows"),
    [((True, False), 1), ((False, False), 2)],
    ids=("listed-unlisted", "unlisted-unlisted"),
)
def test_reference_censoring_marks_each_adjacent_unlisted_pair_once(
    listed: tuple[bool, bool], expected_unlisted_rows: int,
) -> None:
    rows = [
        {
            "chunk": chunk,
            "nodes": chunk * 50,
            "actions": [7],
            "action_reference_listed": [is_listed],
            "emitted_action": 7,
            "emitted_reference_listed": is_listed,
            "uci": "a2a3",
        }
        for chunk, is_listed in enumerate(listed, start=1)
    ]

    details = controller_module._trajectory_reference_censoring("position", rows)

    assert details is not None
    summary = controller_module._reference_censoring_summary([details])
    assert summary["unlisted_emitted_row_count"] == expected_unlisted_rows
    assert summary["censored_transition_count"] == 1
    assert details["censored_transitions"] == [{
        "from_chunk": 1, "to_chunk": 2, "horizon_nodes": 100,
    }]


@pytest.mark.parametrize(
    ("unlisted_indices", "expected_unlisted_rows", "expected_censored_transitions"),
    [
        ((1,), 1, 2),
        ((0, 1), 2, 2),
    ],
    ids=("listed-unlisted", "unlisted-unlisted"),
)
def test_finite_multipv_censoring_preserves_rows_but_disqualifies_the_bank(
    tmp_path: Path,
    unlisted_indices: tuple[int, ...],
    expected_unlisted_rows: int,
    expected_censored_transitions: int,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    for index in unlisted_indices:
        _mark_emitted_reference_unlisted(rows[index])
    details = controller_module._trajectory_reference_censoring(
        str(rows[0]["key"]), rows,
    )
    assert details is not None
    censoring = controller_module._reference_censoring_summary([details])
    assert censoring["unlisted_emitted_row_count"] == expected_unlisted_rows
    assert censoring["censored_transition_count"] == expected_censored_transitions
    assert censoring["passed"] is False
    _rewrite_bank(bank, meta, rows)
    manifest = json.loads(meta.read_text())
    manifest["decision_grade"] = False
    manifest["reference_censoring"] = censoring
    meta.write_text(json.dumps(manifest))

    transitions, info = load_transitions(bank, methodology_smoke=True)

    assert len(transitions) == 3
    assert info["decision_grade"] is False
    assert info["reference_censoring"] == censoring
    assert len(bank.read_text().splitlines()) == 4


def test_decision_grade_row_rejects_an_unlisted_emitted_move(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)
    row = json.loads(bank.read_text().splitlines()[0])
    _mark_emitted_reference_unlisted(row)

    with pytest.raises(ValueError, match="decision label is censored"):
        controller_module._validate_decision_grade_row(
            row, 1, require_full_root_support=True,
        )


def test_decision_grade_manifest_rejects_finite_multipv_censoring(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["reference_censoring"] = {
        "kind": "finite_multipv_unlisted_emitted_move",
        "scope": "completed_trajectory_decision_labels",
        "decision_labels_require_listed_emitted_moves": True,
        "passed": False,
        "affected_trajectory_count": 1,
        "unlisted_emitted_row_count": 1,
        "censored_transition_count": 1,
        "affected_trajectories": [{
            "key": "position",
            "unlisted_emitted_rows": [{
                "chunk": 1, "nodes": 50, "emitted_action": 1, "uci": "a2a3",
            }],
            "censored_transitions": [{
                "from_chunk": 1, "to_chunk": 2, "horizon_nodes": 100,
            }],
        }],
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="finite-MultiPV censoring"):
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
    manifest.update({"row_count": 8, "position_count": 2, "requested_position_count": 2})
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
    rows = [json.loads(json.dumps(row)) for row in original]
    rows[-1] = json.loads(json.dumps(original[2]))
    other_board = chess.Board(
        "rn3bnr/ppp1k2p/5pp1/Pb1pp3/1P3PPq/B7/N1PPP1BP/R2QK1NR w KQ - 4 14"
    )
    other_move = next(iter(other_board.legal_moves))
    other_action = int(move_to_index(other_move, other_board))
    for chunk, template in enumerate(original, start=1):
        row = json.loads(json.dumps(template))
        row.update({
            "key": position_key(other_board), "fen": other_board.fen(),
            "game_id": 4,
            "group_id": "\0".join(("/snapshot", "4")),
            "chunk": chunk, "nodes": chunk * 50, "elapsed_ms": float(chunk),
            "root_actions": [other_action], "root_visits": [chunk * 50],
            "root_visit_shares": [1.0], "root_child_q": [0.1],
            "root_child_q_observed": [True],
            "root_action_regret_cp": [0.0],
            "root_action_reference_cp": [90.0],
            "root_action_reference_listed": [True],
            "deep_reference_best_cp": 90.0,
            "deep_reference_move_cp": {other_move.uci(): 90.0},
            "regret_cp": 0.0, "regret_score": 0.0,
            "emitted_action": other_action, "uci": other_move.uci(),
            "pv_actions": [other_action], "pv_uci": [other_move.uci()],
            "visit_gap": 1.0, "visit_entropy": 0.0, "q_gap": None,
            "stable_chunks": chunk - 1,
            "complexity_predicate_continue": chunk < 3,
            "q_drift": None if chunk == 1 else 0.0,
            "visit_churn": None if chunk == 1 else 0.0,
            "phase": phase_bucket(chess.popcount(other_board.occupied)),
            "piece_count": chess.popcount(other_board.occupied),
            "legal_move_count": 1,
        })
        rows.append(row)
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest.update({"row_count": 8, "position_count": 2, "requested_position_count": 2})
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="complete consecutive manifest horizon"):
        load_transitions(bank)


def test_loader_accepts_production_forced_move_stability_semantics(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    original = [json.loads(line) for line in bank.read_text().splitlines()]
    forced_board = chess.Board(
        "rn3bnr/ppp1k2p/5pp1/Pb1pp3/1P3PPq/B7/N1PPP1BP/R2QK1NR w KQ - 4 14"
    )
    forced_move = next(iter(forced_board.legal_moves))
    assert forced_board.legal_moves.count() == 1
    forced_action = int(move_to_index(forced_move, forced_board))
    rows = []
    for chunk in (1, 2, 3, 4):
        row = json.loads(json.dumps(original[min(chunk - 1, 1)]))
        row.update({
            "key": position_key(forced_board), "fen": forced_board.fen(),
            "chunk": chunk, "nodes": chunk * 50, "elapsed_ms": float(chunk),
            "root_actions": [forced_action],
            "root_visits": [chunk * 50], "root_visit_shares": [1.0],
            "root_child_q": [0.1], "root_child_q_observed": [True],
            "root_action_regret_cp": [0.0],
            "root_action_reference_cp": [90.0],
            "root_action_reference_listed": [True],
            "deep_reference_best_cp": 90.0,
            "deep_reference_move_cp": {forced_move.uci(): 90.0},
            "regret_cp": 0.0, "regret_score": 0.0,
            "emitted_action": forced_action, "uci": forced_move.uci(),
            "pv_actions": [forced_action], "pv_uci": [forced_move.uci()],
            "visit_gap": 1.0, "visit_entropy": 0.0, "q_gap": None,
            "stable_chunks": chunk - 1,
            "complexity_predicate_continue": chunk < 3,
            "phase": phase_bucket(chess.popcount(forced_board.occupied)),
            "piece_count": chess.popcount(forced_board.occupied),
            "legal_move_count": 1,
        })
        rows.append(row)
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest.update({"row_count": 4, "chunk_count": 4})
    manifest["requested_search"]["max_chunks"] = 4
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    transitions, _ = load_transitions(bank)

    assert len(transitions) == 3
    assert transitions[-1].complexity_continue is False


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("root_child_q", [0.1, float("nan")], "root child Q"),
        ("root_action_regret_cp", [10.0, float("nan")], "root action regrets"),
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
    if field in ("root_child_q", "root_action_regret_cp"):
        assert isinstance(value, list)
        expanded = list(rows[-1][field])
        expanded[-1] = value[-1]
        value = expanded
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


@pytest.mark.parametrize("location", ["root", "later_pv"])
def test_loader_strictly_rejects_undecodable_native_actions(
    tmp_path: Path, location: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    if location == "root":
        rows[-1]["root_actions"][1] = 99_999
    else:
        rows[-1]["pv_actions"].append(99_999)
        rows[-1]["pv_uci"].append("a7a6")
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="action cannot be decoded"):
        load_transitions(bank)


def test_loader_requires_full_walker_root_support(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    for row in rows:
        zero_index = row["root_visits"].index(0)
        for field in (
            "root_actions", "root_visits", "root_visit_shares", "root_child_q",
            "root_child_q_observed", "root_action_regret_cp",
            "root_action_reference_cp", "root_action_reference_listed",
        ):
            row[field].pop(zero_index)
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match="every legal action"):
        load_transitions(bank)


def test_loader_rejects_decreasing_accumulated_root_visits(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    row = rows[1]
    nonzero = [i for i, visit in enumerate(row["root_visits"]) if visit > 0]
    emitted_index = row["root_actions"].index(row["emitted_action"])
    other_index = next(index for index in nonzero if index != emitted_index)
    row["root_visits"][emitted_index] = 19
    row["root_visits"][other_index] = 81
    row["root_visit_shares"][emitted_index] = 0.19
    row["root_visit_shares"][other_index] = 0.81
    row["visit_gap"] = -0.62
    row["visit_entropy"] = float(-(0.19 * np.log(0.19) + 0.81 * np.log(0.81)))
    row["visit_churn"] = 0.21
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match="root visits decrease"):
        load_transitions(bank)


def test_loader_requires_root_visits_to_account_for_completed_nodes(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    row = rows[-1]
    nonzero = [i for i, visit in enumerate(row["root_visits"]) if visit > 0]
    row["root_visits"][nonzero[0]] -= 1
    total = sum(row["root_visits"])
    row["root_visit_shares"] = [visit / total for visit in row["root_visits"]]
    emitted_index = row["root_actions"].index(row["emitted_action"])
    alternatives = [
        share for index, share in enumerate(row["root_visit_shares"])
        if index != emitted_index
    ]
    row["visit_gap"] = row["root_visit_shares"][emitted_index] - max(alternatives)
    positive = np.asarray([s for s in row["root_visit_shares"] if s > 0.0])
    row["visit_entropy"] = float(-(positive * np.log(positive)).sum())
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match="completed nodes"):
        load_transitions(bank)


def test_gumbel_zero_visit_exception_requires_a_truly_forced_position(
    tmp_path: Path,
) -> None:
    from scripts.analyze_chunk_controller import _validate_decision_grade_row

    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)
    row = json.loads(bank.read_text().splitlines()[0])
    emitted_index = row["root_actions"].index(row["emitted_action"])
    for field in (
        "root_actions", "root_child_q", "root_action_regret_cp",
        "root_action_reference_cp", "root_action_reference_listed",
    ):
        row[field] = [row[field][emitted_index]]
    row["root_visits"] = [0]
    row["root_visit_shares"] = [0.0]
    row["root_child_q_observed"] = [False]
    row["visit_gap"] = 0.0
    row["visit_entropy"] = 0.0
    row["q_gap"] = None

    with pytest.raises(ValueError, match="completed nodes"):
        _validate_decision_grade_row(row, 1, require_full_root_support=False)


def test_gumbel_zero_visit_exception_accepts_a_truly_forced_position(
    tmp_path: Path,
) -> None:
    from scripts.analyze_chunk_controller import _validate_decision_grade_row

    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)
    row = json.loads(bank.read_text().splitlines()[0])
    board = chess.Board(
        "rn3bnr/ppp1k2p/5pp1/Pb1pp3/1P3PPq/B7/N1PPP1BP/R2QK1NR w KQ - 4 14"
    )
    move = next(iter(board.legal_moves))
    assert board.legal_moves.count() == 1
    action = int(move_to_index(move, board))
    row.update({
        "key": position_key(board), "fen": board.fen(),
        "phase": phase_bucket(chess.popcount(board.occupied)),
        "piece_count": chess.popcount(board.occupied), "legal_move_count": 1,
        "root_actions": [action], "root_visits": [0], "root_visit_shares": [0.0],
        "root_child_q": [0.0], "root_child_q_observed": [False],
        "root_action_regret_cp": [0.0], "root_action_reference_cp": [90.0],
        "root_action_reference_listed": [True],
        "deep_reference_best_cp": 90.0, "deep_reference_move_cp": {move.uci(): 90.0},
        "regret_cp": 0.0, "regret_score": 0.0, "regret_vs_final_cp": 0.0,
        "emitted_action": action, "uci": move.uci(), "pv_actions": [action],
        "pv_uci": [move.uci()], "visit_gap": 0.0, "visit_entropy": 0.0,
        "q_gap": None, "root_q": 0.0,
    })

    _validate_decision_grade_row(row, 1, require_full_root_support=False)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("key", "invented", "position key disagrees"),
        ("phase", 1, "phase disagrees"),
        ("source", 7, "source must be"),
    ],
)
def test_loader_recomputes_position_identity_and_domains(
    tmp_path: Path, field: str, value: object, match: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows[0][field] = value
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match=match):
        load_transitions(bank)


def test_loader_disqualifies_incomplete_search_exclusions(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest.update({
        "requested_position_count": 2,
        "excluded_position_count": 1,
        "incomplete_exclusion_count": 1,
        "excluded_positions": [{
            "key": "missing",
            "chunks_observed": 1,
            "chunks_required": 4,
            "reason": "incomplete_search",
            "search_result": {
                "nodes": 50, "tbhits": 0, "root_declined": "declined",
                "score_mate": None, "board_game_over": False,
            },
        }],
    })
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="accounting is inconsistent"):
        load_transitions(bank)


def test_loader_rejects_multinode_terminal_shortcut_exclusion(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest.update({
        "requested_position_count": 2,
        "excluded_position_count": 1,
        "excluded_positions": [{
            "key": "missing",
            "chunks_observed": 0,
            "chunks_required": 4,
            "reason": "production_terminal_shortcut",
            "search_result": {
                "nodes": 2, "tbhits": 0, "root_declined": None,
                "score_mate": 1, "board_game_over": False,
            },
        }],
    })
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="accounting is inconsistent"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("row_index", "field", "value", "match"),
    [
        (0, "visit_entropy", 0.1, "visit_entropy disagrees"),
        (0, "regret_cp", 11.0, "regret_cp disagrees"),
        (0, "regret_score", 0.5, "regret_score disagrees"),
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
    if field == "root_visits":
        assert isinstance(value, list)
        assert len(value) == 2
        expanded = list(rows[row_index][field])
        nonzero = [i for i, visit in enumerate(expanded) if visit > 0]
        expanded[nonzero[0]], expanded[nonzero[1]] = value
        value = expanded
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


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("regret_cp", -1.0, "audit regret domain"),
        ("regret_score", 1.1, r"outside \[0, 1\]"),
        ("visit_gap", 1.1, r"outside \[-1, 1\]"),
        ("root_q", -1.1, r"outside \[-1, 1\]"),
        ("q_gap", 2.1, r"outside \[-2, 2\]"),
        ("q_drift", 2.1, r"outside \[0, 2\]"),
        ("visit_churn", 1.1, r"outside \[0, 1\]"),
    ],
)
def test_loader_rejects_finite_values_outside_metric_domains(
    tmp_path: Path, field: str, value: float, match: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    target = rows[-1] if field in {"q_drift", "visit_churn"} else rows[0]
    target[field] = value
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match=match):
        load_transitions(bank)


def test_output_path_cannot_replace_the_bank_or_manifest(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    checkpoint = tmp_path / "trainer.pt"
    tablebases = tmp_path / "syzygy_6"
    _require_safe_output_path(bank, meta, tmp_path / "report.json")
    with pytest.raises(ValueError, match="must not overwrite"):
        _require_safe_output_path(bank, meta, bank)
    with pytest.raises(ValueError, match="must not overwrite"):
        _require_safe_output_path(bank, meta, meta)
    manifest = {
        "checkpoint": {"path": str(checkpoint)},
        "preregistration": {"path": str(tmp_path / "preregister.json")},
        "syzygy": {"directories": [{"path": str(tablebases)}]},
    }
    with pytest.raises(ValueError, match="consumed input artifact"):
        _require_safe_output_path(bank, meta, checkpoint, manifest=manifest)
    with pytest.raises(ValueError, match="consumed input artifact"):
        _require_safe_output_path(
            bank, meta, tmp_path / "preregister.json", manifest=manifest,
        )
    with pytest.raises(ValueError, match="Syzygy"):
        _require_safe_output_path(
            bank, meta, tablebases / "report.json", manifest=manifest,
        )


def test_producer_output_paths_cannot_replace_inputs_or_tablebases(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit.jsonl"
    tablebases = tmp_path / "tb"
    tablebases.mkdir()
    with pytest.raises(SystemExit, match="aliases"):
        _require_safe_output_paths(
            audit, tmp_path / "out.meta.json",
            protected_files=[audit], protected_directories=[tablebases],
        )
    with pytest.raises(SystemExit, match="Syzygy"):
        _require_safe_output_paths(
            tablebases / "bank.jsonl", tmp_path / "out.meta.json",
            protected_files=[audit], protected_directories=[tablebases],
        )


def test_output_guards_reject_linked_worktree_git_metadata(tmp_path: Path) -> None:
    from scripts.repo_output_guard import git_control_paths

    repo_root = Path(__file__).resolve().parents[1]
    git_directories = [path for path in git_control_paths(repo_root) if path.is_dir()]
    assert git_directories

    for git_directory in git_directories:
        target = git_directory / "HEAD"
        with pytest.raises(ValueError, match="repository-control"):
            _require_safe_output_path(
                tmp_path / "bank.jsonl",
                tmp_path / "bank.jsonl.meta.json",
                target,
            )
        with pytest.raises(SystemExit, match="repository-control"):
            _require_safe_output_paths(
                target,
                tmp_path / "out.meta.json",
                protected_files=[],
                protected_directories=[],
            )


def test_output_guard_fails_closed_on_fatal_git_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from scripts import repo_output_guard as guard

    monkeypatch.setattr(guard, "git_control_paths", lambda _root: ())
    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=128),
    )

    assert guard.repo_controlled_output(tmp_path / "report.json", tmp_path) is True


def test_producer_output_cannot_overwrite_an_imported_tracked_source(
    tmp_path: Path,
) -> None:
    from scripts import analyze_chunk_controller as controller

    with pytest.raises(SystemExit, match="tracked or repository-control"):
        _require_safe_output_paths(
            Path(controller.__file__),
            tmp_path / "out.meta.json",
            protected_files=[],
            protected_directories=[],
        )


def test_producer_output_lock_serializes_the_bank_manifest_pair(tmp_path: Path) -> None:
    output = tmp_path / "bank.jsonl"
    first = _acquire_output_lock(output)
    try:
        with pytest.raises(SystemExit, match="another producer"):
            _acquire_output_lock(output)
    finally:
        first.close()

    retry = _acquire_output_lock(output)
    retry.close()


def test_producer_output_locks_serialize_overlapping_pairs(tmp_path: Path) -> None:
    output = tmp_path / "bank.jsonl"
    meta = Path(str(output) + ".meta.json")
    first = _acquire_output_locks(output, meta)
    try:
        with pytest.raises(SystemExit, match="another producer holds"):
            _acquire_output_locks(meta, Path(str(meta) + ".meta.json"))
    finally:
        for handle in reversed(first):
            handle.close()

    retry = _acquire_output_locks(meta, Path(str(meta) + ".meta.json"))
    for handle in reversed(retry):
        handle.close()


def test_producer_never_overwrites_frozen_evidence_pair(tmp_path: Path) -> None:
    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    output.write_text("old bank\n")
    meta.write_text("old manifest\n")

    with pytest.raises(SystemExit, match="immutable or incomplete evidence"):
        _require_new_output_pair(output, meta, overwrite=False)
    with pytest.raises(SystemExit, match="overwrite is disabled"):
        _require_new_output_pair(output, meta, overwrite=True)

    assert output.read_text() == "old bank\n"
    assert meta.read_text() == "old manifest\n"


def test_producer_accepts_a_new_evidence_pair_path(tmp_path: Path) -> None:
    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"

    assert _require_new_output_pair(output, meta, overwrite=False) is False


def test_producer_recovers_bank_published_before_its_manifest(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }
    producer._write_json_staged(pending_meta, manifest)
    producer._publish_output(pending_output, output)

    assert not meta.exists()
    assert pending_meta.exists()
    assert _require_new_output_pair(output, meta, overwrite=False) is True
    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest
    assert not pending_output.exists()
    assert not pending_meta.exists()


def test_producer_prepares_both_artifacts_before_pair_publication(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }

    producer._publish_evidence_pair(
        pending_output, output, pending_meta, meta, manifest,
    )

    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest
    assert not pending_output.exists()
    assert not pending_meta.exists()


def test_pair_publication_preserves_preexisting_pending_manifest(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("new bank\n")
    pending_meta.write_text("existing pending manifest\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }

    with pytest.raises(FileExistsError):
        producer._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    assert pending_output.read_text() == "new bank\n"
    assert pending_meta.read_text() == "existing pending manifest\n"
    assert not output.exists()
    assert not meta.exists()


def test_producer_atomic_json_preserves_preexisting_staging_file(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "manifest.json"
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    staging.write_text("other process staging\n")

    with pytest.raises(FileExistsError):
        producer._write_json_atomic(output, {"new": True})

    assert staging.read_text() == "other process staging\n"
    assert not output.exists()


def test_analyzer_atomic_json_preserves_preexisting_staging_file(tmp_path: Path) -> None:
    output = tmp_path / "analysis.json"
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    staging.write_text("other process staging\n")

    with pytest.raises(FileExistsError):
        controller_module._write_json_atomic(output, "{}")

    assert staging.read_text() == "other process staging\n"
    assert not output.exists()


def test_producer_recovers_fully_staged_pair_before_either_publish(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }
    producer._write_json_staged(pending_meta, manifest)

    assert _require_new_output_pair(output, meta, overwrite=False) is True
    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest
    assert not pending_output.exists()
    assert not pending_meta.exists()


def test_recovery_cli_does_not_require_search_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }
    producer._write_json_staged(pending_meta, manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        ["backtest_chunk_trajectory", "--recover-publication", "--out", str(output)],
    )

    producer.main()

    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest


def test_recovery_cli_runs_before_project_or_native_imports(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }
    producer._write_json_staged(pending_meta, manifest)

    shadow = tmp_path / "shadow"
    shadow_package = shadow / "chess_anti_engine"
    shadow_package.mkdir(parents=True)
    (shadow_package / "__init__.py").write_text(
        'raise RuntimeError("search runtime imported")\n'
    )
    repo_root = Path(producer.__file__).resolve().parents[1]
    python_path = [str(shadow), str(repo_root)]
    inherited_python_path = os.environ.get("PYTHONPATH")
    if inherited_python_path:
        python_path.append(inherited_python_path)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(python_path)}

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(producer.__file__).resolve()),
            "--recover-publication",
            "--out", str(output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "search runtime imported" not in completed.stderr
    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest
    assert not pending_output.exists()
    assert not pending_meta.exists()


def test_recovery_cli_rejects_unknown_options_without_publishing(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    default_output = tmp_path / "runs/backtest/chunk_trajectory.jsonl"
    default_output.parent.mkdir(parents=True)
    default_meta = Path(str(default_output) + ".meta.json")
    pending_output = producer._pending_output_path(default_output)
    pending_meta = producer._pending_manifest_path(default_meta)
    pending_output.write_text("completed default bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, default_output),
    }
    producer._write_json_staged(pending_meta, manifest)
    intended_output = tmp_path / "intended/bank.jsonl"
    repo_root = Path(producer.__file__).resolve().parents[1]
    python_path = [str(repo_root)]
    inherited_python_path = os.environ.get("PYTHONPATH")
    if inherited_python_path:
        python_path.append(inherited_python_path)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(python_path)}

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(producer.__file__).resolve()),
            "--recover-publication",
            "--ouut", str(intended_output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "unrecognized arguments: --ouut" in completed.stderr
    assert pending_output.read_text() == "completed default bank\n"
    assert json.loads(pending_meta.read_text()) == manifest
    assert not default_output.exists()
    assert not default_meta.exists()
    assert not intended_output.exists()


def test_import_never_dispatches_recovery_from_host_argv(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = Path(str(output) + ".meta.json")
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }
    producer._write_json_staged(pending_meta, manifest)
    repo_root = Path(producer.__file__).resolve().parents[1]
    python_path = [str(repo_root)]
    inherited_python_path = os.environ.get("PYTHONPATH")
    if inherited_python_path:
        python_path.append(inherited_python_path)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(python_path)}
    program = (
        "import sys; "
        f"sys.argv = ['host', '--recover-publication', '--out', {str(output)!r}]; "
        "import scripts.backtest_chunk_trajectory"
    )

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert pending_output.read_text() == "completed bank\n"
    assert json.loads(pending_meta.read_text()) == manifest
    assert not output.exists()
    assert not meta.exists()


def test_producer_refuses_unprepared_pending_bank(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    producer._pending_output_path(output).write_text("partial bank\n")

    with pytest.raises(SystemExit, match="incomplete evidence"):
        _require_new_output_pair(output, meta, overwrite=False)


def test_git_ignored_output_guard_distinguishes_repository_paths(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("/ignored/\n")

    assert producer._git_ignored_or_outside(repo / "ignored/bank.jsonl", repo)
    assert not producer._git_ignored_or_outside(repo / "bank.jsonl", repo)
    assert producer._git_ignored_or_outside(tmp_path / "outside.jsonl", repo)


def test_producer_rejects_output_artifacts_that_would_dirty_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    monkeypatch.setattr(
        producer.publication_module,
        "_git_ignored_or_outside",
        lambda _path, _root: False,
    )

    with pytest.raises(SystemExit, match="must be Git-ignored or outside"):
        _require_safe_output_paths(
            tmp_path / "bank.jsonl",
            tmp_path / "bank.jsonl.meta.json",
            protected_files=[],
            protected_directories=[],
        )


def test_producer_requires_enough_source_games_for_canonical_bootstrap() -> None:
    from scripts import backtest_chunk_trajectory as producer

    with pytest.raises(SystemExit, match="at least 9 distinct source games"):
        producer._require_analyzable_source_groups(
            {f"position-{game}": f"snapshot\0game-{game}" for game in range(8)},
            methodology_smoke=False,
        )

    producer._require_analyzable_source_groups(
        {f"position-{game}": f"snapshot\0game-{game}" for game in range(9)},
        methodology_smoke=False,
    )
    producer._require_analyzable_source_groups(
        {"a": None}, methodology_smoke=True,
    )


def test_producer_marks_post_collection_group_loss_non_decision_grade() -> None:
    from scripts import backtest_chunk_trajectory as producer

    requested_groups = {
        f"position-{game}": f"snapshot\0game-{game}" for game in range(9)
    }
    completed_groups = {
        f"snapshot\0game-{game}": f"snapshot\0game-{game}" for game in range(8)
    }

    producer._require_analyzable_source_groups(
        requested_groups, methodology_smoke=False,
    )
    assert producer._source_game_group_count(completed_groups) == 8
    assert producer._source_group_resolution_passed(completed_groups) is False


def test_excluded_position_evidence_preserves_partial_raw_snapshots() -> None:
    from scripts import backtest_chunk_trajectory as producer

    position = SimpleNamespace(
        key="position",
        fen=chess.Board().fen(),
        phase=1,
        source=2,
        best_cp=42,
        move_cp={"e2e4": 42, "d2d4": 31},
    )
    snapshots = [{
        "nodes": 50,
        "actions": [1, 2],
        "visits": [30, 20],
        "child_q": [0.2, 0.1],
        "pv_actions": [1],
        "pv_uci": ["e2e4"],
    }]

    evidence = producer._excluded_position_evidence(
        position,
        source_dir="/snapshot",
        source_shard="s0.zarr",
        game_id=7,
        group_id="/snapshot\0" + "7",
        chunks_required=4,
        snapshots=snapshots,
        reason="incomplete_search",
        search_result={"nodes": 50},
    )

    assert evidence["chunks_observed"] == 1
    assert evidence["partial_observations"] == snapshots
    assert evidence["deep_reference_move_cp"] == {"e2e4": 42.0, "d2d4": 31.0}
    assert evidence["search_result"] == {"nodes": 50}


def test_post_collection_failure_preserves_complete_pending_bank_with_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    pending_output = producer._pending_output_path(output)
    meta = Path(str(output) + ".meta.json")
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("row one\nrow two\n")
    censoring = controller_module._reference_censoring_summary([])

    def fail_after_collection() -> None:
        producer._ACTIVE_PENDING_EVIDENCE = {
            "collection_complete": True,
            "pending_output": pending_output,
            "output": output,
            "pending_manifest": pending_meta,
            "output_locks": (),
            "provenance": {"schema": producer._SCHEMA},
            "row_count": 2,
            "position_count": 1,
            "requested_position_count": 1,
            "requested_max_positions": 1,
            "excluded_positions": [],
            "reference_censoring_details": [],
        }
        raise RuntimeError("final provenance snapshot failed")

    monkeypatch.setattr(producer, "_main", fail_after_collection)

    with pytest.raises(RuntimeError, match="final provenance snapshot failed"):
        producer.main()

    assert pending_output.read_text() == "row one\nrow two\n"
    manifest = json.loads(pending_meta.read_text())
    assert manifest["decision_grade"] is False
    assert manifest["complete"] is False
    assert manifest["trajectory_collection_complete"] is True
    assert manifest["failure_stage"] == "post_collection_finalization"
    assert manifest["finalization_error"] == {
        "type": "RuntimeError",
        "message": "final provenance snapshot failed",
    }
    assert manifest["raw_observations_preserved"] is True
    assert manifest["reference_censoring"] == censoring
    assert manifest["output"] == producer._prepared_output_artifact(
        pending_output, output,
    )


def test_producer_rejects_foreign_loaded_python_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    foreign = tmp_path / "foreign_search.py"
    foreign.write_text("FOREIGN = True\n")
    monkeypatch.setitem(
        sys.modules,
        "chess_anti_engine.foreign_review_fixture",
        SimpleNamespace(__file__=str(foreign)),
    )
    repo_root = Path(producer.__file__).resolve().parents[1]
    monkeypatch.setattr(
        producer,
        "_producer_git_file_at_commit",
        lambda _commit, relative: (
            (repo_root / relative).read_bytes()
            if (repo_root / relative).is_file() else None
        ),
    )

    with pytest.raises(
        SystemExit, match=r"chess_anti_engine\.foreign_review_fixture",
    ):
        producer._producer_python_source_artifacts("a" * 40, require_tracked=True)


@pytest.mark.parametrize("module_name", controller_module._NATIVE_MODULES)
def test_producer_rejects_native_binary_built_from_another_revision(
    module_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    repo_root = Path(producer.__file__).resolve().parents[1]
    dependencies = {
        relative_path: (repo_root / relative_path).read_bytes()
        for relative_path in controller_module.extension_spec(module_name).dependencies
    }
    copied = controller_module.native_build_attestation(
        module_name, "b" * 40, dependencies,
    )
    loaded = SimpleNamespace(
        BUILD_ATTESTATION_SCHEMA=copied["schema"],
        BUILD_MODULE_NAME=copied["module"],
        BUILD_SOURCE_GIT_SHA=copied["source_git_sha"],
        BUILD_INPUT_SHA256=copied["input_sha256"],
    )
    monkeypatch.setattr(
        producer,
        "_producer_git_file_at_commit",
        lambda _commit, relative: dependencies.get(relative),
    )

    observed = producer._loaded_native_build_attestation(
        loaded, module_name, "a" * 40,
    )

    assert observed["current_inputs_match_revision"] is True
    assert observed["matches_producer_revision"] is False


@pytest.mark.parametrize("module_name", controller_module._NATIVE_MODULES)
def test_producer_rejects_native_binary_built_from_altered_then_restored_inputs(
    module_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    repo_root = Path(producer.__file__).resolve().parents[1]
    dependencies = {
        relative_path: (repo_root / relative_path).read_bytes()
        for relative_path in controller_module.extension_spec(module_name).dependencies
    }
    built_from = dict(dependencies)
    changed_path = next(iter(built_from))
    built_from[changed_path] += b" locally modified for build"
    altered = controller_module.native_build_attestation(
        module_name, "a" * 40, built_from,
    )
    loaded = SimpleNamespace(
        BUILD_ATTESTATION_SCHEMA=altered["schema"],
        BUILD_MODULE_NAME=altered["module"],
        BUILD_SOURCE_GIT_SHA=altered["source_git_sha"],
        BUILD_INPUT_SHA256=altered["input_sha256"],
    )
    monkeypatch.setattr(
        producer,
        "_producer_git_file_at_commit",
        lambda _commit, relative: dependencies.get(relative),
    )

    observed = producer._loaded_native_build_attestation(
        loaded, module_name, "a" * 40,
    )

    assert observed["current_inputs_match_revision"] is True
    assert observed["matches_producer_revision"] is False


@pytest.mark.parametrize("module_name", controller_module._NATIVE_MODULES)
def test_producer_rejects_missing_native_build_attestation(
    module_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    repo_root = Path(producer.__file__).resolve().parents[1]
    dependencies = {
        relative_path: (repo_root / relative_path).read_bytes()
        for relative_path in controller_module.extension_spec(module_name).dependencies
    }
    monkeypatch.setattr(
        producer,
        "_producer_git_file_at_commit",
        lambda _commit, relative: dependencies.get(relative),
    )

    observed = producer._loaded_native_build_attestation(
        SimpleNamespace(), module_name, "a" * 40,
    )

    assert observed["matches_producer_revision"] is False


def test_real_producer_source_inventory_round_trips_through_analyzer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    importlib.import_module("chess_anti_engine.uci.model_loader")
    repo_root = Path(producer.__file__).resolve().parents[1]

    def tracked_bytes(_commit: str, relative: str) -> bytes | None:
        path = repo_root / relative
        return path.read_bytes() if path.is_file() else None

    monkeypatch.setattr(producer, "_producer_git_file_at_commit", tracked_bytes)
    monkeypatch.setattr(controller_module, "_git_file_at_commit", tracked_bytes)

    sources = producer._producer_python_source_artifacts(
        "a" * 40, require_tracked=True,
    )

    assert sources["scripts"]["repo_relative_path"] == "scripts/__init__.py"
    assert sources["scripts"]["size"] == 0
    assert controller_module._producer_sources_match_revision(sources, "a" * 40)


def test_producer_requires_driver_provenance_before_decision_grade_search() -> None:
    from scripts import backtest_chunk_trajectory as producer

    with pytest.raises(RuntimeError, match="NVIDIA driver provenance"):
        producer._require_nvidia_driver_provenance(None, methodology_smoke=False)
    producer._require_nvidia_driver_provenance("600.1", methodology_smoke=False)
    producer._require_nvidia_driver_provenance(None, methodology_smoke=True)


@pytest.mark.parametrize(
    "name",
    [".bank.jsonl.lock", ".bank.jsonl.tmp-12345", "..tmp-bank.tmp-12345"],
)
def test_producer_output_rejects_internal_output_namespaces(
    tmp_path: Path, name: str,
) -> None:
    with pytest.raises(SystemExit, match="lock/staging namespace"):
        _require_safe_output_paths(
            tmp_path / name,
            tmp_path / "out.meta.json",
            protected_files=[],
            protected_directories=[],
        )


def test_analyzer_output_cannot_replace_producer_staging_file(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"

    with pytest.raises(ValueError, match="lock/staging namespace"):
        _require_safe_output_path(bank, meta, tmp_path / ".bank.jsonl.tmp-12345")


def test_hidden_output_staging_name_is_reserved(tmp_path: Path) -> None:
    from scripts.repo_output_guard import reserved_output_path

    output = tmp_path / ".tmp-bank"
    staging = output.with_name(f".{output.name}.tmp-12345")

    assert staging.name == "..tmp-bank.tmp-12345"
    assert reserved_output_path(staging)


def test_internal_namespace_guard_checks_symlink_lexical_name(tmp_path: Path) -> None:
    target = tmp_path / "ordinary-name"
    target.write_text("existing\n")
    alias = tmp_path / ".bank.jsonl.tmp-12345"
    alias.symlink_to(target)

    with pytest.raises(SystemExit, match="lock/staging namespace"):
        _require_safe_output_paths(
            alias,
            tmp_path / "out.meta.json",
            protected_files=[],
            protected_directories=[],
        )
    with pytest.raises(ValueError, match="lock/staging namespace"):
        _require_safe_output_path(
            tmp_path / "bank.jsonl",
            tmp_path / "bank.jsonl.meta.json",
            alias,
        )


def test_publish_output_detects_replacement_before_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    private = tmp_path / ".bank.tmp"
    output = tmp_path / "bank.jsonl"
    private.write_text("producer bytes\n")
    real_link = producer.os.link

    def replacing_publish(source: Path, destination: Path) -> None:
        real_link(source, destination)
        destination.write_text("other producer bytes\n")

    monkeypatch.setattr(producer.os, "link", replacing_publish)

    with pytest.raises(RuntimeError, match="differs from its private output"):
        _publish_output(private, output)


def test_publish_output_never_clobbers_a_destination_that_appeared(
    tmp_path: Path,
) -> None:
    private = tmp_path / ".bank.tmp"
    output = tmp_path / "bank.jsonl"
    private.write_text("new bank\n")
    output.write_text("existing bank\n")

    with pytest.raises(RuntimeError, match="immutable evidence"):
        _publish_output(private, output)

    assert output.read_text() == "existing bank\n"
    assert private.read_text() == "new bank\n"


def test_manifest_publication_never_clobbers_an_existing_manifest(
    tmp_path: Path,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    manifest = tmp_path / "bank.jsonl.meta.json"
    manifest.write_text("old manifest\n")

    with pytest.raises(RuntimeError, match="immutable evidence"):
        producer._write_json_atomic(manifest, {"new": True})

    assert manifest.read_text() == "old manifest\n"


def test_producer_applies_shared_uci_search_ranges() -> None:
    _validate_registry_search_values(
        "walker_puct", {"chunk_sims": 32, "c_puct": 1.75},
    )
    with pytest.raises(SystemExit, match="chunk_sims"):
        _validate_registry_search_values(
            "walker_puct", {"chunk_sims": 1, "c_puct": 1.75},
        )


def test_producer_rejects_chunk_size_before_checkpoint_or_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    monkeypatch.setattr(
        sys,
        "argv",
        ["backtest_chunk_trajectory", "--checkpoint", "unused", "--chunk-sims", "1"],
    )
    monkeypatch.setattr(
        producer,
        "_checkpoint_file",
        lambda _path: pytest.fail("checkpoint resolution ran before range validation"),
    )

    with pytest.raises(SystemExit, match="production UCI registry"):
        producer.main()


def test_decision_grade_requires_enough_bootstrap_samples() -> None:
    _require_bootstrap_resolution(1, methodology_smoke=True)
    _require_bootstrap_resolution(1000, methodology_smoke=False)
    with pytest.raises(ValueError, match="at least 1000"):
        _require_bootstrap_resolution(999, methodology_smoke=False)


def test_evidence_verdict_is_scoped_to_the_fresh_tree_screen() -> None:
    from scripts.analyze_chunk_controller import _evidence_verdict, _is_canonical_decision_rule

    assert _is_canonical_decision_rule(
        n_folds=5, bootstrap_samples=2000, seed=0,
        allocation_fraction=0.2, min_capture_gain=0.05,
        min_oracle_headroom=1e-4, min_bootstrap_valid_fraction=0.95,
    ) is True
    assert _is_canonical_decision_rule(
        n_folds=5, bootstrap_samples=2000, seed=1,
        allocation_fraction=0.2, min_capture_gain=0.05,
        min_oracle_headroom=1e-4, min_bootstrap_valid_fraction=0.95,
    ) is False

    assert _evidence_verdict(
        evidence_inputs_decision_grade=True,
        canonical_preregistered_rule=True,
        source_group_resolution_passed=True,
        statistical_gate_passed=True,
    ) == "ADVANCE_TO_CLOCK_HISTORY_REUSED_TREE_BANK"
    assert _evidence_verdict(
        evidence_inputs_decision_grade=True,
        canonical_preregistered_rule=True,
        source_group_resolution_passed=True,
        statistical_gate_passed=False,
    ) == "NO_ADVANCE_FROM_FRESH_TREE_FIXED_NODE_SCREEN"
    assert _evidence_verdict(
        evidence_inputs_decision_grade=False,
        canonical_preregistered_rule=True,
        source_group_resolution_passed=False,
        statistical_gate_passed=True,
    ) == "METHODOLOGY_SMOKE_ONLY"
    assert _evidence_verdict(
        evidence_inputs_decision_grade=True,
        canonical_preregistered_rule=False,
        source_group_resolution_passed=False,
        statistical_gate_passed=True,
    ) == "NONCANONICAL_RULE_DIAGNOSTIC_ONLY"
    assert _evidence_verdict(
        evidence_inputs_decision_grade=True,
        canonical_preregistered_rule=True,
        source_group_resolution_passed=False,
        statistical_gate_passed=False,
    ) == "INSUFFICIENT_SOURCE_GAME_GROUPS"


def test_analyzer_main_skips_grouped_analysis_for_undersized_bank(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    one_game = [_transition("position", 1, 100, 0.0, current=True)]
    info = {
        "decision_grade": True,
        "preregistered_design": True,
        "manifest": {"producer_git_sha": "a" * 40},
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_chunk_controller",
            "--in", str(tmp_path / "bank.jsonl"),
            "--meta", str(tmp_path / "bank.jsonl.meta.json"),
        ],
    )
    monkeypatch.setattr(controller_module, "_analyzer_source_artifacts", dict)
    monkeypatch.setattr(controller_module, "_git_state", lambda: ("b" * 40, False))
    monkeypatch.setattr(controller_module, "_require_safe_output_path", lambda *_a, **_k: None)
    monkeypatch.setattr(
        controller_module,
        "load_transitions",
        lambda *_a, **_k: (one_game, info),
    )
    monkeypatch.setattr(
        controller_module,
        "analyze",
        lambda *_a, **_k: pytest.fail("undersized bank entered grouped analysis"),
    )
    monkeypatch.setattr(
        controller_module,
        "_analyzer_provenance",
        lambda *_a: {
            "decision_grade": True,
            "git_sha": "b" * 40,
            "final_git_sha": "b" * 40,
        },
    )

    controller_module.main()

    payload = json.loads(capsys.readouterr().out)
    analysis = payload["analysis"]
    assert analysis["verdict"] == "INSUFFICIENT_SOURCE_GAME_GROUPS"
    assert analysis["analysis_skipped"] == "insufficient_source_game_groups"
    assert analysis["source_game_group_count"] == 1
    assert analysis["grouped_analysis_possible"] is False
    assert analysis["source_group_resolution_passed"] is False
    assert analysis["evidence_decision_grade"] is False


def test_analyzer_main_runs_grouped_analysis_for_two_game_smoke_bank(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    two_games = [
        _transition(key, game, horizon, float(game), current=True)
        for key, game in (("first", 1), ("second", 2))
        for horizon in (100, 150, 200)
    ]
    info = {
        "decision_grade": False,
        "preregistered_design": False,
        "manifest": {"producer_git_sha": "a" * 40},
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_chunk_controller",
            "--in", str(tmp_path / "bank.jsonl"),
            "--meta", str(tmp_path / "bank.jsonl.meta.json"),
            "--methodology-smoke",
            "--folds", "2",
            "--bootstrap-samples", "1",
            "--allocation-fraction", "0.5",
        ],
    )
    monkeypatch.setattr(controller_module, "_analyzer_source_artifacts", dict)
    monkeypatch.setattr(controller_module, "_git_state", lambda: ("b" * 40, False))
    monkeypatch.setattr(controller_module, "_require_safe_output_path", lambda *_a, **_k: None)
    monkeypatch.setattr(
        controller_module,
        "load_transitions",
        lambda *_a, **_k: (two_games, info),
    )
    monkeypatch.setattr(
        controller_module,
        "_analyzer_provenance",
        lambda *_a: {
            "decision_grade": True,
            "git_sha": "b" * 40,
            "final_git_sha": "b" * 40,
        },
    )

    controller_module.main()

    analysis = json.loads(capsys.readouterr().out)["analysis"]
    assert analysis["verdict"] == "METHODOLOGY_SMOKE_ONLY"
    assert "reachable_rollout" in analysis
    assert analysis["source_game_group_count"] == 2
    assert analysis["grouped_analysis_possible"] is True
    assert analysis["source_group_resolution_passed"] is False


def test_grouped_analysis_requires_multiple_held_horizon_training_rows() -> None:
    two_games_one_horizon = [
        _transition("first", 1, 100, 0.0, current=True),
        _transition("second", 2, 100, 0.0, current=True),
    ]

    assert controller_module._grouped_analysis_possible(
        two_games_one_horizon, 2,
    ) is False


def test_analyzer_main_reports_nonrectangular_smoke_bank_without_entering_rollout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    uneven = [
        _transition(key, game, horizon, float(game), current=True)
        for key, game, horizons in (
            ("first", 1, (100, 150, 200)),
            ("second", 2, (100, 150, 200)),
            ("short", 3, (100, 150)),
        )
        for horizon in horizons
    ]
    info = {
        "decision_grade": False,
        "preregistered_design": False,
        "manifest": {"producer_git_sha": "a" * 40},
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_chunk_controller",
            "--in", str(tmp_path / "bank.jsonl"),
            "--meta", str(tmp_path / "bank.jsonl.meta.json"),
            "--methodology-smoke",
            "--folds", "2",
            "--bootstrap-samples", "1",
            "--allocation-fraction", "0.5",
        ],
    )
    monkeypatch.setattr(controller_module, "_analyzer_source_artifacts", dict)
    monkeypatch.setattr(controller_module, "_git_state", lambda: ("b" * 40, False))
    monkeypatch.setattr(controller_module, "_require_safe_output_path", lambda *_a, **_k: None)
    monkeypatch.setattr(
        controller_module,
        "load_transitions",
        lambda *_a, **_k: (uneven, info),
    )
    monkeypatch.setattr(
        controller_module,
        "analyze",
        lambda *_a, **_k: pytest.fail("nonrectangular bank entered grouped rollout"),
    )
    monkeypatch.setattr(
        controller_module,
        "_analyzer_provenance",
        lambda *_a: {
            "decision_grade": True,
            "git_sha": "b" * 40,
            "final_git_sha": "b" * 40,
        },
    )

    controller_module.main()

    analysis = json.loads(capsys.readouterr().out)["analysis"]
    assert analysis["analysis_skipped"] == "nonrectangular_key_by_horizon_layout"
    assert analysis["grouped_analysis_possible"] is False
    assert analysis["grouped_analysis_preflight"] == {
        "passed": False,
        "reasons": ["nonrectangular_key_by_horizon_layout"],
        "source_game_group_count": 3,
        "horizon_count": 3,
    }


def test_analyze_cannot_advance_with_an_undersampled_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import analyze_chunk_controller as controller

    rows = [
        _transition(key, game, horizon, gain, current=gain > 0.0)
        for key, game, gain in (
            ("a", 1, -1.0),
            ("b", 2, -1.0),
            ("c", 3, 1.0),
            ("d", 4, 1.0),
        )
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
    assert result["statistical_gate_passed"] is False
    assert "verdict" not in result

    positive = analyze(
        rows,
        n_folds=2,
        bootstrap_samples=1000,
        seed=0,
        allocation_fraction=0.5,
        min_capture_gain=0.0,
        min_oracle_headroom=0.0,
        min_bootstrap_valid_fraction=0.5,
    )

    assert positive["statistical_gate_passed"] is True
    assert positive["evaluated_rule"]["grouped_cv_folds"] == 2
    assert positive["evaluated_rule"]["bootstrap_seed"] == 0
    assert "verdict" not in positive


def test_analyze_requires_M1_improvement_at_every_reachable_rung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import analyze_chunk_controller as controller

    rows = [
        _transition("a", 1, 100, 0.0, current=False),
        _transition("a", 1, 150, 0.0, current=False),
        _transition("b", 2, 100, 0.0, current=False),
        _transition("b", 2, 150, 0.0, current=False),
        _transition("c", 3, 100, 0.0, current=False),
        _transition("c", 3, 150, -1.0, current=True),
        _transition("d", 4, 100, 10.0, current=True),
        _transition("d", 4, 150, 0.0, current=False),
    ]
    m0 = np.zeros(len(rows), dtype=np.float64)
    m1 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 10.0, 0.0])

    monkeypatch.setattr(
        controller,
        "held_horizon_predictions",
        lambda _rows, model, *, n_folds=5: (m0 if model == "M0" else m1, []),
    )
    monkeypatch.setattr(
        controller,
        "cluster_bootstrap_delta",
        lambda *_args, **_kwargs: {
            "requested_samples": 1000, "valid_samples": 1000,
            "valid_fraction": 1.0, "mean": 1.0, "lower_95": 1.0,
            "upper_95": 1.0,
        },
    )

    result = analyze(
        rows, n_folds=2, bootstrap_samples=1000, seed=4,
        allocation_fraction=0.5, min_capture_gain=0.0,
        min_oracle_headroom=0.0, min_bootstrap_valid_fraction=0.5,
    )

    assert result["M1_minus_M0_oracle_capture"] > 0.0
    assert result["reachable_stage_rule_passed"] is False
    assert result["statistical_gate_passed"] is False


def test_analyzer_provenance_requires_a_stable_clean_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import analyze_chunk_controller as controller

    repo_root = Path(controller.__file__).resolve().parents[1]
    sources = controller._analyzer_source_artifacts()
    monkeypatch.setattr(controller, "_analyzer_source_artifacts", lambda: sources)
    monkeypatch.setattr(controller, "_git_state", lambda: ("b" * 40, False))
    monkeypatch.setattr(
        controller,
        "_git_file_at_commit",
        lambda _commit, relative_path: (repo_root / relative_path).read_bytes(),
    )

    clean = controller._analyzer_provenance(sources, "b" * 40, False)
    dirty = controller._analyzer_provenance(sources, "b" * 40, True)

    assert clean["decision_grade"] is True
    assert clean["sources_match_git_revision"] is True
    assert clean["numpy_version"]
    assert clean["python_chess_version"]
    assert dirty["decision_grade"] is False


def test_analyzer_provenance_rejects_helper_from_another_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import analyze_chunk_controller as controller

    repo_root = Path(controller.__file__).resolve().parents[1]
    sources = controller._analyzer_source_artifacts()
    foreign_oracle = tmp_path / "reachable_oracle.py"
    foreign_oracle.write_bytes(
        Path(controller.solve_reachable_oracle.__code__.co_filename).read_bytes()
    )
    sources["scripts.reachable_oracle"] = controller._artifact_snapshot(foreign_oracle)
    monkeypatch.setattr(controller, "_analyzer_source_artifacts", lambda: sources)
    monkeypatch.setattr(controller, "_git_state", lambda: ("b" * 40, False))
    monkeypatch.setattr(
        controller,
        "_git_file_at_commit",
        lambda _commit, relative_path: (repo_root / relative_path).read_bytes(),
    )

    provenance = controller._analyzer_provenance(sources, "b" * 40, False)

    assert provenance["sources_stable"] is True
    assert provenance["sources_match_git_revision"] is False
    assert provenance["decision_grade"] is False
    assert provenance["source_revision_bindings"]["scripts.reachable_oracle"] == {
        "repo_relative_path": None,
        "matches_reported_git_revision": False,
    }


def test_analyzer_revision_is_authenticated_independently_of_bank_producer() -> None:
    analyzer = {
        "decision_grade": True,
        "git_sha": "b" * 40,
        "sources": {"analyzer": {"sha256": "c" * 64}},
    }

    assert controller_module._decision_grade_evidence_inputs(
        bank_decision_grade=True,
        analyzer_provenance=analyzer,
    ) is True


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


def test_loader_rejects_internally_contradictory_model_search_contract(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    for name in ("requested_model_search_contract", "realized_model_search_contract"):
        manifest[name]["model_input_history_encoding"] = "lc0_root"
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="model/search encoding contract"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value"),
    [("passed", False), ("changed", ["checkpoint"]), ("final_git_dirty", True)],
)
def test_loader_requires_stable_consumed_artifacts_and_checkout(
    tmp_path: Path, field: str, value: object,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["artifact_stability"][field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="changed during collection"):
        load_transitions(bank)


def test_loader_authenticates_and_parses_one_bank_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    original_read_bytes = Path.read_bytes
    reads: dict[Path, int] = {}

    def racing_read_bytes(path: Path) -> bytes:
        resolved = path.resolve()
        reads[resolved] = reads.get(resolved, 0) + 1
        payload = original_read_bytes(path)
        if resolved == bank.resolve():
            path.write_text('{"replacement": true}\n')
        return payload

    monkeypatch.setattr(Path, "read_bytes", racing_read_bytes)

    transitions, _ = load_transitions(bank)

    assert len(transitions) == 3
    assert reads[bank.resolve()] == 1
    assert reads[meta.resolve()] == 1


def test_loader_requires_loaded_cboard_native_provenance(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["lc0_extension"]["cboard_encode_full"] = False
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="CBoard encoding extension"):
        load_transitions(bank)


def test_analyzer_output_cannot_alias_loaded_cboard_extension(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())

    with pytest.raises(ValueError, match="consumed input artifact"):
        _require_safe_output_path(
            bank, meta, Path("/_lc0_ext.so"), manifest=manifest,
        )


def test_analyzer_output_cannot_overwrite_imported_oracle_source(tmp_path: Path) -> None:
    from scripts import reachable_oracle

    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())

    with pytest.raises(ValueError, match="tracked or repository-control"):
        _require_safe_output_path(
            bank, meta, Path(reachable_oracle.__file__), manifest=manifest,
        )


def test_loader_requires_exact_resolved_cuda_device_identity(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["runtime"]["resolved_model_parameter_devices"] = ["cuda:1"]
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="CUDA runtime/device provenance"):
        load_transitions(bank)


def test_loader_rejects_raw_devices_that_contradict_resolved_identity(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["runtime"]["evaluator_device"] = "cuda:1"
    manifest["runtime"]["model_parameter_devices"] = ["cuda:1"]
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="CUDA runtime/device provenance"):
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
            str(game), game, horizon, (game - 5) / horizon,
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
        _transition(str(game), game, horizon, 0.0, current=True)
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
