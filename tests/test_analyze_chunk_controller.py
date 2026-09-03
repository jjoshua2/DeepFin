from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_chunk_controller import (
    Transition,
    _require_bootstrap_resolution,
    _require_safe_output_path,
    _update_stability,
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
    assert "realized_search_values" in source
    assert "_require_search_take_effect" in source


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
    m0 = np.asarray([row.gain * 0.2 for row in rows])
    m1 = np.asarray([row.gain for row in rows])
    del m0, m1
    alpha_profile = {100: 1.0, 150: 1.0}

    first = cluster_bootstrap_delta(
        rows,
        m0_alpha_profile=alpha_profile,
        m1_alpha_profile=alpha_profile,
        allocation_fraction=0.5, samples=50, seed=7,
    )
    second = cluster_bootstrap_delta(
        rows,
        m0_alpha_profile=alpha_profile,
        m1_alpha_profile=alpha_profile,
        allocation_fraction=0.5, samples=50, seed=7,
    )
    assert first == second


def _write_bank(path: Path, *, correct_gap: bool) -> Path:
    rows = []
    for chunk, regret in ((1, 0.2), (2, 0.1)):
        rows.append({
            "schema": "deepfin.chunk_trajectory.v2",
            "key": "k", "source_dir": "/snapshot", "shard": "s0.zarr",
            "game_id": 3, "group_id": "/snapshot\0s0.zarr\0" + "3",
            "chunk": chunk, "nodes": chunk * 50,
            "regret_score": regret, "visit_gap": -0.2 if correct_gap else 0.2,
            "root_actions": [1, 2], "root_visit_shares": [0.4, 0.6],
            "emitted_action": 1, "uci": "a2a3", "bestmove_flip": False,
            "stable_chunks": chunk - 1, "visit_entropy": 0.6, "q_gap": None,
            "complexity_predicate_continue": True,
            "q_drift": None if chunk == 1 else 0.0,
            "visit_churn": None if chunk == 1 else 0.0, "root_q": 0.0,
            "phase": 1, "piece_count": 10, "legal_move_count": 5,
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
        },
        "producer_git_sha": "a" * 40,
        "producer_git_dirty": False,
        "producer_script": {
            "path": "/producer.py", "size": 1, "mtime_ns": 1, "sha256": "a" * 64,
        },
        "checkpoint": {
            "path": "/trainer.pt", "size": 1, "mtime_ns": 1, "sha256": "b" * 64,
        },
        "audit_set": {
            "path": "/audit.jsonl", "size": 1, "mtime_ns": 1, "sha256": "c" * 64,
        },
        "matched_rows": {
            "path": "/matched.npz", "size": 1, "mtime_ns": 1, "sha256": "d" * 64,
        },
        "mcts_extension": {
            "path": "/_mcts_tree.so", "size": 1, "mtime_ns": 1,
            "sha256": "e" * 64, "abi_version": 9, "required_abi_version": 9,
        },
        "syzygy": {
            "path": "/tb/a:/tb/b",
            "directories": [
                {"path": "/tb/a", "rtbw_count": 1, "rtbz_count": 1,
                 "total_bytes": 1, "inventory_sha256": "f" * 64},
                {"path": "/tb/b", "rtbw_count": 1, "rtbz_count": 1,
                 "total_bytes": 1, "inventory_sha256": "1" * 64},
            ],
        },
        "row_count": 2,
        "chunk_count": 2,
        "position_count": 1,
        "requested_position_count": 1,
        "excluded_position_count": 0,
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
        },
        "realized_evaluator": {
            "stack": (
                "BatchCoalescingDispatcher("
                "ThreadSafeGPUDispatcher(DirectGPUEvaluator))"
            ),
            "direct_max_batch": 256, "outer_max_batch": 256, "n_slots": 2,
            "input_bf16": False, "legal_bf16": False,
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


def test_loader_rejects_nonfinite_or_nonnumeric_regret(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows[-1]["regret_score"] = "bad"
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="regret_score must be a finite number"):
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

    def counted_fit(x: np.ndarray, y: np.ndarray, alpha: float):
        nonlocal calls
        calls += 1
        return original(x, y, alpha)

    monkeypatch.setattr(controller, "_fit_ridge", counted_fit)
    controller.cluster_bootstrap_delta(
        rows,
        m0_alpha_profile={100: 1.0, 150: 1.0},
        m1_alpha_profile={100: 1.0, 150: 1.0},
        allocation_fraction=0.5,
        samples=5,
        seed=11,
    )

    assert calls >= 4, "each usable replicate must refit both models"


def test_bootstrap_fails_closed_when_oracle_headroom_is_undefined() -> None:
    rows = [
        _transition(f"{game}-{horizon}", game, horizon, 0.0, current=True)
        for game in range(6)
        for horizon in (100, 150)
    ]
    result = cluster_bootstrap_delta(
        rows,
        m0_alpha_profile={100: 1.0, 150: 1.0},
        m1_alpha_profile={100: 1.0, 150: 1.0},
        allocation_fraction=0.5,
        samples=25,
        seed=3,
        min_oracle_headroom=1e-4,
    )

    assert result["requested_samples"] == 25
    assert result["valid_samples"] == 0
    assert result["valid_fraction"] == 0.0
    assert result["lower_95"] is None
