"""CPU-only finite cross-unit actors; canonical science and aggregate geometry."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

from scripts import bt4_root_policy_worker as worker


@pytest.fixture(autouse=True)
def native_history() -> Any:
    previous = worker.rep_fix.current()
    worker.rep_fix.apply(True, boards_discarded=True)
    yield
    worker.rep_fix.apply(previous is True, boards_discarded=True)


def fixture_module() -> Any:
    spec = importlib.util.spec_from_file_location("unit_fixture", Path(__file__).with_name("test_bt4_root_policy_worker.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_units(tmp_path: Path, *, games: int = 2) -> tuple[Any, Any, Any, dict[str, Any]]:
    fixture = fixture_module()
    root = tmp_path / "fixture"
    root.mkdir()
    original = fixture.spec(root)
    out = tmp_path / "coordinator"
    handle = fixture.fake_tablebase()
    units = {f"unit_{i}": (replace(original, out=out / f"unit_{i}", games=games, seed=14 + i * 17), handle)
             for i in range(2)}
    args = {"max_units": 2, "max_live_games": 4, "target_rows": 4, "max_rows": 8,
            "batch_wait_ms": 100, "max_writes": 1, "deadline_seconds": 30}
    return fixture, out, units, args


def test_cross_unit_four_ply_raw_and_rng_parity_and_exact_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, out, original, args = small_units(tmp_path, games=6)
    moves = ("f2f3", "e7e5", "g2g4", "d8h4")
    sizes = []
    class MateEvaluator:
        def evaluate_roots(self, boards: list[chess.Board], x_batch: np.ndarray) -> list[Any]:
            sizes.append(len(boards))
            return [fixture.FakeEvaluator(moves[board.ply()]).evaluate_roots([board], x[None])[0]
                    for board, x in zip(boards, x_batch)]
    states = []
    canonical = worker.BT4RootPolicyStepper
    class TrackedStepper(canonical):
        def __init__(self, boards: Any, rngs: Any, **kwargs: Any) -> None:
            super().__init__(boards, rngs, **kwargs)
            states.append(rngs)
    monkeypatch.setattr(worker, "BT4RootPolicyStepper", TrackedStepper)
    units = {name: (replace(spec, initial_fen=chess.STARTING_FEN, temperature=1, max_plies=8), handle)
             for name, (spec, handle) in original.items()}
    expected = {}
    evaluator = MateEvaluator()
    for name, (spec, handle) in units.items():
        start = len(states)
        control = replace(spec, out=tmp_path / (name + "_control"), parallel_games=1)
        worker.run_worker(control, evaluator, handle)
        expected[name] = (control, {i: json.dumps(rng.bit_generator.state, sort_keys=True)
                                   for group in states[start:] for i, rng in group.items()})
    states.clear()
    sizes.clear()
    canonical_pool = worker._make_pooled_unit
    unit_rngs: dict[str, dict[int, Any]] = {}
    def make_pool(*values: Any, **options: Any) -> Any:
        pool = canonical_pool(*values, **options)
        make_stepper = pool.make_stepper
        def make(game_id: int, rng: Any) -> Any:
            unit_rngs.setdefault(pool.namespace, {})[game_id] = rng
            return make_stepper(game_id, rng)
        pool.make_stepper = make
        return pool
    monkeypatch.setattr(worker, "_make_pooled_unit", make_pool)
    actual = worker.run_pooled_units(units, evaluator, out, **args)
    assert actual["status"] == "COMPLETE_NOT_ADMISSION"
    assert actual["peak_live_games"] == 4
    assert max(sizes) == 4  # greater than either canonical unit's capacity
    # Pools construct game0/1 per unit, then independent refill. Bind tracked RNG
    # states to each unit's exact SeedSequence rather than completion order.
    assert {name: {i: json.dumps(rng.bit_generator.state, sort_keys=True) for i, rng in values.items()}
            for name, values in unit_rngs.items()} == {name: values for name, (_, values) in expected.items()}
    for name, receipts in actual["unit_receipts"].items():
        assert [receipt["game_id"] for receipt in receipts] == list(range(6))
        for receipt in receipts:
            meta, arrays = fixture.read_game(units[name][0].out / "games" / receipt["path"])
            control_meta, control_arrays = fixture.read_game(expected[name][0].out / "games" / receipt["path"])
            assert meta == control_meta
            assert all(np.array_equal(arrays[key], control_arrays[key]) for key in arrays)
            assert [row["move_uci"] for row in meta["rows"]] == list(moves)
            assert all(row["teacher"]["history_rep_fix"] is True for row in meta["rows"])
    sizes.clear()
    resumed = worker.run_pooled_units(units, evaluator, out, resume=True, **args)
    assert not sizes
    assert resumed["logical_batch_histogram"] == {}
    assert resumed["unit_receipts"] == actual["unit_receipts"]
    changed = dict(units)
    spec, handle = changed["unit_1"]
    changed["unit_1"] = replace(spec, seed=123), handle
    with pytest.raises(ValueError, match="resume"):
        worker.run_pooled_units(changed, evaluator, out, resume=True, **args)
    with pytest.raises(ValueError, match="roster"):
        worker.run_pooled_units(dict(reversed(list(units.items()))), evaluator, out, resume=True, **args)


@pytest.mark.parametrize("rows", [512, 768, 1024])
def test_real_research_units_admit_distinct_full_actor_batches_then_stop(tmp_path: Path, rows: int) -> None:
    fixture = fixture_module()
    original = fixture.spec(tmp_path)
    out = tmp_path / "coordinator"
    count = rows // 64
    handle = fixture.fake_tablebase()
    units = {f"unit_{i}": (replace(original, out=out / f"unit_{i}", seed=14 + i,
                                  games=128, parallel_games=64, max_plies=400,
                                  research_capacity_128x400=True), handle) for i in range(count)}
    sizes = []
    class StopEvaluator(fixture.FakeEvaluator):
        def evaluate_roots(self, boards: Any, x_batch: np.ndarray) -> Any:
            sizes.append(len(boards))
            assert len({id(board) for board in boards}) == rows
            assert x_batch.shape == (rows, 175, 8, 8)
            values = super().evaluate_roots(boards, x_batch)
            (out / "STOP").write_text("CPU geometry fixture; stop before move/RNG application\n")
            return values
    result = worker.run_pooled_units(
        units, StopEvaluator(), out, max_units=count, max_live_games=rows,
        target_rows=rows, max_rows=rows, batch_wait_ms=20000,
        max_writes=1, deadline_seconds=60,
    )
    assert sizes == [rows]
    assert result["logical_batch_histogram"] == {rows: 1}
    assert result["peak_live_games"] == rows
    assert result["status"] == "PAUSED_NOT_ADMISSION"
    assert all(not receipts for receipts in result["unit_receipts"].values())
    assert not list(out.glob("unit_*/games/*.npz"))
    roster = json.loads((out / "cross_unit_launch.json").read_text())["units"]
    assert len(roster) == count
    assert all(json.loads((spec.out / "launch.json").read_text())["games"] == 128 for spec, _ in units.values())


def test_cross_unit_stop_restart_and_postpublication_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, out, units, args = small_units(tmp_path)
    class StopEvaluator(fixture.FakeEvaluator):
        def evaluate_roots(self, boards: Any, x_batch: Any) -> Any:
            result = super().evaluate_roots(boards, x_batch)
            (out / "STOP").write_text("hold admitted CPU results\n")
            return result
    paused = worker.run_pooled_units(units, StopEvaluator(), out, **args)
    assert paused["status"] == "PAUSED_NOT_ADMISSION"
    assert not list(out.glob("unit_*/games/*.npz"))
    (out / "STOP").unlink()
    original = worker._atomic_json
    failed = []
    def fail_checkpoint(path: Path, document: Any) -> None:
        if path.name.endswith("checkpoint.json") and not failed:
            failed.append(path)
            raise OSError("raw published before checkpoint acknowledgment")
        original(path, document)
    with monkeypatch.context() as patch:
        patch.setattr(worker, "_atomic_json", fail_checkpoint)
        with pytest.raises(OSError, match="raw published"):
            worker.run_pooled_units(units, fixture.FakeEvaluator(), out, resume=True, **args)
    raw_before = {path: worker.file_sha256(path) for path in out.glob("unit_*/games/*.npz")}
    assert raw_before
    resumed = worker.run_pooled_units(units, fixture.FakeEvaluator(), out, resume=True, **args)
    assert all(worker.file_sha256(path) == sha for path, sha in raw_before.items())
    assert all([receipt["game_id"] for receipt in receipts] == [0, 1]
               for receipts in resumed["unit_receipts"].values())
    assert len(list(out.glob("unit_*/games/*.npz"))) == 4
    assert len(list(out.glob("unit_*/games/*.checkpoint.json"))) == 4


def test_cross_unit_preflight_rejects_unbounded_or_overlapping_units(tmp_path: Path) -> None:
    fixture, out, units, args = small_units(tmp_path)
    actors: dict[str, Any] = {**args, "max_live_games": 3}
    budget: dict[str, Any] = {**args, "max_units": 1}
    with pytest.raises(ValueError, match="aggregate"):
        worker.run_pooled_units(units, fixture.FakeEvaluator(), out, **actors)
    with pytest.raises(ValueError, match="budgets"):
        worker.run_pooled_units(units, fixture.FakeEvaluator(), out, **budget)
    overlapping = {"unit_0": units["unit_0"], "unit_1": units["unit_0"]}
    with pytest.raises(ValueError, match="nonoverlapping"):
        worker.run_pooled_units(overlapping, fixture.FakeEvaluator(), out, **args)
    changed = dict(units)
    spec, handle = units["unit_1"]
    changed["unit_1"] = replace(spec, model_sha256="b" * 64), handle
    with pytest.raises(ValueError, match="shared teacher"):
        worker.run_pooled_units(changed, fixture.FakeEvaluator(), out, **args)
    assert not out.exists()


def test_cross_unit_asymmetric_lengths_progress_under_row_backpressure(tmp_path: Path) -> None:
    fixture, out, original, args = small_units(tmp_path, games=7)
    moves = ("f2f3", "e7e5", "g2g4", "d8h4")
    units = dict(original)
    spec, handle = units["unit_1"]
    units["unit_1"] = replace(spec, initial_fen=chess.STARTING_FEN, temperature=1), handle
    calls = []
    class MixedEvaluator:
        def evaluate_roots(self, boards: list[chess.Board], x_batch: np.ndarray) -> list[Any]:
            calls.append(tuple(board.ply() for board in boards))
            return [fixture.FakeEvaluator("b1c2" if len(board.piece_map()) == 7 else moves[board.ply()])
                    .evaluate_roots([board], x[None])[0] for board, x in zip(boards, x_batch)]
    evaluator = MixedEvaluator()
    controls = {}
    for name, (spec, handle) in units.items():
        control = replace(spec, out=tmp_path / (name + "_control"), parallel_games=1)
        worker.run_worker(control, evaluator, handle)
        controls[name] = control
    calls.clear()
    limited: dict[str, Any] = {**args, "target_rows": 1, "max_rows": 1, "batch_wait_ms": 0}
    frozen = dict(units)
    def mutate_caller_roster() -> None:
        units.clear()
    result = worker.run_pooled_units(units, evaluator, out, control=mutate_caller_roster, **limited)
    units = frozen
    assert result["status"] == "COMPLETE_NOT_ADMISSION"
    assert calls
    assert all(len(call) == 1 for call in calls)
    assert result["logical_batch_histogram"] == {1: 35}
    for name, receipts in result["unit_receipts"].items():
        assert [receipt["game_id"] for receipt in receipts] == list(range(7))
        for receipt in receipts:
            metadata, arrays = fixture.read_game(units[name][0].out / "games" / receipt["path"])
            expected, expected_arrays = fixture.read_game(controls[name].out / "games" / receipt["path"])
            assert metadata == expected
            assert all(np.array_equal(arrays[key], expected_arrays[key]) for key in arrays)
