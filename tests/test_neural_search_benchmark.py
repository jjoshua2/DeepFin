from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.neural_work import NeuralWorkLedger
from scripts.bench_neural_search import PREFIX, parse_report, run, search


def _lines(*, rows: int = 3, physical: int = 8, simulations: int = 7, wall: float = 1.0) -> list[str]:
    ledger = NeuralWorkLedger()
    ledger.submit(0, rows)
    ledger.dispatch(0, (0,), physical_rows=physical)
    ledger.complete(0)
    ledger.resolve(0, "accepted")
    ledger.simulations_completed(simulations)
    report = ledger.snapshot(wall)
    report["stop_code"] = 0
    return [PREFIX + json.dumps(report), f"info nodes {simulations} string test", "bestmove e2e4"]


def test_fixed_budget_is_neither_forward_calls_simulations_nor_padding() -> None:
    lines = _lines()
    report = parse_report(lines, kind="evals", budget=3)
    assert report["comparable"]
    assert report["counters"]["forward_calls"] == 1
    for wrong in (1, 7, 8):
        assert not parse_report(lines, kind="evals", budget=wrong)["comparable"]


@pytest.mark.parametrize(("field", "value", "message"), [
    ("accepted_neural_rows", 8, "contradict"),
    ("executed_real_rows", 8, "contradict"),
    ("useful_eps", 8, "mismatch"),
    ("executed_wasted_rows", 8, "contradict"),
    ("padded_rows", 0, "padding"),
    ("forward_calls", 3, "histogram"),
    ("completed_simulations", 8, "nodes"),
    ("wall_seconds", float("nan"), "wall_seconds"),
    ("executed_real_rows", True, "counter"),
])
def test_malformed_report_cannot_be_success(field: str, value: Any, message: str) -> None:
    lines = _lines()
    report = json.loads(lines[0][len(PREFIX):])
    report[field] = value
    lines[0] = PREFIX + json.dumps(report)
    with pytest.raises(ValueError, match=message):
        parse_report(lines, kind="evals", budget=3)


def test_missing_duplicate_and_unknown_reports_fail_closed() -> None:
    lines = _lines()
    for broken in (lines[1:], [*lines, lines[0]]):
        with pytest.raises(ValueError, match="exactly one"):
            parse_report(broken, kind="evals", budget=3)
    lines[0] = PREFIX + '{"schema":"other"}'
    with pytest.raises(ValueError, match="schema"):
        parse_report(lines, kind="evals", budget=3)


def test_wall_underfill_and_nonpreemptible_overrun_are_not_fair_comparisons() -> None:
    assert parse_report(_lines(), kind="movetime", budget=1000)["comparable"]
    assert not parse_report(_lines(wall=0.9), kind="movetime", budget=1000)["comparable"]
    result = parse_report(_lines(wall=1.2), kind="movetime", budget=1000)
    assert result["budget_reached"]
    assert result["budget_overrun"] == pytest.approx(0.2)
    assert not result["comparable"]


def test_zero_clock_interval_is_not_infinite_eps() -> None:
    report = parse_report(_lines(wall=0), kind="evals", budget=3)
    assert report["comparable"]
    assert report["counters"]["useful_eps"] is None


class _Client:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def sync(self, command: str) -> list[str]:
        self.commands.append(command)
        return ["readyok"]

    def send(self, command: str) -> None:
        self.commands.append(command)

    def until(self, _prefix: str, *, timeout: float) -> list[str]:
        assert timeout > 0
        return _lines()


def test_real_driver_sends_eval_limit_and_preserves_history(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter((0.0, 1.0))
    monkeypatch.setattr("scripts.bench_neural_search.time.perf_counter", lambda: next(ticks))
    client: Any = _Client()
    position = "position startpos moves g1f3 g8f6 f3g1 f6g8"
    result = search(client, position, kind="evals", budget=3, profile=True, timeout=2, wall_tolerance_ms=10)
    assert client.commands == [position, "go evals 3 depth 32 profile\n"]
    assert result["position"] == position
    assert result["decision_wall_seconds"] >= 0
    with pytest.raises(ValueError, match="single full"):
        search(client, position + "\nquit", kind="evals", budget=3, profile=False, timeout=2, wall_tolerance_ms=10)


def test_paired_run_rejects_different_models_before_starting(tmp_path: Path) -> None:
    config = tmp_path / "engines.json"
    base = {"label": "a", "command": ["unused"], "model_id": "one", "encoding_id": "v2", "source_revision": "test"}
    config.write_text(json.dumps([base, {**base, "label": "b", "model_id": "two"}]))
    with pytest.raises(ValueError, match="same declared model"):
        run(config, tmp_path / "positions", tmp_path / "output", evals=3, milliseconds=1000,
            repeats=1, profile=False, timeout=2, wall_tolerance_ms=10)
    assert not (tmp_path / "output").exists()


def test_benchmark_end_to_end_processes_paired_modes(tmp_path: Path) -> None:
    fake = tmp_path / "fake_engine.py"
    fake.write_text('''import json, sys, time
for command in sys.stdin:
    words = command.split()
    if words[0] == "uci":
        print("id name Bend standalone test fixture")
        print("uciok", flush=True)
    elif words[0] == "isready":
        print("readyok", flush=True)
    elif words[0] == "go":
        rows = int(words[2]) if words[1] == "evals" else 3
        wall = .001 if words[1] == "evals" else int(words[2]) / 1000
        time.sleep(wall)
        report = {"schema": "deepfin.neural-work.v1", "wall_seconds": wall,
                  "completed_simulations": rows+1, "forward_calls": 1,
                  "dispatched_real_rows": rows, "executed_real_rows": rows,
                  "accepted_neural_rows": rows, "padded_rows": 8-rows,
                  "failed_forward_rows": 0, "cancelled_rows": 0, "stale_rows": 0,
                  "rejected_rows": 0, "failed_rows": 0, "unresolved_rows": 0,
                  "executed_wasted_rows": 0, "stop_code": 0,
                  "real_batch_histogram": {str(rows):1}, "physical_batch_histogram":{"8":1},
                  "useful_eps": rows/wall, "executed_eps": rows/wall}
        print("info string neural_work " + json.dumps(report))
        print("info nodes", rows+1, "string fixture")
        print("bestmove e2e4", flush=True)
    elif words[0] == "quit":
        break
''')
    config = tmp_path / "engines.json"
    config.write_text(json.dumps([{"label": label, "command": [sys.executable, str(fake)],
                                   "model_id": "same", "encoding_id": "same", "source_revision": label}
                                  for label in ("a", "b")]))
    positions = tmp_path / "positions.txt"
    positions.write_text("position startpos moves g1f3 g8f6\n")
    output = tmp_path / "observations.jsonl"
    # Functional subprocess coverage, not a host-scheduling performance test.
    # Strict deadline behavior is tested with controlled clocks above/below.
    assert run(config, positions, output, evals=3, milliseconds=10, repeats=2, profile=False,
               timeout=2, wall_tolerance_ms=1000)
    observations = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(observations) == 8  # four process blocks x two modes; warmup excluded
    assert [r["engine"]["label"] for r in observations] == ["a", "a", "b", "b", "b", "b", "a", "a"]
    assert {r["budget_kind"] for r in observations} == {"evals", "movetime"}
    assert all(r["comparable"] for r in observations)
    with pytest.raises(FileExistsError):
        run(config, positions, output, evals=3, milliseconds=10, repeats=1, profile=False,
            timeout=2, wall_tolerance_ms=2)


def test_internal_clock_cannot_hide_slow_decision_output(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter((0.0, 1.2))
    monkeypatch.setattr("scripts.bench_neural_search.time.perf_counter", lambda: next(ticks))
    client: Any = _Client()
    result = search(client, "position startpos", kind="movetime", budget=1000,
                    profile=False, timeout=2, wall_tolerance_ms=10)
    assert result["counters"]["wall_seconds"] == 1
    assert result["decision_budget_overrun_seconds"] == pytest.approx(.2)
    assert not result["comparable"]


def test_unresolved_requests_cannot_be_a_completed_budget_comparison() -> None:
    lines = _lines()
    report = json.loads(lines[0][len(PREFIX):])
    report["unresolved_rows"] = 1
    lines[0] = PREFIX + json.dumps(report)
    assert not parse_report(lines, kind="evals", budget=3)["comparable"]


def test_logical_resolution_cannot_hide_an_unfinished_physical_batch() -> None:
    lines = _lines()
    report = json.loads(lines[0][len(PREFIX):])
    # One completed three-row forward and one unconfirmed singleton. The old
    # parser trusted unresolved_rows=0 instead of reconciling physical work.
    report.update(forward_calls=2, dispatched_real_rows=4,
                  real_batch_histogram={"1": 1, "3": 1}, physical_batch_histogram={"1": 1, "8": 1})
    lines[0] = PREFIX + json.dumps(report)
    result = parse_report(lines, kind="movetime", budget=1000)
    assert not result["comparable"]
    assert result["unconfirmed_forward_rows"] == 1


def test_a_forward_cannot_be_both_executed_and_failed() -> None:
    lines = _lines()
    report = json.loads(lines[0][len(PREFIX):])
    report["failed_forward_rows"] = 1
    lines[0] = PREFIX + json.dumps(report)
    with pytest.raises(ValueError, match="forward rows contradict"):
        parse_report(lines, kind="movetime", budget=1000)


@pytest.mark.parametrize("kind", ["evals", "movetime"])
@pytest.mark.parametrize("outcome", ["rejected", "failed", "cancelled", "stale"])
def test_discarded_work_costs_compute_but_cannot_certify_a_successful_comparison(kind: str, outcome: str) -> None:
    ledger = NeuralWorkLedger()
    ledger.submit(0, 2)
    ledger.submit(1)
    ledger.dispatch(0, (0, 1), physical_rows=8)
    ledger.complete(0)
    ledger.resolve(0, "accepted")
    ledger.resolve(1, outcome)
    ledger.simulations_completed(2)
    report = ledger.snapshot(1.0)
    report["stop_code"] = 0
    lines = [PREFIX + json.dumps(report), "info nodes 2 string test", "bestmove e2e4"]
    result = parse_report(lines, kind=kind, budget=3 if kind == "evals" else 1000)
    assert result["budget_reached"]
    assert not result["comparable"]
    assert result["unconfirmed_forward_rows"] == 0
    assert result["counters"]["executed_real_rows"] == 3
    assert result["counters"]["accepted_neural_rows"] == 2
    assert result["counters"]["useful_eps"] == 2


def test_wasted_rows_require_a_nonaccepted_disposition() -> None:
    lines = _lines()
    report = json.loads(lines[0][len(PREFIX):])
    report.update(accepted_neural_rows=2, useful_eps=2, executed_wasted_rows=1)
    lines[0] = PREFIX + json.dumps(report)
    with pytest.raises(ValueError, match="logical dispositions"):
        parse_report(lines, kind="evals", budget=3)


@pytest.mark.parametrize("line", ["bestmove ", "bestmove garbage", "bestmove e2e4 unexpected",
                                  "bestmove e2e4 notponder e7e5", "bestmove e2e4 ponder garbage"])
def test_malformed_bestmove_is_a_report_error(line: str) -> None:
    lines = _lines()
    lines[-1] = line
    with pytest.raises(ValueError, match="bestmove"):
        parse_report(lines, kind="evals", budget=3)


@pytest.mark.parametrize(("position", "move"), [
    ("position startpos", "a1a8"),  # syntactically valid but blocked rook
    ("position startpos", "e7e5"),  # wrong side to move
    ("position startpos", "0000"),  # cannot substitute no move for a legal root
    ("position startpos moves e2e4", "e2e4"),  # validate the actual history, not startpos
    ("position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "h8g8"),  # checkmate
])
def test_search_rejects_illegal_bestmove_before_banking_success(
    monkeypatch: pytest.MonkeyPatch, position: str, move: str,
) -> None:
    ticks = iter((0.0, 1.0))
    monkeypatch.setattr("scripts.bench_neural_search.time.perf_counter", lambda: next(ticks))
    lines = _lines()
    lines[-1] = "bestmove " + move
    client: Any = _Client()
    monkeypatch.setattr(client, "until", lambda _prefix, **_kwargs: lines)
    with pytest.raises(ValueError, match="illegal bestmove"):
        search(client, position, kind="evals", budget=3, profile=False, timeout=2, wall_tolerance_ms=10)


def test_terminal_null_bestmove_still_has_zero_neural_work(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter((0.0, 1.0))
    monkeypatch.setattr("scripts.bench_neural_search.time.perf_counter", lambda: next(ticks))
    ledger = NeuralWorkLedger()
    ledger.simulations_completed()
    report = ledger.snapshot(1.0)
    report["stop_code"] = 0
    lines = [PREFIX + json.dumps(report), "info nodes 1 string test", "bestmove 0000"]
    client: Any = _Client()
    monkeypatch.setattr(client, "until", lambda _prefix, **_kwargs: lines)
    result = search(client, "position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
                    kind="evals", budget=1, profile=False, timeout=2, wall_tolerance_ms=10)
    assert result["bestmove"] == "0000"
    assert result["counters"]["executed_real_rows"] == 0
    assert not result["budget_reached"]
    assert not result["comparable"]


@pytest.mark.parametrize("position", [
    "position startposgarbage", "position startpos extra",
    "position startpos moves e2e5", "position startpos moves 0000",
    "position fen 8/8/8/8/8/8/8/8 w - - 0 1", "position fen not-a-fen",
])
def test_invalid_corpus_fails_before_engine_launch_or_output_creation(tmp_path: Path, position: str) -> None:
    config = tmp_path / "engines.json"
    config.write_text(json.dumps([{"label": "one", "command": ["must-not-be-launched"],
                                   "model_id": "same", "encoding_id": "same", "source_revision": "test"}]))
    positions = tmp_path / "positions.txt"
    positions.write_text("position startpos\n" + position + "\n")
    output = tmp_path / "observations.jsonl"
    with pytest.raises(ValueError, match=r"position|history|UCI"):
        run(config, positions, output, evals=3, milliseconds=10, repeats=1, profile=False,
            timeout=2, wall_tolerance_ms=10)
    assert not output.exists()


@pytest.mark.parametrize("line", ["info nodes ", "info nodes garbage", "info nodes -1"])
def test_malformed_final_nodes_is_a_report_error(line: str) -> None:
    lines = _lines()
    lines[1] = line
    with pytest.raises(ValueError, match="final nodes"):
        parse_report(lines, kind="evals", budget=3)


@pytest.mark.parametrize(("position", "move"), [
    ("position startpos moves", "e2e4"),  # empty optional history remains compatible
    ("position startpos moves e2e4", "c7c5"),
    ("position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
    ("position startpos moves e2e4 a7a6 e4e5 d7d5", "e5d6"),
    ("position fen 7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8q"),
])
def test_legal_history_castling_en_passant_and_promotion_remain_comparable(
    monkeypatch: pytest.MonkeyPatch, position: str, move: str,
) -> None:
    ticks = iter((0.0, 1.0))
    monkeypatch.setattr("scripts.bench_neural_search.time.perf_counter", lambda: next(ticks))
    lines = _lines()
    lines[-1] = "bestmove " + move
    client: Any = _Client()
    monkeypatch.setattr(client, "until", lambda _prefix, **_kwargs: lines)
    result = search(client, position, kind="evals", budget=3, profile=False, timeout=2, wall_tolerance_ms=10)
    assert result["bestmove"] == move
    assert result["comparable"]
