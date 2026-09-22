"""Paired native-search observations at fixed NN rows AND fixed wall time.

Engines must implement the standalone neural_work v1 report and go evals.
This is a measurement driver, not a Python search coordinator or Elo estimator.
Use full UCI position commands to preserve history. See docs/neural_work.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import time
from pathlib import Path
from typing import Any

from chess_anti_engine.neural_work import SCHEMA
from native.bend_engine.standalone.verify import Client

PREFIX = "info string neural_work "


def _integer(report: dict[str, Any], key: str) -> int:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"missing/nonnegative-integer counter: {key}")
    return value


def _histogram(report: dict[str, Any], key: str) -> tuple[int, int]:
    hist = report.get(key)
    if not isinstance(hist, dict):
        raise ValueError(f"missing histogram: {key}")
    calls = rows = 0
    for size, count in hist.items():
        if not isinstance(size, str) or not size.isdecimal() or int(size) < 1:
            raise ValueError(f"invalid batch size in {key}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"invalid frequency in {key}")
        calls += count
        rows += int(size) * count
    return calls, rows


def parse_report(lines: list[str], *, kind: str, budget: int, wall_tolerance_ms: int = 10) -> dict[str, Any]:
    """Reject missing/contradictory counters; never derive NN work from nodes/calls."""
    reports = [line[len(PREFIX):] for line in lines if line.startswith(PREFIX)]
    if len(reports) != 1:
        raise ValueError("expected exactly one neural_work report")
    report = json.loads(reports[0])
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError("unsupported neural_work schema")
    if kind not in ("evals", "movetime") or budget <= 0 or wall_tolerance_ms < 0:
        raise ValueError("invalid comparison budget")
    keys = ("completed_simulations", "forward_calls", "dispatched_real_rows", "executed_real_rows",
            "accepted_neural_rows", "padded_rows", "failed_forward_rows", "cancelled_rows", "stale_rows",
            "failed_rows", "rejected_rows", "unresolved_rows", "executed_wasted_rows", "stop_code")
    counts = {key: _integer(report, key) for key in keys}
    dispatched, executed, accepted = (counts[key] for key in (
        "dispatched_real_rows", "executed_real_rows", "accepted_neural_rows"))
    if not accepted <= executed <= dispatched:
        raise ValueError("acceptance/execution/dispatch counters contradict each other")
    if counts["unresolved_rows"] == 0 and accepted + counts["executed_wasted_rows"] != executed:
        raise ValueError("resolved useful/wasted rows contradict executed rows")
    real_calls, real_rows = _histogram(report, "real_batch_histogram")
    physical_calls, physical_rows = _histogram(report, "physical_batch_histogram")
    if real_calls != counts["forward_calls"] or physical_calls != real_calls or real_rows != dispatched:
        raise ValueError("forward histogram does not reconcile with dispatched real rows")
    if physical_rows != dispatched + counts["padded_rows"]:
        raise ValueError("physical histogram does not reconcile with padding")
    wall = report.get("wall_seconds")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0:
        raise ValueError("invalid wall_seconds")
    for key, numerator in (("useful_eps", accepted), ("executed_eps", executed)):
        value = report.get(key)
        if wall == 0:
            if value is not None:
                raise ValueError("zero-duration EPS must be null")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isclose(
            value, numerator / wall, rel_tol=2e-5, abs_tol=1e-5,
        ):
            raise ValueError(f"{key} denominator/numerator mismatch")
    nodes = [int(line.split()[2]) for line in lines if line.startswith("info nodes ")]
    bestmoves = [line.split()[1] for line in lines if line.startswith("bestmove ")]
    if nodes != [counts["completed_simulations"]] or len(bestmoves) != 1:
        raise ValueError("missing/duplicate final nodes or bestmove")
    errors = sum(counts[key] for key in ("failed_forward_rows", "cancelled_rows", "stale_rows", "failed_rows", "unresolved_rows"))
    if kind == "evals":
        reached = dispatched == executed == budget
        overrun = max(0, dispatched - budget)
    else:
        reached = wall >= budget / 1000
        overrun = max(0.0, wall - budget / 1000)
    comparable = (reached and counts["stop_code"] == 0 and errors == 0
                  and (overrun == 0 if kind == "evals" else overrun <= wall_tolerance_ms / 1000 + 1e-9))
    return {"counters": report, "bestmove": bestmoves[0], "budget_kind": kind, "budget": budget,
            "budget_reached": reached, "budget_overrun": overrun, "comparable": comparable,
            "overrun_unit": "real_rows" if kind == "evals" else "seconds"}


def search(client: Client, position: str, *, kind: str, budget: int, profile: bool,
           timeout: float, wall_tolerance_ms: int) -> dict[str, Any]:
    if not position.startswith(("position startpos", "position fen ")) or "\n" in position or "\r" in position:
        raise ValueError("positions must be single full UCI position commands")
    ready = client.sync(position)
    if ready != ["readyok"]:
        raise ValueError(f"engine rejected position: {ready}")
    started = time.perf_counter()
    client.send(f"go {kind} {budget} depth 32{' profile' if profile else ''}\n")
    lines = client.until("bestmove ", timeout=timeout)
    decision_seconds = time.perf_counter() - started
    result = parse_report(lines, kind=kind, budget=budget, wall_tolerance_ms=wall_tolerance_ms)
    if result["counters"]["wall_seconds"] > decision_seconds + .002:
        raise ValueError("engine wall interval exceeds observed command-to-bestmove time")
    result["decision_wall_seconds"] = decision_seconds
    # A timely internal search cannot hide slow go preparation or result output.
    # Use the externally observed decision interval for wall-time fairness too.
    decision_overrun = max(0.0, decision_seconds - budget / 1000) if kind == "movetime" else None
    result["decision_budget_overrun_seconds"] = decision_overrun
    if decision_overrun is not None:
        result["comparable"] &= decision_overrun <= wall_tolerance_ms / 1000 + 1e-9
    # Keep compact results, not megabytes of per-leaf diagnostic priors.
    result["position"] = position
    return result


def run(config: Path, positions_file: Path, output: Path, *, evals: int, milliseconds: int,
        repeats: int, profile: bool, timeout: float, wall_tolerance_ms: int) -> bool:
    engines = json.loads(config.read_text())
    if not isinstance(engines, list) or not engines:
        raise ValueError("engine config must be a nonempty list")
    labels: set[str] = set()
    for engine in engines:
        for field in ("label", "model_id", "encoding_id", "source_revision"):
            if not isinstance(engine.get(field), str) or not engine[field]:
                raise ValueError(f"engine requires declared {field}")
        command = engine.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError("command must be a nonempty argv list (no shell)")
        if engine["label"] in labels:
            raise ValueError("engine labels must be unique")
        labels.add(engine["label"])
    if len({(e["model_id"], e["encoding_id"]) for e in engines}) != 1:
        raise ValueError("paired search comparisons require the same declared model and encoding")
    positions = [line.strip() for line in positions_file.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    if not positions or repeats < 1 or not 1 <= evals <= 65536 or not 1 <= milliseconds <= 60000:
        raise ValueError("empty corpus or invalid bounded run settings")
    corpus_sha256 = hashlib.sha256(positions_file.read_bytes()).hexdigest()
    all_comparable = True
    # Exclusive creation avoids overwriting banked observations.
    with output.open("x") as sink:
        for repeat in range(repeats):
            order = engines if repeat % 2 == 0 else list(reversed(engines))
            for engine in order:
                client = Client(engine["command"])
                try:
                    # Fresh process per block; warmup is explicitly excluded.
                    search(client, "position startpos", kind="evals", budget=1, profile=False,
                           timeout=timeout, wall_tolerance_ms=wall_tolerance_ms)
                    for index, position in enumerate(positions):
                        for kind, budget in (("evals", evals), ("movetime", milliseconds)):
                            observation = {"engine": engine, "repeat": repeat, "position_index": index,
                                           "profile": profile, "warmup_searches": 1,
                                           "corpus_sha256": corpus_sha256}
                            try:
                                result = search(client, position, kind=kind, budget=budget, profile=profile,
                                                timeout=timeout, wall_tolerance_ms=wall_tolerance_ms)
                            except (ValueError, AssertionError, TimeoutError, queue.Empty) as exc:
                                observation.update({"error": str(exc), "budget_kind": kind, "budget": budget,
                                                    "comparable": False, "position": position})
                                sink.write(json.dumps(observation, sort_keys=True) + "\n")
                                sink.flush()
                                raise
                            observation.update(result)
                            sink.write(json.dumps(observation, sort_keys=True) + "\n")
                            sink.flush()
                            all_comparable &= result["comparable"]
                finally:
                    client.close()
    return all_comparable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evals", type=int, default=256)
    parser.add_argument("--movetime-ms", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--wall-tolerance-ms", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    passed = run(args.engines, args.positions, args.output, evals=args.evals, milliseconds=args.movetime_ms,
                 repeats=args.repeats, profile=args.profile, timeout=args.timeout,
                 wall_tolerance_ms=args.wall_tolerance_ms)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
