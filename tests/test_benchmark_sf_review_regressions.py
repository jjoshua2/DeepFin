"""Synthetic regression coverage for the benchmark's five review findings.

These checks run the real validator, controller and value-worker summary with
fake processes/UCI responses. They do not generate positions or run Stockfish.
"""
from __future__ import annotations

import builtins
import hashlib
import itertools
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import benchmark_sf_generation as generation
from scripts import benchmark_sf_values as values


def _plan(tmp_path: Path) -> dict[str, Any]:
    out = tmp_path / "out"
    cells = [
        {
            "id": policy, "policy": policy, "concurrency": 4, "seconds": 600,
            "command": [
                "python", "--out-dir", str(out / policy), "--workers", "4",
                "--worker-concurrency", "4", "--nice", "19", "--shard-rows", "256",
                "--staircase", "all:8" if policy == "d8" else "all:9,8:10,4:12",
                "--staircase-policy", "fixed" if policy == "d8" else "g10",
            ],
        }
        for policy in ("g10", "d8")
    ]
    return {
        "status": "READY_BOUNDED_CPU_SCREEN", "profile": "confirmation",
        "cpu_budget_seconds": 10, "wall_budget_seconds": 1800,
        "affinity": [0], "launch_disk_gib": 100, "memory_gib": 40,
        "output_limit_bytes": 2 * 2**30, "out": str(out),
        "runtime": str(generation.REPO_ROOT), "runtime_head": "fixture-head",
        "pins": [], "cells": cells, "env": {},
    }


def _mock_git(monkeypatch):
    monkeypatch.setattr(
        generation.subprocess, "check_output", lambda *_a, **_k: "fixture-head\n"
    )
    monkeypatch.setattr(generation.subprocess, "run", lambda *_a, **_k: None)


def test_registered_generation_plan_still_validates(tmp_path, monkeypatch):
    _mock_git(monkeypatch)
    generation.validate(_plan(tmp_path))


def test_generation_refuses_foreign_decoder_runtime_before_git(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    plan["runtime"] = str(tmp_path / "other-checkout")
    monkeypatch.setattr(
        generation.subprocess, "check_output",
        lambda *_a, **_k: pytest.fail("foreign runtime reached git validation"),
    )
    with pytest.raises(ValueError, match="runtime differs from benchmark checkout"):
        generation.validate(plan)


@pytest.mark.parametrize("cell_id", ["", ".", "..", "../escape", "nested/cell", "/escaped", None])
def test_generation_refuses_nonlocal_cell_ids(tmp_path, monkeypatch, cell_id):
    _mock_git(monkeypatch)
    plan = _plan(tmp_path)
    cell = plan["cells"][0]
    cell["id"] = cell_id
    # Even matching the escaped output argument must not validate the plan.
    cell["command"][2] = str(Path(plan["out"]) / str(cell_id))
    with pytest.raises(ValueError, match="invalid cell id"):
        generation.validate(plan)


def test_confirmation_refuses_reordered_cells(tmp_path, monkeypatch):
    _mock_git(monkeypatch)
    plan = _plan(tmp_path)
    plan["cells"].reverse()
    with pytest.raises(ValueError, match="confirmation order differs"):
        generation.validate(plan)


@pytest.mark.parametrize("cutoff_cell", [0, 1])
def test_cpu_cutoff_keeps_only_previously_qualified_cells(tmp_path, monkeypatch, cutoff_cell):
    plan = _plan(tmp_path)
    _mock_git(monkeypatch)
    launched = []
    stopped = []
    decoded = []
    clock = itertools.count()

    class Child:
        def __init__(self, *_args, **_kwargs):
            self.pid = 1000 + len(launched)
            self.returncode = 0
            self.polls = 0
            launched.append(self.pid)

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

    class Owned:
        def __init__(self, pid, _baseline):
            self.pid = pid

        def sample(self):
            return 10.0 if self.pid == 1000 + cutoff_cell else 1.0

        def stop(self, child):
            stopped.append(child.pid)

    def readout(root, _depth, checkpoint):
        checkpoint()
        decoded.append(root.name)
        return {"eligible_rows": 3, "banked_rows": 3}

    monkeypatch.setattr(generation, "time", SimpleNamespace(
        monotonic=lambda: float(next(clock)), sleep=lambda _seconds: None,
    ))
    monkeypatch.setattr(generation, "resource", SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _who: SimpleNamespace(ru_utime=0.0, ru_stime=0.0),
    ))
    monkeypatch.setattr(generation.os, "sched_setaffinity", lambda *_a: None)
    monkeypatch.setattr(generation.os, "nice", lambda *_a: 19)
    monkeypatch.setattr(generation.os, "getpriority", lambda *_a: 19)
    monkeypatch.setattr(generation.signal, "signal", lambda *_a: None)
    monkeypatch.setattr(generation.shutil, "disk_usage", lambda _p: SimpleNamespace(free=2**50))
    monkeypatch.setattr(generation, "memory", lambda: 2**50)
    monkeypatch.setattr(generation, "child_baseline", dict)
    monkeypatch.setattr(generation, "OwnedProcesses", Owned)
    monkeypatch.setattr(generation.subprocess, "Popen", Child)
    monkeypatch.setattr(generation, "closure_snapshot", lambda _p: {})
    monkeypatch.setattr(generation, "closed_readout", readout)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    report = generation.execute(plan)
    expected = [] if cutoff_cell == 0 else ["g10"]
    assert report["status"] == "BOUNDED_PARTIAL_SCREEN"
    assert [cell["id"] for cell in report["cells"]] == expected
    assert decoded == expected
    assert report["cpu_seconds_observed"] >= plan["cpu_budget_seconds"]
    assert set(stopped) == set(launched)
    out = Path(plan["out"])
    assert json.loads((out / "complete.json").read_text()) == report
    assert not (out / "failed.json").exists()
    assert not (out / f'{plan["cells"][cutoff_cell]["id"]}.result.json').exists()


@pytest.mark.parametrize("sample_count", [64, 128])
def test_value_worker_reports_even_sample_medians(tmp_path, monkeypatch, sample_count):
    row = {
        "fen": "fixture", "history_root_fen": "fixture", "history_uci": [],
        "history_root_reason": "fixture", "game_id": 1, "ply": 0,
    }
    sample = {
        "source": "fixture", "source_row_index": 0, "row": row,
        "row_sha256": hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    source = tmp_path / "samples.jsonl"
    source.write_text((json.dumps(sample) + "\n") * sample_count)
    out = tmp_path / "out"
    out.mkdir()
    closed = []
    moves = [SimpleNamespace(uci=lambda: "a1a2"), SimpleNamespace(uci=lambda: "a1b1")]

    class LegalMoves:
        def __init__(self, items):
            self.items = items

        def __iter__(self):
            return iter(self.items)

        def count(self):
            return len(self.items)

    board = SimpleNamespace(is_valid=lambda: True, legal_moves=LegalMoves(moves))

    class Engine:
        def __init__(self, *_args, **_kwargs):
            pass

        def _send(self, _command):
            pass

        def close(self):
            closed.append(self)

    class Searcher:
        def __init__(self, *, engine, **_kwargs):
            self.engine = engine

        def new_game(self):
            self.engine._send("ucinewgame")

        def stream(self, _history, *, depth, multipv):
            self.engine._send("position fen fixture")
            self.engine._send(f"go depth {depth}")
            return [depth, multipv]

    def parse(lines, *, expected_lines):
        depth, width = lines
        assert width == expected_lines
        block = SimpleNamespace(
            complete=True, depth=depth, nodes_at_depth=10,
            lines=[SimpleNamespace(rank=i + 1, move=move.uci(), effective_cp=0, nodes=10)
                   for i, move in enumerate(moves[:width])],
        )
        return SimpleNamespace(blocks=[block])

    corpus = SimpleNamespace(
        StaircaseSearcher=Searcher, parse_staircase=lambda _text: None,
        RowHistory=SimpleNamespace,
        position_command=lambda _history: "position fen fixture",
        parse_depth_blocks=parse,
        deepest_block_with_width=lambda blocks, **_kwargs: (blocks[0], True),
    )
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "chess_anti_engine.stockfish.uci":
            return SimpleNamespace(StockfishUCI=Engine)
        if name == "scripts" and "gen_sf_rooted_corpus" in fromlist:
            return SimpleNamespace(gen_sf_rooted_corpus=corpus)
        if name == "scripts" and "derive_corpus_targets" in fromlist:
            return SimpleNamespace(derive_corpus_targets=SimpleNamespace(board_from_row=lambda _r: board))
        return real_import(name, globals, locals, fromlist, level)

    ticks = iter(value for i in range(sample_count * 3)
                 for value in (1000.0 * i, 1000.0 * i + i + 1))
    monkeypatch.setattr(values, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    # Scope fake imports to this invocation; no native dependency or process runs.
    with monkeypatch.context() as scope:
        scope.setattr(builtins, "__import__", fake_import)
        values.worker({
            "sample": str(source), "out": str(out), "rows": sample_count,
            "stockfish": "unused-fake", "syzygy_path": "",
            "arms": [["root_d8", 8, 1], ["root_d10", 10, 1], ["all_d8", 8, "all"]],
        })
    records = [json.loads(line) for line in (out / "labels.jsonl").read_text().splitlines()]
    summary = json.loads((out / "worker.complete.json").read_text())
    assert len(closed) == 3
    assert len(records) == sample_count * 3
    for arm, result in summary["arms"].items():
        durations = [record["elapsed_seconds"] for record in records if record["arm"] == arm]
        assert result["median_seconds"] == statistics.median(durations)
        assert result["median_seconds"] != sorted(durations)[len(durations) // 2]
        assert result["search_and_reset_seconds"] == sum(durations)
