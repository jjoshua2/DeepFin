from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random

import chess
import numpy as np
import pytest

from chess_anti_engine.eval.sprt import SprtMonitor
from chess_anti_engine.utils.game_log import GameLogWriter
from scripts import arena_standard as arena
from scripts import bt4_recipe_readout as tool


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def put(path, value):
    path.write_text(json.dumps(value) + "\n")
    return pin(path)


@pytest.fixture(scope="module")
def panel():
    rng, entries, seen = random.Random(6431), [], set()
    while len(entries) < 500:
        board = chess.Board()
        for _ in range(16):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if (
            len(board.move_stack) != 16
            or board.is_game_over()
            or board.legal_moves.count() < 2
            or board.fen() in seen
        ):
            continue
        entries.append(
            {
                "root_fen": board.root().fen(),
                "moves": [m.uci() for m in board.move_stack],
                "fen": board.fen(),
            }
        )
        seen.add(board.fen())
    return entries


@pytest.fixture
def make(tmp_path, monkeypatch, panel):
    monkeypatch.setattr(arena, "git_sha", lambda: "a" * 40)
    monkeypatch.setattr(arena, "production_config_record", dict)
    common = {
        "candidate": {"path": str(tmp_path / "B100.pt"), "sha256": "b" * 64},
        "reference": {"path": str(tmp_path / "H20.pt"), "sha256": "c" * 64},
        "book": {"path": str(tmp_path / "book.zip"), "sha256": "d" * 64},
        "runtime": {
            **put(tmp_path / "runtime.json", {"commit": "a" * 40}),
            "git_sha": "a" * 40,
        },
        "preregistration": put(tmp_path / "registration.json", {"fixed": True}),
    }
    opening = put(tmp_path / "openings.json", panel)

    def build(
        name="low",
        *,
        values=None,
        low=True,
        order=None,
        orphan=False,
        reason="max_seconds",
        loop="rolling",
    ):
        folder = tmp_path / name
        folder.mkdir()
        values = values if values is not None else [0.0, 2.0] * (250 if low else 64)
        ids = list(range(len(values))) if order is None else order
        side = arena.SideSearch(
            shape="training",
            source="fixture",
            gumbel={"policy_temp": 1.0},
            vloss_weight=1,
            target_batch=0,
        )
        settings = arena.arena_game_log_settings(
            mode="matched_sims",
            candidate=common["candidate"]["path"],
            reference=common["reference"]["path"],
            games=1000 if low else 256,
            seed=42,
            openings_path=common["book"]["path"],
            openings_kind="book",
            opening_plies=16,
            sims_candidate=100 if low else 400,
            sims_reference=100 if low else 400,
            ms_per_move=None,
            max_plies=300,
            temperature=0.1,
            gumbel_add_noise=True,
            search_candidate=side,
            search_reference=side,
            volatility_candidate=None,
            uci_args="",
            syzygy_path=None,
            tb_max_pieces=6,
        )
        execution = {
            "loop": loop,
            "compile": "on",
            "eval_hoist": "4096",
            "eval_max_batch": 4096,
            "eval_leaf_cap_uncapped": 4096,
            "max_concurrent_games": 128,
            "arena_pool_size": 128,
            "max_seconds": 5370.0,
            "hard_seconds": 5400,
        }
        bank, result_path = folder / "games.jsonl", folder / "result.jsonl"
        monitor = SprtMonitor(
            tool.SPEC,
            pairs_cap=500,
            granularity="pair" if loop == "rolling" else "chunk",
        )
        writer = GameLogWriter(
            bank,
            driver="arena_standard",
            settings=settings,
            info={"sprt": tool.SPEC.as_record()} if low else None,
        )

        def row(pair, half, score):
            white = score if half == 0 else 1 - score
            writer.write_game(
                {
                    "pair_id": pair,
                    "half": half,
                    "a_is_white": half == 0,
                    "opening_index": pair,
                    "opening_fen": panel[pair]["fen"],
                    "start_fen": panel[pair]["fen"],
                    "result": {0.0: "0-1", 0.5: "1/2-1/2", 1.0: "1-0"}[white],
                    "score_candidate": score,
                    "seed": 42,
                    "loop": loop,
                    "compile": "on",
                    "eval_hoist": "4096",
                }
            )

        for i in ids:
            value = values[i]
            row(i, 0, min(1.0, value))
            row(i, 1, max(0.0, value - 1.0))
            if low:
                monitor.update([value], pair_ids=[i])
        if orphan:
            row(205, 0, 0.5)
            monitor.inflight_games = [(205, 1), (206, 0), (206, 1)]
        writer.close()
        if low:
            monitor.not_started_games = (
                1000 - 2 * len(ids) - int(orphan) - len(monitor.inflight_games)
            )
            monitor.finalize(stop_reason="cap" if len(ids) == 500 else reason)
        scores = monitor.pair_scores if low else values
        record = arena.build_result_record(
            arena.summarize_pentanomial(arena.pentanomial_counts(scores)),
            mode="matched_sims",
            candidate=settings["candidate"],
            reference=settings["reference"],
            openings_path=settings["openings"],
            opening_plies=16,
            sims_candidate=settings["sims_candidate"],
            sims_reference=settings["sims_reference"],
            ms_per_move=None,
            temperature=0.1,
            gumbel_add_noise=True,
            max_plies=300,
            seed=42,
            device="cuda",
            duration_s=100.0,
            search_candidate=side,
            search_reference=side,
            games_requested=settings["games"],
            max_seconds=5370.0,
            truncated=len(scores) < (500 if low else 128),
            game_log=str(bank),
            game_log_fingerprint=tool.settings_fingerprint(settings),
            compile_setting="on",
            compile_values=["on"],
            hoist_setting="4096",
            hoist_values=["4096"],
            eval_max_batch=4096,
            eval_leaf_cap_uncapped=4096,
            eval_leaf_cap_bound=False,
            max_concurrent_games=128,
            arena_pool=128,
            sprt=monitor.as_record() if low else None,
        )
        command = ["python", "scripts/arena_standard.py"]
        flags = {
            "--candidate": settings["candidate"],
            "--reference": settings["reference"],
            "--games": str(settings["games"]),
            "--mode": "matched_sims",
            "--sims": str(settings["sims_candidate"]),
            "--seed": "42",
            "--openings": settings["openings"],
            "--opening-plies": "16",
            "--max-plies": "300",
            "--temperature": "0.1",
            "--search-shape": "training",
            "--cand-gumbel": "policy_temp=1.0",
            "--ref-gumbel": "policy_temp=1.0",
            "--compile": "on",
            "--games-out": str(bank),
            "--out": str(result_path),
            "--max-concurrent-games": "128",
            "--eval-max-batch": "4096",
            "--max-seconds": "5370",
        }
        for k, v in flags.items():
            command.extend([k, v])
        if loop == "chunked":
            command.append("--no-rolling")
        if low:
            command.extend(
                [
                    "--sprt",
                    "elo0=0,elo1=15,alpha=.05,beta=.10,first_pairs=128,step_pairs=64",
                ]
            )
        record["argv"] = command[1:]
        launch = {
            "settings": settings,
            "execution": execution,
            "opening_panel": opening,
            "identities": common,
            "command": command,
            "candidate_role": "B100",
            "reference_role": "H20",
        }
        process = {
            "process_complete": True,
            "exit_code": 0,
            "command": command,
            "hard_seconds": 5400,
            "started_unix": 1.0,
            "ended_unix": 5371.0 if low and reason == "max_seconds" else 101.0,
            "stage_seconds": 5370.0 if low and reason == "max_seconds" else 100.0,
            "gpu_seconds": 5370.0 if low and reason == "max_seconds" else 100.0,
        }
        return {
            "schema": 1,
            "mode": "low_sprt" if low else "high_fixed128",
            "bank": pin(bank),
            "result": put(result_path, record),
            "process": put(folder / "process.json", process),
            "launch": put(folder / "launch.json", launch),
            "opening_panel": opening,
            "expected_settings": settings,
            "expected_execution": execution,
        }

    return build


def mutate(manifest, member, change):
    path = Path(manifest[member]["path"])
    obj = json.loads(path.read_text())
    change(obj)
    manifest[member] = put(path, obj)


def test_delayed_prefix_crossing_preserves_speculation_orphan_and_inflight(make):
    m = make(values=[2.0] * 200, order=[*range(1, 200), 0], orphan=True)
    report = tool.read_cell(m)
    assert report["sprt"]["verdict"] == "H1"
    assert report["sprt"]["pairs"] == 128
    assert report["sprt"]["speculative_completed_pair_ids"] == list(range(128, 200))
    assert report["raw_finished_games"] == 401
    assert report["orphan_finished_halves"] == [[205, 0]]
    assert len(report["unstarted_game_ids"]) == 596


@pytest.mark.parametrize(
    ("values", "reason", "verdict"),
    [
        ([0.0] * 128, "max_seconds", "H0"),
        ([0.0, 2.0] * 250, "cap", "INCONCLUSIVE"),
        ([0.0, 2.0] * 80, "max_seconds", "INCONCLUSIVE"),
    ],
)
def test_negative_and_uncrossed_valid_stops_are_not_operational_failures(
    make, values, reason, verdict
):
    report = tool.read_cell(make(values=values, reason=reason))
    assert report["sprt"]["verdict"] == verdict
    assert report["status"] == "VALID_CELL"
    assert report["sprt"]["pairs"] == (128 if verdict == "H0" else len(values))


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["sprt"].__setitem__("llr", 99.0),
        lambda r: r["sprt"].__setitem__("scored_pair_ids", list(range(192))),
        lambda r: r["sprt"].__setitem__("not_started_games", 0),
        lambda r: r["sprt"].__setitem__("inflight_games", [[0, 0]]),
        lambda r: r["sprt"].__setitem__("llr_trajectory", [[192, 3.0]]),
        lambda r: r.__setitem__("resumed_pairs", 1),
        lambda r: r.__setitem__("score", 0.99),
    ],
)
def test_forged_terminal_prefix_and_accounting_refused(make, change):
    m = make(values=[2.0] * 200, orphan=True)
    mutate(m, "result", change)
    with pytest.raises(tool.InvalidCell):
        tool.read_cell(m)


def test_missing_core_and_duplicate_or_torn_games_refused(make):
    m = make(values=[0.0, 2.0] * 100, order=[*range(127), *range(128, 200)])
    with pytest.raises(tool.InvalidCell, match="first128"):
        tool.read_cell(m)
    good = make(name="good")
    path = Path(good["bank"]["path"])
    lines = path.read_text().splitlines()
    path.write_text("\n".join([*lines, lines[-1]]) + "\n")
    good["bank"] = pin(path)
    with pytest.raises(tool.InvalidCell, match="duplicate"):
        tool.read_cell(good)
    path.write_text("\n".join(lines) + '\n{"kind":')
    good["bank"] = pin(path)
    with pytest.raises(tool.InvalidCell, match="torn"):
        tool.read_cell(good)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.__setitem__("exit_code", 124),
        lambda p: p["command"].__setitem__(p["command"].index("--sims") + 1, "400"),
    ],
)
def test_bad_stage_or_header_only_spoof_refused(make, change):
    m = make()
    mutate(m, "process", change)
    process = json.loads(Path(m["process"]["path"]).read_text())
    if process["exit_code"] == 0:
        mutate(
            m,
            "launch",
            lambda launch: launch.__setitem__("command", process["command"]),
        )
    with pytest.raises(tool.InvalidCell):
        tool.read_cell(m)


def test_high_fixed_probe_recertifies_low_and_same_history_core(make, tmp_path):
    low = make(values=[0.0, 2.0] * 80)
    high = make(name="high", low=False)
    high["low_manifest"] = put(tmp_path / "low.json", low)
    report = tool.read_cell(high)
    assert report["result"]["pairs"] == 128
    assert report["sprt"] is None
    assert report["fixed_core_cross_budget"]["score_advantage_400_minus_100"] == 0
    broken = copy.deepcopy(low)
    mutate(broken, "process", lambda p: p.__setitem__("exit_code", 7))
    high["low_manifest"] = put(tmp_path / "broken-low.json", broken)
    with pytest.raises(tool.InvalidCell):
        tool.read_cell(high)


def test_wrong_high_mapping_and_history_refused(make, tmp_path):
    low = make()
    high = make(name="high", low=False)
    high["low_manifest"] = put(tmp_path / "low.json", low)
    path = Path(high["bank"]["path"])
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    lines[1]["opening_fen"] = lines[1]["start_fen"] = lines[3]["opening_fen"]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    high["bank"] = pin(path)
    with pytest.raises(tool.InvalidCell, match="mapping"):
        tool.read_cell(high)
    entries = json.loads(Path(low["opening_panel"]["path"]).read_text())
    entries[0]["moves"][0] = "a1a8"
    low["opening_panel"] = put(Path(low["opening_panel"]["path"]), entries)
    mutate(
        low, "launch", lambda r: r.__setitem__("opening_panel", low["opening_panel"])
    )
    with pytest.raises(ValueError, match=r"illegal|invalid"):
        tool.read_cell(low)


def test_fixed_probe_rejects_sequential_terminal_and_recursive_manifest(make, tmp_path):
    m = make(name="high", low=False)
    m["low_manifest"] = put(tmp_path / "recursive.json", m)
    with pytest.raises(tool.InvalidCell, match="directly pinned low"):
        tool.read_cell(m)
    mutate(m, "result", lambda r: r.__setitem__("sprt", {}))
    with pytest.raises(tool.InvalidCell, match="fixed probe"):
        tool.read_cell(m)


@pytest.mark.parametrize("loop", ["rolling", "chunked"])
def test_expected_command_parses_through_actual_arena_cli(make, monkeypatch, loop):
    m = make(loop=loop)
    command = json.loads(Path(m["launch"]["path"]).read_text())["command"]
    seen = {}
    monkeypatch.setattr(arena, "run_arena", lambda **kw: seen.update(kw))
    monkeypatch.setattr("sys.argv", command[1:])
    arena.main()
    assert seen["rolling"] is (loop == "rolling")
    assert seen["sims_candidate"] == seen["sims_reference"] == 100
    assert seen["games"] == 1000
    assert seen["sprt"] == tool.SPEC
    assert seen["search_candidate"].gumbel["policy_temp"] == 1.0
    assert seen["search_reference"].gumbel["policy_temp"] == 1.0
    assert seen["resume"] is False
    assert tool.read_cell(m)["status"] == "VALID_CELL"


def test_speculative_orphan_row_is_validated_even_outside_stopping_prefix(make):
    m = make(values=[2.0] * 200, orphan=True)
    path = Path(m["bank"]["path"])
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    lines[-1]["score_candidate"] = 1.0
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    m["bank"] = pin(path)
    with pytest.raises(tool.InvalidCell, match="score/result"):
        tool.read_cell(m)


@pytest.mark.parametrize(
    "extra", [["--sims=400"], ["--sim=400"], ["--resume=true"], ["--device=cpu"]]
)
def test_effective_cli_override_refused(make, extra):
    m = make()
    mutate(m, "process", lambda p: p["command"].extend(extra))
    command = json.loads(Path(m["process"]["path"]).read_text())["command"]
    mutate(m, "launch", lambda p: p.__setitem__("command", command))
    mutate(m, "result", lambda p: p.__setitem__("argv", command[1:]))
    with pytest.raises(tool.InvalidCell):
        tool.read_cell(m)


@pytest.mark.parametrize(
    "changes",
    [
        {"ended_unix": 10001.0},
        {"ended_unix": 6001.0, "stage_seconds": 6000.0, "gpu_seconds": 6000.0},
        {"stage_seconds": float("nan")},
        {"gpu_seconds": float("inf")},
        {"gpu_seconds": 50.0},
        {"ended_unix": 101.0, "stage_seconds": 100.0, "gpu_seconds": 100.0},
    ],
)
def test_process_charge_and_actual_deadline_refused(make, changes):
    m = make(values=[0.0, 2.0] * 80)
    mutate(m, "process", lambda p: p.update(changes))
    with pytest.raises(tool.InvalidCell):
        tool.read_cell(m)


def test_registered_interaction_resamples_aligned_pairs(make, tmp_path):
    low_values = [0.0, 2.0] * 64
    high_values = [2.0, 0.0, 1.0, 2.0] * 32
    low = make(values=low_values)
    high = make(name="high", low=False, values=high_values)
    high["low_manifest"] = put(tmp_path / "low.json", low)
    contrast = tool.read_cell(high)["fixed_core_cross_budget"]
    # Independent row-at-a-time paired resampling; reversing or independently
    # shuffling one side would destroy covariance and change this interval.
    delta = (np.array(high_values) - np.array(low_values)) / 2
    rng = np.random.Generator(np.random.PCG64(20260903))
    draws = [float(delta[rng.integers(0, 128, size=128)].mean()) for _ in range(10000)]
    assert (
        contrast["paired_bootstrap_ci95"] == np.percentile(draws, [2.5, 97.5]).tolist()
    )
    assert contrast["score_advantage_400_minus_100"] == float(delta.mean())
    assert contrast["bootstrap"]["samples"] == 10000
    assert "paired_normal_ci95" not in contrast
