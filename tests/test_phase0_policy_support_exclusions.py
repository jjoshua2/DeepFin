"""Bounded support drops retain exact source identity and cannot hide corruption."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import chess
import pytest

from scripts import derive_corpus_targets as derive
from tests.test_derive_corpus_targets import (
    FEN_W, HISTORY_MOVES, depth_options, full_width_phase, history_row, run_derive,
)
from tests.test_derive_parallel import shard_content, write_split_corpus


def support_miss(row: dict[str, Any]) -> None:
    lines = row["phases"][0]["per_depth"][0]["lines"]
    # Observed defect: full rank slots survive, but a move replaces another slot.
    lines[-1][1] = lines[0][1]


def flags(budget: int, workers: int = 1) -> tuple[str, ...]:
    return ("--policy-observation", "phase0", "--max-policy-support-misses", str(budget),
            "--workers", str(workers), "--rows-per-shard", "3",
            "--seed", "9", "--row-provenance",
            *(("--spill-chunk-rows", "2") if workers > 1 else ()))


def exclusions(output: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in
            (output / derive.POLICY_SUPPORT_MISSES_FILE).read_text().splitlines()]


def test_serial_and_spawn_keep_identical_survivors_and_separate_drop_counts(tmp_path: Path) -> None:
    rows = [history_row(game_id=i, result=None if i == 2 else 1.0) for i in range(8)]
    support_miss(rows[1])
    support_miss(rows[5])
    source = write_split_corpus(tmp_path, rows, [4, 4])
    outputs = [tmp_path / "serial", tmp_path / "spawn"]
    summaries = [run_derive(source, output, "uniform-d9", *flags(2, workers))
                 for workers, output in zip((1, 2), outputs)]
    assert shard_content(outputs[0]) == shard_content(outputs[1])
    assert exclusions(outputs[0]) == exclusions(outputs[1])
    for output, summary in zip(outputs, summaries):
        realized = summary["realized"]
        assert realized["rows_read"] == 8
        assert realized["rows_written"] == 5
        assert realized["rows_dropped_no_result"] == 1
        assert realized["rows_dropped_envelope"] == 0
        assert realized["rows_dropped_policy_support"] == 2
        assert realized["input_key_verified"] == 5
        assert realized["row_schema_counts"] == {str(derive.ROW_SCHEMA_HISTORY): 5}
        assert sum(realized["history_slots_filled_histogram"].values()) == 5
        records = exclusions(output)
        assert records == realized["policy_support_exclusions"]
        assert [(r["source_shard"], r["source_row"], r["game_id"], r["ply"])
                for r in records] == [("w00-00000.jsonl.zst", 1, 1, 8),
                                       ("w00-00001.jsonl.zst", 1, 5, 8)]
        assert all(r["source_dir"] == str(source.resolve()) and
                   r["full_history_input_key_verified"] is True for r in records)
        assert records[0]["missing_moves"]
        assert records[0]["duplicate_moves"]
        assert records[1]["duplicate_moves"]
        assert records[1]["missing_moves"]


@pytest.mark.parametrize("workers", [1, 2])
def test_budget_is_global_across_lanes_and_failure_keeps_evidence(tmp_path: Path, workers: int) -> None:
    rows = [history_row(game_id=i) for i in range(4)]
    support_miss(rows[0])
    support_miss(rows[2])
    source = write_split_corpus(tmp_path, rows, [2, 2])
    output = tmp_path / "refused"
    with pytest.raises(derive.CorpusIntegrityError, match="max-policy-support-misses"):
        run_derive(source, output, "uniform-d9", *flags(1, workers))
    assert not (output / derive.SUMMARY_NAME).exists()
    assert [r["game_id"] for r in exclusions(output)] == [0, 2]


@pytest.mark.parametrize("corruption", ["illegal_extra", "rank", "score", "key", "history", "source",
                                        "truncated", "appended", "width", "searchmoves"])
def test_support_budget_never_hides_other_integrity_errors(tmp_path: Path, corruption: str) -> None:
    row = history_row()
    support_miss(row)
    lines = row["phases"][0]["per_depth"][0]["lines"]
    if corruption == "illegal_extra":
        lines[-1][1] = "a1a8"
    elif corruption == "rank":
        lines[0][0] = 7
    elif corruption == "score":
        lines[0][2] = float("nan")
    elif corruption == "key":
        row["input_key"] = "0" * 32
    elif corruption == "history":
        row["history_uci"] = []
    elif corruption == "truncated":
        lines.pop()
    elif corruption == "appended":
        lines.append([len(lines) + 1, *lines[0][1:]])
    elif corruption == "width":
        row["phases"][0]["width_realized"] -= 1
    elif corruption == "searchmoves":
        row["phases"][0]["searchmoves"] = [lines[0][1]]
    else:
        row["run"]["config_sha256"] = "0" * 64
    source = write_split_corpus(tmp_path, [row, history_row(game_id=1)], [2])
    output = tmp_path / "refused"
    with pytest.raises(derive.CorpusIntegrityError):
        run_derive(source, output, "uniform-d9", *flags(1))
    assert not (output / derive.SUMMARY_NAME).exists()
    assert not (output / derive.POLICY_SUPPORT_MISSES_FILE).exists()


def test_default_stays_strict_and_zero_adds_no_metadata(tmp_path: Path) -> None:
    source = write_split_corpus(tmp_path, [history_row()], [1])
    first = run_derive(source, tmp_path / "default", "uniform-d9")
    explicit = run_derive(source, tmp_path / "zero", "uniform-d9", "--max-policy-support-misses", "0")
    assert shard_content(tmp_path / "default") == shard_content(tmp_path / "zero")
    assert first["realized"] == explicit["realized"]
    assert "max_policy_support_misses" not in first
    assert "rows_dropped_policy_support" not in first["realized"]
    row = history_row()
    support_miss(row)
    invalid = write_split_corpus(tmp_path, [row], [1], name="bad_source")
    with pytest.raises(derive.CorpusIntegrityError, match="legal move set"):
        run_derive(invalid, tmp_path / "strict", "uniform-d9", "--policy-observation", "phase0")


def test_unsupported_opt_in_refused_before_derivation() -> None:
    options = depth_options(derive.parse_scheme("uniform-d9"))
    with pytest.raises(ValueError, match="phase0"):
        replace(options, max_policy_support_misses=1)
    with pytest.raises(ValueError, match=">= 0"):
        replace(options, max_policy_support_misses=-1)


def test_actual_spawn_cli_propagates_original_fatal_error(tmp_path: Path) -> None:
    rows = [history_row(game_id=i) for i in range(4)]
    support_miss(rows[0])
    rows[0]["input_key"] = "0" * 32
    source = write_split_corpus(tmp_path, rows, [2, 2])
    output = tmp_path / "refused"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "PYTHONPATH": str(Path(derive.__file__).resolve().parents[1])}
    process = subprocess.run(
        [sys.executable, str(Path(derive.__file__).resolve()), "--corpus", str(source),
         "--out", str(output), "--scheme", "uniform-d9", "--temp", "1", *flags(1, 2)],
        env=env, capture_output=True, text=True, timeout=45, check=False,
    )
    assert process.returncode != 0
    assert "CorpusIntegrityError" in process.stderr
    assert "input_key" in process.stderr
    assert "Can't get attribute" not in process.stderr
    assert not (output / derive.SUMMARY_NAME).exists()


def test_grouped_support_drops_keep_tail_and_worker_handoff_identity(tmp_path: Path) -> None:
    rows = []
    for game_id in range(2):
        for ply in range(5, 9):
            board = chess.Board(FEN_W)
            for move in HISTORY_MOVES[:ply]:
                board.push_uci(move)
            values = {move.uci(): float(i) for i, move in enumerate(board.legal_moves)}
            rows.append(history_row(game_id=game_id, ply=ply, history_moves=HISTORY_MOVES[:ply],
                                    phases=[full_width_phase(board.fen(), {9: values})]))
    support_miss(rows[2])
    support_miss(rows[3])  # Actual game tail, just over the raw-shard boundary.
    source = write_split_corpus(tmp_path, rows, [3, 2, 3])
    outputs = [tmp_path / "serial", tmp_path / "spawn"]
    summaries = [run_derive(source, output, "uniform-d9", *flags(2, workers),
                            "--value-scheme", "qzphase")
                 for workers, output in zip((1, 2), outputs)]
    assert shard_content(outputs[0]) == shard_content(outputs[1])
    assert exclusions(outputs[0]) == exclusions(outputs[1])
    for summary in summaries:
        assert summary["realized"]["rows_written"] == 6
        assert summary["realized"]["rows_dropped_policy_support"] == 2
        assert summary["realized"]["value_scheme_realized"]["games_with_dropped_tail"] == 1
    for shard in outputs[0].glob("shard_*.zarr"):
        name = "row_provenance.npz"
        assert (shard / name).read_bytes() == (outputs[1] / shard.name / name).read_bytes()
