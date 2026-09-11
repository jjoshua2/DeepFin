"""Saved G10 value selection reaches stored WDL without altering any other array."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import pytest
import zarr

from scripts import adaptive_sf_value as adaptive
from scripts import derive_corpus_targets as derive
from scripts import gen_sf_rooted_corpus as gen
from scripts.lc0_control_train import value_scheme_identity_problems
from tests.test_derive_corpus_targets import (
    depth_options,
    full_width_phase,
    history_row,
    narrowed_phase,
    run_derive,
    write_corpus,
)


def g10_row(*, extended: bool = False, game_id: int = 0) -> dict:
    row = history_row(game_id=game_id)
    moves = list(chess.Board(row["fen"]).legal_moves)
    base = {m.uci(): float(100 - i * 10) for i, m in enumerate(moves)}
    selected = list(base)[:8]
    d10 = {m: float(300 - i * (5 if extended else 30)) for i, m in enumerate(selected)}
    row["phases"] = [
        full_width_phase(row["fen"], {9: base}),
        narrowed_phase(
            {9: {m: float(150 - i * 10) for i, m in enumerate(selected)}, 10: d10},
            depth_requested=10,
        ),
    ]
    if extended:
        row["phases"].append(
            narrowed_phase(
                {12: {m: float(500 - i * 20) for i, m in enumerate(selected[:4])}},
                index=2,
                depth_requested=12,
            )
        )
    row["staircase_gate"] = gen.StaircaseGateDecision(
        5.0 if extended else 30.0,
        extended,
        "margin_at_or_below_threshold" if extended else "margin_above_threshold",
        10,
    ).as_row()
    return row


def selected(row: dict) -> tuple:
    return adaptive.select(row, {m.uci() for m in chess.Board(row["fen"]).legal_moves})


@pytest.mark.parametrize(("extended", "depth"), [(False, 10), (True, 12)])
def test_saved_gate_and_ancestor_rosters_select_exact_depth(
    extended: bool, depth: int
) -> None:
    row = g10_row(extended=extended)
    values, actual, reason = selected(row)
    assert actual == depth
    assert reason == f"selected_d{depth}"
    assert values
    assert max(values.values()) == (500 if extended else 300)
    # Subset membership alone is insufficient: generator uses the precise prior ranked topK.
    wrong = copy.deepcopy(row)
    wrong["phases"][1]["searchmoves"].reverse()
    assert selected(wrong) == (None, 9, "fallback:phase1_identity_or_ancestor_roster")


def test_single_move_d10_is_explicit_valid_stop() -> None:
    fen = "7k/5K2/8/8/8/8/8/8 b - - 0 1"
    moves = {m.uci(): 100.0 for m in chess.Board(fen).legal_moves}
    assert len(moves) == 1
    row = {
        "fen": fen,
        "phases": [
            full_width_phase(fen, {9: moves}),
            narrowed_phase({10: dict.fromkeys(moves, 200.0)}, depth_requested=10),
        ],
        "staircase_gate": gen.StaircaseGateDecision(
            None, False, "fewer_than_two_moves", 10
        ).as_row(),
    }
    assert selected(row) == (
        dict.fromkeys(moves, 200.0),
        10,
        "selected_single_move_d10",
    )


@pytest.mark.parametrize("kind", ["roster", "gate", "incomplete", "missing"])
def test_invalid_later_observations_use_explicit_baseline_fallback(kind: str) -> None:
    row = g10_row(extended=True)
    if kind == "roster":
        row["phases"][2]["per_depth"][0]["lines"][0][1] = "a1a1"
    elif kind == "gate":
        row["staircase_gate"]["margin_cp"] = 99
    elif kind == "incomplete":
        row["phases"][2]["per_depth"][0]["complete"] = False
    else:
        row["phases"].pop()
    values, depth, reason = selected(row)
    assert values is None
    assert depth == 9
    assert reason.startswith("fallback:")


def test_invalid_baseline_or_experiment_identity_is_fatal() -> None:
    row = g10_row()
    row["phases"][0]["per_depth"][0]["lines"][0][2] = float("nan")
    with pytest.raises(ValueError, match="invalid_roster"):
        selected(row)
    row = g10_row()
    row["staircase_gate"]["policy"] = "fixed"
    with pytest.raises(ValueError, match="identity"):
        selected(row)


@pytest.mark.parametrize("workers", [1, 2])
def test_actual_writer_changes_only_wdl_and_preserves_row_order(
    tmp_path: Path, workers: int
) -> None:
    rows = [g10_row(game_id=0), g10_row(extended=True, game_id=1), g10_row(game_id=2)]
    rows[2]["phases"][1]["per_depth"][1]["complete"] = False
    # Existing policy support exclusions must be identical in both arms.
    excluded = g10_row(game_id=3)
    lines = excluded["phases"][0]["per_depth"][0]["lines"]
    lines[-1][1] = lines[0][1]
    rows.append(excluded)
    source = write_corpus(tmp_path, rows)
    common = (
        "--policy-observation",
        "phase0",
        "--row-provenance",
        "--max-policy-support-misses",
        "1",
        "--workers",
        str(workers),
    )
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    run_derive(source, control, "uniform-d9", *common)
    summary = run_derive(
        source,
        candidate,
        "uniform-d9",
        *common,
        "--sf-value-selector",
        adaptive.SELECTOR,
    )
    apath = next(control.glob("shard_*.zarr"))
    bpath = next(candidate.glob("shard_*.zarr"))
    a = zarr.open_group(str(apath), mode="r")
    b = zarr.open_group(str(bpath), mode="r")
    assert len(list(a.array_keys())) == 17
    assert set(a.array_keys()) == set(b.array_keys())
    for name in a.array_keys():
        assert isinstance(name, str)
        if name != "search_wdl":
            np.testing.assert_array_equal(a[name][:], b[name][:])
            ad = apath / name
            bd = bpath / name
            assert {
                str(p.relative_to(ad)): p.read_bytes()
                for p in ad.rglob("*")
                if p.is_file()
            } == {
                str(p.relative_to(bd)): p.read_bytes()
                for p in bd.rglob("*")
                if p.is_file()
            }
    assert set(np.asarray(a["game_id"][:]).tolist()) == {0, 1, 2}
    assert (control / derive.POLICY_SUPPORT_MISSES_FILE).read_bytes() == (
        candidate / derive.POLICY_SUPPORT_MISSES_FILE
    ).read_bytes()
    assert summary["realized"]["rows_dropped_policy_support"] == 1
    assert not np.array_equal(
        np.asarray(a["search_wdl"][:]), np.asarray(b["search_wdl"][:])
    )
    i = int(np.flatnonzero(np.asarray(b["game_id"][:]) == 2)[0])
    np.testing.assert_array_equal(a["search_wdl"][i], b["search_wdl"][i])
    assert summary["scheme"]["value_source"] == adaptive.VALUE_SOURCE
    assert b.attrs["derive_value_source"] == adaptive.VALUE_SOURCE
    assert value_scheme_identity_problems([control, candidate])
    assert summary["realized"]["sf_value_selection_counts"] == {
        "fallback:incomplete_roster": 1,
        "selected_d10": 1,
        "selected_d12": 1,
    }


def test_opt_in_scope_and_old_full_width_depth_rule() -> None:
    scheme = replace(
        derive.parse_scheme("uniform-d9"),
        policy_observation="phase0",
        sf_value_selector=adaptive.SELECTOR,
    )
    with pytest.raises(ValueError, match="value-depth"):
        derive.with_value_depth(scheme, 12)
    with pytest.raises(ValueError, match="phase0"):
        replace(scheme, policy_observation="latest-phase")
    with pytest.raises(ValueError, match="search"):
        replace(depth_options(scheme), value_scheme="qz50")


def test_missing_requested_block_falls_back_but_broken_row_schema_is_fatal() -> None:
    row = g10_row(extended=True)
    row["phases"][2]["per_depth"] = []
    assert selected(row) == (None, 9, "fallback:phase2_requested_block")
    row = g10_row(extended=True)
    del row["phases"][2]["per_depth"][0]["lines"]
    with pytest.raises(KeyError, match="lines"):
        selected(row)
