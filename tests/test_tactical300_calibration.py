"""Retrospective Tactical300 calibration against saved G10 deeper searches."""

from __future__ import annotations

import copy

import chess
import numpy as np
import pytest

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import compact_index_for_move
from scripts import tactical300_calibration as calibration
from tests.test_adaptive_sf_value import g10_row


def _policy_for(row: dict, masses: dict[str, float]) -> np.ndarray:
    board = chess.Board(row["fen"])
    policy = np.zeros(COMPACT_POLICY_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        policy[compact_index_for_move(board, move)] = float(masses.get(move.uci(), 0.0))
    total = float(policy.sum(dtype=np.float64))
    assert total > 0
    policy /= total
    return policy


def _moves(row: dict) -> list[str]:
    return [line[1] for line in row["phases"][0]["per_depth"][0]["lines"]]


def _set_d9_scores(row: dict, first: float, second: float) -> tuple[str, str]:
    lines = row["phases"][0]["per_depth"][0]["lines"]
    a, b = str(lines[0][1]), str(lines[1][1])
    lines[0][2] = first
    lines[1][2] = second
    for index, line in enumerate(lines[2:], 2):
        line[2] = second - 20.0 * index
    return a, b


def test_exact_300_is_not_threshold300_but_301_is() -> None:
    row = g10_row(extended=False)
    a, b = _set_d9_scores(row, 500.0, 200.0)
    result = calibration.analyze_row(row, _policy_for(row, {b: 0.9, a: 0.1}))
    assert result["kind"] == "ordinary"
    assert result["d9_gap_cp"] == 300.0
    assert result["disagreement"] is True
    aggregate = calibration._new_aggregate()
    calibration.aggregate_row(aggregate, result)
    assert aggregate["thresholds"]["300"]["eligible_gap"] == 0

    row = g10_row(extended=False)
    a, b = _set_d9_scores(row, 501.0, 200.0)
    result = calibration.analyze_row(row, _policy_for(row, {b: 0.9, a: 0.1}))
    aggregate = calibration._new_aggregate()
    calibration.aggregate_row(aggregate, result)
    assert aggregate["thresholds"]["300"]["eligible_gap"] == 1
    assert aggregate["thresholds"]["300"]["disagreements"] == 1


def test_d12_can_vindicate_bt4_against_large_d9_gap() -> None:
    row = g10_row(extended=True)
    a, b = _set_d9_scores(row, 600.0, 100.0)
    d12 = row["phases"][2]["per_depth"][0]["lines"]
    assert {line[1] for line in d12} >= {a, b}
    for line in d12:
        if line[1] == a:
            line[2] = 100.0
        elif line[1] == b:
            line[2] = 500.0
        else:
            line[2] = 0.0
    result = calibration.analyze_row(row, _policy_for(row, {b: 0.8, a: 0.2}))
    assert result["final_depth"] == 12
    assert result["d9_gap_cp"] == 500.0
    assert result["disagreement"] is True
    assert result["roster_case"] == "both_complete"
    assert result["pairwise"] == "bt4"
    assert result["global_best"] == "bt4"

    aggregate = calibration._new_aggregate()
    calibration.aggregate_row(aggregate, result)
    cell = aggregate["thresholds"]["300"]
    assert (cell["adjudicable"], cell["bt4_pairwise_wins"]) == (1, 1)
    assert aggregate["threshold300_by_confidence"]["ge_0.75"]["bt4_pairwise_wins"] == 1
    assert aggregate["threshold300_by_depth"]["12"]["bt4_pairwise_wins"] == 1


def test_d10_can_reorder_roster_and_confirm_d9() -> None:
    row = g10_row(extended=False)
    a, b = _set_d9_scores(row, 700.0, 100.0)
    d10 = [block for block in row["phases"][1]["per_depth"] if block["depth"] == 10][0]["lines"]
    # The saved requested roster is fixed, but returned rank order may differ.
    a_index = next(i for i, line in enumerate(d10) if line[1] == a)
    b_index = next(i for i, line in enumerate(d10) if line[1] == b)
    d10[a_index][2] = 700.0
    d10[b_index][2] = 100.0
    d10.sort(key=lambda line: -float(line[2]))
    for rank, line in enumerate(d10, 1):
        line[0] = rank
    row["staircase_gate"]["margin_cp"] = float(d10[0][2]) - float(d10[1][2])
    result = calibration.analyze_row(row, _policy_for(row, {b: 0.8, a: 0.2}))
    assert result["final_depth"] == 10
    assert result["pairwise"] == "d9"
    assert result["global_best"] == "d9"


def test_absent_bt4_move_is_not_given_an_invented_deeper_score() -> None:
    row = g10_row(extended=True)
    moves = _moves(row)
    a = moves[0]
    outside_final = moves[6]
    _set_d9_scores(row, 700.0, 100.0)
    policy = _policy_for(row, {outside_final: 0.95, a: 0.05})
    result = calibration.analyze_row(row, policy)
    assert result["disagreement"] is True
    assert result["final_depth"] == 12
    assert result["roster_case"] in {"d9_only_scored", "d9_complete_bt4_partial"}
    assert result["pairwise"] is None
    aggregate = calibration._new_aggregate()
    calibration.aggregate_row(aggregate, result)
    cell = aggregate["thresholds"]["300"]
    assert cell["adjudicable"] == 0
    assert cell["bt4_pairwise_wins"] == 0


def test_winning_mate_persistence_is_categorical() -> None:
    row = g10_row(extended=True)
    moves = _moves(row)
    mate, other = moves[0], moves[1]
    row["phases"][0]["per_depth"][0]["lines"][0][2] = 99900.0
    d12 = row["phases"][2]["per_depth"][0]["lines"]
    for line in d12:
        line[2] = 99900.0 if line[1] == mate else 0.0
    result = calibration.analyze_row(row, _policy_for(row, {other: 0.9, mate: 0.1}))
    assert result["kind"] == "winning_mate"
    assert result["bt4_agrees_with_d9_winning_mate"] is False
    assert result["final_has_winning_mate"] is True
    assert result["final_preserves_d9_winning_mate"] is True
    assert result["bt4_top_is_final_winning_mate"] is False


def test_losing_mate_row_is_not_folded_into_centipawn_thresholds() -> None:
    row = g10_row(extended=False)
    lines = row["phases"][0]["per_depth"][0]["lines"]
    lines[-1][2] = -99900.0
    top = str(lines[0][1])
    result = calibration.analyze_row(row, _policy_for(row, {top: 1.0}))
    assert result["kind"] == "losing_mate_present"
    aggregate = calibration._new_aggregate()
    calibration.aggregate_row(aggregate, result)
    assert aggregate["losing_mate_rows"] == 1
    assert all(cell["eligible_gap"] == 0 for cell in aggregate["thresholds"].values())


def test_decision_requires_full_coverage_and_minimum_denominator() -> None:
    aggregate = calibration._new_aggregate()
    cell = aggregate["thresholds"]["300"]
    cell["adjudicable"] = calibration.MIN_ADJUDICABLE
    cell["bt4_pairwise_wins"] = 50
    assert calibration.decision_from_aggregate(aggregate, full_source_coverage=False)["verdict"] == "PARTIAL_NO_DECISION"

    cell["adjudicable"] = calibration.MIN_ADJUDICABLE - 1
    assert calibration.decision_from_aggregate(aggregate, full_source_coverage=True)["verdict"] == "INSUFFICIENT_ADJUDICABILITY"

    cell["adjudicable"] = calibration.MIN_ADJUDICABLE
    cell["bt4_pairwise_wins"] = 50
    blocked = calibration.decision_from_aggregate(aggregate, full_source_coverage=True)
    assert blocked["bt4_pairwise_win_rate"] == pytest.approx(0.05)
    assert blocked["verdict"] == "BLOCK_D9_ONLY_REQUIRE_DEPTH_STABILITY"

    cell["bt4_pairwise_wins"] = 49
    allowed = calibration.decision_from_aggregate(aggregate, full_source_coverage=True)
    assert allowed["verdict"] == "NO_5PCT_BLOCK_CALIBRATION_STILL_REQUIRED_FOR_ADMISSION"
    assert allowed["training_admission"] is False


def test_invalid_g10_identity_remains_fatal() -> None:
    row = g10_row()
    row = copy.deepcopy(row)
    row["staircase_gate"]["policy"] = "different"
    move = _moves(row)[0]
    with pytest.raises(ValueError, match="identity"):
        calibration.analyze_row(row, _policy_for(row, {move: 1.0}))
