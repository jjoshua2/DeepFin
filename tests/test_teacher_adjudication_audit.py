"""Offline teacher-adjudication geometry and selection tests."""
from __future__ import annotations

import copy

import chess
import numpy as np
import pytest

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import compact_index_for_move
from scripts import teacher_adjudication_audit as audit
from tests.test_adaptive_sf_value import g10_row


def _policy(board: chess.Board, weights: dict[str, float]) -> np.ndarray:
    result = np.zeros(COMPACT_POLICY_SIZE, dtype=np.float64)
    for move in board.legal_moves:
        result[compact_index_for_move(board, move)] = weights.get(move.uci(), 0.0)
    total = float(result.sum())
    assert total > 0
    return result / total


def _set_d9(row: dict, scores: dict[str, float]) -> None:
    block = next(block for block in row["phases"][0]["per_depth"] if block["depth"] == 9)
    for line in block["lines"]:
        line[2] = float(scores[str(line[1])])


def test_conditional_regret_reports_roster_coverage() -> None:
    board = chess.Board()
    moves = [move.uci() for move in board.legal_moves]
    policy = _policy(board, {moves[0]: 0.5, moves[1]: 0.25, moves[2]: 0.25})
    metric = audit.conditional_policy_metrics(
        board,
        policy,
        {moves[0]: 100.0, moves[1]: 0.0},
    )
    assert metric is not None
    assert metric["scored_mass"] == pytest.approx(0.75)
    assert metric["conditional_regret_cp"] == pytest.approx(100.0 / 3.0)
    assert metric["conditional_best_mass"] == pytest.approx(2.0 / 3.0)


def test_js_arithmetic_and_geometric_are_well_defined() -> None:
    board = chess.Board()
    mapping, legal = audit._legal_map(board)
    moves = list(mapping)
    left = _policy(board, {moves[0]: 0.8, moves[1]: 0.2})
    right = _policy(board, {moves[0]: 0.2, moves[1]: 0.8})
    assert audit.js_divergence(left, left, legal) == pytest.approx(0.0)
    assert audit.js_divergence(left, right, legal) > 0
    arithmetic = audit.arithmetic_mix(left, right, legal)
    assert arithmetic[mapping[moves[0]]] == pytest.approx(0.5)
    assert arithmetic[mapping[moves[1]]] == pytest.approx(0.5)
    geometric = audit.geometric_mix(left, right, legal)
    assert geometric is not None
    assert geometric[mapping[moves[0]]] == pytest.approx(0.5)
    assert geometric[mapping[moves[1]]] == pytest.approx(0.5)


def test_tactical300_preview_moves_half_inferior_mass_only_above_300cp() -> None:
    row = g10_row(extended=False)
    board = chess.Board(row["fen"])
    moves = [move.uci() for move in board.legal_moves]
    base = _policy(board, {moves[0]: 0.1, moves[1]: 0.9})
    scores = dict.fromkeys(moves, -1000.0)
    scores[moves[0]] = 500.0
    scores[moves[1]] = 199.0
    _set_d9(row, scores)
    changed = audit.tactical300_preview(row, base)
    mapping, _legal = audit._legal_map(board)
    assert changed[mapping[moves[0]]] == pytest.approx(0.55)
    assert changed[mapping[moves[1]]] == pytest.approx(0.45)

    exact = copy.deepcopy(row)
    scores[moves[1]] = 200.0
    _set_d9(exact, scores)
    unchanged = audit.tactical300_preview(exact, base)
    np.testing.assert_allclose(unchanged, base)


def test_ranking_constraints_measure_deeper_reversal_without_inventing_missing_moves() -> None:
    aggregate = audit.new_aggregate()
    d9 = {"a": 500.0, "b": 0.0, "c": -100.0}
    audit._update_ranking(aggregate, d9, {"a"}, {"a": 10.0, "b": 20.0})
    cell = aggregate["ranking"]["300"]
    assert cell["constraints"] == 2
    assert cell["contradicted"] == 1
    assert cell["unscored"] == 1


def test_routing_reports_search_fraction_and_reversal_capture() -> None:
    aggregate = audit.new_aggregate()
    audit._update_routing(
        aggregate,
        disagreement=True,
        gap=400.0,
        top_probability=0.8,
        reversal=True,
    )
    audit._update_routing(
        aggregate,
        disagreement=False,
        gap=400.0,
        top_probability=0.8,
        reversal=False,
    )
    final = audit.finalize(aggregate)
    conflict = final["routing"]["bt4_disagrees_d9"]
    assert conflict["search_fraction"] == pytest.approx(0.5)
    assert conflict["reversal_capture_rate"] == pytest.approx(1.0)


def test_selector_is_bounded_deterministic_and_unique() -> None:
    first = audit.Selector(quota=2)
    second = audit.Selector(quota=2)
    identities = [
        {"source_dir": "/tmp/source", "derived_shard": "shard_000000.zarr", "derived_row": i,
         "game_id": i, "ply": 2 * i, "raw_shard": "w00.jsonl.zst", "physical_row": i,
         "input_key": f"{i:064x}"}
        for i in range(10)
    ]
    for identity in identities:
        first.add("d9_bt4_large_conflict", identity)
    for identity in reversed(identities):
        second.add("d9_bt4_large_conflict", identity)
    assert first.selected() == second.selected()
    assert len(first.selected()) == 2
    assert len({item["derived_row"] for item in first.selected()}) == 2


def test_value_losses_and_registered_mixture_are_probability_grounded() -> None:
    target = np.asarray([0.8, 0.1, 0.1])
    perfect = audit._wdl_loss(target, target)
    wrong = audit._wdl_loss(np.asarray([0.1, 0.1, 0.8]), target)
    assert perfect[0] == pytest.approx(0.0)
    assert perfect[1] < wrong[1]


def test_selection_strata_prioritize_known_reversal() -> None:
    assert audit._selection_stratum(disagreement=True, gap=500.0, reversal=True) == "deeper_reversal"
    assert audit._selection_stratum(disagreement=True, gap=500.0, reversal=False) == "d9_bt4_large_conflict"
    assert audit._selection_stratum(disagreement=True, gap=100.0, reversal=False) == "d9_bt4_other_conflict"
    assert audit._selection_stratum(disagreement=False, gap=500.0, reversal=False) == "agreement_control"
