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


def test_zero_coverage_and_missing_policy_have_separate_denominators() -> None:
    board = chess.Board()
    moves = [m.uci() for m in board.legal_moves]
    aggregate = audit.new_aggregate()
    cell = aggregate['policy']['bt4']
    zero = audit.conditional_policy_metrics(board, _policy(board, {moves[0]: 1}), {moves[1]: 10})
    covered = audit.conditional_policy_metrics(board, _policy(board, {moves[1]: 1}), {moves[1]: 10})
    audit._add_policy_metric(cell, zero)
    audit._add_policy_metric(cell, covered)
    audit._add_policy_metric(cell, None)
    audit._add_policy_metric(cell, None, policy_present=False)
    result = audit.finalize(aggregate)['policy']['bt4']
    assert result['rows'] == 4
    assert result['coverage_rows'] == 2
    assert result['zero_coverage_rows'] == result['regret_rows'] == 1
    assert result['invalid_final_rows'] == result['missing_policy_rows'] == 1
    assert result['mean_scored_mass'] == .5
    assert result['mean_conditional_regret_cp'] == 0


def test_unknown_reversal_is_not_a_negative_or_a_search_savings_claim() -> None:
    aggregate = audit.new_aggregate()
    for reversal in (True, False, None):
        audit._update_position_strata(aggregate, ['quiet'], reversal=reversal, bt4_regret=None)
        audit._update_routing(aggregate, disagreement=reversal is None, gap=400,
                              top_probability=.8, reversal=reversal)
    result = audit.finalize(aggregate)
    assert result['position_strata']['quiet']['reversal_rate'] == .5
    assert result['position_strata']['quiet']['unscored_rows'] == 1
    route = result['routing']['bt4_disagrees_d9']
    assert route['search_fraction'] == 0
    assert route['selection_fraction_all_ordinary'] == pytest.approx(1 / 3)
    assert route['unscored'] == 1


def test_tied_best_set_does_not_validate_every_winner_or_duplicate_inferior_mass() -> None:
    aggregate = audit.new_aggregate()
    row = audit._update_ranking(aggregate, {'a': 500, 'b': 500, 'c': 0, 'd': -100},
                                {'a', 'b'}, {'a': 100, 'b': -100, 'c': 0},
                                {'a': .1, 'b': .2, 'c': .3, 'd': .4})['300']
    assert row['confirmed'] == 1
    assert row['all_winner']['contradicted'] == 1
    assert row['all_winner']['contradicted_bt4_mass'] == .3
    assert row['all_winner']['unscored_bt4_mass'] == .4
    assert row['inferior_bt4_mass'] == pytest.approx(.7)
    assert row['constraints'] == 2


def test_row_bank_distinguishes_sf_margin_from_bt4_top_gap_and_missing_ceres() -> None:
    import json
    row = g10_row(extended=False)
    board = chess.Board(row['fen'])
    moves = [m.uci() for m in board.legal_moves]
    scores = dict.fromkeys(moves, -400.0)
    scores[moves[0]], scores[moves[1]] = 50.0, 45.0
    _set_d9(row, scores)
    # Keep exact G10 rank roster/gate; only the phase0 scores change here.
    aggregate = audit.new_aggregate()
    ref = {'source_dir': '/tmp/raw', 'source_namespace': 'qualified-namespace',
           'source_shard': 'w00.jsonl.zst', 'source_row': 3, 'worker_id': 0,
           'game_id': 7, 'ply': 3, 'input_key': 'a' * 32, 'stored_input_key': 'b' * 32}
    bank = audit.analyze_row(aggregate, audit.Selector(), raw_row=row,
                             raw_bt4=_policy(board, {moves[2]: 1}).astype('float32'),
                             ref=ref, derived_shard='shard_000000.zarr', derived_row=2,
                             derived_wdl=np.array([.3, .4, .3]))
    assert bank['position_strata'] == audit._position_strata(board, row)
    assert set(bank['position_strata']) <= set(aggregate['position_strata'])
    assert bank['d9_best_minus_next_lower_cp'] == 5
    assert bank['d9_best_minus_bt4_top_best_cp'] == 450
    assert bank['d9_best_count'] == 1
    assert bank['identity']['source_namespace'] == 'qualified-namespace'
    assert bank['final_depth'] in (9, 10, 12)
    assert bank['final_reason']
    assert bank['policy']['ceres'] is None
    assert aggregate['policy']['ceres']['missing_policy_rows'] == 1
    assert bank['ranking_best_set_and_all_winner']['300']['inferior_bt4_mass'] == 1
    json.dumps(bank, allow_nan=False)
