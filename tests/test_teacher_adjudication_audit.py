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


def test_native_wdl_admission_and_actual_feed_guard(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import zarr
    from tests.test_g10_native_wdl_reuse import fixture
    args, _ = fixture(tmp_path, monkeypatch)
    # The synthetic graph calls its head 'value'; production remains /output/wdl.
    monkeypatch.setattr(audit, 'NATIVE_WDL_HEAD', 'value')
    native = audit.NativeWDLManifest(Path(args.native_wdl_manifest),
        args.expected_native_wdl_manifest_sha256, Path(args.sf_source),
        args.expected_sf_summary_sha256, args.expected_onnx_sha256)
    source = Path(args.sf_source)
    group = zarr.open_group(str(source / 'shard_000000.zarr'), mode='r')
    x = np.asarray(group['x'][:])
    values = native.load('shard_000000.zarr', group, x)
    np.testing.assert_array_equal(values, np.tile([.125, .25, .625], (len(x), 1)))
    native.guard()
    corrupt = x.copy()
    corrupt[:, 104] = 1 - corrupt[:, 104]  # Real history/meta input differs, IDs unchanged.
    with pytest.raises(ValueError, match='feed identity'):
        native.load('shard_000000.zarr', group, corrupt)
    with pytest.raises(ValueError, match='source/model/head'):
        audit.NativeWDLManifest(Path(args.native_wdl_manifest),
            args.expected_native_wdl_manifest_sha256, source,
            args.expected_sf_summary_sha256, 'f' * 64)
    with pytest.raises(ValueError, match='source/model/head'):
        audit.NativeWDLManifest(Path(args.native_wdl_manifest),
            args.expected_native_wdl_manifest_sha256, tmp_path / 'wrong-source',
            args.expected_sf_summary_sha256, args.expected_onnx_sha256)
    manifest = json.loads(Path(args.native_wdl_manifest).read_text())
    historical = Path(next(iter(manifest['invocations'][0]['producer'].values()))['path'])
    historical.write_text(historical.read_text() + '\n# changed')
    with pytest.raises(ValueError, match='evidence changed'):
        native.guard()


def test_optional_native_values_reach_real_audit_bank_without_policy_changes(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import zarr
    from tests.test_g10_native_wdl_reuse import fixture
    from tests.test_bt4_derived_wdl_sidecar import Session
    def distinct_values(_self, _names, feed):
        n = len(feed['input'])
        win = np.arange(n) / 8 + .125
        return [np.column_stack((win, np.full(n, .25), .75 - win)).astype('float16')]
    monkeypatch.setattr(Session, 'run', distinct_values)
    args, _ = fixture(tmp_path, monkeypatch)
    # This shared source fixture is a real derived/joined corpus with a synthetic
    # G10 admission stamp, not an adaptive-search fixture. Keep its actual d9
    # ruler; do not pretend the test exercises production adaptive selection.
    def baseline_ruler(row, legal):
        audit.adaptive.validate_baseline(row, legal)
        return audit.tactical._d9_scores(row, legal), 9, 'synthetic_d9_fixture'
    monkeypatch.setattr(audit.adaptive, 'select', baseline_ruler)
    monkeypatch.setattr(audit, 'NATIVE_WDL_HEAD', 'value')
    real_native = audit.NativeWDLManifest
    # Existing fixtures deliberately use different synthetic policy/value graphs.
    # Keep real admission/cache/feed validation, supplying its known graph identity.
    def native_fixture(path, digest, source, summary_sha, _policy_model):
        return real_native(path, digest, source, summary_sha, args.expected_onnx_sha256)
    monkeypatch.setattr(audit, 'NativeWDLManifest', native_fixture)
    source = Path(args.sf_source)
    cmanifest = tmp_path / 'ceres.json'
    cmanifest.write_text(json.dumps({'schema': 1, 'source': str(source),
        'source_summary_sha256': args.expected_sf_summary_sha256, 'entries': []}))
    # Only the Ceres provider boundary is synthetic; audit geometry and native
    # original-source qualification run normally against the real tiny corpus.
    def load_ceres(_self, _shard, group, rows):
        legal = np.asarray(group['legal_mask'][:]) != 0
        counts = legal.sum(axis=1)
        return {'legal_offsets': np.r_[0, np.cumsum(counts)],
                'legal_indices': np.nonzero(legal)[1],
                'policy_logits': np.zeros(int(counts.sum()), dtype='float16'),
                'value_logits': np.tile([1, 0, -1], (rows, 1)).astype('float16'),
                'value2_logits': np.tile([-1, 0, 1], (rows, 1)).astype('float16')}
    monkeypatch.setattr(audit.CeresManifest, 'load', load_ceres)
    manifest = tmp_path / 'adapter.json'
    from typing import Any
    options: dict[str, Any] = {'expected_manifest_sha256': audit.file_sha256(manifest),
               'ceres_manifest_path': cmanifest,
               'expected_ceres_manifest_sha256': audit.file_sha256(cmanifest)}
    baseline = audit.audit(manifest, out=tmp_path / 'audit_baseline', **options)
    changed = audit.audit(manifest, out=tmp_path / 'audit_native',
        native_wdl_manifest_path=Path(args.native_wdl_manifest),
        expected_native_wdl_manifest_sha256=args.expected_native_wdl_manifest_sha256, **options)
    assert baseline['metrics']['policy'] == changed['metrics']['policy']
    assert all(v['rows'] == 0 for v in baseline['metrics']['value'].values())
    assert all(v['rows'] == changed['rows'] == 3 for v in changed['metrics']['value'].values())
    rows = [json.loads(line) for line in (tmp_path / 'audit_native' / audit.ROW_BANK).read_text().splitlines()]
    assert all(r['policy_complementarity_gt300'] is not None for r in rows)
    for r in rows:
        detail = r['policy_complementarity_gt300']
        assert detail['flagged_mass']['bt4'] == pytest.approx(r['ranking_best_set_and_all_winner']['300']['inferior_bt4_mass'])
        assert len(detail['move_observations']) > 0
    assert all(r['value_inclusion'] == 'included' and len(r['value_losses']) == 6 for r in rows)
    assert all(sum(r['value_ruler_wdl']) == pytest.approx(1) for r in rows)
    expected = set(changed['metrics']['value'])
    assert all(set(r['value_losses']) == expected for r in rows)
    for row in rows:
        native = zarr.open_group(str(Path(args.wdl) / row['identity']['derived_shard']), mode='r')
        prediction = np.asarray(native['bt4_wdl_raw'][row['identity']['derived_row']])
        assert row['value_losses']['bt4_native']['brier'] == pytest.approx(
            audit._wdl_loss(prediction, np.asarray(row['value_ruler_wdl']))[0])
    for name, cell in changed['metrics']['value'].items():
        assert cell['brier_sum'] == pytest.approx(sum(r['value_losses'][name]['brier'] for r in rows))


def test_complementarity_tracks_same_moves_not_just_top_agreement() -> None:
    d9 = {'a': 50.0, 'b': 45.0, 'c': -400.0, 'd': -500.0}
    mapping = {m: i for i, m in enumerate(d9)}
    bt4 = np.pad([.1, .1, .7, .1], (0, COMPACT_POLICY_SIZE - 4))
    final = {'a': 40.0, 'b': 45.0, 'c': -300.0, 'd': -400.0}
    removed = audit.policy_complementarity(d9, {'a'}, final, mapping, bt4,
                                           np.pad([.1, .7, .1, .1], (0, COMPACT_POLICY_SIZE - 4)))
    shifted = audit.policy_complementarity(d9, {'a'}, final, mapping, bt4,
                                           np.pad([.1, .1, .1, .7], (0, COMPACT_POLICY_SIZE - 4)))
    assert removed['flagged_mass']['ceres'] == pytest.approx(.2)
    assert shifted['flagged_mass']['ceres'] == pytest.approx(.8)
    for bank in (removed, shifted):
        for cell in [bank['flagged_mass'], *bank['partitions'].values()]:
            assert cell['arithmetic50'] == pytest.approx((cell['bt4'] + cell['ceres']) / 2)
        assert sum(c['moves'] for c in bank['partitions'].values()) == 2
    assert removed['partitions']['confirmed']['bt4'] == pytest.approx(.8)
    assert removed['move_observations'][2]['policy_index'] == 2
    # SF's best-next gap is only 5cp: these constraints are distinct from T300's gate.
    assert audit.tactical._ordinary_d9_best(d9)[1] == 5


def test_complementarity_conservative_ties_and_narrowed_missing_scores() -> None:
    d9 = {'a': 500.0, 'b': 500.0, 'c': 0.0, 'd': -100.0, 'e': -200.0}
    mapping = {m: i for i, m in enumerate(d9)}
    p = np.pad([.1, .1, .2, .3, .3], (0, COMPACT_POLICY_SIZE - 5))
    bank = audit.policy_complementarity(d9, {'a', 'b'},
        {'a': 100.0, 'b': -100.0, 'c': 0.0, 'd': -100.0}, mapping, p, p)
    assert bank['partitions']['contradicted']['bt4'] == .2
    assert bank['partitions']['ties']['bt4'] == .3
    assert bank['partitions']['unavailable']['bt4'] == .3
    assert bank['move_observations'][-1]['final_effective_cp'] is None
    missing_winner = audit.policy_complementarity(d9, {'a', 'b'},
        {'a': 100.0, 'c': 0.0, 'd': -100.0}, mapping, p, p)
    assert missing_winner['partitions']['unavailable']['bt4'] == pytest.approx(.8)
    old = audit._update_ranking(audit.new_aggregate(), d9, {'a', 'b'},
        {'a': 100.0, 'b': -100.0, 'c': 0.0, 'd': -100.0}, {m: float(p[i]) for m, i in mapping.items()})['300']['all_winner']
    for name, cell in bank['partitions'].items():
        assert cell['bt4'] == pytest.approx(old[('unscored' if name == 'unavailable' else name) + '_bt4_mass'])
