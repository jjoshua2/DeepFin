"""Cheap fail-closed report tests; no engine build/model execution in pytest."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from native.bend_engine.multi_root.verify import ZERO, histogram, integer, obj, parse


def report() -> tuple[dict, dict]:
    root = {'root': 1, 'completed_simulations': 1, 'executed_real_rows': 1,
            'accepted_neural_rows': 1, 'rule_draw_replies': 0, 'used_nodes': 1,
            'stop_code': 0, 'simulation_budget': 4, 'neural_budget': 0,
            'neural_budget_met': None, 'searched_move': True, 'bestmove': 'e2e4'}
    work = {'schema': 'deepfin.multi-root-work.v1', 'scope': 'bounded_cohort', 'roots': 1,
            'wall_seconds': 0.1, 'completed_simulations': 1, 'forward_calls': 1,
            'dispatched_real_rows': 1, 'executed_real_rows': 1, 'accepted_neural_rows': 1,
            'physical_rows': 4, 'padded_rows': 3, 'useful_eps': 10, 'executed_eps': 10,
            'real_batch_histogram': {'1': 1}, 'physical_batch_histogram': {'4': 1},
            'gathering_seconds': 0.01, 'backend_and_transport_seconds': 0.02,
            'normalization_and_backup_seconds': 0.01,
            'phase_seconds': dict.fromkeys(('queue_wait', 'h2d', 'gpu', 'd2h')),
            'clock_resolution_seconds': 0.001, 'warmup_excluded': False, **dict.fromkeys(ZERO, 0)}
    return root, work


def output(root: dict, work: dict) -> str:
    return 'info string cohort_root ' + json.dumps(root) + '\ninfo string cohort_work ' + json.dumps(work) + '\n'


def check(root: dict, work: dict) -> dict:
    return parse(output(root, work), 1, 4, 4, 0, False)


def test_partial_batch_report() -> None:
    r, w = report()
    result = check(r, w)
    assert result['work']['padded_rows'] == 3
    assert result['work']['accepted_neural_rows'] == 1


@pytest.mark.parametrize('bad', [False, True, -1, 2**32, 1.0, '1', None])
def test_integers_are_strict(bad: object) -> None:
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|expected|missing|duplicate|unsupported|extra|noncanonical|unmeasured|incomplete|absent|EPS|wrong"):
        integer(bad)


@pytest.mark.parametrize('key', ZERO)
def test_incomplete_work_is_not_success(key: str) -> None:
    r, w = report()
    w[key] = 1
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|expected|missing|duplicate|unsupported|extra|noncanonical|unmeasured|incomplete|absent|EPS|wrong"):
        check(r, w)


@pytest.mark.parametrize(('key', 'bad'), [
    ('schema', 'deepfin.neural-work.v1'), ('scope', 'broker'), ('roots', 2),
    ('accepted_neural_rows', 2), ('executed_real_rows', 2), ('dispatched_real_rows', 2),
    ('completed_simulations', 2), ('forward_calls', 2), ('physical_rows', 5), ('padded_rows', 0),
    ('wall_seconds', float('nan')), ('wall_seconds', -1), ('wall_seconds', True),
    ('useful_eps', 40), ('useful_eps', float('inf')), ('executed_eps', True),
    ('real_batch_histogram', {'01': 1}), ('real_batch_histogram', {'0': 1}),
    ('real_batch_histogram', {'5': 1}), ('real_batch_histogram', {'1': True}),
    ('physical_batch_histogram', {'2': 2}), ('warmup_excluded', True),
    ('phase_seconds', {'gpu': 0}), ('clock_resolution_seconds', 0.000001),
    ('backend_and_transport_seconds', 0.2), ('gathering_seconds', -1),
])
def test_bad_cohort_metrics_fail(key: str, bad: object) -> None:
    r, w = report()
    w[key] = bad
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|expected|missing|duplicate|unsupported|extra|noncanonical|unmeasured|incomplete|absent|EPS|wrong"):
        check(r, w)


@pytest.mark.parametrize(('key', 'bad'), [
    ('root', 0), ('root', True), ('root', 2), ('completed_simulations', 5),
    ('executed_real_rows', 0), ('accepted_neural_rows', 2), ('used_nodes', 0),
    ('used_nodes', 4097), ('stop_code', 2), ('rule_draw_replies', 1),
    ('simulation_budget', 3), ('neural_budget', 3), ('neural_budget_met', True),
    ('searched_move', 1), ('bestmove', '0000'), ('bestmove', 'a9b8'), ('bestmove', 'e2e4x'),
])
def test_bad_root_report_fails(key: str, bad: object) -> None:
    r, w = report()
    r[key] = bad
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|expected|missing|duplicate|unsupported|extra|noncanonical|unmeasured|incomplete|absent|EPS|wrong"):
        check(r, w)


def test_missing_duplicate_extra_and_nonobject_records_fail() -> None:
    r, w = report()
    text = output(r, w)
    for bad in ['', text.splitlines()[0], text + text, 'garbage\n' + text,
                'info string cohort_root []\n' + text, text.replace('"root": 1', '"root": 1,"root": 1')]:
        with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|expected|missing|duplicate|unsupported|extra|noncanonical|unmeasured|incomplete|absent|EPS|wrong"):
            parse(bad, 1, 4, 4, 0, False)


def test_diagnostics_require_complete_nodes_and_actual_batch_histogram() -> None:
    r, w = report()
    prefix = 'info string cohort_batch 1 1 4\ninfo string cohort_node 1 0 ' + json.dumps([0] * 29) + '\n'
    text = prefix + output(r, w)
    assert parse(text, 1, 4, 4, 0, True)['nodes'][1][0] == [0] * 29
    for bad in [output(r, w), text.replace('1 1 4', '1 2 4'), text.replace('1 0 [', '1 1 ['),
                text.replace('cohort_node 1', 'cohort_node 2')]:
        with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|expected|missing|duplicate|unsupported|extra|noncanonical|unmeasured|incomplete|absent|EPS|wrong"):
            parse(bad, 1, 4, 4, 0, True)
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|expected|missing|duplicate|unsupported|extra|noncanonical|unmeasured|incomplete|absent|EPS|wrong"):
        parse(text, 1, 4, 4, 0, False)


def test_explicit_neural_budget_underfill_stays_visible() -> None:
    r, w = report()
    r['neural_budget'] = 3
    r['neural_budget_met'] = False
    assert parse(output(r, w), 1, 4, 4, 3, False)['roots'][1]['neural_budget_met'] is False
    r['neural_budget_met'] = True
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|expected|missing|duplicate|unsupported|extra|noncanonical|unmeasured|incomplete|absent|EPS|wrong"):
        parse(output(r, w), 1, 4, 4, 3, False)


def test_zero_time_is_unknown_eps_not_infinite_throughput() -> None:
    r, w = report()
    w.update(wall_seconds=0, useful_eps=None, executed_eps=None, gathering_seconds=0,
             backend_and_transport_seconds=0, normalization_and_backup_seconds=0)
    check(r, w)
    w['useful_eps'] = 0
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|expected|missing|duplicate|unsupported|extra|noncanonical|unmeasured|incomplete|absent|EPS|wrong"):
        check(r, w)


def test_helpers_reject_wrong_containers_and_key_aliases() -> None:
    for text in ['[]', 'null', '1', '{"a":1,"a":2}']:
        with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|expected|missing|duplicate|unsupported|extra|noncanonical|unmeasured|incomplete|absent|EPS|wrong"):
            obj(text)
    for value in [[], None, {1: 1}, {'١': 1}, {'1': 0}]:
        with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|expected|missing|duplicate|unsupported|extra|noncanonical|unmeasured|incomplete|absent|EPS|wrong"):
            histogram(deepcopy(value), 4)
