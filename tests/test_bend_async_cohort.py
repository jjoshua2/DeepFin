"""Cheap failure cases for the async cohort report
     never builds an engine."""
from __future__ import annotations

import json

import pytest

from native.bend_engine.multi_root.verify import ZERO, parse


def reports() -> tuple[dict, dict]:
    root = {'root': 1, 'completed_simulations': 1, 'executed_real_rows': 2,
            'dispatched_real_rows': 2, 'accepted_neural_rows': 1, 'cancelled_rows': 1,
            'cancel_requested': True, 'rule_draw_replies': 0, 'used_nodes': 2,
            'stop_code': 2, 'simulation_budget': 4, 'neural_budget': 2,
            'neural_budget_met': False, 'searched_move': True, 'bestmove': 'e2e4'}
    work = {**dict.fromkeys(ZERO, 0), 'schema':'deepfin.multi-root-async-work.v1',
            'scope':'bounded_async_cohort', 'roots':1, 'wall_seconds':0.1,
            'completed_simulations':1, 'forward_calls':2, 'dispatched_real_rows':2,
            'executed_real_rows':2, 'accepted_neural_rows':1, 'cancelled_rows':1,
            'executed_wasted_rows':1, 'physical_rows':8, 'padded_rows':6,
            'useful_eps':10, 'executed_eps':20, 'real_batch_histogram':{'1':2},
            'physical_batch_histogram':{'4':2}, 'gathering_seconds':None,
            'backend_and_transport_seconds':None, 'normalization_and_backup_seconds':None,
            'phase_seconds':dict.fromkeys(('queue_wait','h2d','gpu','d2h')),
            'clock_resolution_seconds':0.001, 'warmup_excluded':False}
    return root, work


def output(r: dict, w: dict) -> str:
    return 'info string cohort_root '+json.dumps(r)+'\ninfo string cohort_work '+json.dumps(w)+'\n'


def check(r: dict, w: dict) -> dict:
    return parse(output(r,w), 1,4,4,2,False,asynchronous=True)


def test_cancelled_compute_is_counted_but_not_useful() -> None:
    r,w=reports()
    assert check(r,w)['work']['executed_eps'] == 2*check(r,w)['work']['useful_eps']
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|refunded|schema|phase|EPS|unsupported|unexpected"):
        parse(output(r,w),1,4,4,2,False)


@pytest.mark.parametrize(('key','bad'), [
    ('schema','deepfin.multi-root-work.v1'),('scope','bounded_cohort'),
    ('accepted_neural_rows',2),('cancelled_rows',0),('executed_wasted_rows',0),
    ('executed_real_rows',1),('dispatched_real_rows',1),('forward_calls',1),
    ('useful_eps',20),('executed_eps',10),('physical_rows',2),('padded_rows',0),
    ('gathering_seconds',0.0),('backend_and_transport_seconds',0.01),
    ('normalization_and_backup_seconds',0),('warmup_excluded',True),
    ('unconfirmed_forward_rows',1),('unresolved_rows',1),('failed_rows',1),
    ('failed_forward_rows',1),('stale_rows',1),('rejected_rows',1),
])
def test_contradictory_accounting_fails(key: str,bad: object) -> None:
    r,w=reports()
    w[key]=bad
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|refunded|schema|phase|EPS|unsupported|unexpected"):
        check(r,w)


@pytest.mark.parametrize('key',['cancelled_rows','dispatched_real_rows','executed_real_rows','accepted_neural_rows'])
@pytest.mark.parametrize('bad',[None,True,False,1.0,'1',-1,2**32])
def test_root_counts_are_strict(key: str,bad: object) -> None:
    r,w=reports()
    r[key]=bad
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|refunded|schema|phase|EPS|unsupported|unexpected"):
        check(r,w)


@pytest.mark.parametrize('bad',[False,None,0,1,'true'])
def test_cancelled_output_requires_explicit_request(bad: object) -> None:
    r,w=reports()
    r['cancel_requested']=bad
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|refunded|schema|phase|EPS|unsupported|unexpected"):
        check(r,w)


def test_cancelled_admission_is_not_refunded() -> None:
    r,w=reports()
    r['neural_budget']=1
    r['neural_budget_met']=True
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|refunded|schema|phase|EPS|unsupported|unexpected"):
        parse(output(r,w),1,4,4,1,False,asynchronous=True)


@pytest.mark.parametrize('line',['cohort_ready','cohort_control cancel 1','cohort_control stop',
                                 'cohort_control quit','cohort_control error invalid-root',
                                 'cohort_control error invalid-command','cohort_control error malformed-line'])
def test_control_records_do_not_become_neural_work(line: str) -> None:
    r,w=reports()
    result=parse('info string '+line+'\n'+output(r,w),1,4,4,2,False,asynchronous=True)
    assert result['work']['accepted_neural_rows']==1


@pytest.mark.parametrize('line',['cohort_ready extra','cohort_control cancel 0','cohort_control cancel 2',
                                 'cohort_control cancel 01','cohort_control unexpected','cohort_control'])
def test_malformed_control_records_fail(line: str) -> None:
    r,w=reports()
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|refunded|schema|phase|EPS|unsupported|unexpected"):
        parse('info string '+line+'\n'+output(r,w),1,4,4,2,False,asynchronous=True)


def test_missing_async_phase_is_not_unknown_measurement() -> None:
    r,w=reports()
    del w['backend_and_transport_seconds']
    with pytest.raises(ValueError, match=r"invalid|mismatch|reconcile|refunded|schema|phase|EPS|unsupported|unexpected"):
        check(r,w)
