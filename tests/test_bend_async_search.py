"""Cheap report/identity failure cases only; full native execution is opt-in."""
from __future__ import annotations

import json

import pytest

from native.bend_engine.async_probe.verify_search import RETIRED, WORK, one_report, snapshot


def decision() -> dict[str, int]:
    return {'search_epoch': 3, 'dispatched_real_rows': 2, 'executed_real_rows': 1,
            'accepted_neural_rows': 1, 'cancelled_rows': 1, 'unconfirmed_forward_rows': 1}


def lines(report: object) -> list[str]:
    return [WORK + json.dumps(report), 'bestmove e2e4']


def check(rows: list[str]) -> None:
    snapshot(rows, 3, dispatched=2, executed=1, accepted=1, cancelled=1)


def test_physical_retirement_cannot_replace_or_duplicate_decision() -> None:
    report = decision()
    rows = [RETIRED + json.dumps({**report, 'search_epoch': 2}), *lines(report)]
    check(rows)
    assert one_report(rows, RETIRED, 2)['search_epoch'] == 2
    with pytest.raises(AssertionError):
        check([RETIRED + json.dumps(report), 'bestmove e2e4'])
    with pytest.raises(AssertionError):
        check([*lines(report), WORK + json.dumps(report)])


@pytest.mark.parametrize('bad', [True, 3.0, None, '3', 2, 4])
def test_search_epoch_must_be_exact_integer(bad: object) -> None:
    with pytest.raises(AssertionError):
        check(lines({**decision(), 'search_epoch': bad}))


@pytest.mark.parametrize('key', ['dispatched_real_rows', 'executed_real_rows',
                                 'accepted_neural_rows', 'cancelled_rows', 'unconfirmed_forward_rows'])
@pytest.mark.parametrize('bad', [False, True, None, -1, 4, '1'])
def test_counter_mismatch_fails_even_when_boolean_compares_equal(key: str, bad: object) -> None:
    with pytest.raises(AssertionError):
        check(lines({**decision(), key: bad}))


@pytest.mark.parametrize('bad', [None, [], 3, True])
def test_nonobject_report_rejected(bad: object) -> None:
    with pytest.raises(AssertionError):
        check(lines(bad))


@pytest.mark.parametrize('moves', [[], ['bestmove e2e4', 'bestmove d2d4']])
def test_exactly_one_move_per_decision(moves: list[str]) -> None:
    with pytest.raises(AssertionError):
        check([WORK + json.dumps(decision()), *moves])
