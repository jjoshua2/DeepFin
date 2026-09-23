"""Cheap deadline-report corruption checks; never compiles or runs a model."""
from __future__ import annotations

import pytest

from native.bend_engine.multi_root.verify import parse
from tests.test_bend_async_cohort import output, reports


def v2() -> tuple[dict, dict]:
    root, work = reports()
    root.update(deadline_offset_ms=25, deadline_expired=True)
    work.update(schema='deepfin.multi-root-async-work.v2', deadline_expired_mask=2)
    return root, work


def check(root: dict, work: dict, commands: str = 'deadline 1 25\nexpired 1') -> dict:
    text = ''.join('info string cohort_control '+s+'\n' for s in commands.splitlines())
    return parse(text+output(root, work), 1, 4, 4, 2, False, asynchronous=True)


def test_expired_physical_work_not_refunded() -> None:
    r, w = v2()
    result = check(r, w)
    assert result['work']['accepted_neural_rows'] == 1
    assert result['work']['executed_real_rows'] == 2
    assert result['roots'][1]['deadline_expired'] is True


@pytest.mark.parametrize('bad', [None, False, True, -1, 1.0, '2', 0, 1, 4, 2**32])
def test_expiry_mask_is_exact(bad: object) -> None:
    r, w = v2()
    w['deadline_expired_mask'] = bad
    with pytest.raises(ValueError, match=r"invalid|mismatch|missing|expiry|expired|requires|revived|control|range"):
        check(r, w)


@pytest.mark.parametrize('bad', [None, 1, 0, 'true', False])
def test_expiry_status_must_match_notice(bad: object) -> None:
    r, w = v2()
    r['deadline_expired'] = bad
    with pytest.raises(ValueError, match=r"invalid|mismatch|missing|expiry|expired|requires|revived|control|range"):
        check(r, w)


@pytest.mark.parametrize('bad', [None, False, True, -1, 25.0, '25', 2**63])
def test_due_time_is_exact_and_acknowledged(bad: object) -> None:
    r, w = v2()
    r['deadline_offset_ms'] = bad
    with pytest.raises(ValueError, match=r"invalid|mismatch|missing|expiry|expired|requires|revived|control|range"):
        check(r, w)


@pytest.mark.parametrize('field', ['deadline_offset_ms', 'deadline_expired'])
def test_missing_root_field_rejected(field: str) -> None:
    r, w = v2()
    del r[field]
    with pytest.raises(ValueError, match=r"invalid|mismatch|missing|expiry|expired|requires|revived|control|range"):
        check(r, w)


@pytest.mark.parametrize('commands', [
    '', 'deadline 1 25', 'expired 1', 'expired 1\ndeadline 1 25',
    'deadline 1 25\nexpired 1\nexpired 1',
    'deadline 1 25\nexpired 1\ndeadline 1 0',
    'deadline 0 25', 'deadline 2 25', 'deadline 01 25',
    'deadline 1 -1', 'deadline 1 3600001', 'deadline 1 0.5',
    'deadline 1 1 extra', 'expired 0', 'expired 2', 'expired 01', 'expired 1 extra',
])
def test_invalid_or_incomplete_notices_rejected(commands: str) -> None:
    r, w = v2()
    with pytest.raises(ValueError, match=r"invalid|mismatch|missing|expiry|expired|requires|revived|control|range"):
        check(r, w, commands)


def test_shortening_notices_allowed_before_expiry() -> None:
    r, w = v2()
    check(r, w, 'deadline 1 1000\ndeadline 1 25\nexpired 1')


def test_manual_stop_does_not_imply_deadline_expiry() -> None:
    r, w = v2()
    r.update(deadline_offset_ms=None, deadline_expired=False)
    w['deadline_expired_mask'] = 0
    check(r, w, 'cancel 1')
    r['deadline_offset_ms'] = 10000
    check(r, w, 'deadline 1 10000\ncancel 1')


def test_v1_cannot_smuggle_deadline_metadata() -> None:
    r, w = v2()
    w['schema'] = 'deepfin.multi-root-async-work.v1'
    with pytest.raises(ValueError, match='requires async v2'):
        check(r, w)


@pytest.mark.parametrize('bad', [None, 0, 1, False])
def test_expired_root_must_be_stopped(bad: object) -> None:
    r, w = v2()
    r['stop_code'] = bad
    with pytest.raises(ValueError, match=r"invalid|mismatch|missing|expiry|expired|requires|revived|control|range"):
        check(r, w)
