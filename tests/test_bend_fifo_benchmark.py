"""Measurement admission tests; no native benchmark runs in ordinary pytest."""
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from native.bend_engine.multi_root import benchmark_fifo as bench


def panel() -> list[dict[str, Any]]:
    return [{'case': case, 'async': mode, 'pair': pair, 'arm': arm,
             'order': (pair + index) % 2, 'repeats': 2, 'work_sha256': case + str(mode),
             'process_seconds': 4.0 if arm == 'list' else 2.0,
             'coordinator_seconds': 2.0 if arm == 'list' else 1.0}
            for case in bench.CASES for mode in (False, True)
            for pair in range(bench.PAIRS) for index, arm in enumerate(bench.ARMS)]


def test_complete_paired_measurements() -> None:
    result = bench.summarize(panel())
    assert len(result) == 6
    for group in result.values():
        for field in ('coordinator_seconds', 'process_seconds'):
            assert group[field]['median_list_over_fifo'] == 2.0
            assert group[field]['paired_ratios'] == [2.0] * 6
            assert group[field]['decision'] == 'consistent_over_5pct_improvement'


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'unknown', 'work', 'repeats', 'order'])
def test_mismatched_panel_rejected(fault: str) -> None:
    rows = panel()
    if fault == 'missing':
        rows.pop()
    elif fault == 'duplicate':
        rows[-1] = dict(rows[0])
    elif fault == 'unknown':
        rows[-1]['case'] = 'unknown'
    elif fault == 'work':
        rows[-1]['work_sha256'] = 'different'
    elif fault == 'repeats':
        rows[-1]['repeats'] = 3
    else:
        rows[-1]['order'] ^= 1
    with pytest.raises(ValueError, match='panel|work|order'):
        bench.summarize(rows)


@pytest.mark.parametrize('value', [-1.0, float('nan'), float('inf'), True])
@pytest.mark.parametrize('field', ['process_seconds', 'coordinator_seconds'])
def test_invalid_duration_rejected(value: float, field: str) -> None:
    rows = panel()
    rows[0][field] = value
    with pytest.raises(ValueError, match='time'):
        bench.summarize(rows)


def test_zero_process_duration_rejected() -> None:
    rows = panel()
    rows[0]['process_seconds'] = 0.0
    with pytest.raises(ValueError, match='duration'):
        bench.summarize(rows)


def test_impossible_internal_time_rejected() -> None:
    rows = panel()
    rows[0]['coordinator_seconds'] = 9.0
    with pytest.raises(ValueError, match='exceeds'):
        bench.summarize(rows)


@pytest.mark.parametrize('duration', [0.0, 0.01, 0.199])
def test_short_internal_sample_has_no_ratio(duration: float) -> None:
    rows = panel()
    rows[0]['coordinator_seconds'] = duration
    group = bench.summarize(rows)['single/sync']
    assert group['coordinator_seconds']['reliable'] is False
    assert group['coordinator_seconds']['median_list_over_fifo'] is None
    assert group['coordinator_seconds']['paired_ratios'] is None
    assert group['process_seconds']['reliable'] is True


def test_repetition_does_not_hide_quantization() -> None:
    rows = panel()
    for row in rows:
        row['repeats'] = 8
        row['coordinator_seconds'] = 0.3  # Above aggregate floor, below 50ms/run.
    assert bench.summarize(rows)['single/sync']['coordinator_seconds']['reliable'] is False


@pytest.mark.parametrize('factor', [0.5, 1.0, 1.03])
def test_negative_and_inconclusive_results_retained(factor: float) -> None:
    rows = panel()
    for row in rows:
        if row['arm'] == 'list':
            row['coordinator_seconds'] = factor
    metric = bench.summarize(rows)['single/sync']['coordinator_seconds']
    assert metric['median_list_over_fifo'] == factor
    expected = 'consistent_over_5pct_regression' if factor == 0.5 else 'inconclusive_at_5pct'
    assert metric['decision'] == expected


def test_one_contrary_pair_blocks_consistent_improvement() -> None:
    rows = panel()
    rows[0]['coordinator_seconds'] = 0.5
    metric = bench.summarize(rows)['single/sync']['coordinator_seconds']
    assert metric['median_list_over_fifo'] == 2.0
    assert metric['decision'] == 'inconclusive_at_5pct'


def test_execute_disables_trace_and_uses_requested_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}
    monkeypatch.setenv('DEEPFIN_BEND_MODEL_TRACE', 'unwanted.bin')
    monkeypatch.setenv('DEEPFIN_MULTI_TEST_FAULT', 'nan')
    monkeypatch.setenv('DEEPFIN_COHORT_ASYNC', '0')

    def run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        seen.update({'cmd': cmd, **kwargs})
        return SimpleNamespace(returncode=0, stderr='', stdout='output')

    def parse(text: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        assert text == 'output'
        assert args == (1, 4, 256, 0, False)
        assert kwargs == {'asynchronous': True}
        return {'work': {}}

    monkeypatch.setattr(bench.subprocess, 'run', run)
    monkeypatch.setattr(bench, 'parse', parse)
    parsed, wall, hashed = bench.execute(Path('runner'), ('startpos',), True, False)
    assert parsed == {'work': {}} and wall >= 0 and len(hashed) == 64
    assert seen['cmd'][1:] == ['--threads', '1', '--', '256', '8', '0', '0', 'startpos']
    assert seen['env']['DEEPFIN_COHORT_ASYNC'] == '1'
    assert 'DEEPFIN_BEND_MODEL_TRACE' not in seen['env']
    assert 'DEEPFIN_MULTI_TEST_FAULT' not in seen['env']
    assert seen['timeout'] == 60 and seen['input'] == ''
