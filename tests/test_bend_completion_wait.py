"""Admission of raw paired wait measurements; native code runs only explicitly."""
from copy import deepcopy
from typing import Any

import pytest

from native.bend_engine.multi_root import benchmark_wait as bench


def panel() -> list[dict[str, Any]]:
    rows = []
    for case in bench.CASES:
        for mode in (False, True):
            for pair in range(bench.PAIRS):
                for index, arm in enumerate(bench.ARMS):
                    observation = {'wall_seconds': 0.5 if arm == 'sleep' else 0.25,
                                   'cpu_seconds': 0.1, 'coordinator_seconds': 0.15,
                                   'stdout_sha256': 'a' * 64}
                    rows.append({'case': case, 'async': mode, 'pair': pair, 'arm': arm,
                                 'order': (pair + index) % 2, 'repeats': 1,
                                 'work_sha256': case + str(mode), 'observations': [observation],
                                 **{k: observation[k] for k in ('wall_seconds', 'cpu_seconds', 'coordinator_seconds')}})
    return rows


def test_paired_gain_and_cpu() -> None:
    summary = bench.summarize(panel())
    assert len(summary) == 6
    for entry in summary.values():
        assert entry['median_sleep_over_notify'] == 2.0
        assert entry['decision'] == 'consistent_over_5pct_improvement'
        assert entry['notify_over_sleep_cpu'] == 1.0


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'work', 'order', 'repeats', 'mode_type', 'pair_type'])
def test_incoherent_panel_rejected(fault: str) -> None:
    rows = panel()
    if fault == 'missing':
        rows.pop()
    elif fault == 'duplicate':
        rows[-1] = deepcopy(rows[0])
    else:
        key, value = {'work': ('work_sha256', 'other'), 'order': ('order', 1),
                      'repeats': ('repeats', 0), 'mode_type': ('async', 0),
                      'pair_type': ('pair', False)}[fault]
        rows[0][key] = value
    with pytest.raises(ValueError, match=r'panel|work|repetition'):
        bench.summarize(rows)


@pytest.mark.parametrize('field', ['wall_seconds', 'cpu_seconds', 'coordinator_seconds'])
@pytest.mark.parametrize('value', [-1.0, float('inf'), float('nan'), True])
def test_invalid_raw_duration(field: str, value: float) -> None:
    rows = panel()
    rows[0]['observations'][0][field] = value
    with pytest.raises(ValueError, match='duration'):
        bench.summarize(rows)


def test_aggregate_must_match_raw_samples() -> None:
    rows = panel()
    rows[0]['wall_seconds'] = 5
    with pytest.raises(ValueError, match='aggregate'):
        bench.summarize(rows)


def test_short_groups_keep_negative_result_without_ratio() -> None:
    rows = panel()
    rows[0]['wall_seconds'] = rows[0]['observations'][0]['wall_seconds'] = 0.19
    entry = bench.summarize(rows)['single/sync']
    assert entry['reliable'] is False
    assert entry['paired_ratios'] is None
    assert entry['median_sleep_over_notify'] is None
    assert entry['decision'] == 'below_measurement_floor'


@pytest.mark.parametrize(('duration', 'decision'), [(0.2, 'consistent_over_5pct_regression'),
                                                   (0.25, 'inconclusive_at_5pct'),
                                                   (0.255, 'inconclusive_at_5pct')])
def test_regression_and_inconclusive_panels(duration: float, decision: str) -> None:
    rows = panel()
    for row in rows:
        if row['arm'] == 'sleep':
            row['wall_seconds'] = row['observations'][0]['wall_seconds'] = duration
    assert bench.summarize(rows)['single/sync']['decision'] == decision


def test_one_contrary_pair_blocks_consistent_gain() -> None:
    rows = panel()
    rows[0]['wall_seconds'] = rows[0]['observations'][0]['wall_seconds'] = 0.2
    assert bench.summarize(rows)['single/sync']['decision'] == 'inconclusive_at_5pct'


def test_internal_timer_cannot_exceed_process_timer() -> None:
    rows = panel()
    rows[0]['observations'][0]['coordinator_seconds'] = 10
    with pytest.raises(ValueError, match='exceeds'):
        bench.summarize(rows)
