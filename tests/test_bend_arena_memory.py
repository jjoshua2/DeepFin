"""Arena footprint measurement admission; ordinary pytest does not run the screen."""
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from native.bend_engine.multi_root import benchmark_arena_memory as bench


def panel() -> list[dict[str, Any]]:
    return [{'roots': roots, 'simulations': sims, 'capacity': cap, 'sample': sample,
             'order': (ci - sample) % len(bench.CAPACITIES), 'physical_capacity': bench.physical_capacity(cap),
             'peak_rss_kib': 10000 + bench.physical_capacity(cap) * roots,
             'process_seconds': 0.5, 'coordinator_seconds': 0.1,
             'work_sha256': 'a' * 64, 'stdout_sha256': 'b' * 64}
            for roots in bench.ROOT_COUNTS for sims in bench.SIMULATIONS
            for ci, cap in enumerate(bench.CAPACITIES) for sample in range(bench.REPEATS)]


def test_complete_panel_uses_all_raw_rss_observations() -> None:
    rows = panel()
    selected = [r for r in rows if (r['roots'], r['simulations'], r['capacity']) == (1, 1, 4096)]
    for row, rss in zip(selected, (100, 200, 100000, 300, 400), strict=True):
        row['peak_rss_kib'] = rss
    summary = bench.summarize(rows)
    assert len(summary) == 20
    assert summary[0]['samples'] == 5
    assert summary[0]['peak_rss_kib_median'] == 300
    assert summary[0]['peak_rss_kib_min'] == 100
    assert summary[0]['peak_rss_kib_max'] == 100000


@pytest.mark.parametrize(('value', 'physical'), [(1, 4096), (4096, 4096), (4097, 8192),
                                                (8192, 8192), (8193, 16384), (65536, 65536)])
def test_physical_capacity(value: int, physical: int) -> None:
    assert bench.physical_capacity(value) == physical


@pytest.mark.parametrize('text', ['0 0', '-1 0', '10 1', 'True 0', '1.2 0', 'NaN 0',
                                  '100', '100 0 extra', '', '١٠٠ 0', f'{(1 << 40) + 1} 0'])
def test_bad_rss_report_rejected(text: str) -> None:
    with pytest.raises(ValueError, match=r'RSS|child'):
        bench.rss_result(text)


def test_independent_child_maxima_are_not_differenced() -> None:
    assert [bench.rss_result(s) for s in ('10000 0\n', '5000 0\n', '8000 0\n')] == [10000, 5000, 8000]


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'unknown', 'order', 'physical', 'work',
                                    'float_identity', 'bool_identity', 'digest', 'rss_bool', 'rss_zero',
                                    'negative_time', 'nan_time', 'inf_time', 'bool_time', 'zero_time', 'internal_time'])
def test_bad_panel_rejected(fault: str) -> None:
    rows = panel()
    if fault == 'missing':
        rows.pop()
    elif fault == 'duplicate':
        rows[-1] = deepcopy(rows[0])
    else:
        key, value = {'unknown': ('capacity', 32768), 'order': ('order', 4),
                      'physical': ('physical_capacity', 4097), 'work': ('work_sha256', 'c' * 64),
                      'float_identity': ('sample', 0.0), 'bool_identity': ('sample', False),
                      'digest': ('stdout_sha256', 'not-a-hash'), 'rss_bool': ('peak_rss_kib', True),
                      'rss_zero': ('peak_rss_kib', 0), 'negative_time': ('process_seconds', -1),
                      'nan_time': ('process_seconds', float('nan')), 'inf_time': ('process_seconds', float('inf')),
                      'bool_time': ('process_seconds', True), 'zero_time': ('process_seconds', 0),
                      'internal_time': ('coordinator_seconds', 2)}[fault]
        rows[0][key] = value
    with pytest.raises(ValueError, match=r'arena|work'):
        bench.summarize(rows)


def search_result() -> dict[str, Any]:
    return {'roots': {1: {'completed_simulations': 64, 'stop_code': 0, 'used_nodes': 100}},
            'nodes': {1: {0: [7]}}, 'events': [('reply', [1])],
            'work': {'completed_simulations': 64, 'wall_seconds': 1.0, 'accepted_neural_rows': 64}}


def test_only_named_timing_and_diagnostic_fields_omitted() -> None:
    parsed = search_result()
    checked = bench.checked_work(parsed, 1, 64, diagnostics=False)
    assert 'nodes' not in checked
    assert 'events' not in checked
    assert checked['work'] == {'completed_simulations': 64, 'accepted_neural_rows': 64}
    parsed['work']['new_unrecognized_accounting'] = 10
    assert bench.checked_work(parsed, 1, 64, diagnostics=False) != checked
    assert bench.checked_work(parsed, 1, 64, diagnostics=True)['nodes'] == parsed['nodes']


@pytest.mark.parametrize('fault', ['short', 'stopped', 'total', 'root'])
def test_incomplete_search_is_not_a_memory_win(fault: str) -> None:
    parsed = search_result()
    if fault == 'short':
        parsed['roots'][1]['completed_simulations'] = 63
    elif fault == 'stopped':
        parsed['roots'][1]['stop_code'] = 1
    elif fault == 'total':
        parsed['work']['completed_simulations'] = 1
    else:
        parsed['roots'][2] = parsed['roots'].pop(1)
    with pytest.raises(ValueError, match=r'work|identit'):
        bench.checked_work(parsed, 1, 64, diagnostics=False)


def test_execute_uses_fresh_per_child_rss_and_disables_inherited_modes(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}
    monkeypatch.setenv('DEEPFIN_COHORT_ASYNC', '1')
    monkeypatch.setenv('DEEPFIN_COHORT_ARENA_NODES', '65536')
    monkeypatch.setenv('DEEPFIN_BEND_MODEL_TRACE', 'unwanted.bin')
    monkeypatch.setenv('DEEPFIN_MULTI_TEST_FAULT', 'nan')

    class Child:
        returncode = 0

        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            seen.update({'argv': argv, **kwargs})
            Path(argv[argv.index('-o') + 1]).write_text('15000 0\n')

        def __enter__(self) -> 'Child':
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def communicate(self, timeout: int) -> tuple[str, str]:
            assert timeout == 60
            return 'checked-output', ''

    def parse(text: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        assert text == 'checked-output'
        assert args == (1, 4, 64, 0, False)
        assert kwargs == {'arena_nodes': 8192}
        parsed = search_result()
        parsed['work']['wall_seconds'] = 0.0
        return parsed

    monkeypatch.setattr(bench.subprocess, 'Popen', Child)
    monkeypatch.setattr(bench.verify, 'parse', parse)
    _, observation = bench.execute(Path('runner'), Path('/usr/bin/time'), tmp_path / 'one', 1, 64, 8192,
                                   diagnostics=False)
    assert observation['peak_rss_kib'] == 15000
    assert seen['argv'][1:3] == ['-f', '%M %x']
    assert seen['env']['DEEPFIN_COHORT_ASYNC'] == '0'
    assert seen['env']['DEEPFIN_COHORT_ARENA_NODES'] == '8192'
    assert 'DEEPFIN_BEND_MODEL_TRACE' not in seen['env']
    assert 'DEEPFIN_MULTI_TEST_FAULT' not in seen['env']
    with pytest.raises(FileExistsError):
        bench.execute(Path('runner'), Path('/usr/bin/time'), tmp_path / 'one', 1, 64, 8192, diagnostics=False)
