"""Admission checks for equal-work timing; these tests do not run a timing panel."""
from dataclasses import replace
import subprocess

import pytest

from native.bend_engine.u64_map_probe import move_cache_benchmark as b


def small() -> b.Case:
    return b.Case('small', 1, (b.legal.START,), (b.legal.START,))


def rendered(arm: str = 'cached', rounds: int = 3, diag: bool = True) -> str:
    order = [[1487, 1999]]
    head = f'configuration {b.ARMS.index(arm)} 1 {rounds} 0 {int(diag)}'
    lines = [head]
    if diag:
        lines.extend(f'trace {i} 1487 1999' for i in range(rounds))
    routes = ' '.join(map(str, b.route_counts(small(), arm, rounds)))
    lines.append(f'sample 100 {rounds} {2 * rounds} {b.repeated_checksum(order, rounds)} {routes}')
    return '\n'.join(lines) + '\n'


def test_all_hit_ratios_keep_the_measured_boards_identical() -> None:
    cases = b.cases()
    assert len(cases) == 8
    assert len({case.positions for case in cases[:5]}) == 1
    for case, hits in zip(cases[:5], (0, 2, 4, 6, 8), strict=True):
        assert len(case.prime) == 8
        assert b.route_counts(case, 'cached', 3) == (hits * 3, 0, (8 - hits) * 3)
        assert b.route_counts(case, 'direct', 3) == (0, 0, 24)
    assert b.route_counts(cases[5], 'cached', 3) == (0, 24, 0)
    assert b.route_counts(cases[-1], 'cached', 3) == (3, 0, 0)


def test_unprimed_warm_cache_fills_once_and_then_hits() -> None:
    case = replace(b.cases()[4], prime=())
    assert b.route_counts(case, 'cached', 3) == (16, 8, 0)
    assert b.route_counts(case, 'cached', 0) == (0, 0, 0)


@pytest.mark.parametrize('rounds', [0, 1, 2, 3, 127])
def test_affine_repetition_matches_direct_fold(rounds: int) -> None:
    order = [[1487, 1999], [], [131071]]
    result = 0
    for _ in range(rounds):
        for moves in order:
            for token in [*(m + 1 for m in moves), 131072 + len(moves)]:
                result = (result * b.MULT + token) & b.MASK
    assert b.repeated_checksum(order, rounds) == result


@pytest.mark.parametrize('rounds', [-1, True, 1.0, b.MAX_ROUNDS + 1])
def test_invalid_repetition_never_reaches_an_unbounded_loop(rounds: int) -> None:
    with pytest.raises(ValueError, match='cycle count'):
        b.repeated_checksum([[]], rounds)
    with pytest.raises(ValueError, match='cycle count'):
        b.encode(small(), 'cached', rounds)


def test_reference_requires_complete_legal_set_and_exact_order() -> None:
    order = [[1487, 1999]]
    parsed = b.parse(rendered(), small(), 'cached', 3, True, order, [{1487, 1999}])
    assert (parsed['requests'], parsed['moves'], parsed['hits']) == (3, 6, 3)
    with pytest.raises(ValueError, match='ordering'):
        b.parse(rendered().replace('1487 1999', '1999 1487', 1), small(), 'cached', 3, True, order, [{1487, 1999}])
    with pytest.raises(ValueError, match='independent legal oracle'):
        b.parse(rendered().replace('1487 1999', '1487 1487', 1), small(), 'cached', 3, True, order, [{1487, 1999}])


@pytest.mark.parametrize('fault', ['requests', 'moves', 'checksum', 'hits', 'fills', 'bypasses', 'configuration', 'extra', 'newline'])
def test_less_or_different_work_is_not_a_speedup(fault: str) -> None:
    text = rendered()
    lines = text.splitlines()
    positions = {'requests': 2, 'moves': 3, 'checksum': 4, 'hits': 5, 'fills': 6, 'bypasses': 7}
    if fault in positions:
        fields = lines[-1].split()
        fields[positions[fault]] = str(int(fields[positions[fault]]) + 1)
        lines[-1] = ' '.join(fields)
    elif fault == 'configuration':
        lines[0] = 'configuration 1 16 3 0 1'
    elif fault == 'extra':
        lines.append(lines[-1])
    text = '\n'.join(lines) + ('' if fault == 'newline' else '\n')
    with pytest.raises(ValueError, match='benchmark'):
        b.parse(text, small(), 'cached', 3, True, [[1487, 1999]], [{1487, 1999}])


@pytest.mark.parametrize('case', [replace(small(), bits=0), replace(small(), bits=True),
                                  replace(small(), positions=()), replace(small(), cold=True),
                                  replace(small(), prime=(b.legal.START + ';startpos',))])
def test_bad_workload_admission(case: b.Case) -> None:
    with pytest.raises(ValueError, match='benchmark'):
        b.encode(case, 'cached', 1)


def panel() -> list[dict[str, object]]:
    return [{'case': 'small', 'arm': arm, 'rounds': 128, 'phase': 'measurement', 'pair': pair,
             'order': (pair + i) % 2, 'milliseconds': 200 if arm == 'direct' else 100,
             'requests': 128, 'moves': 256, 'checksum': 7, 'workload_sha256': 'a' * 64,
             'peak_rss_kib': 10000}
            for pair in range(b.PAIRS) for i, arm in enumerate(b.ARMS)]


def test_balanced_panel_reports_all_pairwise_ratios() -> None:
    summary = b.summarize(panel(), ['small'])['small']
    assert summary['decision'] == 'cache_faster'
    assert summary['ratios'] == [2.0] * 6
    assert summary['median_direct_over_cached'] == 2.0


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'work', 'order', 'type', 'negative'])
def test_incoherent_measurements_rejected(fault: str) -> None:
    rows = panel()
    if fault == 'missing':
        rows.pop()
    elif fault == 'duplicate':
        rows[-1] = dict(rows[0])
    else:
        key, value = {'work': ('requests', 1), 'order': ('order', 1),
                      'type': ('milliseconds', 100.0), 'negative': ('milliseconds', -1)}[fault]
        rows[0][key] = value
    with pytest.raises(ValueError, match=r'benchmark|work'):
        b.summarize(rows, ['small'])


@pytest.mark.parametrize(('ms', 'decision'), [(300, 'cache_slower'), (200, 'inconclusive_at_5pct'),
                                             (20, 'below_measurement_floor')])
def test_regressions_and_short_samples_are_retained(ms: int, decision: str) -> None:
    rows = panel()
    for row in rows:
        if row['arm'] == 'cached':
            row['milliseconds'] = ms
    result = b.summarize(rows, ['small'])['small']
    assert result['decision'] == decision
    if ms < b.MIN_MS:
        assert result['ratios'] is None


@pytest.mark.parametrize('text', ['0 0', '1 1', '-1 0', '100', '100 0 extra', '١٠٠ 0'])
def test_invalid_child_memory_record(text: str) -> None:
    with pytest.raises(ValueError, match=r'RSS|status'):
        b.rss(text)


def test_child_rss_values_are_independent_not_cumulative_differences() -> None:
    assert [b.rss(text) for text in ('10000 0', '5000 0', '8000 0')] == [10000, 5000, 8000]


@pytest.mark.parametrize(('code', 'stdout', 'stderr'), [
    (0, '', 'invalid benchmark bounds\n'), (-11, '', 'invalid benchmark bounds\n'),
    (2, 'unexpected', 'invalid benchmark bounds\n'), (2, '', 'other failure\n'),
    (2, '', ''), (2, '', 'invalid benchmark bounds\nextra\n'),
])
def test_native_admission_requires_exact_rejection(code: int, stdout: str, stderr: str) -> None:
    with pytest.raises(ValueError, match='not rejected as intended'):
        b.check_invalid(subprocess.CompletedProcess(['driver'], code, stdout, stderr))


def test_exact_native_admission_rejection() -> None:
    b.check_invalid(subprocess.CompletedProcess(['driver'], 2, '', 'invalid benchmark bounds\n'))


def test_all_seven_native_invalid_inputs_are_retained() -> None:
    bad = b.invalid_inputs()
    assert len(bad) == len(set(bad)) == 7
    assert tuple(s.partition('|')[0] for s in bad) == (
        '2 4 1 0 0', '1 0 1 0 0', '1 17 1 0 0', '1 4 65537 0 0',
        '1 4 4 0 1', '1 4 1 1 0', '1 4 1 0 0')
    assert len(bad[-1].split('|')[2].split(';')) == 129
