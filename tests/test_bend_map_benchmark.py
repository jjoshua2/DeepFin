"""Benchmark admission and dictionary model; ordinary pytest does not time native maps."""
from dataclasses import replace

import pytest

from native.bend_engine.u64_map_probe import benchmark as b


def small() -> b.Workload:
    return b.Workload('small', 2, ((0, 0), (1 << 63, b.MASK)),
        ((1, 7, 0), (1, 0, 3), (0, 0, 0), (1, 0, 0),
         (2, 1 << 63, 0), (0, 1 << 63, 0), (1, 1 << 63, b.MASK)))


def output(rounds: int = 3, ms: int = 100) -> str:
    tokens, table = b.model(small())
    lines = [f'sample {rounds} {ms} {b.checksum(tokens, rounds)} {len(table)}']
    lines += [f'entry {k >> 32} {k & b.MASK} {v}' for k, v in table.items()]
    return '\n'.join([*lines, 'end']) + '\n'


def test_dictionary_tags_cover_full_update_delete_zero_and_miss() -> None:
    tokens, final = b.model(small())
    assert tokens == [3, 2, 25, 23, (7 * b.MASK + 6) & b.MASK, 5, 1]
    assert final == dict(small().initial)


@pytest.mark.parametrize('rounds', [0, 1, 2, 3, 7, 127, 1024])
def test_affine_checksum_matches_direct_operation_fold(rounds: int) -> None:
    tokens, _ = b.model(small())
    expected = 0
    for _ in range(rounds):
        for token in tokens:
            expected = (expected * b.MULT + token) & b.MASK
    assert b.checksum(tokens, rounds) == expected
    assert b.parse(output(rounds), small(), rounds) == 100


@pytest.mark.parametrize('edit', ['missing', 'extra', 'checksum', 'rounds', 'size', 'negative_time',
                                   'nan_time', 'high_half', 'value', 'duplicate', 'range', 'end'])
def test_corrupt_native_result_rejected(edit: str) -> None:
    text = output()
    if edit == 'missing':
        text = '\n'.join(text.splitlines()[1:]) + '\n'
    elif edit == 'extra':
        text += 'end\n'
    elif edit == 'checksum':
        fields = text.splitlines()
        header = fields[0].split()
        header[3] = str(int(header[3]) ^ 1)
        text = '\n'.join([' '.join(header), *fields[1:]]) + '\n'
    else:
        old, new = {'rounds': ('sample 3', 'sample 4'), 'size': (' 2\nentry', ' 1\nentry'),
                    'negative_time': (' 100 ', ' -1 '), 'nan_time': (' 100 ', ' NaN '),
                    'high_half': ('2147483648', '0'), 'value': ('entry 0 0 0', 'entry 0 0 9'),
                    'duplicate': ('entry 2147483648 0 4294967295', 'entry 0 0 0'),
                    'range': ('4294967295', '4294967296'), 'end': ('end\n', 'done\n')}[edit]
        assert old in text
        text = text.replace(old, new, 1)
    with pytest.raises(ValueError, match=r'map|oracle'):
        b.parse(text, small(), 3)


def test_non_neutral_cycle_not_exponentiated() -> None:
    changed = replace(small(), ops=((1, 0, 99),))
    with pytest.raises(ValueError, match='preserve'):
        b.parse(output(), changed, 3)


def panel() -> list[dict[str, object]]:
    return [{'case': 'small', 'arm': arm, 'pair': pair, 'order': (pair + i) % 2,
             'phase': 'measurement', 'rounds': 1024, 'input_sha256': 'a' * 64,
             'milliseconds': 100 if arm == 'hash' else 200}
            for pair in range(b.PAIRS) for i, arm in enumerate(b.ARMS)]


def test_balanced_measurement_panel() -> None:
    summary = b.summarize(panel(), ['small'])['small']
    assert summary['reliable'] is True
    assert summary['median_scan_over_hash'] == 2
    assert summary['decision'] == 'hash_over_5pct_faster'


@pytest.mark.parametrize('fault', ['drop', 'duplicate', 'work', 'rounds', 'order', 'float_time', 'negative_time'])
def test_incoherent_panel_rejected(fault: str) -> None:
    rows = panel()
    if fault == 'drop':
        rows.pop()
    elif fault == 'duplicate':
        rows[-1] = dict(rows[0])
    else:
        key, value = {'work': ('input_sha256', 'b' * 64), 'rounds': ('rounds', 32),
                      'order': ('order', 1), 'float_time': ('milliseconds', 100.0),
                      'negative_time': ('milliseconds', -1)}[fault]
        rows[0][key] = value
    with pytest.raises(ValueError, match=r'panel|work|order|duration'):
        b.summarize(rows, ['small'])


@pytest.mark.parametrize(('ms', 'decision'), [(40, 'below_measurement_floor'),
                                             (100, 'inconclusive_at_5pct'),
                                             (75, 'scan_over_5pct_faster')])
def test_negative_results_and_short_samples_retained(ms: int, decision: str) -> None:
    rows = panel()
    for row in rows:
        if row['arm'] == 'scan':
            row['milliseconds'] = ms
    summary = b.summarize(rows, ['small'])['small']
    assert summary['decision'] == decision
    if ms < b.MIN_MS:
        assert summary['median_scan_over_hash'] is None
        assert summary['paired_ratios'] is None


def test_workloads_are_full_width_neutral_and_include_cluster() -> None:
    cases = b.workloads()
    assert len(cases) == len({c.name for c in cases}) == 15
    for case in cases:
        assert len(case.ops) == 256
        assert b.model(case)[1] == dict(case.initial)
        assert len(b.encode(case, 32)) < 100000
        assert len(case.initial) == 1 << (case.bits - 1)
        if case.name.startswith('cluster'):
            assert all(b.bucket(k, 127) == 126 for k, _ in case.initial)
        if not case.name.startswith('random'):
            assert all(k & b.MASK == 0 for k, _ in case.initial)
