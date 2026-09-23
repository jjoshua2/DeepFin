"""Fail-closed native-output contracts, independent of a local Bend installation."""
from collections import deque

import pytest

from native.bend_engine.collections_probe.owning_benchmark import (
    ARMS, MASK, SIZES, checksum, expected_ring_trace, expected_roots,
    parse_sample, timing_summary, verify_ring_trace,
)


def sample_text(arm: int = 0, size: int = 3, steps: int = 7, ms: int = 100) -> str:
    return '\n'.join([f'sample {arm} {size} {steps} {ms} {checksum(size, steps)}',
                      *expected_roots(size, steps), 'length 0']) + '\n'


@pytest.mark.parametrize('size', [1, 3, 16])
@pytest.mark.parametrize('steps', [0, 1, 7, 100])
def test_round_robin_oracle(size: int, steps: int) -> None:
    q = deque((i, 0) for i in range(size))
    total = 0
    for _ in range(steps):
        root, visits = q.popleft()
        total = ((total * 1664525 + 1013904223) & MASK) ^ root ^ visits
        q.append((root, visits + 1))
    assert checksum(size, steps) == total
    rows = [line.split() for line in expected_roots(size, steps)]
    assert [(int(row[1]), int(row[4])) for row in rows] == list(q)
    assert parse_sample(sample_text(0, size, steps), 0, size, steps) == 100


@pytest.mark.parametrize(('old', 'new'), [
    ('sample 0', 'sample 1'), ('sample 0 3', 'sample 0 4'),
    (' 7 100 ', ' 8 100 '), (' 100 ', ' -1 '), (' 100 ', ' nan '),
    ('root 1 1', 'root 2 1'), ('4096 2 4096', '4096 3 4096'),
    (' 2 1 0 20 ', ' 9 1 0 20 '), ('2779096484', '0'), ('length 0', 'length 1'),
])
def test_corrupt_owning_sample_rejected(old: str, new: str) -> None:
    text = sample_text()
    assert old in text
    with pytest.raises(ValueError, match='owning sample'):
        parse_sample(text.replace(old, new, 1), 0, 3, 7)


@pytest.mark.parametrize('edit', ['drop', 'extra', 'reorder', 'checksum'])
def test_bad_sample_structure(edit: str) -> None:
    lines = sample_text().splitlines()
    if edit == 'drop':
        lines.pop(1)
    elif edit == 'extra':
        lines.append(lines[-1])
    elif edit == 'reorder':
        lines[1], lines[2] = lines[2], lines[1]
    else:
        fields = lines[0].split()
        fields[-1] = str(int(fields[-1]) ^ 1)
        lines[0] = ' '.join(fields)
    with pytest.raises(ValueError, match='owning sample'):
        parse_sample('\n'.join(lines) + '\n', 0, 3, 7)


def test_ring_oracle_boundaries() -> None:
    text = expected_ring_trace()
    verify_ring_trace(text)
    assert text.startswith('capacity 0\nempty\nsize 0\nempty\nsize 0\nreject ')
    assert text.endswith('invalid 4097\ninvalid 4294967295\n')
    with pytest.raises(ValueError, match='bounded ring trace'):
        verify_ring_trace(text.replace('reject', 'accept', 1))


def timing_rows() -> list[dict[str, int | str]]:
    return [{'phase': 'measurement', 'size': size, 'arm': arm,
             'milliseconds': ms, 'steps': 65536, 'index': index}
            for size in SIZES for index in range(6)
            for arm, ms in zip(ARMS, (200, 100, 50), strict=True)]


def test_balanced_reliable_timings() -> None:
    summary = timing_summary(timing_rows())
    assert summary['16'] == {'reliable': True, 'median_ms': {'list': 200, 'fifo': 100, 'ring': 50},
                             'list_over_fifo': 2.0, 'fifo_over_ring': 2.0}


@pytest.mark.parametrize('fault', ['zero', 'short', 'missing', 'calibration'])
def test_unreliable_timings_have_no_ratio(fault: str) -> None:
    rows = timing_rows()
    if fault == 'missing':
        rows.pop(0)
    elif fault == 'calibration':
        rows[0]['phase'] = 'calibration'
    else:
        rows[0]['milliseconds'] = 0 if fault == 'zero' else 49
    summary = timing_summary(rows)
    assert summary['1'] == {'reliable': False, 'median_ms': {'list': 200, 'fifo': 100, 'ring': 50},
                            'list_over_fifo': None, 'fifo_over_ring': None}
