"""Cheap parser/failure controls, not model export or native compilation in pytest."""
from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from native.bend_engine.batch_backend import verify


def good_output(batch: int = 4) -> str:
    lines = []
    for step, rows in enumerate(verify.row_counts(batch), 1):
        lines += [f'batch {step} {rows} {batch} {rows * verify.LOGITS}',
                  'values ' + ' '.join(['1065353216'] * (rows * verify.LOGITS)),
                  'tail 2143289344']
    return '\n'.join(lines) + '\n'


@pytest.mark.parametrize('batch', verify.BATCHES)
def test_all_batch_shapes(batch: int) -> None:
    outputs = verify.decode_results(good_output(batch), batch)
    assert [output.shape for output in outputs] == [(rows, verify.LOGITS) for rows in verify.row_counts(batch)]
    for output in outputs:
        np.testing.assert_array_equal(output, 1)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'physical', 'real', 'count', 'nonfinite',
                                     'tail', 'negative', 'large', 'nondecimal', 'empty', 'label'])
def test_reject_malformed_results(mutation: str) -> None:
    lines = good_output().splitlines()
    if mutation == 'missing':
        lines.pop()
    elif mutation == 'extra':
        lines.append('batch 6 1 4 1861')
    elif mutation in ('physical', 'real', 'count'):
        parts = lines[0].split()
        index = {'physical': 3, 'real': 2, 'count': 4}[mutation]
        parts[index] = '2'
        lines[0] = ' '.join(parts)
    elif mutation == 'tail':
        lines[2] = 'tail 0'
    else:
        wrong = {'nonfinite': '2143289344', 'negative': '-1', 'large': '4294967296',
                 'nondecimal': 'one', 'empty': '', 'label': '1065353216'}[mutation]
        lines[1] = lines[1].replace('1065353216', wrong, 1)
        if mutation == 'label':
            lines[1] = lines[1].replace('values ', 'wrong ', 1)
    expected_error = {'missing': 'five complete batch records', 'extra': 'five complete batch records',
                      'physical': 'shape/count', 'real': 'shape/count', 'count': 'shape/count',
                      'nonfinite': 'nonfinite', 'tail': 'nonfinite', 'negative': 'invalid output words',
                      'large': 'out-of-range', 'nondecimal': 'invalid output words',
                      'empty': 'invalid output words', 'label': 'missing output/tail'}[mutation]
    with pytest.raises(ValueError, match=expected_error):
        verify.decode_results('\n'.join(lines), 4)


@pytest.mark.parametrize('batch', [0, 3, 32, True])
def test_invalid_fixed_batch(batch: int) -> None:
    with pytest.raises(ValueError, match='fixed batch'):
        verify.row_counts(batch)


@pytest.mark.parametrize('channels', [146, 175])
def test_actual_padded_trace_has_separate_shapes(channels: int) -> None:
    physical = np.zeros((4, channels, 8, 8), dtype='<f4')
    physical[:1] = verify.expected_input(2, 1, channels)
    output = np.ones((1, verify.LOGITS), dtype='<f4')
    record = struct.pack('<6I', 0x44464232, 2, 4, 1, channels, verify.LOGITS) + physical.tobytes() + output.tobytes()
    a, b = verify.trace_record(io.BytesIO(record), 2, 4, 1, channels)
    np.testing.assert_array_equal(a, physical)
    np.testing.assert_array_equal(b, output)
    for broken in [record[:12], record[:-1], bytes(24) + record[24:]]:
        with pytest.raises(ValueError, match="trace"):
            verify.trace_record(io.BytesIO(broken), 2, 4, 1, channels)


def test_full_then_partial_repeats_identical_first_row() -> None:
    np.testing.assert_array_equal(verify.expected_input(1, 4, 175)[:1], verify.expected_input(5, 1, 175))
