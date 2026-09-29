"""Cheap parser/failure tests only; native and model execution are opt-in."""
from pathlib import Path
import struct

import pytest

from native.bend_engine.async_probe import verify_model as gate


def output(asynchronous: bool) -> str:
    lines = []
    for step in range(1, 7):
        if asynchronous and step == 3:
            lines.append('cancelled 3')
        else:
            lines.append('result ' + str(step) + ' 3f800000' * gate.WIDTH)
    return '\n'.join([*lines, ('done 5 1' if asynchronous else 'done 6 0')]) + '\n'


@pytest.mark.parametrize('asynchronous', [False, True])
def test_valid_model_output(asynchronous: bool) -> None:
    parsed = gate.parse_output(output(asynchronous), asynchronous=asynchronous)
    assert list(parsed) == ([1, 2, 4, 5, 6] if asynchronous else [1, 2, 3, 4, 5, 6])
    assert all(row == struct.pack('<f', 1.0) * gate.WIDTH for row in parsed.values())


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('change', ['missing', 'extra', 'order', 'width', 'hex', 'summary', 'duplicate'])
def test_malformed_publication_rejected(asynchronous: bool, change: str) -> None:
    lines = output(asynchronous).splitlines()
    if change == 'missing':
        lines.pop(0)
    elif change == 'extra':
        lines.insert(1, lines[0])
    elif change == 'order':
        lines[0], lines[1] = lines[1], lines[0]
    elif change == 'width':
        lines[0] += ' 3f800000'
    elif change == 'hex':
        lines[0] = lines[0].replace('3f800000', 'xxxxxxxx', 1)
    elif change == 'summary':
        lines[-1] = 'done 6 1'
    else:
        lines[1] = lines[0]
    with pytest.raises(AssertionError):
        gate.parse_output('\n'.join(lines), asynchronous=asynchronous)


@pytest.mark.parametrize('replacement', ['result 3' + ' 3f800000' * gate.WIDTH,
                                         'cancelled 4', 'cancelled 3 extra', 'failed 3'])
def test_cancelled_model_cannot_publish_or_change_disposition(replacement: str) -> None:
    with pytest.raises(AssertionError):
        gate.parse_output(output(True).replace('cancelled 3', replacement), asynchronous=True)


def test_expected_audit() -> None:
    gate.parse_audit(gate.AUDIT + '\n')


@pytest.mark.parametrize('text', ['', gate.AUDIT + '\nextra', gate.AUDIT + '\n' + gate.AUDIT,
                                  gate.AUDIT.replace('calls=6', 'calls=5'),
                                  gate.AUDIT.replace('input_changes=0', 'input_changes=1'),
                                  gate.AUDIT.replace('output_changes=0', 'output_changes=1'),
                                  gate.AUDIT.replace('allocations=1', 'allocations=6')])
def test_bad_or_missing_model_audit(text: str) -> None:
    with pytest.raises(AssertionError):
        gate.parse_audit(text)


def trace_bytes(channels: int) -> bytes:
    return b''.join(struct.pack('<4I', 0x44464c31, step, channels * 64, gate.WIDTH)
                    + gate.expected_input(step, channels) + struct.pack('<f', 1.0) * gate.WIDTH
                    for step in range(1, 7))


@pytest.mark.parametrize('channels', [146, 175])
def test_complete_physical_trace_includes_cancelled_forward(tmp_path: Path, channels: int) -> None:
    trace = tmp_path / 'physical.bin'
    trace.write_bytes(trace_bytes(channels))
    records = gate.read_trace(trace, channels)
    assert len(records) == 6
    assert records[0] == records[4]
    assert records[2][0] != records[0][0]
    assert struct.unpack_from('<f', records[0][0])[0] == -115 / 128


@pytest.mark.parametrize('change', ['truncated_header', 'truncated_output', 'extra', 'magic',
                                    'sequence', 'input_shape', 'output_shape', 'missing_cancelled'])
def test_bad_physical_trace_rejected(tmp_path: Path, change: str) -> None:
    channels = 146
    data = trace_bytes(channels)
    length = 16 + (channels * 64 + gate.WIDTH) * 4
    if change == 'truncated_header':
        data = data[:10]
    elif change == 'truncated_output':
        data = data[:-1]
    elif change == 'extra':
        data += b'x'
    elif change == 'missing_cancelled':
        data = data[:length * 2] + data[length * 3:]
    else:
        offset = {'magic': 0, 'sequence': 4, 'input_shape': 8, 'output_shape': 12}[change]
        data = data[:offset] + struct.pack('<I', 0) + data[offset + 4:]
    trace = tmp_path / 'physical.bin'
    trace.write_bytes(data)
    with pytest.raises(AssertionError):
        gate.read_trace(trace, channels)
