"""CPU-only failure/parser controls; these do NOT certify CUDA execution."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from native.bend_engine.batch_backend import verify_cuda as gate


def manifest() -> dict[str, object]:
    return {'format': 'deepfin-tuple-policy-wdl-checkpoint-v3', 'device': 'cuda',
            'dtype': 'bfloat16', 'device_index': 0, 'checkpoint_sha256': 'a' * 64, 'weights_key': 'model'}


@pytest.mark.parametrize('index', [0, 1, 127])
def test_explicit_index(index: int) -> None:
    assert gate.cuda_index({**manifest(), 'device_index': index}) == index


@pytest.mark.parametrize(('key', 'value'), [('device', 'cpu'), ('dtype', 'float32'), ('dtype', 'float16'),
    ('device_index', -1), ('device_index', 128), ('device_index', True), ('device_index', '0'),
    ('checkpoint_sha256', 'x'), ('weights_key', 'missing'), ('format', 'old')])
def test_reject_target_inference(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=r'unsupported|invalid|missing|CUDA gate'):
        gate.cuda_index({**manifest(), key: value})


@pytest.mark.parametrize(('atol', 'rtol'), [(-1.0, 0.0), (0.0, -1.0), (float('nan'), 0.0),
                                          (0.0, float('inf')), (float('inf'), 0.0)])
def test_tolerances_fail_closed(atol: float, rtol: float) -> None:
    with pytest.raises(ValueError, match='predeclared tolerances'):
        gate.tolerance_pair(atol, rtol)


def test_zero_tolerance_is_allowed() -> None:
    assert gate.tolerance_pair(0.0, 0.0) == (0.0, 0.0)


def trace() -> bytes:
    return (struct.pack('<8I', 0x44464333, 1, 4, 1, 146, 1861, 1, 16)
            + np.full(4 * 146 * 64, 0x3f80, dtype='<u2').tobytes()
            + np.arange(1861, dtype='<f4').tobytes())


def test_bf16_trace_is_not_half_or_float32() -> None:
    x, y = gate.trace_record(BytesIO(trace()), 1, 4, 1, 146, 1)
    assert x.shape == (4, 146, 8, 8)
    assert x.dtype == np.uint16
    assert np.all(x == 0x3f80)
    np.testing.assert_array_equal(y[0], np.arange(1861, dtype=np.float32))


@pytest.mark.parametrize('field', range(8))
def test_corrupted_header_rejected(field: int) -> None:
    raw = bytearray(trace())
    raw[field * 4] ^= 1
    with pytest.raises(ValueError, match='header'):
        gate.trace_record(BytesIO(raw), 1, 4, 1, 146, 1)


@pytest.mark.parametrize('length', [0, 4, 31, 32, 100, -1])
def test_truncated_trace_rejected(length: int) -> None:
    with pytest.raises(ValueError, match='trace'):
        gate.trace_record(BytesIO(trace()[:length]), 1, 4, 1, 146, 1)


@pytest.mark.parametrize('tracing', [False, True])
def test_strict_native_audit(tracing: bool) -> None:
    text = ('native-buffer-audit calls=5 input_changes=0 output_changes=0 input_tensor_allocations=1\n'
            f'native-cuda-audit device_index=1 stream_nondefault=1 pinned_buffers={3 if tracing else 2} completion_events=1\n')
    gate.validate_audit(text, 1, tracing=tracing)
    for bad in ('', text + text, text + 'unexpected\n', text.replace('stream_nondefault=1', 'stream_nondefault=0'),
                text.replace('input_changes=0', 'input_changes=1'), text.replace('device_index=1', 'device_index=0')):
        with pytest.raises(ValueError, match='audit mismatch'):
            gate.validate_audit(bad, 1, tracing=tracing)


def test_no_gpu_is_failure_not_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='no CPU fallback'):
        gate.require_device(0)


def test_unavailable_device_index_is_not_remapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    with pytest.raises(RuntimeError, match='unavailable'):
        gate.require_device(1)


def test_failed_execution_writes_failure_not_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import sys
    paths = {name: tmp_path / name for name in ('binary', 'package', 'checkpoint', 'report')}
    for name in ('binary', 'package', 'checkpoint'):
        paths[name].write_text('protected source')
    argv = ['verify_cuda']
    for name, path in paths.items():
        argv += ['--' + name, str(path)]
    argv += ['--atol', '0.01', '--rtol', '0.01']
    monkeypatch.setattr(sys, 'argv', argv)

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError('requested CUDA device is unavailable; no CPU fallback')

    monkeypatch.setattr(gate, 'verify_model', fail)
    with pytest.raises(SystemExit, match='2'):
        gate.main()
    report = json.loads(paths['report'].read_text())
    assert report['status'] == 'failed'
    assert report['cuda_model_qualified'] is False
    assert report['atol'] == 0.01
    assert report['rtol'] == 0.01
    for name in ('binary', 'package', 'checkpoint'):
        assert paths[name].read_text() == 'protected source'
    # A second invocation must not overwrite the recorded failed run either.
    with pytest.raises(FileExistsError):
        gate.main()


def test_report_cannot_replace_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    package = tmp_path / 'model.pt2'
    package.write_text('protected package')
    monkeypatch.setattr(sys, 'argv', ['verify_cuda', '--binary', str(tmp_path/'binary'),
        '--checkpoint', str(tmp_path/'checkpoint'), '--package', str(package),
        '--report', str(package), '--atol', '0.01', '--rtol', '0.01'])
    with pytest.raises(ValueError, match='protected input'):
        gate.main()
    assert package.read_text() == 'protected package'
