"""Opt-in native batch-boundary checks; no search, acceptance or speed claim.

The model gate uses an explicitly supplied checkpoint and independently evaluated
singleton rows. Its trace contains the actual padded CPU tensor passed to AOTI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import BinaryIO

import numpy as np

BATCHES = (1, 2, 4, 8, 16)
LOGITS = 1861
ATOL, RTOL = 2e-6, 2e-5


def row_counts(batch: int) -> list[int]:
    if type(batch) is not int or batch not in BATCHES:
        raise ValueError('unsupported fixed batch')
    return [batch, 1, max(1, batch - 1), batch, 1]


def words(text: str, count: int) -> np.ndarray:
    values = text.split()
    if len(values) != count or any(not s.isascii() or not s.isdecimal() for s in values):
        raise ValueError('invalid output words/count')
    integers = [int(s) for s in values]
    if any(n > 0xffffffff for n in integers):
        raise ValueError('out-of-range F32 bits')
    return np.asarray(integers, dtype=np.uint32).view(np.float32)


def decode_results(stdout: str, batch: int) -> list[np.ndarray]:
    lines = stdout.splitlines()
    if len(lines) != 15:
        raise ValueError('expected exactly five complete batch records')
    outputs = []
    for step, rows in enumerate(row_counts(batch), 1):
        metadata, values, tail = lines[(step - 1) * 3:step * 3]
        if metadata != f'batch {step} {rows} {batch} {rows * LOGITS}':
            raise ValueError('batch shape/count metadata mismatch')
        if not values.startswith('values ') or not tail.startswith('tail '):
            raise ValueError('missing output/tail record')
        result = words(values[len('values '):], rows * LOGITS).reshape(rows, LOGITS)
        if not np.isfinite(result).all() or not np.isnan(words(tail[len('tail '):], 1)[0]):
            raise ValueError('nonfinite logical output or overwritten tail')
        outputs.append(result)
    return outputs


def expected_input(step: int, rows: int, channels: int) -> np.ndarray:
    if channels not in (146, 175):
        raise ValueError('unsupported channels')
    effective_step = 1 if step == 5 else step
    return (((np.arange(rows * channels * 64, dtype=np.uint32) + effective_step * 7) % 31)
            .astype(np.float32) * np.float32(0.0625)).reshape(rows, channels, 8, 8)


def trace_record(stream: BinaryIO, step: int, batch: int, rows: int,
                 channels: int) -> tuple[np.ndarray, np.ndarray]:
    header = stream.read(24)
    expected = (0x44464232, step, batch, rows, channels, LOGITS)
    if len(header) != 24 or struct.unpack('<6I', header) != expected:
        raise ValueError('batch trace header mismatch')
    input_count = batch * channels * 64
    raw = stream.read(4 * (input_count + rows * LOGITS))
    if len(raw) != 4 * (input_count + rows * LOGITS):
        raise ValueError('truncated batch trace')
    data = np.frombuffer(raw, dtype='<f4').copy()
    return data[:input_count].reshape(batch, channels, 8, 8), data[input_count:].reshape(rows, LOGITS)


def clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith('DEEPFIN_BEND_')
            and k != 'DEEPFIN_BATCH_TEST_FAIL'}


def run(binary: Path, env: dict[str, str], mode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(binary.resolve()), '--threads', '1', '--', str(mode)],
                          env=env, capture_output=True, text=True, timeout=60, check=False)


def failure_controls(binary: Path, env: dict[str, str]) -> int:
    # No trace/audit during negative tests, so a rejected request cannot create
    # misleading successful rows or collide with an existing trace destination.
    env = {k: v for k, v in env.items() if k not in ('DEEPFIN_BEND_MODEL_TRACE', 'DEEPFIN_BEND_BUFFER_AUDIT')}
    for mode in (1, 2, 3, 4):
        result = run(binary, env, mode)
        if (result.returncode != 2 or result.stdout
                or result.stderr.strip() != 'native batch buffer/row contract failed'):
            raise ValueError(f'negative batch transport control failed: {mode}: {result}')
    return 4


def verify_transport(binary: Path, batch: int, channels: int) -> dict[str, object]:
    env = clean_env()
    result = run(binary, env)
    if result.returncode:
        raise ValueError(result.stderr)
    outputs = decode_results(result.stdout, batch)
    expected_log = []
    for step, (rows, output) in enumerate(zip(row_counts(batch), outputs, strict=True), 1):
        first = expected_input(step, rows, channels)[:, 0, 0, 0]
        np.testing.assert_array_equal(output, first[:, None] + np.arange(LOGITS, dtype=np.float32))
        expected_log.append(f'transport-call={step} real={rows} physical={batch}')
    if result.stderr.splitlines() != expected_log:
        raise ValueError('transport callback log mismatch')
    failures = failure_controls(binary, env)
    failed = run(binary, {**env, 'DEEPFIN_BATCH_TEST_FAIL': '1'})
    if failed.returncode != 2 or failed.stdout or failed.stderr.strip() != 'requested test backend failure':
        raise ValueError('backend failure did not propagate')
    return {'status': 'passed', 'scope': 'test backend only', 'batch': batch, 'channels': channels,
            'calls': 5, 'real_rows': sum(row_counts(batch)), 'physical_rows': 5 * batch,
            'failure_controls': failures + 1}


def verify_model(binary: Path, package: Path, checkpoint: Path) -> dict[str, object]:
    import torch
    from native.bend_engine.neural_probe.backend import package_manifest
    from native.bend_engine.neural_probe.checkpoint import load_checkpoint
    from chess_anti_engine.inference import _policy_output

    torch.set_num_threads(2)
    manifest, encoding = package_manifest(package)
    if manifest['device'] != 'cpu' or manifest['dtype'] != 'float32':
        raise ValueError('this gate qualifies only CPU F32; CUDA is a separate gate')
    batch = manifest['batch']
    if type(batch) is not int or batch not in BATCHES:
        raise ValueError('unsupported fixed model batch')
    loaded = load_checkpoint(checkpoint, weights_key=str(manifest['weights_key']))
    if loaded.identity['checkpoint_sha256'] != manifest['checkpoint_sha256'] or loaded.encoding != encoding:
        raise ValueError('checkpoint/encoding identity mismatch')
    env = {**clean_env(), 'DEEPFIN_BEND_MODEL_PACKAGE': str(package.resolve()),
           'DEEPFIN_BEND_BUFFER_AUDIT': '1'}
    maximum = 0.0
    with tempfile.TemporaryDirectory(prefix='deepfin-batch-') as tmp:
        trace = Path(tmp) / 'trace.bin'
        result = run(binary, {**env, 'DEEPFIN_BEND_MODEL_TRACE': str(trace)})
        if result.returncode:
            raise ValueError(result.stderr)
        outputs = decode_results(result.stdout, batch)
        if result.stderr.splitlines() != ['native-buffer-audit calls=5 input_changes=0 output_changes=0 input_tensor_allocations=1']:
            raise ValueError('model buffer reuse audit mismatch: ' + result.stderr)
        with trace.open('rb') as stream, torch.no_grad():
            for step, (rows, output) in enumerate(zip(row_counts(batch), outputs, strict=True), 1):
                physical, traced = trace_record(stream, step, batch, rows, encoding.channels)
                expected = expected_input(step, rows, encoding.channels)
                np.testing.assert_array_equal(physical[:rows].view(np.uint32), expected.view(np.uint32))
                # Exact +0 padding after full batches; stale/NaN padding must fail.
                np.testing.assert_array_equal(physical[rows:].view(np.uint32), 0)
                np.testing.assert_array_equal(output.view(np.uint32), traced.view(np.uint32))
                for row in range(rows):
                    reference = loaded.model(torch.from_numpy(expected[row:row + 1].copy()))
                    want = torch.cat((_policy_output(reference), reference['wdl']), dim=1).float().numpy()[0]
                    np.testing.assert_allclose(output[row], want, atol=ATOL, rtol=RTOL)
                    maximum = max(maximum, float(np.max(np.abs(output[row] - want))))
            if stream.read(1):
                raise ValueError('unexpected trailing native forward')
        np.testing.assert_allclose(outputs[0][0], outputs[-1][0], atol=ATOL, rtol=RTOL)
    failures = failure_controls(binary, env)
    real = sum(row_counts(batch))
    return {'status': 'passed', 'scope': 'native CPU batching on synthetic tensor inputs, not search or CUDA',
            'batch': batch, 'channels': encoding.channels, 'forward_calls': 5,
            'executed_real_rows': real, 'physical_rows': 5 * batch, 'padded_rows': 5 * batch - real,
            'accepted_neural_rows': None, 'useful_eps': None, 'failure_controls': failures,
            'input_changes': 0, 'output_changes': 0, 'input_tensor_allocations': 1,
            'max_logit_absolute_error': maximum, 'atol': ATOL, 'rtol': RTOL,
            'checkpoint_sha256': loaded.identity['checkpoint_sha256'],
            'package_sha256': hashlib.sha256(package.read_bytes()).hexdigest(),
            'executable_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'parameter_count': loaded.identity['parameter_count'], 'torch_version': str(torch.__version__)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--transport-only', action='store_true')
    parser.add_argument('--batch', type=int, choices=BATCHES)
    parser.add_argument('--channels', type=int, choices=(146, 175))
    parser.add_argument('--package', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.transport_only:
        if args.batch is None or args.channels is None or args.package or args.checkpoint:
            parser.error('transport-only requires --batch and --channels, not model paths')
        report = verify_transport(args.binary, args.batch, args.channels)
    else:
        if args.package is None or args.checkpoint is None or args.batch or args.channels:
            parser.error('model qualification requires --package and --checkpoint, not shape overrides')
        report = verify_model(args.binary, args.package, args.checkpoint)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
