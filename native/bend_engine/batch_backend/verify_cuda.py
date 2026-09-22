"""Explicit CUDA/BF16 native-versus-eager gate; never substitutes a CPU run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from typing import BinaryIO

import numpy as np

from .verify import (BATCHES, LOGITS, clean_env, decode_results, expected_input,
                     failure_controls, row_counts, run)
from native.bend_engine.neural_probe.backend import execution_spec
from native.bend_engine.neural_probe.qualification import protect_inputs, write_report


def cuda_index(manifest: dict[str, object]) -> int:
    device, dtype, index = execution_spec(manifest)
    if (device, dtype) != ('cuda', 'bfloat16'):
        raise ValueError('CUDA gate requires an explicit CUDA/BF16 checkpoint package')
    return index


def tolerance_pair(atol: float, rtol: float) -> tuple[float, float]:
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError('predeclared tolerances must be finite and nonnegative')
    return atol, rtol


def require_device(index: int) -> dict[str, object]:
    import torch
    if type(index) is not int or not 0 <= index <= 127:
        raise ValueError('invalid CUDA device index')
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError('requested CUDA device is unavailable; no CPU fallback or qualification')
    properties = torch.cuda.get_device_properties(index)
    if properties.major < 8:
        raise RuntimeError('native CUDA backend requires hardware BF16 support (SM80+)')
    return {'device_index': index, 'name': properties.name,
            'compute_capability': [properties.major, properties.minor],
            'torch_version': str(torch.__version__), 'cuda_build': torch.version.cuda}


def trace_record(stream: BinaryIO, step: int, batch: int, rows: int,
                 channels: int, index: int) -> tuple[np.ndarray, np.ndarray]:
    if batch not in BATCHES or not 1 <= rows <= batch or channels not in (146, 175) or not 0 <= index <= 127:
        raise ValueError('invalid CUDA trace dimensions')
    header = stream.read(32)
    expected = (0x44464333, step, batch, rows, channels, LOGITS, index, 16)
    if len(header) != 32 or struct.unpack('<8I', header) != expected:
        raise ValueError('CUDA BF16 trace header mismatch')
    inputs, outputs = batch * channels * 64, rows * LOGITS
    raw = stream.read(2 * inputs + 4 * outputs)
    if len(raw) != 2 * inputs + 4 * outputs:
        raise ValueError('truncated CUDA trace')
    # Keep BF16 input bits as U16; do not reinterpret them as F16 or F32.
    physical = np.frombuffer(raw, dtype='<u2', count=inputs).copy().reshape(batch, channels, 8, 8)
    result = np.frombuffer(raw, dtype='<f4', offset=2 * inputs).copy().reshape(rows, LOGITS)
    return physical, result


def validate_audit(stderr: str, index: int, *, tracing: bool) -> None:
    expected = ['native-buffer-audit calls=5 input_changes=0 output_changes=0 input_tensor_allocations=1',
                f'native-cuda-audit device_index={index} stream_nondefault=1 pinned_buffers={3 if tracing else 2} completion_events=1']
    if stderr.splitlines() != expected:
        raise ValueError('CUDA model buffer/stream audit mismatch: ' + stderr)


def verify_model(binary: Path, package: Path, checkpoint: Path, *, atol: float, rtol: float) -> dict[str, object]:
    import torch
    from chess_anti_engine.inference import _policy_output
    from native.bend_engine.neural_probe.backend import package_manifest
    from native.bend_engine.neural_probe.checkpoint import load_checkpoint
    from native.bend_engine.neural_probe.qualification import verified_package

    atol, rtol = tolerance_pair(atol, rtol)
    torch.set_num_threads(2)
    manifest, encoding = package_manifest(package)
    index = cuda_index(manifest)
    device = require_device(index)  # fail before loading/moving any checkpoint
    batch = manifest['batch']
    if type(batch) is not int or batch not in BATCHES:
        raise ValueError('unsupported CUDA batch')
    loaded = load_checkpoint(checkpoint, weights_key=str(manifest['weights_key']))
    verified_package(package, loaded, batch=batch, device='cuda', device_index=index)
    model = loaded.model.to(device=torch.device('cuda', index), dtype=torch.bfloat16).eval()
    env = {**clean_env(), 'DEEPFIN_BEND_MODEL_PACKAGE': str(package.resolve()),
           'DEEPFIN_BEND_BUFFER_AUDIT': '1'}
    maximum = 0.0
    with tempfile.TemporaryDirectory(prefix='deepfin-cuda-batch-') as tmp:
        trace = Path(tmp) / 'trace.bin'
        result = run(binary, {**env, 'DEEPFIN_BEND_MODEL_TRACE': str(trace)})
        if result.returncode:
            raise ValueError('native CUDA execution failed: ' + result.stderr)
        outputs = decode_results(result.stdout, batch)
        validate_audit(result.stderr, index, tracing=True)
        with trace.open('rb') as stream, torch.no_grad(), torch.cuda.device(index):
            for step, (rows, output) in enumerate(zip(row_counts(batch), outputs, strict=True), 1):
                physical, traced = trace_record(stream, step, batch, rows, encoding.channels, index)
                expected = expected_input(step, rows, encoding.channels)
                bits = torch.from_numpy(expected).to(torch.bfloat16).view(torch.uint16).numpy()
                np.testing.assert_array_equal(physical[:rows], bits)
                np.testing.assert_array_equal(physical[rows:], 0)
                np.testing.assert_array_equal(output.view(np.uint32), traced.view(np.uint32))
                for row in range(rows):
                    reference = model(torch.from_numpy(expected[row:row + 1].copy()).to(device=f'cuda:{index}', dtype=torch.bfloat16))
                    want = torch.cat((_policy_output(reference), reference['wdl']), dim=1).float().cpu().numpy()[0]
                    np.testing.assert_allclose(output[row], want, atol=atol, rtol=rtol)
                    maximum = max(maximum, float(np.max(np.abs(output[row] - want))))
            if stream.read(1):
                raise ValueError('unexpected trailing CUDA forward')
        np.testing.assert_allclose(outputs[0][0], outputs[-1][0], atol=atol, rtol=rtol)
        # Exercise the ordinary no-trace path as well, not only diagnostic D2H.
        quiet = run(binary, env)
        if quiet.returncode:
            raise ValueError('quiet CUDA execution failed: ' + quiet.stderr)
        validate_audit(quiet.stderr, index, tracing=False)
        for one, two in zip(outputs, decode_results(quiet.stdout, batch), strict=True):
            np.testing.assert_allclose(one, two, atol=atol, rtol=rtol)
    failures = failure_controls(binary, env)
    real = sum(row_counts(batch))
    return {'status': 'passed', 'cuda_model_qualified': True,
            'scope': 'native versus eager BF16 on synthetic tensors; not FP32 fidelity, chess strength or all inputs',
            'device': device, 'batch': batch, 'channels': encoding.channels,
            'forward_calls': 10, 'executed_real_rows': 2 * real,
            'physical_rows': 10 * batch, 'padded_rows': 10 * batch - 2 * real,
            'accepted_neural_rows': None, 'useful_eps': None, 'failure_controls': failures,
            'max_logit_absolute_error': maximum, 'atol': atol, 'rtol': rtol,
            'checkpoint_sha256': loaded.identity['checkpoint_sha256'],
            'weights_key': manifest['weights_key'], 'training_provenance': 'not inferred',
            'parameter_count': loaded.identity['parameter_count'],
            'package_sha256': hashlib.sha256(package.read_bytes()).hexdigest(),
            'executable_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'package', 'checkpoint', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--atol', type=float, required=True, help='Predeclare before execution; no inferred BF16 tolerance')
    parser.add_argument('--rtol', type=float, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint / 'trainer.pt' if args.checkpoint.is_dir() else args.checkpoint
    protect_inputs(args.report, [checkpoint, args.package, args.package.with_suffix('.json'), args.binary])
    # Reserve a NEW report before running. Never overwrite an earlier PASS.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as output:
        output.write('{"status":"started","cuda_model_qualified":false}\n')
    try:
        report = verify_model(args.binary, args.package, checkpoint, atol=args.atol, rtol=args.rtol)
    except Exception as error:
        write_report(args.report, {'status': 'failed', 'cuda_model_qualified': False,
                                  'error': str(error), 'atol': args.atol if math.isfinite(args.atol) else None,
                                  'rtol': args.rtol if math.isfinite(args.rtol) else None})
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    write_report(args.report, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
