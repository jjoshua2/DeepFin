"""Explicit real CPU-model qualification of the native async adapter.

Runs direct/worker and quiet/traced controls in distinct native processes. These
are deterministic synthetic input tensors, not chess-selected leaves or a speed
measurement. The cancelled callback must execute but must not publish its output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile

WIDTH = 1861
STEPS = 6
WORD = re.compile(r'[0-9a-f]{8}')
AUDIT = 'native-buffer-audit calls=6 input_changes=0 output_changes=0 input_tensor_allocations=1'


def parse_output(text: str, *, asynchronous: bool) -> dict[int, bytes]:
    lines = text.splitlines()
    expected = [1, 2, 4, 5, 6] if asynchronous else list(range(1, STEPS + 1))
    rows: dict[int, bytes] = {}
    assert len(lines) == STEPS + 1, 'missing/extra output record'
    for step, line in enumerate(lines[:-1], 1):
        fields = line.split()
        if asynchronous and step == 3:
            assert fields == ['cancelled', '3'], 'cancelled output was published or unaccounted'
            continue
        assert len(fields) == WIDTH + 2, 'output row width'
        assert fields[:2] == ['result', str(step)], 'out-of-order/duplicate result'
        assert all(WORD.fullmatch(word) for word in fields[2:]), 'malformed raw output word'
        rows[step] = struct.pack('<' + 'I' * WIDTH, *(int(word, 16) for word in fields[2:]))
    assert list(rows) == expected
    assert lines[-1] == ('done 5 1' if asynchronous else 'done 6 0'), 'wrong dispositions'
    return rows


def parse_audit(text: str) -> None:
    assert text.splitlines() == [AUDIT], 'missing/malformed/extra audit or changed model storage'


def read_trace(path: Path, channels: int) -> list[tuple[bytes, bytes]]:
    count = channels * 64
    assert channels in (146, 175), 'unsupported channels'
    records = []
    with path.open('rb') as stream:
        for step in range(1, STEPS + 1):
            raw = stream.read(16)
            assert len(raw) == 16, 'missing physical forward'
            assert struct.unpack('<4I', raw) == (0x44464c31, step, count, WIDTH), 'trace identity/shape'
            inputs, outputs = stream.read(count * 4), stream.read(WIDTH * 4)
            assert len(inputs) == count * 4, 'truncated physical input'
            assert len(outputs) == WIDTH * 4, 'truncated physical output'
            records.append((inputs, outputs))
        assert stream.read(1) == b'', 'extra physical forward'
    return records


def expected_input(step: int, channels: int) -> bytes:
    pattern = 1 if step == 5 else step
    return struct.pack('<' + 'f' * (channels * 64),
                       *((((i * 17 + pattern * 13) % 257) - 128) / 128 for i in range(channels * 64)))


def verify(binary: Path, package: Path, checkpoint: Path) -> dict[str, object]:
    # Expensive model dependencies are outside cheap parser tests/imports.
    import numpy as np
    import torch
    from native.bend_engine.neural_probe.checkpoint import EagerReference, load_checkpoint

    torch.set_num_threads(2)
    manifest = json.loads(package.with_suffix('.json').read_text())
    assert manifest['batch'] == 1
    assert manifest['device'] == 'cpu'
    assert manifest['dtype'] == 'float32'
    assert hashlib.sha256(package.read_bytes()).hexdigest() == manifest['sha256']
    loaded = load_checkpoint(checkpoint, weights_key=manifest['weights_key'])
    assert loaded.identity['checkpoint_sha256'] == manifest['checkpoint_sha256']
    channels = loaded.encoding.channels
    assert channels == manifest['channels']
    eager = EagerReference(loaded.model, torch.device('cpu'), torch.float32)
    observations: dict[str, dict[int, bytes]] = {}
    traces: dict[str, list[tuple[bytes, bytes]]] = {}
    max_error = 0.0
    with tempfile.TemporaryDirectory(prefix='deepfin-async-model-') as tmp:
        for mode in ('sync', 'async'):
            for traced in (False, True):
                name = mode + ('-traced' if traced else '-quiet')
                trace = Path(tmp) / (name + '.trace')
                env = {**os.environ, 'DEEPFIN_BEND_MODEL_PACKAGE': str(package.resolve()),
                       'DEEPFIN_BEND_BUFFER_AUDIT': '1'}
                env.pop('DEEPFIN_BEND_MODEL_TRACE', None)
                if traced:
                    env['DEEPFIN_BEND_MODEL_TRACE'] = str(trace)
                run = subprocess.run([str(binary.resolve()), mode], env=env, text=True,
                                     capture_output=True, timeout=120, check=True)
                parse_audit(run.stderr)
                observations[name] = parse_output(run.stdout, asynchronous=(mode == 'async'))
                if traced:
                    traces[mode] = read_trace(trace, channels)
                else:
                    assert not trace.exists()
        assert observations['sync-quiet'] == observations['sync-traced']
        assert observations['async-quiet'] == observations['async-traced']
        assert observations['async-quiet'] == {i: y for i, y in observations['sync-quiet'].items() if i != 3}
        assert traces['sync'] == traces['async'], 'thread/cancellation changed actual model inputs or raw outputs'
        assert traces['sync'][0] == traces['sync'][4], 'repeated inputs/results changed across intervening work'
        for step, (x_bytes, y_bytes) in enumerate(traces['sync'], 1):
            assert x_bytes == expected_input(step, channels), 'snapshot corruption or incorrect input'
            assert y_bytes == observations['sync-quiet'][step], 'caller publication differs from native trace'
            x = np.frombuffer(x_bytes, dtype='<f4').copy().reshape(1, channels, 8, 8)
            with torch.inference_mode():
                reference = eager(torch.from_numpy(x))
            want = torch.cat((reference['policy'][0], reference['wdl'][0])).numpy()
            got = np.frombuffer(y_bytes, dtype='<f4')
            assert np.isfinite(got).all()
            np.testing.assert_allclose(got, want, atol=2e-6, rtol=2e-5)
            max_error = max(max_error, float(np.max(np.abs(got - want))))
    return {
        'status': 'passed', 'scope': 'native CPU singleton worker/model adapter; synthetic inputs, not UCI/search/GPU',
        'native_processes': 4, 'forwards_per_process': STEPS,
        'worker_processes': 2, 'worker_completed_per_process': 5, 'worker_cancelled_per_process': 1,
        'direct_worker_trace_bits_identical': True, 'quiet_traced_outputs_identical': True,
        'snapshot_and_repeated_input_checks': True, 'stable_bridge_buffers': True,
        'bridge_input_tensor_allocations_per_process': 1,
        'max_logit_absolute_error': max_error, 'atol': 2e-6, 'rtol': 2e-5,
        'package_sha256': manifest['sha256'], 'checkpoint_sha256': manifest['checkpoint_sha256'],
        'parameter_count': loaded.identity['parameter_count'], 'torch_version': str(torch.__version__),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('binary', 'package', 'checkpoint', 'report'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    assert not args.report.exists(), 'refusing to overwrite a report'
    result = verify(args.binary, args.package, args.checkpoint)
    with args.report.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
