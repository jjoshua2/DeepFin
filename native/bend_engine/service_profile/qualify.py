"""Explicit source-only native service-probe checks, without Torch or a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

from .profile import object_json, validate_native

HERE = Path(__file__).resolve().parent


def qualify(out: Path, cxx: str) -> dict[str, object]:
    results = []
    for batch, channels, sanitize in [(b, c, False) for b in (1, 2, 4, 8, 16) for c in (146, 175)] + [
            (4, 146, True), (4, 175, True)]:
        d = out / f'b{batch}-c{channels}-{"ubsan" if sanitize else "normal"}'
        d.mkdir()
        (d / 'model_contract.h').write_text(
            f'#define DEEPFIN_MODEL_BATCH {batch}\n#define DEEPFIN_MODEL_CHANNELS {channels}\n'
            '#define DEEPFIN_MODEL_CUDA 0\n#define DEEPFIN_MODEL_PROFILE 2\n'
            '#define DEEPFIN_MODEL_SHA256 "' + 'a' * 64 + '"\n'
            '#define DEEPFIN_CHECKPOINT_SHA256 "' + 'b' * 64 + '"\n')
        binary = d / 'probe'
        command = [cxx, '-std=c++20', '-O1', '-ffp-contract=off', '-I', str(d),
                   str(HERE / 'probe.cpp'), str(HERE / 'fake_backend.cpp'), '-lcrypto', '-o', str(binary)]
        if sanitize:
            command += ['-fsanitize=undefined', '-fno-sanitize-recover=all']
        build = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
        (d / 'build.txt').write_text(build.stdout + build.stderr)
        if build.returncode:
            raise RuntimeError(build.stderr)
        inputs = [0.0] * (batch * channels * 64)
        for i in range(batch):
            inputs[i * channels * 64] = i + 0.25
        expected = [i + 0.25 + j / 4096 for i in range(batch) for j in range(1861)]
        x, y = d / 'input.f32', d / 'expected.f32'
        x.write_bytes(struct.pack('<' + 'f' * len(inputs), *inputs))
        y.write_bytes(struct.pack('<' + 'f' * len(expected), *expected))
        identity = {'batch': batch, 'channels': channels, 'package_sha256': 'a' * 64,
                    'checkpoint_sha256': 'b' * 64,
                    'input_sha256': hashlib.sha256(x.read_bytes()).hexdigest(),
                    'reference_sha256': hashlib.sha256(y.read_bytes()).hexdigest()}
        env = {k: v for k, v in os.environ.items() if not k.startswith(('DEEPFIN_', 'SERVICE_TEST_'))}
        args = [str(binary), str(x), str(y), '2', '3', str(batch - 1)]
        nominal = subprocess.run(args, check=False, env=env, capture_output=True, text=True, timeout=30)
        if nominal.returncode or nominal.stderr:
            raise ValueError(nominal.stderr)
        r = object_json(nominal.stdout)
        validate_native(r, identity, 2, 3, batch - 1)
        (d / 'nominal.json').write_text(json.dumps(r, indent=2) + '\n')
        rejected = 0
        for fault in ('startup', 'fail', 'throw', 'nonfinite', 'mismatch', 'tail'):
            child = subprocess.run(args, check=False, env={**env, 'SERVICE_TEST_FAULT': fault},
                                   capture_output=True, text=True, timeout=30)
            if child.returncode != 2 or child.stdout or 'native service profile:' not in child.stderr:
                raise ValueError('native negative control failed: ' + fault)
            rejected += 1
        for name in ('DEEPFIN_BEND_MODEL_TRACE', 'DEEPFIN_BEND_BUFFER_AUDIT'):
            child = subprocess.run(args, check=False, env={**env, name: '1'}, capture_output=True, text=True, timeout=30)
            if child.returncode != 2 or child.stdout or 'disable trace/audit' not in child.stderr:
                raise ValueError('trace/audit contaminated measurement')
            rejected += 1
        for index, value in ((3, '0'), (3, '33'), (4, '1'), (4, '257'), (5, str(batch)), (3, '+1'), (4, '03')):
            bad = args.copy()
            bad[index] = value
            child = subprocess.run(bad, check=False, env=env, capture_output=True, text=True, timeout=30)
            if child.returncode != 2 or child.stdout:
                raise ValueError('invalid arguments accepted')
            rejected += 1
        for target in (x, y):
            original = target.read_bytes()
            for bad in (original[:-1], original + b'x', bytes.fromhex('0000c07f') + original[4:]):
                target.write_bytes(bad)
                child = subprocess.run(args, check=False, env=env, capture_output=True, text=True, timeout=30)
                if child.returncode != 2 or child.stdout:
                    raise ValueError('bad input/reference accepted')
                rejected += 1
            target.write_bytes(original)
        results.append({'batch': batch, 'channels': channels, 'ubsan': sanitize,
                        'native_calls_checked': batch * 5, 'negative_controls': rejected, 'status': 'passed'})
    return {'status': 'passed', 'scope': 'native timer/probe with deterministic callback, not a model',
            'cases': results}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cxx', default='clang++')
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    try:
        report = qualify(a.out, a.cxx)
    except Exception as error:
        (a.out / 'report.json').write_text(json.dumps({'status': 'failed', 'error': str(error)}))
        raise
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
