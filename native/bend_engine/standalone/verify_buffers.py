"""Opt-in compiled transport/array-logit regressions. No real model is implied."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess


def run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, env=env, timeout=30, check=False)


def verify(transport: Path, array_logits: Path, list_logits: Path) -> dict[str, object]:
    env = os.environ.copy()
    env.pop('DEEPFIN_PROBE_FAIL', None)
    env.pop('DEEPFIN_BEND_NATIVE_DIAGNOSTICS', None)
    command = [str(transport.resolve()), '--threads', '1']
    for mode, count in [(0, 9344), (6, 11200)]:
        for diag in ['0', '1']:
            p = run([*command, '--', str(mode)], {**env, 'DEEPFIN_BEND_NATIVE_DIAGNOSTICS': diag})
            assert p.returncode == 0, p.stderr
            lines = p.stdout.splitlines()
            assert lines[:2] == ['profile=1', f'diagnostics={diag}']
            want = [f'{i} {count} {struct.unpack("<I", struct.pack("<f", float(i + 1860)))[0]}'
                    for i in range(1, 5)]
            assert lines[2:] == want, p.stdout
            assert p.stderr.splitlines() == [f'buffer-reuse-call={i} rows={count}' for i in range(1, 5)]
    failures = 0
    for mode in range(1, 6):
        p = run([*command, '--', str(mode)], env)
        assert p.returncode == 2, (mode, p)
        assert 'buffer contract failed' in p.stderr, (mode, p)
        assert 'buffer-reuse-call' not in p.stderr
        failures += 1
    for setting in [{'DEEPFIN_PROBE_FAIL': '1'}, {'DEEPFIN_BEND_NATIVE_DIAGNOSTICS': 'bogus'}]:
        p = run(command, {**env, **setting})
        assert p.returncode == 2, p
        assert 'buffer-reuse-call' not in p.stderr
        failures += 1
    p = run([str(list_logits.resolve()), '--threads', '1'], env)
    q = run([str(array_logits.resolve()), '--threads', '1'], env)
    assert p.returncode == q.returncode == 0, (p.stderr, q.stderr)
    old, new = p.stdout.splitlines(), q.stdout.splitlines()
    assert len(old) == 11
    assert len(new) == 15
    assert new[:11] == old, 'buffer conversion changed finite checks or numeric results'
    for mode, line in enumerate(new[11:], 11):
        assert list(map(int, line.split())) == [mode, 1, 0, 0, 0, 0, 0, 0]
    return {'status': 'passed', 'reused_buffer_roundtrips': 16, 'transport_failures_rejected': failures,
            'bit_exact_list_array_cases': 11, 'additional_array_contract_rejections': 4,
            'scope': 'generated native effect with test backend; not LibTorch performance'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transport', type=Path, required=True)
    p.add_argument('--array-logits', type=Path, required=True)
    p.add_argument('--list-logits', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    report = verify(args.transport, args.array_logits, args.list_logits)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
