"""Opt-in verification of the compiled async worker and Bend foreign-effect probe.

Uses subprocesses and actual native threads, but only deterministic test callbacks.
No checkpoint, GPU, standalone search or performance claim is involved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def verify(slot_binary: Path, bend_binary: Path, material_binary: Path) -> dict[str, object]:
    slot = subprocess.run([str(slot_binary.resolve())], capture_output=True, text=True,
                          timeout=20, check=True)
    if slot.stderr:
        raise AssertionError(slot.stderr)
    result = json.loads(slot.stdout)
    assert result['status'] == 'passed', result
    assert result['assertions'] >= 440, result
    env = {**os.environ, 'DEEPFIN_BEND_ASYNC': '1'}
    run = subprocess.run([str(bend_binary.resolve()), '--threads', '1'], env=env,
                         capture_output=True, text=True, timeout=15, check=True)
    assert not run.stderr, run.stderr
    assert run.stdout.splitlines() == [
        'submitted 1', 'cancelled 3', 'submitted 2', 'complete 2', 'duplicate 0', 'passed',
    ], run.stdout
    rejected = 0
    for value in (None, '0', '2', 'true', '1 '):
        bad_env = dict(env)
        if value is None:
            bad_env.pop('DEEPFIN_BEND_ASYNC', None)
        else:
            bad_env['DEEPFIN_BEND_ASYNC'] = value
        bad = subprocess.run([str(bend_binary.resolve()), '--threads', '1'], env=bad_env,
                             capture_output=True, text=True, timeout=10, check=False)
        assert bad.returncode == 2, bad
        assert not bad.stdout, bad
        assert bad.stderr, bad
        rejected += 1
    absent = subprocess.run([str(material_binary.resolve()), '--threads', '1'], env=env,
                            capture_output=True, text=True, timeout=10, check=False)
    assert absent.returncode == 2, absent
    assert not absent.stdout, absent
    assert absent.stderr, absent
    return {'status': 'passed', 'worker': result, 'bend_cancel_then_reuse': True,
            'invalid_or_disabled_options_rejected': rejected,
            'missing_native_link_rejected': True,
            'scope': 'bounded worker and generated Bend/C boundary only; no search/model/GPU run'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slot', type=Path, required=True)
    parser.add_argument('--bend', type=Path, required=True)
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.slot, args.bend, args.material)
    with args.report.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
