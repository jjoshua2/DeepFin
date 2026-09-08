#!/usr/bin/env python3
"""One fixed, copy-only archive invocation; caller supplies outer timeout/deadline."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

STATE = Path(__file__).resolve().parent
PLAN = STATE / 'plan.json'


def build_command(plan, deadline):
    remaining = deadline - time.time()
    if not math.isfinite(deadline) or not 30 < remaining <= 14400:
        raise ValueError('absolute deadline must leave 30..14400 seconds')
    return ['/usr/bin/timeout', '--signal=TERM', '--kill-after=30s', f'{remaining - 30:.6f}s',
            '/usr/bin/flock', '--exclusive', '--nonblock', '--no-fork', plan['preparation_lock'],
            '/usr/bin/time', '-v', '-o', str(STATE / 'run' / 'resources.txt'),
            '/usr/bin/ionice', '-c', '3', '/usr/bin/nice', '-n', '19',
            '/usr/bin/taskset', '-c', '0,1', '/usr/bin/python3.10', str(STATE / 'archive_G20T1.py'),
            '--plan', str(PLAN), '--pool', plan['pools'][0]['name'], '--execute',
            '--deadline', str(deadline)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    raw = PLAN.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.plan_sha256:
        raise ValueError('plan bytes changed')
    plan = json.loads(raw)
    for pin in plan['input_pins']:
        if hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest() != pin['sha256']:
            raise ValueError(f'changed input: {pin["path"]}')
    command = build_command(plan, args.deadline)
    if (STATE / 'STOP').exists() or (STATE.parent / 'STOP').exists():
        raise RuntimeError('STOP before archive launch')
    run = STATE / 'run'
    run.mkdir(exist_ok=False)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    with (run / 'started.json').open('x') as f:
        json.dump({'status': 'STARTED_COPY_ONLY', 'started_unix': time.time(),
                   'deadline_unix': args.deadline, 'plan_sha256': args.plan_sha256,
                   'argv': command, 'source_retained': True}, f, indent=2)
        f.write('\n')
    log = os.open(run / 'driver.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.close(log)
    os.execve(command[0], command, env)


if __name__ == '__main__':
    main()
