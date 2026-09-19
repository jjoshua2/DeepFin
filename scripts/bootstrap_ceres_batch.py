#!/usr/bin/env python3
"""Run a pinned sequential Ceres coverage plan, stopping at the first failure."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from bootstrap_experiment_operator import dump, registered_spec, terminate_owned_group


class BatchInterrupted(RuntimeError):
    pass


def stop_signal(_signum, _frame):
    raise BatchInterrupted('batch termination requested')


def available_ram_gib() -> float:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024**2
    return 0


def allocated_bytes(roots: list[str]) -> int:
    def allocation(path: Path) -> int:
        try:
            return path.lstat().st_blocks * 512
        except FileNotFoundError:
            # Writers atomically rename temporary chunks and shard directories.
            # This is a sampled bound; the replacement appears on the next scan.
            return 0

    def walk_error(error: OSError) -> None:
        if not isinstance(error, FileNotFoundError):
            raise error

    total = 0
    for root in roots:
        for directory, _, files in os.walk(root, followlinks=False, onerror=walk_error):
            total += allocation(Path(directory))
            for name in files:
                total += allocation(Path(directory) / name)
    return total


def resource_check(plan: dict, *, startup: bool = False) -> None:
    free = shutil.disk_usage(plan['disk_path']).free / 2**30
    floor = plan['startup_disk_gib'] if startup else plan['disk_floor_gib']
    if free < floor:
        raise RuntimeError(f'disk available {free:.2f} GiB below {floor}')
    ram = available_ram_gib()
    ram_floor = plan['startup_ram_gib'] if startup else plan['running_ram_gib']
    if ram < ram_floor:
        raise RuntimeError(f'RAM available {ram:.2f} GiB below {ram_floor}')
    size = allocated_bytes(plan['output_roots'])
    if size > plan['output_cap_bytes']:
        raise RuntimeError(f'allocated batch output {size} exceeds aggregate cap')


def verify_completion(completion: dict) -> dict:
    result = json.loads(Path(completion['path']).read_text())
    if result[completion['status_key']] != completion['expected']:
        raise ValueError('qualification status mismatch: '+completion['path'])
    return result


def run(plan: dict) -> int:
    output = Path(plan['completion_path'])
    if output.exists():
        raise ValueError('batch already has a completion record')
    started = time.monotonic()
    deadline = started + plan['internal_seconds']
    completed = []
    active = None
    receipt = {'status': 'INCOMPLETE', 'completed_blocks': completed,
               'started_unix': time.time()}
    try:
        verify_completion(plan['prerequisite'])
        resource_check(plan, startup=True)
        for block in plan['blocks']:
            if deadline - time.monotonic() < block['wall_seconds'] + 30:
                raise RuntimeError('remaining shared time cannot fit next registered block')
            spec = registered_spec(block)
            resource_check(plan)
            receipt['active_block'] = block['id']
            dump(output.with_name('progress.json'), receipt)
            env = os.environ.copy()
            env.update(spec.get('env', {}))
            for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'NUMEXPR_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS'):
                env[key] = '2'
            with Path(block['log_path']).open('w') as handle:
                active = subprocess.Popen(spec['argv'], cwd=spec['cwd'], env=env,
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                block_deadline = min(deadline, time.monotonic()+block['wall_seconds']-30)
                next_resource = 0.0
                while active.poll() is None:
                    now = time.monotonic()
                    if now >= block_deadline:
                        raise RuntimeError('registered block exceeded time allowance')
                    if now >= next_resource:
                        resource_check(plan)
                        next_resource = now + 30
                    time.sleep(min(2, block_deadline-now))
                if active.returncode != 0:
                    raise RuntimeError(f'block {block["id"]} exited {active.returncode}')
                active = None
            verify_completion(spec['completion'])
            resource_check(plan)
            completed.append(block['id'])
            receipt['active_block'] = None
            dump(output.with_name('progress.json'), receipt)
        receipt['status'] = plan['success_status']
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        receipt['reason'] = str(exc)
        return 1
    finally:
        if active is not None:
            # Ignore repeated outer TERM while cleaning the newly owned child group.
            old_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                terminate_owned_group(active, grace=20)
            finally:
                signal.signal(signal.SIGTERM, old_handler)
        receipt['elapsed_seconds'] = time.monotonic()-started
        receipt['ended_unix'] = time.time()
        dump(output, receipt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.expected_plan_sha256:
        raise ValueError('batch plan hash mismatch')
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    return run(json.loads(raw))


if __name__ == '__main__':
    raise SystemExit(main())
