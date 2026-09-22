"""Queued, bounded actual-trainer ZIP/NVMe pair. Requires an approved frozen plan."""
from __future__ import annotations
import argparse
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any

if __package__:
    from .run_packed_trainer_preparation import digest, inventory, stop_owned
else:
    # Direct script execution has no package; its sibling is on sys.path.
    from run_packed_trainer_preparation import digest, inventory, stop_owned  # pyright: ignore[reportImplicitRelativeImport]

GIB = 1024**3


def run_stage(command, *, cwd, env, log, guard, lease_fd):
    guard()
    child = None
    try:
        with log.open('xb') as stream:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, pass_fds=(lease_fd,))
            while child.poll() is None:
                guard()
                time.sleep(0.5)
            if child.returncode:
                raise RuntimeError(f'owned GPU child {child.pid} exit {child.returncode}')
            guard()
    finally:
        if child is not None:
            # Deliver pending stop signals only after TERM/KILL and reap finish.
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGALRM})
            try:
                stop_owned(child)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def dependency_gate(plan):
    items = json.loads(Path(plan['queue']).read_text())['items']
    matches = [item for item in items if item['id'] == 'bt4_pipeline_prefetch_20260921']
    if len(matches) != 1 or matches[0]['status'] != 'logged':
        raise ValueError('BT4 queue predecessor is not successfully logged')
    item = matches[0]
    if item['command_file'] != plan['bt4_descriptor'] or item['command_sha256'] != digest(plan['bt4_descriptor']):
        raise ValueError('BT4 queue descriptor binding differs')
    dependency = json.loads(Path(plan['bt4_descriptor']).read_text())['completion']
    if json.loads(Path(dependency['path']).read_text())[dependency['status_key']] != dependency['expected']:
        raise ValueError('queued BT4 predecessor has not completed')
    terminal = json.loads((Path(item['out']) / 'parent_outer_terminal.json').read_text())
    if terminal['returncode'] != 0:
        raise ValueError('BT4 outer supervisor failed')


def qualified_inputs(plan):
    preparation = json.loads(Path(plan['preparation_receipt']).read_text())
    qualified = json.loads(Path(plan['qualification']).read_text())
    terminal = json.loads(Path(plan['cpu_complete']).read_text())
    if terminal['status'] != 'PASS_CPU_QUALIFICATION_NOT_GPU_READY' or terminal['qualification_sha256'] != digest(plan['qualification']):
        raise ValueError('CPU qualification supervisor did not authenticate success')
    if preparation['status'] != 'PASS_BYTES_REQUIRES_FULL_STREAM_QUALIFICATION' or qualified['status'] != 'PASS_MATCHED_PACKED_ZARR_SAMPLER':
        raise ValueError('source preparation or full tensor parity not qualified')
    return preparation, qualified


def qualified_batch_count(qualified, rows):
    if len(qualified['runs']) != 2 or any(run['rows'] != rows for run in qualified['runs']):
        raise ValueError('qualified rows or arm count mismatch')
    plans = [run['plan'] for run in qualified['runs']]
    if any(plan['rows_planned'] != rows or plan['shards'] != 256 or plan['sources'] != 35 for plan in plans):
        raise ValueError('qualified plan coverage mismatch')
    batches = plans[0]['batches_planned']
    if batches <= 0 or plans[1]['batches_planned'] != batches:
        raise ValueError('qualified batch counts differ')
    return batches


def validate_arm(summary, observation, *, rows, batches):
    sampling = summary['sampling']
    if not sampling['complete'] or sampling['rows_realized'] != rows:
        raise ValueError('actual epoch coverage incomplete')
    if sampling['batches_realized'] != batches:
        raise ValueError('actual epoch batch count mismatch')
    if observation['status'] != 'TRAINER_RETURNED_SUCCESS':
        raise ValueError('observer did not record successful trainer return')
    if len(observation['batches']) != batches or sum(b['rows'] for b in observation['batches']) != rows:
        raise ValueError('observed batch coverage mismatch')
    windows = summary['train_window_metrics']
    if len(windows) <= 2 or any(w['steps_requested'] != 88 for w in windows[:2]):
        raise ValueError('missing preregistered two warmup windows')
    if sum(w['train_steps_done'] for w in windows) != batches:
        raise ValueError('optimizer step count differs from exact epoch')
    if sum(w['train_samples_seen'] for w in windows) != rows:
        raise ValueError('trained sample count differs from exact epoch')
    if any(not math.isfinite(w['loss']) for w in windows):
        raise ValueError('nonfinite training loss')
    kept = windows[2:]
    seconds = sum(w['train_time_s'] for w in kept)
    samples = sum(w['train_samples_seen'] for w in kept)
    if not math.isfinite(seconds) or seconds <= 0 or samples <= 0:
        raise ValueError('invalid steady timing denominator')
    return {'steady_samples': samples, 'steady_train_seconds': seconds,
        'steady_rows_per_second': samples / seconds,
        'steady_prefetch_wait_seconds': sum(w['batch_prefetch_wait_s'] for w in kept),
        'full_probe_wall_seconds': observation['wall_seconds'],
        'first_batch_seconds': observation['first_batch_seconds'],
        'observer_seconds': observation['observer_seconds']}


def compare(arms, *, rows, batches) -> dict[str, Any]:
    results = {name: validate_arm(value['summary'], value['observation'], rows=rows, batches=batches)
               for name, value in arms.items()}
    external, local = [arms[name]['observation'] for name in ('external_zip', 'nvme_directory')]
    if external['initial_model_sha256'] != local['initial_model_sha256']:
        raise ValueError('paired initial weights differ')
    def sequence(observation):
        return [(b['rows'], b['order_sha256']) for b in observation['batches']]
    if sequence(external) != sequence(local):
        raise ValueError('actual source-qualified game/ply batch order differs')
    ratio = results['external_zip']['steady_rows_per_second'] / results['nvme_directory']['steady_rows_per_second']
    return {'arms': results, 'steady_external_over_nvme': ratio,
            'storage_screen_pass': ratio >= 0.90,
            'threshold': 0.90, 'scope': 'One fixed-order cache-affected trainer pair; no cold-500M claim'}


def authenticate_sources(preparation, roots, guard):
    # Authenticate every current source member, archive and staged path against
    # the banked byte qualification. This also warms caches, as preregistered.
    for key, suffix in [('nvme_directory', '.zarr'), ('external_zip', '.zarr.zip')]:
        expected = [f'shard_{i:06d}{suffix}' for i in range(len(preparation['records']))]
        if sorted(p.name for p in roots[key].iterdir()) != expected:
            raise ValueError('staged source roster differs from qualified bank')
    for index, record in enumerate(preparation['records']):
        guard()
        source, archive = Path(record['source']), Path(record['zip'])
        for key, suffix, target in [('nvme_directory', '.zarr', source), ('external_zip', '.zarr.zip', archive)]:
            if (roots[key] / f'shard_{index:06d}{suffix}').resolve(strict=True) != target:
                raise ValueError('staged source binding drift')
        members = sorted(p for p in source.rglob('*') if p.is_file())
        if [str(p.relative_to(source)) for p in members] != sorted(record['members']):
            raise ValueError('source member roster drift')
        for member in members:
            guard()
            if digest(member) != record['members'][str(member.relative_to(source))]:
                raise ValueError('qualified source bytes drift')
        if digest(archive) != record['zip_sha256']:
            raise ValueError('qualified ZIP bytes drift')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--review-sha256', required=True)
    a = p.parse_args()
    started = time.monotonic()
    def interrupted(signum, _frame):
        raise RuntimeError(f'pair supervisor signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, 5392)
    if digest(a.plan) != a.plan_sha256 or digest(a.review) != a.review_sha256:
        raise ValueError('pair plan/review pin mismatch')
    review = json.loads(a.review.read_text())
    if review['status'] != 'APPROVED' or review['plan_sha256'] != a.plan_sha256:
        raise ValueError('review does not approve exact GPU pair plan')
    plan = json.loads(a.plan.read_text())
    if plan['status'] != 'APPROVED_FOR_QUEUED_GPU_PAIR':
        raise ValueError('GPU pair plan not approved')
    os.sched_setaffinity(0, {12, 13})
    os.nice(19)
    runtime, out = Path(plan['runtime']), Path(plan['out'])
    def repin():
        for name, version in plan['dependency_versions'].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f'host dependency version drift: {name}')
        for pin in plan['pins']:
            if digest(pin['path']) != pin['sha256']:
                raise ValueError(f'pair pin drift: {pin["path"]}')
        if inventory(runtime) != plan['runtime_files']:
            raise ValueError('pair runtime content/membership drift')
    repin()
    head = subprocess.check_output(['git', '-C', str(runtime), 'rev-parse', 'HEAD'], timeout=10, text=True).strip()
    if head != plan['runtime_commit']:
        raise ValueError('runtime commit mismatch')
    dependency_gate(plan)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    preparation, qualified = qualified_inputs(plan)
    expected_rows = plan['rows']
    batches = qualified_batch_count(qualified, expected_rows)
    out.mkdir(parents=True)
    status: dict[str, Any] = {'status': 'INCOMPLETE', 'plan_sha256': a.plan_sha256, 'stages': []}
    arm_start = None
    def guard():
        now = time.monotonic()
        if now - started >= 5392 or (arm_start is not None and now - arm_start >= 2692):
            raise RuntimeError('pair90min/arm45min cap with cleanup reserve')
        if (out / 'STOP').exists():
            raise RuntimeError('STOP requested')
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(mem['MemAvailable'].split()[0]) * 1024 < 40 * GIB:
            raise RuntimeError('40GiB available RAM floor')
        for root in (out, Path(plan['external_root'])):
            if shutil.disk_usage(root).free < 150 * GIB:
                raise RuntimeError('150GiB disk floor')
        used = plan['preparation_new_bytes']
        for base, _dirs, files in os.walk(out, followlinks=False):
            for name in files:
                path = Path(base) / name
                if not path.is_symlink():
                    used += path.stat().st_size
        if used > 10 * GIB:
            raise RuntimeError('10GiB combined preparation+GPU outputs cap')
        status['combined_new_disk_bytes'] = used
    env = dict(os.environ)
    for key in ('PYTHONOPTIMIZE', 'PYTHONHOME', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(dict.fromkeys(('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                             'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS'), '2'))
    env.update(CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(runtime))
    roots = {k: Path(v) for k, v in plan['roots'].items()}
    arms = {}
    lease = None
    try:
        lease = Path(plan['gpu_lock']).open('a')  # noqa: SIM115 -- closed after owned-child cleanup in finally
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard()
        authenticate_sources(preparation, roots, guard)
        for name in ('external_zip', 'nvme_directory'):
            guard()
            # Admission check is read-only: never stop somebody else's CUDA process.
            gpu_pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], timeout=10, text=True).strip()
            if gpu_pids:
                raise RuntimeError(f'GPU already has compute processes: {gpu_pids}')
            work = out / (name + '_work')
            work.mkdir()
            result = out / name
            observation = out / (name + '_observation.json')
            child_env = dict(env, TORCHINDUCTOR_CACHE_DIR=str(work / 'inductor'),
                             XDG_CACHE_HOME=str(work / 'cache'), TMPDIR=str(work))
            command = [plan['python'], plan['probe'], '--runtime', str(runtime),
                '--driver-sha256', plan['driver_sha256'], '--receipt', str(observation),
                '--gpu-lease-path', plan['gpu_lock'], '--gpu-lease-fd', str(lease.fileno()), '--',
                '--config', plan['config'], '--shards', str(roots[name]), '--out-dir', str(result),
                '--steps', '0', '--epochs', '1', '--sampling-mode', 'game_epoch',
                '--seed', '121', '--batch-size', '512', '--train-window-steps', '88',
                '--epoch-plan-workers', '2', '--epoch-load-workers', '2',
                '--epoch-max-working-set-gib', '12', '--allow-partial-corpus', '--allow-invalid-control']
            if name == 'external_zip':
                command.append('--allow-packed-zarr')
            arm_start = time.monotonic()
            signal.setitimer(signal.ITIMER_REAL, max(0.001, min(2692, 5392 - (arm_start - started))))
            status['stages'].append({'name': name, 'command': command, 'status': 'STARTED'})
            (out / 'progress.json').write_text(json.dumps(status, indent=2))
            run_stage(command, cwd=work, env=child_env, log=out / f'{name}.log', guard=guard, lease_fd=lease.fileno())
            status['stages'][-1].update(status='PASS', wall_seconds=time.monotonic() - arm_start)
            arm_start = None
            signal.setitimer(signal.ITIMER_REAL, max(0.001, 5392 - (time.monotonic() - started)))
            arms[name] = {'summary': json.loads((result / 'summary.json').read_text()),
                          'observation': json.loads(observation.read_text())}
            validate_arm(arms[name]['summary'], arms[name]['observation'], rows=expected_rows, batches=batches)
            repin()
            authenticate_sources(preparation, roots, guard)
        status['comparison'] = compare(arms, rows=expected_rows, batches=batches)
        guard()
        status['status'] = 'PASS_EXACT_PACKED_TRAINER_PAIR'
    except BaseException as error:
        status.update(status='INCOMPLETE', error=repr(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if lease is not None:
            lease.close()
        status['wall_seconds'] = time.monotonic() - started
        (out / 'complete.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
