"""Queued, bounded NVMe host-overlap OFF/ON pair. Requires an approved frozen plan."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

if __package__:
    from .run_packed_trainer_preparation import digest, inventory
else:
    # Direct script execution has no package; its sibling is on sys.path.
    from run_packed_trainer_preparation import digest, inventory  # pyright: ignore[reportImplicitRelativeImport]

GIB = 1024**3
RUNTIME_COMMIT = '502cd02e072471c901255f3fdb580d6ea7b826d0'
RUNTIME_ROOT = '/tmp/deepfin-factorial58-runtime'
INPUT_ROOT = '/home/josh/chess-artifacts/operations/packed-trainer256-20260921/directory'
CONFIG_SHA256 = '413dbea9dcde2774eafc2fde706e639fef9e944e301717b938b39b4729633de2'
PREPARATION_SHA256 = 'e42f319ccd2c10176979bb6dd6b5217b1c29510814df1ec42f263adb71b832f6'


def fixed_plan_contract(plan):
    if digest(__file__) != plan['runner_sha256']:
        raise ValueError('reviewed GPU supervisor bytes differ')
    if plan['runtime_commit'] != RUNTIME_COMMIT:
        raise ValueError('preregistered original runtime required')
    if Path(plan['runtime']).resolve(strict=True) != Path(RUNTIME_ROOT).resolve(strict=True):
        raise ValueError('preregistered frozen runtime path required')
    if plan['root'] != INPUT_ROOT or plan['rows'] != 1963948:
        raise ValueError('preregistered original NVMe bank required')
    input_root = Path(INPUT_ROOT).resolve(strict=True)
    output_path = Path(plan['out']).resolve(strict=False)
    if (output_path.is_relative_to(input_root)
            or output_path.parent.stat().st_dev != input_root.stat().st_dev):
        raise ValueError('GPU output must be separate from input on the NVMe filesystem')
    if digest(plan['config']) != CONFIG_SHA256 or digest(plan['preparation_receipt']) != PREPARATION_SHA256:
        raise ValueError('preregistered original config/source receipt required')
    if digest(Path(plan['runtime']) / 'scripts/lc0_control_train.py') != plan['driver_sha256']:
        raise ValueError('frozen trainer driver digest differs')
    if digest(plan['probe']) != plan['probe_sha256']:
        raise ValueError('companion observer digest differs')
    if os.path.abspath(sys.executable) != os.path.abspath(plan['python']):
        raise ValueError('GPU supervisor and child Python interpreters differ')


class OutputBudget:
    """Account output bytes without walking the tree for each verified input.

    Call refresh at immutable phase boundaries. While an owned child writes,
    poll checks shallow output/checkpoint files each 0.5s and performs a complete
    scan at most once per 15s. Nested compiler-cache growth is sampled by the
    periodic complete scan, not protected by a filesystem quota.
    """

    def __init__(self, out, preparation_bytes, status, cheap_guard):
        self.out = out
        self.preparation_bytes = preparation_bytes
        self.status = status
        self.cheap_guard = cheap_guard
        self.next_scan = 0.0
        self.full_used = preparation_bytes
        self.shallow_at_full = 0

    def shallow_bytes(self):
        used = 0
        for base in (self.out, self.out / 'OFF', self.out / 'ON',
                     self.out / 'OFF/recovery', self.out / 'ON/recovery'):
            if not base.is_dir():
                continue
            with os.scandir(base) as entries:
                for entry in entries:
                    if entry.is_file(follow_symlinks=False):
                        try:
                            metadata = entry.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            continue  # an atomic checkpoint rename crossed this sample
                        used += max(metadata.st_size, metadata.st_blocks * 512)
        return used

    def refresh(self):
        self.cheap_guard()
        used = self.preparation_bytes
        for base, _dirs, files in os.walk(self.out, followlinks=False):
            self.cheap_guard()
            for index, name in enumerate(files):
                if index % 256 == 0:
                    self.cheap_guard()
                path = Path(base) / name
                if not path.is_symlink():
                    try:
                        metadata = path.stat()
                    except FileNotFoundError:
                        continue
                    used += max(metadata.st_size, metadata.st_blocks * 512)
                if used > 32 * GIB:
                    raise RuntimeError('32GiB combined preparation+GPU outputs cap')
        if used > 32 * GIB:
            raise RuntimeError('32GiB combined preparation+GPU outputs cap')
        self.cheap_guard()
        self.status['combined_new_disk_bytes'] = used
        self.full_used = used
        self.shallow_at_full = self.shallow_bytes()
        self.next_scan = time.monotonic() + 15.0

    def poll(self):
        self.cheap_guard()
        used = self.full_used + max(0, self.shallow_bytes() - self.shallow_at_full)
        if used > 32 * GIB:
            raise RuntimeError('32GiB sampled combined output cap')
        self.status['combined_new_disk_bytes'] = used
        if time.monotonic() >= self.next_scan:
            self.refresh()


def stop_owned_group(child):
    """Signal the owned group before reaping its leader, preventing ID reuse."""
    def send(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
            break
        time.sleep(0.05)
    send(signal.SIGKILL)
    child.wait(timeout=3)


def run_stage(command, *, cwd, env, log, guard, lease_fd):
    guard()
    child = None
    try:
        with log.open('xb') as stream:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, pass_fds=(lease_fd,))
            while os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                guard()
                time.sleep(0.5)
            guard()
    finally:
        if child is not None:
            # WNOWAIT keeps the leader unreaped, so its group ID cannot be
            # recycled before stop_owned signals the owned group.
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGALRM})
            try:
                stop_owned_group(child)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    if child is not None and child.returncode:
        raise RuntimeError(f'owned GPU child {child.pid} exit {child.returncode}')


def dependency_gate(plan):
    if digest(plan['queue']) != plan['queue_sha256']:
        raise ValueError('reviewed prerequisite queue bytes differ')
    items = json.loads(Path(plan['queue']).read_text())['items']
    matches = [item for item in items if item['id'] == plan['prerequisite_id']]
    if len(matches) != 1 or matches[0]['status'] != 'logged':
        raise ValueError('prerequisite arena is not successfully logged')
    item = matches[0]
    descriptor = plan['prerequisite_descriptor']
    if (item['command_file'] != descriptor
            or item['command_sha256'] != digest(descriptor)
            or digest(descriptor) != plan['prerequisite_descriptor_sha256']):
        raise ValueError('prerequisite descriptor binding differs')
    completion = json.loads(Path(descriptor).read_text())['completion']
    if digest(completion['path']) != plan['prerequisite_completion_sha256']:
        raise ValueError('reviewed prerequisite completion bytes differ')
    if json.loads(Path(completion['path']).read_text())[completion['status_key']] != completion['expected']:
        raise ValueError('prerequisite arena has not passed')
    terminal_path = Path(item['out']) / 'parent_outer_terminal.json'
    if digest(terminal_path) != plan['prerequisite_outer_terminal_sha256']:
        raise ValueError('reviewed prerequisite outer terminal bytes differ')
    terminal = json.loads(terminal_path.read_text())
    if terminal['returncode'] != 0:
        raise ValueError('prerequisite outer supervisor failed')


def qualified_inputs(plan):
    if (digest(plan['qualification']) != plan['qualification_sha256']
            or digest(plan['cpu_complete']) != plan['cpu_complete_sha256']):
        raise ValueError('reviewed CPU qualification receipt bytes differ')
    preparation = json.loads(Path(plan['preparation_receipt']).read_text())
    qualified = json.loads(Path(plan['qualification']).read_text())
    terminal = json.loads(Path(plan['cpu_complete']).read_text())
    if terminal['status'] != 'PASS_CPU_QUALIFICATION_NOT_GPU_ADMITTED' or terminal['qualification_sha256'] != digest(plan['qualification']):
        raise ValueError('CPU qualification completion not authenticated')
    if qualified['status'] != 'PASS_EXACT_HOST_OVERLAP_CPU_QUALIFICATION':
        raise ValueError('exact-runtime OFF/ON CPU qualification missing')
    if qualified['runtime_commit'] != plan['runtime_commit']:
        raise ValueError('CPU qualification runtime differs')
    if qualified['preparation_sha256'] != digest(plan['preparation_receipt']):
        raise ValueError('CPU qualification source binding differs')
    if preparation['status'] != 'PASS_BYTES_REQUIRES_FULL_STREAM_QUALIFICATION':
        raise ValueError('source preparation unqualified')
    if qualified['root'] != plan['root'] or qualified['config_sha256'] != digest(plan['config']):
        raise ValueError('CPU qualification root/config differs')
    if os.path.abspath(qualified['python_executable']) != os.path.abspath(plan['python']):
        raise ValueError('CPU qualification Python interpreter differs')
    for name, version in qualified['dependency_distribution_versions'].items():
        if plan['dependency_versions'].get(name) != version:
            raise ValueError(f'CPU qualification dependency differs: {name}')
    qualified_batch_count(qualified, plan['rows'])
    return preparation, qualified


def qualified_batch_count(qualified, rows):
    runs = qualified['runs']
    if len(runs) != 2 or [r['enabled'] for r in runs] != [False, True]:
        raise ValueError('need fixed OFF/ON CPU qualification')
    for run in runs:
        if run['rows'] != rows or not run['complete'] or run['same_game_repeats_max'] != 0:
            raise ValueError('CPU qualification coverage or repeat failure')
        schedule = run['batch_rows']
        if schedule != [512] * 3752 + [511] * 84:
            raise ValueError('CPU qualified batch schedule differs')
        if bool(run['host_overlap_reserve_bytes'] > 0) != run['enabled']:
            raise ValueError('CPU overlap reservation ignored')
        if run['overlap_batches_consumed'] != (3836 if run['enabled'] else 0):
            raise ValueError('CPU frozen overlap iterator consumption differs')
        expected_thread = 'exact-host' if run['enabled'] else 'MainThread'
        for key in ('producer_threads', 'preparation_threads'):
            names = run[key]
            if not names or any((not name.startswith(expected_thread) if run['enabled']
                                 else name != expected_thread) for name in names):
                raise ValueError('CPU OFF/ON producer thread path differs')
        plan, receipt = run['plan'], run['receipt']
        if (plan['rows_planned'] != rows or plan['batches_planned'] != 3836
                or plan['full_batches_planned'] != 3752 or plan['ragged_batches_planned'] != 84
                or plan['min_batch_rows_planned'] != 511 or plan['batch_size'] != 512
                or plan['seed'] != 121 or plan['max_working_set_bytes'] != 12 * GIB
                or plan['peak_working_set_bytes_planned'] > 12 * GIB
                or receipt['rows_realized'] != rows or receipt['batches_realized'] != 3836
                or receipt['peak_working_set_bytes'] > 12 * GIB
                or receipt['realized_sha256'] != plan['plan_sha256']
                or receipt['corpus_sha256'] != plan['corpus_sha256']):
            raise ValueError('CPU plan or realized receipt differs from fixed exact epoch')
    for key in ('batch_rows', 'raw_sequence_sha256', 'prepared_sequence_sha256',
                'order_sequence_sha256'):
        if runs[0][key] != runs[1][key]:
            raise ValueError('CPU OFF/ON ordered tensor/augmentation parity differs')
    return 3836


def validate_arm(summary, observation, *, rows, batches):
    sampling = summary['sampling']
    if sampling['peak_working_set_bytes'] > 12 * GIB:
        raise ValueError('sampled working set exceeds preregistered cap')
    if not sampling['complete'] or sampling['rows_realized'] != rows:
        raise ValueError('actual epoch coverage incomplete')
    if sampling['batches_realized'] != batches:
        raise ValueError('actual epoch batch count mismatch')
    if observation['status'] != 'TRAINER_RETURNED_SUCCESS':
        raise ValueError('observer did not record successful trainer return')
    if (len(observation['update_losses']) != batches
            or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in observation['update_losses'])):
        raise ValueError('missing or nonfinite individual optimizer update loss')
    if ([b['rows'] for b in observation['batches']] != [512] * 3752 + [511] * 84
            or sum(b['rows'] for b in observation['batches']) != rows):
        raise ValueError('observed batch coverage mismatch')
    windows = summary['train_window_metrics']
    if [w['steps_requested'] for w in windows] != [88] * 43 + [52]:
        raise ValueError('fixed window request schedule differs')
    if sum(w['train_steps_done'] for w in windows) != batches:
        raise ValueError('optimizer step count differs from exact epoch')
    if sum(w['train_samples_seen'] for w in windows) != rows:
        raise ValueError('trained sample count differs from exact epoch')
    if any(not math.isfinite(value) for w in windows for value in w.values()
           if isinstance(value, (int, float))):
        raise ValueError('nonfinite window metric')
    if any(w['grad_nonfinite_skip_rate'] != 0 or w['transient_cuda_retry_batches'] != 0
           or w['batches_drawn'] != w['train_steps_done'] for w in windows):
        raise ValueError('skipped or retried optimizer update')
    if len(windows) != 44 or [w['window_index'] for w in windows] != list(range(1, 45)):
        raise ValueError('expected all 44 fixed windows')
    if [w['train_samples_seen'] for w in windows] != [
        sum(b['rows'] for b in observation['batches'][i:i + 88])
        for i in range(0, batches, 88)
    ]:
        raise ValueError('window sample counts differ from observed batch sequence')
    kept = windows[1:42]
    if any(w['train_steps_done'] != 88 or w['train_samples_seen'] != 45056 for w in kept):
        raise ValueError('fixed windows2–42 are not full batches')
    seconds = sum(w['train_time_s'] for w in kept)
    samples = sum(w['train_samples_seen'] for w in kept)
    if not math.isfinite(seconds) or seconds <= 0 or samples <= 0:
        raise ValueError('invalid steady timing denominator')
    return {'steady_samples': samples, 'steady_train_seconds': seconds,
        'steady_rows_per_second': samples / seconds,
        'steady_seconds_per_update': seconds / 3608,
        'steady_prefetch_wait_seconds': sum(w['batch_prefetch_wait_s'] for w in kept),
        'full_probe_wall_seconds': observation['full_process_wall_seconds'],
        'first_batch_seconds': observation['first_batch_seconds'],
        'observer_seconds': observation['observer_seconds']}


def observed_order_digest(observation):
    ordered = hashlib.sha256()
    for batch in observation['batches']:
        ordered.update(batch['order_sha256'].encode())
    return ordered.hexdigest()


def compare(arms, *, rows, batches) -> dict[str, Any]:
    results = {name: validate_arm(value['summary'], value['observation'], rows=rows, batches=batches)
               for name, value in arms.items()}
    off, on = [arms[name]['observation'] for name in ('OFF', 'ON')]
    for key in ('initial_model_sha256', 'initial_optimizer_sha256', 'final_model_sha256', 'final_optimizer_sha256'):
        if off[key] != on[key]:
            raise ValueError(f'exact OFF/ON state parity differs: {key}')
    if [(b['rows'], b['order_sha256']) for b in off['batches']] != [(b['rows'], b['order_sha256']) for b in on['batches']]:
        raise ValueError('actual OFF/ON input order differs')
    if off['update_losses'] != on['update_losses']:
        raise ValueError('actual OFF/ON individual loss parity differs')
    for name, enabled in [('OFF', False), ('ON', True)]:
        value = arms[name]
        if value['observation']['host_batch_overlap'] is not enabled:
            raise ValueError('trainer overlap setting ignored')
        if value['observation']['overlap_batches_consumed'] != (batches if enabled else 0):
            raise ValueError('actual overlap iterator did not consume expected batches')
        reserve = value['summary']['sampling'].get('host_overlap_reserve_bytes', 0)
        if bool(reserve > 0) != enabled:
            raise ValueError('actual overlap reservation ignored')
        for metric in ('peak_rss_bytes', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'):
            if value['observation'][metric] <= 0:
                raise ValueError('missing positive memory telemetry')
    ratio = results['ON']['steady_seconds_per_update'] / results['OFF']['steady_seconds_per_update']
    wall_ratio = results['ON']['full_probe_wall_seconds'] / results['OFF']['full_probe_wall_seconds']
    return {'arms': results, 'on_over_off_steady_time': ratio,
            'on_over_off_full_wall': wall_ratio,
            'overlap_screen_pass': ratio <= 0.95 and wall_ratio <= 1.05,
            'scope': 'Fixed OFF then ON pair, exact parity required, windows2–42 preregistered'}


def authenticate_sources(preparation, root, guard):
    expected = [f'shard_{i:06d}.zarr' for i in range(len(preparation['records']))]
    if sorted(p.name for p in root.iterdir()) != expected:
        raise ValueError('NVMe source roster drift')
    for name, record in zip(expected, preparation['records'], strict=True):
        guard()
        source = Path(record['source'])
        if (root / name).resolve(strict=True) != source:
            raise ValueError('NVMe source binding drift')
        members = sorted(p for p in source.rglob('*') if p.is_file())
        if [str(p.relative_to(source)) for p in members] != sorted(record['members']):
            raise ValueError('source member roster drift')
        for index, member in enumerate(members):
            if index % 256 == 0:
                guard()
            if digest(member) != record['members'][str(member.relative_to(source))]:
                raise ValueError('qualified source bytes drift')


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
    signal.setitimer(signal.ITIMER_REAL, 592)  # bounded preflight before GPU work
    if digest(a.plan) != a.plan_sha256 or digest(a.review) != a.review_sha256:
        raise ValueError('pair plan/review pin mismatch')
    review = json.loads(a.review.read_text())
    if review['status'] != 'APPROVED' or review['plan_sha256'] != a.plan_sha256:
        raise ValueError('review does not approve exact GPU pair plan')
    plan = json.loads(a.plan.read_text())
    if plan['status'] != 'APPROVED_FOR_QUEUED_HOST_OVERLAP_PAIR':
        raise ValueError('GPU pair plan not approved')
    fixed_plan_contract(plan)
    os.sched_setaffinity(0, {14, 15})
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
        completed_arm_seconds = sum(stage.get('wall_seconds', 0.0) for stage in status['stages'])
        if arm_start is None and now - started - completed_arm_seconds >= 592:
            raise RuntimeError('10min aggregate non-training admission/report cap')
        if now - started >= 9592 or (arm_start is not None and now - arm_start >= 4492):
            raise RuntimeError('pair160min/arm75min cap with cleanup reserve')
        if (out / 'STOP').exists():
            raise RuntimeError('STOP requested')
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(mem['MemAvailable'].split()[0]) * 1024 < 32 * GIB:
            raise RuntimeError('32GiB available RAM floor')
        for root in (out, Path(plan['root'])):
            if shutil.disk_usage(root).free < 150 * GIB:
                raise RuntimeError('150GiB disk floor')
    output_budget = OutputBudget(out, int(preparation['new_bytes_upper_bound']), status, guard)
    env = dict(os.environ)
    for key in ('PYTHONOPTIMIZE', 'PYTHONHOME', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(dict.fromkeys(('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                             'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS'), '2'))
    env.update(TORCHINDUCTOR_COMPILE_THREADS='2', CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(runtime))
    root = Path(plan['root'])
    arms = {}
    lease = None
    try:
        lease = Path(plan['gpu_lock']).open('a')  # noqa: SIM115 -- closed after owned-child cleanup in finally
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output_budget.refresh()
        authenticate_sources(preparation, root, guard)
        for name in ('OFF', 'ON'):
            output_budget.refresh()
            for distribution, version in plan['dependency_versions'].items():
                if importlib.metadata.version(distribution) != version:
                    raise ValueError(f'host dependency version drift before {name}: {distribution}')
            if status['combined_new_disk_bytes'] >= 28 * GIB:
                raise RuntimeError('28GiB next-arm admission ceiling')
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
                             TRITON_CACHE_DIR=str(work / 'triton'),
                             TORCH_EXTENSIONS_DIR=str(work / 'extensions'),
                             CUDA_CACHE_PATH=str(work / 'cuda_cache'),
                             XDG_CACHE_HOME=str(work / 'cache'), TMPDIR=str(work))
            command = [plan['python'], plan['probe'], '--runtime', str(runtime),
                '--driver-sha256', plan['driver_sha256'], '--receipt', str(observation),
                '--gpu-lease-path', plan['gpu_lock'], '--gpu-lease-fd', str(lease.fileno()), '--',
                '--config', plan['config'], '--shards', str(root), '--out-dir', str(result),
                '--steps', '0', '--epochs', '1', '--sampling-mode', 'game_epoch',
                '--seed', '121', '--batch-size', '512', '--train-window-steps', '88',
                '--epoch-plan-workers', '2', '--epoch-load-workers', '2',
                '--epoch-max-working-set-gib', '12', '--allow-partial-corpus', '--allow-invalid-control']
            if name == 'ON':
                command.append('--epoch-host-batch-overlap')
            arm_start = time.monotonic()
            signal.setitimer(signal.ITIMER_REAL, max(0.001, min(4492, 9592 - (arm_start - started))))
            status['stages'].append({'name': name, 'command': command, 'status': 'STARTED'})
            (out / 'progress.json').write_text(json.dumps(status, indent=2))
            run_stage(command, cwd=work, env=child_env, log=out / f'{name}.log', guard=output_budget.poll, lease_fd=lease.fileno())
            status['stages'][-1].update(status='PASS', wall_seconds=time.monotonic() - arm_start)
            arm_start = None
            signal.setitimer(signal.ITIMER_REAL, max(0.001, min(9592 - (time.monotonic() - started),
                592 - (time.monotonic() - started - sum(stage['wall_seconds'] for stage in status['stages'])))))
            output_budget.refresh()
            arms[name] = {'summary': json.loads((result / 'summary.json').read_text()),
                          'observation': json.loads(observation.read_text())}
            arms[name]['observation']['full_process_wall_seconds'] = status['stages'][-1]['wall_seconds']
            observation = arms[name]['observation']
            if observation['final_step'] != batches or digest(result / 'checkpoint.pt') != observation['final_checkpoint_sha256']:
                raise ValueError('final checkpoint identity or update count differs')
            validate_arm(arms[name]['summary'], arms[name]['observation'], rows=expected_rows, batches=batches)
            expected_order = qualified['runs'][0 if name == 'OFF' else 1]['order_sequence_sha256']
            if observed_order_digest(observation) != expected_order:
                raise ValueError(f'{name} GPU row order differs from CPU qualification')
        authenticate_sources(preparation, root, guard)
        for name in ('OFF', 'ON'):
            if digest(out / name / 'checkpoint.pt') != arms[name]['observation']['final_checkpoint_sha256']:
                raise ValueError('final checkpoint drift')
        repin()
        status['comparison'] = compare(arms, rows=expected_rows, batches=batches)
        output_budget.refresh()
        status['status'] = ('COMPLETE_SCREEN_PASS' if status['comparison']['overlap_screen_pass']
                            else 'COMPLETE_SCREEN_FAIL_RETAIN_OFF')
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
