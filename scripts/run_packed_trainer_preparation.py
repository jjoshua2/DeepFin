"""Hard-bounded CPU-only pack+full-stream qualification; never launches training."""
from __future__ import annotations
import argparse
from collections.abc import Generator
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
from typing import Any

GIB = 1024**3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    root = root.resolve()
    result = {}
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != '.git')
        for name in sorted(files + [d for d in dirs if (Path(base) / d).is_symlink()]):
            path = Path(base) / name
            if path.name == '.git':
                continue
            rel = str(path.relative_to(root))
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(root):
                    raise ValueError(f'external runtime symlink: {path}')
                result[rel] = {'symlink': os.readlink(path)}
            else:
                result[rel] = {'sha256': digest(path)}
    return result


@contextmanager
def _defer_cleanup_stops() -> Generator[None]:
    """Defer raising stop callbacks until the owned group is killed/reaped.

    pthread_sigmask protects only its caller: another existing unblocked thread
    can receive a process-directed stop and schedule its Python callback on main.
    No signal is rerouted. Keep original dispositions and redeliver the first
    observed stop after cleanup, coalescing standard signals into at most3keys.
    Python handlers only execute on main, so background cleanup needs no trap.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
    pending: dict[int, Any] = {}
    def deferred(signum, frame):
        pending.setdefault(signum, frame)
    try:
        for sig, handler in handlers.items():
            if handler != signal.SIG_IGN:
                signal.signal(sig, deferred)
        yield
    finally:
        restoration_error: BaseException | None = None
        for sig, handler in handlers.items():
            try:
                signal.signal(sig, handler)
            except BaseException as exc:
                # A restored stop callback can raise during later restoration.
                # Restore every disposition before propagating that cancellation.
                if restoration_error is None:
                    restoration_error = exc
        if restoration_error is not None:
            raise restoration_error
        if pending:
            signum, frame = next(iter(pending.items()))
            handler = handlers[signal.Signals(signum)]
            if callable(handler):
                handler(signum, frame)
            elif handler == signal.SIG_DFL:
                os.kill(os.getpid(), signum)  # original process disposition, after cleanup


def stop_owned(child):
    # A reaped leader can leave descendants in its owned session/process group.
    def send(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
    with _defer_cleanup_stops():
        try:
            send(signal.SIGTERM)
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        finally:
            try:
                send(signal.SIGKILL)
            finally:
                child.wait(timeout=3)


def run_stage(command, *, cwd, env, log, guard):
    guard()
    child = None
    try:
        with log.open('xb') as stream:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                guard()
                time.sleep(0.5)
            if child.returncode:
                raise RuntimeError(f'owned child {child.pid} exit {child.returncode}')
            guard()
    finally:
        if child is not None:
            stop_owned(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--review-sha256', required=True)
    a = parser.parse_args()
    start = time.monotonic()
    def interrupted(signum, _frame):
        raise RuntimeError(f'supervisor received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, 1192)
    if digest(a.review) != a.review_sha256:
        raise ValueError('review receipt digest mismatch')
    review = json.loads(a.review.read_text())
    if review['status'] != 'APPROVED' or review['plan_sha256'] != a.plan_sha256:
        raise ValueError('review does not approve this exact supervisor plan')
    if digest(a.plan) != a.plan_sha256:
        raise ValueError('supervisor plan digest mismatch')
    plan = json.loads(a.plan.read_text())
    if plan['status'] != 'APPROVED_FOR_CPU_PREPARATION':
        raise ValueError('plan is not approved for CPU preparation')
    runtime = Path(plan['runtime'])
    head = subprocess.check_output(['git', '-C', str(runtime), 'rev-parse', 'HEAD'], timeout=10, text=True).strip()
    if head != plan['runtime_commit']:
        raise ValueError('runtime commit mismatch')
    if inventory(runtime) != plan['runtime_files']:
        raise ValueError('runtime content or roster drift')
    for pin in plan['pins']:
        if digest(pin['path']) != pin['sha256']:
            raise ValueError(f'pin drift: {pin["path"]}')
    selection = Path(plan['selection_plan'])
    data = json.loads(selection.read_text())
    control = Path(plan['control'])
    out, external = Path(data['output']), Path(data['external'])
    for root in (control, out, external):
        if root.exists() or root.is_symlink():
            raise FileExistsError(root)
    control.mkdir(parents=True)
    os.sched_setaffinity(0, {12, 13})
    os.nice(19)
    status: dict[str, Any] = {'status': 'INCOMPLETE', 'plan_sha256': a.plan_sha256, 'stages': []}

    def guard():
        if time.monotonic() - start >= 1192:
            raise RuntimeError('20min aggregate CPU deadline (cleanup reserved)')
        if (control / 'STOP').exists() or (out / 'STOP').exists():
            raise RuntimeError('STOP requested')
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(mem['MemAvailable'].split()[0]) * 1024 < 40 * GIB:
            raise RuntimeError('40GiB available RAM floor')
        for root in (control, external.parent):
            if shutil.disk_usage(root).free < 150 * GIB:
                raise RuntimeError('150GiB disk floor')
        used = plan['reserved_new_bytes']
        for root in (control, out, external):
            for base, _dirs, files in os.walk(root, followlinks=False):
                for name in files:
                    path = Path(base) / name
                    if not path.is_symlink():
                        used += path.stat().st_size
        if used > 10 * GIB:
            raise RuntimeError('10GiB aggregate new disk cap')
        status['new_disk_bytes'] = used

    env = dict(os.environ)
    for key in ('PYTHONOPTIMIZE', 'PYTHONHOME', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(dict.fromkeys(('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS'), '2'))
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(runtime))
    prep = [plan['python'], plan['preparer'], '--cohort-plan', data['cohort_plan'],
        '--cohort-plan-sha256', data['cohort_plan_sha256'], '--selection-plan', str(selection),
        '--selection-plan-sha256', digest(selection), '--reuse', data['reuse'],
        '--out', str(out), '--external', str(external), '--prepare']
    qualification = [plan['python'], plan['qualifier'], '--runtime', str(runtime),
        '--directory', str(out / 'directory'), '--packed', str(out / 'packed'),
        '--result', str(control / 'qualification.json')]
    try:
        for name, command in [('pack', prep), ('qualify', qualification)]:
            tick = time.monotonic()
            status['stages'].append({'name': name, 'command': command, 'status': 'STARTED'})
            (control / 'progress.json').write_text(json.dumps(status, indent=2))
            run_stage(command, cwd=runtime, env=env, log=control / f'{name}.log', guard=guard)
            status['stages'][-1].update(status='PASS', seconds=time.monotonic() - tick)
        receipt = json.loads((control / 'qualification.json').read_text())
        if receipt['status'] != 'PASS_MATCHED_PACKED_ZARR_SAMPLER':
            raise ValueError('full stream qualification did not pass')
        if len(receipt['runs']) != 2 or any(run['rows'] != data['rows'] for run in receipt['runs']):
            raise ValueError('qualification rows differ from immutable selection')
        guard()
        if inventory(runtime) != plan['runtime_files']:
            raise ValueError('runtime drift during preparation')
        for pin in plan['pins']:
            if digest(pin['path']) != pin['sha256']:
                raise ValueError('helper or input pin drift during preparation')
        guard()
        status['qualification_sha256'] = digest(control / 'qualification.json')
        guard()
        status['status'] = 'PASS_CPU_PREPARATION_NOT_GPU_READY'
    except BaseException as error:
        status['status'] = 'INCOMPLETE'
        status['error'] = repr(error)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        status['seconds'] = time.monotonic() - start
        (control / 'complete.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
