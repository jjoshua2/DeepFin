"""One reviewed qualification-only correction under an amended20min CPU budget."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

if __package__:
    from .run_packed_trainer_preparation import digest, inventory, run_stage
else:
    # Direct script execution has no package; its sibling is on sys.path.
    from run_packed_trainer_preparation import digest, inventory, run_stage  # pyright: ignore[reportImplicitRelativeImport]

GIB = 1024**3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--review-sha256', required=True)
    a = p.parse_args()
    started = time.monotonic()
    def interrupted(signum, _frame):
        raise RuntimeError(f'requalification signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, 1192)
    os.sched_setaffinity(0, {12, 13})
    os.nice(19)
    if digest(a.plan) != a.plan_sha256 or digest(a.review) != a.review_sha256:
        raise ValueError('plan or review pin mismatch')
    review = json.loads(a.review.read_text())
    if review['status'] != 'APPROVED' or review['plan_sha256'] != a.plan_sha256:
        raise ValueError('review does not approve this exact correction')
    plan = json.loads(a.plan.read_text())
    if plan['status'] != 'APPROVED_FOR_AMENDED_CPU_QUALIFICATION':
        raise ValueError('amended qualification is not approved')
    runtime, out = Path(plan['runtime']), Path(plan['out'])
    def repin():
        for pin in plan['pins']:
            if digest(pin['path']) != pin['sha256']:
                raise ValueError(f'input/runtime helper pin drift: {pin["path"]}')
        if inventory(runtime) != plan['runtime_files']:
            raise ValueError('runtime content/membership drift')
    repin()
    head = subprocess.check_output(['git', '-C', str(runtime), 'rev-parse', 'HEAD'], timeout=10, text=True).strip()
    if head != plan['runtime_commit']:
        raise ValueError('runtime commit mismatch')
    failed = json.loads(Path(plan['original_complete']).read_text())
    if failed['status'] != 'INCOMPLETE' or failed['seconds'] > 627:
        raise ValueError('original CPU time accounting differs')
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    status = {'status': 'INCOMPLETE', 'plan_sha256': a.plan_sha256}
    def guard():
        if time.monotonic() - started >= 1192 or (out / 'STOP').exists():
            raise RuntimeError('amended CPU deadline or STOP')
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(mem['MemAvailable'].split()[0]) * 1024 < 40 * GIB:
            raise RuntimeError('40GiB available RAM floor')
        for path in (out, Path(plan['external'])):
            if shutil.disk_usage(path).free < 150 * GIB:
                raise RuntimeError('150GiB disk floor')
        used = plan['prior_and_runtime_new_bytes']
        for path in out.rglob('*'):
            if path.is_file() and not path.is_symlink():
                used += path.stat().st_size
        if used > 10 * GIB:
            raise RuntimeError('10GiB total new bytes cap')
        status['new_disk_bytes'] = used
    env = dict(os.environ)
    for key in ('PYTHONOPTIMIZE', 'PYTHONHOME', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(dict.fromkeys(('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                             'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS'), '2'))
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(runtime))
    try:
        run_stage([plan['python'], plan['worker'], '--plan', str(a.plan)], cwd=runtime,
                  env=env, log=out / 'qualify.log', guard=guard)
        result = json.loads((out / 'qualification.json').read_text())
        if result['status'] != 'PASS_MATCHED_PACKED_ZARR_SAMPLER' or len(result['runs']) != 2:
            raise ValueError('full paired tensor qualification incomplete')
        if any(run['rows'] != plan['rows'] for run in result['runs']):
            raise ValueError('qualified rows differ')
        repin()
        status['qualification_sha256'] = digest(out / 'qualification.json')
        guard()
        status['status'] = 'PASS_CPU_QUALIFICATION_NOT_GPU_READY'
    except BaseException as error:
        status.update(status='INCOMPLETE', error=repr(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        status['seconds'] = time.monotonic() - started
        status['total_cpu_preparation_seconds'] = failed['seconds'] + status['seconds']
        (out / 'complete.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
