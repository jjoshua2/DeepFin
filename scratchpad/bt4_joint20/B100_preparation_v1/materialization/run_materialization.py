"""Execute one pinned B100 policy materialization; never train or retry."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

STATE = Path(__file__).resolve().parent
BASE = STATE.parents[1] / 'hybrid_endpoint_run01'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_bytes(plan):
    """Sample only new output metadata; subprocess.run kills/reaps a timed-out du."""
    paths = [plan[k] for k in ('output', 'partial') if os.path.lexists(plan[k])]
    if not paths:
        return 0
    for path in paths:
        if Path(path).is_symlink():
            raise ValueError('output path became a symlink')
    result = subprocess.run(['/usr/bin/du', '-s', '-B1', '--', *paths],
                            capture_output=True, text=True, timeout=20, check=True)
    return sum(int(line.split()[0]) for line in result.stdout.splitlines())


def verify_publication(plan):
    out = Path(plan['output'])
    if os.path.lexists(plan['partial']) or out.is_symlink() or not out.is_dir():
        raise ValueError('expected final publication and no partial')
    mix = json.loads((out / 'bt4_policy_mix_summary.json').read_text())
    expected = {'rows': 18910484, 'shards': 2309, 'kind': 'global',
                'alpha': 1.0, 'bt4_temperature': 0.5}
    if any(mix.get(k) != v for k, v in expected.items()):
        raise ValueError('final B100 summary mismatch')
    derive = json.loads((out / 'derive_targets_summary.json').read_text())
    if len(derive['shards']) != 2309 or sum(s['rows'] for s in derive['shards']) != 18910484:
        raise ValueError('final derived row inventory mismatch')


def main():
    plan_path = STATE / 'launch.json'
    if len(sys.argv) != 2 or sha(plan_path) != sys.argv[1]:
        raise ValueError('supply the exact reviewed launch SHA256')
    plan = json.loads(plan_path.read_text())
    for path, expected in plan['pins'].items():
        if sha(path) != expected:
            raise ValueError(f'changed pinned input: {path}')
    if sha(__file__) != plan['supervisor_sha256']:
        raise ValueError('supervisor changed')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=plan['cwd'], text=True).strip() != plan['commit']:
        raise ValueError('frozen checkout HEAD changed')
    def fresh():
        if any(os.path.lexists(plan[k]) for k in ('output', 'partial')):
            raise ValueError('B100 output or partial already exists; no retry')
    fresh()
    if any(p.exists() for p in (BASE / 'STOP', STATE.parent / 'STOP', STATE / 'STOP')):
        raise ValueError('STOP present')
    if shutil.disk_usage(BASE).free < 150 * 1024**3:
        raise ValueError('150 GiB free reserve unavailable')
    with (BASE / 'preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fresh()
        for path, expected in plan['pins'].items():
            if sha(path) != expected:
                raise ValueError(f'changed pinned input after lease: {path}')
        if any((STATE / n).exists() for n in ('status.json', 'mix.log', 'resources.json')):
            raise ValueError('prior materialization receipt exists; no retry')
        status_path = STATE / 'status.json'
        record = {'schema': 1, 'stage': 'B100_materialization', 'status': 'STARTING',
                  'plan_sha256': sha(plan_path), 'supervisor_sha256': sha(__file__),
                  'started_unix': time.time(), 'supervisor_pid': os.getpid(),
                  'argv': plan['argv']}
        with status_path.open('x') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')

        def write():
            temporary = status_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(record, indent=2) + '\n')
            temporary.replace(status_path)

        child = None
        stopped_at = None
        last_size_sample = time.monotonic()

        def signal_owned(sig):
            if child is not None:
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass

        def stop(reason):
            nonlocal stopped_at
            if stopped_at is None:
                stopped_at = time.monotonic()
                record['stop_reason'] = reason
                signal_owned(signal.SIGTERM)

        signal.signal(signal.SIGTERM, lambda *_: stop('supervisor SIGTERM'))
        signal.signal(signal.SIGINT, lambda *_: stop('supervisor SIGINT'))
        try:
            with (STATE / 'mix.log').open('x') as log:
                if stopped_at is not None:
                    raise RuntimeError(record['stop_reason'])
                if any(p.exists() for p in (BASE / 'STOP', STATE.parent / 'STOP', STATE / 'STOP')):
                    raise RuntimeError('STOP present after lease')
                if shutil.disk_usage(BASE).free < 150 * 1024**3:
                    raise RuntimeError('free disk below reserve after lease')
                child = subprocess.Popen(plan['argv'], cwd=plan['cwd'],
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=(lock.fileno(),))
                record.update(status='RUNNING', pid=child.pid)
                write()
                while child.poll() is None:
                    if any(p.exists() for p in (BASE / 'STOP', STATE.parent / 'STOP', STATE / 'STOP')):
                        stop('STOP marker')
                    if shutil.disk_usage(BASE).free < 150 * 1024**3:
                        stop('free disk below 150 GiB')
                    if stopped_at is None and time.monotonic() - last_size_sample >= 300:
                        try:
                            size = output_bytes(plan)
                            record['last_output_allocated_bytes'] = size
                            record['last_output_sample_unix'] = time.time()
                            if size > 32 * 1024**3:
                                stop('provisional output allocation exceeds 32 GiB')
                            write()
                        except Exception as error:
                            stop(f'output size sampling failed: {error!r}')
                        last_size_sample = time.monotonic()
                    if stopped_at is not None:
                        signal_owned(signal.SIGKILL if time.monotonic() - stopped_at >= 30 else signal.SIGTERM)
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                if child.returncode == 0 and stopped_at is None:
                    verify_publication(plan)
                    record['output_summary_sha256'] = sha(Path(plan['output']) / 'bt4_policy_mix_summary.json')
                    record['derive_summary_sha256'] = sha(Path(plan['output']) / 'derive_targets_summary.json')
                    record['final_output_allocated_bytes'] = output_bytes(plan)
                    if record['final_output_allocated_bytes'] > 32 * 1024**3:
                        raise ValueError('published output exceeds provisional 32 GiB cap; retained')
                record.update(status='COMPLETE' if child.returncode == 0 and stopped_at is None else 'FAILED_OR_STOPPED',
                              returncode=child.returncode, completed_unix=time.time())
                write()
        except BaseException as error:
            record.update(status='FAILED_EXCEPTION', error=repr(error), completed_unix=time.time())
            write()
            raise
        finally:
            if child is not None and child.poll() is None:
                signal_owned(signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    signal_owned(signal.SIGKILL)
                    child.wait()
            # A terminal timeout leader can leave descendants in its owned group.
            signal_owned(signal.SIGKILL)
        print(json.dumps(record), flush=True)
        return 0 if record['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
