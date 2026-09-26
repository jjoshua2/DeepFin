"""Execute one pinned H20 continuation, preserving the original failed attempt."""
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
BASE = STATE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    for pid in plan['original_pids']:
        if Path(f'/proc/{pid}').exists():
            raise ValueError('original PID exists; inspect ownership before recovery')
    if Path(plan['output']).exists() or not Path(plan['partial']).is_dir():
        raise ValueError('expected preserved partial and absent final output')
    if (BASE / 'STOP').exists() or (STATE / 'STOP').exists():
        raise ValueError('STOP present')
    if shutil.disk_usage(BASE).free < 150 * 1024**3:
        raise ValueError('150 GiB free reserve unavailable')
    with (BASE / 'preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = STATE / 'status.json'
        record = {'schema': 1, 'stage': 'recovery', 'status': 'STARTING',
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
            with (STATE / 'recovery.log').open('x') as log:
                if stopped_at is not None:
                    raise RuntimeError(record['stop_reason'])
                child = subprocess.Popen(plan['argv'], cwd=plan['cwd'],
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=(lock.fileno(),))
                record.update(status='RUNNING', pid=child.pid)
                write()
                while child.poll() is None:
                    if (BASE / 'STOP').exists() or (STATE / 'STOP').exists():
                        stop('STOP marker')
                    if shutil.disk_usage(BASE).free < 150 * 1024**3:
                        stop('free disk below 150 GiB')
                    if stopped_at is not None:
                        signal_owned(signal.SIGKILL if time.monotonic() - stopped_at >= 30 else signal.SIGTERM)
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
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
