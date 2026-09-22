import os
import sys
import time

import pytest
from scripts.run_packed_trainer_preparation import run_stage


@pytest.mark.parametrize('kind', ['failed_child', 'STOP', 'RAM_floor'])
def test_owned_child_failure_cleanup(tmp_path, kind):
    pidfile = tmp_path / 'pid'
    command = [sys.executable, '-c',
        'import os,time,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); '
        + ('sys.exit(7)' if kind == 'failed_child' else 'time.sleep(30)'), str(pidfile)]
    calls = 0
    def guard():
        nonlocal calls
        calls += 1
        if kind != 'failed_child' and pidfile.exists():
            raise RuntimeError(kind)
        if calls > 20:
            raise RuntimeError('test safety deadline')
    tick = time.monotonic()
    with pytest.raises(RuntimeError):
        run_stage(command, cwd=tmp_path, env=dict(os.environ), log=tmp_path / 'child.log', guard=guard)
    assert time.monotonic() - tick < 10
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert (tmp_path / 'child.log').exists()


def test_exited_leader_descendant_is_terminated(tmp_path):
    pidfile = tmp_path / 'descendant'
    code = ('import subprocess,sys; p=subprocess.Popen([sys.executable,"-c",'
            '"import time;time.sleep(30)"]); '
            'open(sys.argv[1],"w").write(str(p.pid)); sys.exit(7)')
    with pytest.raises(RuntimeError, match='exit 7'):
        run_stage([sys.executable, '-c', code, str(pidfile)], cwd=tmp_path,
                  env=dict(os.environ), log=tmp_path / 'child.log', guard=lambda: None)
    pid = int(pidfile.read_text())
    from pathlib import Path
    for _ in range(20):
        stat = Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split()[2] == 'Z':
            break
        time.sleep(0.05)
    else:
        pytest.fail('owned descendant survived exited leader cleanup')
