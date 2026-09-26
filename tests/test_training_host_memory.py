"""Host guard exercises an actual coordinator and nested owned CPU process group."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts import training_host_memory as memory


def test_available_memory_thresholds_and_missing_field(monkeypatch):
    monkeypatch.setattr(Path, 'read_text', lambda _: 'MemAvailable: 41943040 kB\n')
    assert memory.require_available(32) == 40 * 1024**3
    with pytest.raises(RuntimeError, match='48 GiB'):
        memory.require_available(48)
    monkeypatch.setattr(Path, 'read_text', lambda _: 'MemTotal: 99999999 kB\n')
    with pytest.raises(RuntimeError, match='unavailable'):
        memory.require_available(32)


def test_low_memory_allows_coordinator_nested_group_cleanup(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    script = '''
import signal,sys,time
from pathlib import Path
from scripts import bt4_direct_screen as arena
arena.RUNTIME = Path.cwd()
arena.disk_guard = lambda p: None
arena.environment = lambda gpu=False: {'CUDA_VISIBLE_DEVICES': ''}
def interrupted(*args): raise InterruptedError('host TERM')
signal.signal(signal.SIGTERM, interrupted)
try:
    arena.run_owned_stage([sys.executable, '-c', "from pathlib import Path; import time; Path('ready').touch(); time.sleep(25)"],
        Path('stage'), 35, None, 'training', {}, manifest={'fixture': True})
finally:
    time.sleep(.3)  # The time leader has already exited; group cleanup must wait.
    Path('coordinator_cleanup').touch()
'''
    child = subprocess.Popen(['/usr/bin/time', '-v', '/usr/bin/nice', '-n', '19',
                              '/usr/bin/ionice', '-c', '3', sys.executable, '-c', script], cwd=tmp_path,
                             env={**os.environ, 'PYTHONPATH': str(root), 'CUDA_VISIBLE_DEVICES': ''},
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(25)'])
    deadline = time.monotonic() + 10

    def available(_):
        if (tmp_path / 'ready').exists():
            raise RuntimeError('test low available memory')
        if time.monotonic() > deadline:
            raise RuntimeError('fixture did not start')
        return 64 * 1024**3

    monkeypatch.setattr(memory, 'require_available', available)
    try:
        with pytest.raises(RuntimeError, match='test low available memory'):
            memory.wait_guarded(child, interval=.02)
        assert child.poll() is not None
        assert not memory._group_alive(child.pid)
        assert (tmp_path / 'coordinator_cleanup').exists()
        failed = json.loads((tmp_path / 'stage/failed.json').read_text())
        assert failed['complete'] is False
        process = json.loads((tmp_path / 'stage/process.json').read_text())
        with pytest.raises(ProcessLookupError):
            os.killpg(process['supervisor_pid'], 0)
        assert unrelated.poll() is None
    finally:
        if child.poll() is None:
            os.killpg(child.pid, 9)
            child.wait()
        unrelated.terminate()
        unrelated.wait()
