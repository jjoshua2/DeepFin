"""CPU-only operational failures; no checkpoint loading or GPU work."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/bt4_direct_screen.py'
spec = importlib.util.spec_from_file_location('bt4_direct_screen_test', SCRIPT)
assert spec is not None
assert spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def manifest(tmp_path) -> dict[str, Any]:
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps({'runtime': {'executable': sys.executable}}))
    return {'schema': 1, 'output': str(tmp_path / 'new'), 'sims': 100, 'hard_seconds': 3600,
            'candidate': dict(zip(('role', 'path', 'sha256'),
                                 (str(x) for x in launcher.CHECKPOINTS['candidate']))),
            'reference': dict(zip(('role', 'path', 'sha256'),
                                 (str(x) for x in launcher.CHECKPOINTS['reference']))),
            'book': {'path': str(launcher.BOOK), 'sha256': launcher.BOOK_SHA},
            'runtime_manifest': {'path': str(runtime), 'sha256': launcher.sha(runtime)},
            'preregistration': {'path': str(tmp_path / 'prereg.md'), 'sha256': 'pending'},
            'launcher_sha256': launcher.sha(SCRIPT)}


def test_protocol_and_new_output(tmp_path):
    m = manifest(tmp_path)
    launcher.validate(m)
    cmd = launcher.command(m)
    for flag, expected in (('--candidate', m['candidate']['path']), ('--reference', m['reference']['path']),
                           ('--sims', '100'), ('--seed', '42'), ('--openings', str(launcher.BOOK)),
                           ('--compile', 'on'), ('--eval-max-batch', '4096')):
        assert cmd[cmd.index(flag) + 1] == expected
    Path(m['output']).mkdir()
    with pytest.raises(ValueError, match='must be new'):
        launcher.validate(m)


@pytest.mark.parametrize('change', ['candidate', 'book', 'execution', 'incomplete', 'search'])
def test_extra_reader_contract_rejects_mismatch(tmp_path, monkeypatch, change):
    m = manifest(tmp_path)
    shape = {'shape': 'training', 'source': 'fixture', 'gumbel': {'topk': 16, 'c_scale': .1, 'policy_temp': 1.0},
             'vloss_weight': 1, 'target_batch': 0, 'tree_reuse': 'cold'}
    qualified = tmp_path / 'qualified.json'
    qualified.write_text(json.dumps({'cells': {'C20T05:100': {'settings': {
        'search_candidate': shape, 'search_reference': shape}}}}))
    monkeypatch.setattr(launcher, 'QUALIFIED_READOUT', qualified)
    monkeypatch.setattr(launcher, 'QUALIFIED_READOUT_SHA', launcher.sha(qualified))
    cell: dict[str, Any] = {'settings': {'candidate': m['candidate']['path'], 'openings': m['book']['path']},
            'execution': ['on', '4096'], 'result': {'games': 1000, 'pairs': 500}}
    for side in ('search_candidate', 'search_reference'):
        cell['settings'][side] = launcher.qualified_search()
    launcher.check_cell(m, cell)
    if change == 'candidate':
        cell['settings']['candidate'] = m['reference']['path']
    elif change == 'book':
        cell['settings']['openings'] = '/fresh_confirmation_reserved'
    elif change == 'execution':
        cell['execution'][1] = '2048'
    elif change == 'search':
        # Both sides drift identically: matching each other is insufficient.
        for side in ('search_candidate', 'search_reference'):
            cell['settings'][side]['gumbel']['topk'] = 32
    else:
        cell['result']['pairs'] = 499
    with pytest.raises(ValueError, match=r"differs|incomplete"):
        launcher.check_cell(m, cell)


def test_pin_rejects_mutated_bytes(tmp_path):
    file = tmp_path / 'model'
    file.write_bytes(b'original')
    pin = launcher.sha(file)
    file.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed identity'):
        launcher.pin(file, pin)


def test_timeout_survives_coordinator_death_and_holds_lease(tmp_path):
    """Real processes prove inherited flock and surviving deadline, without GPU."""
    lease_path = tmp_path / 'gpu.lock'
    ready = tmp_path / 'ready.json'
    code = '''import fcntl,json,os,subprocess,sys,time
lease=open(sys.argv[1], 'a')
fcntl.flock(lease,fcntl.LOCK_EX)
child=subprocess.Popen(json.loads(sys.argv[3]),
                       start_new_session=True,pass_fds=(lease.fileno(),))
open(sys.argv[2],'w').write(json.dumps({'supervisor':child.pid}))
time.sleep(60)
'''
    wrapped = launcher.timeout_command([sys.executable, '-c', 'import time; time.sleep(60)'], 32)
    coordinator = subprocess.Popen([sys.executable, '-c', code, str(lease_path), str(ready), json.dumps(wrapped)])
    supervisor = None
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        supervisor = json.loads(ready.read_text())['supervisor']
        coordinator.kill()
        coordinator.wait(timeout=2)
        with lease_path.open('a') as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            deadline = time.monotonic() + 5
            while True:
                try:
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    assert time.monotonic() < deadline, 'surviving timeout did not release GPU lease'
                    time.sleep(.05)
    finally:
        if coordinator.poll() is None:
            coordinator.kill()
            coordinator.wait(timeout=2)
        if supervisor is not None:
            try:
                os.killpg(supervisor, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_cleanup_only_owned_group():
    owned = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        launcher.cleanup(owned)
        assert owned.poll() is not None
        assert unrelated.poll() is None
    finally:
        launcher.cleanup(unrelated)


def test_child_environment_overrides_ambient_live_config(monkeypatch):
    monkeypatch.setenv('CHESS_ANTI_ENGINE_LIVE_CONFIG', '/unrelated/live.yaml')
    monkeypatch.setenv('BLOSC_NTHREADS', '64')
    code = 'import os,json; print(json.dumps({k:os.environ[k] for k in ("CHESS_ANTI_ENGINE_LIVE_CONFIG","BLOSC_NTHREADS")}))'
    result = json.loads(subprocess.check_output([sys.executable, '-c', code],
                                                env=launcher.environment(True), text=True))
    assert result == {'CHESS_ANTI_ENGINE_LIVE_CONFIG': str(launcher.RUNTIME / 'configs/pbt2_small.yaml'),
                      'BLOSC_NTHREADS': '2'}


def test_read_bank_stdout_is_only_json(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    def noisy_reader(_manifest):
        print('[shape] diagnostic')
        return {'match_complete': True}
    monkeypatch.setattr(launcher, 'read_bank', noisy_reader)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--read-bank', str(path)])
    launcher.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {'match_complete': True}
    assert '[shape] diagnostic' in captured.err


def test_queued_lease_waits_for_owner_release(tmp_path):
    lease_path = tmp_path / 'lease'
    ready, acquired = tmp_path / 'ready', tmp_path / 'acquired'
    code = """import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('launcher',sys.argv[1])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open(sys.argv[2],'a') as lease:
    Path(sys.argv[3]).touch()
    module.acquire_gpu_lease(lease)
    Path(sys.argv[4]).touch()
"""
    with lease_path.open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX)
        child = subprocess.Popen([sys.executable, '-c', code, str(SCRIPT), str(lease_path), str(ready), str(acquired)])
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            assert ready.exists()
            time.sleep(.1)
            assert child.poll() is None
            assert not acquired.exists()
            fcntl.flock(owner, fcntl.LOCK_UN)
            assert child.wait(timeout=5) == 0
            assert acquired.exists()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)
