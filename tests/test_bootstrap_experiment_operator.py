import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location('bootstrap_operator', Path(__file__).parents[1] / 'scripts/bootstrap_experiment_operator.py')
op = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(op)


def bank(tmp_path):
    item = {'id': 'test', 'out': str(tmp_path), 'games': 2, 'status': 'running'}
    rows = [{'kind': 'game', 'pair_id': 0, 'half': i, 'a_is_white': i == 0,
             'opening_fen': 'same', 'score_candidate': .5} for i in range(2)]
    (tmp_path / 'arena.games.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    result = {'games': 2, 'pairs': 1, 'truncated': False, 'score': .5}
    return item, result


def test_complete_bank_and_truncated_rejection(tmp_path):
    item, result = bank(tmp_path)
    op.validate_arena_bank(item, result)
    result['truncated'] = True
    with pytest.raises(ValueError, match='truncated'):
        op.validate_arena_bank(item, result)


def test_summary_cannot_hide_missing_game(tmp_path):
    item, result = bank(tmp_path)
    path = tmp_path / 'arena.games.jsonl'
    path.write_text(path.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError, match='incomplete'):
        op.validate_arena_bank(item, result)


def test_duplicate_half_rejected(tmp_path):
    item, result = bank(tmp_path)
    path = tmp_path / 'arena.games.jsonl'
    path.write_text(path.read_text().splitlines()[0]+'\n'+path.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError, match='duplicate'):
        op.validate_arena_bank(item, result)


def test_deadline_includes_termination_grace(tmp_path, monkeypatch):
    monkeypatch.setattr(op, 'LOOP', tmp_path)
    monkeypatch.setattr(op.time, 'time', lambda: 100)
    op.dump(tmp_path/'STATE.json', {'deadline_unix': 170})
    assert op.launch_budget({'max_seconds': 30}) == 30
    with pytest.raises(ValueError, match='deadline'):
        op.launch_budget({'max_seconds': 31})


def test_gpu_probe_failure_blocks(monkeypatch):
    def fail():
        raise subprocess.TimeoutExpired('nvidia-smi', 10)
    monkeypatch.setattr(op, 'gpu_apps', fail)
    assert op.gpu_busy().startswith('probe_failed:')


def test_dead_wrapper_never_frees_gpu_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(op, 'LOOP', tmp_path)
    monkeypatch.setattr(op, 'pid_alive', lambda pid: False)
    monkeypatch.setattr(op, 'gpu_apps', lambda: '')
    item = {'id': 'lost', 'status': 'running', 'out': str(tmp_path)}
    (tmp_path/'parent_outer.pid').write_text('123')
    op.harvest_arena(item, {'completed': []})
    assert item['status'] == 'needs_recovery'
    op.dump(tmp_path/'queue.json', {'items': [item]})
    assert op.gpu_busy() == 'unresolved_job:lost'


def test_atomic_write_retains_old_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path/'queue.json'
    op.dump(path, {'old': True})
    def fail(*_args):
        raise OSError('injected filesystem error')
    monkeypatch.setattr(op.os, 'replace', fail)
    with pytest.raises(OSError, match="injected"):
        op.dump(path, {'new': True})
    assert json.loads(path.read_text()) == {'old': True}
    assert list(tmp_path.iterdir()) == [path]


def test_uci_incomplete_bank_rejected(tmp_path):
    item = {'out': str(tmp_path), 'games': 2}
    (tmp_path/'match.games.jsonl').write_text(json.dumps({'kind': 'game', 'game_index': 0, 'score_a': 1})+'\n')
    with pytest.raises(ValueError, match='incomplete'):
        op.validate_match_bank(item, {'wins': 2, 'draws': 0, 'losses': 0})


def test_arena_launch_caps_threads_and_persists_intent(tmp_path, monkeypatch):
    loop = tmp_path/'loop'
    loop.mkdir()
    item = {'id': 'new', 'out': str(tmp_path/'out'), 'games': 2, 'sims': 1,
            'seed': 1, 'max_seconds': 100, 'status': 'queued',
            'candidate': 'candidate.pt', 'reference': 'reference.pt'}
    monkeypatch.setattr(op, 'LOOP', loop)
    monkeypatch.setattr(op, 'RUNTIME', tmp_path)
    monkeypatch.setattr(op, 'gpu_apps', lambda: '')
    monkeypatch.setattr(op, 'mem_avail_gib', lambda: 64)
    monkeypatch.setattr(op, 'disk_free_gib', lambda: 200)
    op.dump(loop/'STATE.json', {'deadline_unix': op.time.time()+1000})
    op.dump(loop/'queue.json', {'items': [item]})
    class Child:
        pid = 12345
    def start(*_args, **_kwargs):
        assert op.load(loop/'queue.json')['items'][0]['status'] == 'launching'
        return Child()
    monkeypatch.setattr(op.subprocess, 'Popen', start)
    op.launch_arena(item)
    wrapper = (Path(item['out'])/'launch_parent.sh').read_text()
    assert 'TORCHINDUCTOR_COMPILE_THREADS=2' in wrapper
    assert 'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2' in wrapper
    assert '--kill-after=30s 100s' in wrapper
    assert item['status'] == 'running'


def test_registered_runner_and_qualification(tmp_path, monkeypatch):
    import hashlib
    import sys
    monkeypatch.setattr(op, 'LOOP', tmp_path)
    qualification = tmp_path/'qualified.json'
    spec = {'argv': [sys.executable, '-c',
        'import json,os,sys; assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"]=="2"; '
        'open(sys.argv[1],"w").write(json.dumps({"status":"PASS"}))', str(qualification)],
        'cwd': str(tmp_path), 'env': {}, 'pins': [],
        'completion': {'path': str(qualification), 'status_key': 'status', 'expected': 'PASS'}}
    descriptor = tmp_path/'registered.json'
    descriptor.write_text(json.dumps(spec))
    item = {'id': 'label', 'command_file': str(descriptor),
        'command_sha256': hashlib.sha256(descriptor.read_bytes()).hexdigest(),
        'out': str(tmp_path), 'status': 'running'}
    assert op.registered_spec(item) == spec
    op.dump(tmp_path/'command.json', {'spec': spec, 'out': str(tmp_path), 'max_seconds': 5})
    assert op.run_registered(tmp_path/'command.json') == 0
    state = {'completed': []}
    op.harvest_registered(item, state)
    assert item['status'] == 'logged'
    assert state['completed'] == ['label']
    item['status'] = 'running'
    qualification.write_text('{"status":"FAIL"}')
    op.harvest_registered(item, state)
    assert item['status'] == 'failed'
    item['command_sha256'] = 'bad'
    with pytest.raises(ValueError, match='hash mismatch'):
        op.registered_spec(item)


def test_group_killed_even_when_leader_exits(monkeypatch):
    import signal
    clock = [0.0]
    signals = []
    class ExitedLeader:
        pid = 123
        def poll(self):
            return 0
        def wait(self):
            return 0
    monkeypatch.setattr(op.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(op.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    monkeypatch.setattr(op.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    op.terminate_owned_group(ExitedLeader(), grace=.2)
    assert signals[0] == (123, signal.SIGTERM)
    assert signals[-1] == (123, signal.SIGKILL)
