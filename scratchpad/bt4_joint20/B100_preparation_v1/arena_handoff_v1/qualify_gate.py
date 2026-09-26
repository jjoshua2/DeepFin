"""Disposable gate fixtures only; never calls main, exec, or an arena."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

w = load('wait_fixture', HERE / 'wait_and_exec.py')
a = load('arena_fixture', '/tmp/deepfin-bt4-hybrid-tools/scripts/bt4_direct_screen.py')
original = w.read(HERE / 'plan.json')
checks = []
with tempfile.TemporaryDirectory() as tmp:
    state = Path(tmp)
    def put(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    m = w.read(original['h20_manifest'])
    m.update(state=str(state), run=str(state / 'model'))
    put(state / 'manifest.json', m)
    put(state / 'realized_schedule.json', {'synthetic': True})
    t = {'complete': True, 'role': 'H20', 'run': m['run'],
         'checkpoint': {'role': 'H20', 'path': str(Path(m['run']) / 'checkpoint.pt'), 'sha256': 'synthetic'},
         'schedule': {'path': str(state / 'realized_schedule.json'), 'sha256': w.sha(state / 'realized_schedule.json')}}
    put(state / 'training.complete.json', t)
    owner = {'pid': 100, 'start_ticks': 42, 'cmdline': ['synthetic'], 'cwd': tmp, 'parent': 99}
    put(state / 'training/process.json', {'process_complete': True, 'exit_code': 0, 'owner_pid': 100})
    role, path, digest = a.CHECKPOINTS['candidate']
    d = {'schema': 2, 'output': str(state / 'C20T05.s100'), 'sims': 100, 'games': 1000,
         'hard_seconds': 5400, 'candidate': t['checkpoint'],
         'reference': {'role': role, 'path': str(path), 'sha256': digest},
         'candidate_training': {'path': str(state / 'training.complete.json'), 'sha256': w.sha(state / 'training.complete.json')},
         'book': {'path': str(a.BOOK), 'sha256': a.BOOK_SHA}, 'runtime_manifest': m['runtime_manifest'],
         'preregistration': m['preregistration'], 'launcher_sha256': m['arena_launcher_sha256'], 'reader': m['reader']}
    put(state / 'C20T05.s100/manifest.json', d)
    put(state / 'C20T05.s100.manifest.json', d)
    cmd = a.command(d)
    wrapped = a.timeout_command(cmd, 5400)
    p = {'owner_pid': 100, 'arena_pid': 102, 'supervisor_pid': 101, 'command': cmd,
         'arena_cmdline': cmd, 'supervisor_command': wrapped, 'cwd': str(a.RUNTIME), 'complete': False}
    put(state / 'C20T05.s100/process.json', p)
    identities = {100: owner, 101: {'pid': 101, 'parent': 100, 'cmdline': wrapped, 'cwd': str(a.RUNTIME)},
                  102: {'pid': 102, 'parent': 101, 'cmdline': cmd, 'cwd': str(a.RUNTIME)}}
    w.identity = lambda pid: identities[pid]
    plan = {'coordinator': dict(owner), 'state': tmp, 'training_pids': [],
            'stop_paths': [str(state / 'STOP')], 'h20_manifest': str(state / 'manifest.json')}
    assert w.gate(plan, a)['arena']['pid'] == 102
    checks.append('synthetic registered live-arena gate accepted; real pinned command builder')
    plan['training_pids'] = [os.getpid()]
    assert w.gate(plan, a) is None
    checks.append('still-live training PID blocks handoff')
    plan['training_pids'] = []
    def rejected(label):
        try:
            w.gate(plan, a)
        except ValueError:
            checks.append(label)
        else:
            raise AssertionError(label)
    identities[102]['parent'] = 999
    rejected('unrelated arena ancestry refused')
    identities[102]['parent'] = 101
    identities[102]['cmdline'] = ['wrong arena']
    rejected('wrong live arena command refused')
    identities[102]['cmdline'] = cmd
    identities[100]['start_ticks'] = 43
    rejected('reused coordinator PID refused')
    identities[100]['start_ticks'] = 42
    (state / 'STOP').touch()
    rejected('STOP refused')
print(json.dumps({'checks': checks, 'runner_sha256': w.sha(HERE / 'wait_and_exec.py'),
                  'scope': 'Synthetic receipts/process identities; actual command constructor; no main/exec or live launch'}, indent=2))
