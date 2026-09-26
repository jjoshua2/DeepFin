"""One fixed B100 handoff; default is inspection, --arm requires reviewed plan SHA."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(pid):
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    require(fields[0] not in ('Z', 'X'), 'terminal process')
    return {'pid': pid, 'start_ticks': int(fields[19]), 'parent': int(fields[1]),
            'cwd': str((p / 'cwd').resolve(strict=True)),
            'cmdline': (p / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')}


def pins(plan):
    for path, expected in plan['pins'].items():
        require(sha(path) == expected, f'pin changed: {path}')


def gate(plan, arena):
    owner = identity(plan['coordinator']['pid'])
    require(owner == plan['coordinator'], 'coordinator identity changed')
    state = Path(plan['state'])
    for path in plan['stop_paths']:
        require(not Path(path).exists(), f'stop/failure/terminal marker: {path}')
    # Even PID reuse remains a conservative refusal to start.
    if any(Path(f'/proc/{pid}').exists() for pid in plan['training_pids']):
        return None
    completed = state / 'training.complete.json'
    process = state / 'C20T05.s100/process.json'
    if not completed.exists() or not process.exists():
        return None
    m, t, p = read(plan['h20_manifest']), read(completed), read(process)
    require(t.get('complete') is True and t['role'] == 'H20', 'training not qualified H20')
    require(t['run'] == m['run'] and t['checkpoint']['role'] == 'H20'
            and t['checkpoint']['path'] == str(Path(m['run']) / 'checkpoint.pt'), 'wrong training run')
    require(t['schedule']['path'] == str(state / 'realized_schedule.json')
            and sha(t['schedule']['path']) == t['schedule']['sha256'], 'schedule proof changed')
    training = read(state / 'training/process.json')
    require(training.get('process_complete') is True and training['exit_code'] == 0
            and training['owner_pid'] == owner['pid'], 'training process not closed')
    require(not p.get('process_complete') and not p.get('complete'), 'arena already terminal')
    if 'arena_pid' not in p or 'supervisor_pid' not in p:
        return None
    direct = read(state / 'C20T05.s100/manifest.json')
    require(direct == read(state / 'C20T05.s100.manifest.json'), 'arena manifests differ')
    expected = {'schema': 2, 'output': str(state / 'C20T05.s100'), 'sims': 100,
                'hard_seconds': m['arena_seconds'], 'games': 1000, 'candidate': t['checkpoint'],
                'reference': dict(zip(('role', 'path', 'sha256'),
                                     (str(x) for x in arena.CHECKPOINTS['candidate']))),
                'candidate_training': {'path': str(completed), 'sha256': sha(completed)},
                'book': {'path': str(arena.BOOK), 'sha256': arena.BOOK_SHA},
                'runtime_manifest': m['runtime_manifest'], 'preregistration': m['preregistration'],
                'launcher_sha256': m['arena_launcher_sha256'], 'reader': m['reader']}
    require(direct == expected, 'unregistered first arena')
    child, supervisor = identity(p['arena_pid']), identity(p['supervisor_pid'])
    command = arena.command(direct)
    require(p['owner_pid'] == owner['pid'] and child['parent'] == supervisor['pid']
            and supervisor['parent'] == owner['pid'], 'arena ancestry differs')
    require(child['cmdline'] == p['arena_cmdline'] == p['command'] == command
            and supervisor['cmdline'] == p['supervisor_command'] == arena.timeout_command(command, 5400),
            'arena command differs')
    require(child['cwd'] == supervisor['cwd'] == p['cwd'] == str(arena.RUNTIME), 'arena cwd differs')
    return {'coordinator': owner, 'arena': child, 'supervisor': supervisor,
            'training_complete_sha256': sha(completed), 'arena_process_sha256': sha(process)}


def main():
    plan_path = HERE / 'plan.json'
    if len(sys.argv) == 1:
        print('PREPARED ONLY; --arm <plan SHA256> required')
        return
    require(len(sys.argv) == 3 and sys.argv[1] == '--arm' and sha(plan_path) == sys.argv[2], 'reviewed plan required')
    plan = read(plan_path)
    pins(plan)
    run = HERE / 'run'
    run.mkdir(exist_ok=False)  # One attempt, including a failed or interrupted wait.
    started = time.monotonic()
    status = {'pid': os.getpid(), 'started_unix': time.time(), 'plan_sha256': sha(plan_path), 'status': 'WAITING'}
    def write(name, value):
        with (run / name).open('x') as f:
            json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    try:
        write('started.json', status)
        spec = importlib.util.spec_from_file_location('fixed_h20_arena', plan['arena_script'])
        arena = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arena)
        while time.monotonic() - started < 14400:
            pins(plan)
            evidence = gate(plan, arena)
            if evidence:
                pins(plan)
                require(gate(plan, arena) is not None, 'arena ended before handoff')
                command = shlex.split(Path(plan['command_file']).read_text())
                write('handoff.json', {**status, 'status': 'EXEC_HANDOFF', 'observed_unix': time.time(),
                                      'evidence': evidence, 'command': command})
                os.chdir(plan['b100_cwd'])
                os.execv(command[0], command)
            time.sleep(min(15, max(0, 14400 - (time.monotonic() - started))))
        raise TimeoutError('four-hour wait expired without live registered arena')
    except BaseException as error:
        write('failed.json', {**status, 'status': 'NO_HANDOFF', 'ended_unix': time.time(), 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
