#!/usr/bin/env python3
"""Run one explicitly registered checkpoint comparison with a pinned manifest.

Legacy C/G and E0T05/C profiles coexist with fixed H20/B100/G50 development cells.

No training, resume, automatic retry or promotion. GNU timeout inherits the GPU
lease and survives coordinator death. SIGKILL can leave an incomplete receipt;
reconcile that receipt only after the timeout/arena group has stopped.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(os.environ.get('CHESS_EXPERIMENT_ROOT', Path.home() / 'projects/chess')).expanduser().resolve()
RUNTIME = ROOT / '.dev/worktree/wise-cloud'
HEAD = '7ec261509fb7345cf1ca0ad73809193fc2749bb1'
READER = Path('/tmp/deepfin-bt4-prior-one/scripts/bt4_joint_readout.py')
READER_SHA = 'bdb6b2e2a2dc04c087b7ae36628211015aaba50ac96c4110a5f372c56cefd79f'
BOOK = ROOT / 'data/opening_books/8moves_v3_plus_policybeam_final145cp_plus_uho2024_060_110_plus_2move_thinbeam_dedup.pgn.zip'
QUALIFIED_READOUT = ROOT / 'scratchpad/bt4_joint20/sf_close_run02/completed_s100_readout.json'
QUALIFIED_READOUT_SHA = '353d95c0f94cad1c1be1f9b23f6449d8cad50c6a41523c2b9ab5415f5cd6c12e'
BOOK_SHA = '70d0dfa50a6b1191f1db702a093a4911b7433c612a30599517c2c2d92f0cea7c'
CHECKPOINTS = {
    'candidate': ('C20T05', ROOT / 'runs/armB/qtemp_0.0005_hist_20m_bt4_sfclose_C20T05_epoch_v1/checkpoint.pt',
                  '8a355d29f7d3eee5deec4b3a16a6625d23baebe1302e3a8f2dea9136939e1db3'),
    'reference': ('G20T05', ROOT / 'runs/armB/qtemp_0.0005_hist_20m_bt4_global_G20T05_epoch_v2/checkpoint.pt',
                  'bd8c208a95247373f423be0649329e9c100db64ab1b5a5e68fd7a6aec3769a74'),
}
# Fixed development profiles; each cell is (reference role, simulations, games).
REGISTERED_COMPARISONS = {
    'H20': (('C20T05', 100, 1000), ('G20T05', 100, 1000), ('C20T05', 400, 500)),
    'B100': (('C20T05', 100, 1000), ('C20T05', 400, 500)),
    'G50': (('C20T05', 100, 1000),),
}
EXTENDED_READER = Path(__file__).resolve().with_name('bt4_joint_readout.py')
RESERVE = 150 * 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    temp.replace(path)


def pin(path, digest):
    require(sha(path) == digest, f'changed identity: {path}')


def reader_identity(m):
    return (Path(m['reader']['path']), m['reader']['sha256']) if m['candidate']['role'] in REGISTERED_COMPARISONS else (READER, READER_SHA)


def validate(m):
    keys = {'schema', 'output', 'sims', 'hard_seconds', 'candidate', 'reference',
            'book', 'runtime_manifest', 'preregistration', 'launcher_sha256'}
    role = m.get('candidate', {}).get('role')
    registered = role in REGISTERED_COMPARISONS
    trained = registered or role == 'E0T05'
    extra = {'candidate_training'} if trained else set()
    if registered:
        extra |= {'reader', 'games'}
        require(m['schema'] == 2, 'registered development cells require schema2')
        require(type(m['games']) is int and (m['reference']['role'], m['sims'], m['games']) in REGISTERED_COMPARISONS[role], 'unregistered candidate/reference/budget cell')
        require(m['hard_seconds'] == 5400, 'registered arena cap is 5400 seconds')
        require(m['reader']['path'] == str(EXTENDED_READER) and set(m['reader']) == {'path', 'sha256'}, 'wrong extended reader')
        if m['games'] == 500:
            extra |= {'opening_anchor'}
            require(set(m['opening_anchor']) == {'bank', 'completion'}, 'probe needs completed C100 opening anchor')
            for item in m['opening_anchor'].values():
                require(set(item) == {'path', 'sha256'} and Path(item['path']).is_absolute(), 'invalid opening anchor pin')
    else:
        require(m['schema'] == 1, 'legacy manifest requires schema1')
    require(set(m) == keys | extra, 'manifest keys differ')
    require(type(m['sims']) is int and m['sims'] in (100, 400), 'select 100 or 400 simulations')
    cap = m['hard_seconds']
    require(type(cap) in (int, float) and math.isfinite(cap) and cap > 30, 'hard_seconds must exceed 30s cleanup allowance')
    output = Path(m['output'])
    require(output.is_absolute() and output == output.resolve(), 'output must be canonical absolute path')
    require(not output.exists() and not output.is_symlink(), 'output must be new; no adoption/resume')
    inputs = [RUNTIME, reader_identity(m)[0], BOOK, Path(__file__).resolve()]
    if trained:
        controls = {value[0]: value for value in CHECKPOINTS.values()}
        reference = m['reference']['role']
        require(reference in controls and (registered or reference == 'C20T05'), 'wrong C reference or unregistered reference')
        ref_role, path, digest = controls[reference]
        require(m['reference'] == {'role': ref_role, 'path': str(path), 'sha256': digest}, 'wrong C reference or G reference')
        require(set(m['candidate']) == {'role', 'path', 'sha256'}, 'invalid trained candidate identity')
        candidate = Path(m['candidate']['path'])
        require(candidate.is_absolute() and candidate == candidate.resolve() and candidate.name == 'checkpoint.pt',
                'candidate must be a canonical completed checkpoint')
        require(registered or m['sims'] == 100, 'E0T05 profile is registered only at 100 simulations')
        require(set(m['candidate_training']) == {'path', 'sha256'}, 'missing candidate training receipt')
        inputs += [candidate, path, Path(m['candidate_training']['path']).resolve()]
    else:
        for side, (known_role, path, digest) in CHECKPOINTS.items():
            require(m[side] == {'role': known_role, 'path': str(path), 'sha256': digest}, f'wrong {side} checkpoint')
            inputs.append(path)
    require(m['book'] == {'path': str(BOOK), 'sha256': BOOK_SHA}, 'use original seed42 book')
    for key in ('runtime_manifest', 'preregistration'):
        item = m[key]
        require(set(item) == {'path', 'sha256'} and Path(item['path']).is_absolute(), f'invalid {key}')
        inputs.append(Path(item['path']).resolve())
    if 'opening_anchor' in m:
        inputs.extend(Path(item['path']).resolve() for item in m['opening_anchor'].values())
    require(all(output != p and output not in p.parents and p not in output.parents for p in inputs), 'input/output overlap')


def environment(gpu=False):
    inherited = {k: v for k, v in os.environ.items() if k != 'CHESS_LIVE_PRODUCTION_CONFIG'}
    return {**inherited, 'PYTHONPATH': str(RUNTIME), 'CUDA_VISIBLE_DEVICES': '0' if gpu else '',
            'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
            'OPENBLAS_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2', 'BLOSC_NTHREADS': '2',
            'CHESS_ANTI_ENGINE_LIVE_CONFIG': str(RUNTIME / 'configs/pbt2_small.yaml')}


def qualified_search():
    pin(QUALIFIED_READOUT, QUALIFIED_READOUT_SHA)
    settings = read(QUALIFIED_READOUT)['cells']['C20T05:100']['settings']
    require(settings['search_candidate'] == settings['search_reference'], 'qualified search sides differ')
    return settings['search_candidate']


def check_pins(m):
    pin(__file__, m['launcher_sha256'])
    pin(*reader_identity(m))
    qualified_search()
    for key in ('candidate', 'reference', 'book', 'runtime_manifest', 'preregistration'):
        pin(m[key]['path'], m[key]['sha256'])
    if 'candidate_training' in m:
        verify_candidate_training(m)
    if 'opening_anchor' in m:
        for item in m['opening_anchor'].values():
            pin(item['path'], item['sha256'])
        anchor = read(m['opening_anchor']['completion']['path'])
        require(anchor['complete'] is True and anchor['games_sha256'] == m['opening_anchor']['bank']['sha256'], 'opening anchor incomplete or bank differs')
    return runtime_identity(m['runtime_manifest'])


def verify_candidate_training(m):
    evidence = m['candidate_training']
    pin(evidence['path'], evidence['sha256'])
    receipt = read(evidence['path'])
    require(receipt['complete'] is True and receipt['role'] == m['candidate']['role'], 'candidate training incomplete')
    require(receipt['checkpoint'] == m['candidate'], 'candidate checkpoint differs from completed training')
    run = Path(m['candidate']['path']).parent
    require(receipt['run'] == str(run), 'candidate run differs')
    pin(run / 'summary.json', receipt['summary_sha256'])
    pin(receipt['schedule']['path'], receipt['schedule']['sha256'])
    require(receipt['canonical_plan_sha256'] == 'dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f',
            'candidate schedule differs from registered source/C')


def runtime_identity(identity):
    pin(identity['path'], identity['sha256'])
    frozen = read(identity['path'])
    require(frozen['identities']['heads'][str(RUNTIME)] == HEAD, 'unqualified runtime manifest')
    require(subprocess.check_output(['git', '-C', str(RUNTIME), 'rev-parse', 'HEAD'], text=True).strip() == HEAD, 'runtime moved')
    require(not subprocess.check_output(['git', '-C', str(RUNTIME), 'status', '--porcelain', '--untracked-files=no'], text=True).strip(), 'runtime tracked edits')
    rt = frozen['runtime']
    modules = {'chess_anti_engine.encoding._features_ext', 'chess_anti_engine.encoding._lc0_ext',
               'chess_anti_engine.mcts._mcts_tree', 'chess_anti_engine.nnue._nnue_ext'}
    require(set(rt['native_extensions']) == modules and set(rt['native_extension_sha256']) == set(rt['native_extensions'].values()), 'missing native pins')
    require(rt['torch'] == '2.11.0+cu128' and rt['cuda'] == '12.8' and rt['numpy'] == '1.26.2'
            and rt['python'].startswith('3.10.12'), 'unqualified runtime versions')
    for path, digest in rt['native_extension_sha256'].items():
        pin(path, digest)
    return rt


def runtime_probe(rt):
    code = f"""import contextlib,json,sys
with contextlib.redirect_stdout(sys.stderr):
    import importlib,torch,numpy
    from scripts.arena_standard import apply_search_overrides,resolve_search_shape
    modules={list(rt['native_extensions'])!r}
    actual=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
        cuda=torch.version.cuda,numpy=numpy.__version__,
        search=apply_search_overrides(resolve_search_shape('training'),spec='policy_temp=1.0').as_record(),
        native_extensions={{m:importlib.import_module(m).__file__ for m in modules}})
print(json.dumps(actual))
"""
    actual = json.loads(subprocess.check_output([rt['executable'], '-c', code], cwd=RUNTIME,
                                               env=environment(), text=True, timeout=60))
    require(actual.pop('search') == qualified_search(), 'resolved search differs from qualified C100')
    require(actual == {k: v for k, v in rt.items() if k != 'native_extension_sha256'}, 'actual runtime differs')
    return actual


def command(m):
    python = read(m['runtime_manifest']['path'])['runtime']['executable']
    out = Path(m['output'])
    return [python, 'scripts/arena_standard.py', '--candidate', m['candidate']['path'],
            '--reference', m['reference']['path'], '--games', str(m.get('games', 1000)), '--mode', 'matched_sims',
            '--search-shape', 'training', '--cand-gumbel', 'policy_temp=1.0', '--ref-gumbel', 'policy_temp=1.0',
            '--sims', str(m['sims']), '--seed', '42', '--openings', str(BOOK), '--opening-plies', '16',
            '--max-plies', '300', '--temperature', '0.1', '--max-concurrent-games', '128',
            '--eval-max-batch', '4096', '--compile', 'on', '--no-rolling', '--label',
            f"{m['candidate']['role']}_vs_{m['reference']['role']}",
            '--max-seconds', str(m['hard_seconds'] - 30), '--games-out', str(out / 'arena.games.jsonl'),
            '--out', str(out / 'arena.results.jsonl')]


def timeout_command(cmd, seconds):
    # timeout is its own process-group leader; without --foreground it signals
    # the entire owned group, including descendants. KILL follows TERM by 30s.
    return ['/usr/bin/timeout', '--signal=TERM', '--kill-after=30s', f'{seconds - 30}s', *cmd]


def cleanup(child):
    for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
    require(child.poll() is not None, 'owned supervisor could not be reaped')


def check_cell(m, cell):
    require(cell['settings']['candidate'] == m['candidate']['path'], 'candidate path differs')
    require(cell['settings']['openings'] == m['book']['path'], 'book path differs')
    require(cell['execution'] == ['on', '4096'], 'execution differs from qualified compile/batch')
    expected = qualified_search()
    require(all(cell['settings'].get(side) == expected for side in ('search_candidate', 'search_reference')),
            'realized search differs from qualified C100')
    games = m.get('games', 1000)
    if m['candidate']['role'] == 'E0T05' or m['candidate']['role'] in REGISTERED_COMPARISONS:
        require(cell['raw_game_rows'] == games and cell['superseded_orphan_rows'] == 0, 'trained candidate requires no orphan rows')
    require(cell['result']['games'] == games and cell['result']['pairs'] == games // 2, 'incomplete arena')


def read_bank(m):
    reader, digest = reader_identity(m)
    pin(reader, digest)
    # Load only the role-independent reader; build_report assumes E0 for sf-close.
    sys.path.insert(0, str(RUNTIME))
    spec = importlib.util.spec_from_file_location('qualified_bt4_reader', reader)
    if spec is None or spec.loader is None:
        raise ValueError('qualified reader cannot be loaded')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = {'expected_pairs': m['games'] // 2} if m['candidate']['role'] in REGISTERED_COMPARISONS else {}
    cell = module.read_arm(Path(m['output']) / 'arena.games.jsonl', reference=Path(m['reference']['path']),
                           seed=42, sims=m['sims'], prior_temperature=1.0, **options)
    check_cell(m, cell)
    if 'opening_anchor' in m:
        anchor = m['opening_anchor']
        for item in anchor.values():
            pin(item['path'], item['sha256'])
        completion = read(anchor['completion']['path'])
        require(completion['complete'] is True and completion['games_sha256'] == anchor['bank']['sha256'], 'opening anchor incomplete')
        baseline = module.read_arm(Path(anchor['bank']['path']), reference=CHECKPOINTS['candidate'][1], seed=42, sims=100)
        check_cell({**m, 'sims': 100, 'games': 1000}, baseline)
        require(cell['openings'] == baseline['openings'][:250], 'probe openings differ from first250 C100 pairs')
    return {'candidate_role': m['candidate']['role'], 'reference_role': m['reference']['role'], 'match_complete': True,
            **{k: v for k, v in cell.items() if k not in ('scores', 'openings')},
            'promotion': 'NONE; exploratory direct checkpoint screen'}


def disk_guard(output):
    for path in (ROOT, output.parent):
        require(shutil.disk_usage(path).free >= RESERVE, f'disk below 150 GiB reserve: {path}')


def acquire_gpu_lease(lease):
    # One queued attempt. The cap and output state start only after acquisition.
    fcntl.flock(lease, fcntl.LOCK_EX)


def run_owned_stage(cmd, out, seconds, lease_fd, stage, metadata, *, manifest, stop_paths=(), cwd=None, env=None) -> dict[str, Any]:
    """One bounded process group; caller owns the inherited GPU lease and pin checks."""
    require((cwd is None) == (env is None), 'explicit stage cwd and env must be supplied together')
    stage_cwd = RUNTIME if cwd is None else Path(cwd)
    stage_env = environment(lease_fd is not None) if env is None else dict(env)
    require(stage_cwd.is_absolute() and stage_cwd.is_dir(), 'invalid stage cwd')
    require(not any(p.exists() for p in (out / 'STOP', *stop_paths)), 'stop requested before stage')
    out.mkdir(exist_ok=False)
    write(out / 'manifest.json', manifest)
    wrapped = timeout_command(cmd, seconds)
    started = time.monotonic()
    receipt = {**metadata, 'complete': False, 'owner_pid': os.getpid(), 'started_unix': time.time(),
               'command': cmd, 'supervisor_command': wrapped, 'cwd': str(stage_cwd), 'hard_seconds': seconds}
    child = None
    try:
        write(out / 'process.json', receipt)
        with (out / f'{stage}.log').open('x') as log:
            child = subprocess.Popen(wrapped, cwd=stage_cwd, env=stage_env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(() if lease_fd is None else (lease_fd,)))
            receipt['supervisor_pid'] = child.pid
            write(out / 'process.json', receipt)
            while child.poll() is None:
                if f'{stage}_pid' not in receipt and time.monotonic() - started < 10:
                    children = Path(f'/proc/{child.pid}/task/{child.pid}/children')
                    ids = children.read_text().split() if children.exists() else []
                    if ids:
                        try:
                            cmdline = Path(f'/proc/{ids[0]}/cmdline').read_bytes().decode().strip('\0').split('\0')
                        except FileNotFoundError:
                            pass
                        else:
                            if cmdline != wrapped:
                                receipt[f'{stage}_pid'] = int(ids[0])
                                receipt[f'{stage}_cmdline'] = cmdline
                                write(out / 'process.json', receipt)
                            elif cmdline == wrapped and f'{stage}_preexec_cmdline' not in receipt:
                                # A forked timeout child can still expose its parent's
                                # argv before exec. Preserve it without calling it the
                                # observed workload; sample again within the window.
                                receipt[f'{stage}_preexec_cmdline'] = cmdline
                                receipt[f'{stage}_preexec_pid'] = int(ids[0])
                                write(out / 'process.json', receipt)
                disk_guard(out)
                require(not any(p.exists() for p in (out / 'STOP', *stop_paths)), 'stop requested')
                time.sleep(1)
            require(child.returncode == 0, f'{stage}/timeout exited {child.returncode}')
        cleanup(child)
        elapsed = time.monotonic() - started
        receipt.update(ended_unix=time.time(), stage_seconds=elapsed,
                       gpu_seconds=elapsed if lease_fd is not None else 0.0,
                       exit_code=child.returncode, process_complete=True)
        write(out / 'process.json', receipt)
        return receipt
    except BaseException as error:
        if child is not None:
            cleanup(child)
        write(out / 'failed.json', {**receipt, 'complete': False, 'ended_unix': time.time(),
                                   'stage_seconds': time.monotonic() - started,
                                   'gpu_seconds': time.monotonic() - started if lease_fd is not None else 0.0,
                                   'error': str(error)})
        raise


def execute(m, *, stop_paths=()):
    validate(m)
    require(not any(p.exists() for p in stop_paths), 'stop requested before arena')
    out = Path(m['output'])
    require(out.parent.is_dir(), 'create output parent before launch')
    disk_guard(out)
    rt = check_pins(m)
    actual = runtime_probe(rt)
    with (ROOT / 'scratchpad/gpu0_experiment.lock').open('a') as lease:
        acquire_gpu_lease(lease)
        require(not any(p.exists() for p in stop_paths), 'stop requested while waiting for arena')
        require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                            text=True, timeout=10).strip(), 'competing GPU process; no launch')
        check_pins(m)
        disk_guard(out)
        receipt = run_owned_stage(command(m), out, m['hard_seconds'], lease.fileno(), 'arena',
            {'runtime': actual, 'models': {k: m[k] for k in ('candidate', 'reference')}, 'book': m['book'],
             'qualified_search_readout_sha256': QUALIFIED_READOUT_SHA,
             'live_config': environment()['CHESS_ANTI_ENGINE_LIVE_CONFIG'], 'launcher_sha256': m['launcher_sha256']}, manifest=m, stop_paths=stop_paths)
    # The GPU lease is released for CPU-only readout; no subsequent GPU stage here.
    try:
        check_pins(m)
        result = subprocess.check_output([rt['executable'], str(Path(__file__).resolve()), '--read-bank',
                                          str(out / 'manifest.json')], cwd=RUNTIME,
                                         env=environment(), text=True, timeout=120)
        report = json.loads(result)
        require(report['match_complete'] is True, 'incomplete bank is not a result')
        write(out / 'readout.json', report)
        write(out / 'complete.json', {**receipt, 'complete': True, 'games_sha256': sha(out / 'arena.games.jsonl'),
                                     'readout_sha256': sha(out / 'readout.json'), 'reader_sha256': reader_identity(m)[1]})
    except BaseException as error:
        write(out / 'failed.json', {**receipt, 'complete': False, 'error': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--read-bank', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.read_bank:
        with contextlib.redirect_stdout(sys.stderr):
            report = read_bank(read(args.read_bank))
        print(json.dumps(report, allow_nan=False))
        return
    require(args.manifest is not None, '--manifest required')
    m = read(args.manifest)
    validate(m)
    if not args.execute:
        print(json.dumps({'execute': False, 'command': timeout_command(command(m), m['hard_seconds'])}, indent=2))
        return
    def interrupted(signum, _frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    execute(m)


if __name__ == '__main__':
    main()
