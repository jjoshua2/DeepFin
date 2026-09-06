#!/usr/bin/env python3
"""Plan one C20T05 versus G20T05 arena; --execute requires an explicit pinned manifest.

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

ROOT = Path('/home/josh/projects/chess')
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


def validate(m):
    require(set(m) == {'schema', 'output', 'sims', 'hard_seconds', 'candidate', 'reference',
                       'book', 'runtime_manifest', 'preregistration', 'launcher_sha256'}, 'manifest keys differ')
    require(m['schema'] == 1 and type(m['sims']) is int and m['sims'] in (100, 400), 'select 100 or 400 simulations')
    cap = m['hard_seconds']
    require(type(cap) in (int, float) and math.isfinite(cap) and cap > 30, 'hard_seconds must exceed 30s cleanup allowance')
    output = Path(m['output'])
    require(output.is_absolute() and output == output.resolve(), 'output must be canonical absolute path')
    require(not output.exists() and not output.is_symlink(), 'output must be new; no adoption/resume')
    inputs = [RUNTIME, READER, BOOK, Path(__file__).resolve()]
    for side, (role, path, digest) in CHECKPOINTS.items():
        require(m[side] == {'role': role, 'path': str(path), 'sha256': digest}, f'wrong {side} checkpoint')
        inputs.append(path)
    require(m['book'] == {'path': str(BOOK), 'sha256': BOOK_SHA}, 'use original seed42 book')
    for key in ('runtime_manifest', 'preregistration'):
        item = m[key]
        require(set(item) == {'path', 'sha256'} and Path(item['path']).is_absolute(), f'invalid {key}')
        inputs.append(Path(item['path']).resolve())
    require(all(output != p and output not in p.parents and p not in output.parents for p in inputs), 'input/output overlap')


def environment(gpu=False):
    return {**os.environ, 'PYTHONPATH': str(RUNTIME), 'CUDA_VISIBLE_DEVICES': '0' if gpu else '',
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
    pin(READER, READER_SHA)
    qualified_search()
    for key in ('candidate', 'reference', 'book', 'runtime_manifest', 'preregistration'):
        pin(m[key]['path'], m[key]['sha256'])
    frozen = read(m['runtime_manifest']['path'])
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
            '--reference', m['reference']['path'], '--games', '1000', '--mode', 'matched_sims',
            '--search-shape', 'training', '--cand-gumbel', 'policy_temp=1.0', '--ref-gumbel', 'policy_temp=1.0',
            '--sims', str(m['sims']), '--seed', '42', '--openings', str(BOOK), '--opening-plies', '16',
            '--max-plies', '300', '--temperature', '0.1', '--max-concurrent-games', '128',
            '--eval-max-batch', '4096', '--compile', 'on', '--no-rolling', '--label', 'C20T05_vs_G20T05',
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
    require(cell['result']['games'] == 1000 and cell['result']['pairs'] == 500, 'incomplete arena')


def read_bank(m):
    pin(READER, READER_SHA)
    # Load only the role-independent reader; build_report assumes E0 for sf-close.
    sys.path.insert(0, str(RUNTIME))
    spec = importlib.util.spec_from_file_location('qualified_bt4_reader', READER)
    if spec is None or spec.loader is None:
        raise ValueError('qualified reader cannot be loaded')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cell = module.read_arm(Path(m['output']) / 'arena.games.jsonl', reference=Path(m['reference']['path']),
                           seed=42, sims=m['sims'], prior_temperature=1.0)
    check_cell(m, cell)
    return {'candidate_role': 'C20T05', 'reference_role': 'G20T05', 'match_complete': True,
            **{k: v for k, v in cell.items() if k not in ('scores', 'openings')},
            'promotion': 'NONE; exploratory direct checkpoint screen'}


def disk_guard(output):
    for path in (ROOT, output.parent):
        require(shutil.disk_usage(path).free >= RESERVE, f'disk below 150 GiB reserve: {path}')


def acquire_gpu_lease(lease):
    # One queued attempt. The cap and output state start only after acquisition.
    fcntl.flock(lease, fcntl.LOCK_EX)


def execute(m):
    validate(m)
    out = Path(m['output'])
    require(out.parent.is_dir(), 'create output parent before launch')
    disk_guard(out)
    rt = check_pins(m)
    actual = runtime_probe(rt)
    with (ROOT / 'scratchpad/gpu0_experiment.lock').open('a') as lease:
        acquire_gpu_lease(lease)
        require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                            text=True, timeout=10).strip(), 'competing GPU process; no launch')
        check_pins(m)
        disk_guard(out)
        out.mkdir(exist_ok=False)
        write(out / 'manifest.json', m)
        cmd = command(m)
        wrapped = timeout_command(cmd, m['hard_seconds'])
        started = time.monotonic()
        receipt = {'complete': False, 'owner_pid': os.getpid(), 'started_unix': time.time(),
                   'command': cmd, 'supervisor_command': wrapped, 'cwd': str(RUNTIME), 'runtime': actual,
                   'models': {k: m[k] for k in ('candidate', 'reference')}, 'book': m['book'],
                   'hard_seconds': m['hard_seconds'], 'qualified_search_readout_sha256': QUALIFIED_READOUT_SHA,
                   'live_config': environment()['CHESS_ANTI_ENGINE_LIVE_CONFIG'], 'launcher_sha256': m['launcher_sha256']}
        child = None
        try:
            write(out / 'process.json', receipt)
            with (out / 'arena.log').open('x') as log:
                child = subprocess.Popen(wrapped, cwd=RUNTIME, env=environment(True), stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lease.fileno(),))
                receipt['supervisor_pid'] = child.pid
                write(out / 'process.json', receipt)
                while child.poll() is None:
                    if 'arena_pid' not in receipt:
                        children = Path(f'/proc/{child.pid}/task/{child.pid}/children')
                        ids = children.read_text().split() if children.exists() else []
                        if ids:
                            receipt['arena_pid'] = int(ids[0])
                            receipt['arena_cmdline'] = Path(f'/proc/{ids[0]}/cmdline').read_bytes().decode().strip('\0').split('\0')
                            write(out / 'process.json', receipt)
                    disk_guard(out)
                    require(not (out / 'STOP').exists(), 'stop requested')
                    time.sleep(1)
                require(child.returncode == 0, f'arena/timeout exited {child.returncode}')
            cleanup(child)
            receipt.update(ended_unix=time.time(), gpu_seconds=time.monotonic() - started,
                           exit_code=child.returncode, process_complete=True)
            write(out / 'process.json', receipt)
            check_pins(m)
            result = subprocess.check_output([rt['executable'], str(Path(__file__).resolve()), '--read-bank',
                                              str(out / 'manifest.json')], cwd=RUNTIME,
                                             env=environment(), text=True, timeout=120)
            report = json.loads(result)
            require(report['match_complete'] is True, 'incomplete bank is not a result')
            write(out / 'readout.json', report)
            write(out / 'complete.json', {**receipt, 'complete': True, 'games_sha256': sha(out / 'arena.games.jsonl'),
                                         'readout_sha256': sha(out / 'readout.json'), 'reader_sha256': READER_SHA})
        except BaseException as error:
            if child is not None:
                cleanup(child)
            write(out / 'failed.json', {**receipt, 'complete': False, 'ended_unix': time.time(),
                                       'gpu_seconds': time.monotonic() - started, 'error': str(error)})
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
