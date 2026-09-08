#!/usr/bin/env python3
"""Fixed CPU pilot. --freeze prepares binding only; --execute requires its SHA."""
import argparse
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
REG = STATE / 'preregistration.json'
AUDIT = [Path('/home/josh/projects/chess/data/audit_set_v1.jsonl'),
         Path('/home/josh/projects/chess/scratchpad/audit_d9_labels/audit_d9_labels.jsonl'),
         Path('/home/josh/projects/chess/data/lc0/bt4_audit_cache_topk256_20260817.jsonl')]
AUDIT_SHA = ['d8e26efa0b010450abf9374693afc45027db6d146571785ab897af5061144df2',
             '0f56bdc0aa453b6dbfdad5cf1744e4937b3eeae274f6b1051d683dd4e8aa4f64',
             '622cdfeda7d71c211e57719ba4d0807252934e6c46d0acfadcc098c251168294']
BOOT = "import runpy,sys; from numcodecs import blosc; blosc.set_nthreads(2); p=sys.argv.pop(1); sys.argv[0]=p; runpy.run_path(p,run_name='__main__')"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(cwd, *args):
    return subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()


def identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def verify(plan):
    require(git(plan['checkout'], 'rev-parse', 'HEAD') == plan['commit'], 'checkout revision changed')
    require(not git(plan['checkout'], 'diff', '--name-only', 'HEAD'), 'tracked runtime changed')
    for path, expected in plan['pins'].items():
        require(sha(path) == expected, f'pin changed: {path}')


def usage():
    total = 0
    for root, dirs, files in os.walk(STATE):
        for name in [*dirs, *files]:
            p = Path(root) / name
            require(not p.is_symlink(), f'pilot output symlink: {p}')
        total += sum((Path(root) / name).stat().st_size for name in files)
    return total


def guard():
    require(not (STATE / 'STOP').exists(), 'STOP requested')
    require(usage() < 8 * 1024**3, '8GiB pilot output/cache cap reached')
    require(shutil.disk_usage(STATE).free >= 150 * 1024**3, '150GiB disk reserve reached')


def freeze(args):
    require(args.checkout and args.commit and args.python and args.runtime_pins, 'freeze requires all explicit runtime bindings')
    cwd = Path(args.checkout).resolve()
    require(git(cwd, 'rev-parse', 'HEAD') == args.commit and len(args.commit) == 40, 'need final merged exact checkout commit')
    require(not git(cwd, 'diff', '--name-only', 'HEAD'), 'dirty tracked checkout')
    # Parent supplies reviewed transitive native/interpreter/dependency pins; no guessed runtime.
    runtime = read(args.runtime_pins)
    require(runtime['status'] == 'qualified' and runtime['checkout'] == str(cwd)
            and runtime['commit'] == args.commit and runtime['python'] == args.python, 'runtime qualification binding differs')
    pins = dict(runtime['pins'])
    require(str(Path(args.python).resolve()) in pins, 'resolved interpreter must be pinned')
    require(any(p.endswith('.so') for p in pins), 'native runtime pins required')
    pins.update({str(p): sha(p) for p in [REG, Path(__file__).resolve(), Path(args.runtime_pins).resolve(), Path('/usr/bin/timeout')]})
    for p, expected in zip(AUDIT, AUDIT_SHA):
        require(sha(p) == expected, 'frozen audit bank changed')
        pins[str(p)] = expected
    for source in read(REG)['sources']:
        for key in ('source_manifest', 'receipt_snapshot'):
            item = source[key]
            pins[item['path']] = item['sha256']
        attrs = Path(source['sidecar_dir']) / 'w00-00000.bt4.zarr' / '.zattrs'
        pins[str(attrs)] = sha(attrs)
    plan = {'schema': 1, 'status': 'PREPARED_NOT_LAUNCHED', 'checkout': str(cwd), 'commit': args.commit,
            'python': args.python, 'pins': pins, 'registration': str(REG),
            'command_builder_sha256': sha(__file__), 'audit': 'same bank reanalysis, descriptive, pilot record binding',
            'limits': read(REG)['resource_limits']}
    verify(plan)
    write(STATE / 'launch.json', plan)
    print(sha(STATE / 'launch.json'))


def command(plan, script, *args):
    return [plan['python'], '-c', BOOT, str(Path(plan['checkout']) / 'scripts' / script), *map(str, args)]


def stage(name, argv, plan):
    guard()
    log = STATE / (name + '.log')
    start = time.time()
    with log.open('xb') as out:
        child = subprocess.Popen(argv, cwd=plan['checkout'], stdout=out, stderr=subprocess.STDOUT)
        try:
            write(STATE / (name + '.started.json'), {'pid': child.pid, 'argv': argv, 'start_unix': start})
            while child.poll() is None:
                guard()
                time.sleep(1)
            require(child.returncode == 0, f'{name} exit {child.returncode}')
            guard()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    write(STATE / (name + '.completed.json'), {'pid': child.pid, 'argv': argv, 'start_unix': start,
                                             'end_unix': time.time(), 'exit_code': child.returncode})


def source_check(source):
    from scripts import derive_corpus_targets as derive
    record = derive.read_corpus_record(Path(source['source_dir']))
    require(record.shards[0].name == source['source_shard'], 'registered source is not first resolved shard')
    require(int(derive.shard_row_counts(record)[0]) == source['raw_rows'], 'first shard row claim differs')
    require(sha(record.shards[0]) == source['source_sha256_recorded'], 'registered raw payload changed')


def compare(source):
    import numpy as np
    import zarr
    from numcodecs import blosc
    from scripts import corpus_row_provenance as provenance
    blosc.set_nthreads(2)
    original = Path(source['derived_output'])
    summary = read(original / 'derive_targets_summary.json')
    paths = sorted(original.glob('shard_*.zarr'))
    names = {path.name for path in paths}
    require(names == {Path(item['path']).name for item in summary['shards']}
            and len(paths) == len(summary['shards']), 'source shard inventory differs')
    for key in ('C_output', 'H_output'):
        require(names == {p.name for p in Path(source[key]).glob('shard_*.zarr')},
                f'{key} shard inventory differs')
    rows = 0
    digest = hashlib.sha256()
    for path in paths:
        group = zarr.open_group(str(path), mode='r')
        n = int(group['x'].shape[0])
        refs = provenance.read(path / provenance.FILENAME, rows=n)
        for ref in refs:
            require(ref['source_dir'] == source['source_dir'] and ref['source_shard'] == source['source_shard'], 'foreign source row')
            digest.update(json.dumps(ref, sort_keys=True).encode())
        for key in ('C_output', 'H_output'):
            other_path = Path(source[key]) / path.name
            other = zarr.open_group(str(other_path), mode='r')
            require(set(group.array_keys()) == set(other.array_keys()), 'non-policy array set differs')
            require(sha(path / provenance.FILENAME) == sha(other_path / provenance.FILENAME), 'source-qualified input schedule differs')
            for field in group.array_keys():
                if field == 'policy_target':
                    continue
                require(group[field].shape == other[field].shape and group[field].dtype == other[field].dtype,
                        f'non-policy shape/dtype differs: {key}/{path.name}/{field}')
                for start in range(0, n, 256):
                    require(np.array_equal(group[field][start:start+256], other[field][start:start+256], equal_nan=True),
                            f'non-policy payload changed: {key}/{path.name}/{field}')
        rows += n
    require(rows == summary['realized']['rows_written'] and 0 < rows <= source['raw_rows'], 'emitted row total differs')
    write(Path(source['derived_output']).parent / 'all_row_qualification.json',
          {'status': 'complete', 'rows': rows, 'physical_rows': source['raw_rows'],
           'excluded_rows': source['raw_rows'] - rows, 'derive_realized': summary['realized'],
           'source_qualified_input_schedule_sha256': digest.hexdigest(),
           'scope': 'All non-policy payloads and original-source row sequence; not training schedule execution or strength.'})


def worker(plan):
    verify(plan)
    reg = read(REG)
    raw_ids = {}
    for source in reg['sources']:
        for field in ('derived_output', 'adapted_output', 'rank_output', 'C_output', 'H_output'):
            p = Path(source[field])
            require(not p.exists() and not Path(str(p) + '.writing').exists(), 'existing output/partial refused')
        raw = Path(source['source_dir']) / source['source_shard']
        raw_ids[str(raw)] = identity(raw)
        stage(source['source_id'] + '.source_check', [plan['python'], __file__, '--source-check', source['source_id']], plan)
    for label, scope, alpha in [('C', 'sf-cp-window', '1'), ('H', 'c20-global', '0.2')]:
        stage('audit_' + label, command(plan, 'bt4_policy_mix.py', 'audit', '--audit-set', AUDIT[0], '--d9-labels', AUDIT[1],
              '--bt4-cache', AUDIT[2], '--json', STATE / ('audit_' + label + '.json'), '--boot', '10000', '--seed', '20260903',
              '--alpha', alpha, '--scope', scope, '--bt4-temperature', '0.5', '--sf-rank-cap', '3', '--sf-cp-window', '20',
              '--sf-audit-mode', 'descriptive', '--experiment-record', REG), plan)
    for source in reg['sources']:
        name = source['source_id']
        parent = Path(source['derived_output']).parent
        parent.mkdir(exist_ok=True)
        stage(name + '.derive', command(plan, 'derive_corpus_targets.py', '--corpus', source['source_dir'], '--out', source['derived_output'],
              '--limit', source['raw_rows'], *reg['derive_common_args']), plan)
        summary_path = Path(source['derived_output']) / 'derive_targets_summary.json'
        summary = read(summary_path)
        rows, shards = summary['realized']['rows_written'], len(summary['shards'])
        receipt = json.loads(Path(source['receipt_snapshot']['path']).read_text())
        attrs = read(Path(source['sidecar_dir']) / receipt['sidecar'] / '.zattrs')
        mapping = {'schema': 1, 'derived_summary': {'path': str(summary_path), 'sha256': sha(summary_path)},
                   'teacher': {'onnx': {'path': attrs['onnx_path'], 'sha256': receipt['onnx_sha256']},
                               'policy_output': receipt['policy_output'], 'providers': receipt['providers'], 'remap': receipt['remap_provenance']},
                   'sources': [{'source_dir': source['source_dir'], 'sidecar_dir': source['sidecar_dir'],
                                'manifest': source['source_manifest'], 'receipts': source['receipt_snapshot']}]}
        mapping_path = parent / 'adapter_manifest.json'
        write(mapping_path, mapping)
        stage(name + '.adapt', command(plan, 'adapt_raw_bt4_sidecars.py', '--manifest', mapping_path, '--expected-manifest-sha256', sha(mapping_path),
              '--out', source['adapted_output'], '--max-index-bytes', reg['resource_limits']['adapter_index_cache_bytes']), plan)
        common = ['--expected-rows', rows, '--expected-shards', shards, '--expected-source-summary-sha256', sha(summary_path)]
        stage(name + '.rank', command(plan, 'sf_d9_rank_sidecar.py', '--raw', source['source_dir'], '--shards', source['derived_output'],
              '--out', source['rank_output'], '--limit', source['raw_rows'], '--top-k', '3', '--seed', '0', '--rows-per-shard', '8192',
              '--max-provenance-cache-bytes', '16777216', *common), plan)
        for label, scope, alpha in [('C', 'sf-cp-window', '1'), ('H', 'c20-global', '0.2')]:
            extra = []
            if label == 'H':
                c = Path(source['C_output'])
                extra = ['--c20-parent', c, '--expected-c20-summary-sha256', sha(c / 'derive_targets_summary.json'),
                         '--expected-c20-mix-sha256', sha(c / 'bt4_policy_mix_summary.json')]
            stage(name + '.' + label, command(plan, 'bt4_policy_mix.py', 'mix', '--shards', source['derived_output'], '--sidecar', source['adapted_output'],
                  '--out', source[label + '_output'], '--sf-rank-sidecar', source['rank_output'], '--alpha', alpha, '--scope', scope,
                  '--bt4-temperature', '0.5', '--sf-rank-cap', '3', '--sf-cp-window', '20', '--audit-receipt', STATE / ('audit_' + label + '.json'),
                  '--sf-audit-mode', 'descriptive', '--experiment-record', REG, *common, *extra), plan)
        stage(name + '.compare', [plan['python'], __file__, '--compare', name], plan)
    verify(plan)
    for source in reg['sources']:
        raw = Path(source['source_dir']) / source['source_shard']
        require(identity(raw) == raw_ids[str(raw)] and sha(raw) == source['source_sha256_recorded'], 'raw source changed during pilot')
    guard()
    write(STATE / 'worker_complete.json', {'status': 'complete', 'end_unix': time.time(), 'raw_storage_identities': raw_ids})


def main():
    parser = argparse.ArgumentParser()
    for key in ('checkout', 'commit', 'python', 'runtime-pins', 'source-check', 'compare', 'expected-plan-sha256'):
        parser.add_argument('--' + key)
    for key in ('freeze', 'execute', 'worker'):
        parser.add_argument('--' + key, action='store_true')
    args = parser.parse_args()
    if args.freeze:
        return freeze(args)
    if args.source_check or args.compare:
        source = next(s for s in read(REG)['sources'] if s['source_id'] == (args.source_check or args.compare))
        return source_check(source) if args.source_check else compare(source)
    require(args.execute or args.worker, 'Choose --freeze or --execute; default does not run')
    require(sha(STATE / 'launch.json') == args.expected_plan_sha256, 'explicit launch plan SHA required')
    plan = read(STATE / 'launch.json')
    if args.worker:
        return worker(plan)
    start = time.time()
    guard()
    require(not (STATE / 'started.json').exists(), 'single attempt already started')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=plan['checkout'],
               CHESS_ANTI_ENGINE_LIVE_CONFIG=str(Path(plan['checkout']) / 'configs/pbt2_small.yaml'))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS'):
        env[key] = '2'
    remaining = 1770 - (time.time() - start)
    require(remaining > 0, 'preflight exhausted wall cap')
    argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=30s', f'{remaining:.3f}s', '/usr/bin/nice', '-n', '19',
            '/usr/bin/ionice', '-c', '3', '/usr/bin/taskset', '-c', '0,1', plan['python'], __file__, '--worker',
            '--expected-plan-sha256', args.expected_plan_sha256]
    with (STATE / 'worker.log').open('xb') as out:
        child = subprocess.Popen(argv, env=env, cwd=plan['checkout'], stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            write(STATE / 'started.json', {'pid': child.pid, 'start_unix': start, 'argv': argv, 'plan_sha256': args.expected_plan_sha256})
            code = child.wait()
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        finally:
            if child.poll() is not None:
                # Only this newly owned supervisor group; remove any surviving failed-stage descendants.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    complete = code == 0 and (STATE / 'worker_complete.json').exists() and time.time() - start <= 1800
    write(STATE / ('completed.json' if complete else 'failed.json'),
          {'status': 'complete' if complete else 'failed', 'exit_code': code, 'start_unix': start,
           'end_unix': time.time(), 'plan_sha256': args.expected_plan_sha256})
    require(complete, 'pilot failed; preserve all logs/partials, no retry')


if __name__ == '__main__':
    main()
