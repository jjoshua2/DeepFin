#!/usr/bin/env python3
"""Fixed 64-shard common-data batch. --freeze prepares binding only; --execute requires its SHA."""
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
            require(not p.is_symlink(), f'common-batch output symlink: {p}')
        total += sum((Path(root) / name).stat().st_size for name in files)
    return total


def guard():
    require(not (STATE / 'STOP').exists(), 'STOP requested')
    require(usage() < 8 * 1024**3, '8GiB common-batch output/cache cap reached')
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
    pins[str(Path('/usr/bin/time'))] = sha('/usr/bin/time')
    pins[str(STATE / 'python_bootstrap/sitecustomize.py')] = sha(STATE / 'python_bootstrap/sitecustomize.py')
    pins[read(REG)['assessment']['path']] = read(REG)['assessment']['sha256']
    for source in read(REG)['sources']:
        for key in ('source_manifest', 'receipt_snapshot'):
            item = source[key]
            pins[item['path']] = item['sha256']
        for entry in source['selection']:
            attrs = Path(source['sidecar_dir']) / entry['source_shard'].replace('.jsonl.zst', '.bt4.zarr') / '.zattrs'
            pins[str(attrs)] = sha(attrs)
    plan = {'schema': 1, 'status': 'PREPARED_NOT_LAUNCHED', 'checkout': str(cwd), 'commit': args.commit,
            'python': args.python, 'pins': pins, 'registration': str(REG),
            'command_builder_sha256': sha(__file__), 'scope': 'derive/adapt/rank only; no audit, recipe mixing or training',
            'limits': read(REG)['resource_limits']}
    verify(plan)
    write(STATE / 'launch.json', plan)
    print(sha(STATE / 'launch.json'))


def command(plan, script, *args):
    return [plan['python'], str(Path(plan['checkout']) / 'scripts' / script), *map(str, args)]


def descendants(pid):
    found = {}
    pending = [pid]
    while pending:
        current = pending.pop()
        try:
            children = Path(f'/proc/{current}/task/{current}/children').read_text().split()
        except FileNotFoundError:
            continue
        for item in children:
            child = int(item)
            if child in found:
                continue
            try:
                argv = Path(f'/proc/{child}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            except FileNotFoundError:
                continue
            found[child] = argv
            pending.append(child)
    return found


def stage(name, argv, plan):
    guard()
    log = STATE / (name + '.log')
    metrics_path = STATE / (name + '.time.json')
    require(not metrics_path.exists(), 'existing timing refused')
    timed_argv = ['/usr/bin/time', '-o', str(metrics_path), '-f',
                  '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M,"filesystem_inputs":%I,"filesystem_outputs":%O,"exit_code":%x}', *argv]
    start = time.time()
    observed = {}
    with log.open('xb') as out:
        child = subprocess.Popen(timed_argv, cwd=plan['checkout'], stdout=out, stderr=subprocess.STDOUT)
        try:
            write(STATE / (name + '.started.json'), {'pid': child.pid, 'argv': timed_argv, 'start_unix': start})
            while child.poll() is None:
                observed.update(descendants(child.pid))
                guard()
                time.sleep(1)
            require(child.returncode == 0, f'{name} exit {child.returncode}')
            guard()
        except BaseException:
            # The outer owned GNU-timeout session also contains multiprocessing lanes.
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    metrics = read(metrics_path)
    require(metrics['exit_code'] == 0, 'GNU time child status differs')
    workers = [pid for pid, text in observed.items() if 'multiprocessing.spawn import spawn_main' in text]
    if name.endswith('.derive'):
        require(len(workers) == 2, f'did not observe exactly two persistent spawn workers: {workers}')
    write(STATE / (name + '.completed.json'), {'pid': child.pid, 'argv': timed_argv, 'start_unix': start,
          'end_unix': time.time(), 'exit_code': child.returncode, 'resources': metrics,
          'observed_owned_descendants': observed, 'observed_spawn_worker_pids': workers})


def source_check(source):
    from scripts import derive_corpus_targets as derive
    record = derive.read_corpus_record(Path(source['source_dir']))
    chosen = source['selection']
    require([p.name for p in record.shards[:32]] == [e['source_shard'] for e in chosen], 'registered prefix differs')
    require(list(derive.shard_row_counts(record)[:32]) == [e['raw_rows'] for e in chosen], 'prefix row claims differ')
    for entry, path in zip(chosen, record.shards):
        require(path.stat().st_size == entry['raw_bytes'], 'registered raw size differs')
        require(sha(path) == entry['source_sha256'], 'registered raw payload changed')


def snapshot_derived(source):
    from scripts.adapt_raw_bt4_sidecars import storage_identity
    root = Path(source['derived_output'])
    summary = read(root / 'derive_targets_summary.json')
    require(summary['realized']['rows_read'] == source['raw_rows'], 'raw row accounting differs')
    require(summary['realized']['rows_dropped_envelope'] == 0, 'unexpected envelope drops')
    write(root.parent / 'derived_identity.json', {'source_summary_sha256': sha(root / 'derive_targets_summary.json'),
          'storage_identity': storage_identity(root), 'realized': summary['realized']})


def qualify(source):
    import numpy as np
    import zarr
    from numcodecs import blosc
    from scripts import corpus_row_provenance as provenance, bt4_policy_mix as mix
    from scripts.adapt_raw_bt4_sidecars import storage_identity
    blosc.set_nthreads(2)
    root = Path(source['derived_output'])
    before = read(root.parent / 'derived_identity.json')
    require(storage_identity(root) == before['storage_identity'], 'derived inputs changed during adapter/rank')
    summary_path = root / 'derive_targets_summary.json'
    summary = read(summary_path)
    summary_sha = sha(summary_path)
    require(summary_sha == before['source_summary_sha256'], 'derived summary changed')
    paths = sorted(root.glob('shard_*.zarr'))
    names = {p.name for p in paths}
    require(names == {e['path'] for e in summary['shards']} and len(paths) == len(summary['shards']), 'source membership differs')
    side_root, rank_root = Path(source['adapted_output']), Path(source['rank_output'])
    for other in (side_root, rank_root):
        require({p.name for p in other.glob('shard_*.zarr')} == names, 'sidecar membership differs')
    side = read(side_root / mix.SIDECAR_SUMMARY)
    rank = read(rank_root / 'sf_d9_rank_sidecar_summary.json')
    n = summary['realized']['rows_written']
    require(side['rows'] == rank['rows'] == n and side['source_dir'] == rank['source_dir'] == str(root), 'sidecar summary source/count differs')
    require(rank['source_derive_summary_sha256'] == summary_sha and rank['top_k'] == 3
            and rank['raw_dir'] == source['source_dir'] and rank['row_provenance']['raw_shards_read_once'] == 32
            and rank['raw_rows_read'] == source['raw_rows'] and rank['row_provenance']['policy_observation'] == 'phase0'
            and rank['row_provenance']['value_observation'] == 'latest-phase', 'rank observation or raw coverage differs')
    side_payloads = {e['path']: e for e in side['adapter']['written_shards']}
    require(set(side_payloads) == names, 'adapter payload lineage inventory differs')
    by_name = {e['source_shard']: e for e in source['selection']}
    covered = {name: bytearray(e['raw_rows']) for name, e in by_name.items()}
    digest = hashlib.sha256()
    rows = 0
    for path in paths:
        group = zarr.open_group(str(path), mode='r')
        count = int(group['x'].shape[0])
        stamp = dict(group.attrs)['derive_row_provenance']
        require(sha(path / provenance.FILENAME) == stamp['sha256'], 'row provenance changed')
        refs = provenance.read(path / provenance.FILENAME, rows=count)
        for ref in refs:
            require(ref['source_dir'] == source['source_dir'] and ref['source_shard'] in by_name, 'foreign original source')
            bitmap = covered[ref['source_shard']]
            offset = ref['source_row']
            require(0 <= offset < len(bitmap) and not bitmap[offset], 'duplicate/out-of-bounds source row')
            bitmap[offset] = 1
            digest.update(json.dumps(ref, sort_keys=True).encode())
        _, keys, key_sha, policy_sha = mix._sidecar_identity(group, path)
        attrs = mix._validate_sidecar(side_root / path.name, source_path=path, source_keys=keys,
                   source_key_sha=key_sha, source_policy_sha=policy_sha, onnx_sha=side['onnx']['sha256'],
                   providers=side['providers'], policy_output=side['policy_output'])
        require(attrs['source_dir'] == str(root), 'BT4 original derived parent differs')
        mix._validate_sf_rank_sidecar(rank_root / path.name, source_group=group, source_path=path,
                    source_summary_sha256=summary_sha, required_top_k=3)
        policy = zarr.open_group(str(side_root / path.name), mode='r')[mix.SIDECAR_POLICY_FIELD]
        payload_digest = hashlib.sha256()
        for start in range(0, count, 256):
            p = np.asarray(policy[start:start+256])
            payload_digest.update(np.ascontiguousarray(p).tobytes())
            legal = np.asarray(group['legal_mask'][start:start+256]) != 0
            require(np.all(p[~legal] == 0) and np.allclose(p.sum(axis=1,dtype=np.float64),1,atol=2e-6,rtol=0), 'BT4 legal normalized mass differs')
        require(payload_digest.hexdigest() == attrs['bt4_policy_sha256'] == side_payloads[path.name]['bt4_policy_sha256'], 'adapted payload changed after publication')
        require(attrs['row_provenance_sha256'] == stamp['sha256'] and attrs['source_derive_summary_sha256'] == summary_sha, 'adapter source lineage differs')
        rows += count
    require(rows == n and source['raw_rows'] - rows == summary['realized']['rows_dropped_no_result'], 'survival/exclusion accounting differs')
    require(storage_identity(root) == before['storage_identity'], 'derived storage changed during qualification')
    write(root.parent / 'common_input_qualification.json', {'status': 'complete', 'rows': rows,
          'physical_rows': source['raw_rows'], 'derive_realized': summary['realized'],
          'per_raw_shard_survivors': {name: sum(bits) for name,bits in covered.items()},
          'source_qualified_input_sequence_sha256': digest.hexdigest(), 'unchanged_derived_storage_identity': before['storage_identity'],
          'summary_pins': {str(p): sha(p) for p in [summary_path, side_root / mix.SIDECAR_SUMMARY, rank_root / 'sf_d9_rank_sidecar_summary.json']},
          'scope': 'All emitted rows admitted by source-bound BT4/rank consumers, unique source physical rows and immutable derived non-policy lineage; no training schedule execution.'})


def worker(plan):
    verify(plan)
    reg = read(REG)
    raw_ids = {}
    for source in reg['sources']:
        for field in ('derived_output', 'adapted_output', 'rank_output'):
            p = Path(source[field])
            require(not p.exists() and not Path(str(p) + '.writing').exists(), 'existing output/partial refused')
        for entry in source['selection']:
            raw = Path(source['source_dir']) / entry['source_shard']
            raw_ids[str(raw)] = identity(raw)
        stage(source['source_id'] + '.source_check', [plan['python'], __file__, '--source-check', source['source_id']], plan)
    for source in reg['sources']:
        name = source['source_id']
        parent = Path(source['derived_output']).parent
        parent.mkdir(exist_ok=True)
        stage(name + '.derive', command(plan, 'derive_corpus_targets.py', '--corpus', source['source_dir'], '--out', source['derived_output'],
              '--limit', source['raw_rows'], *reg['derive_common_args']), plan)
        stage(name + '.snapshot', [plan['python'], __file__, '--snapshot', name], plan)
        summary_path = Path(source['derived_output']) / 'derive_targets_summary.json'
        summary = read(summary_path)
        rows, shards = summary['realized']['rows_written'], len(summary['shards'])
        receipts = [json.loads(line) for line in Path(source['receipt_snapshot']['path']).read_text().splitlines()]
        receipt = receipts[0]
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
              '--max-provenance-cache-bytes', reg['resource_limits']['rank_index_cache_bytes'], *common), plan)
        stage(name + '.qualify', [plan['python'], __file__, '--qualify', name], plan)
    verify(plan)
    for source in reg['sources']:
        for entry in source['selection']:
            raw = Path(source['source_dir']) / entry['source_shard']
            require(identity(raw) == raw_ids[str(raw)] and sha(raw) == entry['source_sha256'], 'raw source changed during common batch')
    guard()
    write(STATE / 'worker_complete.json', {'status': 'complete', 'end_unix': time.time(), 'raw_storage_identities': raw_ids})


def main():
    parser = argparse.ArgumentParser()
    for key in ('checkout', 'commit', 'python', 'runtime-pins', 'source-check', 'snapshot', 'qualify', 'expected-plan-sha256'):
        parser.add_argument('--' + key)
    for key in ('freeze', 'execute', 'worker'):
        parser.add_argument('--' + key, action='store_true')
    args = parser.parse_args()
    if args.freeze:
        return freeze(args)
    if args.source_check or args.snapshot or args.qualify:
        name = args.source_check or args.snapshot or args.qualify
        source = next(s for s in read(REG)['sources'] if s['source_id'] == name)
        if args.source_check:
            return source_check(source)
        if args.snapshot:
            return snapshot_derived(source)
        return qualify(source)
    require(args.execute or args.worker, 'Choose --freeze or --execute; default does not run')
    require(sha(STATE / 'launch.json') == args.expected_plan_sha256, 'explicit launch plan SHA required')
    plan = read(STATE / 'launch.json')
    if args.worker:
        return worker(plan)
    start = time.time()
    guard()
    require(not (STATE / 'started.json').exists(), 'single attempt already started')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(STATE / 'python_bootstrap') + os.pathsep + plan['checkout'],
               CHESS_ANTI_ENGINE_LIVE_CONFIG=str(Path(plan['checkout']) / 'configs/pbt2_small.yaml'))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS'):
        env[key] = '2'
    remaining = 7170 - (time.time() - start)
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
    complete = code == 0 and (STATE / 'worker_complete.json').exists() and time.time() - start <= 7200
    write(STATE / ('completed.json' if complete else 'failed.json'),
          {'status': 'complete' if complete else 'failed', 'exit_code': code, 'start_unix': start,
           'end_unix': time.time(), 'plan_sha256': args.expected_plan_sha256})
    require(complete, 'common batch failed; preserve all logs/partials, no retry')


if __name__ == '__main__':
    main()
