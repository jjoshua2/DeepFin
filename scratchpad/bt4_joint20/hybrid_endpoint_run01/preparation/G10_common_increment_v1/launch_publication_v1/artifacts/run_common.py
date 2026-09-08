#!/usr/bin/env python3
"""Fixed nonoverlapping 64-shard incremental common-data batch. --freeze prepares binding only; --execute requires its SHA."""
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


_last_size_check = float('-inf')


def guard(*, force_usage=False):
    global _last_size_check
    require(not any(p.exists() for p in [STATE / 'STOP', STATE.parent / 'STOP']), 'STOP requested')
    require(shutil.disk_usage(STATE).free >= 150 * 1024**3, '150GiB disk reserve reached')
    now = time.monotonic()
    if force_usage or now - _last_size_check >= 60:
        require(usage() < 8 * 1024**3, '8GiB common-batch output/cache cap reached')
        _last_size_check = now


def selected(source):
    item = source['selection']
    require(sha(item['path']) == item['sha256'], 'selection manifest changed')
    return read(item['path'])['shards']


def source_storage(source):
    item = source['source_metadata']
    require(sha(item['path']) == item['sha256'], 'source metadata snapshot changed')
    for entry in read(item['path']):
        path = Path(entry['source_path'])
        require(path.is_file() and not path.is_symlink(), 'selected raw is not regular')
        expected = [entry[k] for k in ['device','inode','bytes','mtime_ns','ctime_ns']]
        require(identity(path) == expected, f'selected raw storage changed: {path}')


def selection_proof(source):
    item = source['selection']
    selected(source)
    return {**read(item['path']), 'path': str(Path(item['path']).resolve()),
            'sha256': item['sha256'],
            'order': 'original corpus shard order; limit applies after selection'}



def validate_registration(reg):
    require(reg['physical_rows'] == 532389 and reg['closed_shards'] == 64, 'wrong fixed batch')
    require({s['source_id']:s['physical_rows'] for s in reg['sources']} ==
            {'run06_g10':266491,'run07_g10_companion4':265898} and len(reg['sources']) == 2,
            'unregistered source population')
    require(reg['derive_options'] == {'scheme':'uniform-d9','policy_observation':'phase0',
            'value_observation':'latest-phase','value_scheme':'search','temp':.0005,'floor':0,
            'seed':0,'rows_per_shard':8192,'workers':2,'row_provenance':True}, 'unregistered derivation options')
    limits = reg['limits']
    require(limits['wall_seconds_including_kill'] == 7200 and limits['cpu_affinity'] == [2,3]
            and limits['numeric_threads'] == 2 and limits['nice'] == 19 and limits['ionice_class'] == 3
            and limits['CUDA_VISIBLE_DEVICES'] == '' and limits['new_output_cache_bytes'] == 8*1024**3
            and limits['minimum_free_bytes'] == 150*1024**3, 'unregistered resource bounds')
    for source in reg['sources']:
        shards = selected(source)
        require([e['source_shard'] for e in shards] == [f'w00-{i:05d}.jsonl.zst' for i in range(32,64)]
                and sum(e['rows'] for e in shards) == source['physical_rows'], 'unregistered selected shard set')
        require(source['support_drop_ceiling'] == 64 and source['missing_result_count_ceiling'] == source['physical_rows']//50,
                'unregistered exclusion bounds')


def freeze(args):
    validate_registration(read(REG))
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
    require(runtime.get('features', {}).get('closed_shard_selection') is True
            and runtime.get('features', {}).get('support_exclusion_requires_result') is True,
            'runtime must qualify selected-shard APIs and result-bearing exclusion guard')
    for script in ['derive_corpus_targets.py', 'sf_d9_rank_sidecar.py', 'adapt_raw_bt4_sidecars.py']:
        require(str(cwd / 'scripts' / script) in pins, f'missing runtime script pin: {script}')
    pins[str(STATE / 'qualification_contract.md')] = sha(STATE / 'qualification_contract.md')
    pins[str(STATE / 'metadata_capture.json')] = sha(STATE / 'metadata_capture.json')
    for source in read(REG)['sources']:
        for key in ('source_manifest', 'selection', 'closed_bt4_receipts', 'closed_source_progress', 'source_metadata'):
            item = source[key]
            pins[item['path']] = item['sha256']
        source_storage(source)
        for entry in read(source['source_metadata']['path']):
            attrs = Path(entry['sidecar_path']) / '.zattrs'
            snapshot = entry['sidecar_attrs_snapshot']
            require(sha(attrs) == snapshot['sha256'], 'selected teacher metadata changed')
            pins[str(attrs)] = snapshot['sha256']
    plan = {'schema': 1, 'status': 'PREPARED_NOT_LAUNCHED', 'checkout': str(cwd), 'commit': args.commit,
            'python': args.python, 'pins': pins, 'registration': str(REG),
            'command_builder_sha256': sha(__file__), 'scope': 'derive/adapt/rank only; no audit, recipe mixing or training',
            'limits': read(REG)['limits'], 'sampling_guards_seconds': {'STOP_and_free': 5, 'output_size': 60}, 'features': runtime['features']}
    verify(plan)
    write(STATE / 'launch.json', plan)
    print(sha(STATE / 'launch.json'))


def command(plan, script, *args):
    require(script in {'derive_corpus_targets.py','adapt_raw_bt4_sidecars.py','sf_d9_rank_sidecar.py'}, 'unregistered tool')
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
    require(name in {f'{source}.{kind}' for source in ['run06_g10','run07_g10_companion4'] for kind in ['derive','snapshot','adapt','rank','qualify']}, 'unregistered stage')
    guard(force_usage=True)
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
                time.sleep(5)
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


def actual_exclusions(source, summary):
    require(summary.get('source_selection') == selection_proof(source), 'derived selection differs')
    realized = summary['realized']
    support = realized.get('policy_support_exclusions', [])
    ns, nn, nr, nw = [realized.get(k) for k in ['rows_dropped_policy_support','rows_dropped_no_result','rows_read','rows_written']]
    require(all(type(n) is int and n >= 0 for n in [ns,nn,nr,nw]), 'invalid drop/row counters')
    require(ns == len(support) <= source['support_drop_ceiling'] == 64, 'support drop cap/count differs')
    require(summary.get('max_policy_support_misses') == 64, 'wrong support budget')
    require(nn <= source['missing_result_count_ceiling'] == source['physical_rows']//50, 'missing-result cap exceeded')
    require(realized['rows_dropped_envelope'] == 0 and nr == source['physical_rows'] and nw == nr-ns-nn,
            'unexpected survival/drop accounting')
    if ns:
        require(summary.get('policy_support_misses_file') == 'policy_support_misses.jsonl', 'missing actual support ledger')
        path = Path(source['derived_output']) / 'policy_support_misses.jsonl'
        require(not path.is_symlink() and path.is_file(), 'support ledger is not regular')
        require([json.loads(line) for line in path.read_text().splitlines()] == support, 'support ledger differs from summary')
    else:
        require(summary.get('policy_support_misses_file') is None, 'unexpected zero-count support ledger')
    universe = {e['source_shard']:e['rows'] for e in selected(source)}
    excluded = set()
    for ref in support:
        key = (ref['source_shard'],ref['source_row'])
        require(ref['source_dir'] == source['source_dir'] and key[0] in universe
                and type(key[1]) is int and 0 <= key[1] < universe[key[0]] and key not in excluded,
                'foreign, duplicate or out-of-range support reference')
        require(ref['reason'] == 'selected_phase0_policy_support' and ref['policy_depth'] == 9
                and ref['full_history_input_key_verified'] is True, 'unsupported exclusion reason')
        excluded.add(key)
    return excluded


def verify_rank_accounting(source, summary, rank):
    realized = summary['realized']
    require(rank.get('source_selection') == selection_proof(source), 'rank selection differs')
    require(rank['raw_rows_read'] == source['physical_rows']
            and rank['rows_dropped_no_result'] == realized['rows_dropped_no_result']
            and rank['row_provenance'].get('rows_dropped_policy_support',0) == realized['rows_dropped_policy_support']
            and rank['rows'] == realized['rows_written'], 'independent rank/drop accounting differs')
    require(rank['row_provenance']['join'] == 'source-qualified-physical-row-and-full-history-keys-v1',
            'missing independent raw eligibility/injectivity proof')


def verify_complement(source, summary, rank, covered):
    verify_rank_accounting(source, summary, rank)
    survived = sum(sum(bits) for bits in covered.values())
    omitted = source['physical_rows'] - survived
    require(survived == summary['realized']['rows_written']
            and omitted == summary['realized']['rows_dropped_policy_support'] + rank['rows_dropped_no_result'],
            'injective eligible mapping does not cover complete survivor complement')
    return omitted


def stop_owned_group(child, grace=5):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Group members can survive the group leader; always reap the owned group.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def snapshot_derived(source):
    from scripts.adapt_raw_bt4_sidecars import storage_identity
    root = Path(source['derived_output'])
    summary = read(root / 'derive_targets_summary.json')
    require(summary['realized']['rows_read'] == source['physical_rows'], 'raw row accounting differs')
    require(summary['realized']['rows_dropped_envelope'] == 0, 'unexpected envelope drops')
    actual_exclusions(source, summary)
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
    excluded = actual_exclusions(source, summary)
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
            and rank['raw_rows_read'] == source['physical_rows'] and rank['row_provenance']['policy_observation'] == 'phase0'
            and rank['row_provenance']['value_observation'] == 'latest-phase', 'rank observation or raw coverage differs')
    verify_rank_accounting(source, summary, rank)
    side_payloads = {e['path']: e for e in side['adapter']['written_shards']}
    require(set(side_payloads) == names, 'adapter payload lineage inventory differs')
    by_name = {e['source_shard']: e for e in selected(source)}
    covered = {name: bytearray(e['rows']) for name, e in by_name.items()}
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
            require((ref['source_shard'], offset) not in excluded, 'excluded source row survived')
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
    require(rows == n, 'emitted row total differs')
    omitted = verify_complement(source, summary, rank, covered)
    require(storage_identity(root) == before['storage_identity'], 'derived storage changed during qualification')
    write(root.parent / 'common_input_qualification.json', {'status': 'complete', 'rows': rows,
          'physical_rows': source['physical_rows'], 'derive_realized': summary['realized'],
          'source_selection': selection_proof(source), 'independent_rank_missing_result_rows': rank['rows_dropped_no_result'],
          'verified_support_exclusion_rows': len(excluded), 'omitted_rows': omitted,
          'complement_proof': 'rank eligibility plus injective full-history physical joins and exact universe cardinality; omitted IDs reconstructible from selection universe minus emitted refs',
          'per_raw_shard_survivors': {name: sum(bits) for name,bits in covered.items()},
          'source_qualified_input_sequence_sha256': digest.hexdigest(), 'unchanged_derived_storage_identity': before['storage_identity'],
          'summary_pins': {str(p): sha(p) for p in [summary_path, side_root / mix.SIDECAR_SUMMARY, rank_root / 'sf_d9_rank_sidecar_summary.json']},
          'scope': 'All emitted rows admitted by source-bound BT4/rank consumers, unique source physical rows and immutable derived non-policy lineage; no training schedule execution.'})


def worker(plan):
    verify(plan)
    reg = read(REG)
    validate_registration(reg)
    raw_ids = {}
    for source in reg['sources']:
        for field in ('derived_output', 'adapted_output', 'rank_output'):
            p = Path(source[field])
            require(not p.exists() and not Path(str(p) + '.writing').exists(), 'existing output/partial refused')
        for entry in selected(source):
            raw = Path(source['source_dir']) / entry['source_shard']
            raw_ids[str(raw)] = identity(raw)
        source_storage(source)
    for source in reg['sources']:
        name = source['source_id']
        parent = Path(source['derived_output']).parent
        parent.mkdir(exist_ok=True)
        stage(name + '.derive', command(plan, 'derive_corpus_targets.py', '--corpus', source['source_dir'], '--out', source['derived_output'],
              '--limit', source['physical_rows'], '--max-policy-support-misses', source['support_drop_ceiling'],
              '--source-shards', source['selection']['path'],
              '--scheme','uniform-d9','--policy-observation','phase0','--value-observation','latest-phase',
              '--value-scheme','search','--temp','0.0005','--floor','0','--seed','0','--rows-per-shard','8192','--workers','2','--row-provenance'), plan)
        stage(name + '.snapshot', [plan['python'], __file__, '--snapshot', name, '--expected-plan-sha256',sha(STATE / 'launch.json')], plan)
        summary_path = Path(source['derived_output']) / 'derive_targets_summary.json'
        summary = read(summary_path)
        rows, shards = summary['realized']['rows_written'], len(summary['shards'])
        receipts = [json.loads(line) for line in Path(source['closed_bt4_receipts']['path']).read_text().splitlines()]
        receipt = receipts[0]
        attrs = read(Path(source['sidecar_dir']) / receipt['sidecar'] / '.zattrs')
        mapping = {'schema': 1, 'derived_summary': {'path': str(summary_path), 'sha256': sha(summary_path)},
                   'teacher': {'onnx': {'path': attrs['onnx_path'], 'sha256': receipt['onnx_sha256']},
                               'policy_output': receipt['policy_output'], 'providers': receipt['providers'], 'remap': receipt['remap_provenance']},
                   'sources': [{'source_dir': source['source_dir'], 'sidecar_dir': source['sidecar_dir'],
                                'manifest': source['source_manifest'], 'receipts': source['closed_bt4_receipts']}]}
        mapping_path = parent / 'adapter_manifest.json'
        write(mapping_path, mapping)
        stage(name + '.adapt', command(plan, 'adapt_raw_bt4_sidecars.py', '--manifest', mapping_path, '--expected-manifest-sha256', sha(mapping_path),
              '--out', source['adapted_output'], '--max-index-bytes', reg['limits']['adapter_index_cache_bytes']), plan)
        common = ['--expected-rows', rows, '--expected-shards', shards, '--expected-source-summary-sha256', sha(summary_path)]
        stage(name + '.rank', command(plan, 'sf_d9_rank_sidecar.py', '--raw', source['source_dir'], '--shards', source['derived_output'],
              '--out', source['rank_output'], '--limit', source['physical_rows'], '--top-k', '3', '--seed', '0', '--rows-per-shard', '8192',
              '--max-provenance-cache-bytes', reg['limits']['rank_index_cache_bytes'], '--source-shards',source['selection']['path'], *common), plan)
        stage(name + '.qualify', [plan['python'], __file__, '--qualify', name, '--expected-plan-sha256',sha(STATE / 'launch.json')], plan)
    verify(plan)
    for source in reg['sources']:
        for entry in selected(source):
            raw = Path(source['source_dir']) / entry['source_shard']
            require(identity(raw) == raw_ids[str(raw)], 'raw source changed during common batch')
    guard(force_usage=True)
    write(STATE / 'worker_complete.json', {'status': 'complete', 'end_unix': time.time(), 'raw_storage_identities': raw_ids})


def main():
    def terminate(_sig, _frame):
        raise SystemExit('termination requested')
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser()
    for key in ('checkout', 'commit', 'python', 'runtime-pins', 'snapshot', 'qualify', 'expected-plan-sha256'):
        parser.add_argument('--' + key)
    for key in ('freeze', 'execute', 'worker'):
        parser.add_argument('--' + key, action='store_true')
    args = parser.parse_args()
    if args.freeze:
        return freeze(args)
    if args.snapshot or args.qualify:
        require(sha(STATE / 'launch.json') == args.expected_plan_sha256, 'internal stage requires pinned launch')
        name = args.snapshot or args.qualify
        source = next(s for s in read(REG)['sources'] if s['source_id'] == name)
        if args.snapshot:
            return snapshot_derived(source)
        return qualify(source)
    require(args.execute or args.worker, 'Choose --freeze or --execute; default does not run')
    require(sha(STATE / 'launch.json') == args.expected_plan_sha256, 'explicit launch plan SHA required')
    plan = read(STATE / 'launch.json')
    if args.worker:
        return worker(plan)
    start = time.time()
    verify(plan)
    guard(force_usage=True)
    require(not (STATE / 'started.json').exists(), 'single attempt already started')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(STATE / 'python_bootstrap') + os.pathsep + plan['checkout'],
               CHESS_ANTI_ENGINE_LIVE_CONFIG=str(Path(plan['checkout']) / 'configs/pbt2_small.yaml'))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS'):
        env[key] = '2'
    remaining = 7170 - (time.time() - start)
    require(remaining > 0, 'preflight exhausted wall cap')
    argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=30s', f'{remaining:.3f}s', '/usr/bin/nice', '-n', '19',
            '/usr/bin/ionice', '-c', '3', '/usr/bin/taskset', '-c', '2,3', plan['python'], __file__, '--worker',
            '--expected-plan-sha256', args.expected_plan_sha256]
    with (STATE / 'worker.log').open('xb') as out:
        child = subprocess.Popen(argv, env=env, cwd=plan['checkout'], stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            write(STATE / 'started.json', {'pid': child.pid, 'start_unix': start, 'argv': argv, 'plan_sha256': args.expected_plan_sha256})
            while child.poll() is None:
                guard()
                time.sleep(5)
            code = child.returncode
        except BaseException as exc:
            stop_owned_group(child)
            write(STATE / 'failed.json', {'status':'failed','error':repr(exc),
                  'exit_code':child.returncode,'start_unix':start,'end_unix':time.time(),
                  'plan_sha256':args.expected_plan_sha256})
            raise
        finally:
            if child.poll() is not None:
                stop_owned_group(child)
    complete = code == 0 and (STATE / 'worker_complete.json').exists() and time.time() - start <= 7200
    write(STATE / ('completed.json' if complete else 'failed.json'),
          {'status': 'complete' if complete else 'failed', 'exit_code': code, 'start_unix': start,
           'end_unix': time.time(), 'plan_sha256': args.expected_plan_sha256})
    require(complete, 'common batch failed; preserve all logs/partials, no retry')


if __name__ == '__main__':
    main()
