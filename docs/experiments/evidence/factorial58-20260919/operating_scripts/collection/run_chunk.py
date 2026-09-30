"""Prepared G10 G10_worker01_postderive_recovery_v1__run07_g10_companion4 full cohort; unchanged fixed32 collector, one shard per chunk."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STATE = Path(__file__).resolve().parent

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def stamp(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]

def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def validate_operations(p):
    helper = p['telemetry_helper']
    require(isinstance(helper, dict) and set(helper) == {'path', 'sha256'}
            and Path(helper['path']).is_absolute() and Path(helper['path']).is_file()
            and sha(helper['path']) == helper['sha256'], 'telemetry helper pin differs')
    require(len(p['operations']) == 1, 'exact one chunk operation required')
    operation = p['operations'][0]
    start = p['start_shard']
    width = p['max_shards']
    require(type(start) is int and type(width) is int and 0 <= start and 1 <= width <= 4,
            'bounded grouped shard range required')
    summary = Path(p['selection']['source']) / 'derive_targets_summary.json'
    require(sha(summary) == p['selection']['summary_sha256'], 'source summary pin differs')
    source_specs = json.loads(summary.read_text())['shards']
    require(len(source_specs) == p['selection']['source_shards'], 'source shard count differs')
    expected_selection = source_specs[start:start + width]
    require(len(expected_selection) == width, 'incomplete selected shard range')
    require(operation['selection'] == p['selection']['shards'] == expected_selection,
            'canonical remaining-chunk selection differs')
    real = sum(x['rows'] for x in expected_selection)
    padding = sum((-x['rows']) % 512 for x in expected_selection)
    counts = {'real_rows': real, 'padding_rows': padding, 'calls': (real + padding) // 512, 'input_rows': real + padding}
    require(p['counts'] == operation['counts'] == counts, 'chunk counts differ')
    require(p['profile'] == 'ceres-c3-batched-compact-v1', 'full-shard profile differs')
    argv = operation['collector_argv']
    expected = {'--source': p['selection']['source'],
        '--expected-source-summary-sha256': p['selection']['summary_sha256'],
        '--onnx': p['model']['path'], '--expected-onnx-sha256': p['model']['sha256'],
        '--out': p['output'], '--start-shard': str(start), '--max-shards': str(width),
        '--wdl-output': 'value', '--wdl-output-kind': 'logits', '--batch-size': '512',
        '--threads': '2', '--gpu-mem-gb': '16', '--gpu-lock': p['gpu_lock'],
        '--minimum-free-gib': '150', '--max-output-gib': '0.125', '--max-seconds': '300',
        '--audited-source-manifest': p['audited_qualification']['path'],
        '--expected-audited-source-manifest-sha256': p['audited_qualification']['sha256'],
        '--stop': str(STATE / 'STOP')}
    require(len(argv) == 2 * len(expected) + 2, 'unexpected chunk arguments')
    for key, value in expected.items():
        require(argv.count(key) == 1 and argv[argv.index(key) + 1] == value,
                'chunk argv differs: ' + key)
    require(argv.count('--retain-value2') == argv.count('--pad-final-batch') == 1,
            'explicit dual/padding flags required')


def run_selected(p, collector, shared, deadline, budget):
    receipts = []
    for operation in p['operations']:
        budget()
        args = collector.build_parser().parse_args(operation['collector_argv'])
        args.max_seconds = deadline - time.time()
        require(args.max_seconds > 60, 'insufficient remaining stage budget')
        args.pilot_counts = operation['counts']
        collector.validate_args(args)
        began = time.time()
        status = shared.run(args, child_target=collector.child)
        require(status == 0, 'collector failed')
        budget()
        invocation = Path(args.invocation)
        complete = json.loads((invocation / 'completed.json').read_text())
        require(complete.get('complete') is True and complete['profile'] == p['profile']
                and complete['selection'] == operation['selection']
                and complete['rows'] == operation['counts']['real_rows']
                and complete['shards'] == complete['new_shards'] == len(operation['selection'])
                and 'fragments' not in complete and 'new_fragments' not in complete,
                'pilot completion scope')
        require(complete['collection_counts'] == complete['new_collection_counts'] == operation['counts'],
                'pilot completion counts')
        timings = json.loads((invocation / 'call_timings.json').read_text())
        require(timings['counts'] == operation['counts'] and
                len(timings['seconds']) == operation['counts']['calls'], 'missing exact call observation')
        require((invocation / 'first_loaded_project.json').exists() and
                (invocation / 'final_loaded_project.json').exists() and
                (invocation / 'final_loaded_maps.txt').exists(), 'missing final process observation')
        receipts.append({'selection': operation['selection'], 'counts': operation['counts'],
                         'invocation': str(invocation), 'completion_sha256': sha(invocation / 'completed.json'),
                         'call_timings_sha256': sha(invocation / 'call_timings.json'),
                         'supervisor_elapsed_seconds': time.time() - began,
                         'session_run_sum_seconds': timings['sum_seconds']})
    require({k: sum(x['counts'][k] for x in receipts) for k in p['counts']} == p['counts'],
            'final aggregate counts differ')
    return receipts


def main():
    global STATE
    start = time.time()
    deadline = start + 300
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', required=True, type=Path)
    ap.add_argument('--expected-plan-sha256', required=True)
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    require(a.state.is_absolute() and a.state.is_dir(), 'absolute prepared state directory')
    STATE = a.state
    require(sha(STATE / 'plan.json') == a.expected_plan_sha256, 'plan hash')
    p = json.loads((STATE / 'plan.json').read_text())
    require(p['status'] == 'PREPARED_NOT_LAUNCHED', 'plan status')
    validate_operations(p)
    if not a.execute:
        print(json.dumps({'status': p['status'], 'operations': p['operations'],
                          'environment': p['environment']}))
        return
    out = Path(p['output'])
    require(not out.exists(), 'pilot output is not fresh; no automatic resume')
    require(not (STATE / 'actual_start.json').exists(), 'attempt already exists')
    require(os.path.abspath(sys.executable) == p['python'], 'actual interpreter')
    require(str(Path(sys.executable).resolve()) == p['python_resolved'], 'interpreter resolved')
    require(str(Path(sys.prefix).resolve()) == p['python_prefix'], 'environment prefix')
    require(sorted(os.sched_getaffinity(0)) == p['cpu_affinity'], 'CPU affinity')
    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('PYTHONHOME'), 'inherited loader override')
    for k, v in p['environment'].items():
        require(os.environ.get(k) == v, 'environment differs: ' + k)
    write(STATE / 'actual_start.json', {'pid': os.getpid(), 'started_unix': start,
          'deadline_unix': deadline, 'plan_sha256': a.expected_plan_sha256,
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
    samples = []
    def budget():
        require(time.time() < deadline - 30, 'inclusive deadline')
        require(not any(Path(x).exists() for x in p['stop_paths']), 'STOP')
    try:
        budget()
        available = next(int(x.split()[1]) * 1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        require(available >= 48 * 2**30, 'startup available memory below48GiB')
        require(subprocess.check_output(['git', '-C', p['runtime'], 'rev-parse', 'HEAD'], text=True).strip() == p['runtime_head'], 'runtime head')
        subprocess.run(['git', '-C', p['runtime'], 'diff', '--exit-code', 'HEAD', '--'], check=True, stdout=subprocess.DEVNULL)
        for item in p['small_pins']:
            budget()
            require(sha(item['path']) == item['sha256'], 'source/evidence pin: ' + item['path'])
        for item in p['library_pins']:
            require(stamp(item['path']) == item['stat'], 'prepared library changed: ' + item['path'])
        require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == p['boot_id'], 'prepared WSL mount namespace belongs to another boot')
        require(all(v['mount_line'] in Path('/proc/self/mountinfo').read_text().splitlines() for v in p['wsl_aliases'].values()), 'prepared WSL mount namespace differs before import')
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import wsl_mapped_libraries as mapped
        sys.path.insert(0, p['runtime'])
        from scripts import ceres_derived_sidecar as collector
        installed = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in importlib.metadata.distributions(path=[p['site_packages']])}
        require(installed == p['package_versions'], 'installed package identities changed')
        import torch
        from numcodecs.blosc import set_nthreads, get_nthreads
        torch.set_num_threads(2)
        set_nthreads(2)
        require(torch.get_num_threads() == 2 and get_nthreads() == 2, 'actual numeric thread settings')
        sys.path.insert(0, str(Path(p['mapped_validator']).parent))
        spec = importlib.util.spec_from_file_location('saved_ceres_maps', p['mapped_validator'])
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        shared = collector.shared
        original_guard = shared.guard_resources
        last_sample = [0.0]
        def check_aggregate(extra_bytes=0):
            require(shared.output_size(Path(p['output'])) + sum(q.stat().st_size for q in STATE.iterdir() if q.is_file()) + extra_bytes <= p['aggregate_output_limit_bytes'], 'aggregate bank/state output cap')
        helper = p['telemetry_helper']
        require(sha(helper['path']) == helper['sha256'], 'telemetry helper changed before import')
        telemetry_spec = importlib.util.spec_from_file_location('_ceres_gpu_telemetry', helper['path'])
        require(telemetry_spec is not None and telemetry_spec.loader is not None, 'telemetry helper loader missing')
        telemetry = importlib.util.module_from_spec(telemetry_spec)
        telemetry_spec.loader.exec_module(telemetry)
        def query_telemetry(label, command):
            def observed_failure(observation):
                record = {'unix': time.time(), 'pid': os.getpid(), 'query': label, 'argv': command,
                          'telemetry_helper_sha256': helper['sha256'], **observation}
                # Separate append-only files survive a collector child failure. The
                # existing bank/state aggregate guard includes every telemetry file.
                path = STATE / f'telemetry_failures_{os.getpid()}.jsonl'
                with path.open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                check_aggregate()
            return telemetry.query(command, check_budget=budget, on_failure=observed_failure)
        def guarded(args):
            budget()
            original_guard(args)
            now = time.monotonic()
            if now - last_sample[0] < 5:
                return
            last_sample[0] = now
            check_aggregate()
            available = next(int(x.split()[1]) * 1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
            require(available >= 32 * 2**30, 'host available memory below32GiB')
            used = int(query_telemetry('memory.used', [p['nvidia_smi'], '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits']))
            require(used <= 24576, 'sampled device memory above prepared cap')
            rss = int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))) * 1024
            require(rss <= 12 * 2**30, 'sampled process RSS above12GiB')
            child = Path(args.invocation) / 'child_started.json'
            child_rss = None
            if child.exists():
                childpid = json.loads(child.read_text())['pid']
                try:
                    child_rss = int(next(x.split()[1] for x in Path(f'/proc/{childpid}/status').read_text().splitlines() if x.startswith('VmRSS:'))) * 1024
                except (FileNotFoundError, StopIteration):
                    pass
                require(child_rss is None or child_rss <= 12 * 2**30, 'sampled child RSS above12GiB')
            samples.append({'unix': time.time(), 'device_memory_MiB': used, 'self_rss': rss, 'child_rss': child_rss})
        shared.guard_resources = guarded
        original_open = collector.open_teacher
        def observed_open(args):
            # Called only inside the actual collector child after its own lease acquisition.
            hardware = query_telemetry('hardware_identity', [p['nvidia_smi'], '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader,nounits'])
            require(hardware == p['gpu_identity'], 'GPU/driver identity changed')
            write(Path(args.invocation) / 'lease_observation.json', {'pid': os.getpid(), 'unix': time.time(), 'gpu_identity': hardware, 'scope': 'Observed after collector acquired its shared GPU lock'})
            session = original_open(args)
            calls = []
            first = None
            expected_counts = args.pilot_counts
            expected_calls = expected_counts['calls']
            class Observed:
                def __getattr__(self, name):
                    return getattr(session, name)
                def run(self, names, feed):
                    nonlocal first
                    require(len(calls) < expected_calls, 'too many physical calls')
                    require(names == p['heads'], 'requested heads differ')
                    begin = time.perf_counter()
                    result = session.run(names, feed)
                    calls.append(time.perf_counter() - begin)
                    if first is None or len(calls) == expected_calls:
                        text = Path('/proc/self/maps').read_text()
                        label = 'first' if first is None else 'final'
                        (Path(args.invocation) / (label + '_loaded_maps.txt')).write_text(text)
                        candidates = mapped.mapped_candidates(text, old.mapped_candidates)
                        if first is None:
                            first = mapped.validate(candidates, p['library_pins'], budget, p['wsl_aliases'], Path('/proc/self/mountinfo').read_text(), old)
                            write(Path(args.invocation) / 'loaded_libraries.json', first)
                        else:
                            require(all(v['mount_line'] in Path('/proc/self/mountinfo').read_text().splitlines() for v in p['wsl_aliases'].values()), 'final WSL mount changed')
                            expected = {str(Path(x['path']).resolve()): x for x in first}
                            for x in candidates:
                                q = str(Path(x['path']).resolve())
                                require(q in expected and not x['deleted'] and stamp(x['path']) == expected[q]['stat'] and x['inode'] == expected[q]['stat'][1], 'new/changed final loaded library')
                                expected_device = p['wsl_aliases'][q]['mapped_device'] if q in p['wsl_aliases'] else '{:x}:{:x}'.format(os.major(expected[q]['stat'][0]), os.minor(expected[q]['stat'][0]))
                                require(mapped.device(x['device']) == expected_device, 'final mapping device differs')
                        modules = {}
                        for name, module in tuple(sys.modules.items()):
                            if name.startswith(('chess_anti_engine', 'scripts')) and getattr(module, '__file__', None):
                                f = Path(module.__file__)
                                require(f.is_relative_to(Path(p['runtime'])), 'project module imported outside frozen runtime')
                                modules[name] = {'path': str(f), 'sha256': sha(f)}
                        write(Path(args.invocation) / (label + '_loaded_project.json'), modules)
                    if len(calls) == expected_calls:
                        write(Path(args.invocation) / 'call_timings.json', {'counts': expected_counts, 'batch': 512, 'seconds': calls, 'sum_seconds': sum(calls), 'scope': 'Synchronous session.run only; excludes conversion, gather, storage, guards and session setup'})
                    return result
            return Observed()
        collector.open_teacher = observed_open
        completed_invocations = run_selected(p, collector, shared, deadline, budget)
        require(sha(STATE / 'plan.json') == a.expected_plan_sha256, 'plan changed')
        require(sha(helper['path']) == helper['sha256'], 'telemetry helper changed before completion')
        for item in p['small_pins']:
            budget()
            require(sha(item['path']) == item['sha256'], 'input changed before completion')
        subprocess.run(['git', '-C', p['runtime'], 'diff', '--exit-code', 'HEAD', '--'], check=True, stdout=subprocess.DEVNULL)
        completed = {'status': 'COMPLETE_CHUNK_PENDING_INDEPENDENT_REVIEW', 'ended_unix': time.time(), 'elapsed_seconds': time.time()-start, 'plan_sha256': a.expected_plan_sha256, 'invocations': completed_invocations, 'counts': p['counts'], 'session_run_sum_seconds': sum(x['session_run_sum_seconds'] for x in completed_invocations), 'samples': samples, 'limitations': p['limitations']}
        budget()
        check_aggregate(len(json.dumps(completed, indent=2, allow_nan=False).encode()) + 1)
        write(STATE / 'completed.json', completed)
    except BaseException as e:
        write(STATE / 'failed.json', {'status': 'FAILED_NO_AUTOMATIC_RETRY', 'ended_unix': time.time(), 'error': repr(e), 'samples': samples})
        raise

if __name__ == '__main__':
    main()
