"""One paired raw identity-cache preparation observation; no inference or full adaptation."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    plan_path = HERE / 'plan.json'
    assert len(sys.argv) == 2 and sha(plan_path) == sys.argv[1]
    plan = json.loads(plan_path.read_text())
    for path, digest in plan['pins'].items():
        assert sha(path) == digest, path
    out = HERE / 'run'
    out.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))
    import numpy as np
    import torch
    from numcodecs import blosc
    blosc.set_nthreads(2)
    assert torch.get_num_threads() == blosc.get_nthreads() == 2
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    assert sorted(os.sched_getaffinity(0)) == [4, 5]
    from scripts import adapt_raw_bt4_sidecars as candidate
    baseline = load('measurement_baseline_adapter', HERE / 'baseline_adapt_raw_bt4_sidecars.py')
    baseline.raw = load('measurement_baseline_raw', HERE / 'baseline_bt4_raw_corpus_sidecar.py')
    manifest = json.loads(Path(plan['manifest']).read_text())
    source = manifest['sources'][0]
    ref = {'source_dir': source['source_dir'], 'source_shard': plan['source_shard'],
           'source_config_sha256': plan['source_config_sha256'],
           'source_namespace': candidate.namespace(Path(source['source_dir']), plan['source_config_sha256'])}
    raw_path = Path(source['source_dir']) / plan['source_shard']
    sidecar = Path(source['sidecar_dir']) / candidate.raw.sidecar_name(plan['source_shard'])
    before = {str(p): candidate.storage_identity(p) for p in (raw_path, sidecar)}
    results, arrays, verified = {}, [], []
    for name, module in [('baseline', baseline), ('candidate', candidate)]:
        inputs = module.RawInputs(manifest, max_rows=plan['rows'], max_index_bytes=1024**2)
        inputs.cache_dir = out / (name + '_cache')
        inputs.cache_dir.mkdir()
        original_verify = module.raw.verify_shard
        verify_seconds = []
        def timed_verify(*args, **kwargs):
            start = time.perf_counter()
            value = original_verify(*args, **kwargs)
            verify_seconds.append(time.perf_counter() - start)
            return value
        module.raw.verify_shard = timed_verify
        started, cpu = time.perf_counter(), time.process_time()
        try:
            _, records = inputs.get(ref)
        finally:
            module.raw.verify_shard = original_verify
        elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu
        assert len(verify_seconds) == 1 and len(records) == plan['rows']
        arrays.append(np.array(records))
        verified.append(dict(inputs.verified))
        results[name] = {'wall_seconds': elapsed, 'cpu_seconds': cpu,
                         'verifier_seconds': verify_seconds[0], 'remaining_get_seconds': elapsed - verify_seconds[0],
                         'identity_sha256': hashlib.sha256(records.tobytes()).hexdigest(),
                         'index_disk_bytes': inputs.index_bytes,
                         'max_rss_kib_cumulative_process': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        del records, inputs
        gc.collect()
    assert np.array_equal(*arrays) and verified[0] == verified[1]
    assert before == {str(p): candidate.storage_identity(p) for p in (raw_path, sidecar)}
    for path, digest in plan['pins'].items():
        assert sha(path) == digest, path
    bytes_written = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    assert bytes_written < 1024**3
    assert not torch.cuda.is_initialized()
    report = {'status': 'COMPLETE', 'rows': plan['rows'], 'source_shard': str(raw_path),
              'plan_sha256': sha(plan_path), 'results': results, 'exact_identity_parity': True,
              'exact_verified_receipt_parity': True, 'source_storage_unchanged': before,
              'speedup_ratio': results['baseline']['wall_seconds'] / results['candidate']['wall_seconds'],
              'wall_reduction_fraction': 1 - results['candidate']['wall_seconds'] / results['baseline']['wall_seconds'],
              'temporary_file_bytes': bytes_written, 'python': sys.version,
              'numpy': np.__version__, 'torch': torch.__version__, 'threads': torch.get_num_threads(),
              'blosc_threads': blosc.get_nthreads(), 'affinity': sorted(os.sched_getaffinity(0)),
              'cuda_initialized': torch.cuda.is_initialized(),
              'scope': 'One old-first pair of actual RawInputs.get calls on one original closed shard; no policy gather/output, complete adapter run, inference, throughput confidence interval or parallel scaling claim.'}
    with (out / 'completed.json').open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
