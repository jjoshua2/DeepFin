#!/usr/bin/env python3
"""Hash-pinned, exclusive-GPU BT4 serial/prefetch comparison; no live adoption."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import threading
from typing import Any

import numpy as np
import zarr

ORDER = ('A', 'B', 'C', 'C', 'B', 'A')


def require(ok: Any, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def pin(ref: dict[str, Any]) -> None:
    require(sha(ref['path']) == ref['sha256'], f"pin changed: {ref['path']}")


def publish(path: Path, value: Any) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.writing')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def runtime_check(variant: dict[str, Any]) -> None:
    root = Path(variant['runtime'])
    require(subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
            == variant['head'], 'runtime head changed')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    expected = {str(Path(p['path']).absolute()) for p in variant['pins']}
    require(bool(expected), 'empty runtime pins')
    actual = set()
    links = {}
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != '.git']
        for name in dirs:
            p = Path(directory) / name
            if p.is_symlink():
                links[str(p)] = os.readlink(p)
                require(root.resolve() in p.resolve(strict=True).parents, 'runtime directory link escapes')
        for name in files:
            p = Path(directory) / name
            if p != root / '.git':
                actual.add(str(p.absolute()))
    require(links == variant.get('directory_links', {}), 'runtime directory links changed')
    require(actual == expected, 'runtime file membership changed')
    for ref in variant['pins']:
        require(str(Path(ref['path']).resolve(strict=True)) == ref['resolved_path'], 'runtime link target changed')
        pin(ref)


def dependencies(plan: dict[str, Any]) -> None:
    items = json.loads(Path(plan['queue']).read_text())['items']
    for dependency in plan['dependencies']:
        matches = [item for item in items if item['id'] == dependency['id']]
        require(len(matches) == 1 and matches[0]['status'] == 'logged', 'factorial dependency not successful')
        item = matches[0]
        require(item['command_file'] == dependency['path']
                and item['command_sha256'] == dependency['sha256'], 'dependency descriptor changed')
        pin(dependency)
        spec = json.loads(Path(dependency['path']).read_text())
        completion = spec['completion']
        result = json.loads(Path(completion['path']).read_text())
        require(result[completion['status_key']] == completion['expected'], 'dependency completion failed')
        terminal = json.loads((Path(item['out']) / 'parent_outer_terminal.json').read_text())
        require(terminal['returncode'] == 0, 'dependency supervisor failed')


def terminate(child: subprocess.Popen[Any]) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    require(child.poll() is not None, 'owned child did not reap')


def checked_child(command: list[str], cwd: Path, env: dict[str, str], log: Path, guard: Any,
                  lease_fd: int | None = None) -> None:
    guard()
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                 pass_fds=() if lease_fd is None else (lease_fd,))
        try:
            while child.poll() is None:
                guard()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            require(child.returncode == 0, f'worker failed: {child.returncode}')
        finally:
            terminate(child)


def compare(runs: list[dict[str, Any]], roots: list[Path], guard: Any = None) -> dict[str, Any]:
    require([r['variant'] for r in runs] == list(ORDER), 'wrong fixed run order')
    require(len(roots) == len(ORDER), 'missing output roots')
    require(len({r['input_sequence_sha256'] for r in runs}) == 1, 'actual inference inputs differ')
    require(len({tuple(r['raw_output_sha256']) for r in runs}) == 1,
            'raw outputs differ; exact-output screen inconclusive, no tolerance widening')
    baseline: Any = zarr.open_group(str(roots[0]), mode='r')
    keys = sorted(baseline.array_keys())
    for path in roots[1:]:
        bank: Any = zarr.open_group(str(path), mode='r')
        require(sorted(bank.array_keys()) == keys, 'output schema changed')
        for name in keys:
            require(bank[name].shape == baseline[name].shape and bank[name].dtype == baseline[name].dtype,
                    'output layout differs')
            for start in range(0, baseline[name].shape[0], 512):
                if guard is not None:
                    guard()
                require(np.array_equal(bank[name][start:start+512], baseline[name][start:start+512]),
                        f'output values differ: {name}')
    medians = {v: {key: statistics.median(r[key] for r in runs if r['variant'] == v)
                   for key in ('producer_seconds', 'verification_seconds', 'producer_and_verification_seconds',
                               'session_setup_seconds', 'session_run_seconds', 'worker_seconds')}
               for v in ('A', 'B', 'C')}
    ratio = medians['B']['producer_and_verification_seconds'] / medians['C']['producer_and_verification_seconds']
    return {'status': 'PASS_EXACT_BT4_PIPELINE_SCREEN', 'medians': medians,
            'prefetch_over_optimized_serial': ratio,
            'prefetch_producer_only_ratio': medians['B']['producer_seconds'] / medians['C']['producer_seconds'],
            'prefetch_decision': 'PASS_5_PERCENT_SCREEN' if ratio >= 1 / .95 else 'NO_5_PERCENT_GAIN',
            'optimized_serial_over_original': medians['A']['producer_and_verification_seconds'] / medians['B']['producer_and_verification_seconds'],
            'scope': 'One banked shard, warm sessions, two observations per variant; no fleet/500M extrapolation.'}


def worker(plan: dict[str, Any], run_index: int) -> None:
    require(plan['status'] == 'FROZEN_REVIEWED_READY', 'worker plan unready')
    require(Path(plan['harness']['path']).resolve() == Path(__file__).resolve(), 'foreign harness path')
    pin(plan['harness'])
    lease_fd = int(os.environ['BT4_PIPELINE_GPU_LEASE_FD'])
    held = os.fstat(lease_fd)
    lock = Path(plan['gpu_lock']).stat()
    require((held.st_dev, held.st_ino) == (lock.st_dev, lock.st_ino), 'foreign inherited GPU lease')
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=5)
    require(not apps.strip(), 'GPU occupied before worker startup')
    variant_name = ORDER[run_index]
    variant = plan['variants'][variant_name]
    runtime_check(variant)
    for ref in plan['data_pins']:
        pin(ref)
    sys.path.insert(0, variant['runtime'])
    from scripts import bt4_raw_corpus_sidecar as raw
    from scripts.ceres_derived_sidecar import provider_proof
    import onnxruntime as ort
    from numcodecs.blosc import set_nthreads
    set_nthreads(2)
    require(sys.version == plan['python_version'], 'Python version changed')
    for package, version in plan['versions'].items():
        require(importlib.metadata.version(package) == version, f'runtime version changed: {package}')
    out = Path(plan['out']) / f'{run_index}_{variant_name}'
    out.mkdir()
    started = time.perf_counter()
    source = raw.load_sources([plan['source_id'] + '=' + plan['source']], out / 'unused_namespace')[0]
    shard = Path(plan['shard'])
    require(dict(zip((p.name for p in source.inventory.shards), source.inventory.shard_rows))[shard.name]
            == plan['rows'], 'selected source coverage changed')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.enable_profiling = True
    options.profile_file_prefix = str(out / 'ort_profile')
    tick = time.perf_counter()
    session = ort.InferenceSession(plan['onnx'], options,
        providers=[('CUDAExecutionProvider', {'device_id': 0, 'gpu_mem_limit': 8 * 2**30}), 'CPUExecutionProvider'],
        enable_fallback=False)
    session.disable_fallback()
    setup_seconds = time.perf_counter() - tick
    require(session.get_providers() == ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'provider fallback')
    actual = session.get_provider_options()['CUDAExecutionProvider']
    require(int(actual['device_id']) == 0 and int(actual['gpu_mem_limit']) == 8 * 2**30, 'CUDA options changed')
    require(len(session.get_inputs()) == 1 and session.get_inputs()[0].type in ('tensor(float16)', 'tensor(float)')
            and list(session.get_inputs()[0].shape[1:]) == [112, 8, 8], 'teacher input layout changed')
    input_name = session.get_inputs()[0].name
    dtype = np.dtype('float16' if session.get_inputs()[0].type == 'tensor(float16)' else 'float32')
    policy_name = session.get_outputs()[raw.resolve_policy_output(session, None)].name
    contract = raw.resolve_wdl_output(session, {'output': '/output/wdl', 'kind': 'probabilities'}, policy_name)
    if contract is None:
        raise RuntimeError('native WDL missing')
    import itertools
    with_rows = getattr(raw, 'iter_bt4_input_rows', raw.derive.iter_corpus_rows)
    iterator: Any = with_rows(shard)
    try:
        initial = list(itertools.islice(iterator, 128))
    finally:
        iterator.close()
    planes, _, _, _, _ = raw.encode_rows(initial, source=source)
    feed = raw.x_to_lc0_planes(planes, input_history_encoding=raw.derive.INPUT_HISTORY_ENCODING).astype(dtype)
    names = [policy_name, contract['output']]
    session.run(names, {input_name: feed})
    profile = Path(session.end_profiling())
    proof = provider_proof(json.loads(profile.read_text()))
    publish(out / 'provider.json', {**proof, 'scope': 'First batch128 warmup; same session/partition for timed calls.',
                                  'profile_sha256': sha(profile), 'options': actual})
    del initial, planes, feed
    digest = hashlib.sha256()
    output_hashes = [hashlib.sha256(), hashlib.sha256()]
    timings: list[float] = []
    batch_rows: list[int] = []
    main_thread = threading.get_ident()

    class Observed:
        def run(self, requested: list[str], feeds: dict[str, np.ndarray]) -> Any:
            require(threading.get_ident() == main_thread, 'inference left main thread')
            require(requested == names and list(feeds) == [input_name], 'inference endpoints changed')
            value = feeds[input_name]
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes(order='C'))
            batch_rows.append(len(value))
            tick = time.perf_counter()
            result = session.run(requested, feeds)
            timings.append(time.perf_counter() - tick)
            require(len(result) == 2, 'missing neural output')
            for h, array in zip(output_hashes, result, strict=True):
                if not isinstance(array, np.ndarray):
                    raise RuntimeError('non-dense neural output')
                h.update(str((array.dtype.str, array.shape)).encode())
                h.update(array.tobytes(order='C'))
            return result

    pending = raw.PendingShard(source, shard, plan['rows'], out / 'sidecar.zarr')
    kwargs = {'cpu_prefetch': True} if variant['prefetch'] else {}
    tick = time.perf_counter()
    raw.label_shard(pending, sess=Observed(), input_name=input_name, input_dtype=dtype,
        providers=session.get_providers(), policy_name=policy_name, onnx_path=Path(plan['onnx']),
        onnx_sha256=plan['onnx_sha256'], remap_stamp=raw.remap_provenance(), batch_size=128,
        wdl_output=contract, **kwargs)
    producer_seconds = time.perf_counter() - tick
    expected_batches = [min(128, plan['rows'] - offset) for offset in range(0, plan['rows'], 128)]
    require(batch_rows == expected_batches, 'inference batch boundaries changed')
    tick = time.perf_counter()
    raw.verify_shard(pending, onnx_sha256=plan['onnx_sha256'], expected_policy_output=policy_name,
        expected_providers=session.get_providers(), expected_remap=raw.functional_remap_identity(raw.remap_provenance()),
        batch_size=128, expected_wdl=contract)
    verification_seconds = time.perf_counter() - tick
    for ref in plan['data_pins']:
        pin(ref)
    publish(out / 'complete.json', {'status': 'PASS_VERIFIED_PIPELINE_RUN', 'variant': variant_name,
        'rows': plan['rows'], 'batch_rows': batch_rows, 'input_sequence_sha256': digest.hexdigest(),
        'raw_output_sha256': [h.hexdigest() for h in output_hashes], 'producer_seconds': producer_seconds,
        'verification_seconds': verification_seconds,
        'producer_and_verification_seconds': producer_seconds + verification_seconds,
        'session_setup_seconds': setup_seconds, 'session_run_seconds': sum(timings),
        'session_call_seconds': timings, 'worker_seconds': time.perf_counter() - started})


def execute(plan: dict[str, Any], plan_ref: dict[str, Any]) -> None:
    require(plan['status'] == 'FROZEN_REVIEWED_READY', 'plan is not ready')
    require(plan['order'] == list(ORDER) and plan['rows'] == 8236, 'registered design changed')
    require(Path(plan['harness']['path']).resolve() == Path(__file__).resolve(), 'foreign harness path')
    pin(plan['harness'])
    dependencies(plan)
    for variant in plan['variants'].values():
        runtime_check(variant)
    for ref in plan['data_pins']:
        pin(ref)
    out = Path(plan['out'])
    require(not out.exists(), 'fresh output required')
    started = time.monotonic()
    receipt: dict[str, Any] = {'status': 'INCOMPLETE', 'plan': plan_ref}
    with Path(plan['gpu_lock']).open('a') as lease:
        requested_stop: list[int] = []
        prior = {sig: signal.signal(sig, lambda s, _f: requested_stop.append(s))
                 for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=5)
            require(not apps.strip(), 'GPU is occupied')
            os.sched_setaffinity(0, {14, 15})
            os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
            out.mkdir(parents=True)

            last_resource_check = 0.0

            def guard() -> None:
                nonlocal last_resource_check
                require(not requested_stop, 'signal requested stop')
                require(time.monotonic() - started < 1140, '19-minute inner cap')
                require(not any(Path(p).exists() for p in plan['stop_paths']), 'STOP requested')
                now = time.monotonic()
                if now - last_resource_check < 2:
                    return
                last_resource_check = now
                available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                     if line.startswith('MemAvailable:'))) * 1024
                require(available >= 40 * 2**30, 'RAM floor')
                require(shutil.disk_usage(out).free >= 150 * 2**30, 'disk floor')
                require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) < 2 * 2**30, 'output size cap')
                used = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.used',
                           '--format=csv,noheader,nounits'], text=True, timeout=5).strip())
                require(used < 24576, 'GPU memory cap')

            guard()
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'PYTHONDONTWRITEBYTECODE': '1',
                   'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2',
                   'BT4_PIPELINE_GPU_LEASE_FD': str(lease.fileno())}
            for key in ('PYTHONOPTIMIZE', 'PYTHONHOME', 'LD_PRELOAD'):
                env.pop(key, None)
            runs = []
            roots = []
            for index, name in enumerate(ORDER):
                guard()
                env['PYTHONPATH'] = plan['variants'][name]['runtime']
                checked_child(['/usr/bin/python3', str(Path(__file__).resolve()), '--plan', plan_ref['path'],
                               '--sha256', plan_ref['sha256'], '--worker', str(index)],
                              Path(plan['variants'][name]['runtime']), env, out / f'{index}_{name}.log', guard, lease.fileno())
                run_out = out / f'{index}_{name}'
                result = json.loads((run_out / 'complete.json').read_text())
                require(result['status'] == 'PASS_VERIFIED_PIPELINE_RUN' and result['rows'] == plan['rows'],
                        'worker qualification failed')
                runs.append(result)
                roots.append(run_out / 'sidecar.zarr')
            guard()
            receipt.update(compare(runs, roots, guard), runs=runs)
            for variant in plan['variants'].values():
                runtime_check(variant)
                guard()
            for ref in plan['data_pins']:
                pin(ref)
                guard()
            guard()
        except BaseException as error:
            receipt.update(status='INCOMPLETE', error=repr(error))
            raise
        finally:
            for sig, handler in prior.items():
                signal.signal(sig, handler)
            receipt['elapsed_seconds'] = time.monotonic() - started
            publish(Path(plan['completion']), receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--worker', type=int, choices=range(6))
    args = parser.parse_args()
    ref = {'path': str(args.plan.resolve()), 'sha256': args.sha256}
    pin(ref)
    plan = json.loads(args.plan.read_text())
    if args.execute:
        execute(plan, ref)
    else:
        worker(plan, args.worker)


if __name__ == '__main__':
    main()
