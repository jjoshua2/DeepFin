"""One closed raw shard, matched BT4 batching, no production-bank mutation."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import bt4_raw_corpus_sidecar as raw


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(2**20), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def provider_proof(events, batch=32):
    # Ceres and BT4 share the ONNX provider contract: only bounded integer
    # shape work may remain on CPU, and CUDA must execute neural kernels.
    from scripts.ceres_derived_sidecar import provider_proof as validate_placement
    result = validate_placement(events)
    return {**result, 'batch_size': batch,
            'scope': f'First batch{batch} call; later calls share graph partition.'}



def worker(plan, batch):
    import onnxruntime as ort
    from numcodecs.blosc import set_nthreads
    set_nthreads(2)
    out = Path(plan['out']) / str(batch)
    out.mkdir()
    start = time.perf_counter()
    for ref in plan['data_pins']:
        assert sha(ref['path']) == ref['sha256'], ref['path']
    source = raw.load_sources([plan['source_id'] + '=' + plan['source']], out / 'unused_namespace')[0]
    shard = Path(plan['shard'])
    assert dict(zip((p.name for p in source.inventory.shards), source.inventory.shard_rows))[shard.name] == plan['rows']
    t = time.perf_counter()
    rows = list(raw.derive.iter_corpus_rows(shard))
    assert len(rows) == plan['rows']
    decode_seconds = time.perf_counter() - t
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.enable_profiling = True
    options.profile_file_prefix = str(out / 'ort_profile')
    t = time.perf_counter()
    session = ort.InferenceSession(plan['onnx'], options,
        providers=[('CUDAExecutionProvider', {'device_id': 0, 'gpu_mem_limit': 8 * 2**30}), 'CPUExecutionProvider'],
        enable_fallback=False)
    session.disable_fallback()
    setup_seconds = time.perf_counter() - t
    assert session.get_providers() == ['CUDAExecutionProvider', 'CPUExecutionProvider']
    actual = session.get_provider_options()['CUDAExecutionProvider']
    assert int(actual['gpu_mem_limit']) == 8 * 2**30
    assert int(actual['device_id']) == 0
    input_name = session.get_inputs()[0].name
    dtype = np.float16 if session.get_inputs()[0].type == 'tensor(float16)' else np.float32
    policy_name = session.get_outputs()[raw.resolve_policy_output(session, None)].name
    contract = raw.resolve_wdl_output(session, {'output': '/output/wdl', 'kind': 'probabilities'}, policy_name)
    assert contract is not None, 'Required WDL output missing'
    t = time.perf_counter()
    feeds = []
    for offset in range(0, len(rows), 512):
        planes, _, _, _, _ = raw.encode_rows(rows[offset:offset+512], source=source)
        feeds.append(raw.x_to_lc0_planes(planes, input_history_encoding=raw.derive.INPUT_HISTORY_ENCODING).astype(dtype))
        del planes
    feed = np.concatenate(feeds)
    del feeds, rows
    prepare_seconds = time.perf_counter() - t
    names = [policy_name, contract['output']]
    session.run(names, {input_name: feed[:batch]})
    profile_path = Path(session.end_profiling())
    proof = provider_proof(json.loads(profile_path.read_text()), batch=batch)
    write(out / 'provider_proof.json', {**proof, 'profile_sha256': sha(profile_path), 'batch': batch,
        'providers': session.get_providers(), 'provider_options': actual, 'ort_version': ort.__version__})
    timings = []
    outputs = [[], []]
    for offset in range(0, len(feed), batch):
        t = time.perf_counter()
        result = session.run(names, {input_name: feed[offset:offset+batch]})
        timings.append(time.perf_counter() - t)
        for i, array in enumerate(result):
            assert isinstance(array, np.ndarray), 'Expected dense neural output'
            assert np.isfinite(array).all()
            outputs[i].append(array)
    for name, arrays in zip(('policy_raw', 'wdl_raw'), outputs):
        np.save(out / (name + '.npy'), np.concatenate(arrays))
    del feed, outputs
    gc.collect()
    stages = {'encode_rows': 0., 'fingerprints': 0., 'lc0_conversion': 0., 'legal_policy': 0., 'session_run': 0.}
    for attr, key in [('encode_rows', 'encode_rows'), ('position_fingerprints', 'fingerprints'),
                      ('x_to_lc0_planes', 'lc0_conversion'), ('legal_move_probabilities', 'legal_policy')]:
        original = getattr(raw, attr)
        def wrapped(*args, _fn=original, _key=key, **kwargs):
            t = time.perf_counter()
            try:
                return _fn(*args, **kwargs)
            finally:
                stages[_key] += time.perf_counter() - t
        setattr(raw, attr, wrapped)
    class Observed:
        def run(self, *args, **kwargs):
            t = time.perf_counter()
            try:
                return session.run(*args, **kwargs)
            finally:
                stages['session_run'] += time.perf_counter() - t
    pending = raw.PendingShard(source, shard, plan['rows'], out / 'sidecar.zarr')
    t = time.perf_counter()
    raw.label_shard(pending, sess=Observed(), input_name=input_name, input_dtype=np.dtype(dtype),
        providers=session.get_providers(), policy_name=policy_name, onnx_path=Path(plan['onnx']),
        onnx_sha256=plan['onnx_sha256'], remap_stamp=raw.remap_provenance(), batch_size=batch,
        wdl_output=contract)
    end_to_end = time.perf_counter() - t
    for ref in plan['data_pins']:
        assert sha(ref['path']) == ref['sha256']
    write(out / 'complete.json', {'status': 'COMPLETE_BATCH', 'batch': batch, 'rows': plan['rows'],
        'decode_seconds': decode_seconds, 'prepare_seconds': prepare_seconds, 'session_setup_seconds': setup_seconds,
        'inference_seconds': sum(timings), 'inference_rows_per_second': plan['rows']/sum(timings),
        'end_to_end_seconds': end_to_end, 'end_to_end_rows_per_second': plan['rows']/end_to_end,
        'stages': stages, 'other_decode_map_write_seconds': end_to_end-sum(stages.values()),
        'worker_seconds': time.perf_counter()-start, 'wdl_contract': contract,
        'scope': 'One ordered screen; end-to-end excludes session startup and bank discovery.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--batch', type=int, choices=[128, 256, 512], required=True)
    args = p.parse_args()
    worker(json.loads(Path(args.plan).read_text()), args.batch)
