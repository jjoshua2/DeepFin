#!/usr/bin/env python3
"""Fixed32 C3 compact legal logits and primary WDL logits from qualified stored x.

Approximate research backend only; no value2, native value blend or strength claim.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
from typing import Any
from collections.abc import Callable

import numpy as np
import zarr
from numcodecs import Blosc
from numcodecs.blosc import set_nthreads

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chess_anti_engine.encoding import ceres_tpg as tpg
from chess_anti_engine.moves import leela_index as mapping
from scripts import bt4_derived_wdl_sidecar as shared
from scripts.sf_policy_rewrite import require

PROFILE = 'ceres-c3-fixed32-primary-compact-v1'
MODEL_SHA = '44aa02c775456f18ed464e33fc37b8e4abf58d7bf8f4cfb3ff19492e32e56df3'
BATCH = 32
COLUMNS = (*shared.COLUMNS, 'legal_mask', 'has_legal_mask')
DTYPES = {'legal_offsets': 'uint32', 'legal_indices': 'uint16', 'policy_logits': 'float16',
          'value_logits': 'float16', 'row_index': 'uint64', 'game_id': 'int64',
          'ply_index': 'int32', 'tpg_feed_sha256': 'uint8'}
CPU_SHAPE_OPS = {'Shape', 'Size', 'Gather', 'Unsqueeze', 'Squeeze', 'Concat', 'Slice',
                 'Cast', 'Reshape', 'Add', 'Mul', 'Div', 'Equal', 'Where'}
BACKEND: dict[str, Any] = {'onnxruntime': '1.29.0', 'numpy': '2.2.6', 'batch_size': 32,
           'optimization': 'extended', 'execution': 'sequential', 'threads': [2, 1],
           'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
           'provider_options': {'device_id': 0, 'gpu_mem_limit': 8 * 2**30,
                                'arena_extend_strategy': 'kSameAsRequested', 'use_tf32': 0},
           'input': ['squares_byte', 'uint8', [64, 137]],
           'outputs': ['policy', 'value'], 'native_dtype': 'float16',
           'value_order': ['win', 'draw', 'loss'], 'value_pov': 'side_to_move',
           'value_interpretation': 'primary raw logits; local head contract, no native blend'}


def validate_args(args: argparse.Namespace) -> None:
    require(args.batch_size == BATCH and args.threads == 2, 'requires fixed32 and two threads')
    require(args.gpu_mem_gb == 8 and args.gpu_lock and Path(args.gpu_lock).is_absolute(),
            'requires explicit shared GPU lock and accepted8GiB CUDA arena')
    require(args.expected_onnx_sha256 == MODEL_SHA, 'requires accepted C3 model hash')
    require(args.wdl_output == 'value' and args.wdl_output_kind == 'logits',
            'requires primary value logits only')


def provider_proof(events: list[dict[str, Any]]) -> dict[str, Any]:
    neural = 0
    cpu = []
    for event in events:
        args = event.get('args', {})
        provider = args.get('provider')
        if not provider:
            continue
        op = args.get('op_name')
        if provider == 'CUDAExecutionProvider':
            neural += int(op in ('Gemm', 'MatMul', 'FusedMatMul', 'FusedGemm'))
        elif provider == 'CPUExecutionProvider':
            outputs = args.get('output_type_shape')
            require(op in CPU_SHAPE_OPS and isinstance(outputs, list) and bool(outputs),
                    'unapproved CPU kernel')
            elements = 0
            for output in outputs:
                require(isinstance(output, dict) and bool(output), 'missing CPU output type')
                for dtype, shape in output.items():
                    require(dtype in ('int64', 'int32', 'bool') and isinstance(shape, list)
                            and all(type(n) is int and n >= 0 for n in shape),
                            'CPU floating or dynamic compute')
                    elements += math.prod(shape)
            require(elements <= 4096, 'large CPU shape output')
            cpu.append({'op': op, 'outputs': outputs})
        else:
            raise ValueError('unexpected profiled provider')
    require(neural > 0, 'no CUDA neural compute proof')
    return {'CUDA_neural_kernel_events': neural, 'CPU_shape_events': cpu,
            'scope': 'First fixed32 call in this session; later calls share graph partition.'}


def open_teacher(args: argparse.Namespace) -> Any:
    import onnxruntime as ort
    require(ort.__version__ == BACKEND['onnxruntime'] and np.__version__ == BACKEND['numpy'],
            'unqualified collector runtime versions')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'requires registered GPU0 visibility')
    require('CUDAExecutionProvider' in ort.get_available_providers(), 'CUDA unavailable')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.enable_profiling = True
    options.profile_file_prefix = str(Path(args.invocation) / 'ort_profile')
    session = ort.InferenceSession(args.onnx, sess_options=options,
        providers=[('CUDAExecutionProvider', BACKEND['provider_options']), 'CPUExecutionProvider'],
        enable_fallback=False)
    session.disable_fallback()
    require(session.get_providers() == BACKEND['providers'], 'session fallback')
    realized = session.get_provider_options()['CUDAExecutionProvider']
    require(all(str(realized[k]) == str(v) for k, v in BACKEND['provider_options'].items()),
            'CUDA provider options differ')
    inputs = session.get_inputs()
    require(len(inputs) == 1 and inputs[0].name == 'squares_byte'
            and inputs[0].type == 'tensor(uint8)' and list(inputs[0].shape[1:]) == [64, 137],
            'teacher input differs')
    outputs = {x.name: x for x in session.get_outputs()}
    for name, width in [('policy', 1858), ('value', 3)]:
        require(name in outputs and outputs[name].type == 'tensor(float16)'
                and len(outputs[name].shape) == 2 and outputs[name].shape[-1] == width,
                'teacher output differs')
    shared.raw.atomic_json(Path(args.invocation) / 'session.json',
        {'backend': BACKEND, 'providers': session.get_providers(), 'options': realized,
         'python': sys.version, 'executable': sys.executable, 'ort_build': ort.get_build_info(),
         'qualification': 'Session created; first-call provider proof still required'})
    return session


def namespace(args: argparse.Namespace) -> dict[str, Any]:
    return {'profile': PROFILE, 'source': str(Path(args.source).resolve()),
            'summary_sha256': args.expected_source_summary_sha256,
            **shared.g10_binding(args), 'model_sha256': args.expected_onnx_sha256,
            'backend': BACKEND,
            'producer': {str(Path(p).resolve().relative_to(Path(__file__).resolve().parents[1])):
                         shared.file_sha256(p) for p in (__file__, shared.__file__,
                         tpg.__file__, mapping.__file__)}}


def expected_binding(args: argparse.Namespace, spec: dict[str, Any], state: str) -> dict[str, Any]:
    return {**namespace(args), 'shard': spec['path'], 'rows': spec['rows'],
            'source_storage_identity': state}


def verify_cached(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    state = shared.storage_identity(path)
    group: Any = zarr.open_group(str(path), mode='r')
    attrs = dict(group.attrs)
    require(attrs.get('complete') is True and attrs.get('binding') == expected,
            'completed Ceres binding differs')
    n = expected['rows']
    require(n > 0 and n % BATCH == 0 and set(group.array_keys()) == set(DTYPES),
            'Ceres array inventory/count differs')
    arrays = {}
    for key, dtype in DTYPES.items():
        array = group[key]
        require(array.dtype == np.dtype(dtype), 'native Ceres array dtype differs')
        shared.complete_chunks(array)
        arrays[key] = np.asarray(array[:])
    offsets = arrays['legal_offsets']
    require(bool(offsets.shape == (n + 1,) and offsets[0] == 0
            and np.all(np.diff(offsets.astype('int64')) > 0)), 'invalid legal offsets')
    count = int(offsets[-1])
    shapes = {'legal_offsets': (n + 1,), 'legal_indices': (count,), 'policy_logits': (count,),
              'value_logits': (n, 3), 'row_index': (n,), 'game_id': (n,),
              'ply_index': (n,), 'tpg_feed_sha256': (n, 32)}
    require(all(arrays[k].shape == shape for k, shape in shapes.items()), 'Ceres array shape differs')
    require(np.array_equal(arrays['row_index'], np.arange(n)), 'Ceres row order differs')
    require(bool(np.all(arrays['game_id'] >= 0) and np.all(arrays['ply_index'] >= 0)), 'negative identity')
    require(np.isfinite(arrays['policy_logits']).all() and np.isfinite(arrays['value_logits']).all(),
            'nonfinite Ceres logits')
    for i in range(n):
        indices = arrays['legal_indices'][int(offsets[i]):int(offsets[i + 1])]
        require(bool(np.all(indices < 1858) and np.all(np.diff(indices.astype('int64')) > 0)),
                'invalid compact legal roster')
    require({k: shared.raw.sha_array(v) for k, v in arrays.items()} == attrs['array_sha256'],
            'Ceres array digest differs')
    require(set(attrs['source_array_sha256']) == set(COLUMNS), 'source content proof absent')
    proof = attrs['provider_proof']
    require(proof['CUDA_neural_kernel_events'] > 0, 'provider proof absent')
    require(shared.storage_identity(path) == state, 'Ceres bank changed during verification')
    return attrs


def label_shard(args: argparse.Namespace, spec: dict[str, Any], summary: dict[str, Any],
                session: Any, guard: Callable[[], None], qualify: Callable[[], dict[str, Any]]) -> None:
    source = Path(args.source) / spec['path']
    state = shared.storage_identity(source)
    n = spec['rows']
    require(0 < n <= 8192 and n % BATCH == 0, 'partial/oversized fixed32 pilot shard refused; no padding/drop')
    group = shared.source_arrays(source, summary, n)
    for key in ('legal_mask', 'has_legal_mask'):
        require(key in group, 'missing source legal mask')
        shared.complete_chunks(group[key])
    legal = np.asarray(group['legal_mask'][:])
    require(bool(legal.shape == (n, 1858) and group['has_legal_mask'].shape == (n,)
            and np.all((legal == 0) | (legal == 1))
            and np.all(np.asarray(group['has_legal_mask'][:]) == 1)), 'invalid source legal mask')
    counts = (legal != 0).sum(axis=1, dtype=np.uint64)
    require(bool(np.all(counts > 0) and int(counts.sum()) < 2**32), 'empty or excessive legal roster')
    offsets = np.r_[np.uint64(0), np.cumsum(counts)].astype('uint32')
    payload = {'legal_offsets': offsets,
               'legal_indices': np.nonzero(legal)[1].astype('uint16'),
               'policy_logits': np.empty(int(offsets[-1]), dtype='float16'),
               'value_logits': np.empty((n, 3), dtype='float16'),
               'row_index': np.arange(n, dtype='uint64'),
               'game_id': np.empty(n, dtype='int64'), 'ply_index': np.empty(n, dtype='int32'),
               'tpg_feed_sha256': np.empty((n, 32), dtype='uint8')}
    hashes = {k: hashlib.sha256() for k in COLUMNS}
    proof = None
    for start in range(0, n, BATCH):
        guard()
        end = start + BATCH
        batch = {k: np.asarray(group[k][start:end]) for k in COLUMNS}
        for k, digest in hashes.items():
            digest.update(batch[k].tobytes(order='C'))
        require(bool(np.all(batch['has_game_id'] == 1) and np.all(batch['has_ply_index'] == 1)),
                'missing source row identity')
        for name, dtype in [('game_id', np.int64), ('ply_index', np.int32)]:
            require(bool(np.all(batch[name] >= 0) and np.all(batch[name] <= np.iinfo(dtype).max)),
                    'source identity outside storage range')
            payload[name][start:end] = batch[name]
        feed = tpg.stored_x_to_ceres_tpg_bytes(batch['x'], input_history_encoding=shared.HISTORY,
                                             history_rep_fix=True)
        fetched = session.run(['policy', 'value'], {'squares_byte': feed})
        require(len(fetched) == 2, 'missing teacher outputs')
        policy, value = fetched
        require(policy.dtype == value.dtype == np.dtype('float16')
                and policy.shape == (BATCH, 1858) and value.shape == (BATCH, 3)
                and np.isfinite(policy).all() and np.isfinite(value).all(), 'invalid native outputs')
        proof = qualify()
        gather = mapping.leela_gather_indices(*tpg.ceres_tpg_gather_context(feed))
        for local, row in enumerate(range(start, end)):
            lo, hi = int(offsets[row]), int(offsets[row + 1])
            indices = payload['legal_indices'][lo:hi]
            slots = gather[local, indices]
            require(bool(np.all((slots >= 0) & (slots < 1858)) and len(set(slots.tolist())) == len(slots)),
                    'invalid legal Leela mapping')
            payload['policy_logits'][lo:hi] = policy[local, slots]
        payload['value_logits'][start:end] = value
        payload['tpg_feed_sha256'][start:end] = shared.row_digests(feed)
    guard()
    require(shared.storage_identity(source) == state, 'source changed during Ceres collection')
    target = Path(args.out) / spec['path']
    writing = target.with_name(target.name + '.writing')
    require(not os.path.lexists(target) and not os.path.lexists(writing), 'output/partial exists')
    result: Any = zarr.open_group(str(writing), mode='w')
    for name, values in payload.items():
        result.create_dataset(name, data=values, chunks=(min(len(values), 512), *values.shape[1:]),
                              compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE))
    result.attrs.update(complete=True, binding=expected_binding(args, spec, state),
        array_sha256={k: shared.raw.sha_array(v) for k, v in payload.items()},
        source_array_sha256={k: v.hexdigest() for k, v in hashes.items()}, provider_proof=proof,
        history_lineage='Inherited stored history and source qualification; no raw replay or native repetition oracle.')
    verify_cached(writing, expected_binding(args, spec, state))
    guard()
    require(shared.storage_identity(source) == state, 'source changed before publication')
    os.replace(writing, target)


def produce(args: argparse.Namespace) -> None:
    validate_args(args)
    with shared.raw.advisory_lease(Path(args.out) / '.writer.lock', poll_seconds=1,
                                   description='Ceres writer'):
        summary, specs = shared.source_inventory(args)
        require(all(0 < s['rows'] <= 8192 and s['rows'] % BATCH == 0 for s in specs),
                'fixed32 pilot requires divisible shards of at most8192 rows; no padding/drop')
        marker = Path(args.out) / 'ceres_source.json'
        wanted = namespace(args)
        if marker.exists():
            require(json.loads(marker.read_text()) == wanted, 'Ceres source namespace differs')
        else:
            require(not list(Path(args.out).glob('shard_*.zarr*'))
                    and not (Path(args.out) / 'g10_common_source.json').exists(),
                    'existing foreign or unbound output namespace')
            shared.raw.atomic_json(marker, wanted)
        shared.guard_resources(args)
        model_state = shared.storage_identity(Path(args.onnx))
        require(shared.file_sha256(args.onnx) == MODEL_SHA, 'teacher model differs')
        require(shared.storage_identity(Path(args.onnx)) == model_state, 'model changed during hashing')
        todo = []
        for spec in specs:
            target = Path(args.out) / spec['path']
            require(not target.with_name(target.name + '.writing').exists(), 'partial Ceres shard exists')
            if target.exists():
                source_state = shared.storage_identity(Path(args.source) / spec['path'])
                verify_cached(target, expected_binding(args, spec, source_state))
            else:
                todo.append(spec)

        def guard() -> None:
            shared.guard_resources(args)
            require(shared.storage_identity(Path(args.onnx)) == model_state, 'model changed')
            require(json.loads(marker.read_text()) == wanted, 'Ceres namespace changed')
            shared.check_g10_pin(args)

        if todo:
            gpu_path = Path(args.gpu_lock)
            gpu_path.parent.mkdir(parents=True, exist_ok=True)
            gpu_fd = os.open(gpu_path, os.O_CREAT | os.O_RDWR, 0o600)
            # Child owns this descriptor until process exit, including CUDA teardown.
            fcntl.flock(gpu_fd, fcntl.LOCK_EX)
            guard()
            session = open_teacher(args)
            proof: dict[str, Any] | None = None

            def qualify() -> dict[str, Any]:
                nonlocal proof
                require(session.get_providers() == BACKEND['providers'], 'provider fallback during inference')
                if proof is None:
                    profile = Path(session.end_profiling())
                    proof = provider_proof(json.loads(profile.read_text()))
                    proof['profile_sha256'] = shared.file_sha256(profile)
                    shared.raw.atomic_json(Path(args.invocation) / 'provider_proof.json', proof)
                return proof

            for spec in todo:
                label_shard(args, spec, summary, session, guard, qualify)
        guard()
        require(shared.file_sha256(Path(args.source) / shared.SUMMARY)
                == args.expected_source_summary_sha256, 'source summary changed')
        if getattr(args, 'g10_admission', None) is not None:
            current = shared.g10.admit(Path(args.g10_common_qualification),
                args.expected_g10_common_qualification_sha256, Path(args.source), summary)
            require(shared.g10.same(current, args.g10_admission), 'G10 admission changed')
        for spec in specs:
            attrs = dict(zarr.open_group(str(Path(args.out) / spec['path']), mode='r').attrs)
            require(shared.storage_identity(Path(args.source) / spec['path'])
                    == attrs['binding']['source_storage_identity'], 'source changed before completion')
        guard()
        shared.raw.atomic_json(Path(args.invocation) / 'child_completed.json',
            {'schema': 1, 'complete': True, 'profile': PROFILE, 'selection': specs,
             'rows': sum(s['rows'] for s in specs), 'shards': len(specs), 'new_shards': len(todo),
             'namespace': wanted, 'scope': 'Compact primary Ceres teacher bank only; no training qualification'})


def child(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    validate_args(args)
    set_nthreads(2)
    shared.raw.atomic_json(Path(args.invocation) / 'child_started.json',
        {'pid': os.getpid(), 'profile': PROFILE, 'affinity': sorted(os.sched_getaffinity(0)),
         'blosc_threads': shared.get_nthreads(), 'threads': 2})
    produce(args)


def build_parser() -> argparse.ArgumentParser:
    parser = shared.build_parser()
    parser.description = __doc__
    parser.set_defaults(batch_size=BATCH, gpu_mem_gb=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    return shared.run(args, child_target=child)


if __name__ == '__main__':
    raise SystemExit(main())
