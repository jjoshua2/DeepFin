"""Read-only runtime/import/admission qualification; never construct an ORT session."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    p = Path(path).resolve()
    s = p.stat()
    return {'path': str(p), **{k: getattr(s, k) for k in
            ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA must be hidden for this qualification')
    sys.path.insert(0, str(args.runtime))
    from scripts import bt4_raw_corpus_sidecar as raw
    from scripts import bt4_policy_dump as dump
    import numpy as np
    import torch
    import onnxruntime as ort
    import numcodecs.blosc as blosc
    import chess
    torch.set_num_threads(2)
    blosc.set_nthreads(2)
    functional = raw.functional_remap_identity(raw.remap_provenance())
    libraries = {}
    for name in ('numpy', 'torch', 'onnxruntime', 'zarr', 'numcodecs', 'chess',
                 'onnxruntime.capi.onnxruntime_pybind11_state'):
        module = importlib.import_module(name)
        libraries[name] = {'version': str(getattr(module, '__version__', None)),
                           'file': identity(module.__file__)}
    admissions = []
    project = Path('/home/josh/projects/chess')
    for name in ('run06_g10', 'run07_g10_companion4'):
        source = project / 'data/nnue_bootstrap' / name
        sidecars = project / 'data/lc0/bt4_policy_sidecars/g10_raw' / name
        manifest = json.loads((source / 'manifest.json').read_text())
        target = sidecars / 'w00-00000.bt4.zarr'
        attrs_path = target / '.zattrs'
        before = identity(attrs_path)
        attrs = json.loads(attrs_path.read_text())
        raw_path = source / attrs['source_shard']
        source_stat = identity(raw_path)
        inventory = raw.derive.ProgressInventory(shards=(raw_path,),
            shard_rows=(attrs['source_rows_claimed'],), rows_claimed=attrs['source_rows_claimed'],
            progress_files=(), torn_tail_files=(), unlisted_on_disk=())
        spec = raw.SourceSpec(name, source, sidecars, manifest, sha(source/'manifest.json'), inventory)
        pending = raw.PendingShard(spec, raw_path, attrs['source_rows_claimed'], target)
        readback = raw.validate_existing(pending,
            onnx_sha256='1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0',
            policy_output='/output/policy', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            functional_remap=functional)
        snapshot = project / 'scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_common_batch_v1' / f'{name}.closed_receipts.jsonl'
        with snapshot.open() as stream:
            banked = next(json.loads(line) for line in stream
                          if json.loads(line)['source_shard'] == raw_path.name)
        expected = raw.receipt_from_attrs(readback, target)
        if banked != expected:
            raise ValueError(f'banked closed receipt mismatch: {name}')
        if before != identity(attrs_path) or source_stat != identity(raw_path):
            raise ValueError('closed input changed during metadata admission')
        admissions.append({'source_id': name, 'sidecar': str(target), 'attrs_sha256': sha(attrs_path),
            'snapshot_sha256': sha(snapshot), 'source_stat': source_stat,
            'banked_receipt_exact': True, 'ordinary_validate_existing_pass': True})
    moves_check = []
    for fen in ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
                'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1'):
        board = chess.Board(fen)
        logits = np.linspace(-4, 3, raw.COMPACT_POLICY_SIZE, dtype=np.float32)
        ucis, probabilities = dump.legal_move_policy(board, logits)
        if hasattr(dump, 'legal_move_probabilities'):
            moves, reused = dump.legal_move_probabilities(board, logits)
            assert [m.uci() for m in moves] == ucis
            np.testing.assert_array_equal(reused, probabilities)
        moves_check.append({'fen': fen, 'uci': ucis, 'probabilities_sha256': raw.sha_array(probabilities)})
    if torch.cuda.is_initialized():
        raise ValueError('qualification initialized CUDA')
    result = {'status': 'PASS_IMPORTS_AND_METADATA_ADMISSION_NO_GPU_SESSION', 'runtime': str(args.runtime),
        'python': sys.version, 'interpreter': identity(sys.executable), 'interpreter_sha256': sha(sys.executable),
        'libraries': libraries, 'available_providers': ort.get_available_providers(),
        'functional_remap': functional, 'admissions': admissions, 'helper_smoke': moves_check,
        'torch_threads': torch.get_num_threads(), 'blosc_threads': blosc.get_nthreads(),
        'cuda_initialized': False, 'ort_session_created': False,
        'loaded_raw_path': raw.__file__, 'loaded_dump_path': dump.__file__}
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(result['status'], args.runtime)


if __name__ == '__main__':
    main()
