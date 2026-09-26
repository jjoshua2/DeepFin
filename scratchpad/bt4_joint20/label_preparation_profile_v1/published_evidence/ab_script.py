"""One old/new CPU comparison of source-extracted actual labeler loops."""
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

STATE = Path(__file__).resolve().parent
PLAN = STATE / 'plan.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_function(path, scope):
    tree = ast.parse(path.read_text())
    label = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'label_shard')
    batch = next(node for node in label.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_batch')
    loops = [node for node in batch.body if isinstance(node, ast.For)]
    if len(loops) != 1 or ast.unparse(loops[0].iter) != 'enumerate(boards)':
        raise ValueError('unexpected actual labeler loop shape')
    function = ast.parse('def project(boards, output):\n    cursor = 0\n    entropy_sum = top1_sum = 0.0\n    legal_moves_sum = 0\n    bt4_policy = np.zeros((len(boards), COMPACT_POLICY_SIZE), dtype=np.float32)\n').body[0]
    function.body.extend([loops[0], ast.parse('return bt4_policy, entropy_sum, top1_sum, legal_moves_sum').body[0]])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), scope)
    return scope['project']


def run():
    plan = json.loads(PLAN.read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or set(os.sched_getaffinity(0)) != {0}:
        raise ValueError('CPU/GPU bounds differ')
    for name, digest in plan['pins'].items():
        if sha(name) != digest:
            raise ValueError(f'changed input: {name}')
    sys.path.insert(0, plan['runtime'])
    from scripts import bt4_raw_corpus_sidecar as raw
    from scripts import bt4_policy_dump as dump
    import chess
    import numpy as np
    import numcodecs.blosc as blosc
    import torch
    torch.set_num_threads(2)
    blosc.set_nthreads(2)
    source = Path(plan['source'])

    def identity():
        s = source.stat()
        return {k: getattr(s, k) for k in plan['source_stat']}

    if identity() != plan['source_stat']:
        raise ValueError('source stat changed')
    iterator = raw.derive.iter_corpus_rows(source)
    try:
        rows = list(itertools.islice(iterator, 1024))
    finally:
        iterator.close()
    manifest = json.loads((source.parent / 'manifest.json').read_text())
    spec = raw.SourceSpec('run06_g10', source.parent, STATE / 'unused', manifest,
        sha(source.parent / 'manifest.json'), raw.derive.ProgressInventory(
            shards=(source,), shard_rows=(8236,), rows_claimed=8236,
            progress_files=(), torn_tail_files=(), unlisted_on_disk=()))
    planes, boards, keys, games, plies = raw.encode_rows(rows, source=spec)
    prior = json.loads(Path(plan['prior_completed']).read_text())
    for name, array in [('planes_float32', planes), ('input_keys', keys), ('games', games), ('plies', plies)]:
        if raw.sha_array(array) != prior['identity_hashes'][name]:
            raise ValueError(f'profile input bank mismatch: {name}')
    if len(rows) != 1024:
        raise ValueError('short bank')
    fixed_logits = np.broadcast_to(np.linspace(-4, 3, raw.COMPACT_POLICY_SIZE, dtype=np.float32),
                                   (1024, raw.COMPACT_POLICY_SIZE)).copy()
    namespace = {'np': np, 'chess': chess, 'compact_index_for_move': raw.compact_index_for_move,
                 'COMPACT_POLICY_SIZE': raw.COMPACT_POLICY_SIZE, 'pending': SimpleNamespace(path=source)}
    old_scope = {**namespace, 'leela_index_for_move': dump.leela_index_for_move}
    old_tree = ast.parse((STATE / 'baseline_dump.py').read_text())
    old_helper = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'legal_move_policy')
    exec(compile(ast.Module(body=[old_helper], type_ignores=[]), str(STATE / 'baseline_dump.py'), 'exec'), old_scope)
    old = project_function(STATE / 'baseline_raw.py', old_scope)
    new_scope = {**namespace, 'leela_index_for_move': dump.leela_index_for_move}
    new_tree = ast.parse((Path(plan['candidate']) / 'scripts/bt4_policy_dump.py').read_text())
    new_helper = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'legal_move_probabilities')
    exec(compile(ast.Module(body=[new_helper], type_ignores=[]), str(Path(plan['candidate']) / 'scripts/bt4_policy_dump.py'), 'exec'), new_scope)
    new = project_function(Path(plan['candidate']) / 'scripts/bt4_raw_corpus_sidecar.py', new_scope)
    results = {}
    outputs = []
    # Fixed order, one invocation each. No warmup, repetitions or cProfile.
    for name, function in [('old', old), ('new', new)]:
        wall, cpu = time.perf_counter(), time.process_time()
        output = function(boards, fixed_logits)
        results[name] = {'wall_seconds': time.perf_counter() - wall,
                         'cpu_seconds': time.process_time() - cpu}
        outputs.append(output)
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    if outputs[0][1:] != outputs[1][1:]:
        raise ValueError('aggregate statistics changed')
    if identity() != plan['source_stat']:
        raise ValueError('source changed during comparison')
    for name, digest in plan['pins'].items():
        if sha(name) != digest:
            raise ValueError(f'input changed during comparison: {name}')
    receipt = {'status': 'PASS_EXACT_CPU_POSTPROCESSING_PARITY', 'rows': len(rows), 'timings': results,
        'relative_wall_reduction': 1 - results['new']['wall_seconds'] / results['old']['wall_seconds'],
        'old_over_new_wall': results['old']['wall_seconds'] / results['new']['wall_seconds'],
        'policy_sha256': raw.sha_array(outputs[0][0]), 'fixed_logits_sha256': raw.sha_array(fixed_logits),
        'entropy_sum': outputs[0][1], 'top1_sum': outputs[0][2], 'legal_moves_sum': outputs[0][3],
        'source_stat_after': identity(), 'plan_sha256': sha(PLAN), 'runtime': {'python': sys.version,
        'numpy': np.__version__, 'torch': torch.__version__, 'torch_threads': torch.get_num_threads(),
        'blosc_threads': blosc.get_nthreads(), 'affinity': sorted(os.sched_getaffinity(0)),
        'cuda_initialized': torch.cuda.is_initialized()},
        'limitations': 'Single old-then-new observation, no repetition, fixed artificial logits shared across rows. Exact actual postprocessing loops only; excludes JSON/history, model inference, serialization and end-to-end throughput. Order/cache effects are uncontrolled; no statistical speed claim.'}
    with (STATE / 'run/completed.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    run()
