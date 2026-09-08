"""One fixed CPU-only input-preparation profile; no session, labels or writes to inputs."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import pstats
import stat
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    s = Path(path).stat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError(f"not a regular input: {path}")
    return {key: getattr(s, key) for key in
            ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def run(plan):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA must be hidden')
    if plan['rows'] != 1024 or len(os.sched_getaffinity(0)) > 2:
        raise ValueError('fixed row/CPU bound differs')
    for path, pin in plan['pins'].items():
        if sha(path) != pin:
            raise ValueError(f'changed pinned input: {path}')
    source = Path(plan['source']['path'])
    before = identity(source)
    if before != plan['source']['stat']:
        raise ValueError('raw source identity changed since preparation')
    sys.path.insert(0, plan['runtime'])
    # Same imported functions as label_shard; imports happen outside profile timing.
    from scripts import bt4_raw_corpus_sidecar as raw
    import chess
    import numpy as np
    import numcodecs.blosc as blosc
    import torch
    from chess_anti_engine.moves.leela_index import leela_index_for_move
    from chess_anti_engine.encoding import rep_fix
    threads_before = {'torch': torch.get_num_threads(), 'blosc': blosc.get_nthreads()}
    torch.set_num_threads(2)
    blosc.set_nthreads(2)
    manifest_path = Path(plan['source']['manifest'])
    manifest = json.loads(manifest_path.read_text())
    attrs = json.loads(Path(plan['source']['closed_sidecar_attrs']).read_text())
    if (manifest['row_schema'] != raw.corpus.ROW_SCHEMA
            or manifest['config_sha256'] != attrs['source_config_sha256']
            or attrs['source_shard'] != source.name
            or attrs['source_sha256'] != plan['source']['banked_sha256']
            or attrs['source_file_bytes'] != before['st_size']
            or attrs['source_file_mtime_ns'] != before['st_mtime_ns']):
        raise ValueError('closed-shard source/schema evidence mismatch')
    inventory = raw.derive.ProgressInventory(
        shards=(source,), shard_rows=(attrs['source_rows_claimed'],),
        rows_claimed=attrs['source_rows_claimed'], progress_files=(),
        torn_tail_files=(), unlisted_on_disk=(),
    )
    spec = raw.SourceSpec('run06_g10', source.parent, Path(plan['output']),
                          manifest, sha(manifest_path), inventory)
    out = Path(plan['output'])
    write(out / 'started.json', {'pid': os.getpid(), 'argv': sys.argv,
                                'start_unix': time.time(), 'source_before': before})
    stages = {}
    profiler = cProfile.Profile()

    def measured(name, fn):
        wall, cpu = time.perf_counter(), time.process_time()
        value = fn()
        stages[name] = {'wall_seconds': time.perf_counter() - wall,
                        'cpu_seconds': time.process_time() - cpu}
        return value

    try:
        profiler.enable()

        def decode():
            iterator = raw.derive.iter_corpus_rows(source)
            try:
                return list(itertools.islice(iterator, plan['rows']))
            finally:
                iterator.close()

        rows = measured('decode_zstd_utf8_json', decode)
        if len(rows) != plan['rows']:
            raise ValueError('short source prefix')
        planes, boards, keys, games, plies = measured('history_encode_and_key_validation',
                                                     lambda: raw.encode_rows(rows, source=spec))
        fingerprints = measured('position_fingerprints', lambda: raw.position_fingerprints(
            planes, input_history_encoding=raw.derive.INPUT_HISTORY_ENCODING))
        features = measured('convert_lc0_planes', lambda: raw.x_to_lc0_planes(
            planes, input_history_encoding=raw.derive.INPUT_HISTORY_ENCODING))

        def legal_mapping():
            # The three actual mapping passes surrounding legal_move_policy in
            # label_shard. No fabricated logits or softmax/probability assertions.
            result = []
            for board in boards:
                ucis = [move.uci() for move in board.legal_moves]
                teacher = [leela_index_for_move(board, chess.Move.from_uci(u)) for u in ucis]
                compact = [raw.compact_index_for_move(board, chess.Move.from_uci(u)) for u in ucis]
                expected = {raw.compact_index_for_move(board, move) for move in board.legal_moves}
                if (any(i < 0 for i in teacher) or len(compact) != len(expected)
                        or len(set(compact)) != len(compact) or set(compact) != expected
                        or any(i < 0 or i >= raw.COMPACT_POLICY_SIZE for i in compact)):
                    raise ValueError('legal map mismatch')
                result.append({'uci': ucis, 'teacher': teacher, 'compact': compact})
            return result

        mappings = measured('legal_mapping_only_no_logits', legal_mapping)
        profiler.disable()
        if identity(source) != before:
            raise ValueError('source changed while reading prefix')
        for path, pin in plan['pins'].items():
            if sha(path) != pin:
                raise ValueError(f'pinned input changed: {path}')
        namespace = hashlib.sha256(json.dumps(
            [str(source.parent.resolve()), manifest['config_sha256']], separators=(',', ':')
        ).encode()).hexdigest()
        with (out / 'rows.jsonl').open('x') as stream:
            for i, row in enumerate(rows):
                stream.write(json.dumps({'source_namespace': namespace, 'source_shard': source.name,
                    'source_row': i, 'worker_id': row['worker_id'], 'game_id': int(games[i]),
                    'ply': int(plies[i]), 'input_key': keys[i].tobytes().hex(),
                    'fingerprint': fingerprints[i].hex()}, sort_keys=True) + '\n')
        write(out / 'legal_mapping.json', mappings)
        profiler.dump_stats(str(out / 'profile.pstats'))
        text = io.StringIO()
        pstats.Stats(profiler, stream=text).strip_dirs().sort_stats('cumulative').print_stats(60)
        (out / 'profile.txt').write_text(text.getvalue())
        stats = pstats.Stats(profiler)
        functions = [{'file': key[0], 'line': key[1], 'function': key[2], 'primitive_calls': value[0],
                      'total_calls': value[1], 'self_seconds': value[2], 'cumulative_seconds': value[3]}
                     for key, value in stats.stats.items()]
        write(out / 'functions.json', sorted(functions, key=lambda f: -f['cumulative_seconds']))
        write(out / 'completed.json', {'status': 'COMPLETE_CPU_SUBSET_PROFILE', 'rows': len(rows),
            'end_unix': time.time(), 'stages': stages, 'source_before': before,
            'source_after': identity(source), 'banked_source_sha256': plan['source']['banked_sha256'],
            'source_hash_scope': 'Banked closed-shard SHA and prior completed source check; this run checks frozen stat identity and original input keys for the first1024 rows, not a fresh whole-file SHA.',
            'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                'affinity': sorted(os.sched_getaffinity(0)), 'threads_before': threads_before,
                'threads_after': {'torch': torch.get_num_threads(), 'blosc': blosc.get_nthreads()},
                'rep_fix_process_state': rep_fix.current(), 'history_encoding': raw.derive.INPUT_HISTORY_ENCODING,
                'extra_features': raw.derive.INPUT_EXTRA_FEATURES, 'row_schema': raw.corpus.ROW_SCHEMA},
            'identity_hashes': {'planes_float32': raw.sha_array(planes), 'lc0_planes': raw.sha_array(features),
                'input_keys': raw.sha_array(keys), 'games': raw.sha_array(games), 'plies': raw.sha_array(plies),
                'rows_jsonl': sha(out / 'rows.jsonl'), 'legal_mapping': sha(out / 'legal_mapping.json')},
            'limitations': 'One sequential prefix, cProfile overhead, no GPU/inference/softmax/sidecar writes or full labeler timing. No throughput or pipeline-bottleneck conclusion.'})
    except BaseException as exc:
        profiler.disable()
        profiler.dump_stats(str(out / 'failed_profile.pstats'))
        write(out / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
                                   'stages_completed': stages, 'end_unix': time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()))
