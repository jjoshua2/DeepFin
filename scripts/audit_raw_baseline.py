"""Bounded receipt-selected G10 baseline eligibility; no derivation or model construction."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any
from collections.abc import Callable

from scripts import adaptive_sf_value as baseline
from scripts import derive_corpus_targets as derive
from scripts import training_host_memory as memory

MAX_SHARDS = 192
MAX_DIAGNOSTICS = 512 * 1024**2


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(path: Path, guard: Callable[[], None]) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            guard()
            result.update(block)
    return result.hexdigest()


def read_pin(ref: dict[str, Any], guard: Callable[[], None]) -> dict[str, Any]:
    require(set(ref) == {'path', 'sha256'}, 'invalid pin')
    path = Path(ref['path'])
    require(path.is_absolute() and path == path.resolve(), 'noncanonical pin')
    require(digest(path, guard) == ref['sha256'], 'changed pin: ' + str(path))
    return json.loads(path.read_text())


def inspector() -> derive.TargetDeriver:
    scheme = replace(derive.parse_scheme('uniform-d9'), policy_observation='phase0', value_observation='latest-phase')
    # No Q mapping/target construction is invoked by this audit.
    options = derive.DeriveOptions(scheme, .0005, 1., 1., 0, 0, 8192, 0)
    return derive.TargetDeriver(options)


def inspect_row(row: dict[str, Any], config: str, worker: int, tool: derive.TargetDeriver) -> dict[str, Any]:
    """Identity faults are fatal even for otherwise excluded/no-result rows."""
    require(type(row.get('schema')) is int and row['schema'] == derive.ROW_SCHEMA_HISTORY, 'history schema required')
    for key in ('worker_id', 'game_id', 'ply'):
        require(type(row.get(key)) is int and row[key] >= 0, 'invalid row identity ' + key)
    require(row['worker_id'] == worker, 'row worker/shard mismatch')
    derive._check_row_identity(row, config)
    board = tool._board_for(row)
    require(board.is_valid(), 'invalid row board')
    require(row.get('search_key') == derive.corpus.search_key(board), 'search identity mismatch')
    tool._verify_input_key(row, tool._encode(board), count_emitted=False)
    if row.get('result') is not None:
        require(type(row['result']) in (int, float), 'invalid result type')
        derive.wdl_target_from_result(row['result'])
    legal = {move.uci() for move in board.legal_moves}
    policy_error = value_error = None
    values = None
    policy_stage = 'phase0_block'
    try:
        tool._validate_selected_policy_block(row, board)
        policy_stage = 'phase0_support'
        bank = derive.RowBank(row)
        values = derive.apply_scheme(bank, tool.options.scheme)
        tool._check_support(board, values, row)
    except (ValueError, KeyError, IndexError, TypeError, derive.CorpusIntegrityError, derive.EnvelopeMiss) as exc:
        policy_error = policy_stage + ':' + type(exc).__name__
    try:
        # This validates each consumed later d9 block, plus baseline phase0.
        # It never chooses adaptive/deeper labels or repairs duplicate moves.
        baseline.validate_baseline(row, legal)
        if values is not None and policy_error is None:
            selected = tool._value_view(derive.RowBank(row), values)
            require(set(selected.moves) == legal, 'composite value support differs')
    except (ValueError, KeyError, IndexError, TypeError, derive.CorpusIntegrityError, derive.EnvelopeMiss) as exc:
        value_error = str(exc) if type(exc) is ValueError else type(exc).__name__
    return {'policy_error': policy_error, 'value_error': value_error,
            'no_result': row.get('result') is None,
            'eligible': policy_error is None and value_error is None and row.get('result') is not None}


def selection(manifest: dict[str, Any], guard: Callable[[], None]) -> list[dict[str, Any]]:
    require(manifest['schema'] == 1, 'manifest schema')
    require(manifest['teacher_sha256'] == '1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0', 'registered teacher required')
    collection = read_pin(manifest['collection'], guard)
    require(collection['status'] == 'BOUNDED_RAW_LABEL_COLLECTION_COMPLETE', 'collection not complete')
    entries = collection['receipts']
    require(0 < len(entries) <= MAX_SHARDS and collection['new_shards'] == len(entries), 'receipt count')
    require(sum(row['positions'] for row in entries) == collection['new_rows'], 'receipt row total')
    sources = {}
    for source in manifest['sources']:
        root = Path(source['source_dir'])
        require(root.is_absolute() and root == root.resolve(), 'source root aliases')
        require(source['id'] not in sources, 'duplicate source ID')
        require(Path(source['manifest']['path']) == root / 'manifest.json', 'source manifest path')
        launch = read_pin(source['manifest'], guard)
        require(launch['row_schema'] == derive.ROW_SCHEMA_HISTORY, 'source row schema')
        require(launch.get('history_rep_fix') is True, 'source history regime')
        require(all(root != prior[0] for prior in sources.values()), 'duplicate source root')
        sources[source['id']] = (root, launch)
    result = []
    seen = set()
    for receipt in entries:
        sid, name = receipt['source_id'], receipt['source_shard']
        match = re.fullmatch(r'w(\d+)-\d+\.jsonl\.(?:zst|gz)', name)
        require(sid in sources and match is not None and (sid, name) not in seen, 'invalid/duplicate receipt selection')
        if match is None:
            raise ValueError('invalid shard name')
        seen.add((sid, name))
        root, launch = sources[sid]
        require(type(receipt['positions']) is int and receipt['positions'] > 0, 'invalid raw row count')
        require(receipt['wdl']['output'] == '/output/wdl' and receipt['wdl']['kind'] == 'probabilities', 'joint native WDL required')
        require(receipt['wdl']['dtype'] == 'float32' and receipt['wdl']['order'] == ['win', 'draw', 'loss']
                and receipt['wdl']['pov'] == 'side_to_move' and receipt['wdl']['rows'] == receipt['positions'], 'native WDL layout')
        require(receipt['onnx_sha256'] == manifest['teacher_sha256'], 'teacher mismatch')
        result.append({'source_id': sid, 'source_dir': str(root), 'source_shard': name,
                       'source_namespace': hashlib.sha256(json.dumps([str(root), launch['config_sha256']], separators=(',', ':')).encode()).hexdigest(),
                       'config_sha256': launch['config_sha256'], 'source_sha256': receipt['source_sha256'],
                       'rows': receipt['positions'], 'worker': int(match.group(1))})
    return result


def audit(manifest: dict[str, Any], out: Path, guard: Callable[[], None], *, diagnostic_cap: int = MAX_DIAGNOSTICS) -> dict[str, Any]:
    require(0 < diagnostic_cap <= MAX_DIAGNOSTICS, 'diagnostic cap')
    entries = selection(manifest, guard)
    out = out.resolve()
    require(all(not out.is_relative_to(Path(e['source_dir'])) and not Path(e['source_dir']).is_relative_to(out) for e in entries), 'output overlaps raw source')
    out.mkdir()  # fresh, failed evidence is never adopted
    tool = inspector()
    totals: Counter[str] = Counter()
    shards = []
    diagnostic_bytes = 0
    current: dict[str, Any] = {}
    try:
        with (out / 'rejected.jsonl').open('wb') as diagnostics:
            for entry in entries:
                guard()
                path = Path(entry['source_dir']) / entry['source_shard']
                stamp = path.stat()
                require(digest(path, guard) == entry['source_sha256'], 'raw source hash mismatch')
                counts: Counter[str] = Counter()
                reasons: Counter[str] = Counter()
                for index, row in enumerate(derive.iter_corpus_rows(path)):
                    guard()
                    current = {**entry, 'source_row': index, 'game_id': row.get('game_id'), 'ply': row.get('ply')}
                    require(index < entry['rows'], 'raw rows exceed receipt')
                    item = inspect_row(row, entry['config_sha256'], entry['worker'], tool)
                    counts['physical_rows'] += 1
                    counts['eligible_rows'] += int(item['eligible'])
                    counts['no_result_rows'] += int(item['no_result'])
                    counts['required_baseline_rejected_rows'] += int(not item['no_result'] and not item['eligible'])
                    for kind in ('policy', 'value'):
                        error = item[kind + '_error']
                        counts[kind + '_rejected_rows'] += int(error is not None)
                        if error is not None:
                            reasons[kind + ':' + error] += 1
                    if not item['eligible']:
                        payload = (json.dumps({**current, **item}, sort_keys=True) + '\n').encode()
                        diagnostic_bytes += len(payload)
                        require(diagnostic_bytes <= diagnostic_cap, 'diagnostic byte cap')
                        diagnostics.write(payload)
                require(counts['physical_rows'] == entry['rows'], 'raw row count mismatch')
                now = path.stat()
                require((stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns) ==
                        (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns), 'raw source changed')
                bad = counts['required_baseline_rejected_rows'] > 0
                counts['whole_shard_collateral_eligible_rows'] = counts['eligible_rows'] if bad else 0
                counts['whole_shard_retained_rows'] = 0 if bad else counts['eligible_rows']
                totals.update(counts)
                shards.append({**entry, 'counts': dict(counts), 'reasons': dict(reasons), 'whole_shard_eligible': not bad})
                (out / 'progress.json').write_text(json.dumps({'complete_shards': len(shards), 'counts': dict(totals)}) + '\n')
        guard()
        report = {'status': 'PASS_BASELINE_ELIGIBILITY_NOT_DERIVATION', 'counts': dict(totals), 'shards': shards,
                  'diagnostic_bytes': diagnostic_bytes, 'manifest': manifest,
                  'semantics': 'phase0 uniform-d9 policy; original composite latest-phase d9 value. Policy and composite-value counters overlap; value validator also requires phase0. No scores repaired or adaptive targets selected.'}
        (out / 'complete.json').write_text(json.dumps(report, indent=2) + '\n')
        return report
    except BaseException as exc:
        (out / 'failure.json').write_text(json.dumps({'status': 'FAILED_NOT_ELIGIBILITY_PASS', 'error': str(exc),
            'current': current, 'complete_shards': len(shards), 'counts': dict(totals)}, indent=2) + '\n')
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'GPU must be hidden')
    require(len(os.sched_getaffinity(0)) <= 2, 'at most two CPUs')
    require(all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')), 'two numeric threads')
    memory.require_available(48)
    deadline = min(args.deadline_unix, time.time() + 1740)
    last_resources = 0.
    def guard() -> None:
        nonlocal last_resources
        now = time.time()
        require(now < deadline, 'audit deadline')
        if now - last_resources < 2:
            return
        last_resources = now
        memory.require_available(32)
        require(shutil.disk_usage(args.out.parent).free >= 150 * 1024**3, 'disk reserve')
    manifest = read_pin({'path': str(args.manifest.resolve()), 'sha256': args.expected_manifest_sha256}, guard)
    audit(manifest, args.out, guard)


if __name__ == '__main__':
    main()
