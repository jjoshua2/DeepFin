"""Receipt-selected streaming audit fixtures; no model or real corpus reads."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import chess
import pytest

from scripts import audit_raw_baseline as tool
from tests.test_derive_corpus_targets import history_row, full_width_phase, narrowed_phase


def row() -> dict[str, Any]:
    item = history_row()
    moves = [m.uci() for m in chess.Board(item['fen']).legal_moves]
    values = {m: float(100 - i) for i, m in enumerate(moves)}
    item['phases'] = [full_width_phase(item['fen'], {9: values}),
        narrowed_phase({9: dict.fromkeys(moves[:3], 300.0), 10: dict.fromkeys(moves[:3], 400.0)}, depth_requested=10)]
    return item


def inspect(item: dict[str, Any]) -> dict[str, Any]:
    return tool.inspect_row(item, item['run']['config_sha256'], item['worker_id'], tool.inspector())


def test_later_duplicate_rejected_despite_valid_phase0() -> None:
    item = row()
    lines = item['phases'][1]['per_depth'][0]['lines']
    lines[1][1] = lines[0][1]
    result = inspect(item)
    assert result['policy_error'] is None
    assert result['value_error'] == 'invalid_roster'
    assert result['eligible'] is False


def test_benign_reemission_and_unused_depth_do_not_change_eligibility() -> None:
    item = row()
    assert inspect(item)['eligible'] is True
    # Parser anomaly metadata is not itself an ambiguity in the consumed block.
    item['phases'][1]['anomalies'] = {'reemitted': 100, 'disagreeing': 0}
    unused = copy.deepcopy(item['phases'][1]['per_depth'][1])
    unused['depth'] = 3
    unused['lines'][1][1] = unused['lines'][0][1]
    item['phases'][1]['per_depth'].append(unused)
    assert inspect(item)['eligible'] is True


def test_duplicate_policy_and_composite_counters_are_independent() -> None:
    item = row()
    item['phases'][0]['per_depth'][0]['lines'][1][1] = item['phases'][0]['per_depth'][0]['lines'][0][1]
    result = inspect(item)
    assert result['policy_error'] is not None
    assert result['value_error'] is not None


@pytest.mark.parametrize('fault', ['input', 'history', 'source', 'worker', 'result'])
def test_identity_failure_is_fatal_even_without_result(fault: str) -> None:
    item = row()
    item['result'] = None
    config, worker = item['run']['config_sha256'], item['worker_id']
    if fault == 'input':
        item['input_key'] = 'bad'
    elif fault == 'history':
        item['history_root_fen'] = chess.STARTING_FEN
        item['history_uci'] = []
        item['fen'] = '8/8/8/8/8/8/8/8 w - - 0 1'
    elif fault == 'source':
        config = 'other'
    elif fault == 'worker':
        worker += 1
    else:
        item['result'] = 2.
    with pytest.raises((ValueError, tool.derive.CorpusIntegrityError), match=r'input|fen|replay|board|config|worker|result|position'):
        tool.inspect_row(item, config, worker, tool.inspector())


def put(path: Path, value: Any) -> dict[str, str]:
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def manifest(tmp_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = tmp_path / 'raw'
    source.mkdir()
    path = source / 'w00-00000.jsonl.gz'
    with gzip.open(path, 'wt') as stream:
        for item in rows:
            stream.write(json.dumps(item) + '\n')
    launch = put(source / 'manifest.json', {'row_schema': 3, 'history_rep_fix': True, 'config_sha256': rows[0]['run']['config_sha256']})
    teacher = '1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0'
    receipt = {'source_id': 'run06', 'source_shard': path.name, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'positions': len(rows), 'onnx_sha256': teacher, 'wdl': {'output': '/output/wdl', 'kind': 'probabilities', 'dtype': 'float32', 'order': ['win', 'draw', 'loss'], 'pov': 'side_to_move', 'rows': len(rows)}}
    collection = put(tmp_path / 'collection.json', {'status': 'BOUNDED_RAW_LABEL_COLLECTION_COMPLETE', 'new_shards': 1,
        'new_rows': len(rows), 'receipts': [receipt]})
    return {'schema': 1, 'collection': collection, 'teacher_sha256': teacher,
        'sources': [{'id': 'run06', 'source_dir': str(source), 'manifest': launch}]}


def test_streaming_counts_collateral_and_source_qualified_reasons(tmp_path: Path) -> None:
    good, bad = row(), row()
    bad['phases'][1]['per_depth'][0]['lines'][1][1] = bad['phases'][1]['per_depth'][0]['lines'][0][1]
    out = tmp_path / 'audit'
    result = tool.audit(manifest(tmp_path, [good, bad]), out, lambda: None)
    counts = result['counts']
    assert counts['eligible_rows'] == counts['whole_shard_collateral_eligible_rows'] == 1
    assert counts['whole_shard_retained_rows'] == counts['policy_rejected_rows'] == 0
    assert counts['value_rejected_rows'] == 1
    rejected = json.loads((out / 'rejected.jsonl').read_text())
    assert rejected['source_row'] == 1
    assert rejected['source_shard'] == 'w00-00000.jsonl.gz'
    assert len(rejected['source_namespace']) == 64


@pytest.mark.parametrize('fault', ['hash', 'rows', 'duplicate', 'cap', 'deadline'])
def test_receipt_bounds_and_failure_preserve_no_pass(tmp_path: Path, fault: str) -> None:
    bad = row()
    bad['phases'][1]['per_depth'][0]['lines'][1][1] = bad['phases'][1]['per_depth'][0]['lines'][0][1]
    m = manifest(tmp_path, [bad])
    c = json.loads(Path(m['collection']['path']).read_text())
    if fault == 'hash':
        c['receipts'][0]['source_sha256'] = '0' * 64
    elif fault == 'rows':
        c['receipts'][0]['positions'] = c['new_rows'] = c['receipts'][0]['wdl']['rows'] = 2
    elif fault == 'duplicate':
        c['receipts'] *= 2
        c['new_shards'] = c['new_rows'] = 2
    m['collection'] = put(Path(m['collection']['path']), c)
    def guard() -> None:
        if fault == 'deadline':
            raise ValueError('deadline')
    out = tmp_path / 'audit'
    with pytest.raises(ValueError, match=r'hash|count|duplicate|cap|deadline'):
        tool.audit(m, out, guard, diagnostic_cap=1 if fault == 'cap' else tool.MAX_DIAGNOSTICS)
    assert not (out / 'complete.json').exists()



def test_no_result_rejection_does_not_discard_valid_shard_rows(tmp_path: Path) -> None:
    good, absent = row(), row()
    absent['result'] = None
    absent['phases'][1]['per_depth'][0]['lines'][1][1] = absent['phases'][1]['per_depth'][0]['lines'][0][1]
    result = tool.audit(manifest(tmp_path, [good, absent]), tmp_path / 'audit', lambda: None)
    assert result['counts']['no_result_rows'] == 1
    assert result['counts']['whole_shard_retained_rows'] == 1
    assert result['counts']['whole_shard_collateral_eligible_rows'] == 0
