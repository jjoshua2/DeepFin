"""Audited middle-row exclusions keep physical provenance through real writers."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import adapt_raw_bt4_sidecars as adapter
from scripts import bt4_raw_corpus_sidecar as raw
from scripts.bt4_policy_dump import remap_provenance
from tests.test_adapt_raw_bt4_sidecars import Teacher
from scripts import audit_raw_baseline as audit
from scripts import baseline_row_exclusions as exclusions
from scripts import corpus_row_provenance as refs
from scripts import derive_corpus_targets as derive
from tests.test_audit_raw_baseline import row, put
from tests.test_derive_parallel import write_split_corpus
from tests.test_derive_corpus_targets import run_derive


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = [row() for _ in range(5)]
    for i, r in enumerate(rows):
        r['game_id'] = i
    rows[1]['phases'][0]['per_depth'][0]['lines'][1][1] = rows[1]['phases'][0]['per_depth'][0]['lines'][0][1]
    rows[3]['result'] = None
    source = write_split_corpus(tmp_path, rows, [3, 2])
    entries = [{'source_shard': p.name, 'rows': count, 'source_sha256': exclusions.sha(p)}
               for p, count in zip(sorted(source.glob('*.jsonl.zst')), [3, 2])]
    source_pin = {'path': str(source / 'manifest.json'), 'sha256': exclusions.sha(source / 'manifest.json')}
    selection = put(tmp_path / 'selection.json', {'schema': 1, 'source_dir': str(source),
        'source_config_sha256': rows[0]['run']['config_sha256'], 'source_manifest_sha256': source_pin['sha256'], 'shards': entries})
    teacher = '1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0'
    receipts = [{'source_id': 'fixture', 'source_shard': e['source_shard'], 'source_sha256': e['source_sha256'],
        'positions': e['rows'], 'onnx_sha256': teacher, 'wdl': {'output': '/output/wdl', 'kind': 'probabilities',
        'dtype': 'float32', 'order': ['win', 'draw', 'loss'], 'pov': 'side_to_move', 'rows': e['rows']}} for e in entries]
    collection = put(tmp_path / 'collection.json', {'status': 'BOUNDED_RAW_LABEL_COLLECTION_COMPLETE', 'new_shards': 2,
        'new_rows': 5, 'receipts': receipts})
    audit.audit({'schema': 1, 'collection': collection, 'teacher_sha256': teacher,
        'sources': [{'id': 'fixture', 'source_dir': str(source), 'manifest': source_pin}]}, tmp_path / 'audit', lambda: None)
    def pin(p: Path) -> dict[str, str]:
        return {'path': str(p), 'sha256': exclusions.sha(p)}
    p = tmp_path / 'exclusions.json'
    put(p, {'schema': 1, 'audit': pin(tmp_path / 'audit/complete.json'),
        'diagnostics': pin(tmp_path / 'audit/rejected.jsonl'), 'selection': selection})
    return source, Path(selection['path']), p


def test_sequential_parallel_exact_survivors_and_physical_offsets(tmp_path: Path) -> None:
    source, selection, exclusion = fixture(tmp_path)
    outputs = []
    for workers in (1, 2):
        out = tmp_path / f'derived{workers}'
        result = run_derive(source, out, 'uniform-d9', '--source-shards', str(selection),
            '--baseline-exclusions', str(exclusion), '--row-provenance', '--workers', str(workers),
            '--policy-observation', 'phase0', '--value-observation', 'latest-phase', '--rows-per-shard', '3', '--seed', '9')
        assert result['realized']['rows_read'] == 5
        assert result['realized']['rows_dropped_baseline_audit'] == result['realized']['rows_dropped_no_result'] == 1
        assert result['realized']['rows_written'] == 3
        shard = out / 'shard_000000.zarr'
        group: Any = zarr.open_group(str(shard), mode='r')
        references = refs.read(shard / refs.FILENAME, rows=3)
        assert sorted((r['source_shard'], r['source_row']) for r in references) == [
            ('w00-00000.jsonl.zst', 0), ('w00-00000.jsonl.zst', 2), ('w00-00001.jsonl.zst', 1)]
        assert group.attrs['derive_baseline_exclusions']['excluded_rows'] == 1
        outputs.append((np.asarray(group['x'][:]), np.asarray(group['policy_target'][:]), references))
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    assert outputs[0][2] == outputs[1][2]

    # Actual unchanged adapter joins synthetic teacher outputs by original offsets.
    spec = raw.load_sources([f'fixture={source}'], tmp_path / 'raw_labels')[0]
    spec.out_dir.mkdir(parents=True)
    remap = remap_provenance()
    receipt_path = tmp_path / 'teacher_receipts.jsonl'
    teacher: dict[str, Any] = {'onnx': {'path': str(tmp_path / 'synthetic.onnx'), 'sha256': 'a' * 64},
               'policy_output': 'policy', 'providers': ['synthetic_CPU_fixture'], 'remap': remap}
    for path, count in zip(spec.inventory.shards, spec.inventory.shard_rows):
        pending = raw.PendingShard(spec, path, count, spec.out_dir / raw.sidecar_name(path.name))
        attrs = raw.label_shard(pending, sess=Teacher(), input_name='input', input_dtype=np.dtype(np.float32),
            providers=teacher['providers'], policy_name='policy', onnx_path=Path(teacher['onnx']['path']),
            onnx_sha256='a' * 64, remap_stamp=remap, batch_size=128)
        raw.append_receipt(receipt_path, raw.receipt_from_attrs(attrs, pending.target))
    def pin(p: Path) -> dict[str, str]:
        return {'path': str(p), 'sha256': exclusions.sha(p)}
    m = tmp_path / 'adapter.json'
    put(m, {'schema': 1, 'derived_summary': pin(out / derive.SUMMARY_NAME), 'teacher': teacher,
        'sources': [{'source_dir': str(source), 'sidecar_dir': str(spec.out_dir),
                     'manifest': pin(source / 'manifest.json'), 'receipts': pin(receipt_path)}]})
    target = tmp_path / 'adapted'
    adapter.adapt(m, expected_manifest_sha256=exclusions.sha(m), out=target)
    adapted: Any = zarr.open_group(str(target / 'shard_000000.zarr'), mode='r')
    for i, ref in enumerate(outputs[1][2]):
        original: Any = zarr.open_group(str(spec.out_dir / raw.sidecar_name(ref['source_shard'])), mode='r')
        np.testing.assert_array_equal(adapted['bt4_policy'][i], original[raw.POLICY_FIELD][ref['source_row']])


@pytest.mark.parametrize('fault', ['duplicate', 'unknown', 'index', 'game', 'incomplete', 'roster', 'hash'])
def test_bad_audit_binding_is_refused(tmp_path: Path, fault: str) -> None:
    _source, selection, path = fixture(tmp_path)
    m = json.loads(path.read_text())
    if fault in ('duplicate', 'unknown', 'index', 'game'):
        p = Path(m['diagnostics']['path'])
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        if fault == 'duplicate':
            rows.append(rows[0])
        elif fault == 'unknown':
            rows[0]['source_shard'] = 'unknown'
        elif fault == 'index':
            rows[0]['source_row'] = 999
        else:
            rows[0]['game_id'] = -1
        p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        m['diagnostics']['sha256'] = exclusions.sha(p)
        a = json.loads(Path(m['audit']['path']).read_text())
        a['diagnostic_bytes'] = p.stat().st_size
        m['audit'] = put(Path(m['audit']['path']), a)
    elif fault == 'incomplete':
        a = json.loads(Path(m['audit']['path']).read_text())
        a['status'] = 'RUNNING'
        m['audit'] = put(Path(m['audit']['path']), a)
    elif fault == 'roster':
        s = json.loads(selection.read_text())
        s['shards'].pop()
        m['selection'] = put(selection, s)
    else:
        m['audit']['sha256'] = '0' * 64
    put(path, m)
    with pytest.raises(ValueError, match="baseline exclusions"):
        exclusions.load(path)


def test_unused_and_wrong_runtime_identity_fail(tmp_path: Path) -> None:
    source, _selection, p = fixture(tmp_path)
    plan = exclusions.load(p)
    pending = plan.pending(list(source.glob('*.jsonl.zst')))
    bad = row()
    bad['game_id'] = 999
    with pytest.raises(ValueError, match='identity'):
        exclusions.consume(pending, source / 'w00-00000.jsonl.zst', 1, bad)
    pending = plan.pending(list(source.glob('*.jsonl.zst')))
    assert not exclusions.consume(pending, source / 'w00-00000.jsonl.zst', 0, row())
    with pytest.raises(ValueError, match='unused'):
        exclusions.require(not pending, 'unused exclusion IDs')


def test_other_skip_or_value_regimes_refused(tmp_path: Path) -> None:
    _source, _selection, p = fixture(tmp_path)
    base = audit.inspector().options
    with pytest.raises(ValueError, match='baseline exclusions require'):
        replace(base, baseline_exclusions=exclusions.load(p), row_provenance=True, limit=1)


@pytest.mark.parametrize('workers', [1, 2])
def test_wrong_audited_game_fails_actual_derivation(tmp_path: Path, workers: int) -> None:
    source, selection, path = fixture(tmp_path)
    m = json.loads(path.read_text())
    p = Path(m['diagnostics']['path'])
    diagnostic = [json.loads(line) for line in p.read_text().splitlines()]
    diagnostic[0]['game_id'] = 999
    p.write_text(''.join(json.dumps(r) + '\n' for r in diagnostic))
    m['diagnostics']['sha256'] = exclusions.sha(p)
    a = json.loads(Path(m['audit']['path']).read_text())
    a['diagnostic_bytes'] = p.stat().st_size
    m['audit'] = put(Path(m['audit']['path']), a)
    put(path, m)
    out = tmp_path / 'refused'
    with pytest.raises(ValueError, match='row identity mismatch'):
        run_derive(source, out, 'uniform-d9', '--source-shards', str(selection),
            '--baseline-exclusions', str(path), '--row-provenance', '--workers', str(workers),
            '--policy-observation', 'phase0')
    assert not (out / derive.SUMMARY_NAME).exists()
