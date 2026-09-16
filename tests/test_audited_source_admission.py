"""Saved audit identity remains attached to derived rows and reused teacher labels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import audited_source_admission as tool
from scripts import baseline_row_exclusions as exclusions


def put(path: Path, data: Any) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return {'path': str(path), 'sha256': tool.exclusions.sha(path)}


@pytest.fixture
def lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = tmp_path / 'derived'
    selected: dict[str, Any] = {'source_dir': str(tmp_path / 'raw'), 'source_config_sha256': 'f' * 64,
                'shards': [{'source_shard': 'w00-00001.jsonl.zst'}]}
    selection = {**selected, **put(tmp_path / 'selection.json', selected)}
    audit = put(tmp_path / 'audit.json', {'shards': [{
        'source_dir': selected['source_dir'], 'source_shard': 'w00-00001.jsonl.zst',
        'counts': {'eligible_rows': 3}}]})
    exclusion = put(tmp_path / 'exclusions.json', {'audit': audit})
    proof: dict[str, Any] = {**exclusion, 'physical_rows': 5, 'eligible_rows': 3, 'no_result_rows': 1, 'excluded_rows': 1,
             'pins': [audit]}
    monkeypatch.setattr(tool.exclusions, 'load', lambda _: exclusions.Exclusions(proof, selected, ()))
    summary = {'baseline_exclusions': proof, 'source_selection': selection,
               'realized': {'rows_written': 3, 'rows_read': 5, 'rows_dropped_no_result': 1,
                            'rows_dropped_baseline_audit': 1, 'rows_dropped_envelope': 0, 'input_key_verified': 3},
               'scheme': {'canonical': 'uniform-d9', 'policy_observation': 'phase0', 'value_observation': 'latest-phase'},
               'value_scheme': {'name': 'search'}, 'temp_requested': .0005, 'floor_requested': 0,
               'shards': [{'path': 'shard_000000.zarr', 'rows': 3, 'row_provenance': {'sha256': 'a' * 64}}]}
    derived = put(root / 'derive_targets_summary.json', summary)
    adapter_manifest = put(tmp_path / 'adapter_manifest.json', {'source': derived})
    adapter = put(tmp_path / 'adapter' / 'bt4_policy_sidecar_summary.json', {
        'rows': 3, 'adapted_wdl': {'rows': 3, 'new_teacher_evaluations': 0},
        'adapter': {'derived_summary': derived, 'manifest': adapter_manifest,
                    'written_shards': [{'path': 'shard_000000.zarr', 'rows': 3, 'row_provenance_sha256': 'a' * 64}]}})
    def process(name: str, command: list[str]) -> dict[str, str]:
        return put(tmp_path / (name + '.json'), {'process_complete': True, 'exit_code': 0, 'gpu_seconds': 0,
                                                'command': command})
    manifest = {'schema': 1, 'profile': tool.PROFILE, 'derived_summary': derived, 'adapter_summary': adapter,
                'derivation_process': process('derive_process', ['derive', '--out', str(root), '--corpus', selected['source_dir'],
                    '--source-shards', selection['path'], '--baseline-exclusions', proof['path']]),
                'adapter_process': process('adapter_process', ['adapt', '--out', str(Path(adapter['path']).parent),
                    '--manifest', adapter_manifest['path'], '--expected-manifest-sha256', adapter_manifest['sha256']])}
    return root, summary, manifest


def run(root: Path, summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    ref = put(root.parent / 'admission.json', manifest)
    return tool.admit(Path(ref['path']), ref['sha256'], root, summary)


def test_saved_audit_admits_only_retained_rows(lineage: Any) -> None:
    root, summary, manifest = lineage
    result = run(root, summary, manifest)
    assert result['rows'] == 3
    assert result['raw_shards'] == ['w00-00001.jsonl.zst']
    assert result['per_raw_shard_survivors'] == {'w00-00001.jsonl.zst': 3}
    assert 'corpus_complete' not in result


@pytest.mark.parametrize('fault', ['failed_derive', 'wrong_corpus', 'wrong_adapter', 'wrong_offsets', 'new_inference', 'wrong_adapter_command', 'missing_retained_row'])
def test_foreign_or_partial_lineage_is_rejected(lineage: Any, fault: str) -> None:
    root, summary, manifest = lineage
    key = 'derivation_process' if fault in ('failed_derive', 'wrong_corpus') else 'adapter_summary'
    if fault == 'wrong_adapter_command':
        key = 'adapter_process'
    data = exclusions.read(manifest[key])
    if fault == 'failed_derive':
        data['exit_code'] = 1
    elif fault == 'wrong_corpus':
        data['command'][data['command'].index('--corpus') + 1] = '/other'
    elif fault == 'wrong_adapter':
        data['adapter']['derived_summary']['sha256'] = 'wrong'
    elif fault == 'wrong_offsets':
        data['adapter']['written_shards'][0]['row_provenance_sha256'] = 'wrong'
    elif fault == 'new_inference':
        data['adapted_wdl']['new_teacher_evaluations'] = 3
    elif fault == 'wrong_adapter_command':
        data['command'][-1] = 'wrong'
    elif fault == 'missing_retained_row':
        data['rows'] = 2
    manifest[key] = put(Path(manifest[key]['path']), data)
    with pytest.raises(ValueError, match='baseline exclusions:'):
        run(root, summary, manifest)
