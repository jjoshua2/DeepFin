"""Real small G10/native-WDL writer with explicitly synthetic matching receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import matched_sf_value as matched
from scripts import bt4_value_rewrite as tool
from tests.test_g10_native_wdl_reuse import fixture as original_fixture, pin, write


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    args, _ = original_fixture(tmp_path, monkeypatch)
    candidate = tmp_path / 'adaptive'
    shutil.copytree(args.sf_source, candidate)
    summary = json.loads((candidate / tool.DERIVE_SUMMARY).read_text())
    records, witnesses = [], {}
    for spec in summary['shards']:
        name = spec['path']
        group: Any = zarr.open_group(str(candidate / name), mode='a')
        params = dict(group.attrs['derive_scheme_params'])
        params.update(sf_value_selector=matched.SELECTOR, value_source=matched.VALUE_SOURCE)
        group.attrs.update(derive_scheme_params=params, derive_value_source=matched.VALUE_SOURCE)
        group['search_wdl'][:] = np.broadcast_to(np.array([.125, .25, .625], dtype='float16'), (spec['rows'], 3))
        records.append({'shard': name, 'rows': spec['rows'],
                        'provenance_sha256': tool.wdl.file_sha256(candidate / name / 'row_provenance.npz'),
                        'nonvalue_decoded_sha256': {column: hashlib.sha256(np.asarray(group[column][:]).tobytes()).hexdigest()
                                                   for column in tool.ARRAYS - {'search_wdl'}}})
        witnesses[name] = {'attrs': pin(candidate / name / '.zattrs'),
                           'search_wdl_sha256': hashlib.sha256(np.asarray(group['search_wdl'][:]).tobytes()).hexdigest()}
    summary['value_scheme']['q_definition'] = 'synthetic adaptive values for consumer test'
    write(candidate / tool.DERIVE_SUMMARY, summary)
    runtime = Path(tool.__file__).resolve().parent.parent
    argv = ['python', str(runtime / 'scripts/derive_corpus_targets.py'), '--out', str(candidate),
            '--sf-value-selector', matched.SELECTOR, '--value-scheme', 'search', '--scheme', 'uniform-d9',
            '--policy-observation', 'phase0', '--value-observation', 'latest-phase', '--limit', str(summary['limit_requested'])]
    plan = {'baseline': str(args.sf_source), 'runtime_cwd': str(runtime), 'derive_argv': argv,
            'pins': {str(args.sf_source / tool.DERIVE_SUMMARY): args.expected_sf_summary_sha256,
                     **{str(runtime / role): tool.wdl.file_sha256(runtime / role)
                        for role in ('scripts/derive_corpus_targets.py', 'scripts/adaptive_sf_value.py')}}}
    planpath = tmp_path / 'plan.json'
    write(planpath, plan)
    receipt = {'status': 'COMPLETE_MATCHED_DERIVATION_NOT_TRAINING_ADMISSION', 'returncode': 0,
               'plan_sha256': pin(planpath)['sha256'], 'derive_summary_sha256': pin(candidate / tool.DERIVE_SUMMARY)['sha256'],
               'argv': argv, 'rows': 3, 'checked_shards': records}
    receiptpath = tmp_path / 'completed.json'
    write(receiptpath, receipt)
    manifest: dict[str, Any] = {'schema': 1, 'profile': matched.PROFILE, 'original_source': str(args.sf_source),
                'original_summary_sha256': args.expected_sf_summary_sha256,
                'candidate_source': str(candidate), 'candidate_summary': pin(candidate / tool.DERIVE_SUMMARY),
                'launch_plan': pin(planpath), 'completed_receipt': pin(receiptpath), 'shards': witnesses}
    args.matched_sf_manifest = tmp_path / 'matched.json'
    write(args.matched_sf_manifest, manifest)
    args.expected_matched_sf_manifest_sha256 = pin(args.matched_sf_manifest)['sha256']
    return args, manifest


def test_actual_writer_uses_adaptive_sf_preserves_teacher_and_shared_recipe(tmp_path, monkeypatch):
    results = []
    for cohort in ('a', 'b'):
        root = tmp_path / cohort
        root.mkdir()
        args, manifest = fixture(root, monkeypatch)
        teacher_state = tool.wdl.storage_identity(args.wdl)
        result = tool.rewrite(args)
        results.append(result)
        assert tool.wdl.storage_identity(args.wdl) == teacher_state
        assert result['sf_source_dir'] == str(args.sf_source)
        assert result['native_wdl_reuse']['new_teacher_evaluations'] == 0
        assert result['matched_sf']['candidate_source'] == manifest['candidate_source']
        for spec in result['outputs']:
            name = spec['path']
            old: Any = zarr.open_group(str(args.source / name), mode='r')
            new: Any = zarr.open_group(str(args.out / name), mode='r')
            adaptive: Any = zarr.open_group(str(Path(manifest['candidate_source']) / name), mode='r')
            teacher: Any = zarr.open_group(str(args.wdl / name), mode='r')
            want = tool.target(np.asarray(adaptive['search_wdl'][:]), np.asarray(teacher['bt4_wdl_raw'][:]), .5)
            baseline = tool.target(np.asarray(old['search_wdl'][:]), np.asarray(teacher['bt4_wdl_raw'][:]), .5)
            np.testing.assert_array_equal(new['search_wdl'][:], want)
            assert not np.array_equal(want, baseline)
            for column in tool.ARRAYS - {'search_wdl'}:
                assert np.asarray(old[column][:]).tobytes() == np.asarray(new[column][:]).tobytes()
        assert result['value_source'] != tool.value_source(args.expected_onnx_sha256, args.wdl_output, .5)
    assert results[0]['value_source'] == results[1]['value_source']
    assert results[0]['matched_sf']['manifest'] != results[1]['matched_sf']['manifest']


@pytest.mark.parametrize('defect', ['stale_wdl', 'nonvalue', 'provenance', 'wrong_original',
                                  'selector', 'missing_coverage', 'receipt_pin', 'during_write', 'invalid_wdl'])
def test_rejects_bad_matched_evidence(tmp_path, monkeypatch, defect):
    args, manifest = fixture(tmp_path, monkeypatch)
    path = Path(manifest['candidate_source']) / 'shard_000000.zarr'
    group: Any = zarr.open_group(str(path), mode='a')
    if defect == 'stale_wdl':
        group['search_wdl'][0] = [.25, .25, .5]
    elif defect == 'invalid_wdl':
        group['search_wdl'][0] = [1, 1, 1]
        manifest['shards']['shard_000000.zarr']['search_wdl_sha256'] = hashlib.sha256(np.asarray(group['search_wdl'][:]).tobytes()).hexdigest()
    elif defect == 'nonvalue':
        group['game_id'][0] = 999
    elif defect == 'provenance':
        (path / 'row_provenance.npz').write_bytes(b'wrong source-qualified rows')
    elif defect == 'wrong_original':
        manifest['original_source'] = str(args.source)
    elif defect == 'selector':
        group.attrs['derive_value_source'] = 'deepest_phase_covering'
        manifest['shards']['shard_000000.zarr']['attrs'] = pin(path / '.zattrs')
    elif defect == 'missing_coverage':
        manifest['shards'].pop('shard_000001.zarr')
    elif defect == 'receipt_pin':
        Path(manifest['completed_receipt']['path']).write_text('{}')
    else:
        target = tool.target
        changed = False
        def mutate(*a, **kw):
            nonlocal changed
            if not changed:
                changed = True
                group['search_wdl'][0] = [.25, .25, .5]
            return target(*a, **kw)
        monkeypatch.setattr(tool, 'target', mutate)
    write(args.matched_sf_manifest, manifest)
    args.expected_matched_sf_manifest_sha256 = pin(args.matched_sf_manifest)['sha256']
    with pytest.raises(ValueError, match=r'matched SF|adaptation pin|source or sidecar changed'):
        tool.rewrite(args)
    assert not args.out.exists()
