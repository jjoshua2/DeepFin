"""Actual G10 derive/qualification/cache and value writer; synthetic teacher only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import bt4_value_rewrite as tool
from scripts import g10_native_wdl_reuse as reuse
from tests.test_bt4_derived_wdl_sidecar import Session, install
from tests.test_g10_wdl_admission import fixture as g10_fixture


def pin(path: Path) -> dict[str, str]:
    return {'path': str(path), 'sha256': tool.wdl.file_sha256(path)}


def write(path: Path, body: Any) -> None:
    path.write_text(json.dumps(body, sort_keys=True))


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    wargs, qualification = g10_fixture(tmp_path)
    install(monkeypatch, Session())
    # Two genuine invocation receipts covering disjoint parts of the same cohort.
    wargs.max_shards = 1
    for start in (0, 1):
        wargs.start_shard = start
        assert tool.wdl.run(wargs) == 0
    sf = Path(wargs.source)
    source = tmp_path / 'B100'
    shutil.copytree(sf, source)
    summary = json.loads((sf / tool.DERIVE_SUMMARY).read_text())
    recipe = {'kind': 'global', 'algorithm': 'legal-normalized-global-arithmetic-v1',
              'alpha': 1.0, 'bt4_temperature': 0.5, 'rows': 3, 'expected_shards': 2,
              'source_dir': str(sf), 'source_derive_summary_sha256': wargs.expected_source_summary_sha256,
              'mutated_arrays': ['policy_target']}
    for shard in source.glob('*.zarr'):
        group: Any = zarr.open_group(str(shard), mode='a')
        group.attrs.update(policy_target_mix_kind='global', policy_target_mix_alpha=1.0,
                           policy_target_mix_bt4_temperature=0.5)
        # The native-WDL reuse path is independent of the frozen policy values.
        policy = np.asarray(group['policy_target'][:])
        legal = np.asarray(group['legal_mask'][:])
        from scripts.bt4_policy_mix import mix_policy_targets
        group['policy_target'][:] = mix_policy_targets(
            policy, np.broadcast_to(np.arange(1, 1859), policy.shape).astype('float32'),
            legal, alpha=1.0, scope='global', bt4_temperature=0.5)
    summary['policy_target_postprocess'] = recipe
    write(source / tool.POLICY_SUMMARY, recipe)
    write(source / tool.DERIVE_SUMMARY, summary)
    # Model a historical source freeze whose source hashes differ from today's
    # writer. These are explicit synthetic evidence, not an actual old rollout.
    historical_dir = tmp_path / 'historical'
    historical_dir.mkdir()
    producer = {}
    runtime = Path(tool.__file__).resolve().parent.parent
    for index, role in enumerate(sorted(reuse.PRODUCERS)):
        frozen = historical_dir / f'{index}.py'
        frozen.write_bytes((runtime / role).read_bytes() + b'\n# synthetic prior source freeze\n')
        producer[role] = pin(frozen)
    historical_g10 = pin(runtime / 'scripts/g10_wdl_admission.py')
    entries: list[dict[str, Any]] = []
    for completed_path in sorted((Path(wargs.out) / 'invocations').glob('*/completed.json')):
        completed = json.loads(completed_path.read_text())
        attrs_pins = {}
        for spec in completed['selection']:
            path = Path(wargs.out) / spec['path'] / '.zattrs'
            attrs = json.loads(path.read_text())
            attrs['binding']['producer'] = {k: v['sha256'] for k, v in producer.items()}
            write(path, attrs)
            attrs_pins[spec['path']] = pin(path)
        entries.append({'completed': pin(completed_path),
                        'started': pin(completed_path.parent / 'started.json'),
                        'producer': producer, 'g10_admission_script': historical_g10,
                        'attributes': attrs_pins})
    manifest_path = tmp_path / 'native.json'
    manifest: dict[str, Any] = {'schema': 1, 'profile': reuse.PROFILE, 'source_dir': str(sf),
                'wdl_dir': wargs.out, 'source_summary_sha256': wargs.expected_source_summary_sha256,
                'onnx_sha256': wargs.expected_onnx_sha256, 'wdl_output': 'value',
                'invocations': entries}
    write(manifest_path, manifest)
    args = tool.build_parser().parse_args([
        '--source', str(source), '--sf-source', str(sf), '--wdl', wargs.out,
        '--out', str(tmp_path / 'value50'), '--alpha', '.5', '--batch-size', '2',
        '--minimum-free-gib', '0', '--expected-source-summary-sha256',
        tool.wdl.file_sha256(source / tool.DERIVE_SUMMARY),
        '--expected-policy-summary-sha256', tool.wdl.file_sha256(source / tool.POLICY_SUMMARY),
        '--expected-sf-summary-sha256', wargs.expected_source_summary_sha256,
        '--expected-onnx-sha256', wargs.expected_onnx_sha256, '--wdl-output', 'value',
        '--g10-common-qualification', str(qualification),
        '--expected-g10-common-qualification-sha256', tool.wdl.file_sha256(qualification),
        '--native-wdl-manifest', str(manifest_path),
        '--expected-native-wdl-manifest-sha256', tool.wdl.file_sha256(manifest_path)])
    return args, manifest


def test_g10_historical_native_cache_reaches_real_value_writer(tmp_path, monkeypatch):
    args, manifest = fixture(tmp_path, monkeypatch)
    original_sf = tool.wdl.storage_identity(args.sf_source)
    original_side = tool.wdl.storage_identity(args.wdl)
    result = tool.rewrite(args)
    assert result['rows'] == 3
    assert result['shards'] == 2
    assert result['native_wdl_reuse']['new_teacher_evaluations'] == 0
    assert result['g10_common_admission']['source_dir'] == str(args.sf_source)
    assert tool.wdl.storage_identity(args.sf_source) == original_sf
    assert tool.wdl.storage_identity(args.wdl) == original_side
    for output in result['outputs']:
        old: Any = zarr.open_group(str(args.source / output['path']), mode='r')
        new: Any = zarr.open_group(str(args.out / output['path']), mode='r')
        side: Any = zarr.open_group(str(args.wdl / output['path']), mode='r')
        assert side.attrs['binding']['producer'] == {
            k: v['sha256'] for k, v in manifest['invocations'][0]['producer'].items()}
        for name in tool.ARRAYS - {'search_wdl'}:
            assert np.asarray(old[name][:]).tobytes() == np.asarray(new[name][:]).tobytes()
        sf = np.asarray(old['search_wdl'][:], dtype='float64')
        bt4 = np.asarray(side['bt4_wdl_raw'][:], dtype='float64')
        expected = (.5 * sf / sf.sum(1, keepdims=True) + .5 * bt4 / bt4.sum(1, keepdims=True)).astype('float16')
        np.testing.assert_array_equal(new['search_wdl'][:], expected)


@pytest.mark.parametrize('defect', ['source', 'producer', 'head', 'overlap', 'missing',
                                  'incomplete', 'attrs_binding', 'no_qualification', 'both_modes'])
def test_rejects_wrong_historical_evidence_before_publication(tmp_path, monkeypatch, defect):
    args, manifest = fixture(tmp_path, monkeypatch)
    if defect == 'source':
        manifest['source_dir'] = str(tmp_path / 'another_source')
    elif defect == 'producer':
        entry = manifest['invocations'][0]
        # A validly pinned current producer is not the producer in old attrs.
        entry['producer']['scripts/bt4_derived_wdl_sidecar.py'] = pin(Path(tool.wdl.__file__).resolve())
    elif defect == 'head':
        args.wdl_output = 'another_head'
    elif defect == 'overlap':
        manifest['invocations'].append(copy.deepcopy(manifest['invocations'][0]))
    elif defect == 'missing':
        manifest['invocations'].pop()
    elif defect == 'incomplete':
        entry = manifest['invocations'][0]
        path = Path(entry['completed']['path'])
        receipt = json.loads(path.read_text())
        receipt['complete'] = False
        write(path, receipt)
        entry['completed'] = pin(path)
    elif defect == 'attrs_binding':
        entry = manifest['invocations'][0]
        name = next(iter(entry['attributes']))
        path = Path(entry['attributes'][name]['path'])
        attrs = json.loads(path.read_text())
        attrs['binding']['unattested'] = True
        write(path, attrs)
        entry['attributes'][name] = pin(path)
    elif defect == 'no_qualification':
        args.g10_common_qualification = None
        args.expected_g10_common_qualification_sha256 = None
    else:
        args.wdl_adapter_manifest = args.native_wdl_manifest
        args.expected_wdl_adapter_manifest_sha256 = args.expected_native_wdl_manifest_sha256
    write(args.native_wdl_manifest, manifest)
    args.expected_native_wdl_manifest_sha256 = tool.wdl.file_sha256(args.native_wdl_manifest)
    with pytest.raises(ValueError, match=r'native|G10'):
        tool.rewrite(args)
    assert not args.out.exists()
    assert not args.out.with_name(args.out.name + '.writing').exists()
