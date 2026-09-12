"""Tiny actual derived/collected shards; only the teacher inference is fake."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import ceres_target_mix as tool
from tests.test_ceres_derived_sidecar import Session, setup


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[argparse.Namespace, dict[str, Any]]:
    cargs = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(tool.ceres, 'open_teacher', lambda _: Session(tmp_path))
    tool.ceres.produce(cargs)
    source = Path(cargs.source)
    shard = source / 'shard_000000.zarr'
    group: Any = zarr.open_group(str(shard), mode='r')
    bank_path = Path(cargs.out) / shard.name
    bank: Any = zarr.open_group(str(bank_path), mode='r')
    raw_path = tmp_path / 'raw_bt4' / shard.name
    raw: Any = zarr.open_group(str(raw_path), mode='w')
    encoding, keys, key_sha, policy_sha = tool.bt4._sidecar_identity(group, shard)
    legal = group['legal_mask'][:]
    probabilities = legal * np.arange(1, 1859, dtype=np.float64)
    probabilities = (probabilities / probabilities.sum(axis=1, keepdims=True)).astype('float32')
    raw.create_dataset(tool.bt4.SIDECAR_KEY_FIELD, data=keys, chunks=keys.shape)
    raw.create_dataset(tool.bt4.SIDECAR_POLICY_FIELD, data=probabilities, chunks=(8, 1858))
    raw.attrs.update(bt4_policy_sidecar_schema=1, source_shard=shard.name, source_dir=str(source), positions=len(keys),
        source_key_sha256=key_sha, source_policy_sha256=policy_sha,
        input_history_encoding=encoding, policy_encoding='lc0_1858', policy_size=1858,
        onnx_sha256='b' * 64, providers=['CUDAExecutionProvider'], policy_output='policy',
        teacher_evaluations_per_position=1, search_nodes=0, stored_dtype='float32')
    binding = dict(bank.attrs)['binding']
    manifest: dict[str, Any] = {'schema': 1, 'source': str(source),
        'source_summary_sha256': cargs.expected_source_summary_sha256,
        'teachers': {'bt4': {'model_sha256': 'b' * 64, 'providers': ['CUDAExecutionProvider'],
                            'policy_output': 'policy'},
                    'ceres': {k: binding[k] for k in ('model_sha256', 'profile', 'backend')}},
        'entries': [{'shard': shard.name, 'bt4': str(raw_path), 'ceres': str(bank_path),
                     'ceres_binding': binding,
                     'bt4_storage_identity': tool.shared.storage_identity(raw_path)}]}
    summary = {'schema': 1, 'kind': 'bt4_raw_legal_policy_sidecar', 'source_dir': str(source),
        'source_shards': 1, 'sidecar_shards': 1, 'rows': len(keys), 'policy_encoding': 'lc0_1858',
        'onnx': {'path': '/fixture/bt4.onnx', 'sha256': 'b' * 64}, 'policy_output': 'policy',
        'providers': ['CUDAExecutionProvider'], 'teacher_evaluations_per_position': 1,
        'search_nodes': 0, 'stored_dtype': 'float32',
        'remap': {'commit': 'fixture', 'dirty': False, 'blobs': {'mapping.py': 'fixture'}}}
    summary_path = raw_path.parent / tool.bt4.SIDECAR_SUMMARY
    summary_path.write_text(json.dumps(summary))
    summary_sha = tool.shared.file_sha256(summary_path)
    monkeypatch.setattr(tool, 'LEGACY_BT4_SUMMARY_SHA', summary_sha)
    manifest['bt4_lineage'] = {'mode': 'legacy-root-position-v1',
                             'summary': {'path': str(summary_path), 'sha256': summary_sha}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    args = tool.build_parser().parse_args(['--manifest', str(path), '--expected-manifest-sha256',
        tool.shared.file_sha256(path), '--out', str(tmp_path / 'mixed'), '--minimum-free-gib', '0'])
    return args, manifest


def test_probability_mixing_and_single_sharpening():
    b = np.array([[0.2, 0.8, 0.]])
    logits = np.log(np.array([[0.6, 0.4, 0.1]]))
    legal = np.array([[1, 1, 0]])
    result = tool.policy_target(b, logits, legal, bt4_weight=.5,
                               bt4_temperature=.5, ceres_temperature=.5)
    expected = .5 * np.array([.04, .64]) / .68 + .5 * np.array([.36, .16]) / .52
    np.testing.assert_allclose(result[0, :2], expected)
    assert result[0, 2] == 0
    assert not np.allclose(result[0, :2], np.array([.2 * .6, .8 * .4])**2 /
                           sum(np.array([.2 * .6, .8 * .4])**2))


def test_real_rewrite_and_consumer(tmp_path, monkeypatch):
    from chess_anti_engine.replay.shard import load_shard_arrays
    args, manifest = fixture(tmp_path, monkeypatch)
    result = tool.rewrite(args)
    out = Path(args.out)
    assert result['complete']
    assert result['rows'] == 32
    assert result['bt4_lineage']['full_historical_input_provenance'] == 'inherited_from_pinned_collection'
    assert result['bt4_lineage']['full_input_digest_verified_shards'] == 0
    assert result['producer_sha256'] == tool.producer_pins()
    original = Path(manifest['source']) / 'shard_000000.zarr'
    target = out / original.name
    before, after = tool.copies.file_map(original), tool.copies.file_map(target)
    # Exercise the exact extracted shard path independently after full-manifest
    # admission, with smaller batches that exercise partial-chunk writes.
    accepted, source_summary, specs = tool.read_manifest(Path(args.manifest), args.expected_manifest_sha256)
    single = tmp_path / 'single_shard'
    single.mkdir()
    shard_args = argparse.Namespace(**{**vars(args), 'batch_size': 8})
    proof, states, metrics = tool.rewrite_shard(
        shard_args, manifest=accepted, original=source_summary, spec=specs[0],
        entry=accepted['entries'][0], writing=single, weight=.5,
        temperatures={'bt4': .5, 'ceres': .5}, guard=lambda: None)
    direct: Any = zarr.open_group(str(single / original.name), mode='r')
    full: Any = zarr.open_group(str(target), mode='r')
    assert dict(direct.attrs) == dict(full.attrs)
    for name in tool.ARRAYS:
        a, b = np.asarray(direct[name][:]), np.asarray(full[name][:])
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    assert proof['policy_target_sha256'] == result['outputs'][0]['policy_target_sha256']
    assert proof['changed_rows'] == result['outputs'][0]['changed_rows']
    assert metrics['max_mass_error'] == result['max_stored_mass_error']
    assert metrics['max_tv'] == result['max_stored_total_variation']
    assert metrics['support_lost'] == result['support_lost_move_entries']
    assert metrics['full_input_digest_verified_shards'] == result['bt4_lineage']['full_input_digest_verified_shards']
    assert all(tool.shared.storage_identity(path) == state for path, state in states.items())
    for name, digest in before.items():
        if name != '.zattrs' and not name.startswith('policy_target/'):
            assert after[name] == digest
    loaded, _ = load_shard_arrays(target)
    group: Any = zarr.open_group(str(target), mode='r')
    np.testing.assert_array_equal(loaded['policy_target'], group['policy_target'][:])
    old, _ = load_shard_arrays(original)
    assert not np.array_equal(loaded['policy_target'], old['policy_target'])
    import torch
    from chess_anti_engine.replay.dataset import collate_arrays
    from chess_anti_engine.train.losses import compute_loss
    losses, grads = [], []
    for arrays in (old, loaded):
        batch = collate_arrays(arrays, device='cpu')
        logits = torch.linspace(-2, 2, 1858).repeat(32, 1).requires_grad_()
        loss = compute_loss({'policy': logits, 'wdl': torch.zeros((32, 3))}, batch,
                            search_wdl_frac=1.0, sf_wdl_frac=0.0)
        loss['policy_ce'].backward()
        grads.append(logits.grad)
        losses.append(loss)
    assert not torch.equal(grads[0], grads[1])
    torch.testing.assert_close(losses[0]['wdl_ce'], losses[1]['wdl_ce'])
    assert losses[1]['search_wdl_effective_rows'].item() == 32
    derived = json.loads((out / tool.DERIVE_SUMMARY).read_text())
    assert derived['policy_target_postprocess'] == {k: v for k, v in result.items() if k != 'outputs'}
    with pytest.raises(ValueError, match='output or partial exists'):
        tool.rewrite(args)


@pytest.mark.parametrize('defect', ['duplicate', 'selected', 'input', 'row', 'legal', 'missing_chunk'])
def test_reject_misalignment_before_publication(tmp_path, monkeypatch, defect):
    args, manifest = fixture(tmp_path, monkeypatch)
    entry = manifest['entries'][0]
    bank: Any = zarr.open_group(entry['ceres'], mode='a')
    if defect == 'duplicate':
        manifest['entries'].append(entry.copy())
    elif defect == 'selected':
        entry['ceres_binding']['selected_rows'] = {}
    elif defect == 'input':
        source: Any = zarr.open_group(str(Path(manifest['source']) / entry['shard']), mode='a')
        source['x'][0, 0, 0, 0] += 1
        # Even a refreshed storage stamp cannot hide mismatched source content.
        entry['ceres_binding']['source_storage_identity'] = tool.shared.storage_identity(
            Path(manifest['source']) / entry['shard'])
        bank.attrs['binding'] = entry['ceres_binding']
    elif defect in ('row', 'legal'):
        name = 'game_id' if defect == 'row' else 'legal_indices'
        values = bank[name][:]
        values[0] += 1
        bank[name][:] = values
        hashes = dict(bank.attrs['array_sha256'])
        hashes[name] = tool.shared.raw.sha_array(values)
        bank.attrs['array_sha256'] = hashes
    else:
        array = bank['policy_logits']
        del array.chunk_store[array._chunk_key((0,))]
    path = Path(args.manifest)
    path.write_text(json.dumps(manifest))
    args.expected_manifest_sha256 = tool.shared.file_sha256(path)
    message = {'duplicate': 'every source shard', 'selected': 'selected fragments',
               'input': 'provenance mismatch|source array digest', 'row': 'row identity differs',
               'legal': 'legal roster', 'missing_chunk': 'missing stored chunk'}[defect]
    with pytest.raises(ValueError, match=message):
        tool.rewrite(args)
    assert not Path(args.out).exists()


def test_failure_preserves_partial_without_complete_receipt(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(tool, 'policy_target', lambda *a, **kw: (_ for _ in ()).throw(ValueError('stop')))
    with pytest.raises(ValueError, match='stop'):
        tool.rewrite(args)
    partial = Path(args.out + '.writing')
    assert partial.is_dir()
    assert not (partial / tool.SUMMARY).exists()
    assert not Path(args.out).exists()


@pytest.mark.parametrize('mode', ['legacy-root-position-v1', 'stored-x-v1'])
def test_old_root_keys_do_not_prove_full_history(tmp_path, monkeypatch, mode):
    args, manifest = fixture(tmp_path, monkeypatch)
    manifest['bt4_lineage']['mode'] = mode
    entry = manifest['entries'][0]
    source: Any = zarr.open_group(str(Path(manifest['source']) / entry['shard']), mode='a')
    attrs = dict(zarr.open_group(entry['bt4'], mode='r').attrs)
    old_x = source['x'][:]
    old_keys = tool.bt4._source_keys(old_x, tool.shared.HISTORY)
    if mode == 'stored-x-v1':
        attrs.update(source_stored_x_sha256=tool.shared.raw.sha_array(old_x),
                     source_derive_summary_sha256=manifest['source_summary_sha256'])
        assert tool.verify_bt4_full_input(source, attrs, manifest)
    source['x'][0, 13, 0, 0] += 1
    np.testing.assert_array_equal(old_keys, tool.bt4._source_keys(source['x'][:], tool.shared.HISTORY))
    if mode == 'stored-x-v1':
        with pytest.raises(ValueError, match='BT4 full input digest differs'):
            tool.verify_bt4_full_input(source, attrs, manifest)
        # Explicit legacy selection cannot bypass a stronger proof already present.
        manifest['bt4_lineage']['mode'] = 'legacy-root-position-v1'
        with pytest.raises(ValueError, match='BT4 full input digest differs'):
            tool.verify_bt4_full_input(source, attrs, manifest)
    else:
        assert not tool.verify_bt4_full_input(source, attrs, manifest)
        _, _, specs = tool.read_manifest(Path(args.manifest), args.expected_manifest_sha256)
        receipt = tool.verify_bt4_lineage(manifest, specs)
        assert receipt['full_historical_input_provenance'] == 'inherited_from_pinned_collection'
        assert receipt['full_input_digest_verified_shards'] == 0


@pytest.mark.parametrize('defect', ['summary_pin', 'counts', 'remap', 'membership', 'missing_mode'])
def test_explicit_legacy_collection_binding(tmp_path, monkeypatch, defect):
    _args, manifest = fixture(tmp_path, monkeypatch)
    pin = manifest['bt4_lineage']['summary']
    path = Path(pin['path'])
    summary = json.loads(path.read_text())
    if defect == 'summary_pin':
        manifest['bt4_lineage']['summary']['sha256'] = '0' * 64
    elif defect == 'counts':
        summary['rows'] -= 1
    elif defect == 'remap':
        summary.pop('remap')
    elif defect == 'membership':
        (path.parent / 'shard_999999.zarr').mkdir()
    else:
        manifest['bt4_lineage']['mode'] = 'implicit'
    if defect in ('counts', 'remap'):
        path.write_text(json.dumps(summary))
        pin['sha256'] = tool.shared.file_sha256(path)
        monkeypatch.setattr(tool, 'LEGACY_BT4_SUMMARY_SHA', pin['sha256'])
    specs = [{'path': 'shard_000000.zarr', 'rows': 32}]
    with pytest.raises(ValueError, match=r'BT4|remap'):
        tool.verify_bt4_lineage(manifest, specs)
