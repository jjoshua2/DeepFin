"""Actual compressed-shard joining and failure cases; no teacher inference."""
from __future__ import annotations

import argparse
import json
import shutil
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import ceres_joint_recipe as joint
from scripts import bt4_two_epoch_train as train


def fixture(tmp_path):
    original = tmp_path / 'sf' / 'shard_000000.zarr'
    original.parent.mkdir()
    g: Any = zarr.open_group(str(original), mode='w')
    for name in joint.ARRAYS:
        shape = (3, 1858) if name in ('policy_target', 'legal_mask') else (3, 3) if name in ('wdl_target', 'search_wdl') else (3, 2) if name == 'x' else (3,)
        a = np.ones(shape, dtype=np.float16)
        if name == 'legal_mask':
            a[:, 2:] = 0
        if name == 'policy_target':
            a[:] = 0
            a[:, 0] = 1
        if name in ('search_wdl', 'wdl_target'):
            a[:] = [.25, .5, .25]
        g.create_dataset(name, data=a, chunks=shape)
    g.attrs.update(derive_value_scheme='sf', derive_value_source='fixture')
    policy, value = tmp_path / 'policy' / original.name, tmp_path / 'value' / original.name
    shutil.copytree(original, policy)
    shutil.copytree(original, value)
    pg: Any = zarr.open_group(str(policy), mode='a')
    pg['policy_target'][:, :2] = np.tile([.25, .75], (3, 1))
    pg.attrs['policy_target_mix_kind'] = 'bt4-ceres'
    vg: Any = zarr.open_group(str(value), mode='a')
    vg['search_wdl'][:] = np.tile([.5, .25, .25], (3, 1))
    vg.attrs.update(derive_value_scheme='sf50-bt4-native25-ceres-dual25',
                    derive_value_source='mixed', value_target_postprocess={'weights': joint.value.WEIGHTS})
    spec = {'path': original.name, 'rows': 3}
    proofs = [{**spec, 'files_manifest_sha256': joint.file_digest(joint.copies.file_map(p))} for p in (policy, value)]
    return original, policy, value, tmp_path / 'joined' / original.name, spec, *proofs


def test_actual_join_preserves_fifteen_arrays_and_exact_parent_targets(tmp_path):
    args = fixture(tmp_path)
    before = [joint.copies.file_map(p) for p in args[:3]]
    receipt = joint.combine_shard(*args)
    output: Any = zarr.open_group(str(args[3]), mode='r')
    np.testing.assert_array_equal(output['policy_target'][:, :2], np.tile([.25, .75], (3, 1)))
    np.testing.assert_array_equal(output['search_wdl'][:], np.tile([.5, .25, .25], (3, 1)))
    assert joint.unchanged(joint.copies.file_map(args[3])) == joint.unchanged(before[0])
    assert [joint.copies.file_map(p) for p in args[:3]] == before
    assert output.attrs['policy_target_mix_kind'] == 'bt4-ceres'
    assert output.attrs['value_target_postprocess']['weights'] == joint.value.WEIGHTS
    assert receipt['files_manifest_sha256'] == joint.file_digest(joint.copies.file_map(args[3]))


@pytest.mark.parametrize('defect', ['parent_tamper', 'non_target', 'mass', 'illegal', 'missing_chunk', 'duplicate_output'])
def test_actual_join_rejects_unqualified_input(tmp_path, defect):
    args: list[Any] = list(fixture(tmp_path))
    p: Any = zarr.open_group(str(args[1]), mode='a')
    if defect == 'duplicate_output':
        args[3].mkdir(parents=True)
    elif defect == 'non_target':
        p['game_id'][0] = 99
    elif defect == 'mass':
        p['policy_target'][0, 0] = .5
    elif defect == 'illegal':
        p['policy_target'][0, :3] = [.25, .5, .25]
    elif defect == 'missing_chunk':
        (args[1] / 'game_id' / '0').unlink()
    else:
        p['policy_target'][0, :2] = [.5, .5]
    # Except the explicit corruption test, let the manifest authenticate the
    # defective parent so semantic/non-target checks themselves must catch it.
    if defect != 'parent_tamper':
        args[5]['files_manifest_sha256'] = joint.file_digest(joint.copies.file_map(args[1]))
    with pytest.raises(ValueError, match=r"parent|target|illegal|chunk|output shard"):
        joint.combine_shard(*args)


def test_named_profile_uses_joint_admission_without_relaxing_old_profiles(monkeypatch):
    called = []
    monkeypatch.setattr(joint, 'verify_training_recipe', lambda prep: called.append(prep) or 'joint')
    assert train.verify_recipe({'marker': True}, joint.PROFILE) == 'joint'
    assert called == [{'marker': True}]


def test_materialize_qualify_and_actual_training_admission(tmp_path, monkeypatch):
    original, policy, values, _dest, spec, pp, vp = fixture(tmp_path)
    sf = {'shards': [spec]}
    pd = {**sf, 'policy_target_postprocess': {'weights': {'bt4': .5, 'ceres': .5}}}
    vd = {**sf, 'value_scheme': {'name': joint.value.VALUE_SCHEME},
          'value_target_postprocess': {'weights': joint.value.WEIGHTS}}
    for root, derived in ((original.parent, sf), (policy.parent, pd), (values.parent, vd)):
        (root / joint.DERIVE).write_text(json.dumps(derived))
    for root, proof in ((policy.parent, pp), (values.parent, vp)):
        (root / 'parent.json').write_text(json.dumps({'outputs': [proof]}))
    monkeypatch.setattr(joint.recipes, 'SOURCE', original.parent)
    monkeypatch.setitem(joint.recipes.CORPORA, 'CeresB50', policy.parent)
    monkeypatch.setitem(joint.recipes.CORPORA, 'B100CeresV25', values.parent)
    monkeypatch.setattr(joint, 'ROWS', 3)
    monkeypatch.setattr(joint, 'SHARDS', 1)
    # Parent teacher qualification has its own real metadata admission tests;
    # this fixture supplies qualified parents and exercises every subsequent
    # compressed copy, publication, qualification and training-admission step.
    monkeypatch.setattr(joint, 'validate_parents', lambda manifest: (sf, {'outputs': [pp]}, {'outputs': [vp]}))
    manifest: dict[str, Any] = {'schema': 1, 'profile': joint.PROFILE, 'source': joint.ref(original.parent / joint.DERIVE)}
    for role, root in (('policy', policy.parent), ('value', values.parent)):
        manifest[role] = {'derive': joint.ref(root / joint.DERIVE), 'recipe': joint.ref(root / 'parent.json')}
    mp = tmp_path / 'manifest.json'
    mp.write_text(json.dumps(manifest))
    corpus = tmp_path / 'complete'
    joint.materialize(argparse.Namespace(manifest=str(mp), expected_manifest_sha256=joint.sha(mp),
                                        out=str(corpus), max_seconds=30, minimum_free_gib=0, max_output_gib=1))
    qualification = tmp_path / 'qualification.json'
    joint.qualify(corpus, qualification)
    prep = {'source': manifest['source'], 'derive_summary': joint.ref(corpus / joint.DERIVE),
            'recipe_summary': joint.ref(corpus / joint.SUMMARY), 'data_qualification': joint.ref(qualification)}
    assert train.verify_recipe(prep, joint.PROFILE) == corpus
    group: Any = zarr.open_group(str(corpus / spec['path']), mode='a')
    group['search_wdl'][0] = [.25, .25, .5]
    with pytest.raises(ValueError, match='output changed'):
        train.verify_recipe(prep, joint.PROFILE)
