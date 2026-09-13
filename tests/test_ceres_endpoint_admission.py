"""Endpoint recipe cannot be admitted under the half-mixture identity."""
from __future__ import annotations

import copy

import pytest

from scripts import bt4_one_epoch_screen as epoch
from scripts import ceres_target_mix as producer


@pytest.mark.parametrize(('profile', 'weight'), [('CeresB50', .5), ('Ceres100', 0.)])
def test_exact_policy_recipe_and_cross_profile_rejection(tmp_path, monkeypatch, profile, weight):
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    for name in ('CeresB50', 'Ceres100'):
        monkeypatch.setitem(epoch.CORPORA, name, corpus)
    specs = [{'path': f'shard_{i:06d}.zarr', 'rows': 8192} for i in range(2309)]
    specs[-1]['rows'] = 3348
    source = {'shards': specs}
    monkeypatch.setattr(epoch.arena, 'read', lambda _path: source)
    monkeypatch.setattr(epoch.arena, 'pin', lambda *_args: None)
    pins = {str(tmp_path / 'ceres_target_mix.py'): 'e' * 64}
    m = {'profile': profile, 'ceres_producer_pins': pins}
    recipe = {'schema': 1, 'complete': True, 'status': 'COMPLETE', 'kind': 'bt4-ceres-policy',
        'algorithm': producer.ALGORITHM, 'weights': {'bt4': weight, 'ceres': 1 - weight},
        'temperatures': {'bt4': .5, 'ceres': .5}, 'source_dir': str(epoch.SOURCE),
        'source_summary_sha256': epoch.COMMON_PINS[str(epoch.SOURCE / 'derive_targets_summary.json')],
        'rows': 18910484, 'shards': 2309, 'mutated_arrays': ['policy_target'],
        'unchanged_arrays': sorted(producer.ARRAYS - {'policy_target'}),
        'teachers': {'bt4': {'model_sha256': '1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0',
                         'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'policy_output': '/output/policy'},
                  'ceres': {'model_sha256': '44aa02c775456f18ed464e33fc37b8e4abf58d7bf8f4cfb3ff19492e32e56df3',
                            'profile': 'ceres-c3-fixed32-compact-v2', 'backend': copy.deepcopy(epoch.CERES_BACKEND)}},
        'max_stored_mass_error': 0., 'max_stored_total_variation': 0.,
        'bt4_lineage': {'mode': 'legacy-root-position-v1', 'summary': {
            'path': str(epoch.ROOT / 'data/lc0/bt4_policy_sidecars/armB_qtemp0005_hist20m/bt4_policy_sidecar_summary.json'),
            'sha256': producer.LEGACY_BT4_SUMMARY_SHA},
            'root_position_and_history_regime_verified': True,
            'full_historical_input_provenance': 'inherited_from_pinned_collection',
            'full_input_digest_verified_shards': 0}, 'producer_sha256': pins,
        'outputs': [{**row, **dict.fromkeys(('source_storage_identity', 'bt4_storage_identity',
            'ceres_storage_identity', 'policy_target_sha256', 'files_manifest_sha256', 'attrs_sha256'), 'f' * 64)}
                 for row in specs]}
    epoch.verify_ceres_recipe(m, recipe, source)
    other = {**m, 'profile': 'Ceres100' if profile == 'CeresB50' else 'CeresB50'}
    with pytest.raises(ValueError, match='mixture or historical source'):
        epoch.verify_ceres_recipe(other, recipe, source)
    wrong_source = {**recipe, 'source_summary_sha256': '0' * 64}
    with pytest.raises(ValueError, match='mixture or historical source'):
        epoch.verify_ceres_recipe(m, wrong_source, source)
