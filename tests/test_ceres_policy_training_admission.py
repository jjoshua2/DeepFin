"""Admission consumes the new producer contract without changing training runtime.

Only metadata fixtures are used here; actual arrays/loss are tested by the producer.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import pytest

from scripts import bt4_one_epoch_screen as epoch
from scripts import ceres_target_mix as producer
from tests.test_bt4_one_epoch_screen import training_only_manifest
from tests.test_bt4_value_training_admission import prepared as value_prepared


def prepared(tmp_path, monkeypatch):
    m, files, _value, _derived, pins = value_prepared(tmp_path, monkeypatch)
    corpus = epoch.corpus_for(m)
    monkeypatch.setitem(epoch.CORPORA, epoch.CERES_PROFILE, corpus)
    m['profile'] = epoch.CERES_PROFILE
    m['input_pins'][str(corpus / producer.SUMMARY)] = m['input_pins'].pop(
        str(corpus / 'bt4_value_rewrite_summary.json'))
    m['ceres_producer_pins'] = {str(tmp_path / 'ceres_target_mix.py'): 'c' * 64,
                              str(tmp_path / 'ceres_derived_sidecar.py'): 'd' * 64}
    source = files[str(epoch.SOURCE / producer.DERIVE_SUMMARY)]
    backend = producer.ceres.backend(argparse.Namespace(pad_final_batch=True, retain_value2=True))
    recipe: dict[str, Any] = {
        'schema': 1, 'status': 'COMPLETE', 'complete': True,
        'kind': 'bt4-ceres-policy', 'algorithm': producer.ALGORITHM,
        'weights': {'bt4': .5, 'ceres': .5},
        'temperatures': {'bt4': .5, 'ceres': .5},
        'mutated_arrays': ['policy_target'],
        'unchanged_arrays': sorted(producer.ARRAYS - {'policy_target'}),
        'source_dir': str(epoch.SOURCE),
        'source_summary_sha256': epoch.COMMON_PINS[str(epoch.SOURCE / producer.DERIVE_SUMMARY)],
        'manifest_sha256': 'e' * 64, 'producer_sha256': dict(m['ceres_producer_pins']),
        'bt4_lineage': {
            'mode': 'legacy-root-position-v1',
            'summary': {'path': str(epoch.ROOT / 'data/lc0/bt4_policy_sidecars/armB_qtemp0005_hist20m/bt4_policy_sidecar_summary.json'),
                        'sha256': producer.LEGACY_BT4_SUMMARY_SHA},
            'root_position_and_history_regime_verified': True,
            'full_historical_input_provenance': 'inherited_from_pinned_collection',
            'full_input_digest_verified_shards': 0,
        },
        'teachers': {
            'bt4': {'model_sha256': '1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0',
                    'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                    'policy_output': '/output/policy'},
            'ceres': {'model_sha256': producer.ceres.MODEL_SHA,
                      'profile': producer.ceres.EXTENDED_PROFILE, 'backend': backend}},
        'rows': 18910484, 'shards': 2309,
        'max_stored_mass_error': .0001, 'max_stored_total_variation': .0001,
        'support_lost_move_entries': 20,
        'outputs': [{**spec, 'changed_rows': spec['rows'],
                     **dict.fromkeys(('source_storage_identity', 'bt4_storage_identity', 'ceres_storage_identity', 'policy_target_sha256', 'files_manifest_sha256', 'attrs_sha256'), 'f' * 64)} for spec in source['shards']],
    }
    derived = {**source, 'policy_target_postprocess': {k: v for k, v in recipe.items() if k != 'outputs'}}
    files[str(corpus / producer.SUMMARY)] = recipe
    files[str(corpus / producer.DERIVE_SUMMARY)] = derived
    qualification = files[m['data_qualification']['path']]
    qualification['profile'] = epoch.CERES_PROFILE
    qualification['rewrite_summary']['path'] = str(corpus / producer.SUMMARY)
    return m, files, recipe, derived, pins


def test_valid_producer_receipt_preserves_historical_training_command(tmp_path, monkeypatch):
    m, files, _recipe, _derived, pins = prepared(tmp_path, monkeypatch)
    epoch.validate(m)
    assert epoch.check_pins(m) == {'same_training_runtime': True}
    assert set(m['ceres_producer_pins'].items()) <= set(pins)
    assert epoch.comparisons(m) == ()
    files[m['runtime_manifest']['path']] = {'runtime': {'executable': sys.executable}}
    command = epoch.train_command(m)
    original = epoch.train_command(training_only_manifest(tmp_path))
    command[command.index('--shards') + 1] = original[original.index('--shards') + 1]
    assert command == original


@pytest.mark.parametrize('defect', ['weight', 'temperature', 'bt4_teacher', 'ceres_teacher',
    'lineage', 'backend', 'value', 'mutated_value', 'producer', 'missing_output', 'missing_hash', 'incomplete', 'partial'])
def test_recipe_mismatch_is_rejected_at_real_admission(tmp_path, monkeypatch, defect):
    m, _files, recipe, derived, _pins = prepared(tmp_path, monkeypatch)
    if defect == 'weight':
        recipe['weights'] = {'bt4': .75, 'ceres': .25}
    elif defect == 'temperature':
        recipe['temperatures']['bt4'] = .25
    elif defect == 'bt4_teacher':
        recipe['teachers']['bt4']['model_sha256'] = '0' * 64
    elif defect == 'lineage':
        recipe['bt4_lineage']['full_historical_input_provenance'] = 'stored_x_digest_verified'
    elif defect == 'backend':
        recipe['teachers']['ceres']['backend'] = {'arbitrary': True}
    elif defect == 'ceres_teacher':
        recipe['teachers']['ceres']['model_sha256'] = '0' * 64
    elif defect == 'value':
        derived['value_scheme'] = {'name': 'sf50-bt4-ceres'}
    elif defect == 'mutated_value':
        recipe['mutated_arrays'].append('search_wdl')
    elif defect == 'producer':
        recipe['producer_sha256'].pop(next(iter(recipe['producer_sha256'])))
    elif defect == 'missing_output':
        recipe['outputs'].pop()
    elif defect == 'missing_hash':
        recipe['outputs'][0].pop('policy_target_sha256')
    elif defect == 'incomplete':
        recipe['complete'] = False
    elif defect == 'partial':
        corpus = epoch.corpus_for(m)
        corpus.with_name(corpus.name + '.writing').mkdir()
    derived['policy_target_postprocess'] = {k: v for k, v in recipe.items() if k != 'outputs'}
    epoch.validate(m)
    with pytest.raises(ValueError, match='Ceres'):
        epoch.check_pins(m)


@pytest.mark.parametrize('defect', ['missing_pins', 'relative_pin', 'schema2'])
def test_manifest_requires_explicit_frozen_producer_and_training_only(tmp_path, monkeypatch, defect):
    m, _files, _recipe, _derived, _pins = prepared(tmp_path, monkeypatch)
    if defect == 'missing_pins':
        m.pop('ceres_producer_pins')
    elif defect == 'relative_pin':
        m['ceres_producer_pins'] = {'ceres_target_mix.py': 'c' * 64}
    else:
        m['schema'] = 2
    with pytest.raises(ValueError, match='Ceres'):
        epoch.validate(m)
