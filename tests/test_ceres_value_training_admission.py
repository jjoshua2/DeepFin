"""Three-teacher value admission metadata; actual storage/loss tested beside producer."""
from __future__ import annotations

import copy
import sys
from typing import Any

import pytest

from scripts import bt4_one_epoch_screen as epoch
from scripts import ceres_value_mix as producer
from tests.test_bt4_one_epoch_screen import training_only_manifest
from tests.test_bt4_value_training_admission import prepared as value_prepared


def prepared(tmp_path, monkeypatch):
    m, files, _old, _derived, pins = value_prepared(tmp_path, monkeypatch)
    corpus = epoch.corpus_for(m)
    monkeypatch.setitem(epoch.CORPORA, epoch.CERES_VALUE_PROFILE, corpus)
    m['profile'] = epoch.CERES_VALUE_PROFILE
    m['input_pins'][str(corpus / producer.SUMMARY)] = m['input_pins'].pop(str(corpus / 'bt4_value_rewrite_summary.json'))
    m['ceres_producer_pins'] = {str(tmp_path / 'ceres_value_mix.py'): 'c' * 64,
                              str(tmp_path / 'ceres_target_mix.py'): 'd' * 64}
    parent = epoch.CORPORA['B100']
    original = files[str(parent / producer.DERIVE_SUMMARY)]
    sf = files[str(epoch.SOURCE / producer.DERIVE_SUMMARY)]
    recipe: dict[str, Any] = {'schema': 1, 'status': 'COMPLETE', 'complete': True,
        'kind': 'sf-bt4-ceres-value', 'algorithm': producer.ALGORITHM,
        'weights': copy.deepcopy(producer.WEIGHTS), 'ceres_value_profile': copy.deepcopy(producer.CERES_VALUE_PROFILE),
        'wdl_order': 'WDL', 'wdl_pov': 'side_to_move', 'wdl_kind': 'probabilities',
        'mutated_arrays': ['search_wdl'], 'unchanged_arrays': sorted(producer.ARRAYS - {'search_wdl'}),
        'source_dir': str(parent), 'sf_source_dir': str(epoch.SOURCE),
        'source_summary_sha256': epoch.B100_PARENT_PINS[producer.DERIVE_SUMMARY],
        'source_policy_summary_sha256': epoch.B100_PARENT_PINS[producer.POLICY_SUMMARY],
        'sf_summary_sha256': epoch.COMMON_PINS[str(epoch.SOURCE / producer.DERIVE_SUMMARY)],
        'manifest_sha256': 'e' * 64, 'producer_sha256': dict(m['ceres_producer_pins']),
        'teachers': {'bt4': {'onnx': '/fixture/BT4.onnx', 'model_sha256': producer.BT4_MODEL,
            'requested_wdl': {'kind': 'probabilities', 'output': '/output/wdl'},
            'producer': copy.deepcopy(producer.HISTORICAL_BT4_PRODUCER)},
            'ceres': {'model_sha256': producer.ceres.MODEL_SHA,
                      'profile': producer.ceres.EXTENDED_PROFILE, 'backend': copy.deepcopy(epoch.CERES_BACKEND)}},
        'value_scheme': producer.VALUE_SCHEME, 'value_source': producer.value_source(),
        'rows': 18910484, 'shards': 2309, 'changed_rows': 18000000,
        'max_stored_mass_error': .0001, 'max_stored_l1_error': .0001,
        'outputs': [{**spec, 'changed_rows': spec['rows'],
            **dict.fromkeys(('source_storage_identity', 'sf_storage_identity', 'bt4_storage_identity',
                            'ceres_storage_identity', 'search_wdl_sha256', 'attrs_sha256', 'files_manifest_sha256'),
                           'f' * 64)} for spec in sf['shards']]}
    derived = {**original, 'value_scheme': {'name': producer.VALUE_SCHEME, 'source': producer.value_source()},
               'value_target_postprocess': {k: v for k, v in recipe.items() if k != 'outputs'}}
    files[str(corpus / producer.SUMMARY)] = recipe
    files[str(corpus / producer.DERIVE_SUMMARY)] = derived
    qualification = files[m['data_qualification']['path']]
    qualification['profile'] = epoch.CERES_VALUE_PROFILE
    qualification['rewrite_summary']['path'] = str(corpus / producer.SUMMARY)
    return m, files, recipe, derived, pins


def test_valid_value_recipe_keeps_exact_historical_train_command(tmp_path, monkeypatch):
    m, files, _recipe, _derived, pins = prepared(tmp_path, monkeypatch)
    epoch.validate(m)
    assert epoch.check_pins(m) == {'same_training_runtime': True}
    assert set(m['ceres_producer_pins'].items()) <= set(pins)
    assert {(str(epoch.CORPORA['B100'] / k), v) for k, v in epoch.B100_PARENT_PINS.items()} <= set(pins)
    assert epoch.comparisons(m) == ()
    files[m['runtime_manifest']['path']] = {'runtime': {'executable': sys.executable}}
    command = epoch.train_command(m)
    original = epoch.train_command(training_only_manifest(tmp_path))
    command[command.index('--shards') + 1] = original[original.index('--shards') + 1]
    assert command == original


@pytest.mark.parametrize('defect', ['weights', 'head_temperature', 'head_weight', 'pov', 'order',
    'historical_producer', 'backend', 'policy', 'output', 'proof', 'incomplete', 'missing_pins', 'schema2'])
def test_reject_three_teacher_value_contract_drift(tmp_path, monkeypatch, defect):
    m, _files, recipe, derived, _pins = prepared(tmp_path, monkeypatch)
    if defect == 'weights':
        recipe['weights'] = {'sf': .75, 'bt4': .125, 'ceres': .125}
    elif defect == 'head_temperature':
        recipe['ceres_value_profile']['primary_temperature'] = .5
    elif defect == 'head_weight':
        recipe['ceres_value_profile']['secondary_weight'] = .5
    elif defect in ('pov', 'order'):
        recipe['wdl_' + defect] = 'white' if defect == 'pov' else 'LDW'
    elif defect == 'historical_producer':
        recipe['teachers']['bt4']['producer']['scripts/bt4_derived_wdl_sidecar.py'] = '0' * 64
    elif defect == 'backend':
        recipe['teachers']['ceres']['backend']['outputs'] = ['policy', 'value']
    elif defect == 'policy':
        derived['policy_target_postprocess'] = {'kind': 'CeresB50'}
    elif defect == 'output':
        recipe['outputs'].pop()
    elif defect == 'proof':
        recipe['outputs'][0].pop('search_wdl_sha256')
    elif defect == 'incomplete':
        recipe['status'] = 'WRITING'
    elif defect == 'missing_pins':
        m.pop('ceres_producer_pins')
    else:
        m['schema'] = 2
    derived['value_target_postprocess'] = {k: v for k, v in recipe.items() if k != 'outputs'}
    if defect in ('missing_pins', 'schema2'):
        with pytest.raises(ValueError, match='Ceres'):
            epoch.validate(m)
    else:
        epoch.validate(m)
        with pytest.raises(ValueError, match='Ceres'):
            epoch.check_pins(m)
