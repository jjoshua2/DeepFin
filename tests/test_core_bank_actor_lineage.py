"""Adjacent actor lineage controls; synthetic closure and inert serving only."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import core_finite_collect as collect
from tests.test_core_finite_collect import protocol, roster


def frozen(path: Path, value: Any) -> dict[str, str]:
    collect.write_json(path, value)
    path.chmod(0o444)
    return {'path': str(path), 'sha256': collect.digest(path)}


def bank_predecessor(tmp_path: Path, p: dict[str, Any]) -> dict[str, Any]:
    """A faithful emitted metadata shape, not a real trained model/authority."""
    actor = Path(p['actor']['path'])
    if actor.name != 'learner.pt':
        actor = tmp_path / 'bank-parent' / 'learner.pt'
        actor.parent.mkdir()
        actor.write_bytes(b'synthetic actor bytes; inert loader only')
        actor.chmod(0o444)
    checkpoint = {'path': str(actor), 'sha256': collect.digest(actor)}
    architecture = p['actor']['architecture']
    run, source = '33333333-3333-4333-8333-333333333333', 'f' * 40
    contract: dict[str, Any] = {'schema': 'core_generation_bank_learning_v1', 'run_uuid': run, 'source_revision': source,
        'source_files': dict.fromkeys(collect.BANK_PARENT_SOURCE_FILES, 'a' * 64),
        'architecture': architecture, 'initial_model': {'path': str(tmp_path / 'historical.pt'), 'sha256': 'e' * 64},
        'initial_actor_step': 9000, 'first_generation': 2, 'batch_size': 2,
        'games_per_generation': 1, 'max_games_per_generation': 1, 'updates_per_segment': [2, 2],
        'replay_generations': 4, 'seed': 7, 'runtime': {'device': 'cpu'}, 'lr': {}, 'mirror_prob': 0.5,
        'limits': {}, 'bank_limits': {}}
    cref = frozen(tmp_path / 'bank-parent-contract.json', contract)
    banks = [{'root_acceptance': {'path': str(tmp_path / 'historic-root.json'), 'sha256': '1' * 64},
              'independent_review': {'path': str(tmp_path / 'historic-review.json'), 'sha256': '2' * 64}}]
    stage = {'schema': 'core_generation_bank_stage_v1', 'contract_sha256': cref['sha256'], 'segment': 1,
             'banks': banks, 'parent': None, 'output': str(actor.parent)}
    sref = frozen(tmp_path / 'bank-parent-stage.json', stage)
    boundary = {'status': 'COMPLETED_CORE_GENERATION_BANK_SEGMENT_V1', 'run_uuid': run, 'source_revision': source,
        'contract_sha256': cref['sha256'], 'stage': sref, 'parent': None, 'segment': 1, 'logical_generation': 2,
        'step_start': 0, 'step': 2, 'applied_updates': 2, 'applied_views': 4, 'drawn_rows': 4,
        'accepted_banks': banks, 'bank_identities': [{'run_uuid': p['run_uuid'], 'collection_uuid': p['collection_uuid'],
            'generation': 2, 'source_revision': p['source']['revision'], 'architecture': architecture,
            'actor_sha256': 'e' * 64, 'actor_checkpoint_step': 9000}],
        'initial_model': contract['initial_model'], 'initial_actor_step': 9000, 'checkpoint': checkpoint,
        'collector_lineage_qualified': False}
    bpath = actor.parent / 'boundary.json'
    if bpath.exists():
        bpath.unlink()  # Replace only this test's fictional old-family fixture.
    bref = frozen(bpath, boundary)
    command = ['python', '-m', 'scripts.core_generation_bank_learner', '--contract', cref['path'],
               '--contract-sha256', cref['sha256'], '--stage', sref['path'], '--stage-sha256', sref['sha256']]
    terminal = {'status': 'PASS', 'returncode': 0, 'owned_processes_closed': True, 'custody': 'OWNED_PROCESSES_CLOSED',
                'supervisor_sha256': 'd' * 64, 'command': command, 'source_revision': 'd' * 40}
    tref = frozen(tmp_path / 'bank-parent-terminal.json', terminal)
    Path(tref['path']).chmod(0o644)  # Stable independently owned terminals may be writable.
    p['generation'] = 3
    p['actor'] = {'kind': 'bank_learner', **checkpoint, 'architecture': architecture, 'expected_checkpoint_step': 2,
        'predecessor': {'boundary': bref, 'owner_terminal': tref, 'owner_sha256': 'd' * 64, 'command': command},
        'lineage': {'schema': collect.BANK_LINEAGE_SCHEMA, 'run_uuid': run, 'source_revision': source,
                    'contract_sha256': cref['sha256'], 'segment': 1, 'logical_generation': 2}}
    return {'contract': contract, 'stage': stage, 'boundary': boundary, 'terminal': terminal, 'protocol': p}


def replace(ref: dict[str, str], value: Any) -> dict[str, str]:
    path = Path(ref['path'])
    path.unlink()
    return frozen(path, value)


def test_public_validator_joins_real_counter_domains_and_separate_sources(tmp_path: Path):
    p = protocol(tmp_path)
    case = bank_predecessor(tmp_path, p)
    collect.validate_protocol(p, roster())
    parent = collect.closed_bank_predecessor(p)
    assert p['generation'] == 3
    assert parent['logical_generation'] == 2
    assert parent['segment'] == 1
    assert parent['step'] == p['actor']['expected_checkpoint_step'] == 2
    assert parent['initial_actor_step'] == 9000
    assert p['run_uuid'] != parent['run_uuid']
    assert p['source']['revision'] != parent['source_revision']
    assert case['terminal']['source_revision'] != parent['source_revision']
    assert parent['collector_lineage_qualified'] is False


@pytest.mark.parametrize('fault', ['schema', 'extra', 'bool_segment', 'float_logical', 'wrong_run', 'wrong_source',
    'wrong_contract', 'wrong_segment', 'wrong_logical', 'wrong_step', 'architecture', 'closure', 'old_command',
    'history', 'logical_history', 'dose', 'pending', 'failure', 'writable', 'symlink', 'old_kind'])
def test_lineage_mismatches_fail_before_actor_work(tmp_path: Path, fault: str):
    p = protocol(tmp_path)
    case = bank_predecessor(tmp_path, p)
    actor, lineage, binding = p['actor'], p['actor']['lineage'], p['actor']['predecessor']
    if fault == 'schema':
        lineage['schema'] = 'unchecked_parent_v1'
    elif fault == 'extra':
        lineage['allow_foreign_parent'] = True
    elif fault == 'bool_segment':
        lineage['segment'] = True
    elif fault == 'float_logical':
        lineage['logical_generation'] = 2.0
    elif fault == 'wrong_run':
        lineage['run_uuid'] = p['run_uuid']
    elif fault == 'wrong_source':
        lineage['source_revision'] = p['source']['revision']
    elif fault == 'wrong_contract':
        lineage['contract_sha256'] = '0' * 64
    elif fault == 'wrong_segment':
        lineage['segment'] = 2
    elif fault == 'wrong_logical':
        lineage['logical_generation'] = 1
    elif fault == 'wrong_step':
        actor['expected_checkpoint_step'] = 9000
    elif fault == 'architecture':
        actor['architecture'] = {**actor['architecture'], 'another_field': 1}
    elif fault == 'closure':
        case['terminal']['owned_processes_closed'] = False
        binding['owner_terminal'] = replace(binding['owner_terminal'], case['terminal'])
    elif fault == 'old_command':
        binding['command'][2] = 'scripts.core_finite_learner'
    elif fault in {'history', 'logical_history', 'dose'}:
        if fault == 'history':
            case['boundary']['accepted_banks'] = []
        elif fault == 'logical_history':
            case['boundary']['bank_identities'][0]['generation'] = 1
        else:
            case['boundary']['step'] = 3
        binding['boundary'] = replace(binding['boundary'], case['boundary'])
    elif fault == 'pending':
        frozen(Path(actor['path']).parent / 'boundary.pending.json', {})
    elif fault == 'failure':
        frozen(collect.failure_marker(Path(actor['path']).parent), {})
    elif fault == 'writable':
        Path(binding['boundary']['path']).chmod(0o644)
    elif fault == 'symlink':
        alias = tmp_path / 'alias.json'
        alias.symlink_to(binding['boundary']['path'])
        binding['boundary'] = {**binding['boundary'], 'path': str(alias)}
    else:
        actor['kind'] = 'learner'
    with pytest.raises(ValueError, match=r'bank|predecessor'):
        collect.validate_protocol(p, roster())


@pytest.mark.parametrize('fault', ['none', 'missing', 'wrong_source', 'wrong_segment', 'extra', 'float_segment'])
def test_actual_load_capture_requires_tag_before_one_constructor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str):
    import torch
    from chess_anti_engine.uci import model_loader

    p = protocol(tmp_path)
    bank_predecessor(tmp_path, p)
    tag = collect.bank_lineage_identity(p['actor'])
    payload: dict[str, Any] = {'model': {'tensor.' + str(i): torch.tensor([float(i)]) for i in range(494)},
        'arch': deepcopy(p['actor']['architecture']), 'step': 2, collect.BANK_CHECKPOINT_KEY: dict(tag)}
    if fault == 'missing':
        del payload[collect.BANK_CHECKPOINT_KEY]
    elif fault == 'wrong_source':
        payload[collect.BANK_CHECKPOINT_KEY]['source_revision'] = p['source']['revision']
    elif fault == 'wrong_segment':
        payload[collect.BANK_CHECKPOINT_KEY]['segment'] = 2
    elif fault == 'extra':
        payload[collect.BANK_CHECKPOINT_KEY]['allow'] = True
    elif fault == 'float_segment':
        payload[collect.BANK_CHECKPOINT_KEY]['segment'] = 1.0
    loads, builds = [], []

    def load(*_args: Any, **_kwargs: Any) -> Any:
        loads.append(True)
        return payload

    class InertActor:
        def state_dict(self) -> dict[str, Any]:
            return payload['model']

        def parameters(self) -> list[Any]:
            return [SimpleNamespace(device='cuda:0')]

    def build(_config: Any) -> Any:
        builds.append(True)
        return InertActor()

    def public_loader(path: Path, **_kwargs: Any) -> Any:
        torch.load(path, map_location='cuda:0', weights_only=True)
        return model_loader.build_model(p['actor']['architecture'])

    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(model_loader, 'build_model', build)
    monkeypatch.setattr(model_loader, 'load_model_from_checkpoint', public_loader)
    if fault == 'none':
        _, serving = collect.load_actor(p)
        assert serving['bank_learner_tag'] == tag
        assert serving['actor_lineage'] == p['actor']['lineage']
        assert serving['predecessor_boundary'] == p['actor']['predecessor']['boundary']
        assert serving['checkpoint_step'] == 2
        assert serving['logical_generation'] == 3
        assert serving['bank_learner_tag']['logical_generation'] == 2
        assert serving['model_build_count'] == 1
        assert builds == [True]
    else:
        with pytest.raises(ValueError, match=r'loaded bank actor family tag'):
            collect.load_actor(p)
        assert builds == []
    assert loads == [True]


def test_actual_public_cli_dispatch_keeps_existing_argv_and_consumes_lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    p = protocol(tmp_path)
    bank_predecessor(tmp_path, p)
    p['roster'] = frozen(tmp_path / 'roster.json', roster())
    starts = tmp_path / 'starts.epd'
    starts.write_text('fictional source-only starts')
    p['starts'] = {'path': str(starts), 'sha256': collect.digest(starts)}
    p['resolved_profile_sha256'] = collect.hashlib.sha256(collect.canonical({})).hexdigest()
    ref = frozen(tmp_path / 'protocol.json', p)
    ad = frozen(tmp_path / 'admission.json', {})
    calls = []
    monkeypatch.setattr(collect, 'exclusion_inputs', lambda _p: None)
    monkeypatch.setattr(collect, 'verify_execution', lambda *_args: calls.append('admission'))
    monkeypatch.setattr(collect, 'source_guard', lambda _p: {})
    monkeypatch.setattr(collect, 'build_configs', lambda _p, _r: ({}, {}))

    def execute(actual: dict[str, Any], *_args: Any) -> int:
        calls.append(actual['actor']['lineage'])
        return 0

    monkeypatch.setattr(collect, 'execute', execute)
    monkeypatch.setattr(sys, 'argv', ['collector', '--execute', '--protocol', ref['path'], '--protocol-sha256', ref['sha256'],
                                    '--admission', ad['path'], '--admission-sha256', ad['sha256']])
    assert collect.main() == 0
    assert calls == ['admission', p['actor']['lineage']]


def audit_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, legacy: bool = False) -> Any:
    """Reuse real native/raw/Zarr/FakeTB audit, not a replacement audit body."""
    from scripts import core_finite_admit as audit
    from tests.test_core_finite_admit import entry_case

    case = entry_case(tmp_path, monkeypatch, generation=3, legacy=legacy)
    _, p = audit.learning.read_ref(case.protocol['producer_protocol'])
    bank_predecessor(tmp_path, p)
    p['source']['sha256'] = dict(case.spec['source_files'])
    pref = frozen(tmp_path / 'bank-producer.json', p)
    case.protocol['producer_protocol'] = pref
    case.spec['actor_checkpoint_step'] = 2
    producer_root = Path(p['output_root'])
    _, serving = audit.learning.read_ref(audit.ref(producer_root / 'actor-serving.json'))
    serving.update(checkpoint_step=2, actor_kind='bank_learner', actor_lineage=p['actor']['lineage'],
                   bank_learner_tag=collect.bank_lineage_identity(p['actor']), predecessor_boundary=p['actor']['predecessor']['boundary'])
    _, readout = audit.learning.read_ref(case.protocol['producer_readout'])
    producer_root.chmod(0o755)
    serving_ref = replace(audit.ref(producer_root / 'actor-serving.json'), serving)
    readout.update(protocol_sha256=pref['sha256'], actor_serving=serving_ref)
    readout_ref = replace(case.protocol['producer_readout'], readout)
    case.protocol['producer_readout'] = readout_ref
    ad = frozen(tmp_path / 'bank-producer-admission.json', {'status': 'ROOT_ADMITS_CORE_GENERATION_V1',
        'protocol_sha256': pref['sha256'], 'run_uuid': p['run_uuid'], 'generation': p['generation'],
        'source_revision': p['source']['revision'], 'actor_sha256': p['actor']['sha256']})
    command = ['python', '-m', 'scripts.core_finite_collect', '--execute', '--protocol', pref['path'], '--protocol-sha256', pref['sha256'],
               '--admission', ad['path'], '--admission-sha256', ad['sha256']]
    _, terminal = audit.learning.read_ref(case.protocol['producer_owner']['terminal'])
    terminal['command'] = command
    case.protocol['producer_owner']['terminal'] = frozen(tmp_path / 'bank-producer-terminal.json', terminal)
    case.protocol['producer_owner']['command'] = command
    case.protocol['producer_inventory'] = frozen(tmp_path / 'bank-inventory.json',
        {'path': str(producer_root), 'files': collect.inventory(producer_root, seal=True)})
    case.protocol['runtime'] = {'owner': {'path': '/synthetic-cpu-owner.py', 'sha256': 'd' * 64}, 'seconds': 300,
                                'inner_seconds': 30, 'cpu_affinity': [20, 22], 'python': '/python'}
    return SimpleNamespace(case=case, producer=p, serving=serving, producer_root=producer_root)


@pytest.mark.parametrize('fault', ['none', 'missing_tag', 'wrong_lineage', 'wrong_boundary'])
def test_actual_data_public_entry_requires_serving_lineage_and_preserves_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str):
    from scripts import core_finite_admit as audit
    from scripts import core_generation_data_admit as data
    from scripts import core_generation_bank_acceptance as bank_api
    from tests.test_core_finite_admit import bind_data_case

    fixture = audit_case(tmp_path, monkeypatch)
    case = fixture.case
    if fault != 'none':
        serving = fixture.serving
        if fault == 'missing_tag':
            del serving['bank_learner_tag']
        elif fault == 'wrong_lineage':
            serving['actor_lineage'] = {**serving['actor_lineage'], 'segment': 2}
        else:
            serving['predecessor_boundary'] = {'path': str(tmp_path / 'wrong.json'), 'sha256': '0' * 64}
        _, readout = audit.learning.read_ref(case.protocol['producer_readout'])
        fixture.producer_root.chmod(0o755)
        readout['actor_serving'] = replace(audit.ref(fixture.producer_root / 'actor-serving.json'), serving)
        case.protocol['producer_readout'] = replace(case.protocol['producer_readout'], readout)
        case.protocol['producer_inventory'] = frozen(tmp_path / 'bad-serving-inventory.json',
            {'path': str(fixture.producer_root), 'files': collect.inventory(fixture.producer_root, seal=True)})
    pref = bind_data_case(tmp_path, case)
    if fault != 'none':
        with pytest.raises(ValueError, match=r'bank actor serving lineage'):
            data.execute(pref)
        assert not (Path(case.protocol['output_root']) / 'generation.json').exists()
        return
    result = data.execute(pref)
    assert result['status'] == 'AUDITED_CORE_DATA_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1'
    assert result['logical_generation'] == 3
    assert result['actor_checkpoint_step'] == 2
    assert fixture.serving['logical_generation'] == 3
    assert fixture.serving['actor_lineage']['logical_generation'] == 2
    _, candidate = audit.learning.read_ref(result['generation_manifest'])
    assert candidate['schema'] == 'CORE_TRAIN_GENERATION_DATA_CANDIDATE_V1'
    assert candidate['learning_admitted'] is False
    _, wrapper = audit.learning.read_ref(candidate['games'][0]['producer_proof'])
    assert wrapper['status'] == 'VERIFIED_RESOLVED_CORE_DATA_GAME_CANDIDATE_V1'
    # Fictional root/independent receipts exercise the real metadata bank API.
    # They are not actual resource-owner or real-data qualification receipts.
    command = ['/python', '-B', '-m', 'scripts.core_generation_data_admit', '--protocol', pref['path'], '--protocol-sha256', pref['sha256']]
    source = {'revision': case.spec['source_revision'], 'checkout': str(audit.learning.REPO), 'source_files': case.spec['source_files']}
    terminal = frozen(tmp_path / 'fake-audit-terminal.json', {'status': 'PASS', 'returncode': 0,
        'owned_processes_closed': True, 'custody': 'OWNED_PROCESSES_CLOSED', 'resource': 'cpu', 'seconds': 300,
        'wall_seconds': 1.0, 'supervisor_sha256': 'd' * 64, 'command': command, 'source_revision': 'a' * 40,
        'cwd': '/synthetic-owner-checkout', 'known_processes': {'123': 456}, 'supervisor_pid': 122})
    ad = frozen(tmp_path / 'fake-audit-admission.json', {'status': bank_api.ADMISSION_STATUS, 'audit_protocol': pref,
        'audit_source': source, 'command': command, 'owner': case.protocol['runtime']['owner'], 'resource': 'cpu', 'seconds': 300})
    root = {'status': bank_api.ROOT_STATUS, 'audit_protocol': pref, 'data_spec': case.protocol['data_spec'],
        'readout': audit.ref(Path(case.protocol['output_root']) / 'readout.json'), 'candidate': result['generation_manifest'],
        'audit_source': source, 'learning_admitted': False, 'audit_owner': {'terminal': terminal, 'admission': ad,
        'command': command, 'source_revision': 'a' * 40, 'cwd': '/synthetic-owner-checkout',
        'closure': {'supported_eof': True, 'exit_code': 0, 'supervisor_closed': True, 'owned_births_closed': True,
                    'supervisor_pid': 122, 'known_processes': {'123': 456}}}}
    root_ref = frozen(tmp_path / 'fake-root-acceptance.json', root)
    review_ref = frozen(tmp_path / 'fake-independent-review.json', {'status': bank_api.REVIEW_STATUS, 'root_acceptance': root_ref,
        'findings': [], **{key: root[key] for key in ('audit_protocol', 'data_spec', 'readout', 'candidate')}})
    view = bank_api.validate_accepted_generation(root_ref, review_ref, max_file_bytes=4 << 20, max_total_bytes=16 << 20, max_games=1)
    assert view.identity['generation'] == 3
    assert view.identity['actor_checkpoint_step'] == 2
    assert view.identity['source_revision'] != fixture.producer['actor']['lineage']['source_revision']
    assert len(view.games) == 1
    assert view.evidence['learning_admitted'] is False


def test_legacy_recipe_audit_rejects_bank_actor_without_new_schema_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from scripts import core_finite_admit as audit

    fixture = audit_case(tmp_path, monkeypatch, legacy=True)
    case = fixture.case
    c = {'schema': 'core_finite_learning_v1', 'run_uuid': case.spec['run_uuid'], 'source_revision': case.spec['source_revision'],
        'source_files': case.spec['source_files'], 'architecture': case.spec['architecture'], 'donor': case.actor,
        'batch_size': 1, 'games_per_generation': 1, 'max_games_per_generation': 1, 'updates_per_generation': [1],
        'replay_generations': 4, 'seed': 0, 'mirror_prob': 0,
        'runtime': {'device': 'cpu', 'use_amp': False, 'torch_version': 'unused', 'numpy_version': 'unused', 'torch_threads': 2, 'cuda_devices': 0},
        'lr': {'peak': 1e-4, 'floor': 1e-4, 'warmup_steps': 0, 'warmup_start': 1e-4, 'first_period': 2},
        'limits': {'max_rows': 1, 'max_decoded_window_bytes': 8 << 20}}
    p = {**case.protocol, 'schema': 'CORE_FINITE_ADMISSION_V1', 'learner_contract': frozen(tmp_path / 'recipe.json', c)}
    del p['data_spec']
    with pytest.raises(ValueError, match=r'requires recipe-independent data audit'):
        audit.execute(frozen(tmp_path / 'legacy-bank-actor-audit.json', p))
    assert not (Path(p['output_root']) / 'generation.json').exists()


@pytest.mark.parametrize('raw', [b'{"key":1,"key":2}', b'{"key":NaN}', b'{"key":1e999}'])
def test_new_metadata_reader_refuses_duplicate_and_nonfinite_json(tmp_path: Path, raw: bytes):
    path = tmp_path / 'invalid.json'
    path.write_bytes(raw)
    path.chmod(0o444)
    ref = {'path': str(path), 'sha256': collect.digest(path)}
    with pytest.raises(ValueError, match=r'duplicate|nonfinite'):
        collect._bank_checked(ref)
