"""Anchored actor capture controls: real decode, inert CPU serving, no CUDA."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts import core_finite_collect as collect
from tests.test_core_bank_actor_lineage import audit_case, bank_predecessor, frozen, replace
from tests.test_core_finite_collect import protocol
from tests.test_frozen_policy_anchor import disabled


def project(p: dict[str, Any]) -> None:
    """Rehash a strict fictional closure graph; do not mock lineage predicates."""
    actor = p['actor']
    binding = actor['predecessor']
    boundary = collect._bank_checked(binding['boundary'])
    stage = collect._bank_checked(boundary['stage'])
    old_contract_ref = {'path': binding['command'][-7], 'sha256': boundary['contract_sha256']}
    contract = collect._bank_checked(old_contract_ref)
    identity = disabled()
    contract.update(schema='core_generation_bank_policy_anchor_v1', policy_anchor=identity)
    for name in ('scripts/core_generation_bank_policy_anchor.py', 'chess_anti_engine/train/frozen_policy_anchor.py', 'chess_anti_engine/policy_anchor_spec.py'):
        contract['source_files'][name] = 'a' * 64
    cref = replace(old_contract_ref, contract)
    stage.update(schema='core_generation_bank_policy_anchor_stage_v1', contract_sha256=cref['sha256'])
    sref = replace(boundary['stage'], stage)
    boundary.update(status='COMPLETED_CORE_GENERATION_BANK_POLICY_ANCHOR_SEGMENT_V1', contract_sha256=cref['sha256'], stage=sref, policy_anchor=identity)
    binding['boundary'] = replace(binding['boundary'], boundary)
    command = binding['command']
    command[-9] = 'scripts.core_generation_bank_policy_anchor'
    command[-5] = cref['sha256']
    command[-1] = sref['sha256']
    terminal = collect._bank_checked(binding['owner_terminal'], readonly=False)
    terminal['command'] = command
    binding['owner_terminal'] = replace(binding['owner_terminal'], terminal)
    actor.update(kind=collect.ANCHOR_BANK_KIND, lineage={**actor['lineage'], 'schema': collect.ANCHOR_BANK_LINEAGE_SCHEMA,
        'contract_sha256': cref['sha256'], 'policy_anchor': identity})


@pytest.mark.parametrize('fault', ['none', 'missing', 'reference', 'old_family'])
def test_actual_decode_and_capture_refuse_before_constructor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str):
    from chess_anti_engine.uci import model_loader

    p = protocol(tmp_path)
    bank_predecessor(tmp_path, p)
    project(p)
    actor = p['actor']
    tag = collect.bank_lineage_identity(actor)
    payload: dict[str, Any] = {'model': {'tensor.' + str(i): torch.tensor([float(i)]) for i in range(494)},
        'arch': deepcopy(actor['architecture']), 'step': 2, collect.ANCHOR_BANK_CHECKPOINT_KEY: tag,
        'frozen_reference_policy_anchor_v1': disabled()}
    if fault == 'missing':
        del payload[collect.ANCHOR_BANK_CHECKPOINT_KEY]
    elif fault == 'reference':
        payload['frozen_reference_policy_anchor_v1'] = {'wrong': 'reference'}
    elif fault == 'old_family':
        payload[collect.BANK_CHECKPOINT_KEY] = {key: value for key, value in tag.items() if key != 'policy_anchor'}
    path = Path(actor['path'])
    path.chmod(0o644)
    torch.save(payload, path)
    path.chmod(0o444)
    actor['sha256'] = collect.digest(path)
    boundary = collect._bank_checked(actor['predecessor']['boundary'])
    boundary['checkpoint'] = {'path': str(path), 'sha256': actor['sha256']}
    actor['predecessor']['boundary'] = replace(actor['predecessor']['boundary'], boundary)
    real_load = torch.load
    loads, builds = [], []
    def load(*args: Any, **kwargs: Any) -> Any:
        loads.append(True)
        return real_load(*args, **kwargs)
    class InertActor:
        def state_dict(self) -> dict[str, Any]:
            return payload['model']
        def parameters(self) -> list[Any]:
            return [SimpleNamespace(device='cuda:0')]
    def build(_config: Any) -> Any:
        builds.append(True)
        return InertActor()
    def cpu_loader(checkpoint: Path, **_kwargs: Any) -> Any:
        torch.load(checkpoint, map_location='cpu', weights_only=True)
        return model_loader.build_model(actor['architecture'])
    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(model_loader, 'build_model', build)
    monkeypatch.setattr(model_loader, 'load_model_from_checkpoint', cpu_loader)
    if fault == 'none':
        _, serving = collect.load_actor(p)
        assert serving['policy_anchor_bank_tag'] == tag
        assert serving['actor_kind'] == collect.ANCHOR_BANK_KIND
        assert serving['logical_generation'] == 3
        assert serving['policy_anchor_bank_tag']['logical_generation'] == 2
        assert 'bank_learner_tag' not in serving
        assert builds == [True]
    else:
        with pytest.raises(ValueError, match=r'loaded.*family|loaded reference identity'):
            collect.load_actor(p)
        assert builds == []
    assert loads == [True]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault', ['none', 'tag'])
def test_real_data_auditor_consumes_new_serving_lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str):
    from scripts import core_finite_admit as audit
    from scripts import core_generation_data_admit as data
    from tests.test_core_finite_admit import bind_data_case

    f = audit_case(tmp_path, monkeypatch)
    p, case, root = f.producer, f.case, f.producer_root
    project(p)
    pref = frozen(tmp_path / 'anchored-producer.json', p)
    case.protocol['producer_protocol'] = pref
    serving = f.serving
    serving.update(actor_kind=collect.ANCHOR_BANK_KIND, actor_lineage=p['actor']['lineage'],
        policy_anchor_bank_tag=collect.bank_lineage_identity(p['actor']), predecessor_boundary=p['actor']['predecessor']['boundary'])
    del serving['bank_learner_tag']
    if fault == 'tag':
        serving['policy_anchor_bank_tag'] = {'wrong': 'identity'}
    _, readout = audit.learning.read_ref(case.protocol['producer_readout'])
    root.chmod(0o755)
    readout.update(protocol_sha256=pref['sha256'], actor_serving=replace(audit.ref(root / 'actor-serving.json'), serving))
    case.protocol['producer_readout'] = replace(case.protocol['producer_readout'], readout)
    ad = frozen(tmp_path / 'anchored-producer-admission.json', {'status': 'ROOT_ADMITS_CORE_GENERATION_V1',
        'protocol_sha256': pref['sha256'], 'run_uuid': p['run_uuid'], 'generation': p['generation'],
        'source_revision': p['source']['revision'], 'actor_sha256': p['actor']['sha256']})
    command = ['python', '-m', 'scripts.core_finite_collect', '--execute', '--protocol', pref['path'], '--protocol-sha256', pref['sha256'],
        '--admission', ad['path'], '--admission-sha256', ad['sha256']]
    _, terminal = audit.learning.read_ref(case.protocol['producer_owner']['terminal'])
    terminal['command'] = command
    case.protocol['producer_owner'].update(terminal=frozen(tmp_path / 'anchored-producer-terminal.json', terminal), command=command)
    case.protocol['producer_inventory'] = frozen(tmp_path / 'anchored-inventory.json', {'path': str(root), 'files': collect.inventory(root, seal=True)})
    protocol_ref = bind_data_case(tmp_path, case)
    if fault == 'tag':
        with pytest.raises(ValueError, match='bank actor serving lineage'):
            data.execute(protocol_ref)
    else:
        result = data.execute(protocol_ref)
        assert result['logical_generation'] == 3
        assert result['actor_checkpoint_step'] == 2
        assert result['learning_admitted'] is False
        assert result['candidate_only'] is True
