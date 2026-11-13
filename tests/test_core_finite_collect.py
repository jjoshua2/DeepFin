"""Collector boundary controls; no actual actor, engine, tablebase or CUDA job."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import core_finite_collect as collect
from chess_anti_engine.verified_outcome import VerifiedOutcome, require_verified

RUN = '11111111-1111-4111-8111-111111111111'
COLLECTION = '22222222-2222-4222-8222-222222222222'


def protocol(tmp_path: Path) -> dict[str, Any]:
    return {
        'schema': 'CORE_FINITE_GENERATION_V1', 'run_uuid': RUN, 'collection_uuid': COLLECTION,
        'generation': 1, 'attempts': 2, 'concurrent_slots': 2, 'seed': 7,
        'actor': {'kind': 'donor', 'path': str(tmp_path / 'actor.pt'),
                  'sha256': collect.INITIAL_DONOR_SHA256, 'expected_checkpoint_step': 680276,
                  'predecessor': None, 'architecture': {'_schema_version': 20, 'input_extra_features': 'v2_threats'}},
        'source': {'checkout': str(tmp_path / 'source'), 'revision': 'a' * 40}, 'output_root': str(tmp_path / 'output'),
        'runtime': {'device': 'cuda:0', 'cpu_affinity': [20, 22], 'threads': 2, 'inner_seconds': 100},
        'owner': {'seconds': 120},
        'serving': {'device': 'cuda:0', 'use_amp': True, 'amp_dtype': 'bfloat16'},
        'stockfish': {'workers': 2, 'nodes': 5000, 'multipv': 1, 'hash_mb': 16},
    }


def roster() -> list[dict[str, Any]]:
    return [{'start_ordinal': i, 'bank': 'train', 'physical_game_id': COLLECTION + ':' + str(i),
             'root_fen': 'root', 'opening_moves': ['e2e4'] if i == 0 else ['d2d4']} for i in range(2)]


def test_descriptor_uses_declared_counts_and_keeps_historical_donor_step(tmp_path: Path) -> None:
    p = protocol(tmp_path)
    collect.validate_protocol(p, roster())
    assert p['actor']['expected_checkpoint_step'] == 680276
    p['attempts'] = p['concurrent_slots'] = 1
    collect.validate_protocol(p, roster()[:1])


@pytest.mark.parametrize('fault', ['reordered', 'holdout', 'duplicate_history', 'wrong_physical', 'wrong_donor'])
def test_fixed_roster_refuses_replacements_and_foreign_actor(tmp_path: Path, fault: str) -> None:
    p, games = protocol(tmp_path), roster()
    if fault == 'reordered':
        games.reverse()
    elif fault == 'holdout':
        games[0]['bank'] = 'holdout'
    elif fault == 'duplicate_history':
        games[1]['opening_moves'] = games[0]['opening_moves']
    elif fault == 'wrong_physical':
        games[0]['physical_game_id'] = RUN + ':0'
    else:
        p['actor']['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        collect.validate_protocol(p, games)


def predecessor(tmp_path: Path, p: dict[str, Any]) -> dict[str, Any]:
    def write(name: str, value: Any) -> dict[str, str]:
        path = tmp_path / name
        collect.write_json(path, value)
        return {'path': str(path), 'sha256': collect.digest(path)}

    contract = write('contract.json', {'run_uuid': RUN, 'source_revision': p['source']['revision']})
    stage = write('stage.json', {'generation': 1, 'contract_sha256': contract['sha256'],
                  'generations': [], 'parent': None, 'output': str(tmp_path)})
    p['actor']['path'] = str(tmp_path / 'learner.pt')
    boundary = write('boundary.json', {'status': 'COMPLETED_CORE_GENERATION_V1', 'run_uuid': RUN,
        'source_revision': p['source']['revision'], 'generation': 1, 'contract_sha256': contract['sha256'],
        'stage': stage, 'parent': None, 'generation_manifests': [], 'final_step': 512,
        'final_checkpoint': {'path': p['actor']['path'], 'sha256': 'f' * 64}})
    command = ['python', '-m', 'scripts.core_finite_learner', '--contract', contract['path'],
               '--contract-sha256', contract['sha256'], '--stage', stage['path'], '--stage-sha256', stage['sha256']]
    terminal = write('terminal.json', {'status': 'PASS', 'returncode': 0, 'owned_processes_closed': True,
        'custody': 'OWNED_PROCESSES_CLOSED', 'supervisor_sha256': 'b' * 64, 'command': command})
    return {'boundary': boundary, 'owner_terminal': terminal, 'owner_sha256': 'b' * 64, 'command': command}


def test_later_actor_must_be_previous_closed_final_not_selected_checkpoint(tmp_path: Path) -> None:
    p = protocol(tmp_path)
    binding = predecessor(tmp_path, p)
    p['generation'] = 2
    p['actor'].update(kind='learner', sha256='f' * 64, expected_checkpoint_step=512, predecessor=binding)
    collect.validate_protocol(p, roster())
    p['actor']['expected_checkpoint_step'] = 8
    with pytest.raises(ValueError, match='previous-generation'):
        collect.validate_protocol(p, roster())


def test_predecessor_child_pass_does_not_substitute_for_owned_closure(tmp_path: Path) -> None:
    p = protocol(tmp_path)
    binding = predecessor(tmp_path, p)
    binding['owner_sha256'] = 'c' * 64
    with pytest.raises(ValueError, match='not closed PASS'):
        collect.closed_predecessor(binding)


def test_predecessor_failure_marker_refuses_published_child_boundary(tmp_path: Path) -> None:
    p = protocol(tmp_path)
    binding = predecessor(tmp_path, p)
    tmp_path.with_name(tmp_path.name + '.failed.json').write_text('{}')
    with pytest.raises(ValueError, match='predecessor stage/contract/history'):
        collect.closed_predecessor(binding)


def test_exclusion_refs_fail_before_actor_work(tmp_path: Path) -> None:
    prior = tmp_path / 'prior.json'
    playing = tmp_path / 'playing.json'
    collect.write_json(prior, {'history_sha256': [], 'stored_input_float16_sha256': [],
                              'coverage_scope': 'explicit synthetic empty control', 'donor_pretraining_exposure': 'UNKNOWN'})
    collect.write_json(playing, [{'id': 'held-out-start', 'input_sha256': '0' * 64, 'mirror_sha256': '1' * 64}])
    p = {'exclusions': {'prior': {'path': str(prior), 'sha256': collect.digest(prior)},
         'playing_starts': {'path': str(playing), 'sha256': collect.digest(playing)},
         'playing_start_count': 1, 'prior_overlap_action': 'report', 'playing_overlap_action': 'refuse'}}
    assert collect.exclusion_inputs(p)[1][0]['id'] == 'held-out-start'
    p['exclusions']['playing_starts']['sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='bound file changed'):
        collect.exclusion_inputs(p)


def test_inventory_hashes_exact_files_and_seals_without_renaming(tmp_path: Path) -> None:
    shard = tmp_path / 'shard'
    (shard / 'x').mkdir(parents=True)
    (shard / 'x' / '0').write_bytes(b'rowbytes')
    (shard / '.zgroup').write_text('{}')
    files = collect.inventory(shard, seal=True)
    assert [x['path'] for x in files] == ['.zgroup', 'x/0']
    assert all((shard / x['path']).stat().st_mode & 0o222 == 0 for x in files)
    assert files[1]['sha256'] == collect.digest(shard / 'x' / '0')
    file = tmp_path / 'single.npz'
    file.write_bytes(b'npz')
    assert collect.inventory(file)[0]['path'] == '.'


def test_inventory_refuses_symlinked_payload(tmp_path: Path) -> None:
    (tmp_path / 'original').write_bytes(b'payload')
    shard = tmp_path / 'shard'
    shard.mkdir()
    (shard / 'x').symlink_to(tmp_path / 'original')
    with pytest.raises(ValueError, match='symlink'):
        collect.inventory(shard)


def test_proof_cannot_be_replaced_by_serialized_boolean() -> None:
    with pytest.raises(ValueError, match='actual verified-outcome'):
        require_verified({'resolved': True})
    with pytest.raises(ValueError, match='actual verified-outcome'):
        require_verified(VerifiedOutcome({'resolved': True}, '0' * 64, object()))


def test_actual_actor_loader_observer_links_served_tensor_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real collector observer around inert public-loader chain; no CUDA allocation."""
    import torch
    from chess_anti_engine.uci import model_loader

    p = protocol(tmp_path)
    actor = Path(p['actor']['path'])
    actor.write_bytes(b'bounded-test-actor-container')
    actor.chmod(0o444)
    p['actor']['sha256'] = collect.digest(actor)
    state = {'tensor.' + str(i): torch.tensor([float(i)]) for i in range(494)}
    p['actor']['architecture']['phase_piece_thresholds'] = [12, 20]
    checkpoint_arch = deepcopy(p['actor']['architecture'])
    checkpoint_arch['phase_piece_thresholds'] = (12, 20)
    payload = {'model': state, 'arch': checkpoint_arch, 'step': 680276}

    class InertActor:
        def state_dict(self) -> dict[str, Any]:
            return state

        def parameters(self) -> list[Any]:
            return [SimpleNamespace(device='cuda:0')]

    monkeypatch.setattr(torch, 'load', lambda *_a, **_k: payload)
    monkeypatch.setattr(model_loader, 'build_model', lambda _cfg: InertActor())

    def inert_loader(path: Path, **_kwargs: Any) -> Any:
        torch.load(path, weights_only=True, map_location='cuda:0')
        return model_loader.build_model(p['actor']['architecture'])

    monkeypatch.setattr(model_loader, 'load_model_from_checkpoint', inert_loader)
    _, serving = collect.load_actor(p)
    assert serving['checkpoint_step'] == 680276
    assert serving['model_build_count'] == 1
    assert serving['pre_state_sha256'] == serving['checkpoint_model_state_sha256']
    assert serving['actor_sha256'] != serving['pre_state_sha256']
    p['actor']['expected_checkpoint_step'] = 0
    with pytest.raises(ValueError, match='historical checkpoint step'):
        collect.load_actor(p)


def test_served_state_digest_includes_values_names_and_shapes() -> None:
    import torch

    state = {'weight': torch.tensor([1.0, 2.0])}
    assert collect.state_sha256(state) == collect.state_sha256({'_orig_mod.weight': state['weight']})
    changed = deepcopy(state)
    changed['weight'][1] = 3
    assert collect.state_sha256(state) != collect.state_sha256(changed)
    with pytest.raises(ValueError, match='ambiguous'):
        collect.state_sha256({**state, '_orig_mod.weight': state['weight']})


def test_hashed_json_refuses_ambiguous_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / 'duplicate.json'
    path.write_text('{"generation":1,"generation":2}')
    with pytest.raises(ValueError, match='duplicate JSON identity key'):
        collect.checked({'path': str(path), 'sha256': collect.digest(path)})


@pytest.mark.parametrize('inherited', [None, '1', '0'])
def test_source_guard_forces_nonlocking_git_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                inherited: str | None) -> None:
    repo = tmp_path / 'source'
    monkeypatch.setattr(collect, '__file__', str(repo / 'scripts/core_finite_collect.py'))
    if inherited is None:
        monkeypatch.delenv('GIT_OPTIONAL_LOCKS', raising=False)
    else:
        monkeypatch.setenv('GIT_OPTIONAL_LOCKS', inherited)
    monkeypatch.setenv('CORE_COLLECTOR_TEST_CONTEXT', 'preserved')
    calls = []

    def git(command: list[str], *, cwd: Path, env: dict[str, str]) -> bytes:
        assert cwd == repo
        assert env['GIT_OPTIONAL_LOCKS'] == '0'
        assert env['CORE_COLLECTOR_TEST_CONTEXT'] == 'preserved'
        calls.append(command)
        return b'abc\n' if command[1] == 'rev-parse' else b''

    monkeypatch.setattr(collect.subprocess, 'check_output', git)
    with pytest.raises(ValueError, match='incomplete source fingerprint'):
        collect.source_guard({'source': {'checkout': str(repo), 'revision': 'abc', 'sha256': {}}})
    assert calls == [['git', 'rev-parse', 'HEAD'], ['git', 'status', '--porcelain']]
    assert collect.os.environ.get('GIT_OPTIONAL_LOCKS') == inherited


def test_late_final_seal_fault_banks_immutable_external_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / 'generation'
    out.mkdir()
    original_inventory = collect.inventory

    def fail_after_seal(path: Path, *, seal: bool = False) -> list[dict[str, Any]]:
        original_inventory(path, seal=seal)
        assert path.stat().st_mode & 0o222 == 0
        raise OSError('injected late seal/fsync failure')

    monkeypatch.setattr(collect, 'inventory', fail_after_seal)
    with pytest.raises(OSError, match='injected late seal'):
        collect.seal_readout(out, {'status': 'COLLECTED_CORE_GENERATION_V1', 'run_uuid': RUN, 'generation': 1})
    marker = collect.failure_marker(out)
    assert marker.parent == out.parent
    assert marker.parent != out
    assert marker.stat().st_mode & 0o222 == 0
    failure = collect.checked({'path': str(marker), 'sha256': collect.digest(marker)})
    assert failure['phase'] == 'final_output_seal'
    assert failure['status'] == 'FAILED_CORE_GENERATION_V1'
    assert failure['learning_admitted'] is failure['root_acceptance_issued'] is False
    # A child readout can exist, but the external failure remains authoritative.
    assert (out / 'readout.json').exists()
    previous = marker.read_bytes()
    collect.mark_failure(out, {'later_error': 'must not overwrite first failure'})
    assert marker.read_bytes() == previous
