"""Synthetic serialized authority fixtures, never a new legal/TB proof issuer."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pytest
import zarr
from scripts import retained_search_target_census as audit
FICTION = 'SYNTHETIC_NOT_REAL_AUTHORITY'

def ref(path: Path, value: Any) -> dict[str, str]:
    path.write_bytes(audit.canonical(value))
    path.chmod(292)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def inventory(root: Path) -> list[dict[str, Any]]:
    return [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file()]

def seal(root: Path) -> None:
    for p in sorted(root.rglob('*'), reverse=True):
        p.chmod(365 if p.is_dir() else 292)
    root.chmod(365)

def fixture(tmp_path: Path, *, fault: str='') -> dict[str, Any]:
    """Two physical games,3+1rows; real storage/schema, explicitly fictional proof docs."""
    bank = tmp_path / 'bank'
    bank.mkdir()
    bindings = tmp_path / 'bindings'
    bindings.mkdir()
    (run, collection, source, actor, served) = ('run-fiction', 'collection-fiction', '1' * 40, '2' * 64, '3' * 64)
    limits = {'max_metadata_bytes': 128 << 20, 'max_inventory_files': 128000, 'max_inventory_bytes': 32 << 30, 'max_raw_decoded_bytes_per_game': 64 << 20, 'max_shard_decoded_bytes_per_game': 64 << 20, 'max_total_decoded_bytes': 32 << 30}
    config = {'game': {'policy_encoding': 'lc0_1858', 'history_rep_fix': True, 'policy_row_retention': 'all_eligible', 'search_wdl_draw_mode': 'net_raw', 'input_history_encoding': 'v2_threats', 'categorical_bins': 2}}
    ref(bank / 'resolved-profile.json', config)
    serving = ref(bank / 'serving.json', {'actor_sha256': actor, 'pre_state_sha256': served, 'post_state_sha256': served, 'checkpoint_model_state_sha256': served})
    producer = {'run_uuid': run, 'collection_uuid': collection, 'source': {'revision': source}, 'generation': 1, 'actor': {'sha256': actor}, 'attempts': 2, 'output_root': str(bank), 'resolved_profile_sha256': audit.digest(config), 'owner': {'sha256': '4' * 64, 'seconds': 3600}, 'fixture': FICTION}
    producer_ref = ref(bindings / 'producer.json', producer)

    def terminal(command: list[str], resource: str, seconds: int) -> dict[str, Any]:
        return {'status': 'PASS', 'returncode': 0, 'owned_processes_closed': True, 'custody': 'OWNED_PROCESSES_CLOSED', 'command': command, 'supervisor_sha256': '4' * 64, 'resource': resource, 'seconds': seconds, 'known_processes': {'99999999': 1}, 'fixture': FICTION}
    producer_command = ['fake', '--protocol', producer_ref['path'], '--protocol-sha256', producer_ref['sha256']]
    producer_terminal = ref(bindings / 'producer-terminal.json', terminal(producer_command, 'gpu', 3600))
    completed = {}
    staged_games = []
    for (g, (plies, target, result, end)) in enumerate([([0, 2, 4], [[0.5, 0.25, 0.25], [0.1, 0.2, 0.7], [1, 0, 0]], '1-0', 'NATURAL'), ([1], [[0, 1, 0]], '1/2-1/2', 'STRICT_TB_ADJUDICATED')]):
        n = len(plies)
        physical = collection + ':' + str(g)
        game_id = 100 + g
        directory = bank / ('game-' + str(g))
        directory.mkdir()
        shard = directory / 'shard'
        rows = [{'physical_row_id': physical + ':' + str(i), 'ply': ply, 'move_offset': ply, 'history_sha256': '5' * 64, 'stored_input_float16_sha256': '6' * 64, 'stored_input_float16_mirror_sha256': '7' * 64} for (i, ply) in enumerate(plies)]
        records = [{'move_offset': ply, 'ply_index': ply, 'pov_color': ply % 2 == 0, 'has_policy': True, 'is_sf_refute_opp': False} for ply in plies]
        actual: dict[str, Any] = {'game_id': game_id, 'retained_samples': n, 'records': records, 'result': result, 'natural_result': result if g == 0 else '*', 'resolution': 'natural' if g == 0 else 'tablebase_adjudication', 'total_plies': 6 if g == 0 else 2, 'played_move_indices': [0] * (6 if g == 0 else 2), 'start': {'root_fen': 'SYNTHETIC_FEN', 'opening_moves': [], 'start_ply': 0}}
        if fault == 'unknown_distance' and g == 0:
            del actual['total_plies']
            del actual['played_move_indices']
        if fault == 'bad_distance' and g == 0:
            actual['total_plies'] += 1
        if fault == 'bad_record_offset' and g == 0:
            records[1]['move_offset'] = 1
        common: dict[str, Any] = {'run_uuid': run, 'collection_uuid': collection, 'generation': 1, 'source_revision': source, 'actor_sha256': actor, 'served_state_sha256': served, 'physical_game_id': physical, 'game_id': game_id, 'rows': n, 'bank': 'train', 'scientific_class': end}
        proof = {'schema': 'CORE_ACTUALLY_VERIFIED_OUTCOME_V1', **common, 'resolved': True, 'result': result, 'retained_rows': n, 'audit_sha256': audit.digest(actual), 'fixture': FICTION}
        policy = np.zeros((n, 1858), dtype=np.float32)
        policy[:, 0] = 1
        raw: dict[str, Any] = {'physical_row_id': np.array([r['physical_row_id'] for r in rows]), 'policy_target_raw': policy, 'outcome_wdl': np.array([0] * n if g == 0 else [1], dtype=np.int8), 'pov_white': np.array([ply % 2 == 0 for ply in plies], dtype=np.uint8), 'search_wdl_raw': np.array(target, dtype=np.float32), 'has_search_raw': np.ones(n, dtype=np.uint8), 'categorical_raw': np.zeros((n, 2), dtype=np.float32), 'has_categorical_raw': np.zeros(n, dtype=np.uint8)}
        if g == 0:
            if fault == 'wrong_row':
                raw['physical_row_id'][1] = physical + ':0'
            if fault == 'bad_pov':
                raw['pov_white'][0] = 0
            if fault == 'wrong_outcome':
                raw['outcome_wdl'][0] = 2
            if fault == 'missing_mask':
                raw['has_search_raw'][0] = 0
            if fault == 'invalid_mask':
                raw['has_search_raw'][0] = 2
            if fault == 'nan':
                raw['search_wdl_raw'][0, 0] = np.nan
            if fault == 'negative':
                raw['search_wdl_raw'][0] = [-0.1, 0.5, 0.6]
            if fault == 'bad_mass':
                raw['search_wdl_raw'][0] = [0.2, 0.2, 0.2]
            if fault == 'unsafe_unused_header':
                raw['policy_target_raw'] = np.array([[object()]], dtype=object)
        np.savez_compressed(directory / 'raw-targets.npz', **raw)
        arrays: dict[str, Any] = {'search_wdl': raw['search_wdl_raw'].astype(np.float16), 'has_search_wdl': raw['has_search_raw'], 'wdl_target': raw['outcome_wdl'].copy(), 'game_id': np.full(n, game_id, dtype=np.int64), 'ply_index': np.array(plies, dtype=np.int32)}
        for key in ('has_game_id', 'has_ply_index', 'has_policy', 'is_network_turn', 'has_is_network_turn', 'is_selfplay', 'has_is_selfplay', 'has_legal_mask'):
            arrays[key] = np.ones(n, dtype=np.uint8)
        if fault == 'quantization' and g == 0:
            arrays['search_wdl'][0, 0] += np.float16(0.01)
        if fault == 'wrong_stored_game' and g == 0:
            arrays['game_id'][0] = 101
        if fault == 'wrong_stored_ply' and g == 0:
            arrays['ply_index'][0] = 1
        if fault == 'wrong_stored_mask' and g == 0:
            arrays['has_search_wdl'] = np.zeros(n, dtype=np.uint8)
        group = zarr.open_group(str(shard), mode='w')
        group.attrs.update({'run_id': collection, 'model_sha256': actor, 'positions': n, 'games': 1, 'history_rep_fix': True, 'input_history_encoding': 'v2_threats', 'policy_encoding': 'lc0_1858'})
        for (key, value) in arrays.items():
            group.create_dataset(key, data=value, chunks=value.shape)
        shard_ref = {'path': str(shard), 'files': inventory(shard)}
        validation = {'schema': 'VERIFIED_CORE_GAME_METADATA_V1', **common, 'full_history_verified': True, 'resolved': True, 'raw_targets_verified': True, 'usable_shard': True, 'rows_detail': rows, 'all_eligible_policy_rows_retained': True, 'policy_row_retention': 'all_eligible', 'verified_outcome_sha256': audit.digest(proof), 'shard_inventory_sha256': audit.digest(shard_ref['files'])}
        provenance = {'schema': 'CORE_GAME_PROVENANCE_V1', **common, 'actual': actual, 'planned': {'physical_game_id': physical, 'root_fen': 'SYNTHETIC_FEN', 'opening_moves': []}}
        if fault == 'wrong_source' and g == 0:
            provenance['source_revision'] = '8' * 40
        docs = {'provenance': ('provenance.json', provenance), 'validation': ('validation.json', validation), 'outcome_proof': ('outcome-proof.json', {'schema': 'CORE_PRODUCING_OUTCOME_PROOF_V1', **common, 'verification': proof, 'proof_sha256': audit.digest(proof)}), 'raw_metadata': ('raw-targets.json', {'schema': 'CORE_RAW_TARGETS_V1', **common, 'raw_targets_verified': True, 'verified_outcome': proof, 'verified_outcome_sha256': audit.digest(proof)})}
        evidence = {key: ref(directory / name, value) for (key, (name, value)) in docs.items()}
        item = {**common, 'path': str(directory), 'inventory': inventory(directory), 'shard': shard_ref, 'validation': copy.deepcopy(validation), 'evidence': evidence}
        completed[str(g)] = item
        staged_games.append((common, shard_ref, evidence, validation))
    readout = {'status': 'COLLECTED_CORE_GENERATION_V1', 'protocol_sha256': producer_ref['sha256'], 'run_uuid': run, 'collection_uuid': collection, 'generation': 1, 'attempts': 2, 'source_revision': source, 'actor_sha256': actor, 'all_planned_attempts_accounted': True, 'training_steps': 0, 'root_acceptance_issued': False, 'learning_admitted': False, 'completed': completed, 'actor_serving': serving}
    readout_ref = ref(bank / 'readout.json', readout)
    full = ref(bindings / 'producer-inventory.json', {'path': str(bank), 'files': inventory(bank)})
    audit_protocol = {'limits': limits, 'producer_protocol': producer_ref, 'producer_readout': readout_ref, 'producer_inventory': full, 'runtime': {'owner': {'sha256': '4' * 64}, 'seconds': 1800}, 'output_root': str(bindings), 'producer_owner': {'terminal': producer_terminal, 'command': producer_command, 'owner_sha256': '4' * 64}}
    audit_ref = ref(bindings / 'audit.json', audit_protocol)
    games = []
    for (g, (common, shard_ref, evidence, validation)) in enumerate(staged_games):
        label = 'NATURAL' if g == 0 else 'TB_ADJUDICATED'
        wrapper = {'status': 'VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1', **{k: v for (k, v) in common.items() if k not in ('collection_uuid', 'scientific_class')}, 'label_class': label, 'resolved': True, 'full_history_verified': True, 'raw_targets_verified': True, 'audit_protocol': audit_ref, 'evidence': evidence, 'shard_inventory_sha256': validation['shard_inventory_sha256']}
        wrapper_ref = ref(bindings / ('game-' + str(g) + '.json'), wrapper)
        games.append({'physical_game_id': common['physical_game_id'], 'game_id': common['game_id'], 'bank': 'train', 'rows': common['rows'], 'label_class': label, 'shard': shard_ref, 'producer_proof': wrapper_ref})
    if fault == 'wrong_role':
        games[0]['bank'] = 'hold'
    if fault == 'duplicate_game':
        games[1]['physical_game_id'] = games[0]['physical_game_id']
    generation = {'schema': 'ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1', 'run_uuid': run, 'source_revision': source, 'generation': 1, 'actor_sha256': actor, 'actor_serving': serving, 'games': games, 'audit_protocol': audit_ref}
    gen_ref = ref(bindings / 'generation.json', generation)
    report = ref(bindings / 'readout.json', {'status': 'AUDITED_CORE_TRAIN_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1', 'root_postclosure_promotion_required': True, 'audit_protocol': audit_ref, 'generation_manifest': gen_ref, 'eligible_games': 2, 'eligible_rows': 4})
    command = ['fake', '-m', 'scripts.core_finite_admit', '--protocol', audit_ref['path'], '--protocol-sha256', audit_ref['sha256']]
    cpu_terminal = terminal(command, 'cpu', 1800)
    if fault == 'failed_cpu':
        cpu_terminal['status'] = 'FAILED'
    term_ref = ref(bindings / 'audit-terminal.json', cpu_terminal)
    promotion = ref(bindings / 'promotion.json', {'status': 'ROOT_PROMOTES_CLOSED_AUDIT_GENERATION_FOR_RESEARCH_LEARNER_V1', 'generation': gen_ref, 'attempts': 2, 'eligible_games': 2, 'eligible_rows': 4, 'audit_protocol': audit_ref, 'audit_closed_owner': term_ref, 'audit_readout': report, 'exact_command': command})
    authority = ref(bindings / 'authority.json', {'status': 'ROOT_AUTHENTICATES_EXISTING_FIXED_TRAIN_BANK_FOR_RETAINED_SEARCH_V1', 'generation_manifest': gen_ref, 'new_data_acceptance_issued': False, 'data_identity': {'run_uuid': run, 'source_revision': source}, 'original_acceptance_refs': [] if fault == 'foreign_authority' else [promotion], 'fixture': FICTION})
    selection = ref(bindings / 'selection.json', {'status': 'ROOT_SELECTS_PROSPECTIVE_MODEL_FREE_ALL_BANK_WDL_TARGET_CENSUS_V1', 'study_uuid': 'study-fiction', 'data_authority': authority, 'data_scope': {'generation_manifest_sha256': gen_ref['sha256'], 'physical_TRAIN_games': 2, 'rows': 4, 'expected_search_target_available_rows': 4, 'recorded_game_end_classes': {'NATURAL': 1, 'STRICT_TB_ADJUDICATED': 1}}, 'fixture': FICTION})
    clarification = ref(bindings / 'clarification.json', {'status': 'ROOT_CLARIFIES_PRECOMPUTE_TARGET_CENSUS_STATISTIC_IMPLEMENTATION_V1', 'selection': selection, 'quantiles': {'levels': list(audit.QUANTILES), 'method': 'NumPy quantile method=linear'}, 'fixture': FICTION})
    seal(bank)
    seal(bindings)
    return {'study_uuid': 'study-fiction', 'selection': selection, 'clarification': clarification, 'data_authority': authority, 'promotion': promotion, 'generation_manifest': gen_ref, 'output': str(tmp_path / 'report'), 'fixture': FICTION}

def test_all_bank_known_metrics_and_denominators(tmp_path: Path) -> None:
    p = fixture(tmp_path)
    result = audit.census(p)
    cell = result['statistics']['overall']['archived_f32']
    assert result['physical_games'] == 2
    assert result['rows'] == result['available_rows'] == 4
    assert cell['physical_games'] == 2
    assert cell['row_means']['full_vector_squared_residual'] == pytest.approx(0.42875, abs=1e-07)
    assert cell['row_means']['Q_squared_residual'] == pytest.approx(0.780625, abs=1e-07)
    assert cell['row_means']['draw_squared_residual'] == pytest.approx(0.025625, abs=1e-07)
    assert cell['equal_game_means']['Q_residual'] == pytest.approx(-0.3916666667, abs=1e-07)
    assert cell['row_means']['Q_residual'] == pytest.approx(-0.5875, abs=1e-07)
    assert cell['sums']['zero_W_or_L'] == 2
    assert cell['sums']['numerical_boundary_mass'] == 2
    assert cell['quantiles']['Q'][4] == 0.125
    empty = result['statistics']['outcome:L']['archived_f32']
    assert empty['rows'] == empty['physical_games'] == 0
    assert empty['row_means'] is None
    assert empty['equal_game_means'] is None
    per_game = json.loads((Path(p['output']) / 'per-game.json').read_text())
    assert [v['rows'] for v in per_game] == [3, 1]
    assert per_game[0]['distance_counts'] == {'0-4': 2, '5-16': 1}

def test_f16_quantization_and_f32_normalization_are_distinct(tmp_path: Path) -> None:
    result = audit.census(fixture(tmp_path))
    cells = result['statistics']['overall']
    assert cells['stored_f16_widened']['mass_max'] > 1
    assert cells['archived_f32']['row_means']['full_vector_squared_residual'] != cells['stored_f16_widened']['row_means']['full_vector_squared_residual']
    assert cells['stored_f16_widened']['row_means']['full_vector_squared_residual'] != cells['normalized_f16_source_f32']['row_means']['full_vector_squared_residual']
    assert result['stored_f16_metrics'].startswith('algebraic')

def test_missing_distance_is_explicit_unknown_without_row_loss(tmp_path: Path) -> None:
    result = audit.census(fixture(tmp_path, fault='unknown_distance'))
    cell = result['statistics']['distance:UNKNOWN']['archived_f32']
    assert cell['rows'] == 3
    assert cell['physical_games'] == 1
    assert result['rows'] == 4

@pytest.mark.parametrize(('fault', 'message'), [('wrong_role', 'complete TRAIN population'), ('duplicate_game', 'complete TRAIN population'), ('failed_cpu', 'exact closed input owner'), ('foreign_authority', 'original root bank authority'), ('wrong_source', 'per-game source/role/common identity'), ('wrong_row', 'stored/raw physical row identity'), ('bad_pov', 'raw POV/physical ply'), ('wrong_outcome', 'outcome result/POV mismatch'), ('quantization', 'exact F32-to-F16'), ('missing_mask', 'missing/invalid search availability'), ('invalid_mask', 'missing/invalid search availability'), ('wrong_stored_mask', 'missing/invalid search availability'), ('nan', 'invalid original F32'), ('negative', 'invalid original F32'), ('bad_mass', 'invalid original F32'), ('wrong_stored_game', 'stored/raw physical row identity'), ('wrong_stored_ply', 'stored/raw physical row identity'), ('bad_distance', 'inconsistent exact stop distance'), ('bad_record_offset', 'raw POV/physical ply'), ('unsafe_unused_header', 'raw field dtype/shape|unsafe raw array header')])
def test_malformed_bank_is_unqualified_without_clean_tables(tmp_path: Path, fault: str, message: str) -> None:
    p = fixture(tmp_path, fault=fault)
    with pytest.raises(ValueError, match=message):
        audit.census(p)
    assert not (Path(p['output']) / 'readout.json').exists()
    assert not (Path(p['output']) / 'per-game.json').exists()

def test_wrong_metadata_hash_refuses_before_numeric_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = fixture(tmp_path)
    p['selection']['sha256'] = '0' * 64

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError('numeric payload decode')
    monkeypatch.setattr(audit, 'raw_targets', forbidden)
    with pytest.raises(ValueError, match='input SHA/custody'):
        audit.census(p)

def test_added_inventory_file_refuses_before_numeric_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = fixture(tmp_path)
    bank = tmp_path / 'bank'
    bank.chmod(493)
    (bank / 'foreign').write_text('foreign')
    bank.chmod(365)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError('numeric payload decode')
    monkeypatch.setattr(audit, 'raw_targets', forbidden)
    with pytest.raises(ValueError, match='added/omitted inventory'):
        audit.census(p)

def test_metadata_cap_is_checked_before_any_hash_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'oversized'
    path.write_bytes(b'large')
    path.chmod(292)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError('opened oversized input')
    monkeypatch.setattr(audit.os, 'open', forbidden)
    with pytest.raises(ValueError, match='input byte cap before'):
        audit.Inputs().json({'path': str(path), 'sha256': '0' * 64}, cap=1)

def test_conflicting_cached_hash_refuses(tmp_path: Path) -> None:
    path = tmp_path / 'metadata'
    r = ref(path, {'fixture': FICTION})
    inputs = audit.Inputs()
    inputs.json(r)
    with pytest.raises(ValueError, match='conflicting input reference'):
        inputs.json({**r, 'sha256': '0' * 64})

def test_late_custody_failure_has_no_clean_tables(tmp_path: Path) -> None:
    p = fixture(tmp_path)

    def refuse() -> None:
        raise ValueError('late guard')
    with pytest.raises(ValueError, match='late guard'):
        audit.census(p, final_guard=refuse)
    marker = Path(p['output']).with_name('report.failed.json')
    assert json.loads(marker.read_text())['status'] == 'UNQUALIFIED_WDL_TARGET_CENSUS'
    assert not (Path(p['output']) / 'readout.json').exists()

def test_no_policy_or_categorical_payload_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = fixture(tmp_path)
    with np.load(tmp_path / 'bank/game-0/raw-targets.npz', allow_pickle=False) as archive:
        cls = type(archive)
    original = cls.__getitem__

    def selected(self: Any, key: str) -> Any:
        assert key not in ('policy_target_raw', 'categorical_raw', 'has_categorical_raw')
        return original(self, key)
    monkeypatch.setattr(cls, '__getitem__', selected)
    assert audit.census(p)['rows'] == 4

def test_output_inside_another_checkout_refuses(tmp_path: Path) -> None:
    other = tmp_path / 'other'
    other.mkdir()
    (other / '.git').write_text('gitdir: fictional')
    with pytest.raises(ValueError, match='fresh external disjoint output'):
        audit.fresh_output(other / 'report', [])

def test_crossed_end_distance_strata_equal_game_denominator(tmp_path: Path) -> None:
    result = audit.census(fixture(tmp_path))
    cell = result['statistics']['end_distance:NATURAL:0-4']['archived_f32']
    assert cell['rows'] == 2
    assert cell['physical_games'] == 1
    assert cell['equal_game_means'] == cell['row_means']

def test_zarr_unsafe_codec_refuses_before_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / 'unsafe'
    field = root / 'search_wdl'
    field.mkdir(parents=True)
    (field / '.zarray').write_text(json.dumps({'dtype': '|O', 'shape': [1, 3], 'chunks': [1, 3], 'filters': None, 'compressor': None}))

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError('opened unsafe Zarr')
    monkeypatch.setattr(zarr, 'open_group', forbidden)
    with pytest.raises(ValueError, match='unsafe Zarr dtype/filter'):
        audit.stored_arrays({'path': str(root)}, 1, 64 << 20)

def test_padded_zarr_allocation_cap_precedes_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / 'oversized'
    field = root / 'search_wdl'
    field.mkdir(parents=True)
    (field / '.zarray').write_text(json.dumps({'dtype': '<f2', 'shape': [1, 3], 'chunks': [1000000, 3], 'filters': None, 'compressor': None}))

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError('opened oversized Zarr')
    monkeypatch.setattr(zarr, 'open_group', forbidden)
    with pytest.raises(ValueError, match='Zarr padded decoded bound'):
        audit.stored_arrays({'path': str(root)}, 1, 1000)

def test_changed_authenticated_input_refuses_final_custody(tmp_path: Path) -> None:
    path = tmp_path / 'input'
    r = ref(path, {'fixture': FICTION})
    inputs = audit.Inputs()
    inputs.json(r)
    path.chmod(420)
    path.write_text('changed')
    path.chmod(292)
    with pytest.raises(ValueError, match='final input custody'):
        inputs.final()

def test_writable_input_refuses(tmp_path: Path) -> None:
    path = tmp_path / 'input'
    r = ref(path, {'fixture': FICTION})
    path.chmod(420)
    with pytest.raises(ValueError, match='writable input'):
        audit.Inputs().json(r)
