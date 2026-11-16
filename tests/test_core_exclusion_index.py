"""Tiny fictional readonly metadata graphs; never real exposure/bank authority."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from scripts import core_exclusion_index as index
from scripts import core_finite_collect as collect


def sealed(path: Path, body: Any) -> dict[str, str]:
    if path.exists():
        path.chmod(0o644)
    path.write_bytes(json.dumps(body, sort_keys=True, separators=(',', ':')).encode())
    path.chmod(0o444)
    return {'path': str(path), 'sha256': collect.digest(path)}


def case(tmp_path: Path, *, fault: str = ''):
    rows = [('history_sha256', ['1' * 64, '2' * 64]), ('history_sha256', ['3' * 64]),
            ('stored_input_float16_sha256', ['a' * 64, 'b' * 64])]
    if fault == 'duplicate_inside':
        rows[0][1][1] = rows[0][1][0]
    elif fault == 'duplicate_across':
        rows[1][1][0] = rows[0][1][1]
    elif fault == 'duplicate_domain':
        rows[2][1][0] = rows[0][1][0]
    elif fault == 'unsorted':
        rows[0][1].reverse()
    elif fault == 'invalid_sha':
        rows[0][1][0] = 'x' * 64
    chunks = []
    for ordinal, (kind, values) in enumerate(rows):
        body = {'schema': index.CHUNK_SCHEMA, 'kind': kind, 'values': values}
        if fault == 'recursive' and ordinal == 0:
            body = {'schema': index.SCHEMA, 'chunks': []}
        ref = sealed(tmp_path / ('chunk-' + str(ordinal) + '.json'), body)
        chunks.append({'kind': kind, **ref, 'bytes': Path(ref['path']).stat().st_size, 'count': len(values)})
    if fault == 'order':
        chunks[0], chunks[1] = chunks[1], chunks[0]
    elif fault == 'kind_order':
        chunks.reverse()
    elif fault == 'same_path':
        chunks[1]['path'] = chunks[0]['path']
    elif fault == 'count_bool':
        chunks[0]['count'] = True
    elif fault == 'bytes_bool':
        chunks[0]['bytes'] = True
    elif fault == 'wrong_bytes':
        chunks[0]['bytes'] += 1
    manifest = {'schema': index.SCHEMA, 'chunks': chunks, 'unique_counts': {'history_sha256': 3, 'stored_input_float16_sha256': 2},
                'serialized_chunk_bytes': sum(row['bytes'] for row in chunks), 'coverage_scope': 'fictional tiny explicit test scope',
                'donor_pretraining_exposure': 'UNKNOWN'}
    if fault == 'count':
        manifest['unique_counts']['history_sha256'] += 1
    elif fault == 'total':
        manifest['serialized_chunk_bytes'] += 1
    elif fault == 'coverage':
        manifest['coverage_scope'] = ''
    elif fault == 'pretraining':
        manifest['donor_pretraining_exposure'] = 'PROVEN_UNSEEN'
    elif fault == 'manifest_extra':
        manifest['fallback'] = True
    elif fault == 'too_many':
        manifest['chunks'] = chunks * 43
    envelope = {'schema': index.SCHEMA, 'manifest': sealed(tmp_path / 'manifest.json', manifest)}
    if fault == 'missing':
        Path(chunks[0]['path']).unlink()
    elif fault == 'hash':
        chunks[0]['sha256'] = 'e' * 64
        envelope['manifest'] = sealed(tmp_path / 'manifest.json', manifest)
    elif fault == 'writable':
        Path(chunks[0]['path']).chmod(0o644)
    elif fault == 'symlink':
        link = tmp_path / 'linked.json'
        link.symlink_to(chunks[0]['path'])
        chunks[0]['path'] = str(link)
        envelope['manifest'] = sealed(tmp_path / 'manifest.json', manifest)
    return envelope, manifest


def protocol(tmp_path: Path, envelope: dict[str, Any]) -> dict[str, Any]:
    playing = sealed(tmp_path / 'playing.json', [{'id': 'fiction-playing', 'input_sha256': 'c' * 64, 'mirror_sha256': 'd' * 64}])
    return {'source': {'sha256': {'scripts/core_exclusion_index.py': collect.digest(Path(index.__file__))}},
            'exclusions': {'prior': envelope, 'playing_starts': playing, 'playing_start_count': 1,
                           'prior_overlap_action': 'refuse', 'playing_overlap_action': 'refuse'}}


def test_actual_chunk_reader_returns_existing_flat_membership_api(tmp_path: Path):
    envelope, _ = case(tmp_path)
    actual = index.load(envelope)
    assert actual == {'history_sha256': ['1' * 64, '2' * 64, '3' * 64], 'stored_input_float16_sha256': ['a' * 64, 'b' * 64],
                      'coverage_scope': 'fictional tiny explicit test scope', 'donor_pretraining_exposure': 'UNKNOWN'}
    p = protocol(tmp_path, envelope)
    assert collect.exclusion_inputs(p)[0] == actual


@pytest.mark.parametrize('fault', ['duplicate_inside', 'duplicate_across', 'duplicate_domain', 'unsorted', 'invalid_sha',
    'recursive', 'order', 'kind_order', 'same_path', 'count_bool', 'bytes_bool', 'wrong_bytes', 'count', 'total', 'coverage',
    'pretraining', 'manifest_extra', 'too_many', 'missing', 'hash', 'writable', 'symlink'])
def test_rehashed_bad_graph_never_returns_partial_membership(tmp_path: Path, fault: str):
    envelope, _ = case(tmp_path, fault=fault)
    expected = {
        'duplicate_inside': '^duplicate/unsorted exclusion identity$', 'duplicate_across': '^duplicate/unsorted exclusion identity$',
        'duplicate_domain': '^duplicate/unsorted exclusion identity$', 'unsorted': '^duplicate/unsorted exclusion identity$',
        'invalid_sha': '^invalid exclusion SHA256$', 'recursive': '^exclusion chunk schema/count/bytes differ$',
        'order': '^duplicate/unsorted exclusion identity$', 'kind_order': '^exclusion chunk kind order differs$',
        'same_path': '^duplicate/recursive exclusion chunk path$', 'count_bool': '^invalid exclusion count/bytes$',
        'bytes_bool': '^invalid exclusion count/bytes$', 'wrong_bytes': '^exclusion chunk schema/count/bytes differ$',
        'count': '^exclusion declared totals differ$', 'total': '^exclusion declared totals differ$',
        'coverage': '^invalid exclusion manifest$', 'pretraining': '^invalid exclusion manifest$', 'manifest_extra': '^invalid exclusion manifest$',
        'too_many': '^invalid exclusion chunk/count roster$', 'missing': r"^\[Errno 2\] No such file or directory: ",
        'hash': '^bound file changed: ', 'writable': '^exclusion metadata must be readonly regular files$',
        'symlink': '^exclusion metadata must be canonical and external$',
    }
    with pytest.raises((ValueError, FileNotFoundError), match=expected[fault]):
        index.load(envelope)


@pytest.mark.parametrize('field', ['manifest', 'chunk', 'chunks', 'serialized_chunks'])
def test_exact_small_limit_boundary_and_one_under(tmp_path: Path, field: str):
    envelope, manifest = case(tmp_path)
    limits = index._Limits(manifest=Path(envelope['manifest']['path']).stat().st_size,
        chunk=max(item['bytes'] for item in manifest['chunks']), chunks=3, serialized_chunks=manifest['serialized_chunk_bytes'])
    assert index._load(envelope, limits)['history_sha256'] == ['1' * 64, '2' * 64, '3' * 64]
    reduced = replace(limits, **{field: getattr(limits, field) - 1})
    expected = {
        'manifest': '^exclusion metadata byte cap$', 'chunk': '^exclusion chunk byte cap$',
        'chunks': '^invalid exclusion chunk/count roster$', 'serialized_chunks': '^exclusion serialized chunk total cap$',
    }
    with pytest.raises(ValueError, match=expected[field]):
        index._load(envelope, reduced)


def test_explicit_empty_index_is_not_missing_or_imputed_exposure(tmp_path: Path):
    ref = sealed(tmp_path / 'empty.json', {'schema': index.SCHEMA, 'chunks': [],
        'unique_counts': dict.fromkeys(index.KINDS, 0), 'serialized_chunk_bytes': 0,
        'coverage_scope': 'explicit fictional empty set, not a universal unseen claim', 'donor_pretraining_exposure': 'UNKNOWN'})
    result = index.load({'schema': index.SCHEMA, 'manifest': ref})
    assert result['history_sha256'] == []
    assert result['stored_input_float16_sha256'] == []


def test_legacy_flat_route_and_collision_decision_are_unchanged(tmp_path: Path):
    envelope, _ = case(tmp_path)
    p = protocol(tmp_path, envelope)
    completed = {0: {'validation': {'rows_detail': [{'physical_row_id': 'fiction:0:0', 'history_sha256': '1' * 64,
        'stored_input_float16_sha256': 'a' * 64, 'stored_input_float16_mirror_sha256': 'f' * 64}]}}}
    chunked = collect.exclusions_gate(completed, p)
    prior = index.load(envelope)
    prior_ref = sealed(tmp_path / 'flat.json', prior)
    Path(prior_ref['path']).chmod(0o644)  # Legacy route historically permits stable writable metadata.
    p['exclusions']['prior'] = prior_ref
    flat = collect.exclusions_gate(completed, p)
    assert flat == chunked
    assert flat['qualified'] is False
    p['exclusions']['prior_overlap_action'] = 'report'
    assert collect.exclusions_gate(completed, p)['qualified'] is True


def test_bad_chunk_direct_execute_stops_before_actor_or_engine(tmp_path: Path, monkeypatch):
    envelope, _ = case(tmp_path, fault='missing')
    p = protocol(tmp_path, envelope)
    p['schema'] = 'CORE_FINITE_GENERATION_V1'
    actor_calls = []
    monkeypatch.setattr(collect, 'load_actor', lambda *_: actor_calls.append('forbidden'))
    with pytest.raises(FileNotFoundError):
        collect.execute(p, 'a' * 64, [], cast(collect.CollectionConfigs, {}), {}, {})
    assert not actor_calls


@pytest.mark.parametrize('fault', ['unknown_schema', 'extra_envelope', 'missing_helper', 'wrong_helper'])
def test_opt_in_cannot_fallback_or_import_unpinned_helper(tmp_path: Path, fault: str):
    envelope, _ = case(tmp_path)
    p = protocol(tmp_path, envelope)
    if fault == 'unknown_schema':
        envelope['schema'] = 'unknown'
    elif fault == 'extra_envelope':
        envelope['fallback'] = True
    elif fault == 'missing_helper':
        p['source']['sha256'].clear()
    else:
        p['source']['sha256']['scripts/core_exclusion_index.py'] = 'e' * 64
    expected = {
        'unknown_schema': '^unsupported chunked exclusion envelope$', 'extra_envelope': '^unsupported chunked exclusion envelope$',
        'missing_helper': '^chunked exclusion helper source mismatch$', 'wrong_helper': '^chunked exclusion helper source mismatch$',
    }
    with pytest.raises(ValueError, match=expected[fault]):
        collect.exclusion_inputs(p)


def test_duplicate_manifest_keys_and_changed_input_refused(tmp_path: Path, monkeypatch):
    envelope, manifest = case(tmp_path)
    real = collect.checked
    def changed(ref):
        value = real(ref)
        if ref == envelope['manifest']:
            Path(ref['path']).chmod(0o644)
        return value
    monkeypatch.setattr(collect, 'checked', changed)
    with pytest.raises(ValueError, match=r"^exclusion metadata must be readonly regular files$"):
        index.load(envelope)
    monkeypatch.setattr(collect, 'checked', real)
    path = Path(envelope['manifest']['path'])
    path.chmod(0o644)
    raw = json.dumps(manifest).encode().replace(b'"schema":', b'"schema":"duplicate","schema":', 1)
    path.write_bytes(raw)
    path.chmod(0o444)
    with pytest.raises(ValueError, match='duplicate JSON'):
        index.load({'schema': index.SCHEMA, 'manifest': {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}})


def rehash_graph(directory: Path) -> None:
    """Rebind this test's complete fictional metadata DAG, including CLI hashes."""
    bodies = {str(path): json.loads(path.read_bytes()) for path in directory.rglob('*.json')}
    changes: dict[str, str] = {}
    for _ in range(24):
        modified = False
        for name, body in list(bodies.items()):
            def update(value):
                if isinstance(value, dict):
                    if set(value) == {'path', 'sha256'} and value['path'] in bodies:
                        return {'path': value['path'], 'sha256': collect.digest(value['path'])}
                    return {key: update(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [update(item) for item in value]
                if isinstance(value, str):
                    return changes.get(value, value)
                return value
            path = Path(name)
            before = collect.digest(path)
            updated = update(body)
            ref = sealed(path, updated)
            bodies[name] = updated
            if ref['sha256'] != before:
                changes[before] = ref['sha256']
                modified = True
        if not modified:
            return
    raise AssertionError('fictional metadata graph did not converge')


@pytest.mark.parametrize(('holdout', 'missing'), [(False, ''), (True, ''), (False, 'producer'), (True, 'audit')])
def test_bank_metadata_requires_historical_chunk_helper_pins_without_reopening_chunks(tmp_path: Path, holdout: bool, missing: str):
    from scripts import core_generation_bank_acceptance as accepted
    from tests.test_core_generation_bank_acceptance import graph

    root, review, _ = graph(tmp_path, holdout=holdout)
    spec_path, producer_path = tmp_path / 'spec.json', tmp_path / 'producer.json'
    spec, producer = json.loads(spec_path.read_bytes()), json.loads(producer_path.read_bytes())
    helper = 'scripts/core_exclusion_index.py'
    historical = '9' * 64  # Historical source assertion, not current-consumer file bytes.
    if missing != 'audit':
        spec['source_files'][helper] = historical
    if missing != 'producer':
        producer['source']['sha256'][helper] = historical
    producer['exclusions'] = {'prior': {'schema': index.SCHEMA, 'manifest': {'path': '/NEVER-OPEN/exclusion-index.json', 'sha256': '8' * 64}}}
    sealed(spec_path, spec)
    sealed(producer_path, producer)
    root_body = json.loads(Path(root['path']).read_bytes())
    root_body['audit_source']['source_files'] = spec['source_files']
    sealed(Path(root['path']), root_body)
    rehash_graph(tmp_path)
    root = {'path': root['path'], 'sha256': collect.digest(root['path'])}
    review = {'path': review['path'], 'sha256': collect.digest(review['path'])}
    if missing:
        with pytest.raises(ValueError, match=r'chunked exclusion|invalid digest'):
            accepted._validate_generation(root, review, max_file_bytes=1 << 20, max_total_bytes=8 << 20, max_games=3, holdout=holdout)
    else:
        view = accepted._validate_generation(root, review, max_file_bytes=1 << 20, max_total_bytes=8 << 20, max_games=3, holdout=holdout)
        assert view.evidence['exclusion_index']['source_sha256'] == historical
        assert view.evidence['exclusion_index']['manifest']['path'] == '/NEVER-OPEN/exclusion-index.json'
        assert view.evidence['learning_admitted'] is False


def test_opt_in_requires_producer_helper_pin_before_asset_guard(tmp_path: Path, monkeypatch):
    source = tmp_path / 'source'
    monkeypatch.setattr(collect, '__file__', str(source / 'scripts/core_finite_collect.py'))
    monkeypatch.setattr(collect.subprocess, 'check_output', lambda argv, **_k: b'a' * 40 if 'rev-parse' in argv else b'')
    p = {'schema': 'CORE_FINITE_GENERATION_V1', 'actor': {'kind': 'donor'},
         'source': {'checkout': str(source), 'revision': 'a' * 40, 'sha256': dict.fromkeys(collect.REQUIRED_SOURCE_FILES, 'b' * 64)},
         'exclusions': {'prior': {'schema': index.SCHEMA, 'manifest': {'path': '/NEVER-OPEN/index.json', 'sha256': 'c' * 64}}}}
    with pytest.raises(ValueError, match='incomplete chunked exclusion source fingerprint'):
        collect.source_guard(p)


@pytest.mark.parametrize(('holdout', 'missing'), [(False, ''), (True, ''), (False, 'producer'), (True, 'audit')])
def test_actual_public_data_audits_consume_new_index_and_require_matching_source(tmp_path: Path, monkeypatch, holdout: bool, missing: str):
    from scripts import core_generation_data_admit as train
    from scripts import core_generation_holdout_admit as hold
    from tests.test_core_finite_admit import entry_case, bind_data_case

    fixture = entry_case(tmp_path, monkeypatch, holdout=holdout)
    index_dir = tmp_path / 'index'
    index_dir.mkdir()
    envelope, _ = case(index_dir)
    p = fixture.protocol
    producer_path = Path(p['producer_protocol']['path'])
    producer = json.loads(producer_path.read_bytes())
    helper = 'scripts/core_exclusion_index.py'
    current_hash = collect.digest(Path(index.__file__))
    if missing != 'producer':
        producer['source']['sha256'][helper] = current_hash
    if missing != 'audit':
        fixture.spec['source_files'][helper] = current_hash
    producer['exclusions']['prior'] = envelope
    producer_ref = sealed(producer_path, producer)
    admission_path = tmp_path / 'producer-admission.json'
    admission = json.loads(admission_path.read_bytes())
    admission['protocol_sha256'] = producer_ref['sha256']
    admission_ref = sealed(admission_path, admission)
    command = list(p['producer_owner']['command'])
    command[command.index('--protocol-sha256') + 1] = producer_ref['sha256']
    command[command.index('--admission-sha256') + 1] = admission_ref['sha256']
    terminal_path = Path(p['producer_owner']['terminal']['path'])
    terminal = json.loads(terminal_path.read_bytes())
    terminal['command'] = command
    p['producer_owner'].update(command=command, terminal=sealed(terminal_path, terminal))
    readout_path = Path(p['producer_readout']['path'])
    readout = json.loads(readout_path.read_bytes())
    readout['protocol_sha256'] = producer_ref['sha256']
    p['producer_readout'] = sealed(readout_path, readout)
    p['producer_protocol'] = producer_ref
    if not missing:
        overlap = collect.exclusions_gate({0: fixture.fixture.item}, producer)
        producer_root = Path(producer['output_root'])
        sealed(producer_root / 'input-exclusions.json', overlap)
        p['producer_inventory'] = sealed(Path(p['producer_inventory']['path']),
            {'path': str(producer_root), 'files': collect.inventory(producer_root, seal=True)})
    pref = bind_data_case(tmp_path, fixture)
    entry = hold if holdout else train
    if missing:
        with pytest.raises(ValueError, match='chunked exclusion audit source pin'):
            entry.main(['--protocol', pref['path'], '--protocol-sha256', pref['sha256']])
    else:
        assert entry.main(['--protocol', pref['path'], '--protocol-sha256', pref['sha256']]) == 0
        report = json.loads((Path(p['output_root']) / 'readout.json').read_bytes())
        assert report['models_constructed'] == report['forwards'] == report['optimizer_updates'] == 0
        assert report['status'] == ('AUDITED_CORE_HOLDOUT_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1' if holdout
                                    else 'AUDITED_CORE_DATA_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1')
