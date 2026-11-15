"""Authenticated, model-free all-bank retained-search WDL target descriptions.

No Torch/project/native chess imports. Archived F32, stored F16 and the F32
source-formula normalization are distinct; no target or training file is written.
"""
from __future__ import annotations
import argparse
from collections import Counter
from collections.abc import Callable
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import subprocess
import sys
from typing import Any
import zipfile
REPO = Path(__file__).resolve().parents[1]
MAX_METADATA = 128 << 20
QUANTILES = (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)
REPRESENTATIONS = ('archived_f32', 'stored_f16_widened', 'normalized_f16_source_f32')
END_CLASSES = ('NATURAL', 'STRICT_TB_ADJUDICATED')
DISTANCES = ('0-4', '5-16', '17-64', '65+', 'UNKNOWN')
COMPRESSORS = {'blosc', 'zstd', 'lz4', 'gzip', 'zlib', 'bz2', 'lzma'}
METRICS = ('Q', 'D', 'mass', 'Q_residual', 'Q_absolute_residual', 'Q_squared_residual', 'draw_residual', 'draw_absolute_residual', 'draw_squared_residual', 'full_vector_squared_residual', 'negative_sum_p_log_p', 'zero_W', 'zero_L', 'zero_W_or_L', 'numerical_boundary_mass')
RAW_KEYS = {'physical_row_id', 'policy_target_raw', 'outcome_wdl', 'pov_white', 'search_wdl_raw', 'has_search_raw', 'categorical_raw', 'has_categorical_raw'}

def require(ok: object, message: str) -> None:
    if not ok:
        raise ValueError(message)

def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()

def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for (key, item) in pairs:
        require(key not in value, 'duplicate JSON key')
        value[key] = item
    return value

def stamp(path: Path, *, readonly: bool=True) -> tuple[int, ...]:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and (not path.is_symlink()), 'ordinary input file')
    require(not readonly or not info.st_mode & 146, 'writable input')
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode)

class Inputs:
    """Authenticate before decode, cache only custody stamps, never input arrays."""

    def __init__(self) -> None:
        self.seen: dict[Path, tuple[int, ...]] = {}
        self.bytes_hashed = 0
        self.hashes: dict[Path, str] = {}

    def file(self, path: Path, expected: str, *, cap: int, readonly: bool=True) -> None:
        require(path.is_absolute() and path.resolve() == path, 'absolute non-symlink input path')
        before = stamp(path, readonly=readonly)
        require(before[2] <= cap, 'input byte cap before read/hash')
        if path not in self.seen:
            h = hashlib.sha256()
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC), 'rb') as stream:
                require(stamp_fd(stream.fileno()) == before, 'opened input identity')
                for block in iter(lambda : stream.read(1 << 20), b''):
                    h.update(block)
                require(stamp_fd(stream.fileno()) == before, 'input read custody')
            require(h.hexdigest() == expected and stamp(path, readonly=readonly) == before, 'input SHA/custody')
            self.seen[path] = before
            self.bytes_hashed += before[2]
        else:
            require(self.seen[path] == before, 'input changed since authentication')
            require(self.hashes[path] == expected, 'conflicting input reference')
        self.hashes[path] = expected

    def json(self, ref: dict[str, Any], cap: int=MAX_METADATA, *, readonly: bool=True) -> Any:
        require(set(ref) == {'path', 'sha256'}, 'authenticated reference fields')
        path = Path(ref['path'])
        self.file(path, ref['sha256'], cap=cap, readonly=readonly)
        with path.open('rb') as stream:
            data = stream.read(cap + 1)
        require(len(data) <= cap and stamp(path, readonly=readonly) == self.seen[path], 'metadata changed during parse')
        return json.loads(data, object_pairs_hook=unique_object, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

    def inventory(self, value: dict[str, Any], limits: dict[str, int]) -> None:
        root = Path(value['path'])
        files = value['files']
        require(root.is_absolute() and root.resolve() == root and root.is_dir() and (not root.is_symlink()) and (not root.stat().st_mode & 146), 'readonly inventory root')
        require(isinstance(files, list) and 0 < len(files) <= limits['max_inventory_files'], 'inventory file count')
        names = [row['path'] for row in files]
        require(names == sorted(set(names)), 'duplicate/unsorted inventory')
        require(sum(row['bytes'] for row in files) <= limits['max_inventory_bytes'], 'inventory total bytes')
        actual: list[str] = []
        for (directory, dirs, entries) in os.walk(root, followlinks=False):
            base = Path(directory)
            require(not any((base / name).is_symlink() for name in dirs + entries), 'inventory symlink')
            actual.extend((base / name).relative_to(root).as_posix() for name in entries)
            require(len(actual) <= limits['max_inventory_files'], 'undeclared inventory size')
        require(sorted(actual) == names, 'added/omitted inventory member')
        for row in files:
            name = PurePosixPath(row['path'])
            require(set(row) == {'path', 'bytes', 'sha256'} and (not name.is_absolute()) and ('..' not in name.parts) and ('\\' not in str(name)) and (type(row['bytes']) is int) and (row['bytes'] >= 0), 'unsafe inventory entry')
            path = root / str(name)
            require(stamp(path)[2] == row['bytes'], 'inventory size mismatch')
            self.file(path, row['sha256'], cap=limits['max_inventory_bytes'])

    def final(self) -> None:
        for (path, before) in self.seen.items():
            require(stamp(path, readonly=False) == before, 'final input custody')

def stamp_fd(fd: int) -> tuple[int, ...]:
    s = os.fstat(fd)
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)

def closed(terminal: dict[str, Any], command: list[str], owner: str, resource: str, seconds: int) -> None:
    require(terminal['status'] == 'PASS' and type(terminal['returncode']) is int and (terminal['returncode'] == 0) and (terminal['owned_processes_closed'] is True) and (terminal['custody'] == 'OWNED_PROCESSES_CLOSED') and (terminal['command'] == command) and (terminal['supervisor_sha256'] == owner) and (terminal['resource'] == resource) and (terminal['seconds'] == seconds) and bool(terminal['known_processes']), 'exact closed input owner required')

def authenticate(p: dict[str, Any], inputs: Inputs) -> dict[str, Any]:
    """Candidate generation schema never substitutes for postclosure promotion."""
    selection = inputs.json(p['selection'])
    require(selection['status'] == 'ROOT_SELECTS_PROSPECTIVE_MODEL_FREE_ALL_BANK_WDL_TARGET_CENSUS_V1' and selection['study_uuid'] == p['study_uuid'] and (selection['data_authority'] == p['data_authority']), 'selection/data authority mismatch')
    clarification = inputs.json(p['clarification'])
    require(clarification['status'] == 'ROOT_CLARIFIES_PRECOMPUTE_TARGET_CENSUS_STATISTIC_IMPLEMENTATION_V1' and clarification['selection'] == p['selection'] and (clarification['quantiles']['levels'] == list(QUANTILES)) and (clarification['quantiles']['method'] == 'NumPy quantile method=linear'), 'precompute statistic clarification')
    scope = selection['data_scope']
    require(scope['generation_manifest_sha256'] == p['generation_manifest']['sha256'], 'selection generation mismatch')
    authority = inputs.json(p['data_authority'])
    require(authority['status'] == 'ROOT_AUTHENTICATES_EXISTING_FIXED_TRAIN_BANK_FOR_RETAINED_SEARCH_V1' and authority['new_data_acceptance_issued'] is False and (authority['generation_manifest'] == p['generation_manifest']) and (p['promotion'] in authority['original_acceptance_refs']), 'original root bank authority required')
    promotion = inputs.json(p['promotion'])
    require(promotion['status'] == 'ROOT_PROMOTES_CLOSED_AUDIT_GENERATION_FOR_RESEARCH_LEARNER_V1' and promotion['generation'] == p['generation_manifest'] and (promotion['attempts'] == promotion['eligible_games'] == scope['physical_TRAIN_games']) and (promotion['eligible_rows'] == scope['rows']), 'original postclosure promotion mismatch')
    audit = inputs.json(promotion['audit_protocol'])
    limits = audit['limits']
    require(all(type(v) is int and v > 0 for v in limits.values()) and limits['max_metadata_bytes'] <= MAX_METADATA and (limits['max_raw_decoded_bytes_per_game'] <= 64 << 20) and (limits['max_shard_decoded_bytes_per_game'] <= 64 << 20), 'inherited bounded input limits')
    cap = limits['max_metadata_bytes']
    terminal = inputs.json(promotion['audit_closed_owner'], cap, readonly=False)
    closed(terminal, promotion['exact_command'], audit['runtime']['owner']['sha256'], 'cpu', audit['runtime']['seconds'])
    require(terminal['command'][-6:] == ['-m', 'scripts.core_finite_admit', '--protocol', promotion['audit_protocol']['path'], '--protocol-sha256', promotion['audit_protocol']['sha256']], 'audit child protocol mismatch')
    report = inputs.json(promotion['audit_readout'], cap)
    require(report['status'] == 'AUDITED_CORE_TRAIN_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1' and report['audit_protocol'] == promotion['audit_protocol'] and (report['generation_manifest'] == p['generation_manifest']) and (report['root_postclosure_promotion_required'] is True) and (report['eligible_games'] == scope['physical_TRAIN_games']) and (report['eligible_rows'] == scope['rows']), 'audit readout mismatch')
    audit_output = Path(audit['output_root'])
    require(Path(p['generation_manifest']['path']) == audit_output / 'generation.json' and Path(promotion['audit_readout']['path']) == audit_output / 'readout.json', 'audit output namespace')
    generation = inputs.json(p['generation_manifest'], cap)
    identity = authority['data_identity']
    require(generation['schema'] == 'ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1' and generation['audit_protocol'] == promotion['audit_protocol'] and all(generation[k] == identity[k] for k in ('source_revision', 'run_uuid')), 'generation identity mismatch')
    games = generation['games']
    require(len(games) == scope['physical_TRAIN_games'] and sum(g['rows'] for g in games) == scope['rows'] and (len({g['physical_game_id'] for g in games}) == len(games)) and (len({g['game_id'] for g in games}) == len(games)) and all(g['bank'] == 'train' and type(g['rows']) is int and (g['rows'] > 0) for g in games), 'complete TRAIN population/unique games')
    producer = inputs.json(audit['producer_protocol'], cap)
    require(all(producer[k] == generation[k] for k in ('run_uuid', 'generation')) and producer['source']['revision'] == identity['source_revision'] and (producer['actor']['sha256'] == generation['actor_sha256']), 'original producer identity')
    po = audit['producer_owner']
    pt = inputs.json(po['terminal'], cap, readonly=False)
    closed(pt, po['command'], po['owner_sha256'], 'gpu', producer['owner']['seconds'])
    require(po['owner_sha256'] == producer['owner']['sha256'], 'producer owner source')
    for flag in ('--protocol', '--protocol-sha256'):
        expected = audit['producer_protocol']['path' if flag == '--protocol' else 'sha256']
        require(po['command'].count(flag) == 1 and po['command'][po['command'].index(flag) + 1] == expected, 'producer child protocol mismatch')
    full = inputs.json(audit['producer_inventory'], cap)
    require(full['path'] == producer['output_root'], 'producer inventory root')
    inputs.inventory(full, limits)
    readout = inputs.json(audit['producer_readout'], cap)
    require(readout['status'] == 'COLLECTED_CORE_GENERATION_V1' and readout['protocol_sha256'] == audit['producer_protocol']['sha256'] and all(readout[k] == producer[k] for k in ('run_uuid', 'collection_uuid', 'generation', 'attempts')) and (readout['source_revision'] == identity['source_revision']) and (readout['actor_sha256'] == generation['actor_sha256']) and (readout['all_planned_attempts_accounted'] is True) and (readout['training_steps'] == 0) and (readout['root_acceptance_issued'] is False) and (readout['learning_admitted'] is False), 'original readout identity')
    root = Path(producer['output_root'])
    require(not root.with_name(root.name + '.failed.json').exists() and (not (root / 'failure.json').exists()) and (not list(root.glob('.attempt-*'))) and (not list(root.glob('failed-*.json'))), 'failed/partial producer')
    resolved_path = root / 'resolved-profile.json'
    resolved = inputs.json({'path': str(resolved_path), 'sha256': inputs.hashes[resolved_path]}, cap)
    require(digest(resolved) == producer['resolved_profile_sha256'], 'resolved config identity')
    cfg = resolved['game']
    require(cfg['policy_encoding'] == 'lc0_1858' and cfg['history_rep_fix'] is True and (cfg['policy_row_retention'] == 'all_eligible') and (cfg['search_wdl_draw_mode'] == 'net_raw'), 'original fixed encoding/retention/draw route')
    serving = inputs.json(readout['actor_serving'], cap)
    require(generation['actor_serving'] == readout['actor_serving'] and serving['actor_sha256'] == generation['actor_sha256'] and (serving['pre_state_sha256'] == serving['post_state_sha256'] == serving['checkpoint_model_state_sha256']), 'inherited actor-serving proof')
    return {'scope': scope, 'generation': generation, 'producer': producer, 'readout': readout, 'audit_ref': promotion['audit_protocol'], 'limits': limits, 'config': cfg, 'served': serving['pre_state_sha256'], 'audit_output': str(audit_output)}

def raw_targets(path: Path, n: int, bins: int, cap: int) -> dict[str, Any]:
    """Check every NPY header; decode only identity/outcome/search columns."""
    import numpy as np
    expected = {'policy_target_raw': (np.dtype('f4'), (n, 1858)), 'outcome_wdl': (np.dtype('i1'), (n,)), 'pov_white': (np.dtype('u1'), (n,)), 'search_wdl_raw': (np.dtype('f4'), (n, 3)), 'has_search_raw': (np.dtype('u1'), (n,)), 'categorical_raw': (np.dtype('f4'), (n, bins)), 'has_categorical_raw': (np.dtype('u1'), (n,))}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == 8 and {v.filename for v in entries} == {v + '.npy' for v in RAW_KEYS} and (sum(v.file_size for v in entries) <= cap), 'raw archive members/decoded bound')
        for item in entries:
            require(not item.flag_bits & 1 and item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), 'raw archive codec')
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'NPY header version')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                (shape, fortran, dtype) = reader(stream)
                require(not fortran and (not dtype.hasobject) and (dtype.fields is None), 'unsafe raw array header')
                key = item.filename[:-4]
                if key == 'physical_row_id':
                    require(shape == (n,) and dtype.kind == 'U', 'raw physical ID header')
                else:
                    require((dtype, shape) == expected[key], 'raw field dtype/shape: ' + key)
                require(stream.tell() + math.prod(shape) * dtype.itemsize == item.file_size, 'NPY member length')
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in ('physical_row_id', 'outcome_wdl', 'pov_white', 'search_wdl_raw', 'has_search_raw')}

def stored_arrays(shard: dict[str, Any], n: int, cap: int) -> Any:
    """Safety checks precede Zarr materialization; inherit original full legality proof."""
    import numpy as np
    import zarr
    root = Path(shard['path'])
    total = 0
    metadata = sorted(root.rglob('.zarray'))
    require(bool(metadata), 'no stored array metadata')
    for path in metadata:
        require(path.parent.parent == root and path.stat().st_size <= 65536, 'Zarr metadata cap/location')
        value = json.loads(path.read_bytes(), object_pairs_hook=unique_object)
        dtype = np.dtype(value['dtype'])
        (shape, chunks) = (value['shape'], value['chunks'])
        require(dtype.kind in 'biuf' and (not dtype.hasobject) and (not value.get('filters')), 'unsafe Zarr dtype/filter')
        compressor = value.get('compressor')
        require(compressor is None or compressor.get('id') in COMPRESSORS, 'unsafe Zarr compressor')
        require(isinstance(shape, list) and isinstance(chunks, list) and (1 <= len(shape) <= 4) and (shape[0] == n) and (len(shape) == len(chunks)) and all(type(x) is int and x > 0 for x in shape + chunks), 'Zarr shape/chunk bound')
        total += math.prod((a + b - 1) // b * b for (a, b) in zip(shape, chunks, strict=True)) * dtype.itemsize
        require(total <= cap, 'Zarr padded decoded bound')
    return zarr.open_group(str(root), mode='r')

def stop_distance(row: dict[str, Any], audit: dict[str, Any]) -> int | None:
    """Distance to observed natural stop or adjudication, never future mate."""
    (total, played) = (audit.get('total_plies'), audit.get('played_move_indices'))
    require(total is None or (type(total) is int and total >= 0), 'malformed total ply')
    require(played is None or isinstance(played, list), 'malformed played trace')
    if total is None or played is None:
        return None
    distance = total - row['ply']
    require(0 <= row['move_offset'] < len(played) and distance >= 0 and (distance == len(played) - row['move_offset']), 'inconsistent exact stop distance')
    return distance

def distance_bin(distance: int | None) -> str:
    if distance is None:
        return 'UNKNOWN'
    return '0-4' if distance <= 4 else '5-16' if distance <= 16 else '17-64' if distance <= 64 else '65+'

def game_vectors(item: dict[str, Any], context: dict[str, Any], inputs: Inputs) -> tuple[Any, Any, Any, list[str]]:
    import numpy as np
    n = item['rows']
    limits = context['limits']
    cap = limits['max_metadata_bytes']
    (generation, producer) = (context['generation'], context['producer'])
    ordinal = int(item['physical_game_id'].rsplit(':', 1)[1])
    completed = context['readout']['completed'][str(ordinal)]
    require(item['physical_game_id'] == producer['collection_uuid'] + ':' + str(ordinal), 'physical game ordinal')
    directory = Path(completed['path'])
    require(directory == Path(producer['output_root']) / ('game-' + str(ordinal)) and item['shard'] == completed['shard'], 'game/shard path join')
    inputs.inventory({'path': str(directory), 'files': completed['inventory']}, limits)
    require(Path(item['producer_proof']['path']) == Path(context['audit_output']) / ('game-' + str(ordinal) + '.json'), 'original audited game wrapper path')
    wrapper = inputs.json(item['producer_proof'], cap)
    require(wrapper['status'] == 'VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1' and wrapper['resolved'] is True and (wrapper['full_history_verified'] is True) and (wrapper['raw_targets_verified'] is True) and (wrapper['audit_protocol'] == context['audit_ref']) and (wrapper['evidence'] == completed['evidence']), 'qualified per-game wrapper')
    docs = {k: inputs.json(v, cap) for (k, v) in wrapper['evidence'].items()}
    require(set(docs) == {'provenance', 'validation', 'outcome_proof', 'raw_metadata'}, 'complete inherited proof set')
    (provenance, validation, proof, metadata) = (docs[k] for k in ('provenance', 'validation', 'outcome_proof', 'raw_metadata'))
    end_class = 'STRICT_TB_ADJUDICATED' if item['label_class'] == 'TB_ADJUDICATED' else item['label_class']
    require(end_class in END_CLASSES, 'unresolved/unknown end class')
    common = {'run_uuid': producer['run_uuid'], 'collection_uuid': producer['collection_uuid'], 'source_revision': generation['source_revision'], 'generation': generation['generation'], 'actor_sha256': generation['actor_sha256'], 'served_state_sha256': context['served'], 'physical_game_id': item['physical_game_id'], 'game_id': item['game_id'], 'rows': n, 'bank': 'train', 'scientific_class': end_class}
    for doc in [completed, *docs.values()]:
        require(all(doc.get(k) == v and type(doc.get(k)) is type(v) for (k, v) in common.items()), 'per-game source/role/common identity')
    require(all(wrapper[k] == common[k] for k in common if k not in ('collection_uuid', 'scientific_class')) and wrapper['label_class'] == item['label_class'], 'wrapper game identity')
    require(validation == completed['validation'] and validation['schema'] == 'VERIFIED_CORE_GAME_METADATA_V1' and (validation['full_history_verified'] is True) and (validation['resolved'] is True) and (validation['raw_targets_verified'] is True) and (validation['usable_shard'] is True) and (validation['all_eligible_policy_rows_retained'] is True) and (validation['policy_row_retention'] == 'all_eligible') and (metadata['raw_targets_verified'] is True), 'retention/validation mismatch')
    audit = provenance['actual']
    verified = proof['verification']
    require(provenance['schema'] == 'CORE_GAME_PROVENANCE_V1' and metadata['schema'] == 'CORE_RAW_TARGETS_V1' and (proof['schema'] == 'CORE_PRODUCING_OUTCOME_PROOF_V1') and (verified['audit_sha256'] == digest(audit)) and (proof['proof_sha256'] == digest(verified)) and (metadata['verified_outcome'] == verified) and (metadata['verified_outcome_sha256'] == proof['proof_sha256']) and (validation['verified_outcome_sha256'] == proof['proof_sha256']) and (verified['scientific_class'] == end_class) and (verified['resolved'] is True) and (verified['result'] == audit['result']) and (verified['retained_rows'] == n), 'inherited outcome/audit hash join')
    require((end_class == 'NATURAL' and audit['resolution'] == 'natural' and (audit['natural_result'] == audit['result'])) or (end_class == 'STRICT_TB_ADJUDICATED' and audit['resolution'] == 'tablebase_adjudication' and (audit['natural_result'] == '*')), 'recorded end class/resolution')
    require(all(verified[k] == common[k] for k in ('run_uuid', 'collection_uuid', 'generation', 'source_revision', 'physical_game_id', 'game_id', 'actor_sha256', 'served_state_sha256')), 'verified proof identity')
    require(provenance['planned']['physical_game_id'] == item['physical_game_id'] and audit['start']['root_fen'] == provenance['planned']['root_fen'] and (audit['start']['opening_moves'] == provenance['planned']['opening_moves']), 'planned original history')
    rows = validation['rows_detail']
    records = audit['records']
    require(len(rows) == n and audit['retained_samples'] == n and (audit['game_id'] == item['game_id']) and (not any(v['is_sf_refute_opp'] for v in records)), 'retained record denominator')
    eligible = [v for v in records if v['has_policy'] and (not v['is_sf_refute_opp'])]
    require(len(eligible) == n and [v['ply'] for v in rows] == [v['ply_index'] for v in eligible] and (len({v['ply'] for v in rows}) == len({v['move_offset'] for v in rows}) == n), 'chronological physical ply/move join')
    raw_path = directory / 'raw-targets.npz'
    require(raw_path in inputs.seen and digest(item['shard']['files']) == validation['shard_inventory_sha256'] == wrapper['shard_inventory_sha256'], 'authenticated raw/shard inventory join')
    raw = raw_targets(raw_path, n, context['config']['categorical_bins'], limits['max_raw_decoded_bytes_per_game'])
    arrays = stored_arrays(item['shard'], n, limits['max_shard_decoded_bytes_per_game'])
    attrs = dict(arrays.attrs)
    require(attrs['run_id'] == producer['collection_uuid'] and attrs['model_sha256'] == generation['actor_sha256'] and (attrs['positions'] == n) and (attrs['games'] == 1) and (attrs['history_rep_fix'] is True) and (attrs['input_history_encoding'] == context['config']['input_history_encoding']) and (attrs['policy_encoding'] == 'lc0_1858'), 'stored encoding identity')
    specs = {'search_wdl': ('f2', (n, 3)), 'has_search_wdl': ('u1', (n,)), 'game_id': ('i8', (n,)), 'ply_index': ('i4', (n,)), 'wdl_target': ('i1', (n,))}
    for (key, (dtype, shape)) in specs.items():
        require(key in arrays and arrays[key].dtype == np.dtype(dtype) and (arrays[key].shape == shape), 'stored field dtype/shape: ' + key)
    for key in ('has_game_id', 'has_ply_index', 'has_policy', 'is_network_turn', 'has_is_network_turn', 'is_selfplay', 'has_is_selfplay', 'has_legal_mask'):
        require(key in arrays and arrays[key].dtype == np.dtype('u1') and (arrays[key].shape == (n,)) and bool((np.asarray(arrays[key]) == 1).all()), 'stored row qualification flag: ' + key)
    require(np.array_equal(raw['physical_row_id'], [item['physical_game_id'] + ':' + str(i) for i in range(n)]) and np.array_equal(np.asarray(arrays['game_id']), np.full(n, item['game_id'])) and np.array_equal(np.asarray(arrays['ply_index']), [r['ply'] for r in rows]), 'stored/raw physical row identity')
    require(bool((raw['has_search_raw'] == 1).all()) and np.array_equal(np.asarray(arrays['has_search_wdl']), raw['has_search_raw']), 'missing/invalid search availability')
    values = raw['search_wdl_raw']
    stored = np.asarray(arrays['search_wdl'])
    require(bool(np.isfinite(values).all()) and bool((values >= 0).all()) and bool((np.abs(values.sum(axis=1) - 1) < 1e-05).all()), 'invalid original F32 search simplex')
    require(stored.tobytes() == values.astype(np.float16).tobytes(), 'exact F32-to-F16 target quantization')
    outcome = raw['outcome_wdl']
    require(bool(np.isin(outcome, [0, 1, 2]).all()) and np.array_equal(np.asarray(arrays['wdl_target']), outcome), 'genuine stored outcome')
    bins = []
    result = audit['result']
    require(result in ('1-0', '0-1', '1/2-1/2'), 'resolved outcome value')
    for (i, (row, record)) in enumerate(zip(rows, eligible, strict=True)):
        require(type(row['ply']) is int and row['ply'] >= 0 and (type(row['move_offset']) is int) and (row['move_offset'] >= 0) and (row['ply'] == audit['start']['start_ply'] + row['move_offset']) and (record['move_offset'] == row['move_offset']) and (type(record['pov_color']) is bool) and (raw['pov_white'][i] in (0, 1)) and (bool(raw['pov_white'][i]) == record['pov_color'] == (row['ply'] % 2 == 0)), 'raw POV/physical ply')
        expected_outcome = 1 if result == '1/2-1/2' else 0 if (result == '1-0') == record['pov_color'] else 2
        require(int(outcome[i]) == expected_outcome, 'outcome result/POV mismatch')
        require(row['physical_row_id'] == str(raw['physical_row_id'][i]) and all(isinstance(row[k], str) and len(row[k]) == 64 for k in ('history_sha256', 'stored_input_float16_sha256', 'stored_input_float16_mirror_sha256')), 'inherited row/history/input identities')
        bins.append(distance_bin(stop_distance(row, audit)))
    return (values, stored.astype(np.float32), outcome, bins)

def metrics(vector: Any, outcome: Any) -> Any:
    import numpy as np
    p = vector.astype(np.float64)
    y = np.eye(3, dtype=np.float64)[outcome]
    q = p[:, 0] - p[:, 2]
    d = p[:, 1]
    qerr = q - (y[:, 0] - y[:, 2])
    derr = d - y[:, 1]
    logs = np.zeros_like(p)
    np.log(p, out=logs, where=p > 0)
    return np.column_stack((q, d, p.sum(axis=1), qerr, np.abs(qerr), qerr ** 2, derr, np.abs(derr), derr ** 2, ((p - y) ** 2).sum(axis=1), -(p * logs).sum(axis=1), p[:, 0] == 0, p[:, 2] == 0, (p[:, 0] == 0) | (p[:, 2] == 0), np.abs(np.abs(q) - (1 - d)) <= 1e-06))

def groups() -> list[str]:
    return ['overall', *['end:' + s for s in END_CLASSES], *['outcome:' + s for s in ('W', 'D', 'L')], *['distance:' + s for s in DISTANCES], *['end_distance:' + e + ':' + s for e in END_CLASSES for s in DISTANCES]]

def aggregate(banked: list[dict[str, Any]]) -> dict[str, Any]:
    """Only scalar metrics survive each game; no input/policy/target corpus cache."""
    import numpy as np
    result = {}
    for group in groups():
        cells = {}
        for rep in REPRESENTATIONS:
            cells_for_games = [g['metrics'][rep][group] for g in banked if group in g['metrics'][rep]]
            if not cells_for_games:
                cells[rep] = {'rows': 0, 'physical_games': 0, 'sums': None, 'row_means': None, 'equal_game_means': None, 'quantiles': None}
                continue
            rows = np.concatenate(cells_for_games, axis=0)
            cells[rep] = {'rows': len(rows), 'physical_games': len(cells_for_games), 'sums': dict(zip(METRICS, rows.sum(axis=0).tolist(), strict=True)), 'row_means': dict(zip(METRICS, rows.mean(axis=0).tolist(), strict=True)), 'equal_game_means': dict(zip(METRICS, np.stack([r.mean(axis=0) for r in cells_for_games]).mean(axis=0).tolist(), strict=True)), 'quantiles': {k: np.quantile(rows[:, i], QUANTILES, method='linear').tolist() for (i, k) in enumerate(METRICS)}, 'finite_rows': len(rows), 'nonnegative_vector_rows': len(rows), 'mass_min': float(rows[:, 2].min()), 'mass_max': float(rows[:, 2].max())}
        result[group] = cells
    return result

def fresh_output(output: Path, roots: list[Path]) -> None:
    require(output.is_absolute() and output.resolve() == output and (not output.exists()) and (not output.is_symlink()) and (not output.with_name(output.name + '.failed.json').exists()) and (not output.with_name(output.name + '.failed.json').is_symlink()) and (not any((q / '.git').exists() or (q / '.git').is_symlink() for q in (output, *output.parents))) and all(not output.is_relative_to(q) and (not q.is_relative_to(output)) for q in roots), 'fresh external disjoint output')

def write_json(path: Path, value: Any) -> None:
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)

def census(p: dict[str, Any], *, final_guard: Callable[[], None]=lambda : None) -> dict[str, Any]:
    """All gates precede usable output. A technical fault emits UNQUALIFIED only."""
    inputs = Inputs()
    output = Path(p['output'])
    context = authenticate(p, inputs)
    original = Path(context['producer']['output_root'])
    roots = [original, *[path if path.parent == output.parent else path.parent for path in inputs.seen]]
    fresh_output(output, roots)
    output.mkdir()
    try:
        import numpy as np
        banked = []
        per_game = []
        end_counts: Counter[str] = Counter()
        available = 0
        games = context['generation']['games']
        require(set(context['readout']['completed']) == {str(i) for i in range(len(games))}, 'complete original attempts')
        for game in games:
            (raw, stored, outcome, bins) = game_vectors(game, context, inputs)
            n = game['rows']
            available += n
            clipped = np.maximum(stored, np.float32(0))
            normalized = clipped / np.maximum(clipped.sum(axis=1, keepdims=True, dtype=np.float32), np.float32(1e-08))
            end = 'STRICT_TB_ADJUDICATED' if game['label_class'] == 'TB_ADJUDICATED' else game['label_class']
            end_counts[end] += 1
            which: dict[str, Any] = {'overall': np.ones(n, dtype=bool), 'end:' + end: np.ones(n, dtype=bool)}
            for (y, label) in enumerate(('W', 'D', 'L')):
                which['outcome:' + label] = outcome == y
            for b in DISTANCES:
                mask = np.array([v == b for v in bins])
                which['distance:' + b] = mask
                which['end_distance:' + end + ':' + b] = mask
            data = {}
            sums = {}
            for (rep, vector) in zip(REPRESENTATIONS, (raw, stored, normalized), strict=True):
                m = metrics(vector, outcome)
                require(bool(np.isfinite(m).all()), 'nonfinite derived metric')
                data[rep] = {key: m[mask] for (key, mask) in which.items() if bool(mask.any())}
                sums[rep] = {key: {'rows': len(a), 'sums': dict(zip(METRICS, a.sum(axis=0).tolist(), strict=True))} for (key, a) in data[rep].items()}
            banked.append({'metrics': data})
            per_game.append({'physical_game_id': game['physical_game_id'], 'rows': n, 'end_class': end, 'distance_counts': dict(Counter(bins)), 'representations': sums})
            del raw, stored, normalized, clipped, outcome
        scope = context['scope']
        require(available == scope['expected_search_target_available_rows'] == scope['rows'] and dict(end_counts) == scope['recorded_game_end_classes'], 'full availability/end-class denominators')
        stats = aggregate(banked)
        inputs.final()
        final_guard()
        result = {'status': 'QUALIFIED_ALL_BANK_WDL_TARGET_DESCRIPTIONS_ROOT_CLOSURE_REQUIRED', 'study_uuid': p['study_uuid'], 'selection': p['selection'], 'clarification': p['clarification'], 'generation_manifest': p['generation_manifest'], 'physical_games': len(games), 'rows': available, 'available_rows': available, 'missing_rows': 0, 'invalid_rows': 0, 'end_class_games': dict(end_counts), 'quantile_levels': list(QUANTILES), 'quantile_method': 'linear_unweighted_finite_rows', 'equal_game_scope': 'mean of per-game means among G_s games with >=1 row in that stratum; no CIs', 'metric_arithmetic': 'F64 after originalF32/storedF16-widening/source-formula-F32-normalization', 'stored_f16_metrics': 'algebraic for unnormalized vector, including negative_sum_p_log_p', 'boundary_mass_is_not_clipping_frequency': True, 'distance_is_to_recorded_stop_not_future_TB_ending': True, 'full_legal_outcome_proof': 'Inherited exact accepted original proofs, no new game/TB/native replay', 'source_to_Torch_GPU_bitidentity': 'UNQUALIFIED_NOT_REPLAYED', 'playing_strength_or_training_adoption': False, 'statistics': stats, 'software': {'python': sys.version, 'numpy': np.__version__}, 'input_files_authenticated': len(inputs.seen), 'input_bytes_hashed': inputs.bytes_hashed}
        write_json(output / 'per-game.json', per_game)
        write_json(output / 'readout.json', result)
        output.chmod(0o555)
        return result
    except BaseException as error:
        marker = output.with_name(output.name + '.failed.json')
        for name in ('per-game.json', 'readout.json'):
            owned = output / name
            if owned.exists() or owned.is_symlink():
                owned.unlink()
        write_json(marker, {'status': 'UNQUALIFIED_WDL_TARGET_CENSUS', 'error': repr(error), 'study_uuid': p['study_uuid'], 'clean_statistics_emitted': False})
        raise

def source_guard(source: dict[str, Any]) -> None:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True, env=env).strip() == source['revision'], 'reader source revision')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, env=env), 'dirty reader source')
    required = {'scripts/retained_search_target_census.py', 'tests/test_retained_search_target_census.py'}
    require(set(source['files']) == required, 'exact reader/test source pins')
    for (name, expected) in source['files'].items():
        Inputs().file(REPO / name, expected, cap=4 << 20, readonly=False)

def runtime_guard(p: dict[str, Any], protocol: dict[str, str], publication: dict[str, str], admission: dict[str, str]) -> None:
    require(p['schema'] == 'RETAINED_SEARCH_TARGET_CENSUS_V1' and p['runtime']['inner_seconds'] == 540 and (p['runtime']['owner_seconds'] == 600) and __debug__, 'fixed census CPU caps/schema')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.sched_getaffinity(0) == {20, 22} and all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')), 'CPU-only regime')
    require(not any(n == k or n.startswith(k + '.') for n in sys.modules for k in ('torch', 'onnxruntime', 'chess_anti_engine')), 'forbidden runtime import')
    source_guard(p['reader_source'])
    inputs = Inputs()
    pub = inputs.json(publication, 4 << 20)
    ad = inputs.json(admission, 4 << 20)
    require(pub['status'] == 'ROOT_ACCEPTS_MAIN_MERGED_RETAINED_SEARCH_TARGET_CENSUS_PREREGISTRATION_V1' and pub['state'] == 'MAIN_MERGED' and (pub['protocol'] == protocol) and (pub['selection'] == p['selection']) and (pub['source'] == p['reader_source']), 'main prereg/source publication')
    owner = p['runtime']['owner']
    parent = Path('/proc', str(os.getppid()), 'cmdline').read_bytes().decode().rstrip('\x00').split('\x00')
    inputs.file(Path(owner['path']), owner['sha256'], cap=4 << 20, readonly=False)
    require(owner['path'] in parent and parent.count('--resource') == parent.count('--seconds') == parent.count('--name') == parent.count('--') == 1 and (parent[parent.index('--resource') + 1] == 'cpu') and (parent[parent.index('--seconds') + 1] == '600') and (parent[parent.index('--name') + 1] == p['runtime']['stage']), 'direct root CPU600 owner')
    child = parent[parent.index('--') + 1:]
    require(child.count('--admission-sha256') == 1, 'admission command')
    child[child.index('--admission-sha256') + 1] = 'THIS_ADMISSION_SHA256'
    require(ad['status'] == 'ROOT_ADMITS_RETAINED_SEARCH_TARGET_CENSUS_CPU_V1' and ad['protocol'] == protocol and (ad['publication'] == publication) and (ad['argv_admission_sha256_placeholder'] == child), 'exact root census admission')

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('protocol', 'protocol-sha256', 'publication', 'publication-sha256', 'admission', 'admission-sha256'):
        ap.add_argument('--' + key, required=True)
    a = ap.parse_args()
    pref = {'path': a.protocol, 'sha256': a.protocol_sha256}
    pub = {'path': a.publication, 'sha256': a.publication_sha256}
    ad = {'path': a.admission, 'sha256': a.admission_sha256}
    p = Inputs().json(pref, 4 << 20)
    runtime_guard(p, pref, pub, ad)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('census inner540')))
    signal.alarm(540)
    try:

        def final_guard() -> None:
            require(Inputs().json(pref, 4 << 20) == p, 'protocol changed')
            runtime_guard(p, pref, pub, ad)
        census(p, final_guard=final_guard)
    finally:
        signal.alarm(0)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
