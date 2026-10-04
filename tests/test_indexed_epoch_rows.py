"""Small actual-loader regressions; no production corpus/model/GPU reads."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer, _shard_content_sha256
from chess_anti_engine.replay.indexed_rows import IndexedArray, RowIndexSelection
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import ShardMeta, load_shard_arrays, samples_to_arrays, save_local_shard_arrays


def write_source(root, groups=16):
    root.mkdir()
    samples = []
    for row in range(2 * groups):
        policy = np.zeros(COMPACT_POLICY_SIZE, dtype=np.float32)
        policy[row % COMPACT_POLICY_SIZE] = 1
        legal = (policy != 0).astype(np.uint8)
        samples.append(ReplaySample(x=np.full((146, 8, 8), row % 17, dtype=np.float32),
                                    policy_target=policy, legal_mask=legal, wdl_target=row % 3,
                                    priority=float(row + 1), has_policy=True,
                                    game_id=row // 2, ply_index=row))
    shard = root / 'shard_000000.zarr'
    save_local_shard_arrays(shard, arrs=samples_to_arrays(samples),
                            meta=ShardMeta(positions=len(samples), policy_encoding='lc0_1858',
                                           policy_size=COMPACT_POLICY_SIZE))
    return shard


def manifest(tmp_path, source, retained=None, additional=None):
    if retained is None:
        retained = list(range(0, 32, 2))
    if additional is None:
        additional = list(range(1, 32, 2))
    n = 32
    masks = []
    for rows in (retained, additional):
        values = np.zeros(n, dtype=np.uint8)
        values[rows] = 1
        masks.append(np.packbits(values, bitorder='little').tobytes())
    mask = tmp_path / 'mask.bin'
    mask.write_bytes(b''.join(masks))
    data = {'status': 'PASS_FROZEN_NESTED_PHYSICAL_ROW_SELECTION_ONLY',
            'small_rows': len(retained), 'large_rows': len(retained) + len(additional),
            'cohorts': [{'cohort': 'cohort00', 'source_rows': n, 'bytes_per_half': 4,
                         'rows_per_half': len(retained), 'mask_path': str(mask),
                         'sha256': hashlib.sha256(mask.read_bytes()).hexdigest(),
                         'shards': [{'path': str(source), 'rows': n, 'cohort_row_offset': 0,
                                     'content_sha256': _shard_content_sha256(source)}]}]}
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(data))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def open_buffer(root, pin, arm='retained', counter=None, cap=64 << 20, seed=131):
    return GameAwareEpochBuffer(shard_dir=root, batch_size=8, seed=seed,
                                input_planes=146, input_history_encoding='legacy',
                                history_rep_fix=False, mirror_augmentation=False,
                                plan_workers=1, load_workers=1, max_working_set_bytes=cap,
                                objective_mask_counter=counter,
                                row_index_selection={'path': str(pin[0]), 'sha256': pin[1], 'arm': arm})


def drain(buffer):
    batches = [buffer.sample_batch_arrays(8) for _ in range(buffer.num_batches)]
    assert buffer.receipt()['complete']
    return batches


def setup(tmp_path):
    root = tmp_path / 'source'
    shard = write_source(root)
    return root, manifest(tmp_path, shard)


def test_actual_rows_targets_histories_and_nested_epoch_dose(tmp_path):
    root, pin = setup(tmp_path)
    small = open_buffer(root, pin)
    assert small.plan.rows == 16
    small_batches = drain(small)
    rows = np.concatenate([b['ply_index'] for b in small_batches])
    assert sorted(rows.tolist()) == list(range(0, 32, 2))
    for batch in small_batches:
        assert len(np.unique(batch['game_id'])) == len(batch['game_id'])
        for i, original in enumerate(batch['ply_index']):
            assert np.all(batch['x'][i] == original % 17)
            assert batch['policy_target'][i, original % COMPACT_POLICY_SIZE] == 1
            assert batch['legal_mask'][i, original % COMPACT_POLICY_SIZE] == 1
            assert batch['wdl_target'][i] == original % 3
    second = open_buffer(root, pin, seed=132)
    union = open_buffer(root, pin, 'union')
    large_rows = np.concatenate([b['ply_index'] for b in drain(union)])
    assert sorted(large_rows.tolist()) == list(range(32))
    assert small.num_batches + second.num_batches == union.num_batches == 4
    assert small.plan.ragged_batches == second.plan.ragged_batches == union.plan.ragged_batches == 0
    assert sorted(np.concatenate([b['ply_index'] for b in drain(second)]).tolist()) == sorted(rows.tolist())
    assert small.receipt()['row_index_persistent_array_copies'] is False


def test_objective_populations_selected_and_realized(tmp_path):
    root, pin = setup(tmp_path)
    def count(arrays):
        return {'policy': float(np.asarray(arrays['has_policy']).sum()),
                'selected_priority': float(np.asarray(arrays['priority']).sum())}
    buffer = open_buffer(root, pin, counter=count)
    assert dict(buffer.plan.objective_mask_weights) == {'policy': 16.0, 'selected_priority': 256.0}
    drain(buffer)


def test_deterministic_schedule_and_separate_augmentation_rng(tmp_path):
    root, pin = setup(tmp_path)
    a, b = open_buffer(root, pin), open_buffer(root, pin)
    assert a.plan.plan_sha256 == b.plan.plan_sha256
    assert np.array_equal(a.rng.random(32), b.rng.random(32))
    assert [x['ply_index'].tolist() for x in drain(a)] == [x['ply_index'].tolist() for x in drain(b)]
    other = open_buffer(root, pin, 'union')
    assert a.plan.corpus_sha256 != other.plan.corpus_sha256


def test_original_decode_memory_is_priced(tmp_path):
    root, pin = setup(tmp_path)
    buffer = open_buffer(root, pin)
    record = buffer._records[0]
    assert record.indexed_source_decoded_bytes > record.decoded_bytes
    assert record.validated_load_bytes == 5 * record.indexed_source_decoded_bytes + record.decoded_bytes
    with pytest.raises(ValueError, match=r'working|needs|limit|exceed'):
        open_buffer(root, pin, cap=record.decoded_bytes * 5)


def test_original_source_drift_fails_before_training(tmp_path):
    root, pin = setup(tmp_path)
    buffer = open_buffer(root, pin)
    (root / 'shard_000000.zarr/.zattrs').write_text('{}')
    with pytest.raises(RuntimeError, match='changed'):
        buffer.sample_batch_arrays(8)


def test_mask_hash_and_manifest_hash_reject_drift(tmp_path):
    _root, pin = setup(tmp_path)
    with pytest.raises(ValueError, match='manifest hash'):
        RowIndexSelection(pin[0], '0' * 64, 'retained')
    (tmp_path / 'mask.bin').write_bytes(b'\0' * 8)
    with pytest.raises(ValueError, match='mask hash'):
        RowIndexSelection(*pin, 'retained')


def test_source_coverage_and_content_pin_rejected(tmp_path):
    root, pin = setup(tmp_path)
    selection = RowIndexSelection(*pin, 'retained')
    with pytest.raises(ValueError, match='coverage'):
        selection.check_paths([])
    with pytest.raises(ValueError, match='coverage'):
        selection.check_paths([root / 'shard_000000.zarr'] * 2)
    (root / 'shard_000000.zarr/.zattrs').write_text('{}')
    with pytest.raises(ValueError, match='content'):
        open_buffer(root, pin)


@pytest.mark.parametrize('mutation', ['overlap', 'offset', 'quota', 'padding'])
def test_malformed_masks_fail_closed(tmp_path, mutation):
    _root, pin = setup(tmp_path)
    data = json.loads(pin[0].read_bytes())
    cohort = data['cohorts'][0]
    mask = Path(cohort['mask_path'])
    if mutation == 'overlap':
        mask.write_bytes(b'\xff' * 8)
    elif mutation == 'offset':
        cohort['shards'][0]['cohort_row_offset'] = 1
    elif mutation == 'quota':
        cohort['rows_per_half'] = 17
    else:
        cohort['source_rows'] = 31
        cohort['shards'][0]['rows'] = 31
    cohort['sha256'] = hashlib.sha256(mask.read_bytes()).hexdigest()
    pin[0].write_text(json.dumps(data))
    sha = hashlib.sha256(pin[0].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=r'overlap|padding|offset|quota'):
        RowIndexSelection(pin[0], sha, 'union')


def test_lazy_view_preserves_index_order_and_decodes_only_requested_rows():
    class Source:
        shape, dtype, chunks = (12, 2), np.dtype('<f2'), (4, 2)
        def __init__(self):
            self.calls = []
        def get_orthogonal_selection(self, index):
            self.calls.append(index[0].tolist() if hasattr(index[0], 'tolist') else index[0])
            return np.arange(24, dtype='<f2').reshape(12, 2)[index]
    source = Source()
    view = IndexedArray(source, np.array([2, 5, 9], dtype='<u4'))
    assert source.calls == []
    assert view.shape == (3, 2)
    assert np.array_equal(view[1:2], np.array([[10, 11]], dtype='<f2'))
    assert source.calls == [[5]]
    assert np.array_equal(view[:, 1], np.array([5, 11, 19], dtype='<f2'))


def pin_cohorts(tmp_path, paths, picks):
    from chess_anti_engine.replay import target_overlay as storage
    cohorts = []
    small = large = 0
    for ordinal, (path, retained, additional, count) in enumerate(picks):
        values = []
        for rows in (retained, additional):
            bits = np.zeros(count, dtype=np.uint8)
            bits[rows] = 1
            values.append(np.packbits(bits, bitorder='little').tobytes())
        mask = tmp_path / f'c{ordinal}.bin'
        mask.write_bytes(b''.join(values))
        content = (storage.overlay_content_sha256(path) if storage.has_overlay(path)
                   else _shard_content_sha256(path))
        cohorts.append({'source_rows': count, 'bytes_per_half': (count + 7) // 8,
                        'rows_per_half': len(retained), 'mask_path': str(mask),
                        'sha256': hashlib.sha256(mask.read_bytes()).hexdigest(),
                        'shards': [{'path': str(path), 'rows': count, 'cohort_row_offset': 0,
                                    'content_sha256': content}]})
        small += len(retained)
        large += len(retained) + len(additional)
    assert {str(p) for p in paths} == {str(p[0]) for p in picks}
    data = {'status': 'PASS_FROZEN_NESTED_PHYSICAL_ROW_SELECTION_ONLY',
            'small_rows': small, 'large_rows': large, 'cohorts': cohorts}
    path = tmp_path / 'selection-multi.json'
    path.write_text(json.dumps(data))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_175_plane_qualified_overlay_keeps_original_replacements(tmp_path):
    from tests.test_target_overlay_v2_main_port import _qualified
    from scripts.lc0_control_train import stage_shards
    roots, targets, ref = _qualified(tmp_path)
    pin = pin_cohorts(tmp_path, targets, [(p, [0], [1], 2) for p in targets])
    stage = tmp_path / 'stage'
    stage_shards(roots, stage)
    buffer = GameAwareEpochBuffer(shard_dir=stage, batch_size=2, seed=131,
                                  input_planes=175, input_history_encoding='lc0_root_legacy_meta',
                                  history_rep_fix=True, mirror_augmentation=True,
                                  plan_workers=1, load_workers=1,
                                  overlay_storage_qualification=ref,
                                  row_index_selection={'path': str(pin[0]), 'sha256': pin[1], 'arm': 'retained'})
    assert [r.path for r in sorted(buffer._records, key=lambda r: str(r.path))] == sorted(targets)
    batch = buffer.sample_batch_arrays(2)
    assert buffer.receipt()['complete']
    assert batch['x'].shape == (2, 175, 8, 8)
    policies = sorted(float(v) for v in batch['policy_target'][:, 0])
    assert policies == [0.625, 0.75]
    assert sorted(float(v) for v in batch['search_wdl'][:, 0]) == [0.125, 0.25]
    assert np.all(batch['ply_index'] == 0)
    assert buffer.plan.history_rep_fix


def test_source_namespaces_and_cross_shard_same_game_are_preserved(tmp_path):
    from scripts.lc0_control_train import stage_shards
    root = tmp_path / 'one-source'
    first = write_source(root)
    second = root / 'shard_000001.zarr'
    import shutil
    shutil.copytree(first, second)
    other = write_source(tmp_path / 'other-source')
    paths = [first, second, other]
    pin = pin_cohorts(tmp_path, paths, [(p, list(range(0, 32, 2)), list(range(1, 32, 2)), 32) for p in paths])
    stage = tmp_path / 'stage'
    stage_shards([root, other.parent], stage)
    buffer = open_buffer(stage, pin)
    records = {r.path: r for r in buffer._records}
    assert np.array_equal(records[first].game_keys, records[second].game_keys)
    assert not np.intersect1d(records[first].game_keys, records[other].game_keys).size
    assert buffer.plan.game_count == 32
    batches = drain(buffer)
    assert sum(len(b['x']) for b in batches) == 48
    assert all(len(np.unique(b['game_id'])) == len(b['game_id']) for b in batches)


def test_zero_selected_shard_is_covered_but_not_loaded(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    first = write_source(root)
    import shutil
    second = root / 'shard_000001.zarr'
    shutil.copytree(first, second)
    pin = pin_cohorts(tmp_path, [first, second], [(first, list(range(0, 32, 2)), list(range(1, 32, 2)), 32),
                                             (second, [], [], 32)])
    buffer = open_buffer(root, pin)
    assert buffer.receipt()['row_index_source_shards'] == 2
    assert buffer.plan.shard_count == 1
    import chess_anti_engine.replay.game_epoch as module
    original = module._storage_load
    def load(path, *args, **kwargs):
        if path == second and not kwargs.get('lazy', False):
            pytest.fail('zero-selected shard eagerly decoded')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(module, '_storage_load', load)
    drain(buffer)


def test_normal_path_matches_unmodified_base_code(tmp_path):
    # Frozen oracle from immutable base 0386c2c2986385eb7109fb5f0a7072870b5076bc.
    root, _pin = setup(tmp_path)
    actual = GameAwareEpochBuffer(shard_dir=root, batch_size=8, seed=131,
                                  input_planes=146, input_history_encoding='legacy',
                                  history_rep_fix=False, mirror_augmentation=False,
                                  plan_workers=1, load_workers=1)
    expected = [[14, 17, 28, 22, 18, 30, 27, 13],
                [7, 8, 2, 11, 20, 26, 23, 5],
                [21, 4, 1, 15, 9, 29, 10, 24],
                [6, 3, 12, 0, 25, 16, 31, 19]]
    assert actual.num_batches == 4
    assert actual.plan.ragged_batches == 0
    for rows in expected:
        batch = actual.sample_batch_arrays(8)
        assert batch['ply_index'].tolist() == rows
        for i, row in enumerate(rows):
            assert np.all(batch['x'][i] == row % 17)
            assert batch['policy_target'][i, row % COMPACT_POLICY_SIZE] == 1
            assert np.count_nonzero(batch['policy_target'][i]) == 1
            assert batch['legal_mask'][i, row % COMPACT_POLICY_SIZE] == 1
            assert batch['wdl_target'][i] == row % 3
            assert batch['priority'][i] == row + 1
    assert actual.receipt()['complete']


def test_metadata_reserve_is_fail_closed_before_mask_decode(tmp_path):
    _root, pin = setup(tmp_path)
    with pytest.raises(ValueError, match='metadata reserve'):
        RowIndexSelection(*pin, 'union', max_metadata_bytes=1)
    selection = RowIndexSelection(*pin, 'union')
    assert selection.metadata_reserve_bytes > sum(s.indices.nbytes for s in selection.shards.values())



def test_close_releases_index_arrays_without_changing_receipt(tmp_path):
    root, pin = setup(tmp_path)
    buffer = open_buffer(root, pin)
    drain(buffer)
    before = buffer.receipt()
    buffer.close()
    assert buffer._row_index_selection is None
    assert all(record.row_selection is None for record in buffer._records)
    assert buffer.receipt() == before
    buffer.close()
    assert buffer.receipt() == before


def test_true_empty_source_reservation_is_hash_bound_and_not_scheduled(tmp_path):
    root, pin = setup(tmp_path)
    original, _ = load_shard_arrays(root / 'shard_000000.zarr')
    empty = {name: np.asarray(value)[:0] if np.asarray(value).ndim >= 1
             and np.asarray(value).shape[0] == 32 else np.asarray(value)
             for name, value in original.items()}
    # Quarantine reservations may prune optional per-row identity fields.
    for name in ('game_id', 'has_game_id', 'ply_index', 'has_ply_index'):
        empty.pop(name, None)
    source = root / 'shard_000001.zarr'
    save_local_shard_arrays(source, arrs=empty)
    doc = json.loads(pin[0].read_text())
    doc['cohorts'][0]['shards'].append({'path': str(source), 'rows': 0,
        'cohort_row_offset': 32, 'content_sha256': _shard_content_sha256(source)})
    pin[0].write_text(json.dumps(doc))
    bound = (pin[0], hashlib.sha256(pin[0].read_bytes()).hexdigest())
    buffer = open_buffer(root, bound)
    rows = np.concatenate([b['ply_index'] for b in drain(buffer)])
    assert sorted(rows.tolist()) == list(range(0, 32, 2))
    assert buffer.plan.shard_count == 1
    buffer.close()
    (source / 'source-drift').write_text('changed')
    with pytest.raises(ValueError, match='source content differs'):
        open_buffer(root, bound)


def test_dense_large_shard_index_scratch_rejected_before_mask_read(tmp_path, monkeypatch):
    _root, pin = setup(tmp_path)
    doc = json.loads(pin[0].read_text())
    cohort = doc['cohorts'][0]
    cohort.update(source_rows=10_000_000, bytes_per_half=1_250_000,
                  rows_per_half=5_000_000)
    cohort['shards'][0]['rows'] = 10_000_000
    doc.update(small_rows=5_000_000, large_rows=10_000_000)
    pin[0].write_text(json.dumps(doc))
    digest = hashlib.sha256(pin[0].read_bytes()).hexdigest()
    read = Path.read_bytes
    def no_mask_read(path):
        if path == Path(cohort['mask_path']):
            pytest.fail('oversized index scratch must reject before mask allocation')
        return read(path)
    monkeypatch.setattr(Path, 'read_bytes', no_mask_read)
    with pytest.raises(ValueError, match='metadata reserve exceeds'):
        RowIndexSelection(pin[0], digest, 'union', max_metadata_bytes=100_000_000)
