"""Content-bound replay upgrade failure memo; no real replay data or threads."""
from __future__ import annotations

from collections import deque
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from chess_anti_engine.replay import disk_buffer, threat_upgrade
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.upgrade_failure_cache import UpgradeFailureCache


def _chunk(planes: int = 175) -> dict[str, np.ndarray]:
    x = np.zeros((2, planes, 8, 8), dtype=np.float16)
    x[:, 112] = 1  # Live v1 planes with a zero post-v1 block require repair.
    return {'x': x, '_input_history_encoding': np.asarray('legacy')}


def _buffer(target: int = 175, *, enabled: bool = True) -> DiskReplayBuffer:
    buf = DiskReplayBuffer.__new__(DiskReplayBuffer)
    buf._input_planes = target
    buf._upgrade_v1_planes = enabled
    buf._upgrade_failures = 0
    buf._upgrade_failure_cache = UpgradeFailureCache()
    return buf


def _recomputed(x, history_encoding=None, *, version='v2_threats'):
    del history_encoding
    width = {'v2_threats': 175, 'v3_checks': 179}[version]
    return np.zeros((len(x), width - 112, 8, 8), dtype=np.float32)


def test_repeated_equal_width_failure_preserves_arrays_and_counts():
    buf = _buffer()
    arrs = _chunk()
    original = arrs['x'].copy()
    with patch.object(threat_upgrade, 'recompute_extra_planes', side_effect=_recomputed) as decode:
        for _ in range(4):
            incoming = {key: value.copy() for key, value in arrs.items()}
            assert buf._pad_x_planes(incoming) is incoming
            np.testing.assert_array_equal(incoming['x'], original)
        assert decode.call_count == 1
    assert buf._upgrade_failures == 4


def test_changed_x_same_shape_and_dtype_retries():
    buf, arrs = (_buffer(), _chunk())
    with patch.object(threat_upgrade, 'recompute_extra_planes', side_effect=_recomputed) as decode:
        buf._pad_x_planes(arrs)
        arrs['x'][0, 0, 0, 0] = 1
        buf._pad_x_planes(arrs)
        assert decode.call_count == 2


def test_repaired_content_is_not_hidden_by_previous_failure():
    buf, arrs = (_buffer(), _chunk())
    with patch.object(threat_upgrade, 'recompute_extra_planes', side_effect=_recomputed) as decode:
        buf._pad_x_planes(arrs)
        arrs['x'][:, 112:146] = 0
        arrs['x'][:, 146:] = 1
        assert buf._pad_x_planes(arrs) is arrs
        assert decode.call_count == 1
    assert buf._upgrade_failures == 1


def test_root_presence_and_content_invalidate():
    buf, arrs = (_buffer(), _chunk())
    with patch.object(threat_upgrade, 'recompute_extra_planes', side_effect=_recomputed) as decode:
        buf._pad_x_planes(arrs)
        arrs['x_lc0_root'] = arrs['x'].copy()
        buf._pad_x_planes(arrs)
        arrs['x_lc0_root'][0, 0, 0, 0] = 1
        buf._pad_x_planes(arrs)
        assert decode.call_count == 3


def test_history_and_target_invalidate():
    buf, arrs = (_buffer(), _chunk())
    with patch.object(threat_upgrade, 'recompute_extra_planes', side_effect=_recomputed) as decode:
        buf._pad_x_planes(arrs)
        arrs['_input_history_encoding'] = np.asarray('lc0_root')
        buf._pad_x_planes(arrs)
        buf._input_planes = 179
        out = buf._pad_x_planes(arrs)
        assert out['x'].shape[1] == 179
        assert decode.call_count == 3


def test_new_buffer_retries_and_does_not_persist_failure():
    arrs = _chunk()
    with patch.object(threat_upgrade, 'recompute_extra_planes', side_effect=_recomputed) as decode:
        _buffer()._pad_x_planes(arrs)
        _buffer()._pad_x_planes(arrs)
        assert decode.call_count == 2


def test_unknown_value_error_is_not_cached():
    buf, arrs = (_buffer(), _chunk())
    with patch.object(disk_buffer, 'upgrade_arrays_to_planes', side_effect=ValueError('temporary')) as decode:
        buf._pad_x_planes(arrs)
        buf._pad_x_planes(arrs)
        assert decode.call_count == 2


def test_non_value_error_still_propagates():
    buf, arrs = (_buffer(), _chunk())
    with patch.object(disk_buffer, 'upgrade_arrays_to_planes', side_effect=RuntimeError('temporary')):
        with pytest.raises(RuntimeError, match='temporary'):
            buf._pad_x_planes(arrs)
    assert buf._upgrade_failures == 0


def test_input_mutation_during_failed_attempt_is_not_cached():
    buf, arrs = (_buffer(), _chunk())

    def mutate(*_args):
        arrs['x'][0, 0, 0, 0] += 1
        raise threat_upgrade.PlaneUpgradeValidationError('mismatch')
    with patch.object(disk_buffer, 'upgrade_arrays_to_planes', side_effect=mutate) as decode:
        buf._pad_x_planes(arrs)
        buf._pad_x_planes(arrs)
        assert decode.call_count == 2


def test_successful_repair_is_not_cached_as_failure():
    buf, arrs = (_buffer(), _chunk())

    def valid(x, *_args):
        out = np.ones((len(x), 63, 8, 8), dtype=np.float32)
        out[:, :34] = x[:, 112:146]
        return out
    with patch.object(threat_upgrade, 'recompute_extra_planes', side_effect=valid) as decode:
        first = buf._pad_x_planes(arrs)
        second = buf._pad_x_planes(arrs)
        assert first['x'][:, 146:].all()
        np.testing.assert_array_equal(first['x'], second['x'])
        assert decode.call_count == 2
    assert buf._upgrade_failures == 0
    assert not arrs['x'][:, 146:].any()


def test_disabled_path_has_no_fingerprint_or_decode_cost():
    buf, arrs = (_buffer(enabled=False), _chunk())
    with patch.object(UpgradeFailureCache, 'fingerprint', side_effect=AssertionError('fingerprint')):
        assert buf._pad_x_planes(arrs) is arrs


def test_wider_stored_data_still_raises():
    with pytest.raises(ValueError, match='refusing to truncate'):
        _buffer()._pad_x_planes(_chunk(179))


def test_cross_version_failure_preserves_prefix_and_padding():
    buf, arrs = (_buffer(), _chunk(146))
    with patch.object(threat_upgrade, 'recompute_extra_planes', side_effect=_recomputed) as decode:
        for _ in range(2):
            out = buf._pad_x_planes(arrs)
            np.testing.assert_array_equal(out['x'][:, :146], arrs['x'])
            assert not out['x'][:, 146:].any()
        assert decode.call_count == 1


def test_general_version_validation_failure_is_cached():
    buf, arrs = (_buffer(179), _chunk(179))
    with patch.object(threat_upgrade, 'recompute_extra_planes', side_effect=_recomputed) as decode:
        buf._pad_x_planes(arrs)
        buf._pad_x_planes(arrs)
        assert decode.call_count == 1


def test_noncontiguous_equivalent_content_hits_same_cache():
    arrs = _chunk()
    noncontiguous = np.asfortranarray(arrs['x'])
    assert not noncontiguous.flags.c_contiguous
    assert UpgradeFailureCache.fingerprint(arrs, 175, 'v2_threats') == (
        UpgradeFailureCache.fingerprint({**arrs, 'x': noncontiguous}, 175, 'v2_threats')
    )


def test_dtype_shape_target_and_version_are_part_of_identity():
    arrs = _chunk()
    original = UpgradeFailureCache.fingerprint(arrs, 175, 'v2_threats')
    variants = [
        ({**arrs, 'x': arrs['x'].astype(np.float32)}, 175, 'v2_threats'),
        ({**arrs, 'x': arrs['x'][:1]}, 175, 'v2_threats'),
        (arrs, 179, 'v3_checks'),
        (arrs, 175, 'different-decoder-version'),
    ]
    for chunk, target, version in variants:
        assert original != UpgradeFailureCache.fingerprint(chunk, target, version)


def test_unsupported_fingerprint_does_not_memoize():
    arrs = _chunk()
    arrs['x'] = arrs['x'].astype(object)
    assert UpgradeFailureCache.fingerprint(arrs, 175, 'v2_threats') is None
    cache = UpgradeFailureCache()
    cache.remember(None)
    assert not cache.contains(None)


def test_unknown_or_oversized_history_retries_without_cache():
    buf, arrs = (_buffer(), _chunk())
    arrs['_input_history_encoding'] = np.asarray('x' * 257)
    with patch.object(disk_buffer, 'upgrade_arrays_to_planes', side_effect=ValueError('history')) as decode:
        buf._pad_x_planes(arrs)
        buf._pad_x_planes(arrs)
        assert decode.call_count == 2


def test_lru_is_bounded_and_eviction_allows_retry():
    cache = UpgradeFailureCache(max_entries=2)
    cache.remember(b'one')
    cache.remember(b'two')
    assert cache.contains(b'one')
    cache.remember(b'three')
    assert not cache.contains(b'two')
    assert cache.contains(b'one')
    assert cache.contains(b'three')


def test_zero_rows_can_be_fingerprinted():
    arrs = _chunk()
    arrs['x'] = arrs['x'][:0]
    assert isinstance(UpgradeFailureCache.fingerprint(arrs, 175, 'v2_threats'), bytes)


def _exercise_add_many_arrays(planes: int, expected_decodes: int):
    buf, arrs = _buffer(), _chunk(planes)
    arrs['policy_target'] = np.zeros((2, 8), dtype=np.float16)
    arrs['priority'] = np.ones(2, dtype=np.float32)
    arrs['wdl_target'] = np.zeros(2, dtype=np.int8)
    buf._shuffle_buf = deque()
    buf._shuffle_sizes = deque()
    buf._shuffle_offsets = deque()
    buf._shuffle_size_total = 0
    buf._shuffle_priority_store = np.zeros(0, dtype=np.float32)
    buf._shuffle_wdl_store = np.zeros(0, dtype=np.int8)
    buf._write_buf_rows = 0
    buf._shard_size = 1000
    with ExitStack() as stack:
        for name in ('_reject_if_read_only', '_claim_writer', '_trim_shuffle_buf',
                     '_accumulate_priority_mass', '_enforce_window'):
            stack.enter_context(patch.object(DiskReplayBuffer, name, create=True))
        stack.enter_context(patch.object(DiskReplayBuffer, '_shape_shuffle_priority',
                                        side_effect=lambda _a, pri: pri, create=True))
        write = stack.enter_context(patch.object(DiskReplayBuffer, '_append_write_arrays', create=True))
        stack.enter_context(patch.object(disk_buffer, 'prune_storage_arrays', side_effect=lambda a: a, create=True))
        stack.enter_context(patch.object(disk_buffer, 'sparsify_chunk', side_effect=lambda a: a, create=True))
        decode = stack.enter_context(patch.object(threat_upgrade, 'recompute_extra_planes', side_effect=_recomputed))
        buf.add_many_arrays(arrs)
        buf.add_many_arrays(arrs)
        assert decode.call_count == expected_decodes
        assert buf._upgrade_failures == 4
        assert buf._shuffle_size_total == 4
        assert write.call_count == 2
        for stored in buf._shuffle_buf:
            assert stored['x'].shape == (2, 175, 8, 8)
            np.testing.assert_array_equal(stored['x'][:, :planes], arrs['x'])
            assert not stored['x'][:, 146:].any()
        np.testing.assert_array_equal(write.call_args.args[0]['x'][:, :planes], arrs['x'])


def test_add_many_arrays_reuses_failure_for_hot_append_without_dropping_rows():
    _exercise_add_many_arrays(175, expected_decodes=1)


def test_narrow_add_many_arrays_retries_once_for_the_padded_identity():
    # Padding146 to175 changes the actual content key; both identities are
    # memoized independently. This is not one decode for every add call.
    _exercise_add_many_arrays(146, expected_decodes=2)


def test_native_equal_width_and_dropout_chunks_skip_failure_hashing():
    for width in (175, 179, 183):
        for dropout in (False, True):
            arrs = _chunk(width)
            if dropout:
                arrs['x'][:] = 0
            else:
                arrs['x'][:, 146] = 1
            arrs['x_lc0_root'] = arrs['x'].copy()
            buffer = _buffer(width)
            with patch.object(buffer._upgrade_failure_cache, 'fingerprint', side_effect=AssertionError('no healthy hash')):
                assert buffer._pad_x_planes(arrs) is arrs
            assert buffer._upgrade_failures == 0


def test_root_only_zero_padding_still_checks_and_memoizes_validation_failure():
    arrs = _chunk()
    arrs['x_lc0_root'] = arrs['x'].copy()
    arrs['x'][:, 146] = 1
    buffer = _buffer()
    with patch.object(threat_upgrade, 'recompute_extra_planes', side_effect=_recomputed) as decode:
        for _ in range(3):
            assert buffer._pad_x_planes(arrs) is arrs
    assert decode.call_count == 1
    assert buffer._upgrade_failures == 3


def test_equal_width_malformed_root_shape_retains_value_error_fallback():
    arrs = _chunk()
    arrs['x_lc0_root'] = np.zeros((3, 175, 8, 8), dtype=np.float16)
    buffer = _buffer()
    with patch.object(buffer._upgrade_failure_cache, 'fingerprint', side_effect=AssertionError('no invalid-shape hash')):
        assert buffer._pad_x_planes(arrs) is arrs
    assert buffer._upgrade_failures == 1
