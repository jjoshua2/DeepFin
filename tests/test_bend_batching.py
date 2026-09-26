"""Deterministic broker contracts. No models, compiler, search or sleeping."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from native.bend_engine.neural_probe.batching import Batcher, Key


def prepared(count: int = 2, *, capacity: int = 8) -> Batcher:
    b = Batcher(4, 146, capacity=capacity, max_wait=2)
    for i in range(count):
        b.register(i, 1)
        b.submit(Key(i, 1, 1, 0), np.full((1, 146, 8, 8), i + 1, dtype=np.float32),
                 np.array([0, 1]), now=0, deadline=10)
    return b


def outputs() -> tuple[np.ndarray, np.ndarray]:
    p = np.zeros((4, 1858), dtype=np.float32)
    w = np.zeros((4, 3), dtype=np.float32)
    for i in range(4):
        w[i, 0] = i
    return p, w


def test_partial_flush_and_padding_have_no_phantom_completions() -> None:
    b = prepared()
    assert b.dispatch(1.999) is None
    batch = b.dispatch(2)
    assert batch is not None
    np.testing.assert_array_equal(batch.x[:, 0, 0, 0], [1, 2, 0, 0])
    replies = b.complete(batch, *outputs(), now=3)
    assert [r.key.session for r in replies] == [0, 1]
    assert replies[0].wdl != replies[1].wdl
    assert b.reserved == 0
    assert not b.pending


def test_full_batch_does_not_wait() -> None:
    b = prepared(4)
    batch = b.dispatch(0)
    assert batch is not None
    assert len(batch.jobs) == 4
    assert b.dispatch(100) is None


def test_backpressure_counts_queued_and_cancelled_inflight_rows() -> None:
    b = prepared(4, capacity=4)
    batch = b.dispatch(0)
    assert batch is not None
    b.cancel(Key(0, 1, 1, 0))
    b.register(0, 2)
    with pytest.raises(BufferError, match='capacity'):
        b.submit(Key(0, 2, 1, 0), np.zeros((1, 146, 8, 8)), np.array([0]), now=1, deadline=10)
    b.complete(batch, *outputs(), now=2)
    b.submit(Key(0, 2, 1, 0), np.zeros((1, 146, 8, 8)), np.array([0]), now=2, deadline=10)
    assert b.reserved == 1


def test_late_cancelled_row_cannot_touch_restarted_epoch() -> None:
    b = prepared()
    old = b.dispatch(2)
    assert old is not None
    assert b.cancel(Key(0, 1, 1, 0)).status == 'cancelled'
    b.register(0, 2)
    key = Key(0, 2, 1, 0)  # node/request deliberately reused in the newer epoch
    b.submit(key, np.zeros((1, 146, 8, 8)), np.array([0]), now=3, deadline=10)
    replies = b.complete(old, *outputs(), now=4)
    assert [r.key.session for r in replies] == [1]
    assert b.pending[0].key == key
    with pytest.raises(ValueError, match=r'stale|duplicate'):
        b.complete(old, *outputs(), now=4)


@pytest.mark.parametrize('inflight', [False, True])
def test_expiration_and_cancellation_do_not_reappear(inflight: bool) -> None:
    b = prepared()
    batch = b.dispatch(2) if inflight else None
    assert len(b.expire(9.99)) == 0
    expired = b.expire(10)
    assert len(expired) == 2
    assert all(r.status == 'expired' for r in expired)
    if batch is not None:
        assert b.complete(batch, *outputs(), now=11) == []
    assert not b.pending
    assert b.reserved == 0


def test_expiration_is_checked_again_on_scatter() -> None:
    b = prepared()
    batch = b.dispatch(2)
    assert batch is not None
    replies = b.complete(batch, *outputs(), now=10)
    assert [r.status for r in replies] == ['expired', 'expired']


@pytest.mark.parametrize('bad', ['shape', 'nan', 'infinity'])
def test_bad_output_is_atomic_then_fails_all_active_rows(bad: str) -> None:
    b = prepared()
    batch = b.dispatch(2)
    assert batch is not None
    p, w = outputs()
    if bad == 'shape':
        p = p[:3]
    else:
        w[1, 0] = float(bad if bad == 'nan' else 'inf')
    with pytest.raises(ValueError, match='batched model'):
        b.complete(batch, p, w, now=3)
    assert len(b.pending) == 2
    assert b.flight is batch
    assert [r.status for r in b.fail(batch)] == ['backend_error', 'backend_error']
    assert not b.pending


def test_foreign_batch_cannot_consume_current_reservation() -> None:
    b = prepared()
    batch = b.dispatch(2)
    assert batch is not None
    with pytest.raises(ValueError, match='foreign'):
        b.complete(replace(batch), *outputs(), now=3)
    assert b.flight is batch


def test_input_and_action_ownership() -> None:
    b = Batcher(4, 146)
    b.register(1, 1)
    x, actions = np.ones((1, 146, 8, 8)), np.array([0, 1])
    b.submit(Key(1, 1, 1, 0), x, actions, now=0, deadline=10)
    x[:] = 999
    actions[:] = 2
    batch = b.dispatch(1)
    assert batch is not None
    assert batch.x[0, 0, 0, 0] == 1
    np.testing.assert_array_equal(batch.jobs[0].actions, [0, 1])


@pytest.mark.parametrize('key', [Key(0, 1, 1, 0), Key(0, 2, 1, 0), Key(99, 1, 1, 0)])
def test_duplicate_stale_or_unregistered_request(key: Key) -> None:
    b = prepared(1)
    with pytest.raises(ValueError, match=r'identity|outstanding'):
        b.submit(key, np.zeros((1, 146, 8, 8)), np.array([0]), now=1, deadline=10)


def test_one_pending_request_per_session_even_with_new_sequence() -> None:
    b = prepared(1)
    with pytest.raises(ValueError, match='outstanding'):
        b.submit(Key(0, 1, 2, 1), np.zeros((1, 146, 8, 8)), np.array([0]), now=1, deadline=10)
    with pytest.raises(ValueError, match='outstanding'):
        b.register(0, 2)


@pytest.mark.parametrize('batch', [0, 3, 32, True])
def test_invalid_batch(batch: int) -> None:
    with pytest.raises(ValueError, match='batch'):
        Batcher(batch, 146)


def test_session_registry_is_bounded() -> None:
    b = Batcher(4, 146, max_sessions=1)
    b.register(0, 1)
    with pytest.raises(BufferError, match='session capacity'):
        b.register(1, 1)
    with pytest.raises(ValueError, match='epoch'):
        b.register(0, 1)


@pytest.mark.parametrize('deadline', [0, -1, float('nan'), float('inf')])
def test_bad_deadlines(deadline: float) -> None:
    b = Batcher(4, 146)
    b.register(0, 1)
    with pytest.raises(ValueError, match='deadline'):
        b.submit(Key(0, 1, 1, 0), np.zeros((1, 146, 8, 8)), np.array([0]), now=0, deadline=deadline)


def test_fifo_survives_cancellation_and_full_batch() -> None:
    b = prepared(5)
    b.cancel(Key(1, 1, 1, 0))
    batch = b.dispatch(0)
    assert batch is not None
    assert [p.key.session for p in batch.jobs] == [0, 2, 3, 4]


def test_batch_package_manifest_never_reinterprets_v1(tmp_path) -> None:
    import hashlib
    import json
    import torch
    from native.bend_engine.neural_probe.backend import BATCH_FORMAT, FORMAT, package_manifest
    p = tmp_path / 'dummy.pt2'
    p.write_bytes(b'parser only, never executed')
    data = {'format': BATCH_FORMAT, 'batch': 4, 'row_independent': True,
            'policy_width': 1858, 'channels': 146, 'input_history_encoding': 'lc0_root',
            'input_extra_features': 'v1', 'history_rep_fix': True,
            'torch_version': str(torch.__version__), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    p.with_suffix('.json').write_text(json.dumps(data))
    assert package_manifest(p)[0]['batch'] == 4
    data['format'] = FORMAT
    p.with_suffix('.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='batch one'):
        package_manifest(p)
    data['format'], data['row_independent'] = BATCH_FORMAT, False
    p.with_suffix('.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='independent rows'):
        package_manifest(p)
