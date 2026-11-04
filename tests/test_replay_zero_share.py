"""T5: zero sharing is a no-read/no-write no-op, not full-shard ingest."""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from chess_anti_engine.replay import DiskReplayBuffer
from chess_anti_engine.tune import replay_exchange as exchange


def _share(fraction, buf, root):
    return exchange._share_top_replay_each_iteration(
        config={}, recipient_trial_dir=root, replay_shard_dir=root,
        buf=cast(DiskReplayBuffer, buf), top_k_trials=1, within_best_frac=.05,
        min_metric=0., source_skip_newest=0, shard_size=6, holdout_fraction=0.,
        max_unseen_iters_per_source=1, share_fraction=fraction,
    )


def test_zero_sharing_does_not_discover_or_touch_donors(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('zero sharing must not discover/read/write or ingest')

    for name in ['_select_top_trial_snapshots', '_load_import_state',
                 '_load_shard_arrays_with_retry', 'atomic_write_text']:
        monkeypatch.setattr(exchange, name, forbidden)
    buf = SimpleNamespace(add_many_arrays=forbidden)
    result = _share(0., buf, tmp_path)
    assert result == dict.fromkeys([
        'source_trials_selected', 'source_trials_ingested',
        'source_trials_skipped_repeat', 'source_shards_loaded',
        'source_samples_ingested'], 0)


def test_zero_direct_ingest_is_no_read_and_preserves_counters(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('zero fraction must not load a shard')

    monkeypatch.setattr(exchange, '_load_shard_arrays_with_retry', forbidden)
    summary = {'source_shards_loaded': 2, 'source_samples_ingested': 7}
    assert not exchange._ingest_one_shard(
        Path('unused'), buf=cast(DiskReplayBuffer, object()),
        share_fraction=0., summary=summary,
    )
    assert summary == {'source_shards_loaded': 2, 'source_samples_ingested': 7}


@pytest.mark.parametrize(('fraction', 'count'), [(1., 6), (.5, 3), (.01, 1)])
def test_positive_share_preserves_ingest_and_progress(tmp_path, monkeypatch, fraction, count):
    arrays = {'x': np.arange(6).reshape(6, 1), 'priority': np.ones(6)}
    received = []
    written = []
    buf = SimpleNamespace(rng=np.random.default_rng(4), add_many_arrays=received.append)
    monkeypatch.setattr(exchange, '_select_top_trial_snapshots',
                        lambda **_: [{'trial_dir': str(tmp_path), 'iter': 3}])
    monkeypatch.setattr(exchange, '_load_import_state', lambda _: {})
    monkeypatch.setattr(exchange, '_all_trial_snapshots', lambda _: [{'iter': 3}])
    monkeypatch.setattr(exchange, '_resolve_source_shard_dir', lambda *a, **kw: (tmp_path, True))
    monkeypatch.setattr(exchange, '_collect_iter_shards', lambda **_: [Path('synthetic')])
    monkeypatch.setattr(exchange, '_load_shard_arrays_with_retry', lambda *a, **kw: (arrays, {}))
    monkeypatch.setattr(exchange, 'atomic_write_text', lambda *args: written.append(args))
    result = _share(fraction, buf, tmp_path)
    assert result['source_trials_selected'] == 1
    assert result['source_trials_ingested'] == 1
    assert result['source_shards_loaded'] == 1
    assert result['source_samples_ingested'] == count
    assert len(received) == 1
    assert len(received[0]['x']) == count
    assert len(set(received[0]['x'].ravel().tolist())) == count
    assert len(written) == 1
    assert json.loads(written[0][1]) == {str(tmp_path.resolve()): 3}
