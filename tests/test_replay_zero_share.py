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


# n=6 except where stated. .43 -> n*f = 2.58: round() gives 3, int() gives 2.
# n=5, f=.5 -> 2.5 and n=7, f=.5 -> 3.5 pin half-to-even (2 and 4): a
# round-half-up rule would give 3 and 4, int() 2 and 3.
@pytest.mark.parametrize(('fraction', 'count', 'n'), [
    (1., 6, 6), (.5, 3, 6), (.01, 1, 6), (.43, 3, 6), (.5, 2, 5), (.5, 4, 7),
])
def test_positive_share_preserves_ingest_and_progress(tmp_path, monkeypatch, fraction, count, n):
    arrays = {'x': np.arange(n).reshape(n, 1), 'priority': np.ones(n)}
    received = []
    written = []
    buf = SimpleNamespace(rng=np.random.default_rng(4), add_many_arrays=received.append)
    monkeypatch.setattr(exchange, '_select_top_trial_snapshots',
                        lambda **_: [{'trial_dir': str(tmp_path), 'iter': 3}])
    monkeypatch.setattr(exchange, '_load_import_state', lambda _: {})
    monkeypatch.setattr(exchange, '_all_trial_snapshots', lambda _: [{'iter': 3}])
    monkeypatch.setattr(exchange, '_resolve_source_shard_dir', lambda *_a, **_kw: (tmp_path, True))
    monkeypatch.setattr(exchange, '_collect_iter_shards', lambda **_: [Path('synthetic')])
    monkeypatch.setattr(exchange, '_load_shard_arrays_with_retry', lambda *_a, **_kw: (arrays, {}))
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


BAD_FRACTIONS = [-.5, -1e-9, float('nan'), float('inf'), -float('inf'), 1.5, 1.0000001, True, False]


@pytest.mark.parametrize('fraction', BAD_FRACTIONS, ids=repr)
def test_out_of_range_share_fails_closed_without_touching_donors(
    tmp_path, monkeypatch, capsys, fraction,
):
    """A value that skipped the loader (restored config, direct caller) shares
    NOTHING and says so; it neither ingests the whole shard nor raises."""
    def forbidden(*_args, **_kwargs):
        pytest.fail('an out-of-range fraction must not discover/read/ingest')

    for name in ['_select_top_trial_snapshots', '_load_import_state',
                 '_load_shard_arrays_with_retry', 'atomic_write_text']:
        monkeypatch.setattr(exchange, name, forbidden)
    buf = SimpleNamespace(add_many_arrays=forbidden)
    result = _share(fraction, buf, tmp_path)
    assert result['source_samples_ingested'] == 0
    assert result['source_shards_loaded'] == 0
    if fraction is not False:  # False == 0.0: the quiet "share off" path
        assert 'replay share DISABLED' in capsys.readouterr().out


@pytest.mark.parametrize('fraction', BAD_FRACTIONS, ids=repr)
def test_out_of_range_direct_ingest_reads_nothing(monkeypatch, fraction):
    def forbidden(*_args, **_kwargs):
        pytest.fail('an out-of-range fraction must not load a shard')

    monkeypatch.setattr(exchange, '_load_shard_arrays_with_retry', forbidden)
    summary = {'source_shards_loaded': 0, 'source_samples_ingested': 0}
    assert not exchange._ingest_one_shard(
        Path('unused'), buf=cast(DiskReplayBuffer, object()),
        share_fraction=fraction, summary=summary,
    )
    assert summary == {'source_shards_loaded': 0, 'source_samples_ingested': 0}


# ── loader: launch is fatal, a live reload is rejected whole ──────────────────

_LOADER_BAD = [-0.5, 1.5, float('nan'), float('inf'), True, False, '-0.5', 'nan', '1e3', 'abc', [0.5]]


@pytest.mark.parametrize('value', _LOADER_BAD, ids=repr)
def test_loader_refuses_bad_share_fraction_at_launch(value):
    from chess_anti_engine.utils import flatten_run_config_defaults

    with pytest.raises(ValueError, match='exploit_replay_share_fraction'):
        flatten_run_config_defaults({'tune': {'exploit_replay_share_fraction': value}})


@pytest.mark.parametrize(('value', 'realized'), [
    (0, 0.0), (0.0, 0.0), (1, 1.0), (0.5, 0.5), ('0.5', 0.5), ('5e-1', 0.5),
])
def test_loader_accepts_in_range_share_fraction(value, realized):
    from chess_anti_engine.tune.trial_config import TrialConfig
    from chess_anti_engine.utils import flatten_run_config_defaults

    flat = flatten_run_config_defaults({'tune': {'exploit_replay_share_fraction': value}})
    assert TrialConfig.from_dict(flat).exploit_replay_share_fraction == realized


@pytest.mark.parametrize('bad', ['-0.5', '.nan', '1.5', 'true'])
def test_live_reload_with_bad_share_fraction_keeps_previous_config(tmp_path, caplog, bad):
    import logging

    from chess_anti_engine.tune.trainable_config_ops import _reload_yaml_into_config

    yaml_path = tmp_path / 'live.yaml'
    # A concurrent legitimate edit rides along: the WHOLE reload is rejected.
    yaml_path.write_text(
        f'tune:\n  exploit_replay_share_fraction: {bad}\n'
        f'  exploit_replay_top_k_trials: 9\n', encoding='utf-8')
    config: dict = {'exploit_replay_share_fraction': 0.5, 'exploit_replay_top_k_trials': 5}
    with caplog.at_level(logging.WARNING):
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert config == {'exploit_replay_share_fraction': 0.5, 'exploit_replay_top_k_trials': 5}
    assert 'YAML reload failed' in caplog.text
    assert 'exploit_replay_share_fraction' in caplog.text

    # Positive control: the same reload with a valid value IS applied.
    yaml_path.write_text(
        'tune:\n  exploit_replay_share_fraction: 0.25\n'
        '  exploit_replay_top_k_trials: 9\n', encoding='utf-8')
    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert config['exploit_replay_share_fraction'] == 0.25
    assert config['exploit_replay_top_k_trials'] == 9
