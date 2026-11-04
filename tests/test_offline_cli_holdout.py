"""BO1: ordinary CLI routes independent eval selection without training."""
import sys
from pathlib import Path

import pytest

from scripts import offline_replay_epoch as epoch


@pytest.mark.parametrize(
    ('extra', 'expected'),
    [
        ([], ['t0', 't1', 't2']),
        (['--eval-replay-dir', 'holdout'], ['h0', 'h1', 'h2']),
        (['--eval-max-shards', '1'], ['t0']),
        (['--eval-max-shards', '1', '--eval-newest-shards'], ['t2']),
        (['--eval-replay-dir', 'holdout', '--eval-max-shards', '2'], ['h0', 'h1']),
        (['--eval-replay-dir', 'holdout', '--eval-max-shards', '1',
          '--eval-newest-shards'], ['h2']),
    ],
)
def test_ordinary_cli_routes_eval_selection(tmp_path, monkeypatch, extra, expected):
    pools = {'train': list(map(Path, ['t0', 't1', 't2'])),
             'holdout': list(map(Path, ['h0', 'h1', 'h2']))}
    monkeypatch.setattr(epoch, 'load_yaml_file', lambda _: {})
    monkeypatch.setattr(epoch, 'iter_shard_paths', lambda directory: pools[str(directory)])
    selected = []
    marker = {}

    def load_eval(**kwargs):
        selected.extend(str(p) for p in kwargs['shard_paths'])
        return marker

    def train(**kwargs):
        assert kwargs['shard_paths'] == [Path('t2')]
        assert kwargs['eval_arrs'] is marker
        return {'candidate': kwargs['candidate']}

    monkeypatch.setattr(epoch, '_load_eval_arrs', load_eval)
    monkeypatch.setattr(epoch, '_train_candidate', train)
    monkeypatch.setattr(sys, 'argv', ['offline', '--replay-dir', 'train',
        '--out-dir', str(tmp_path / 'out'), '--candidates', 'adamw',
        '--max-shards', '1', '--newest-shards', *extra])
    epoch.main()
    assert selected == expected


def test_empty_explicit_holdout_does_not_fall_back_to_training(tmp_path, monkeypatch):
    monkeypatch.setattr(epoch, 'load_yaml_file', lambda _: {})
    monkeypatch.setattr(epoch, 'iter_shard_paths',
                        lambda directory: [Path('train-shard')] if directory == 'train' else [])
    monkeypatch.setattr(epoch, '_train_candidate',
                        lambda **_: pytest.fail('training must not start without eval data'))
    monkeypatch.setattr(epoch, 'load_shard_arrays',
                        lambda *a, **kw: pytest.fail('must not read training shard for holdout'))
    monkeypatch.setattr(sys, 'argv', ['offline', '--replay-dir', 'train',
        '--eval-replay-dir', 'empty', '--out-dir', str(tmp_path / 'out'),
        '--candidates', 'adamw'])
    with pytest.raises(ValueError, match='eval'):
        epoch.main()
