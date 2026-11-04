"""BO3 retention contract, using a deliberately scaled tiny initial ceiling."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chess_anti_engine.replay import ReplayBuffer, ReplaySample
from scripts import train_bootstrap as bootstrap


@pytest.mark.parametrize('max_positions', [0, 1, 4, 10])
def test_cli_retains_every_requested_row_before_training(tmp_path, monkeypatch, max_positions):
    for name in ['1', '2', '3']:
        (tmp_path / f'{name}.npz').touch()
    loaded_ids = []
    observed = []
    saved = []

    def load(path):
        shard = int(Path(path).stem)
        ids = [2 * shard, 2 * shard + 1]
        loaded_ids.extend(ids)
        policy = np.zeros(1858, dtype=np.float32)
        policy[0] = 1
        return [ReplaySample(x=np.zeros((146, 8, 8), dtype=np.float32),
                             policy_target=policy, wdl_target=0, game_id=i)
                for i in ids], {}

    def buffer(capacity, **kwargs):
        # Exercise real storage/eviction, without allocating >2M samples.
        return ReplayBuffer(3 if capacity == 2_000_000 else capacity, **kwargs)

    def train(buf, **kwargs):
        expected = loaded_ids if max_positions == 0 else loaded_ids[:max_positions]
        actual = np.concatenate([chunk['game_id'] for chunk in buf._chunks]).tolist()
        assert actual == expected
        assert len(buf) == len(expected)
        assert kwargs['steps'] == len(expected)
        if max_positions:
            assert buf.capacity == max_positions
        observed.extend(actual)
        return SimpleNamespace(loss=0., wdl_loss=0., moves_left_loss=0.)

    monkeypatch.setattr(bootstrap, 'load_yaml_file', lambda _: {'device': 'cpu'})
    monkeypatch.setattr(bootstrap, 'build_model', lambda _: object())
    monkeypatch.setattr(bootstrap, 'Trainer', lambda *a, **kw: SimpleNamespace(
        train_steps=train, save=lambda _: saved.append(True)))
    monkeypatch.setattr(bootstrap, 'ReplayBuffer', buffer)
    monkeypatch.setattr(bootstrap, 'load_npz', load)
    monkeypatch.setattr(sys, 'argv', ['bootstrap', '--config', 'unused',
        '--bootstrap-dir', str(tmp_path), '--out', str(tmp_path / 'unused.pt'),
        '--loaders', '1', '--batch-size', '1', '--max-positions', str(max_positions)])
    bootstrap.main()
    assert len(observed) == (6 if max_positions == 0 else min(6, max_positions))
    assert saved == [True]
