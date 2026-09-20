from __future__ import annotations

import json

import numpy as np
import pytest
import zarr

from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer
from scripts import lc0_control_train as train
from scripts.lc0_data_to_rows import shard_dir_search_wdl_coverage
from tests.test_lc0_control_drivers import _tiny_config, _write_game_rows
from tests.test_packed_zarr_epoch import pack


def argv(tmp_path, roots):
    return [
        '--config', str(_tiny_config(tmp_path)), '--shards', *map(str, roots),
        '--out-dir', str(tmp_path / 'run'), '--steps', '0', '--batch-size', '4',
        '--sampling-mode', 'game_epoch', '--allow-packed-zarr',
        '--device', 'cpu', '--no-compile', '--allow-arch-drift',
        '--allow-invalid-control',
    ]


def source(root):
    return _write_game_rows(root, [(i, i) for i in range(8)])


def zipped(tmp_path, attrs=None):
    src = source(tmp_path / 'source')
    shard = src / 'shard_000000.zarr'
    if attrs:
        zarr.open_group(str(shard), mode='a').attrs.update(attrs)
    target = tmp_path / 'packed'
    target.mkdir()
    pack(shard, target / (shard.name + '.zip'))
    return target


def test_staging_mixed_roster_preserves_resolved_source_partitions(tmp_path):
    first = source(tmp_path / 'first')
    other = source(tmp_path / 'other')
    packed = zipped(tmp_path)
    # Directory and ZIP can coexist, provided their shard indices differ.
    (first / 'shard_000001.zarr.zip').symlink_to(packed / 'shard_000000.zarr.zip')
    staged = tmp_path / 'stage'
    assert train.stage_shards([first, other], staged, allow_packed_zarr=True) == 3
    assert [p.name for p in sorted(staged.iterdir())] == [
        'shard_000000.zarr', 'shard_000001.zarr.zip', 'shard_000002.zarr',
    ]
    assert [p.resolve().parent for p in sorted(staged.iterdir())] == [first, packed, other]
    with pytest.raises(ValueError, match='require --allow-packed-zarr'):
        train.stage_shards([first], tmp_path / 'refused')
    with pytest.raises(ValueError, match='require --allow-packed-zarr'):
        train.read_value_stamps([first])
    (first / 'shard_000000.zarr.zip').symlink_to(packed / 'shard_000000.zarr.zip')
    with pytest.raises(ValueError, match='duplicate shard index'):
        train.stage_shards([first], tmp_path / 'duplicate', allow_packed_zarr=True)


@pytest.mark.parametrize('extra', [
    ['--sampling-mode', 'replacement'],
    ['--overlay-storage-qualification', '/absent'],
    ['--expected-overlay-storage-qualification-sha256', '0' * 64],
])
def test_cli_refuses_unsupported_combinations_before_loading(tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        train.main(argv(tmp_path, [tmp_path / 'absent']) + extra)
    assert exc.value.code == 2


@pytest.mark.parametrize(('attrs', 'pattern'), [
    ({'derive_corpus_row_schema': 3, 'zero_history': False}, 'mixes input-history'),
    ({'derive_value_scheme': 'qz50'}, 'value'),
    ({'corpus_complete': False}, 'PARTIAL corpus'),
])
def test_real_preflight_reads_and_refuses_bad_zip_stamps(tmp_path, attrs, pattern):
    packed = zipped(tmp_path, attrs)
    plain = source(tmp_path / 'plain')
    with pytest.raises(SystemExit, match=pattern):
        train.main(argv(tmp_path, [plain, packed]))
    assert not (tmp_path / 'run').exists()


def test_zip_coverage_stays_lazy_and_reaches_preflight(tmp_path, monkeypatch):
    packed = zipped(tmp_path)
    original = zarr.Array.__getitem__

    def guarded(self, selection):
        assert self.path not in ('x', 'policy', 'policy_target'), self.path
        return original(self, selection)

    monkeypatch.setattr(zarr.Array, '__getitem__', guarded)
    assert shard_dir_search_wdl_coverage(packed, allow_packed_zarr=True) == (8, 8)
    # An absent label on one packed row must be observed, not defaulted away.
    raw = tmp_path / 'source' / 'shard_000000.zarr'
    group = zarr.open_group(str(raw), mode='a')
    group['has_search_wdl'][0] = 0
    (packed / 'shard_000000.zarr.zip').unlink()
    pack(raw, packed / 'shard_000000.zarr.zip')
    assert shard_dir_search_wdl_coverage(packed, allow_packed_zarr=True) == (7, 8)
    with pytest.raises(SystemExit, match='PARTIALLY search_wdl-labelled'):
        train.main(argv(tmp_path, [packed]))


def test_real_cpu_train_preserves_two_epoch_packed_opt_in_and_receipt(tmp_path, monkeypatch):
    packed = zipped(tmp_path)
    plain = source(tmp_path / 'plain')
    calls = []

    class ObservedBuffer(GameAwareEpochBuffer):
        def __init__(self, **kwargs):
            calls.append(kwargs.copy())
            super().__init__(**kwargs)

    monkeypatch.setattr(train, 'GameAwareEpochBuffer', ObservedBuffer)
    assert train.main([*argv(tmp_path, [plain, packed]), '--epochs', '2']) == 0
    assert len(calls) == 2
    assert all(c['allow_packed_zarr'] is True for c in calls)
    assert calls[1]['seed'] == calls[0]['seed'] + 1
    summary = json.loads((tmp_path / 'run' / 'summary.json').read_text())
    assert summary['realized_replay_after_guard']['applied']['allow_packed_zarr'] is True
    assert summary['sampling']['complete'] is True
    assert summary['corpus']['label_coverage']['search_wdl'] == {'labelled_rows': 16, 'rows': 16}
    assert summary['steps_realized'] == 8
    assert all(c['shard_dir'] == tmp_path / 'run' / 'staged_shards' for c in calls)
    assert np.isfinite(summary['metrics']['train_time_s'])
