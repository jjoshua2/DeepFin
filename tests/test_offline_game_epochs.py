"""Small real training passes qualify uninterrupted multi-epoch orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from scripts import lc0_control_train as driver
from tests.test_lc0_control_drivers import _tiny_config, _write_game_rows


def arguments(tmp_path: Path, output: Path, *, epochs: int = 2) -> list[str]:
    shards = _write_game_rows(tmp_path / 'rows', [(game, game) for game in range(20)])
    return [
        '--config', str(_tiny_config(tmp_path)), '--shards', str(shards),
        '--out-dir', str(output), '--steps', '0', '--batch-size', '4',
        '--sampling-mode', 'game_epoch', '--epochs', str(epochs),
        '--epoch-plan-workers', '2', '--epoch-load-workers', '2',
        '--train-window-steps', '2', '--device', 'cpu', '--no-compile',
        '--allow-arch-drift', '--allow-invalid-control',
    ]


def test_two_epochs_use_every_row_and_keep_optimizer_and_rng_continuity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[int, list[int]] = {}
    identities: list[tuple[int, int, int, int]] = []
    sample = driver.GameAwareEpochBuffer.sample_batch_arrays
    train = driver.Trainer.train_steps

    def capture_rows(self: Any, *args: Any, **kwargs: Any) -> Any:
        batch = sample(self, *args, **kwargs)
        seen.setdefault(self.plan.seed, []).extend(np.asarray(batch['game_id']).tolist())
        return batch

    def capture_state(self: Any, buf: Any, **kwargs: Any) -> Any:
        identities.append((id(self), id(self.opt), id(buf.rng), self.step))
        return train(self, buf, **kwargs)

    monkeypatch.setattr(driver.GameAwareEpochBuffer, 'sample_batch_arrays', capture_rows)
    monkeypatch.setattr(driver.Trainer, 'train_steps', capture_state)
    out = tmp_path / 'two'
    assert driver.main(arguments(tmp_path, out)) == 0
    result = json.loads((out / 'summary.json').read_text())
    assert sorted(seen) == [0, 1]
    assert sorted(seen[0]) == sorted(seen[1]) == list(range(20))
    assert seen[0] != seen[1]
    assert len({entry[:3] for entry in identities}) == 1
    assert [entry[3] for entry in identities] == [0, 2, 4, 5, 7, 9]
    assert result['steps'] == result['steps_realized'] == 10
    assert result['train_windows'] == 6
    assert [row['steps_requested'] for row in result['train_window_metrics']] == [2, 2, 1, 2, 2, 1]
    sampling = result['sampling']
    assert sampling['mode'] == 'game_epochs'
    assert sampling['complete'] is True
    assert sampling['rows_realized'] == 40
    assert sampling['batches_realized'] == 10
    assert [record['sampling']['seed'] for record in sampling['epochs']] == [0, 1]
    assert all(record['sampling']['complete'] for record in sampling['epochs'])
    assert all(record['sampling']['plan_sha256'] == record['sampling']['realized_sha256']
               for record in sampling['epochs'])
    assert result['valid_control'] is False
    assert result['mid_checkpoint']['step'] == 5
    assert result['mid_checkpoint']['step'] in [row['steps_cumulative'] for row in result['train_window_metrics']]
    assert not any('actual train-window endpoint' in reason for reason in result['validity_problems'])
    assert {entry['role'] for entry in result['checkpoints']} == {'mid', 'last', 'epoch1'}
    first = torch.load(out / 'checkpoint_epoch1.pt', map_location='cpu', weights_only=False)
    final = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert first['step'] == 5
    assert final['step'] == 10
    assert not (out / 'checkpoint_epoch1.pending.pt').exists()

    # The new default still follows the same first pass, including its short window.
    single = tmp_path / 'one'
    argv = arguments(tmp_path, single, epochs=1)
    index = argv.index('--epochs')
    del argv[index:index + 2]
    assert driver.main(argv) == 0
    standalone = torch.load(single / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert all(torch.equal(value, standalone['model'][key]) for key, value in first['model'].items())
    assert not (single / 'checkpoint_epoch1.pt').exists()
    assert json.loads((single / 'summary.json').read_text())['sampling']['mode'] == 'game_epoch'


def test_incomplete_second_epoch_cannot_publish_final_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = driver.GameAwareEpochBuffer.sample_batch_arrays

    def fail_second(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self.plan.seed == 1:
            raise RuntimeError('injected second epoch failure')
        return sample(self, *args, **kwargs)

    monkeypatch.setattr(driver.GameAwareEpochBuffer, 'sample_batch_arrays', fail_second)
    out = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='injected second epoch'):
        driver.main(arguments(tmp_path, out))
    assert not (out / 'summary.json').exists()
    assert not (out / 'checkpoint.pt').exists()
    assert not (out / 'checkpoint_epoch1.pt').exists()
    assert not (out / 'checkpoint_mid.pt').exists()
    assert (out / 'checkpoint_epoch1.pending.pt').exists()
    assert 'checkpoint_epoch1.pending.pt' in driver.existing_run_artifacts(out)


@pytest.mark.parametrize('extra', [
    ['--epochs', '0'], ['--epochs', '2'],
    ['--epochs', '2', '--sampling-mode', 'game_epoch'],
])
def test_invalid_multi_epoch_cli_refuses_before_data_or_output(
    tmp_path: Path, extra: list[str],
) -> None:
    out = tmp_path / 'absent'
    with pytest.raises(SystemExit) as failure:
        driver.main(['--shards', str(tmp_path / 'missing'), '--out-dir', str(out),
                     '--steps', '10', *extra])
    assert failure.value.code == 2
    assert not out.exists()


def test_changed_corpus_between_epochs_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zarr

    save = driver.Trainer.save
    out = tmp_path / 'changed'
    argv = arguments(tmp_path, out)

    def change_after_first(self: Any, path: Path) -> None:
        save(self, path)
        if path.name == 'checkpoint_epoch1.pending.pt':
            group = zarr.open_group(str(tmp_path / 'rows/shard_000000.zarr'), mode='r+')
            group['x'][0, 0, 0, 0] = 0.125

    monkeypatch.setattr(driver.Trainer, 'save', change_after_first)
    with pytest.raises(RuntimeError, match='corpus or batch count changed'):
        driver.main(argv)
    assert not (out / 'checkpoint.pt').exists()
    assert not (out / 'checkpoint_epoch1.pt').exists()
    assert not (out / 'summary.json').exists()


def test_fewer_optimizer_steps_than_consumed_batches_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = driver.Trainer.train_steps

    def underreport(self: Any, *args: Any, **kwargs: Any) -> Any:
        metrics = train(self, *args, **kwargs)
        metrics.train_steps_done -= 1
        return metrics

    monkeypatch.setattr(driver.Trainer, 'train_steps', underreport)
    out = tmp_path / 'missing_step'
    with pytest.raises(RuntimeError, match='optimizer window incomplete'):
        driver.main(arguments(tmp_path, out))
    assert not (out / 'summary.json').exists()
    assert not (out / 'checkpoint.pt').exists()
    assert not (out / 'checkpoint_epoch1.pending.pt').exists()


@pytest.mark.parametrize(('epochs', 'fraction', 'expected'), [(2, 0.5, 36935), (3, 2 / 3, 73870)])
def test_ragged_real_corpus_midpoint_uses_epoch_window_endpoints(
    epochs: int, fraction: float, expected: int,
) -> None:
    endpoints = driver.game_epoch_window_endpoints(epoch_steps=36935, epochs=epochs, window=88)
    mid = driver.mid_step_on_epoch_boundary(endpoints=endpoints, frac=fraction)
    assert mid == expected
    assert mid in endpoints[:-1]
    assert mid % 88 != 0


@pytest.mark.parametrize(('mid', 'off_boundary'), [(36935, False), (36960, True)])
def test_validity_uses_real_epoch_endpoints_instead_of_global_modulo(
    mid: int, off_boundary: bool,
) -> None:
    problems = driver.control_validity_problems(
        allow_leak=False, allow_arch_drift=False, has_purity_receipt=True,
        steps=73870, warmup_steps=1000, mid_saved_at_step=mid,
        mid_checkpoint_frac=0.5, window_steps=88, cadence_problem=None,
        device='cuda', configured_device='cuda', batch_size=512,
        configured_batch_size=512, live_config_unread=False, live_game_frac=0.0,
        window_endpoints=driver.game_epoch_window_endpoints(epoch_steps=36935, epochs=2, window=88),
    )
    assert any('actual train-window endpoint' in reason for reason in problems) == off_boundary
