import json

import pytest
import torch

from chess_anti_engine.utils.atomic import atomic_write
from scripts import bootstrap_recovery as recovery


class TinyTrainer:
    def __init__(self):
        self.model = torch.nn.Linear(2, 1)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, 2)
        self.step = 0

    def update(self):
        self.opt.zero_grad()
        self.model(torch.ones(1, 2)).sum().backward()
        self.opt.step()
        self.scheduler.step()
        self.step += 1

    def save(self, path):
        state = {'model': self.model.state_dict(), 'opt': self.opt.state_dict(),
                 'scheduler': self.scheduler.state_dict(), 'step': self.step,
                 'peak_lr': .01, 'zclip': {'count': self.step}}
        atomic_write(path, lambda p: torch.save(state, p))


def save(rolling, trainer):
    return rolling.maybe_save(trainer, progress={'window_steps': 1, 'epoch_index': 1},
                             metrics={'train_steps_done': 1, 'train_samples_seen': 1})


def test_full_state_round_trip_and_hourly_retention(tmp_path):
    now = [0.0]
    rolling = recovery.RollingRecoveryCheckpoints(tmp_path, clock=lambda: now[0])
    trainer = TinyTrainer()
    trainer.update()
    first = save(rolling, trainer)
    rng = torch.get_rng_state().clone()
    now[0] = 3599
    assert save(rolling, trainer) is None
    assert torch.equal(torch.get_rng_state(), rng)
    for time_value in (3600, 7200):
        now[0] = time_value
        trainer.update()
        latest = save(rolling, trainer)
    assert not first.exists()
    snapshots = json.loads((tmp_path / 'latest.json').read_text())['snapshots']
    assert len(snapshots) == 2
    state = torch.load(latest / 'checkpoint.pt', weights_only=False)
    restored = TinyTrainer()
    restored.model.load_state_dict(state['model'])
    restored.opt.load_state_dict(state['opt'])
    restored.scheduler.load_state_dict(state['scheduler'])
    assert restored.scheduler.state_dict() == trainer.scheduler.state_dict()
    assert state['step'] == 3
    assert state['zclip']['count'] == 3
    for key in state['model']:
        assert torch.equal(restored.model.state_dict()[key], trainer.model.state_dict()[key])
    trainer.update()
    restored.step = 3
    restored.update()
    for actual, expected in zip(restored.model.parameters(), trainer.model.parameters()):
        assert torch.equal(actual, expected)  # Adam moments and LR really restored.
    assert torch.load(latest / 'rng.pt', weights_only=False)['torch_cpu'].dtype == torch.uint8
    manifest = json.loads((latest / 'manifest.json').read_text())
    assert 'fresh explicitly seeded' in manifest['resume_semantics']


@pytest.mark.parametrize("after_publication", [False, True])
def test_failed_commit_preserves_previous_good_snapshot(tmp_path, monkeypatch, after_publication):
    now = [0.0]
    rolling = recovery.RollingRecoveryCheckpoints(tmp_path, clock=lambda: now[0])
    trainer = TinyTrainer()
    trainer.update()
    first = save(rolling, trainer)
    original_index = (tmp_path / 'latest.json').read_bytes()
    original_checkpoint = (first / 'checkpoint.pt').read_bytes()
    original_write = recovery.atomic_write
    def fail_index(path, writer, **kwargs):
        if path.name == 'latest.json':
            if after_publication:
                original_write(path, writer, **kwargs)
            raise OSError('disk full')
        return original_write(path, writer, **kwargs)
    monkeypatch.setattr(recovery, 'atomic_write', fail_index)
    now[0] = 3600
    trainer.update()
    with pytest.raises(OSError, match='disk full'):
        save(rolling, trainer)
    if not after_publication:
        assert (tmp_path / 'latest.json').read_bytes() == original_index
    for name in json.loads((tmp_path / 'latest.json').read_text())['snapshots']:
        assert (tmp_path / name / 'checkpoint.pt').is_file()
    assert (first / 'checkpoint.pt').read_bytes() == original_checkpoint
    assert not list(tmp_path.glob('.writing-*'))


def test_invalid_window_is_not_a_recovery_point(tmp_path):
    rolling = recovery.RollingRecoveryCheckpoints(tmp_path)
    trainer = TinyTrainer()
    assert rolling.maybe_save(trainer, progress={'window_steps': 1},
                              metrics={'train_steps_done': 1, 'loss': float('nan')}) is None
    assert not (tmp_path / 'latest.json').exists()
