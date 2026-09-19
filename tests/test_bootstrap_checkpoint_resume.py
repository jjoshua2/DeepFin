from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from chess_anti_engine.utils import sha256_file
from scripts.bootstrap_checkpoint_resume import resume_bootstrap


def donor(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2)
    model(torch.ones(1, 2)).sum().backward()
    opt.step()
    scheduler.step()
    state = {'model': model.state_dict(), 'opt': opt.state_dict(),
             'scheduler': scheduler.state_dict(), 'step': 9, 'peak_lr': 0.01,
             'zclip': {'count': 9}}
    path = tmp_path / 'donor.pt'
    torch.save(state, path)
    target = torch.nn.Linear(2, 1)
    restored = torch.optim.AdamW(target.parameters(), lr=0.01)
    schedule = torch.optim.lr_scheduler.StepLR(restored, step_size=2)
    trainer = SimpleNamespace(model=target, opt=restored, _scheduler=schedule,
                              step=0, _peak_lr=0.01, zclip_state_dict=lambda: {'count': 9})
    def load(_path):
        target.load_state_dict(state['model'])
        restored.load_state_dict(state['opt'])
        schedule.load_state_dict(state['scheduler'])
        trainer.step = 9
    trainer.load = load
    return trainer, path, state


def test_restores_moments_scheduler_step_and_new_rng(tmp_path):
    trainer, path, state = donor(tmp_path)
    result = resume_bootstrap(trainer, path, sha256_file(path), 9, 102)
    assert result['step_start'] == 9
    assert trainer.opt.state_dict()['state']
    assert trainer._scheduler.state_dict() == state['scheduler']
    assert torch.initial_seed() == 102


def test_rejects_silent_optimizer_fallback(tmp_path):
    trainer, path, _ = donor(tmp_path)
    load = trainer.load
    def bad_load(p):
        load(p)
        trainer.opt.state.clear()
    trainer.load = bad_load
    with pytest.raises(ValueError, match='optimizer'):
        resume_bootstrap(trainer, path, sha256_file(path), 9, 102)


def test_rejects_wrong_identity_and_architecture(tmp_path):
    trainer, path, _ = donor(tmp_path)
    with pytest.raises(ValueError, match='hash mismatch'):
        resume_bootstrap(trainer, path, '0'*64, 9, 102)
    with pytest.raises(ValueError, match='matching positive step'):
        resume_bootstrap(trainer, path, sha256_file(path), 8, 102)
    trainer.model = torch.nn.Linear(3, 1)
    with pytest.raises(ValueError, match='identical model'):
        resume_bootstrap(trainer, path, sha256_file(path), 9, 102)


def test_real_epoch_continuation_keeps_global_steps_and_each_boundary(tmp_path):
    import json
    from scripts import lc0_control_train as driver
    from tests.test_offline_game_epochs import arguments
    first = tmp_path / 'first'
    argv = [*arguments(tmp_path, first, epochs=1), '--seed', '101']
    assert driver.main(argv) == 0
    path = first / 'checkpoint.pt'
    initial = torch.load(path, map_location='cpu', weights_only=False)['step']
    second = tmp_path / 'continued'
    argv[argv.index('--out-dir') + 1] = str(second)
    argv[argv.index('--epochs') + 1] = '3'
    argv[argv.index('--seed') + 1] = '102'
    argv += ['--resume-checkpoint', str(path), '--resume-checkpoint-sha256', sha256_file(path), '--resume-step', str(initial)]
    assert driver.main(argv) == 0
    summary = json.loads((second / 'summary.json').read_text())
    assert summary['continuation']['step_start'] == initial
    assert summary['continuation']['step_end'] == initial + summary['steps_realized']
    checkpoints = ['checkpoint_epoch1.pt', 'checkpoint_epoch2.pt', 'checkpoint.pt']
    assert [torch.load(second / name, map_location='cpu', weights_only=False)['step'] for name in checkpoints] == [initial*2, initial*3, initial*4]
    assert not list(second.glob('*.pending.pt'))
