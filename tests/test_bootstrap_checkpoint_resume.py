from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from chess_anti_engine.utils import sha256_file
from scripts.bootstrap_checkpoint_resume import resume_bootstrap


def donor(tmp_path: Path) -> tuple[Any, Path, dict[str, Any]]:
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2)
    model(torch.ones(1, 2)).sum().backward()
    opt.step()
    scheduler.step()
    state = {'model': model.state_dict(), 'opt': opt.state_dict(),
             'scheduler': scheduler.state_dict(), 'step': 9, 'peak_lr': 0.01,
             'zclip': {'count': 9}, 'opt_param_names': ['weight', 'bias']}
    path = tmp_path / 'donor.pt'
    torch.save(state, path)
    target = torch.nn.Linear(2, 1)
    restored = torch.optim.AdamW(target.parameters(), lr=0.01)
    schedule = torch.optim.lr_scheduler.StepLR(restored, step_size=2)
    trainer = SimpleNamespace(model=target, opt=restored, _scheduler=schedule,
                              step=0, _peak_lr=0.01, zclip_state_dict=lambda: {'count': 9},
                              _swa_model=None,
                              _optimizer_param_names=lambda: ['weight', 'bias'],
                              _donor_optimizer_param_names=lambda checkpoint: checkpoint.get('opt_param_names'))
    def load(_path: Path, *, exact_resume: bool = False) -> None:
        assert exact_resume
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
    def bad_load(p: Path, *, exact_resume: bool = False) -> None:
        load(p, exact_resume=exact_resume)
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
    assert summary['continuation']['additional_steps'] == initial * 3
    assert [row['sampling']['seed'] for row in summary['sampling']['epochs']] == [102, 103, 104]
    assert {entry['role'] for entry in summary['checkpoints']} == {'mid', 'last', 'epoch1', 'epoch2'}
    for entry in summary['checkpoints']:
        assert sha256_file(Path(entry['path'])) == entry['sha256']
    assert summary['additional_epoch_checkpoints'][0]['sha256'] == sha256_file(second / 'checkpoint_epoch2.pt')
    assert sha256_file(path) == summary['continuation']['sha256']


@pytest.mark.parametrize('missing', ['model', 'opt', 'scheduler', 'step', 'peak_lr', 'zclip'])
def test_incomplete_checkpoint_is_refused(tmp_path: Path, missing: str) -> None:
    trainer, path, state = donor(tmp_path)
    del state[missing]
    torch.save(state, path)
    with pytest.raises(ValueError, match='complete checkpoint state'):
        resume_bootstrap(trainer, path, sha256_file(path), 9, 102)


@pytest.mark.parametrize('component', ['model', 'scheduler', 'zclip', 'step', 'peak_lr'])
def test_partial_restore_is_refused(tmp_path: Path, component: str) -> None:
    trainer, path, _ = donor(tmp_path)
    load = trainer.load

    def partial_load(p: Path, *, exact_resume: bool = False) -> None:
        load(p, exact_resume=exact_resume)
        if component == 'model':
            with torch.no_grad():
                trainer.model.weight.add_(1)
        elif component == 'scheduler':
            trainer._scheduler.last_epoch += 1
        elif component == 'zclip':
            trainer.zclip_state_dict = lambda: {'count': 0}
        elif component == 'step':
            trainer.step = 0
        else:
            trainer._peak_lr = 0.5

    trainer.load = partial_load
    with pytest.raises(ValueError, match=component):
        resume_bootstrap(trainer, path, sha256_file(path), 9, 102)


def test_same_shaped_optimizer_slots_must_keep_their_names(tmp_path: Path) -> None:
    trainer, path, state = donor(tmp_path)
    state['opt_param_names'] = ['bias', 'weight']
    torch.save(state, path)
    with pytest.raises(ValueError, match='optimizer parameter identities'):
        resume_bootstrap(trainer, path, sha256_file(path), 9, 102)
    assert trainer.step == 0


def test_changed_checkpoint_during_restore_is_refused(tmp_path: Path) -> None:
    trainer, path, state = donor(tmp_path)
    load = trainer.load

    def changed_load(p: Path, *, exact_resume: bool = False) -> None:
        load(p, exact_resume=exact_resume)
        torch.save({**state, 'changed': True}, p)

    trainer.load = changed_load
    with pytest.raises(ValueError, match='changed while loading'):
        resume_bootstrap(trainer, path, sha256_file(path), 9, 102)


@pytest.mark.parametrize('extra', [
    ['--resume-step', '9'],
    ['--resume-checkpoint-sha256', '0' * 64],
    ['--resume-checkpoint', 'missing.pt'],
    ['--resume-checkpoint', 'missing.pt', '--resume-checkpoint-sha256', 'bad',
     '--resume-step', '9', '--sampling-mode', 'game_epoch'],
    ['--resume-checkpoint', 'missing.pt', '--resume-checkpoint-sha256', '0' * 64,
     '--resume-step', '0', '--sampling-mode', 'game_epoch'],
    ['--resume-checkpoint', 'missing.pt', '--resume-checkpoint-sha256', '0' * 64,
     '--resume-step', '9'],
])
def test_invalid_resume_cli_refuses_before_output(tmp_path: Path, extra: list[str]) -> None:
    from scripts import lc0_control_train as driver

    out = tmp_path / 'absent'
    with pytest.raises(SystemExit) as failure:
        driver.main(['--shards', str(tmp_path / 'missing'), '--out-dir', str(out),
                     '--steps', '0', *extra])
    assert failure.value.code == 2
    assert not out.exists()


@pytest.mark.parametrize('name', ['checkpoint_epoch2.pt', 'checkpoint_epoch3.pending.pt'])
def test_every_epoch_artifact_blocks_output_reuse(tmp_path: Path, name: str) -> None:
    from scripts import lc0_control_train as driver

    (tmp_path / name).touch()
    assert name in driver.existing_run_artifacts(tmp_path)
