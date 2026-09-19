"""Strict same-architecture bootstrap continuation, with explicit new epoch RNG."""
from pathlib import Path
from typing import Any

import torch

from chess_anti_engine.train.trainer import strip_compile_prefix
from chess_anti_engine.utils import sha256_file


def require_equal(actual: Any, expected: Any, name: str) -> None:
    if isinstance(expected, torch.Tensor):
        same = isinstance(actual, torch.Tensor) and actual.shape == expected.shape and actual.dtype == expected.dtype and torch.equal(actual.detach().cpu(), expected.detach().cpu())
    elif isinstance(expected, dict):
        same = isinstance(actual, dict) and actual.keys() == expected.keys()
        if same:
            for key in expected:
                require_equal(actual[key], expected[key], f'{name}.{key}')
    elif isinstance(expected, (list, tuple)):
        same = isinstance(actual, type(expected)) and len(actual) == len(expected)
        if same:
            for index, value in enumerate(expected):
                require_equal(actual[index], value, f'{name}[{index}]')
    else:
        same = actual == expected
    if not same:
        raise ValueError(f'checkpoint continuation did not restore {name} exactly')


def resume_bootstrap(trainer: Any, path: Path, expected_sha256: str, expected_step: int, epoch_seed: int) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError('resume checkpoint hash mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    required = {'model', 'opt', 'scheduler', 'step', 'peak_lr', 'zclip'}
    if not required <= checkpoint.keys() or checkpoint['step'] != expected_step or expected_step <= 0:
        raise ValueError('resume requires complete checkpoint state and matching positive step')
    current = strip_compile_prefix(trainer.model.state_dict())
    if current.keys() != checkpoint['model'].keys() or any(current[k].shape != v.shape or current[k].dtype != v.dtype for k, v in checkpoint['model'].items()):
        raise ValueError('resume requires identical model keys, shapes and dtypes')
    trainer.load(path)
    require_equal(strip_compile_prefix(trainer.model.state_dict()), checkpoint['model'], 'model')
    require_equal(trainer.opt.state_dict(), checkpoint['opt'], 'optimizer')
    require_equal(trainer._scheduler.state_dict(), checkpoint['scheduler'], 'scheduler')
    require_equal(trainer.zclip_state_dict(), checkpoint['zclip'], 'zclip')
    require_equal(trainer.step, expected_step, 'step')
    require_equal(trainer._peak_lr, checkpoint['peak_lr'], 'peak_lr')
    if sha256_file(path) != expected_sha256:
        raise ValueError('resume checkpoint changed while loading')
    # Trainer.save does not promise a complete RNG snapshot. This is an explicit
    # new epoch order, not a bitwise reproduction of an uninterrupted process.
    torch.manual_seed(epoch_seed)
    return {'checkpoint': str(path.resolve()), 'sha256': expected_sha256,
            'step_start': expected_step, 'epoch_seed_start': epoch_seed,
            'restored': ['model', 'optimizer', 'scheduler', 'step', 'peak_lr', 'zclip'],
            'rng': 'Torch reset to explicit epoch seed; sampler uses successive epoch seeds; uninterrupted RNG replay not claimed'}
