"""Small checkpoint/manifest contracts; no compilation, search or full chess model."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig
from native.bend_engine.neural_probe import checkpoint as cp
from native.bend_engine.neural_probe.backend import CHECKPOINT_FORMAT, execution_spec
from native.bend_engine.neural_probe.checkpoint_probe import tolerances


def payload() -> dict:
    cfg = ModelConfig(kind='tiny', input_history_encoding='lc0_root', history_rep_fix=False)
    return {'arch': {'_schema_version': ARCH_SCHEMA_VERSION, **asdict(cfg)},
            'model': {'weight': torch.ones(2, 2), 'bias': torch.ones(2)}, 'step': 7}


def test_explicit_saved_weights_and_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cp, 'build_model', lambda cfg: torch.nn.Linear(2, 2))
    data = payload()
    data['swa_model'] = {k: v * 2 for k, v in data['model'].items()}
    path = tmp_path / 'trainer.pt'
    torch.save(data, path)
    normal = cp.load_checkpoint(tmp_path)
    swa = cp.load_checkpoint(path, weights_key='swa_model')
    assert torch.equal(normal.model.state_dict()['weight'], torch.ones(2, 2))
    assert torch.equal(swa.model.state_dict()['weight'], torch.full((2, 2), 2.0))
    assert normal.identity['checkpoint_sha256'] == swa.identity['checkpoint_sha256']
    assert normal.identity['weights_key'] != swa.identity['weights_key']
    assert normal.identity['checkpoint_step'] == 7


@pytest.mark.parametrize('change', ['arch', 'schema', 'future', 'unknown', 'encoding', 'missing', 'extra', 'shape', 'nan', 'empty', 'swa'])
def test_strict_checkpoint_rejects_ambiguous_or_incomplete(monkeypatch, tmp_path: Path, change: str) -> None:
    monkeypatch.setattr(cp, 'build_model', lambda cfg: torch.nn.Linear(2, 2))
    data = payload()
    if change == 'arch':
        del data['arch']
    elif change == 'schema':
        data['arch']['_schema_version'] = True
    elif change == 'future':
        data['arch']['_schema_version'] = ARCH_SCHEMA_VERSION + 1
    elif change == 'unknown':
        data['arch']['unsupported_feature'] = True
    elif change == 'encoding':
        del data['arch']['input_history_encoding']
    elif change == 'missing':
        del data['model']['bias']
    elif change == 'extra':
        data['model']['unexpected'] = torch.zeros(1)
    elif change == 'shape':
        data['model']['weight'] = torch.ones(3, 3)
    elif change == 'nan':
        data['model']['weight'][0, 0] = float('nan')
    elif change == 'empty':
        data['model'] = {}
    path = tmp_path / 'trainer.pt'
    torch.save(data, path)
    with pytest.raises((ValueError, RuntimeError), match=r'arch|metadata|state|nonfinite|size|key'):
        cp.load_checkpoint(path, weights_key='swa_model' if change == 'swa' else 'model')


def test_conflicting_tied_weights_are_not_silently_overwritten(monkeypatch, tmp_path: Path) -> None:
    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(2, 2)
            self.b = self.a
    monkeypatch.setattr(cp, 'build_model', lambda cfg: Shared())
    data = payload()
    data['model'] = {k: v.clone() for k, v in Shared().state_dict().items()}
    data['model']['b.weight'] += 1
    path = tmp_path / 'trainer.pt'
    torch.save(data, path)
    with pytest.raises(ValueError, match='alias/dtype conflict'):
        cp.load_checkpoint(path)


@pytest.mark.parametrize('name', ['policy', 'policy_own'])
def test_real_policy_alias_is_used_instead_of_auxiliary_heads(name: str) -> None:
    class Output(torch.nn.Module):
        def forward(self, x):
            return {name: x, 'policy_sf': -x, 'wdl': x + 1}
    p, w = cp.OutputTuple(Output())(torch.ones(1, 3))
    assert torch.equal(p, torch.ones(1, 3))
    assert torch.equal(w, torch.full((1, 3), 2.0))


def spec() -> dict[str, object]:
    return {'format': CHECKPOINT_FORMAT, 'device': 'cpu', 'dtype': 'float32',
            'device_index': 0, 'checkpoint_sha256': 'a' * 64, 'weights_key': 'model'}


@pytest.mark.parametrize(('key', 'value'), [('device', 'auto'), ('dtype', 'bfloat16'), ('device_index', -1),
                                          ('device_index', 1), ('device_index', True), ('checkpoint_sha256', 'x' * 64),
                                          ('checkpoint_sha256', None), ('weights_key', 'guess')])
def test_manifest_does_not_guess_execution_contract(key: str, value: object) -> None:
    data = spec()
    data[key] = value
    with pytest.raises(ValueError, match=r'contract|index|fingerprint|key'):
        execution_spec(data)


def test_old_and_explicit_device_contracts() -> None:
    assert execution_spec({'format': 'old'}) == ('cpu', 'float32', 0)
    assert execution_spec(spec()) == ('cpu', 'float32', 0)
    data = spec()
    data.update(device='cuda', dtype='bfloat16', device_index=1)
    assert execution_spec(data) == ('cuda', 'bfloat16', 1)


def test_cuda_never_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='no CPU fallback'):
        cp.target('cuda', 0)
    with pytest.raises(ValueError, match='explicit predeclared'):
        tolerances('cuda', None, None)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1])
def test_tolerances_are_validated(value: float) -> None:
    with pytest.raises(ValueError, match='finite and nonnegative'):
        tolerances('cpu', value, 0)


@pytest.mark.parametrize('batch', [1, 2, 4, 8, 16])
def test_group_limits_follow_package_bucket(batch: int) -> None:
    from native.bend_engine.neural_probe.batch_probe import group_limits
    limit, capacity = group_limits(batch, None)
    assert limit == batch
    assert batch <= capacity <= 16
    assert group_limits(batch, 1) == (1, capacity)


@pytest.mark.parametrize(('batch', 'rows'), [(3, None), (True, None), (1, 4), (4, 0), (4, True)])
def test_group_limits_reject_invalid_requests(batch: int, rows: int | None) -> None:
    from native.bend_engine.neural_probe.batch_probe import group_limits
    with pytest.raises(ValueError, match=r'group batch|row limit'):
        group_limits(batch, rows)
