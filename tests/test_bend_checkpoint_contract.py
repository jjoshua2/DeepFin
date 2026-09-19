"""Small checkpoint/manifest guards. No AOT compilation, inference or search."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig
from chess_anti_engine.moves.encode import COMPACT_TO_FULL_POLICY
from native.bend_engine.neural_probe import checkpoint as ck
from native.bend_engine.neural_probe.backend import package_manifest
from native.bend_engine.neural_probe.checkpoint_probe import compare_probabilities


def architecture() -> dict:
    return {**asdict(ModelConfig(kind='tiny', input_history_encoding='lc0_root',
                                input_extra_features='v1', history_rep_fix=True)),
            '_schema_version': ARCH_SCHEMA_VERSION}


@pytest.mark.parametrize(('device', 'dtype'), [('cuda', 'bfloat16'), ('cpu', 'bfloat16'),
                                         ('cuda:00', 'float32'), ('cuda:-1', 'float32'),
                                         ('mps', 'float32'), ('cpu', 'float16')])
def test_execution_contract_rejects_ambiguous_targets(device: str, dtype: str) -> None:
    with pytest.raises(ValueError, match=r'device|execution'):
        ck.device_contract(device, dtype)


def test_explicit_cuda_never_silently_becomes_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='no CPU fallback'):
        ck.execution_manifest('cuda:0', 'bfloat16')


@pytest.mark.parametrize('change', ['missing', 'newer', 'unknown', 'legacy', 'relations', 'policy'])
def test_checkpoint_architecture_fails_closed(change: str) -> None:
    arch = architecture()
    if change == 'missing':
        del arch['input_history_encoding']
    elif change == 'newer':
        arch['_schema_version'] = ARCH_SCHEMA_VERSION + 1
    elif change == 'unknown':
        arch['invented_topology_field'] = True
    elif change == 'legacy':
        arch['input_history_encoding'] = 'legacy'
    elif change == 'relations':
        arch['policy_dynamic_relations'] = True
    else:
        arch['policy_encoding'] = 'invented'
    with pytest.raises(ValueError, match=r'identity|newer|unknown|history|relation|policy'):
        ck.checkpoint_config(arch)


def test_checkpoint_identity_and_inference_only_override() -> None:
    arch = architecture()
    arch['use_gradient_checkpointing'] = True
    cfg, encoding = ck.checkpoint_config(arch)
    assert cfg.use_gradient_checkpointing is False
    assert encoding.input_history_encoding == 'lc0_root'
    assert encoding.channels == 146
    assert encoding.history_rep_fix is True
    assert arch['use_gradient_checkpointing'] is True


def test_exact_saved_weights_not_fresh_initialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = torch.nn.Linear(2, 3)
    state = {k: torch.full_like(v, 0.25) for k, v in module.state_dict().items()}
    monkeypatch.setattr(ck, 'build_model', lambda cfg: torch.nn.Linear(2, 3))
    path = tmp_path / 'model.pt'
    torch.save({'arch': architecture(), 'model': state}, path)
    loaded = ck.load_checkpoint(path)
    for key, value in loaded.model.state_dict().items():
        assert torch.equal(value, state[key])
    assert loaded.sha256 == ck.fingerprint(path)
    assert not loaded.model.training


@pytest.mark.parametrize('bad', ['bare', 'missing', 'extra', 'shape', 'nonfinite'])
def test_incomplete_weights_never_get_a_random_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    state = torch.nn.Linear(2, 3).state_dict()
    monkeypatch.setattr(ck, 'build_model', lambda cfg: torch.nn.Linear(2, 3))
    if bad == 'missing':
        state.pop('bias')
    elif bad == 'extra':
        state['extra'] = torch.zeros(1)
    elif bad == 'shape':
        state['weight'] = torch.zeros(3, 3)
    elif bad == 'nonfinite':
        state['weight'][0, 0] = float('nan')
    path = tmp_path / 'model.pt'
    torch.save(state if bad == 'bare' else {'arch': architecture(), 'model': state}, path)
    with pytest.raises((ValueError, RuntimeError), match=r'arch|Missing|Unexpected|size mismatch|nonfinite'):
        ck.load_checkpoint(path)


def manifest() -> dict:
    cfg, encoding = ck.checkpoint_config(architecture())
    model_config = asdict(cfg)
    return {'format': ck.CHECKPOINT_FORMAT, 'torch_version': str(torch.__version__),
            'model_config': model_config, 'model_config_sha256': ck.digest_json(model_config),
            'checkpoint_sha256': 'f' * 64, 'weights': 'checkpoint-supplied',
            'output_order': ['policy', 'wdl'], 'row_independent': True,
            'batch': 4, 'channels': 146, 'policy_width': 1858,
            **asdict(encoding), **ck.execution_manifest('cpu', 'float32')}


@pytest.mark.parametrize(('field', 'value'), [('device', 'cuda'), ('dtype', 'bfloat16'),
    ('model_config_sha256', '0'*64), ('checkpoint_sha256', ''), ('history_rep_fix', False),
    ('row_independent', False), ('output_order', ['wdl', 'policy']), ('weights', 'seeded-untrained'),
    ('cuda_capability', [9, 0]), ('cxx11_abi', 'yes'), ('torch_cuda', 'incompatible')])
def test_v3_manifest_rejects_wrong_identity_and_runtime(tmp_path: Path, field: str, value: object) -> None:
    path = tmp_path / 'fake.pt2'
    path.write_bytes(b'parser only; never executed')
    data = manifest()
    data['sha256'] = ck.fingerprint(path)
    path.with_suffix('.json').write_text(json.dumps(data))
    assert package_manifest(path)[1].channels == 146
    data[field] = value
    path.with_suffix('.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r'device|execution|fingerprint|encoding|contract|weights|capability|ABI'):
        package_manifest(path)


def test_probability_comparison_detects_wrong_head_and_row() -> None:
    actions = np.asarray(COMPACT_TO_FULL_POLICY[:3], dtype=np.int64)
    p, w = np.zeros((1, 1858)), np.zeros((1, 3))
    assert compare_probabilities((p, w), (p, w), actions) == (0.0, 0.0)
    bad_p, bad_w = p.copy(), w.copy()
    bad_p[0, 0], bad_w[0, 2] = 30, 30
    policy_tv, wdl_tv = compare_probabilities((bad_p, bad_w), (p, w), actions)
    assert policy_tv > 0.6
    assert wdl_tv > 0.6


def test_checkpoint_is_detected_changing_during_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'model.pt'
    torch.save({'arch': architecture(), 'model': torch.nn.Linear(2, 3).state_dict()}, path)
    real_load = torch.load
    def changing(*args, **kwargs):
        value = real_load(*args, **kwargs)
        with path.open('ab') as stream:
            stream.write(b'concurrent modification')
        return value
    monkeypatch.setattr(torch, 'load', changing)
    with pytest.raises(ValueError, match='changed while'):
        ck.load_checkpoint(path)
