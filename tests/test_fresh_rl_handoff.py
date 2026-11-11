from __future__ import annotations

import copy
import hashlib
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model
from chess_anti_engine.train import Trainer, trainer_kwargs_from_config
from chess_anti_engine.run import _build_tune_config_dict
from chess_anti_engine.tune.distributed_runtime import _publish_distributed_trial_state
from chess_anti_engine.tune.fresh_rl_donor import MODE, load_fresh_donor
from chess_anti_engine.tune.harness import (
    _filter_param_space_against_tuner_pkl,
    _patch_experiment_state_for_resume,
    run_tune,
)
from chess_anti_engine.tune.startup_intent import (
    SCHEMA,
    StartupIntentError,
    contract_digest,
    enforce_startup,
)
from chess_anti_engine.tune.trainable import _build_trial_model_config
from chess_anti_engine.tune.trainable_config_ops import _reload_yaml_into_config
from chess_anti_engine.tune.trainable_init import _maybe_load_bootstrap
from chess_anti_engine.tune.trial_config import RestoreResult, TrialConfig
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults


def _assert_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert (left.dtype, left.shape) == (right.dtype, right.shape)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_equal(a, b)
    else:
        assert left == right


def _fixture(tmp_path: Path) -> tuple[dict, dict, Trainer]:
    model_cfg = ModelConfig(
        embed_dim=32, num_layers=1, num_heads=4, use_smolgen=False,
        input_history_encoding='lc0_root_legacy_meta', history_rep_fix=True,
        aux_policy_head_dim=16, policy_embedding_mode='linear',
        categorical_head_coupled=True,
    )
    arch = {'_schema_version': ARCH_SCHEMA_VERSION, **asdict(model_cfg)}
    source = build_model(model_cfg)
    payload = {'arch': arch, 'model': copy.deepcopy(source.state_dict()), 'step': 9000,
               'opt': {'must_not_restore': True}, 'scheduler': {'must_not_restore': True}}
    path = tmp_path / 'donor.pt'
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    settings = {'lr': 3e-6, 'optimizer': 'aurora', 'lr_schedule': 'sqrt_release',
                'batch_size': 2, 'accum_steps': 1}
    config = {**settings, **asdict(model_cfg), 'model': model_cfg.kind,
              'bootstrap_checkpoint': str(path), 'bootstrap_checkpoint_sha256': digest,
              'bootstrap_handoff_mode': MODE, 'bootstrap_model_contract': arch,
              'bootstrap_zero_policy_heads': False, 'bootstrap_reinit_volatility_heads': False,
              'rl_startup_intent': 'fresh_bootstrap', 'device': 'cpu',
              'warmup_steps': 8, 'warmup_lr_start': 3e-7, 'use_amp': False,
              'use_compile': False, 'aurora_cuda_graphs': False,
              'matrix_optimizer_scope': 'mlp_out', 'matrix_lr_multiplier': 20.0}
    config['rl_startup_contract'] = {
        'schema': SCHEMA, 'run_uuid': '11111111-1111-4111-8111-111111111111',
        'donor_sha256': digest, 'model_contract_sha256': contract_digest(arch), 'settings': settings,
    }
    trainer = Trainer(build_model(model_cfg), model_config=model_cfg,
                      **trainer_kwargs_from_config(config))
    return config, payload, trainer


def _load(config: dict, trainer: Trainer) -> None:
    load_fresh_donor(config=config, trainer=trainer, checkpoint=None,
                     restore=RestoreResult(), torch_module=torch, schema_version=ARCH_SCHEMA_VERSION)


def _rewrite_payload(config: dict, payload: dict) -> None:
    path = Path(config['bootstrap_checkpoint'])
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config['bootstrap_checkpoint_sha256'] = digest
    config['rl_startup_contract']['donor_sha256'] = digest


def test_config_to_trainer_bootstrap_and_first_publication(tmp_path: Path) -> None:
    config, payload, trainer = _fixture(tmp_path)
    # Exercise the YAML loader, TrialConfig and production model mapping together.
    architecture = dict(asdict(trainer._model_config))
    architecture['gradient_checkpointing'] = architecture.pop('use_gradient_checkpointing')
    history_rep_fix = architecture.pop('history_rep_fix')
    top_keys = {key: config[key] for key in (
        'rl_startup_intent', 'rl_startup_contract', 'bootstrap_handoff_mode',
        'bootstrap_checkpoint', 'bootstrap_checkpoint_sha256', 'bootstrap_model_contract',
        'bootstrap_zero_policy_heads', 'bootstrap_reinit_volatility_heads',
    )}
    train_keys = {*config['rl_startup_contract']['settings'], 'warmup_steps', 'warmup_lr_start',
                  'matrix_optimizer_scope', 'matrix_lr_multiplier'}
    raw = {'model': architecture, 'selfplay': {'history_rep_fix': history_rep_fix}, 'train': {
        key: config[key] for key in train_keys
    }, **top_keys}
    loaded = flatten_run_config_defaults(yaml.safe_load(yaml.safe_dump(raw)))
    parsed = Namespace(**(loaded | {
        'device': 'cpu', 'no_amp': True, 'work_dir': str(tmp_path / 'work'),
        'eval_mcts_simulations': 16, 'eval_sf_nodes': 1000, 'search_optimizer_choices': None,
    }))
    resolved = _build_tune_config_dict(parsed)
    tc = TrialConfig.from_dict(resolved)
    assert _build_trial_model_config(tc) == trainer._model_config
    model_cfg = _build_trial_model_config(tc)
    trainer = Trainer(build_model(model_cfg), model_config=model_cfg,
                      **trainer_kwargs_from_config(resolved))
    optimizer, scheduler = trainer.opt, trainer._scheduler
    before = copy.deepcopy((optimizer.state_dict(), scheduler.state_dict(), trainer.step))
    _maybe_load_bootstrap(tc=tc, trainer=trainer, device='cpu', ckpt=None, restore=RestoreResult())
    _assert_equal(trainer.model.state_dict(), payload['model'])
    _assert_equal((optimizer.state_dict(), scheduler.state_dict(), trainer.step), before)
    assert trainer.opt is optimizer
    assert trainer._scheduler is scheduler
    assert trainer.step == 0
    assert trainer._swa_model is not None
    _assert_equal(trainer._swa_model.module.state_dict(), payload['model'])
    _publish_distributed_trial_state(
        trainer=trainer, config=config, model_cfg=trainer._model_config,
        server_root=tmp_path / 'server', trial_id='fresh', training_iteration=0,
        trainer_step=0, sf_nodes=1000, mcts_simulations=16, wdl_regret=0.2,
    )
    published = torch.load(tmp_path / 'server/trials/fresh/publish/latest_model.pt',
                           map_location='cpu', weights_only=True)
    _assert_equal(published['model'], payload['model'])


@pytest.mark.parametrize('fault', ['digest', 'missing', 'shape', 'dtype', 'nan', 'inf',
                                  'negative_inf', 'arch', 'extra', 'non_tensor'])
def test_refusals_preserve_unpublished_recipient(tmp_path: Path, fault: str) -> None:
    config, payload, trainer = _fixture(tmp_path)
    original = copy.deepcopy(trainer.model.state_dict())
    key = next(iter(payload['model']))
    if fault == 'digest':
        config['bootstrap_checkpoint_sha256'] = '0' * 64
        config['rl_startup_contract']['donor_sha256'] = '0' * 64
    else:
        if fault == 'missing':
            payload['model'].pop(key)
        elif fault == 'extra':
            payload['model']['unknown'] = torch.zeros(1)
        elif fault == 'shape':
            payload['model'][key] = torch.zeros(1)
        elif fault == 'dtype':
            payload['model'][key] = payload['model'][key].double()
        elif fault == 'non_tensor':
            payload['model'][key] = [1, 2]
        elif fault == 'arch':
            payload['arch']['history_rep_fix'] = False
        else:
            number = {'nan': float('nan'), 'inf': float('inf'), 'negative_inf': -float('inf')}[fault]
            payload['model'][key].fill_(number)
        _rewrite_payload(config, payload)
    pattern = {'digest': 'SHA256', 'missing': 'keys', 'extra': 'keys', 'shape': 'shape',
               'dtype': 'dtype', 'arch': 'model contract|architecture'}.get(fault, 'non-tensor|nonfinite')
    with pytest.raises(ValueError, match=pattern):
        _load(config, trainer)
    _assert_equal(trainer.model.state_dict(), original)
    assert trainer.fresh_rl_donor_provenance is None


def test_missing_file_refuses_in_actual_bootstrap_consumer(tmp_path: Path) -> None:
    config, _, trainer = _fixture(tmp_path)
    Path(config['bootstrap_checkpoint']).unlink()
    with pytest.raises(FileNotFoundError):
        _maybe_load_bootstrap(tc=TrialConfig.from_dict(config), trainer=trainer,
                              device='cpu', ckpt=None, restore=RestoreResult())


def test_outer_compile_prefix_is_normalized(tmp_path: Path) -> None:
    config, payload, trainer = _fixture(tmp_path)
    payload['model'] = {'_orig_mod.' + key: value for key, value in payload['model'].items()}
    _rewrite_payload(config, payload)
    _load(config, trainer)
    _assert_equal(trainer.model.state_dict(), {key.removeprefix('_orig_mod.'): value
                                              for key, value in payload['model'].items()})


@pytest.mark.parametrize('conflicting', [False, True])
def test_tied_parameter_aliases_and_buffers(tmp_path: Path, conflicting: bool) -> None:
    config, payload, trainer = _fixture(tmp_path)
    tied = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.zeros(3))
    counter = torch.zeros(1, dtype=torch.int64)
    tied.register_parameter('first', parameter)
    tied.register_parameter('second', parameter)
    tied.register_buffer('counter', counter)
    trainer.model = tied
    payload['model'] = {'first': torch.tensor([1., 2., 3.]),
                        'second': torch.tensor([4., 5., 6.] if conflicting else [1., 2., 3.]),
                        'counter': torch.tensor([7], dtype=torch.int64)}
    _rewrite_payload(config, payload)
    if conflicting:
        with pytest.raises(ValueError, match='aliases disagree'):
            _load(config, trainer)
        assert torch.equal(parameter, torch.zeros(3))
        assert torch.equal(counter, torch.zeros(1, dtype=torch.int64))
    else:
        _load(config, trainer)
        assert tied.get_parameter('first') is tied.get_parameter('second')
        _assert_equal(tied.state_dict(), payload['model'])


def test_changing_file_is_rejected_before_recipient_mutation(tmp_path: Path, monkeypatch) -> None:
    config, _, trainer = _fixture(tmp_path)
    original = copy.deepcopy(trainer.model.state_dict())
    real_load = torch.load

    def changing_load(*args: Any, **kwargs: Any) -> Any:
        result = real_load(*args, **kwargs)
        with Path(config['bootstrap_checkpoint']).open('ab') as stream:
            stream.write(b'changed')
        return result

    monkeypatch.setattr(torch, 'load', changing_load)
    with pytest.raises(ValueError, match='changed during authenticated load'):
        _load(config, trainer)
    _assert_equal(trainer.model.state_dict(), original)


@pytest.mark.parametrize('conflict', ['resume', 'salvage', 'policy_reset', 'checkpoint', 'exact_restore'])
def test_fresh_start_conflicts_are_refused(tmp_path: Path, conflict: str) -> None:
    config, _, _ = _fixture(tmp_path)
    checkpoint = None
    if conflict == 'resume':
        config['resume'] = True
    elif conflict == 'salvage':
        config['salvage_seed_pool_dir'] = 'pool'
    elif conflict == 'policy_reset':
        config['bootstrap_zero_policy_heads'] = True
    elif conflict == 'checkpoint':
        checkpoint = object()
    else:
        config['rl_startup_intent'] = 'exact_restore'
    def validate() -> None:
        TrialConfig.from_dict(config)
        enforce_startup(config, checkpoint)
    with pytest.raises(ValueError, match=r'resume|salvage|must be false|checkpoint|restore'):
        validate()


def test_hot_reload_cannot_drop_strict_startup_contract(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    before = copy.deepcopy(config)
    yaml_path = tmp_path / 'config.yaml'
    yaml_path.write_text('train:\n  lr: 0.00001\n')
    with pytest.raises(StartupIntentError):
        _reload_yaml_into_config(config, str(yaml_path))
    assert config == before


@pytest.mark.parametrize('surface', ['driver', 'state_overlay', 'pickle_filter'])
def test_harness_refuses_strict_resume_before_io(tmp_path: Path, surface: str) -> None:
    config, _, _ = _fixture(tmp_path)
    state_file = tmp_path / 'state.json'
    state_file.write_text('unchanged saved state')

    def attempt_resume() -> None:
        if surface == 'driver':
            run_tune(base_config=config, work_dir=tmp_path, num_samples=1, resume=True)
        elif surface == 'state_overlay':
            _patch_experiment_state_for_resume(state_file=state_file, param_space=config)
        else:
            _filter_param_space_against_tuner_pkl(config, experiment_path=tmp_path)

    with pytest.raises(StartupIntentError, match='resume'):
        attempt_resume()
    assert state_file.read_text() == 'unchanged saved state'
    assert not (tmp_path / 'tune').exists()


@pytest.mark.parametrize('key', ['lr', 'optimizer', 'lr_schedule', 'batch_size', 'accum_steps'])
def test_bound_startup_settings_cannot_change(tmp_path: Path, key: str) -> None:
    config, _, _ = _fixture(tmp_path)
    config[key] = 'changed'
    with pytest.raises(StartupIntentError, match='changed or missing bound setting: ' + key):
        TrialConfig.from_dict(config)
