"""Two-epoch admission and completion; no candidate training or corpus scan."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.replay.game_epoch import GameEpochPlan, MIRROR_AUGMENTATION_BATCH_COPIES, _balanced_batch_rows
from scripts import bt4_two_epoch_train as train


def write(path: Path, value: Any) -> dict[str, str]:
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': train.owned.sha(path)}


def plan(seed: int) -> dict[str, Any]:
    # Use the actual producer's serialization, without reading a corpus.
    return GameEpochPlan(
        rows=train.ROWS, batches=train.BATCHES, full_batches=36699, ragged_batches=236,
        min_batch_rows=511, shard_count=train.SHARDS, source_count=1, game_count=train.GAMES,
        batch_size=512, policy_size=1858, input_history_encoding='lc0_root_legacy_meta', history_rep_fix=True,
        seed=seed, load_workers=2, max_working_set_bytes=12*1024**3, peak_working_set_bytes=10*1024**3,
        mirror_augmentation=True, mirror_working_set_batch_copies=MIRROR_AUGMENTATION_BATCH_COPIES, collation_working_set_batch_copies=3,
        validated_load_payload_copies=5, corpus_sha256='c'*64, objective_mask_weights=(('policy', 123.0),),
        plan_sha256=str(seed+1)*64, load_counts=np.array([], dtype=np.int64), batch_rows=np.array([], dtype=np.int64),
        resident_bytes_after_batch=np.array([], dtype=np.int64),
    ).as_dict()


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    m = {'schema': 1, 'scope': train.SCOPE, 'profile': 'B100', 'state': str(tmp_path/'state'), 'run': str(tmp_path/'run'),
             'training_seconds': 30000, 'plan_workers': 3, 'load_workers': 2, 'max_working_set_bytes': 12*1024**3,
             'runtime_qualification': {'path': str(tmp_path/'runtime.json'), 'sha256': 'a'*64},
             'preregistration': {'path': str(tmp_path/'prereg.json'), 'sha256': 'b'*64},
             'preparation': {'path': str(tmp_path/'prep.json'), 'sha256': 'd'*64},
             'launcher_sha256': 'e'*64, 'stage_helper_sha256': 'f'*64, 'recipe_helper_sha256': '0'*64}
    prep: dict[str, Any] = {'schema': 1, 'status': 'PASS_TWO_EPOCH_PREPARATION', 'profile': 'B100', 'corpus': str(tmp_path/'corpus'),
                'runtime_qualification': m['runtime_qualification'], 'preregistration': m['preregistration'],
                'plan_workers': 3, 'load_workers': 2, 'max_working_set_bytes': m['max_working_set_bytes'],
                'epochs': [{'epoch_index': i+1, 'source_logical_order_sha256': str(i+3)*64,
                             'corpus_logical_order_sha256': str(i+3)*64, 'source_plan_sha256': str(i+5)*64, 'plan': plan(i)} for i in range(2)]}
    (tmp_path/'corpus').mkdir()
    bank = Path(__file__).resolve().parents[1] / 'scratchpad/bt4_joint20/publication_20260908_followup/training-horizon-runtime/py310_fixture/test_two_epochs_use_every_row_0/two/summary.json'
    actual_window = json.loads(bank.read_text())['train_window_metrics'][0]
    assert type(actual_window['transient_cuda_retry_batches']) is float
    batch_rows = _balanced_batch_rows(rows=train.ROWS, batch_size=512, max_game_rows=100)
    windows, records = [], []
    count = (train.BATCHES+87)//88
    for epoch, entry in enumerate(prep['epochs']):
        for i in range(count):
            steps = min(88, train.BATCHES-i*88)
            samples = int(batch_rows[i*88:i*88+steps].sum())
            windows.append({'window_index': len(windows)+1, 'epoch_index': epoch+1, 'epoch_window_index': i+1,
                'steps_requested': steps, 'train_steps_done': steps, 'steps_cumulative': epoch*train.BATCHES+min((i+1)*88, train.BATCHES),
                'epoch_steps_cumulative': min((i+1)*88, train.BATCHES), 'transient_cuda_retry_batches': actual_window['transient_cuda_retry_batches'],
                'grad_nonfinite_skip_rate': 0.0, 'loss': 1., 'grad_norm_mean': 2., 'train_samples_seen': samples})
        records.append({'epoch_index': epoch+1, 'steps_start': epoch*train.BATCHES, 'steps_end': (epoch+1)*train.BATCHES,
            'window_count': count, 'sampling': dict(entry['plan'], complete=True, plan_workers=3, load_workers=2,
            rows_realized=train.ROWS, batches_realized=train.BATCHES, decoded_rows_resident=0, decoded_bytes_resident=0,
            same_game_repeats_max=0, realized_sha256=entry['plan']['plan_sha256'], peak_working_set_bytes=10*1024**3)})
    summary = {'seed': 0, 'batch_size': 512, 'configured_batch_size': 512, 'warmup_steps': 1000, 'train_window_steps': 88,
        'steps': 2*train.BATCHES, 'steps_realized': 2*train.BATCHES, 'compute_loss_calls': 2*train.BATCHES,
        'train_windows': len(windows), 'train_window_metrics': windows, 'corpus': {'shard_dirs': [prep['corpus']]},
        'realized_after_guard': {'device': 'cuda', 'use_compile': True}, 'valid_control': False,
        'validity_problems': ['historical exact-epoch sampling deviation'], 'checkpoints': [], 'sampling': {
        'mode': 'game_epochs', 'complete': True, 'epochs_requested': 2, 'epochs_completed': 2,
        'sampling_seed_rule': 'seed + zero_based_epoch_index', 'augmentation_rng': 'continuous_from_epoch_one',
        'corpus_sha256': 'c'*64, 'rows_realized': 2*train.ROWS, 'batches_realized': 2*train.BATCHES, 'loss_normalization': train.LOSS, 'epochs': records}}
    return m, prep, summary


def test_actual_plan_serialization_and_ragged_two_epoch_completion(fixture: Any) -> None:
    m, prep, summary = fixture
    train.validate(m)
    train.verify_plans(m, prep)
    train.verify_summary(summary, m, prep)
    run = Path(m['run'])
    run.mkdir()
    for role, name in [('epoch1', 'checkpoint_epoch1.pt'), ('last', 'checkpoint.pt')]:
        path = run/name
        path.write_bytes(role.encode())
        summary['checkpoints'].append({'role': role, 'path': str(path), 'sha256': train.owned.sha(path)})
    write(run/'summary.json', summary)
    result = train.complete(m, prep)
    assert result['two_epoch_training_complete'] is True
    assert result['historical_valid_control'] is False
    assert not {'complete', 'checkpoint', 'canonical_plan_sha256'} & result.keys()
    assert result['epochs'][0]['steps_end'] == result['epochs'][1]['steps_start'] == 36935
    assert summary['train_window_metrics'][419]['steps_requested'] == 63
    assert summary['train_window_metrics'][420]['steps_requested'] == 88
    (run/'checkpoint_epoch1.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):
        train.complete(m, prep)


@pytest.mark.parametrize('mutation', [
    lambda s: s['sampling'].update(mode='game_epoch'),
    lambda s: s['sampling'].update(augmentation_rng='reseeded'),
    lambda s: s['sampling'].update(loss_normalization='historical'),
    lambda s: s['sampling']['epochs'][1]['sampling'].update(complete=False),
    lambda s: s['sampling']['epochs'][1]['sampling'].update(seed=0),
    lambda s: s['sampling']['epochs'][1]['sampling'].update(realized_sha256='f'*64),
    lambda s: s['sampling']['epochs'][1]['sampling'].update(objective_mask_weights={'policy': 124.0}),
    lambda s: s['sampling']['epochs'][1]['sampling'].update(decoded_bytes_resident=1),
    lambda s: s['sampling']['epochs'][1]['sampling'].update(same_game_repeats_max=1),
    lambda s: s['sampling']['epochs'][1]['sampling'].update(peak_working_set_bytes=13*1024**3),
    lambda s: s['train_window_metrics'][420].update(epoch_window_index=420),
    lambda s: s['train_window_metrics'][420].update(steps_cumulative=37048),
    lambda s: s['train_window_metrics'][421].update(window_index=421),
    lambda s: s['train_window_metrics'][420].update(transient_cuda_retry_batches=1),
    lambda s: s['train_window_metrics'][420].update(train_steps_done=87),
    lambda s: s['train_window_metrics'][420].update(train_samples_seen=45055),
    lambda s: s['train_window_metrics'][420].update(train_samples_seen=1),
    lambda s: s['train_window_metrics'][420].update(loss=float('nan')),
    lambda s: s['train_window_metrics'][420].update(grad_norm_mean=float('inf')),
    lambda s: s.update(warmup_steps=2000),
])
def test_incomplete_or_changed_trajectory_refused(fixture: Any, mutation: Any) -> None:
    m, prep, summary = fixture
    mutation(summary)
    with pytest.raises((RuntimeError, ValueError)):
        train.verify_summary(summary, m, prep)


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(status='PENDING'),
    lambda p: p.update(profile='H20'),
    lambda p: p.update(load_workers=16),
    lambda p: p['epochs'][1]['plan'].update(seed=0),
    lambda p: p['epochs'][1]['plan'].update(policy_size=4672),
    lambda p: p['epochs'][1]['plan'].update(history_rep_fix=False),
    lambda p: p['epochs'][1]['plan'].update(input_history_encoding='other'),
    lambda p: p['epochs'][1]['plan'].update(mirror_augmentation=False),
    lambda p: p['epochs'][1]['plan'].update(corpus_sha256='e'*64),
    lambda p: p['epochs'][1]['plan'].update(peak_working_set_bytes_planned=13*1024**3),
    lambda p: p['epochs'][1].update(corpus_logical_order_sha256='f'*64),
    lambda p: p['epochs'][1]['plan'].pop('objective_mask_weights'),
])
def test_preparation_refuses_missing_proof_or_resource_drift(fixture: Any, mutation: Any) -> None:
    m, prep, _summary = fixture
    mutation(prep)
    with pytest.raises((RuntimeError, ValueError)):
        train.verify_plans(m, prep)


def test_explicit_resources_and_single_uninterrupted_command(fixture: Any) -> None:
    m, prep, _ = fixture
    cmd = train.train_command(m, {'runtime': {'executable': '/qualified/python'}}, Path(prep['corpus']))
    assert cmd.count('scripts/lc0_control_train.py') == 1
    for flag, value in [('--epochs', '2'), ('--epoch-plan-workers', '3'), ('--epoch-load-workers', '2'),
                        ('--epoch-max-working-set-gib', '12.0'), ('--seed', '0'), ('--train-window-steps', '88')]:
        assert cmd[cmd.index(flag)+1] == value
    assert not any('resume' in word or 'arena' in word for word in cmd)
    assert train.environment(Path('/qualified'), gpu=False)['CUDA_VISIBLE_DEVICES'] == ''


@pytest.mark.parametrize(('stop', 'fail'), [(False, False), (True, False), (False, True)])
def test_owned_stage_wiring_stop_failure_and_release_before_completion(
    fixture: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: bool, fail: bool,
) -> None:
    m, prep, _ = fixture
    runtime = tmp_path/'runtime'
    runtime.mkdir()
    rt = {'python': 'fixture', 'executable': '/fixture/python', 'torch': 'fixture', 'cuda': 'fixture', 'numpy': 'fixture',
              'native_extensions': {}, 'native_extension_sha256': {}}
    q = {'root': str(runtime), 'runtime': rt}
    monkeypatch.setattr(train, 'check_inputs', lambda _m: (q, prep, Path(prep['corpus'])))
    monkeypatch.setattr(train.owned, 'ROOT', tmp_path)
    (tmp_path/'scratchpad').mkdir()
    monkeypatch.setattr(train.owned, 'disk_guard', lambda _path: None)
    monkeypatch.setattr(train.subprocess, 'check_output', lambda cmd, **_kw: '' if cmd[0] == 'nvidia-smi' else json.dumps({k:v for k,v in rt.items() if k != 'native_extension_sha256'}))
    held: list[Any] = []
    def acquire(lease: Any) -> None:
        held.append(lease)
        if stop:
            (Path(m['state'])/'STOP').touch()
    monkeypatch.setattr(train.owned, 'acquire_gpu_lease', acquire)
    stages: list[Any] = []
    def stage(cmd: Any, _out: Any, _seconds: Any, fd: Any, _label: Any, _metadata: Any, **kwargs: Any) -> dict[str, Any]:
        assert not held[0].closed
        assert fd == held[0].fileno()
        assert kwargs['cwd'] == runtime
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0'
        assert kwargs['stop_paths'] == (Path(m['state'])/'STOP',)
        stages.append(cmd)
        write(Path(m['state'])/'training_threads.json', {'after': {'torch': 2, 'blosc': 2, 'compile': 2}, 'dynamo_suppress_errors': False,
              'driver': str(runtime/'scripts/lc0_control_train.py'), 'argv': cmd[4:]})
        if fail:
            raise RuntimeError('injected owned failure')
        return {'gpu_seconds': 1}
    monkeypatch.setattr(train.owned, 'run_owned_stage', stage)
    def completion(*_args: Any) -> dict[str, Any]:
        assert held[0].closed
        return {'scope': train.SCOPE, 'two_epoch_training_complete': True}
    monkeypatch.setattr(train, 'complete', completion)
    if stop or fail:
        with pytest.raises((RuntimeError, ValueError)):
            train.execute(m)
        assert (Path(m['state'])/'failed.json').is_file()
        assert not (Path(m['state'])/'two_epoch_training.complete.json').exists()
    else:
        train.execute(m)
        assert (Path(m['state'])/'two_epoch_training.complete.json').is_file()
    assert len(stages) == (0 if stop else 1)
    assert not (Path(m['state'])/'training.complete.json').exists()
    assert not (Path(m['state'])/'complete.json').exists()


def test_no_flag_old_coordinator_admission_still_rejects_two_epoch_manifest(fixture: Any) -> None:
    with pytest.raises((ValueError, KeyError)):
        train.recipes.validate(fixture[0])


def test_missing_qualification_is_refused_before_probe_or_stage(fixture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    m, _, _ = fixture
    for path, key in ((train.__file__, 'launcher_sha256'), (train.owned.__file__, 'stage_helper_sha256'), (train.recipes.__file__, 'recipe_helper_sha256')):
        m[key] = train.owned.sha(path)
    monkeypatch.setattr(train.subprocess, 'check_output', lambda *_a, **_k: pytest.fail('runtime work before qualification'))
    with pytest.raises(FileNotFoundError):
        train.execute(m)
    assert not Path(m['state']).exists()


@pytest.mark.parametrize('defect', ['none', 'wrong_score_space', 'changed_nonpolicy', 'wrong_qualification'])
def test_genuine_softsf_helper_admitted_without_bt4_or_old_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str,
) -> None:
    from tests.test_bt4_one_epoch_screen import softsf_files
    old, files, pins = softsf_files(tmp_path, monkeypatch)
    corpus = train.recipes.CORPORA['SoftSF10']
    recipe_ref = {'path': str(corpus/'sf_policy_rewrite_summary.json'), 'sha256': 'a'*64}
    derive_ref = {'path': str(corpus/'derive_targets_summary.json'), 'sha256': 'b'*64}
    prep = {'source': {'path': str(train.recipes.SOURCE/'derive_targets_summary.json'),
             'sha256': train.recipes.COMMON_PINS[str(train.recipes.SOURCE/'derive_targets_summary.json')]},
            'data_qualification': old['data_qualification'], 'derive_summary': derive_ref, 'recipe_summary': recipe_ref}
    if defect == 'wrong_score_space':
        files[recipe_ref['path']]['score_space'] = 'q'
    elif defect == 'changed_nonpolicy':
        files[derive_ref['path']]['value_scheme'] = {'name': 'replacement'}
    elif defect == 'wrong_qualification':
        files[old['data_qualification']['path']]['profile'] = 'B100'
    if defect == 'none':
        assert train.verify_recipe(prep, 'SoftSF10') == corpus
        assert (recipe_ref['path'], recipe_ref['sha256']) in pins
        assert not any('bt4_policy_mix_summary' in path for path, _ in pins)
    else:
        with pytest.raises(ValueError, match=r'SoftSF10|qualification'):
            train.verify_recipe(prep, 'SoftSF10')


@pytest.mark.parametrize('bad', [True, float('nan'), 0.1])
def test_retry_metric_rejects_boolean_nonfinite_or_positive_value(fixture: Any, bad: Any) -> None:
    m, prep, summary = fixture
    summary['train_window_metrics'][0]['transient_cuda_retry_batches'] = bad
    with pytest.raises(ValueError, match='skipped or retried'):
        train.verify_summary(summary, m, prep)


@pytest.mark.parametrize('defect', ['none', 'cpu_only', 'other_head', 'other_numpy', 'eager', 'one_epoch'])
def test_actual_runtime_receipt_shapes_fail_closed(tmp_path: Path, defect: str) -> None:
    rt = {'python': 'fixture', 'executable': '/fixture/python', 'torch': 'fixture', 'cuda': '12.8', 'numpy': '1.26.2'}
    cpu = {'status': 'PASS_CPU_TRAINING_IMPORTS', 'head': train.TRAINING_HEAD, 'runtime': str(tmp_path),
           'cuda_initialized': False, **{k:v for k,v in rt.items() if k != 'cuda'}, 'cuda_build': rt['cuda']}
    sampling = {'complete': True, 'mode': 'game_epochs', 'epochs_requested': 2, 'epochs_completed': 2,
                'rows_realized': 2048, 'batches_realized': 4}
    summary = {'sampling': sampling, 'trainable_params': 61444448, 'batch_size': 512, 'steps_realized': 4,
               'compute_loss_calls': 4, 'realized_after_guard': {'device': 'cuda', 'use_compile': True}}
    cuda: dict[str, Any] = {'status': 'PASS_COMPILED_CUDA_TWO_EPOCH_PROBE',
            'runtime': {**{k:v for k,v in cpu.items() if k not in ('status', 'runtime')}, 'path': str(tmp_path), 'dynamo_suppress_errors': False},
            'model_parameters': 61444448, 'sampling': sampling,
            'compile': {'unique_graphs': 1, 'frames_ok': 1, 'inductor_graph_cache_events': 1, 'generated_kernel_count_delta': 0},
            'summary': write(tmp_path/'summary.json', summary)}
    if defect == 'cpu_only':
        cuda['status'] = 'PASS_CPU_TRAINING_IMPORTS'
    elif defect == 'other_head':
        cuda['runtime']['head'] = 'a'*40
    elif defect == 'other_numpy':
        cuda['runtime']['numpy'] = '2.2.6'
    elif defect == 'eager':
        cuda['compile']['unique_graphs'] = 0
    elif defect == 'one_epoch':
        cuda['sampling']['epochs_completed'] = 1
    q = {'root': str(tmp_path), 'runtime': rt, 'cpu_qualification': write(tmp_path/'cpu.json', cpu),
         'cuda_qualification': write(tmp_path/'cuda.json', cuda)}
    if defect == 'none':
        train.verify_runtime_evidence(q)
    else:
        with pytest.raises(ValueError, match=r'CUDA|capture'):
            train.verify_runtime_evidence(q)


def test_real_child_bootstrap_normalizes_threads_and_reaches_actual_driver_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = Path(__file__).resolve().parents[1]
    record = tmp_path/'threads.json'
    monkeypatch.setenv('TORCHDYNAMO_SUPPRESS_ERRORS', '1')
    env = train.environment(runtime, gpu=False)
    assert 'TORCHDYNAMO_SUPPRESS_ERRORS' not in env
    env['TORCHINDUCTOR_COMPILE_THREADS'] = '7'
    result = subprocess.run([sys.executable, '-c', train.CHILD_BOOTSTRAP, str(record),
                             'scripts/lc0_control_train.py', '--help'], cwd=runtime, env=env,
                            text=True, capture_output=True, timeout=90, check=False)
    assert result.returncode == 0, result.stderr
    assert '--epochs' in result.stdout
    observed = json.loads(record.read_text())
    assert observed['before']['compile'] == 7
    assert observed['after'] == {'torch': 2, 'blosc': 2, 'compile': 2}
    assert observed['argv'] == ['scripts/lc0_control_train.py', '--help']
    assert observed['driver'] == str(runtime/'scripts/lc0_control_train.py')
    assert observed['cuda_initialized'] is False
    assert observed['dynamo_suppress_errors'] is False
