"""Small saved-summary and subprocess fixtures; no training/data/GPU workload."""
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('bt4_direct_screen', SCRIPTS / 'bt4_direct_screen.py')
assert spec is not None
assert spec.loader is not None
arena = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arena)
sys.modules['bt4_direct_screen'] = arena
spec = importlib.util.spec_from_file_location('bt4_one_epoch_screen', SCRIPTS / 'bt4_one_epoch_screen.py')
assert spec is not None
assert spec.loader is not None
epoch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(epoch)


def training_fixture(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'checkpoint.pt').write_bytes(b'fixture weights')
    sampling = {'mode': 'game_epoch', 'complete': True, 'seed': 0, 'batch_size': 512,
                'rows_planned': 18910484, 'rows_realized': 18910484, 'batches_planned': 36935,
                'batches_realized': 36935, 'shards': 2309, 'games': 97968, 'plan_workers': 16,
                'load_workers': 16, 'same_game_repeats_max': 0, 'decoded_rows_resident': 0,
                'plan_sha256': 'physical', 'realized_sha256': 'physical'}
    summary: dict[str, Any] = {'sampling': sampling, 'seed': 0, 'batch_size': 512, 'train_window_steps': 88,
               'steps_realized': 36935, 'compute_loss_calls': 36935,
               'corpus': {'shard_dirs': [str(epoch.CORPUS)]}, 'valid_control': False,
               'validity_problems': ['historical purity limitation'],
               'checkpoints': [{'role': 'last', 'path': str(run / 'checkpoint.pt'),
                                'sha256': arena.sha(run / 'checkpoint.pt')}],
               'train_window_metrics': [{'grad_nonfinite_skip_rate': 0., 'transient_cuda_retry_batches': 0.,
                                         'loss': 2., 'grad_norm_mean': 1.} for _ in range(420)]}
    report: dict[str, Any] = {'verifier_sha256': epoch.INPUT_PINS[str(epoch.VERIFIER)], 'seed': 0,
              'batch_size': 512, 'runtime': {'numpy': '1.26.2'},
              'source_plan': {'plan_sha256': epoch.CANONICAL, 'rows_planned': 18910484, 'batches_planned': 36935},
              'arms': {'E0T05': {'corpus': str(epoch.CORPUS), 'staging': 'verified actual',
                      'training_completion_verified': True, 'metadata_matches_source': True,
                      'canonical_plan_sha256': epoch.CANONICAL, 'physical_plan_sha256': 'physical'}}}
    return run, summary, report


def save_fixture(run, summary, report):
    (run / 'summary.json').write_text(json.dumps(summary))
    report['arms']['E0T05']['summary_sha256'] = arena.sha(run / 'summary.json')
    path = run.parent / 'schedule.json'
    path.write_text(json.dumps(report))
    return path


@pytest.mark.parametrize('mutation', ['none', 'skipped', 'retried', 'wrong_physical', 'prospective_only', 'wrong_checkpoint'])
def test_completion_requires_actual_epoch_and_schedule(tmp_path, mutation):
    run, summary, report = training_fixture(tmp_path)
    if mutation == 'skipped':
        summary['train_window_metrics'][0]['grad_nonfinite_skip_rate'] = .1
    elif mutation == 'retried':
        summary['train_window_metrics'][0]['transient_cuda_retry_batches'] = 1
    elif mutation == 'wrong_physical':
        report['arms']['E0T05']['physical_plan_sha256'] = 'other'
    elif mutation == 'prospective_only':
        report['arms']['E0T05']['staging'] = 'prospective only'
        report['arms']['E0T05']['training_completion_verified'] = False
    elif mutation == 'wrong_checkpoint':
        summary['checkpoints'][0]['sha256'] = 'other'
    path = save_fixture(run, summary, report)
    if mutation == 'none':
        result = epoch.completed_training({'run': str(run)}, path)
        assert result['complete'] is True
        assert result['historical_valid_control'] is False
        assert result['checkpoint']['role'] == 'E0T05'
    else:
        with pytest.raises(ValueError, match=r'skipped/retried|schedule differ|stage differs|checkpoint differs'):
            epoch.completed_training({'run': str(run)}, path)


@pytest.mark.parametrize('exit_code', [0, 7])
def test_owned_stage_closes_charge_and_preserves_failed_state(tmp_path, monkeypatch, exit_code):
    monkeypatch.setattr(arena, 'RUNTIME', tmp_path)
    monkeypatch.setattr(arena, 'disk_guard', lambda _path: None)
    monkeypatch.setattr(arena, 'environment', lambda _gpu=False: {'CUDA_VISIBLE_DEVICES': ''})
    out = tmp_path / 'stage'
    with (tmp_path / 'lease').open('a') as lease:
        command = [sys.executable, '-c', f'import sys; print("fixture"); sys.exit({exit_code})']
        if exit_code:
            with pytest.raises(ValueError, match='exited 7'):
                arena.run_owned_stage(command, out, 35, lease.fileno(), 'training', {}, manifest={'fixture': True})
            assert arena.read(out / 'failed.json')['complete'] is False
            assert not (out / 'complete.json').exists()
        else:
            receipt = arena.run_owned_stage(command, out, 35, lease.fileno(), 'training', {}, manifest={'fixture': True})
            assert receipt['process_complete'] is True
            assert receipt['exit_code'] == 0
            assert receipt['gpu_seconds'] > 0
            assert receipt['complete'] is False  # Process success alone does not qualify an epoch.
        assert arena.read(out / 'manifest.json') == {'fixture': True}
        assert 'fixture' in (out / 'training.log').read_text()


def test_esharp_profile_requires_training_receipt_and_c_reference(tmp_path):
    role, checkpoint, digest = arena.CHECKPOINTS['candidate']
    m: dict[str, Any] = {'schema': 1, 'output': str(tmp_path / 'arena'), 'sims': 100, 'hard_seconds': 5400,
         'candidate': {'role': 'E0T05', 'path': str(tmp_path / 'run/checkpoint.pt'), 'sha256': 'new'},
         'reference': {'role': role, 'path': str(checkpoint), 'sha256': digest},
         'candidate_training': {'path': str(tmp_path / 'training.complete.json'), 'sha256': 'completion'},
         'book': {'path': str(arena.BOOK), 'sha256': arena.BOOK_SHA},
         'runtime_manifest': {'path': str(tmp_path / 'runtime.json'), 'sha256': 'runtime'},
         'preregistration': {'path': str(tmp_path / 'prereg.md'), 'sha256': 'prereg'}, 'launcher_sha256': 'script'}
    arena.validate(m)
    m['reference']['sha256'] = 'different'
    with pytest.raises(ValueError, match='wrong C reference'):
        arena.validate(m)
    m['reference']['sha256'] = digest
    del m['candidate_training']
    with pytest.raises(ValueError, match='manifest keys differ'):
        arena.validate(m)


def test_arena_stop_set_during_lease_wait_prevents_launch(tmp_path, monkeypatch):
    stop = tmp_path / 'STOP'
    (tmp_path / 'scratchpad').mkdir()
    monkeypatch.setattr(arena, 'ROOT', tmp_path)
    monkeypatch.setattr(arena, 'validate', lambda _m: None)
    monkeypatch.setattr(arena, 'check_pins', lambda _m: {})
    monkeypatch.setattr(arena, 'runtime_probe', lambda _rt: {})
    monkeypatch.setattr(arena, 'disk_guard', lambda _path: None)
    monkeypatch.setattr(arena, 'acquire_gpu_lease', lambda _lease: stop.touch())
    out = tmp_path / 'arena'
    with pytest.raises(ValueError, match='stop requested while waiting'):
        arena.execute({'output': str(out)}, stop_paths=(stop,))
    assert not out.exists()


def test_cpu_schedule_supervision_hides_gpu_and_has_no_gpu_charge(tmp_path, monkeypatch):
    monkeypatch.setattr(arena, 'RUNTIME', tmp_path)
    monkeypatch.setattr(arena, 'disk_guard', lambda _path: None)
    command = [sys.executable, '-c', 'import os; assert os.environ["CUDA_VISIBLE_DEVICES"] == ""']
    receipt = arena.run_owned_stage(command, tmp_path / 'schedule', 35, None, 'schedule', {}, manifest={})
    assert receipt['gpu_seconds'] == 0
    assert receipt['stage_seconds'] > 0
    assert receipt['process_complete'] is True


def test_parent_stop_prevents_any_owned_stage(tmp_path):
    stop = tmp_path / 'STOP'
    stop.touch()
    out = tmp_path / 'schedule'
    with pytest.raises(ValueError, match='stop requested before stage'):
        arena.run_owned_stage([], out, 35, None, 'schedule', {}, manifest={}, stop_paths=(stop,))
    assert not out.exists()
