"""Small saved-summary and subprocess fixtures; no training/data/GPU workload."""
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import bt4_direct_screen as arena
from scripts import bt4_one_epoch_screen as epoch


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
               'train_windows': 420,
               'train_window_metrics': [{'grad_nonfinite_skip_rate': 0., 'transient_cuda_retry_batches': 0.,
                    'loss': 2., 'grad_norm_mean': 1., 'window_index': i + 1, 'steps_requested': min(88, 36935 - i * 88),
                    'train_steps_done': min(88, 36935 - i * 88), 'steps_cumulative': min((i + 1) * 88, 36935),
                    'train_samples_seen': 45056 if i < 419 else 18910484 - 419 * 45056} for i in range(420)]}
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
        checkpoint = result['checkpoint']
        assert isinstance(checkpoint, dict)
        assert checkpoint['role'] == 'E0T05'
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


@pytest.mark.parametrize('mode', ['script', 'module'])
def test_epoch_cli_and_module_import_the_sibling_launcher(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    command = [str(root / 'scripts/bt4_one_epoch_screen.py')] if mode == 'script' else ['-m', 'scripts.bt4_one_epoch_screen']
    result = subprocess.run([sys.executable, *command, '--help'],
                            cwd=tmp_path if mode == 'script' else root,
                            env={**os.environ, 'PYTHONPATH': '', 'CUDA_VISIBLE_DEVICES': ''},
                            text=True, capture_output=True, check=True, timeout=10)
    assert '--manifest' in result.stdout
    assert '--execute' in result.stdout


def registered_manifest(tmp_path, role='H20') -> dict[str, Any]:
    corpus = epoch.CORPORA[role]
    evidence = {'path': str(tmp_path / 'evidence.json'), 'sha256': 'a' * 64}
    return {'schema': 2, 'profile': role, 'state': str(tmp_path / 'state'), 'run': str(tmp_path / 'run'),
            'training_seconds': 16200, 'arena_seconds': 5400, 'total_seconds': epoch.TOTAL_CAPS[role],
            'runtime_manifest': evidence, 'preregistration': evidence, 'prospective_schedule': evidence,
            'launcher_sha256': 'a' * 64, 'arena_launcher_sha256': 'a' * 64,
            'reader': {'path': str(arena.EXTENDED_READER), 'sha256': 'a' * 64},
            'data_qualification': evidence, 'comparisons': [list(c) for c in arena.REGISTERED_COMPARISONS[role]],
            'input_pins': {**epoch.COMMON_PINS, str(corpus / 'bt4_policy_mix_summary.json'): 'a' * 64,
                           str(corpus / 'derive_targets_summary.json'): 'b' * 64}}


@pytest.mark.parametrize('role', ['H20', 'B100', 'G50'])
def test_registered_profile_train_command_and_completed_role(tmp_path, role):
    m = registered_manifest(tmp_path, role)
    epoch.validate(m)
    Path(m['runtime_manifest']['path']).write_text(json.dumps({'runtime': {'executable': sys.executable}}))
    cmd = epoch.train_command(m)
    assert cmd[cmd.index('--shards') + 1] == str(epoch.CORPORA[role])
    assert cmd[cmd.index('--seed') + 1] == '0'
    assert cmd[cmd.index('--steps') + 1] == '0'
    run, summary, report = training_fixture(tmp_path)
    summary['corpus']['shard_dirs'] = [str(epoch.CORPORA[role])]
    report['arms'][role] = report['arms'].pop('E0T05')
    report['arms'][role]['corpus'] = str(epoch.CORPORA[role])
    (run / 'summary.json').write_text(json.dumps(summary))
    report['arms'][role]['summary_sha256'] = arena.sha(run / 'summary.json')
    schedule = tmp_path / 'schedule.json'
    schedule.write_text(json.dumps(report))
    result = epoch.completed_training(m, schedule)
    assert result['role'] == role
    checkpoint = result['checkpoint']
    assert isinstance(checkpoint, dict)
    assert checkpoint['role'] == role
    assert result['historical_valid_control'] is False
    report['arms'][role]['physical_plan_sha256'] = 'wrong'
    schedule.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='schedule differ'):
        epoch.completed_training(m, schedule)


@pytest.mark.parametrize('mutation', ['order', 'extra', 'cap', 'source_pin', 'corpus_pin'])
def test_registered_profiles_reject_unregistered_work(tmp_path, mutation):
    m = registered_manifest(tmp_path)
    if mutation == 'order':
        m['comparisons'].reverse()
    elif mutation == 'extra':
        m['comparisons'].append(['G20T05', 400, 500])
    elif mutation == 'cap':
        m['total_seconds'] = 108000  # Global authorization is not reusable per arm.
    elif mutation == 'source_pin':
        m['input_pins'][str(epoch.SOURCE / 'derive_targets_summary.json')] = 'b' * 64
    else:
        m['input_pins'][str(epoch.CORPORA['H20'] / 'derive_targets_summary.json')] = 'pending'
    with pytest.raises(ValueError, match=r'order differs|budget differs|input pins differ|hashes required'):
        epoch.validate(m)


@pytest.mark.parametrize('stop_after', [None, 2])
def test_registered_sequence_reuses_training_and_stops_between_cells(tmp_path, monkeypatch, stop_after):
    m = registered_manifest(tmp_path)
    root = tmp_path / 'root'
    (root / 'scratchpad').mkdir(parents=True)
    monkeypatch.setattr(epoch, 'ROOT', root)
    monkeypatch.setattr(epoch, 'validate', lambda _: None)
    monkeypatch.setattr(epoch, 'check_pins', lambda _: {'executable': sys.executable})
    monkeypatch.setattr(epoch, 'verify_schedule', lambda *a, **kw: None)
    monkeypatch.setattr(arena, 'runtime_probe', lambda _: {})
    monkeypatch.setattr(arena, 'disk_guard', lambda _: None)
    monkeypatch.setattr(epoch.subprocess, 'check_output', lambda *a, **kw: '')
    Path(m['prospective_schedule']['path']).write_text('{}')
    stages, cells = [], []
    def stage(*args, **_kwargs):
        stages.append(args[4])
        return {'gpu_seconds': 10 if args[4] == 'training' else 0}
    monkeypatch.setattr(arena, 'run_owned_stage', stage)
    monkeypatch.setattr(epoch, 'train_command', lambda _: ['frozen-trainer'])
    candidate = {'role': 'H20', 'path': str(Path(m['run']) / 'checkpoint.pt'), 'sha256': 'c' * 64}
    monkeypatch.setattr(epoch, 'completed_training', lambda *_: {'complete': True, 'checkpoint': candidate})
    def match(cell, **_kwargs):
        cells.append((cell['reference']['role'], cell['sims'], cell['games']))
        assert cell['candidate'] == candidate
        out = Path(cell['output'])
        out.mkdir()
        arena.write(out / 'complete.json', {'complete': True, 'gpu_seconds': 20, 'games_sha256': 'd' * 64})
        if cell['games'] == 500:
            assert cell['opening_anchor']['bank']['path'].endswith('C20T05.s100/arena.games.jsonl')
        if len(cells) == stop_after:
            (Path(m['state']) / 'STOP').touch()
    monkeypatch.setattr(arena, 'execute', match)
    if stop_after:
        with pytest.raises(ValueError, match='stop requested'):
            epoch.execute(m)
        assert len(cells) == 2
        assert not (Path(m['state']) / 'complete.json').exists()
    else:
        epoch.execute(m)
        assert cells == [('C20T05', 100, 1000), ('G20T05', 100, 1000), ('C20T05', 400, 500)]
        result = arena.read(Path(m['state']) / 'complete.json')
        assert result['gpu_seconds'] == 70
        assert len(result['arena_receipts']) == 3
    assert stages == ['training', 'schedule']


@pytest.mark.parametrize('mutation', ['none', 'failed', 'corpus', 'hash', 'role'])
def test_data_qualification_binds_success_and_final_recipe(tmp_path, mutation):
    m = registered_manifest(tmp_path)
    corpus = epoch.corpus_for(m)
    receipt: dict[str, Any] = {'schema': 1, 'status': 'PASS_REGISTERED_CORPUS_QUALIFICATION', 'profile': 'H20',
               'corpus': str(corpus), 'rows': 18910484, 'shards': 2309,
               'source': {'path': str(epoch.SOURCE), 'derive_sha256': epoch.COMMON_PINS[str(epoch.SOURCE / 'derive_targets_summary.json')]},
               'derive_summary': {'path': str(corpus / 'derive_targets_summary.json'), 'sha256': 'b' * 64},
               'mix_summary': {'path': str(corpus / 'bt4_policy_mix_summary.json'), 'sha256': 'a' * 64}}
    if mutation == 'failed':
        receipt['status'] = 'FAILED'
    elif mutation == 'corpus':
        receipt['corpus'] = str(epoch.CORPORA['B100'])
    elif mutation == 'role':
        receipt['profile'] = 'B100'
    elif mutation == 'hash':
        receipt['derive_summary']['sha256'] = 'c' * 64
    Path(m['data_qualification']['path']).write_text(json.dumps(receipt))
    if mutation == 'none':
        epoch.verify_data_qualification(m)
    else:
        with pytest.raises(ValueError, match='data qualification failed'):
            epoch.verify_data_qualification(m)


@pytest.mark.parametrize('mutation', ['duplicate', 'skipped', 'cumulative', 'samples'])
def test_registered_window_cadence_rejects_duplicate_or_skipped_work(tmp_path, mutation):
    _, summary, _ = training_fixture(tmp_path)
    epoch.verify_window_cadence(summary)
    windows = summary['train_window_metrics']
    if mutation == 'duplicate':
        windows[1] = dict(windows[0])
    elif mutation == 'skipped':
        windows[2]['train_steps_done'] = 87
    elif mutation == 'cumulative':
        windows[-1]['steps_cumulative'] -= 1
    else:
        windows[-1]['train_samples_seen'] -= 1
    with pytest.raises(ValueError, match=r'cadence/cumulative|sample total'):
        epoch.verify_window_cadence(summary)
