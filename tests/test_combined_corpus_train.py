"""Combined training wiring and completed evidence; no training/model execution."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import combined_corpus_train as tool
from scripts import bt4_package_readout as reader


def put(path: Path, value: Any) -> dict[str, str]:
    path.write_text(json.dumps(value))
    return tool.pin(path)


@pytest.fixture
def completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[str, Any]]:
    monkeypatch.setattr(tool, 'ROWS', 21)
    panel = put(tmp_path / 'panel.json', [{}] * 256)
    monkeypatch.setattr(tool, 'PANEL_SHA', panel['sha256'])
    cohorts, mapping = [], []
    roots: dict[str, list[str]] = {'B100': [], 'V50': []}
    for i in range(21):
        cohort: dict[str, Any] = {'id': f'c{i}', 'roots': {}}
        row: dict[str, Any] = {'cohort': f'c{i}', 'shard_index': 0, 'rows': 1, 'namespace': str(i), 'paths': {}}
        for arm in ('source', 'B100', 'V50'):
            root = tmp_path / f'{arm}_{i}'
            root.mkdir()
            shard = root / 'shard_000000.zarr'
            shard.mkdir()
            row['paths'][arm] = str(shard)
            cohort['roots'][arm] = {'summary': put(root / 'derive_targets_summary.json', {'rows': 1})}
            if arm != 'source':
                roots[arm].append(str(root))
        cohorts.append(cohort)
        mapping.append(row)
    manifest = put(tmp_path / 'corpus.json', {'seed': 101, 'batch_size': 512,
                    'expected_rows': 21, 'expected_shards': 21, 'cohorts': cohorts})
    plans = {arm: {'seed': 101, 'batch_size': 512, 'mode': 'game_epoch', 'rows_planned': 21,
                 'batches_planned': 1, 'shards': 21, 'games': 21, 'plan_sha256': arm * 16}
             for arm in ('source', 'B100', 'V50')}
    report: dict[str, Any] = {'status': 'PASS_CORPUS_SET_PROSPECTIVE_NOT_TRAINING', 'manifest_sha256': manifest['sha256'],
              'mapping': mapping, 'rows': 21, 'seed': 101, 'batch_size': 512, 'trainer_shards': roots,
              'training_role_to_arm': tool.ROLES, 'ordered_source_columns': [{'rows': 1, 'game_id_sha256': str(i)} for i in range(21)],
              'runtime': {'python': '3.10.12 qualified', 'numpy': '1.26.2', 'torch': '2.11.0+cu128'},
              'arms': {arm: {'physical_plan': plan, 'canonical_plan_sha256': plans['source']['plan_sha256'],
                            'metadata_matches_source': True, 'training_completed': False} for arm, plan in plans.items()}}
    prospective = put(tmp_path / 'prospective.json', report)
    runtime = {'executable': '/frozen/python', **report['runtime']}
    runtime_pin = put(tmp_path / 'runtime.json', {'runtime': runtime})
    evidence: dict[str, Any] = {}
    for side, role in zip(('candidate', 'reference'), ('Combined35M_V50', 'Combined35M_SF100')):
        arm = tool.ROLES[role]
        run = tmp_path / role
        run.mkdir()
        staging = run / 'staged_shards'
        staging.mkdir()
        for i, row in enumerate(mapping):
            (staging / f'shard_{i:06d}.zarr').symlink_to(row['paths'][arm], target_is_directory=True)
        checkpoint = {'role': role, 'path': str(run / 'checkpoint.pt'), 'sha256': side * 8}
        plan = plans[arm]
        summary = {'seed': 101, 'batch_size': 512, 'warmup_steps': 1000, 'train_window_steps': 88,
                   'steps_realized': 1, 'compute_loss_calls': 1, 'corpus': {'shard_dirs': roots[arm]},
                   'sampling': {**plan, 'rows_realized': 21, 'batches_realized': 1, 'complete': True,
                                'plan_workers': 2, 'load_workers': 2, 'same_game_repeats_max': 0,
                                'decoded_rows_resident': 0, 'realized_sha256': plan['plan_sha256']},
                   'train_windows': 1, 'train_window_metrics': [{'window_index': 1, 'steps_requested': 1,
                        'train_steps_done': 1, 'steps_cumulative': 1, 'train_samples_seen': 21,
                        'grad_nonfinite_skip_rate': 0, 'transient_cuda_retry_batches': 0, 'loss': .5, 'grad_norm_mean': 1.}],
                   'checkpoints': [{**checkpoint, 'role': 'last'}], 'valid_control': False,
                   'validity_problems': ['historical comparison limitations retained']}
        summary_pin = put(run / 'summary.json', summary)
        receipt = {'schema': 1, 'complete': True, 'profile': tool.PROFILE, 'role': role, 'run': str(run),
                   'corpus_manifest': manifest, 'prospective': prospective, 'opening_panel': panel,
                   'runtime_manifest': runtime_pin, 'preregistration': {'path': '/registered', 'sha256': 'a' * 64},
                   'checkpoint': checkpoint, 'summary_sha256': summary_pin['sha256'],
                   'physical_plan_sha256': plan['plan_sha256'], 'canonical_plan_sha256': plans['source']['plan_sha256'],
                   'actual_staging_verified': True, 'actual_game_columns_verified': True,
                   'actual_staging_sha256': tool.verify_staging(run, report, role),
                   'actual_game_columns': report['ordered_source_columns'], 'training_charge_seconds': 100.,
                   'input_pins': {str(tool.owned.RUNTIME / k): v for k, v in tool.FROZEN.items()},
                   'code_pins': {str(Path(tool.__file__).resolve()): tool.owned.sha(tool.__file__)},
                   'historical_valid_control': False, 'historical_validity_problems': summary['validity_problems']}
        receipt['training_process'] = put(run / 'process.json', {'process_complete': True, 'exit_code': 0,
                     'gpu_seconds': 100., 'hard_seconds': 21600, 'command': tool.train_command(receipt, report, runtime['executable']),
                     'cwd': str(tool.owned.RUNTIME), 'runtime': runtime, 'input_pins': receipt['code_pins']})
        evidence[side] = checkpoint
        evidence[side + '_training'] = put(run / 'complete.json', receipt)
    return evidence, report


def test_actual_multiroot_arguments_and_pair_completion(completed: Any) -> None:
    evidence, report = completed
    assert tool.matched_training_pair(evidence) == ('Combined35M_V50', 'Combined35M_SF100')
    reference = tool.read_pin(evidence['reference_training'])
    command = tool.train_command(reference, report, '/frozen/python')
    assert command[command.index('--shards') + 1:command.index('--out-dir')] == report['trainer_shards']['B100']
    assert command[command.index('--seed') + 1] == '101'
    assert command[command.index('--epoch-plan-workers') + 1] == command[command.index('--epoch-load-workers') + 1] == '2'


@pytest.mark.parametrize('mutation', ['realization', 'seed', 'workers', 'window', 'roots', 'physical', 'columns', 'staging', 'command', 'process', 'role'])
def test_completed_evidence_rejects_wrong_schedule_or_execution(completed: Any, mutation: str) -> None:
    evidence, _ = completed
    receipt = tool.read_pin(evidence['candidate_training'])
    summary_path = Path(receipt['run']) / 'summary.json'
    summary = json.loads(summary_path.read_text())
    process = tool.read_pin(receipt['training_process'])
    if mutation == 'realization':
        summary['sampling']['realized_sha256'] = 'wrong'
    elif mutation == 'seed':
        summary['seed'] = 0
    elif mutation == 'workers':
        summary['sampling']['load_workers'] = 16
    elif mutation == 'window':
        summary['train_window_metrics'][0]['train_steps_done'] = 0
    elif mutation == 'roots':
        summary['corpus']['shard_dirs'].reverse()
    elif mutation == 'physical':
        receipt['physical_plan_sha256'] = 'wrong'
    elif mutation == 'columns':
        receipt['actual_game_columns'][0]['game_id_sha256'] = 'changed'
    elif mutation == 'staging':
        receipt['actual_staging_sha256'] = 'wrong'
    elif mutation == 'command':
        process['command'][process['command'].index('--shards') + 1] = '/source-proof-only'
    elif mutation == 'process':
        process['exit_code'] = 137
    else:
        receipt['role'] = 'B100'
    receipt['summary_sha256'] = put(summary_path, summary)['sha256']
    receipt['training_process'] = put(Path(receipt['training_process']['path']), process)
    with pytest.raises(ValueError, match=r'differs|incomplete|nonfinite|training'):
        tool.verify_completed(receipt)


def test_staging_rejects_swapped_same_basename_roots_and_extra(completed: Any) -> None:
    evidence, report = completed
    run = Path(evidence['reference']['path']).parent
    staged = run / 'staged_shards'
    tool.verify_staging(run, report, 'Combined35M_SF100')
    first, second = staged / 'shard_000000.zarr', staged / 'shard_000001.zarr'
    targets = first.resolve(), second.resolve()
    first.unlink()
    second.unlink()
    first.symlink_to(targets[1], target_is_directory=True)
    second.symlink_to(targets[0], target_is_directory=True)
    with pytest.raises(ValueError, match='target/order'):
        tool.verify_staging(run, report, 'Combined35M_SF100')
    first.unlink()
    second.unlink()
    first.symlink_to(targets[0], target_is_directory=True)
    second.symlink_to(targets[1], target_is_directory=True)
    (staged / 'unregistered').touch()
    with pytest.raises(ValueError, match='roster'):
        tool.verify_staging(run, report, 'Combined35M_SF100')


def test_fixed_package_requires_matched_training_and_pretraining_panel(completed: Any) -> None:
    evidence, _ = completed
    receipt = tool.read_pin(evidence['candidate_training'])
    contract = {**evidence, 'training': {k: v for k, v in evidence.items() if k.endswith('_training')},
                'pairs': 256, 'sims': 400, 'candidate_prior_temperature': 1., 'reference_prior_temperature': 1.,
                'opening_panel': receipt['opening_panel'], 'execution': {'loop': 'rolling', 'compile': 'on',
                    'eval_max_batch': 4096, 'max_concurrent_games': 128}}
    reader.verify_combined_training(contract)
    for key, value in [('pairs', 128), ('sims', 100), ('candidate_prior_temperature', .5),
                       ('opening_panel', {'path': '/another', 'sha256': 'b' * 64})]:
        changed = copy.deepcopy(contract)
        changed[key] = value
        with pytest.raises(ValueError, match=r'differs|incomplete|nonfinite|training'):
            reader.verify_combined_training(changed)


def test_optional_stage_guard_aborts_owned_child_and_preserves_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool.owned, 'disk_guard', lambda _: None)
    def reject() -> None:
        raise RuntimeError('simulated host headroom failure')
    with pytest.raises(RuntimeError, match='headroom'):
        tool.owned.run_owned_stage([sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path / 'stage',
                                  60, None, 'test', {}, manifest={}, cwd=tmp_path, env=dict(os.environ), guard=reject)
    failure = json.loads((tmp_path / 'stage/failed.json').read_text())
    assert failure['complete'] is False
    process = json.loads((tmp_path / 'stage/process.json').read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(process['supervisor_pid'], 0)


def test_default_plan_is_read_only_and_invalid_seed_rejects_before_outputs(completed: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    evidence, report = completed
    manifest = tool.read_pin(evidence['reference_training'])
    manifest.update(state=str(tmp_path / 'fresh_state'), run=str(tmp_path / 'fresh_run'),
                    training_seconds=21600, coordinator_seconds=27000, stop_paths=[str(tmp_path / 'STOP')])
    manifest['preregistration'] = put(tmp_path / 'prereg.json', {'registered': True})
    manifest['code_pins'] = {str(Path(p).resolve()): tool.owned.sha(p) for p in
                            (tool.__file__, tool.owned.__file__, tool.original.__file__, tool.memory.__file__, tool.schedule.__file__)}
    # Isolate frozen source bytes from this machine's actual historical runtime.
    monkeypatch.setattr(tool, 'FROZEN', {})
    path = tmp_path / 'launch.json'
    put(path, manifest)
    monkeypatch.setattr(sys, 'argv', ['combined_corpus_train', '--manifest', str(path)])
    tool.main()
    result = json.loads(capsys.readouterr().out)
    assert result['execute'] is False
    assert not Path(manifest['state']).exists()
    assert not Path(manifest['run']).exists()
    report['seed'] = 0
    manifest['prospective'] = put(Path(manifest['prospective']['path']), report)
    put(path, manifest)
    with pytest.raises(ValueError, match='seed differs'):
        tool.main()
    assert not Path(manifest['state']).exists()
    assert not Path(manifest['run']).exists()
