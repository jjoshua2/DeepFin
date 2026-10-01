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
        cohort: dict[str, Any] = {'id': f'c{i}', 'roots': {}, 'identity_kind': 'historical-single-source' if i == 0 else 'qualified-g10-selection'}
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
    manifest = put(tmp_path / 'corpus.json', {'kind': 'matched-b100-sf-native50-corpus-set', 'seed': 101, 'batch_size': 512,
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
    partial = {arm: {'partial': True, 'allow_partial_corpus': True, 'incomplete_shards': {
        Path(row['paths'][arm]).parent.name + '/' + Path(row['paths'][arm]).name: {'derive_run_finalized': True}
        for row in mapping[1:]}} for arm in ('B100', 'V50')}
    subset = put(tmp_path / 'subset.json', {'schema': 1, 'status': 'PASS_SELECTED_COMPLETE_UNION_FROZEN_PREFLIGHT',
        'corpus_manifest': manifest, 'prospective': prospective, 'runtime_manifest': runtime_pin, 'trainer_shards': roots,
        'partial_corpus': partial, 'flags': {'allow_partial_corpus': True, 'allow_leak': False, 'allow_mixed_history': False},
        'config': {'path': str(tool.owned.RUNTIME / 'configs/lc0_positive_control.yaml'),
                   'sha256': tool.FROZEN['configs/lc0_positive_control.yaml']}})
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
                   'steps_realized': 1, 'compute_loss_calls': 1, 'corpus': {'shard_dirs': roots[arm], 'partial_corpus': partial[arm]},
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
                   'runtime_manifest': runtime_pin, 'selected_subset_qualification': subset, 'preregistration': {'path': '/registered', 'sha256': 'a' * 64},
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
    assert '--allow-partial-corpus' in command
    assert '--allow-leak' not in command
    assert '--allow-mixed-history' not in command
    assert command[command.index('--epoch-plan-workers') + 1] == command[command.index('--epoch-load-workers') + 1] == '2'


@pytest.mark.parametrize('mutation', ['realization', 'seed', 'workers', 'window', 'roots', 'physical', 'columns', 'staging', 'partial_original', 'partial_unfinished', 'partial_summary', 'command', 'process', 'role'])
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
    elif mutation in ('partial_original', 'partial_unfinished'):
        qual = tool.read_pin(receipt['selected_subset_qualification'])
        entries = qual['partial_corpus']['V50']['incomplete_shards']
        if mutation == 'partial_original':
            entries['V50_0/shard_000000.zarr'] = {'derive_run_finalized': True}
        else:
            entries[next(iter(entries))]['derive_run_finalized'] = False
        receipt['selected_subset_qualification'] = put(Path(receipt['selected_subset_qualification']['path']), qual)
    elif mutation == 'partial_summary':
        summary['corpus']['partial_corpus']['allow_partial_corpus'] = False
    elif mutation == 'command':
        process['command'][process['command'].index('--shards') + 1] = '/source-proof-only'
    elif mutation == 'process':
        process['exit_code'] = 137
    else:
        receipt['role'] = 'B100'
    receipt['summary_sha256'] = put(summary_path, summary)['sha256']
    receipt['training_process'] = put(Path(receipt['training_process']['path']), process)
    with pytest.raises(ValueError, match=r'differs|incomplete|nonfinite|training|partial|finalized'):
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
    contract = {**evidence, 'profile': tool.PROFILE, 'training': {k: v for k, v in evidence.items() if k.endswith('_training')},
                'pairs': 256, 'sims': 400, 'seed': 20260913, 'candidate_prior_temperature': 1., 'reference_prior_temperature': 1.,
                'opening_panel': receipt['opening_panel'], 'execution': {'loop': 'rolling', 'compile': 'on',
                    'eval_max_batch': 4096, 'max_concurrent_games': 128}}
    reader.verify_combined_training(contract)
    for key, value in [('pairs', 128), ('sims', 100), ('seed', 42), ('candidate_prior_temperature', .5),
                       ('opening_panel', {'path': '/another', 'sha256': 'b' * 64})]:
        changed = copy.deepcopy(contract)
        changed[key] = value
        with pytest.raises(ValueError, match=r'differs|incomplete|nonfinite|training|partial|finalized'):
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
    real_pin = tool.owned.pin
    monkeypatch.setattr(tool.owned, 'pin', lambda path, digest: None if str(path).startswith(str(tool.owned.RUNTIME)) else real_pin(path, digest))
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


def add_value_masks(summary: dict[str, Any]) -> None:
    summary['realized'] = {
        'sf_wdl_frac (realized)': 0., 'search_wdl_frac (realized)': 1.,
        'game_frac (intended outcome share)': 0., 'sf_effective_frac (labelled sf mass)': 0.,
        'search_effective_frac (labelled search mass)': 1., 'leaked_from_sf': 0.,
        'leaked_from_search': 0., 'leaked_to_outcome': 0., 'outcome_borne_frac (game_frac + leak)': 0.,
    }
    summary['realized_categorical'] = {'rebuild_categorical_target (realized)': 1.,
        'categorical_target column present': 0., 'rebuild applies to this batch': 0.,
        'sf_labelled_frac': 0., 'search_labelled_frac': 1., 'categorical outcome_borne_frac': 0.}
    for window in summary['train_window_metrics']:
        window.update(categorical_loss=0., sf_eval_loss=0.)


@pytest.fixture
def native_completed(completed: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence, old_report = completed
    prior = tool.read_pin(evidence['candidate_training'])
    prior_summary_path = Path(prior['run']) / 'summary.json'
    prior_summary = json.loads(prior_summary_path.read_text())
    add_value_masks(prior_summary)
    prior['summary_sha256'] = put(prior_summary_path, prior_summary)['sha256']
    prior_pin = put(Path(evidence['candidate_training']['path']), prior)
    # Exercise the real completed-evidence checks in the reference verifier below.
    monkeypatch.setattr(tool, 'legacy_v50', lambda receipt, ref: tool.verify_completed(receipt))
    manifest = tool.read_pin(prior['corpus_manifest'])
    manifest['kind'] = tool.schedule.V100_KIND
    report = copy.deepcopy(old_report)
    report['training_role_to_arm'] = {tool.V100_ROLE: 'V100'}
    report['trainer_shards'] = {'V100': []}
    for cohort, row in zip(manifest['cohorts'], report['mapping'], strict=True):
        root = tmp_path / ('V100_' + cohort['id'])
        root.mkdir()
        (root / 'shard_000000.zarr').mkdir()
        cohort['roots']['V100'] = {'summary': put(root / 'derive_targets_summary.json', {'rows': 1})}
        del cohort['roots']['V50']
        row['paths']['V100'] = str(root / 'shard_000000.zarr')
        del row['paths']['V50']
        report['trainer_shards']['V100'].append(str(root))
    manifest_pin = put(tmp_path / 'native_corpus.json', manifest)
    report['manifest_sha256'] = manifest_pin['sha256']
    report['arms']['V100'] = report['arms'].pop('V50')
    report['arms']['V100']['physical_plan']['plan_sha256'] = 'native-physical'
    report_pin = put(tmp_path / 'native_prospective.json', report)
    partial = {'partial': True, 'allow_partial_corpus': True, 'incomplete_shards': {
        Path(row['paths']['V100']).parent.name + '/shard_000000.zarr': {'derive_run_finalized': True}
        for row in report['mapping'][1:]}}
    qualification = tool.read_pin(prior['selected_subset_qualification'])
    qualification.update(corpus_manifest=manifest_pin, prospective=report_pin,
        trainer_shards=report['trainer_shards'], partial_corpus={'V100': partial},
        coverage={'V100': {'search_wdl': {'rows': 21, 'labelled_rows': 21}, 'sf_wdl': {'rows': 21, 'labelled_rows': 0}}})
    qualification_pin = put(tmp_path / 'native_subset.json', qualification)
    receipt = copy.deepcopy(prior)
    run = tmp_path / tool.V100_ROLE
    run.mkdir()
    (run / 'staged_shards').mkdir()
    for i, row in enumerate(report['mapping']):
        (run / 'staged_shards' / f'shard_{i:06d}.zarr').symlink_to(row['paths']['V100'], target_is_directory=True)
    receipt.update(role=tool.V100_ROLE, run=str(run), corpus_manifest=manifest_pin, prospective=report_pin,
        selected_subset_qualification=qualification_pin, previous_training=prior_pin,
        previous_verifier={'path': '/historical/verifier.py', 'sha256': tool.LEGACY_VERIFIER_SHA},
        physical_plan_sha256='native-physical', actual_staging_sha256=tool.verify_staging(run, report, tool.V100_ROLE))
    receipt['checkpoint'] = {'role': tool.V100_ROLE, 'path': str(run / 'checkpoint.pt'), 'sha256': 'native-checkpoint'}
    summary = copy.deepcopy(prior_summary)
    summary['corpus'] = {'shard_dirs': report['trainer_shards']['V100'], 'partial_corpus': partial}
    summary['sampling'].update(plan_sha256='native-physical', realized_sha256='native-physical')
    summary['checkpoints'] = [{**receipt['checkpoint'], 'role': 'last'}]
    receipt['summary_sha256'] = put(run / 'summary.json', summary)['sha256']
    process = tool.read_pin(prior['training_process'])
    process['command'] = tool.train_command(receipt, report, '/frozen/python')
    receipt['training_process'] = put(run / 'process.json', process)
    return receipt, report


def test_native_completed_pair_keeps_actual_reference_and_masks(native_completed: Any, tmp_path: Path) -> None:
    receipt, report = native_completed
    assert tool.verify_completed(receipt) == report
    reference = tool.read_pin(receipt['previous_training'])
    evidence = {'candidate': receipt['checkpoint'], 'reference': reference['checkpoint'],
        'candidate_training': put(tmp_path / 'native_complete.json', receipt), 'reference_training': receipt['previous_training']}
    assert tool.matched_training_pair(evidence) == (tool.V100_ROLE, 'Combined35M_V50')
    command = tool.train_command(receipt, report, '/frozen/python')
    assert command[command.index('--shards') + 1:command.index('--out-dir')] == report['trainer_shards']['V100']


@pytest.mark.parametrize('mutation', ['outcome', 'categorical', 'missing_loss', 'coverage', 'columns', 'policy_root', 'role'])
def test_native_rejects_changed_supervision_or_source(native_completed: Any, mutation: str) -> None:
    receipt, report = native_completed
    summary_path = Path(receipt['run']) / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if mutation == 'outcome':
        summary['realized']['outcome_borne_frac (game_frac + leak)'] = .1
    elif mutation == 'categorical':
        summary['realized_categorical']['categorical_target column present'] = 1.
    elif mutation == 'missing_loss':
        del summary['train_window_metrics'][0]['sf_eval_loss']
    elif mutation == 'coverage':
        q = tool.read_pin(receipt['selected_subset_qualification'])
        q['coverage']['V100']['search_wdl']['labelled_rows'] = 20
        receipt['selected_subset_qualification'] = put(Path(receipt['selected_subset_qualification']['path']), q)
    elif mutation == 'columns':
        report['ordered_source_columns'][0]['game_id_sha256'] = 'other-game'
        receipt['prospective'] = put(Path(receipt['prospective']['path']), report)
    elif mutation == 'policy_root':
        manifest = tool.read_pin(receipt['corpus_manifest'])
        manifest['cohorts'][0]['roots']['B100']['summary']['path'] = '/other/policy/derive_targets_summary.json'
        receipt['corpus_manifest'] = put(Path(receipt['corpus_manifest']['path']), manifest)
        report['manifest_sha256'] = receipt['corpus_manifest']['sha256']
        receipt['prospective'] = put(Path(receipt['prospective']['path']), report)
    else:
        receipt['role'] = 'Combined35M_V50'
    receipt['summary_sha256'] = put(summary_path, summary)['sha256']
    with pytest.raises(ValueError, match=r'differs|admitted'):
        tool.verify_completed(receipt)


def test_historical_bridge_requires_pinned_source_and_real_completed_checks(completed: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence, report = completed
    receipt = tool.read_pin(evidence['candidate_training'])
    # Portable historical-shaped module: the actual completed-evidence consumer,
    # not a stub success function. Production accepts only the fixed historical SHA.
    module = tmp_path / 'historical.py'
    module.write_text('from scripts.combined_corpus_train import verify_completed\n')
    ref = tool.pin(module)
    monkeypatch.setattr(tool, 'LEGACY_VERIFIER_SHA', ref['sha256'])
    receipt['code_pins'][ref['path']] = ref['sha256']
    process = tool.read_pin(receipt['training_process'])
    process['input_pins'] = receipt['code_pins']
    receipt['training_process'] = put(Path(receipt['training_process']['path']), process)
    assert tool.legacy_v50(receipt, ref) == report
    process['exit_code'] = 1
    receipt['training_process'] = put(Path(receipt['training_process']['path']), process)
    with pytest.raises(ValueError, match='process incomplete'):
        tool.legacy_v50(receipt, ref)
    with pytest.raises(ValueError, match='verifier source'):
        tool.legacy_v50(receipt, {**ref, 'sha256': '0' * 64})


def test_expansion_requires_exact_historical_prefix_and_audited_extras(completed: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence, _ = completed
    prior = tool.read_pin(evidence['candidate_training'])
    old = tool.read_pin(prior['corpus_manifest'])
    new = copy.deepcopy(old)
    new['kind'] = tool.schedule.EXPANSION_KIND
    new['cohorts'].append({'id': 'new', 'identity_kind': 'audited-g10-selection'})
    m = {'previous_training': evidence['candidate_training'], 'previous_verifier': {},
         'opening_panel': prior['opening_panel'], 'runtime_manifest': prior['runtime_manifest'],
         'corpus_manifest': put(tmp_path / 'expanded.json', new)}
    verified = []
    monkeypatch.setattr(tool, 'legacy_v50', lambda receipt, ref: verified.append(receipt))
    tool.expansion_predecessor(m)
    assert verified == [prior]
    new['cohorts'][1]['roots']['V50']['summary']['sha256'] = 'retargeted'
    m['corpus_manifest'] = put(tmp_path / 'expanded.json', new)
    with pytest.raises(ValueError, match='preserved35M'):
        tool.expansion_predecessor(m)
    new['cohorts'] = [*copy.deepcopy(old['cohorts']), {'id': 'new', 'identity_kind': 'qualified-g10-selection'}]
    m['corpus_manifest'] = put(tmp_path / 'expanded.json', new)
    with pytest.raises(ValueError, match='new audited'):
        tool.expansion_predecessor(m)


def test_expansion_budget_is_explicit_and_old_role_cannot_use_it() -> None:
    m = {'profile': tool.EXPANSION_PROFILE, 'role': tool.EXPANSION_ROLE}
    assert tool.budgets(m) == (32400, 43200)
    assert tool.arm_for(tool.EXPANSION_ROLE) == 'V50'
    m['role'] = 'Combined35M_V50'
    with pytest.raises(ValueError, match='expanded role'):
        tool.budgets(m)
    assert tool.budgets({'profile': tool.PROFILE, 'role': 'Combined35M_V50'}) == (21600, 27000)


def test_expansion_admits_measured_rows_but_refuses_raw_upper_bound(completed: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence, report = completed
    prior = tool.read_pin(evidence['candidate_training'])
    manifest = tool.read_pin(prior['corpus_manifest'])
    manifest['kind'] = tool.schedule.EXPANSION_KIND
    rows = 50_000_003
    extra = {'id': 'new', 'identity_kind': 'audited-g10-selection', 'roots': {
        arm: {'summary': {'path': str(tmp_path / ('new_' + arm) / 'derive_targets_summary.json')}}
        for arm in ('source', 'B100', 'V50')}}
    manifest['cohorts'].append(extra)
    manifest.update(expected_rows=rows, expected_shards=22)
    row = {'cohort': 'new', 'rows': rows - 21, 'paths': {
        arm: str(tmp_path / ('new_' + arm) / 'shard_000000.zarr') for arm in ('source', 'B100', 'V50')}}
    report['mapping'].append(row)
    report['training_role_to_arm'] = {tool.EXPANSION_ROLE: 'V50'}
    report['trainer_shards'] = {'V50': [*report['trainer_shards']['V50'], str(tmp_path / 'new_V50')]}
    report['rows'] = rows
    report['arms']['V50']['physical_plan'].update(rows_planned=rows, shards=22)
    m: dict[str, Any] = {**prior, 'profile': tool.EXPANSION_PROFILE, 'role': tool.EXPANSION_ROLE,
         'previous_training': evidence['candidate_training'], 'previous_verifier': {},
         'corpus_manifest': put(tmp_path / 'expanded.json', manifest)}
    report['manifest_sha256'] = m['corpus_manifest']['sha256']
    m['prospective'] = put(tmp_path / 'expanded_report.json', report)
    monkeypatch.setattr(tool, 'legacy_v50', lambda *args: {})
    assert tool.admission(m)['rows'] == rows
    # Merely increasing expected_rows without retained-row coverage is rejected.
    manifest['expected_rows'] += 100
    m['corpus_manifest'] = put(tmp_path / 'expanded.json', manifest)
    report['manifest_sha256'] = m['corpus_manifest']['sha256']
    m['prospective'] = put(tmp_path / 'expanded_report.json', report)
    with pytest.raises(ValueError, match='registered combined rows'):
        tool.admission(m)
    manifest['expected_rows'] = 49_999_999
    m['corpus_manifest'] = put(tmp_path / 'expanded.json', manifest)
    report['manifest_sha256'] = m['corpus_manifest']['sha256']
    m['prospective'] = put(tmp_path / 'expanded_report.json', report)
    with pytest.raises(ValueError, match='expansion row budget'):
        tool.admission(m)
