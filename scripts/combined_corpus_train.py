#!/usr/bin/env python3
"""Explicit seed-101 combined value comparison; inspect by default, no resume.

The prospective proof supplies dimensions, never permission to bypass the frozen
trainer's history, value-source or encoding gates. Completion verifies the actual
staged links and realized physical schedule before inheriting canonical identity.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import bt4_direct_screen as owned
from scripts import bt4_one_epoch_screen as original
from scripts import training_host_memory as memory
from scripts import combined_corpus_schedule as schedule

PROFILE = 'combined35m_value_seed101'
ROLES = {'Combined35M_SF100': 'B100', 'Combined35M_V50': 'V50'}
ROWS = 35314577
PANEL_SHA = '14470ee9bcf5fdfc822bb19988941ecf4bbe2a1ea739d46487d3663b1735340c'
FROZEN = {
    'scripts/lc0_control_train.py': '52d1132689c1cd53a23b63c9274226b467bd9aafdc34548a18db121a03bf9337',
    'configs/lc0_positive_control.yaml': '413dbea9dcde2774eafc2fde706e639fef9e944e301717b938b39b4729633de2',
    'chess_anti_engine/replay/game_epoch.py': '621e5d0764e62cee492688e63e4099ff8cbc0d39ea094b252c3cae31cd74fde3',
}
require = owned.require


def pin(path: Path) -> dict[str, str]:
    return {'path': str(path.resolve()), 'sha256': owned.sha(path)}


def read_pin(item: dict[str, Any]) -> Any:
    require(Path(item['path']).is_absolute(), 'relative evidence path')
    owned.pin(item['path'], item['sha256'])
    return owned.read(item['path'])


def same(actual: Any, expected: Any, label: str) -> None:
    require(json.dumps(actual, sort_keys=True, allow_nan=False)
            == json.dumps(expected, sort_keys=True, allow_nan=False), label + ' differs')


def admission(m: dict[str, Any]) -> dict[str, Any]:
    """Small pinned artifacts only; no corpus admission or game-column rerun."""
    manifest = read_pin(m['corpus_manifest'])
    report = read_pin(m['prospective'])
    same(report['status'], 'PASS_CORPUS_SET_PROSPECTIVE_NOT_TRAINING', 'prospective status')
    same(report['manifest_sha256'], m['corpus_manifest']['sha256'], 'manifest binding')
    same(report['training_role_to_arm'], ROLES, 'scientific role mapping')
    for value in (manifest, report):
        same(value['seed'], 101, 'seed')
        same(value['batch_size'], 512, 'batch size')
    same(report['rows'], ROWS, 'registered combined rows')
    same(manifest['expected_rows'], ROWS, 'manifest rows')
    same(len(manifest['cohorts']), 21, 'logical cohort count')
    mapping = report['mapping']
    same(len(mapping), manifest['expected_shards'], 'mapping shard count')
    same(sum(row['rows'] for row in mapping), ROWS, 'mapping row count')
    require(all(type(row['rows']) is int and row['rows'] > 0 for row in mapping), 'invalid shard rows')
    for arm in ROLES.values():
        roots = report['trainer_shards'][arm]
        same(roots, [str(Path(c['roots'][arm]['summary']['path']).parent) for c in manifest['cohorts']], 'ordered training roots')
        require(len(set(roots)) == 21 and all(Path(root).is_absolute() for root in roots), 'invalid roots')
        paths = [row['paths'][arm] for row in mapping]
        require(len(paths) == len(set(paths)) and all(Path(p).is_absolute() for p in paths), 'aliased shard mapping')
        require(list(dict.fromkeys(str(Path(p).parent) for p in paths)) == roots, 'ordered cohort mapping differs')
        plan = report['arms'][arm]['physical_plan']
        same({k: plan[k] for k in ('seed', 'batch_size', 'rows_planned', 'shards')},
             {'seed': 101, 'batch_size': 512, 'rows_planned': ROWS, 'shards': len(mapping)}, 'prospective dimensions')
        require(report['arms'][arm]['metadata_matches_source'] is True
                and report['arms'][arm]['training_completed'] is False, 'prospective claim differs')
        same(report['arms'][arm]['canonical_plan_sha256'], report['arms']['source']['physical_plan']['plan_sha256'],
             'canonical source schedule')
    runtime = report['runtime']
    require(runtime['python'].startswith('3.10.12') and runtime['numpy'] == '1.26.2'
            and runtime['torch'] == '2.11.0+cu128', 'unqualified prospective runtime')
    return report


def validate(m: dict[str, Any], *, fresh: bool = True) -> dict[str, Any]:
    same(m['schema'], 1, 'schema')
    same(m['profile'], PROFILE, 'profile')
    require(m['role'] in ROLES, 'unknown combined role')
    same(m['training_seconds'], 21600, 'training cap')
    same(m['coordinator_seconds'], 27000, 'coordinator cap')
    require(isinstance(m['stop_paths'], list) and m['stop_paths']
            and all(isinstance(p, str) and Path(p).is_absolute() for p in m['stop_paths']), 'missing/relative STOP paths')
    for key in ('state', 'run'):
        path = Path(m[key])
        require(path.is_absolute() and path == path.resolve() and not path.is_symlink()
                and path.parent.is_dir(), 'invalid output parent')
        require(not fresh or not path.exists(), 'existing output; no automatic resume')
    require(Path(m['state']) != Path(m['run']) and Path(m['state']) not in Path(m['run']).parents
            and Path(m['run']) not in Path(m['state']).parents, 'overlapping outputs')
    same(m['opening_panel']['sha256'], PANEL_SHA, 'prelaunch opening panel')
    panel = read_pin(m['opening_panel'])
    same(len(panel), 256, 'prelaunch opening pairs')
    owned.pin(m['preregistration']['path'], m['preregistration']['sha256'])
    for module in (__file__, owned.__file__, original.__file__, memory.__file__, schedule.__file__):
        owned.pin(module, m['code_pins'][str(Path(module).resolve())])
    for suffix, digest in FROZEN.items():
        owned.pin(owned.RUNTIME / suffix, digest)
    report = admission(m)
    if m['role'] == 'Combined35M_V50':
        prior = read_pin(m['previous_training'])
        verify_completed(prior)
        same(prior['role'], 'Combined35M_SF100', 'sequential SF100 predecessor')
        for key in ('corpus_manifest', 'prospective', 'opening_panel', 'runtime_manifest', 'preregistration'):
            same(prior[key], m[key], 'predecessor ' + key)
    else:
        require('previous_training' not in m, 'SF100 must be the first fresh arm')
    return report


def train_command(m: dict[str, Any], report: dict[str, Any], python: str) -> list[str]:
    return [python, 'scripts/lc0_control_train.py', '--config', 'configs/lc0_positive_control.yaml',
            '--shards', *report['trainer_shards'][ROLES[m['role']]], '--out-dir', m['run'],
            '--steps', '0', '--batch-size', '512', '--sampling-mode', 'game_epoch',
            '--epoch-plan-workers', '2', '--epoch-load-workers', '2', '--seed', '101',
            '--device', 'cuda', '--train-window-steps', '88', '--allow-invalid-control']


def summary_contract(summary: dict[str, Any], report: dict[str, Any], role: str) -> None:
    arm = ROLES[role]
    plan = report['arms'][arm]['physical_plan']
    expected = {**plan, 'complete': True, 'rows_realized': plan['rows_planned'],
                'batches_realized': plan['batches_planned'], 'plan_workers': 2, 'load_workers': 2,
                'same_game_repeats_max': 0, 'decoded_rows_resident': 0,
                'realized_sha256': plan['plan_sha256']}
    for key, value in expected.items():
        same(summary['sampling'].get(key), value, 'actual sampling ' + key)
    for key, value in {'seed': 101, 'batch_size': 512, 'warmup_steps': 1000, 'train_window_steps': 88,
                       'steps_realized': plan['batches_planned'], 'compute_loss_calls': plan['batches_planned']}.items():
        same(summary.get(key), value, 'actual training ' + key)
    same(summary['corpus']['shard_dirs'], report['trainer_shards'][arm], 'actual ordered training roots')
    original.verify_window_cadence(summary)
    require(all(w['grad_nonfinite_skip_rate'] == 0 and w['transient_cuda_retry_batches'] == 0
                and math.isfinite(w['loss']) and math.isfinite(w['grad_norm_mean'])
                for w in summary['train_window_metrics']), 'nonfinite, skipped or retried training')


def verify_staging(run: Path, report: dict[str, Any], role: str) -> str:
    """Exact links/order, including foreign extras; never open a shard payload."""
    expected = [Path(row['paths'][ROLES[role]]) for row in report['mapping']]
    staged = run / 'staged_shards'
    actual = sorted(staged.iterdir())
    same([p.name for p in actual], [f'shard_{i:06d}.zarr' for i in range(len(expected))], 'actual staged roster')
    require(all(p.is_symlink() and p.resolve(strict=True) == target.resolve(strict=True)
                for p, target in zip(actual, expected)), 'actual staged target/order differs')
    return hashlib.sha256(json.dumps([str(p.resolve()) for p in actual], separators=(',', ':')).encode()).hexdigest()


def verify_actual_columns(run: Path, report: dict[str, Any], role: str) -> dict[str, Any]:
    roster_sha = verify_staging(run, report, role)
    require('chess_anti_engine.replay.game_epoch' not in sys.modules, 'sampler already imported')
    for suffix, digest in FROZEN.items():
        owned.pin(owned.RUNTIME / suffix, digest)
    sys.path.insert(0, str(owned.RUNTIME))
    torch = importlib.import_module('torch')
    torch.set_num_threads(2)
    epoch = importlib.import_module('chess_anti_engine.replay.game_epoch')
    require(Path(str(epoch.__file__)).resolve() == owned.RUNTIME / 'chess_anti_engine/replay/game_epoch.py', 'wrong actual sampler')
    paths = sorted((run / 'staged_shards').iterdir())
    def guard() -> None:
        memory.require_available(memory.RUNNING_GIB)
        owned.disk_guard(run)
    records, columns = schedule.scan_columns(epoch, paths, report['mapping'], guard)
    schedule.canonical_records(records, report['mapping'], ROLES[role])
    same(columns, report['ordered_source_columns'], 'realized full ordered game columns')
    same(verify_staging(run, report, role), roster_sha, 'staging stability')
    return {'actual_staging_sha256': roster_sha, 'actual_game_columns': columns}


def completed_training(m: dict[str, Any], report: dict[str, Any], charge: dict[str, Any]) -> dict[str, Any]:
    run, role = Path(m['run']), m['role']
    summary = owned.read(run / 'summary.json')
    summary_contract(summary, report, role)
    realized = verify_actual_columns(run, report, role)
    roster_sha = realized['actual_staging_sha256']
    checkpoint = {'role': role, **pin(run / 'checkpoint.pt')}
    require(any(c['role'] == 'last' and c['path'] == checkpoint['path'] and c['sha256'] == checkpoint['sha256']
                for c in summary['checkpoints']), 'last checkpoint differs')
    plan = report['arms'][ROLES[role]]
    receipt = {'schema': 1, 'profile': PROFILE, 'complete': True, 'role': role, 'run': str(run),
               'checkpoint': checkpoint, 'summary_sha256': owned.sha(run / 'summary.json'),
               'canonical_plan_sha256': plan['canonical_plan_sha256'],
               'physical_plan_sha256': plan['physical_plan']['plan_sha256'],
               'actual_staging_sha256': roster_sha, 'actual_staging_verified': True,
               'actual_game_columns_verified': True, 'actual_game_columns': realized['actual_game_columns'],
               'training_charge_seconds': charge['gpu_seconds'],
               'training_process': pin(Path(m['state']) / 'training/process.json'),
               'input_pins': {str(owned.RUNTIME / k): v for k, v in FROZEN.items()},
               'historical_valid_control': summary['valid_control'],
               'historical_validity_problems': summary['validity_problems'],
               'proof': 'Exact actual staged-link order and frozen trainer physical plan/realization equal the admitted prospective arm; canonical identity is inherited from its game-column proof. No feature/target revalidation.'}
    receipt.update({k: m[k] for k in ('corpus_manifest', 'prospective', 'opening_panel', 'preregistration', 'runtime_manifest', 'code_pins')})
    return receipt


def verify_completed(receipt: dict[str, Any]) -> dict[str, Any]:
    """Match admission reads small completion evidence, never models or corpora."""
    require(receipt['complete'] is True and receipt['profile'] == PROFILE and receipt['role'] in ROLES,
            'incomplete or foreign combined training')
    report = admission(receipt)
    role, run = receipt['role'], Path(receipt['run'])
    require(run.is_absolute(), 'relative completed run')
    summary = read_pin({'path': str(run / 'summary.json'), 'sha256': receipt['summary_sha256']})
    summary_contract(summary, report, role)
    same(receipt['physical_plan_sha256'], report['arms'][ROLES[role]]['physical_plan']['plan_sha256'], 'completed physical plan')
    same(receipt['canonical_plan_sha256'], report['arms'][ROLES[role]]['canonical_plan_sha256'], 'completed canonical plan')
    require(receipt['actual_staging_verified'] is True and receipt['actual_game_columns_verified'] is True,
            'actual staging/game-column proof missing')
    same(receipt['actual_game_columns'], report['ordered_source_columns'], 'actual ordered game columns')
    expected_paths = [str(Path(row['paths'][ROLES[role]]).resolve()) for row in report['mapping']]
    expected_roster_sha = hashlib.sha256(json.dumps(expected_paths, separators=(',', ':')).encode()).hexdigest()
    same(receipt['actual_staging_sha256'], expected_roster_sha, 'recorded actual staging mapping')
    charge = receipt['training_charge_seconds']
    require(type(charge) in (int, float) and math.isfinite(charge) and 0 < charge <= 21600, 'training cap exceeded')
    process = read_pin(receipt['training_process'])
    require(process['process_complete'] is True and process['exit_code'] == 0, 'training process incomplete')
    same(process['gpu_seconds'], charge, 'process training charge')
    same(process['hard_seconds'], 21600, 'process budget')
    rt = read_pin(receipt['runtime_manifest'])['runtime']
    same(process['command'], train_command(receipt, report, rt['executable']), 'actual training command')
    same(process['cwd'], str(owned.RUNTIME), 'actual training runtime')
    same(process['runtime'], {k: v for k, v in rt.items() if k != 'native_extension_sha256'}, 'actual runtime versions')
    same(process['input_pins'], receipt['code_pins'], 'actual coordinator dependencies')
    owned.pin(__file__, receipt['code_pins'][str(Path(__file__).resolve())])
    for suffix, digest in FROZEN.items():
        same(receipt['input_pins'].get(str(owned.RUNTIME / suffix)), digest, 'frozen training pin')
    checkpoint = receipt['checkpoint']
    same(checkpoint['role'], role, 'checkpoint role')
    same(checkpoint['path'], str(run / 'checkpoint.pt'), 'checkpoint path')
    require(any(c['role'] == 'last' and c['path'] == checkpoint['path'] and c['sha256'] == checkpoint['sha256']
                for c in summary['checkpoints']), 'completed summary checkpoint differs')
    for key, source in (('historical_valid_control', 'valid_control'), ('historical_validity_problems', 'validity_problems')):
        same(receipt[key], summary[source], 'historical limitations')
    same(receipt['opening_panel']['sha256'], PANEL_SHA, 'frozen panel')
    read_pin(receipt['opening_panel'])
    return report


def matched_training_pair(evidence: dict[str, Any]) -> tuple[str, str]:
    roles = ('Combined35M_V50', 'Combined35M_SF100')
    receipts = []
    for side, role in zip(('candidate', 'reference'), roles):
        receipt = read_pin(evidence[side + '_training'])
        verify_completed(receipt)
        same(receipt['role'], role, 'combined match direction')
        same(receipt['checkpoint'], evidence[side], 'combined match checkpoint')
        receipts.append(receipt)
    for key in ('corpus_manifest', 'prospective', 'canonical_plan_sha256', 'opening_panel', 'preregistration', 'runtime_manifest'):
        same(receipts[0][key], receipts[1][key], 'matched ' + key)
    require(evidence['candidate']['path'] != evidence['reference']['path']
            and evidence['candidate']['sha256'] != evidence['reference']['sha256'], 'candidate is reference')
    return roles


def execute(m: dict[str, Any]) -> None:
    started = time.monotonic()
    report = validate(m)
    memory.require_available(memory.STARTUP_GIB)
    rt = owned.runtime_identity(m['runtime_manifest'])
    actual_runtime = original.training_runtime_probe(rt)
    state = Path(m['state'])
    def guard() -> None:
        require(time.monotonic() - started < 26940, 'coordinator budget exhausted')
        require(not any(Path(p).exists() for p in [str(state / 'STOP'), *m['stop_paths']]), 'STOP requested')
        owned.disk_guard(state)
        memory.require_available(memory.RUNNING_GIB)
    guard()
    state.mkdir(exist_ok=False)
    owned.write(state / 'manifest.json', m)
    try:
        with (owned.ROOT / 'scratchpad/gpu0_experiment.lock').open('a') as lease:
            while True:
                guard()
                require(time.monotonic() - started + 21600 + 1800 + 60 <= 27000, 'insufficient training/completion allowance')
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(2)
            validate(m, fresh=False)
            require(not Path(m['run']).exists(), 'run appeared while waiting')
            require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                                text=True, timeout=10).strip(), 'competing GPU process')
            env = owned.environment(gpu=True)
            for key in ('PYTHONOPTIMIZE', 'PYTHONHOME', 'LD_PRELOAD'):
                env.pop(key, None)
            charge = owned.run_owned_stage(train_command(m, report, rt['executable']), state / 'training', 21600,
                lease.fileno(), 'training', {'runtime': actual_runtime, 'input_pins': m['code_pins']},
                manifest=m, stop_paths=(*(Path(p) for p in m['stop_paths']), state / 'STOP'),
                cwd=owned.RUNTIME, env=env, guard=guard)
        guard()
        # Input proof/manifest pins remain fixed; only metadata links and summary
        # are inspected after training, not a second full corpus pass.
        same(admission(m), report, 'prospective stability')
        command = ['/usr/bin/nice', '-n', '19', '/usr/bin/ionice', '-c', '3', '/usr/bin/taskset', '-c', '0,1',
                   rt['executable'], str(Path(__file__).resolve()), '--manifest', str(state / 'manifest.json'), '--verify-completed']
        owned.run_owned_stage(command, state / 'completion', 1800, None, 'completion', {}, manifest=m,
            stop_paths=(*(Path(p) for p in m['stop_paths']), state / 'STOP'),
            cwd=Path(__file__).resolve().parents[1], env=owned.environment(), guard=guard)
        receipt = owned.read(state / 'training.complete.json')
        verify_completed(receipt)
        same(receipt['training_charge_seconds'], charge['gpu_seconds'], 'owned completion charge')
        guard()
        owned.write(state / 'complete.json', {'complete': True, 'scope': 'training_only',
                    'training_receipt': pin(state / 'training.complete.json'), 'promotion': 'NONE'})
    except BaseException as error:
        owned.write(state / 'failed.json', {'complete': False, 'error': str(error), 'partial_artifacts_preserved': True})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--verify-completed', action='store_true')
    args = parser.parse_args()
    require(not sys.flags.optimize and os.environ.get('PYTHONOPTIMIZE') in (None, '', '0'), 'optimized Python refused')
    m = owned.read(args.manifest)
    def interrupted(signum: int, _frame: Any) -> None:
        raise RuntimeError(f'coordinator interrupted: {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(signum, interrupted)
    if args.verify_completed:
        signal.alarm(1770)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'completion must hide GPU')
        report = validate(m, fresh=False)
        state = Path(m['state'])
        require(not (state / 'training.complete.json').exists(), 'completion already exists')
        receipt = completed_training(m, report, owned.read(state / 'training/process.json'))
        verify_completed(receipt)
        owned.write(state / 'training.complete.json', receipt)
    elif args.execute:
        signal.alarm(26970)
        execute(m)
    else:
        report = validate(m)
        print(json.dumps({'execute': False, 'command': train_command(m, report,
                         read_pin(m['runtime_manifest'])['runtime']['executable'])}, indent=2))


if __name__ == '__main__':
    main()
