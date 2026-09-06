#!/usr/bin/env python3
"""One registered E0T05 epoch followed by its direct C20T05 screen; plan by default.

Existing published targets only. No mixing, resume, retries or automatic promotion.
The prospective and realized schedule checks use the frozen seed-zero verifier.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import subprocess

import bt4_direct_screen as arena

ROOT = arena.ROOT
CORPUS = ROOT / 'data/nnue_derived/armB/qtemp_0.0005_hist_20m_bt4_toptie_t050'
SOURCE = ROOT / 'data/nnue_derived/armB/qtemp_0.0005_hist_20m'
VERIFIER = ROOT / 'scratchpad/bt4_joint20/global_run02/verify_matched_schedule.py'
CANONICAL = 'dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f'
C_ROLE, C_CHECKPOINT, C_SHA = arena.CHECKPOINTS['candidate']
INPUT_PINS = {
    str(SOURCE / 'derive_targets_summary.json'): '391837e49773465edced77bfd13f4084edc60feeff0484078280873d942e50ef',
    str(CORPUS / 'bt4_policy_mix_summary.json'): 'a85ba1403c2477018bdc59b622311094027ca17cf045dc78090f6c0ee9f5463d',
    str(CORPUS / 'derive_targets_summary.json'): '4354bafe8e1435bb030faaa161810bbfff667a2b83c647940ad371eb2ab1caf1',
    str(VERIFIER): '4e8e27e861021dd4c75a1a21404ff9aed0d55775e0212def585bc21b1d8cde18',
    str(C_CHECKPOINT.parent / 'summary.json'): 'cacc3652db3deb2bcd12cf69b0493d3f48d7ef2555c3ea4dc7502fedff3313fd',
    str(ROOT / 'scratchpad/bt4_joint20/sf_close_run02/optional_sharpened_ties_data_qualification.json'):
        '48897206e4af6d37bb4fa351cebe1e1d4b493e07b63126fd8a65332bd85af8c1',
    str(arena.RUNTIME / 'scripts/lc0_control_train.py'): '52d1132689c1cd53a23b63c9274226b467bd9aafdc34548a18db121a03bf9337',
    str(arena.RUNTIME / 'configs/lc0_positive_control.yaml'): '413dbea9dcde2774eafc2fde706e639fef9e944e301717b938b39b4729633de2',
    str(arena.RUNTIME / 'chess_anti_engine/replay/game_epoch.py'): '621e5d0764e62cee492688e63e4099ff8cbc0d39ea094b252c3cae31cd74fde3',
}


def validate(m):
    arena.require(set(m) == {'schema', 'state', 'run', 'training_seconds', 'arena_seconds', 'total_seconds',
                            'runtime_manifest', 'preregistration', 'prospective_schedule',
                            'launcher_sha256', 'arena_launcher_sha256', 'input_pins'}, 'one-epoch manifest keys differ')
    arena.require(m['schema'] == 1, 'unsupported one-epoch manifest')
    for key in ('training_seconds', 'arena_seconds', 'total_seconds'):
        arena.require(type(m[key]) in (int, float) and math.isfinite(m[key]) and m[key] > 30, f'invalid {key}')
    arena.require(m['training_seconds'] == 16200 and m['arena_seconds'] == 5400
                  and m['total_seconds'] == 21600, 'registered 4.5h training + 1.5h arena budget differs')
    arena.require(m['input_pins'] == INPUT_PINS, 'published corpus/source/config identity differs')
    outputs = [Path(m[k]) for k in ('state', 'run')]
    inputs = [CORPUS, SOURCE, arena.RUNTIME, C_CHECKPOINT, Path(__file__).resolve(), Path(arena.__file__).resolve()]
    for key in ('runtime_manifest', 'preregistration', 'prospective_schedule'):
        arena.require(set(m[key]) == {'path', 'sha256'} and Path(m[key]['path']).is_absolute(), f'invalid {key}')
        inputs.append(Path(m[key]['path']).resolve())
    for i, out in enumerate(outputs):
        arena.require(out.is_absolute() and out == out.resolve() and out.parent.is_dir(), 'canonical output with existing parent required')
        arena.require(not out.exists() and not out.is_symlink(), f'new output required: {out}')
        for other in inputs + outputs[i+1:]:
            arena.require(out != other and out not in other.parents and other not in out.parents, 'overlapping input/output paths')


def check_pins(m):
    arena.pin(__file__, m['launcher_sha256'])
    arena.pin(arena.__file__, m['arena_launcher_sha256'])
    for path, digest in INPUT_PINS.items():
        arena.pin(path, digest)
    arena.pin(C_CHECKPOINT, C_SHA)
    arena.pin(arena.BOOK, arena.BOOK_SHA)
    arena.pin(arena.READER, arena.READER_SHA)
    for key in ('preregistration', 'prospective_schedule'):
        arena.pin(m[key]['path'], m[key]['sha256'])
    mix = arena.read(CORPUS / 'bt4_policy_mix_summary.json')
    arena.require(arena.read(CORPUS / 'derive_targets_summary.json')['policy_target_postprocess'] == mix,
                  'published recipe lineage differs')
    return arena.runtime_identity(m['runtime_manifest'])


def verify_schedule(report, *, prospective):
    arena.require(report['verifier_sha256'] == INPUT_PINS[str(VERIFIER)] and report['seed'] == 0
                  and report['batch_size'] == 512 and report['runtime']['numpy'] == '1.26.2', 'schedule verifier/runtime differs')
    plan = report['source_plan']
    arena.require(plan['plan_sha256'] == CANONICAL and plan['rows_planned'] == 18910484
                  and plan['batches_planned'] == 36935, 'source canonical epoch differs')
    required = {'E0T05', 'C'} if prospective else {'E0T05'}
    arena.require(set(report['arms']) == required, 'schedule arms differ')
    for role, arm in report['arms'].items():
        expected = CORPUS if role == 'E0T05' else C_CHECKPOINT.parent
        if role == 'C':
            arena.require(arm['summary_sha256'] == INPUT_PINS[str(expected / 'summary.json')]
                          and arm['training_completion_verified'] is True, 'C completed schedule differs')
        else:
            arena.require(arm['corpus'] == str(CORPUS), 'E0T05 schedule corpus differs')
            arena.require(arm['staging'] == ('prospective only' if prospective else 'verified actual'), 'schedule stage differs')
            if not prospective:
                arena.require(arm['training_completion_verified'] is True, 'E0T05 realized schedule incomplete')
        arena.require(arm['metadata_matches_source'] is True and arm['canonical_plan_sha256'] == CANONICAL,
                      f'{role}: ordered source/game schedule differs')


def train_command(m):
    python = arena.read(m['runtime_manifest']['path'])['runtime']['executable']
    return [python, 'scripts/lc0_control_train.py', '--config', 'configs/lc0_positive_control.yaml',
            '--shards', str(CORPUS), '--out-dir', m['run'], '--steps', '0', '--batch-size', '512',
            '--sampling-mode', 'game_epoch', '--epoch-plan-workers', '16', '--epoch-load-workers', '16',
            '--seed', '0', '--device', 'cuda', '--train-window-steps', '88', '--allow-invalid-control']


def completed_training(m, schedule_path):
    run = Path(m['run'])
    summary = arena.read(run / 'summary.json')
    report = arena.read(schedule_path)
    verify_schedule(report, prospective=False)
    arm = report['arms']['E0T05']
    arena.require(arm['summary_sha256'] == arena.sha(run / 'summary.json'), 'verified training summary changed')
    sampling = summary['sampling']
    expected = {'mode': 'game_epoch', 'complete': True, 'seed': 0, 'batch_size': 512,
                'rows_planned': 18910484, 'rows_realized': 18910484, 'batches_planned': 36935,
                'batches_realized': 36935, 'shards': 2309, 'games': 97968, 'plan_workers': 16,
                'load_workers': 16, 'same_game_repeats_max': 0, 'decoded_rows_resident': 0}
    arena.require(all(sampling.get(k) == v for k, v in expected.items()), 'incomplete/mismatched exact epoch')
    arena.require(sampling['plan_sha256'] == sampling['realized_sha256'] == arm['physical_plan_sha256'],
                  'actual staging and realized training schedule differ')
    arena.require(summary['seed'] == 0 and summary['batch_size'] == 512 and summary['train_window_steps'] == 88
                  and summary['steps_realized'] == summary['compute_loss_calls'] == 36935
                  and summary['corpus']['shard_dirs'] == [str(CORPUS)], 'training settings differ')
    windows = summary['train_window_metrics']
    arena.require(len(windows) == 420 and all(w['grad_nonfinite_skip_rate'] == 0
                  and w['transient_cuda_retry_batches'] == 0 and math.isfinite(w['loss'])
                  and math.isfinite(w['grad_norm_mean']) for w in windows), 'training skipped/retried or nonfinite windows')
    checkpoint = {'role': 'E0T05', 'path': str(run / 'checkpoint.pt'), 'sha256': arena.sha(run / 'checkpoint.pt')}
    arena.require(any(c['role'] == 'last' and c['path'] == checkpoint['path'] and c['sha256'] == checkpoint['sha256']
                      for c in summary['checkpoints']), 'completed checkpoint differs')
    return {'complete': True, 'role': 'E0T05', 'run': str(run), 'checkpoint': checkpoint,
            'summary_sha256': arm['summary_sha256'], 'canonical_plan_sha256': CANONICAL,
            'physical_plan_sha256': arm['physical_plan_sha256'],
            'schedule': {'path': str(schedule_path), 'sha256': arena.sha(schedule_path)},
            'historical_valid_control': summary['valid_control'], 'historical_validity_problems': summary['validity_problems']}


def execute(m):
    validate(m)
    rt = check_pins(m)
    actual = arena.runtime_probe(rt)
    verify_schedule(arena.read(m['prospective_schedule']['path']), prospective=True)
    state, run = Path(m['state']), Path(m['run'])
    arena.disk_guard(state)
    arena.disk_guard(run)
    state.mkdir(exist_ok=False)
    arena.write(state / 'manifest.json', m)
    try:
        with (ROOT / 'scratchpad/gpu0_experiment.lock').open('a') as lease:
            arena.acquire_gpu_lease(lease)
            check_pins(m)
            arena.require(not run.exists() and not (state / 'STOP').exists(), 'existing run or stop requested')
            arena.disk_guard(state)
            arena.disk_guard(run)
            arena.require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                                       text=True, timeout=10).strip(), 'competing GPU process')
            charge = arena.run_owned_stage(train_command(m), state / 'training', m['training_seconds'], lease.fileno(),
                'training', {'runtime': actual, 'launcher_sha256': m['launcher_sha256'], 'input_pins': INPUT_PINS},
                manifest=m, stop_paths=(state / 'STOP',))
        arena.require(0 < charge['gpu_seconds'] <= m['training_seconds'], 'training charge exceeds cap')
        check_pins(m)
        # No GPU lease during metadata-only schedule reconstruction. Raw BT4 may run.
        schedule = state / 'realized_schedule.json'
        command = ['/usr/bin/nice', '-n', '19', '/usr/bin/ionice', '-c', '3', '/usr/bin/taskset', '-c', '0,1',
                   rt['executable'], str(VERIFIER), '--run', f'E0T05={run}', '--output', str(schedule)]
        arena.run_owned_stage(command, state / 'schedule', 1800, None, 'schedule',
                              {'gpu_stage': False}, manifest=m, stop_paths=(state / 'STOP',))
        check_pins(m)
        completion = completed_training(m, schedule)
        completion.update(training_charge_seconds=charge['gpu_seconds'], input_pins=INPUT_PINS)
        arena.write(state / 'training.complete.json', completion)
        direct = {'schema': 1, 'output': str(state / 'arena'), 'sims': 100, 'hard_seconds': m['arena_seconds'],
                  'candidate': completion['checkpoint'], 'reference': {'role': C_ROLE, 'path': str(C_CHECKPOINT), 'sha256': C_SHA},
                  'candidate_training': {'path': str(state / 'training.complete.json'), 'sha256': arena.sha(state / 'training.complete.json')},
                  'book': {'path': str(arena.BOOK), 'sha256': arena.BOOK_SHA}, 'runtime_manifest': m['runtime_manifest'],
                  'preregistration': m['preregistration'], 'launcher_sha256': m['arena_launcher_sha256']}
        arena.require(not (state / 'STOP').exists(), 'stop requested before arena')
        arena.write(state / 'arena_manifest.json', direct)
        arena.execute(direct, stop_paths=(state / 'STOP',))
        arena_done = arena.read(state / 'arena/complete.json')
        total = charge['gpu_seconds'] + arena_done['gpu_seconds']
        arena.require(arena_done['complete'] is True and total <= m['total_seconds'], 'arena incomplete or total charge exceeded')
        arena.write(state / 'complete.json', {'complete': True, 'gpu_seconds': total,
                    'training_receipt_sha256': arena.sha(state / 'training.complete.json'),
                    'arena_receipt_sha256': arena.sha(state / 'arena/complete.json'), 'promotion': 'NONE; apply preregistered rule'})
    except BaseException as error:
        arena.write(state / 'failed.json', {'complete': False, 'error': str(error), 'partial_artifacts_preserved': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    m = arena.read(args.manifest)
    validate(m)
    if not args.execute:
        print(json.dumps({'execute': False, 'training_command': train_command(m),
                          'arena': '100 simulations / 1000 games E0T05 vs pinned C, after verified completed epoch'}, indent=2))
        return
    def interrupted(signum, _frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    execute(m)


if __name__ == '__main__':
    main()
