"""Fail-closed E-only admission; draft plans never acquire the GPU or launch."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

TOTAL_ROWS = 58090688
TOTAL_SHARDS = 7108
TOTAL_STEPS = 113459
INITIAL = '9f367063cf16f82f3fe82f726a2b0e39ab262d91f8f9a640e6b06cafc7c8a4bf'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_pin(pin):
    require(isinstance(pin, dict) and isinstance(pin.get('sha256'), str)
            and len(pin['sha256']) == 64, 'future pin is not frozen')
    require(sha(pin['path']) == pin['sha256'], 'pin changed: ' + pin['path'])
    return json.loads(Path(pin['path']).read_text())


def pin_files(pin, retained):
    result = read_pin(pin)
    retained.append(dict(pin))
    return result


def exact_training(summary):
    sampling = summary['sampling']
    require(summary['seed'] == 121 and summary['batch_size'] == 512
            and summary['steps_realized'] == TOTAL_STEPS and summary.get('continuation') is None,
            'D must be fresh seed121/batch512/113459 updates')
    require(sampling['complete'] and sampling['rows_planned'] == TOTAL_ROWS
            and sampling['rows_realized'] == TOTAL_ROWS and sampling['batches_realized'] == TOTAL_STEPS,
            'D exact epoch incomplete')


def admit(plan):
    require(plan['status'] == 'FROZEN_READY' and plan['arm'] == 'E', 'E draft is not admitted')
    require(plan['expected_steps'] == TOTAL_STEPS, 'E update budget changed')
    retained = []
    source = pin_files(plan['source_D_plan'], retained)
    require(source['status'] == 'FROZEN_READY' and source['arm'] == 'D', 'source D plan differs')
    expected = list(source['command_prefix'])
    expected[expected.index('--out-dir') + 1] = plan['out']
    require(plan['command_prefix'] == expected and plan['env'] == source['env'], 'E training command/env differs from D')
    require(plan['runtime'] == source['runtime'] == '/tmp/deepfin-factorial58-runtime'
            and plan['runtime_head'] == source['runtime_head'] == '502cd02e072471c901255f3fdb580d6ea7b826d0',
            'E must retain D training runtime; no loader optimization adoption')
    require((plan['training_seconds'], plan['pause_seconds'], plan['internal_seconds'])
            == (72000, 7200, 72600), 'D training/pause/cleanup budgets differ')
    require(plan['gpu_lock'] == source['gpu_lock'], 'shared GPU lease differs')
    future = plan['future_pins']
    require(plan['dataset_complete'] == future['preparation_complete']['path'], 'E completion path differs')
    d = pin_files(future['D_complete'], retained)
    terminal = pin_files(future['D_terminal'], retained)
    require(terminal['returncode'] == 0 and d['status'] == 'PASS_FACTORIAL58_ARM'
            and d['arm'] == 'D' and d['plan_sha256'] == plan['source_D_plan']['sha256'], 'D terminal/lineage incomplete')
    summary = pin_files(d['summary'], retained)
    exact_training(summary)
    require(d['checkpoint'] == future['D_checkpoint'], 'D checkpoint is not the frozen completed checkpoint')
    require(sha(d['checkpoint']['path']) == d['checkpoint']['sha256'], 'D checkpoint changed')
    retained.append(d['checkpoint'])
    init = pin_files(d['initial_state'], retained)
    anchor_pin = plan['initial_anchor_pin']
    anchor = pin_files(anchor_pin, retained)
    require(anchor_pin['path'] == plan['initial_anchor'] and init['seed'] == anchor['seed'] == 121
            and init['tensor_sha256'] == anchor['tensor_sha256'] == d['tensor_sha256'] == INITIAL,
            'D/A fresh initialization identity differs')

    # Successful wrappers and actual complete color-swapped banks are required.
    # Merely seeing a queued item or a rolling Elo estimate is insufficient.
    sys.path.insert(0, plan['arena_operator_runtime'])
    from bootstrap_experiment_operator import validate_arena_bank
    for name, reference in [('D_C', 'C'), ('D_B', 'B')]:
        receipt = pin_files(future[name + '_complete'], retained)
        wrapper = pin_files(future[name + '_terminal'], retained)
        arena_plan = pin_files(plan['comparison_plans'][name], retained)
        require(wrapper['returncode'] == 0 and receipt['status'] == 'PASS_FACTORIAL58_256_GAME_ARENA'
                and receipt['plan_sha256'] == plan['comparison_plans'][name]['sha256'], name + ' incomplete/wrong plan')
        bound = receipt['bound']
        require(bound['candidate']['arm'] == 'D' and bound['reference']['arm'] == reference
                and bound['candidate']['checkpoint'] == d['checkpoint'], name + ' wrong donors')
        bank = Path(arena_plan['out']) / 'arena.games.jsonl'
        require(sha(bank) == receipt['bank_sha256'], name + ' bank changed')
        retained.append({'path': str(bank), 'sha256': receipt['bank_sha256']})
        validate_arena_bank({'games': 256, 'out': arena_plan['out']}, receipt['result'])
        for donor in bound.values():
            for key in ['checkpoint', 'summary', 'initial_state', 'receipt', 'terminal', 'plan']:
                pin = donor[key]
                require(sha(pin['path']) == pin['sha256'], name + ' bound donor changed')
                retained.append(pin)

    complete = pin_files(future['preparation_complete'], retained)
    require(complete['status'] == 'COMPLETE_SFFREE_35_COHORTS'
            and (complete['rows'], complete['shards'], len(complete['cohorts'])) == (TOTAL_ROWS, TOTAL_SHARDS, 35),
            'full E target preparation incomplete')
    require(complete['qualified'] == future['qualified'] and complete['plan'] == future['builder_plan'],
            'E preparation/qualification/builder lineage differs')
    builder = pin_files(future['builder_plan'], retained)
    require(builder['runtime'] != plan['runtime'], 'builder runtime must not replace D training runtime')
    require(builder['output'] == plan['target_output'], 'E output differs from completed builder plan')
    roots = plan['expected_roots']
    require(roots == [str(Path(plan['target_output']) / f'cohort{i:02d}' / 'E') for i in range(35)],
            'E root order differs')
    counts = []
    for i, pin in enumerate(complete['cohorts']):
        require(pin['path'] == str(Path(roots[i]).parent / 'complete.json'), 'E cohort receipt order differs')
        cohort = pin_files(pin, retained)
        require(cohort['status'] == 'COMPLETE_SFFREE_TARGET_COHORT' and cohort['root'] == roots[i], 'E cohort incomplete')
        counts.append((cohort['rows'], cohort['shards']))
    require(sum(r for r, _ in counts) == TOTAL_ROWS and sum(s for _, s in counts) == TOTAL_SHARDS,
            'E cohort totals differ')
    qualified = pin_files(future['qualified'], retained)
    require(qualified['schema'] == 2 and qualified['status'] == 'PASS_IMMUTABLE_TARGET_OVERLAY_STORAGE_QUALIFICATION'
            and qualified['rows'] == TOTAL_ROWS and len(qualified['shards']) == TOTAL_SHARDS
            and qualified['roots'] == roots, 'E full storage qualification differs')
    d_qualified = pin_files(plan['D_qualification'], retained)
    require(qualified['base_seals'] == d_qualified['base_seals'], 'E and D base identity/order differs')
    for seal in qualified['base_seals']:
        pin_files(seal, retained)
    def roster(receipt):
        indices = {root: i for i, root in enumerate(receipt['roots'])}
        return [(indices[str(Path(e['path']).parent)], Path(e['path']).name, e['rows']) for e in receipt['shards']]
    require(roster(qualified) == roster(d_qualified), 'E/D shard and row ordering differs')
    require(all(Path(root).is_dir() for root in roots), 'E qualified root missing')
    # Existing D training runtime performs full live tree/content/semantic
    # verification via --overlay-storage-qualification before it trains.
    for pin in retained:
        require(sha(pin['path']) == pin['sha256'], 'admission input changed during read')
    return roots, future['qualified'], retained, anchor
