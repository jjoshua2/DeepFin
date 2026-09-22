"""Bounded second-seed replication; no result-dependent branch selection."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

from admission121 import admit as admit121, read_pin, require, sha, TOTAL_ROWS, TOTAL_STEPS


def read(path):
    return json.loads(Path(path).read_text())


def bind_train(donor, retained):
    plan = read_pin(donor['plan'])
    complete = read(donor['complete'])
    terminal = read(donor['terminal'])
    require(plan['status'] == 'FROZEN_READY' and plan['arm'] == donor['arm'], 'donor plan differs')
    require(terminal['returncode'] == 0 and complete['status'] == 'PASS_FACTORIAL58_ARM'
            and complete['arm'] == donor['arm'] and complete['plan_sha256'] == donor['plan']['sha256'],
            'donor incomplete or lineage differs')
    for path in [donor['complete'], donor['terminal']]:
        retained.append({'path': path, 'sha256': sha(path)})
    retained.append(donor['plan'])
    for key, filename in [('summary', 'summary.json'), ('initial_state', 'initial_state.json'), ('checkpoint', 'checkpoint.pt')]:
        pin = complete[key]
        require(pin['path'] == str(Path(plan['out']) / filename) and sha(pin['path']) == pin['sha256'],
                'donor artifact namespace/hash differs')
        retained.append(pin)
    facts = read(complete['summary']['path'])
    sampling = facts['sampling']
    require(facts['seed'] == 122 and facts['batch_size'] == 512 and facts['steps_realized'] == TOTAL_STEPS
            and facts.get('continuation') is None, 'donor seed/update contract differs')
    require(sampling['complete'] and sampling['rows_planned'] == sampling['rows_realized'] == TOTAL_ROWS
            and sampling['batches_realized'] == TOTAL_STEPS and sampling['same_game_repeats_max'] == 0
            and sampling['plan_sha256'] == sampling['realized_sha256'], 'donor exact schedule differs')
    initial = read(complete['initial_state']['path'])
    require(initial['seed'] == 122 and initial['tensor_sha256'] == complete['tensor_sha256'], 'donor initial identity differs')
    return initial


def admit(plan):
    require(plan['status'] == 'FROZEN_READY' and plan['arm'] in ['D122', 'E122'], 'invalid replica')
    source = read_pin(plan['source_E121_plan'])
    roots, qualification, retained, _ = admit121(source)
    retained.append(plan['source_E121_plan'])
    expected = list(source['command_prefix'])
    expected[expected.index('--out-dir') + 1] = plan['out']
    expected[expected.index('--seed') + 1] = '122'
    require(plan['command_prefix'] == expected, 'replica command changed beyond seed/output')
    for key in ['runtime', 'runtime_head', 'env', 'gpu_lock', 'training_seconds', 'pause_seconds', 'internal_seconds']:
        require(plan[key] == source[key], 'replica runtime/budget changed: ' + key)
    require(plan['expected_steps'] == TOTAL_STEPS, 'replica update budget differs')
    # E121-versus-D121 must finish before either replication; score never selects work.
    arena_plan = read_pin(plan['prerequisite_arena']['plan'])
    arena = read(plan['prerequisite_arena']['complete'])
    terminal = read(plan['prerequisite_arena']['terminal'])
    require(arena['status'] == 'PASS_FACTORIAL58_256_GAME_ARENA' and terminal['returncode'] == 0
            and arena['plan_sha256'] == plan['prerequisite_arena']['plan']['sha256'], 'E121-D121 incomplete')
    require(arena['bound']['candidate']['arm'] == 'E' and arena['bound']['reference']['arm'] == 'D', 'wrong prerequisite match')
    require(arena['bound']['candidate']['plan'] == plan['source_E121_plan'], 'wrong E121 training source')
    from bootstrap_experiment_operator import validate_arena_bank
    validate_arena_bank({'games': 256, 'out': arena_plan['out']}, arena['result'])
    bank = str(Path(arena_plan['out']) / 'arena.games.jsonl')
    require(sha(bank) == arena['bank_sha256'], 'prerequisite bank changed')
    retained.append({'path': bank, 'sha256': arena['bank_sha256']})
    retained.append(plan['prerequisite_arena']['plan'])
    for key in ['complete', 'terminal']:
        path = plan['prerequisite_arena'][key]
        retained.append({'path': path, 'sha256': sha(path)})
    for donor in arena['bound'].values():
        for key in ['checkpoint', 'summary', 'initial_state', 'receipt', 'terminal', 'plan']:
            pin = donor[key]
            require(sha(pin['path']) == pin['sha256'], 'prerequisite donor changed')
            retained.append(pin)
    anchor = None
    if plan['arm'] == 'D122':
        qualified = read_pin(source['D_qualification'])
        roots, qualification = qualified['roots'], source['D_qualification']
        require(plan['initial_anchor'] == str(Path(plan['out']) / 'initial_state.json'), 'D122 must establish fresh anchor')
    else:
        anchor = bind_train(plan['paired_D122'], retained)
        donor_plan = read_pin(plan['paired_D122']['plan'])
        require(plan['initial_anchor'] == str(Path(donor_plan['out']) / 'initial_state.json'), 'E122 anchor differs from D122')
    for pin in retained:
        require(sha(pin['path']) == pin['sha256'], 'replication admission input changed')
    return roots, qualification, retained, anchor
