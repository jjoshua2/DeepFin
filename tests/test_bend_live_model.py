"""Cheap live-model transcript/identity failures; model execution is opt-in."""
from __future__ import annotations

import json

import pytest

from native.bend_engine.multi_root.verify_live_model import BEGIN, OPEN, audit, project
from native.bend_engine.multi_root.verify_live import CONTROL, ROOT, WORK


def rows(diagnostics: bool = True) -> list[str]:
    output = [OPEN]
    for gen in (1, 2):
        output += [CONTROL + f'pending-install 1 {gen}', CONTROL + f'admitted 1 {gen}']
        if diagnostics:
            output += [f'info string cohort_batch {gen} 1 4', 'info string native_path 1 1 0',
                       'info string native_reply 1 1 0 0 0 0 0']
        output += [BEGIN + f'1 {gen}']
        if diagnostics:
            output += ['info string cohort_node 1 0 ' + json.dumps([0] * 29)]
        r = {'schema': 'deepfin.live-root.v1', 'slot': 1, 'generation': gen,
             'completed_simulations': 1, 'dispatched_real_rows': 1, 'executed_real_rows': 1,
             'accepted_neural_rows': 1, 'cancelled_rows': 0, 'cancel_requested': False,
             'deadline_expired': False, 'deadline_offset_ms': None, 'arena_capacity': 4096,
             'arena_physical_slots': 4096, 'used_nodes': 1, 'simulation_budget': 4, 'neural_budget': 0}
        output += [ROOT + json.dumps(r)]
    output += [CONTROL + 'quit', WORK + json.dumps({
        'schema': 'deepfin.live-cohort-work.v1', 'reported_generations': 2, 'batch_size': 4,
        'completed_simulations': 2, 'forward_calls': 2, 'dispatched_real_rows': 2,
        'executed_real_rows': 2, 'accepted_neural_rows': 2, 'cancelled_rows': 0,
        'executed_wasted_rows': 0, 'physical_rows': 8, 'padded_rows': 6,
        'unconfirmed_forward_rows': 0, 'unresolved_rows': 0, 'useful_eps': None,
        'warmup_excluded': False})]
    return output


def check(lines: list[str], diagnostics: bool = True) -> dict:
    return project(lines, [(1, 1), (1, 2)], 4, diagnostics)


def test_generations_remap_only_identity_not_values() -> None:
    result = check(rows())
    paths = [values for name, values in result['events'] if name == 'native_path']
    assert paths == [[1, 1, 0], [2, 1, 0]]
    assert result['nodes'][1][0] == result['nodes'][2][0] == [0] * 29
    assert result['row_counts'] == [1, 1]


def test_quiet_markers_are_not_tree_diagnostics() -> None:
    assert check(rows(False), False)['nodes'] == {}
    with pytest.raises(ValueError, match=r'diagnostic'):
        check(rows(), False)


@pytest.mark.parametrize('plan', [[], [(1, 1), (1, 1)], [(1, 2), (1, 1)], [(2, 1), (1, 2)], [(0, 1)], [(1, True)]])
def test_wrong_generation_plan(plan: list[tuple[int, int]]) -> None:
    with pytest.raises(ValueError, match=r'identit|admission|integer|generation|reservation'):
        project(rows(), plan, 4, True)


@pytest.mark.parametrize('batch', [True, 0, 1, 3, 8, 32])
def test_wrong_bound_batch(batch: int) -> None:
    with pytest.raises(ValueError, match=r'integer|batch'):
        project(rows(), [(1, 1), (1, 2)], batch, True)


@pytest.mark.parametrize('field', ['cancel_requested', 'deadline_expired', 'deadline_offset_ms',
                                 'arena_capacity', 'arena_physical_slots', 'simulation_budget', 'neural_budget'])
def test_root_configuration_mutation(field: str) -> None:
    lines = rows()
    index = next(i for i, line in enumerate(lines) if line.startswith(ROOT))
    value = json.loads(lines[index].removeprefix(ROOT))
    value[field] = 1
    lines[index] = ROOT + json.dumps(value)
    with pytest.raises(ValueError, match=r'arena|configuration|cancellation'):
        check(lines)


@pytest.mark.parametrize('extra', [OPEN, CONTROL + 'error stale-identity', CONTROL + 'cancel 1 1',
                                  'info string unknown 0', 'info string native_reply 9 1 0',
                                  'info string native_reply 1 1 0', 'info string cohort_batch 5 1 4',
                                  CONTROL + 'pending-install 1 3'])
def test_duplicate_unknown_unscoped_or_late_events(extra: str) -> None:
    lines = rows()
    lines.insert(-2, extra)
    with pytest.raises(ValueError, match=r'plan|generation|output|batch'):
        check(lines)


def test_missing_snapshot_and_extra_physical_batch_fail() -> None:
    with pytest.raises(ValueError, match=r'snapshot'):
        check([x for x in rows() if not x.startswith('info string cohort_node ')])
    lines = rows()
    lines.insert(-2, 'info string cohort_batch 3 1 4')
    with pytest.raises(ValueError, match=r'reconcile'):
        check(lines)


def test_duplicate_json_keys_fail() -> None:
    lines = rows()
    lines[-1] = lines[-1].replace('"batch_size": 4', '"batch_size": 4, "batch_size": 4')
    with pytest.raises(ValueError, match=r'duplicate JSON'):
        check(lines)


@pytest.mark.parametrize('text', ['', 'native-buffer-audit calls=1 input_changes=0 output_changes=0 input_tensor_allocations=1',
    'native-buffer-audit calls=2 input_changes=1 output_changes=0 input_tensor_allocations=1',
    'native-buffer-audit calls=2 input_changes=0 output_changes=0 input_tensor_allocations=2',
    'native-buffer-audit calls=2 input_changes=0 output_changes=0 input_tensor_allocations=1\nextra'])
def test_bad_audit(text: str) -> None:
    with pytest.raises(ValueError, match=r'audit mismatch'):
        audit(text, 2)


def test_good_audit() -> None:
    audit('native-buffer-audit calls=2 input_changes=0 output_changes=0 input_tensor_allocations=1\n', 2)


def test_completed_remove_and_readmission() -> None:
    lines = rows()
    at = lines.index(CONTROL + 'pending-install 1 2')
    lines[at:at] = [CONTROL + 'pending-remove 1 1', CONTROL + 'removed 1 1']
    assert check(lines)['row_counts'] == [1, 1]


@pytest.mark.parametrize('events', [
    ['removed 1 1'], ['pending-remove 1 2'], ['pending-remove 2 1'],
    ['pending-remove 1 1'], ['pending-remove 1 1', 'pending-remove 1 1'],
    ['pending-remove 1 1', 'removed 1 2'],
])
def test_bad_removal_sequence(events: list[str]) -> None:
    lines = rows()
    at = lines.index(CONTROL + 'pending-install 1 2')
    lines[at:at] = [CONTROL + event for event in events]
    with pytest.raises(ValueError, match=r'remov|admission'):
        check(lines)


def test_removal_before_model_result_and_unfinished_final_removal() -> None:
    lines = rows()
    lines.insert(lines.index(CONTROL + 'admitted 1 1') + 1, CONTROL + 'pending-remove 1 1')
    with pytest.raises(ValueError, match=r'removal'):
        check(lines)
    lines = rows()
    lines.insert(-2, CONTROL + 'pending-remove 1 2')
    with pytest.raises(ValueError, match=r'incomplete'):
        check(lines)
