"""Reject schedule changes even when aggregate work and final trees are equal."""
from copy import deepcopy
from typing import Any

import pytest

from native.bend_engine.multi_root.verify_fifo import (
    check_probe, compare, probe_expected, semantic_view,
)


def observation() -> dict[str, Any]:
    return {'roots': {1: {'completed': 1}, 2: {'completed': 1}},
            'nodes': {1: {0: [1, 2]}, 2: {0: [3, 4]}},
            'events': [('native_path', [1, 1, 0]), ('native_path', [2, 1, 0]),
                       ('cohort_batch', [1, 2, 4])],
            'work': {'executed_real_rows': 2, 'padded_rows': 2, 'wall_seconds': 0.2,
                     'phase_seconds': {'queue_wait': None}, 'warmup_excluded': False}}


def test_only_explicit_time_values_are_ignored() -> None:
    left = observation()
    right = deepcopy(left)
    right['work']['wall_seconds'] = 0.8
    assert compare(left, right) == compare(left, left)
    assert 'phase_seconds' in semantic_view(right)['work']


@pytest.mark.parametrize('fault', ['order', 'drop', 'duplicate', 'ticket', 'root_order',
                                    'tree', 'accounting', 'new_field', 'phase', 'warmup'])
def test_semantic_corruption_rejected(fault: str) -> None:
    left = observation()
    right = deepcopy(left)
    if fault == 'order':
        right['events'][0], right['events'][1] = right['events'][1], right['events'][0]
    elif fault == 'drop':
        right['events'].pop()
    elif fault == 'duplicate':
        right['events'].append(right['events'][0])
    elif fault == 'ticket':
        right['events'][0][1][1] += 1
    elif fault == 'root_order':
        right['roots'] = dict(reversed(list(right['roots'].items())))
    elif fault == 'tree':
        right['nodes'][1][0][0] += 1
    elif fault == 'accounting':
        right['work']['padded_rows'] = 0
    elif fault == 'new_field':
        right['work']['new_counter'] = 1
    elif fault == 'phase':
        right['work']['phase_seconds']['queue_wait'] = 0
    else:
        right['work']['warmup_excluded'] = True
    with pytest.raises(ValueError, match='FIFO scheduler differs'):
        compare(left, right)


def test_probe_boundaries() -> None:
    text = probe_expected()
    check_probe(text)
    assert text.startswith('case 0 0 0\nstate 1 0\n')
    assert len([line for line in text.splitlines() if line.startswith('case ')]) == 120
    assert 'root 16 0 4096 1 2 0' in text


@pytest.mark.parametrize(('old', 'new'), [('state 1 0', 'state 0 0'),
                                          ('root 16 0 4096 1 2 0', 'root 16 0 4096 1 0 0'),
                                          ('case 1 0 0', 'case 2 0 0')])
def test_probe_faults_rejected(old: str, new: str) -> None:
    text = probe_expected()
    assert old in text
    with pytest.raises(ValueError, match='FIFO adapter order'):
        check_probe(text.replace(old, new, 1))
