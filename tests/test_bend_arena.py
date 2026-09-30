"""Bounded arena configuration/output contracts; native execution is opt-in."""
from copy import deepcopy
from typing import Any

import pytest

from native.bend_engine.multi_root import verify
from native.bend_engine.multi_root.verify_arena import expected_probe, outcome, probe_result
from tests.test_bend_multi_root import output, report


def banner(cap: int) -> str:
    return f'info string cohort_arena {cap} {max(4096, 1 << (cap - 1).bit_length())}\n'


@pytest.mark.parametrize('cap', [1, 20, 21, 4097, 8192, 16385, 65536])
def test_requested_capacity_and_physical_storage(cap: int) -> None:
    root, work = report()
    root['used_nodes'] = cap
    parsed = verify.parse(banner(cap) + output(root, work), 1, 4, 4, 0, False, arena_nodes=cap)
    assert parsed['roots'][1]['used_nodes'] == cap


def test_default_output_and_bounds_unchanged() -> None:
    root, work = report()
    text = output(root, work)
    assert verify.parse(text, 1, 4, 4, 0, False) == verify.parse(text, 1, 4, 4, 0, False, arena_nodes=4096)
    with pytest.raises(ValueError, match='arena declaration'):
        verify.parse(banner(4096) + text, 1, 4, 4, 0, False)


@pytest.mark.parametrize('capacity', [0, 65537, 4294967295, True, False, 8192.0, '8192'])
def test_invalid_capacity_rejected(capacity: Any) -> None:
    root, work = report()
    with pytest.raises(ValueError, match='invalid integer'):
        verify.parse(output(root, work), 1, 4, 4, 0, False, arena_nodes=capacity)


@pytest.mark.parametrize('kind', ['absent', 'duplicate', 'wrong_logical', 'wrong_physical', 'late', 'extra'])
def test_capacity_must_be_observed_exactly_once(kind: str) -> None:
    root, work = report()
    prefix = banner(8192)
    text = prefix + output(root, work)
    if kind == 'absent':
        text = output(root, work)
    elif kind == 'duplicate':
        text = prefix + text
    elif kind == 'wrong_logical':
        text = text.replace('8192 8192', '4097 8192')
    elif kind == 'wrong_physical':
        text = text.replace('8192 8192', '8192 4096')
    elif kind == 'late':
        lines = output(root, work).splitlines()
        text = lines[0] + '\n' + prefix + lines[1] + '\n'
    else:
        text = text.replace('8192 8192', '8192 8192 ignored')
    with pytest.raises(ValueError, match=r'arena|declaration'):
        verify.parse(text, 1, 4, 4, 0, False, arena_nodes=8192)


def test_spare_physical_slots_do_not_increase_logical_capacity() -> None:
    root, work = report()
    root['used_nodes'] = 4098
    with pytest.raises(ValueError, match='invalid integer'):
        verify.parse(banner(4097) + output(root, work), 1, 4, 4, 0, False, arena_nodes=4097)


@pytest.mark.parametrize(('old', 'new'), [
    ('config 4096', 'config 8192'), ('8192 8192 8192', '8192 8192 4096'),
    ('first 0 101', 'first 0 777'), ('4294967295 305419896', '0 305419896'),
    ('active 4096 4097 8192 1 8192', 'active 4096 4097 8192 1 4096'),
    ('idle 4096 1 4096 0 4096 20 4', 'idle 4096 1 4096 1 4096 21 0'),
    ('deadline 0 1 65536 0 65536', 'deadline 0 1 65536 0 4096'),
])
def test_native_probe_corruption_rejected(old: str, new: str) -> None:
    text = expected_probe()
    assert old in text
    with pytest.raises(ValueError, match='arena probe differs'):
        probe_result(text.replace(old, new, 1))


def test_probe_has_all_boundaries_and_final_transaction() -> None:
    text = expected_probe()
    probe_result(text)
    assert len(text.splitlines()) == 76
    assert 'last 65535 65559 ' in text
    assert 'arena 4294967295 1 1 1 0 4096 4' in text
    with pytest.raises(ValueError, match='arena probe differs'):
        probe_result('\n'.join(text.splitlines()[:-1]) + '\n')


def observed_search() -> tuple[dict[str, Any], dict[str, Any]]:
    return ({'roots': {1: {'stop_code': 1, 'completed_simulations': 184, 'used_nodes': 4090}}},
            {'roots': {1: {'stop_code': 0, 'completed_simulations': 256, 'used_nodes': 5600}}})


def test_capacity_gate_requires_realized_search_work() -> None:
    before, after = observed_search()
    assert outcome(before, after) == {'default_simulations': 184, 'default_nodes': 4090,
                                     'larger_simulations': 256, 'larger_nodes': 5600}


@pytest.mark.parametrize(('arm', 'key', 'bad'), [
    (0, 'stop_code', 0), (0, 'completed_simulations', 256), (0, 'used_nodes', 4097),
    (1, 'stop_code', 1), (1, 'completed_simulations', 255),
    (1, 'used_nodes', 4096), (1, 'used_nodes', 8193),
])
def test_accepted_setting_without_effect_is_not_success(arm: int, key: str, bad: int) -> None:
    results = deepcopy(observed_search())
    results[arm]['roots'][1][key] = bad
    with pytest.raises(ValueError, match='actual search gate'):
        outcome(*results)
