"""Independent dictionary/counter and trace-admission contracts for ID interning."""
import pytest

from native.bend_engine.u64_map_probe import registry as r
from native.bend_engine.u64_map_probe.run_probe import MASK


def test_revisits_do_not_consume_ids() -> None:
    case = r.Case('stable', 3, 7, [('g', 0), ('r', 0), ('r', 0), ('r', 1), ('g', 0)])
    assert r.expected(case) == (
        'begin 3 7\ng 0 0 missing size 0 next 7\nr 0 0 assigned 7 size 1 next 8\n'
        'r 0 0 known 7 size 1 next 8\nr 0 1 assigned 8 size 2 next 9\n'
        'g 0 0 value 7 size 2 next 9\nend size 2 next 9\n')


def test_full_table_does_not_advance_and_hits_still_work() -> None:
    case = r.Case('full', 1, 0, [('r', 0), ('r', 1), ('r', 0), ('g', 1)])
    assert r.expected(case) == (
        'begin 1 0\nr 0 0 assigned 0 size 1 next 1\nr 0 1 full size 1 next 1\n'
        'r 0 0 known 0 size 1 next 1\ng 0 1 missing size 1 next 1\nend size 1 next 1\n')


def test_max_id_is_issued_once_then_only_known_keys_succeed() -> None:
    case = r.Case('last', 3, MASK, [('r', 0), ('r', 1), ('r', 0), ('g', 0), ('g', 1)])
    assert r.expected(case) == (
        f'begin 3 {MASK}\nr 0 0 assigned {MASK} size 1 next none\n'
        f'r 0 1 exhausted size 1 next none\nr 0 0 known {MASK} size 1 next none\n'
        f'g 0 0 value {MASK} size 1 next none\ng 0 1 missing size 1 next none\nend size 1 next none\n')


def test_id_exhaustion_precedes_table_full_for_new_keys() -> None:
    case = r.Case('both', 1, MASK, [('r', 0), ('r', 1)])
    text = r.expected(case)
    assert 'exhausted' in text
    assert 'full' not in text


def test_encounters_allocate_dense_first_seen_ids() -> None:
    case = r.encounter_case('walk', [1 << 63, 0, 1 << 63, MASK, 0])
    text = r.expected(case)
    assert 'r 2147483648 0 known 0 size 2 next 2\n' in text
    assert 'r 0 4294967295 assigned 2 size 3 next 3\n' in text
    assert text.endswith('g 2147483648 0 value 0 size 3 next 3\ng 0 0 value 1 size 3 next 3\n'
                         'g 0 4294967295 value 2 size 3 next 3\nend size 3 next 3\n')


@pytest.mark.parametrize('keys', [[], [0] * 521, [-1], [1 << 64], [True]])
def test_invalid_encounters_rejected(keys: list[int]) -> None:
    with pytest.raises(ValueError, match=r'budget|operation'):
        r.encounter_case('invalid', keys)


@pytest.mark.parametrize('value', [-1, True, 1.0, 1 << 32])
@pytest.mark.parametrize('field', ['bits', 'first'])
def test_invalid_constructor_transport(field: str, value: int) -> None:
    case = r.Case('invalid', 4, 0, [])._replace(**{field: value})
    with pytest.raises(ValueError, match='configuration'):
        r.encode(case)


@pytest.mark.parametrize('bits', [0, 17, MASK])
def test_rejected_constructor(bits: int) -> None:
    assert r.expected(r.Case('invalid', bits, MASK, [])) == f'invalid {bits} {MASK}\n'


@pytest.mark.parametrize('op', [('d', 0), ('r', -1), ('r', 1 << 64), ('r', True), ('g', 1.0)])
def test_invalid_operations(op: tuple[str, int]) -> None:
    with pytest.raises(ValueError, match='operation'):
        r.encode(r.Case('invalid', 3, 0, [op]))


@pytest.mark.parametrize(('old', 'new'), [('known 0', 'known 1'), ('assigned 0', 'assigned 1'),
                                        ('known', 'assigned'), ('next 1', 'next 2'),
                                        ('size 1', 'size 2'), ('r 0 0', 'r 1 0')])
def test_corrupt_trace_rejected(old: str, new: str) -> None:
    case = r.Case('trace', 2, 0, [('r', 0), ('r', 0)])
    text = r.expected(case)
    assert old in text
    with pytest.raises(ValueError, match='trace differs'):
        r.verify(text.replace(old, new, 1), case)


def test_incomplete_or_extra_trace_rejected() -> None:
    case = r.Case('trace', 2, 0, [('r', 0)])
    text = r.expected(case)
    for changed in (text[:-1], text + 'end size 1 next 1\n', 'begin 2 0\nend size 1 next 1\n'):
        with pytest.raises(ValueError, match='trace differs'):
            r.verify(changed, case)


def test_fixtures_and_transport_are_reproducible() -> None:
    cases = r.fixtures()
    assert cases == r.fixtures()
    assert len(cases) == len({c.name for c in cases}) == 35
    assert any(c.first == MASK for c in cases)
    for case in cases:
        assert len(r.encode(case)) <= 100_000
        r.verify(r.expected(case), case)


def test_large_transport_rejected() -> None:
    with pytest.raises(ValueError, match='budget'):
        r.encode(r.Case('large', 16, 0, [('r', (1 << 64) - 1)] * 5000))
