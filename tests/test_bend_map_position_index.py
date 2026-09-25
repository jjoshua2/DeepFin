"""Exact identity, not hash equality, controls structural position IDs."""
import pytest

from native.bend_engine.u64_map_probe import position_index as p

ZERO = (0,) * 8 + (1, 0, 64)


def test_equal_hash_different_identity_gets_different_ids() -> None:
    other = (1 << 63, *ZERO[1:])
    case = p.Case('collision', 3, [p.Op('r', 0, ZERO), p.Op('r', 0, other),
                                 p.Op('r', 0, ZERO), p.Op('g', 0, other)])
    assert p.expected(case) == (
        'begin 3\nr 0 assigned 0 size 1\nr 1 assigned 1 size 2\n'
        'r 2 known 0 size 2\ng 3 value 1 size 2\nend 2\n')


def test_collision_records_consume_capacity_and_known_still_works() -> None:
    other = (1, *ZERO[1:])
    case = p.Case('full', 1, [p.Op('r', 7, ZERO), p.Op('r', 7, other),
                            p.Op('g', 7, other), p.Op('r', 7, ZERO)])
    assert p.expected(case) == (
        'begin 1\nr 0 assigned 0 size 1\nr 1 full size 1\n'
        'g 2 missing size 1\nr 3 known 0 size 1\nend 1\n')


@pytest.mark.parametrize('field', range(11))
def test_each_canonical_field_is_part_of_identity(field: int) -> None:
    other = list(ZERO)
    other[field] ^= (1 << 63) if field < 8 else 1
    case = p.replay('field', [(7, ZERO), (7, tuple(other))])
    assert p.expected(case).endswith('g 8 value 1 size 2\ng 9 value 0 size 2\nend 2\n')


def test_hash_stability_is_a_caller_precondition() -> None:
    # The index does not normalize or repair an inconsistent caller hash.
    case = p.Case('hash', 2, [p.Op('r', 0, ZERO), p.Op('r', 1, ZERO)])
    assert 'r 1 assigned 1 size 2\n' in p.expected(case)


@pytest.mark.parametrize('bits', [0, 17, p.MASK])
def test_invalid_size_is_not_an_allocated_empty_index(bits: int) -> None:
    assert p.expected(p.Case('invalid', bits, [])) == f'invalid {bits}\n'


@pytest.mark.parametrize('bits', [-1, True, 1.0, 1 << 32])
def test_invalid_capacity_transport(bits: int) -> None:
    with pytest.raises(ValueError, match='capacity'):
        p.encode(p.Case('bad', bits, []))


@pytest.mark.parametrize('key', [-1, True, 1 << 64])
def test_invalid_hash_transport(key: int) -> None:
    with pytest.raises(ValueError, match='operation'):
        p.encode(p.Case('bad', 3, [p.Op('r', key, ZERO)]))


@pytest.mark.parametrize('identity', [ZERO[:-1], (*ZERO, 0), (-1, *ZERO[1:]),
                                    (1 << 64, *ZERO[1:]), (True, *ZERO[1:]),
                                    (*ZERO[:-1], 1 << 32)])
def test_invalid_field_transport(identity: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match='identity'):
        p.encode(p.Case('bad', 3, [p.Op('r', 0, identity)]))


def test_transport_keeps_high_bits_and_absent_ep_code() -> None:
    case = p.Case('words', 3, [p.Op('r', (1 << 64) - 1, (1 << 63, *ZERO[1:]))])
    text = p.encode(case)
    assert text.startswith('3;r 4294967295 4294967295 2147483648 0 ')
    assert text.endswith(' 1 0 64')
    assert len(text.split(';')[1].split()) == 22


@pytest.mark.parametrize(('old', 'new'), [('assigned 1', 'assigned 0'), ('known 0', 'known 1'),
                                        ('size 2', 'size 1'), ('end 2', 'end 1'),
                                        ('g 8 value 1', 'g 8 missing')])
def test_corrupt_collision_result_rejected(old: str, new: str) -> None:
    case = p.replay('two', [(0, ZERO), (0, (1, *ZERO[1:]))])
    text = p.expected(case)
    assert old in text
    with pytest.raises(ValueError, match='complete-identity oracle'):
        p.verify(text.replace(old, new, 1), case)


def test_missing_or_extra_rows_are_not_accepted() -> None:
    case = p.replay('one', [(0, ZERO)])
    text = p.expected(case)
    for altered in (text[:-1], text + 'end 1\n', 'begin 7\nend 1\n'):
        with pytest.raises(ValueError, match='complete-identity oracle'):
            p.verify(altered, case)


def test_fixtures_preserve_all_eleven_fields_and_capacity_cases() -> None:
    cases = p.fixtures()
    assert cases == p.fixtures()
    assert len(cases) == len({c.name for c in cases}) == 26
    field_case = next(c for c in cases if c.name == 'every-field-same-hash')
    assert len({op.identity for op in field_case.ops}) == 12
    assert {op.key for op in field_case.ops} == {0}
    assert len([c for c in cases if c.name.startswith('collision-full-')]) == 4
    for case in cases:
        p.verify(p.expected(case), case)


def test_large_transport_rejected() -> None:
    with pytest.raises(ValueError, match='budget'):
        p.encode(p.Case('large', 16, [p.Op('r', (1 << 64) - 1, ((1 << 64) - 1,) * 8 + (1, 15, 64))] * 1000))
