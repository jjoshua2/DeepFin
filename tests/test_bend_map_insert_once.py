"""Dictionary semantics for fused lookup/admission, independent of native layout."""
import pytest

from native.bend_engine.u64_map_probe import insert_once as once
from native.bend_engine.u64_map_probe.run_probe import Case, MASK, Op


def test_insert_once_never_replaces_even_at_capacity() -> None:
    case = Case('one', 1, [Op('e', 0, 0), Op('e', 0, MASK), Op('g', 0), Op('e', 1, 7),
                            Op('g', 1), Op('g', 0), Op('d', 0), Op('e', 1, 9)])
    text = ('begin 1\ne 0 0 inserted 0 1\ne 0 0 existing 0 1\ng 0 0 value 0 1\n'
            'e 0 1 full 1\ng 0 1 missing 1\ng 0 0 value 0 1\nd 0 0 value 0 0\n'
            'e 0 1 inserted 9 1\nend 1\n')
    assert once.expected(case) == text
    once.verify(text, case)


def test_ordinary_put_and_remove_keep_their_meaning() -> None:
    case = Case('mixed', 2, [Op('e', 0, 10), Op('p', 0, 20), Op('e', 0, 30),
                              Op('d', 0), Op('e', 0, 40), Op('g', 0)])
    assert once.expected(case) == (
        'begin 2\ne 0 0 inserted 10 1\np 0 0 replaced 10 1\ne 0 0 existing 20 1\n'
        'd 0 0 value 20 0\ne 0 0 inserted 40 1\ng 0 0 value 40 1\nend 1\n')


def test_encounters_keep_first_id_not_latest_proposal() -> None:
    case = once.from_encounters('revisit', [1 << 63, 0, 1 << 63, (1 << 64) - 1, 0])
    text = once.expected(case)
    assert 'e 2147483648 0 existing 0 2\n' in text
    assert 'e 0 0 existing 1 3\n' in text
    assert text.endswith('g 2147483648 0 value 0 3\ng 0 0 value 1 3\ng 4294967295 4294967295 value 3 3\nend 3\n')
    assert case.ops[4].value == 2
    assert case.ops[8].value == 4


@pytest.mark.parametrize('keys', [[], [0] * 521, [-1], [1 << 64], [True]])
def test_bad_encounter_stream_rejected(keys: list[int]) -> None:
    with pytest.raises(ValueError, match=r'budget|operation'):
        once.from_encounters('bad', keys)


@pytest.mark.parametrize('op', [Op('x', 0), Op('e', -1), Op('e', 1 << 64),
                                Op('e', 0, -1), Op('e', 0, 1 << 32), Op('e', True), Op('e', 0, True)])
def test_bad_input_rejected(op: Op) -> None:
    with pytest.raises(ValueError, match='operation'):
        once.encode(Case('bad', 4, [op]))


@pytest.mark.parametrize('bits', [-1, True, 1.0, 1 << 32])
def test_invalid_capacity_type_or_transport(bits: int) -> None:
    with pytest.raises(ValueError, match='capacity'):
        once.encode(Case('bad', bits, []))


@pytest.mark.parametrize('bits', [0, 17, MASK])
def test_invalid_constructor_has_exact_record(bits: int) -> None:
    assert once.expected(Case('constructor', bits, [])) == f'invalid {bits}\n'


@pytest.mark.parametrize(('old', 'new'), [('existing 3', 'existing 7'), ('existing', 'inserted'),
                                        ('inserted 3', 'inserted 7'), ('end 1', 'end 0'),
                                        ('e 0 0', 'e 1 0'), ('existing 3', 'full')])
def test_corrupt_output_rejected(old: str, new: str) -> None:
    case = Case('corrupt', 1, [Op('e', 0, 3), Op('e', 0, 7), Op('g', 0)])
    text = once.expected(case)
    assert old in text
    with pytest.raises(ValueError, match='trace differs'):
        once.verify(text.replace(old, new, 1), case)


def test_output_must_be_complete() -> None:
    case = Case('length', 1, [Op('e', 0, 3)])
    text = once.expected(case)
    for changed in (text[:-1], text + 'end 1\n', 'begin 1\nend 1\n'):
        with pytest.raises(ValueError, match='trace differs'):
            once.verify(changed, case)


def test_fixture_coverage_and_reproducibility() -> None:
    cases = once.fixtures()
    assert cases == once.fixtures()
    assert len(cases) == len({c.name for c in cases}) == 16
    assert len([c for c in cases if c.name.startswith('mixed-')]) == 4
    for case in cases:
        assert len(once.encode(case)) <= 100_000
        once.verify(once.expected(case), case)


def test_large_transport_rejected() -> None:
    with pytest.raises(ValueError, match='budget'):
        once.encode(Case('large', 4, [Op('e', (1 << 64) - 1, MASK)] * 4000))
