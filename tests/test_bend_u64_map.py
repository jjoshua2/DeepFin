"""Independent trace/admission checks; ordinary pytest does not compile Bend."""
import pytest

from native.bend_engine.u64_map_probe.run_probe import Case, Op, bucket, encode, expected, fixtures, verify


def test_full_replacement_and_zero_value() -> None:
    case = Case('edge', 1, [Op('p', 0, 0), Op('p', 1 << 32, 1), Op('p', 0, 7),
                            Op('g', 0), Op('d', 0), Op('d', 0), Op('p', 1 << 32, 9)])
    text = ('begin 1\np 0 0 added 1\np 1 0 full 1\np 0 0 replaced 0 1\n'
            'g 0 0 value 7 1\nd 0 0 value 7 0\nd 0 0 missing 0\np 1 0 added 1\nend 1\n')
    assert expected(case) == text
    verify(text, case)


def test_full_width_keys_and_max_value() -> None:
    maximum = (1 << 64) - 1
    case = Case('limbs', 3, [Op('p', 0, 0), Op('p', 1 << 63, 1),
                             Op('p', maximum, (1 << 32) - 1), Op('g', maximum)])
    assert expected(case).endswith('g 4294967295 4294967295 value 4294967295 3\nend 3\n')
    assert encode(case).endswith('p 4294967295 4294967295 4294967295;g 4294967295 4294967295')


@pytest.mark.parametrize('bits', [0, 17, (1 << 32) - 1])
def test_invalid_constructor_record(bits: int) -> None:
    assert expected(Case('invalid', bits, [])) == f'invalid {bits}\n'


@pytest.mark.parametrize('bits', range(1, 17))
def test_valid_constructor_record(bits: int) -> None:
    assert expected(Case('valid', bits, [])) == f'begin {bits}\nend 0\n'


@pytest.mark.parametrize('fault', ['missing', 'extra', 'value', 'size', 'disposition', 'high-half'])
def test_changed_native_trace_rejected(fault: str) -> None:
    case = Case('oracle', 3, [Op('p', 1 << 63, 42), Op('g', 1 << 63)])
    text = expected(case)
    if fault == 'missing':
        text = '\n'.join(text.splitlines()[1:]) + '\n'
    elif fault == 'extra':
        text += 'end 1\n'
    else:
        old, new = {'value': ('value 42', 'value 41'), 'size': ('end 1', 'end 2'),
                    'disposition': ('added', 'full'), 'high-half': ('2147483648', '0')}[fault]
        text = text.replace(old, new, 1)
    with pytest.raises(ValueError, match='map trace differs'):
        verify(text, case)


@pytest.mark.parametrize('op', [Op('x', 0), Op('p', -1), Op('p', 1 << 64),
                                Op('p', 0, -1), Op('p', 0, 1 << 32)])
def test_invalid_fixture_operands(op: Op) -> None:
    with pytest.raises(ValueError, match='invalid fixture operation'):
        encode(Case('bad', 4, [op]))


def test_oversized_fixture_transport_rejected() -> None:
    with pytest.raises(ValueError, match='bounded environment'):
        encode(Case('large', 4, [Op('p', (1 << 64) - 1, (1 << 32) - 1)] * 4000))


def test_fixture_coverage_and_high_half_collision() -> None:
    cases = fixtures()
    assert len({c.name for c in cases}) == len(cases) == 32
    for case in cases:
        assert len(encode(case)) <= 100_000
        verify(expected(case), case)
    cluster = next(c for c in cases if c.name == 'high-half-wrapped-cluster')
    inserted = cluster.ops[:8]
    assert len({op.key for op in inserted}) == 8
    assert all(op.key & ((1 << 32) - 1) == 0 for op in inserted)
    assert all(bucket(op.key, 15) == 14 for op in inserted)
    assert all(op.action == 'p' for op in inserted)
    assert len([c for c in cases if c.name.startswith('churn-')]) == 4
