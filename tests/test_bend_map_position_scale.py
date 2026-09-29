"""Full-record population, saturation and per-operation evidence admission."""
import subprocess

import pytest

from native.bend_engine.u64_map_probe import position_scale as p


def text(bits: int = 2, width: int = 1) -> str:
    return '\n'.join(p.expected_lines(bits, width)) + '\n'


def test_small_sequence_has_exact_assignment_readback_and_saturation() -> None:
    observed = text()
    assert observed.startswith(
        'position-scale 2 1 2\ng 0 11 0 missing size 0\nr 0 11 0 assigned 0 size 1\n'
        'r 1 11 1 assigned 1 size 2\ng 1 11 1 value 1 size 2\ng 0 11 0 value 0 size 2\n'
        'r 0 11 0 known 0 size 2\nr 1 11 1 known 1 size 2\nr 2 11 0 full size 2\n'
        'g 2 11 0 missing size 2\nr 3 11 3 full size 2\ng 3 11 3 missing size 2\n')
    assert observed.endswith('g 0 11 0 value 0 size 2\ng 1 11 1 value 1 size 2\nend 2\n')
    assert p.verify(observed, 2, 1) == 59


def test_shared_hash_still_assigns_distinct_records_and_counts_identities() -> None:
    observed = text(2, 8)
    assert 'r 1 11 0 assigned 1 size 2\n' in observed
    assert 'g 1 11 0 value 1 size 2\ng 0 11 0 value 0 size 2\n' in observed
    assert 'r 2 11 0 full size 2\n' in observed
    assert observed.endswith('g 0 11 0 value 0 size 2\ng 1 11 0 value 1 size 2\nend 2\n')


@pytest.mark.parametrize(('bits', 'width'), p.SCENARIOS)
def test_full_population_operation_counts_and_permutation(bits: int, width: int) -> None:
    n = p.population(bits, width)
    ops = list(p.operations(bits, width))
    assert ops[1:n + 1] == [p.Op('r', i, 11, i // width) for i in range(n)]
    assert len(ops) == 4 * n + 49
    assert len(list(p.expected_lines(bits, width))) == 4 * n + 51
    assert sorted(op.record for op in ops[-n:]) == list(range(n))
    assert len({p.identity(i, 11) for i in range(n)}) == n
    assert len({p.key(i // width) for i in range(n)}) == (n + width - 1) // width


@pytest.mark.parametrize('field', range(11))
def test_mismatches_change_only_the_selected_full_identity_field(field: int) -> None:
    for i in (0, 32767):
        before, changed = p.identity(i, 11), p.identity(i, field)
        assert [j for j in range(11) if before[j] != changed[j]] == [field]
    assert f'g 0 {field} 0 missing size 2\nr 0 {field} 0 full size 2\n' in text()


def test_maximum_record_population_is_not_just_maximum_hash_allocation() -> None:
    assert p.population(16, 8) == 32768
    assert len({p.key(i // 8) for i in range(32768)}) == 4096
    assert (16, 32768) not in p.SCENARIOS  # No unbudgeted quadratic maximum-chain run.
    assert (10, 512) in p.SCENARIOS       # Explicit bounded long-chain control.


@pytest.mark.parametrize(('bits', 'width'), [(0, 1), (1, 1), (17, 1), (16, 32768),
                                             (2, 0), (10, 2), (True, 1), (2, True),
                                             (2.0, 1), (2, 1.0)])
def test_invalid_configuration(bits: int, width: int) -> None:
    with pytest.raises(ValueError, match='configuration'):
        p.population(bits, width)


@pytest.mark.parametrize(('old', 'new'), [
    ('r 1 11 1 assigned 1', 'r 1 11 1 assigned 0'),
    ('g 1 11 1 value 1', 'g 1 11 1 value 0'),
    ('r 0 11 0 known 0', 'r 0 11 0 full'),
    ('r 2 11 0 full', 'r 2 11 0 assigned 2'),
    ('g 2 11 0 missing', 'g 2 11 0 value 0'),
    ('g 0 0 0 missing', 'g 0 0 0 value 0'),
    ('size 2', 'size 1'), ('end 2', 'end 1'),
])
def test_corrupt_record_chain_id_or_size_cannot_pass(old: str, new: str) -> None:
    baseline = text()
    assert old in baseline
    with pytest.raises(ValueError, match='position scale mismatch'):
        p.verify(baseline.replace(old, new, 1), 2, 1)


@pytest.mark.parametrize('fault', ['missing', 'extra', 'reordered', 'newline', 'wrong_configuration'])
def test_partial_or_misidentified_traces_are_rejected(fault: str) -> None:
    baseline = text()
    lines = baseline.splitlines()
    if fault == 'missing':
        lines.pop(3)
    elif fault == 'extra':
        lines.append('end 2')
    elif fault == 'reordered':
        lines[2], lines[3] = lines[3], lines[2]
    elif fault == 'wrong_configuration':
        lines[0] = 'position-scale 16 8 32768'
    altered = baseline[:-1] if fault == 'newline' else '\n'.join(lines) + '\n'
    with pytest.raises(ValueError, match='position scale'):
        p.verify(altered, 2, 1)


@pytest.mark.parametrize(('code', 'stdout', 'stderr'), [
    (0, '', 'invalid position scale configuration\n'),
    (-11, '', 'invalid position scale configuration\n'),
    (2, 'unexpected', 'invalid position scale configuration\n'),
    (2, '', 'other failure\n'), (2, '', ''),
])
def test_unrelated_failures_are_not_configuration_rejections(code: int, stdout: str, stderr: str) -> None:
    with pytest.raises(ValueError, match='not rejected as intended'):
        p.check_invalid(subprocess.CompletedProcess(['probe'], code, stdout, stderr))


def test_exact_native_configuration_diagnostic() -> None:
    p.check_invalid(subprocess.CompletedProcess(['probe'], 2, '', 'invalid position scale configuration\n'))
