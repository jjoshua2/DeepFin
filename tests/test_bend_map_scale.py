"""Full-capacity trace admission; ordinary pytest never compiles or times maps."""
from pathlib import Path
import sys

import pytest

from native.bend_engine.u64_map_probe import scale


def text(bits: int = 2) -> str:
    return '\n'.join(scale.expected_lines(bits)) + '\n'


def test_small_trace_checks_full_replacement_removal_and_retry() -> None:
    observed = text()
    assert observed.startswith(
        'map 2 2\np 0 added 1\np 1 added 2\np 2 full 2\ng 2 missing 2\n'
        'p 0 replaced 0 2\np 1 replaced 2246822519 2\n'
        'e 0 existing 4294967295 2\ne 1 existing 2048144776 2\ne 2 full 2\n')
    assert 'd 0 value 4294967295 1\ng 0 missing 1\ng 1 value 2048144776 1\np 2 added 2\n' in observed
    assert 'd 1 value 2048144776 1\nd 2 value 198677742 0\n' in observed
    assert 'p 0 added 1\ne 0 existing 7 1\ng 0 value 7 1\nd 0 value 7 0\nmap-end 0\n' in observed
    assert scale.verify(observed, 2) == 55


def test_full_registry_preserves_aborted_slot_and_last_id() -> None:
    observed = text()
    assert ('registry 0\nr 0 assigned 0 1 1\na 1 reserved 1 aborted 1 1\n'
            'g 1 missing 1 1\nc 2 reserved 1 assigned 1 2 2\ng 1 missing 2 2\nr 1 full 2 2\n') in observed
    assert ('a 1 reserved 4294967295 aborted 1 4294967295\ng 1 missing 1 4294967295\n'
            'c 2 reserved 4294967295 assigned 4294967295 2 none\n') in observed
    assert observed.endswith('r 2 known 4294967295 2 none\ng 2 value 4294967295 2 none\n'
                             'c 3 exhausted 2 none\nregistry-end 2 none\n')


@pytest.mark.parametrize('bits', [2, 10, 16])
def test_exact_scale_counts_and_advertised_entry_limit(bits: int) -> None:
    n = scale.entry_limit(bits)
    operations = list(scale.map_operations(n))
    assert operations[:n] == [('p', i, 0) for i in range(n)]
    assert operations[n] == ('p', n, 0)
    assert len(operations) == 10 * n + 7
    assert len(list(scale.registry_operations(n))) == 3 * n + 5
    assert len(list(scale.expected_lines(bits))) == 16 * n + 23
    if bits == 16:
        assert n == 32768
        assert len({scale.key(i) for i in range(n + n // 2 + 2)}) == n + n // 2 + 2


@pytest.mark.parametrize('bits', [-1, 0, 1, 17, 65536, True, 16.0])
def test_invalid_scale_exponent(bits: int) -> None:
    with pytest.raises(ValueError, match='scale exponent'):
        scale.entry_limit(bits)


@pytest.mark.parametrize(('old', 'new'), [
    ('p 1 added 2', 'p 1 full 1'), ('p 2 full 2', 'p 2 added 3'),
    ('p 1 replaced 2246822519', 'p 1 replaced 0'),
    ('e 0 existing 4294967295', 'e 0 existing 0'),
    ('g 0 missing 1', 'g 0 value 0 1'), ('p 2 added 2', 'p 2 added 1'),
    ('map-end 0', 'map-end 1'), ('a 1 reserved 1 aborted 1 1', 'a 1 reserved 1 aborted 1 2'),
    ('c 2 reserved 1 assigned 1', 'c 2 reserved 1 assigned 0'),
    ('r 1 full 2 2', 'r 1 full 2 3'), ('c 0 known 0', 'c 0 known 1'),
    ('registry-end 2 none', 'registry-end 2 0'),
])
def test_every_outcome_value_and_counter_must_match(old: str, new: str) -> None:
    baseline = text()
    assert old in baseline
    with pytest.raises(ValueError, match='scale trace mismatch'):
        scale.verify(baseline.replace(old, new, 1), 2)


@pytest.mark.parametrize('fault', ['missing', 'extra', 'reordered', 'newline', 'wrong_size'])
def test_partial_or_misidentified_run_is_not_a_pass(fault: str) -> None:
    baseline = text()
    rows = baseline.splitlines()
    if fault == 'missing':
        rows.pop(20)
    elif fault == 'extra':
        rows.append('registry-end 2 none')
    elif fault == 'reordered':
        rows[1], rows[2] = rows[2], rows[1]
    elif fault == 'wrong_size':
        rows[0] = 'map 16 32768'
    altered = baseline[:-1] if fault == 'newline' else '\n'.join(rows) + '\n'
    with pytest.raises(ValueError, match=r'scale output|scale trace'):
        scale.verify(altered, 2)


def test_command_keeps_failure_diagnostics(tmp_path: Path) -> None:
    commands = scale.Commands(tmp_path)
    with pytest.raises(RuntimeError, match='exit=0'):
        commands.run([sys.executable, '-c', 'import sys; print("warning", file=sys.stderr)'], 'bad')
    assert (tmp_path / 'bad.stderr').read_text() == 'warning\n'
    assert commands.records[0]['timed_out'] is False


def test_command_uses_only_selected_scale_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('DEEPFIN_MAP_SCALE_BITS', 'invalid')
    monkeypatch.setenv('BEND_U64_CORRUPT', '1')
    commands = scale.Commands(tmp_path)
    script = 'import os; print(os.getenv("DEEPFIN_MAP_SCALE_BITS")); print(os.getenv("BEND_U64_CORRUPT"))'
    assert commands.run([sys.executable, '-c', script], 'plain') == 'None\nNone\n'
    assert commands.run([sys.executable, '-c', script], 'selected', 16) == '16\nNone\n'
