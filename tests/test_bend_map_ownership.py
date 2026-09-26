"""Ownership-diagnostic admission and actual-payload output contracts."""
from pathlib import Path
import subprocess
import sys

import pytest

from native.bend_engine.u64_map_probe import reservation_ownership as ownership


def result(code: int = 0, stdout: str = '', stderr: str = '') -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(['bend', 'case.bend', '--check-only'], code, stdout, stderr)


def diagnostic(binder: str = 'q', location: str = 'exercise') -> str:
    return (f'Error:\n- expected : {binder}\n- observed : {binder} (consumed more than once)\n'
            f'Location: {location}\n4>| def exercise(q: R.Reservation):\n')


@pytest.mark.parametrize('binder', ['q', 'r', 's'])
def test_only_expected_ownership_failure_is_accepted(binder: str) -> None:
    ownership.check_compilation(result(1, stderr=diagnostic(binder)), ownership.consumed(binder))


@pytest.mark.parametrize('fault', ['success', 'crash', 'wrong_exit', 'syntax', 'import',
                                  'wrong_binder', 'wrong_location', 'stdout', 'second_error'])
def test_unrelated_compiler_failures_do_not_count_as_ownership_checks(fault: str) -> None:
    code, stdout, stderr = 1, '', diagnostic()
    if fault in ('success', 'crash', 'wrong_exit'):
        code = {'success': 0, 'crash': -11, 'wrong_exit': 2}[fault]
    elif fault == 'syntax':
        stderr = 'Error:\ninvalid syntax\nLocation: exercise\n'
    elif fault == 'import':
        stderr = 'Error:\nmissing ./IdRegistry.bend\n'
    elif fault == 'wrong_binder':
        stderr = diagnostic('other')
    elif fault == 'wrong_location':
        stderr = diagnostic(location='unrelated')
    elif fault == 'stdout':
        stdout = 'All terms check.\n'
    else:
        stderr += 'Error:\nsecond error\n'
    with pytest.raises(ValueError, match='expected ownership diagnostic'):
        ownership.check_compilation(result(code, stdout, stderr), ownership.consumed('q'))


@pytest.mark.parametrize(('code', 'stdout', 'stderr'), [(1, '', diagnostic()), (0, '', ''),
                                                       (0, 'All terms check.\n', 'warning\n'),
                                                       (0, 'All terms check.\nextra\n', '')])
def test_valid_controls_must_check_cleanly(code: int, stdout: str, stderr: str) -> None:
    with pytest.raises(ValueError, match='valid ownership program'):
        ownership.check_compilation(result(code, stdout, stderr), None)


def test_successful_control() -> None:
    ownership.check_compilation(result(stdout='All terms check.\n'), None)


def test_each_misuse_has_an_unchanged_valid_control() -> None:
    assert len(ownership.MISUSES) == 9
    assert len({c.name for c in ownership.MISUSES}) == 9
    for case in ownership.MISUSES:
        positive = ownership.program(case, invalid=False)
        negative = ownership.program(case, invalid=True)
        assert positive != negative
        assert positive.replace(case.old, case.new) == negative
        assert positive.startswith('import Base\nimport ./IdRegistry.bend as R\n')
    assert set(ownership.VALID_ONLY) == {'branch-exclusive', 'candidate-returned-owner', 'affine-drop'}


def test_changed_use_site_is_not_silently_unmutated() -> None:
    case = ownership.MISUSES[0]._replace(old='missing-site')
    with pytest.raises(ValueError, match='use site changed'):
        ownership.program(case, invalid=True)


def test_payload_state_includes_final_id_and_actual_updated_bytes() -> None:
    text = ownership.expected_payload()
    ownership.check_payload(text)
    assert len(text.splitlines()) == 52
    assert 'aborted 0 4294967295\nunpublished missing\nassigned 4294967295\ncommitted 1 none\n' in text
    assert text.count('payload-last 4294967295 305419896\n') == 4
    assert 'payload-first 2779096485 0\n' in text


@pytest.mark.parametrize(('old', 'new'), [('unpublished missing', 'unpublished value 0'),
                                        ('aborted 0 7', 'aborted 0 8'),
                                        ('assigned 7', 'assigned 8'),
                                        ('committed 1 none', 'committed 1 0'),
                                        ('known 7', 'known 8'),
                                        ('payload-first 2779096485 0', 'payload-first 0 0'),
                                        ('payload-last 4294967295 305419896', 'payload-last 0 0'),
                                        ('end\n', '')])
def test_wrong_publication_counter_or_payload_is_rejected(old: str, new: str) -> None:
    text = ownership.expected_payload()
    assert old in text
    with pytest.raises(ValueError, match='owned-payload transaction'):
        ownership.check_payload(text.replace(old, new, 1))


def test_payload_requires_complete_output() -> None:
    with pytest.raises(ValueError, match='owned-payload transaction'):
        ownership.check_payload(ownership.expected_payload() + 'extra\n')


def test_expected_compile_failure_is_retained_not_automatically_accepted(tmp_path: Path) -> None:
    commands = ownership.Commands(tmp_path)
    completed = commands.run([sys.executable, '-c', 'import sys; print("bad", file=sys.stderr); sys.exit(1)'], 'bad')
    assert completed.returncode == 1
    assert (tmp_path / 'bad.stderr').read_text() == 'bad\n'
    assert commands.records[0]['timed_out'] is False
    with pytest.raises(ValueError, match='expected ownership diagnostic'):
        ownership.check_compilation(completed, ownership.consumed('q'))


def test_success_with_diagnostics_fails_clean_command(tmp_path: Path) -> None:
    commands = ownership.Commands(tmp_path)
    with pytest.raises(RuntimeError, match='exit=0'):
        commands.clean([sys.executable, '-c', 'import sys; print("warning", file=sys.stderr)'], 'warning')
    assert (tmp_path / 'warning.stderr').read_text() == 'warning\n'


def test_copy_annotation_cannot_promote_reservation_to_data() -> None:
    ownership.check_compilation(result(1, stderr=ownership.TYPE_ERROR + '4>| def exercise(+q: R.Reservation):\n'),
                                ownership.TYPE_ERROR)
    with pytest.raises(ValueError, match='expected ownership diagnostic'):
        ownership.check_compilation(result(1, stderr=diagnostic()), ownership.TYPE_ERROR)
