"""Session cache controls and observable reuse; native search checks are explicit."""
import subprocess

import pytest

from native.bend_engine.session_probe import cache_probe as p


@pytest.mark.parametrize('line', ['cache 17 0 0 0', 'cache 0 1 0 0', 'cache 0 0 1 0',
                                  'cache 0 0 0 1', 'cache 1 0 2 0', 'cache 6 0 33 0',
                                  'cache 6 -1 0 0', 'cache 6 NaN 0 0', 'cache 6 0 0',
                                  'cache 6 0 0 0 extra', 'other 6 0 0 0'])
def test_invalid_stats_are_not_evidence_of_reuse(line: str) -> None:
    with pytest.raises(ValueError, match=r'cache|nondecimal'):
        p.parse_stats(line)


@pytest.mark.parametrize('line', ['cache 0 0 0 0', 'cache 1 1 1 30', 'cache 6 16 16 0',
                                  'cache 16 0 32768 1'])
def test_valid_reports(line: str) -> None:
    assert p.parse_stats(line) == tuple(map(int, line.split()[1:]))


def test_reuse_and_clear_observations() -> None:
    p.check_reuse((6, 0, 16, 0), (6, 16, 16, 0), (6, 0, 16, 0))


@pytest.mark.parametrize('phase', [0, 1, 2])
def test_ignored_setting_or_clear_fails(phase: int) -> None:
    observations = [(6, 0, 16, 0), (6, 16, 16, 0), (6, 0, 16, 0)]
    observations[phase] = (6, 0, 0, 0)
    with pytest.raises(ValueError, match='reuse/clear'):
        p.check_reuse(*observations)


def test_recreation_each_epoch_is_not_persistent_reuse() -> None:
    with pytest.raises(ValueError, match='reuse/clear'):
        p.check_reuse((6, 0, 16, 0), (6, 0, 16, 0), (6, 0, 16, 0))


def test_environment_does_not_leak_inherited_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(p.SETTING, '16')
    monkeypatch.setenv('BEND_U64_CORRUPT', '1')
    assert p.SETTING not in p.environment(None)
    assert p.environment('0')[p.SETTING] == '0'
    assert p.environment('6')[p.SETTING] == '6'
    assert 'BEND_U64_CORRUPT' not in p.environment('6')


@pytest.mark.parametrize(('code', 'stderr'), [(0, 'invalid cache command\n'), (-11, 'invalid cache command\n'),
                                            (2, 'wrong error\n'), (2, 'invalid cache command'),
                                            (2, 'invalid cache command\nextra\n')])
def test_unrelated_failure_is_not_a_successful_rejection(code: int, stderr: str) -> None:
    with pytest.raises(ValueError, match='unexpected session-cache rejection'):
        p.check_invalid(subprocess.CompletedProcess([], code, '', stderr), 'invalid cache command')


def test_expected_native_rejection() -> None:
    p.check_invalid(subprocess.CompletedProcess([], 2, '', 'invalid cache command\n'), 'invalid cache command')


def test_transcript_fingerprint_includes_order_values_and_input() -> None:
    original = [('in', 'reply 1 1 0\n'), ('out', 'node 0 0'), ('out', 'best 42')]
    digest = p.fingerprint(original)
    assert len(digest) == 64
    for changed in (original[::-1], original[1:], [*original[:-1], ('out', 'best 43')],
                    [('in', 'reply 1 2 0\n'), *original[1:]]):
        assert p.fingerprint(changed) != digest


def test_invalid_settings_include_sign_space_overflow_and_non_ascii() -> None:
    assert {'', '-1', '+1', ' 6', '6 ', '17', '4294967296', '١'} <= set(p.INVALID)


def test_only_the_existing_foreign_boundary_is_accepted() -> None:
    names = ['Job.load', 'Command.read', 'Reply.read', 'step_path_checked', 'step_traced',
             'step_read', 'step_prepared', 'cached_step', 'step', 'run', 'begin_valid', 'begin',
             'command', 'dispatch', 'command_dispatch', 'serve', 'validated', 'start', 'main']
    stderr = 'All terms check, but 19 defs rely on unsafe or foreign code:\n'
    stderr += ''.join('- ' + name + '\n' for name in names)
    p.check_generation(subprocess.CompletedProcess([], 0, '', stderr), False)
    for code, out, err in ((1, '', stderr), (0, 'unexpected', stderr), (0, '', stderr + 'warning\n'),
                           (0, '', stderr.replace('Reply.read', 'Unexpected.effect')), (0, '', '')):
        with pytest.raises(ValueError, match='compiler diagnostic'):
            p.check_generation(subprocess.CompletedProcess([], code, out, err), False)
