"""Explicit CPU target, capability admission and diagnostic propagation contracts."""
from pathlib import Path

import pytest

from native.bend_engine.u64_map_probe import cpu_target as target


def test_target_is_explicit_and_not_host_auto_selection() -> None:
    assert target.TARGET_NAME == 'bmi2-popcnt'
    assert target.TARGET_FLAGS == ('-march=x86-64', '-mpopcnt', '-mbmi2')


@pytest.mark.parametrize('text', ['', 'bmi2=0 popcnt=1\n', 'bmi2=1 popcnt=0\n',
                                  'bmi2=0 popcnt=0\n', 'bmi2=1 popcnt=1',
                                  'bmi2=1 popcnt=1\nextra\n'])
def test_unknown_or_unsupported_host_is_not_a_skip(text: str) -> None:
    with pytest.raises(ValueError, match='no target fallback'):
        target.check_host(text)


def test_exact_host_capabilities() -> None:
    target.check_host('bmi2=1 popcnt=1\n')


def test_baseline_precedes_target_and_report_records_flags(tmp_path: Path) -> None:
    calls: list[tuple[list[str], str]] = []

    def command(argv: list[str], stage: str) -> str:
        calls.append((argv, stage))
        return {'cpu-host-run': 'bmi2=1 popcnt=1\n', 'cpu-target-run': 'bmi2-popcnt-ok\n'}.get(stage, '')

    report = target.qualify('clang', tmp_path, command)
    assert [stage for _, stage in calls] == ['cpu-host-build', 'cpu-host-run', 'cpu-target-build', 'cpu-target-run']
    assert '-mbmi2' not in calls[0][0]
    assert all(flag in calls[2][0] for flag in target.TARGET_FLAGS)
    assert report['flags'] == list(target.TARGET_FLAGS)
    assert report['instruction_check'] == 'passed'
    assert (tmp_path / 'cpu-target.c').read_text() == target.TARGET_SOURCE


def test_unsupported_host_never_builds_or_runs_target(tmp_path: Path) -> None:
    stages: list[str] = []

    def command(_argv: list[str], stage: str) -> str:
        stages.append(stage)
        return 'bmi2=0 popcnt=1\n' if stage == 'cpu-host-run' else ''

    with pytest.raises(ValueError, match='host check failed'):
        target.qualify('clang', tmp_path, command)
    assert stages == ['cpu-host-build', 'cpu-host-run']


@pytest.mark.parametrize('stage', ['cpu-host-build', 'cpu-host-run', 'cpu-target-build', 'cpu-target-run'])
def test_command_failure_or_diagnostic_is_not_swallowed(tmp_path: Path, stage: str) -> None:
    def command(_argv: list[str], label: str) -> str:
        if label == stage:
            raise RuntimeError('compiler diagnostic or execution failure')
        return 'bmi2=1 popcnt=1\n' if label == 'cpu-host-run' else ''

    with pytest.raises(RuntimeError, match='compiler diagnostic'):
        target.qualify('clang', tmp_path, command)


@pytest.mark.parametrize('text', ['', 'bmi2-popcnt-ok', 'bmi2-popcnt-ok\nextra\n', 'wrong\n'])
def test_instruction_probe_needs_exact_result(tmp_path: Path, text: str) -> None:
    def command(_argv: list[str], stage: str) -> str:
        return {'cpu-host-run': 'bmi2=1 popcnt=1\n', 'cpu-target-run': text}.get(stage, '')

    with pytest.raises(ValueError, match='instruction check failed'):
        target.qualify('clang', tmp_path, command)
