"""Status reads one requested run, never whichever global Ray tree is newest."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.script_loading import load_script_module


class Fixture:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch):
        self.root = root
        self.ray = root / 'ray'
        self.proc = root / 'proc'
        self.cwd = root / 'status-cwd'
        self.cwd.mkdir()
        monkeypatch.chdir(self.cwd)
        self.module = load_script_module('status.py', 'o7_status_fixture')
        self.commands: list[list[str]] = []
        self.ps: list[str] = []

        def path(*parts: Any) -> Path:
            result = Path(*parts)
            if result == Path('/tmp/ray'):
                return self.ray
            if result.is_relative_to('/proc'):
                return self.proc / result.relative_to('/proc')
            if result.is_absolute() and not result.is_relative_to(root):
                raise AssertionError(f'unexpected non-fixture path: {result}')
            return result

        def output(argv: list[str], **_kwargs: Any) -> str:
            self.commands.append(argv)
            if argv == ['ps', 'aux']:
                return '\n'.join(self.ps)
            if argv[:2] == ['du', '-sh']:
                assert Path(argv[2]).resolve().is_relative_to(root)
                return '1K inert fixture\n'
            raise AssertionError(f'unexpected external command: {argv}')

        monkeypatch.setattr(self.module, 'Path', path)
        monkeypatch.setattr(self.module.subprocess, 'check_output', output)
        self.monkeypatch = monkeypatch

    def driver(self, run: Path, *, pid: int = 2222, cwd: Path | None = None,
               config: str | None = None, work_dir: str | None = None,
               extra: tuple[str, ...] = ()) -> Path:
        cwd = cwd or self.root
        cwd.mkdir(parents=True, exist_ok=True)
        config_path = cwd / (config or f'{pid}.yaml')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(f'work_dir: {json.dumps(str(run))}\n')
        argv = ['python', '-m', 'chess_anti_engine.run', '--config', str(config_path)]
        if work_dir is not None:
            argv += ['--work-dir', work_dir]
        argv += list(extra)
        directory = self.proc / str(pid)
        directory.mkdir(parents=True)
        (directory / 'cmdline').write_bytes(('\0'.join(argv) + '\0').encode())
        (directory / 'cwd').symlink_to(cwd, target_is_directory=True)
        self.ps.append(f'user {pid} 0.0 0.1 100 100 ? S 00:00 0:00 ' + ' '.join(argv))
        return config_path

    def trial(self, base: Path, *, run: Path, marker: str, iteration: int,
              trial_id: str = 'same_00000', stamp: float = 100,
              metadata: bool = True) -> Path:
        trial = base / f'train_trial_{trial_id}'
        trial.mkdir(parents=True)
        (trial / 'result.json').write_text(json.dumps({
            'trial_id': trial_id, 'iter': iteration,
            'opponent_strength': 111 if marker == 'REQUESTED' else 999,
            'config': {'work_dir': str(run / 'tune')},
        }) + '\n')
        if metadata:
            (trial / 'params.json').write_text(json.dumps({'work_dir': str(run / 'tune')}))
        (trial / 'progress.csv').write_text('iteration\n')
        log = trial / 'distributed_workers/worker_00/worker.log'
        log.parent.mkdir(parents=True)
        log.write_text(f'2026-01-01 01:02:03 INFO batch done: {marker}\n')
        for p in (base, trial, trial / 'result.json', log):
            os.utime(p, (stamp, stamp))
        return trial

    def run(self, run: Path | None, capsys: pytest.CaptureFixture[str]) -> str:
        self.monkeypatch.setattr('sys.argv', ['status.py'] + (['--run', str(run)] if run else []))
        self.module.main()
        return capsys.readouterr().out


@pytest.mark.parametrize('active', [False, True])
def test_requested_run_ignores_unrelated_process_and_ray_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], active: bool,
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested, unrelated = tmp_path / 'requested', tmp_path / 'unrelated'
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    f.trial(f.ray / 'session_new_2222/artifacts/experiment', run=unrelated,
            marker='UNRELATED', iteration=99, stamp=999)
    if active:
        f.driver(unrelated)
    out = f.run(requested, capsys)
    assert 'REQUESTED' in out
    assert 'UNRELATED' not in out
    assert 'PID 2222' not in out
    assert '999.0' not in out
    assert '111.0' in out


@pytest.mark.parametrize('explicit', [False, True])
def test_matching_driver_and_ray_metadata_preserve_useful_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], explicit: bool,
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'requested'
    requested.mkdir()
    f.driver(requested)
    f.trial(f.ray / 'session_new_2222/artifacts/experiment', run=requested,
            marker='REQUESTED', iteration=11)
    out = f.run(requested if explicit else None, capsys)
    assert 'REQUESTED' in out
    assert 'PID 2222' in out
    assert '111.0' in out


def set_argv(f: Fixture, pid: int, *args: str) -> None:
    (f.proc / str(pid) / 'cmdline').write_bytes(('\0'.join(args) + '\0').encode())


@pytest.mark.parametrize('equals', [False, True])
def test_quoted_relative_paths_cli_override_use_process_not_reporter_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], equals: bool,
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    driver_cwd = tmp_path / 'driver space'
    requested = driver_cwd / 'chosen run'
    cfg = f.driver(tmp_path / 'yaml-wrong-run', cwd=driver_cwd, config='config space/run.yaml')
    args = (['--config=config space/run.yaml', '--work-dir=chosen run'] if equals
            else ['--config', 'config space/run.yaml', '--work-dir', 'chosen run'])
    set_argv(f, 2222, 'python3', '-u', '-m', 'chess_anti_engine.run', *args)
    trial = f.trial(f.ray / 'session_x/artifacts/experiment', run=requested,
                    marker='REQUESTED', iteration=11)
    # Relative trial metadata uses the observed driver's cwd, not status cwd.
    (trial / 'params.json').write_text(json.dumps({'work_dir': 'chosen run/tune'}))
    out = f.run(None, capsys)
    assert 'PID 2222' in out
    assert 'REQUESTED' in out
    assert str(cfg) in out
    assert 'yaml-wrong-run' not in out
    assert f.commands[-1] == ['du', '-sh', str(requested)]


@pytest.mark.parametrize('yaml_content', ['', 'work_dir: null\n', 'seed: 42\n'])
def test_actual_default_is_runs_not_config_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], yaml_content: str,
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'runs'
    cfg = f.driver(requested, config='not-the-run-name.yaml')
    cfg.write_text(yaml_content)
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    out = f.run(None, capsys)
    assert 'REQUESTED' in out
    assert 'not-the-run-name' not in out.split('Config:')[0]
    assert f.commands[-1] == ['du', '-sh', str(requested)]


def test_cli_without_config_and_repeated_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'requested'
    f.driver(requested)
    set_argv(f, 2222, 'python', '-m', 'chess_anti_engine.run',
             '--work-dir', str(tmp_path / 'old'), '--work-dir=' + str(requested), '--mode=tune')
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    out = f.run(None, capsys)
    assert 'REQUESTED' in out
    assert 'PID 2222' in out
    assert 'Config:' not in out


@pytest.mark.parametrize('kind', ['missing-cmdline', 'missing-cwd', 'missing-config', 'bad-yaml',
                                   'list-yaml', 'nonstring-root', 'salvage', 'abbreviated',
                                   'missing-value', 'not-module', 'invalid-encoding'])
def test_unknown_process_metadata_never_guesses_run_or_exposes_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], kind: str,
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'requested'
    cfg = f.driver(requested)
    proc = f.proc / '2222'
    if kind.startswith('missing-') and kind != 'missing-value':
        {'missing-cmdline': proc / 'cmdline', 'missing-cwd': proc / 'cwd',
         'missing-config': cfg}[kind].unlink()
    elif kind in ('bad-yaml', 'list-yaml', 'nonstring-root'):
        cfg.write_text({'bad-yaml': 'secret: [TOP_SECRET', 'list-yaml': '[TOP_SECRET]',
                        'nonstring-root': 'work_dir: [TOP_SECRET]'}[kind])
    elif kind == 'invalid-encoding':
        (proc / 'cmdline').write_bytes(b'\xff')
    else:
        extras = {'salvage': ['--mode', 'salvage'], 'abbreviated': ['--work', str(requested)],
                  'missing-value': ['--work-dir'], 'not-module': ['-c', 'TOP_SECRET']}[kind]
        set_argv(f, 2222, 'python', '-m' if kind != 'not-module' else '-c',
                 'chess_anti_engine.run', '--config', str(cfg), *extras)
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    inferred = f.run(None, capsys)
    assert 'Use --run' in inferred
    assert 'TOP_SECRET' not in inferred
    explicit = f.run(requested, capsys)
    assert 'REQUESTED' in explicit
    assert 'PID 2222' not in explicit
    assert 'TOP_SECRET' not in explicit
    assert 'no matching driver observed' in explicit


@pytest.mark.parametrize('same_root', [False, True])
def test_multiple_drivers_never_arbitrarily_attach_first_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], same_root: bool,
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested, other = tmp_path / 'requested', tmp_path / 'other'
    f.driver(requested, pid=1111)
    f.driver(requested if same_root else other, pid=2222)
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    out = f.run(requested, capsys)
    assert 'REQUESTED' in out
    if same_root:
        assert 'multiple matching drivers' in out
        assert 'PID 1111' not in out
        assert 'REQUESTED' in f.run(None, capsys)
    else:
        assert 'PID 1111' in out
        assert 'PID 2222' not in out
        assert 'Use --run' in f.run(None, capsys)


def test_unreadable_candidate_blocks_default_inference_but_not_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'requested'
    f.driver(requested, pid=1111)
    f.driver(tmp_path / 'unknown', pid=2222)
    (f.proc / '2222/cmdline').unlink()
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    assert 'Use --run' in f.run(None, capsys)
    assert 'REQUESTED' in f.run(requested, capsys)


@pytest.mark.parametrize('metadata', ['missing', 'bad-json', 'wrong', 'relative-unbound'])
def test_pid_suffix_never_proves_ray_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], metadata: str,
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested, other = tmp_path / 'requested', tmp_path / 'other'
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    trial = f.trial(f.ray / 'session_new_2222/artifacts/experiment', run=other,
                    marker='UNRELATED', iteration=99, stamp=999)
    params = trial / 'params.json'
    if metadata == 'missing':
        params.unlink()
    elif metadata == 'bad-json':
        params.write_text('{')
    elif metadata == 'relative-unbound':
        params.write_text(json.dumps({'work_dir': 'requested/tune'}))
    out = f.run(requested, capsys)
    assert 'REQUESTED' in out
    assert 'UNRELATED' not in out


def test_stale_yaml_cannot_attach_old_ray_artifacts_to_new_configured_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    old, new = tmp_path / 'old', tmp_path / 'new'
    cfg = f.driver(old)
    f.trial(f.ray / 'session_new_2222/artifacts/experiment', run=old,
            marker='OLD_RUNTIME', iteration=99)
    cfg.write_text(f'work_dir: {json.dumps(str(new))}\n')
    out = f.run(None, capsys)
    assert 'observed metadata' in out
    assert 'No result.json' in out
    assert 'OLD_RUNTIME' not in out
    # Explicit old root can still use absolute realized trial metadata, without
    # claiming the driver currently observed through changed YAML belongs to it.
    out = f.run(old, capsys)
    assert 'OLD_RUNTIME' in out
    assert 'PID 2222' not in out


def test_each_ray_trial_requires_binding_and_logs_use_selected_trial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested, other = tmp_path / 'requested', tmp_path / 'other'
    experiment = f.ray / 'session_old_1111/artifacts/experiment'
    selected = f.trial(experiment, run=requested, marker='REQUESTED', iteration=11)
    f.trial(experiment, run=other, marker='MIXED_UNRELATED', iteration=99, trial_id='other_00000')
    # Same trial ID in newer global session and durable sibling cannot replace
    # the selected row or its throughput, even when their log timestamps win.
    f.trial(f.ray / 'session_new_2222/artifacts/experiment', run=other,
            marker='UNRELATED', iteration=99, stamp=999)
    f.trial(requested / 'old-snapshot', run=requested, marker='OLD_SIBLING', iteration=2, stamp=1000)
    # Force experiment ordering independently of directory insertion effects.
    os.utime(experiment.parent.parent, (100, 100))
    os.utime(f.ray / 'session_new_2222', (999, 999))
    out = f.run(requested, capsys)
    assert 'REQUESTED' in out
    assert 'UNRELATED' not in out
    assert 'OLD_SIBLING' not in out
    assert str(selected) not in out


def test_single_result_selection_snapshot_survives_later_global_ray_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested, other = tmp_path / 'requested', tmp_path / 'other'
    f.trial(f.ray / 'session_first/artifacts/experiment', run=requested,
            marker='REQUESTED', iteration=11)
    original = f.module._find_trial_result_jsons
    selections = []

    def select(*args: Any, **kwargs: Any) -> list[Path]:
        selections.append(True)
        assert len(selections) == 1
        selected = original(*args, **kwargs)
        f.trial(f.ray / 'session_later/artifacts/experiment', run=other,
                marker='LATER_UNRELATED', iteration=99, stamp=999)
        return selected

    monkeypatch.setattr(f.module, '_find_trial_result_jsons', select)
    out = f.run(requested, capsys)
    assert selections == [True]
    assert 'REQUESTED' in out
    assert 'LATER_UNRELATED' not in out


def test_run_alias_and_glob_characters_do_not_change_log_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'requested[1]'
    requested.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(requested, target_is_directory=True)
    f.driver(requested)
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    out = f.run(alias, capsys)
    assert 'REQUESTED' in out
    assert 'PID 2222' in out
    assert f.commands[-1] == ['du', '-sh', str(requested)]


def test_duplicate_command_lines_are_distinct_driver_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'requested'
    f.driver(requested, pid=1111, config='shared.yaml')
    f.driver(requested, pid=2222, config='shared.yaml')
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    out = f.run(None, capsys)
    assert 'multiple matching drivers' in out
    assert 'REQUESTED' in out


def test_matching_drivers_with_different_cwds_do_not_guess_relative_ray_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'requested'
    f.driver(requested, pid=1111, cwd=tmp_path / 'cwd-one')
    f.driver(requested, pid=2222, cwd=tmp_path / 'cwd-two')
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    ray = f.trial(f.ray / 'session_new/artifacts/experiment', run=requested,
                  marker='AMBIGUOUS', iteration=99)
    (ray / 'params.json').write_text(json.dumps({'work_dir': '../requested/tune'}))
    out = f.run(requested, capsys)
    assert 'REQUESTED' in out
    assert 'AMBIGUOUS' not in out


def test_winning_result_path_not_trial_id_glob_controls_worker_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'requested'
    f.trial(requested / 'new', run=requested, marker='REQUESTED', iteration=11, stamp=100)
    f.trial(requested / 'old', run=requested, marker='LOSING_SAME_ID', iteration=2, stamp=999)
    out = f.run(requested, capsys)
    assert 'REQUESTED' in out
    assert 'LOSING_SAME_ID' not in out


def test_missing_result_trial_id_uses_same_fallback_for_table_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'requested'
    trial = f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    result = trial / 'result.json'
    row = json.loads(result.read_text())
    del row['trial_id']
    result.write_text(json.dumps(row) + '\n')
    out = f.run(requested, capsys)
    assert 'same_00000' in out
    assert 'REQUESTED' in out


def test_ps_failure_retains_explicit_durable_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'requested'
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    original = f.module.subprocess.check_output

    def output(argv: list[str], **kwargs: Any) -> str:
        if argv == ['ps', 'aux']:
            raise OSError('inert inaccessible process table')
        return original(argv, **kwargs)

    monkeypatch.setattr(f.module.subprocess, 'check_output', output)
    out = f.run(requested, capsys)
    assert 'REQUESTED' in out
    assert 'no matching driver observed' in out


def test_no_matching_data_does_not_fall_back_to_global_ray(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested, other = tmp_path / 'requested', tmp_path / 'other'
    f.driver(requested)
    f.trial(f.ray / 'session_new_2222/artifacts/experiment', run=other,
            marker='UNRELATED', iteration=99)
    out = f.run(requested, capsys)
    assert 'No result.json' in out
    assert 'UNRELATED' not in out


def test_literal_tilde_in_config_is_not_expanded_like_explicit_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / '~/requested'
    config = f.driver(requested)
    config.write_text('work_dir: ~/requested\n')
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    out = f.run(None, capsys)
    assert 'REQUESTED' in out
    assert f.commands[-1] == ['du', '-sh', str(requested)]


def test_configless_driver_uses_actual_default_under_its_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'runs'
    f.driver(requested)
    set_argv(f, 2222, 'python', '-m', 'chess_anti_engine.run')
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    out = f.run(None, capsys)
    assert 'REQUESTED' in out
    assert 'Config:' not in out
    assert f.commands[-1] == ['du', '-sh', str(requested)]


def test_yaml_mode_and_last_config_override_are_observed_before_cli_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'requested'
    config = f.driver(requested)
    config.write_text(f'work_dir: {json.dumps(str(requested))}\nmode: salvage\n')
    first = tmp_path / 'first.yaml'
    first.write_text(f'work_dir: {json.dumps(str(tmp_path / "wrong"))}\n')
    set_argv(f, 2222, 'python', '-m', 'chess_anti_engine.run',
             '--config', str(first), '--config=' + str(config))
    assert 'Use --run' in f.run(None, capsys)
    set_argv(f, 2222, 'python', '-m', 'chess_anti_engine.run',
             '--config', str(first), '--config=' + str(config), '--mode', 'train')
    f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    assert 'REQUESTED' in f.run(None, capsys)


def test_selected_worker_log_is_streamed_without_materializing_growing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    f = Fixture(tmp_path, monkeypatch)
    requested = tmp_path / 'requested'
    trial = f.trial(requested / 'tune', run=requested, marker='REQUESTED', iteration=11)
    log = trial / 'distributed_workers/worker_00/worker.log'
    log.write_text('2026-01-01 00:00:00 INFO batch done: EARLIER\n'
                   '2026-01-01 01:02:03 INFO batch done: REQUESTED\n'
                   'not a batch\n')
    original = Path.read_text

    def guarded(path: Path, *args: Any, **kwargs: Any) -> str:
        assert path.name != 'worker.log', 'growing worker logs must be streamed'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', guarded)
    out = f.run(requested, capsys)
    assert 'REQUESTED' in out
    assert 'EARLIER' not in out
