"""The shared gate must execute its advertised scope and propagate failures."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import pytest
import torch

from scripts import validate


@pytest.fixture(autouse=True)
def _platform_string_is_cached() -> None:
    """platform.platform() shells out once per process (``uname -p``) and caches.

    Tests here replace ``subprocess.run``; without the cache each would depend
    on an earlier test having run ``validate.main`` unpatched.
    """
    platform.platform()


@pytest.fixture(autouse=True)
def _no_change_scoping(monkeypatch: pytest.MonkeyPatch) -> None:
    # On a CI pull request these are set, and main() would shell out to git
    # through the subprocess.run fakes below; scoping has its own test file.
    for name in ("GITHUB_ACTIONS", "GITHUB_EVENT_NAME", "GITHUB_BASE_REF", validate.DESELECT_LIST_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("suite", ["cpu", "pext", "capped"])
def test_each_test_run_isolates_inherited_caches_and_cleans_up_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suite: str,
) -> None:
    shared_cache = tmp_path / "live-cache"
    shared_cache.mkdir()
    sentinel = shared_cache / "active-artifact"
    sentinel.write_text("preserve")
    for name in ("DEEPFIN_COMPILE_CACHE", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.setenv(name, str(shared_cache))
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "16")
    monkeypatch.setenv("MAX_JOBS", "16")
    monkeypatch.setattr(validate, "preflight", lambda *_args: None)
    report = tmp_path / "child-environment.json"
    reported_keys = (
        "DEEPFIN_COMPILE_CACHE", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_COMPILE_THREADS", "MAX_JOBS",
    )
    caches: list[Path] = []
    for exit_code in (0, 7):
        code = (
            "import json, os; from pathlib import Path; "
            "root = Path(os.environ['DEEPFIN_COMPILE_CACHE']); "
            "assert root.is_dir(); "
            "(root / 'test-artifact').write_text('temporary'); "
            f"Path({str(report)!r}).write_text(json.dumps({{key: os.environ[key] for key in {reported_keys!r}}})); "
            f"raise SystemExit({exit_code})"
        )
        monkeypatch.setattr(validate, "suite_command", lambda _suite, code=code: [sys.executable, "-c", code])
        assert validate.main([suite, "--threads", "2", "--expect-backend",
                              "pext" if suite == "pext" else "magic",
                              "--report-dir", str(tmp_path / "reports")]) == exit_code
        env = json.loads(report.read_text())
        cache = Path(env["DEEPFIN_COMPILE_CACHE"])
        assert cache != shared_cache
        assert env["TORCHINDUCTOR_CACHE_DIR"] == str(cache / "compile_cache" / "torchinductor")
        assert env["TRITON_CACHE_DIR"] == str(cache / "compile_cache" / "triton")
        assert env["TORCHINDUCTOR_COMPILE_THREADS"] == env["MAX_JOBS"] == "1"
        assert not cache.exists()
        caches.append(cache)
    assert caches[0] != caches[1]
    assert sentinel.read_text() == "preserve"
    assert list(shared_cache.iterdir()) == [sentinel]
    assert os.environ["DEEPFIN_COMPILE_CACHE"] == str(shared_cache)


@pytest.mark.parametrize("explicit_compiler", [None, "/chosen/toolchain/c++"])
def test_linux_compile_environment_avoids_path_compiler_but_respects_explicit_choice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, explicit_compiler: str | None,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "access", lambda *_args: True)
    env = {"PATH": "/newer-gcc/bin:/usr/bin"}
    if explicit_compiler is not None:
        env["CXX"] = explicit_compiler
    validate.configure_compile_environment(env, tmp_path)
    assert env["CXX"] == (explicit_compiler or "/usr/bin/c++")


def test_capped_regression_overrides_ci_auto_and_child_thread_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAE_TEST_THREADS", "auto")
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = validate.validation_environment("capped", "auto", "magic")
    assert env["CAE_TEST_THREADS"] == "2"
    assert all(env[key] == "2" for key in validate.THREAD_VARS)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["OMP_NUM_THREADS"] == "16"
    command = validate.suite_command("capped")
    assert command[-1] == "tests/test_pytest_thread_cap.py"
    assert "test_gen_random_selfplay_shards.py::" in command[-3]
    assert "test_gen_nnue_selfplay_shards.py::" in command[-2]


@pytest.mark.parametrize("suite", ["cpu", "pext", "capped", "lint"])
def test_runner_propagates_real_child_failure_and_prints_regime(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, suite: str,
) -> None:
    monkeypatch.delenv("CAE_TEST_THREADS", raising=False)
    monkeypatch.delenv("CAE_EXPECT_SLIDER_BACKEND", raising=False)
    monkeypatch.setattr(validate, "preflight", lambda *_args: None)
    # A real failed child demonstrates that the runner cannot print PASS after
    # a failed lint/test command, including an otherwise empty child stdout.
    monkeypatch.setattr(validate, "suite_command", lambda _suite: [sys.executable, "-c", "raise SystemExit(7)"])
    assert validate.main([suite, "--report-dir", str(tmp_path / "reports")]) == 7
    out = capsys.readouterr()
    assert f"{suite}: FAIL" in out.out
    assert "threads=2" in out.out
    assert "CUDA_VISIBLE_DEVICES=''" in out.out
    assert "exit 7" in out.out
    assert "rerun" in out.err
    assert "PASS" not in out.out


def test_missing_native_build_stops_before_pytest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    monkeypatch.setattr(validate, "ROOT", tmp_path)
    report = tmp_path / "artifacts/validation/cpu.xml"
    report.parent.mkdir(parents=True)
    report.write_text("stale passing result")
    called: list[object] = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **_kwargs: called.append(args))
    assert validate.main(["cpu", "--threads", "2", "--expect-backend", "magic"]) == 1
    assert not called
    assert not report.exists()
    assert "freshness check failed" in capsys.readouterr().err


def test_cpu_wheel_requirement_rejects_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    with pytest.raises(RuntimeError, match="CPU wheel required"):
        validate.preflight("lint", "magic", True)
    # CUDA development environments may still run the CPU suite, which hides
    # devices in the child. The wheel assertion is a separate CI requirement.
    validate.preflight("lint", "magic", False)


@pytest.mark.parametrize("matching_prefix", [False, True])
def test_lint_checks_environment_prefix_before_launching_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], matching_prefix: bool,
) -> None:
    checkout_venv = tmp_path / ".venv"
    checkout_venv.mkdir()
    monkeypatch.setattr(validate, "ROOT", tmp_path)
    # Keep sys.executable unchanged: two environments can point at the same
    # base interpreter while supplying different versions of the lint tools.
    monkeypatch.setattr(sys, "prefix", str(checkout_venv if matching_prefix else tmp_path / "other-env"))
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    result = validate.main(["lint", "--threads", "2"])
    output = capsys.readouterr()
    if matching_prefix:
        assert result == 0
        assert commands == [["bash", "scripts/lint.sh"]]
    else:
        assert result == 1
        assert not commands
        assert "lint environment mismatch" in output.err
        assert str(checkout_venv / "bin/python") in output.err
        assert "PASS" not in output.out


def test_pext_cannot_silently_validate_magic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAE_EXPECT_SLIDER_BACKEND", "magic")
    with pytest.raises(SystemExit) as exc:
        validate.main(["pext"])
    assert exc.value.code == 2


def test_backend_mismatch_fails_before_test_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import check_c_extensions_fresh

    monkeypatch.setattr(check_c_extensions_fresh, "check_extensions", lambda _root: [])

    class NativeModule:
        SLIDER_BACKEND = "magic"
        __file__ = str(validate.ROOT / "chess_anti_engine/encoding/_lc0_ext.so")

    monkeypatch.setattr(validate.importlib, "import_module", lambda _name: NativeModule)
    with pytest.raises(RuntimeError, match="expected pext, found magic"):
        validate.preflight("pext", "pext", False)


def test_named_commands_keep_whole_repo_lint_and_full_cpu_scope() -> None:
    assert validate.suite_command("lint") == ["bash", "scripts/lint.sh"]
    command = validate.suite_command("cpu")
    assert command[:3] == [sys.executable, "-m", "pytest"]
    assert command[command.index("-m", 3) + 1] == "not slow"
    assert not any(arg.startswith("tests/") for arg in command)
    command = validate.suite_command("pext")
    paths = [arg for arg in command if arg.startswith("tests/")]
    assert paths
    assert all((validate.ROOT / path).is_file() for path in paths)
    assert "tests/test_slider_tables.py" in command
    assert "tests/test_c_extension_freshness.py" in command


@pytest.mark.parametrize("passes", [True, False])
def test_runner_saves_real_pytest_results_and_timings_even_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, passes: bool,
) -> None:
    probe = tmp_path / "test_probe.py"
    probe.write_text(f"def test_observed_behavior():\n    assert {passes!r}\n")
    previous = tmp_path / "artifacts/validation/cpu.xml"
    previous.parent.mkdir(parents=True)
    previous.write_text("previous run")
    command = [*validate.suite_command("cpu"), str(probe)]
    monkeypatch.setattr(validate, "ROOT", tmp_path)
    monkeypatch.setattr(validate, "preflight", lambda *_args: None)
    monkeypatch.setattr(validate, "suite_command", lambda _suite: command.copy())
    assert validate.main(["cpu", "--threads", "2", "--report-dir", "results"]) == (0 if passes else 1)
    report = ET.parse(tmp_path / "results/cpu.xml")
    cases = report.findall(".//testcase")
    assert len(cases) == 1
    assert cases[0].get("name") == "test_observed_behavior"
    assert float(cases[0].attrib["time"]) >= 0
    assert (cases[0].find("failure") is None) == passes
    assert previous.read_text() == "previous run"


def test_ci_invokes_shared_suites_and_canary_keeps_same_lint_scope() -> None:
    import yaml

    ci: dict[str, Any] = yaml.safe_load((validate.ROOT / ".github/workflows/ci.yml").read_text())
    commands = [step.get("run", "") for job in ci["jobs"].values() for step in job["steps"]]
    for suite in ("cpu", "pext", "capped", "lint", "join-race"):
        assert any(f"scripts/validate.py {suite} --require-cpu-wheel" in command for command in commands)
    assert not any("python -m pytest" in command or "ruff check" in command for command in commands)
    canary: dict[str, Any] = yaml.safe_load((validate.ROOT / ".github/workflows/lint-canary.yml").read_text())
    assert any("scripts/validate.py lint --require-cpu-wheel" in step.get("run", "")
               for step in canary["jobs"]["canary"]["steps"])


def test_record_only_fast_path_keeps_the_full_suites_as_the_default() -> None:
    import yaml

    ci: dict[str, Any] = yaml.safe_load((validate.ROOT / ".github/workflows/ci.yml").read_text())
    jobs = ci["jobs"]
    flag = "needs.classify.outputs.records_only"
    assert jobs["classify"]["outputs"] == {"records_only": "${{ steps.scope.outputs.records_only }}"}
    # The shards run unless the flag is exactly 'true' (unset on any classifier failure),
    # and also when the classify job itself failed.
    shards = jobs["test-shard"]
    assert shards["needs"] == "classify"
    assert shards["if"] == f"${{{{ !cancelled() && {flag} != 'true' }}}}"
    steps = {step.get("name"): step for step in jobs["test"]["steps"]}
    records = steps["Record-reading tests (record-only pull request)"]
    assert records["if"] == f"{flag} == 'true'"
    assert "scripts/validate.py records --require-cpu-wheel" in records["run"]
    (scope,) = jobs["classify"]["steps"]
    assert scope["id"] == "scope"
    classify = scope["run"]
    assert "'^docs/experiments/[^/]+\\.md$'" in classify
    assert '[ "$n" -gt 0 ] && [ "$other" -eq 0 ]' in classify
    assert "previous_filename" in classify


def test_ci_shards_the_cpu_suite_and_the_required_check_needs_every_shard() -> None:
    import yaml

    ci: dict[str, Any] = yaml.safe_load((validate.ROOT / ".github/workflows/ci.yml").read_text())
    shards = ci["jobs"]["test-shard"]
    assert shards["strategy"]["fail-fast"] is False
    matrix = shards["strategy"]["matrix"]["shard"]
    count = len(matrix)
    assert matrix == list(range(1, count + 1))
    steps = {step.get("name"): step for step in shards["steps"]}
    cpu = steps["Ordinary CPU suite (one shard)"]
    assert "if" not in cpu
    assert cpu["run"] == f"python scripts/validate.py cpu --require-cpu-wheel --shard ${{{{ matrix.shard }}}}/{count}"
    assert steps["Capped local regression"]["if"] == "matrix.shard == 1"
    assert steps["Upload test timings and results"]["with"]["name"] == "cpu-test-results-shard-${{ matrix.shard }}"
    # `test` is the required check: it always runs, and off the record-only path
    # anything but a successful matrix (failed, cancelled, skipped) fails it.
    test = ci["jobs"]["test"]
    assert test["needs"] == ["classify", "test-shard"]
    assert test["if"] == "always()"
    gate = test["steps"][0]
    assert gate["if"] == "needs.classify.outputs.records_only != 'true'"
    assert gate["env"] == {"SHARDS_RESULT": "${{ needs.test-shard.result }}"}
    assert '[ "$SHARDS_RESULT" = "success" ]' in gate["run"].splitlines()
    assert "continue-on-error" not in gate
    # After the gate, the shards' reports must show the serial phase ran tests.
    download, serial = test["steps"][1:3]
    assert download["if"] == serial["if"] == gate["if"]
    assert download["uses"].startswith("actions/download-artifact@")
    assert download["with"] == {"pattern": "cpu-test-results-shard-*", "path": "shard-reports"}
    assert "shard-reports/*/cpu-serial.xml" in serial["run"]
    assert '[ "$n" -gt 0 ]' in serial["run"].splitlines()[-1]
    assert "continue-on-error" not in download
    assert "continue-on-error" not in serial
    assert "continue-on-error" not in test


def _ci_shard_count() -> int:
    import yaml

    ci: dict[str, Any] = yaml.safe_load((validate.ROOT / ".github/workflows/ci.yml").read_text())
    return len(ci["jobs"]["test-shard"]["strategy"]["matrix"]["shard"])


def test_shards_partition_every_test_file_in_both_phases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Read from the pytest commands the runner launches, one run per shard."""
    monkeypatch.delenv("CAE_TEST_WORKERS", raising=False)
    monkeypatch.setattr(validate, "preflight", lambda *_args: None)
    monkeypatch.setattr(validate, "apply_change_scope", lambda *_args: frozenset())
    seen: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(validate.subprocess, "run", fake_run)
    count = _ci_shard_count()
    # Every file pytest would collect by its default python_files patterns.
    tests = validate.ROOT / "tests"
    collected = sorted({path.relative_to(validate.ROOT).as_posix()
                        for pattern in ("test_*.py", "*_test.py") for path in tests.rglob(pattern)})
    assert len(collected) > 800
    timings = json.loads((validate.ROOT / validate.SHARD_DURATIONS).read_text())
    assigned: list[str] = []
    for index in range(1, count + 1):
        seen.clear()
        assert validate.main(["cpu", "--threads", "auto", "--shard", f"{index}/{count}",
                              "--report-dir", str(tmp_path)]) == 0
        parallel, serial = ([arg for arg in command if arg.startswith("tests/")] for command in seen)
        assert parallel == serial
        assert parallel
        assigned += parallel
    assert sorted(assigned) == collected
    # Files the timings do not list are assigned too, and a stale timing adds nothing.
    extra = [*collected, "tests/test_not_in_the_timings.py", "tests/sub/other_test.py"]
    assert not {*extra[-2:]} & {*timings["serial"], *timings["xdist"]}
    stale = {phase: {**files, "tests/test_deleted_long_ago.py": 500.0} for phase, files in timings.items()}
    for durations in (timings, stale, {}):
        shards = validate.assign_shards(extra, durations, count)
        assert len(shards) == count
        assert sorted(name for shard in shards for name in shard) == sorted(extra)
    assert validate.assign_shards(collected, timings, count) == validate.assign_shards(reversed(collected), timings, count)


@pytest.mark.parametrize("value", ["0/10", "11/10", "1/0", "3", "1/10/2", "a/b", "-1/10", " 1/10", "1.0/10", ""])
def test_shard_rejects_malformed_and_out_of_range_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str,
) -> None:
    def no_launch(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a rejected --shard must not launch anything")

    monkeypatch.setattr(validate, "preflight", no_launch)
    monkeypatch.setattr(validate.subprocess, "run", no_launch)
    with pytest.raises(SystemExit) as exit_info:
        validate.main(["cpu", f"--shard={value}", "--report-dir", str(tmp_path)])
    assert exit_info.value.code == 2


def test_shard_applies_only_to_the_cpu_suite(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        validate.main(["capped", "--shard", "1/2", "--report-dir", str(tmp_path)])
    assert exit_info.value.code == 2
    assert validate.parse_shard("10/10") == (10, 10)


def test_shard_universe_is_recursive_and_takes_both_pytest_name_patterns(
    tmp_path: Path, pytestconfig: pytest.Config,
) -> None:
    for name in ("tests/test_top.py", "tests/sub/test_nested.py", "tests/b_test.py", "tests/sub/deep/c_test.py",
                 "tests/helper.py", "tests/conftest.py", "scripts/test_outside.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("")
    assert validate.shard_universe(tmp_path) == [
        "tests/b_test.py", "tests/sub/deep/c_test.py", "tests/sub/test_nested.py", "tests/test_top.py"]
    # The hand-written universe matches pytest only while these stay as they are.
    assert pytestconfig.getini("testpaths") == ["tests"]
    assert pytestconfig.getini("python_files") == ["test_*.py", "*_test.py"]


@pytest.mark.parametrize(
    ("deselect", "phase_codes", "expected"),
    [
        # pytest exits 5 when a phase selected no test. One empty phase passes
        # when the other ran tests, whichever it is and whatever the files are.
        ((), (5, 0), 0),
        ((), (0, 5), 0),
        ((), (0, 0), 0),
        # A real failure is never hidden by the other phase being empty or green.
        ((), (5, 1), 1),
        ((), (1, 5), 1),
        ((), (0, 2), 2),
        ((), (3, 0), 3),
        ((), (1, 2), 1),
        # A shard that ran nothing at all fails while any of its files is selected.
        ((), (5, 5), 5),
        (("tests/test_b.py",), (5, 5), 5),
        (("tests/test_b.py",), (0, 5), 0),
        # Change scoping deselected every file of the shard.
        (("tests/test_a.py", "tests/test_b.py"), (5, 5), 0),
        (("tests/test_a.py", "tests/test_b.py"), (5, 2), 2),
    ],
)
def test_an_empty_shard_phase_passes_and_a_shard_that_ran_nothing_does_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, deselect: tuple[str, ...],
    phase_codes: tuple[int, int], expected: int,
) -> None:
    monkeypatch.delenv("CAE_TEST_WORKERS", raising=False)
    (tmp_path / "tests").mkdir()
    for stem in ("a", "b"):
        (tmp_path / f"tests/test_{stem}.py").write_text("def test_x():\n    pass\n")
    (tmp_path / validate.SHARD_DURATIONS).write_text("{}")
    monkeypatch.setattr(validate, "ROOT", tmp_path)
    monkeypatch.setattr(validate, "preflight", lambda *_args: None)
    monkeypatch.setattr(validate, "apply_change_scope", lambda *_args: frozenset(deselect))
    codes = iter(phase_codes)
    monkeypatch.setattr(validate.subprocess, "run",
                        lambda command, **_kwargs: subprocess.CompletedProcess(command, next(codes)))
    assert validate.main(["cpu", "--threads", "auto", "--shard", "1/1", "--report-dir", "results"]) == expected
    # Unsharded, an empty phase stays a failure.
    if 5 in phase_codes:
        codes = iter(phase_codes)
        assert validate.main(["cpu", "--threads", "auto", "--report-dir", "results"]) != 0
    # One process for both phases: 5 passes only when no file is selected at all.
    codes = iter((5,))
    single = validate.main(["cpu", "--threads", "2", "--shard", "1/1", "--report-dir", "results"])
    assert single == (0 if len(deselect) == 2 else 5)


_MARK_SERIAL_CONFTEST = """
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "subprocess" in item.path.read_text():
            item.add_marker(pytest.mark.serial)
"""


@pytest.mark.parametrize(
    ("lone_xdist_file", "serial_passes", "expected"),
    [
        # A test file that defines no test (tests/test_run_bootstrap.py was one).
        ('"""Helpers only."""\n', True, 0),
        ("import pytest\n\n\n@pytest.mark.slow\ndef test_x():\n    pass\n", True, 0),
        ('import pytest\n\npytest.skip("not here", allow_module_level=True)\n', True, 0),
        ('"""Helpers only."""\n', False, 1),
    ],
)
def test_a_file_with_no_runnable_test_cannot_fail_its_shard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lone_xdist_file: str, serial_passes: bool, expected: int,
) -> None:
    """Real pytest runs: the only xdist-phase file of a shard selects nothing.

    This failed shard 8/10 once a new file moved the partition so that
    tests/test_run_bootstrap.py, then test-less, was the shard's one non-serial file.
    """
    monkeypatch.delenv("CAE_TEST_WORKERS", raising=False)
    (tmp_path / "tests").mkdir()
    (tmp_path / "conftest.py").write_text(_MARK_SERIAL_CONFTEST)
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    serial\n    slow\n")
    (tmp_path / "tests/test_serial.py").write_text(
        f"import subprocess\n\n\ndef test_runs():\n    assert subprocess and {serial_passes!r}\n")
    (tmp_path / "tests/test_lone.py").write_text(lone_xdist_file)
    (tmp_path / validate.SHARD_DURATIONS).write_text("{}")
    monkeypatch.setattr(validate, "ROOT", tmp_path)
    monkeypatch.setattr(validate, "preflight", lambda *_args: None)
    monkeypatch.setattr(validate, "apply_change_scope", lambda *_args: frozenset())
    # Two phases without needing xdist here: only the split matters, and plain
    # pytest reports an empty selection with the same exit code 5.
    monkeypatch.setattr(validate, "parallel_args", lambda *_args: ["-p", "no:cacheprovider"])
    assert validate.main(["cpu", "--threads", "auto", "--shard", "1/1", "--report-dir", "results"]) == expected
    assert not ET.parse(tmp_path / "results/cpu.xml").findall(".//testcase[@name='test_x']")
    assert len(ET.parse(tmp_path / "results/cpu-serial.xml").findall(".//testcase[@name='test_runs']")) == 1


def test_change_scope_hands_the_shard_the_same_deselection_it_hands_pytest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(validate, "changed_files", lambda *_args: (["docs/x.md"], "ok"))
    monkeypatch.setattr(validate, "script_test_deselection",
                        lambda *_args: (frozenset({"tests/test_b.py"}), "one"))
    env: dict[str, str] = {}
    assert validate.apply_change_scope(env, None, tmp_path) == frozenset({"tests/test_b.py"})
    assert Path(env[validate.DESELECT_LIST_ENV]).read_text().split() == ["tests/test_b.py"]
    monkeypatch.setattr(validate, "changed_files", lambda *_args: (None, "not a pull request"))
    assert validate.apply_change_scope({}, None, tmp_path) == frozenset()


def test_every_test_that_reads_the_records_is_in_the_records_suite() -> None:
    readers = {f"tests/{path.name}" for path in (validate.ROOT / "tests").glob("test_*.py")
               if "docs/experiments" in path.read_text()}
    assert readers <= set(validate.RECORD_TESTS), sorted(readers - set(validate.RECORD_TESTS))
    assert validate.suite_command("records")[-len(validate.RECORD_TESTS):] == list(validate.RECORD_TESTS)


def test_only_the_cpu_suite_is_spread_across_processes() -> None:
    assert validate.parallel_args("cpu", "auto") == ["-n", "auto", "--dist", "loadfile"]
    assert validate.parallel_args("cpu", "3") == ["-n", "3", "--dist", "loadfile"]
    assert validate.parallel_args("cpu", "1") == []
    for suite in ("capped", "pext", "lint"):
        assert validate.parallel_args(suite, "auto") == []


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--threads", "auto"], ["-n", "auto", "--dist", "loadfile"]),
        (["--threads", "2"], []),
        (["--threads", "2", "--workers", "4"], ["-n", "4", "--dist", "loadfile"]),
        (["--threads", "auto", "--workers", "1"], []),
    ],
)
def test_worker_default_follows_the_thread_regime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str], expected: list[str],
) -> None:
    monkeypatch.delenv("CAE_TEST_WORKERS", raising=False)
    monkeypatch.setattr(validate, "preflight", lambda *_args: None)
    seen: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(validate.subprocess, "run", fake_run)
    assert validate.main(["cpu", *argv, "--report-dir", str(tmp_path)]) == 0
    extra = [arg for arg in seen[0] if arg in {"-n", "--dist", "loadfile"} or arg in {"auto", "4"}]
    assert extra == expected


@pytest.mark.parametrize(
    ("phase_codes", "expected"),
    [((0, 0), 0), ((0, 3), 3), ((3, 0), 3), ((1, 2), 1)],
)
def test_parallel_cpu_run_adds_a_one_process_serial_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase_codes: tuple[int, int], expected: int,
) -> None:
    monkeypatch.delenv("CAE_TEST_WORKERS", raising=False)
    monkeypatch.setattr(validate, "preflight", lambda *_args: None)
    seen: list[list[str]] = []
    envs: list[dict[str, str]] = []
    codes = iter(phase_codes)

    def fake_run(command: list[str], *, env: dict[str, str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        envs.append(env)
        return subprocess.CompletedProcess(command, next(codes))

    monkeypatch.setattr(validate.subprocess, "run", fake_run)
    # Either phase's failure is the run's result, and both phases always run.
    assert validate.main(["cpu", "--threads", "auto", "--report-dir", str(tmp_path)]) == expected
    parallel, serial = seen
    # Workers are the parallelism in the xdist phase; the serial phase is uncapped.
    assert envs[0]["CAE_TEST_THREADS"] == envs[0]["OMP_NUM_THREADS"] == "1"
    assert envs[1]["CAE_TEST_THREADS"] == "auto"
    assert parallel[parallel.index("-m", 3) + 1] == "(not slow) and not serial"
    assert serial[serial.index("-m", 3) + 1] == "(not slow) and serial"
    assert "-n" in parallel
    assert "-n" not in serial
    assert f"--junitxml={tmp_path / 'cpu.xml'}" in parallel
    assert f"--junitxml={tmp_path / 'cpu-serial.xml'}" in serial


def test_one_process_cpu_run_keeps_a_single_unsplit_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv("CAE_TEST_WORKERS", raising=False)
    monkeypatch.setattr(validate, "preflight", lambda *_args: None)
    seen: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(validate.subprocess, "run", fake_run)
    assert validate.main(["cpu", "--threads", "2", "--report-dir", str(tmp_path)]) == 0
    (command,) = seen
    assert command[command.index("-m", 3) + 1] == "not slow"


def test_known_files_keep_their_phase_and_the_marker_reaches_their_tests(request: pytest.FixtureRequest) -> None:
    """An empty serial phase passes in a shard, so the marking must not rot unseen."""
    from tests.serial_phase import runs_in_serial_phase

    tests = validate.ROOT / "tests"
    for name in ("test_uci_degrade_gaps_subprocess.py", "test_gpu_gate_take_effect.py",
                 "test_validation_runner.py", "test_selfplay_label_overlap.py"):
        assert runs_in_serial_phase(tests / name), name
    for name in ("test_simple_rl_collect_game_resume.py", "test_param_count.py"):
        assert not runs_in_serial_phase(tests / name), name
    # This file starts processes; tests/conftest.py must have marked this test.
    assert request.node.get_closest_marker("serial") is not None


def test_serial_phase_rule_and_explicit_list(tmp_path: Path) -> None:
    from tests.serial_phase import SERIAL_TEST_FILES, runs_in_serial_phase

    missing = sorted(name for name in SERIAL_TEST_FILES if not (validate.ROOT / "tests" / name).is_file())
    assert not missing
    cases = {
        "test_a.py": ("import subprocess\n", True),
        "test_b.py": ("import signal\nsignal.alarm(1)\n", True),
        "test_c.py": ("import threading\nassert threading.enumerate()\n", True),
        "test_d.py": ("import threading\nthreading.Event()\n", False),
        "test_e.py": ("x = 1\n", False),
    }
    for name, (source, expected) in cases.items():
        path = tmp_path / name
        path.write_text(source)
        assert runs_in_serial_phase(path) is expected, name
    listed = tmp_path / sorted(SERIAL_TEST_FILES)[0]
    listed.write_text("x = 1\n")
    assert runs_in_serial_phase(listed)


@pytest.mark.parametrize("argv", [["--threads", "auto"], ["--threads", "2"]])
def test_a_failed_preflight_leaves_no_earlier_serial_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str],
) -> None:
    monkeypatch.delenv("CAE_TEST_WORKERS", raising=False)
    stale = tmp_path / "cpu-serial.xml"
    stale.write_text("earlier PASS")

    def failing_preflight(*_args: object) -> None:
        raise RuntimeError("no cpu wheel")

    monkeypatch.setattr(validate, "preflight", failing_preflight)
    assert validate.main(["cpu", *argv, "--report-dir", str(tmp_path)]) == 1
    assert not stale.exists()


def test_join_race_suite_runs_the_join_tests_and_refuses_an_unexposed_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = validate.suite_command("join-race")
    assert "tests/test_thread_join.py" in command
    assert "tests/test_thread_join_sites.py" in command
    assert validate.parallel_args("join-race", "auto") == []
    monkeypatch.setattr(validate, "python_version", lambda: (3, 13))
    with pytest.raises(RuntimeError, match=r"CPython < 3\.13"):
        validate.preflight("join-race", "magic", False)
