"""The shared gate must execute its advertised scope and propagate failures."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import pytest
import torch

from scripts import validate


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
    for suite in ("cpu", "pext", "capped", "lint"):
        assert any(f"scripts/validate.py {suite} --require-cpu-wheel" in command for command in commands)
    assert not any("python -m pytest" in command or "ruff check" in command for command in commands)
    canary: dict[str, Any] = yaml.safe_load((validate.ROOT / ".github/workflows/lint-canary.yml").read_text())
    assert any("scripts/validate.py lint --require-cpu-wheel" in step.get("run", "")
               for step in canary["jobs"]["canary"]["steps"])
