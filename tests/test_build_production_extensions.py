from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import build_production_extensions as builder
from scripts.build_production_extensions import build_environment, select_compiler


REPO_ROOT = Path(__file__).resolve().parents[1]


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_select_compiler_prefers_explicit_gcc15(tmp_path: Path) -> None:
    explicit = _executable(tmp_path / "toolchain/gcc")
    home = tmp_path / "home"
    _executable(home / ".local/gcc-15.3/bin/gcc")

    selected = select_compiler(
        env={"CAE_GCC15_CC": str(explicit), "CC": "/not/executable"},
        home=home,
    )

    assert selected == str(explicit.resolve())


def test_select_compiler_uses_validated_home_install(tmp_path: Path) -> None:
    compiler = _executable(tmp_path / "home/.local/gcc-15.3/bin/gcc")
    assert select_compiler(env={"PATH": ""}, home=tmp_path / "home") == str(compiler.resolve())


def test_select_compiler_does_not_silently_fall_back_to_system_gcc(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="validated GCC 15 compiler not found"):
        select_compiler(
            env={"PATH": os.defpath, "CC": "/usr/bin/gcc"},
            home=tmp_path / "empty-home",
        )


def test_build_environment_forces_native_lto() -> None:
    env = build_environment(
        compiler="/opt/gcc15/bin/gcc",
        base={
            "PATH": os.defpath,
            "CAE_EXT_NATIVE": "0",
            "CAE_EXT_LTO": "0",
            "CAE_EXT_SANITIZE": "address",
            "CFLAGS": "-fprofile-generate",
            "LDFLAGS": "-fprofile-generate",
            "LDSHARED": "/tmp/instrumented-linker",
        },
    )
    assert env["CC"] == "/opt/gcc15/bin/gcc"
    assert env["LDSHARED"] == "/opt/gcc15/bin/gcc -shared"
    assert env["CAE_EXT_NATIVE"] == "1"
    assert env["CAE_EXT_LTO"] == "1"
    assert "CAE_EXT_SANITIZE" not in env
    assert "CFLAGS" not in env
    assert "LDFLAGS" not in env


def test_build_environment_embeds_revision_bound_input_digests() -> None:
    digests = {
        "chess_anti_engine.encoding._features_ext": "b" * 64,
        "chess_anti_engine.encoding._lc0_ext": "c" * 64,
        "chess_anti_engine.mcts._mcts_tree": "d" * 64,
    }
    env = build_environment(
        compiler="/opt/gcc15/bin/gcc",
        base={
            "CAE_EXT_BUILD_GIT_SHA": "stale",
            "CAE_EXT_BUILD_INPUT_DIGESTS": "stale",
        },
        source_git_sha="a" * 40,
        input_digests=digests,
    )

    assert env["CAE_EXT_BUILD_GIT_SHA"] == "a" * 40
    assert json.loads(env["CAE_EXT_BUILD_INPUT_DIGESTS"]) == digests


def test_build_environment_never_inherits_an_unpaired_stale_attestation() -> None:
    env = build_environment(
        compiler="/opt/gcc15/bin/gcc",
        base={
            "CAE_EXT_BUILD_GIT_SHA": "a" * 40,
            "CAE_EXT_BUILD_INPUT_DIGESTS": "{}",
        },
    )

    assert "CAE_EXT_BUILD_GIT_SHA" not in env
    assert "CAE_EXT_BUILD_INPUT_DIGESTS" not in env


def test_build_script_direct_invocation_needs_no_pythonpath() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "scripts/build_production_extensions.py", "--help"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Build host-optimized" in completed.stdout


def test_main_rejects_dependency_changed_by_build_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    script = repo_root / "scripts/build_production_extensions.py"
    script.parent.mkdir(parents=True)
    script.write_text("# synthetic entrypoint\n")
    dependency_bytes: dict[str, bytes] = {}
    for module in builder.NATIVE_BUILD_ATTESTED_MODULES:
        for relative_path in builder.extension_spec(module).dependencies:
            source = dependency_bytes.setdefault(
                relative_path, f"committed:{relative_path}\n".encode(),
            )
            path = repo_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source)
    source_sha = "a" * 40
    build_ran = False
    readback_ran = False

    monkeypatch.setattr(builder, "__file__", str(script))
    monkeypatch.setattr(builder, "select_compiler", lambda **_kwargs: "/gcc15")
    monkeypatch.setattr(builder, "clean_source_git_sha", lambda _root: source_sha)
    monkeypatch.setattr(
        builder,
        "_git_file_at_commit",
        lambda _root, _sha, relative_path: dependency_bytes[relative_path],
    )
    monkeypatch.setattr(sys, "argv", [str(script)])

    def fake_run(command: list[str], **_kwargs: object) -> None:
        nonlocal build_ran, readback_ran
        if "setup.py" in command:
            build_ran = True
            changed = repo_root / builder.extension_spec(
                builder.NATIVE_BUILD_ATTESTED_MODULES[0]
            ).dependencies[0]
            original = changed.read_bytes()
            original_stat = changed.stat()
            changed.write_bytes(original + b"changed during build\n")
            changed.write_bytes(original)
            os.utime(
                changed,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
        else:
            readback_ran = True

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="input identities changed"):
        builder.main()

    assert build_ran is True
    assert readback_ran is False
