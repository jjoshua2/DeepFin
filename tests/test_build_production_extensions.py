from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.build_production_extensions import build_environment, select_compiler


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
    # ⚑ --build-id=sha1 is load-bearing, not decoration: gcc15 builds without
    # it ship no GNU build-id and the provenance gate refuses the .so
    # (a41cd9d18). An expectation without the flag re-opens that hole.
    assert env["LDSHARED"] == "/opt/gcc15/bin/gcc -shared -Wl,--build-id=sha1"
    assert env["CAE_EXT_NATIVE"] == "1"
    assert env["CAE_EXT_LTO"] == "1"
    assert "CAE_EXT_SANITIZE" not in env
    assert "CFLAGS" not in env
    assert "LDFLAGS" not in env
