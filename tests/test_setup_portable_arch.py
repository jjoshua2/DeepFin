"""The portable extension build names the x86-64 baseline instead of trusting the
compiler's default, which Ubuntu 26.04 raised to x86-64-v3 (BMI2/AVX2 on)."""
from __future__ import annotations

import ast
import platform
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_helper() -> Callable[..., object]:
    """Pull the pure helper out of setup.py; importing setup.py would run setup()."""
    tree = ast.parse((REPO_ROOT / "setup.py").read_text())
    node = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_portable_arch_args"
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "setup.py", "exec"), namespace)
    helper = namespace["_portable_arch_args"]
    assert callable(helper)
    return helper


def test_portable_x86_build_names_the_baseline() -> None:
    helper = _load_helper()
    assert helper(native=False, machine="x86_64", cflags="") == ["-march=x86-64"]
    assert helper(native=False, machine="AMD64", cflags="-mbmi2") == ["-march=x86-64"]


def test_native_other_arch_and_explicit_march_are_left_alone() -> None:
    helper = _load_helper()
    assert helper(native=True, machine="x86_64", cflags="") == []
    assert helper(native=False, machine="aarch64", cflags="") == []
    assert helper(native=False, machine="x86_64", cflags="-O2 -march=x86-64-v3") == []


def test_both_compile_arg_builders_use_the_arch_helper() -> None:
    source = (REPO_ROOT / "setup.py").read_text()
    assert source.count("args += _arch_args()") == 2
    assert source.count('"-march=native"') == 1


def _defines_bmi2(*flags: str) -> bool:
    out = subprocess.run(
        ["gcc", *flags, "-dM", "-E", "-x", "c", "/dev/null"],
        check=True, capture_output=True, text=True,
    ).stdout
    return "__BMI2__" in out


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_gcc_honours_the_baseline_and_keeps_an_explicit_feature_flag() -> None:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        pytest.skip("x86-64 flag semantics")
    # Whatever this compiler defaults to, naming the baseline turns BMI2 off...
    assert not _defines_bmi2("-march=x86-64-v3", "-march=x86-64")
    # ...and the PEXT CI leg's CFLAGS=-mbmi2, which precedes it, still wins.
    assert _defines_bmi2("-mbmi2", "-march=x86-64")
