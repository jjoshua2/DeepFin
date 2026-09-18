from __future__ import annotations

import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "native" / "bend_engine" / "aoti_probe"
BUILD_SCRIPT = PROBE_DIR / "build_probe.sh"
PACKAGE_SCRIPT = PROBE_DIR / "build_test_package.py"


def _first_executable(candidates: list[str | None]) -> str | None:
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        found = shutil.which(raw)
        if found:
            return found
    return None


def _bend_bin() -> str | None:
    return _first_executable(
        [
            os.environ.get("BEND_BIN"),
            "bend",
            str(ROOT / "build" / "bend_toolchain" / "bin" / "bend"),
            str(Path.home() / ".bend" / "bin" / "bend"),
        ]
    )


def _cmake_bin() -> str | None:
    return _first_executable([os.environ.get("CMAKE"), "cmake"])


def _cc_bin() -> str | None:
    return _first_executable(
        [os.environ.get("BEND_AOTI_CC"), os.environ.get("CC"), "clang", "cc"]
    )


def _cxx_bin() -> str | None:
    return _first_executable(
        [os.environ.get("BEND_AOTI_CXX"), "clang++", "c++"]
    )


def _require_bend_aoti_probe() -> bool:
    raw = os.environ.get("CAE_REQUIRE_BEND_AOTI_PROBE", "").strip().lower()
    return raw not in {"", "0", "false", "no", "n", "off"}


def _toolchain_available() -> bool:
    return (
        _bend_bin() is not None
        and _cmake_bin() is not None
        and _cc_bin() is not None
        and _cxx_bin() is not None
    )


_PROBE_TOOLCHAIN_REASON = "Bend/CMake/clang native toolchain is not installed"

pytestmark = pytest.mark.skipif(
    (not _require_bend_aoti_probe()) and (not _toolchain_available()),
    reason=_PROBE_TOOLCHAIN_REASON,
)


def _run_checked(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"command failed with exit {result.returncode}: {' '.join(args)}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result


def _float_bits(value: np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _checksum(words: list[int]) -> int:
    acc = 0
    for word in words:
        acc = ((acc * 2654435761) & 0xFFFFFFFF) ^ (word & 0xFFFFFFFF)
    return acc


def _expected(seed: int) -> dict[str, int]:
    x = np.arange(seed * 8 - 7, seed * 8 + 1, dtype=np.float32)
    y = x * np.float32(2.0) + np.float32(1.0)
    words = [_float_bits(v) for v in y]
    return {"count": len(words), "checksum": _checksum(words)}


def _parse(stdout: str) -> dict[int, dict[str, int]]:
    rows: dict[int, dict[str, int]] = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line.startswith("seed="):
            continue
        fields = dict(part.split("=", 1) for part in line.split())
        seed = int(fields.pop("seed"))
        rows[seed] = {key: int(value) for key, value in fields.items()}
    return rows


def _needed_libraries(binary: Path) -> list[str]:
    result = _run_checked(["readelf", "-d", str(binary)])
    names: list[str] = []
    for line in result.stdout.splitlines():
        if "(NEEDED)" not in line:
            continue
        start = line.find("[")
        end = line.rfind("]")
        if start >= 0 and end > start:
            names.append(line[start + 1 : end])
    return names


def test_bend_can_run_native_aoti_package(tmp_path: Path) -> None:
    bend = _bend_bin()
    cmake = _cmake_bin()
    cc = _cc_bin()
    cxx = _cxx_bin()
    assert bend is not None, _PROBE_TOOLCHAIN_REASON
    assert cmake is not None, "cmake is required to link the Bend AOTI probe"
    assert cc is not None, "clang/cc is required to compile Bend-emitted C"
    assert cxx is not None, "clang++ is required to compile the AOTI C++ bridge"

    package = tmp_path / "probe_model.pt2"
    package_env = os.environ.copy()
    package_env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor_cache"))
    package_env.setdefault("CUDA_VISIBLE_DEVICES", "")
    usr_gxx = Path("/usr/bin/g++")
    if usr_gxx.is_file() and os.access(usr_gxx, os.X_OK):
        package_env["BEND_AOTI_PACKAGE_CXX"] = str(usr_gxx)

    _run_checked(
        [sys.executable, str(PACKAGE_SCRIPT), "--out", str(package)],
        env=package_env,
    )
    assert package.is_file()

    build_env = os.environ.copy()
    build_env["BEND_AOTI_PROBE_BUILD_DIR"] = str(tmp_path / "native_build")
    build_env["BEND_BIN"] = bend
    build_env["CMAKE"] = cmake
    build_env["BEND_AOTI_CC"] = cc
    build_env["BEND_AOTI_CXX"] = cxx
    build_env["PYTHON"] = sys.executable
    built = _run_checked([str(BUILD_SCRIPT)], env=build_env)
    binary = Path(built.stdout.strip().splitlines()[-1])
    assert binary.is_file(), built.stdout

    needed = _needed_libraries(binary)
    python_libs = [name for name in needed if "python" in name.lower()]
    assert python_libs == [], (
        "probe binary linked a Python runtime, which this architecture gate "
        f"forbids: {python_libs}\nNEEDED={needed}"
    )

    run_env = os.environ.copy()
    run_env["DEEPFIN_AOTI_PROBE_PACKAGE"] = str(package)
    run_env.setdefault("CUDA_VISIBLE_DEVICES", "")
    run = _run_checked([str(binary)], env=run_env)
    observed = _parse(run.stdout)
    detail = f"--- stdout ---\n{run.stdout}\n--- stderr ---\n{run.stderr}"

    assert set(observed) == {0, 1, 7}, detail
    for seed in observed:
        assert observed[seed] == _expected(seed), detail
    assert "AOTI probe failed" not in run.stderr, detail
