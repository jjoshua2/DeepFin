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


def _toolchain_available() -> bool:
    return (
        shutil.which("bend") is not None
        and shutil.which("cmake") is not None
        and shutil.which("clang") is not None
        and shutil.which("clang++") is not None
    )


_REQUIRE = os.environ.get("CAE_REQUIRE_BEND_AOTI_PROBE") == "1"
pytestmark = pytest.mark.skipif(
    not _toolchain_available() and not _REQUIRE,
    reason="Bend/CMake/clang native toolchain is not installed",
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


def test_bend_can_run_native_aoti_package(tmp_path: Path) -> None:
    package = tmp_path / "probe_model.pt2"
    package_env = os.environ.copy()
    package_env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor_cache"))

    _run_checked(
        [sys.executable, str(PACKAGE_SCRIPT), "--out", str(package)],
        env=package_env,
    )
    assert package.is_file()

    build_env = os.environ.copy()
    build_env["BEND_AOTI_PROBE_BUILD_DIR"] = str(tmp_path / "native_build")
    built = _run_checked([str(BUILD_SCRIPT)], env=build_env)
    binary = Path(built.stdout.strip().splitlines()[-1])
    assert binary.is_file(), built.stdout

    run_env = os.environ.copy()
    run_env["DEEPFIN_AOTI_PROBE_PACKAGE"] = str(package)
    run = _run_checked([str(binary)], env=run_env)
    observed = _parse(run.stdout)

    assert set(observed) == {0, 1, 7}, run.stdout
    for seed in observed:
        assert observed[seed] == _expected(seed)
