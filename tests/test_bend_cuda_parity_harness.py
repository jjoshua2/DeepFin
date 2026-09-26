from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.moves.encode import COMPACT_TO_FULL_POLICY


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "native" / "bend_engine" / "cuda_parity" / "run_parity.py"
BUILD_SCRIPT = ROOT / "native" / "bend_engine" / "cuda_parity" / "build_probe.sh"

_spec = importlib.util.spec_from_file_location("deepfin_bend_cuda_parity", MODULE_PATH)
assert _spec is not None
assert _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)


def test_infer_bucket_from_deepfin_package_name() -> None:
    assert _mod.infer_bucket(Path("chess_b16.pt2"), None) == 16
    assert _mod.infer_bucket(Path("chess_b1190.pt2"), None) == 1190
    assert _mod.infer_bucket(Path("anything.pt2"), 7) == 7


def test_cuda_fixture_uses_real_cboard_encoding(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    _mod.build_fixture(path, batch=3, input_planes=146, hist_mode=1)
    header = _mod.read_fixture_header(path)

    assert header == {
        "version": 1,
        "batch": 3,
        "planes": 146,
        "full_policy_width": 4672,
        "map_count": 1858,
    }

    header_bytes = len(_mod.MAGIC) + 5 * 4
    expected_bytes = (
        header_bytes
        + 1858 * 4
        + 3 * 146 * 8 * 8 * 2
    )
    assert path.stat().st_size == expected_bytes

    with path.open("rb") as f:
        f.seek(header_bytes)
        mapping = np.frombuffer(f.read(1858 * 4), dtype="<u4").copy()
    assert np.array_equal(
        mapping, np.asarray(COMPACT_TO_FULL_POLICY, dtype="<u4")
    )

    board = chess.Board(_mod.FIXTURE_FENS[0])
    encoded = np.asarray(
        CBoard.from_board(board).encode_full(1, 34),
        dtype=np.float32,
    )
    want = np.asarray(_mod._bf16_bits(encoded), dtype="<u2").reshape(-1)
    payload_off = header_bytes + 1858 * 4
    got = np.fromfile(path, dtype="<u2", offset=payload_off, count=146 * 64)
    assert np.array_equal(got, want)


def test_summary_is_stable_bit_level_contract() -> None:
    values = np.asarray([0.0, 1.0, -2.0, 3.5], dtype=np.float32)
    first = _mod.summarize_f32(values)
    second = _mod.summarize_f32(values.copy())
    assert first == second
    assert first[0] == 4
    assert first[1] != 0
    assert first[2] != 0


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


def _require_cuda_parity_compile() -> bool:
    raw = os.environ.get("CAE_REQUIRE_BEND_CUDA_PARITY", "").strip().lower()
    return raw not in {"", "0", "false", "no", "n", "off"}


def _compile_toolchain() -> bool:
    return (
        _mod._bend_bin() is not None
        and _first_executable([os.environ.get("CMAKE"), "cmake"]) is not None
        and _first_executable(
            [os.environ.get("BEND_CUDA_PARITY_CC"), os.environ.get("CC"), "clang"]
        )
        is not None
        and _first_executable([os.environ.get("BEND_CUDA_PARITY_CXX"), "clang++"])
        is not None
    )


@pytest.mark.skipif(
    (not _require_cuda_parity_compile()) and (not _compile_toolchain()),
    reason="Bend/CMake/clang native toolchain is not installed",
)
def test_cuda_parity_binary_compiles_without_python_runtime(tmp_path: Path) -> None:
    bend = _mod._bend_bin()
    cmake = _first_executable([os.environ.get("CMAKE"), "cmake"])
    cc = _first_executable(
        [os.environ.get("BEND_CUDA_PARITY_CC"), os.environ.get("CC"), "clang"]
    )
    cxx = _first_executable([os.environ.get("BEND_CUDA_PARITY_CXX"), "clang++"])
    assert bend is not None
    assert cmake is not None
    assert cc is not None
    assert cxx is not None

    env = os.environ.copy()
    env["BEND_CUDA_PARITY_BUILD_DIR"] = str(tmp_path / "native_build")
    env["BEND_BIN"] = bend
    env["CMAKE"] = cmake
    env["BEND_CUDA_PARITY_CC"] = cc
    env["BEND_CUDA_PARITY_CXX"] = cxx
    env["PYTHON"] = sys.executable
    env["BEND_NO_TELEMETRY"] = "1"
    result = subprocess.run(
        [str(BUILD_SCRIPT)],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"compile failed ({result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    binary = Path(result.stdout.strip().splitlines()[-1])
    assert binary.is_file(), result.stdout

    readelf = subprocess.run(
        ["readelf", "-d", str(binary)],
        check=False,
        text=True,
        capture_output=True,
    )
    if readelf.returncode != 0:
        pytest.fail(f"readelf failed:\n{readelf.stderr}")
    needed: list[str] = []
    for line in readelf.stdout.splitlines():
        if "(NEEDED)" not in line:
            continue
        start = line.find("[")
        end = line.rfind("]")
        if start >= 0 and end > start:
            needed.append(line[start + 1 : end])
    assert needed, "readelf -d produced no NEEDED entries"
    assert any("libtorch" in name for name in needed), needed
    python_libs = [name for name in needed if "python" in name.lower()]
    assert python_libs == [], python_libs
