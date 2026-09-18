from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

import chess
import numpy as np
import torch

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.inference import _aoti_load_package, _policy_output_full
from chess_anti_engine.moves.encode import (
    COMPACT_POLICY_SIZE,
    COMPACT_TO_FULL_POLICY,
    POLICY_SIZE,
)


MAGIC = b"DFCUDA3\0"
VERSION = 1
FIXTURE_FENS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "k3r3/8/8/8/8/8/8/4K3 w - - 0 1",
    "r2q1rk1/pp2bppp/2npbn2/2p5/4P3/2NP1N2/PPQ1BPPP/R1B2RK1 w - - 4 10",
)


def infer_bucket(package: Path, explicit: int | None) -> int:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("--batch must be positive")
        return explicit
    match = re.search(r"chess_b(\d+)\.pt2$", package.name)
    if match is None:
        raise ValueError(
            f"cannot infer fixed batch size from {package.name!r}; pass --batch"
        )
    return int(match.group(1))


def _bf16_bits(array: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    return tensor.to(torch.bfloat16).view(torch.uint16).numpy(force=True)


def build_fixture(
    path: Path,
    *,
    batch: int,
    input_planes: int,
    hist_mode: int = 1,
) -> None:
    if input_planes <= 112:
        raise ValueError("input_planes must include LC0 112 plus feature planes")
    n_extra = input_planes - 112
    if n_extra not in (34, 63):
        raise ValueError(
            f"unsupported DeepFin feature-plane count {n_extra}; expected 34 or 63"
        )

    rows: list[np.ndarray] = []
    for i in range(batch):
        board = chess.Board(FIXTURE_FENS[i % len(FIXTURE_FENS)])
        cboard = CBoard.from_board(board)
        encoded = np.asarray(
            cboard.encode_full(hist_mode, n_extra),
            dtype=np.float32,
        )
        if encoded.shape != (input_planes, 8, 8):
            raise AssertionError(
                f"CBoard encoded {encoded.shape}, expected {(input_planes, 8, 8)}"
            )
        rows.append(encoded)

    inputs = _bf16_bits(np.stack(rows, axis=0))
    mapping = np.asarray(COMPACT_TO_FULL_POLICY, dtype="<u4")
    if mapping.shape != (COMPACT_POLICY_SIZE,):
        raise AssertionError(
            f"compact policy map shape {mapping.shape} != {(COMPACT_POLICY_SIZE,)}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(
            struct.pack(
                "<5I",
                VERSION,
                batch,
                input_planes,
                POLICY_SIZE,
                COMPACT_POLICY_SIZE,
            )
        )
        f.write(mapping.tobytes(order="C"))
        f.write(np.asarray(inputs, dtype="<u2").tobytes(order="C"))


def read_fixture_header(path: Path) -> dict[str, int]:
    with path.open("rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("bad fixture magic")
        raw = f.read(struct.calcsize("<5I"))
        if len(raw) != struct.calcsize("<5I"):
            raise ValueError("truncated fixture header")
        version, batch, planes, full_width, map_count = struct.unpack("<5I", raw)
    return {
        "version": version,
        "batch": batch,
        "planes": planes,
        "full_policy_width": full_width,
        "map_count": map_count,
    }


def summarize_f32(array: np.ndarray) -> tuple[int, int, int]:
    values = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
    bits = values.view(np.uint32)
    count = int(bits.size)
    xors = int(np.bitwise_xor.reduce(bits, initial=np.uint32(0)))
    indices = np.arange(1, count + 1, dtype=np.uint64)
    mix = int(
        np.sum(bits.astype(np.uint64) * indices, dtype=np.uint64)
        & np.uint64(0xFFFFFFFF)
    )
    return count, xors, mix


def python_reference(
    package: Path,
    fixture: Path,
    *,
    device_index: int,
) -> dict[str, int]:
    header = read_fixture_header(fixture)
    batch = header["batch"]
    planes = header["planes"]
    map_count = header["map_count"]
    header_bytes = len(MAGIC) + struct.calcsize("<5I") + map_count * 4

    bits = np.fromfile(fixture, dtype="<u2", offset=header_bytes)
    expected = batch * planes * 64
    if bits.size != expected:
        raise ValueError(f"fixture has {bits.size} bf16 values, expected {expected}")
    bits = bits.reshape(batch, planes, 8, 8)

    device = torch.device("cuda", device_index)
    input_cpu = torch.from_numpy(np.ascontiguousarray(bits)).view(torch.bfloat16)
    input_cuda = input_cpu.to(device=device)

    with torch.cuda.device(device_index), torch.no_grad():
        model = _aoti_load_package(str(package))
        out = model(input_cuda)
        raw_policy = out["policy"] if "policy" in out else out["policy_own"]
        dense_policy = _policy_output_full(out).detach().float().cpu().numpy()
        wdl = out["wdl"].detach().float().cpu().numpy()
        torch.cuda.synchronize(device)

    policy_count, policy_xor, policy_mix = summarize_f32(dense_policy)
    wdl_count, wdl_xor, wdl_mix = summarize_f32(wdl)
    return {
        "batch": batch,
        "raw_policy_width": int(raw_policy.shape[-1]),
        "policy_count": policy_count,
        "policy_xor": policy_xor,
        "policy_mix": policy_mix,
        "wdl_count": wdl_count,
        "wdl_xor": wdl_xor,
        "wdl_mix": wdl_mix,
    }


def _run_checked(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result


def parse_native_summary(stdout: str) -> dict[str, int]:
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line.startswith("batch="):
            continue
        fields = dict(part.split("=", 1) for part in line.split())
        return {key: int(value) for key, value in fields.items()}
    raise ValueError(f"native Bend output had no summary line:\n{stdout}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a real DeepFin CUDA .pt2 package through Python and native Bend/C++ AOTI."
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--input-planes", type=int, required=True)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--hist-mode", type=int, default=1)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build/bend_cuda_parity"),
    )
    args = parser.parse_args()

    package = args.package.resolve()
    if not package.is_file():
        raise FileNotFoundError(package)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA parity requires torch.cuda.is_available()")
    if args.device_index < 0 or args.device_index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {args.device_index} is unavailable; "
            f"device_count={torch.cuda.device_count()}"
        )

    batch = infer_bucket(package, args.batch)
    root = Path(__file__).resolve().parents[3]
    build_script = Path(__file__).with_name("build_probe.sh")

    with tempfile.TemporaryDirectory(prefix="deepfin_bend_cuda_") as tmp:
        fixture = Path(tmp) / "deepfin_cuda_fixture.bin"
        build_fixture(
            fixture,
            batch=batch,
            input_planes=args.input_planes,
            hist_mode=args.hist_mode,
        )

        expected = python_reference(
            package,
            fixture,
            device_index=args.device_index,
        )

        env = os.environ.copy()
        env["BEND_CUDA_PARITY_BUILD_DIR"] = str(args.build_dir.resolve())
        built = _run_checked([str(build_script)], env=env)
        binary = Path(built.stdout.strip().splitlines()[-1])
        if not binary.is_file():
            raise RuntimeError(f"native build did not produce {binary}")

        run_env = os.environ.copy()
        run_env["DEEPFIN_AOTI_CUDA_PACKAGE"] = str(package)
        run_env["DEEPFIN_AOTI_CUDA_FIXTURE"] = str(fixture)
        run_env["DEEPFIN_AOTI_DEVICE_INDEX"] = str(args.device_index)
        native = _run_checked([str(binary)], env=run_env)
        observed = parse_native_summary(native.stdout)

    if observed != expected:
        keys = sorted(set(expected) | set(observed))
        detail = "\n".join(
            f"  {key}: python={expected.get(key)} native={observed.get(key)}"
            for key in keys
            if expected.get(key) != observed.get(key)
        )
        raise SystemExit(f"CUDA parity FAILED:\n{detail}")

    print(
        "CUDA parity PASS: "
        f"package={package} batch={batch} planes={args.input_planes} "
        f"raw_policy_width={expected['raw_policy_width']} "
        f"policy_count={expected['policy_count']} "
        f"wdl_count={expected['wdl_count']}"
    )


if __name__ == "__main__":
    main()
