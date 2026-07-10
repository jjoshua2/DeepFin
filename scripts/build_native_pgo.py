#!/usr/bin/env python3
"""Build native extensions with LTO and a fresh representative GCC profile."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _build(env: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, default=ROOT / "build" / "pgo")
    parser.add_argument("--training-iterations", type=int, default=200)
    args = parser.parse_args()
    if args.training_iterations <= 0:
        raise SystemExit("--training-iterations must be positive")

    profile_dir = args.profile_dir.resolve()
    build_root = (ROOT / "build").resolve()
    if profile_dir != build_root and build_root not in profile_dir.parents:
        raise SystemExit("--profile-dir must be inside the repository build directory")
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    profile_dir.mkdir(parents=True)

    env = os.environ.copy()
    env.update({
        "CAE_EXT_NATIVE": "1",
        "CAE_EXT_LTO": "1",
        "PYTHONPATH": str(ROOT),
        "OMP_NUM_THREADS": "1",
    })
    env.pop("CAE_EXT_SANITIZE", None)
    env.pop("LDFLAGS", None)
    env["CFLAGS"] = f"-fprofile-generate={profile_dir} -fprofile-update=atomic"
    _build(env)
    subprocess.run(
        [
            sys.executable, "scripts/train_native_pgo.py",
            "--iterations", str(args.training_iterations),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

    env["CFLAGS"] = f"-fprofile-use={profile_dir} -fprofile-correction"
    env["CAE_EXT_WERROR"] = "1"
    _build(env)


if __name__ == "__main__":
    main()
