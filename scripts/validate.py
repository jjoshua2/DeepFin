#!/usr/bin/env python3
"""Run the same named validation suites locally and in CI.

Examples: python scripts/validate.py cpu; python scripts/validate.py lint
PEXT requires a separately built BMI2 extension set; this command never rebuilds.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PEXT_TESTS = (
    "tests/test_slider_tables.py",
    "tests/test_perft.py",
    "tests/test_cboard_move_parity.py",
    "tests/test_cboard_terminal.py",
    "tests/test_check_resolver.py",
    "tests/test_fastq_see.py",
    "tests/test_fastq_search.py",
    "tests/test_qsearch_dag_parity.py",
    "tests/test_nnue_position_dag.py",
    "tests/test_fuzz_smoke.py",
    "tests/test_c_extension_freshness.py",
)
CAPPED_TESTS = (
    "tests/test_gen_random_selfplay_shards.py::test_generated_shard_loads_through_the_real_replay_buffer",
    "tests/test_gen_nnue_selfplay_shards.py::test_native_shard_loads_through_the_real_replay_buffer",
    "tests/test_pytest_thread_cap.py",
)
THREAD_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def suite_command(suite: str) -> list[str]:
    if suite == "lint":
        return ["bash", "scripts/lint.sh"]
    paths = PEXT_TESTS if suite == "pext" else CAPPED_TESTS if suite == "capped" else ()
    return [sys.executable, "-m", "pytest", "-m", "not slow", "--tb=short", *paths]


def validation_environment(suite: str, threads: str, backend: str) -> dict[str, str]:
    env = dict(os.environ)
    # lint.sh resolves command-line tools from PATH when there is no local venv.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CAE_TEST_THREADS"] = "2" if suite == "capped" else threads
    env["CAE_EXPECT_SLIDER_BACKEND"] = backend
    if env["CAE_TEST_THREADS"] != "auto":
        for name in THREAD_VARS:
            env[name] = env["CAE_TEST_THREADS"]
    return env


def preflight(suite: str, backend: str, require_cpu_wheel: bool) -> None:
    checkout_venv = ROOT / ".venv"
    if suite == "lint" and checkout_venv.is_dir() and Path(sys.prefix).resolve() != checkout_venv.resolve():
        # lint.sh intentionally prefers this checkout's .venv/bin tools over
        # PATH. Venv executables can share a base-python symlink, so compare
        # environment prefixes rather than executable targets.
        raise RuntimeError(
            f"lint environment mismatch: invoking environment is {sys.prefix}, "
            f"but scripts/lint.sh selects tools from {checkout_venv}. "
            f"Run: {checkout_venv / 'bin/python'} scripts/validate.py lint"
        )
    import torch

    print(f"torch={torch.__version__} cuda_build={torch.version.cuda}", flush=True)
    if require_cpu_wheel and torch.version.cuda is not None:
        raise RuntimeError("CPU wheel required; install the locked CPU environment before validating")
    if suite == "lint":
        return
    # Running by filename puts scripts/, not the checkout root, on sys.path.
    sys.path.insert(0, str(ROOT))
    from scripts.check_c_extensions_fresh import check_extensions

    issues = check_extensions(ROOT)
    if issues:
        raise RuntimeError(
            "C extension freshness check failed:\n  " + "\n  ".join(issues)
            + "\nRebuild in this checkout with this interpreter and the intended slider recipe; "
            "see docs/development.md."
        )
    for name in ("encoding._lc0_ext", "mcts._mcts_tree", "nnue._nnue_ext"):
        module = importlib.import_module(f"chess_anti_engine.{name}")
        actual = module.SLIDER_BACKEND
        source = module.__file__
        print(f"{name}: backend={actual} file={source}", flush=True)
        if source is None or not Path(source).resolve().is_relative_to(ROOT):
            raise RuntimeError(f"{name} imported from another checkout; reinstall this checkout")
        if actual != backend:
            raise RuntimeError(f"{name}: expected {backend}, found {actual}; rebuild the intended slider arm")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("cpu", "pext", "capped", "lint"))
    parser.add_argument("--threads", default=os.environ.get("CAE_TEST_THREADS", "2"),
                        help="Positive count or auto; defaults to CAE_TEST_THREADS or 2. capped always uses 2.")
    parser.add_argument("--expect-backend", choices=("magic", "pext"),
                        default=os.environ.get("CAE_EXPECT_SLIDER_BACKEND"),
                        help="Defaults to magic, or pext for the pext suite.")
    parser.add_argument("--require-cpu-wheel", action="store_true")
    args = parser.parse_args(argv)
    if args.threads != "auto" and (not args.threads.isdecimal() or int(args.threads) < 1):
        parser.error("--threads must be a positive integer or auto")
    backend = args.expect_backend or ("pext" if args.suite == "pext" else "magic")
    if args.suite == "pext" and backend != "pext":
        parser.error("the pext suite requires --expect-backend pext")
    env = validation_environment(args.suite, args.threads, backend)
    command = suite_command(args.suite)
    started = time.monotonic()
    print(f"suite={args.suite} root={ROOT}\npython={sys.executable} {platform.python_version()} "
          f"platform={platform.platform()}\nthreads={env['CAE_TEST_THREADS']} "
          f"expected_backend={backend} CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']!r}", flush=True)
    print(f"command: {shlex.join(command)}", flush=True)
    try:
        preflight(args.suite, backend, args.require_cpu_wheel)
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        code = result.returncode
    except (ImportError, OSError, RuntimeError) as exc:
        print(f"validation preflight/launch failed: {exc}", file=sys.stderr, flush=True)
        code = 1
    elapsed = time.monotonic() - started
    print(f"{args.suite}: {'PASS' if code == 0 else 'FAIL'} in {elapsed:.2f}s (exit {code})", flush=True)
    if code:
        print("Fix the reported failure and rerun the same validation command; "
              "the selected suite did not pass.", file=sys.stderr)
    return code if code >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
