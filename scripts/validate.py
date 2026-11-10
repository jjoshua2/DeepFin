#!/usr/bin/env python3
"""Run the same named validation suites locally and in CI.

Examples: python scripts/validate.py cpu; python scripts/validate.py lint
PEXT requires a separately built BMI2 extension set; this command never rebuilds.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
import importlib
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
# Newline-separated paths changed by the pull request; tests/conftest.py reads it.
CHANGED_FILES_ENV = "CAE_CHANGED_FILES"
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
    return [sys.executable, "-m", "pytest", "-m", "not slow", "--tb=short", "--durations=20", *paths]


def parallel_args(suite: str, workers: str) -> list[str]:
    """pytest-xdist arguments for the full CPU suite.

    Only the ``cpu`` suite is spread across processes: ``capped`` and ``pext``
    are short and ``capped`` exists to observe one process's thread setting.
    ``loadfile`` keeps every test of a file in one worker, so module-scoped
    fixtures and file-local shared state behave as they do serially.
    """
    if suite != "cpu" or workers == "1":
        return []
    return ["-n", workers, "--dist", "loadfile"]


def split_serial_phase(command: list[str]) -> tuple[list[str], list[str]]:
    """Split a pytest command into its xdist phase and its one-process phase.

    Tests marked ``serial`` read process-wide state (live threads, the SIGALRM
    handler) or assume an unloaded host, so they run afterwards in one process.
    """
    marker = command.index("-m", 3) + 1
    parallel, serial = list(command), list(command)
    parallel[marker] = f"({command[marker]}) and not serial"
    serial[marker] = f"({command[marker]}) and serial"
    return parallel, serial


# Change-scoped selection of script-only test files. Any doubt runs everything:
# an unmappable or unsafe changed path, a git failure, or no pull request at all.
_SCRIPT_FILE = re.compile(r"scripts/([\w-]+)\.(?:py|sh)")
_TEST_FILE = re.compile(r"tests/(test_\w*)\.py")
# `scripts.X` (imports, importlib strings, `-m`), a `X.py`/`X.sh` file name in
# any spelling of its path, and the bare imports a script run by filename uses.
_SCRIPT_REF = re.compile(
    r"\bscripts\.(\w+)|([\w-]+)\.(?:py|sh)\b|^[ \t]*(?:from[ \t]+\.?|import[ \t]+)(\w+)(?![\w.])", re.MULTILINE,
)
_SCRIPTS_FROM_IMPORT = re.compile(r"\bfrom\s+scripts\s+import\s+(\([^)]*\)|[^\n]*)")
_TEST_REF = re.compile(r"\b(?:tests\.|from\s+\.)(test_\w+)|\b(test_\w+)\.py\b")
_PACKAGE_REF = re.compile(r"\bchess_anti_engine\b")
# A test that reads docs/, tests/ or scans scripts/ depends on files no
# reference names, so it is never treated as script-only.
_TREE_READ = re.compile(
    r"[\"'](?:docs|tests)[\"']|\bdocs/|\btests/(?!test_\w*\.py)"
    r"|[\"']scripts[\"']\s*\)?\s*\.\s*(?:r?glob|iterdir)\(|(?:glob|listdir|walk|scandir)\(\s*[\"'][^\"']*scripts"
)


def _script_refs(text: str) -> set[str]:
    refs = {name for match in _SCRIPT_REF.finditer(text) for name in match.groups() if name}
    for match in _SCRIPTS_FROM_IMPORT.finditer(text):
        refs.update(re.findall(r"\w+", match.group(1)))
    return refs


def _test_refs(text: str) -> set[str]:
    return {name for match in _TEST_REF.finditer(text) for name in match.groups() if name}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def script_test_deselection(
    changed: Iterable[str], root: Path, read: Callable[[Path], str] = _read_text,
) -> tuple[frozenset[str] | None, str]:
    """Script-only test files (``tests/test_*.py``) the change set cannot affect.

    Returns ``(None, reason)`` when scoping is not permitted, so everything runs.
    A script-only file references an existing ``scripts/`` module and never
    mentions ``chess_anti_engine``. Scoping is permitted only when every changed
    path is under ``scripts/`` (as a top-level ``.py``/``.sh`` module), is a
    ``tests/test_*.py`` file, is under ``docs/`` or is a top-level ``*.md``;
    anything else may reach every test. A file is affected when it, or any
    script or test file it references transitively, changed.
    """
    paths = {path.strip().replace("\\", "/") for path in changed if path.strip()}
    if not paths:
        return None, "no changed files"
    seeds: set[str] = set()
    for path in sorted(paths):
        if path.startswith("docs/") or ("/" not in path and path.endswith(".md")):
            continue
        if match := _TEST_FILE.fullmatch(path):
            seeds.add(f"t:{match[1]}")
        elif path.startswith("scripts/"):
            match = _SCRIPT_FILE.fullmatch(path)
            if match is None or match[1] == "__init__":
                return None, f"cannot map {path} to a script module"
            seeds.add(f"s:{match[1]}")
        else:
            return None, f"{path} may affect any test"
    try:
        scripts = sorted(path for path in (root / "scripts").iterdir()
                         if path.is_file() and path.suffix in (".py", ".sh") and path.stem != "__init__")
        tests = sorted((root / "tests").glob("test_*.py"))
        top_md = sorted(path.name for path in root.glob("*.md"))
        sources = {path: read(path) for path in (*scripts, *tests)}
    except OSError as exc:
        return None, f"cannot read the tree: {exc}"
    existing = {path.stem for path in scripts}
    script_names = existing | {seed[2:] for seed in seeds if seed.startswith("s:")}
    test_names = {path.stem for path in tests} | {seed[2:] for seed in seeds if seed.startswith("t:")}
    top_md_read = re.compile("|".join(rf"[\"'/]{re.escape(name)}\b" for name in top_md) or r"(?!)")
    dependents: dict[str, set[str]] = {}
    candidates: list[Path] = []
    script_paths = set(scripts)
    for path in (*scripts, *tests):
        text = sources[path]
        node = f"{'s' if path in script_paths else 't'}:{path.stem}"
        script_refs = _script_refs(text)
        for ref in {f"s:{name}" for name in script_refs & script_names} | {
                f"t:{name}" for name in _test_refs(text) & test_names}:
            dependents.setdefault(ref, set()).add(node)
        if (node.startswith("t:") and script_refs & existing and not _PACKAGE_REF.search(text)
                and not _TREE_READ.search(text) and not top_md_read.search(text)):
            candidates.append(path)
    affected, frontier = set(seeds), list(seeds)
    while frontier:
        for node in dependents.get(frontier.pop(), ()):
            if node not in affected:
                affected.add(node)
                frontier.append(node)
    deselected = frozenset(f"tests/{path.name}" for path in candidates if f"t:{path.stem}" not in affected)
    return deselected, f"{len(paths)} changed files; {len(candidates)} script-only test files"


def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=300).stdout
    return [line for line in out.splitlines() if line.strip()]


def changed_files(changed_vs: str | None, environ: Mapping[str, str]) -> tuple[list[str] | None, str]:
    """The change set to scope script-only tests by, or ``(None, reason)``.

    ``--changed-vs REF`` diffs a local branch from its merge base with REF and
    adds uncommitted and untracked files. On a GitHub pull request the checkout
    is the depth-1 merge commit, so a two-dot diff against the fetched base tip
    is the PR's change set (or a superset). Any other run, or any git failure,
    returns None so that everything runs.
    """
    try:
        if changed_vs:
            files = _git("diff", "--name-only", "--no-renames", f"{changed_vs}...HEAD")
            files += _git("diff", "--name-only", "--no-renames", "HEAD")
            files += _git("ls-files", "--others", "--exclude-standard")
        elif environ.get("GITHUB_EVENT_NAME") == "pull_request" and environ.get("GITHUB_BASE_REF"):
            _git("fetch", "--no-tags", "--depth=1", "origin", environ["GITHUB_BASE_REF"])
            files = _git("diff", "--name-only", "--no-renames", "FETCH_HEAD", "HEAD")
        else:
            return None, "not a pull request (use --changed-vs REF to opt in locally)"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"change detection failed: {exc}"
    if not files:
        return None, "no changed files detected"
    return sorted(set(files)), "ok"


def validation_environment(suite: str, threads: str, backend: str) -> dict[str, str]:
    env = dict(os.environ)
    # lint.sh resolves command-line tools from PATH when there is no local venv.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CAE_TEST_THREADS"] = "2" if suite == "capped" else threads
    env["CAE_EXPECT_SLIDER_BACKEND"] = backend
    # Only main() may scope a run; an inherited change set must not.
    env.pop(CHANGED_FILES_ENV, None)
    if env["CAE_TEST_THREADS"] != "auto":
        for name in THREAD_VARS:
            env[name] = env["CAE_TEST_THREADS"]
    return env


def configure_compile_environment(env: dict[str, str], cache_root: Path) -> None:
    """Keep test compilation away from live artifacts and compiler overrides."""
    env["DEEPFIN_COMPILE_CACHE"] = str(cache_root)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "compile_cache" / "torchinductor")
    env["TRITON_CACHE_DIR"] = str(cache_root / "compile_cache" / "triton")
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    env["MAX_JOBS"] = "1"
    # A newer compiler placed on PATH can emit libstdc++ requirements the
    # installed Torch runtime cannot load. Prefer Linux's system compiler;
    # retain an explicit CXX when validating a deliberately chosen toolchain.
    if sys.platform.startswith("linux") and os.access("/usr/bin/c++", os.X_OK):
        env.setdefault("CXX", "/usr/bin/c++")


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


def apply_change_scope(env: dict[str, str], changed_vs: str | None) -> bool:
    """Hand the change set to pytest when scoping is permitted; print the outcome."""
    files, reason = changed_files(changed_vs, os.environ)
    deselected = None
    if files is not None:
        deselected, reason = script_test_deselection(files, ROOT)
    if files is None or deselected is None:
        print(f"change scoping: off ({reason}); running every test file", flush=True)
        return False
    print(f"change scoping: on ({reason}); deselecting {len(deselected)} unaffected script-only "
          "test files", flush=True)
    if not deselected:
        return False
    env[CHANGED_FILES_ENV] = "\n".join(files)
    return True


def phase_code(code: int, scoped: bool) -> int:
    """pytest's 'no tests collected' (5) is a pass only when scoping emptied the phase."""
    if code == 5 and scoped:
        print("change scoping deselected every test in this phase; treating it as a pass", flush=True)
        return 0
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("cpu", "pext", "capped", "lint"))
    parser.add_argument("--threads", default=os.environ.get("CAE_TEST_THREADS", "2"),
                        help="Positive count or auto; defaults to CAE_TEST_THREADS or 2. capped always uses 2.")
    parser.add_argument("--workers", default=os.environ.get("CAE_TEST_WORKERS"),
                        help="pytest-xdist processes for the cpu suite: a positive count or auto. Defaults to "
                             "CAE_TEST_WORKERS, else auto when threads is auto (CI) and 1 otherwise, so a "
                             "capped local run stays in one process.")
    parser.add_argument("--expect-backend", choices=("magic", "pext"),
                        default=os.environ.get("CAE_EXPECT_SLIDER_BACKEND"),
                        help="Defaults to magic, or pext for the pext suite.")
    parser.add_argument("--require-cpu-wheel", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/validation"),
                        help="JUnit XML directory, relative to the checkout unless absolute; one file per suite.")
    parser.add_argument("--changed-vs", metavar="REF",
                        help="cpu suite: skip script-only test files that the changes since the merge base "
                             "with REF (plus uncommitted and untracked files) cannot affect, as CI does on a "
                             "pull request. See docs/development.md.")
    args = parser.parse_args(argv)
    if args.changed_vs and args.suite != "cpu":
        parser.error("--changed-vs applies only to the cpu suite")
    if args.threads != "auto" and (not args.threads.isdecimal() or int(args.threads) < 1):
        parser.error("--threads must be a positive integer or auto")
    workers = args.workers or ("auto" if args.threads == "auto" else "1")
    if workers != "auto" and (not workers.isdecimal() or int(workers) < 1):
        parser.error("--workers must be a positive integer or auto")
    backend = args.expect_backend or ("pext" if args.suite == "pext" else "magic")
    if args.suite == "pext" and backend != "pext":
        parser.error("the pext suite requires --expect-backend pext")
    env = validation_environment(args.suite, args.threads, backend)
    command = suite_command(args.suite)
    xdist = parallel_args(args.suite, workers)
    serial_command: list[str] | None = None
    if xdist:
        command, serial_command = split_serial_phase(command)
        command += xdist
    started = time.monotonic()
    print(f"suite={args.suite} root={ROOT}\npython={sys.executable} {platform.python_version()} "
          f"platform={platform.platform()}\nthreads={env['CAE_TEST_THREADS']} workers={workers} "
          f"expected_backend={backend} CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']!r}", flush=True)
    try:
        with ExitStack() as stack:
            if args.suite != "lint":
                report = ROOT / args.report_dir / f"{args.suite}.xml"
                report.parent.mkdir(parents=True, exist_ok=True)
                # A preflight failure must not leave an earlier PASS report.
                report.unlink(missing_ok=True)
                command.append(f"--junitxml={report}")
                print(f"junit_report={report}", flush=True)
                if serial_command is not None:
                    serial_report = report.with_name(f"{args.suite}-serial.xml")
                    serial_report.unlink(missing_ok=True)
                    serial_command.append(f"--junitxml={serial_report}")
                cache_root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="deepfin-validate-")))
                configure_compile_environment(env, cache_root)
                print(f"compile_cache={cache_root} CXX={env.get('CXX', 'PATH default')} "
                      "compile_threads=1 MAX_JOBS=1", flush=True)
            preflight(args.suite, backend, args.require_cpu_wheel)
            scoped = args.suite == "cpu" and apply_change_scope(env, args.changed_vs)
            print(f"executing: {shlex.join(command)}", flush=True)
            code = phase_code(subprocess.run(command, cwd=ROOT, env=env, check=False).returncode, scoped)
            if serial_command is not None:
                # Always run it: a failure in one phase must not hide the other's.
                print(f"executing: {shlex.join(serial_command)}", flush=True)
                serial_code = subprocess.run(serial_command, cwd=ROOT, env=env, check=False).returncode
                code = code or phase_code(serial_code, scoped)
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
