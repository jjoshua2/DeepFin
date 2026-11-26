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
import json
import os
from pathlib import Path
import platform
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
# Path of a file listing, one per line, the test files tests/conftest.py deselects.
DESELECT_LIST_ENV = "CAE_DESELECT_LIST"
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
# The OS-thread join tests (chess_anti_engine/utils/thread_join.py). Meaningful
# only on CPython < 3.13, where Thread.join can return before the OS thread has
# run its thread-local destructors; the join-race CI leg runs them on 3.10 with
# the engine's minimal install plus the client extras the worker sites import
# (.github/join-race-py310-requirements.txt).
JOIN_RACE_TESTS = (
    "tests/test_thread_join.py",
    "tests/test_thread_join_sites.py",
    "tests/test_uci_degrade_gaps.py::test_join_waits_until_the_os_thread_ran_its_tls_destructors",
)
# What a pull request that changes only docs/experiments/*.md can break: the
# tests that read those records. CI runs this suite instead of capped + cpu for
# such a PR (.github/workflows/ci.yml, "Classify the change").
# tests/test_validation_runner.py fails when a test file mentions
# docs/experiments and is missing here.
RECORD_TESTS = (
    "tests/test_core_d2_short_horizon_metadata.py",
    "tests/test_core_k0_saved_endpoint_recovery.py",
    "tests/test_no_absolute_home_paths.py",
    "tests/test_validation_runner.py",
    "tests/test_verify_evidence_stamps.py",
)
# Per-file test seconds measured in CI, by phase; scripts/ci_file_durations.py
# regenerates it from the junit reports. It only balances the shards: a file it
# does not list is still assigned (at the median), so a stale file cannot drop a test.
SHARD_DURATIONS = "tests/ci_file_durations.json"
# One xdist second costs this much wall time next to one serial-phase second.
XDIST_WEIGHT = 0.25
THREAD_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def suite_command(suite: str) -> list[str]:
    if suite == "lint":
        return ["bash", "scripts/lint.sh"]
    paths = {"pext": PEXT_TESTS, "capped": CAPPED_TESTS, "join-race": JOIN_RACE_TESTS,
             "records": RECORD_TESTS}.get(suite, ())
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


def parse_shard(text: str) -> tuple[int, int]:
    """``K/N`` as ``(K, N)`` with ``1 <= K <= N``; anything else is a ValueError."""
    match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", text)
    if match is None or int(match[1]) > int(match[2]):
        raise ValueError(f"--shard must be K/N with 1 <= K <= N, got {text!r}")
    return int(match[1]), int(match[2])


def shard_universe(root: Path) -> list[str]:
    """Every file pytest collects under ``tests/`` (its default ``python_files``)."""
    tests = root / "tests"
    found = {*tests.rglob("test_*.py"), *tests.rglob("*_test.py")}
    return sorted(path.relative_to(root).as_posix() for path in found if path.is_file())


def shard_cost(serial: float, xdist: float, xdist_max: float) -> float:
    """Predicted wall seconds of one shard's tests.

    The serial phase is one process, so its seconds add up. The xdist phase
    spreads files over workers but ``--dist loadfile`` keeps a file on one
    worker, so it cannot finish before its longest file does.
    """
    return serial + max(xdist_max, xdist * XDIST_WEIGHT)


def assign_shards(files: Iterable[str], durations: Mapping[str, Mapping[str, float]], count: int) -> list[list[str]]:
    """Partition ``files`` into ``count`` shards of balanced predicted wall time.

    Deterministic greedy packing: longest file first, each into the shard whose
    cost after taking it is lowest (ties to the lowest shard). The partition is
    over ``files`` — the files on disk — never over the timings: a file with no
    timing gets the median as serial-phase seconds, the phase that is not
    spread over workers.
    """
    serial, xdist = durations.get("serial", {}), durations.get("xdist", {})
    known = [float(value) for value in (*serial.values(), *xdist.values())]
    median = statistics.median(known) if known else 1.0
    timed: list[tuple[float, float, str]] = []
    for name in sorted(set(files)):
        in_serial, in_xdist = float(serial.get(name, 0.0)), float(xdist.get(name, 0.0))
        if name not in serial and name not in xdist:
            in_serial = median
        timed.append((in_serial, in_xdist, name))
    timed.sort(key=lambda item: (-(item[0] + item[1]), item[2]))
    shards: list[list[str]] = [[] for _ in range(count)]
    loads = [(0.0, 0.0, 0.0)] * count
    for in_serial, in_xdist, name in timed:
        after = [(load[0] + in_serial, load[1] + in_xdist, max(load[2], in_xdist)) for load in loads]
        best = min(range(count), key=lambda index: (shard_cost(*after[index]), index))
        loads[best] = after[best]
        shards[best].append(name)
    return [sorted(shard) for shard in shards]


def shard_files(root: Path, index: int, count: int) -> tuple[list[str], set[str]]:
    """The test files of shard ``index`` (1-based) of ``count``, and its timed phases.

    The second value names the phases (``xdist``, ``serial``) in which the
    timings place at least one of the shard's files: such a phase must run tests.
    """
    try:
        durations = json.loads((root / SHARD_DURATIONS).read_text(encoding="utf-8"))
        files = assign_shards(shard_universe(root), durations, count)[index - 1]
        return files, {phase for phase in ("xdist", "serial") if set(files) & set(durations.get(phase, {}))}
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"cannot read {SHARD_DURATIONS}: {exc}") from exc


def shard_phase_code(code: int, phase: str, timed_phases: set[str]) -> int:
    """Exit code of one phase of a sharded run.

    pytest exits 5 when it selected no test. A shard may hold no file of one
    phase (for example only serial files), so 5 passes there; where the timings
    place a file of the shard in the phase, 5 stays a failure.
    """
    if code == 5 and phase not in timed_phases:
        print(f"{phase} phase: no test file of this shard runs in it", flush=True)
        return 0
    return code


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
_TESTS_FROM_IMPORT = re.compile(r"\bfrom\s+(?:tests|\.)\s+import\s+(\([^)]*\)|[^\n]*)")
# Any quoted name: catches dynamic imports such as import_module("scripts." + NAME).
# It only adds dependency edges, never makes a file script-only.
_QUOTED_NAME = re.compile(r"[\"']([\w-]+)[\"']")
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
    refs = {name for match in _TEST_REF.finditer(text) for name in match.groups() if name}
    for match in _TESTS_FROM_IMPORT.finditer(text):
        refs.update(re.findall(r"\w+", match.group(1)))
    return refs


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
    anything else may reach every test, and so does a change to this file,
    which must not judge its own edit. A file is affected when it, or any
    script or test file it references transitively, changed.
    """
    paths = {path.strip().replace("\\", "/") for path in changed if path.strip()}
    if not paths:
        return None, "no changed files"
    seeds: set[str] = set()
    for path in sorted(paths):
        if path == "scripts/validate.py":
            return None, f"{path} defines this rule"
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
        # A deleted top-level md still counts: a test reading it must run.
        top_md = sorted({path.name for path in root.glob("*.md")}
                        | {path for path in paths if "/" not in path and path.endswith(".md")})
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
        quoted = set(_QUOTED_NAME.findall(text))
        for ref in {f"s:{name}" for name in (script_refs | quoted) & script_names} | {
                f"t:{name}" for name in (_test_refs(text) | quoted) & test_names}:
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
    is the PR's merge commit at depth 1: its parents are fetched and it is
    diffed against its first parent, the base it was merged onto, which is
    exactly the PR's change set. Diffing against the base branch's current tip
    instead would drop any file the base later changed identically. A HEAD that
    is not a two-parent merge, any other run, or any git failure returns None
    so that everything runs.
    """
    try:
        if changed_vs:
            files = _git("diff", "--name-only", "--no-renames", f"{changed_vs}...HEAD")
            files += _git("diff", "--name-only", "--no-renames", "HEAD")
            files += _git("ls-files", "--others", "--exclude-standard")
        elif environ.get("GITHUB_ACTIONS") == "true" and environ.get("GITHUB_EVENT_NAME") == "pull_request":
            # GITHUB_ACTIONS too: a --depth fetch would make a local full clone shallow.
            (head,) = _git("rev-parse", "HEAD")
            # GITHUB_SHA is the PR's test merge. Any other two-parent HEAD (a PR
            # head that merged main in) would diff as what main brought, not the PR.
            if head != environ.get("GITHUB_SHA"):
                return None, "HEAD is not the pull request's merge commit"
            _git("fetch", "--no-tags", "--depth=2", "origin", head)
            if len(_git("rev-list", "--parents", "-n1", "HEAD")[0].split()) != 3:
                return None, "HEAD is not a two-parent merge commit"
            files = _git("diff", "--name-only", "--no-renames", "HEAD^1", "HEAD")
        else:
            return None, "not a pull request (use --changed-vs REF to opt in locally)"
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
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
    env.pop(DESELECT_LIST_ENV, None)
    if env["CAE_TEST_THREADS"] != "auto":
        for name in THREAD_VARS:
            env[name] = env["CAE_TEST_THREADS"]
    return env


def parallel_environment(env: dict[str, str]) -> dict[str, str]:
    """One compute thread per xdist worker: the workers are the parallelism.

    Uncapped workers would each start a thread per core and starve the tests
    that measure wall-clock time.
    """
    out = dict(env)
    out["CAE_TEST_THREADS"] = "1"
    for name in THREAD_VARS:
        out[name] = "1"
    return out


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


def python_version() -> tuple[int, int]:
    """The running interpreter's (major, minor); a seam for the join-race refusal test."""
    return sys.version_info[0], sys.version_info[1]


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
    if suite == "join-race" and python_version() >= (3, 13):
        # FAIL, never skip: from 3.13 Thread.join joins the OS thread itself,
        # so these tests pass with or without the wait they guard.
        raise RuntimeError(
            f"the join-race suite needs CPython < 3.13 (this is {platform.python_version()}): "
            "only there can Thread.join return before the OS thread has exited"
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


def apply_change_scope(env: dict[str, str], changed_vs: str | None, workdir: Path) -> None:
    """Compute the deselection once and hand pytest its file list; print the outcome.

    Every pytest process (each xdist worker and the serial phase) only reads
    the list, so the rule runs once per validation.
    """
    files, reason = changed_files(changed_vs, os.environ)
    deselected = None
    if files is not None:
        deselected, reason = script_test_deselection(files, ROOT)
    if files is None or deselected is None:
        print(f"change scoping: off ({reason}); running every test file", flush=True)
        return
    print(f"change scoping: on ({reason}); deselecting {len(deselected)} unaffected script-only "
          "test files", flush=True)
    if deselected:
        listing = workdir / "deselected-test-files.txt"
        listing.write_text("".join(f"{path}\n" for path in sorted(deselected)), encoding="utf-8")
        env[DESELECT_LIST_ENV] = str(listing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("cpu", "pext", "capped", "lint", "join-race", "records"))
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
    parser.add_argument("--shard", metavar="K/N",
                        help="cpu suite: run only the test files of shard K of N (both phases). The N shards "
                             "partition every test file; CI runs them as parallel jobs.")
    args = parser.parse_args(argv)
    if args.changed_vs and args.suite != "cpu":
        parser.error("--changed-vs applies only to the cpu suite")
    shard: tuple[int, int] | None = None
    timed_phases: set[str] = set()
    if args.shard is not None:
        if args.suite != "cpu":
            parser.error("--shard applies only to the cpu suite")
        try:
            shard = parse_shard(args.shard)
        except ValueError as exc:
            parser.error(str(exc))
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
                # Removed even when this run has no serial phase: a stale one
                # from an earlier run must not sit beside a fresh report.
                serial_report = report.with_name(f"{args.suite}-serial.xml")
                serial_report.unlink(missing_ok=True)
                if serial_command is not None:
                    serial_command.append(f"--junitxml={serial_report}")
                cache_root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="deepfin-validate-")))
                configure_compile_environment(env, cache_root)
                print(f"compile_cache={cache_root} CXX={env.get('CXX', 'PATH default')} "
                      "compile_threads=1 MAX_JOBS=1", flush=True)
            preflight(args.suite, backend, args.require_cpu_wheel)
            if shard is not None:
                # Explicit paths, so each shard collects only its own files. A
                # phase left with no test exits 5 and fails the shard.
                selected, timed_phases = shard_files(ROOT, *shard)
                print(f"shard {shard[0]}/{shard[1]}: {len(selected)} test files", flush=True)
                if not selected:
                    raise RuntimeError(f"shard {shard[0]}/{shard[1]} has no test files")
                for phase in (command, serial_command):
                    if phase is not None:
                        phase.extend(selected)
            if args.suite == "cpu":
                scope_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="deepfin-scope-")))
                apply_change_scope(env, args.changed_vs, scope_dir)
            print(f"executing: {shlex.join(command)}", flush=True)
            result = subprocess.run(command, cwd=ROOT, env=parallel_environment(env) if xdist else env, check=False)
            code = result.returncode
            if serial_command is not None:
                # Always run it: a failure in one phase must not hide the other's.
                print(f"executing: {shlex.join(serial_command)}", flush=True)
                serial_code = subprocess.run(serial_command, cwd=ROOT, env=env, check=False).returncode
                if shard is not None and (code, serial_code) != (5, 5):
                    code = shard_phase_code(code, "xdist", timed_phases)
                    serial_code = shard_phase_code(serial_code, "serial", timed_phases)
                code = code or serial_code
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
