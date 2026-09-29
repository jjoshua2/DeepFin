"""Opt-in single-thread runtime-call counters; never a timing or live-memory profile."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from native.bend_engine.legal_probe.benchmark import (
    BUILD_MODES, CASES, build_benchmarks, digest, parse_sample,
)
from native.bend_engine.legal_probe.run_probe import (
    HERE, ROOT, check_compiler, command, fen_position, request,
)

COUNTERS = {
    "heap_alloc": "INLINE Loc heap_alloc(Env e, Cls cls) {",
    "rfc_wrap": "OUTLINE Term rfc_wrap(Env e, Term t, u32 cnt) {",
    "rfc_bump": "INLINE void rfc_bump(Env e, Loc r, u32 k) {",
    "term_drop": "FAR void term_drop(Env e, Term t) {",
}
ROW = re.compile("profile " + " ".join(rf"{key}=([0-9]+)" for key in COUNTERS))


def replace_once(text: str, needle: str, replacement: str) -> str:
    if text.count(needle) != 1:
        raise ValueError(f"profile anchor missing or ambiguous: {needle}")
    return text.replace(needle, replacement)


def instrument(text: str) -> str:
    """Edit only disposable generated C; fail closed if the pinned ABI changes."""
    decls = "\n#if !DEVICE\nstatic int profile_active;\n"
    decls += "".join(f"static uint64_t profile_{key};\n" for key in COUNTERS)
    decls += "#endif\n"
    first = next(iter(COUNTERS.values()))
    text = replace_once(text, first, decls + first)
    for key, anchor in COUNTERS.items():
        body = f"\n#if !DEVICE\n  profile_{key} += (uint64_t)profile_active;\n#endif"
        text = replace_once(text, anchor, anchor + body)
    start = "    perft_clock_start();"
    reset = "\n".join(f"    profile_{key} = 0;" for key in COUNTERS)
    text = replace_once(text, start, start + "\n" + reset + "\n    profile_active = 1;")
    stop = '    perft_clock_finish("bend-pext",'
    text = replace_once(text, stop, "    profile_active = 0;\n" + stop)
    fmt = " ".join(f'{key}=%" PRIu64 "' for key in COUNTERS)
    args = ", ".join(f"profile_{key}" for key in COUNTERS)
    emit = f'    fprintf(stderr, "profile {fmt}\\n", {args});\n'
    anchor = "    /* Keep the last table owner alive until AFTER the measurement. */"
    return replace_once(text, anchor, emit + anchor)


def parse_counters(text: str) -> dict[str, int]:
    match = ROW.fullmatch(text.strip())
    if match is None:
        raise ValueError("missing or malformed profile counters")
    values = dict(zip(COUNTERS, map(int, match.groups()), strict=True))
    if any(v >= 1 << 64 for v in values.values()):
        raise ValueError("overflowed profile counters")
    if values["rfc_wrap"] > values["heap_alloc"]:
        raise ValueError("wrapper count exceeds heap requests")
    return values


def profile(source: Path, bun: str, cc: str, mode: str, cases: list[str],
            baseline: Path | None) -> dict[str, object]:
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("distinct profile cases required")
    pin = check_compiler(source)
    report: dict[str, object] = {
        "schema": 1, "compiler": pin, "clang": command([cc, "--version"]).splitlines()[0],
        "build_mode": mode, "threads": 1,
        "scope": "measured traversal only, excluding warmup/setup/printing/table destruction; "
                 "runtime function calls, not malloc calls, live bytes or time attribution",
        "timing": "instrumented timing is intentionally omitted; use benchmark.py uninstrumented",
        "core_sha256": digest(HERE / "Chess.bend"),
        "baseline_core_sha256": digest(baseline) if baseline else None,
    }
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="deepfin-bend-alloc-profile-") as tmp:
        work = Path(tmp)
        binaries, commands = build_benchmarks(source, work, bun, cc, BUILD_MODES[mode], baseline)
        generated_hashes: dict[str, str] = {}
        for name, binary in binaries.items():
            if name == "cboard":
                continue
            generated = binary.parent / "bench.c"
            generated_hashes[name] = digest(generated)
            profiled = binary.parent / "profile.c"
            profiled.write_text(instrument(generated.read_text()))
            profiled_binary = binary.parent / "profile"
            compile_args = [cc, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O3",
                            *BUILD_MODES[mode], "-I", str(binary.parent / "legal_probe"),
                            str(profiled), str(work / "support.o"), "-pthread", "-lm",
                            "-o", str(profiled_binary)]
            command(compile_args)
            commands.append(compile_args)
            for case in cases:
                fen, depth, expected = CASES[case]
                result = subprocess.run([str(profiled_binary), "--threads", "1"],
                    cwd=ROOT, input=request(fen_position(fen), depth), text=True,
                    capture_output=True, timeout=120, check=False)
                if result.returncode:
                    raise RuntimeError(f"profile failed: {result.stdout}\n{result.stderr}")
                parse_sample(result.stdout, depth, expected, bend=True)
                results.append({"engine": name, "case": case, "depth": depth,
                                "nodes": expected, "counters": parse_counters(result.stderr)})
        report["generated_sha256"] = generated_hashes
        report["build_commands"] = commands
    report["results"] = results
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-root", type=Path, default=ROOT / "build/bend_u64_toolchain/source")
    parser.add_argument("--bun", default=os.environ.get("BUN", "bun"))
    parser.add_argument("--cc", default=os.environ.get("CC", "clang"))
    parser.add_argument("--mode", choices=BUILD_MODES, default="native")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--baseline-chess", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        for executable in (args.bun, args.cc):
            if shutil.which(executable) is None:
                raise ValueError(f"executable not found: {executable}")
        result = profile(args.compiler_root.resolve(), args.bun, args.cc, args.mode,
                         args.cases, args.baseline_chess)
        text = json.dumps(result, indent=2) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text)
        print(text, end="")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Bend allocation profile failed: {exc}\n")


if __name__ == "__main__":
    main()
