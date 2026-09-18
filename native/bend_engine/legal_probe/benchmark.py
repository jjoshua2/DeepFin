"""Opt-in, paired in-process traversal timing; never invoked by ordinary pytest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile

from native.bend_engine.legal_probe.run_probe import (
    CANONICAL, HERE, ROOT, START, check_compiler, command, fen_position, request,
)

CASES = {"startpos": (START, 5, 4865609),
         "kiwipete": (CANONICAL[1][1], 4, 4085603)}
BUILD_MODES = {"generic": [], "native": ["-march=native"]}
ROW = re.compile(r"bench (bend-pext|pext|magic) ([0-9]+) ([0-9]+) ([0-9]+)")


def parse_sample(text: str, depth: int, expected: int, *, bend: bool) -> dict[str, int | str]:
    lines = text.splitlines()
    if len(lines) != 2 or re.fullmatch(r"warmup [0-9]+ [0-9]+", lines[0]) is None:
        raise ValueError("missing/malformed warmup row")
    hi, lo = map(int, lines[0].split()[1:])
    if max(hi, lo) >= 1 << 32 or (hi << 32 | lo) != expected:
        raise ValueError("incorrect warmup count")
    match = ROW.fullmatch(lines[1])
    if match is None:
        raise ValueError("malformed timing row")
    backend, got_depth, nodes, ns = match.groups()
    if (backend == "bend-pext") != bend:
        raise ValueError("incorrect benchmark backend")
    if int(got_depth) != depth or int(nodes) != expected or not 0 < int(ns) < 120_000_000_000:
        raise ValueError("incorrect depth/count or invalid timing interval")
    return {"backend": backend, "nodes": int(nodes), "nanoseconds": int(ns)}


def summary(samples_ns: list[int], nodes: int) -> dict[str, float | int]:
    if not samples_ns or any(n <= 0 for n in samples_ns):
        raise ValueError("positive timing samples required")
    median = statistics.median(samples_ns)
    return {"samples": len(samples_ns), "median_seconds": median / 1e9,
            "min_seconds": min(samples_ns) / 1e9, "max_seconds": max(samples_ns) / 1e9,
            "median_leaf_nodes_per_second": nodes * 1e9 / median}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_benchmarks(source: Path, work: Path, bun: str, cc: str, flags: list[str],
                     baseline: Path | None) -> tuple[dict[str, Path], list[list[str]]]:
    check_compiler(source)
    commands: list[list[str]] = []

    def run(args: list[str]) -> None:
        commands.append(args)
        command(args)

    common = [cc, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O3"]
    support = work / "support.o"
    run([*common, "-I", str(ROOT), "-c", str(HERE / "support.c"), "-o", str(support)])
    binaries: dict[str, Path] = {}
    choices = {"bend": HERE / "Chess.bend"}
    if baseline is not None:
        choices["bend_baseline"] = baseline
    # Mirror just the relative modules. The old core uses the exact same clock,
    # compiler, input, tables and flags as the new core; only Chess.bend differs.
    for name, core in choices.items():
        parent = work / name
        legal = parent / "legal_probe"
        legal.mkdir(parents=True)
        sliders = parent / "bitboard_probe"
        sliders.mkdir()
        shutil.copyfile(HERE.parent / "bitboard_probe/Sliders.bend", sliders / "Sliders.bend")
        for filename in ("bench.bend", "bench_clock.c", "bench_clock.h", "input.c", "support.h"):
            shutil.copyfile(HERE / filename, legal / filename)
        shutil.copyfile(core, legal / "Chess.bend")
        generated = parent / "bench.c"
        run([bun, str(source / "bend2/main.ts"), str(legal / "bench.bend"), "-o", str(generated)])
        binary = parent / "bench"
        run([*common, *flags, "-I", str(legal), str(generated), str(support), "-pthread", "-lm", "-o", str(binary)])
        binaries[name] = binary
    binary = work / "cboard"
    run([*common, *flags, "-DLEGAL_ORACLE", "-DLEGAL_BENCH", "-I", str(ROOT),
         str(HERE / "support.c"), "-pthread", "-lm", "-o", str(binary)])
    binaries["cboard"] = binary
    return binaries, commands


def benchmark(source: Path, bun: str, cc: str, mode: str, cases: list[str],
              samples: int, baseline: Path | None, cpu: int | None) -> dict[str, object]:
    if not 1 <= samples <= 9 or not cases or len(cases) != len(set(cases)):
        raise ValueError("use 1..9 samples and distinct benchmark cases")
    pin = check_compiler(source)
    files = [HERE / n for n in ("Chess.bend", "bench.bend", "bench_clock.c", "bench_clock.h", "support.c", "support.h", "input.c", "benchmark.py")]
    files += [HERE.parent / "bitboard_probe/Sliders.bend"]
    files += sorted((ROOT / "chess_anti_engine/encoding").glob("*.h"))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    report: dict[str, object] = {"schema": 1, "compiler": pin, "sources_sha256": hashes,
        "baseline_chess_sha256": digest(baseline) if baseline is not None else None,
        "clang": command([cc, "--version"]).splitlines()[0], "build_mode": mode,
        "threads": 1, "timing": "CLOCK_MONOTONIC tree traversal only; full warmup each process; setup/printing/table destruction excluded",
        "scope": "single-thread CPU perft, bulk counting at depth one; not MCTS or playing strength"}
    if Path("/proc/cpuinfo").exists():
        report["cpu_model"] = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown")
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    selected = min(affinity) if cpu is None and affinity else cpu
    if selected is not None and (affinity is None or selected not in affinity):
        raise ValueError("requested CPU is not available for affinity")
    report["pinned_cpu"] = selected
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="deepfin-bend-perft-bench-") as tmp:
        binaries, commands = build_benchmarks(source, Path(tmp), bun, cc, BUILD_MODES[mode], baseline)
        report["build_commands"] = commands
        try:
            if selected is not None:
                os.sched_setaffinity(0, {selected})
            for case in cases:
                fen, depth, expected = CASES[case]
                values: dict[str, list[int]] = {name: [] for name in binaries}
                raw: list[dict[str, object]] = []
                names = list(binaries)
                for trial in range(samples):
                    # Balance each complete rotation cycle; reverse alternate cycles.
                    order = names[trial % len(names):] + names[:trial % len(names)]
                    if (trial // len(names)) % 2:
                        order = order[::-1]
                    for name in order:
                        args = [str(binaries[name])] + ([] if name == "cboard" else ["--threads", "1"])
                        output = command(args, input_text=request(fen_position(fen), depth))
                        row = parse_sample(output, depth, expected, bend=name != "cboard")
                        ns = int(row["nanoseconds"])
                        values[name].append(ns)
                        raw.append({"trial": trial, "engine": name, **row})
                stats = {name: summary(times, expected) for name, times in values.items()}
                results.append({"case": case, "fen": fen, "depth": depth, "nodes": expected,
                    "raw": raw, "summary": stats,
                    "bend_over_c_time_ratio": stats["bend"]["median_seconds"] / stats["cboard"]["median_seconds"]})
        finally:
            if affinity is not None:
                os.sched_setaffinity(0, affinity)
    report["results"] = results
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-root", type=Path, default=ROOT / "build/bend_u64_toolchain/source")
    parser.add_argument("--bun", default=os.environ.get("BUN", "bun"))
    parser.add_argument("--cc", default=os.environ.get("CC", "clang"))
    parser.add_argument("--mode", choices=BUILD_MODES, default="native")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--samples", type=int, choices=range(1, 10), default=3)
    parser.add_argument("--baseline-chess", type=Path, help="optional previous Chess.bend source; hash recorded")
    parser.add_argument("--cpu", type=int, help="default: first available CPU on platforms with affinity")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        for executable in (args.bun, args.cc):
            if shutil.which(executable) is None:
                raise ValueError(f"executable not found: {executable}")
        result = benchmark(args.compiler_root.resolve(), args.bun, args.cc, args.mode,
                           args.cases, args.samples, args.baseline_chess, args.cpu)
        text = json.dumps(result, indent=2) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text)
        print(text, end="")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Bend perft benchmark failed: {exc}\n")


if __name__ == "__main__":
    main()
