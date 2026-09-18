#!/usr/bin/env python3
"""Opt-in, CPU-only Bend U64/CBoard parity. Uses only the Python standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIELDS = ("bishop", "square", "cases", "failures", "first", "sum_hi", "sum_lo")
ROW = re.compile(" ".join(rf"{key}=(\d+)" for key in FIELDS))
MODES = {
    "generic": [],
    "portable": ["-DBEND_U64_PORTABLE"],
    "native": ["-march=native"],
    "ubsan": ["-fsanitize=undefined", "-fno-sanitize-recover=all"],
}


def compiler_digest(source: Path) -> str:
    """Fingerprint compiler, prelude and effects, not version labels or git state."""
    files = [source / "bend2" / name for name in ("base.bend", "bend.ts", "comp.ts", "main.ts")]
    files.extend(p for p in (source / "bend2/effs").rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(file.relative_to(source).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def check_compiler(source: Path) -> dict[str, str]:
    pin: dict[str, str] = json.loads((HERE / "toolchain.json").read_text())
    actual = compiler_digest(source)
    if actual != pin["source_sha256"]:
        raise ValueError("compiler sources do not match the pinned U64 fork; "
                         "a stock Bend version label is not sufficient")
    return pin


def relevant_bits(bishop: int, square: int) -> int:
    """Independent geometric count; prevents truncated/empty fixtures passing."""
    directions = ((1, 1), (1, -1), (-1, 1), (-1, -1)) if bishop else (
        (0, 1), (0, -1), (1, 0), (-1, 0))
    bits = 0
    for df, dr in directions:
        file, rank = square % 8, square // 8
        length = 0
        while True:
            file, rank = file + df, rank + dr
            if not (0 <= file < 8 and 0 <= rank < 8):
                break
            length += 1
        bits += max(0, length - 1)
    return bits


def parse_report(text: str) -> list[dict[str, int]]:
    rows: dict[tuple[int, int], dict[str, int]] = {}
    for line in text.splitlines():
        match = ROW.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed slider report: {line!r}")
        row = dict(zip(FIELDS, map(int, match.groups()), strict=True))
        key = row["bishop"], row["square"]
        if key[0] not in (0, 1) or not 0 <= key[1] < 64 or key in rows:
            raise ValueError(f"invalid/duplicate slider row: {key}")
        if row["cases"] != 2 * (1 << relevant_bits(*key)) + 64:
            raise ValueError(f"incomplete fixture traversal: {key}")
        if row["failures"] != 0 or row["first"] != 0:
            raise ValueError(f"incorrect slider result: {row}")
        if row["sum_hi"] >= 1 << 32 or row["sum_lo"] >= 1 << 32:
            raise ValueError("invalid U64 checksum limbs")
        rows[key] = row
    if len(rows) != 128:
        raise ValueError(f"expected 128 square/piece rows, got {len(rows)}")
    return [rows[key] for key in sorted(rows)]


def run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                            capture_output=True, timeout=120, check=False)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def verify(source: Path, bun: str, cc: str, modes: list[str]) -> dict[str, object]:
    pin = check_compiler(source)
    results: list[dict[str, object]] = []
    reference: list[dict[str, int]] | None = None
    clean_env = dict(os.environ)
    clean_env.pop("BEND_U64_CORRUPT", None)
    clean_env["BEND_NO_TELEMETRY"] = "1"
    with tempfile.TemporaryDirectory(prefix="deepfin-bend-u64-") as tmp:
        work = Path(tmp)
        generated = work / "sliders.c"
        run([bun, str(source / "bend2/main.ts"), str(HERE / "main.bend"),
             "-o", str(generated)], env=clean_env)
        for mode in modes:
            flags = MODES[mode]
            obj, binary = work / f"{mode}.o", work / mode
            # The oracle TU needs magic constants, so do not give it BMI2.
            sanitize = MODES["ubsan"] if mode == "ubsan" else []
            run([cc, "-std=c11", "-O3", *sanitize, "-I", str(ROOT), "-c",
                 str(HERE / "fixtures.c"), "-o", str(obj)], env=clean_env)
            run([cc, "-std=c11", "-O3", *flags, str(generated), str(obj),
                 "-pthread", "-lm", "-o", str(binary)], env=clean_env)
            start = time.perf_counter()
            output = run([str(binary), "--threads", "1"], env=clean_env)
            seconds = time.perf_counter() - start
            rows = parse_report(output)
            if reference is not None and rows != reference:
                raise ValueError(f"{mode}: output differs from first build variant")
            reference = rows
            # A broken table must fail the EXECUTABLE, not merely our parser.
            bad = subprocess.run([str(binary), "--threads", "1"], cwd=ROOT,
                                 env={**clean_env, "BEND_U64_CORRUPT": "1"},
                                 text=True, capture_output=True, timeout=120, check=False)
            if bad.returncode != 1 or "U64 slider mismatch:" not in bad.stderr:
                raise ValueError(f"{mode}: corrupt table was not correctly rejected")
            results.append({"mode": mode, "squares": len(rows),
                            "occupancy_cases": sum(row["cases"] for row in rows),
                            "attack_comparisons": 2 * sum(row["cases"] for row in rows),
                            "corruption_rejected": True,
                            "verification_wall_seconds": seconds})
    return {"compiler_repository": pin["repository"], "compiler_revision": pin["revision"],
            "compiler_sources_verified": True,
            "clang": run([cc, "--version"]).splitlines()[0], "results": results,
            "scope": "slider parity including fixture construction; not perft or a throughput benchmark"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-root", type=Path,
                        default=ROOT / "build/bend_u64_toolchain/source")
    parser.add_argument("--bun", default=os.environ.get("BUN", "bun"))
    parser.add_argument("--cc", default=os.environ.get("CC", "clang"))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        for command in (args.bun, args.cc):
            if shutil.which(command) is None:
                raise ValueError(f"executable not found: {command}")
        report = verify(args.compiler_root.resolve(), args.bun, args.cc, args.modes)
        text = json.dumps(report, indent=2) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text)
        print(text, end="")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Bend U64 validation failed: {exc}\n")


if __name__ == "__main__":
    main()
