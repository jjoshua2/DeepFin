"""Actual non-slider initialization/query checks against independent coordinates.

No proof-only functions run in the candidate. Each build mode repeats fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
MODES = {"generic": [], "portable": ["-DBEND_U64_PORTABLE"], "native": ["-march=native"],
         "ubsan": ["-fsanitize=undefined", "-fno-sanitize-recover=all"]}


def run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False,
                            env={**os.environ, "BEND_NO_TELEMETRY": "1", "TERM": "dumb"})
    if result.returncode != 0 or result.stderr:
        raise AssertionError((args[:3], result.returncode, result.stderr[-2000:]))
    return result


def mask(kind: int, origin: int) -> int:
    x, y = origin % 8, origin // 8
    if kind == 0:
        offsets = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)
                   if abs(dx * dy) == 2]
    elif kind == 1:
        offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    else:
        offsets = [(dx, 1 if kind == 2 else -1) for dx in (-1, 1)]
    return sum(1 << ((y + dy) * 8 + x + dx) for dx, dy in offsets
               if 0 <= x + dx < 8 and 0 <= y + dy < 8)


def fixtures() -> list[dict]:
    rows = []
    for init in range(4):
        for kind in range(4):
            for square in range(64):
                value = mask(kind, square)
                for occupancy in (0, 2**64 - 1, (0x9249249249249249 ^ (1 << square))):
                    rows.append({"input": [init, kind, square, occupancy >> 32, occupancy & 0xffffffff],
                                 "expected": [value >> 32, value & 0xffffffff]})
    assert len(rows) == len({tuple(r["input"]) for r in rows}) == 3072
    return rows


def observe(binary: Path, cases: list[dict]) -> list[list[int]]:
    results = []
    for init in range(4):
        group = [c for c in cases if c["input"][0] == init]
        if not group:
            continue
        raw = run([str(binary), str(init), *(str(x) for c in group for x in c["input"][1:])], 90).stdout
        lines = raw.splitlines()
        assert len(lines) == len(group), (init, len(lines), len(group))
        for line in lines:
            fields = line.split()
            assert len(fields) == 3 and fields[0] == "mask", line
            results.append([int(x) for x in fields[1:]])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    bun, cc = os.environ.get("BUN", "bun"), os.environ.get("CC", "clang")
    identity = run([bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)]).stdout
    paths = [*SUITE.glob("*.bend"), SUITE / "focused.js", Path(__file__),
             ENGINE / "legal_probe/Chess.bend", ENGINE / "standalone/Tables.bend"]
    originals = {str(p.relative_to(ENGINE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    cases = fixtures()
    expected = [r["expected"] for r in cases]
    modes, mutants = [], []
    with tempfile.TemporaryDirectory(prefix="deepfin-leaper-geometry-") as td:
        temporary = Path(td)

        def generate(root: Path, name: str) -> Path:
            source = temporary / (name + ".c")
            run([bun, str(compiler / "bend2/main.ts"), str(root / "standalone/proofs/attack_geometry/probe.bend"), "-o", str(source)])
            assert source.is_file()
            return source

        def build(source: Path, name: str, flags: list[str]) -> Path:
            binary = temporary / name
            run([cc, "-std=c11", "-O2", *flags, str(source), "-pthread", "-lm", "-o", str(binary)])
            return binary

        source = generate(ENGINE, "probe")
        invalid = [[], ["x"], ["-1"], ["4294967296"], ["4"], ["0", "0"],
                   ["0", "4", "0", "0", "0"], ["0", "0", "64", "0", "0"],
                   ["0", *(str(v) for _ in range(1025) for v in [0, 0, 0, 0])]]
        for mode, flags in MODES.items():
            binary = build(source, mode, flags)
            actual = observe(binary, cases)
            mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
            assert mismatch is None and len(actual) == len(expected), (mode, mismatch)
            for batch in invalid:
                r = subprocess.run([str(binary), *batch], capture_output=True, text=True, timeout=30, check=False)
                assert r.returncode == 2 and "invalid" in r.stderr + r.stdout and "runtime error" not in r.stderr
            modes.append({"mode": mode, "rows": len(actual), "invalid_rejections": len(invalid),
                          "output_sha256": hashlib.sha256(json.dumps(actual, separators=(",", ":")).encode()).hexdigest()})
            print(f"PASS {mode}: {len(actual)} actual masks after initialization", flush=True)
        for name, filename, old, new in [
            ("wrong-king-query-slot", "legal_probe/Chess.bend",
             "Array.get(U64, table, U32.add(320, sq))", "Array.get(U64, table, U32.add(256, sq))"),
            ("shifted-black-pawn-storage", "standalone/Tables.bend",
             "Array.set(U64, a, U32.add(448, sq), pawns(sq, 4294967295))",
             "Array.set(U64, a, U32.add(447, sq), pawns(sq, 4294967295))"),
            ("reversed-white-pawn-storage", "standalone/Tables.bend",
             "Array.set(U64, a, U32.add(384, sq), pawns(sq, 1))",
             "Array.set(U64, a, U32.add(384, sq), pawns(sq, 4294967295))"),
        ]:
            copy = temporary / name
            shutil.copytree(ENGINE, copy)
            target = copy / filename
            text = target.read_text()
            assert text.count(old) == 1, (name, "nonunique mutation")
            target.write_text(text.replace(old, new))
            binary = build(generate(copy, name), name + "-exe", [])
            actual = observe(binary, cases[:768])
            mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected[:768])) if a != b), None)
            assert mismatch is not None, name
            mutants.append({"name": name, "compiled_and_executed": True, "rejected": True,
                            "row": mismatch, "input": cases[mismatch]["input"],
                            "observed": actual[mismatch], "expected": expected[mismatch]})
    for filename, digest in originals.items():
        assert hashlib.sha256((ENGINE / filename).read_bytes()).hexdigest() == digest, filename
    assert run([bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)]).stdout == identity
    report = {"native_gate": "PASS", "compiler_identity": identity, "cc": run([cc, "--version"]).stdout,
              "rows_per_mode": len(cases), "distinct_inputs": len(cases), "fields_per_mode": len(cases) * 2,
              "masks_per_initialization": 256, "initializations": 4, "occupancies_per_mask": 3,
              "fixture_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "modes": modes, "mutations": mutants, "source_sha256s": originals,
              "scope": "Actual initialization and leaper Chess.attack versus independent coordinates; selected mask queries, not full-buffer/native lifetime proof. Full builder, nonzero seed, partial preceding table loop and planted last-slot sentinel; no proof/model execution."}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
