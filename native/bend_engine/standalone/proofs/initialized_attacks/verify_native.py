"""Actual seeded full-table attack/check queries versus independent forward geometry.

The coordinate/set reference is reused, unmodified, from the accepted attack-witness
verifier. This probe additionally queries queen masks and repeats the complete
callback sequence under public zero-seed and two nonzero-seed full initializations.
No proof-interface Spec, witness scan, or expected decision executes in the candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
REFERENCE = SUITE.parent / "attack_witness/verify_native.py"
_spec = importlib.util.spec_from_file_location("accepted_attack_reference", REFERENCE)
if _spec is None or _spec.loader is None:
    raise ImportError(str(REFERENCE))
Ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Ref)
MODES = Ref.MODES


def run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                            env={**os.environ, "BEND_NO_TELEMETRY": "1", "TERM": "dumb"}, check=False)
    if result.returncode != 0 or result.stderr:
        raise AssertionError((args[:3], result.returncode, result.stderr[-2000:]))
    return result


def board_from_input(inp: list[int]) -> list:
    planes = [(inp[2 + i * 2] << 32) | inp[3 + i * 2] for i in range(8)]
    return [(set(k for k in range(6) if planes[k] & (1 << sq)),
             set(c for c, plane in [(1, planes[6]), (0, planes[7])] if plane & (1 << sq)))
            for sq in range(64)]


def fixtures() -> list[dict]:
    cases = Ref.fixtures()
    for case in cases:
        inp = case["input"]
        queen = sum(1 << sq for sq in Ref.geometry(board_from_input(inp), inp[0], 4, inp[1]))
        old = case["expected"]
        case["expected"] = old[:10] + [queen >> 32, queen & 0xFFFFFFFF] + old[10:]
    return cases


def observe(binary: Path, context: int, cases: list[dict]) -> list[list[int]]:
    outputs = []
    for start in range(0, len(cases), 512):
        batch = cases[start:start + 512]
        raw = run([str(binary), str(context), *[str(x) for case in batch for x in case["input"]]], 90).stdout
        lines = raw.splitlines()
        if len(lines) != len(batch):
            raise AssertionError(("record count", start, len(lines), len(batch)))
        for line in lines:
            fields = line.split()
            assert fields[0] == "case" and len(fields) == 16, line
            outputs.append([int(x) for x in fields[1:]])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    bun, cc = os.environ.get("BUN", "bun"), os.environ.get("CC", "clang")
    identity = run([bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)]).stdout
    originals = {str(p.relative_to(ENGINE)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [*SUITE.glob("*.bend"), SUITE / "focused.js", Path(__file__), REFERENCE,
                           ENGINE / "legal_probe/Chess.bend", ENGINE / "standalone/Tables.bend"]}
    cases = fixtures()
    expected = [c["expected"] for c in cases]
    modes, mutations = [], []
    with tempfile.TemporaryDirectory(prefix="deepfin-initialized-attacks-") as tmpdir:
        tmp = Path(tmpdir)

        def generate(probe: Path, name: str) -> Path:
            out = tmp / (name + ".c")
            run([bun, str(compiler / "bend2/main.ts"), str(probe), "-o", str(out)])
            assert out.is_file()
            return out

        def build(c: Path, name: str, flags: list[str]) -> Path:
            out = tmp / name
            run([cc, "-std=c11", "-O2", *flags, str(c), "-pthread", "-lm", "-o", str(out)])
            return out

        source = generate(SUITE / "probe.bend", "probe")
        invalid = [[], ["3"], ["0", "x"], ["0", "-1"], ["0", "4294967296"], ["0", "0"],
                   ["0", "64", *map(str, cases[0]["input"][1:])],
                   ["0", str(cases[0]["input"][0]), "2", *map(str, cases[0]["input"][2:])],
                   ["0", *map(str, cases[0]["input"]), "0"]]
        for mode, flags in MODES.items():
            binary = build(source, mode, flags)
            contexts = []
            for context in range(3):
                actual = observe(binary, context, cases)
                if actual != expected:
                    raise AssertionError((mode, context, next((i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b)))
                contexts.append({"context": context, "rows": len(actual),
                                 "output_sha256": hashlib.sha256(json.dumps(actual, separators=(",", ":")).encode()).hexdigest()})
                print(f"PASS {mode} context {context}: {len(actual)} complete attack/check observations", flush=True)
            for batch in invalid:
                p = subprocess.run([str(binary), *batch], capture_output=True, text=True, timeout=30, check=False)
                assert p.returncode == 2 and "invalid" in p.stderr + p.stdout and "runtime error" not in p.stderr
            modes.append({"mode": mode, "contexts": contexts, "invalid_rejections": len(invalid)})
        # Each mutation must compile and execute before a wrong numerical result is accepted as detection.
        for name, rel, old, new in [
            ("wrong-pawn-direction", "legal_probe/Chess.bend",
             "attacked_pawn(b, sq, by, attack(0, table, sq, U32.xor(by, 1), occupied(b)))",
             "attacked_pawn(b, sq, by, attack(0, table, sq, by, occupied(b)))"),
            ("queen-omits-diagonal", "legal_probe/Chess.bend", "or_result(rook, slide(table, U32.add(64, sq), occ))", "(table, rook)"),
            ("wrong-king-query-slot", "legal_probe/Chess.bend", "Array.get(U64, table, U32.add(320, sq))", "Array.get(U64, table, U32.add(256, sq))"),
            ("slider-ignores-blockers", "standalone/Tables.bend", "U64.test_bit(occ, U32.to_nat(next))", "False{}"),
        ]:
            copy = tmp / name
            shutil.copytree(ENGINE, copy)
            target = copy / rel
            text = target.read_text()
            assert text.count(old) == 1, (name, text.count(old))
            target.write_text(text.replace(old, new))
            c = generate(copy / "standalone/proofs/initialized_attacks/probe.bend", name)
            binary = build(c, name + "-exe", [])
            observed = observe(binary, 1, cases[:1024])
            mismatch = next((i for i, (a, b) in enumerate(zip(observed, expected[:1024])) if a != b), None)
            assert mismatch is not None, name
            mutations.append({"name": name, "compiled_and_executed": True, "rejected": True,
                              "context": 1, "row": mismatch, "observed": observed[mismatch], "expected": expected[mismatch]})
            print(f"PASS mutation {name}: wrong-value rejection at row {mismatch}", flush=True)
    for path, digest in originals.items():
        assert hashlib.sha256((ENGINE / path).read_bytes()).hexdigest() == digest, path
    assert run([bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)]).stdout == identity
    report = {"native_gate": "PASS", "compiler_identity": identity, "cc": run([cc, "--version"]).stdout,
              "base_requests": len(cases), "initializations": 3, "rows_per_mode": len(cases) * 3,
              "distinct_inputs": len({(ctx, *c["input"]) for ctx in range(3) for c in cases}),
              "fields_per_mode": len(cases) * 3 * 15,
              "singleton_at_target_per_context": sum(c["singleton_at_target"] for c in cases),
              "missing_king_per_context": sum(c["king_count"] == 0 for c in cases),
              "multiple_kings_per_context": sum(c["king_count"] > 1 for c in cases),
              "fixture_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "modes": modes, "mutations": mutations, "source_sha256s": originals,
              "scope": "Actual zero/nonzero full initialization, six piece masks, attacked and bounded in_check; independent forward coordinate/set reference. No proof model, full-buffer or exhaustive-position native result."}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
