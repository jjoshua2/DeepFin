"""Observe table state after actual initialized attack/check calls.

Keeps the qualified producer and forward-coordinate reference unchanged. A temporary
copy of the native probe plants/reads one unused slot. One mutant preserves all query
answers but loses that slot; another preserves the current Boolean and breaks the
following check. This is sampled native state evidence, not a lifetime/full-buffer proof.
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
_SPEC = importlib.util.spec_from_file_location("initialized_attack_reference", SUITE / "verify_native.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("missing qualified initialized-attack reference")
REF = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(REF)
SENTINEL = (826366246, 1398314899)  # 0x31415926 / 0x53589793, both nonzero.
SLOT = 131071
RESULT = "(table, Bool.not(U64.is_zero(U64.and(hits, color(b, by)))))"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise AssertionError(("ambiguous probe/mutation site", before, text.count(before)))
    return text.replace(before, after)


def probe_source(source: str) -> str:
    # Insertion is test instrumentation, not a production table-builder change.
    source = replace_once(source, "def rows(fuel: Nat", '''def state_result(r: Array<U64> & U64) -> IO(Unit):
  (table,value) = r
  IO.print("sentinel " ++ show(value))

def rows(fuel: Nat''')
    source = replace_once(source, "case _ Nil{}: IO.pure(Unit,Unit{})",
                          f"case _ Nil{{}}: state_result(Array.get(U64,table,{SLOT}))")
    return replace_once(source, "rows(512n,ys,initial(c))",
                        f"rows(512n,ys,Array.set(U64,initial(c),{SLOT},U64.from_parts({SENTINEL[0]},{SENTINEL[1]})))")


def fixtures() -> list[dict]:
    cases = []
    for target in range(64):
        for by in (0, 1):
            board = REF.Ref.empty()
            REF.Ref.put(board, target, 5, 1 - by)
            REF.Ref.put(board, target ^ 17, 1, by)
            expected = REF.Ref.expected(board, target, by)
            queen = sum(1 << sq for sq in REF.Ref.geometry(board, target, 4, by))
            cases.append({"input": [target, by, *REF.Ref.encode(board, (1 - by, 15, 64))],
                          "expected": expected[:10] + [queen >> 32, queen & 0xFFFFFFFF] + expected[10:]})
    assert len({tuple(c["input"]) for c in cases}) == len(cases) == 128
    assert cases[0]["expected"][12] == cases[0]["expected"][14] == 1
    return cases


def run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False,
                               env={**os.environ, "BEND_NO_TELEMETRY": "1", "TERM": "dumb"})
    if completed.returncode != 0 or completed.stderr:
        raise AssertionError((args[:3], completed.returncode, completed.stderr[-2500:]))
    return completed


def observe(binary: Path, context: int, cases: list[dict]) -> tuple[list[list[int]], tuple[int, int]]:
    raw = run([str(binary), str(context), *(str(x) for c in cases for x in c["input"])], 90).stdout
    lines = raw.splitlines()
    assert len(lines) == len(cases) + 1, "missing or extra native records"
    sentinel = lines[-1].split()
    assert len(sentinel) == 3 and sentinel[0] == "sentinel", lines[-1]
    values = []
    for line in lines[:-1]:
        fields = line.split()
        assert len(fields) == 16 and fields[0] == "case", line
        row = [int(x) for x in fields[1:]]
        assert all(0 <= x <= 0xFFFFFFFF for x in row)
        values.append(row)
    return values, (int(sentinel[1]), int(sentinel[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    bun, cc = os.environ.get("BUN", "bun"), os.environ.get("CC", "clang")
    identity_command = [bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)]
    identity = run(identity_command).stdout
    paths = [*SUITE.glob("*.bend"), SUITE / "focused.js", SUITE / "verify_native.py", Path(__file__),
             SUITE.parent / "attack_witness/verify_native.py", ENGINE / "legal_probe/Chess.bend",
             ENGINE / "standalone/Tables.bend", ENGINE / "standalone/verify_compiler.js",
             ENGINE / "standalone/toolchain.json"]
    original = {str(p.relative_to(ENGINE)): digest(p) for p in paths}
    cases = fixtures()
    expected = [c["expected"] for c in cases]
    modes, mutations = [], []
    with tempfile.TemporaryDirectory(prefix="initialized-attack-state-") as td:
        temporary = Path(td)
        engine = temporary / "engine"
        shutil.copytree(ENGINE, engine)
        suite = engine / "standalone/proofs/initialized_attacks"
        probe = suite / "table_state_probe.bend"
        assert not probe.exists()
        probe.write_text(probe_source((suite / "probe.bend").read_text()))
        probe_sha = digest(probe)

        def generate(name: str) -> Path:
            source = temporary / (name + ".c")
            run([bun, str(compiler / "bend2/main.ts"), str(probe), "-o", str(source)])
            assert source.is_file()
            return source

        def build(source: Path, name: str, flags: list[str]) -> Path:
            binary = temporary / name
            run([cc, "-std=c11", "-O2", *flags, str(source), "-pthread", "-lm", "-o", str(binary)])
            return binary

        baseline = generate("baseline")
        for mode, flags in REF.MODES.items():
            binary = build(baseline, mode, flags)
            for context in range(3):
                observed, sentinel = observe(binary, context, cases)
                assert observed == expected, (mode, context, "query mismatch")
                assert sentinel == SENTINEL, (mode, context, sentinel)
            modes.append({"mode": mode, "contexts": 3, "rows": len(cases) * 3,
                          "query_fields": len(cases) * 3 * 15, "sentinel_reads": 3,
                          "all_query_values_match": True, "all_sentinels_match": True})
            print(f"PASS {mode}: 384 state-threaded queries; three sentinel reads", flush=True)

        chess = engine / "legal_probe/Chess.bend"
        pristine = chess.read_text()
        for name, replacement in [
            ("lose-returned-table", "(Array.new(U64,17n,U64.zero()), Bool.not(U64.is_zero(U64.and(hits, color(b, by)))))"),
            ("erase-unused-slot-only", f"(Array.set(U64,table,{SLOT},U64.zero()), Bool.not(U64.is_zero(U64.and(hits, color(b, by)))))"),
        ]:
            chess.write_text(replace_once(pristine, RESULT, replacement))
            binary = build(generate(name), name, [])
            observed, sentinel = observe(binary, 1, cases)
            same = observed == expected
            assert observed[0][:14] == expected[0][:14], "first masks/current Boolean must remain correct"
            if name == "lose-returned-table":
                assert observed[0][14] == 0 and expected[0][14] == 1, "subsequent check must detect lost state"
                assert not same
            else:
                assert same, "unused-slot-only corruption must leave EVERY query field unchanged"
            assert sentinel == (0, 0) and sentinel != SENTINEL
            mutations.append({"name": name, "compiled_and_executed": True, "rejected": True,
                              "first_masks_and_attack_boolean_match": True,
                              "all_query_values_match": same, "first_check_observed": observed[0][14],
                              "first_check_expected": expected[0][14], "sentinel_observed": list(sentinel),
                              "sentinel_expected": list(SENTINEL)})
            print(f"PASS mutation {name}: rejected after normal compilation/execution", flush=True)
        chess.write_text(pristine)
    assert {p: digest(ENGINE / p) for p in original} == original
    assert run(identity_command).stdout == identity
    report = {"table_state_gate": "PASS", "compiler_identity": identity, "cc": run([cc, "--version"]).stdout,
              "base_queries": len(cases), "contexts": 3, "registered_law_count_changed": False,
              "registered_source_control_count_changed": False, "unused_slot": SLOT,
              "fixture_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "generated_probe_sha256": probe_sha, "modes": modes, "mutations": mutations,
              "source_sha256s": original,
              "scope": "One unused-slot observation per context plus exact sequential query values; not whole-buffer or native-lifetime correctness. Modes repeat fixtures. Mutation builds use generic C flags."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
