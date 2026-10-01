"""Bounded fail-closed qualification of the actual both-wing projection candidate."""
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

from .qualify import invoke, load_closure, replace, safe
from ..table_preservation._validation import begin_report, require, write_report

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]


def semantic_rejection(result: dict, location: str) -> bool:
    output = result["raw_text"]
    forbidden = (
        "no such file", "RangeError", "Maximum call stack", "more than once",
        "a decreasing self-call", "a defined name", "a parameter or field scrutinee",
        "a pattern (a binder", "out of memory", "OutOfMemory", "WARNING", "unsafe", "TODO",
        "consumed", "erased", "a kind (", "quantity",
    )
    return (
        result["exit_code"] == 1
        and not result["timed_out"]
        and "expected" in output
        and "observed" in output
        and not re.search(r"(?m)^- expected\s*:\s*[-+][A-Za-z_]\w*\s*$", output)
        and not re.search(r"(?m)^- (?:expected|observed)\s*:\s*(?:Data|Type|Quant|Kind\([^\n]*\)|&[012])\s*$", output)
        and bool(re.search(r"Location:\s*(?:[\w./]+\.)*" + re.escape(location) + r"\b", output))
        and not any(word in output for word in forbidden)
    )


def manifest() -> None:
    consumer = (SUITE / "castle_consumer.bend").read_text()
    require("import ../castle_sequence/PROOF.bend as InheritedProof" in consumer,
            "missing inherited initialized proof bodies")
    require("import ./CastleComposition.bend as Composition" in consumer,
            "missing new proof bodies")
    require(re.findall(r"^def (\w+)\(", consumer, re.MULTILINE) == [
        "use_exact_castling_projection", "use_no_duplicate_castles",
    ], "complete consumer inventory")
    require("Composition.exact(d,seed,n,extra,b,white,depth,tables,full,good,turn,one)" in consumer,
            "missing complete exact-pair consumer")
    require("Composition.no_duplicates(d,seed,n,extra,b,white,depth,tables,full,good,turn,one)" in consumer,
            "missing complete uniqueness consumer")
    require("{P.result(Chess.legal_moves(Init.run(d,seed,n,extra),b)) ==" in consumer,
            "consumer must use actual optimized generator")
    require("(Init.run(d,seed,n,extra),P.expected(b,white)) : Array<U64> & List<&2,Chess.Ply>}" in consumer,
            "consumer must retain the complete returned array")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    begin_report(args.report, "castling_projection_gate")
    report: dict = {
        "castling_projection_gate": "NOT_COMPLETED",
        "accepted_candidate_definitions": 0,
        "scope": (
            "Exact both-wing castling projection and Boolean complete-Ply uniqueness "
            "for actual optimized legal_moves under symbolic initialized recipe, "
            "valid Board, Boolean side and home-king singleton premises. "
            "Not actual public-builder or independent history/whole-generator correctness."
        ),
    }
    try:
        compiler = args.compiler.resolve()
        bun = os.environ.get("BUN", "bun")
        pin_command = [bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)]
        pin = subprocess.run(pin_command, capture_output=True, text=True, timeout=30, check=True)
        require(not pin.stderr, "compiler identity warning")
        report["compiler_identity"] = pin.stdout
        manifest()
        closure = load_closure()
        graph = closure(SUITE / "castle_consumer.bend", ENGINE)
        paths = {ENGINE / path for path in graph}
        paths.update(SUITE / name for name in (
            "qualify_castles.py", "test_qualify_castles.py", "qualify.py", "CASTLING.md",
        ))
        paths.update(SUITE.parent / name for name in (
            "table_preservation/focused.py", "table_preservation/_validation.py",
        ))
        paths.update(ENGINE / name for name in (
            "standalone/toolchain.json", "standalone/verify_compiler.js",
        ))

        def identities() -> dict[str, str]:
            return {
                str(path.relative_to(ENGINE)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(paths)
            }

        before = identities()
        report["source_sha256s"] = before
        write_report(args.report, report)
        helper = invoke(bun, compiler, SUITE / "CastleProjection.bend", 180)
        report["projection_helper"] = helper
        write_report(args.report, report)
        require(safe(helper), "projection helper did not return exact safe checker success")
        consumer = invoke(bun, compiler, SUITE / "castle_consumer.bend", 600)
        report["consumer"] = consumer
        write_report(args.report, report)
        require(safe(consumer), "complete consumer did not return exact safe checker success")
        controls: list[dict] = []
        report["controls"] = controls
        mutations = [
            (
                "ordinary-heads-not-erased", "standalone/proofs/generator_contract/CastleSpec.bend",
                "proj_bit(U32.is_eq(C.flag(m),2),m,project(rest))",
                "proj_bit(True{},m,project(rest))", "CastleProjection.bend", "ordinary_cons",
            ),
            (
                "castling-heads-erased", "standalone/proofs/generator_contract/CastleSpec.bend",
                "proj_bit(U32.is_eq(C.flag(m),2),m,project(rest))",
                "proj_bit(False{},m,project(rest))", "CastleProjection.bend", "castle_retain",
            ),
            (
                "actual-final-check-flipped-side", "legal_probe/Chess.bend",
                "filter_after(m, acc, in_check(table, make_move(b, m), get_turn(b)))",
                "filter_after(m, acc, in_check(table, make_move(b, m), U32.xor(get_turn(b), 1)))",
                "CastleProjection.bend", "step",
            ),
            (
                "actual-generator-repeats-kingside", "legal_probe/Chess.bend",
                "filter_prepare(b, castle_side(b, False{}, castle_side(b, True{},\n    scan(",
                "filter_prepare(b, castle_side(b, True{}, castle_side(b, True{},\n    scan(",
                "castle_consumer.bend", "projected",
            ),
            (
                "two-wing-list-repeats-kingside", "standalone/proofs/generator_contract/CastleUnique.bend",
                "S.select(queen_reject,S.move(white,False{}),Nil{})",
                "S.select(queen_reject,S.move(white,True{}),Nil{})",
                "CastleUnique.bend", "selected",
            ),
        ]
        for name, target, old, new, entry_name, location in mutations:
            with tempfile.TemporaryDirectory(prefix="castling-projection-control-") as directory:
                copied = Path(directory) / "engine"
                shutil.copytree(ENGINE, copied, symlinks=True)
                replace(copied / target, old, new)
                entry = copied / "standalone/proofs/generator_contract" / entry_name
                closure(entry, copied)
                result = invoke(bun, compiler, entry, 120)
                controls.append({
                    "name": name, "kind": "source semantic/refinement",
                    "target": target, "expected_location": location,
                    "rejected": semantic_rejection(result, location), "result": result,
                })
                write_report(args.report, report)
                require(semantic_rejection(result, location),
                        "control was not the intended semantic rejection: " + name)
        require(len(controls) == 5, "incomplete control inventory")
        require(before == identities(), "source drift")
        after = subprocess.run(pin_command, capture_output=True, text=True, timeout=30, check=True)
        require(pin.stdout == after.stdout and not after.stderr, "compiler drift")
        report["castling_projection_gate"] = "PASS"
        report["accepted_candidate_definitions"] = 2
    except Exception as error:
        report["failure"] = type(error).__name__ + ": " + str(error)
        write_report(args.report, report)
        raise
    write_report(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
