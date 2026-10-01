"""Bounded fail-closed qualification of forward attack composition."""
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

from ..generator_contract.qualify import invoke, load_closure, replace, safe
from ..table_preservation._validation import begin_report, require, write_report

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]


def semantic_rejection(result: dict, location: str) -> bool:
    output = result["raw_text"]
    forbidden = (
        "no such file", "RangeError", "Maximum call stack", "more than once",
        "a decreasing self-call", "a defined name", "a parameter or field scrutinee",
        "a pattern (a binder", "out of memory", "OutOfMemory", "WARNING", "unsafe", "TODO",
        "consumed", "erased", "a kind (", "quantity", "a filled definition",
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
    consumer = (SUITE / "consumer.bend").read_text()
    require("import ../generator_contract/castle_consumer.bend as QualifiedCastles" in consumer,
            "missing qualified castling dependency bodies")
    require(re.findall(r"^def (\w+)\(", consumer, re.MULTILINE) == [
        "use_geometry", "use_initialized_attack", "use_initialized_check",
        "use_forward_castle_subset",
    ], "complete consumer inventory")
    required = (
        "Bridge.geometry(b,target,by,bound)",
        "Actual.attacked(d,seed,n,extra,b,target,by,depth,tables,full,bound)",
        "Actual.checked(d,seed,n,extra,b,target,side,depth,tables,full,bound,one)",
        "Castles.exact(d,seed,n,extra,b,white,depth,tables,full,good,turn,one)",
        "{Chess.attacked(Init.run(d,seed,n,extra),b,U32.from_nat(target),Bool.to_u32(by)) ==",
        "(Init.run(d,seed,n,extra),Forward.attacked(b,target,by)) : Array<U64> & Bool}",
        "{Projection.result(Chess.legal_moves(Init.run(d,seed,n,extra),b)) ==",
        "(Init.run(d,seed,n,extra),Castles.expected(b,white)) : Array<U64> & List<&2,Chess.Ply>}",
    )
    require(all(text in consumer for text in required), "missing complete actual statement or proof call")
    helper = (SUITE / "helper.bend").read_text()
    require("import ./Fixtures.bend as Fixtures" in helper, "missing fixture positive baseline")
    require("import ./Runtime.bend as Runtime" in helper, "missing runtime positive baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    begin_report(args.report, "forward_attack_gate")
    report: dict = {
        "forward_attack_gate": "NOT_COMPLETED",
        "accepted_candidate_definitions": 0,
        "scope": (
            "Independent forward-source witness equivalence, actual initialized attacked "
            "and singleton in_check pairs, and the actual castling subset with forward checks. "
            "Symbolic initializer and explicit existing local premises; not Tables.build, "
            "history reachability or whole-generator correctness."
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
        graph = closure(SUITE / "consumer.bend", ENGINE)
        graph.update(closure(SUITE / "helper.bend", ENGINE))
        paths = {ENGINE / path for path in graph}
        paths.update(SUITE / name for name in (
            "qualify.py", "test_qualify.py", "README.md",
        ))
        paths.update(SUITE.parent / name for name in (
            "table_preservation/focused.py", "table_preservation/_validation.py",
            "generator_contract/qualify.py",
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
        helper = invoke(bun, compiler, SUITE / "helper.bend", 180)
        report["projection_helper"] = helper
        write_report(args.report, report)
        require(safe(helper), "projection helper did not return exact safe checker success")
        controls: list[dict] = []
        report["controls"] = controls
        mutations = [
            (
                "wrong-forward-pawn-color", "standalone/proofs/forward_attacks/Spec.bend",
                "Bool.pick(G.Leaper,by,G.WhitePawn{},G.BlackPawn{})",
                "Bool.pick(G.Leaper,by,G.BlackPawn{},G.WhitePawn{})",
                "Fixtures.bend", "white_pawn_forward",
            ),
            (
                "omit-upper-source-limb", "standalone/proofs/forward_attacks/Spec.bend",
                "scan(32n,32n,oh,ph,nh,bh,rh,qh,kh,dst,by,occ)",
                "False{}", "Fixtures.bend", "upper_rook",
            ),
            (
                "omit-queen-diagonal-witness", "standalone/proofs/forward_attacks/Spec.bend",
                "Bool.and(Bool.or(b,q),ba)", "Bool.and(b,ba)",
                "Fixtures.bend", "queen_diagonal",
            ),
            (
                "ignore-forward-slider-blockers", "standalone/proofs/forward_attacks/Spec.bend",
                "observe(b,W.select(b,by),target,by,Chess.occupied(b))",
                "observe(b,W.select(b,by),target,by,U64.zero())",
                "Fixtures.bend", "blocked_rook",
            ),
            (
                "actual-pawn-query-not-reversed", "legal_probe/Chess.bend",
                "attacked_pawn(b, sq, by, attack(0, table, sq, U32.xor(by, 1), occupied(b)))",
                "attacked_pawn(b, sq, by, attack(0, table, sq, by, occupied(b)))",
                "Runtime.bend", "unfold",
            ),
        ]
        for name, target, old, new, entry_name, location in mutations:
            with tempfile.TemporaryDirectory(prefix="forward-attack-control-") as directory:
                copied = Path(directory) / "engine"
                shutil.copytree(ENGINE, copied, symlinks=True)
                replace(copied / target, old, new)
                entry = copied / "standalone/proofs/forward_attacks" / entry_name
                closure(entry, copied)
                result = invoke(bun, compiler, entry, 60)
                controls.append({
                    "name": name, "kind": "source semantic/refinement",
                    "target": target, "expected_location": location,
                    "rejected": semantic_rejection(result, location), "result": result,
                })
                write_report(args.report, report)
                require(semantic_rejection(result, location),
                        "control was not the intended semantic rejection: " + name)
        require(len(controls) == 5, "incomplete control inventory")
        consumer = invoke(bun, compiler, SUITE / "consumer.bend", 1050)
        report["consumer"] = consumer
        write_report(args.report, report)
        require(safe(consumer), "complete consumer did not return exact safe checker success")
        require(before == identities(), "source drift")
        after = subprocess.run(pin_command, capture_output=True, text=True, timeout=30, check=True)
        require(pin.stdout == after.stdout and not after.stderr, "compiler drift")
        report["forward_attack_gate"] = "PASS"
        report["accepted_candidate_definitions"] = 4
    except Exception as error:
        report["failure"] = type(error).__name__ + ": " + str(error)
        write_report(args.report, report)
        raise
    write_report(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
