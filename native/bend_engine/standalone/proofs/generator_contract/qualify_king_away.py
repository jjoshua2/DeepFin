"""Bounded fail-closed gate for arbitrary-table king-away castling exclusion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .qualify import invoke, load_closure, replace, safe
from .qualify_castles import semantic_rejection
from ..table_preservation._validation import begin_report, require, write_report

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]


def manifest() -> None:
    consumer = (SUITE / "king_away_consumer.bend").read_text()
    require("import ../castle_sequence/PROOF.bend as InheritedProof" in consumer,
            "inherited proof bodies must be loaded")
    require("import ./KingAway.bend as Away" in consumer, "candidate bodies missing")
    require("Away.exact(table,b,white,sq,turn,one,bound,away)" in consumer,
            "complete consumer call missing")
    require("{P.result(Chess.legal_moves(table,b)) == (table,Nil{}) : Array<U64> & List<&2,Chess.Ply>}" in consumer,
            "complete actual-generator pair proposition missing")
    require(consumer.count("def use_exact_king_away(") == 1, "consumer inventory")
    require("table: Array<U64>,+b: Chess.Board,+white: Bool,+sq: Nat" in consumer,
            "arbitrary array and Board domain missing")
    require("one: {K.kings(b,white) == U64.bit(sq) : U64}" in consumer,
            "moving king singleton missing")
    require("bound: {Nat.is_lt(sq,64n) == True{} : Bool}" in consumer, "square bound missing")
    require("away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool}" in consumer,
            "off-home premise missing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    begin_report(args.report, "king_away_gate")
    report: dict = {"king_away_gate": "NOT_COMPLETED", "accepted_candidate_definitions": 0,
                    "scope": "Actual optimized legal_moves empty flag-2 projection and complete array equality; arbitrary table/Board, Boolean side, bounded off-home moving-king singleton."}
    try:
        compiler = args.compiler.resolve()
        bun = os.environ.get("BUN", "bun")
        pin_command = [bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)]
        pin = subprocess.run(pin_command, capture_output=True, text=True, timeout=30, check=True)
        require(not pin.stderr, "compiler identity warning")
        report["compiler_identity"] = pin.stdout
        manifest()
        closure = load_closure()
        graph = closure(SUITE / "king_away_consumer.bend", ENGINE)
        paths = {ENGINE / path for path in graph}
        paths.update(SUITE / name for name in ("qualify_king_away.py", "test_qualify_king_away.py"))
        paths.update(ENGINE / name for name in ("standalone/toolchain.json", "standalone/verify_compiler.js"))
        paths.update(SUITE / name for name in ("qualify.py", "qualify_castles.py"))
        paths.update(SUITE.parent / name for name in ("table_preservation/focused.py", "table_preservation/_validation.py"))

        def identities() -> dict[str, str]:
            return {str(path.relative_to(ENGINE)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(paths)}

        before = identities()
        report["source_sha256s"] = before
        write_report(args.report, report)
        helper = invoke(bun, compiler, SUITE / "KingAwayGuard.bend", 60)
        report["guard_helper"] = helper
        write_report(args.report, report)
        require(safe(helper), "guard helper must return exact safe success")
        ordinary = invoke(bun, compiler, SUITE / "KingAwayFilter.bend", 60)
        report["ordinary_helper"] = ordinary
        write_report(args.report, report)
        require(safe(ordinary), "ordinary scan/filter helper must return exact safe success")
        controls: list[dict] = []
        report["controls"] = controls
        mutations = [
            ("actual-home-king-guard-removed", "legal_probe/Chess.bend",
             "Bool.and(U64.test_bit(U64.and(get_kings(b), own), U32.to_nat(src)),",
             "Bool.and(True{},", "standalone/proofs/castle_emission/Flow.bend", "side"),
            ("actual-ordinary-scan-injects-castling-tag", "legal_probe/Chess.bend",
             "Con{Ply{src, dst, 0, flag}, tail}",
             "Con{Ply{src, dst, 0, 2}, tail}",
             "standalone/proofs/castle_chain/Scan.bend", "put"),
        ]
        for name, target, old, new, entry_name, location in mutations:
            with tempfile.TemporaryDirectory(prefix="king-away-control-") as directory:
                copied = Path(directory) / "engine"
                shutil.copytree(ENGINE, copied, symlinks=True)
                replace(copied / target, old, new)
                entry = copied / entry_name
                closure(entry, copied)
                result = invoke(bun, compiler, entry, 60)
                rejected = semantic_rejection(result, location)
                controls.append({"name": name, "target": target, "expected_location": location,
                                 "kind": "source semantic/refinement", "rejected": rejected, "result": result})
                write_report(args.report, report)
                require(rejected, "not intended semantic rejection: " + name)
        require(len(controls) == 2, "incomplete controls")
        consumer = invoke(bun, compiler, SUITE / "king_away_consumer.bend", 900)
        report["consumer"] = consumer
        write_report(args.report, report)
        require(safe(consumer), "complete consumer must return exact safe success")
        require(before == identities(), "source drift")
        after = subprocess.run(pin_command, capture_output=True, text=True, timeout=30, check=True)
        require(pin.stdout == after.stdout and not after.stderr, "compiler drift")
        report["king_away_gate"] = "PASS"
        report["accepted_candidate_definitions"] = 1
    except Exception as error:
        report["failure"] = type(error).__name__ + ": " + str(error)
        write_report(args.report, report)
        raise
    write_report(args.report, report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"source_sha256s", "controls", "consumer"}}, indent=2))


if __name__ == "__main__":
    main()
