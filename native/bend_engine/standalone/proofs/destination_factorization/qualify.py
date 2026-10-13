"""Bounded, fail-closed structural destination qualification; no inventory claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import tempfile
import time

from ..generator_contract.qualify import invoke, load_closure, replace
from ..generator_contract.qualify_castles import semantic_rejection
from ..table_preservation._validation import begin_report, require, write_report

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]


def safe_result(result: dict) -> bool:
    return (result["exit_code"] == 0 and not result["timed_out"] and
            result["raw_text"] == "All terms check.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    begin_report(args.report, "destination_factorization_gate")
    report: dict = {"destination_factorization_gate": "NOT_COMPLETED",
              "accepted_candidate_definitions": 0,
              "scope": "Structural identity through actual bit_squares, independent move-block encoding and full-Ply multiplicity, arbitrary-table scan_after pair equality. No independent bitboard coverage or legal_moves correctness.",
              "checks": [], "controls": [], "cpu_limit_seconds": 5400}
    cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_start = time.monotonic()
    original_affinity = os.sched_getaffinity(0)
    before = None
    try:
        # Both pin verification and all child checks inherit at most two CPUs.
        os.sched_setaffinity(0, set(sorted(original_affinity)[:2]))
        os.environ.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                           "MKL_NUM_THREADS": "2", "RAYON_NUM_THREADS": "2"})
        bun = os.environ.get("BUN", "bun")
        compiler = args.compiler.resolve()
        pin = subprocess.run([bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)],
                             capture_output=True, text=True, timeout=30, check=True)
        require(not pin.stderr, "compiler identity warning")
        report["compiler_identity"] = pin.stdout
        closure = load_closure()
        paths = set(closure(SUITE / "consumer.bend", ENGINE))
        paths.update(str(p.relative_to(ENGINE)) for p in SUITE.iterdir() if p.is_file())
        paths.update({"standalone/toolchain.json", "standalone/verify_compiler.js",
                      "standalone/proofs/generator_contract/qualify.py",
                      "standalone/proofs/generator_contract/qualify_castles.py",
                      "standalone/proofs/table_preservation/focused.py",
                      "standalone/proofs/table_preservation/_validation.py"})

        def identities() -> dict[str, str]:
            return {p: hashlib.sha256((ENGINE / p).read_bytes()).hexdigest() for p in sorted(paths)}

        before = identities()
        report["source_sha256s"] = before
        inherited_path = ENGINE.parents[1] / "docs/experiments/evidence/bend-king-away/source.json"
        inherited = json.loads(inherited_path.read_text())["source_sha256s"]
        report["inherited_receipt_sha256"] = hashlib.sha256(inherited_path.read_bytes()).hexdigest()
        report["inherited_inputs"] = len(inherited)
        report["inherited_inputs_unchanged"] = all(hashlib.sha256((ENGINE / p).read_bytes()).hexdigest() == h for p, h in inherited.items())
        require(len(inherited) == 249 and report["inherited_inputs_unchanged"], "inherited source identity changed")
        candidate = "\n".join(p.read_text() for p in SUITE.glob("*.bend"))
        require("\nlaw " not in candidate and "@unsafe" not in candidate and "?" not in candidate, "new unchecked declaration")
        consumer = (SUITE / "consumer.bend").read_text()
        for call in ("F.structural(n,empty,bb,src,pawn,ep,keys,tail)",
                     "E.destinations(n,empty,bb,src,pawn,ep,tail)",
                     "C.destinations(n,empty,bb,src,pawn,ep,query,tail)",
                     "Scan.exact(table,targets,src,pawn,ep,tail)"):
            require(call in consumer, "missing importing consumer call")

        def cpu_used() -> float:
            now = resource.getrusage(resource.RUSAGE_CHILDREN)
            return now.ru_utime + now.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime

        def check(entry: Path, seconds: int) -> dict:
            require(0 < seconds <= 900 and cpu_used() + 2 * seconds <= 5400, "CPU reservation failed")
            result = invoke(bun, compiler, entry, seconds)
            report["cpu_seconds"] = cpu_used()
            report["wall_seconds"] = time.monotonic() - wall_start
            return result

        for name in ("structural_consumer.bend", "consumer.bend"):
            result = check(SUITE / name, 180)
            report["checks"].append({"entry": name, "result": result})
            write_report(args.report, report)
            require(safe_result(result), "consumer did not safely check")
        mutations = [
            ("actual-ordinary-flag", "legal_probe/Chess.bend",
             "Con{Ply{src, dst, 0, flag}, tail}", "Con{Ply{src, dst, 0, 2}, tail}", "Emission.bend", "put"),
            ("actual-promotion-rank", "legal_probe/Chess.bend",
             "Bool.or(U32.is_zero(rank), U32.is_eq(rank, 7))",
             "Bool.or(U32.is_zero(rank), U32.is_eq(rank, 6))", "structural_consumer.bend", "structural"),
            ("actual-en-passant-tag", "legal_probe/Chess.bend",
             "Bool.to_u32(Bool.and(pawn, U32.is_eq(dst, ep_sq)))",
             "Bool.to_u32(Bool.and(pawn, U32.is_eq(dst, 65)))", "structural_consumer.bend", "structural"),
            ("actual-bit-square-key", "legal_probe/Chess.bend",
             "Con{U64.ctz(bb), acc}", "Con{U32.xor(U64.ctz(bb),1), acc}", "structural_consumer.bend", "structural"),
            ("consumer-proof-call-disconnected", "standalone/proofs/destination_factorization/consumer.bend",
             "F.structural(n,empty,bb,src,pawn,ep,keys,tail)", "{==}", "consumer.bend", "use_structural"),
        ]
        for name, target, old, new, entry, location in mutations:
            with tempfile.TemporaryDirectory(prefix="bend-destination-control-") as directory:
                copied = Path(directory) / "engine"
                shutil.copytree(ENGINE, copied, symlinks=True)
                replace(copied / target, old, new)
                checked_entry = copied / "standalone/proofs/destination_factorization" / entry
                closure(checked_entry, copied)
                result = check(checked_entry, 120)
                rejected = semantic_rejection(result, location)
                report["controls"].append({"name": name, "target": target, "expected_location": location,
                                           "rejected": rejected, "result": result})
                write_report(args.report, report)
                require(rejected, "control failed semantic rejection: " + name)
        final_pin = subprocess.run([bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)],
                                   capture_output=True, text=True, timeout=30, check=True)
        require(not final_pin.stderr and final_pin.stdout == pin.stdout, "compiler identity changed during qualification")
        report["compiler_identity_after"] = final_pin.stdout
        require(before == identities(), "source changed during qualification")
        report["source_unchanged"] = True
        report["destination_factorization_gate"] = "PASS"
        report["accepted_candidate_definitions"] = 4
    finally:
        os.sched_setaffinity(0, original_affinity)
        report["cpu_seconds"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime - cpu_start.ru_utime - cpu_start.ru_stime
        report["wall_seconds"] = time.monotonic() - wall_start
        write_report(args.report, report)


if __name__ == "__main__":
    main()
