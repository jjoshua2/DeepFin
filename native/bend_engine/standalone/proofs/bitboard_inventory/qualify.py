"""Bounded, fail-closed qualification of singleton inventory and actual low-bit count step."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

from ..generator_contract.qualify import load_closure, replace, invoke
from ..generator_contract.qualify_castles import semantic_rejection
from ..table_preservation._validation import begin_report, require, write_report

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
REPO = ENGINE.parents[1]
INHERITED_GATE = REPO / "docs/experiments/evidence/bend-bitboard-inventory/inherited-destination-factorization.json"


def safe_result(result: dict) -> bool:
    return (result["exit_code"] == 0 and not result["timed_out"]
            and result["raw_text"] == "All terms check.\n")


def manifest() -> None:
    inventory = (SUITE / "Inventory.bend").read_text()
    proof = (SUITE / "OneBit.bend").read_text()
    consumer = (SUITE / "consumer.bend").read_text()
    clear_step = (SUITE / "ClearStep.bend").read_text()
    names = ["expand_one", "scan_one_bit", "empty_indices", "scan_empty_targets",
             "scan_step_onehot", "scan_step_empty", "scan_one_source", "legal_moves_one_source"]
    import re
    require(re.findall(r"^def (\w+)\(", inventory, re.MULTILINE) == ["singleton_indices"],
            "complete inventory theorem inventory changed")
    require(re.findall(r"^def (\w+)\(", proof, re.MULTILINE) == names,
            "complete theorem inventory changed")
    clear_names = ["one", "clear_model", "and_self", "sub_one", "clear_matches", "is_zero_word",
                   "count_delta", "one_word", "sub_word", "clear_to_and", "clear_actual",
                   "clear_lsb_count_step", "clear_lsb_popcount_step"]
    require(re.findall(r"^def (\w+)\(", clear_step, re.MULTILINE) == clear_names,
            "complete clear-step theorem inventory changed")
    require(not re.search(r"^law\s", clear_step, re.MULTILINE), "unproved clear-step law declaration")
    require("@unsafe" not in clear_step and "?" not in clear_step, "unchecked clear-step shortcut")
    for actual in ("U64.clear_lsb(a)", "U64.popcount(a)", "Word.count(64n,U64.to_word(a))",
                   "Borrow.u64_sub(a,U64.one())", "Step.and_word(a,U64.sub(a,U64.one()))"):
        require(actual in clear_step, "actual clear/popcount proof body disconnected: " + actual)
    require(not re.search(r"^law\s", inventory + proof, re.MULTILINE), "unproved law declaration")
    require("@unsafe" not in inventory + proof and "?" not in inventory + proof, "unchecked shortcut")
    calls = [
        "Inventory.singleton_indices(sq,e)",
        "OneBit.scan_one_bit(table,sq,e,src,pawn,ep,tail)",
        "OneBit.scan_step_onehot(b,src,table,tail,sq,bound,target)",
        "OneBit.scan_step_empty(b,src,table,tail,target)",
        "OneBit.scan_one_source(b,table,src_sq,src_bound,source,dst_sq,dst_bound,target)",
        "OneBit.legal_moves_one_source(b,table,src_sq,src_bound,source,dst_sq,dst_bound,target)",
        "ClearStep.clear_lsb_popcount_step(a)",
    ]
    for call in calls:
        require(call in consumer, "missing importing consumer call: " + call)
    require("Chess.legal_moves(table,b) == Chess.filter_prepare(b,Chess.castle_side(b,False{},Chess.castle_side(b,True{}," in proof,
            "actual optimized legal_moves pipeline disconnected")
    require("Chess.scan_after(src,pawn,ep,tail,(table,U64.bit(sq)))" in proof,
            "actual scan_after disconnected")
    require("Chess.bit_squares(64n,U64.is_zero(U64.bit(sq)),U64.bit(sq),Nil{})" in proof,
            "actual bounded singleton inventory disconnected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    begin_report(args.report, "bitboard_inventory_gate")
    report: dict = {
        "bitboard_inventory_gate": "NOT_COMPLETED",
        "accepted_candidate_definitions": 0,
        "scope": (
            "Actual Chess.bit_squares exact output for zero and each of the 64 bounded singleton U64.bit(sq) masks; "
            "actual scan_after, scan_step, scan and legal_moves consumer equalities for one-hot source/destination "
            "frontiers. Exact Array<U64> and Ply-tail preservation is stated for scan-stage equalities; the legal_moves "
            "equation retains its actual table-threading pipeline from an arbitrary input array. The separate all-U64 "
            "clear_lsb/popcount step is proved, but arbitrary-mask bit_squares membership/cardinality, ctz membership, "
            "piece-target geometry, attack answers and full legal-move correctness are not."
        ),
        "checks": [], "controls": [], "cpu_limit_seconds": 5400,
    }
    cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_start = time.monotonic()
    original_affinity = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, set(sorted(original_affinity)[:2]))
        os.environ.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                           "MKL_NUM_THREADS": "2", "RAYON_NUM_THREADS": "2",
                           "BEND_NO_TELEMETRY": "1", "TERM": "dumb"})
        bun = os.environ.get("BUN") or shutil.which("bun")
        if not bun:
            raise RuntimeError("bun executable was not found")
        compiler = args.compiler.resolve()
        pin_command = [bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)]
        pin = subprocess.run(pin_command, capture_output=True, text=True, timeout=30, check=True)
        require(not pin.stderr, "compiler identity warning")
        report["compiler_identity"] = pin.stdout
        require("aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae" in pin.stdout,
                "unexpected compiler commit")
        closure = load_closure()
        paths = set(closure(SUITE / "consumer.bend", ENGINE))
        paths.update(str(path.relative_to(ENGINE)) for path in SUITE.iterdir() if path.is_file())
        paths.update({
            "standalone/toolchain.json", "standalone/verify_compiler.js",
            "standalone/proofs/generator_contract/qualify.py",
            "standalone/proofs/generator_contract/qualify_castles.py",
            "standalone/proofs/table_preservation/focused.py",
            "standalone/proofs/table_preservation/_validation.py",
            "standalone/proofs/destination_factorization/qualify.py",
        })

        def identities() -> dict[str, str]:
            return {name: hashlib.sha256((ENGINE / name).read_bytes()).hexdigest()
                    for name in sorted(paths)}

        before = identities()
        report["source_sha256s"] = before
        report["source_identity_count"] = len(before)
        prior_path = REPO / "docs/experiments/evidence/bend-king-away/source.json"
        prior = json.loads(prior_path.read_text())
        inherited = prior["source_sha256s"]
        report["prior_king_away_receipt_sha256"] = hashlib.sha256(prior_path.read_bytes()).hexdigest()
        report["prior_input_count"] = len(inherited)
        report["prior_inputs_unchanged"] = all(
            hashlib.sha256((ENGINE / name).read_bytes()).hexdigest() == digest
            for name, digest in inherited.items()
        )
        require(len(inherited) == 249 and report["prior_inputs_unchanged"],
                "prior checked source identity changed")
        manifest()

        inherited_gate = subprocess.run(
            [sys.executable, "-m", "native.bend_engine.standalone.proofs.destination_factorization.qualify",
             str(compiler), "--report", str(INHERITED_GATE)],
            cwd=REPO, capture_output=True, text=True, timeout=900, check=False,
        )
        report["inherited_destination_gate"] = {
            "exit_code": inherited_gate.returncode,
            "receipt": str(INHERITED_GATE),
        }
        require(inherited_gate.returncode == 0, "inherited destination gate failed")
        prior_run = json.loads(INHERITED_GATE.read_text())
        require(prior_run.get("destination_factorization_gate") == "PASS"
                and prior_run.get("accepted_candidate_definitions") == 4
                and len(prior_run.get("controls", [])) == 5
                and all(control.get("rejected") for control in prior_run["controls"]),
                "inherited gate did not recheck all four contracts and five controls")
        report["inherited_destination_gate"]["receipt_sha256"] = hashlib.sha256(
            INHERITED_GATE.read_bytes()
        ).hexdigest()
        write_report(args.report, report)

        def cpu_used() -> float:
            now = resource.getrusage(resource.RUSAGE_CHILDREN)
            return now.ru_utime + now.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime

        def check(entry: Path, seconds: int) -> dict:
            require(0 < seconds <= 900 and cpu_used() + 2 * seconds <= 5400,
                    "CPU reservation failed")
            result = invoke(bun, compiler, entry, seconds)
            report["cpu_seconds"] = cpu_used()
            report["wall_seconds"] = time.monotonic() - wall_start
            return result

        for entry_name in ("Inventory.bend", "OneBit.bend", "ClearStep.bend", "consumer.bend"):
            result = check(SUITE / entry_name, 180)
            report["checks"].append({"entry": entry_name, "safe": safe_result(result), "result": result})
            write_report(args.report, report)
            require(safe_result(result), "proof source did not return exact safe checker output")

        mutations = [
            ("actual-bit-square-key", "legal_probe/Chess.bend",
             "Con{U64.ctz(bb), acc}", "Con{U32.xor(U64.ctz(bb),1), acc}", "Inventory.bend", "singleton_indices"),
            ("actual-destination-budget", "legal_probe/Chess.bend",
             "destinations(64n, U64.is_zero(targets), targets, src, pawn, ep_sq, acc)",
             "destinations(63n, U64.is_zero(targets), targets, src, pawn, ep_sq, acc)", "consumer.bend", "cells"),
            ("actual-scan-step-target-source", "legal_probe/Chess.bend",
             "piece_targets(pawn, table, b, src)", "piece_targets(False{}, table, b, src)", "consumer.bend", "scan_step_onehot"),
            ("actual-clear-lsb-count-step", "standalone/proofs/bitboard_inventory/consumer.bend",
             "U64.to_word(U64.clear_lsb(a))", "U64.to_word(U64.clear_bit(a,1n))",
             "consumer.bend", "use_clear_lsb_popcount_step"),
            ("consumer-proof-call-disconnected", "standalone/proofs/bitboard_inventory/consumer.bend",
             "OneBit.legal_moves_one_source(b,table,src_sq,src_bound,source,dst_sq,dst_bound,target)",
             "{==}", "consumer.bend", "use_actual_legal_moves_one_source"),
        ]
        for name, target, old, new, entry_name, location in mutations:
            with tempfile.TemporaryDirectory(prefix="bend-bitboard-control-") as directory:
                copied = Path(directory) / "engine"
                shutil.copytree(ENGINE, copied, symlinks=True)
                replace(copied / target, old, new)
                checked_entry = copied / "standalone/proofs/bitboard_inventory" / entry_name
                closure(checked_entry, copied)
                result = check(checked_entry, 120)
                rejected = semantic_rejection(result, location)
                report["controls"].append({
                    "name": name, "target": target, "expected_location": location,
                    "rejected": rejected, "result": result,
                })
                write_report(args.report, report)
                require(rejected, "control failed intended semantic rejection: " + name)

        require(len(report["controls"]) == 5, "incomplete bitboard control inventory")
        after = subprocess.run(pin_command, capture_output=True, text=True, timeout=30, check=True)
        require(not after.stderr and pin.stdout == after.stdout, "compiler identity drift")
        report["compiler_identity_after"] = after.stdout
        require(before == identities(), "source changed during qualification")
        report["source_unchanged"] = True
        report["cpu_seconds"] = cpu_used()
        require(report["cpu_seconds"] <= 5400, "task CPU budget exceeded")
        report["wall_seconds"] = time.monotonic() - wall_start
        report["bitboard_inventory_gate"] = "PASS"
        report["accepted_candidate_definitions"] = 10
    except Exception as error:
        report["failure"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        os.sched_setaffinity(0, original_affinity)
        report["cpu_seconds"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime - cpu_start.ru_utime - cpu_start.ru_stime
        report["wall_seconds"] = time.monotonic() - wall_start
        write_report(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
