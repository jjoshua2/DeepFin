"""Fail-closed qualification of exact-array actual EP acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ..generator_contract.qualify import load_closure
from ..fast_full_equivalence.qualify_fast_full import (
    MEMORY_CAP, OUTPUT_CAP, mutate_declaration, now, rejected_at,
    run_check, sha256, write_report,
)

sys.dont_write_bytecode = True
SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
PROJECT = ENGINE.parents[1]
BASE = "049bd4afca105af20bd07b1373070c743e6150b7"
BASE_TREE = "7a8bbc221b738d64bd81433c8b9a8114034745b7"
BRANCH = "proof/bend-accepted-ep-20261006"
PIN = "aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae"
FINGERPRINT = "d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4"
ENTRIES = ("consumer.bend", "Witnesses.bend", "Boundary.bend", "TailWitnesses.bend")
CONTROLS: tuple[tuple[str, str, str, str, str, tuple[str, ...]], ...] = (
    ("force-wrong-flag", "Accept.bend", "required", "tag: {C.flag(m) == 1 : U32}", "tag: {C.flag(m) == 0 : U32}", ("required",)),
    ("disconnect-fast-check-requirement", "Accept.bend", "chosen", "required(sensitive,m,tag)", "{==}", ("chosen",)),
    ("disconnect-prepared-filter", "Accept.bend", "prepared", "Filter.prepare(c,b,xs)", "{==}", ("prepared",)),
    ("disconnect-legal-prefilter", "Accept.bend", "known", "G.prepared_equal(c,b)", "{==}", ("known",)),
    ("disconnect-whole-result-pair", "Public.bend", "legal_lift", "Pair.false_check(c,b,m,Accept.known(c,b,m,here,tag))", "Accept.known(c,b,m,here,tag)", ("legal_lift",)),
    ("disconnect-public-consumer", "consumer.bend", "use_generated", "P.legal(a,b,m,here,tag)", "{==}", ("use_generated",)),
    ("query-child-turn", "consumer.bend", "use_generated", "Chess.get_turn(b)) == (a,False{})", "Chess.get_turn(Chess.make_move(b,m))) == (a,False{})", ("use_generated",)),
    ("disconnect-full-Ply", "consumer.bend", "use_generated", "P.legal(a,b,m,here,tag)", "P.legal(a,b,Chess.Ply{0,0,0,0},here,tag)", ("use_generated",)),
    ("wrong-original-table", "Witnesses.bend", "white_pair", "(ALeaf{U64.bit(43n)},False{})", "(ALeaf{U64.bit(44n)},False{})", ("white_pair",)),
    ("wrong-promotion-member", "Witnesses.bend", "white_member", "C.Ply{36,43,0,1}", "C.Ply{36,43,1,1}", ("white_member",)),
    ("wrong-EP-flag-member", "Witnesses.bend", "white_member", "C.Ply{36,43,0,1}", "C.Ply{36,43,0,0}", ("white_member",)),
    ("erase-geometric-rook-hit", "Boundary.bend", "exposed_rook", "U64.bit(39n) : U64", "U64.zero() : U64", ("exposed_rook",)),
    ("claim-moving-side-checked-false", "Boundary.bend", "moving_side_checked", "== True{} : Bool", "== False{} : Bool", ("moving_side_checked",)),
    ("erase-prepared-duplicates", "TailWitnesses.bend", "prepared_duplicates", "== 2n : Nat", "== 1n : Nat", ("prepared_duplicates",)),
    ("change-prepared-table", "TailWitnesses.bend", "prepared_table", "== ALeaf{U64.bit(43n)}", "== ALeaf{U64.bit(44n)}", ("prepared_table",)),
    ("invent-wrong-promotion", "TailWitnesses.bend", "wrong_promotion_absent", "== 0n : Nat", "== 1n : Nat", ("wrong_promotion_absent",)),
    ("invent-wrong-flag", "TailWitnesses.bend", "wrong_flag_absent", "== 0n : Nat", "== 1n : Nat", ("wrong_flag_absent",)),
    ("retain-rejected-duplicates", "TailWitnesses.bend", "rejected_duplicates", "== 0n : Nat", "== 2n : Nat", ("rejected_duplicates",)),
    ("erase-raw-tail-duplicates", "TailWitnesses.bend", "raw_tail_duplicates", "== 2n : Nat", "== 1n : Nat", ("raw_tail_duplicates",)),
    ("assume-unchecked-tail-safe", "TailWitnesses.bend", "raw_tail_check_true", "True{}) : Array", "False{}) : Array", ("raw_tail_check_true",)),
)
SUITE_PREFIX = "standalone/proofs/accepted_ep/"

def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT), *args],
                                   text=True, timeout=30).strip()

def blobs(ref: str) -> dict[str, str]:
    result = {}
    for line in git("ls-tree", "-r", ref).splitlines():
        metadata, path = line.split("\t", 1)
        _mode, kind, digest = metadata.split()
        if kind == "blob":
            result[path] = digest
    return result

def blob(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()

def strict(result: dict, locations: tuple[str, ...]) -> bool:
    inference = ("cannot infer", "expected : an annotated term", "non-inferrable",
                 "non-inferable", "not inferrable", "not inferable")
    return rejected_at(result, tuple(locations)) and not any(
        word in result["raw_text"].lower() for word in inference)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--checker-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--entry", choices=ENTRIES, action="append")
    args = parser.parse_args()
    if args.report.exists() or args.evidence_dir.exists():
        raise RuntimeError("requires fresh report/evidence paths")
    args.evidence_dir.mkdir(parents=True)
    affinity = os.sched_getaffinity(0)
    report: dict = {"gate": "NOT_COMPLETED", "base": BASE, "base_tree": BASE_TREE,
              "branch": BRANCH, "positive_checks": [], "negative_controls": [],
              "scope": "Exact full-Ply actual legal/prepared membership plus flag1 implies original-array in_check child on old moving side returns (array,False). Arbitrary arrays/Boards/U32 fields/duplicate candidates; prepare starts Nil. No geometric king-safety or unchecked-tail acceptance claim."}
    try:
        if not {1, 3}.issubset(affinity):
            raise RuntimeError("CPUs 1 and 3 unavailable")
        os.sched_setaffinity(0, {1, 3})
        os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                          MKL_NUM_THREADS="2", RAYON_NUM_THREADS="2",
                          BEND_NO_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1")
        compiler = args.compiler.resolve()
        bun = os.environ.get("BUN", str(Path.home() / ".bun/bin/bun"))
        version = subprocess.check_output([bun, "--version"], text=True, timeout=15).strip()
        pin_cmd = [bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)]
        pin = subprocess.run(pin_cmd, capture_output=True, text=True, check=True, timeout=30)
        manifest = json.loads(args.checker_manifest.read_text())
        files = manifest["files"]
        fingerprint = hashlib.sha256(b"".join(
            row["path"].encode() + b"\0" + bytes.fromhex(row["sha256"])
            for row in sorted(files, key=lambda row: row["path"]))).hexdigest()
        if (version != "1.4.2" or pin.stderr or PIN not in pin.stdout
                or FINGERPRINT not in pin.stdout or len(files) != 84
                or fingerprint != FINGERPRINT
                or manifest["source_fingerprint_sha256"] != fingerprint
                or any(not (compiler / r["path"]).is_file()
                       or sha256(compiler / r["path"]) != r["sha256"] for r in files)):
            raise RuntimeError("pinned Bend/Bun/84-file identity mismatch")
        report["compiler"] = {"command": pin_cmd, "identity": pin.stdout,
                              "bun_version": version, "file_count": 84,
                              "fingerprint": fingerprint,
                              "manifest_sha256": sha256(args.checker_manifest)}
        closure = load_closure()
        graph = {entry: sorted(closure(SUITE / entry, ENGINE)) for entry in ENTRIES}
        paths = set().union(*map(set, graph.values()))
        paths.update(str(p.relative_to(ENGINE)) for p in SUITE.iterdir() if p.is_file())
        support = {"standalone/proofs/table_preservation/focused.py"}
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename:
                p = Path(filename).resolve()
                if p.suffix == ".py" and p.is_relative_to(ENGINE):
                    support.add(str(p.relative_to(ENGINE)))
        paths.update(support)
        paths.update({"standalone/toolchain.json", "standalone/verify_compiler.js"})
        before = {p: sha256(ENGINE / p) for p in sorted(paths)}
        report["source_sha256s"] = before
        report["support_sha256s"] = {p: before[p] for p in sorted(support)}
        report["entry_closures"] = graph
        report["local_head"] = git("rev-parse", "HEAD")
        report["local_tree"] = git("rev-parse", "HEAD^{tree}")
        if git("rev-parse", BASE + "^{tree}") != BASE_TREE:
            raise RuntimeError("base tree differs")
        if git("branch", "--show-current") != BRANCH:
            raise RuntimeError("branch differs")
        base_blobs = blobs(BASE)
        reused = {}
        for p in sorted(paths):
            if p.startswith(SUITE_PREFIX):
                continue
            digest = blob(ENGINE / p)
            if base_blobs.get("native/bend_engine/" + p) != digest:
                raise RuntimeError("dependency differs from exact base: " + p)
            reused[p] = digest
        report["reused_dependency_git_blobs"] = reused
        if report["local_head"] != BASE:
            if git("status", "--porcelain", "--ignored"):
                raise RuntimeError("frozen qualification requires clean worktree")
            expected = {"native/bend_engine/" + p for p in paths if p.startswith(SUITE_PREFIX)}
            if set(git("diff", "--name-only", "--no-renames", BASE, "HEAD").splitlines()) != expected:
                raise RuntimeError("delta outside new suite")
            head_blobs = blobs("HEAD")
            if any(head_blobs.get("native/bend_engine/" + p) != blob(ENGINE / p) for p in paths):
                raise RuntimeError("qualified file differs from frozen Git head")
            report["qualified_sources_match_head"] = True
        else:
            report["input_mode"] = "base-overlay"
        report["resource_limits"] = {
            "positive_wall_seconds": 86400, "negative_wall_seconds": 86400,
            "selected_cpus": [1, 3], "threads": 2,
            "address_space_cap_bytes": MEMORY_CAP, "max_rss_kib": MEMORY_CAP // 1024,
            "stdout_stderr_file_cap_bytes": OUTPUT_CAP}
        snapshot = args.evidence_dir / "source-snapshot"
        for p in sorted(paths):
            dest = snapshot / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ENGINE / p, dest)
        write_report(args.report, report)
        for entry in args.entry or ENTRIES:
            name = Path(entry).stem
            result = run_check(name, snapshot / SUITE_PREFIX / entry,
                               compiler, bun, 86400, args.evidence_dir, "1,3")
            report["positive_checks"].append(result)
            write_report(args.report, report)
            if not result["passed"]:
                raise RuntimeError("positive entry failed: " + entry)
        if not args.positive_only:
            if args.entry:
                raise RuntimeError("controls require all positive entries")
            for name, filename, declaration, old, new, locations in CONTROLS:
                mutant = args.evidence_dir / "mutants" / name
                shutil.copytree(snapshot, mutant, symlinks=False)
                target = mutant / SUITE_PREFIX / filename
                baseline = sha256(target)
                mutate_declaration(target, declaration, old, new)
                result = run_check(name, target, compiler, bun, 86400, args.evidence_dir, "1,3")
                result.update(mutation_target=SUITE_PREFIX + filename,
                              mutation_declaration=declaration,
                              baseline_sha256=baseline, mutated_sha256=sha256(target),
                              mutation_anchor=old, replacement=new,
                              checked_entry=SUITE_PREFIX + filename,
                              expected_locations=locations,
                              rejected_at_expected_obligation=strict(result, locations))
                report["negative_controls"].append(result)
                write_report(args.report, report)
                if not result["rejected_at_expected_obligation"]:
                    raise RuntimeError("control failed strict rejection: " + name)
        if {p: sha256(ENGINE / p) for p in sorted(paths)} != before:
            raise RuntimeError("source drift during qualification")
        after = subprocess.run(pin_cmd, capture_output=True, text=True, check=True, timeout=30)
        if after.stderr or after.stdout != pin.stdout:
            raise RuntimeError("compiler drift during qualification")
        report["source_unchanged"] = True
        report["compiler_unchanged"] = True
        report["evidence_sha256s"] = {
            str(p.relative_to(args.evidence_dir)): sha256(p)
            for p in sorted(args.evidence_dir.rglob("*")) if p.is_file()}
        report["gate"] = "PASS"
        report["qualification_complete"] = not args.positive_only
    except Exception as failure:
        report["gate"] = "FAIL"
        report["failure"] = repr(failure)
        raise
    finally:
        os.sched_setaffinity(0, affinity)
        report["finished_at_utc"] = now()
        write_report(args.report, report)

if __name__ == "__main__":
    main()
