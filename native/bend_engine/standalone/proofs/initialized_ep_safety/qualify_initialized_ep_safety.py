"""Fail-closed qualification of initialized actual EP geometry."""
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
BASE = "16b974d71fe454a382bdbd0a9e7e934a941d4191"
BASE_TREE = "3fe688e89eeb6d8fe919d784ffa2fc39bd4b5eb8"
BRANCH = "proof/bend-initialized-ep-safety-20261006"
PIN = "aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae"
FINGERPRINT = "d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4"
ENTRIES = ("Flag.bend", "Frame.bend", "Finish.bend", "Check.bend", "consumer.bend", "Witnesses.bend", "Boundary.bend")
CONTROLS: tuple[tuple[str, str, str, str, str, tuple[str, ...]], ...] = (('disconnect-frame', 'Frame.bend', 'singleton', 'kept,', '{==},', ('singleton',)), ('disconnect-initialized-query', 'Check.bend', 'selected', 'Composition.checked(d,seed,n,extra,Chess.make_move(b,m),q,side,\n      depth,tables,full,bound,Frame.singleton(b,m,q,side,turn,one,frame))', '{==}', ('selected',)), ('query-opposite-side', 'Check.bend', 'selected', 'Chess.make_move(b,m),q,side,\n      depth', 'Chess.make_move(b,m),q,Bool.not(side),\n      depth', ('selected',)), ('disconnect-accepted-pair', 'Geometry.bend', 'actual', 'Accepted.legal(Init.run(d,seed,n,extra),b,m,accepted,tag)', '{==}', ('actual',)), ('disconnect-king-frame', 'Geometry.bend', 'actual', 'EP.legal_member(Init.run(d,seed,n,extra),b,m,good,coherent,origin,Flag.required(m,tag))', '({==},{==})', ('actual',)), ('wrong-EP-flag', 'Geometry.bend', 'actual', '+tag: {C.flag(m) == 1 : U32}', '+tag: {C.flag(m) == 0 : U32}', ('actual',)), ('remove-board-premise', 'Geometry.bend', 'actual', 'good: {Board.valid(b) == True{} : Bool}', 'good: {True{} == True{} : Bool}', ('actual',)), ('remove-EP-coherence', 'Geometry.bend', 'actual', 'coherent: {Position.valid_ep(Chess.get_ep(b),b) == True{} : Bool}', 'coherent: {True{} == True{} : Bool}', ('actual',)), ('remove-singleton', 'Geometry.bend', 'actual', 'one: {F.plane(b,Bool.to_u32(side)) == U64.bit(q) : U64}', 'one: {U64.bit(q) == U64.bit(q) : U64}', ('actual',)), ('remove-square-range', 'Check.bend', 'selected', 'bound: {Nat.is_lt(q,64n) == True{} : Bool}', 'bound: {True{} == True{} : Bool}', ('selected',)), ('wrong-table-depth', 'Check.bend', 'selected', 'depth: {d == 17n : Nat}', 'depth: {d == 16n : Nat}', ('selected',)), ('wrong-table-count', 'Check.bend', 'selected', 'tables: {n == 128n : Nat}', 'tables: {n == 64n : Nat}', ('selected',)), ('wrong-extras-count', 'Check.bend', 'selected', 'full: {extra == 64n : Nat}', 'full: {extra == 63n : Nat}', ('selected',)), ('disconnect-public-consumer', 'consumer.bend', 'use_generated', 'lift(a,d,seed,n,extra,b,m,q,side,initialized,depth,tables,full,good,coherent,turn,bound,one)(here)(tag)', '{==}', ('use_generated',)), ('remove-exact-initialization', 'consumer.bend', 'lift', 'initialized: {a == Init.run(d,seed,n,extra) : Array<U64>}', 'initialized: {a == a : Array<U64>}', ('lift',)), ('remove-original-turn', 'Geometry.bend', 'actual', 'turn: {Chess.get_turn(b) == Bool.to_u32(side) : U32}', 'turn: {Bool.to_u32(side) == Bool.to_u32(side) : U32}', ('actual',)), ('wrong-full-Ply-promotion', 'consumer.bend', 'lift', 'Safety.actual(d,seed,n,extra,b,m,q,side,', 'Safety.actual(d,seed,n,extra,b,Chess.Ply{0,0,1,1},q,side,', ('lift',)), ('erase-discovered-attack', 'Witnesses.bend', 'discovered_child_attacked', '== True{} : Bool', '== False{} : Bool', ('discovered_child_attacked',)), ('claim-destination-attacked', 'Witnesses.bend', 'wrong_square_clear', '== False{} : Bool', '== True{} : Bool', ('wrong_square_clear',)), ('wrong-geometric-opponent', 'Witnesses.bend', 'discovered_child_attacked', '32n,False{}', '32n,True{}', ('discovered_child_attacked',)), ('reverse-old-turn-transport', 'Finish.bend', 'old_turn', '%turn :', '%Equal.sym(U32,Chess.get_turn(b),Bool.to_u32(side),turn) :', ('old_turn',)), ('change-query-return-table', 'Finish.bend', 'geometry', '(a,G.attacked(child,q,Bool.not(side))) : Array<U64> & Bool}', '(ALeaf{U64.zero()},G.attacked(child,q,Bool.not(side))) : Array<U64> & Bool}', ('geometry',)), ('disconnect-exclusion-consumer', 'Boundary.bend', 'exclude', 'Safety.use_generated(a,d,seed,n,extra,b,m,32n,True{},\n      initialized,depth,tables,full,good,coherent,turn,{==},one,here,tag)', '{==}', ('exclude',)), ('disconnect-flag-observer-bridge', 'Flag.bend', 'required', 'same(m)', '{==}', ('required',)), ('remove-discovered-fixture-identity', 'Boundary.bend', 'certificates', 'same: {b == B.discovered() : Chess.Board}', 'same: {b == b : Chess.Board}', ('certificates',)), ('wrong-fixture-full-Ply-promotion', 'Boundary.bend', 'certificates', 'record: {m == Chess.Ply{36,43,0,1} : Chess.Ply}', 'record: {m == Chess.Ply{36,43,1,1} : Chess.Ply}', ('certificates',)))
SUITE_PREFIX = "standalone/proofs/initialized_ep_safety/"

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
              "scope": "Exact initialized-array full-Ply actual legal membership and flag1, valid parent/EP metadata, canonical old turn and bounded singleton original king imply independent target-centred child geometry False. Child singleton and actual whole-pair False are derived. No forward-ray reciprocity or full physical legality claim."}
    try:
        if not {1, 3}.issubset(affinity):
            raise RuntimeError("CPUs 1 and 3 unavailable")
        os.sched_setaffinity(0, {1, 3})
        os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                          MKL_NUM_THREADS="2", RAYON_NUM_THREADS="2",
                          BUN_JSC_jitMemoryReservationSize="67108864", BUN_JSC_forceRAMSize="6442450944", BUN_JSC_validateOptions="true", DO_NOT_TRACK="1",
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
            "selected_cpus": [1, 3], "threads": 2, "runtime_env": {"BUN_JSC_jitMemoryReservationSize": "67108864", "BUN_JSC_forceRAMSize": "6442450944", "BUN_JSC_validateOptions": "true", "DO_NOT_TRACK": "1"},
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
