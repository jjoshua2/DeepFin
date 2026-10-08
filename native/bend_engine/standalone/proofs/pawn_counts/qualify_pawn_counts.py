"""Fail-closed qualification of actual pawn-count preservation through generation and replay."""
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
BASE = "7cddd6019094421cb5e3624cf5544b58e1535609"
BASE_TREE = "a51d235d8bac8ec13a510d6e723988d0728c3081"
BRANCH = "proof/bend-pawn-counts-20261008"
PIN = "aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae"
FINGERPRINT = "d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4"
ENTRIES = ("consumer.bend", "Fixtures.bend", "Initialized.bend")
CONTROLS = (('omit-destination-from-joint-mask', 'Fusion.bend', 'mask_bool', 'Bool.or(Bool.or(s,c),Bool.or(t,r))', 'Bool.or(Bool.or(s,c),r)', 'Fusion.bend', ('target_clear',)), ('remove-owned-source', 'Actual.bend', 'planes', 'owned: {U64.test_bit(Chess.select_u64(turn,w,bl),src) == True{} : Bool}', 'owned: Unit', 'Actual.bend', ('planes',)), ('remove-source-deletion', 'Actual.bend', 'planes', 'erased: {U64.test_bit(remove,src) == True{} : Bool}', 'erased: Unit', 'Actual.bend', ('planes',)), ('color-only-instead-of-joint-ownership', 'Actual.bend', 'planes', 'I.owned(pawns,w,src,source_bound,pawn,owned)', 'owned', 'Actual.bend', ('planes',)), ('disconnect-pawn-decoder', 'Selector.bend', 'choice', 'put: {U32.is_eq(Bool.pick(U32,zero,Chess.piece(b,s),p),0) == True{} : Bool}', 'put: Unit', 'Selector.bend', ('choice',)), ('disconnect-source-pawn-certificate', 'Selector.bend', 'tagged', 'source(b,m,same)', '{==}', 'Selector.bend', ('tagged',)), ('disconnect-intersection-ownership', 'Intersection.bend', 'owned', 'left: {U64.test_bit(a,i) == True{} : Bool}', 'left: Unit', 'Intersection.bend', ('owned',)), ('wrong-active-color', 'Actual.bend', 'white_actual', 'A.white_actual(b,m)', 'A.black_actual(b,m)', 'Actual.bend', ('white_actual',)), ('wrong-inactive-color', 'Actual.bend', 'black_actual', 'A.black_actual(b,m)', 'A.white_actual(b,m)', 'Actual.bend', ('black_actual',)), ('remove-noncastle-exclusion', 'Actual.bend', 'frame', 'no: {U32.is_eq(A.flag(m),2) == False{} : Bool}', 'no: Unit', 'Actual.bend', ('frame',)), ('disconnect-pawn-count-change', 'Bounds.bend', 'preserved', 'change: {Nat.is_le(U.count(new),U.count(old)) == True{} : Bool}', 'change: Unit', 'Bounds.bend', ('preserved',)), ('disconnect-parent-pawn-bound', 'Bounds.bend', 'preserved', 'parent: {U32.is_le(U64.popcount(old),8) == True{} : Bool}', 'parent: Unit', 'Bounds.bend', ('preserved',)), ('claim-nine-within-eight', 'Bounds.bend', 'to_nat8', 'case 9n: Old.false_true({Nat.is_le(9n,8n) == True{} : Bool},known)', 'case 9n: {==}', 'Bounds.bend', ('to_nat8',)), ('disconnect-empty-flag', '../color_counts/Emission.bend', 'destinations', 'coherent: {empty == U64.is_zero(bb) : Bool}', 'coherent: {empty == empty : Bool}', '../color_counts/Emission.bend', ('destinations',)), ('disconnect-source-ownership', '../color_counts/Emission.bend', 'scan', '+set: {All.all_bits_set(keys,Chess.color(b,Chess.get_turn(b))) == True{} : Bool}', '+set: {keys == keys : List<&2,U32>}', '../color_counts/Emission.bend', ('scan',)), ('disconnect-zero-rights', '../color_counts/Emission.bend', 'guard_false', 'zero: {Chess.get_rights(b) == 0 : U32}', 'zero: Unit', '../color_counts/Emission.bend', ('guard_false',)), ('wrong-first-full-ply', 'Apply.bend', 'moved', 'Prior.board(g,m)', 'Prior.board(g,Chess.Ply{8,16,0,0})', 'Apply.bend', ('moved',)), ('disconnect-exact-table', 'Apply.bend', 'found', '(kept,moved(g,w,bl,Pos.find_move(xs,text),', '({==},moved(g,w,bl,Pos.find_move(xs,text),', 'Apply.bend', ('found',)), ('skip-replay-head', 'Replay.bend', 'tail', 'tail(c,w,bl,rest,Protocol.apply_one(text,maybe,O.pack(c)),Apply.one(c,maybe,text,w,bl,good))', 'tail(c,w,bl,rest,(O.pack(c),maybe),({==},good))', 'Replay.bend', ('tail',)), ('disconnect-replay-consumer', 'consumer.bend', 'lift', 'Replay.tail(c,S.white_limit(maybe),S.black_limit(maybe),xs,', 'Replay.tail(c,S.white_limit(maybe),S.black_limit(maybe),Nil{},', 'consumer.bend', ('lift',)), ('wrong-promotion-zero', 'Fixtures.bend', 'lower_ep_promotion', 'Con{Chess.Ply{8,0,1,0}', 'Con{Chess.Ply{8,0,0,0}', 'Fixtures.bend', ('lower_ep_promotion',)), ('wrong-promotion-EP-flag', 'Fixtures.bend', 'upper_ep_promotion', 'Con{Chess.Ply{48,63,1,0}', 'Con{Chess.Ply{48,63,1,1}', 'Fixtures.bend', ('upper_ep_promotion',)), ('wrong-nonpawn-EP-flag', 'Fixtures.bend', 'nonpawn_ep', 'Con{Chess.Ply{8,0,0,0}', 'Con{Chess.Ply{8,0,0,1}', 'Fixtures.bend', ('nonpawn_ep',)), ('wrong-ordinary-EP-flag', 'Fixtures.bend', 'ordinary_ep', 'Con{Chess.Ply{8,16,0,1}', 'Con{Chess.Ply{8,16,0,0}', 'Fixtures.bend', ('ordinary_ep',)), ('drop-duplicate-destination', 'Fixtures.bend', 'twelve_occurrences', '== 12n : Nat', '== 11n : Nat', 'Fixtures.bend', ('twelve_occurrences',)), ('drop-duplicate-command', 'Fixtures.bend', 'duplicates', 'Consumer.use_replay(a,Some{g},Con{text,Con{text,Nil{}}},good)', 'Consumer.use_replay(a,Some{g},Con{text,Nil{}},good)', 'Fixtures.bend', ('duplicates',)), ('wrong-EP-pawn-removal', 'Fixtures.bend', 'ep_removes_both', 'Chess.Ply{48,16,0,1}', 'Chess.Ply{48,16,0,0}', 'Fixtures.bend', ('ep_removes_both',)), ('wrong-raw-turn-color', 'Fixtures.bend', 'raw_turn_moves_black', '== 1n : Nat', '== 0n : Nat', 'Fixtures.bend', ('raw_turn_moves_black',)), ('claim-unowned-raw-move-safe', 'Fixtures.bend', 'raw_unowned_increases', '== False{} : Bool', '== True{} : Bool', 'Fixtures.bend', ('raw_unowned_increases',)), ('claim-missing-rook-castle-safe', 'Fixtures.bend', 'raw_castle_increases', '== False{} : Bool', '== True{} : Bool', 'Fixtures.bend', ('raw_castle_increases',)), ('remove-initialization', 'Initialized.bend', 'use_replay', 'built: {a == Init.run(d,seed,n,extra) : Array<U64>}', 'built: {a == a : Array<U64>}', 'Initialized.bend', ('use_replay',)), ('remove-depth17', 'Initialized.bend', 'use_replay', 'depth: {d == 17n : Nat}', 'depth: {d == d : Nat}', 'Initialized.bend', ('use_replay',)), ('remove-table-count128', 'Initialized.bend', 'use_replay', 'tables: {n == 128n : Nat}', 'tables: {n == n : Nat}', 'Initialized.bend', ('use_replay',)), ('remove-extra64', 'Initialized.bend', 'use_replay', 'full: {extra == 64n : Nat}', 'full: {extra == extra : Nat}', 'Initialized.bend', ('use_replay',)), ('disconnect-composed-pawn-counts', 'Initialized.bend', 'finish', 'Counts.use_replay(O.pack(c),maybe,Con{text,tail},', 'Counts.use_replay(O.pack(c),maybe,Nil{},', 'Initialized.bend', ('finish',)), ('ghost-destination-count-nine', 'Fixtures.bend', 'ghost_cleared', '== 8n : Nat', '== 9n : Nat', 'Fixtures.bend', ('ghost_cleared',)), ('overlapping-inactive-pawn-not-removed', 'Fixtures.bend', 'overlapping_inactive', '== 7n : Nat', '== 8n : Nat', 'Fixtures.bend', ('overlapping_inactive',)))
SUITE_PREFIX = "standalone/proofs/pawn_counts/"

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
              "scope": 'Actual legal full-Ply children under zero rights have nonincreasing Nat counts of both actual pawn/color intersections for arbitrary planes/raw turns/tables. Ordered Protocol.moves preserves zero rights, both actual U32 pawn/color intersection popcounts <=8, and Nat counts bounded by the initial counts from conditional parent zero-right/pawn-count bounds. Actual destination clearing, source-pawn decoder and joint ownership connect the mask algebra to make_move. Source ownership/ranges and destination noncastle flags are derived from exact-base actual producers; arbitrary duplicate-containing command and destination tails, exact table, full first-match Ply and actual scan_after pair/fuel/empty flag are retained. Initialized composition retains prior metadata, back-rank pawn exclusion, color bounds <=16 and all four original initialization certificates. No full Position.valid, king count, nonzero-right castle count preservation, replay acceptance or legal correctness claim.'}
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
        report["per_check_guards"] = []
        def guard(stage: str) -> dict:
            if {p: sha256(ENGINE / p) for p in sorted(paths)} != before:
                raise RuntimeError("source drift at " + stage)
            if {p: sha256(snapshot / p) for p in sorted(paths)} != before:
                raise RuntimeError("snapshot drift at " + stage)
            if sha256(args.checker_manifest) != report["compiler"]["manifest_sha256"]:
                raise RuntimeError("compiler manifest drift at " + stage)
            if any((compiler / r["path"]).is_symlink() or sha256(compiler / r["path"]) != r["sha256"] for r in files):
                raise RuntimeError("compiler file drift at " + stage)
            if git("rev-parse", "HEAD") != report["local_head"] or git("rev-parse", "HEAD^{tree}") != report["local_tree"]:
                raise RuntimeError("Git drift at " + stage)
            if report.get("qualified_sources_match_head") and git("status", "--porcelain", "--ignored"):
                raise RuntimeError("frozen worktree drift at " + stage)
            result = {"stage": stage, "observed_at_utc": now(), "source_count": len(before),
                      "source_hash_map_sha256": hashlib.sha256(json.dumps(before,sort_keys=True,separators=(",", ":")).encode()).hexdigest(),
                      "compiler_file_count": len(files), "compiler_fingerprint": fingerprint,
                      "git_head": report["local_head"], "git_tree": report["local_tree"]}
            report["per_check_guards"].append(result)
            write_report(args.report, report)
            return result
        def mutant_guard(root: Path, target: Path, digest: str) -> None:
            expected = dict(before)
            expected[str(target.relative_to(root))] = digest
            found = {}
            for p in root.rglob("*"):
                if p.is_symlink():
                    raise RuntimeError("mutant symlink")
                if p.is_file():
                    found[str(p.relative_to(root))] = sha256(p)
            if found != expected:
                raise RuntimeError("mutant differs outside exact declared replacement")
        write_report(args.report, report)
        for entry in args.entry or ENTRIES:
            name = Path(entry).stem
            guard("before-positive-" + name)
            result = run_check(name, snapshot / SUITE_PREFIX / entry,
                               compiler, bun, 86400, args.evidence_dir, "1,3")
            guard("after-positive-" + name)
            report["positive_checks"].append(result)
            write_report(args.report, report)
            if not result["passed"]:
                raise RuntimeError("positive entry failed: " + entry)
        if not args.positive_only:
            if args.entry:
                raise RuntimeError("controls require all positive entries")
            for name, filename, declaration, old, new, entry, locations in CONTROLS:
                mutant = args.evidence_dir / "mutants" / name
                shutil.copytree(snapshot, mutant, symlinks=False)
                target = (mutant / SUITE_PREFIX / filename).resolve()
                if not target.is_relative_to(mutant.resolve()):
                    raise RuntimeError("mutation target escapes isolated snapshot")
                baseline = sha256(target)
                mutate_declaration(target, declaration, old, new)
                mutated = sha256(target)
                mutant_guard(mutant,target,mutated)
                guard("before-control-" + name)
                result = run_check(name, (mutant / SUITE_PREFIX / entry).resolve(), compiler, bun, 86400, args.evidence_dir, "1,3")
                guard("after-control-" + name)
                mutant_guard(mutant,target,mutated)
                result.update(mutation_target=SUITE_PREFIX + filename,
                              mutation_declaration=declaration,
                              baseline_sha256=baseline, mutated_sha256=sha256(target),
                              mutation_anchor=old, replacement=new,
                              checked_entry=SUITE_PREFIX + entry,
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
