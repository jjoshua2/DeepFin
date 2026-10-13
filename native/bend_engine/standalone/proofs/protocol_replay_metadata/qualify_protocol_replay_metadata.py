"""Fail-closed qualification of actual nonempty protocol replay ready metadata and exact tables."""
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
BASE = "d8d5cd8caea9923296cb0be096d6d503c5fdf1a9"
BASE_TREE = "2fdb4e270544ade15351dce64eefe9cbf31105ba"
BRANCH = "proof/bend-protocol-replay-metadata-20261007"
PIN = "aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae"
FINGERPRINT = "d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4"
ENTRIES = ("Apply.bend", "Fixtures.bend")
CONTROLS = (('disconnect-generated-bounds', 'Apply.bend', 'legal', 'text,r,kept,rows,canonical,fields)', 'text,r,kept,rows,canonical,Unit{})', 'Apply.bend', ('legal', 'Apply.legal')), ('remove-parent-canonical', 'Apply.bend', 'legal', 'canonical: Turn.canonical(Chess.get_turn(Pos.board(g)))', 'canonical: {True{} == True{} : Bool}', 'Apply.bend', ('legal', 'Apply.legal')), ('wrong-child-side', 'Apply.bend', 'moved', 'converted(Chess.make_move(Pos.board(g),m),Bool.not(white),child)', 'converted(Chess.make_move(Pos.board(g),m),white,child)', 'Apply.bend', ('moved', 'Apply.moved')), ('force-ep-sentinel', 'Spec.bend', 'ready', 'Range.ep(Chess.get_ep(b))', '{Chess.get_ep(b) == 64 : U32}', 'Apply.bend', ('converted', 'Apply.converted')), ('skip-first-application', 'Replay.bend', 'first', 'tail(c,rest,Protocol.apply_one(text,maybe,O.pack(c)),Apply.one(c,maybe,text,good))', 'tail(c,rest,(O.pack(c),maybe),({==},good))', 'Replay.bend', ('first', 'Replay.first')), ('skip-tail-application', 'Replay.bend', 'tail', 'Apply.one(c,maybe,text,S.weaken(maybe,good))', '({==},good)', 'Replay.bend', ('tail', 'Replay.tail')), ('wrong-first-ep', 'Fixtures.bend', 'first_ep', '== 20 : U32', '== 64 : U32', 'Fixtures.bend', ('first_ep',)), ('pretend-empty-repaired', 'Fixtures.bend', 'empty_keeps_bad_ep', '== 65 : U32', '== 64 : U32', 'Fixtures.bend', ('empty_keeps_bad_ep',)), ('wrong-two-move-parity', 'Fixtures.bend', 'second_turn', '== 1 : U32', '== 0 : U32', 'Fixtures.bend', ('second_turn',)), ('stale-second-ep', 'Fixtures.bend', 'second_ep', '== 44 : U32', '== 20 : U32', 'Fixtures.bend', ('second_ep',)), ('accept-second-invalid', 'Fixtures.bend', 'second_invalid', '\n    (ALeaf{U64.zero},None{}) : Array', '\n    (ALeaf{U64.zero},Some{raw()}) : Array', 'Fixtures.bend', ('second_invalid',)), ('disconnect-fen-rows', 'consumer.bend', 'fen_lift', 'Rows.fen(pieces,turn,castle,target,half,fulltext)', 'Unit{}', 'consumer.bend', ('fen_lift',)), ('disconnect-fen-turn', 'consumer.bend', 'fen_lift', 'Turn.fen(pieces,turn,castle,target,half,fulltext)', 'Unit{}', 'consumer.bend', ('fen_lift',)), ('remove-initialization', 'consumer.bend', 'use_fen_replay', 'initialized: {a == Init.run(d,seed,n,extra) : Array<U64>}', 'initialized: {a == a : Array<U64>}', 'consumer.bend', ('use_fen_replay',)))
SUITE_PREFIX = "standalone/proofs/protocol_replay_metadata/"

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
              "scope": 'Nonempty actual Protocol.moves replay preserves exact affine input table and bootstraps conditional Board.valid, canonical turn and EP64or<64 from only parent rows+canonical, no input EP bound. Actual Source.legal bounds+fullPly, Find.all, Position.apply and arbitrary ordered duplicate-containing command tail/None/failure. Public FEN derives Rows.fen/Turn.fen internally through actual validation and retains initialization equation and d17/n128/extra64 in result. No child Position.valid, EP coherence, kings, replay acceptance, legal correctness, arbitrary fast/full equivalence or closed Init.run normalization claim.'}
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
                target = mutant / SUITE_PREFIX / filename
                baseline = sha256(target)
                mutate_declaration(target, declaration, old, new)
                mutated = sha256(target)
                mutant_guard(mutant,target,mutated)
                guard("before-control-" + name)
                result = run_check(name, mutant / SUITE_PREFIX / entry, compiler, bun, 86400, args.evidence_dir, "1,3")
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
