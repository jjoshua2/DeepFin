"""Fail-closed qualification of actual generated-candidate bypass safety and exact fast/full composition."""
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

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
PROJECT = ENGINE.parents[1]
BASE = "02a2620006e5c10ae69139785b0d878894a6fc81"
BASE_TREE = "a90a94238771b306eeda87f7100ecfdcbe737441"
BRANCH = "proof/bend-generated-bypass-safety-20261005"

CONTROLS = (('wrong-actual-post-occupancy', 'Spec.bend', 'post', 'Q.rook(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(Chess.make_move(b,m)))', 'Q.rook(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(b))', ('plain_pairs',)), ('lookup-disconnected-from-scan', 'Spec.bend', 'lookup', 'E.member(m,scan(c,b))', 'E.member(m,candidates(c,b))', ('scan_case',)), ('wrong-ordinary-no-check-query', 'Safety.bend', 'plain_pairs', 'Ordinary.use_no_check(c,b,src,dst,white', 'Ordinary.use_no_check(c,b,dst,src,white', ('plain_pairs',)), ('disconnect-promotion-no-check', 'Safety.bend', 'promotion_pairs', 'Promotion.use_no_check(c,b,src,dst,choice,white', 'Ordinary.use_no_check(c,b,src,dst,choice,white', ('promotion_pairs',)), ('wrong-classified-full-ply', 'Safety.bend', 'scan_case', 'Scan.member(O.pack(c),b,m,fields_here)', 'Scan.member(O.pack(c),b,Chess.Ply{0,0,0,0},fields_here)', ('scan_case',)), ('replace-actual-prepared-rays', 'Safety.bend', 'use_bypass', 'Origin.scan_bypass(c,b,Prepared.rays(c,b),m,here,bypass)', 'Origin.scan_bypass(c,b,U64.zero(),m,here,bypass)', ('use_bypass',)), ('disconnect-occurrence-query', 'Lift.bend', 'occurrences', 'Safety.candidate(c,b,m,white,here', 'Safety.candidate(c,b,Chess.Ply{0,0,0,0},white,here', ('occurrences',)), ('disconnect-actual-table-consumer', 'consumer.bend', 'actual_legal_full', 'Actual.prepared_equal(c,b)', '{==}', ('actual_legal_full',)), ('omit-guarded-castle-forcing', 'Forced.bend', 'castle_step', 'Mask.certificate(b,m,Prepared.rays(c,b),castle)', '{==}', ('castle_step',)), ('wrong-ep-field-contract', 'Forced.bend', 'ep_step', 'ep: Fields.ep(m)', 'ep: Fields.plain(m)', ('ep_step',)), ('wrong-promotion-output-tag', 'Fixtures.bend', 'promotion_repeated_order', 'P.ply(49,57,P.Rook{}) <> P.ply(49,57,P.Bishop{}) <> P.ply(49,57,P.Queen{}) <>', 'P.ply(49,57,P.Rook{}) <> P.ply(49,57,P.Bishop{}) <> P.ply(49,57,P.Rook{}) <>', ('promotion_repeated_order',)), ('wrong-post-turn-check', 'Fixtures.bend', 'ordinary_actual_safe', 'Chess.Ply{19,34,0,0}),1) ==', 'Chess.Ply{19,34,0,0}),0) ==', ('ordinary_actual_safe',)), ('dropped-promotion-duplicate', 'Fixtures.bend', 'promotion_repeated_order', 'P.ply(49,57,P.Knight{}) <> P.ply(49,57,P.Queen{}) <> duplicate_tail()', 'P.ply(49,57,P.Knight{}) <> duplicate_tail()', ('promotion_repeated_order',)), ('wrong-promotion-order', 'Fixtures.bend', 'promotion_repeated_order', 'P.ply(49,57,P.Rook{}) <> P.ply(49,57,P.Bishop{}) <>', 'P.ply(49,57,P.Bishop{}) <> P.ply(49,57,P.Rook{}) <>', ('promotion_repeated_order',)), ('dropped-unclassified-tail', 'Fixtures.bend', 'promotion_repeated_order', '<> duplicate_tail())', '<> Nil{})', ('promotion_repeated_order',)), ('wrong-returned-table', 'Fixtures.bend', 'ordinary_repeated_order', '(O.pack(Ordinary.fixture_cells()),Chess.Ply{19,34,0,0} <>', '(ALeaf{U64.zero()},Chess.Ply{19,34,0,0} <>', ('ordinary_repeated_order',)), ('dropped-ordinary-duplicate', 'Fixtures.bend', 'ordinary_repeated_order', 'Chess.Ply{19,34,0,0} <> Chess.Ply{19,34,0,0} <> Chess.Ply{19,34,0,0} <> duplicate_tail()', 'Chess.Ply{19,34,0,0} <> Chess.Ply{19,34,0,0} <> duplicate_tail()', ('ordinary_repeated_order',)))

FALSE_WITNESSES = frozenset(['dropped-ordinary-duplicate', 'dropped-promotion-duplicate', 'dropped-unclassified-tail', 'wrong-post-turn-check', 'wrong-promotion-order', 'wrong-promotion-output-tag', 'wrong-returned-table'])


def source_hashes(paths: set[str]) -> dict[str, str]:
    return {path: sha256(ENGINE / path) for path in sorted(paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--checker-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    if args.report.exists() or args.evidence_dir.exists():
        raise RuntimeError("qualification requires fresh report and evidence paths")
    prior_evidence = {str(p): sha256(p) for p in (SUITE / "evidence").rglob("*") if p.is_file()}
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {
        "gate": "NOT_COMPLETED",
        "scope": "Actual prefilter candidate-origin bridge, exact full-Ply ordinary/promotion no-check dispatch and EP/guarded-castle forced-full paths; derives per-occurrence bypass safety from explicit valid-board, nonempty original king plane, canonical turn, original-side initial whole-pair False and exact old/actual-post rook/bishop lookup contracts. Composes exact same-table fast/full and actual legal_moves/full equality, preserving ordered duplicates and arbitrary retained tail. No initialized Tables.build, post-board validity, full chess legality or unconditional fast/full claim.",
        "remote_base_commit": BASE,
        "local_base_commit": None,
        "local_base_tree": None,
        "local_base_alias_note": "Exact PR1038 base/tree pins reused dependencies. Supports a base overlay or clean published head; published source files must match HEAD Git blobs.",
        "branch": BRANCH,
        "compiler_pin": {},
        "source_sha256s": {},
        "positive_checks": [],
        "negative_controls": [],
    }
    affinity = os.sched_getaffinity(0)
    try:
        selected_cpus = sorted(affinity)[:2]
        if not selected_cpus:
            raise RuntimeError("no allowed CPUs")
        cpu_affinity = ",".join(map(str, selected_cpus))
        os.sched_setaffinity(0, set(selected_cpus))
        os.environ.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                           "MKL_NUM_THREADS": "2", "RAYON_NUM_THREADS": "2",
                           "BEND_NO_TELEMETRY": "1"})
        bun = os.environ.get("BUN", str(Path.home() / ".bun" / "bin" / "bun"))
        version = subprocess.run([bun, "--version"], capture_output=True, text=True,
                                 check=True, timeout=15)
        report["bun_version"] = version.stdout.strip()
        pin_cmd = [bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)]
        pin = subprocess.run(pin_cmd, capture_output=True, text=True,
                             check=True, timeout=30)
        if pin.stderr:
            raise RuntimeError("compiler verifier produced stderr")
        report["compiler_pin"]["verify_command"] = pin_cmd
        report["compiler_pin"]["identity_output"] = pin.stdout.strip()
        manifest = json.loads(args.checker_manifest.read_text())
        files = manifest["files"]
        ordered = sorted(files, key=lambda row: row["path"])
        fingerprint = hashlib.sha256(b"".join(
            row["path"].encode("utf-8") + b"\0" + bytes.fromhex(row["sha256"])
            for row in ordered
        )).hexdigest()
        file_match = all(
            (compiler / row["path"]).is_file()
            and sha256(compiler / row["path"]) == row["sha256"]
            for row in files
        )
        if (len(files) != 84 or fingerprint != manifest["source_fingerprint_sha256"]
                or fingerprint != "d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4"
                or not file_match or report["bun_version"] != "1.4.2"
                or "aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae" not in pin.stdout
                or "d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4" not in pin.stdout):
            raise RuntimeError("Bend/Bun/checker-tree identity mismatch")
        report["checker_tree"] = {
            "manifest_path": str(args.checker_manifest.resolve()),
            "files": len(files), "fingerprint_sha256": fingerprint,
            "manifest_sha256": sha256(args.checker_manifest),
            "all_file_hashes_match": file_match,
        }
        closure = load_closure()
        paths = set(closure(SUITE / "Fixtures.bend", ENGINE))
        paths.update(str(p.relative_to(ENGINE)) for p in SUITE.iterdir() if p.is_file())
        paths.update({"standalone/toolchain.json", "standalone/verify_compiler.js"})
        support = {
            "standalone/proofs/generator_contract/qualify.py",
            "standalone/proofs/table_preservation/focused.py",
            "standalone/proofs/table_preservation/_validation.py",
            str(Path(__file__).resolve().relative_to(ENGINE)),
        }
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename:
                module_path = Path(filename).resolve()
                if module_path.suffix == ".py" and module_path.is_relative_to(ENGINE):
                    support.add(str(module_path.relative_to(ENGINE)))
        correction_paths: set[str] = set()
        paths.update(support)
        paths.update(correction_paths)
        before = source_hashes(paths)
        report["qualification_support_sha256s"] = source_hashes(support)
        report["source_sha256s"] = before
        report["local_head_commit"] = subprocess.run(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        report["local_head_tree"] = subprocess.run(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD^{tree}"],
            capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        base_tree = subprocess.run(
            ["git", "-C", str(PROJECT), "rev-parse", BASE + "^{tree}"],
            capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        if base_tree != BASE_TREE:
            raise RuntimeError("PR1038 base tree identity mismatch")
        report["local_base_commit"] = BASE
        report["local_base_tree"] = base_tree
        if report["local_head_commit"] == BASE:
            if report["local_head_tree"] != BASE_TREE:
                raise RuntimeError("base overlay HEAD tree identity mismatch")
            report["input_mode"] = "base-overlay"
        else:
            allowed = {
                "native/bend_engine/standalone/proofs/generated_bypass_safety/" + name
                for name in ("Spec.bend", "Origin.bend", "Safety.bend", "Lift.bend", "Forced.bend", "Fixtures.bend", "consumer.bend",
                             "qualify_generated_bypass_safety.py", "README.md")
            }
            allowed.update("native/bend_engine/" + p for p in correction_paths)
            changed = subprocess.run(
                ["git", "-C", str(PROJECT), "diff", "--no-renames", "--name-only", BASE, "HEAD"],
                capture_output=True, text=True, check=True, timeout=30).stdout.splitlines()
            if set(changed) != allowed:
                raise RuntimeError("published HEAD changes outside this proof suite")
            dirty = subprocess.run(
                ["git", "-C", str(PROJECT), "status", "--porcelain"],
                capture_output=True, text=True, check=True, timeout=15).stdout
            if dirty:
                raise RuntimeError("published-head qualification requires clean tracked files")
            report["input_mode"] = "published-head"
            report["published_delta_paths"] = sorted(changed)
            head_listing = subprocess.run(
                ["git", "-C", str(PROJECT), "ls-tree", "-r", "HEAD"],
                capture_output=True, text=True, check=True, timeout=30).stdout
            head_blobs = {}
            for line in head_listing.splitlines():
                metadata, path = line.split("\t", 1)
                _mode, kind, digest = metadata.split()
                if kind == "blob":
                    head_blobs[path] = digest
            qualified = {}
            for path in sorted(paths):
                raw = (ENGINE / path).read_bytes()
                blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                if head_blobs.get("native/bend_engine/" + path) != blob:
                    raise RuntimeError("qualified source differs from HEAD: " + path)
                qualified[path] = blob
            report["qualified_source_git_blobs"] = qualified
            report["qualified_sources_match_head"] = True
        listing = subprocess.run(
            ["git", "-C", str(PROJECT), "ls-tree", "-r", BASE],
            capture_output=True, text=True, check=True, timeout=30).stdout
        base_blobs = {}
        for line in listing.splitlines():
            metadata, path = line.split("\t", 1)
            _mode, kind, digest = metadata.split()
            if kind == "blob":
                base_blobs[path] = digest
        reused = {}
        for path in sorted(paths):
            if path.startswith("standalone/proofs/generated_bypass_safety/") or path in correction_paths:
                continue
            raw = (ENGINE / path).read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            remote_path = "native/bend_engine/" + path
            if base_blobs.get(remote_path) != blob:
                raise RuntimeError("dependency differs from intended base: " + path)
            reused[path] = blob
        report["base_evidence_correction_paths"] = sorted(correction_paths)
        report["reused_dependency_git_blobs"] = reused
        report["reused_dependencies_match_exact_base"] = True
        report["remote_base_tree"] = BASE_TREE
        report["prior_evidence_sha256s"] = prior_evidence
        report["source_identity_sha256"] = hashlib.sha256(
            json.dumps(before, sort_keys=True).encode()).hexdigest()
        report["resource_limits"] = {
            "positive_wall_timeout_seconds": 86400,
            "negative_wall_timeout_seconds": 86400, "cpu_affinity": cpu_affinity,
            "selected_cpus": selected_cpus, "inherited_allowed_cpus": sorted(affinity),
            "OMP_NUM_THREADS": 2, "address_space_cap_bytes": MEMORY_CAP,
            "stdout_stderr_file_cap_bytes": OUTPUT_CAP,
        }
        snapshot = args.evidence_dir / "source-snapshot"
        for path in sorted(paths):
            destination = snapshot / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ENGINE / path, destination)
        (args.evidence_dir / "source-identity.json").write_text(
            json.dumps({"head": report["local_head_commit"], "tree": report["local_head_tree"],
                        "source_sha256s": before,
                        "identity_sha256": report["source_identity_sha256"]}, indent=2) + "\n")
        write_report(args.report, report)

        result = run_check("Fixtures", snapshot / "standalone/proofs/generated_bypass_safety/Fixtures.bend", compiler, bun,
                           86400, args.evidence_dir, cpu_affinity)
        report["positive_checks"].append(result)
        write_report(args.report, report)
        if not result["passed"]:
            raise RuntimeError("positive full fixture/consumer check failed")

        controls = CONTROLS
        false_witnesses = FALSE_WITNESSES
        for name, filename, declaration, old, new, locations in controls:
            target = str((SUITE / filename).resolve().relative_to(ENGINE))
            copy_engine = args.evidence_dir / "mutants" / name
            shutil.copytree(snapshot, copy_engine, symlinks=False)
            target_path = copy_engine / target
            baseline = sha256(target_path)
            mutate_declaration(target_path, declaration, old, new)
            mutated = sha256(target_path)
            entry_filename = "Safety.bend" if filename == "Spec.bend" else filename
            entry = copy_engine / "standalone/proofs/generated_bypass_safety" / entry_filename
            result = run_check(name, entry, compiler, bun, 86400, args.evidence_dir, cpu_affinity)
            result.update({
                "checked_entry": str(entry.relative_to(copy_engine)),
                "control_kind": ("concrete-false-witness" if name in false_witnesses else "contract-coupling"),
                "mutation_target": target, "mutation_declaration": declaration, "baseline_sha256": baseline,
                "mutated_sha256": mutated, "mutation_anchor": old,
                "replacement": new,
                "rejected_at_expected_obligation": rejected_at(result, locations),
                "expected_locations": list(locations),
            })
            report["negative_controls"].append(result)
            write_report(args.report, report)
            if not result["rejected_at_expected_obligation"]:
                raise RuntimeError("negative control did not reject: " + name)

        if any(not Path(p).is_file() or sha256(Path(p)) != h for p,h in prior_evidence.items()):
            raise RuntimeError("prior evidence changed during qualification")
        report["prior_evidence_unchanged"] = True
        if source_hashes(paths) != before:
            raise RuntimeError("qualified source identity changed during run")
        final_pin = subprocess.run(pin_cmd, capture_output=True, text=True,
                                   check=True, timeout=30)
        if final_pin.stderr or final_pin.stdout != pin.stdout:
            raise RuntimeError("compiler identity changed during qualification")
        report["compiler_identity_unchanged"] = True
        report["source_unchanged"] = True
        report["evidence_sha256s"] = {str(p.relative_to(args.evidence_dir)): sha256(p) for p in sorted(args.evidence_dir.rglob("*")) if p.is_file()}
        report["gate"] = "PASS"
    except Exception as failure:
        report["failure"] = repr(failure)
        report["gate"] = "FAIL"
        raise
    finally:
        os.sched_setaffinity(0, affinity)
        report["finished_at_utc"] = now()
        write_report(args.report, report)


if __name__ == "__main__":
    main()
