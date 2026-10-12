"""Fail-closed qualification of actual generated full-Ply fields and bypass classification."""
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
BASE = "4dd45ecec257763a5be9635fd123e3a1fea455b6"
BASE_TREE = "f61e53a88ab1d926bbcf4c7a78161b2d97d57415"
BRANCH = "proof/bend-candidate-bypass-classification-20261005"

CONTROLS: tuple[tuple[str, str, str, str, str, tuple[str, ...]], ...] = (('wrong-promotion-choice', 'Emission.bend', 'put', 'Promotion.Knight{}', 'Promotion.Queen{}', ('put',)), ('wrong-ordinary-ep-flag', 'Emission.bend', 'put', 'Chess.put_move(promote,src,dst,Bool.to_u32(ep),tail)', 'Chess.put_move(promote,src,dst,0,tail)', ('put',)), ('omit-tail-classification', 'Emission.bend', 'put', 'good: Chain.every(m => Fields.scanned(m),tail)', 'good: Unit', ('put',)), ('drop-scan-after-tail', 'Scan.bend', 'after', 'Goal(Chess.scan_after(src,pawn,ep,tail,r))', 'Goal(Chess.scan_after(src,pawn,ep,Nil{},r))', ('after',)), ('wrong-scan-source', 'Scan.bend', 'step', 'Goal(Chess.scan_step(b,src,r))', 'Goal(Chess.scan_step(b,0,r))', ('step',)), ('disconnect-owned-source-scan', 'Scan.bend', 'initial', 'Chess.color(b,Chess.get_turn(b))', 'U64.zero()', ('initial',)), ('wrong-second-castle-side', 'Generated.bend', 'before', 'b,False{},z,pz', 'b,True{},z,pz', ('before',)), ('omit-actual-castle-guard', 'Generated.bend', 'before', 'g => Inr{Inl{(g,{==})}}', 'g => Inr{Inl{({==},{==})}}', ('before',)), ('disconnect-final-prepare', 'Generated.bend', 'after', 'Goal(Chain.suffix(b,r))', 'Goal(sides(b,r))', ('after',)), ('omit-actual-bypass', 'Bypass.bend', 'scanned', 'bypass: {Chess.filter_requires(sensitive,m) == False{} : Bool}', 'bypass: {False{} == False{} : Bool}', ('scanned',)), ('disconnect-guarded-castle-forcing', 'Bypass.bend', 'generated', 'Mask.certificate(b,m,rays,castle)', '{==}', ('generated',)), ('omit-prefilter-member', 'Bypass.bend', 'prefilter', 'here: E.member(m,E.moves(Generated.produced(table,b)))', 'here: Unit', ('prefilter',)), ('wrong-consumer-query', 'consumer.bend', 'prefilter_bypass', 'Bypass.prefilter(table,b,rays,m,here,bypass)', 'Bypass.prefilter(table,b,rays,Chess.Ply{0,0,0,0},here,bypass)', ('prefilter_bypass',)), ('disconnect-legal-filter-consumer', 'consumer.bend', 'legal_bypass', 'Bypass.legal(table,b,rays,m,here,bypass)', 'Bypass.prefilter(table,b,rays,m,here,bypass)', ('legal_bypass',)), ('omit-owned-king-source', 'consumer.bend', 'owned_king_forces', 'owned: {Sensitive.owned_king(b,src) == True{} : Bool}', 'owned: Unit', ('owned_king_forces',)), ('wrong-returned-table', 'consumer.bend', 'multi_after_exact', '(ALeaf{U64.bit(63n)},Chess.Ply{2,56,1,0}', '(ALeaf{U64.zero()},Chess.Ply{2,56,1,0}', ('multi_after_exact',)), ('dropped-duplicate-tail', 'consumer.bend', 'multi_after_exact', 'Chess.Ply{2,10,0,1} <> duplicate_tail()', 'Chess.Ply{2,10,0,1} <> Nil{}', ('multi_after_exact',)), ('wrong-ep-field', 'consumer.bend', 'multi_after_exact', 'Chess.Ply{2,10,0,1} <> duplicate_tail()', 'Chess.Ply{2,10,0,0} <> duplicate_tail()', ('multi_after_exact',)), ('wrong-promotion-ep-field', 'consumer.bend', 'promotion_overrides_ep', 'Promotion.choices(2,56)', 'Chess.Ply{2,56,1,1} <> Chess.Ply{2,56,2,1} <> Chess.Ply{2,56,3,1} <> Chess.Ply{2,56,4,1} <> Nil{}', ('promotion_overrides_ep',)), ('absent-ordinary-shadow', 'consumer.bend', 'fixture_prefilter', 'Promotion.choices(49,57)', 'Chess.Ply{49,57,0,0} <> Nil{}', ('fixture_prefilter',)), ('wrong-legal-order', 'consumer.bend', 'fixture_legal_order', 'Chess.Ply{49,57,4,0} <> Chess.Ply{49,57,3,0}', 'Chess.Ply{49,57,3,0} <> Chess.Ply{49,57,4,0}', ('fixture_legal_order',)), ('raw-flag-two-forced', 'consumer.bend', 'raw_flag_two_can_bypass', '== False{} : Bool', '== True{} : Bool', ('raw_flag_two_can_bypass',)), ('ep-not-forced', 'consumer.bend', 'ep_forces_any_promotion', '== True{} : Bool', '== False{} : Bool', ('ep_forces_any_promotion',)), ('high-king-not-forced', 'consumer.bend', 'high_king_forces_any_fields', '== True{} : Bool', '== False{} : Bool', ('high_king_forces_any_fields',)), ('sanitize-unclassified-tail', 'consumer.bend', 'unclassified_tail_survives', '(ALeaf{U64.zero()},Chess.Ply{99,100,99,99} <> Nil{})', '(ALeaf{U64.zero()},Nil{})', ('unclassified_tail_survives',)), ('ignore-explicit-empty', 'consumer.bend', 'explicit_empty', 'Chess.destinations(64n,True{},multi_targets()', 'Chess.destinations(64n,False{},multi_targets()', ('explicit_empty',)))

FALSE_WITNESSES = frozenset(['absent-ordinary-shadow', 'dropped-duplicate-tail', 'ep-not-forced', 'high-king-not-forced', 'ignore-explicit-empty', 'raw-flag-two-forced', 'sanitize-unclassified-tail', 'wrong-ep-field', 'wrong-legal-order', 'wrong-promotion-ep-field', 'wrong-returned-table'])

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
        "scope": 'Exact full-Ply alternatives for every occurrence in actual scan, both guarded prefilter castle producers and actual legal_moves output: tag0/flag0, tag0/flag1, typed promotion tags1..4/flag0, or actual guarded castle certificate. Actual generated filter_requires False removes EP and guarded castles, leaving exact ordinary or typed promotion witnesses. Arbitrary affine tables/raw boards, source keys and target masks; implementation fuel/empty flags retained; supplied tail occurrence certificates preserved. No Board.valid/canonical-turn/nonempty-king/geometry/lookup premise for classification. No no-check preservation, table-builder contract, post-board validity or full fast/full-equivalence claim.',
        "remote_base_commit": BASE,
        "local_base_commit": None,
        "local_base_tree": None,
        "local_base_alias_note": "Exact PR1037 base/tree pins reused dependencies. Supports a base overlay or clean published head; published source files must match HEAD Git blobs.",
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
        paths = set(closure(SUITE / "consumer.bend", ENGINE))
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
            raise RuntimeError("PR1037 base tree identity mismatch")
        report["local_base_commit"] = BASE
        report["local_base_tree"] = base_tree
        if report["local_head_commit"] == BASE:
            if report["local_head_tree"] != BASE_TREE:
                raise RuntimeError("base overlay HEAD tree identity mismatch")
            report["input_mode"] = "base-overlay"
        else:
            allowed = {
                "native/bend_engine/standalone/proofs/candidate_bypass_classification/" + name
                for name in ("Fields.bend", "Emission.bend", "Scan.bend", "Generated.bend", "Bypass.bend", "consumer.bend",
                             "qualify_candidate_bypass_classification.py", "README.md")
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
            if path.startswith("standalone/proofs/candidate_bypass_classification/") or path in correction_paths:
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

        result = run_check("consumer", snapshot / "standalone/proofs/candidate_bypass_classification/consumer.bend", compiler, bun,
                           86400, args.evidence_dir, cpu_affinity)
        report["positive_checks"].append(result)
        write_report(args.report, report)
        if not result["passed"]:
            raise RuntimeError("positive consumer check failed")

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
            entry = (copy_engine / target if filename in {"Fields.bend", "Emission.bend", "Scan.bend", "Generated.bend", "Bypass.bend"}
                     else copy_engine / "standalone/proofs/candidate_bypass_classification/consumer.bend")
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
