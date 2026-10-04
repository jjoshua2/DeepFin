"""Fail-closed qualification of ordinary king preservation and old/new whole-pair query composition."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from ..generator_contract.qualify import load_closure
from ..fast_full_equivalence.qualify_fast_full import (
    MEMORY_CAP, OUTPUT_CAP, mutate_declaration, now, rejected_at,
    run_check, sha256, write_report,
)

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
PROJECT = ENGINE.parents[1]
BASE = "b42d8188c94936a4de4a687ee0eaf22d2e327cb1"
BASE_TREE = "b93bed3ea89d65a74af0c5ca5bb03b4b96998400"
BRANCH = "proof/bend-ordinary-king-frame-20261004"

CONTROLS: tuple[tuple[str, str, str, str, str, tuple[str, ...]], ...] = (('omit-actual-candidate', 'consumer.bend', 'use_frame', 'here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(generation_table,b)))', 'here: Unit', ('use_frame',)), ('omit-board-consistency', 'consumer.bend', 'use_frame', 'good: {B.valid(b) == True{} : Bool}', 'good: Unit', ('use_frame',)), ('omit-destination-king-exclusion', 'consumer.bend', 'use_frame', 'target_clear: D.disjoint(U64.bit(U32.to_nat(dst)),Chess.get_kings(b))', 'target_clear: Unit', ('use_frame',)), ('invert-actual-bypass', 'consumer.bend', 'use_frame', 'Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{}', 'Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == True{}', ('use_frame',)), ('wrong-special-flag', 'Frame.bend', 'moved', 'Chess.Ply{src,dst,0,0}', 'Chess.Ply{src,dst,0,1}', ('F.kings_update',)), ('wrong-promotion', 'Frame.bend', 'moved', 'Chess.Ply{src,dst,0,0}', 'Chess.Ply{src,dst,5,0}', ('F.kings_update',)), ('omit-canonical-turn', 'consumer.bend', 'use_enemy', 'canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}', 'canonical: Unit', ('use_enemy',)), ('omit-old-rook-contract', 'consumer.bend', 'use_queries', 'old_r: Q.rook(c,K.index(b),Chess.get_turn(b),Chess.occupied(b))', 'old_r: Unit', ('use_queries',)), ('omit-old-bishop-contract', 'consumer.bend', 'use_queries', 'old_d: Q.bishop(c,K.index(b),Chess.get_turn(b),Chess.occupied(b))', 'old_d: Unit', ('use_queries',)), ('omit-new-rook-contract', 'consumer.bend', 'use_queries', 'new_r: Q.rook(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst)))', 'new_r: Unit', ('use_queries',)), ('omit-new-bishop-contract', 'consumer.bend', 'use_queries', 'new_d: Q.bishop(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst)))', 'new_d: Unit', ('use_queries',)), ('wrong-returned-table', 'consumer.bend', 'use_check_square', 'Chess.attacked(O.pack(c),F.moved(b,src,dst),Q.king(b),U32.xor(Chess.get_turn(b),1)) :', 'Chess.attacked(ALeaf{U64.zero()},F.moved(b,src,dst),Q.king(b),U32.xor(Chess.get_turn(b),1)) :', ('use_check_square',)), ('omit-moving-king-presence', 'consumer.bend', 'use_queries', 'U64.is_zero(K.plane(b)) == False{}', 'True{} == True{}', ('use_queries',)), ('disconnect-new-occupancy', 'consumer.bend', 'post_queen', 'r: Q.rook(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst)))', 'r: Q.rook(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(b))', ('post_queen',)), ('actual-member-frame-witness', 'consumer.bend', 'actual_frame', '== U64.bit(8n) : U64', '== U64.bit(32n) : U64', ('actual_frame',)), ('flipped-turn-selects-other-witness', 'consumer.bend', 'next_side_selects_other', '== 56 : U32', '== 8 : U32', ('next_side_selects_other',)), ('capture-erases-rook-witness', 'consumer.bend', 'capture_erases_rook', '== U64.zero() : U64', '== U64.bit(63n) : U64', ('capture_erases_rook',)), ('king-destination-clears-plane-witness', 'consumer.bend', 'king_destination_clears_plane', '== 64 : U32', '== 8 : U32', ('king_destination_clears_plane',)), ('invalid-board-adds-king-witness', 'consumer.bend', 'phantom_adds_king', '== U64.or(U64.bit(8n),U64.bit(10n)) : U64', '== U64.bit(8n) : U64', ('phantom_adds_king',)), ('noncanonical-enemy-adds-target-witness', 'consumer.bend', 'noncanonical_enemy_adds_target', '== True{} : Bool', '== False{} : Bool', ('noncanonical_enemy_adds_target',)), ('noncanonical-pawn-intersection-witness', 'consumer.bend', 'noncanonical_pawn_intersection_empty', '== U64.zero() : U64', '== U64.bit(10n) : U64', ('noncanonical_pawn_intersection_empty',)))
FALSE_WITNESSES = ['actual-member-frame-witness', 'capture-erases-rook-witness', 'flipped-turn-selects-other-witness', 'invalid-board-adds-king-witness', 'king-destination-clears-plane-witness', 'noncanonical-enemy-adds-target-witness', 'noncanonical-pawn-intersection-witness']


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
        "scope": "Actual ordinary make_move preserves the original-side king plane under candidate membership, Board.valid, actual bypass and explicit destination king exclusion. Canonical turns give exact enemy/color-kind erasure. Compose actual in_check square and separate OLD/NEW whole-pair rook/bishop contracts at the preserved square. No post-check False implication, destination provenance discharge, builder validity, king uniqueness or full legality claim.",
        "remote_base_commit": BASE,
        "local_base_commit": None,
        "local_base_tree": None,
        "local_base_alias_note": "Exact PR1031 base/tree pins reused dependencies. Supports a base overlay or clean published head; published source files must match HEAD Git blobs.",
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
            raise RuntimeError("PR1031 base tree identity mismatch")
        report["local_base_commit"] = BASE
        report["local_base_tree"] = base_tree
        if report["local_head_commit"] == BASE:
            if report["local_head_tree"] != BASE_TREE:
                raise RuntimeError("base overlay HEAD tree identity mismatch")
            report["input_mode"] = "base-overlay"
        else:
            allowed = {
                "native/bend_engine/standalone/proofs/ordinary_king_frame/" + name
                for name in ("Algebra.bend", "Frame.bend", "Source.bend", "Enemy.bend",
                             "consumer.bend", "qualify_ordinary_king_frame.py", "README.md")
            }
            allowed.update("native/bend_engine/" + p for p in correction_paths)
            changed = subprocess.run(
                ["git", "-C", str(PROJECT), "diff", "--no-renames", "--name-only", BASE, "HEAD"],
                capture_output=True, text=True, check=True, timeout=30).stdout.splitlines()
            if not changed or set(changed) - allowed:
                raise RuntimeError("published HEAD changes outside this proof suite")
            dirty = subprocess.run(
                ["git", "-C", str(PROJECT), "status", "--porcelain", "--untracked-files=no"],
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
            if path.startswith("standalone/proofs/ordinary_king_frame/") or path in correction_paths:
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

        result = run_check("consumer", SUITE / "consumer.bend", compiler, bun,
                           86400, args.evidence_dir, cpu_affinity)
        report["positive_checks"].append(result)
        write_report(args.report, report)
        if not result["passed"]:
            raise RuntimeError("positive consumer check failed")

        controls = CONTROLS
        false_witnesses = FALSE_WITNESSES
        for name, filename, declaration, old, new, locations in controls:
            target = "standalone/proofs/ordinary_king_frame/" + filename
            with tempfile.TemporaryDirectory(prefix="deepfin-ordinary-king-control-") as tmp:
                copy_engine = Path(tmp) / "engine"
                shutil.copytree(ENGINE, copy_engine, symlinks=True)
                target_path = copy_engine / target
                baseline = sha256(target_path)
                mutate_declaration(target_path, declaration, old, new)
                mutated = sha256(target_path)
                entry = copy_engine / "standalone/proofs/ordinary_king_frame/consumer.bend"
                result = run_check(name, entry, compiler, bun, 86400, args.evidence_dir, cpu_affinity)
                result.update({
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
