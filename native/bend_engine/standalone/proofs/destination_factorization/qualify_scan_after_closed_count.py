"""Fail-closed qualification of the closed actual scan_after per-query count."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import tempfile
import time

from ..generator_contract.qualify import load_closure, replace

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
PROJECT = ENGINE.parents[1]
OUTPUT_CAP = 65536
MEMORY_CAP = 6 * 1024 * 1024 * 1024


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def preexec_limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP, MEMORY_CAP))
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_CAP, OUTPUT_CAP))


def run_check(name: str, entry: Path, compiler: Path, bun: str, seconds: int,
              output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / (name + ".stdout.txt")
    stderr_path = output_dir / (name + ".stderr.txt")
    time_path = output_dir / (name + ".time.txt")
    command = [
        "/usr/bin/time", "-v", "-o", str(time_path),
        "timeout", "--signal=TERM", str(seconds) + "s",
        "taskset", "-c", "0,1", "env",
        "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "MKL_NUM_THREADS=2",
        "RAYON_NUM_THREADS=2", "BEND_NO_TELEMETRY=1", bun, "--smol",
        str(compiler / "bend2/main.ts"), str(entry), "--check-only",
    ]
    started = utcnow()
    start = time.monotonic()
    timed_out = False
    code = None
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr,
                                     timeout=seconds + 15, check=False,
                                     preexec_fn=preexec_limits)
        code = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    elapsed = time.monotonic() - start
    stdout = stdout_path.read_bytes() if stdout_path.exists() else b""
    stderr = stderr_path.read_bytes() if stderr_path.exists() else b""
    raw = stdout + stderr
    text = raw.decode("utf-8", errors="replace")
    time_text = time_path.read_text(errors="replace") if time_path.exists() else ""
    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", time_text)
    result = {
        "name": name, "command": command, "started_at_utc": started,
        "finished_at_utc": utcnow(), "elapsed_wall_seconds": elapsed,
        "wall_timeout_seconds": seconds, "exit_code": code, "timed_out": timed_out,
        "stdout_bytes": len(stdout), "stderr_bytes": len(stderr),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "raw_output_sha256": hashlib.sha256(raw).hexdigest(),
        "max_rss_kib": int(rss_match.group(1)) if rss_match else None,
        "max_output_bytes": OUTPUT_CAP, "address_space_cap_bytes": MEMORY_CAP,
        "cpu_affinity": "0,1", "raw_text": text,
    }
    result["passed"] = (
        code == 0 and not timed_out and text == "All terms check.\n"
        and len(raw) <= OUTPUT_CAP
        and result["max_rss_kib"] is not None
        and result["max_rss_kib"] <= MEMORY_CAP // 1024
    )
    return result


def identity(paths: set[str]) -> dict[str, str]:
    return {p: sha256(ENGINE / p) for p in sorted(paths)}


def control_rejected(result: dict, locations: tuple[str, ...]) -> bool:
    text = result["raw_text"]
    forbidden = ("out of memory", "OutOfMemory", "WARNING", "unsafe",
                 "no such file", "Maximum call stack", "timeout")
    match = re.search(r"Location:\s*([\w./]+)", text)
    return (
        result["exit_code"] == 1 and not result["timed_out"]
        and "expected" in text and "observed" in text and match is not None
        and any(location in match.group(1) for location in locations)
        and not any(word in text for word in forbidden)
        and result["stdout_bytes"] + result["stderr_bytes"] <= OUTPUT_CAP
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--checker-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    report = {
        "gate": "NOT_COMPLETED",
        "scope": "Exact full-Ply count of actual Chess.scan_after for arbitrary U64 targets, U32 query, affine table and duplicate-containing Ply tail; count is factored through actual Chess.destinations and actual bit_squares frequency/range. No target geometry, board validity or legal-move completeness.",
        "base_commit": "123bb1a8201554ae18a8a44966b627d971452f01",
        "branch": "proof/bend-scan-after-predicate-20261003",
        "compiler_pin": {},
        "source_sha256s": {},
        "positive_checks": [],
        "negative_controls": [],
    }
    original_affinity = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, set(sorted(original_affinity)[:2]))
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
            raise RuntimeError("Bend/Bun/checker tree identity mismatch")
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
        before = identity(paths)
        report["source_sha256s"] = before
        git_head = subprocess.run(["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        if git_head != report["base_commit"]:
            raise RuntimeError("source checkout moved off the requested base before qualification")
        report["git_head_before"] = git_head
        report["source_identity_sha256"] = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
        report["resource_limits"] = {
            "positive_wall_timeout_seconds": 86400, "negative_wall_timeout_seconds": 120,
            "cpu_affinity": "two CPUs", "OMP_NUM_THREADS": 2,
            "address_space_cap_bytes": MEMORY_CAP, "stdout_stderr_file_cap_bytes": OUTPUT_CAP,
        }
        write_report(args.report, report)

        for name, entry in (("AfterCount", SUITE / "AfterCount.bend"),
                            ("QueryFactor", SUITE / "QueryFactor.bend"),
                            ("consumer", SUITE / "consumer.bend")):
            result = run_check(name, entry, compiler, bun, 86400, args.evidence_dir)
            report["positive_checks"].append(result)
            write_report(args.report, report)
            if not result["passed"]:
                raise RuntimeError("positive check failed: " + name)

        controls = [
            ("wrong-promotion-rank", "legal_probe/Chess.bend",
             "Bool.and(pawn, Bool.or(U32.is_zero(rank), U32.is_eq(rank, 7)))",
             "Bool.and(pawn, Bool.or(U32.is_zero(rank), U32.is_eq(rank, 6)))",
             ("Factor.structural", "Emission.destinations", "Count.destinations", "actual_emission_count", "use_scan_after_closed_count")),
            ("wrong-en-passant-flag", "legal_probe/Chess.bend",
             "Bool.to_u32(Bool.and(pawn, U32.is_eq(dst, ep_sq)))",
             "Bool.to_u32(Bool.and(pawn, U32.is_eq(dst, 65)))",
             ("Factor.structural", "Emission.destinations", "Count.destinations", "actual_emission_count", "use_scan_after_closed_count")),
            ("dropped-duplicate-tail", "standalone/proofs/destination_factorization/Spec.bend",
             "bump(same(head,query),count(rest,query))",
             "bump(same(head,query),0n)",
             ("Count.block", "duplicate_tail_multiplicity", "use_duplicate_tail_multiplicity", "use_scan_after_closed_count")),
            ("disconnected-scan-consumer", "standalone/proofs/destination_factorization/consumer.bend",
             "QFactor.scan_after_closed_count(table,targets,src,pawn,ep,query,tail)",
             "{==}",
             ("use_scan_after_closed_count",)),
        ]
        for name, target, old, new, locations in controls:
            with tempfile.TemporaryDirectory(prefix="deepfin-scan-after-control-") as tmp:
                copied_engine = Path(tmp) / "engine"
                shutil.copytree(ENGINE, copied_engine, symlinks=True)
                target_path = copied_engine / target
                baseline = sha256(target_path)
                replace(target_path, old, new)
                mutated = sha256(target_path)
                entry = copied_engine / "standalone/proofs/destination_factorization/consumer.bend"
                result = run_check(name, entry, compiler, bun, 120, args.evidence_dir)
                result.update({
                    "mutation_target": target, "baseline_sha256": baseline,
                    "mutated_sha256": mutated, "mutation_anchor": old,
                    "replacement": new,
                    "rejected_at_expected_obligation": control_rejected(result, locations),
                    "expected_locations": list(locations),
                })
                report["negative_controls"].append(result)
                write_report(args.report, report)
                if not result["rejected_at_expected_obligation"]:
                    raise RuntimeError("negative control did not reject semantically: " + name)

        after = identity(paths)
        if after != before:
            raise RuntimeError("source files changed during qualification")
        final_pin = subprocess.run(pin_cmd, capture_output=True, text=True,
                                   check=True, timeout=30)
        if final_pin.stderr or final_pin.stdout != pin.stdout:
            raise RuntimeError("compiler identity changed during qualification")
        report["compiler_identity_unchanged"] = True
        report["source_unchanged"] = True
        report["gate"] = "PASS"
    except Exception as failure:
        report["failure"] = repr(failure)
        report["gate"] = "FAIL"
        raise
    finally:
        os.sched_setaffinity(0, original_affinity)
        report["finished_at_utc"] = utcnow()
        write_report(args.report, report)


if __name__ == "__main__":
    main()
