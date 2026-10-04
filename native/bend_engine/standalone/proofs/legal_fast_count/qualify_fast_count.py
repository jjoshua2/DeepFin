"""Fail-closed qualification of actual fast and prepare occurrence counts."""
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
import sys
import tempfile
import time

from ..generator_contract.qualify import load_closure

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
PROJECT = ENGINE.parents[1]
OUTPUT_CAP = 16 * 1024 * 1024
MEMORY_CAP = 6 * 1024 * 1024 * 1024
BASE = "066fef8640770398daf6e27af804f2374856c066"
BASE_TREE = "c13eb9ba7d2fee34e36a530d49071c15f7467511"
BRANCH = "proof/bend-fast-count-20261004"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    home = str(Path.home())
    def redact(value):
        if isinstance(value, str):
            return value.replace(home, "<HOME>")
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        return value
    temporary.write_text(json.dumps(redact(report), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def limits() -> None:
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
    started = now()
    start = time.monotonic()
    timed_out = False
    code = None
    try:
        if any(p.exists() for p in (stdout_path, stderr_path, time_path)):
            raise RuntimeError("refusing to overwrite checker evidence")
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr,
                                     timeout=seconds + 15, check=False,
                                     preexec_fn=limits)
        code = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    elapsed = time.monotonic() - start
    stdout = stdout_path.read_bytes() if stdout_path.exists() else b""
    stderr = stderr_path.read_bytes() if stderr_path.exists() else b""
    raw = stdout + stderr
    text = raw.decode("utf-8", errors="replace")
    time_text = time_path.read_text(errors="replace") if time_path.exists() else ""
    time_text = time_text.replace(str(Path.home()), "<HOME>")
    if time_path.exists():
        time_path.write_text(time_text)
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", time_text)
    result = {
        "name": name, "command": command, "started_at_utc": started,
        "finished_at_utc": now(), "elapsed_wall_seconds": elapsed,
        "wall_timeout_seconds": seconds, "exit_code": code, "timed_out": timed_out,
        "stdout_bytes": len(stdout), "stderr_bytes": len(stderr),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "raw_output_sha256": hashlib.sha256(raw).hexdigest(),
        "max_rss_kib": int(rss.group(1)) if rss else None,
        "max_output_bytes": OUTPUT_CAP, "address_space_cap_bytes": MEMORY_CAP,
        "cpu_affinity": "0,1", "raw_text": text,
    }
    result["passed"] = (
        code == 0 and not timed_out and text == "All terms check.\n"
        and len(raw) <= OUTPUT_CAP and result["max_rss_kib"] is not None
        and result["max_rss_kib"] <= MEMORY_CAP // 1024
    )
    return result


def source_hashes(paths: set[str]) -> dict[str, str]:
    return {path: sha256(ENGINE / path) for path in sorted(paths)}


def rejected_at(result: dict, locations: tuple[str, ...]) -> bool:
    text = result["raw_text"]
    forbidden = ("out of memory", "outofmemory", "warning", "unsafe",
                 "no such file", "maximum call stack", "timeout",
                 "syntax", "parse error", "not found", "more than once",
                 "consumed", "resource")
    matches = re.findall(r"^Location:\s*([\w./]+)\s*$", text, re.MULTILINE)
    expected = re.search(r"^- expected\s*:\s*(.+)$", text, re.MULTILINE)
    observed = re.search(r"^- observed\s*:\s*(.+)$", text, re.MULTILINE)
    return (
        result["exit_code"] == 1 and not result["timed_out"]
        and len(matches) == 1 and matches[0] in locations
        and text.count("Error:") == 1
        and expected is not None and observed is not None
        and bool(expected.group(1).strip()) and bool(observed.group(1).strip())
        and expected.group(1).strip() != observed.group(1).strip()
        and not any(word in text.lower() for word in forbidden)
        and not re.search(r"\b(?:SIG[A-Z0-9]+|signal \d+)\b", text)
        and result["stdout_bytes"] + result["stderr_bytes"] < OUTPUT_CAP
        and result["max_rss_kib"] is not None
        and result["max_rss_kib"] <= MEMORY_CAP // 1024
    )


def mutate_declaration(path: Path, declaration: str, old: str, new: str) -> None:
    if path.is_symlink():
        raise RuntimeError("mutation target must be an isolated regular file")
    text = path.read_text()
    marker = "def " + declaration + "("
    if text.count(marker) != 1:
        raise RuntimeError("declaration not unique: " + declaration)
    start = text.index(marker)
    end = text.find("\ndef ", start + len(marker))
    if end < 0:
        end = len(text)
    body = text[start:end]
    if body.count(old) != 1:
        raise RuntimeError("declaration-local mutation anchor not unique")
    updated = text[:start] + body.replace(old, new, 1) + text[end:]
    path.write_text(updated)


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
    report = {
        "gate": "NOT_COMPLETED",
        "scope": "Actual Chess.filter_fast and Chess.filter_prepare full-Ply occurrence counts, arbitrary candidate and tail multiplicity, and exact input table preservation. Controls cover only their stated local mutations. No fast/full equivalence or chess safety claim.",
        "remote_base_commit": BASE,
        "local_base_commit": None,
        "local_base_tree": None,
        "local_base_alias_note": "Exact PR1018 HEAD and tree required; every reused dependency is compared with its Git blob at that base.",
        "branch": BRANCH,
        "compiler_pin": {},
        "source_sha256s": {},
        "positive_checks": [],
        "negative_controls": [],
    }
    affinity = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, set(sorted(affinity)[:2]))
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
        paths.update(support)
        before = source_hashes(paths)
        report["qualification_support_sha256s"] = source_hashes(support)
        report["source_sha256s"] = before
        report["local_base_commit"] = subprocess.run(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        report["local_base_tree"] = subprocess.run(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD^{tree}"],
            capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        if report["local_base_commit"] != BASE or report["local_base_tree"] != BASE_TREE:
            raise RuntimeError("publication qualification requires exact PR1018 HEAD/tree")
        listing = subprocess.run(
            ["git", "-C", str(PROJECT), "ls-tree", "-r", BASE],
            capture_output=True, text=True, check=True, timeout=30).stdout
        base_blobs = {}
        for line in listing.splitlines():
            metadata, path = line.split("\t", 1)
            mode, kind, digest = metadata.split()
            if kind == "blob":
                base_blobs[path] = digest
        reused = {}
        for path in sorted(paths):
            if path.startswith("standalone/proofs/legal_fast_count/"):
                continue
            raw = (ENGINE / path).read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            remote_path = "native/bend_engine/" + path
            if base_blobs.get(remote_path) != blob:
                raise RuntimeError("dependency differs from intended base: " + path)
            reused[path] = blob
        report["reused_dependency_git_blobs"] = reused
        report["reused_dependencies_match_exact_base"] = True
        report["remote_base_tree"] = BASE_TREE
        report["prior_evidence_sha256s"] = prior_evidence
        report["source_identity_sha256"] = hashlib.sha256(
            json.dumps(before, sort_keys=True).encode()).hexdigest()
        report["resource_limits"] = {
            "positive_wall_timeout_seconds": 86400,
            "negative_wall_timeout_seconds": 120, "cpu_affinity": "two CPUs",
            "OMP_NUM_THREADS": 2, "address_space_cap_bytes": MEMORY_CAP,
            "stdout_stderr_file_cap_bytes": OUTPUT_CAP,
        }
        write_report(args.report, report)

        result = run_check("consumer", SUITE / "consumer.bend", compiler, bun,
                           86400, args.evidence_dir)
        report["positive_checks"].append(result)
        write_report(args.report, report)
        if not result["passed"]:
            raise RuntimeError("positive consumer check failed")

        controls = [('dropped-unchecked', 'FastCount.bend', 'fast_step_count', 'case False{}: PlySpec.bump(hit,total)', 'case False{}: total', ('FastCount.fast_head_count',)), ('kept-rejected-checked', 'FastCount.bend', 'fast_step_count', 'case True{}: Count.retain_count(reject,hit,total)', 'case True{}: PlySpec.bump(hit,total)', ('FastCount.fast_head_count',)), ('duplicate-count-one', 'consumer.bend', 'duplicate_unchecked_occurrences', ' == 2n : Nat}:', ' == 1n : Nat}:', ('duplicate_unchecked_occurrences',)), ('dropped-tail-count', 'consumer.bend', 'actual_fast_occurrence_count', 'Fast.fast_count_fold(xs,c,b,sensitive,query,PlySpec.count(tail,query)) : Nat}:', 'Fast.fast_count_fold(xs,c,b,sensitive,query,0n) : Nat}:', ('actual_fast_occurrence_count',)), ('removed-production-result', 'consumer.bend', 'actual_fast_occurrence_count', 'After.pair_count(Chess.filter_fast(xs,b,sensitive,(Observe.pack(c),tail)),query) ==', 'PlySpec.count(tail,query) ==', ('actual_fast_occurrence_count',)), ('omitted-kings-mask', 'FastCount.bend', 'choose_count', 'U64.or(rays,Chess.get_kings(b))', 'rays', ('FastCount.choose_spec_count',)), ('wrong-prepare-check', 'FastCount.bend', 'prepare_count', 'FilterSpec.initial(c,b)', 'Bool.not(FilterSpec.initial(c,b))', ('FastCount.actual_prepare_count',))]
        for name, filename, declaration, old, new, locations in controls:
            target = "standalone/proofs/legal_fast_count/" + filename
            with tempfile.TemporaryDirectory(prefix="deepfin-fast-count-control-") as tmp:
                copy_engine = Path(tmp) / "engine"
                shutil.copytree(ENGINE, copy_engine, symlinks=True)
                target_path = copy_engine / target
                baseline = sha256(target_path)
                mutate_declaration(target_path, declaration, old, new)
                mutated = sha256(target_path)
                entry = copy_engine / "standalone/proofs/legal_fast_count/consumer.bend"
                result = run_check(name, entry, compiler, bun, 120, args.evidence_dir)
                result.update({
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
        report["evidence_sha256s"] = {p.name: sha256(p) for p in args.evidence_dir.iterdir() if p.is_file()}
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
