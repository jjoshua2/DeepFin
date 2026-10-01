"""Fail-closed qualification of the actual public-builder equality candidate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from ..table_preservation._validation import begin_report, require, write_report

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]


def load_closure():
    """Use the exact existing policy implementation, without modifying it."""
    path = SUITE.parent / "table_preservation/focused.py"
    spec = importlib.util.spec_from_file_location("generator_contract_closure", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.closure


def safe(result: dict) -> bool:
    return (
        result["exit_code"] == 0
        and not result["timed_out"]
        and result["raw_text"].strip() == "All terms check."
    )


def semantic_rejection(result: dict) -> bool:
    text = result["raw_text"]
    forbidden = (
        "no such file", "RangeError", "Maximum call stack", "more than once",
        "a decreasing self-call", "a defined name", "a parameter or field scrutinee",
        "out of memory", "OutOfMemory", "WARNING", "unsafe", "TODO",
    )
    return (
        result["exit_code"] == 1
        and not result["timed_out"]
        and "expected" in text
        and "observed" in text
        and bool(re.search(r"Location:\s*(?:[\w./]+\.)*public_builder_matches_recipe\b", text))
        and not any(word in text for word in forbidden)
    )


def invoke(bun: str, compiler: Path, entry: Path, seconds: int) -> dict:
    start = time.monotonic()
    command = [bun, "--smol", str(compiler / "bend2/main.ts"), str(entry), "--check-only"]
    try:
        process = subprocess.run(
            command, capture_output=True, timeout=seconds, check=False,
            env={**os.environ, "BEND_NO_TELEMETRY": "1", "TERM": "dumb"},
        )
        code = process.returncode
        raw = process.stdout + process.stderr
        timed_out = False
    except subprocess.TimeoutExpired as failure:
        code = None
        raw = (failure.stdout or b"") + (failure.stderr or b"")
        timed_out = True
    return {
        "command": command, "exit_code": code, "timed_out": timed_out,
        "wall_limit_seconds": seconds, "seconds": time.monotonic() - start,
        "raw_text": raw.decode("utf-8", errors="replace"),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
    }


def replace(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    require(source.count(old) == 1, "nonunique mutation anchor")
    path.write_text(source.replace(old, new))


def manifest(suite: Path) -> None:
    proof = (suite / "Closed.bend").read_text()
    consumer = (suite / "consumer.bend").read_text()
    require(
        re.findall(r"^def (\w+)\(", proof, re.MULTILINE)
        == ["public_builder_matches_recipe"], "candidate definition inventory",
    )
    require(not re.search(r"^law\s", proof, re.MULTILINE), "unproved law declaration")
    require("import ./Closed.bend as Closed" in consumer, "consumer proof import")
    require("Closed.public_builder_matches_recipe()" in consumer, "consumer proof call")
    proposition = "{Tables.build() == Init.run(17n,U64.zero(),128n,64n) : Array<U64>}"
    require(proposition in proof and proposition in consumer, "complete closed proposition")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    begin_report(args.report, "closed_builder_gate")
    report: dict = {
        "closed_builder_gate": "NOT_COMPLETED",
        "new_accepted_law_count": 0,
        "scope": "Actual Tables.build equality only; full generator contract remains open.",
    }
    try:
        compiler = args.compiler.resolve()
        bun = os.environ.get("BUN", "bun")
        pin = [bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)]
        before_pin = subprocess.run(pin, capture_output=True, text=True, check=True, timeout=30)
        require(not before_pin.stderr, "compiler identity warning")
        report["compiler_identity"] = before_pin.stdout
        manifest(SUITE)
        closure = load_closure()
        graph = closure(SUITE / "consumer.bend", ENGINE)
        paths = {ENGINE / p for p in graph}
        paths.update(p for p in SUITE.iterdir() if p.is_file())
        paths.update(SUITE.parent / p for p in (
            "table_preservation/focused.py", "table_preservation/_validation.py",
        ))
        paths.update(ENGINE / p for p in (
            "standalone/toolchain.json", "standalone/verify_compiler.js",
        ))
        def identities() -> dict[str, str]:
            return {
                str(path.relative_to(ENGINE)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(paths)
            }
        before = identities()
        report["source_sha256s"] = before
        write_report(args.report, report)
        consumer = invoke(bun, compiler, SUITE / "consumer.bend", 600)
        report["consumer"] = consumer
        write_report(args.report, report)
        require(safe(consumer), "closed consumer did not return exact safe checker success")
        controls: list[dict] = []
        report["controls"] = controls
        mutations = [
            (
                "public-builder-unused-slot-corruption", "standalone/Tables.bend",
                "  extras(64n, 0, tables(128n, 0, 512, Array.new(U64, 17n, U64.zero())))",
                "  Array.set(U64, extras(64n, 0, tables(128n, 0, 512, "
                "Array.new(U64, 17n, U64.zero()))), 131071, U64.from_u32(1))",
            ),
            (
                "false-seed-equality", "standalone/proofs/generator_contract/Closed.bend",
                "Init.run(17n,U64.zero(),128n,64n)",
                "Init.run(17n,U64.from_u32(1),128n,64n)",
            ),
        ]
        for name, target, old, new in mutations:
            with tempfile.TemporaryDirectory(prefix="closed-builder-control-") as directory:
                copied = Path(directory) / "engine"
                shutil.copytree(ENGINE, copied, symlinks=True)
                replace(copied / target, old, new)
                entry = copied / "standalone/proofs/generator_contract/Closed.bend"
                closure(entry, copied)
                result = invoke(bun, compiler, entry, 180)
                controls.append({
                    "name": name, "kind": "source semantic/refinement",
                    "rejected": semantic_rejection(result), "result": result,
                })
                write_report(args.report, report)
                require(semantic_rejection(result), "control was not an intended semantic rejection: " + name)
        require(before == identities(), "source drift")
        after_pin = subprocess.run(pin, capture_output=True, text=True, check=True, timeout=30)
        require(before_pin.stdout == after_pin.stdout and not after_pin.stderr, "compiler drift")
        report["closed_builder_gate"] = "PASS"
        report["checked_candidate_definition_count"] = 1
    except Exception as error:
        report["failure"] = type(error).__name__ + ": " + str(error)
        write_report(args.report, report)
        raise
    write_report(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
