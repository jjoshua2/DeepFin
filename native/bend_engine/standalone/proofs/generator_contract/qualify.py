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
import signal
import subprocess
import tempfile
import threading
import time

from ..table_preservation._validation import begin_report, require, write_report

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
DEFAULT_CONSUMER_TIMEOUT_SECONDS = 86_400
MAX_CONSUMER_LOG_BYTES = 16 * 1024 * 1024


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
        not result.get("output_limit_exceeded", False)
        and not result.get("output_truncated", False)
        and result["exit_code"] == 0
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
        not result.get("output_limit_exceeded", False)
        and not result.get("output_truncated", False)
        and result["exit_code"] == 1
        and not result["timed_out"]
        and "expected" in text
        and "observed" in text
        and bool(re.search(r"Location:\s*(?:[\w./]+\.)*public_builder_matches_recipe\b", text))
        and not any(word in text for word in forbidden)
    )


def invoke(bun: str, compiler: Path, entry: Path, seconds: int, log_path: Path | None = None) -> dict:
    start = time.monotonic()
    command = [bun, "--smol", str(compiler / "bend2/main.ts"), str(entry), "--check-only"]
    if seconds <= 0:
        raise ValueError("consumer timeout must be positive")
    owned_log = log_path is None
    if log_path is None:
        fd, name = tempfile.mkstemp(prefix="bend-consumer-", suffix=".log")
        os.close(fd)
        log_path = Path(name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    output_limit_exceeded = threading.Event()
    output_digest = hashlib.sha256()
    output_size = 0
    process: subprocess.Popen | None = None
    reader: threading.Thread | None = None
    reader_errors: list[BaseException] = []
    reader_failed = threading.Event()
    original_sigterm = None
    completed = False

    def stop_process(*, force: bool = False) -> None:
        if process is None:
            return
        running = process.poll() is None
        if not running and not force:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            elif running:
                process.terminate()
        except ProcessLookupError:
            pass
        if running:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        if process.poll() is None:
            process.wait()

    def interrupted(signum, frame) -> None:
        del signum, frame
        raise KeyboardInterrupt("proof checker interrupted")

    warning = threading.Timer(
        3600,
        lambda: print(
            "WARNING: proof checker has run for one hour; elapsed time is not semantic progress",
            file=__import__("sys").stderr,
            flush=True,
        ),
    ) if seconds > 3600 else None
    try:
        if threading.current_thread() is threading.main_thread():
            original_sigterm = signal.signal(signal.SIGTERM, interrupted)
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env={**os.environ, "BEND_NO_TELEMETRY": "1", "TERM": "dumb"},
                start_new_session=os.name == "posix",
            )

            def drain_output() -> None:
                nonlocal output_size
                assert process is not None
                assert process.stdout is not None
                try:
                    with process.stdout:
                        while True:
                            chunk = process.stdout.read(64 * 1024)
                            if not chunk:
                                break
                            output_digest.update(chunk)
                            allowed = max(0, MAX_CONSUMER_LOG_BYTES - output_size)
                            if allowed:
                                log.write(chunk[:allowed])
                                log.flush()
                                output_size += min(len(chunk), allowed)
                            if len(chunk) > allowed and not output_limit_exceeded.is_set():
                                output_limit_exceeded.set()
                                stop_process(force=True)
                except BaseException as error:
                    reader_errors.append(error)
                    reader_failed.set()
                    stop_process(force=True)

            reader = threading.Thread(target=drain_output, name="bend-proof-log-drain")
            reader.start()
            if warning is not None:
                warning.start()
            deadline = start + seconds
            while (process.poll() is None and not output_limit_exceeded.is_set()
                   and not reader_failed.is_set()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    continue
            if timed_out or output_limit_exceeded.is_set() or reader_failed.is_set():
                stop_process(force=True)
            reader.join(timeout=min(5, max(0, deadline - time.monotonic())))
            if reader.is_alive():
                timed_out = True
                stop_process(force=True)
                reader.join(timeout=5)
            if reader.is_alive():
                raise RuntimeError("proof checker output pipe did not close after process-group cleanup")
            if reader_errors:
                raise RuntimeError("proof checker output capture failed") from reader_errors[0]
            completed = True
        raw = log_path.read_bytes()[-(1024 * 1024):]
        if owned_log:
            log_path.unlink(missing_ok=True)
    finally:
        if process is not None:
            stop_process(force=not completed)
        if reader is not None and reader.is_alive():
            reader.join(timeout=10)
        if warning is not None:
            warning.cancel()
        if original_sigterm is not None:
            signal.signal(signal.SIGTERM, original_sigterm)
    code = None if timed_out else process.returncode if process is not None else None
    return {
        "command": command, "exit_code": code,
        "timed_out": timed_out, "output_limit_exceeded": output_limit_exceeded.is_set(),
        "output_truncated": output_size > len(raw),
        "wall_limit_seconds": seconds, "seconds": time.monotonic() - start,
        "log_path": str(log_path) if not owned_log else None,
        "log_bytes": log_path.stat().st_size if not owned_log else 0,
        "raw_text": raw.decode("utf-8", errors="replace"),
        "output_sha256": output_digest.hexdigest(),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--consumer-timeout-seconds", type=int,
        default=DEFAULT_CONSUMER_TIMEOUT_SECONDS,
        help="positive checker wall limit (default: 86400 seconds / 24 hours)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require(args.consumer_timeout_seconds > 0, "consumer timeout must be positive")
    begin_report(args.report, "closed_builder_gate")
    report: dict = {
        "closed_builder_gate": "NOT_COMPLETED",
        "new_accepted_law_count": 0,
        "scope": "Actual Tables.build equality only; full generator contract remains open.",
        "default_consumer_timeout_seconds": DEFAULT_CONSUMER_TIMEOUT_SECONDS,
        "effective_consumer_timeout_seconds": args.consumer_timeout_seconds,
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
        consumer = invoke(
            bun, compiler, SUITE / "consumer.bend", args.consumer_timeout_seconds,
            args.report.parent / "closed-builder-consumer.log",
        )
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
