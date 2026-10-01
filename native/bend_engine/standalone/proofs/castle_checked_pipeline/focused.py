"""Opt-in initialized castling producer/filter proof gate; no runtime code mutation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

LAWS = ["initialized_castle_producer_matches_checks", "initialized_castle_pipeline_matches_three_checks", "accepted_castle_pipeline_has_safe_stages"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def closure(entry: Path, engine: Path) -> dict[str, str]:
    """Inspect every path component before resolving imports, including ancestors."""
    result: dict[str, str] = {}

    def visit(path: Path) -> None:
        if any(part.is_symlink() for part in [path, *path.parents]):
            raise ValueError(f"Symlinked raw import path: {path}")
        path = Path(os.path.abspath(path))
        rel = path.relative_to(engine)
        if not (rel.as_posix().startswith("standalone/") or rel.as_posix() in {"legal_probe/Chess.bend", "bitboard_probe/Sliders.bend"}):
            raise ValueError(f"Out of proof scope: {rel}")
        cursor = engine
        for part in rel.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"Symlinked source: {cursor}")
        if not path.is_file() or path.resolve() != path:
            raise ValueError(f"Nonregular source: {path}")
        key = rel.as_posix()
        if key in result:
            return
        raw = path.read_bytes()
        result[key] = sha(raw)
        text = "\n".join(line.split("#", 1)[0] for line in raw.decode().splitlines())
        if "@unsafe" in text or "?" in text:
            raise ValueError(f"Unsafe dependency or proof hole: {key}")
        for imported in re.findall(r"^\s*import\s+(\S+)", text, re.MULTILINE):
            if imported == "Base":
                continue
            if not re.fullmatch(r"\.{1,2}/[A-Za-z0-9_/.]+\.bend", imported):
                raise ValueError(f"Foreign import: {imported}")
            visit(path.parent / imported)

    visit(entry)
    return result


def manifest(suite: Path) -> None:
    laws = re.findall(r"^law (\w+):", (suite / "LAWS.bend").read_text(), re.MULTILINE)
    p = (suite / "PROOF.bend").read_text()
    proofs = re.findall(r"^def Laws\.(\w+)\(", p, re.MULTILINE)
    if laws != LAWS or proofs != LAWS:
        raise ValueError("Incorrect law/proof inventory")
    if "import ./LAWS.bend as Laws" not in p:
        raise ValueError("Missing public law import")
    if "import ./PROOF.bend as Proof" not in (suite / "consumer.bend").read_text():
        raise ValueError("Missing proof body consumer import")


def invoke(bun: str, compiler: Path, entry: Path, limit: int) -> dict:
    start = time.monotonic()
    p = subprocess.run([bun, str(compiler / "bend2/main.ts"), str(entry)],
                       capture_output=True, timeout=limit, check=False,
                       env={**os.environ, "TERM": "dumb", "BEND_NO_TELEMETRY": "1"})
    raw = p.stdout + p.stderr
    return {"exit_code": p.returncode, "raw_text": raw.decode(), "sha256": sha(raw), "seconds": time.monotonic() - start}


def safe(r: dict) -> None:
    if r["exit_code"] != 0 or r["raw_text"].strip() != "All terms check.":
        raise AssertionError(str(r)[-4000:])


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise AssertionError(f"Nonunique mutation site in {path.name}: {old}")
    path.write_text(text.replace(old, new))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--controls-only", action="store_true", help="Partial developer check; never produces focused_gate=PASS")
    args = parser.parse_args()
    suite = Path(__file__).resolve().parent
    engine = suite.parents[2]
    compiler = args.compiler.resolve()
    bun = os.environ.get("BUN", "bun")
    pin = engine / "standalone/verify_compiler.js"
    identity = subprocess.check_output([bun, str(pin), str(compiler)], text=True)
    manifest(suite)
    sources = closure(suite / "consumer.bend", engine)
    sources.update(closure(suite / "probe.bend", engine))
    # Examples are separately checked small witnesses, not expensive closed table reductions.
    if (suite / "Examples.bend").exists():
        sources.update(closure(suite / "Examples.bend", engine))
    for path in [*suite.glob("*.py"), *suite.glob("*.js"), *suite.glob("*.md"), pin, engine / "standalone/toolchain.json"]:
        sources[path.relative_to(engine).as_posix()] = sha(path.read_bytes())
    consumer = None if args.controls_only else invoke(bun, compiler, suite / "consumer.bend", 1500)
    if consumer is not None:
        safe(consumer)
    examples = None
    if not args.controls_only and (suite / "Examples.bend").exists():
        examples = invoke(bun, compiler, suite / "Examples.bend", 180)
        safe(examples)
    controls = []
    semantic = [
        ("actual-ignores-start", "legal_probe/Chess.bend", "Wire.bend",
         "retain_move(Bool.or(start_check, transit_check), m, acc)", "retain_move(transit_check, m, acc)", "start"),
        ("transit-certificate-uses-unchecked-false", "standalone/proofs/castle_checked_pipeline/Wire.bend", "Wire.bend",
         "tail,first,middle,a,t)", "tail,first,False{},a,t)", "producer"),
        ("final-certificate-uses-unchecked-false", "standalone/proofs/castle_checked_pipeline/Wire.bend", "Wire.bend",
         "case False{}: destination(b,white,ks,table,Nil{},last,f)", "case False{}: destination(b,white,ks,table,Nil{},False{},f)", "final_list"),
        ("actual-ignores-destination", "legal_probe/Chess.bend", "Wire.bend",
         "(table, retain_move(check, m, acc))", "(table, retain_move(False{}, m, acc))", "destination"),
        ("actual-alters-returned-table", "legal_probe/Chess.bend", "Wire.bend",
         "(table, retain_move(Bool.or(start_check, transit_check), m, acc))", "(Array.set(U64, table, 0, U64.zero()), retain_move(Bool.or(start_check, transit_check), m, acc))", "start"),
        ("independent-model-drops-retained-move", "standalone/proofs/castle_checked_pipeline/Spec.bend", "Wire.bend",
         "case False{}: Con{m,tail}", "case False{}: Nil{}", "retain"),
        ("decision-omits-final-rejection", "standalone/proofs/castle_checked_pipeline/Decision.bend", "Decision.bend",
         "here: E.member(m,S.keep(Bool.or(Bool.or(a,b),c),h,Nil{}))", "here: E.member(m,S.keep(Bool.or(a,b),h,Nil{}))", "all_false"),
    ]
    for name, file, entry, old, new, location in semantic:
        print("Checking source rejection:", name, flush=True)
        with tempfile.TemporaryDirectory(prefix="castle-pipeline-semantic-") as t:
            copied = Path(t) / "engine"
            shutil.copytree(engine, copied, symlinks=True)
            replace(copied / file, old, new)
            r = invoke(bun, compiler, copied / "standalone/proofs/castle_checked_pipeline" / entry, 180)
            raw = r["raw_text"]
            prohibited = ["no such file", "RangeError", "Maximum call stack", "more than once", "a decreasing self-call", "a defined name", "Segmentation fault"]
            if not (r["exit_code"] == 1 and "expected" in raw and "observed" in raw
                    and re.search(r"Location:\s*" + re.escape(location) + r"\b", raw)
                    and not any(p in raw for p in prohibited)):
                raise AssertionError(f"Not an intended semantic rejection {name}: {r}")
            # Keep original diagnostics; these target new control-flow/certificate bodies.
            controls.append({"name": name, "kind": "source semantic/refinement", "entry": entry, "rejected": True, **r})
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.with_suffix(".partial.json").write_text(json.dumps({"focused_gate": "INCOMPLETE", "completed_controls": controls}, indent=2) + "\n")
    for kind in ["missing-law", "missing-proof", "missing-laws-import", "missing-consumer-proof", "hole", "unsafe", "foreign", "symlink"]:
        with tempfile.TemporaryDirectory(prefix="castle-pipeline-policy-") as t:
            copied = Path(t) / "engine"
            shutil.copytree(engine, copied, symlinks=True)
            s = copied / "standalone/proofs/castle_checked_pipeline"
            if kind == "missing-law":
                replace(s / "LAWS.bend", "law " + LAWS[0] + ":", "def missing:")
            elif kind == "missing-proof":
                replace(s / "PROOF.bend", "def Laws." + LAWS[0] + "(", "def missing(")
            elif kind == "missing-laws-import":
                replace(s / "PROOF.bend", "import ./LAWS.bend as Laws\n", "")
            elif kind == "missing-consumer-proof":
                replace(s / "consumer.bend", "import ./PROOF.bend as Proof\n", "")
            elif kind == "symlink":
                f = s / "Wire.bend"
                f.rename(s / "Wire.original")
                f.symlink_to("Wire.original")
            else:
                with (s / "Wire.bend").open("a") as f:
                    f.write({"hole": "\n?missing\n", "unsafe": "\n@unsafe\ndef fake(): 0\n", "foreign": '\nimport "./oracle.c"\n'}[kind])
            try:
                manifest(s)
                closure(s / "consumer.bend", copied)
            except (AssertionError, ValueError, OSError):
                controls.append({"name": kind, "kind": "manifest/import policy", "rejected": True})
            else:
                raise AssertionError("Policy mutation accepted: " + kind)
    try:
        safe({"exit_code": 0, "raw_text": "All terms check.\nWARNING: unsafe dependency"})
    except AssertionError:
        controls.append({"name": "warning-on-zero-exit", "kind": "synthetic output-wrapper unit; not compiler execution", "rejected": True})
    else:
        raise AssertionError("Unsafe warning output accepted")
    for path, digest in sources.items():
        if sha((engine / path).read_bytes()) != digest:
            raise AssertionError("Source drift: " + path)
    if subprocess.check_output([bun, str(pin), str(compiler)], text=True) != identity:
        raise AssertionError("Compiler identity drift")
    assert len(controls) == 16
    report = {"focused_gate": "CONTROLS_ONLY" if args.controls_only else "PASS", "new_laws": 3,
              "new_controls": 16, "consumer": consumer, "examples": examples,
              "negative_controls": controls, "source_sha256s": dict(sorted(sources.items())),
              "compiler_identity": identity, "complete_aggregate_run": False,
              "scope": "Single guarded route: actual castle_side followed by full filter_legal, threaded returned array; not full optimized legal_moves or attack-direction equivalence."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"source_sha256s", "negative_controls", "consumer", "examples"}}, indent=2))


if __name__ == "__main__":
    main()
