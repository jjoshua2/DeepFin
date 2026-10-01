"""Opt-in full stage-composition check and narrowly classified negative controls."""
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

LAW = "initialized_castling_stage_check_matches_geometry"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_closure(entry: Path, engine: Path) -> dict[str, str]:
    seen: dict[str, str] = {}

    def visit(path: Path) -> None:
        absolute = path.absolute()
        resolved = path.resolve(strict=True)
        if absolute != resolved or not resolved.is_file():
            raise ValueError(f"Nonregular or symlinked source: {path}")
        rel = resolved.relative_to(engine).as_posix()
        if not (rel.startswith("standalone/") or rel in {
            "legal_probe/Chess.bend", "bitboard_probe/Sliders.bend"
        }):
            raise ValueError(f"Out-of-scope import: {rel}")
        if rel in seen:
            return
        raw = resolved.read_bytes()
        seen[rel] = digest(raw)
        code = "\n".join(line.split("#", 1)[0] for line in raw.decode().splitlines())
        if "@unsafe" in code or "?" in code:
            raise ValueError(f"Unsafe source or proof hole: {rel}")
        for imported in re.findall(r"^\s*import\s+(\S+)", code, re.MULTILINE):
            if imported == "Base":
                continue
            if not re.fullmatch(r"\.{1,2}/[A-Za-z0-9_/.]+\.bend", imported):
                raise ValueError(f"Foreign import: {imported}")
            visit((resolved.parent / imported).resolve())

    visit(entry)
    return dict(sorted(seen.items()))


def check_manifest(suite: Path) -> None:
    laws = re.findall(r"^law (\w+):", (suite / "LAWS.bend").read_text(), re.MULTILINE)
    proofs = re.findall(r"^def Laws\.(\w+)\(", (suite / "PROOF.bend").read_text(), re.MULTILINE)
    if laws != [LAW] or proofs != [LAW]:
        raise ValueError("Missing, extra, or duplicate public law/proof")
    if "import ./PROOF.bend as Proof" not in (suite / "consumer.bend").read_text():
        raise ValueError("Consumer does not import proof bodies")


def invoke(bun: str, compiler: Path, entry: Path, timeout: int) -> dict:
    started = time.monotonic()
    result = subprocess.run(
        [bun, str(compiler / "bend2/main.ts"), str(entry)],
        capture_output=True, timeout=timeout,
        env={**os.environ, "BEND_NO_TELEMETRY": "1", "TERM": "dumb"},
        check=False,
    )
    raw = result.stdout + result.stderr
    return {"exit_code": result.returncode, "raw_text": raw.decode(),
            "sha256": digest(raw), "seconds": time.monotonic() - started}


def require_safe(result: dict) -> None:
    if result["exit_code"] != 0 or result["raw_text"].strip() != "All terms check.":
        raise AssertionError(str(result)[-5000:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    suite = Path(__file__).resolve().parent
    engine = suite.parents[2]
    compiler = args.compiler.resolve()
    bun = os.environ.get("BUN", "bun")
    check_manifest(suite)
    before = source_closure(suite / "consumer.bend", engine)
    consumer = invoke(bun, compiler, suite / "consumer.bend", 1500)
    require_safe(consumer)
    controls = []
    mutations = [
        ("omit-consistency", "good: {B.valid(b) == True{} : Bool}", "good: {True{} == True{} : Bool}"),
        ("omit-valid-side", "turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}", "turn: {0 == 0 : U32}"),
        ("omit-initial-singleton", "one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}", "one: {U64.zero() == U64.zero() : U64}"),
        ("omit-producer-guard", "guard: {E.guard(b,ks) == True{} : Bool}", "guard: {True{} == True{} : Bool}"),
        ("check-other-color", "Check.color(K.stage_board(b,white,ks,stage),white)", "Check.color(K.stage_board(b,white,ks,stage),Bool.not(white))"),
    ]
    for name, old, new in mutations:
        with tempfile.TemporaryDirectory(prefix="castle-stage-control-") as directory:
            copied_engine = Path(directory) / "engine"
            shutil.copytree(engine, copied_engine, symlinks=True)
            target = copied_engine / "standalone/proofs/castle_stage_geometry/Inputs.bend"
            text = target.read_text()
            if text.count(old) != 1:
                raise AssertionError(f"Mutation site not unique: {name}")
            target.write_text(text.replace(old, new))
            rejection = invoke(bun, compiler, target, 180)
            output = rejection["raw_text"]
            forbidden = ("no such file", "RangeError", "Maximum call stack", "more than once", "a decreasing self-call")
            if not (rejection["exit_code"] == 1 and "expected" in output and "observed" in output
                    and re.search(r"Location:.*singleton", output) and not any(s in output for s in forbidden)):
                raise AssertionError(f"Not an intended refinement rejection: {name}: {rejection}")
            controls.append({"name": name, "kind": "source semantic/refinement", "rejected": True,
                             "entry": "Inputs.bend", **rejection})
    # This is a wrapper unit, not a Bend compiler execution.
    try:
        require_safe({"exit_code": 0, "raw_text": "All terms check.\nWARNING: unsafe dependency"})
    except AssertionError:
        controls.append({"name": "reject-warning-on-zero-exit", "kind": "synthetic output-wrapper unit", "rejected": True})
    else:
        raise AssertionError("Warning output accepted")
    if source_closure(suite / "consumer.bend", engine) != before:
        raise AssertionError("Source drift during qualification")
    report = {"focused_gate": "PASS", "new_laws": 1, "new_controls": 6,
              "consumer": consumer, "negative_controls": controls, "source_sha256s": before,
              "runner_sha256": digest(Path(__file__).read_bytes()),
              "scope": "Initialized independent target-centred geometric check at each castling stage, not a proof that the result is safe/false."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"source_sha256s", "negative_controls", "consumer"}}, indent=2))


if __name__ == "__main__":
    main()
