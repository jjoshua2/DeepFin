"""Fail-closed helpers for the opt-in slider-reversal checks."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

class InvalidEvidence(ValueError):
    """An observation does not meet the declared verification contract."""

def require(ok: bool, message: str) -> None:
    if not ok:
        raise InvalidEvidence(message)

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def command(args: list[str], timeout: int = 180) -> dict:
    started = time.monotonic()
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False,
                       env={**os.environ, "BEND_NO_TELEMETRY": "1", "TERM": "dumb"})
    return {"args": args, "exit_code": p.returncode, "stdout": p.stdout,
            "stderr": p.stderr, "seconds": time.monotonic()-started}

def success(r: dict) -> str:
    require(r["exit_code"] == 0 and not r["stderr"], str(r)[-5000:])
    return r["stdout"]

def safe(r: dict) -> None:
    require(r["exit_code"] == 0 and (r["stdout"]+r["stderr"]).strip() == "All terms check.", str(r)[-5000:])

def atomic_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(report, f, indent=2); f.write("\n"); name = f.name
    os.replace(name, path)

def closure(entry: Path, engine: Path, seen: dict[str, str] | None = None) -> dict[str,str]:
    if seen is None:
        seen = {}
    path = Path(os.path.abspath(entry))
    require(path.is_file() and path.resolve() == path, "Missing/nonregular/symlinked source: " + str(path))
    rel = path.relative_to(engine).as_posix()
    require(rel.startswith("standalone/") or rel in {"legal_probe/Chess.bend", "bitboard_probe/Sliders.bend"}, "Escaped proof scope: " + rel)
    if rel in seen:
        return seen
    seen[rel] = sha(path.read_bytes())
    code = "\n".join(line.split("#",1)[0] for line in path.read_text().splitlines())
    require("@unsafe" not in code and "?" not in code, "Unsafe/hole dependency: " + rel)
    for name in re.findall(r"^\s*import\s+(\S+)",code,re.MULTILINE):
        if name == "Base":
            continue
        require(re.fullmatch(r"\.{1,2}/[A-Za-z0-9_/.]+\.bend",name) is not None, "Foreign dependency: " + name)
        closure(path.parent/name,engine,seen)
    return seen
