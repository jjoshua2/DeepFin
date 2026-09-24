#!/usr/bin/env python3
"""Re-run a pinned BT4 supervisor's full strict CPU verifier for one saved bank.

This never starts a worker or an inference session. A passing receipt may be
used to qualify an own-game teacher source; it does not qualify throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

STATUS = "PASS_INDEPENDENT_BT4_ORDINARY_BANK_AUDIT"
# Frozen supervisors whose verify_bank performs board/history re-encoding,
# legal-policy checks and natural/rule50 Syzygy terminal adjudication.
REVIEWED_FULL_STRICT_VERIFIERS = frozenset({
    "b1244e59bd703a0294dc23e637c815f9114968f10c7c6ccb51fe71bc4766d277",  # ordinary pilot
    "aa2e16ab1a78973f46839bc6228c4d783ceb6097db55f9f5a5d576fec3103d86",  # ordinary retry
    "379dc054e06bb831c7a50824207740a43bdd726f2497790fe1d065450add240a",  # parallel screen
})


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def bank_identity(bank: Path) -> dict[str, Any]:
    """Refuse a bank changed while its strict verifier was reading it."""
    records = []
    for path in (bank, *sorted(bank.rglob("*"))):
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode),
                f"unsupported bank entry: {path}")
        records.append((path.relative_to(bank).as_posix(), info.st_mode,
                        info.st_dev, info.st_ino, info.st_size,
                        info.st_mtime_ns, info.st_ctime_ns))
    return {"tree_sha256": hashlib.sha256(json.dumps(records).encode()).hexdigest(),
            "entries": len(records)}


def audit_bank(*, auditor_sha256: str, plan_path: Path, plan_sha256: str, verifier_path: Path,
               verifier_sha256: str, stage_name: str, bank: Path,
               terminal_path: Path, terminal_sha256: str,
               output: Path) -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "hide CUDA for CPU audit")
    plan_path = plan_path.resolve(strict=True)
    verifier_path = verifier_path.resolve(strict=True)
    bank = bank.resolve(strict=True)
    terminal_path = terminal_path.resolve(strict=True)
    output = output.resolve()
    auditor_path = Path(__file__).resolve(strict=True)
    require(sha_file(auditor_path) == auditor_sha256, "auditor source SHA differs")
    require(not output.exists() and not output.with_name(output.name + ".writing").exists(),
            "audit output must be fresh")
    require(not output.is_relative_to(bank), "audit receipt must be outside the bank")
    require(sha_file(plan_path) == plan_sha256, "plan SHA differs")
    require(sha_file(verifier_path) == verifier_sha256, "verifier SHA differs")
    require(verifier_sha256 in REVIEWED_FULL_STRICT_VERIFIERS,
            "verifier source lacks a reviewed full-strict profile")
    require(sha_file(terminal_path) == terminal_sha256, "terminal SHA differs")
    plan = read_json(plan_path)
    terminal = read_json(terminal_path)
    require(plan.get("supervisor_sha256") == verifier_sha256,
            "plan does not pin this verifier")
    stages = [stage for stage in plan.get("stages", []) if stage.get("name") == stage_name]
    require(len(stages) == 1, "stage absent or ambiguous")
    stage = stages[0]
    stage_dir = Path(plan["output_root"]).resolve() / stage_name
    require(bank == stage_dir / "bank" and terminal_path == stage_dir / "terminal.json",
            "bank or terminal is outside the planned stage")
    require(terminal.get("plan_sha256") == plan_sha256
            and terminal.get("stage") == stage_name
            and terminal.get("status", "").startswith("PASS_")
            and terminal.get("returncode") == 0
            and type(terminal.get("accepted_rows")) is int
            and terminal["accepted_rows"] > 0,
            "stage terminal is not a completed verified worker")
    summary_path = bank / "summary.json"
    summary_sha256 = sha_file(summary_path)
    require(terminal.get("summary_sha256") == summary_sha256,
            "stage terminal does not pin the current bank summary")
    summary = read_json(summary_path)
    provider_path = bank / "provider_proof.json"
    provider_sha256 = sha_file(provider_path)
    require(summary.get("provider_proof_sha256") == provider_sha256
            and terminal.get("provider_proof_sha256") == provider_sha256,
            "provider proof is not pinned by the bank and terminal")
    initial_identity = bank_identity(bank)

    writing = output.with_name(output.name + ".writing")
    with writing.open("x") as stream:
        stream.write(json.dumps({"status": "INCOMPLETE", "bank": str(bank)}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    spec = importlib.util.spec_from_file_location("bt4_pinned_fullstrict_verifier", verifier_path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(module.__file__ is not None
            and Path(module.__file__).resolve() == verifier_path
            and callable(getattr(module, "verify_bank", None)),
            "verifier origin/function differs")
    facts = module.verify_bank(plan, stage, bank)
    require(isinstance(facts, dict), "verifier did not return facts")
    for key in ("accepted_rows", "attempted_plies", "completed_games", "discarded_games",
                "sixman_games", "natural_games", "summary_sha256",
                "provider_proof_sha256", "pinned_source_origins"):
        require(key in facts, f"independent verifier omitted required fact: {key}")
    require(all(key in terminal and terminal[key] == value for key, value in facts.items()),
            "independent verifier facts differ from stage terminal")
    require(facts["summary_sha256"] == summary_sha256
            and facts["provider_proof_sha256"] == provider_sha256
            and facts["accepted_rows"] == summary["rows_emitted"]
            and facts["accepted_rows"] > 0
            and bool(facts["pinned_source_origins"]),
            "verified facts do not bind bank rows and source origins")
    require(sha_file(plan_path) == plan_sha256
            and sha_file(auditor_path) == auditor_sha256
            and sha_file(verifier_path) == verifier_sha256
            and sha_file(terminal_path) == terminal_sha256
            and sha_file(summary_path) == summary_sha256
            and sha_file(provider_path) == provider_sha256
            and bank_identity(bank) == initial_identity,
            "audit input changed during verification")
    receipt = {
        "status": STATUS, "bank": str(bank), "summary_sha256": summary_sha256,
        "provider_proof_sha256": provider_sha256,
        "bank_identity": initial_identity,
        "accepted_rows": facts["accepted_rows"], "stage": stage_name,
        "plan": {"path": str(plan_path), "sha256": plan_sha256},
        "terminal": {"path": str(terminal_path), "sha256": terminal_sha256},
        "auditor": {"path": str(auditor_path), "sha256": auditor_sha256,
                    "function": "audit_bank"},
        "verifier": {"path": str(verifier_path), "sha256": verifier_sha256,
                     "function": "verify_bank"},
        "facts": facts,
    }
    with writing.open("w") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(writing, output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "verifier", "terminal"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--auditor-sha256", required=True)
    args = parser.parse_args()
    audit_bank(auditor_sha256=args.auditor_sha256,
               plan_path=args.plan, plan_sha256=args.plan_sha256,
               verifier_path=args.verifier, verifier_sha256=args.verifier_sha256,
               stage_name=args.stage, bank=args.bank, terminal_path=args.terminal,
               terminal_sha256=args.terminal_sha256, output=args.out)


if __name__ == "__main__":
    main()
