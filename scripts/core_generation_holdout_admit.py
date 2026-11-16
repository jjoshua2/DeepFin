"""Audit a closed literal HOLDOUT generation; emit evaluation-only candidates.

This emits candidate data only. It does not promote a bank, select an actor,
construct a model, or admit learning. The legacy auditor is a separate entry.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts import core_finite_admit as audit


def execute(protocol_ref: dict[str, str]) -> dict[str, Any]:
    audit.require(Path(__file__).resolve() == audit.learning.REPO / "scripts/core_generation_holdout_admit.py",
                  "foreign data auditor entrypoint")
    return audit.execute_holdout_data(protocol_ref)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args(argv)
    result = execute({"path": args.protocol, "sha256": args.protocol_sha256})
    print(audit.canonical({"status": result["status"], "generation_manifest": result["generation_manifest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
