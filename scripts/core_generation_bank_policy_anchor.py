"""Explicit finite accepted-bank family with a frozen-reference policy anchor."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts import core_generation_bank_learner as bank

CONTRACT = bank.ANCHOR_CONTRACT
STAGE = bank.ANCHOR_STAGE
BOUNDARY = bank.ANCHOR_BOUNDARY
CHECKPOINT_KEY = bank.ANCHOR_CHECKPOINT_KEY
SOURCE_FILES = bank.ANCHOR_SOURCE_FILES


def execute(contract_ref: dict[str, str], stage_ref: dict[str, str]) -> dict[str, Any]:
    bank.core.require(Path(__file__).resolve() == bank.core.REPO / "scripts/core_generation_bank_policy_anchor.py",
                      "foreign policy-anchor entrypoint")
    return bank._execute(contract_ref, stage_ref, _anchored=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "contract-sha256", "stage", "stage-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    result = execute({"path": args.contract, "sha256": args.contract_sha256}, {"path": args.stage, "sha256": args.stage_sha256})
    print(bank.core.canonical({"status": result["status"], "checkpoint": result["checkpoint"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
