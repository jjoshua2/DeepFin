"""Strict fixed-bank retained-search main-WDL capability, not a study recipe."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import Any
import uuid

from scripts import core_finite_learner as core

CONTRACT = "retained_search_main_wdl_learning_v1"
PROFILE = "retained_search_main_wdl_profile_v1"
STAGE = "retained_search_main_wdl_segment_v1"
BOUNDARY = "COMPLETED_RETAINED_SEARCH_MAIN_WDL_SEGMENT_V1"
MODULE = "scripts.retained_search_finite_learner"
EXTRA_SOURCE = ("scripts/retained_search_finite_learner.py", "chess_anti_engine/replay/augment.py",
                "chess_anti_engine/replay/buffer.py", "chess_anti_engine/replay/dataset.py")
CHECKPOINT_KEY = "retained_search_learning_contract"


def validate_profile(p: dict[str, Any]) -> None:
    core.require(set(p) == {"schema", "scope", "search_share", "data_identity", "generation_manifest"}
                 and p["schema"] == PROFILE, "retained-search profile fields/schema")
    core.require(type(p["search_share"]) in (int, float) and p["search_share"] in (0, 1),
                 "retained search share must be zero or one")
    core.require(p["scope"] in {"ROOT_ADMITTED_FIXED_TRAIN_BANK", "SYNTHETIC_ONLY"}, "retained-search scope")
    identity = p["data_identity"]
    core.require(set(identity) == {"run_uuid", "source_revision"}
                 and str(uuid.UUID(identity["run_uuid"])) == identity["run_uuid"]
                 and re.fullmatch("[0-9a-f]{40}", identity["source_revision"]) is not None,
                 "retained-search producing identity")
    ref = p["generation_manifest"]
    core.require(set(ref) == {"path", "sha256"}
                 and re.fullmatch("[0-9a-f]{64}", ref["sha256"]) is not None,
                 "retained-search generation reference")


def checkpoint_identity(c: dict[str, Any], contract_ref: dict[str, str]) -> dict[str, Any]:
    return {"family": CONTRACT, "contract_sha256": contract_ref["sha256"],
            "run_uuid": c["run_uuid"], "source_revision": c["source_revision"], "profile": c["retained_search"]}


class SearchProfile:
    STAGE = STAGE
    BOUNDARY = BOUNDARY

    def validate_and_project(self, c: dict[str, Any], stage: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        core.require(Path(__file__).resolve() == core.REPO / "scripts/retained_search_finite_learner.py",
                     "retained-search caller imported from another checkout")
        core.require(c.get("schema") == CONTRACT and "updates_per_segment" in c
                     and "updates_per_generation" not in c, "not the retained-search learning family")
        validate_profile(c["retained_search"])
        core.require(c["run_uuid"] != c["retained_search"]["data_identity"]["run_uuid"],
                     "learner run must not relabel producing run")
        projected = {**c, "schema": "core_finite_learning_v1", "updates_per_generation": c["updates_per_segment"]}
        del projected["updates_per_segment"], projected["retained_search"]
        core.require(set(EXTRA_SOURCE) <= c["source_files"].keys(), "missing retained-search source pins")
        core.validate_contract(projected)
        projected.update(schema=CONTRACT, retained_search=c["retained_search"])
        core.require(set(stage) == {"schema", "contract_sha256", "segment", "generations", "parent", "output"}
                     and stage["schema"] == STAGE, "retained-search segment fields/schema")
        core.integer(stage["segment"], "segment")
        core.require(stage["generations"] == [c["retained_search"]["generation_manifest"]],
                     "retained-search fixed bank changed")
        _, gen = core.read_ref(stage["generations"][0], max_bytes=core.GENERATION_METADATA_LIMIT)
        fiction = gen.get("qualification_scope") == "SYNTHETIC_NOT_REAL_AUTHORITY"
        core.require(fiction == (c["retained_search"]["scope"] == "SYNTHETIC_ONLY"),
                     "retained-search synthetic/actual scope mismatch")
        internal_stage = {**stage, "generation": stage["segment"]}
        del internal_stage["segment"]
        return projected, internal_stage

    def closed_parent(self, binding: dict[str, Any], c: dict[str, Any], contract_ref: dict[str, str],
                      stage: dict[str, Any]) -> dict[str, Any]:
        core.require(set(binding) == {"boundary", "owner_terminal", "owner_sha256", "command"},
                     "retained-search closed parent fields")
        path, parent = core.read_ref(binding["boundary"])
        _, prior = core.read_ref(parent["stage"])
        output = core.external(prior["output"])
        core.require(path == output / "boundary.json" and not core.failure_marker(output).exists()
                     and not (output / "boundary.pending.json").exists(), "retained-search parent failed/incomplete")
        core.require(parent["status"] == BOUNDARY and prior["schema"] == STAGE
                     and prior["contract_sha256"] == contract_ref["sha256"]
                     and prior["segment"] == parent["segment"] == stage["generation"] - 1
                     and prior["generations"] == stage["generations"]
                     and prior["parent"] == parent["parent"]
                     and parent["retained_search_profile"] == c["retained_search"],
                     "retained-search same-family contract/profile/data parent mismatch")
        suffix = ["-m", MODULE, "--contract", contract_ref["path"], "--contract-sha256", contract_ref["sha256"],
                  "--stage", parent["stage"]["path"], "--stage-sha256", parent["stage"]["sha256"]]
        command = binding["command"]
        core.require(isinstance(command, list) and len(command) > len(suffix)
                     and all(isinstance(v, str) and v for v in command) and command[-len(suffix):] == suffix,
                     "retained-search parent exact command")
        core.require(re.fullmatch("[0-9a-f]{64}", binding["owner_sha256"]) is not None,
                     "retained-search owner source SHA")
        _, terminal = core.read_ref(binding["owner_terminal"], readonly=False)
        core.require(terminal.get("status") == "PASS" and type(terminal.get("returncode")) is int
                     and terminal["returncode"] == 0 and terminal.get("owned_processes_closed") is True
                     and terminal.get("custody") == "OWNED_PROCESSES_CLOSED"
                     and terminal.get("supervisor_sha256") == binding["owner_sha256"]
                     and terminal.get("command") == command, "retained-search parent exact owner did not close")
        core.require(parent["source_revision"] == c["source_revision"], "retained-search parent source")
        return {**parent, "generation": parent["segment"]}

    def validate_arrays(self, arrays: dict[str, Any], rows: int) -> None:
        """Validate stored observations, without normalizing/rebuilding them or drawing."""
        import numpy as np

        value, mask = arrays.get("search_wdl"), arrays.get("has_search_wdl")
        if value is None and mask is None:
            return  # Entire absent component is the existing genuine-outcome fallback.
        core.require(mask is not None and mask.shape == (rows,) and mask.dtype == np.uint8
                     and np.all((mask == 0) | (mask == 1)), "retained-search missing/invalid availability mask")
        if value is None:
            core.require(not np.any(mask), "retained-search claimed label is absent")
            return
        core.require(value.shape == (rows, 3) and value.dtype == np.float16,
                     "retained-search stored vector shape/precision")
        qualified = value[mask == 1].astype(np.float32)
        core.require(np.isfinite(qualified).all() and (qualified >= 0).all()
                     and np.all(np.abs(qualified.sum(axis=1) - 1) <= .002),
                     "retained-search invalid qualified distribution")

    def bind_checkpoint(self, path: Path, c: dict[str, Any], contract_ref: dict[str, str]) -> None:
        """Only this owner's newly saved trusted checkpoint gets the new family tag."""
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        core.require(CHECKPOINT_KEY not in state, "retained-search checkpoint already tagged")
        state[CHECKPOINT_KEY] = checkpoint_identity(c, contract_ref)
        pending = path.with_name(path.name + ".family.pending")
        with pending.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
        core.fsync_directory(path.parent)

    def check_checkpoint(self, ref: dict[str, str], c: dict[str, Any], contract_ref: dict[str, str]) -> None:
        import torch

        path = core.external(ref["path"])
        core.require(core.file_sha(path) == ref["sha256"], "retained-search parent checkpoint hash")
        state = torch.load(path, map_location="cpu", weights_only=True)
        core.require(state.get(CHECKPOINT_KEY) == checkpoint_identity(c, contract_ref),
                     "retained-search checkpoint family/contract/profile mismatch")
        core.require(core.file_sha(path) == ref["sha256"], "retained-search checkpoint changed")

    def decorate_boundary(self, boundary: dict[str, Any], c: dict[str, Any]) -> None:
        share = c["retained_search"]["search_share"]
        mass = boundary["drawn_rows"] * boundary["metrics"]["search_wdl_effective_frac"] * share
        boundary.update(status=BOUNDARY, segment=boundary.pop("generation"),
                        retained_search_profile=c["retained_search"], retained_search_target_mass=mass,
                        genuine_outcome_target_mass=boundary["drawn_rows"] - mass,
                        target_scope="stored retained-search WDL; missing labels fall back to unchanged genuine integer outcome")


def execute(contract_ref: dict[str, str], stage_ref: dict[str, str]) -> dict[str, Any]:
    return core._execute(contract_ref, stage_ref, retained_search=SearchProfile())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--stage-sha256", required=True)
    args = parser.parse_args(argv)
    result = execute({"path": args.contract, "sha256": args.contract_sha256},
                     {"path": args.stage, "sha256": args.stage_sha256})
    print(core.canonical({"status": result["status"], "checkpoint": result["checkpoint"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
