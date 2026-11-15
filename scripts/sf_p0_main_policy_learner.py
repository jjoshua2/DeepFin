"""Versioned fixed-bank reduced-SF current-P0 MAIN-policy capability, no execution defaults."""
from __future__ import annotations

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import re
from typing import Any
import uuid

from scripts import core_finite_learner as core
from scripts import sf_current_p0_finite_learner as admission
from scripts import sf_p0_main_policy_eligibility as eligibility

CONTRACT = "sf_p0_main_policy_learning_v1"
PROFILE = "sf_p0_main_policy_profile_v1"
STAGE = "sf_p0_main_policy_segment_v1"
BOUNDARY = "COMPLETED_SF_P0_MAIN_POLICY_SEGMENT_V1"
MODULE = "scripts.sf_p0_main_policy_learner"
CHECKPOINT_KEY = "sf_p0_main_policy_contract"
EXTRA_SOURCE = (*admission.EXTRA_SOURCE, "scripts/sf_p0_main_policy_learner.py",
                "scripts/sf_p0_main_policy_eligibility.py", "chess_anti_engine/moves/__init__.py",
                "chess_anti_engine/moves/encode.py", "chess_anti_engine/moves/leela_index.py")


def validate_profile(p: dict[str, Any]) -> None:
    core.require(set(p) == {"schema", "scope", "teacher_share", "eligibility", "data_identity",
                            "generation_manifest", "data_authority", "limits"} and p["schema"] == PROFILE, "main-policy profile schema/fields")
    core.require(type(p["teacher_share"]) in (int, float) and p["teacher_share"] in (0, 1),
                 "main-policy teacher share must be zero or one")
    core.require(p["eligibility"] == eligibility.SCHEMA, "main-policy eligibility version")
    core.require(set(p["data_identity"]) == {"run_uuid", "source_revision"}
                 and str(uuid.UUID(p["data_identity"]["run_uuid"])) == p["data_identity"]["run_uuid"]
                 and re.fullmatch("[0-9a-f]{40}", p["data_identity"]["source_revision"]) is not None,
                 "main-policy original producing identity")
    core.require(set(p["generation_manifest"]) == {"path", "sha256"}
                 and re.fullmatch("[0-9a-f]{64}", p["generation_manifest"]["sha256"]) is not None, "main-policy fixed bank reference")
    core.require(set(p["data_authority"]) == {"path", "sha256"}
                 and re.fullmatch("[0-9a-f]{64}", p["data_authority"]["sha256"]) is not None, "main-policy bank authority reference")
    if p["scope"] == "FIXED_ORIGINAL_G1":
        core.require(p["data_identity"] == {"run_uuid": admission.G1_RUN, "source_revision": admission.G1_SOURCE}
                     and p["generation_manifest"]["sha256"] == admission.G1
                     and p["data_authority"]["sha256"] == "2d8570409214a331323d5c1c9b68e46f1308257608ee9a9b39244f24de8513e5", "main-policy original G1 identity")
    else:
        core.require(p["scope"] == "SYNTHETIC_ONLY", "main-policy scope")
    cap = p["limits"]
    core.require(set(cap) == {"metadata_bytes", "game_decoded_bytes", "resident_bytes"}, "main-policy limits")
    for key, value in cap.items():
        core.integer(value, key)
    core.require(cap["metadata_bytes"] <= 128 << 20 and cap["game_decoded_bytes"] <= 64 << 20
                 and cap["resident_bytes"] <= 1 << 30, "main-policy bounded source/carrier inputs")


def checkpoint_identity(c: dict[str, Any], contract_ref: dict[str, str]) -> dict[str, Any]:
    return {"family": CONTRACT, "contract_sha256": contract_ref["sha256"],
            "run_uuid": c["run_uuid"], "source_revision": c["source_revision"], "profile": c["sf_p0_main_policy"]}


class PolicyReplay:
    """One unchanged uniform draw/callback before the existing Trainer mirror."""
    def __init__(self, replay: Any, profile: MainPolicyProfile) -> None:
        self.replay, self.profile = replay, profile
        self.rng = replay.rng
    def __len__(self) -> int:
        return len(self.replay)

    @property
    def membership(self) -> Any:
        return self.replay.membership
    def sample_batch_arrays(self, batch_size: int) -> dict[str, Any]:
        core.require(self.profile.pending is None, "main-policy unconsumed draw")
        batch = self.replay.sample_batch_arrays(batch_size)
        rows = self.profile.pending
        self.profile.pending = None
        core.require(rows is not None and len(rows) == batch_size, "main-policy missing original draw callback")
        return self.profile.overlay(batch, rows)


class MainPolicyProfile:
    STAGE = STAGE
    BOUNDARY = BOUNDARY
    def __init__(self) -> None:
        self.admission = admission.P0Profile()
        self.pending: Any = None
        self.index: dict[str, Any] = {}
        self.counts: Counter[str] = Counter()
        self.bank_counts: Counter[str] = Counter()
        self.batches: list[dict[str, int]] = []
    def validate_and_project(self, c: dict[str, Any], stage: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        core.require(Path(__file__).resolve() == core.REPO / "scripts/sf_p0_main_policy_learner.py",
                     "main-policy caller imported from another source tree")
        core.require(c.get("schema") == CONTRACT and "updates_per_segment" in c
                     and "updates_per_generation" not in c, "not the main-policy learning family")
        validate_profile(c["sf_p0_main_policy"])
        core.require(c["run_uuid"] != c["sf_p0_main_policy"]["data_identity"]["run_uuid"], "learner cannot relabel producer")
        projected = {**c, "schema": "core_finite_learning_v1", "updates_per_generation": c["updates_per_segment"]}
        del projected["updates_per_segment"], projected["sf_p0_main_policy"]
        core.require(set(EXTRA_SOURCE) <= c["source_files"].keys(), "missing main-policy source pins")
        core.validate_contract(projected)
        projected.update(schema=CONTRACT, sf_p0_main_policy=c["sf_p0_main_policy"])
        core.require(set(stage) == {"schema", "contract_sha256", "segment", "generations", "parent", "output", "main_policy_sidecars"}
                     and stage["schema"] == STAGE, "main-policy segment schema/fields")
        core.integer(stage["segment"], "segment")
        core.require(stage["generations"] == [c["sf_p0_main_policy"]["generation_manifest"]], "main-policy fixed bank changed")
        refs = stage["main_policy_sidecars"]
        active = c["sf_p0_main_policy"]["teacher_share"] == 1
        core.require((not active and refs is None) or (active and isinstance(refs, list) and len(refs) == 1),
                     "zero main-policy must bypass sidecar IO; positive requires accepted bank")
        profile = c["sf_p0_main_policy"]
        _, bank = core.read_ref(profile["data_authority"], max_bytes=profile["limits"]["metadata_bytes"])
        core.require(bank["generation_manifest"] == stage["generations"][0]
                     and bank["data_identity"] == profile["data_identity"], "main-policy inherited bank authority join")
        if profile["scope"] == "FIXED_ORIGINAL_G1":
            core.require(bank["status"] == "ROOT_AUTHENTICATES_EXISTING_FIXED_TRAIN_BANK_FOR_RETAINED_SEARCH_V1"
                         and bank["new_data_acceptance_issued"] is False, "main-policy inherited existing bank evidence")
            for ref in bank["original_acceptance_refs"]:
                core.read_ref(ref, max_bytes=profile["limits"]["metadata_bytes"])
        else:
            core.require(bank["status"] == "SYNTHETIC_NOT_REAL_AUTHORITY", "main-policy fictional bank authority")
        _, gen = core.read_ref(stage["generations"][0], max_bytes=core.GENERATION_METADATA_LIMIT)
        if c["sf_p0_main_policy"]["scope"] == "FIXED_ORIGINAL_G1":
            core.require(gen.get("qualification_scope") != "SYNTHETIC_NOT_REAL_AUTHORITY"
                         and len(gen["games"]) == 256 and sum(g["rows"] for g in gen["games"]) == 24827, "main-policy whole original G1")
        else:
            core.require(gen.get("qualification_scope") == "SYNTHETIC_NOT_REAL_AUTHORITY"
                         and len(gen["games"]) == 1 and 0 < gen["games"][0]["rows"] <= 4, "main-policy synthetic scope")
        internal = {**stage, "generation": stage["segment"]}
        del internal["segment"]
        return projected, internal

    def closed_parent(self, binding: dict[str, Any], c: dict[str, Any], contract_ref: dict[str, str],
                      stage: dict[str, Any]) -> dict[str, Any]:
        core.require(set(binding) == {"boundary", "owner_terminal", "owner_sha256", "command"},
                     "main-policy closed parent fields")
        path, parent = core.read_ref(binding["boundary"])
        _, prior = core.read_ref(parent["stage"])
        output = core.external(prior["output"])
        core.require(path == output / "boundary.json" and not core.failure_marker(output).exists()
                     and not (output / "boundary.pending.json").exists(), "main-policy parent failed/incomplete")
        core.require(parent["status"] == BOUNDARY and prior["schema"] == STAGE
                     and prior["contract_sha256"] == contract_ref["sha256"]
                     and prior["segment"] == parent["segment"] == stage["generation"] - 1
                     and prior["generations"] == stage["generations"]
                     and prior["main_policy_sidecars"] == stage["main_policy_sidecars"] == parent["main_policy_sidecars"]
                     and prior["parent"] == parent["parent"]
                     and parent["sf_p0_main_policy_profile"] == c["sf_p0_main_policy"],
                     "main-policy same-family contract/profile/data parent mismatch")
        suffix = ["-m", MODULE, "--contract", contract_ref["path"], "--contract-sha256", contract_ref["sha256"],
                  "--stage", parent["stage"]["path"], "--stage-sha256", parent["stage"]["sha256"]]
        command = binding["command"]
        core.require(isinstance(command, list) and len(command) > len(suffix)
                     and all(isinstance(v, str) and v for v in command) and command[-len(suffix):] == suffix,
                     "main-policy parent exact command")
        core.require(re.fullmatch("[0-9a-f]{64}", binding["owner_sha256"]) is not None,
                     "main-policy owner source SHA")
        _, terminal = core.read_ref(binding["owner_terminal"], readonly=False)
        core.require(terminal.get("status") == "PASS" and type(terminal.get("returncode")) is int
                     and terminal["returncode"] == 0 and terminal.get("owned_processes_closed") is True
                     and terminal.get("custody") == "OWNED_PROCESSES_CLOSED"
                     and terminal.get("supervisor_sha256") == binding["owner_sha256"]
                     and terminal.get("command") == command, "main-policy parent exact owner did not close")
        core.require(parent["source_revision"] == c["source_revision"], "main-policy parent source")
        return {**parent, "generation": parent["segment"]}

    def bind_checkpoint(self, path: Path, c: dict[str, Any], contract_ref: dict[str, str]) -> None:
        """Only this owner's newly saved trusted checkpoint gets the new family tag."""
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        core.require(CHECKPOINT_KEY not in state, "main-policy checkpoint already tagged")
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
        core.require(core.file_sha(path) == ref["sha256"], "main-policy parent checkpoint hash")
        state = torch.load(path, map_location="cpu", weights_only=True)
        core.require(state.get(CHECKPOINT_KEY) == checkpoint_identity(c, contract_ref),
                     "main-policy checkpoint family/contract/profile mismatch")
        core.require(core.file_sha(path) == ref["sha256"], "main-policy checkpoint changed")
    def prepare(self, c: dict[str, Any], stage: dict[str, Any], generations: list[dict[str, Any]]) -> None:
        """Reuse DATA-only carrier authority, then validate policy/TB facts BEFORE model construction."""
        import numpy as np
        from scripts import sf_current_p0_sidecar_schema as schema

        profile = c["sf_p0_main_policy"]
        # Only the existing data-admission helper is reused. Its local value tag
        # selects no loss/target; no old SF learner/profile contract is admitted.
        data_c = {"sf_p0": {"scope": profile["scope"], "value_variant": "native", "limits": profile["limits"]}}
        data_stage = {"generations": stage["generations"], "sf_p0_sidecars": stage["main_policy_sidecars"]}
        self.admission.prepare(data_c, data_stage, generations)
        core.require(not self.index, "main-policy profile already prepared")
        fields = ("policy_f16", "qualified_policy", "legal_mask_u8")
        resident = 0
        for game, folder, rows in self.admission.entries:
            facts = eligibility.join_game(game, generations[0], rows, profile["limits"]["metadata_bytes"])
            decoded = sum(np.dtype(schema.FIELDS[name][0]).itemsize * math.prod(schema.FIELDS[name][1]) * game["rows"] for name in fields)
            core.require(decoded <= profile["limits"]["game_decoded_bytes"]
                         and resident + decoded <= profile["limits"]["resident_bytes"], "main-policy decoded/resident byte cap")
            arrays = {name: schema.load_npy(folder / (name + ".npy"), name, game["rows"],
                      profile["limits"]["game_decoded_bytes"]) for name in fields}
            resident += decoded
            for i, row in enumerate(rows):
                flag = arrays["qualified_policy"][i]
                core.require(int(flag) in (0, 1) and bool(flag) is row["qualified_flags"]["policy"], "main-policy qualification flag identity")
                legal = arrays["legal_mask_u8"][i]
                core.require(np.isin(legal, [0, 1]).all()
                             and schema.array_digest(legal) == row["legal_mask_sha256"], "main-policy carrier legal mask")
                if flag:
                    value = arrays["policy_f16"][i].astype(np.float32)
                    core.require(np.isfinite(value).all() and (value >= 0).all()
                                 and abs(float(value.sum()) - 1) <= .002 and not value[legal == 0].any(),
                                 "main-policy invalid qualified teacher vector")
                self.bank_counts["rows"] += 1
                self.bank_counts["raw_qualified_teacher"] += int(flag)
                self.bank_counts["structurally_eligible"] += int(facts[i]["structurally_eligible"])
                self.bank_counts["replaceable"] += int(bool(flag) and facts[i]["structurally_eligible"])
            source = core.digest(game["shard"]["files"])
            core.require(source not in self.index, "main-policy duplicate source chunk")
            self.index[source] = (rows, arrays, facts)
        self.admission.check_custody()
    def on_draw(self, rows: Any) -> None:
        core.require(self.pending is None, "main-policy duplicate draw callback")
        self.pending = rows
    def wrap(self, replay: Any, c: dict[str, Any]) -> PolicyReplay:
        core.require(c["sf_p0_main_policy"]["teacher_share"] == 1, "zero must not wrap replay")
        core.require(replay.membership == [(1, source, len(entry[0])) for source, entry in self.index.items()],
                     "main-policy original physical membership")
        return PolicyReplay(replay, self)
    def overlay(self, batch: dict[str, Any], drawn: Any) -> dict[str, Any]:
        import numpy as np
        from scripts.sf_current_p0_sidecar_schema import array_digest

        # Copy ONLY policy; every fallback byte, original mask and WDL/diagnostic
        # field is preserved. Trainer later performs its one existing mirror.
        out = dict(batch)
        out["policy_target"] = batch["policy_target"].copy()
        n = len(drawn)
        batch_counts: Counter[str] = Counter(dict.fromkeys(("rows", "original_policy_supervised", "qualified_teacher",
            "structurally_eligible", "replaced", "original_fallback", "replaced_vector_changed", "replaced_normalized_f32_changed"), 0))
        core.require(batch["has_policy"].shape == (n,) and np.isin(batch["has_policy"], [0, 1]).all(),
                     "main-policy original supervision mask")
        for i, draw in enumerate(drawn):
            core.require(draw.generation == 1 and draw.source_sha256 in self.index, "main-policy foreign physical draw")
            rows, arrays, facts = self.index[draw.source_sha256]
            core.require(type(draw.row_index) is int and 0 <= draw.row_index < len(rows), "main-policy draw bounds")
            row, fact = rows[draw.row_index], facts[draw.row_index]
            identity = row["input_identity"]
            core.require(int(batch["game_id"][i]) == row["game_id"] and int(batch["ply_index"][i]) == identity["ply"]
                         and bool(batch["has_policy"][i]) is fact["original_has_policy"], "main-policy original game/ply/mask join")
            core.require(array_digest(np.asarray(batch["x"][i], dtype=np.float16)) == identity["input_F16_sha256"]
                         and array_digest(batch["legal_mask"][i]) == row["legal_mask_sha256"]
                         and np.array_equal(batch["legal_mask"][i], arrays["legal_mask_u8"][draw.row_index]),
                         "main-policy sampled input/legal identity")
            available = bool(arrays["qualified_policy"][draw.row_index])
            active = available and fact["structurally_eligible"] and bool(batch["has_policy"][i])
            if active:
                teacher = arrays["policy_f16"][draw.row_index]
                original = batch["policy_target"][i]
                changed = not np.array_equal(original, teacher)
                a, b = original.astype(np.float32), teacher.astype(np.float32)
                core.require(np.isfinite(a).all() and (a >= 0).all() and a.sum() > 0, "main-policy original target invalid")
                normalized_changed = not np.array_equal(a / a.sum(), b / b.sum())
                self.counts["replaced_vector_changed"] += int(changed)
                self.counts["replaced_normalized_f32_changed"] += int(normalized_changed)
                batch_counts["replaced_vector_changed"] += int(changed)
                batch_counts["replaced_normalized_f32_changed"] += int(normalized_changed)
                out["policy_target"][i] = teacher
            batch_counts["rows"] += 1
            batch_counts["original_policy_supervised"] += int(bool(batch["has_policy"][i]))
            batch_counts["qualified_teacher"] += int(available)
            batch_counts["structurally_eligible"] += int(fact["structurally_eligible"])
            batch_counts["replaced"] += int(active)
            batch_counts["original_fallback"] += int(not active)
            self.counts["drawn_rows"] += 1
            self.counts["original_policy_supervised"] += int(bool(batch["has_policy"][i]))
            self.counts["qualified_teacher"] += int(available)
            self.counts["structurally_eligible"] += int(fact["structurally_eligible"])
            self.counts["replaced"] += int(active)
            self.counts["original_fallback"] += int(not active)
            self.counts["TB_candidate"] += int(fact["TB_applied"] or fact["TB_abstained"])
            self.counts["material_le7"] += int(fact["piece_count"] < eligibility.MIN_PIECES)
            self.counts["status:" + row["source_join"]["status"]] += 1
        self.batches.append({"draw_batch": len(self.batches), **dict(batch_counts)})
        return out
    def decorate_boundary(self, boundary: dict[str, Any], c: dict[str, Any], stage: dict[str, Any]) -> None:
        if c["sf_p0_main_policy"]["teacher_share"]:
            self.admission.check_custody()
            core.require(self.pending is None and self.counts["drawn_rows"] == boundary["drawn_rows"]
                         and self.counts["replaced"] + self.counts["original_fallback"] == boundary["drawn_rows"]
                         and len(self.batches) == boundary["applied_updates"]
                         and sum(b["rows"] for b in self.batches) == boundary["drawn_rows"],
                         "main-policy complete realized batch denominator")
        boundary.update(status=BOUNDARY, segment=boundary.pop("generation"),
                        sf_p0_main_policy_profile=c["sf_p0_main_policy"], main_policy_sidecars=stage["main_policy_sidecars"],
                        main_policy_bank_counts=dict(sorted(self.bank_counts.items())),
                        main_policy_sample_counts=dict(sorted(self.counts.items())), main_policy_batch_counts=self.batches,
                        main_policy_availability_scope=("accepted carrier/drawn rows" if c["sf_p0_main_policy"]["teacher_share"] else "UNMEASURED_ZERO_UNWRAPPED_NO_SIDECAR_IO"),
                        main_policy_replaced_rows=self.counts["replaced"],
                        main_policy_original_rows=boundary["drawn_rows"] - self.counts["replaced"],
                        main_policy_contrast_observed=self.counts["replaced_normalized_f32_changed"] > 0,
                        main_policy_contrast_scope="source-F32 normalized target inequality, not a GPU reduction/gradient bitidentity claim",
                        main_policy_context="F16 reduced current-P0 teacher then unchanged loss normalization; genuine integer outcome/common weights unchanged")


def execute(contract_ref: dict[str, str], stage_ref: dict[str, str]) -> dict[str, Any]:
    return core._execute(contract_ref, stage_ref, main_policy=MainPolicyProfile())


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
