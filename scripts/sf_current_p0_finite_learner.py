"""Explicit fixed-bank SF-P0 main-WDL capability; no scientific recipe defaults."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

from scripts import core_finite_learner as core

CONTRACT = "sf_p0_main_wdl_learning_v1"
PROFILE = "sf_p0_main_wdl_profile_v1"
STAGE = "sf_p0_main_wdl_segment_v1"
BOUNDARY = "COMPLETED_SF_P0_MAIN_WDL_SEGMENT_V1"
MODULE = "scripts.sf_current_p0_finite_learner"
G1 = "35279e2ef19cdf808e4b4e25bb50fbb524750693ee035c1d44080d87a3db7878"
G1_RUN = "b0db4b6b-2bb3-4753-80d6-96825decbbee"
G1_SOURCE = "090b5143315a0faef7ab0ac1a90e2d5cebf3b929"
EXTRA_SOURCE = (
    "scripts/sf_current_p0_finite_learner.py", "scripts/sf_current_p0_sidecar.py",
    "scripts/sf_current_p0_sidecar_schema.py", "scripts/sf_banked_p0_census.py",
    "chess_anti_engine/replay/augment.py", "chess_anti_engine/replay/dataset.py",
)


def validate_profile(p: dict[str, Any]) -> None:
    core.require(set(p) == {"schema", "scope", "alpha", "value_variant", "data_identity",
                            "generation_manifest", "limits"} and p["schema"] == PROFILE, "SF-P0 profile fields/schema")
    alpha = core.number(p["alpha"], "SF-P0 alpha")
    core.require(0 <= alpha <= 1 and p["value_variant"] in {"native", "stored"}, "SF-P0 alpha/variant")
    core.require(set(p["data_identity"]) == {"run_uuid", "source_revision"}, "SF-P0 data identity")
    if p["scope"] == "FIXED_ORIGINAL_G1":
        core.require(p["data_identity"] == {"run_uuid": G1_RUN, "source_revision": G1_SOURCE}
                     and p["generation_manifest"]["sha256"] == G1, "fixed original G1 identity")
    else:
        core.require(p["scope"] == "SYNTHETIC_ONLY"
                     and re.fullmatch("[0-9a-f]{40}", p["data_identity"]["source_revision"]) is not None,
                     "only fixed G1 or explicitly fictional qualification data")
    caps = p["limits"]
    core.require(set(caps) == {"metadata_bytes", "game_decoded_bytes", "resident_bytes"}, "SF-P0 limits")
    for key, value in caps.items():
        core.integer(value, key)
    core.require(caps["metadata_bytes"] <= 128 << 20 and caps["game_decoded_bytes"] <= 64 << 20,
                 "SF-P0 per-file/game caps")


@dataclass
class SidecarGame:
    game: dict[str, Any]
    rows: list[dict[str, Any]]
    arrays: dict[str, Any]


class P0Replay:
    """One original draw/callback, then a copied batch before Trainer's one mirror."""

    def __init__(self, replay: Any, profile: P0Profile) -> None:
        self.replay, self.profile = replay, profile
        self.rng = replay.rng

    def __len__(self) -> int:
        return len(self.replay)

    @property
    def membership(self) -> Any:
        return self.replay.membership

    def sample_batch_arrays(self, batch_size: int) -> dict[str, Any]:
        core.require(self.profile.pending is None, "unconsumed SF-P0 draw")
        batch = self.replay.sample_batch_arrays(batch_size)
        rows = self.profile.pending
        self.profile.pending = None
        core.require(rows is not None and len(rows) == batch_size, "missing original SF-P0 draw callback")
        return self.profile.overlay(batch, rows)


class P0Profile:
    STAGE = STAGE
    BOUNDARY = BOUNDARY

    def __init__(self) -> None:
        self.pending: Any = None
        self.entries: list[tuple[dict[str, Any], Path, list[dict[str, Any]]]] = []
        self.index: dict[str, SidecarGame] = {}
        self.counts: Counter[str] = Counter()
        self.alpha = 0.0
        self.variant = ""
        self.custody: dict[str, Any] | None = None

    def validate_and_project(self, c: dict[str, Any], stage: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Validate new wire schemas, then use the shared engine's internal names."""
        core.require(c.get("schema") == CONTRACT and "updates_per_segment" in c
                     and "updates_per_generation" not in c, "not the SF-P0 learning family")
        validate_profile(c["sf_p0"])
        core.require(c["run_uuid"] != c["sf_p0"]["data_identity"]["run_uuid"], "learner run must not relabel producing run")
        projected = {**c, "schema": "core_finite_learning_v1",
                     "updates_per_generation": c["updates_per_segment"]}
        del projected["updates_per_segment"], projected["sf_p0"]
        core.require(set(EXTRA_SOURCE) <= c["source_files"].keys(), "missing SF-P0 consumer source pins")
        core.validate_contract(projected)
        projected.update(schema=CONTRACT, sf_p0=c["sf_p0"])
        core.require(set(stage) == {"schema", "contract_sha256", "segment", "generations", "parent", "output", "sf_p0_sidecars"}
                     and stage["schema"] == STAGE, "SF-P0 segment fields/schema")
        core.integer(stage["segment"], "segment")
        core.require(stage["generations"] == [c["sf_p0"]["generation_manifest"]], "SF-P0 fixed-bank scope changed")
        active = c["sf_p0"]["alpha"] > 0
        core.require((not active and stage["sf_p0_sidecars"] is None)
                     or (active and isinstance(stage["sf_p0_sidecars"], list) and len(stage["sf_p0_sidecars"]) == 1),
                     "zero SF-P0 must bypass sidecars; nonzero needs complete admitted bank")
        _, gen = core.read_ref(stage["generations"][0], max_bytes=core.GENERATION_METADATA_LIMIT)
        if c["sf_p0"]["scope"] == "FIXED_ORIGINAL_G1":
            core.require(len(gen["games"]) == 256 and sum(g["rows"] for g in gen["games"]) == 24827,
                         "fixed G1 full population")
        else:
            core.require(gen.get("qualification_scope") == "SYNTHETIC_NOT_REAL_AUTHORITY"
                         and len(gen["games"]) == 1 and 0 < gen["games"][0]["rows"] <= 4, "SF-P0 synthetic scope")
        internal_stage = {**stage, "generation": stage["segment"]}
        del internal_stage["segment"]
        return projected, internal_stage

    def closed_parent(self, binding: dict[str, Any], c: dict[str, Any], contract_ref: dict[str, str],
                      stage: dict[str, Any]) -> dict[str, Any]:
        core.require(set(binding) == {"boundary", "owner_terminal", "owner_sha256", "command"}, "SF-P0 closed parent fields")
        path, parent = core.read_ref(binding["boundary"])
        _, prior = core.read_ref(parent["stage"])
        output = core.external(prior["output"])
        core.require(path == output / "boundary.json" and not core.failure_marker(output).exists()
                     and not (output / "boundary.pending.json").exists(), "SF-P0 parent publication failed/incomplete")
        core.require(parent["status"] == BOUNDARY and prior["schema"] == STAGE
                     and prior["contract_sha256"] == contract_ref["sha256"]
                     and prior["segment"] == parent["segment"] == stage["generation"] - 1
                     and prior["generations"] == stage["generations"]
                     and prior["sf_p0_sidecars"] == stage["sf_p0_sidecars"] == parent["sf_p0_sidecars"]
                     and prior["parent"] == parent["parent"] and parent["sf_p0_profile"] == c["sf_p0"],
                     "SF-P0 same-family contract/profile/data parent mismatch")
        suffix = ["-m", MODULE, "--contract", contract_ref["path"], "--contract-sha256", contract_ref["sha256"],
                  "--stage", parent["stage"]["path"], "--stage-sha256", parent["stage"]["sha256"]]
        command = binding["command"]
        core.require(isinstance(command, list) and len(command) > len(suffix)
                     and all(isinstance(v, str) and v for v in command) and command[-len(suffix):] == suffix,
                     "SF-P0 parent exact command")
        _, terminal = core.read_ref(binding["owner_terminal"], readonly=False)
        core.require(re.fullmatch("[0-9a-f]{64}", binding["owner_sha256"]) is not None, "SF-P0 owner source SHA")
        core.require(terminal.get("status") == "PASS" and type(terminal.get("returncode")) is int
                     and terminal["returncode"] == 0 and terminal.get("owned_processes_closed") is True
                     and terminal.get("custody") == "OWNED_PROCESSES_CLOSED"
                     and terminal.get("supervisor_sha256") == binding["owner_sha256"]
                     and terminal.get("command") == command, "SF-P0 parent exact owner did not close")
        core.require(parent["source_revision"] == c["source_revision"], "SF-P0 parent source")
        return {**parent, "generation": parent["segment"]}

    def prepare(self, c: dict[str, Any], stage: dict[str, Any], generations: list[dict[str, Any]]) -> None:
        """Root admission + independent review + BOTH closed phases before arrays."""
        from scripts import sf_banked_p0_census as audit
        from scripts import sf_current_p0_sidecar as carrier

        self.alpha, self.variant = c["sf_p0"]["alpha"], c["sf_p0"]["value_variant"]
        cap = c["sf_p0"]["limits"]["metadata_bytes"]
        refs = stage["sf_p0_sidecars"][0]
        core.require(set(refs) == {"root_acceptance", "independent_review"}, "SF-P0 root/review references")
        _, accepted = core.read_ref(refs["root_acceptance"], max_bytes=cap)
        core.require(accepted["status"] == "ROOT_ACCEPTS_CLOSED_CURRENT_P0_SF_SIDECAR_DATA_ONLY"
                     and accepted["generation_manifest"] == stage["generations"][0]
                     and accepted["scope"] == c["sf_p0"]["scope"]
                     and accepted["training_admitted"] is False, "not root-postclosure accepted SF-P0 carrier")
        if accepted["scope"] == "SYNTHETIC_ONLY":
            core.require(accepted.get("qualification_scope") == "SYNTHETIC_NOT_REAL_AUTHORITY", "synthetic SF-P0 authority scope")
        else:
            core.require(accepted.get("qualification_scope") != "SYNTHETIC_NOT_REAL_AUTHORITY", "fictional SF-P0 authority on real data")
        _, validation = core.read_ref(accepted["validation"], max_bytes=cap)
        _, review = core.read_ref(refs["independent_review"], max_bytes=cap)
        core.require(review["status"] == "ACCEPT_CLOSED_CURRENT_P0_SF_SIDECAR_DATA_ONLY_NO_FINDINGS"
                     and review["findings"] == [] and review["candidate"] == accepted["candidate"]
                     and review["validation"] == accepted["validation"]
                     and review["scope"] == accepted["scope"]
                     and review["root_acceptance"] == refs["root_acceptance"], "SF-P0 independent actual validation required")
        core.read_ref(accepted["post_closure"], readonly=False)
        protocols = {}
        for phase in ("produce", "validate"):
            owner = accepted[phase]
            _, p = core.read_ref(owner["protocol"], max_bytes=cap)
            carrier.phase_authority(p, owner["protocol"], owner["admission"])
            _, terminal = core.read_ref(owner["terminal"], readonly=False)
            core.require(p["phase"] == phase, "SF-P0 producing phase")
            carrier.closed(terminal, owner["command"], p["runtime"]["owner"]["sha256"], p["runtime"]["owner_seconds"])
            core.require(owner["command"][-8:] == ["--protocol", owner["protocol"]["path"], "--protocol-sha256",
                         owner["protocol"]["sha256"], "--admission", owner["admission"]["path"],
                         "--admission-sha256", owner["admission"]["sha256"]], "SF-P0 phase command")
            protocols[phase] = p
        core.require(protocols["validate"]["candidate"] == accepted["candidate"]
                     and protocols["validate"]["producer_owner"] == {
                         "terminal": accepted["produce"]["terminal"], "command": accepted["produce"]["command"],
                         "owner_sha256": protocols["produce"]["runtime"]["owner"]["sha256"]},
                     "SF-P0 validator producer closure link")
        inventory_cap = {"metadata_bytes": cap, "carrier_total_bytes": protocols["produce"]["limits"]["carrier_total_bytes"],
                         "inventory_files": protocols["produce"]["limits"]["inventory_files"]}
        candidate = carrier.candidate_inventory(audit, accepted["candidate"], inventory_cap)
        self.custody = {"acceptance_ref": refs["root_acceptance"], "acceptance": accepted,
                        "candidate": candidate, "inventory_cap": inventory_cap}
        gen = generations[0]
        core.require(candidate["producer_protocol"] == accepted["produce"]["protocol"]
                     and candidate["execution_admission"] == accepted["produce"]["admission"]
                     and candidate["generation_manifest"] == stage["generations"][0]
                     and candidate["source"] == protocols["produce"]["source"] == accepted["source"]
                     and candidate["counts"] == accepted["counts"]
                     and candidate["scope"] == c["sf_p0"]["scope"]
                     and validation["status"] == "VALIDATION_CANDIDATE_CURRENT_P0_SF_SIDECAR_V1"
                     and validation["candidate"] == accepted["candidate"]
                     and validation["protocol"] == accepted["validate"]["protocol"]
                     and validation["execution_admission"] == accepted["validate"]["admission"]
                     and validation["source_payload_and_vector_bytes_recomputed"] is True
                     and validation["rows"] == candidate["rows"] == accepted["rows"] == sum(g["rows"] for g in gen["games"])
                     and validation["physical_games"] == candidate["physical_games"] == accepted["physical_games"] == len(gen["games"]), "SF-P0 complete validation/population")
        core.require(len(candidate["games"]) == len(gen["games"]), "SF-P0 complete game inventory")
        root = Path(accepted["candidate"]["path"]).parent
        total = 0
        for game, info in zip(gen["games"], candidate["games"], strict=True):
            core.require(info["physical_game_id"] == game["physical_game_id"] and info["rows"] == game["rows"], "SF-P0 game order")
            folder = root / info["directory"]
            member = next(f for f in candidate["files"] if f["path"] == info["directory"] + "/rows.json")
            total += member["bytes"]
            core.require(total <= cap, "SF-P0 aggregate row metadata cap")
            _, metadata = core.read_ref({"path": str(folder / "rows.json"), "sha256": member["sha256"]}, max_bytes=cap)
            core.require(metadata["original_shard"] == game["shard"] and metadata["producer_proof"] == game["producer_proof"]
                         and metadata["physical_game_id"] == game["physical_game_id"] and metadata["game_id"] == game["game_id"]
                         and metadata["rows"] == len(metadata["rows_detail"]) == game["rows"], "SF-P0 original shard/proof/rows")
            for i, row in enumerate(metadata["rows_detail"]):
                core.require(row["original_row_index"] == i and row["physical_game_id"] == game["physical_game_id"]
                             and row["game_id"] == game["game_id"]
                             and row["physical_row_id"] == game["physical_game_id"] + ":" + str(i), "SF-P0 row identities")
            self.entries.append((game, folder, metadata["rows_detail"]))

    def on_draw(self, rows: Any) -> None:
        core.require(self.pending is None, "duplicate SF-P0 draw callback")
        self.pending = rows

    def wrap(self, replay: Any, c: dict[str, Any]) -> P0Replay:
        import numpy as np
        from scripts import sf_current_p0_sidecar_schema as schema

        selected = "native_wdl_permille_i16" if self.variant == "native" else "stored_query_wdl_f16"
        flag = "qualified_native_wdl" if self.variant == "native" else "qualified_stored_wdl"
        fields = [selected, flag, "legal_mask_u8", "predecessor_meta_i32", "observed_predecessor_meta",
                  "predecessor_pv_i16", "observed_predecessor_pv"]
        resident = 0
        for game, folder, rows in self.entries:
            decoded = sum(np.dtype(schema.FIELDS[name][0]).itemsize
                          * math.prod(schema.FIELDS[name][1]) * game["rows"] for name in fields)
            core.require(decoded <= c["sf_p0"]["limits"]["game_decoded_bytes"]
                         and resident + decoded <= c["sf_p0"]["limits"]["resident_bytes"], "SF-P0 pre-load decoded byte cap")
            arrays = {name: schema.load_npy(folder / (name + ".npy"), name, game["rows"],
                       c["sf_p0"]["limits"]["game_decoded_bytes"]) for name in fields}
            resident += sum(a.nbytes for a in arrays.values())
            core.require(resident <= c["sf_p0"]["limits"]["resident_bytes"], "SF-P0 resident byte cap")
            source = core.digest(game["shard"]["files"])
            core.require(source not in self.index, "duplicate SF-P0 original source chunk")
            self.index[source] = SidecarGame(game, rows, arrays)
        core.require([(g, source, n) for g, source, n in replay.membership]
                     == [(1, core.digest(game["shard"]["files"]), game["rows"]) for game, _, _ in self.entries],
                     "SF-P0 physical membership mismatch")
        self.check_custody()
        return P0Replay(replay, self)

    def check_custody(self) -> None:
        from scripts import sf_banked_p0_census as audit
        from scripts.sf_current_p0_sidecar import candidate_inventory

        core.require(self.custody is not None, "missing SF-P0 custody")
        assert self.custody is not None
        _, accepted = core.read_ref(self.custody["acceptance_ref"])
        core.require(accepted == self.custody["acceptance"]
                     and candidate_inventory(audit, accepted["candidate"], self.custody["inventory_cap"])
                     == self.custody["candidate"], "SF-P0 carrier custody changed")

    def overlay(self, batch: dict[str, Any], drawn: Any) -> dict[str, Any]:
        import numpy as np

        n = len(drawn)
        out = dict(batch)
        out["sf_wdl"] = np.eye(3, dtype=np.float32)[batch["wdl_target"]]
        out["has_sf_wdl"] = np.zeros(n, dtype=np.uint8)
        out["sf_label_meta"] = np.zeros((n, 6), dtype=np.int32)
        out["has_sf_label_meta"] = np.zeros(n, dtype=np.uint8)
        out["sf_multipv_raw"] = np.tile(np.array([-1, -32768, 0, -1, -1], dtype=np.int16), (n, 48, 1))
        out["has_sf_multipv_raw"] = np.zeros(n, dtype=np.uint8)
        # No policy anchoring/reply-head supervision. Disarm old P1 flags/payloads;
        # only the same-predecessor raw diagnostics above enter health checks.
        for key in ("sf_policy_target", "sf_legal_mask", "sf_move_index"):
            out.pop(key, None)
        for key in ("has_sf_policy", "has_sf_legal_mask", "has_sf_move"):
            out[key] = np.zeros(n, dtype=np.uint8)
        field = "native_wdl_permille_i16" if self.variant == "native" else "stored_query_wdl_f16"
        flag = "qualified_native_wdl" if self.variant == "native" else "qualified_stored_wdl"
        metadata_flag = "native_wdl" if self.variant == "native" else "stored_wdl"
        for i, draw in enumerate(drawn):
            core.require(draw.generation == 1 and draw.source_sha256 in self.index, "foreign SF-P0 physical draw")
            entry = self.index[draw.source_sha256]
            core.require(type(draw.row_index) is int and 0 <= draw.row_index < len(entry.rows), "SF-P0 draw row bounds")
            row, a = entry.rows[draw.row_index], entry.arrays
            identity = row["input_identity"]
            core.require(int(batch["game_id"][i]) == row["game_id"]
                         and int(batch["ply_index"][i]) == identity["ply"], "SF-P0 sampled game/ply identity")
            # array_digest is byte identity, not a re-encoding or teacher query.
            from scripts.sf_current_p0_sidecar_schema import array_digest
            core.require(array_digest(np.asarray(batch["x"][i], dtype=np.float16)) == identity["input_F16_sha256"]
                         and array_digest(batch["legal_mask"][i]) == row["legal_mask_sha256"]
                         and np.array_equal(batch["legal_mask"][i], a["legal_mask_u8"][draw.row_index]), "SF-P0 sampled input/legal identity")
            present = int(a[flag][draw.row_index])
            status = row["source_join"]["status"]
            core.require(present in (0, 1) and bool(present) is row["qualified_flags"][metadata_flag]
                         and (not present or status in {"SOURCE_CONTEXT_REDUCED_LABEL", "SOURCE_CONTEXT_NO_REDUCED_LABEL"}), "SF-P0 qualified mask/status")
            if present:
                value = np.asarray(a[field][draw.row_index], dtype=np.float32)
                if self.variant == "native":
                    core.require(np.all(value >= 0) and value.sum() == 1000, "SF-P0 native counts")
                    value = value / 1000.0
                core.require(np.isfinite(value).all() and np.all(value >= 0) and abs(float(value.sum()) - 1) <= .002, "SF-P0 stored/native distribution")
                out["sf_wdl"][i] = value
                out["has_sf_wdl"][i] = 1
            for target, source, mark in (("sf_label_meta", "predecessor_meta_i32", "observed_predecessor_meta"),
                                         ("sf_multipv_raw", "predecessor_pv_i16", "observed_predecessor_pv")):
                observed = int(a[mark][draw.row_index])
                core.require(observed in (0, 1), "SF-P0 diagnostic presence")
                out[target][i] = a[source][draw.row_index]
                out["has_" + target][i] = observed
            self.counts["rows"] += 1
            self.counts["qualified"] += present
            self.counts["status:" + status] += 1
        return out

    def decorate_boundary(self, boundary: dict[str, Any], c: dict[str, Any], stage: dict[str, Any]) -> None:
        if c["sf_p0"]["alpha"] > 0:
            self.check_custody()
            core.require(self.pending is None and self.counts["rows"] == boundary["drawn_rows"], "SF-P0 applied draw denominator")
            coverage = self.counts["qualified"] / boundary["drawn_rows"]
            core.require(math.isclose(boundary["metrics"]["sf_wdl_effective_frac"], coverage, abs_tol=1e-6),
                         "SF-P0 actual loss availability disagrees with overlay")
        boundary.update(status=BOUNDARY, segment=boundary.pop("generation"), sf_p0_profile=c["sf_p0"],
                        sf_p0_sidecars=stage["sf_p0_sidecars"], sf_p0_sample_counts=dict(sorted(self.counts.items())),
                        sf_p0_teacher_mass=self.counts["qualified"] * c["sf_p0"]["alpha"],
                        sf_p0_outcome_mass=boundary["drawn_rows"] - self.counts["qualified"] * c["sf_p0"]["alpha"],
                        sf_p0_context="current-P0 SF source context; integer genuine outcome unchanged")


def execute(contract_ref: dict[str, str], stage_ref: dict[str, str]) -> dict[str, Any]:
    return core._execute(contract_ref, stage_ref, sf_p0=P0Profile())


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
