"""Strict additive metadata predicates for the separate D2 horizon question.

The historical ruler is imported unchanged. These functions issue no receipts,
read no checkpoint/journal/book payloads, and never authorize an owned phase.
This additive source grants no execution, deployment or qualification authority.
"""
from __future__ import annotations

import hashlib
import math
import stat
import uuid
from pathlib import Path
from typing import Any

from scripts import core_generation_holdout_ruler as ruler

STUDY = "3326c042-1044-4270-8d1e-02ca4c1c2ca5"
LEARNER_SOURCE = "bbb4b725a2d5218ffa4630841dd93e8b2884b58d"
SHORT_COEFFICIENTS = {"S0": 0, "S1": 1}
LONG_LABELS = {"L0": "K0", "L1": "K1"}
LR = {"peak": 3e-6, "floor": 3e-6, "warmup_start": 3e-6, "warmup_steps": 0, "first_period": 4096}
_W = Path.home() / "chess-artifacts" / "operations" / "rl-week-one-20261112-v1"
DESIGN = {"path": str(_W / "prospective-KL-D2-short-horizon-root-design-v1/design.json"),
          "sha256": "4f61d06dd3d08eac7d2091347d682fa3433b2559b236c913d5a4bb48bfe94749"}
RUN_UUIDS = {"S0": "c5e854cf-9ef3-4762-9b58-53f6baf9692e", "S1": "017c63b8-8cba-491d-baac-59acb403b87b"}
LEARNER_PINS = {"path": str(_W / "collector-current-I-asset-qualification-source-v3/source-pins.json"),
                "sha256": "574c855d693676776d3e9d8bba5e24c789d6793e2cda97eee1a1323254852187"}
TRAIN_BANK = {"root_acceptance": {"path": str(_W / "prospective-whole-game-TRAIN-audit-bindings-v1/bank-root-acceptance.json"),
                                  "sha256": "4438815792a927f52d5234fdd4b87d067d7af1e6520c9de73696cd10d1316d19"},
              "independent_review": {"path": str(_W / "prospective-whole-game-TRAIN-audit-bindings-v1/bank-independent-review.json"),
                                     "sha256": "af80aba10f8b695ec8f50978ac169afe3cd6115c3a1a488e94df6ae7a18af951"}}
RUNTIME = {"cuda_devices": 1, "device": "cuda:0", "numpy_version": "2.2.6", "torch_threads": 2,
           "torch_version": "2.14.0+cu130", "use_amp": True}
SHORT_SCHEMA = "core_bank_learner_same_gpu_payload_proof_D2_short_v1"
SHORT_PENDING = "D2_SHORT_HORIZON_GPU_OWN_LIVE_SAVED_PROOF_PENDING_ROOT_CLOSURE_V1"
SHORT_ADMISSION = "ROOT_ADMITS_D2_SHORT_HORIZON_LEARNER_GPU_V1"
SHORT_REVIEW_STATUS = "ACCEPT_D2_SHORT_HORIZON_PASSIVE_PROOF_SOURCE_ONLY_NO_FINDINGS_V1"
# The new route names the actual independently reviewed observer, not a
# status-only substitute or either historical D1/engineering helper.
SHORT_HELPER_SHA256 = "d20c6d918b40740cf8bb9d24c0fe3bd29ae7907e5b2029f957934b074893351f"
SHORT_SOURCE_REVIEW = {"path": str(_W / "prospective-KL-D2-short-horizon-learner-proof-independent-source-review-v2/review.json"),
                       "sha256": "b5e32d81c4c1b0d6fa5be1426fb51a3d521794184090ec875e6fe77b2e26a825"}
SHORT_CONTRACT_TEMPLATES = {
    "S0": {"path": str(_W / "prospective-KL-D2-short-horizon-learner-proof-source-preparation-v2/templates/S0.learner.contract.json"),
           "sha256": "991010a19017cc81800c5bfeece31d2c09c777c84669c7495bf592bb159cf009"},
    "S1": {"path": str(_W / "prospective-KL-D2-short-horizon-learner-proof-source-preparation-v2/templates/S1.learner.contract.json"),
           "sha256": "0f1005001fea396c7751c699ca2730aa3184178cf1028a1466b22275547a1e9b"},
}
ROLE_HELPER_SHA256 = "09e34c57b32b837fa973bee91edecb641b8abdd117051346ab7671aa8f403ba6"
ROLE_SOURCE_REVIEW = {"path": str(_W / "prospective-KL-D2-fresh-screen-A-80-159-role-independent-source-review-v1/review.json"),
                      "sha256": "5837ed5102082f51880fc8d676f67c1156a296b716baafecbdbc900a57bba527"}
SOURCE_ROLES = {
    "root": {"path": str(_W / "prospective-whole-game-fresh-roles-bindings-v1/root-acceptance.json"),
             "sha256": "14c2d63c678f4718c23b40bd84b43957008a232f52bd956058a4253c500da6ca"},
    "independent": {"path": str(_W / "prospective-whole-game-fresh-roles-bindings-v1/independent-review.json"),
                    "sha256": "88eca9455ebd6f045b7a7d6efe8e670756e00a757eff966aecfbb6b3a992252c"},
}
D1_NO_WINNER = {"path": str(_W / "prospective-KL-D1-complete-development-no-winner-readout-v1/readout.json"),
                "sha256": "1c886a2a8e39a3b843c508b5ea2d0edd9343d38f6e524d7cbc9b5481f0b617fc"}
D1_NO_WINNER_INDEPENDENT = {"path": str(_W / "prospective-KL-D1-complete-development-no-winner-independent-review-v1/review.json"),
                            "sha256": "26bdd259e5656a3eb3eec44c5a6afb61e850ededcd11cf81e85ff245351070c6"}
D1_DESIGN = {"path": str(_W / "prospective-KL-development-first-root-design-v1/design.json"),
             "sha256": "98770dcd0ac6dffba73608fcad53a23e5a90c929a8432ed74b6d52d615b1c8dd"}
LONG_BINDINGS = {
    "L0": {
        "root": {"path": str(_W / "prospective-KL-D1-T1-K0-no-clip-corrective-audit-bindings-v1/root-acceptance.json"),
                 "sha256": "71f733ac9a904aae0277f132dbcf58015692298e037fc5dde75f65235b94fc1f"},
        "independent": {"path": str(_W / "prospective-KL-D1-T1-K0-no-clip-corrective-audit-bindings-v1/independent-review.json"),
                        "sha256": "9388a6592b5bf819666567230098512f3756efbcb6920e884d30338907f09b71"}},
    "L1": {
        "root": {"path": str(_W / "prospective-KL-D1-T1-K1-learning-GPU-bindings-v1/root-acceptance.json"),
                 "sha256": "44cf204e7d62116352ecc2bde86d662ddb9367f2620332c83ebdf3c209e46014"},
        "independent": {"path": str(_W / "prospective-KL-D1-T1-K1-learning-GPU-bindings-v1/independent-review.json"),
                        "sha256": "8e5a03ecd1278084751db8ff05808432441dd8841cf97bcd9bcee4429c036bb5"}},
}


def short_contract(c: dict[str, Any], label: str, donor: dict[str, Any],
                   source_files: dict[str, str]) -> None:
    """Validate the fixed short contract against already authenticated inputs."""
    ruler.require(label in SHORT_COEFFICIENTS, "D2 short label differs")
    ruler.require(donor["checkpoint"]["sha256"] == "a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0"
                  and donor["model_state_sha256"] == "9796079cd73bef73ba2f5d5f2ad0779704498832e34a7c2a880f0d0b7f305229",
                  "D2 original donor state differs")
    ruler.bulk_reference(donor["checkpoint"], suffix=".pt")
    keys = {"schema", "run_uuid", "source_revision", "source_files", "architecture", "initial_model",
            "initial_actor_step", "first_generation", "batch_size", "games_per_generation", "max_games_per_generation",
            "updates_per_segment", "replay_generations", "seed", "runtime", "lr", "mirror_prob", "limits",
            "bank_limits", "policy_anchor"}
    ruler.require(set(c) == keys and c["schema"] == ruler.ANCHOR_KEY, "D2 public contract shape differs")
    ruler.require(str(uuid.UUID(c["run_uuid"])) == c["run_uuid"] == RUN_UUIDS[label]
                  and c["source_revision"] == LEARNER_SOURCE and c["source_files"] == source_files
                  and len(source_files) == 219, "D2 learner source/run identity differs")
    ruler.require(c["initial_model"] == donor["checkpoint"] and type(c["initial_actor_step"]) is int
                  and c["initial_actor_step"] == donor["step"] == 680276 and c["architecture"] == donor["architecture"],
                  "D2 complete A39 bootstrap differs")
    ruler.require(type(c["batch_size"]) is int and c["batch_size"] == 32
                  and type(c["seed"]) is int and c["seed"] == 2026111673
                  and type(c["mirror_prob"]) is float and c["mirror_prob"] == 0.5
                  and type(c["updates_per_segment"]) is list and len(c["updates_per_segment"]) == 1
                  and type(c["updates_per_segment"][0]) is int and c["updates_per_segment"] == [512]
                  and ruler.canonical(c["lr"]) == ruler.canonical(LR), "D2 short dose/LR/sampling differs")
    ruler.require(type(c["first_generation"]) is int and c["first_generation"] == 1
                  and type(c["replay_generations"]) is int and c["replay_generations"] == 4
                  and type(c["games_per_generation"]) is int and c["games_per_generation"] == 160
                  and type(c["max_games_per_generation"]) is int and c["max_games_per_generation"] == 192,
                  "D2 full bank horizon differs")
    ruler.require(type(c["limits"]) is dict and all(type(v) is int for v in c["limits"].values())
                  and type(c["bank_limits"]) is dict and all(type(v) is int for v in c["bank_limits"].values())
                  and c["limits"] == {"max_rows": 17568, "max_decoded_window_bytes": 16 << 30}
                  and c["bank_limits"] == {"max_file_bytes": 128 << 20, "max_total_bytes": 4 << 30, "max_games": 192},
                  "D2 complete bank limits differ")
    runtime = c["runtime"]
    ruler.require(type(runtime) is dict and runtime == RUNTIME
                  and runtime["device"] == "cuda:0" and runtime["use_amp"] is True
                  and type(runtime["torch_threads"]) is int and runtime["torch_threads"] == 2
                  and type(runtime["cuda_devices"]) is int and runtime["cuda_devices"] == 1,
                  "D2 learner runtime differs")
    anchor = c["policy_anchor"]
    coefficient = SHORT_COEFFICIENTS[label]
    ruler.require(type(anchor) is dict and set(anchor) == {"schema", "direction", "mask", "coefficient", "reference"}
                  and anchor["schema"] == "frozen_reference_policy_anchor_v1"
                  and anchor["direction"] == "reference_to_student_forward_KL"
                  and anchor["mask"] == "original_network_turn_has_policy_with_explicit_legal_mask_v1"
                  and type(anchor["coefficient"]) in (int, float) and anchor["coefficient"] == coefficient,
                  "D2 policy-only anchor differs")
    expected = None if coefficient == 0 else {"checkpoint": donor["checkpoint"], "architecture": donor["architecture"],
                                             "checkpoint_step": 680276, "historical_source_revision": None}
    ruler.require(anchor["reference"] == expected, "D2 frozen reference differs")


def retained_model(m: ruler.Metadata, binding: dict[str, Any], label: str,
                   donor: dict[str, Any]) -> dict[str, Any]:
    """Resolve a display alias through the unchanged, exact historical route."""
    ruler.require(label in LONG_LABELS and binding == LONG_BINDINGS[label], "D2 retained endpoint identity differs")
    design = m.read(D1_DESIGN)
    ruler.require(design["study_uuid"] == "8045699e-4088-4a41-ae9c-4013d4a55469", "D2 retained D1 design differs")
    value = ruler._d1_model(m, binding, LONG_LABELS[label], donor, design)
    ruler.require(value["step"] == 4096 and value["label"] == LONG_LABELS[label], "D2 retained provenance differs")
    # Preserve the original label, tag, study and source. A display alias is not a
    # replacement delivery and does not turn K0's saved-only proof into same-GPU.
    return value


def slice_projection(projection: dict[str, Any], role: dict[str, Any],
                     public_consumption: dict[str, Any]) -> None:
    """Check slice witnesses after a caller authenticates its future authority.

    This pure predicate is not a closed-role acceptance or a book/history read.
    The producer/helper/owner review adapter remains unissued in this precursor.
    """
    ruler.require(type(role) is dict and set(role) == {"source_role", "original400_book", "original_full_role", "book", "parent_map",
                  "selection", "prefix_start", "pairs", "source_count", "complete_parents", "actual_public_loader_count",
                  "no_skips_dedup_shrink", "parent_order_sha256", "history_order_sha256"}, "D2 slice witness shape differs")
    ruler.require(role["source_role"] == "SCREEN_C_A" and type(role["prefix_start"]) is int and role["prefix_start"] == 80
                  and all(type(role[k]) is int and role[k] == 80 for k in ("pairs", "complete_parents", "actual_public_loader_count"))
                  and type(role["source_count"]) is int and role["source_count"] == 400
                  and role["no_skips_dedup_shrink"] is True
                  and role["selection"] == "exact original ordinals80..159; no rng/subsample/reorder", "D2 fixed slice differs")
    for key in ("original400_book", "book"):
        ruler.bulk_reference(role[key], suffix=".epd")
    for key in ("original_full_role", "parent_map"):
        ruler.bank.reference(role[key])
    for key in ("parent_order_sha256", "history_order_sha256"):
        ruler.bank.sha(role[key])
    ruler.require(type(projection) is dict and set(projection) == {"schema", "study_uuid", "design", "source_role", "original_full_role",
                  "original400_book", "pairs", "games", "scope"}
                  and projection["schema"] == "D2_COMPLETE_FRESH_SLICE_PARENT_PROJECTION_V1"
                  and projection["study_uuid"] == STUDY and projection["design"] == DESIGN
                  and type(projection["pairs"]) is int
                  and all(projection[k] == role[k] for k in ("source_role", "original_full_role", "original400_book", "pairs"))
                  and type(projection["scope"]) is str and bool(projection["scope"]), "D2 slice parent authority differs")
    games = projection["games"]
    ruler.require(type(games) is list and len(games) == 80, "D2 complete slice parents differ")
    for index, game in enumerate(games):
        ruler.require(type(game) is dict and set(game) == {"pair_id", "original_role_ordinal", "parent_id", "parent_member_ids", "source_parents"}
                      and type(game["pair_id"]) is int and game["pair_id"] == index
                      and type(game["original_role_ordinal"]) is int and game["original_role_ordinal"] == index + 80
                      and type(game["parent_id"]) is str and bool(game["parent_id"])
                      and type(game["parent_member_ids"]) is list and bool(game["parent_member_ids"])
                      and all(type(member) is str and member for member in game["parent_member_ids"])
                      and len(set(game["parent_member_ids"])) == len(game["parent_member_ids"])
                      and type(game["source_parents"]) is list
                      and len(game["source_parents"]) == len(game["parent_member_ids"]), "D2 original parent ordinal differs")
    ruler.require(hashlib.sha256(ruler.canonical(games)).hexdigest() == role["parent_order_sha256"], "D2 parent order digest differs")
    ruler.require(type(public_consumption) is dict
                  and set(public_consumption) == {"book", "pairs", "actual_public_reads", "full16_history_order_EPD_exact"}
                  and public_consumption["book"] == role["book"]
                  and type(public_consumption["pairs"]) is int and public_consumption["pairs"] == 80
                  and type(public_consumption["actual_public_reads"]) is int and public_consumption["actual_public_reads"] == 1
                  and public_consumption["full16_history_order_EPD_exact"] is True,
                  "D2 qualified full-history consumption differs")


def _closed_owner(m: ruler.Metadata, owner: dict[str, Any], suffix: list[str], *, role: bool = False) -> dict[str, Any]:
    """Two fixed new admission profiles; historical predicates are intact."""
    resource, seconds = ("cpu", 300) if role else ("gpu", 3600)
    status = "ROOT_ADMITS_KL_DEVELOPMENT_FIRST_D2_PREFIX_CPU_V1" if role else SHORT_ADMISSION
    plan_key, plan_flag = ("protocol", "--protocol") if role else ("observer_plan", "--observer-plan")
    ruler.require(set(owner) == {"terminal", "admission", "command", "source_revision", "cwd", "closure"}, "D2 closed owner shape differs")
    terminal, admission = m.observed(owner["terminal"]), m.read(owner["admission"])
    command = owner["command"]
    ruler.require(type(command) is list and len(command) > len(suffix) and command[-len(suffix):] == suffix
                  and all(type(arg) is str and arg for arg in command), "D2 child suffix differs")
    ruler.bank.sha(owner["source_revision"], 40)
    ruler.require(type(owner["cwd"]) is str and Path(owner["cwd"]).is_absolute(), "D2 owner cwd differs")
    ruler.require(terminal["status"] == "PASS" and type(terminal["returncode"]) is int and terminal["returncode"] == 0
                  and terminal["owned_processes_closed"] is True and terminal["custody"] == "OWNED_PROCESSES_CLOSED"
                  and terminal["resource"] == resource and terminal["command"] == command
                  and terminal["source_revision"] == owner["source_revision"] and terminal["cwd"] == owner["cwd"], "D2 owner not closed GPU PASS")
    wall = terminal["wall_seconds"]
    ruler.require(type(terminal["seconds"]) is int and terminal["seconds"] == seconds
                  and type(wall) in (int, float) and math.isfinite(wall) and 0 <= wall <= seconds, "D2 owner cap/wall differs")
    closure = owner["closure"]
    ruler.require(set(closure) == {"supported_eof", "exit_code", "supervisor_pid", "supervisor_closed", "owned_births_closed", "known_processes"}
                  and closure["supported_eof"] is True and type(closure["exit_code"]) is int and closure["exit_code"] == 0
                  and closure["supervisor_closed"] is True and closure["owned_births_closed"] is True
                  and type(closure["supervisor_pid"]) is int and closure["supervisor_pid"] == terminal["supervisor_pid"]
                  and closure["known_processes"] == terminal["known_processes"] and closure["known_processes"], "D2 closure differs")
    for pid, birth in closure["known_processes"].items():
        ruler.require(type(pid) is str and pid.isdecimal() and str(int(pid)) == pid and int(pid) > 0
                      and type(birth) is int and birth > 0, "D2 process birth differs")
    ruler.require(admission["status"] == status and admission["execution_authorized"] is True
                  and admission["resource"] == resource and type(admission["seconds"]) is int and admission["seconds"] == seconds,
                  "D2 GPU admission differs")
    for flag, value in (("--admission", owner["admission"]["path"]), ("--admission-sha256", owner["admission"]["sha256"]),
                        (plan_flag, admission[plan_key]["path"]),
                        (plan_flag + "-sha256", admission[plan_key]["sha256"])):
        ruler.require(command.count(flag) == 1 and command.index(flag) + 1 < len(command)
                      and command[command.index(flag) + 1] == value, "D2 observer argv differs")
    normalized = list(command)
    normalized[normalized.index("--admission-sha256") + 1] = "SELF_ADMISSION_SHA256"
    ruler.require(admission["normalized_child"] == normalized
                  and admission["owner"]["sha256"] == terminal["supervisor_sha256"], "D2 normalized owner differs")
    return admission


def _prefix_proof(m: ruler.Metadata, saved: dict[str, Any], plan: dict[str, Any], boundary: dict[str, Any],
                  label: str, donor: dict[str, Any]) -> None:
    """Validate a reviewed in-owner sampler witness, never open either journal."""
    witness = saved["draw_prefix_comparison"]
    keys = {"schema", "long_comparator", "long_boundary", "long_journal", "short_journal", "batches", "rows_per_batch", "views",
            "long_batches", "long_views", "population_rows", "population_games", "membership_sha256", "coverage_sha256",
            "long_prefix_sha256", "short_prefix_sha256", "equal", "long_stream_passes", "long_raw_read_bytes",
            "short_stream_passes", "short_raw_read_bytes", "long_pre", "long_post", "long_final", "short_observed", "all_reads_metered", "claims"}
    ruler.require(type(witness) is dict and set(witness) == keys
                  and witness["schema"] == "D2_PAIRED_SAMPLER_PREFIX_COMPARISON_V1", "D2 sampler witness shape differs")
    long_label = "L0" if label == "S0" else "L1"
    ruler.require(plan["long_comparator"] == witness["long_comparator"] == LONG_BINDINGS[long_label], "D2 paired long comparator differs")
    long_model = retained_model(m, plan["long_comparator"], long_label, donor)
    long_root = m.read(plan["long_comparator"]["root"])
    ruler.require(witness["long_boundary"] == long_root["boundary"], "D2 long boundary differs")
    expected_long = {"path": str(Path(long_model["checkpoint"]["path"]).parent / "draws.jsonl"),
                     "sha256": long_model["boundary"]["draws_sha256"]}
    expected_short = {"path": str(Path(saved["checkpoint"]["path"]).parent / "draws.jsonl"), "sha256": boundary["draws_sha256"]}
    ruler.require(witness["long_journal"] == expected_long and witness["short_journal"] == expected_short, "D2 journal descriptor differs")
    for key in ("long_journal", "short_journal"):
        ruler.bulk_reference(witness[key], suffix=".jsonl")
    for key, expected in (("batches", 512), ("rows_per_batch", 32), ("views", 16384), ("long_batches", 4096),
                          ("long_views", 131072), ("population_rows", 17568), ("population_games", 192),
                          ("long_stream_passes", 3), ("short_stream_passes", 1)):
        ruler.require(type(witness[key]) is int and witness[key] == expected, "D2 sampler witness count differs")
    ruler.require(witness["equal"] is True and witness["all_reads_metered"] is True
                  and witness["claims"] == "Physical-row sampler prefix equality only; no mirror/full-RNG/arithmetic identity claim.",
                  "D2 sampler equality/scope differs")
    ruler.require(witness["membership_sha256"] == hashlib.sha256(ruler.canonical(boundary["membership"])).hexdigest()
                  and witness["coverage_sha256"] == hashlib.sha256(ruler.canonical(saved["draw_coverage"])).hexdigest()
                  and ruler.canonical(boundary["membership"]) == ruler.canonical(long_model["boundary"]["membership"]),
                  "D2 sampler population differs")
    ruler.require(witness["long_prefix_sha256"] == witness["short_prefix_sha256"] == boundary["draws_sha256"], "D2 sampler prefix digest differs")
    ruler.bank.sha(witness["long_prefix_sha256"])
    stamps: dict[str, dict[str, Any]] = {}
    for key in ("long_pre", "long_post", "long_final", "short_observed"):
        value = witness[key]
        ruler.require(type(value) is dict and set(value) == {"stamp", "bytes", "sha256"}, "D2 sampler file witness differs")
        observed = value["stamp"]
        ruler.require(type(observed) is dict and set(observed) == {"st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode"}
                      and all(type(v) is int and v >= 0 for v in observed.values())
                      and stat.S_ISREG(observed["st_mode"]) and not observed["st_mode"] & 0o222
                      and type(value["bytes"]) is int and 0 < value["bytes"] == observed["st_size"] <= 64 << 20,
                      "D2 sampler stable-file stamp differs")
        ruler.bank.sha(value["sha256"])
        stamps[key] = value
    ruler.require(stamps["long_pre"] == stamps["long_post"] == stamps["long_final"]
                  and stamps["long_pre"]["sha256"] == expected_long["sha256"]
                  and stamps["short_observed"]["sha256"] == expected_short["sha256"], "D2 sampler file changed")
    ruler.require(type(witness["long_raw_read_bytes"]) is int
                  and witness["long_raw_read_bytes"] == 3 * stamps["long_pre"]["bytes"] <= 192 << 20
                  and type(witness["short_raw_read_bytes"]) is int
                  and witness["short_raw_read_bytes"] == stamps["short_observed"]["bytes"], "D2 sampler read budget differs")
    ruler.require(type(saved["additional_wrapper_read_bytes"]) is int
                  and witness["long_raw_read_bytes"] + witness["short_raw_read_bytes"] <= saved["additional_wrapper_read_bytes"] <= 4 << 30,
                  "D2 total wrapper read budget differs")


def _short_payload(m: ruler.Metadata, root: dict[str, Any], c: dict[str, Any], boundary: dict[str, Any],
                   suffix: list[str], label: str, donor: dict[str, Any]) -> dict[str, Any]:
    """Accept only the new reviewed observer, never project its status to D1."""
    audit = root["payload_audit"]
    ruler.require(type(audit) is dict and set(audit) == {"schema", "readout", "owner"} and audit["schema"] == SHORT_SCHEMA,
                  "D2 requires its normal GPU proof family")
    ruler.require(audit["owner"] == root["learner_owner"], "D2 proof must share complete learner owner")
    admission = _closed_owner(m, audit["owner"], suffix)
    saved = m.read(audit["readout"])
    reconciled = {"model", "optimizer", "scheduler", "zclip", "counters", "rng", "journal", "membership", "policy_anchor"}
    ruler.require(saved["status"] == SHORT_PENDING and saved["payload_schema"] == SHORT_SCHEMA
                  and saved["study_uuid"] == STUDY and saved["arm"] == label
                  and all(saved[k] == root[k] for k in ("contract", "stage", "boundary", "checkpoint", "source_revision", "source_checkout"))
                  and saved["architecture"] == c["architecture"] and type(saved["step"]) is int and saved["step"] == 512
                  and saved["public_suffix"] == suffix and type(saved["reconciled"]) is dict
                  and set(saved["reconciled"]) == reconciled and all(v is True for v in saved["reconciled"].values()),
                  "D2 own-live saved proof incomplete")
    plan = m.read(saved["observer_plan"])
    keys = {"schema", "status", "study_uuid", "arm", "design", "helper", "source_review", "source_pins", "qualification", "assets", "runtime", "owner",
            "stage_name", "contract", "contract_template", "stage", "accepted_bank", "train_protection", "N", "K", "learner_output", "observation",
            "limits", "issued_utc", "deadline_utc", "public_suffix", "long_comparator"}
    ruler.require(set(plan) == keys and plan["schema"] == "SCIENTIFIC_D2_SHORT_HORIZON_PASSIVE_PROOF_V1"
                  and plan["status"] == "BOUND_D2_SHORT_HORIZON_PASSIVE_PROOF_NOT_DELIVERY_AUTHORITY_V1"
                  and plan["study_uuid"] == STUDY and plan["arm"] == label and plan["design"] == DESIGN
                  and admission["observer_plan"] == saved["observer_plan"]
                  and all(saved[k] == plan[k] == admission[k] for k in ("helper", "source_review", "source_pins"))
                  and all(plan[k] == root[k] for k in ("contract", "stage")) and plan["public_suffix"] == suffix
                  and plan["contract_template"] == SHORT_CONTRACT_TEMPLATES[label]
                  and plan["learner_output"] == str(Path(root["checkpoint"]["path"]).parent)
                  and plan["observation"] == audit["readout"]["path"] and plan["accepted_bank"] == TRAIN_BANK
                  and plan["limits"] == {"outer_seconds": 3600, "inner_seconds": 3570, "checkpoint_file_bytes": 8 << 30,
                                         "journal_bytes": 64 << 20, "aggregate_bytes": 4 << 20}
                  and all(type(v) is int for v in plan["limits"].values())
                  and type(saved["outer_seconds"]) is int and saved["outer_seconds"] == 3600
                  and type(saved["absolute_inner_seconds"]) is int and saved["absolute_inner_seconds"] == 3570,
                  "D2 plan/admission proof differs")
    ruler.require(m.read(plan["contract_template"]) == c, "D2 selected contract template differs")
    helper = saved["helper"]
    ruler.require(type(helper) is dict and set(helper) == {"path", "sha256"} and type(helper["path"]) is str
                  and Path(helper["path"]).is_absolute() and Path(helper["path"]).suffix == ".py"
                  and helper["sha256"] == SHORT_HELPER_SHA256 and audit["owner"]["command"].count(helper["path"]) == 1
                  and saved["source_review"] == SHORT_SOURCE_REVIEW, "D2 observer source identity differs")
    ruler.bank.sha(helper["sha256"])
    review = m.read(saved["source_review"])
    ruler.require(review["status"] == SHORT_REVIEW_STATUS and review["helper"] == helper
                  and review["source_pins"] == saved["source_pins"] == LEARNER_PINS and review["findings"] == [],
                  "D2 observer independent source acceptance differs")
    pins = m.read(LEARNER_PINS)
    ruler.require(pins["source_revision"] == root["source_revision"] == LEARNER_SOURCE
                  and pins["source_checkout"] == root["source_checkout"] and pins["files"] == c["source_files"]
                  and type(pins["files"]) is dict and len(pins["files"]) == 219, "D2 learner source closure differs")
    ruler.require(saved["save_before_equals_after"] is True and saved["reference_immutable"] is True
                  and type(saved["additional_final_CPU_checkpoint_decodes"]) is int
                  and saved["additional_final_CPU_checkpoint_decodes"] == 1, "D2 save/reference proof differs")
    ruler.bank.sha(saved["own_live_state_fingerprint"])
    initial = saved["initial"]
    for key, expected in (("state_entries", 494), ("input_planes", 175), ("unique_parameter_numel", 61444448), ("naive_state_numel", 77173088)):
        ruler.require(type(initial["model"][key]) is int and initial["model"][key] == expected, "D2 full494 model facts differ")
    ruler.require(initial["model"]["finite"] is True
                  and initial["actual_student_device"] == "cuda:0" and initial["actual_Trainer_use_amp"] is True
                  and saved["initial_model_state_sha256"] == initial["model"]["state_sha256"] == donor["model_state_sha256"],
                  "D2 original full A39 state differs")
    ruler.bank.sha(saved["initial_model_state_sha256"])
    ruler.bank.integer(saved["optimizer_named_parameters"], "D2 named optimizer parameters")
    ruler.bank.integer(saved["active_optimizer_slots"], "D2 active optimizer slots")
    ruler.require(saved["active_optimizer_slots"] <= saved["optimizer_named_parameters"], "D2 optimizer ownership differs")
    rng = boundary["rng"]
    ruler.require(type(rng) is dict and set(rng) == {"numpy", "mask", "torch_cpu", "torch_cuda", "python"}
                  and type(rng["torch_cuda"]) is list and len(rng["torch_cuda"]) == 1
                  and saved["five_rng_snapshot_sha256"] == hashlib.sha256(ruler.canonical(rng)).hexdigest()
                  and saved["membership_sha256"] == hashlib.sha256(ruler.canonical(boundary["membership"])).hexdigest()
                  and ruler.canonical(initial["membership"]) == ruler.canonical(boundary["membership"])
                  and saved["actual_backend"] == initial["backend"] == boundary["backend"], "D2 five-stream/membership/backend proof differs")
    ruler.require(type(saved["Torch_intra"]) is int and saved["Torch_intra"] == 2
                  and type(saved["Torch_interop"]) is int and saved["Torch_interop"] == 2
                  and type(initial["backend"]["torch_threads"]) is int and initial["backend"]["torch_threads"] == 2
                  and type(initial["backend"]["interop_threads"]) is int and initial["backend"]["interop_threads"] == 2,
                  "D2 thread-two evidence differs")
    coverage = saved["draw_coverage"]
    ruler.require(type(plan["N"]) is int and plan["N"] == 17568 and type(plan["K"]) is int and plan["K"] == 512
                  and type(coverage["population_rows"]) is int and coverage["population_rows"] == 17568
                  and type(coverage["batches"]) is int and coverage["batches"] == 512
                  and type(coverage["views"]) is int and coverage["views"] == boundary["drawn_rows"] == boundary["applied_views"] == 16384
                  and coverage["journal_sha256"] == boundary["draws_sha256"]
                  and type(coverage["views_per_population_row"]) is float and coverage["views_per_population_row"] == 16384 / 17568,
                  "D2 complete sampler coverage differs")
    _prefix_proof(m, saved, plan, boundary, label, donor)
    return saved


def model_delivery(m: ruler.Metadata, binding: dict[str, Any], label: str,
                   donor: dict[str, Any]) -> dict[str, Any]:
    """Explicit D2 dispatch; historical labels and receipts are never relabelled."""
    if label in LONG_LABELS:
        return retained_model(m, binding, label, donor)
    ruler.require(label in SHORT_COEFFICIENTS and set(binding) == {"root", "independent"}, "D2 model family differs")
    root, review = m.read(binding["root"]), m.read(binding["independent"])
    keys = {"status", "contract", "stage", "boundary", "checkpoint", "source_revision", "source_checkout", "learner_owner", "payload_audit"}
    ruler.require(set(root) == keys and root["status"] == ruler.DELIVERY_ROOT
                  and set(review) == {"status", "root_acceptance", "contract", "stage", "boundary", "checkpoint", "payload_audit", "findings"}
                  and review["status"] == ruler.DELIVERY_REVIEW and review["root_acceptance"] == binding["root"] and review["findings"] == []
                  and all(review[k] == root[k] for k in ("contract", "stage", "boundary", "checkpoint", "payload_audit")),
                  "D2 normal GPU delivery envelope differs")
    c, stage, boundary = (m.read(root[k]) for k in ("contract", "stage", "boundary"))
    source = m.read(LEARNER_PINS)
    short_contract(c, label, donor, source["files"])
    ruler.require(stage == {"schema": "core_generation_bank_policy_anchor_stage_v1", "contract_sha256": root["contract"]["sha256"],
                            "segment": 1, "parent": None, "banks": [TRAIN_BANK], "output": str(Path(root["checkpoint"]["path"]).parent)},
                  "D2 exact fresh stage differs")
    ruler.require(boundary["status"] == ruler.BOUNDARY and boundary["contract_sha256"] == root["contract"]["sha256"]
                  and boundary["stage"] == root["stage"] and boundary["checkpoint"] == root["checkpoint"]
                  and boundary["source_revision"] == root["source_revision"] == c["source_revision"]
                  and source["source_checkout"] == root["source_checkout"]
                  and boundary["run_uuid"] == c["run_uuid"] and type(boundary["logical_generation"]) is int
                  and boundary["logical_generation"] == 1 and type(boundary["segment"]) is int and boundary["segment"] == 1
                  and boundary["initial_model"] == c["initial_model"] and type(boundary["initial_actor_step"]) is int
                  and boundary["initial_actor_step"] == 680276 and boundary["parent"] is None
                  and type(boundary["step_start"]) is int and boundary["step_start"] == 0
                  and all(type(boundary[k]) is int and boundary[k] == 512 for k in ("step", "applied_updates"))
                  and all(type(boundary[k]) is int and boundary[k] == 16384 for k in ("applied_views", "drawn_rows"))
                  and boundary["policy_anchor"] == c["policy_anchor"], "D2 complete fresh endpoint differs")
    suffix = ["-m", "scripts.core_generation_bank_policy_anchor", "--contract", root["contract"]["path"], "--contract-sha256", root["contract"]["sha256"],
              "--stage", root["stage"]["path"], "--stage-sha256", root["stage"]["sha256"]]
    saved = _short_payload(m, root, c, boundary, suffix, label, donor)
    ruler.bank.sha(saved["model_state_sha256"])
    ruler.require(boundary["accepted_banks"] == stage["banks"] and type(boundary["bank_identities"]) is list
                  and len(boundary["bank_identities"]) == 1, "D2 actual TRAIN bank horizon differs")
    checkpoint = ruler.core.external(root["checkpoint"]["path"])
    output = Path(stage["output"])
    ruler.require(checkpoint == output / "learner.pt" and not output.with_name(output.name + ".failed.json").exists()
                  and not (output / "boundary.pending.json").exists(), "D2 failed/pending endpoint")
    tag = {k: boundary[k] for k in ("run_uuid", "source_revision", "segment", "logical_generation", "policy_anchor")}
    tag["contract_sha256"] = root["contract"]["sha256"]
    model = {"label": label, "checkpoint": root["checkpoint"], "architecture": c["architecture"], "step": 512,
             "tag": tag, "model_state_sha256": saved["model_state_sha256"], "contract": c, "boundary": boundary, "stage": stage}
    train = ruler.training_population(m, model)
    ruler.require(len(train.games) == 192 and train.evidence["eligible_rows"] == 17568, "D2 full192 TRAIN population differs")
    _short_game_coverage(saved["draw_coverage"], model, train)
    return model


def _short_game_coverage(coverage: dict[str, Any], model: dict[str, Any], train: Any) -> None:
    rows = coverage["per_game_aggregate_counts"]
    ruler.require(type(rows) is list and len(rows) == len(train.games) == 192
                  and type(coverage["games"]) is int and coverage["games"] == 192, "D2 game roster differs")
    draws = unique = zero = population = 0
    for row, game in zip(rows, train.games, strict=True):
        ruler.require(type(row) is dict and set(row) == {"physical_game_id", "population_rows", "draws", "unique_rows", "unique_fraction"}
                      and row["physical_game_id"] == game["physical_game_id"] and type(row["population_rows"]) is int
                      and row["population_rows"] == game["rows"], "D2 physical game differs")
        count = ruler.bank.integer(row["draws"], "D2 game draws", 0)
        seen = ruler.bank.integer(row["unique_rows"], "D2 game unique rows", 0)
        ruler.require(seen <= min(count, game["rows"]) and (seen > 0) == (count > 0)
                      and type(row["unique_fraction"]) is float and row["unique_fraction"] == seen / game["rows"], "D2 game coverage differs")
        draws += count
        unique += seen
        population += game["rows"]
        zero += count == 0
    ruler.require(type(coverage["unique_physical_rows_drawn"]) is int and coverage["unique_physical_rows_drawn"] == unique
                  and type(coverage["zero_draw_games"]) is int and coverage["zero_draw_games"] == zero
                  and population == coverage["population_rows"] == 17568 and draws == coverage["views"] == model["boundary"]["drawn_rows"]
                  and type(coverage["unique_fraction"]) is float and coverage["unique_fraction"] == unique / population,
                  "D2 aggregate game coverage differs")


def role_authority(m: ruler.Metadata, binding: dict[str, Any]) -> dict[str, Any]:
    """Read closed role/exposure receipts, not a book, roster or parent payload."""
    ruler.require(set(binding) == {"root", "independent"}, "D2 role binding differs")
    root, independent = m.read(binding["root"]), m.read(binding["independent"])
    keys = {"status", "study_uuid", "design", "source_roles", "prefixes", "original_six400_books", "protocol", "readout", "owner", "exposure_registry"}
    ruler.require(set(root) == keys and root["status"] == "ROOT_ACCEPTS_CLOSED_D2_FRESH_SCREEN_A_80_159_METADATA_V1"
                  and root["study_uuid"] == STUDY and root["design"] == DESIGN and root["source_roles"] == SOURCE_ROLES
                  and set(independent) == keys | {"root_acceptance", "findings"}
                  and independent["status"] == "ACCEPT_CLOSED_D2_FRESH_SCREEN_A_80_159_METADATA_NO_FINDINGS_V1"
                  and independent["root_acceptance"] == binding["root"] and independent["findings"] == []
                  and all(independent[k] == root[k] for k in keys - {"status"}), "D2 role acceptance differs")
    protocol, report = m.read(root["protocol"]), m.read(root["readout"])
    ruler.require(protocol["schema"] == "D2_FRESH_SLICE_METADATA_CPU_PROTOCOL_V1"
                  and protocol["status"] == "BOUND_D2_FRESH_SLICE_METADATA_ONLY_V1"
                  and protocol["study_uuid"] == STUDY and protocol["design"] == DESIGN
                  and {"root": protocol["role_root"], "independent": protocol["role_independent"]} == SOURCE_ROLES
                  and protocol["exposure_registry"] == root["exposure_registry"]
                  and type(protocol["limits"]["outer_seconds"]) is int and protocol["limits"]["outer_seconds"] == 300
                  and type(protocol["limits"]["inner_seconds"]) is int and protocol["limits"]["inner_seconds"] == 270
                  and report["status"] == "D2_FRESH_SLICE_METADATA_CANDIDATE_PENDING_ROOT_CLOSURE_V1"
                  and report["protocol"] == root["protocol"] and report["design"] == DESIGN
                  and report["role_root"] == protocol["role_root"] and report["role_independent"] == protocol["role_independent"]
                  and report["prefixes"] == root["prefixes"] and report["original_six400_books"] == root["original_six400_books"]
                  and report["exposure_registry"] == root["exposure_registry"]
                  and report["complete_parent_descriptors_preserved"] is True
                  and type(report["controls_passed"]) is int and report["controls_passed"] == 16,
                  "D2 actual fresh-role protocol/readout differs")
    helper, source_review = protocol["helper"], m.read(protocol["source_review"])
    ruler.require(type(helper) is dict and set(helper) == {"path", "sha256"} and type(helper["path"]) is str
                  and Path(helper["path"]).is_absolute() and Path(helper["path"]).suffix == ".py" and helper["sha256"] == ROLE_HELPER_SHA256
                  and protocol["source_review"] == ROLE_SOURCE_REVIEW
                  and source_review["status"] == "ACCEPT_KL_DEVELOPMENT_FIRST_D2_PREFIX_SOURCE_ONLY_NO_FINDINGS_V1"
                  and source_review["helper"] == helper and source_review["controls"] == protocol["controls"]
                  and source_review["source_pins"] == protocol["source_pins"] and source_review["findings"] == [],
                  "D2 role producer source differs")
    owner = root["owner"]
    suffix = [helper["path"], "--protocol", root["protocol"]["path"], "--protocol-sha256", root["protocol"]["sha256"],
              "--admission", owner["admission"]["path"], "--admission-sha256", owner["admission"]["sha256"]]
    admission = _closed_owner(m, owner, suffix, role=True)
    ruler.require(admission["protocol"] == root["protocol"] and all(admission[k] == protocol[k] for k in ("helper", "source_review", "source_pins"))
                  and owner["command"].count(helper["path"]) == 1, "D2 role owner/source joins differ")
    original, original_ind = m.read(SOURCE_ROLES["root"]), m.read(SOURCE_ROLES["independent"])
    ruler.require(original["status"] == "ROOT_ACCEPTS_CLOSED_PROSPECTIVE_WHOLE_GAME_ROLES_V1"
                  and original_ind["status"] == "ACCEPT_CLOSED_PROSPECTIVE_WHOLE_GAME_ROLES_NO_FINDINGS_V1"
                  and original_ind["root_acceptance"] == SOURCE_ROLES["root"] and original_ind["findings"] == []
                  and root["original_six400_books"] == original["books"] and len(original["books"]) == 6,
                  "D2 original role ancestry differs")
    ruler.require(type(root["prefixes"]) is dict and set(root["prefixes"]) == {"SCREEN_C_A"}, "D2 requires one common fresh panel")
    value = root["prefixes"]["SCREEN_C_A"]
    ruler.require(type(value) is dict and set(value) == {"source_role", "original400_book", "original_full_role", "book", "parent_map", "selection",
                  "prefix_start", "pairs", "source_count", "complete_parents", "actual_public_loader_count", "no_skips_dedup_shrink",
                  "parent_order_sha256", "history_order_sha256"}
                  and value["source_role"] == "SCREEN_C_A" and value["original400_book"] == original["books"]["SCREEN_C_A"]
                  and value["original_full_role"] == original["roles"]["SCREEN_C_A"]
                  and type(value["prefix_start"]) is int and value["prefix_start"] == 80
                  and type(value["source_count"]) is int and value["source_count"] == 400
                  and all(type(value[k]) is int and value[k] == 80 for k in ("pairs", "complete_parents", "actual_public_loader_count"))
                  and value["no_skips_dedup_shrink"] is True
                  and value["selection"] == "exact original ordinals80..159; no rng/subsample/reorder", "D2 exact fresh slice differs")
    ruler.bulk_reference(value["book"], suffix=".epd")
    ruler.bank.reference(value["parent_map"])
    ruler.bank.sha(value["parent_order_sha256"])
    ruler.bank.sha(value["history_order_sha256"])
    domains = report["all8role_domains_disjoint"]
    ruler.require(type(domains) is dict and set(domains) == {"group", "parent", "history", "epd", "history_sha", "input"}
                  and all(type(count) is int for count in domains.values())
                  and all(domains[k] == 2656 for k in ("group", "history", "epd", "history_sha"))
                  and domains["input"] == 5312 and domains["parent"] >= 2656, "D2 full role domain counts differ")
    consumed = report["public_consumption"]
    expected = [(ref, 400) for ref in original["books"].values()] + [(value["book"], 80)]
    ruler.require(type(consumed) is list and len(consumed) == 7
                  and all(type(row) is dict and set(row) == {"book", "pairs", "actual_public_reads", "full16_history_order_EPD_exact"}
                          and type(row["pairs"]) is int and type(row["actual_public_reads"]) is int
                          and row["actual_public_reads"] == 1 and row["full16_history_order_EPD_exact"] is True for row in consumed)
                  and all(sum(row == {"book": ref, "pairs": count, "actual_public_reads": 1, "full16_history_order_EPD_exact": True}
                              for row in consumed) == 1 for ref, count in expected), "D2 seven actual public book reads differ")
    _known_exposure(m, root, report)
    return value


def _known_exposure(m: ruler.Metadata, root: dict[str, Any], report: dict[str, Any]) -> None:
    refs = root["exposure_registry"]
    ruler.require(type(refs) is dict and set(refs) == {"root", "independent"}, "D2 known exposure refs differ")
    registry, independent = m.read(refs["root"]), m.read(refs["independent"])
    keys = {"status", "study_uuid", "design", "source_roles", "prior_closed_owner_count", "known_usage", "D1_readout", "D1_review",
            "usage_evidence", "no_other_known_controlled_role_usage", "external_usage_claim", "scope"}
    expected = {"TRAIN": [[0, 192]], "HOLD": [[0, 64]], "SCREEN_C_A": [[0, 80]], "SCREEN_C_B": [[0, 80]],
                "SCREEN_C_A39": [], "CONFIRM_C_A": [], "CONFIRM_C_B": [], "CONFIRM_C_A39": []}
    ruler.require(set(registry) == keys and registry["status"] == "ROOT_SELECTS_KNOWN_EXPOSURE_HISTORY_BEFORE_D2_ROLE_QUALIFICATION_V1"
                  and registry["study_uuid"] == STUDY and registry["design"] == DESIGN and registry["source_roles"] == SOURCE_ROLES
                  and type(registry["prior_closed_owner_count"]) is int and registry["prior_closed_owner_count"] == 406
                  and registry["known_usage"] == expected and registry["no_other_known_controlled_role_usage"] is True
                  and registry["external_usage_claim"] is False
                  and set(independent) == keys | {"root_acceptance", "findings"}
                  and independent["status"] == "ACCEPT_ROOT_KNOWN_EXPOSURE_HISTORY_BEFORE_D2_ROLE_QUALIFICATION_NO_FINDINGS_V1"
                  and independent["root_acceptance"] == refs["root"] and independent["findings"] == []
                  and all(independent[k] == registry[k] for k in keys - {"status"}), "D2 reviewed known exposure registry differs")
    ruler.require(all(type(interval) is list and len(interval) == 2 and all(type(v) is int for v in interval)
                      for intervals in registry["known_usage"].values() for interval in intervals), "D2 strict exposure intervals differ")
    d1_ref, d1_ind = D1_NO_WINNER, D1_NO_WINNER_INDEPENDENT
    ruler.require(registry["D1_readout"] == d1_ref and registry["D1_review"] == d1_ind, "D2 known D1 exposure identity differs")
    d1, d1i = m.read(d1_ref), m.read(d1_ind)
    ruler.require(d1["all_seven_technically_complete"] is True and d1i["readout"] == d1_ref and d1i["findings"] == [], "D2 prior D1 closure differs")
    ruler.require(type(registry["usage_evidence"]) is list and len(registry["usage_evidence"]) <= 32, "D2 bounded exposure evidence differs")
    for evidence in registry["usage_evidence"]:
        m.read(evidence)
    check = report["known_exposure_check"]
    ruler.require(type(check) is dict and set(check) == {"registry", "prior_closed_owner_count", "known_usage", "conservatively_used_rows",
                  "selected_original_ordinals", "known_controlled_domains_disjoint", "selected_domain_counts", "scope"}
                  and check["registry"] == refs and type(check["prior_closed_owner_count"]) is int and check["prior_closed_owner_count"] == 406
                  and check["known_usage"] == expected and type(check["conservatively_used_rows"]) is int and check["conservatively_used_rows"] == 416
                  and type(check["selected_original_ordinals"]) is list and check["selected_original_ordinals"] == [80, 160]
                  and all(type(v) is int for v in check["selected_original_ordinals"]) and check["known_controlled_domains_disjoint"] is True,
                  "D2 actual known-exposure check differs")
    counts = check["selected_domain_counts"]
    ruler.require(type(counts) is dict and set(counts) == {"group", "parent", "history", "epd", "history_sha", "input"}
                  and all(type(count) is int for count in counts.values())
                  and all(counts[k] == 80 for k in ("group", "history", "epd", "history_sha"))
                  and counts["input"] == 160 and counts["parent"] >= 80, "D2 selected exposure counts differ")
