"""One fresh accepted-bank pass with full-state, cumulative capped continuation.

Each cycle has an immutable new population/order and one closed output. There
is no in-window retry, optimizer reset, actor selection, recollection or launch.
The original lineage's persistent ledger continues across contract/study UUIDs.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, cast
import uuid

from chess_anti_engine.policy_anchor_spec import validate_spec
from chess_anti_engine.replay import fixed_population
from chess_anti_engine.replay.capped_cycle import Usage, read_usage, row_dicts
from chess_anti_engine.replay.stored_row_once import ORDER, SAMPLING, QuotaLedger, StoredRowOnceReplay, digest, permutation_for
from scripts import core_finite_learner as core
from scripts import core_generation_bank_acceptance as admission
from scripts import core_generation_bank_learner as bank
from scripts import core_selfplay_union_manifest as union
from scripts import core_selfplay_union_stored_row_once as global_once

CONTRACT = "core_capped_on_policy_cycle_v1"
STAGE = "core_capped_on_policy_cycle_stage_v1"
BOUNDARY = "COMPLETED_CORE_CAPPED_ON_POLICY_CYCLE_V1"
CHECKPOINT_KEY = CONTRACT
SOURCE_FILES = (*global_once.SOURCE_FILES, "scripts/core_closed_once_actor.py",
                "scripts/core_capped_on_policy_cycle.py", "scripts/core_capped_on_policy_cycle_proof.py",
                "chess_anti_engine/replay/capped_cycle.py", "scripts/core_selfplay_union_policy_anchor_stored_row_once_proof.py",
                "chess_anti_engine/policy_anchor_spec.py", "chess_anti_engine/train/frozen_policy_anchor.py")


DeliveryResolver = Callable[[dict[str, Any]], dict[str, Any]]
PayloadValidator = Callable[[dict[str, Any], dict[str, Any], Any], str]
ROOT_FIELDS = {"status", "study_uuid", "method", "contract", "boundary", "closed_binding",
               "entry_source", "entry_source_review", "model"}
MODEL_FIELDS = {"checkpoint", "architecture", "step", "model_state_sha256",
                "model_state_hash_schema", "label", "tag_key", "tag"}
PARENT_TAGS = (CHECKPOINT_KEY, "core_selfplay_union_stored_row_once_v1", "core_selfplay_union_policy_anchor_stored_row_once_v1")


def recipe(c: dict[str, Any]) -> dict[str, Any]:
    return {k: c[k] for k in ("batch_size", "seed", "runtime", "lr", "mirror_prob", "policy_anchor",
                              "retained_search_share", "maximum_presentations", "sampling", "population", "cycle")}


def _validate_contract_metadata(c: dict[str, Any]) -> None:
    core.require(set(c) == {"schema", "study_uuid", "run_uuid", "cycle", "method", "source_revision", "source_files",
        "architecture", "origin", "parent", "initial_model", "initial_model_step", "population", "bank",
        "batch_size", "updates_per_segment", "seed", "runtime", "lr", "mirror_prob", "limits", "bank_limits",
        "sampling", "policy_anchor", "retained_search_share", "maximum_presentations"} and c["schema"] == CONTRACT,
        "invalid capped cycle contract")
    for k in ("study_uuid", "run_uuid"):
        core.require(type(c[k]) is str and str(uuid.UUID(c[k])) == c[k], "cycle UUID")
    core.require(1 <= core.integer(c["cycle"], "cycle") <= 2**31 - 1, "cycle counter bound")
    core.require(type(c["source_files"]) is dict and set(SOURCE_FILES) <= set(c["source_files"]), "cycle source closure")
    core.require(type(c["source_revision"]) is str and re.fullmatch("[0-9a-f]{40}", c["source_revision"]) is not None, "cycle source revision")
    for name, value in cast(dict[str, Any], c["source_files"]).items():
        core.require(type(name) is str and (core.REPO / name).resolve().is_relative_to(core.REPO)
            and not Path(name).is_absolute() and type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None, "cycle source pin shape")
    core.require(type(c["maximum_presentations"]) is int and 1 <= c["maximum_presentations"] <= 2,
                 "cycle maximum2 total presentations")
    b = core.integer(c["batch_size"], "cycle batch size")
    core.integer(c["initial_model_step"], "cycle recorded initial model step", 0)
    n = core.integer(c["population"]["rows"], "cycle population rows")
    core.require(type(c["updates_per_segment"]) is list and len(c["updates_per_segment"]) == 1
        and type(c["updates_per_segment"][0]) is int
        and c["updates_per_segment"] == [(n + b - 1) // b], "cycle one complete finite pass")
    core.require(type(c["retained_search_share"]) in (int, float) and math.isfinite(c["retained_search_share"])
                 and 0 <= c["retained_search_share"] <= 1, "cycle retained search share")
    if c["policy_anchor"] is not None:
        core.require(validate_spec(c["policy_anchor"], c["architecture"]) == c["policy_anchor"], "cycle frozen reference")
    core.require(0 <= core.number(c["mirror_prob"], "cycle mirror probability") <= 1, "cycle mirror probability")
    core.require(0 <= core.integer(c["seed"], "cycle seed", 0) < 2**32, "cycle seed bound")
    rt = c["runtime"]
    core.require(set(rt) == {"device", "use_amp", "torch_version", "numpy_version", "torch_threads", "cuda_devices"}
        and rt["device"] in {"cpu", "cuda:0"} and type(rt["use_amp"]) is bool
        and type(rt["torch_threads"]) is int and rt["torch_threads"] == 2
        and type(rt["cuda_devices"]) is int and rt["cuda_devices"] == (0 if rt["device"] == "cpu" else 1)
        and all(type(rt[k]) is str and rt[k] for k in ("torch_version", "numpy_version")), "cycle runtime")
    lr = c["lr"]
    core.require(set(lr) == {"peak", "floor", "warmup_steps", "warmup_start", "first_period"}
        and core.number(lr["peak"], "cycle LR") > 0 and lr["peak"] == lr["floor"] == lr["warmup_start"]
        and type(lr["warmup_steps"]) is int and lr["warmup_steps"] == 0, "cycle explicit flat LR")
    core.integer(lr["first_period"], "cycle LR horizon")
    limits = c["limits"]
    core.require(set(limits) == {"max_rows", "max_decoded_bytes", "max_file_bytes", "max_total_metadata_bytes",
                               "max_lineage_rows"}, "cycle limits")
    for k, v in limits.items():
        core.integer(v, k)
    core.require(n <= limits["max_rows"] and limits["max_file_bytes"] <= 128 << 20, "cycle row/metadata bounds")
    core.require(set(c["bank_limits"]) == {"max_file_bytes", "max_total_bytes", "max_games"}, "cycle bank metadata limits")
    for k, v in c["bank_limits"].items():
        core.integer(v, k)
    core.require(c["bank_limits"]["max_file_bytes"] <= 128 << 20
        and type(c["bank"]) is dict and set(c["bank"]) == {"root_acceptance", "independent_review"}, "cycle accepted bank binding/cap")
    population = c["population"]
    core.require(set(population) == {"manifest", "plan", "origin_map_sha256", "membership_sha256", "sources", "rows"},
                 "cycle population identity")
    core.integer(population["sources"], "cycle source count")
    sampling = c["sampling"]
    core.require(set(sampling) == {"schema", "order_seed", "expected_rows", "presentations_per_row", "mirror_shares_quota",
        "partial_batch", "lineage_uuid", "quota_ledger", "order", "permutation_sha256"}
        and sampling["schema"] == SAMPLING and type(sampling["expected_rows"]) is int and sampling["expected_rows"] == n
        and type(sampling["presentations_per_row"]) is int and sampling["presentations_per_row"] == 1
        and sampling["mirror_shares_quota"] is True and sampling["partial_batch"] == "EMIT_FINAL_REMAINDER_ONCE_NO_PADDING"
        and sampling["lineage_uuid"] == c["run_uuid"] and sampling["order"] == ORDER
        and type(sampling["permutation_sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", sampling["permutation_sha256"]) is not None,
        "cycle finite permutation/quota recipe")
    core.integer(sampling["order_seed"], "cycle order seed", 0)
    core.external(sampling["quota_ledger"])
    _, method = core.read_ref(c["method"])
    core.require(method["schema"] == "CORE_CAPPED_ON_POLICY_CYCLE_METHOD_V1" and method["study_uuid"] == c["study_uuid"]
                 and method["origin"] == c["origin"] and method["cycle_recipe"] == recipe(c),
                 "cycle recipe differs from caller-selected method")


def validate_contract(c: dict[str, Any]) -> None:
    """Owned virtual-order check; metadata actor/Root joins do not generate it."""
    _validate_contract_metadata(c)
    core.require(c["sampling"]["permutation_sha256"] == digest(permutation_for(c["population"]["rows"], c["sampling"]["order_seed"])),
                 "cycle selected permutation digest")


def compact_metadata(value: Any) -> int:
    size = len((core.canonical(value) + "\n").encode())
    core.require(size <= 4 << 20, "cycle closed metadata exceeds4MiB")
    return size


def source_guard(c: dict[str, Any]) -> None:
    bank.source_guard(c)
    for module, name in ((union, "scripts/core_selfplay_union_manifest.py"),
                         (global_once, "scripts/core_selfplay_union_stored_row_once.py")):
        core.require(Path(inspect.getfile(module)).resolve() == core.REPO / name, "foreign capped cycle helper")
    core.require(Path(__file__).resolve() == core.REPO / "scripts/core_capped_on_policy_cycle.py"
        and Path(inspect.getfile(read_usage)).resolve() == core.REPO / "chess_anti_engine/replay/capped_cycle.py",
        "foreign capped cycle source")


def optimizer_recipe(trainer: Any) -> dict[str, Any]:
    """Actual constructor controls, excluding slot tensors and schedule progress.

    Exact resume restores the parent's controls too. Compare these to the new
    constructor before any draw so a declared recipe cannot be silently ignored.
    Flat LR plus the cumulative horizon guard keeps T_i and every group LR fixed.
    """
    scheduler = trainer._scheduler
    core.require(type(scheduler).__name__ == "CosineAnnealingWarmRestarts"
        and type(scheduler).__module__ == "torch.optim.lr_scheduler", "cycle actual scheduler family")
    value = {"optimizer_type": type(trainer.opt).__module__ + "." + type(trainer.opt).__qualname__,
        "optimizer_names": trainer._optimizer_param_names(),
        "groups": [{key: value for key, value in group.items() if key != "params"} for group in trainer.opt.param_groups],
        "scheduler": {key: scheduler.state_dict()[key] for key in ("T_0", "T_i", "T_mult", "eta_min", "base_lrs")},
        "peak_lr": float(trainer._peak_lr), "grad_clip_mode": trainer.grad_clip_mode}
    return cast(dict[str, Any], json.loads(core.canonical(value)))


def parent_metadata(c: dict[str, Any], *, resolve_delivery: DeliveryResolver) -> dict[str, Any]:
    origin = c["origin"]
    core.require(type(origin) is dict and origin.get("family") in {"model_only", "global_union", "policy_anchor"},
                 "cycle explicit origin family")
    origin = cast(dict[str, Any], origin)
    fresh_origin = origin["family"] == "model_only"
    if fresh_origin:
        core.require(set(origin) == {"schema", "family", "model", "architecture", "recorded_step"}
            and origin["schema"] == "CORE_CAPPED_CYCLE_MODEL_ONLY_ORIGIN_V1"
            and origin["model"] == c["initial_model"] and origin["architecture"] == c["architecture"]
            and type(origin["recorded_step"]) is int and origin["recorded_step"] == c["initial_model_step"],
            "cycle model-only original identity")
        global_once.reference(origin["model"], "cycle initial model")
        if c["cycle"] == 1:
            core.require(c["parent"] is None and c["lr"]["first_period"] > c["updates_per_segment"][0] - 1,
                         "initial cycle must start fresh at step0")
            return {"fresh": True, "student_step": 0, "boundary": None, "contract": None, "root": None,
                    "model": {"checkpoint": origin["model"], "architecture": c["architecture"], "step": origin["recorded_step"]}}
    elif c["cycle"] == 1:
        core.require(c["parent"] == origin, "first cycle must continue the exact selected origin")
    # Validate a decreasing counter before recursive trusted-parent joins.
    if c["cycle"] > 1:
        core.require(type(c["parent"]) is dict and c["parent"]["family"] == "capped_cycle", "cycle parent family")
        parent_descriptor = cast(dict[str, Any], c["parent"])
        _, parent_root = core.read_ref(parent_descriptor["delivery"]["root"])
        _, prior = core.read_ref(parent_root["contract"])
        core.require(type(prior["cycle"]) is int and prior["cycle"] + 1 == c["cycle"], "cycle parent must be strictly adjacent")
    parent = trusted_parent(c["parent"]) if c["cycle"] > 1 else resolve_delivery(c["parent"])
    pc, pb = parent["contract"], parent["boundary"]
    core.require(c["run_uuid"] == pc["run_uuid"] and c["sampling"]["quota_ledger"] == pc["sampling"]["quota_ledger"],
                 "cycle cannot reset original lineage/model/ledger")
    if c["cycle"] > 1:
        core.require(pc["origin"] == origin and pc["maximum_presentations"] == c["maximum_presentations"],
                     "cycle adjacent cap/origin changed")
    if fresh_origin:
        core.require(pc["initial_model"] == origin["model"] and pc["initial_model_step"] == origin["recorded_step"],
                     "cycle model-only origin changed")
    else:
        original = resolve_delivery(origin)["contract"]
        core.require(c["run_uuid"] == original["run_uuid"]
            and c["sampling"]["quota_ledger"] == original["sampling"]["quota_ledger"]
            and c["initial_model"] == original["initial_model"] and c["initial_model_step"] == original["initial_model_step"],
            "cycle cannot reset original once lineage")
    for key in ("architecture", "batch_size", "seed", "runtime", "lr", "mirror_prob"):
        core.require(c[key] == pc[key], "cycle full-state recipe changed: " + key)
    core.require(c["retained_search_share"] == pc.get("retained_search_share", 0.0)
        and c["policy_anchor"] == pc.get("policy_anchor")
        and c["lr"]["first_period"] > pb["step"] + c["updates_per_segment"][0] - 1,
        "cycle targets/reference/schedule horizon changed")
    return {**parent, "fresh": False, "student_step": parent["model"]["step"]}


def metadata(c: dict[str, Any], stage: dict[str, Any], contract_ref: dict[str, str], *, resolve_delivery: DeliveryResolver) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    core.require(set(stage) == {"schema", "contract_sha256", "segment", "cycle", "population", "parent", "output"}
        and stage["schema"] == STAGE and stage["contract_sha256"] == contract_ref["sha256"]
        and type(stage["segment"]) is int and stage["segment"] == 1
        and type(stage["cycle"]) is int and stage["cycle"] == c["cycle"]
        and stage["population"] == c["population"] and stage["parent"] == c["parent"], "cycle stage identity")
    core.require(not core.external(c["sampling"]["quota_ledger"]).is_relative_to(core.external(stage["output"])),
                 "cycle persistent quota ledger inside sealed output")
    parent = parent_metadata(c, resolve_delivery=resolve_delivery)
    accepted = admission.validate_accepted_generation(c["bank"]["root_acceptance"], c["bank"]["independent_review"], **c["bank_limits"])
    identity = bank.plain(accepted.identity)
    core.require(identity["actor_sha256"] == parent["model"]["checkpoint"]["sha256"]
        and identity["actor_checkpoint_step"] == parent["model"]["step"]
        and core.canonical(identity["architecture"]) == core.canonical(c["architecture"]),
        "new bank was not produced by exact adjacent parent actor")
    manifest = union.read_union_manifest(c["population"]["manifest"], c["population"]["plan"],
        max_file_bytes=c["limits"]["max_file_bytes"], max_total_bytes=c["limits"]["max_total_metadata_bytes"])
    population = fixed_population.manifest_identity(manifest, c["population"]["manifest"])
    branch = {"kind": "bank", **c["bank"]}
    core.require(population == {k: v for k, v in c["population"].items() if k != "plan"}
        and manifest["architecture"] == c["architecture"] and not manifest["aliases"]
        and len(manifest["origins"]) == 1 and manifest["origins"][0]["acceptance"] == branch,
        "cycle population must be the exact new accepted bank")
    core.require(sum(g["rows"] for g in accepted.games) == population["rows"]
        and len(accepted.games) == population["sources"], "cycle bank/population totals")
    return manifest, population, parent


def usage_for(c: dict[str, Any], journals: list[dict[str, str]]) -> Usage:
    return read_usage(journals, maximum=c["maximum_presentations"], max_rows=c["limits"]["max_lineage_rows"],
        batch_size=c["batch_size"], max_file_bytes=c["limits"]["max_file_bytes"],
        max_total_bytes=c["limits"]["max_total_metadata_bytes"])


def checkpoint_identity(c: dict[str, Any], ref: dict[str, str]) -> dict[str, Any]:
    return {"contract_sha256": ref["sha256"], **{k: c[k] for k in ("study_uuid", "run_uuid", "source_revision",
        "cycle", "method", "origin", "parent", "sampling", "maximum_presentations", "policy_anchor", "retained_search_share")}}


def checkpoint_tag(c: dict[str, Any], ref: dict[str, str], boundary: dict[str, Any]) -> dict[str, Any]:
    return {"identity": checkpoint_identity(c, ref), **{k: boundary[k] for k in (
        "quota_state", "draw_journals", "quota_ledger_checkpoint", "population_identity", "source_origin_binding", "usage_state")}}


def local_closed_record(contract_ref: dict[str, str], binding: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Closed metadata join; unlike the owned consumer this opens no journals."""
    _, c = core.read_ref(contract_ref)
    _validate_contract_metadata(c)
    core.require(set(binding) == {"boundary", "owner_terminal", "owner_sha256", "command"}, "cycle closed binding")
    path, b = core.read_ref(binding["boundary"])
    _, stage = core.read_ref(b["stage"])
    output = core.external(stage["output"])
    core.require(path == output / "boundary.json" and not core.failure_marker(output).exists()
        and not (output / "boundary.pending.json").exists(), "cycle closed failure/output")
    _, terminal = core.read_ref(binding["owner_terminal"])
    suffix = ["-m", "scripts.core_capped_on_policy_cycle_proof", "--contract", contract_ref["path"],
        "--contract-sha256", contract_ref["sha256"], "--stage", b["stage"]["path"], "--stage-sha256", b["stage"]["sha256"]]
    core.require(binding["command"][-10:] == suffix and terminal["command"] == binding["command"]
        and terminal["status"] == "PASS" and type(terminal["returncode"]) is int and terminal["returncode"] == 0
        and terminal["owned_processes_closed"] is True and terminal["custody"] == "OWNED_PROCESSES_CLOSED"
        and terminal["supervisor_sha256"] == binding["owner_sha256"], "cycle closed owner/command")
    core.require(set(stage) == {"schema", "contract_sha256", "segment", "cycle", "population", "parent", "output"}
        and stage["schema"] == STAGE and stage["contract_sha256"] == contract_ref["sha256"]
        and stage["segment"] == b["segment"] == 1 and stage["cycle"] == b["cycle"] == c["cycle"]
        and stage["population"] == c["population"] and stage["parent"] == b["parent"] == c["parent"], "cycle local closed stage")
    count, n = c["updates_per_segment"][0], c["population"]["rows"]
    core.require(all(type(b[k]) is int for k in ("cycle", "segment", "step_start", "step", "applied_updates", "applied_views", "drawn_rows"))
        and b["status"] == BOUNDARY and b["pass_complete"] is True and b["contract_sha256"] == contract_ref["sha256"]
        and b["study_uuid"] == c["study_uuid"] and b["run_uuid"] == c["run_uuid"] and b["method"] == c["method"]
        and b["source_revision"] == c["source_revision"] and b["sampling"] == c["sampling"]
        and b["policy_anchor"] == c["policy_anchor"] and b["step_start"] >= 0
        and b["step"] == b["step_start"] + count and b["applied_updates"] == count
        and b["applied_views"] == b["drawn_rows"] == n
        and b["population_identity"] == {k:v for k,v in c["population"].items() if k != "plan"}
        and b["source_origin_binding"] == global_once.origin_binding(b["population_identity"])
        and b["quota_state"]["cursor"] == n and b["quota_state"]["batches"] == count and b["quota_state"]["tainted"] is False
        and b["usage_state"]["batches"] == b["step"] and b["usage_state"]["maximum_presentations"] == c["maximum_presentations"]
        and b["checkpoint"]["path"] == str(output / "learner.pt")
        and b["draw_journals"][-1] == {"path":str(output / "draws.jsonl"), "sha256":b["draws_sha256"]},
        "cycle local accepted counters/identity")
    compact_metadata(b)
    return c, b, stage


def closed_record(contract_ref: dict[str, str], binding: dict[str, Any], *, resolve_delivery: DeliveryResolver) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    c, b, stage = local_closed_record(contract_ref, binding)
    manifest, identity, parent = metadata(c, stage, contract_ref, resolve_delivery=resolve_delivery)
    output = core.external(stage["output"])
    count, n = c["updates_per_segment"][0], c["population"]["rows"]
    core.require(all(type(b[k]) is int for k in ("cycle", "step_start", "step", "applied_updates", "applied_views", "drawn_rows")),
                 "cycle closed integer counters")
    core.require(b["status"] == BOUNDARY and b["contract_sha256"] == contract_ref["sha256"] and b["cycle"] == c["cycle"]
        and b["parent"] == c["parent"] and b["step_start"] == parent["student_step"]
        and b["step"] == b["step_start"] + count and b["applied_updates"] == count
        and b["applied_views"] == b["drawn_rows"] == n and b["pass_complete"] is True
        and b["population_identity"] == identity and b["sampling"] == c["sampling"]
        and b["study_uuid"] == c["study_uuid"] and b["run_uuid"] == c["run_uuid"]
        and b["source_revision"] == c["source_revision"] and b["method"] == c["method"]
        and b["policy_anchor"] == c["policy_anchor"], "cycle closed counters/identity")
    expected_membership = [[s["generation"], s["source_sha256"], s["rows"]] for s in manifest["sources"]]
    core.require(b["membership"] == expected_membership
        and b["source_origin_binding"] == global_once.origin_binding(identity)
        and b["checkpoint"]["path"] == str(output / "learner.pt")
        and type(b["metrics"]["train_steps_done"]) is int and type(b["metrics"]["train_samples_seen"]) is int
        and b["metrics"]["train_steps_done"] == count and b["metrics"]["train_samples_seen"] == n
        and all(type(b["metrics"][key]) in (int, float) and b["metrics"][key] == 0
                for key in ("grad_nonfinite_skip_rate", "transient_cuda_retry_batches")),
        "cycle closed membership/metrics/checkpoint aliases")
    core.require(b["retained_search"] == global_once.retained_search_record(c, b["metrics"],
        b["retained_search"]["population_validation"]), "cycle actual target share/coverage record")
    journals = b["draw_journals"]
    prior_journals = [] if parent["fresh"] else parent["boundary"]["draw_journals"]
    core.require(journals[:-1] == prior_journals
        and journals[-1] == {"path": str(output / "draws.jsonl"), "sha256": b["draws_sha256"]}, "cycle closed journal chain")
    return c, b, parent


def validate_closed_segment(contract_ref: dict[str, str], binding: dict[str, Any], *, resolve_delivery: DeliveryResolver) -> dict[str, Any]:
    """Owned consumer reads bounded sealed journals, not checkpoints/shards.

    The cumulative quota check is an actual journal read. Call only within a
    scope covering those lineage journals; this is not a Root-only metadata API.
    """
    c, b, _ = closed_record(contract_ref, binding, resolve_delivery=resolve_delivery)
    validate_contract(c)
    n, count = c["population"]["rows"], c["updates_per_segment"][0]
    journals = b["draw_journals"]

    class Population:
        rng = None
        membership = tuple(tuple(row) for row in b["membership"])
        def __len__(self) -> int:
            return n
    def forbidden(_rows: Any) -> None:
        raise ValueError("closed cycle consumer cannot draw")
    replay = StoredRowOnceReplay(Population(), seed=c["sampling"]["order_seed"], batch_size=c["batch_size"],
        expected_rows=n, on_draw=forbidden)
    cursor = batches = 0
    def selected_order(path: Path, record: dict[str, Any]) -> None:
        nonlocal cursor, batches
        if str(path) != journals[-1]["path"]:
            return
        lo, hi = cursor, min(n, cursor + c["batch_size"])
        expected = row_dicts(tuple(replay.row_at(i) for i in replay.permutation[lo:hi]))
        core.require(record["draw_batch"] == b["step_start"] + batches and record["rows"] == expected,
                     "cycle selected row order differs")
        replay._update_prefix(tuple(replay.row_at(i) for i in replay.permutation[lo:hi]))
        cursor, batches = hi, batches + 1
    usage = read_usage(journals, maximum=c["maximum_presentations"], max_rows=c["limits"]["max_lineage_rows"],
        batch_size=c["batch_size"], max_file_bytes=c["limits"]["max_file_bytes"],
        max_total_bytes=c["limits"]["max_total_metadata_bytes"], on_record=selected_order)
    core.require(core.canonical(usage.state()) == core.canonical(b["usage_state"]) and usage.batches == b["step"], "cycle actual cumulative usage/counters")
    core.require(usage.ledger_checkpoint(c["sampling"]["quota_ledger"]) == b["quota_ledger_checkpoint"],
                 "cycle cumulative journal/ledger bytes mismatch")
    core.require(cursor == n and batches == count and core.canonical(b["quota_state"]) == core.canonical({
        **replay.state(), "cursor": n, "batches": count, "last_batch_rows": n % c["batch_size"] or c["batch_size"]}),
        "cycle final local cursor/order state")
    return {"contract": contract_ref, "boundary": binding["boundary"], "checkpoint": b["checkpoint"],
            "steps": b["step"], "actual_views": n, "pass_complete": True, "usage_state": usage.state()}


def observed_certificate(c: dict[str, Any], b: dict[str, Any], ref: dict[str, str], parent: dict[str, Any] | None) -> dict[str, Any]:
    certificate = b["own_live_certificate"]
    count = c["updates_per_segment"][0]
    core.require(certificate["schema"] == "core_capped_cycle_own_live_saved_payload_proof_v1"
        and certificate["status"] == "CAPPED_CYCLE_OWN_LIVE_SAVED_PAYLOAD_PENDING_ROOT_CLOSURE_V1"
        and certificate["contract"] == ref and certificate["stage"] == b["stage"] and certificate["checkpoint"] == b["checkpoint"]
        and certificate["step"] == b["step"] and certificate["step_start"] == b["step_start"]
        and certificate["applied_updates"] == count and certificate["actual_views"] == b["applied_views"]
        and certificate["usage_state"] == b["usage_state"]
        and certificate["state_handoff"] == b["state_handoff"]
        and certificate["actual_calls"] == {"forward": count, "optimizer_pre": count, "optimizer_post": count, "save": 1}
        and all(type(value) is int for value in certificate["actual_calls"].values())
        and certificate["extra_forwards"] == 0 and certificate["save_before_equals_after"] is True
        and certificate["additional_final_CPU_checkpoint_decodes"] == 1
        and certificate["five_rng_sha256"] == core.digest(b["rng"])
        and certificate["journal_chain"] == b["draw_journals"]
        and certificate["ledger_checkpoint"] == b["quota_ledger_checkpoint"]
        and certificate["population_identity"] == b["population_identity"]
        and certificate["source_origin_binding"] == b["source_origin_binding"]
        and certificate["saved_model"]["finite"] is True
        and certificate["retained_search"] == b["retained_search"]
        and certificate["saved_model_state_sha256"] == certificate["saved_model"]["model_state_sha256"],
        "cycle observed actual model/state/save/usage certificate")
    if parent is None:
        # Only trusted_parent uses this: the paired previous owned consumer has
        # already checked its incoming handoff. Do not recursively reopen history.
        core.require(b["state_handoff"] in {"model_only_fresh_optimizer", "adjacent_full_state_continuation"},
                     "cycle accepted parent handoff family")
    elif parent["fresh"]:
        core.require(b["state_handoff"] == "model_only_fresh_optimizer" and b["step_start"] == 0
            and certificate["parent_initial_state_matches"] is None
            and certificate["initial"]["fresh_optimizer_slots"] == 0
            and certificate["initial"]["initial_student_step"] == 0,
            "cycle initial model-only state was not fresh")
        provenance = certificate["initial"]["initial_donor_provenance"]
        core.require(provenance["mode"] == "fresh_rl_complete_v1"
            and provenance["sha256"] == c["initial_model"]["sha256"] and provenance["architecture"] == c["architecture"],
            "cycle observed original donor provenance")
    else:
        core.require(b["state_handoff"] == "adjacent_full_state_continuation"
            and certificate["parent_initial_state_matches"] is True
            and certificate["initial"]["state"] == parent["boundary"]["own_live_certificate"]["own_live_state_fingerprint"]
            and certificate["initial"]["rng"] == parent["boundary"]["rng"]
            and certificate["initial"]["backend"] == b["backend"] == parent["boundary"]["backend"],
            "cycle adjacent full-state/RNG proof differs")
    anchor = certificate["policy_anchor"]
    core.require(anchor["identity"] == c["policy_anchor"] and anchor["teacher_unchanged"] is True
        and anchor["no_optimizer_ownership"] is True and anchor["no_reference_gradients"] is True
        and anchor["all_reference_modules_eval"] is True, "cycle observed teacher invariants")
    if c["policy_anchor"] is not None and c["policy_anchor"]["coefficient"] > 0:
        core.require(anchor["actual_reference_forwards"] == count and anchor["diagnostics"] == b["policy_anchor_diagnostics"]
            and anchor["diagnostics"]["eligible_rows"] == anchor["diagnostics"]["total_rows"] == b["applied_views"],
            "cycle actual teacher forward/eligible row coverage")
    else:
        core.require(anchor["actual_reference_forwards"] == 0 and anchor["initial_reference"] is None,
                     "cycle zero/null teacher fastpath")
    return certificate


def validate_closed_model_delivery(contract_ref: dict[str, str], binding: dict[str, Any], *, resolve_delivery: DeliveryResolver) -> dict[str, Any]:
    """Future owned delivery consumer: actual bounded journals, no model decode."""
    result = validate_closed_segment(contract_ref, binding, resolve_delivery=resolve_delivery)
    c, b, parent = closed_record(contract_ref, binding, resolve_delivery=resolve_delivery)
    certificate = observed_certificate(c, b, contract_ref, parent)
    result.update(family=CONTRACT, study_uuid=c["study_uuid"], method=c["method"],
        own_live_certificate=certificate, model={"checkpoint": b["checkpoint"], "architecture": c["architecture"],
        "step": b["step"], "model_state_sha256": certificate["saved_model_state_sha256"],
        "model_state_hash_schema": "union_typed_tensor_tree_sha256_v1", "label": "capped_cycle",
        "tag_key": CHECKPOINT_KEY, "tag": checkpoint_tag(c, contract_ref, b)})
    compact_metadata(result)
    return result


def trusted_parent(lineage: dict[str, Any]) -> dict[str, Any]:
    """Bounded adjacent post-CPU Root9/ind11 snapshot, not recursive ancestry.

    The trusted caller pins a genuine accepted pair. Its previous owned consumer
    checked journals/handoff; only local closed metadata and saved proof are read
    again. No prior models, permutations or per-row counts are copied here.
    """
    core.require(set(lineage) == {"schema", "family", "delivery"}
        and lineage["schema"] == "core_closed_once_actor_lineage_v1" and lineage["family"] == "capped_cycle",
        "cycle trusted parent lineage")
    pair = lineage["delivery"]
    core.require(type(pair) is dict and set(pair) == {"root", "independent"}, "cycle accepted delivery pair")
    pair = cast(dict[str, Any], pair)
    _, root = core.read_ref(pair["root"])
    _, independent = core.read_ref(pair["independent"])
    core.require(set(root) == ROOT_FIELDS and set(independent) == ROOT_FIELDS | {"root_acceptance", "findings"}
        and root["status"] == "ROOT_ACCEPTS_CLOSED_CAPPED_ON_POLICY_CYCLE_MODEL_DELIVERY_V1"
        and independent["status"] == "ACCEPT_CLOSED_CAPPED_ON_POLICY_CYCLE_MODEL_DELIVERY_NO_FINDINGS_V1"
        and independent["root_acceptance"] == pair["root"] and independent["findings"] == []
        and all(independent[k] == v for k, v in root.items() if k != "status"), "cycle trusted Root9/ind11 joins")
    _, source = core.read_ref(root["entry_source"])
    _, review = core.read_ref(root["entry_source_review"])
    core.require(review["status"] == "ACCEPT_CAPPED_ON_POLICY_CYCLE_SOURCE_ONLY_NO_FINDINGS_V1"
        and review["source_manifest"] == root["entry_source"] and review["findings"] == []
        and review["owned_source_files"] == source["owned_source_files"], "cycle genuine accepted source pair")
    c, b, _ = local_closed_record(root["contract"], root["closed_binding"])
    cert = observed_certificate(c, b, root["contract"], None)
    model = root["model"]
    core.require(root["boundary"] == root["closed_binding"]["boundary"] and root["method"] == c["method"]
        and root["study_uuid"] == c["study_uuid"] and set(model) == MODEL_FIELDS
        and model["checkpoint"] == b["checkpoint"] and model["architecture"] == c["architecture"]
        and model["step"] == b["step"] and model["tag_key"] == CHECKPOINT_KEY
        and model["tag"] == checkpoint_tag(c, root["contract"], b)
        and model["model_state_hash_schema"] == "union_typed_tensor_tree_sha256_v1"
        and model["model_state_sha256"] == cert["saved_model_state_sha256"], "cycle actual model delivery descriptor")
    return {"root": root, "contract": c, "boundary": b, "model": model}


def accepted_delivery(lineage: dict[str, Any], *, resolve_delivery: DeliveryResolver) -> dict[str, Any]:
    """Metadata-only accepted transport, never repeat the owned consumer.

    The actor bridge supplies its genuine resolver for API/source continuity;
    prior ownership is the caller-pinned Root9/ind11 pair, not recursive replay.
    """
    core.require(Path(inspect.getfile(resolve_delivery)).resolve() == core.REPO / "scripts/core_closed_once_actor.py",
                 "foreign accepted cycle bridge callback")
    return trusted_parent(lineage)


def execute(contract_ref: dict[str, str], stage_ref: dict[str, str], *, observer: global_once.PassiveObserver,
            resolve_delivery: DeliveryResolver, validate_payload: PayloadValidator) -> dict[str, Any]:
    _, c = core.read_ref(contract_ref)
    _, stage = core.read_ref(stage_ref)
    validate_contract(c)
    source_guard(c)
    core.require(all(Path(inspect.getfile(helper)).resolve() == core.REPO / "scripts/core_closed_once_actor.py"
                     for helper in (resolve_delivery, validate_payload)), "foreign cycle parent bridge callback")
    manifest, identity, parent = metadata(c, stage, contract_ref, resolve_delivery=resolve_delivery)
    pb = parent["boundary"]
    journals = [] if parent["fresh"] else pb["draw_journals"]
    usage = usage_for(c, journals)
    membership = tuple((s["generation"], s["source_sha256"], s["rows"]) for s in manifest["sources"])
    usage.check_population(membership)
    core.require(usage.batches == parent["student_step"]
        and (parent["fresh"] or "usage_state" not in pb or usage.state() == pb["usage_state"]),
        "cycle parent usage cannot rewind/reset")
    output = core.external(stage["output"])
    core.require(not core.failure_marker(output).exists(), "failed cycle output cannot be reused")
    output.mkdir(parents=False, exist_ok=False)
    trainer = None
    drawn = applied = batches = 0
    try:
        import numpy as np
        import torch
        from scripts.bootstrap_checkpoint_resume import resume_bootstrap
        from scripts.offline_epoch_resume import _restore_rng, _rng_state

        for helper, name in ((resume_bootstrap, "scripts/bootstrap_checkpoint_resume.py"),
                             (_restore_rng, "scripts/offline_epoch_resume.py"), (_rng_state, "scripts/offline_epoch_resume.py")):
            core.require(Path(inspect.getfile(helper)).resolve() == core.REPO / name, "foreign cycle restore helper")
        core.require((torch.__version__, np.__version__) == (c["runtime"]["torch_version"], c["runtime"]["numpy_version"]), "cycle runtime versions")
        torch.set_num_threads(c["runtime"]["torch_threads"])
        backend = core.backend_profile(torch)
        core.require((parent["fresh"] or backend == pb["backend"])
            and (torch.cuda.device_count() if torch.cuda.is_available() else 0) == c["runtime"]["cuda_devices"], "cycle backend changed")
        random.seed(c["seed"])
        np.random.seed(c["seed"])
        torch.manual_seed(c["seed"])
        rng = np.random.default_rng(c["seed"])
        with QuotaLedger(core.external(c["sampling"]["quota_ledger"]), first=parent["fresh"], expected_journals=journals,
            max_file_bytes=c["limits"]["max_file_bytes"], max_total_bytes=c["limits"]["max_total_metadata_bytes"]) as ledger, (output / "draws.jsonl").open("x") as journal:
            expected_ledger = usage.ledger_checkpoint(c["sampling"]["quota_ledger"]) if parent["fresh"] else pb["quota_ledger_checkpoint"]
            core.require(ledger.checkpoint() == expected_ledger, "cycle parent ledger tail mismatch")
            def on_draw(rows: Any) -> None:
                nonlocal drawn, batches
                values = row_dicts(rows)
                usage.validate_rows(values)
                encoded = (core.canonical({"draw_batch": parent["student_step"] + batches, "rows": values}) + "\n").encode()
                core.require(journal.tell() + len(encoded) <= c["limits"]["max_file_bytes"], "cycle local journal cap")
                ledger.reserve(encoded)
                journal.write(encoded.decode())
                journal.flush()
                os.fsync(journal.fileno())
                usage.accept(values)
                drawn += len(values)
                batches += 1
            loaded = fixed_population.load_fixed_population(c["architecture"], manifest, c["population"]["manifest"],
                rng=rng, max_rows=c["limits"]["max_rows"], max_decoded_bytes=c["limits"]["max_decoded_bytes"])
            core.require(loaded.identity == identity, "cycle loaded population changed")
            search_validation = None
            if c["retained_search_share"] > 0:
                search_validation = global_once.validate_retained_search_population(loaded, np)
            replay = StoredRowOnceReplay(loaded, seed=c["sampling"]["order_seed"], batch_size=c["batch_size"],
                expected_rows=c["population"]["rows"], on_draw=on_draw)
            core.import_custody()
            trainer = core.make_trainer(c, output / "tb", retained_search_share=c["retained_search_share"])
            expected_optimizer_recipe = optimizer_recipe(trainer)
            if parent["fresh"]:
                bank.fresh_model(trainer, c, torch)
            else:
                checkpoint = core.external(parent["model"]["checkpoint"]["path"])
                core.require(not checkpoint.stat().st_mode & 0o222 and core.file_sha(checkpoint) == parent["model"]["checkpoint"]["sha256"], "cycle parent checkpoint custody")
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                actor = {"kind": "closed_once_learner", **parent["model"]["checkpoint"], "architecture": c["architecture"],
                    "expected_checkpoint_step": parent["model"]["step"], "predecessor": parent["root"]["closed_binding"], "lineage": c["parent"]}
                validate_payload(actor, saved, torch)
                del saved
                resume_bootstrap(trainer, checkpoint, parent["model"]["checkpoint"]["sha256"], parent["model"]["step"], c["seed"])
            if c["policy_anchor"] is not None:
                trainer.configure_frozen_policy_anchor(c["policy_anchor"])
            core.require(core.canonical(optimizer_recipe(trainer)) == core.canonical(expected_optimizer_recipe),
                         "actual cycle optimizer/scheduler recipe changed during bootstrap")
            if not parent["fresh"]:
                _restore_rng(pb["rng"], rng, cast(Any, np.random))
                core.require(core.canonical(_rng_state(rng, cast(Any, np.random))) == core.canonical(pb["rng"]), "cycle five RNG restoration")
            start = trainer.step
            context = {"trainer": trainer, "contract": c, "contract_ref": contract_ref, "stage": stage, "stage_ref": stage_ref,
                "rng": rng, "np": np, "torch": torch, "population": loaded, "replay": replay, "backend": backend, "parent": parent}
            context["optimizer_recipe"] = expected_optimizer_recipe
            observer.before_train(context)
            def counted(_opt: Any, _args: Any, _kwargs: Any) -> None:
                nonlocal applied
                applied += 1
            hook = trainer.opt.register_step_post_hook(counted)
            try:
                metrics = trainer.train_steps(replay, batch_size=c["batch_size"], steps=c["updates_per_segment"][0])
            finally:
                hook.remove()
            core.require(trainer.step == start + applied and applied == metrics.train_steps_done == c["updates_per_segment"][0]
                and drawn == metrics.train_samples_seen == len(replay) and replay.remaining == 0
                and metrics.grad_nonfinite_skip_rate == metrics.transient_cuda_retry_batches == 0, "cycle actual update/row/taint counters")
            quota_state, ledger_checkpoint = replay.state(), ledger.checkpoint()
            search_record = global_once.retained_search_record(c, asdict(metrics), search_validation)
        journal_path = output / "draws.jsonl"
        journal_path.chmod(0o444)
        draw_journals = [*journals, {"path": str(journal_path), "sha256": core.file_sha(journal_path)}]
        core.require(usage_for(c, draw_journals).state() == usage.state(), "cycle reserved usage/journal mismatch")
        if c["runtime"]["device"].startswith("cuda"):
            torch.cuda.synchronize()
        checkpoint = output / "learner.pt"
        context["checkpoint"] = checkpoint
        observer.before_save(context)
        trainer.save(checkpoint)
        observer.after_save(context)
        boundary = {"status": BOUNDARY, "study_uuid": c["study_uuid"], "run_uuid": c["run_uuid"], "cycle": c["cycle"], "segment": 1,
            "contract_sha256": contract_ref["sha256"], "source_revision": c["source_revision"], "stage": stage_ref,
            "method": c["method"], "parent": c["parent"], "sampling": c["sampling"], "step_start": start, "step": trainer.step,
            "applied_updates": applied, "applied_views": metrics.train_samples_seen, "drawn_rows": drawn,
            "membership": membership, "quota_state": quota_state, "draw_journals": draw_journals,
            "draws_sha256": draw_journals[-1]["sha256"], "quota_ledger_checkpoint": ledger_checkpoint,
            "usage_state": usage.state(), "population_identity": identity, "source_origin_binding": global_once.origin_binding(identity),
            "pass_complete": True, "metrics": asdict(metrics), "rng": _rng_state(rng, cast(Any, np.random)), "backend": backend,
            "retained_search": search_record,
            "policy_anchor": c["policy_anchor"], "policy_anchor_diagnostics": trainer.frozen_policy_anchor_diagnostics(),
            "state_handoff": "model_only_fresh_optimizer" if parent["fresh"] else "adjacent_full_state_continuation",
            "champion_changed": False, "collector_lineage_qualified": False}
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # Keep real inherited full-state payload; remove only its separately
        # authenticated parent-family metadata before binding this new family.
        for tag in PARENT_TAGS:
            saved.pop(tag, None)
        saved[CHECKPOINT_KEY] = checkpoint_tag(c, contract_ref, boundary)
        pending = output / "learner.pt.family.pending"
        with pending.open("xb") as stream:
            torch.save(saved, stream)
            stream.flush()
            os.fsync(stream.fileno())
        del saved
        os.replace(pending, checkpoint)
        checkpoint.chmod(0o444)
        core.fsync_directory(output)
        boundary["checkpoint"] = {"path": str(checkpoint), "sha256": core.file_sha(checkpoint)}
        boundary["own_live_certificate"] = observer.closed_checkpoint(context, boundary)
        trainer.writer.close()
        source_guard(c)
        core.import_custody()
        final_manifest, final_identity, final_parent = metadata(c, stage, contract_ref, resolve_delivery=resolve_delivery)
        core.require((final_manifest, final_identity, final_parent) == (manifest, identity, parent), "cycle metadata changed during training")
        compact_metadata(boundary)
        core.publish_boundary(output, boundary)
        return boundary
    except BaseException as error:
        try:
            marker = core.failure_marker(output)
            core.write_json(marker, {"status": "FAILED_CAPPED_CYCLE_NO_RECOVERY_V1", "contract": contract_ref,
                "stage": stage_ref, "drawn_rows": drawn, "optimizer_calls": applied, "error": repr(error)})
            marker.chmod(0o444)
            core.fsync_directory(marker.parent)
        finally:
            try:
                observer.failed()
            finally:
                try:
                    if trainer is not None:
                        trainer.writer.close()
                finally:
                    core.seal(output)
        raise
