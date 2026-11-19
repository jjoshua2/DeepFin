"""One uniform stored-row pass over an explicitly pinned historical union.

Additive family: original nonconsecutive collection/generation provenance remains
in the immutable source map. No four-bank FIFO projection, fabricated genealogy,
legacy proof relabeling, runtime adoption or launcher is provided here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import hashlib
import os
from pathlib import Path
import random
import re
from typing import Any, Protocol, cast
import uuid

from scripts import core_finite_learner as core
from scripts import core_generation_bank_learner as bank
from scripts import core_generation_bank_stored_row_once as once
from scripts import core_selfplay_union_manifest as union
from chess_anti_engine.replay import fixed_population
from chess_anti_engine.policy_anchor_spec import CHECKPOINT_KEY as REFERENCE_KEY, validate_spec
from chess_anti_engine.replay.stored_row_once import (
    ORDER, SAMPLING, QuotaLedger, StoredRowOnceReplay, digest, permutation_for,
    rows_after_batches, verify_journals,
)

CONTRACT = "core_selfplay_union_policy_anchor_stored_row_once_v1"
STAGE = "core_selfplay_union_policy_anchor_stored_row_once_stage_v1"
BOUNDARY = "COMPLETED_CORE_SELFPLAY_UNION_POLICY_ANCHOR_STORED_ROW_ONCE_SEGMENT_V1"
CHECKPOINT_KEY = "core_selfplay_union_policy_anchor_stored_row_once_v1"
SOURCE_FILES = (*once.SOURCE_FILES, "scripts/core_selfplay_union_stored_row_once.py",
                "scripts/core_selfplay_union_stored_row_once_proof.py", "chess_anti_engine/replay/fixed_population.py",
                "scripts/core_selfplay_union_manifest.py", "scripts/core_selfplay_union_policy_anchor_stored_row_once.py",
                "scripts/core_selfplay_union_policy_anchor_stored_row_once_proof.py")


class PassiveObserver(Protocol):
    """Explicit in-process observation only; no replacement learner or launcher."""

    def before_train(self, context: dict[str, Any]) -> None: ...
    def before_save(self, context: dict[str, Any]) -> None: ...
    def after_save(self, context: dict[str, Any]) -> None: ...
    def closed_checkpoint(self, context: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]: ...
    def failed(self) -> None: ...


def reference(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"invalid union {label} reference")
    path, sha = value["path"], value["sha256"]
    if not isinstance(path, str) or not isinstance(sha, str) or re.fullmatch("[0-9a-f]{64}", sha) is None:
        raise ValueError(f"invalid union {label} fields")
    core.external(path)
    return {"path": path, "sha256": sha}


def _validate_contract_metadata(c: dict[str, Any]) -> None:
    core.require(set(c) == {"schema", "study_uuid", "run_uuid", "method", "source_revision", "source_files",
        "architecture", "initial_model", "initial_model_step", "batch_size", "updates_per_segment", "seed",
        "runtime", "lr", "mirror_prob", "limits", "population", "sampling", "policy_anchor"} and c["schema"] == CONTRACT,
        "invalid fixed-union contract")
    for key in ("study_uuid", "run_uuid"):
        core.require(type(c[key]) is str and str(uuid.UUID(c[key])) == c[key], "invalid union UUID")
    core.require(type(c["source_revision"]) is str and re.fullmatch("[0-9a-f]{40}", c["source_revision"]) is not None,
                 "invalid union source revision")
    core.require(type(c["source_files"]) is dict and set(SOURCE_FILES) <= set(c["source_files"]), "missing union source pins")
    for name, expected in cast(dict[str, Any], c["source_files"]).items():
        path = core.REPO / name
        core.require(path.resolve().is_relative_to(core.REPO) and not path.is_symlink()
                     and type(expected) is str and re.fullmatch("[0-9a-f]{64}", expected), "union source pin shape")
    donor = reference(c["initial_model"], "initial model")
    anchor = validate_spec(c["policy_anchor"], c["architecture"])
    core.require(anchor == c["policy_anchor"], "union anchor canonical identity differs")
    if anchor["coefficient"] > 0:
        core.require(anchor["reference"]["checkpoint"] == donor
            and anchor["reference"]["checkpoint_step"] == c["initial_model_step"],
            "union anchor must pin the selected model-only bootstrap reference")
    core.integer(c["initial_model_step"], "recorded donor step", 0)
    core.integer(c["batch_size"], "union batch size")
    core.integer(c["seed"], "union seed", 0)
    core.require(c["seed"] < 2**32, "union seed exceeds NumPy legacy range")
    core.require(type(c["updates_per_segment"]) is list and c["updates_per_segment"], "missing union finite segments")
    steps = sum(core.integer(v, "union segment updates") for v in c["updates_per_segment"])
    lr = c["lr"]
    core.require(type(lr) is dict and set(lr) == {"peak", "floor", "warmup_steps", "warmup_start", "first_period"},
                 "invalid union LR recipe")
    peak, floor, start = (core.number(lr[k], k) for k in ("peak", "floor", "warmup_start"))
    warm = core.integer(lr["warmup_steps"], "union warmup", 0)
    period = core.integer(lr["first_period"], "union LR period")
    core.require(peak > 0 and peak == floor == start and warm == 0 and period > steps - 1,
                 "union requires explicit flat LR with no in-horizon restart")
    core.require(0 <= core.number(c["mirror_prob"], "union mirror probability") <= 1, "union mirror bound")
    runtime = c["runtime"]
    core.require(type(runtime) is dict and set(runtime) == {"device", "use_amp", "torch_version", "numpy_version",
        "torch_threads", "cuda_devices"} and runtime["device"] in ("cpu", "cuda:0")
        and type(runtime["use_amp"]) is bool and runtime["torch_threads"] == 2
        and runtime["cuda_devices"] == (0 if runtime["device"] == "cpu" else 1), "invalid union runtime")
    limits = c["limits"]
    core.require(type(limits) is dict and set(limits) == {"max_rows", "max_decoded_bytes", "max_file_bytes", "max_total_metadata_bytes"},
                 "invalid union limits")
    limits = cast(dict[str, Any], limits)
    for key, value in limits.items():
        core.integer(value, key)
    core.require(limits["max_file_bytes"] <= 128 << 20, "union metadata file limit exceeds128MiB")
    population = c["population"]
    core.require(type(population) is dict and set(population) == {"manifest", "plan", "origin_map_sha256",
        "membership_sha256", "sources", "rows"}, "invalid union population binding")
    population = cast(dict[str, Any], population)
    reference(population["manifest"], "manifest")
    reference(population["plan"], "plan")
    for key in ("origin_map_sha256", "membership_sha256"):
        core.require(type(population[key]) is str and re.fullmatch("[0-9a-f]{64}", population[key]) is not None, "union population digest")
    core.integer(population["sources"], "union source count")
    n = core.integer(population["rows"], "union rows")
    core.require(n <= limits["max_rows"], "union population exceeds row cap")
    sampling = c["sampling"]
    core.require(type(sampling) is dict and set(sampling) == {"schema", "order_seed", "expected_rows", "presentations_per_row",
        "mirror_shares_quota", "partial_batch", "lineage_uuid", "quota_ledger", "order", "permutation_sha256"},
        "invalid union sampling recipe")
    seed = core.integer(sampling["order_seed"], "union order seed", 0)
    core.require(sampling["schema"] == SAMPLING and sampling["expected_rows"] == n
        and type(sampling["expected_rows"]) is int and sampling["presentations_per_row"] == 1
        and type(sampling["presentations_per_row"]) is int and sampling["mirror_shares_quota"] is True
        and sampling["partial_batch"] == "EMIT_FINAL_REMAINDER_ONCE_NO_PADDING" and sampling["lineage_uuid"] == c["run_uuid"]
        and sampling["order"] == ORDER and sampling["permutation_sha256"] == digest(permutation_for(n, seed)),
        "union permutation/quota/partial recipe mismatch")
    quota_path = sampling["quota_ledger"]
    core.require(type(quota_path) is str, "union quota path type")
    core.external(cast(str, quota_path))
    core.require(steps == (n + c["batch_size"] - 1) // c["batch_size"], "union finite dose mismatch")
    # The caller authenticates this exact method descriptor; its prose/status is
    # not promoted into a source, data or executable admission by this validator.
    _, method = core.read_ref(reference(c["method"], "method"))
    training = method["training"]
    expected_lr = {key: training["learning_rate"][key] for key in ("peak", "floor", "warmup_steps", "warmup_start", "first_period")}
    core.require(method["study_uuid"] == c["study_uuid"] and method["bootstrap"]["model"] == donor
        and method["bootstrap"]["architecture"] == c["architecture"]
        and method["bootstrap"]["donor_recorded_step"] == c["initial_model_step"]
        and method["bootstrap"]["student_start_step"] == 0
        and method["requested_population"]["stored_sources"] == population["sources"]
        and method["requested_population"]["stored_rows"] == n
        and training["batch_size"] == c["batch_size"] and training["closed_segments_updates"] == c["updates_per_segment"]
        and training["expected_optimizer_updates"] == steps and training["last_batch_real_rows"] == (n % c["batch_size"] or c["batch_size"])
        and training["seed"] == c["seed"] and training["order_seed"] == seed
        and training["mirror_probability"] == c["mirror_prob"] and expected_lr == lr and training["runtime"] == runtime
        and core.number(training["policy_anchor_coefficient"], "method anchor coefficient") == anchor["coefficient"]
        and core.canonical(training["policy_anchor"]) == core.canonical(anchor)
        and training["no_priority_diff_focus_skip_or_extra_pass"] is True
        and training["max_rows"] == limits["max_rows"] and training["max_total_decoded_bytes"] == limits["max_decoded_bytes"],
        "union actual contract differs from caller-pinned method")


def validate_contract(c: dict[str, Any]) -> None:
    _validate_contract_metadata(c)
    donor = reference(c["initial_model"], "initial model")
    model = core.external(donor["path"])
    core.require(not model.stat().st_mode & 0o222 and core.file_sha(model) == donor["sha256"], "union initial model custody")


def source_guard(c: dict[str, Any]) -> None:
    bank.source_guard(c)
    for module, name in ((union, "scripts/core_selfplay_union_manifest.py"),
                         (fixed_population, "chess_anti_engine/replay/fixed_population.py")):
        core.require(Path(inspect.getfile(module)).resolve() == core.REPO / name, "foreign union source helper")
    core.require(Path(__file__).resolve() == core.REPO / "scripts/core_selfplay_union_policy_anchor_stored_row_once.py", "foreign union entry")


def origin_binding(identity: dict[str, Any]) -> dict[str, Any]:
    return {"manifest": identity["manifest"], "origin_map_sha256": identity["origin_map_sha256"]}


def _population_metadata(c: dict[str, Any], stage: dict[str, Any], contract_ref: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    core.require(set(stage) == {"schema", "contract_sha256", "segment", "population", "parent", "output"}
        and stage["schema"] == STAGE and stage["contract_sha256"] == contract_ref["sha256"]
        and stage["population"] == c["population"], "invalid union stage/population binding")
    segment = core.integer(stage["segment"], "union segment")
    core.require(segment <= len(c["updates_per_segment"]), "union segment exceeds finite horizon")
    core.require(not core.external(c["sampling"]["quota_ledger"]).is_relative_to(core.external(stage["output"])),
                 "union quota ledger inside sealed output")
    manifest = union.read_union_manifest(c["population"]["manifest"], c["population"]["plan"],
        max_file_bytes=c["limits"]["max_file_bytes"], max_total_bytes=c["limits"]["max_total_metadata_bytes"])
    identity = fixed_population.manifest_identity(manifest, c["population"]["manifest"])
    core.require(identity == {key: value for key, value in c["population"].items() if key != "plan"}
        and manifest["architecture"] == c["architecture"], "union manifest/origin/membership identity mismatch")
    return manifest, identity


def metadata(c: dict[str, Any], stage: dict[str, Any], contract_ref: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    manifest, identity = _population_metadata(c, stage, contract_ref)
    parent = None
    if stage["segment"] == 1:
        core.require(stage["parent"] is None, "first union segment must bootstrap model only")
    else:
        parent = closed_parent(stage["parent"], c, contract_ref, stage)
        core.require(parent["population_identity"] == identity and parent["source_origin_binding"] == origin_binding(identity),
                     "parent union identity/origin changed")
    return manifest, identity, parent


def checkpoint_identity(c: dict[str, Any], contract_ref: dict[str, str], segment: int) -> dict[str, Any]:
    return {"contract_sha256": contract_ref["sha256"], "run_uuid": c["run_uuid"], "study_uuid": c["study_uuid"],
            "source_revision": c["source_revision"], "segment": segment, "sampling": c["sampling"], "method": c["method"],
            "policy_anchor": c["policy_anchor"]}


def _closed_record_metadata(binding: dict[str, Any], c: dict[str, Any], contract_ref: dict[str, str], stage: dict[str, Any]) -> dict[str, Any]:
    core.require(type(binding) is dict and set(binding) == {"boundary", "owner_terminal", "owner_sha256", "command"}, "invalid union parent")
    path, parent = core.read_ref(binding["boundary"])
    _, prior = core.read_ref(parent["stage"])
    output = core.external(prior["output"])
    core.require(path == output / "boundary.json" and not core.failure_marker(output).exists()
        and not (output / "boundary.pending.json").exists(), "failed/incomplete union parent")
    core.require(set(prior) == {"schema", "contract_sha256", "segment", "population", "parent", "output"}
        and prior["schema"] == STAGE and prior["contract_sha256"] == contract_ref["sha256"]
        and prior["segment"] == parent["segment"] == stage["segment"] - 1
        and prior["population"] == stage["population"] == c["population"] and prior["parent"] == parent["parent"],
        "union original parent stage changed")
    suffix = ["--contract", contract_ref["path"],
              "--contract-sha256", contract_ref["sha256"], "--stage", parent["stage"]["path"], "--stage-sha256", parent["stage"]["sha256"]]
    command = binding["command"]
    core.require(type(command) is list and len(command) > len(suffix) and all(type(arg) is str and arg for arg in command)
        and command[-len(suffix):] == suffix
        and command[-len(suffix)-2:-len(suffix)] in (["-m", "scripts.core_selfplay_union_policy_anchor_stored_row_once"],
            ["-m", "scripts.core_selfplay_union_policy_anchor_stored_row_once_proof"]), "union closed parent command")
    core.require(type(binding["owner_sha256"]) is str and re.fullmatch("[0-9a-f]{64}", binding["owner_sha256"]), "union parent owner hash")
    _, terminal = core.read_ref(binding["owner_terminal"], readonly=False)
    core.require(terminal["status"] == "PASS" and type(terminal["returncode"]) is int and terminal["returncode"] == 0
        and terminal["owned_processes_closed"] is True and terminal["custody"] == "OWNED_PROCESSES_CLOSED"
        and terminal["supervisor_sha256"] == binding["owner_sha256"] and terminal["command"] == command, "union parent not closed PASS")
    core.require(parent["status"] == BOUNDARY and parent["contract_sha256"] == contract_ref["sha256"]
        and parent["run_uuid"] == c["run_uuid"] and parent["study_uuid"] == c["study_uuid"]
        and parent["source_revision"] == c["source_revision"] and parent["sampling"] == c["sampling"] and parent["method"] == c["method"]
        and parent["policy_anchor"] == c["policy_anchor"]
        and parent["step"] == sum(c["updates_per_segment"][:stage["segment"] - 1])
        and parent["initial_model"] == c["initial_model"] and parent["initial_model_step"] == c["initial_model_step"], "union parent family mismatch")
    checkpoint = reference(parent["checkpoint"], "closed checkpoint")
    core.require(core.external(checkpoint["path"]) == output / "learner.pt", "union checkpoint output path")
    return parent


def closed_parent(binding: dict[str, Any], c: dict[str, Any], contract_ref: dict[str, str], stage: dict[str, Any]) -> dict[str, Any]:
    parent = _closed_record_metadata(binding, c, contract_ref, stage)
    checkpoint = core.external(parent["checkpoint"]["path"])
    core.require(not checkpoint.stat().st_mode & 0o222 and core.file_sha(checkpoint) == parent["checkpoint"]["sha256"],
                 "union parent checkpoint custody")
    return parent


def validate_closed_segment(contract_ref: dict[str, str], binding: dict[str, Any], *, require_complete: bool = True) -> dict[str, Any]:
    """Caller-trusted family lineage only; not a model delivery or admission.

    Reconstruct declared row/journal identities from pinned metadata. Do not
    reopen any checkpoint, shard, journal payload or build a model here. The
    actual learner/observer and future Root+ind delivery remain separate proofs.
    """
    _, c = core.read_ref(contract_ref)
    _validate_contract_metadata(c)
    _, candidate = core.read_ref(binding["boundary"])
    _, stage = core.read_ref(candidate["stage"])
    manifest, identity = _population_metadata(c, stage, contract_ref)
    closed = _closed_record_metadata(binding, c, contract_ref, dict(stage, segment=stage["segment"] + 1))
    core.require(closed["population_identity"] == identity and closed["source_origin_binding"] == origin_binding(identity),
                 "closed union identity/origin mismatch")
    membership = tuple((s["generation"], s["source_sha256"], s["rows"]) for s in manifest["sources"])
    expected_start = sum(c["updates_per_segment"][:closed["segment"] - 1])
    expected_updates = c["updates_per_segment"][closed["segment"] - 1]
    core.require(core.canonical(closed["membership"]) == core.canonical(membership)
        and type(closed["step_start"]) is int and closed["step_start"] == expected_start
        and type(closed["applied_updates"]) is int and closed["applied_updates"] == expected_updates,
        "closed union membership/selected segment counters differ")
    class Population:
        rng = None
        def __init__(self) -> None:
            self.membership = membership
        def __len__(self) -> int:
            return sum(row[2] for row in membership)
    def forbidden(_rows: Any) -> None:
        raise ValueError("lineage metadata consumer cannot draw")
    replay = StoredRowOnceReplay(Population(), seed=c["sampling"]["order_seed"], batch_size=c["batch_size"],
                                 expected_rows=c["population"]["rows"], on_draw=forbidden)
    state = closed["quota_state"]
    expected = replay.state()
    core.require(type(state) is dict and set(state) == set(expected), "closed union cursor schema")
    for key in set(expected) - {"cursor", "batches", "last_batch_rows", "row_prefix_sha256"}:
        core.require(core.canonical(state[key]) == core.canonical(expected[key]), "closed union permutation/state changed")
    batches = core.integer(state["batches"], "closed union batches", 0)
    cursor = rows_after_batches(len(replay), c["batch_size"], batches)
    core.require(type(state["cursor"]) is int and state["cursor"] == cursor and batches == closed["step"], "closed union cursor/update mismatch")
    last = cursor - rows_after_batches(len(replay), c["batch_size"], batches - 1) if batches else 0
    core.require(type(state["last_batch_rows"]) is int and state["last_batch_rows"] == last, "closed union last batch mismatch")
    # Expected canonical identities/hashes, without reading raw journal files.
    prefix = hashlib.sha256()
    raw_ledger = hashlib.sha256()
    raw_bytes = 0
    journals = closed["draw_journals"]
    core.require(type(journals) is list and len(journals) == closed["segment"], "closed union journal chain length")
    journals = cast(list[dict[str, str]], journals)
    batch_start = 0
    for index, ref in enumerate(journals):
        reference(ref, "closed journal")
        segment_hash = hashlib.sha256()
        batch_end = batch_start + c["updates_per_segment"][index]
        for batch in range(batch_start, batch_end):
            lo = rows_after_batches(len(replay), c["batch_size"], batch)
            hi = rows_after_batches(len(replay), c["batch_size"], batch + 1)
            rows = [asdict(replay.row_at(i)) for i in replay.permutation[lo:hi]]
            for row in rows:
                prefix.update((core.canonical(row) + "\n").encode())
            encoded = (core.canonical({"draw_batch": batch, "rows": rows}) + "\n").encode()
            segment_hash.update(encoded)
            raw_ledger.update(encoded)
            raw_bytes += len(encoded)
        core.require(segment_hash.hexdigest() == ref["sha256"], "closed union declared journal sequence mismatch")
        batch_start = batch_end
    core.require(batch_start == batches and prefix.hexdigest() == state["row_prefix_sha256"], "closed union declared prefix mismatch")
    core.require(closed["draws_sha256"] == journals[-1]["sha256"]
        and journals[-1]["path"] == str(Path(stage["output"]) / "draws.jsonl"), "closed union current journal hash/path aliases differ")
    core.require(closed["quota_ledger_checkpoint"] == {"path": c["sampling"]["quota_ledger"], "bytes": raw_bytes,
        "sha256": raw_ledger.hexdigest()}, "closed union declared ledger mismatch")
    expected_views = cursor - rows_after_batches(len(replay), c["batch_size"], closed["step_start"])
    core.require(closed["drawn_rows"] == closed["applied_views"] == closed["metrics"]["train_samples_seen"] == expected_views
        and closed["applied_updates"] == closed["metrics"]["train_steps_done"]
        and closed["metrics"]["transient_cuda_retry_batches"] == 0 and closed["metrics"]["grad_nonfinite_skip_rate"] == 0,
        "closed union actual counters differ")
    complete = cursor == len(replay) and closed["segment"] == len(c["updates_per_segment"])
    core.require(closed["pass_complete"] is complete and (complete or not require_complete), "closed union pass incomplete")
    return {"family": CONTRACT, "contract": contract_ref, "checkpoint": closed["checkpoint"], "boundary": binding["boundary"],
        "run_uuid": c["run_uuid"], "study_uuid": c["study_uuid"], "method": c["method"],
        "population_identity": identity, "source_origin_binding": origin_binding(identity), "sampling": c["sampling"],
        "policy_anchor": c["policy_anchor"], "policy_anchor_diagnostics": closed["policy_anchor_diagnostics"],
        "steps": batches, "actual_views": cursor, "pass_complete": complete,
        "scope": "caller-trusted closed lineage metadata only; actual own-live model delivery/Root+ind proof required separately"}


def validate_closed_model_delivery(contract_ref: dict[str, str], binding: dict[str, Any], *, require_complete: bool = True) -> dict[str, Any]:
    """Metadata API for a future adapter; caller must authenticate Root/ind proof.

    Unlike lineage-only validation, this requires the dedicated own-live wrapper
    certificate, without decoding model or journal payloads in the consumer.
    """
    lineage = validate_closed_segment(contract_ref, binding, require_complete=require_complete)
    core.require(binding["command"][-10:-8] == ["-m", "scripts.core_selfplay_union_policy_anchor_stored_row_once_proof"],
                 "union model delivery requires the dedicated observed entry command")
    _, boundary = core.read_ref(binding["boundary"])
    _, c = core.read_ref(contract_ref)
    observed = boundary["own_live_certificate"]
    core.require(observed["schema"] == "core_selfplay_union_policy_anchor_own_live_saved_payload_proof_v1"
        and observed["status"] == "UNION_POLICY_ANCHOR_OWN_LIVE_SAVED_PAYLOAD_OBSERVED_PENDING_ROOT_CLOSURE_V1"
        and observed["contract"] == contract_ref and observed["stage"] == boundary["stage"]
        and observed["checkpoint"] == lineage["checkpoint"] and observed["population_identity"] == lineage["population_identity"]
        and observed["source_origin_binding"] == lineage["source_origin_binding"] and observed["sampling"] == c["sampling"]
        and observed["step"] == lineage["steps"] and observed["step_start"] == boundary["step_start"]
        and observed["actual_views"] == boundary["applied_views"] and observed["cumulative_views"] == lineage["actual_views"]
        and observed["applied_updates"] == boundary["applied_updates"]
        and observed["quota_prefix_sha256"] == boundary["quota_state"]["row_prefix_sha256"]
        and observed["journal_chain"] == boundary["draw_journals"]
        and observed["ledger_checkpoint"] == boundary["quota_ledger_checkpoint"], "union own-live certificate lineage mismatch")
    count = c["updates_per_segment"][boundary["segment"] - 1]
    core.require(observed["actual_calls"] == {"forward": count, "optimizer_pre": count, "optimizer_post": count, "save": 1}
        and observed["save_before_equals_after"] is True and observed["extra_forwards"] == 0
        and observed["additional_final_CPU_checkpoint_decodes"] == 1
        and observed["five_rng_sha256"] == core.digest(boundary["rng"])
        and observed["initial"]["rng"] != {} and observed["initial"]["backend"] == boundary["backend"], "union observed passivity/counter mismatch")
    anchor = observed["policy_anchor"]
    core.require(anchor["identity"] == c["policy_anchor"] == boundary["policy_anchor"]
        and anchor["diagnostics"] == boundary["policy_anchor_diagnostics"]
        and anchor["teacher_unchanged"] is True and anchor["no_optimizer_ownership"] is True
        and anchor["no_reference_gradients"] is True and anchor["all_reference_modules_eval"] is True,
        "union anchor observed identity/frozen custody differs")
    if c["policy_anchor"]["coefficient"] == 0:
        core.require(anchor["initial_reference"] is None and anchor["final_reference_sha256"] is None
            and anchor["actual_reference_forwards"] == 0 and anchor["diagnostics"] is None,
            "zero union anchor did reference work")
    else:
        first, diagnostics = anchor["initial_reference"], anchor["diagnostics"]
        core.require(core.digest(first) == anchor["final_reference_sha256"] and first["checkpoint"] == c["policy_anchor"]["reference"]["checkpoint"]
            and first["model"]["finite"] is True and first["model"]["state_entries"] == len(first["model"]["key_shapes"])
            and first["model"]["key_shape_dtype_sha256"] == core.digest(first["model"]["key_shapes"])
            and core.canonical(first["model"]["architecture"]) == core.canonical(c["architecture"])
            and first["model"]["input_planes"] == observed["saved_model"]["input_planes"]
            and first["model"]["unique_parameter_numel"] > 0
            and type(anchor["actual_reference_forwards"]) is int
            and 0 <= anchor["actual_reference_forwards"] <= count
            and diagnostics["reference_forwards"] == anchor["actual_reference_forwards"]
            and diagnostics["total_rows"] == boundary["applied_views"]
            and 0 <= diagnostics["eligible_rows"] <= diagnostics["total_rows"]
            and core.number(diagnostics["kl_row_sum"], "observed trajectory KL") >= -1e-5 * max(1, diagnostics["eligible_rows"])
            and core.number(diagnostics["weighted_row_sum"], "observed weighted trajectory KL") == c["policy_anchor"]["coefficient"] * diagnostics["kl_row_sum"]
            and diagnostics["mean_kl"] == (diagnostics["kl_row_sum"] / diagnostics["eligible_rows"] if diagnostics["eligible_rows"] else None),
            "union anchor actual teacher/trajectory evidence differs")
    facts = observed["saved_model"]
    core.require(facts["architecture"] == c["architecture"] and facts["state_entries"] == len(facts["key_shapes"])
        and facts["key_shape_dtype_sha256"] == core.digest(facts["key_shapes"])
        and facts["key_shape_dtype_sha256"] == observed["initial"]["model"]["key_shape_dtype_sha256"]
        and facts["model_state_sha256"] == observed["saved_model_state_sha256"] and facts["finite"] is True
        and facts["unique_parameter_numel"] > 0 and facts["input_planes"] > 0, "union observed model metadata mismatch")
    return {**lineage, "own_live_certificate": observed,
        "scope": "caller-trusted closed union model metadata; independent Root delivery/source custody and evaluation admission remain required"}


def execute(contract_ref: dict[str, str], stage_ref: dict[str, str], *, observer: PassiveObserver | None = None) -> dict[str, Any]:
    _, c = core.read_ref(contract_ref)
    _, stage = core.read_ref(stage_ref)
    validate_contract(c)
    source_guard(c)
    manifest, identity, parent = metadata(c, stage, contract_ref)
    output = core.external(stage["output"])
    core.require(not core.failure_marker(output).exists(), "failed bank segment output cannot be reused")
    output.mkdir(parents=False, exist_ok=False)
    trainer = None
    drawn = applied = 0
    draw_batches = 0
    prior_journals: list[dict[str, str]] = []
    try:
        import numpy as np
        import torch
        from scripts.bootstrap_checkpoint_resume import resume_bootstrap
        from scripts.offline_epoch_resume import _restore_rng, _rng_state

        for helper, name in ((resume_bootstrap, "scripts/bootstrap_checkpoint_resume.py"), (_rng_state, "scripts/offline_epoch_resume.py")):
            core.require(Path(inspect.getfile(helper)).resolve() == core.REPO / name, "foreign bank state helper")
        core.require((torch.__version__, np.__version__) == (c["runtime"]["torch_version"], c["runtime"]["numpy_version"]), "bank runtime versions differ")
        torch.set_num_threads(c["runtime"]["torch_threads"])
        backend = core.backend_profile(torch)
        core.require((torch.cuda.device_count() if torch.cuda.is_available() else 0) == c["runtime"]["cuda_devices"], "bank CUDA visibility differs")
        if parent is not None:
            core.require(parent["backend"] == backend, "bank continuation backend changed")
        random.seed(c["seed"])
        np.random.seed(c["seed"])
        torch.manual_seed(c["seed"])
        rng = np.random.default_rng(c["seed"])
        prior_journals = [] if parent is None else parent["draw_journals"]
        with QuotaLedger(core.external(c["sampling"]["quota_ledger"]), first=parent is None,
                         expected_journals=prior_journals, max_file_bytes=c["limits"]["max_file_bytes"],
                         max_total_bytes=c["limits"]["max_total_metadata_bytes"]) as ledger, (output / "draws.jsonl").open("x") as journal:
            if parent is not None:
                core.require(ledger.checkpoint() == parent["quota_ledger_checkpoint"], "quota ledger/checkpoint mismatch")
            def on_draw(rows: Any) -> None:
                nonlocal drawn, draw_batches
                encoded = core.canonical({"draw_batch": sum(c["updates_per_segment"][:stage["segment"] - 1]) + draw_batches,
                                          "rows": [asdict(row) for row in rows]}) + "\n"
                core.require(journal.tell() + len(encoded.encode()) <= c["limits"]["max_file_bytes"], "local draw journal byte cap")
                ledger.reserve(encoded.encode())
                journal.write(encoded)
                journal.flush()
                os.fsync(journal.fileno())
                drawn += len(rows)
                draw_batches += 1

            # Complete actual payload admission/decode before any model construction.
            loaded = fixed_population.load_fixed_population(c["architecture"], manifest, c["population"]["manifest"],
                rng=rng, max_rows=c["limits"]["max_rows"], max_decoded_bytes=c["limits"]["max_decoded_bytes"])
            core.require(loaded.identity == identity, "actual loaded union identity changed")
            replay = StoredRowOnceReplay(loaded, seed=c["sampling"]["order_seed"], batch_size=c["batch_size"],
                                         expected_rows=c["sampling"]["expected_rows"], on_draw=on_draw)
            if parent is not None:
                prior_journals = parent["draw_journals"]
                replay.restore(parent["quota_state"], prior_journals,
                               max_file_bytes=c["limits"]["max_file_bytes"],
                               max_total_bytes=c["limits"]["max_total_metadata_bytes"])
                core.require(parent["quota_state"]["cursor"] == rows_after_batches(len(replay), c["batch_size"], parent["step"]),
                             "parent checkpoint/journal cumulative dose differs")
            prior_steps = 0 if parent is None else parent["step"]
            core.require(prior_steps + c["updates_per_segment"][stage["segment"] - 1] <= replay.total_batches,
                         "stored-row segment overdraw before model construction")
            expected_views = (rows_after_batches(len(replay), c["batch_size"], prior_steps + c["updates_per_segment"][stage["segment"] - 1])
                              - rows_after_batches(len(replay), c["batch_size"], prior_steps))
            core.import_custody()
            trainer = core.make_trainer(c, output / "tb")
            if parent is None:
                bank.fresh_model(trainer, c, torch)
            else:
                checkpoint = core.external(parent["checkpoint"]["path"])
                core.require(core.file_sha(checkpoint) == parent["checkpoint"]["sha256"], "bank parent checkpoint changed")
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                core.require(state.get(CHECKPOINT_KEY) == {"identity": checkpoint_identity(c, contract_ref, stage["segment"] - 1),
                                                             "quota_state": parent["quota_state"], "draw_journals": prior_journals,
                                                             "quota_ledger_checkpoint": parent["quota_ledger_checkpoint"],
                                                             "population_identity": identity, "source_origin_binding": origin_binding(identity)}, "bank checkpoint family tag mismatch")
                core.require("core_selfplay_union_stored_row_once_v1" not in state and bank.CHECKPOINT_KEY not in state and bank.ANCHOR_CHECKPOINT_KEY not in state and once.CHECKPOINT_KEY not in state,
                             "legacy/once checkpoint cannot masquerade as union family")
                core.require(state.get(REFERENCE_KEY) == c["policy_anchor"], "saved union anchor reference identity differs")
                del state
                core.require(core.file_sha(checkpoint) == parent["checkpoint"]["sha256"], "bank parent checkpoint changed during tag read")
                resume_bootstrap(trainer, checkpoint, parent["checkpoint"]["sha256"], parent["step"], c["seed"])
            # Reconstruct the same frozen reference only after student bootstrap/full
            # state restore. Its constructor preserves RNG; restore all five parent
            # streams after both setup operations, before any observation/draw.
            trainer.configure_frozen_policy_anchor(c["policy_anchor"])
            if parent is not None:
                _restore_rng(parent["rng"], rng, cast(Any, np.random))
                core.require(core.canonical(_rng_state(rng, cast(Any, np.random))) == core.canonical(parent["rng"]), "union RNG restoration differs")
            core.import_custody()
            start = trainer.step
            count = c["updates_per_segment"][stage["segment"] - 1]
            context = {"trainer": trainer, "replay": replay, "population": loaded, "contract": c, "stage": stage,
                       "contract_ref": contract_ref, "stage_ref": stage_ref, "backend": backend, "rng": rng,
                       "np": np, "torch": torch, "checkpoint": output / "learner.pt"}
            if observer is not None:
                observer.before_train(context)

            def counted(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
                nonlocal applied
                applied += 1

            hook = trainer.opt.register_step_post_hook(counted)
            try:
                metrics = trainer.train_steps(replay, batch_size=c["batch_size"], steps=count)
            finally:
                hook.remove()
            core.require(trainer.step == start + count and applied == count == metrics.train_steps_done, "bank applied update count mismatch")
            core.require(metrics.grad_nonfinite_skip_rate == 0 and metrics.transient_cuda_retry_batches == 0, "bank skips/retries refused")
            core.require(drawn == expected_views == metrics.train_samples_seen, "bank draw/applied view mismatch")
            quota_ledger_checkpoint = ledger.checkpoint()
        journal_path = output / "draws.jsonl"
        journal_path.chmod(0o444)
        draw_journals = [*prior_journals, {"path": str(journal_path), "sha256": core.file_sha(journal_path)}]
        quota_state = replay.state()
        observed = verify_journals(replay, draw_journals, max_file_bytes=c["limits"]["max_file_bytes"],
                                   max_total_bytes=c["limits"]["max_total_metadata_bytes"])
        core.require(observed == {key: quota_state[key] for key in ("cursor", "batches", "row_prefix_sha256")},
                     "current checkpoint/journal cursor differs")
        core.require(quota_state["cursor"] == rows_after_batches(len(replay), c["batch_size"], trainer.step), "cumulative optimizer/quota dose mismatch")
        if stage["segment"] == len(c["updates_per_segment"]):
            core.require(replay.remaining == 0, "completed pass must exhaust every stored row once")
        if c["runtime"]["device"].startswith("cuda"):
            torch.cuda.synchronize()
        checkpoint = output / "learner.pt"
        if observer is not None:
            observer.before_save(context)
        trainer.save(checkpoint)
        if observer is not None:
            observer.after_save(context)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        tag_key = CHECKPOINT_KEY
        core.require(tag_key not in state, "bank checkpoint already tagged")
        state[tag_key] = {"identity": checkpoint_identity(c, contract_ref, stage["segment"]),
                          "quota_state": quota_state, "draw_journals": draw_journals,
                          "quota_ledger_checkpoint": quota_ledger_checkpoint,
                          "population_identity": identity, "source_origin_binding": origin_binding(identity)}
        pending = checkpoint.with_name("learner.pt.family.pending")
        with pending.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, checkpoint)
        checkpoint.chmod(0o444)
        del state
        trainer.writer.close()
        source_guard(c)
        core.import_custody()
        final_manifest, final_identity, final_parent = metadata(c, stage, contract_ref)
        core.require(final_manifest == manifest and final_identity == identity and final_parent == parent,
                     "accepted bank/parent metadata changed during segment")
        boundary = {"status": BOUNDARY, "run_uuid": c["run_uuid"], "source_revision": c["source_revision"],
            "contract_sha256": contract_ref["sha256"], "stage": stage_ref, "parent": stage["parent"],
            "segment": stage["segment"], "study_uuid": c["study_uuid"], "sampling": c["sampling"], "method": c["method"],
            "quota_state": quota_state, "draw_journals": draw_journals,
            "quota_ledger_checkpoint": quota_ledger_checkpoint, "pass_complete": replay.remaining == 0,
            "population_identity": identity, "source_origin_binding": origin_binding(identity), "initial_model_policy": "model_only_fresh_optimizer",
            "state_handoff": "model_only_bootstrap" if parent is None else "adjacent_full_state_continuation", "initial_model": c["initial_model"], "initial_model_step": c["initial_model_step"],
            "step_start": start, "step": trainer.step, "applied_updates": applied, "applied_views": metrics.train_samples_seen,
            "checkpoint": {"path": str(checkpoint), "sha256": core.file_sha(checkpoint)},
            "drawn_rows": drawn, "draws_sha256": core.file_sha(output / "draws.jsonl"), "membership": replay.membership,
            "metrics": asdict(metrics), "rng": _rng_state(rng, cast(Any, np.random)), "backend": backend,
            "recovery_scope": "same-family closed segment boundary only; interrupted output is terminal/unqualified",
            "champion_changed": False, "collector_lineage_qualified": False,
            "policy_anchor": trainer.frozen_policy_anchor_state(),
            "policy_anchor_diagnostics": trainer.frozen_policy_anchor_diagnostics()}
        core.require(boundary["policy_anchor"] == c["policy_anchor"], "actual union anchor identity changed")
        if observer is not None:
            boundary["own_live_certificate"] = observer.closed_checkpoint(context, boundary)
        # Canonical compact JSON keeps the actual 405013-row permutation bounded;
        # publication seals the checkpoint, journal and certificate together.
        core.require(len((core.canonical(boundary) + "\n").encode()) <= 4 << 20, "union closed boundary exceeds4MiB")
        core.publish_boundary(output, boundary)
        return boundary
    except BaseException as error:
        failure = {"status": "FAILED_SELFPLAY_UNION_POLICY_ANCHOR_STORED_ROW_ONCE_SEGMENT_NO_RECOVERY_V1", "error": repr(error),
                   "contract": contract_ref, "stage": stage_ref, "drawn_rows": drawn, "completed_optimizer_calls": applied}
        try:
            marker = core.failure_marker(output)
            core.write_json(marker, failure)
            marker.chmod(0o444)
            core.fsync_directory(marker.parent)
        finally:
            try:
                if observer is not None:
                    observer.failed()
            finally:
                if trainer is not None:
                    trainer.writer.close()
                core.seal(output)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--stage-sha256", required=True)
    args = parser.parse_args(argv)
    result = execute({"path": args.contract, "sha256": args.contract_sha256}, {"path": args.stage, "sha256": args.stage_sha256})
    print(core.canonical({"status": result["status"], "checkpoint": result["checkpoint"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
