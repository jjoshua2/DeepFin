"""Opt-in approximate bounded-shuffle pass over a pinned self-play union.

Additive family: original nonconsecutive collection/generation provenance remains
in the immutable source map. No four-bank FIFO projection, fabricated genealogy,
legacy proof relabeling, runtime adoption or launcher is provided here.
"""
from __future__ import annotations

import argparse
import bisect
from dataclasses import asdict
import inspect
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import re
from typing import Any, Protocol, cast
import uuid

from scripts import core_finite_learner as core
from scripts import core_generation_bank_learner as bank
from scripts import core_generation_bank_stored_row_once as once
from scripts import core_selfplay_union_stored_row_once as global_union
from scripts import core_selfplay_union_manifest as union
from chess_anti_engine.replay import fixed_population, bounded_shuffle, bounded_shuffle_source
from chess_anti_engine.replay.bounded_shuffle import BoundedShuffleReplay, ReplayLimits, ORDER, audit_selection
from chess_anti_engine.replay.bounded_shuffle_source import BoundedReservationWriter, from_fixed_manifest
from chess_anti_engine.replay.stored_row_once import (
    QuotaLedger, rows_after_batches,
)

SAMPLING = "BOUNDED_SHUFFLE_ONE_PASS_V1"
CONTRACT = "core_selfplay_union_bounded_shuffle_v1"
STAGE = "core_selfplay_union_bounded_shuffle_stage_v1"
BOUNDARY = "COMPLETED_CORE_SELFPLAY_UNION_BOUNDED_SHUFFLE_SEGMENT_V1"
CHECKPOINT_KEY = "core_selfplay_union_bounded_shuffle_v1"
SOURCE_FILES = (*once.SOURCE_FILES, "chess_anti_engine/replay/fixed_population.py",
                "scripts/core_selfplay_union_manifest.py", "scripts/core_selfplay_union_bounded_shuffle.py",
                "scripts/core_selfplay_union_bounded_shuffle_proof.py",
                "scripts/core_selfplay_union_stored_row_once.py", "scripts/core_selfplay_union_stored_row_once_proof.py",
                "chess_anti_engine/replay/bounded_shuffle.py", "chess_anti_engine/replay/bounded_shuffle_source.py")


class TrainerReplay(BoundedShuffleReplay):
    """Optimizer-success acknowledgement separated from final tensor release."""

    borrowed_row_batches = True

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._successful_use = False
        total = 0
        self._source_ends: list[int] = []
        for _, _, count in self.membership:
            total += count
            self._source_ends.append(total)

    def row_at(self, index: int):
        from chess_anti_engine.replay.generation import DrawnRow
        core.require(type(index) is int and 0 <= index < self.total_rows, "bounded physical row index")
        slot = bisect.bisect_right(self._source_ends, index)
        start = self._source_ends[slot - 1] if slot else 0
        generation, sha, _ = self.membership[slot]
        return DrawnRow(generation, sha, index - start)

    def mark_batch_completed(self) -> None:
        core.require(not self._successful_use and self._inflight is not None and not self._tainted,
                     "bounded optimizer acknowledgement differs")
        self._successful_use = True

    def release_completed_batch(self) -> None:
        if self._successful_use:
            self.complete_batch()
            self._successful_use = False

    def abort_consumption(self) -> None:
        self._successful_use = False
        super().abort_consumption()


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
        "runtime", "lr", "mirror_prob", "limits", "population", "sampling"} and c["schema"] == CONTRACT,
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
        "mirror_shares_quota", "partial_batch", "lineage_uuid", "quota_ledger", "order", "limits"},
        "invalid union sampling recipe")
    seed = core.integer(sampling["order_seed"], "union order seed", 0)
    core.require(sampling["schema"] == SAMPLING and sampling["expected_rows"] == n
        and type(sampling["expected_rows"]) is int and sampling["presentations_per_row"] == 1
        and type(sampling["presentations_per_row"]) is int and sampling["mirror_shares_quota"] is True
        and sampling["partial_batch"] == "EMIT_FINAL_REMAINDER_ONCE_NO_PADDING" and sampling["lineage_uuid"] == c["run_uuid"]
        and sampling["order"] == ORDER,
        "bounded shuffle/quota/partial recipe mismatch")
    replay_limits = ReplayLimits(**cast(Any, sampling["limits"]))
    core.require(replay_limits.total_bytes <= limits["max_decoded_bytes"] and replay_limits.buffer_rows >= c["batch_size"],
                 "bounded replay limits exceed selected working-byte/row cap")
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
        and training["sampling_order"] == ORDER and training["replay_limits"] == sampling["limits"]
        and training["mirror_probability"] == c["mirror_prob"] and expected_lr == lr and training["runtime"] == runtime
        and training["policy_anchor_coefficient"] == 0 and training["no_priority_diff_focus_skip_or_extra_pass"] is True
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
    for module, name in ((bounded_shuffle, "chess_anti_engine/replay/bounded_shuffle.py"),
                         (bounded_shuffle_source, "chess_anti_engine/replay/bounded_shuffle_source.py")):
        core.require(Path(inspect.getfile(module)).resolve() == core.REPO / name, "foreign bounded replay source")
    core.require(Path(__file__).resolve() == core.REPO / "scripts/core_selfplay_union_bounded_shuffle.py", "foreign bounded entry")


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
        parent = _closed_record_metadata(stage["parent"], c, contract_ref, stage)
        core.require(parent["population_identity"] == identity and parent["source_origin_binding"] == origin_binding(identity),
                     "parent union identity/origin changed")
    return manifest, identity, parent


def checkpoint_identity(c: dict[str, Any], contract_ref: dict[str, str], segment: int) -> dict[str, Any]:
    return {"contract_sha256": contract_ref["sha256"], "run_uuid": c["run_uuid"], "study_uuid": c["study_uuid"],
            "source_revision": c["source_revision"], "segment": segment, "sampling": c["sampling"], "method": c["method"]}


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
        and command[-len(suffix)-2:-len(suffix)] in (["-m", "scripts.core_selfplay_union_bounded_shuffle"],
            ["-m", "scripts.core_selfplay_union_bounded_shuffle_proof"]), "union closed parent command")
    core.require(type(binding["owner_sha256"]) is str and re.fullmatch("[0-9a-f]{64}", binding["owner_sha256"]), "union parent owner hash")
    _, terminal = core.read_ref(binding["owner_terminal"], readonly=False)
    core.require(terminal["status"] == "PASS" and type(terminal["returncode"]) is int and terminal["returncode"] == 0
        and terminal["owned_processes_closed"] is True and terminal["custody"] == "OWNED_PROCESSES_CLOSED"
        and terminal["supervisor_sha256"] == binding["owner_sha256"] and terminal["command"] == command, "union parent not closed PASS")
    core.require(parent["status"] == BOUNDARY and parent["contract_sha256"] == contract_ref["sha256"]
        and parent["run_uuid"] == c["run_uuid"] and parent["study_uuid"] == c["study_uuid"]
        and parent["source_revision"] == c["source_revision"] and parent["sampling"] == c["sampling"] and parent["method"] == c["method"]
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


def expected_records(c: dict[str, Any], identity: dict[str, Any], membership: tuple[Any, ...], batches: int):
    """Symbolic finite order, independent of queue scheduling and payload NNZ."""
    n, width = c["population"]["rows"], c["batch_size"]
    capacity = c["sampling"]["limits"]["buffer_rows"]
    private = random.Random(c["sampling"]["order_seed"])
    slots: list[int] = []
    upstream = emitted = 0
    prefix = "0" * 64

    ends: list[int] = []
    running = 0
    for _, _, count in membership:
        running += count
        ends.append(running)

    def physical(index: int) -> dict[str, Any]:
        slot = bisect.bisect_right(ends, index)
        generation, sha, _ = membership[slot]
        begin = ends[slot - 1] if slot else 0
        return {"generation": generation, "source_sha256": sha, "row_index": index - begin}

    for batch in range(batches):
        while len(slots) < capacity and upstream < n:
            slots.append(upstream)
            upstream += 1
        rows = []
        for _ in range(min(width, n - emitted)):
            chosen = private.randrange(len(slots))
            index = slots.pop(chosen)
            rows.append(physical(index))
            if upstream < n:
                slots.insert(chosen, upstream)
                upstream += 1
        emitted += len(rows)
        record = {"schema": bounded_shuffle.DRAW_SCHEMA, "order": ORDER, "lineage": c["run_uuid"],
                  "identity_sha256": core.digest(identity), "batch": batch + 1, "emitted": emitted,
                  "previous": prefix, "rows": rows}
        prefix = hashlib.sha256((core.canonical(record) + "\n").encode()).hexdigest()
        yield record


def journal_identity(c: dict[str, Any], identity: dict[str, Any], membership: tuple[Any, ...],
                     state: dict[str, Any], journals: list[dict[str, str]], segment: int, *, read_payload: bool) -> dict[str, Any]:
    """Metadata reconstructs exact hashes; actual resume additionally reads bytes.

    All source/row order depends only on pinned membership, fixed slots and seed.
    A parsed/forged state or a tainted ledger is never trusted as a new prefix.
    """
    state_keys = {"schema", "order", "identity", "membership", "lineage", "seed", "batch_size", "limits",
                  "upstream", "slots", "ready", "emitted", "batches", "spent_hex", "reservation_prefix",
                  "shuffle_state", "mirror_state", "sha256"}
    core.require(type(state) is dict and set(state) == state_keys and len(journals) == segment
                 and type(state["upstream"]) is int and 0 <= state["upstream"] <= c["population"]["rows"]
                 and type(state["seed"]) is int and type(state["batch_size"]) is int
                 and state["schema"] == bounded_shuffle.SCHEMA
                 and state["order"] == ORDER and core.canonical(state["identity"]) == core.canonical(identity)
                 and core.canonical(state["membership"]) == core.canonical([list(s) for s in membership])
                 and state["lineage"] == c["run_uuid"] and state["seed"] == c["sampling"]["order_seed"]
                 and state["batch_size"] == c["batch_size"]
                 and core.canonical(state["limits"]) == core.canonical(c["sampling"]["limits"]),
                 "bounded journal/state family differs")
    mirror = state["mirror_state"]
    core.require(type(mirror) is dict and set(mirror) == {"bit_generator", "state", "has_uint32", "uinteger"}
                 and mirror["bit_generator"] == "PCG64" and type(mirror["state"]) is dict
                 and set(mirror["state"]) == {"state", "inc"}
                 and all(type(v) is int and 0 <= v < 2**128 for v in mirror["state"].values())
                 and cast(int, mirror["state"]["inc"]) % 2 == 1
                 and type(mirror["has_uint32"]) is int and mirror["has_uint32"] in (0, 1)
                 and type(mirror["uinteger"]) is int and 0 <= mirror["uinteger"] < 2**32,
                 "bounded closed mirror RNG shape differs")
    batches = sum(c["updates_per_segment"][:segment])
    core.require(type(state["batches"]) is int and state["batches"] == batches
                 and type(state["emitted"]) is int
                 and state["emitted"] == rows_after_batches(c["population"]["rows"], c["batch_size"], batches),
                 "bounded checkpoint/journal counts differ")
    audit = audit_selection(state, expected_records(c, identity, membership, batches))
    records = iter(expected_records(c, identity, membership, batches))
    all_hash, used = hashlib.sha256(), 0
    for count, reference_value in zip(c["updates_per_segment"][:segment], journals, strict=True):
        reference(reference_value, "bounded journal")
        expected_hash = hashlib.sha256()
        before: os.stat_result | None = None
        stream: Any = None
        path: Path | None = None
        if read_payload:
            path = core.external(reference_value["path"])
            before = path.lstat()
            core.require(stat.S_ISREG(before.st_mode) and not before.st_mode & 0o222
                         and before.st_size <= c["limits"]["max_file_bytes"], "bounded journal custody/cap")
            used += before.st_size  # reserve complete size BEFORE any read
            core.require(used <= c["limits"]["max_total_metadata_bytes"], "bounded journal aggregate cap")
            stream = os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb", buffering=0)
        actual_bytes = 0
        try:
            if stream is not None:
                assert before is not None
                core.require(_stamp(os.fstat(stream.fileno())) == _stamp(before), "bounded journal opened identity")
            for _ in range(count):
                encoded = (core.canonical(next(records)) + "\n").encode()
                core.require(len(encoded) <= 1 << 20, "bounded reservation record cap")
                if stream is not None:
                    core.require(before is not None and actual_bytes + len(encoded) <= before.st_size, "truncated bounded journal")
                    raw = bytearray()
                    while len(raw) < len(encoded):
                        block = stream.read(len(encoded) - len(raw))
                        core.require(bool(block), "truncated bounded journal")
                        raw.extend(block)
                    core.require(bytes(raw) == encoded, "bounded actual journal order/record differs")
                actual_bytes += len(encoded)
                expected_hash.update(encoded)
                all_hash.update(encoded)
            core.require(actual_bytes <= c["limits"]["max_file_bytes"]
                         and expected_hash.hexdigest() == reference_value["sha256"], "bounded declared journal hash/cap differs")
            if stream is not None:
                assert before is not None
                assert path is not None
                core.require(actual_bytes == before.st_size and _stamp(os.fstat(stream.fileno())) == _stamp(before)
                             and _stamp(path.lstat()) == _stamp(before), "bounded journal changed or extra tail")
            else:
                used += actual_bytes
            core.require(used <= c["limits"]["max_total_metadata_bytes"], "bounded journal aggregate cap")
        finally:
            if stream is not None:
                stream.close()
    return {"audit": audit, "ledger": {"path": c["sampling"]["quota_ledger"], "bytes": used, "sha256": all_hash.hexdigest()}}


def _stamp(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode)


def validate_closed_segment(contract_ref: dict[str, str], binding: dict[str, Any], *, require_complete: bool = True) -> dict[str, Any]:
    """Caller-trusted METADATA only; no raw model/shard/journal qualification."""
    _, c = core.read_ref(contract_ref)
    _validate_contract_metadata(c)
    _, candidate = core.read_ref(binding["boundary"])
    _, stage = core.read_ref(candidate["stage"])
    manifest, identity = _population_metadata(c, stage, contract_ref)
    closed = _closed_record_metadata(binding, c, contract_ref, dict(stage, segment=stage["segment"] + 1))
    membership = tuple((s["generation"], s["source_sha256"], s["rows"]) for s in manifest["sources"])
    expected_start = sum(c["updates_per_segment"][:closed["segment"] - 1])
    count = c["updates_per_segment"][closed["segment"] - 1]
    core.require(closed["population_identity"] == identity and closed["source_origin_binding"] == origin_binding(identity)
                 and core.canonical(closed["membership"]) == core.canonical(membership)
                 and type(closed["step_start"]) is int and closed["step_start"] == expected_start
                 and type(closed["applied_updates"]) is int and closed["applied_updates"] == count,
                 "closed bounded population/segment differs")
    checked = journal_identity(c, identity, membership, closed["quota_state"], closed["draw_journals"], closed["segment"], read_payload=False)
    core.require(checked["ledger"] == closed["quota_ledger_checkpoint"] and checked["audit"] == closed["selection_audit"]
                 and closed["draws_sha256"] == closed["draw_journals"][-1]["sha256"]
                 and closed["draw_journals"][-1]["path"] == str(Path(stage["output"]) / "draws.jsonl"), "closed bounded ledger/journal aliases differ")
    core.require(core.canonical(closed["quota_state"]["mirror_state"]) == core.canonical(closed["rng"]["numpy"]),
                 "closed bounded augmentation RNG differs from live boundary")
    for name in ("drawn_rows", "applied_views", "step"):
        core.require(type(closed[name]) is int, "noninteger bounded closed counter")
    for name in ("train_samples_seen", "train_steps_done"):
        core.require(type(closed["metrics"][name]) is int, "noninteger bounded metric counter")
    core.require(type(closed["metrics"]["transient_cuda_retry_batches"]) is float
                 and core.number(closed["metrics"]["transient_cuda_retry_batches"], "bounded retry metric") == 0.0,
                 "bounded retry metric must be actual finite float zero")
    views = closed["quota_state"]["emitted"]
    expected_views = views - rows_after_batches(c["population"]["rows"], c["batch_size"], expected_start)
    core.require(closed["drawn_rows"] == closed["applied_views"] == closed["metrics"]["train_samples_seen"] == expected_views
                 and closed["step"] == closed["metrics"]["train_steps_done"] + expected_start
                 and closed["metrics"]["transient_cuda_retry_batches"] == 0
                 and core.number(closed["metrics"]["grad_nonfinite_skip_rate"], "bounded nonfinite skip rate") == 0, "closed bounded actual counters differ")
    complete = views == c["population"]["rows"] and closed["segment"] == len(c["updates_per_segment"])
    core.require(closed["pass_complete"] is complete and (complete or not require_complete), "closed bounded pass incomplete")
    return {"family": CONTRACT, "contract": contract_ref, "checkpoint": closed["checkpoint"], "boundary": binding["boundary"],
            "run_uuid": c["run_uuid"], "study_uuid": c["study_uuid"], "method": c["method"],
            "population_identity": identity, "source_origin_binding": origin_binding(identity), "sampling": c["sampling"],
            "steps": closed["step"], "actual_views": views, "pass_complete": complete,
            "selection_audit": checked["audit"], "scope": "caller-trusted bounded lineage metadata only; actual Root+ind own-live proof required"}


def validate_closed_model_delivery(contract_ref: dict[str, str], binding: dict[str, Any], *, require_complete: bool = True) -> dict[str, Any]:
    """Metadata API for a future adapter; caller must authenticate Root/ind proof.

    Unlike lineage-only validation, this requires the dedicated own-live wrapper
    certificate, without decoding model or journal payloads in the consumer.
    """
    lineage = validate_closed_segment(contract_ref, binding, require_complete=require_complete)
    core.require(binding["command"][-10:-8] == ["-m", "scripts.core_selfplay_union_bounded_shuffle_proof"],
                 "union model delivery requires the dedicated observed entry command")
    _, boundary = core.read_ref(binding["boundary"])
    _, c = core.read_ref(contract_ref)
    observed = boundary["own_live_certificate"]
    core.require(observed["schema"] == "core_selfplay_union_bounded_own_live_saved_payload_proof_v1"
        and observed["status"] == "BOUNDED_UNION_OWN_LIVE_SAVED_PAYLOAD_OBSERVED_PENDING_ROOT_CLOSURE_V1"
        and observed["contract"] == contract_ref and observed["stage"] == boundary["stage"]
        and observed["checkpoint"] == lineage["checkpoint"] and observed["population_identity"] == lineage["population_identity"]
        and observed["source_origin_binding"] == lineage["source_origin_binding"] and observed["sampling"] == c["sampling"]
        and observed["step"] == lineage["steps"] and observed["step_start"] == boundary["step_start"]
        and observed["actual_views"] == boundary["applied_views"] and observed["cumulative_views"] == lineage["actual_views"]
        and observed["applied_updates"] == boundary["applied_updates"]
        and observed["quota_prefix_sha256"] == boundary["quota_state"]["reservation_prefix"]
        and observed["journal_chain"] == boundary["draw_journals"]
        and observed["ledger_checkpoint"] == boundary["quota_ledger_checkpoint"], "union own-live certificate lineage mismatch")
    count = c["updates_per_segment"][boundary["segment"] - 1]
    core.require(observed["actual_calls"] == {"forward": count, "optimizer_pre": count, "optimizer_post": count, "save": 1}
        and observed["save_before_equals_after"] is True and observed["extra_forwards"] == 0
        and observed["additional_final_CPU_checkpoint_decodes"] == 1
        and observed["five_rng_sha256"] == core.digest(boundary["rng"])
        and observed["bounded_state_sha256"] == core.digest(boundary["quota_state"])
        and observed["initial"]["rng"] != {} and observed["initial"]["backend"] == boundary["backend"], "union observed passivity/counter mismatch")
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
    replay = None
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
                         max_total_bytes=c["limits"]["max_total_metadata_bytes"]) as ledger, (output / "draws.jsonl").open("xb", buffering=0) as journal:
            if parent is not None:
                core.require(ledger.checkpoint() == parent["quota_ledger_checkpoint"], "quota ledger/checkpoint mismatch")
            writer = BoundedReservationWriter(ledger, journal)
            def on_draw(rows: Any, encoded: bytes) -> None:
                nonlocal drawn, draw_batches
                core.require(journal.tell() + len(encoded) <= c["limits"]["max_file_bytes"], "bounded local journal cap")
                writer(rows, encoded)
                drawn += len(rows)
                draw_batches += 1

            membership = tuple((s["generation"], s["source_sha256"], s["rows"]) for s in manifest["sources"])
            if parent is not None:
                checked_parent = journal_identity(c, identity, membership, parent["quota_state"], prior_journals,
                                                  parent["segment"], read_payload=True)
                core.require(checked_parent["ledger"] == ledger.checkpoint() == parent["quota_ledger_checkpoint"],
                             "bounded immutable journal/ledger tail differs before payload restore")
            # Header-derived source budgets are admitted before model creation.
            # Real contents are checked per source before its rows are presented;
            # streaming does NOT pretend to decode the whole population upfront.
            loaded = from_fixed_manifest(c["architecture"], manifest, c["population"]["manifest"], rng=rng)
            core.require(loaded.identity == identity, "actual bounded source identity changed")
            replay = TrainerReplay(loaded, seed=c["sampling"]["order_seed"], batch_size=c["batch_size"],
                                   limits=ReplayLimits(**c["sampling"]["limits"]), lineage=c["run_uuid"], reserve=on_draw)
            if parent is not None:
                replay.restore(parent["quota_state"], journal_rows=parent["quota_state"]["emitted"],
                               journal_prefix=parent["quota_state"]["reservation_prefix"])
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
                core.require(bank.CHECKPOINT_KEY not in state and bank.ANCHOR_CHECKPOINT_KEY not in state and once.CHECKPOINT_KEY not in state and global_union.CHECKPOINT_KEY not in state,
                             "legacy/once checkpoint cannot masquerade as union family")
                del state
                core.require(core.file_sha(checkpoint) == parent["checkpoint"]["sha256"], "bank parent checkpoint changed during tag read")
                resume_bootstrap(trainer, checkpoint, parent["checkpoint"]["sha256"], parent["step"], c["seed"])
                # resume_bootstrap seeds Torch; restore ALL five streams afterwards.
                _restore_rng(parent["rng"], rng, cast(Any, np.random))
                core.require(core.canonical(_rng_state(rng, cast(Any, np.random))) == core.canonical(parent["rng"])
                             and core.canonical(rng.bit_generator.state) == core.canonical(parent["quota_state"]["mirror_state"]),
                             "bounded five-stream/cursor RNG restoration differs")
            core.import_custody()
            start = trainer.step
            count = c["updates_per_segment"][stage["segment"] - 1]
            context = {"trainer": trainer, "replay": replay, "population": loaded, "contract": c, "stage": stage,
                       "contract_ref": contract_ref, "stage_ref": stage_ref, "backend": backend, "rng": rng,
                       "np": np, "torch": torch, "checkpoint": output / "learner.pt"}
            if observer is not None:
                observer.before_train(context)
            replay.start_prefetch()

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
        # Torch preserves tuples but JSON boundaries do not: canonicalize once
        # so the exact saved family tag and later closed parent compare equally.
        quota_state = json.loads(core.canonical(replay.checkpoint()))
        observed = journal_identity(c, identity, membership, quota_state, draw_journals, stage["segment"], read_payload=True)
        core.require(observed["ledger"] == quota_ledger_checkpoint, "current bounded journal/ledger differs")
        core.require(quota_state["emitted"] == rows_after_batches(len(replay), c["batch_size"], trainer.step), "cumulative optimizer/quota dose mismatch")
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
            "selection_audit": observed["audit"], "quota_ledger_checkpoint": quota_ledger_checkpoint, "pass_complete": replay.remaining == 0,
            "population_identity": identity, "source_origin_binding": origin_binding(identity), "initial_model_policy": "model_only_fresh_optimizer",
            "state_handoff": "model_only_bootstrap" if parent is None else "adjacent_full_state_continuation", "initial_model": c["initial_model"], "initial_model_step": c["initial_model_step"],
            "step_start": start, "step": trainer.step, "applied_updates": applied, "applied_views": metrics.train_samples_seen,
            "checkpoint": {"path": str(checkpoint), "sha256": core.file_sha(checkpoint)},
            "drawn_rows": drawn, "draws_sha256": core.file_sha(output / "draws.jsonl"), "membership": replay.membership,
            "metrics": asdict(metrics), "rng": _rng_state(rng, cast(Any, np.random)), "backend": backend,
            "recovery_scope": "same-family closed segment boundary only; interrupted output is terminal/unqualified",
            "champion_changed": False, "collector_lineage_qualified": False}
        if observer is not None:
            boundary["own_live_certificate"] = observer.closed_checkpoint(context, boundary)
        # Compact JSON keeps bounded slots/ready IDs and the spent bitmap finite;
        # publication seals the checkpoint, journal and certificate together.
        core.require(len((core.canonical(boundary) + "\n").encode()) <= 4 << 20, "union closed boundary exceeds4MiB")
        replay.close()
        core.publish_boundary(output, boundary)
        return boundary
    except BaseException as error:
        failure = {"status": "FAILED_SELFPLAY_UNION_BOUNDED_SHUFFLE_SEGMENT_NO_RECOVERY_V1", "error": repr(error),
                   "contract": contract_ref, "stage": stage_ref, "drawn_rows": drawn, "completed_optimizer_calls": applied}
        try:
            marker = core.failure_marker(output)
            core.write_json(marker, failure)
            marker.chmod(0o444)
            core.fsync_directory(marker.parent)
        finally:
            try:
                if replay is not None:
                    replay.abort_consumption()
                    replay.close()
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
